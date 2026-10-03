"""Crisp polygon footprints for large buildings from the landcover class map.

A YOLO OBB is a rectangle.  Large buildings are often L-, U-, T-shaped or
stepped, so the OBB either overshoots into the yard or, when the detector
only boxed a wing, undershoots the roof.  This module turns the landcover
SegFormer's building class, clipped to a margin around the OBB, into a
polygon that follows the roof without looking blobby:

1. the mask is rotated into the OBB frame (walls become axis-aligned);
2. thin bridges to neighbouring roofs are cut by a morphological opening
   and the component(s) under the OBB are kept;
3. wall positions are found as peaks of the mask's edge profiles along each
   axis, at least ``MIN_STEP_M`` apart;
4. the grid those walls define is filled cell by cell (majority of mask
   pixels) and its outline traced: a rectilinear polygon whose edges lie
   on real roof edges, with no notch smaller than a cell;
5. when the rectilinear fit is poor (angled or curved walls) the cleaned
   mask's contour is simplified to straight edges instead;
6. courtyards are kept open by cutting the cell grid through each hole, so
   every output ring is hole-free (DSF facades take one winding).

Everything is pure numpy/OpenCV and deterministic.  Callers decide what to
do when :func:`extract_footprint` returns None (no building evidence).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

MARGIN_FRAC = 0.15        # search this far beyond the OBB (fraction of side)
MARGIN_MIN_M = 4.0
MARGIN_MAX_M = 30.0
MIN_STEP_M = 3.0          # smallest wall-to-wall step kept
MIN_EDGE_FRAC = 0.35      # an edge peak must run this share of MIN_STEP_M...
MIN_HOLE_M2 = 120.0       # smaller holes are filled
MIN_FILL = 0.30           # building pixels / OBB area below this = no evidence
# Growth beyond the OBB must be a real wing: at least this deep.  Oblique
# orthos show a tall building's facade next to its roof, and the landcover
# model labels that strip "building" too; it is a few metres wide along a
# whole side, so a depth test separates it from a wing the detector missed.
WING_MIN_M = 8.0
# ...and all growth together stays under this share of the OBB: past it the
# mask has usually run into a neighbour sharing a wall or an open structure
# (a stadium bowl, a dome's annex), so the carve-only footprint is used.
MAX_GROWTH = 1.2
# A ring must lie this much inside the OBB; detached pieces are neighbours.
MIN_RING_IN_OBB = 0.5
# Irregularity of a traced outline, the sum of:
#   * the share of its perimeter in edges shorter than SHORT_EDGE_M (a
#     jagged or stair-stepped outline is mostly short steps; a clean L or U
#     is long walls -- perimeter alone cannot tell, since any orthogonally
#     convex staircase has its bounding rectangle's perimeter);
#   * |perimeter / rotated-rectangle perimeter - 1| (notches, rounding);
#   * 1 - fit IoU of the outline against its mask;
#   * 1.0 when the outline is not rectilinear (blobs, curves, angled walls)
#     or has more than MAX_REGULAR_VERTICES corners (a staircase tracing of
#     a curve; regular structures at ZL16 stay within about a dozen).
# On Kaohsiung ZL16 samples clean halls, L and stepped shapes scored at or
# below ~0.25 and rough multi-building outlines above.  Callers treat
# anything above MAX_IRREGULARITY as not one regular structure.
SHORT_EDGE_M = 6.0
MAX_REGULAR_VERTICES = 12
MAX_IRREGULARITY = 0.25
RECTILINEAR_MIN_IOU = 0.80
WORK_M_PER_PX = 0.5       # rotated working grid resolution (upsampled)


def _obb_frame(pts):
    """Return (centre, angle_rad, length, width) of a 4-point OBB (px)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    e0 = pts[1] - pts[0]
    e1 = pts[2] - pts[1]
    if np.hypot(*e0) >= np.hypot(*e1):
        axis, length, width = e0, np.hypot(*e0), np.hypot(*e1)
    else:
        axis, length, width = e1, np.hypot(*e1), np.hypot(*e0)
    return pts.mean(axis=0), math.atan2(axis[1], axis[0]), length, width


def _peaks(profile, min_sep, min_strength):
    """Indices of local maxima >= min_strength, greedy by strength."""
    order = np.argsort(-profile, kind="stable")
    order = order[profile[order] >= min_strength]
    blocked = np.zeros(len(profile), dtype=bool)
    taken = []
    for i in order.tolist():
        if not blocked[i]:
            taken.append(i)
            blocked[max(0, i - min_sep + 1):i + min_sep] = True
    return sorted(taken)


def _grid_breaks(mask, axis, min_sep, min_strength):
    """Wall positions along ``axis`` (0 = columns/x, 1 = rows/y)."""
    m = np.asarray(mask, dtype=bool)
    # |diff| of a 0/1 mask is "neighbours differ"
    if axis == 0:
        diff = (m[:, 1:] != m[:, :-1]).sum(axis=0)  # length W-1
    else:
        diff = (m[1:, :] != m[:-1, :]).sum(axis=1)
    # a boundary between pixel i and i+1 sits at coordinate i+1
    prof = np.convolve(diff.astype(np.float64), [0.25, 0.5, 0.25], mode="same")
    brk = [p + 1 for p in _peaks(prof, min_sep, min_strength)]
    n = mask.shape[1] if axis == 0 else mask.shape[0]
    occupied = mask.any(axis=0) if axis == 0 else mask.any(axis=1)
    idx = np.flatnonzero(occupied)
    if idx.size == 0:
        return []
    lo, hi = int(idx[0]), int(idx[-1]) + 1
    brk = [b for b in brk if lo + min_sep // 2 < b < hi - min_sep // 2]
    return sorted(set([lo] + brk + [min(hi, n)]))


def _cells(mask, xs, ys):
    """Boolean cell grid: majority of mask pixels in each (y, x) cell."""
    integ = cv2.integral(mask.astype(np.uint8)).astype(np.int64)
    xs = np.asarray(xs, dtype=np.int64)
    ys = np.asarray(ys, dtype=np.int64)
    corners = integ[np.ix_(ys, xs)]
    s = corners[1:, 1:] - corners[:-1, 1:] - corners[1:, :-1] + corners[:-1, :-1]
    area = np.diff(ys)[:, None] * np.diff(xs)[None, :]
    return (area > 0) & (s * 2 >= area)


def _trace_cells(grid, xs, ys):
    """Outer rings (work-frame px) of each 4-connected group of cells.

    Returns a list of (ring, holes) where holes is the number of holes.
    """
    h, w = grid.shape
    # draw the cell grid 4 px per cell (blocks share their edge pixels);
    # the 8-connected tracer cuts concave corners by one pixel, so contour
    # points are rounded to the nearest cell corner, which folds the cut
    # back onto the corner
    q = 4
    # cell (j, i) covers rows q*j..q*j+q and columns q*i..q*i+q inclusive:
    # a q-fold upsampling, widened by one pixel down and right
    up = np.zeros((q * h + 1, q * w + 1), dtype=bool)
    up[:-1, :-1] = np.repeat(np.repeat(np.asarray(grid, dtype=bool), q, 0), q, 1)
    blk = up.copy()
    blk[1:, :] |= up[:-1, :]
    blk[:, 1:] |= up[:, :-1]
    blk[1:, 1:] |= up[:-1, :-1]
    img = blk.astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    if hierarchy is None:
        return out
    hierarchy = hierarchy[0]
    for k, c in enumerate(contours):
        if hierarchy[k][3] != -1:
            continue
        holes = 0
        child = hierarchy[k][2]
        while child != -1:
            holes += 1
            child = hierarchy[child][0]
        ring = []
        for (cx, cy) in c.reshape(-1, 2):
            ring.append((xs[min(len(xs) - 1, int(round(cx / q)))],
                         ys[min(len(ys) - 1, int(round(cy / q)))]))
        ring = _dedupe_ring(ring)
        if len(ring) >= 3:
            out.append((ring, holes))
    return out


def _dedupe_ring(ring):
    """Drop repeated and collinear vertices of an axis-aligned ring."""
    pts = []
    for p in ring:
        p = (float(p[0]), float(p[1]))
        if not pts or pts[-1] != p:
            pts.append(p)
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    changed = True
    while changed and len(pts) > 3:
        changed = False
        for i in range(len(pts)):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
            cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            if abs(cross) < 1e-9:
                pts.pop(i)
                changed = True
                break
    return pts


def _split_holes(grid):
    """Split cell groups that enclose holes by cutting through each hole.

    Returns a list of boolean grids, each hole-free.
    """
    pending = [grid]
    done = []
    guard = 0
    while pending and guard < 64:
        guard += 1
        g = pending.pop()
        filled = g.copy()
        inv = (~g).astype(np.uint8)
        n, lab = cv2.connectedComponents(inv, connectivity=4)
        hole_cols = None
        for k in range(1, n):
            ys_, xs_ = np.nonzero(lab == k)
            touches = (ys_.min() == 0 or xs_.min() == 0 or
                       ys_.max() == g.shape[0] - 1 or xs_.max() == g.shape[1] - 1)
            if not touches:
                hole_cols = (xs_, ys_)
                break
        if hole_cols is None:
            done.append(g)
            continue
        xs_, ys_ = hole_cols
        # cut across the shorter extent of the whole group, through the hole
        gy, gx = np.nonzero(filled)
        if (gx.max() - gx.min()) >= (gy.max() - gy.min()):
            cut = int(round(xs_.mean()))
            a = g.copy()
            a[:, cut:] = False
            b = g.copy()
            b[:, :cut] = False
        else:
            cut = int(round(ys_.mean()))
            a = g.copy()
            a[cut:, :] = False
            b = g.copy()
            b[:cut, :] = False
        for part in (a, b):
            if part.any():
                pending.append(part)
    return done + pending


def _poly_iou(rings, mask):
    draw = np.zeros(mask.shape, dtype=np.uint8)
    for ring in rings:
        cv2.fillPoly(draw, [np.round(np.asarray(ring)).astype(np.int32)], 1)
    inter = np.count_nonzero(draw & mask)
    union = np.count_nonzero(draw | mask)
    return inter / union if union else 0.0


def extract_footprint(building_mask, obb_pts, m_per_px):
    """Return a footprint for the building under ``obb_pts``, or None.

    Tries the grown footprint first and falls back to carving inside the
    OBB when growth exceeds MAX_GROWTH.  See :func:`_extract`.
    """
    base = _extract_base(building_mask, obb_pts, m_per_px)
    if base is None:
        return None
    res = _extract_finish(base, grow=True)
    if res is not None and res["area_m2"] > MAX_GROWTH * res["obb_area_m2"]:
        # the carve-only retry shares everything up to the growth step
        res = _extract_finish(base, grow=False)
    return res


def _extract(building_mask, obb_pts, m_per_px, grow):
    """Return a footprint for the building under ``obb_pts``, or None.

    ``building_mask``: HxW bool/uint8 map of building pixels (texture px).
    ``obb_pts``: the detection's 4 corners in texture px.
    Result dict: ``rings`` (list of hole-free rings in texture px, float),
    ``kind`` ('rectilinear' | 'simplified'), ``fill`` (building share of the
    OBB), ``area_m2``, ``obb_area_m2``, ``fit_iou`` (rings vs cleaned mask).
    """
    base = _extract_base(building_mask, obb_pts, m_per_px)
    return None if base is None else _extract_finish(base, grow)


def _extract_base(building_mask, obb_pts, m_per_px):
    """Steps of :func:`_extract` that do not depend on ``grow``, or None."""
    m_per_px = float(m_per_px)
    if m_per_px <= 0.0:
        return None
    centre, ang, length, width = _obb_frame(obb_pts)
    if length < 2 or width < 2:
        return None
    obb_area_m2 = length * width * m_per_px * m_per_px
    margin_px = min(MARGIN_MAX_M, max(MARGIN_MIN_M, MARGIN_FRAC * length * m_per_px)) / m_per_px
    half_l = length / 2.0 + margin_px
    half_w = width / 2.0 + margin_px
    scale = m_per_px / WORK_M_PER_PX           # work px per texture px
    out_w = int(math.ceil(2 * half_l * scale)) + 2
    out_h = int(math.ceil(2 * half_w * scale)) + 2
    if out_w * out_h > 4_000_000:
        scale = math.sqrt(4_000_000 / float(out_w * out_h)) * scale
        out_w = int(math.ceil(2 * half_l * scale)) + 2
        out_h = int(math.ceil(2 * half_w * scale)) + 2
    work_m = m_per_px / scale
    ca, sa = math.cos(ang), math.sin(ang)
    # texture px -> work px: rotate by -ang about centre, scale, shift
    fwd = np.array([[ca * scale, sa * scale, 0.0],
                    [-sa * scale, ca * scale, 0.0]])
    fwd[:, 2] = np.array([out_w / 2.0, out_h / 2.0]) - fwd[:, :2] @ centre
    # warp only the source pixels under the work window (+2 px for the
    # bilinear taps): the mask is usually a whole texture, and converting
    # and warping all of it per detection dominated placement time
    bm = np.asarray(building_mask)
    win = np.array([[0.0, 0.0], [out_w, 0.0], [0.0, out_h], [out_w, out_h]])
    inv0 = cv2.invertAffineTransform(fwd)
    win = win @ inv0[:, :2].T + inv0[:, 2]
    sx0 = max(0, int(math.floor(win[:, 0].min())) - 2)
    sy0 = max(0, int(math.floor(win[:, 1].min())) - 2)
    sx1 = min(bm.shape[1], int(math.ceil(win[:, 0].max())) + 3)
    sy1 = min(bm.shape[0], int(math.ceil(win[:, 1].max())) + 3)
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    fwd_crop = fwd.copy()
    fwd_crop[:, 2] += fwd[:, :2] @ np.array([sx0, sy0], dtype=np.float64)
    src = (bm[sy0:sy1, sx0:sx1] != 0).astype(np.uint8) * 255
    work = cv2.warpAffine(src, fwd_crop, (out_w, out_h), flags=cv2.INTER_LINEAR,
                          borderValue=0) >= 128
    # OBB in the work frame is axis-aligned and centred
    ox0 = int(round(out_w / 2.0 - length * scale / 2.0))
    ox1 = int(round(out_w / 2.0 + length * scale / 2.0))
    oy0 = int(round(out_h / 2.0 - width * scale / 2.0))
    oy1 = int(round(out_h / 2.0 + width * scale / 2.0))
    obb_px = max(1, (ox1 - ox0) * (oy1 - oy0))
    fill = np.count_nonzero(work[oy0:oy1, ox0:ox1]) / float(obb_px)
    if fill < MIN_FILL:
        return None

    # clean: cut thin bridges, close gaps, keep what sits under the OBB
    k = max(1, int(round(MIN_STEP_M / work_m)))
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    clean = cv2.morphologyEx(work.astype(np.uint8), cv2.MORPH_OPEN, kern)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kern)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=4)
    core = np.zeros_like(clean)
    sx = int((ox1 - ox0) * 0.2)
    sy = int((oy1 - oy0) * 0.2)
    core[oy0 + sy:oy1 - sy, ox0 + sx:ox1 - sx] = 1
    obb_m = np.zeros_like(clean)
    obb_m[oy0:oy1, ox0:ox1] = 1
    keep = np.zeros(clean.shape, dtype=bool)
    obb_b = obb_m != 0
    core_b = core != 0
    for c in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[c])
        win_c = (slice(y, y + h), slice(x, x + w))
        comp = lab[win_c] == c
        inside = int(np.count_nonzero(comp & obb_b[win_c]))
        if np.any(comp & core_b[win_c]) and inside >= 0.3 * area:
            keep[win_c] |= comp
        elif inside >= 0.6 * area:
            keep[win_c] |= comp
    if not keep.any():
        return None
    return {"keep": keep, "obb_m": obb_m, "work_m": work_m, "fwd": fwd,
            "m_per_px": m_per_px, "fill": fill, "obb_area_m2": obb_area_m2}


def _extract_finish(base, grow):
    """Growth, cell grid, tracing and scoring of :func:`_extract`."""
    keep = base["keep"].copy()
    obb_m = base["obb_m"]
    work_m = base["work_m"]
    fwd = base["fwd"]
    m_per_px = base["m_per_px"]
    fill = base["fill"]
    obb_area_m2 = base["obb_area_m2"]
    outside = keep & (obb_m == 0)
    if not grow:
        keep &= obb_m != 0
    elif outside.any():
        kw = max(1, int(round(WING_MIN_M / work_m)))
        wings = cv2.morphologyEx(
            outside.astype(np.uint8), cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kw))) != 0
        keep = (keep & (obb_m != 0)) | wings
    # fill small holes, keep courtyards
    inv = (~keep).astype(np.uint8)
    hn, hlab, hstats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    min_hole_px = MIN_HOLE_M2 / (work_m * work_m)
    hs = hstats.astype(np.int64)
    small_hole = (
        (hs[:, 0] != 0) & (hs[:, 1] != 0) &
        (hs[:, 0] + hs[:, 2] < inv.shape[1]) &
        (hs[:, 1] + hs[:, 3] < inv.shape[0]) &
        (hs[:, 4] < min_hole_px)
    )
    small_hole[0] = False
    if small_hole.any():
        keep |= small_hole[hlab]
    keep_u8 = keep.astype(np.uint8)

    min_sep = max(2, int(round(MIN_STEP_M / work_m)))
    min_strength = MIN_EDGE_FRAC * min_sep
    xs = _grid_breaks(keep, 0, min_sep, min_strength)
    ys = _grid_breaks(keep, 1, min_sep, min_strength)
    rings_work = []
    kind = "rectilinear"
    if len(xs) >= 2 and len(ys) >= 2:
        grid = _cells(keep, xs, ys)
        min_ring_px = 40.0 / (work_m * work_m)
        for part in _split_holes(grid):
            # corner-touching groups are separate buildings (no pinched rings)
            n_g, lab_g = cv2.connectedComponents(part.astype(np.uint8),
                                                 connectivity=4)
            for g in range(1, n_g):
                for ring, _holes in _trace_cells(lab_g == g, xs, ys):
                    if abs(cv2.contourArea(np.asarray(ring, np.float32))) >= min_ring_px:
                        rings_work.append(ring)
    fit = _poly_iou(rings_work, keep_u8) if rings_work else 0.0
    if fit < RECTILINEAR_MIN_IOU:
        kind = "simplified"
        rings_work = []
        contours, _ = cv2.findContours(keep_u8, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        eps = max(1.0, 1.5 / work_m)
        for c in contours:
            if cv2.contourArea(c) * work_m * work_m < 40.0:
                continue
            ap = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
            if len(ap) >= 3:
                rings_work.append([(float(x), float(y)) for x, y in ap])
        fit = _poly_iou(rings_work, keep_u8) if rings_work else 0.0
        if not rings_work:
            return None
    # detached pieces mostly outside the OBB belong to neighbours
    kept_rings = []
    for ring in rings_work:
        draw = np.zeros(obb_m.shape, dtype=np.uint8)
        cv2.fillPoly(draw, [np.round(np.asarray(ring)).astype(np.int32)], 1)
        tot = np.count_nonzero(draw)
        if tot and np.count_nonzero(draw & obb_m) >= MIN_RING_IN_OBB * tot:
            kept_rings.append(ring)
    rings_work = kept_rings
    if not rings_work:
        return None
    inv_fwd = cv2.invertAffineTransform(fwd)
    rings = []
    area_px = 0.0
    for ring in rings_work:
        arr = np.asarray(ring, dtype=np.float64)
        tex = arr @ inv_fwd[:, :2].T + inv_fwd[:, 2]
        rings.append(tex)
        area_px += abs(cv2.contourArea(tex.astype(np.float32)))
    all_pts = np.vstack(rings).astype(np.float32)
    (_c, (rw, rh), _a) = cv2.minAreaRect(all_pts)
    perimeter = sum(cv2.arcLength(r.astype(np.float32), True) for r in rings)
    short = 0.0
    for r in rings:
        edges = np.linalg.norm(np.roll(r, -1, axis=0) - r, axis=1)
        short += float(edges[edges * m_per_px < SHORT_EDGE_M].sum())
    irregularity = (short / max(perimeter, 1e-6) +
                    abs(perimeter / max(2.0 * (rw + rh), 1e-6) - 1.0) +
                    (1.0 - float(fit)) +
                    (0.0 if kind == "rectilinear" and
                     sum(len(r) for r in rings) <= MAX_REGULAR_VERTICES
                     else 1.0))
    return {
        "irregularity": float(irregularity),
        "rings": rings,
        "kind": kind,
        "fill": float(fill),
        "area_m2": float(area_px * m_per_px * m_per_px),
        "obb_area_m2": float(obb_area_m2),
        "fit_iou": float(fit),
    }


# ---------------------------------------------------------------- segment-first
#
# Segment-first placement (overlay switch O4_BLD_SEG_FIRST) starts from
# the building-class components themselves instead of detector boxes.  At
# ZL16 the landcover model cannot see party walls, so one component can hold
# several buildings.  The programmatic cue left in the ortho is a straight
# line across the roof where the roof colour changes (different roofing, a
# parapet, a height step casting a shadow): split_shared_walls() cuts a
# component there.  Without such a line, wall-sharing buildings stay one.

SEG_OPEN_M = 2.0          # bridges narrower than this are cut first
SPLIT_MIN_PIECE_M = 8.0   # each side of a cut is at least this long
SPLIT_STRIP_M = 4.0       # roof sampled this far on each side of a cut
SPLIT_MIN_CHROMA = 10.0   # mean a*b* difference across the cut (roofing)
# ...or a dark line along the cut (gap, parapet shadow, height step) at
# least this much darker (L*) than both sides.  A brightness change alone
# is a pitched roof's ridge, not a wall.
SPLIT_MIN_GAP_L = 12.0
SPLIT_MIN_SHARE = 0.2     # each piece holds at least this share of the area
SPLIT_MIN_DE = SPLIT_MIN_CHROMA   # kept for cache keys
SPLIT_MIN_LINE = 1.4      # gradient on the cut / median gradient in the roof
SPLIT_MIN_SPAN = 0.8      # the cut crosses the component over this share
SPLIT_MAX_DEPTH = 3
# traced area / its rotated rectangle at or above this -> "rectangle": at
# ZL16 a plain hall often traces with a notch or a chamfered corner, and a
# 4-vertex-only test sent two thirds of such halls to facades
RECT_MIN_FILL = 0.85


def _component_frame(mask):
    """minAreaRect of a mask's pixels -> (corners, rect)."""
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    return cv2.boxPoints(rect), rect


def _rotate_to_frame(arr, rect, interp):
    """Rotate ``arr`` so the rect is axis-aligned.

    Returns the rotated array and the forward affine (texture px -> frame).
    """
    (cx, cy), (w, h), ang = rect
    pad = 4
    out_w = int(math.ceil(w)) + 2 * pad
    out_h = int(math.ceil(h)) + 2 * pad
    m = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    m[0, 2] += out_w / 2.0 - cx
    m[1, 2] += out_h / 2.0 - cy
    rot = cv2.warpAffine(arr, m, (out_w, out_h), flags=interp, borderValue=0)
    return rot, m


def _best_cut(lab, mask, grad, m_per_px, axis):
    """Best straight cut along ``axis`` (0 = vertical line): (axis, pos, dE)."""
    strip = max(2, int(round(SPLIT_STRIP_M / m_per_px)))
    min_piece = max(strip, int(round(SPLIT_MIN_PIECE_M / m_per_px)))
    roof_grad = grad[mask]
    if roof_grad.size < 16:
        return None
    med = float(np.median(roof_grad)) + 1e-3
    m = mask if axis == 0 else mask.T
    L = lab if axis == 0 else lab.transpose(1, 0, 2)
    g = grad if axis == 0 else grad.T
    cols = np.flatnonzero(m.any(axis=0))
    if cols.size == 0:
        return None
    lo, hi = int(cols[0]), int(cols[-1]) + 1
    col_sum = m.sum(axis=0)
    col_area = np.cumsum(col_sum)
    total = float(col_area[-1])
    # the area, share and span gates are integer counts: evaluate them for
    # every column at once and run the colour tests only where they pass
    n_cols = m.shape[1]
    c_all = np.arange(lo + min_piece, hi - min_piece + 1)
    if c_all.size == 0:
        return None
    cs = np.concatenate(([0], col_area))
    a_sum = cs[c_all] - cs[c_all - strip]
    b_sum = cs[np.minimum(c_all + strip, n_cols)] - cs[c_all]
    left_all = col_area[c_all - 1] / total
    both = np.zeros(n_cols + 1, dtype=np.int64)
    both[1:n_cols] = (m[:, :-1] & m[:, 1:]).sum(axis=0)
    span_all = both[c_all] / np.maximum(
        1, np.maximum(col_sum[c_all - 1], col_sum[np.minimum(c_all, n_cols - 1)]))
    gate = ((a_sum >= 4) & (b_sum >= 4) &
            (left_all >= SPLIT_MIN_SHARE) & (left_all <= 1.0 - SPLIT_MIN_SHARE) &
            (span_all >= SPLIT_MIN_SPAN))
    # the cut-line gradient test, estimated in float64 for all columns; the
    # 1% margin dwarfs float32 rounding, so only columns the exact test below
    # would reject are skipped
    line_all = m[:, :-1] & m[:, 1:]
    g_line = np.where(line_all, np.maximum(g[:, :-1], g[:, 1:]), 0.0)
    g_sum = np.zeros(n_cols + 1, dtype=np.float64)
    g_sum[1:n_cols] = g_line.sum(axis=0, dtype=np.float64)
    gl_est = g_sum[c_all] / np.maximum(both[c_all], 1) / med
    gate &= gl_est >= SPLIT_MIN_LINE * 0.99
    # same for the roofing-colour / dark-line test: strip means from column
    # prefix sums.  float32 means of n values in [-128, 128] stray by at
    # most ~n * 1e-5, so the margin keeps every column the test could pass.
    if gate.any():
        lab_sum = np.zeros((n_cols + 1, 3), dtype=np.float64)
        lab_sum[1:] = np.cumsum(
            np.where(m[..., None], L, 0.0).sum(axis=0, dtype=np.float64), axis=0)
        sa = lab_sum[c_all] - lab_sum[c_all - strip]
        sb = lab_sum[np.minimum(c_all + strip, n_cols)] - lab_sum[c_all]
        la_est = sa / np.maximum(a_sum, 1)[:, None]
        lb_est = sb / np.maximum(b_sum, 1)[:, None]
        chroma_est = np.hypot(la_est[:, 1] - lb_est[:, 1], la_est[:, 2] - lb_est[:, 2])
        l_line = np.where(line_all, np.minimum(L[:, :-1, 0], L[:, 1:, 0]), 0.0)
        l_sum = np.zeros(n_cols + 1, dtype=np.float64)
        l_sum[1:n_cols] = l_line.sum(axis=0, dtype=np.float64)
        gap_est = (np.minimum(la_est[:, 0], lb_est[:, 0]) -
                   l_sum[c_all] / np.maximum(both[c_all], 1))
        margin = 0.5 + 3e-5 * np.maximum(a_sum, b_sum)
        gate &= ((chroma_est >= SPLIT_MIN_CHROMA - margin) |
                 (gap_est >= SPLIT_MIN_GAP_L - margin))
    best = None
    for c in c_all[gate].tolist():
        a = m[:, c - strip:c]
        b = m[:, c:c + strip]
        line = m[:, c - 1] & m[:, c]
        la = L[:, c - strip:c][a].mean(axis=0)
        lb = L[:, c:c + strip][b].mean(axis=0)
        chroma = float(np.linalg.norm(la[1:] - lb[1:]))
        line_l = float(np.minimum(L[:, c - 1, 0][line], L[:, c, 0][line]).mean())
        gap = min(la[0], lb[0]) - line_l
        if chroma < SPLIT_MIN_CHROMA and gap < SPLIT_MIN_GAP_L:
            continue
        de = max(chroma, gap)
        gl = float(np.maximum(g[:, c - 1][line], g[:, c][line]).mean()) / med
        if gl < SPLIT_MIN_LINE:
            continue
        if best is None or de > best[2]:
            best = (axis, c, de)
    return best


def split_shared_walls(rgb, mask, m_per_px, depth=0):
    """Split one component at straight roof-colour lines.

    ``rgb`` and ``mask`` (bool) share a shape.  Returns a list of bool masks;
    the input alone when nothing splits.
    """
    if depth >= SPLIT_MAX_DEPTH or rgb is None:
        return [mask]
    if np.count_nonzero(mask) * m_per_px * m_per_px < 2 * SPLIT_MIN_PIECE_M ** 2:
        return [mask]
    _corners, rect = _component_frame(mask)
    mrot, fwd = _rotate_to_frame(mask.astype(np.uint8), rect, cv2.INTER_NEAREST)
    mrot = mrot != 0
    rgbrot, _ = _rotate_to_frame(np.ascontiguousarray(rgb), rect, cv2.INTER_LINEAR)
    lab = cv2.cvtColor(rgbrot, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    gray = cv2.cvtColor(rgbrot, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    # a vertical cut needs x-gradient evidence along it, a horizontal one y
    cut = None
    for grad, axis in ((gx, 0), (gy, 1)):
        c = _best_cut(lab, mrot, grad, m_per_px, axis)
        if c is not None and (cut is None or c[2] > cut[2]):
            cut = c
    if cut is None:
        return [mask]
    axis, pos, _de = cut
    side = np.zeros(mrot.shape, dtype=np.uint8)
    if axis == 0:
        side[:, pos:] = 1
    else:
        side[pos:, :] = 1
    inv = cv2.invertAffineTransform(fwd)
    h, w = mask.shape
    back = cv2.warpAffine(side, inv, (w, h), flags=cv2.INTER_NEAREST,
                          borderValue=0) != 0
    out = []
    for part in (mask & ~back, mask & back):
        if np.count_nonzero(part):
            out.extend(split_shared_walls(rgb, part, m_per_px, depth + 1))
    return out


def trace_piece(mask, m_per_px):
    """Trace one building mask (bool) -> dict or None.

    Adds ``rect`` (the rotated rectangle's 4 corners when the outline is
    one ring filling >= RECT_MIN_FILL of that rectangle, else None) and
    ``frame`` (the mask's rotated bounding rectangle) to extract's result.
    """
    if np.count_nonzero(mask) < 4:
        return None
    corners, _rect = _component_frame(mask)
    res = _extract(mask.astype(np.uint8), corners, m_per_px, grow=False)
    if res is None:
        return None
    rect = None
    if len(res["rings"]) == 1:
        ring = np.asarray(res["rings"][0], dtype=np.float32)
        box = cv2.minAreaRect(ring)
        (_c, (rw, rh), _a) = box
        if abs(cv2.contourArea(ring)) >= RECT_MIN_FILL * max(rw * rh, 1e-6):
            rect = cv2.boxPoints(box).astype(np.float64)
    res["rect"] = rect
    res["frame"] = corners
    return res


def segment_components(building_mask, m_per_px, min_m2, cut_mask=None):
    """Opened building-class components as ((x, y, w, h), bool mask crop).

    ``cut_mask`` (e.g. rasterized OSM roads) is removed first, so a street
    always separates two components.
    """
    bm = (np.asarray(building_mask) != 0).astype(np.uint8)
    if cut_mask is not None:
        bm[np.asarray(cut_mask) != 0] = 0
    k = max(1, int(round(SEG_OPEN_M / m_per_px)))
    op = cv2.morphologyEx(bm, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(op, connectivity=4)
    min_px = min_m2 / (m_per_px * m_per_px)
    out = []
    for c in range(1, n):
        x, y, w, h, a = stats[c]
        if a < min_px:
            continue
        out.append(((int(x), int(y), int(w), int(h)), lab[y:y + h, x:x + w] == c))
    return out


def roof_edge_density(rgb, mask):
    """Share of Canny edge pixels inside a component (eroded by 1 px).

    One roof reads as a few long lines; a block of several buildings as
    many.  On Kaohsiung ZL16 samples single halls and sheds sat at or
    below ~0.26 and multi-building complexes above.  None without pixels.
    """
    if rgb is None:
        return None
    inner = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if np.count_nonzero(inner) < 20:
        return None
    gray = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 140) > 0
    return float(edges[inner].mean())


# Ground look-alikes.  The landcover model sometimes labels running tracks,
# clay pitches, dirt lots and dark fields "building" (bare land is almost
# never predicted on Taiwan ZL16 samples, so the class map cannot veto
# them).  Two cues, both requiring a flat surface (roofs, even brown
# corrugated ones, carry edges): a soil colour, or no colour contrast with
# the non-building ground around the outline.
GROUND_RING_M = 6.0
GROUND_MAX_EDGE = 0.15    # soil test: flat
GROUND_SOIL_A = (3.0, 20.0)
GROUND_SOIL_B = (10.0, 35.0)
GROUND_SOIL_L = (30.0, 72.0)
GROUND_MAX_DE = 6.0       # contrast test: indistinguishable from the ring
GROUND_DE_MAX_EDGE = 0.2


def looks_like_ground(rgb, piece, building, m_per_px):
    """True when a claimed outline is more likely open ground than a roof.

    ``rgb`` (HxWx3), ``piece`` and ``building`` (HxW bool) share a window
    that extends at least GROUND_RING_M around the piece.  Returns
    (bool, reason) with reason 'soil', 'no_contrast' or None.
    """
    if rgb is None:
        return False, None
    piece_u8 = piece.astype(np.uint8)
    inner = cv2.erode(piece_u8, np.ones((3, 3), np.uint8)) > 0
    r = max(2, int(round(GROUND_RING_M / m_per_px)))
    ring = (cv2.dilate(piece_u8, np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0)
    ring &= ~piece & ~building
    if np.count_nonzero(inner) < 10 or np.count_nonzero(ring) < 10:
        return False, None
    rgb = np.ascontiguousarray(rgb)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    edges = cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 60, 140) > 0
    edge = float(edges[inner].mean())
    mean = lab[piece].mean(axis=0)
    if (edge < GROUND_MAX_EDGE and
            GROUND_SOIL_L[0] <= mean[0] <= GROUND_SOIL_L[1] and
            GROUND_SOIL_A[0] <= mean[1] <= GROUND_SOIL_A[1] and
            GROUND_SOIL_B[0] <= mean[2] <= GROUND_SOIL_B[1]):
        return True, "soil"
    de = float(np.linalg.norm(mean - lab[ring].mean(axis=0)))
    if de < GROUND_MAX_DE and edge < GROUND_DE_MAX_EDGE:
        return True, "no_contrast"
    return False, None
