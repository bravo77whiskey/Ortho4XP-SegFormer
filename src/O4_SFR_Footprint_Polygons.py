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
    taken = []
    for i in order.tolist():
        if profile[i] < min_strength:
            break
        if all(abs(i - j) >= min_sep for j in taken):
            taken.append(i)
    return sorted(taken)


def _grid_breaks(mask, axis, min_sep, min_strength):
    """Wall positions along ``axis`` (0 = columns/x, 1 = rows/y)."""
    m = mask.astype(np.int16)
    if axis == 0:
        diff = np.abs(np.diff(m, axis=1)).sum(axis=0)  # length W-1
    else:
        diff = np.abs(np.diff(m, axis=0)).sum(axis=1)
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
    grid = np.zeros((len(ys) - 1, len(xs) - 1), dtype=bool)
    integ = cv2.integral(mask.astype(np.uint8))
    for j in range(len(ys) - 1):
        y0, y1 = ys[j], ys[j + 1]
        for i in range(len(xs) - 1):
            x0, x1 = xs[i], xs[i + 1]
            area = (y1 - y0) * (x1 - x0)
            if area <= 0:
                continue
            s = integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]
            grid[j, i] = s * 2 >= area
    return grid


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
    img = np.zeros((q * h + 1, q * w + 1), dtype=np.uint8)
    for j in range(h):
        for i in range(w):
            if grid[j, i]:
                img[q * j:q * j + q + 1, q * i:q * i + q + 1] = 255
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
    res = _extract(building_mask, obb_pts, m_per_px, grow=True)
    if res is not None and res["area_m2"] > MAX_GROWTH * res["obb_area_m2"]:
        res = _extract(building_mask, obb_pts, m_per_px, grow=False)
    return res


def _extract(building_mask, obb_pts, m_per_px, grow):
    """Return a footprint for the building under ``obb_pts``, or None.

    ``building_mask``: HxW bool/uint8 map of building pixels (texture px).
    ``obb_pts``: the detection's 4 corners in texture px.
    Result dict: ``rings`` (list of hole-free rings in texture px, float),
    ``kind`` ('rectilinear' | 'simplified'), ``fill`` (building share of the
    OBB), ``area_m2``, ``obb_area_m2``, ``fit_iou`` (rings vs cleaned mask).
    """
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
    src = (np.asarray(building_mask) != 0).astype(np.uint8) * 255
    work = cv2.warpAffine(src, fwd, (out_w, out_h), flags=cv2.INTER_LINEAR,
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
    for c in range(1, n):
        comp = lab == c
        area = int(stats[c, cv2.CC_STAT_AREA])
        inside = int(np.count_nonzero(comp & (obb_m != 0)))
        if np.any(comp & (core != 0)) and inside >= 0.3 * area:
            keep |= comp
        elif inside >= 0.6 * area:
            keep |= comp
    if not keep.any():
        return None
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
    for c in range(1, hn):
        x, y, w, h, a = hstats[c]
        if x == 0 or y == 0 or x + w >= inv.shape[1] or y + h >= inv.shape[0]:
            continue
        if a < min_hole_px:
            keep[hlab == c] = True
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
    return {
        "rings": rings,
        "kind": kind,
        "fill": float(fill),
        "area_m2": float(area_px * m_per_px * m_per_px),
        "obb_area_m2": float(obb_area_m2),
        "fit_iou": float(fit),
    }
