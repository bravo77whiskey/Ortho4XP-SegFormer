"""Forest stands: one measured height per irregular patch of canopy.

On ZL18/19 textures forest polygons take a shorter copy of their forest
(O4_Forest_Heights) by the canopy height measured on the ortho. Before
stands, that height was a 20 m Gaussian average cut into 3 m height zones,
so a broad forest whose height drifted inside one zone was planted at one
height from edge to edge.

Here the canopy is cut into stands about STAND_M across before it is
measured, and each stand takes the median crown-top height of its own
measured canopy. Neighbouring stands that land on the same height copy still
make one polygon; where they differ, the polygon splits along the stand edge.

Stand shapes (``shape``):
  * ``watershed`` (default): seeds on a jittered grid flooded (cv2.watershed)
    over a ridge map: height steps in the smoothed crown tops, the line
    between measured and unmeasured canopy, plus fractal noise. Stand edges
    settle on real height steps where there are some and wander through
    flat canopy where there are none; open ground between canopy patches is
    a wall, so a stand never jumps a road.
  * ``warped``: Voronoi cells of the same seeds, read through a fractal
    noise warp, so edges are wavy but ignore the heights.

Stand pieces under MIN_FRAC of a full stand join the stand next to them.
Size and shape were picked on the review set's ZL18/19 textures
(2026-11-05, experiments/veg_stands_20261105): the error against a second
canopy pass kept falling down to 15 m stands, but 15 m made a busy mosaic
and ~40 % more polygon edge than today; watershed beat warped at the same
edge length and planted fewer stray full-height patches.

The seeds come from a generator seeded by the texture name, so a rebuild
plants the same stands. Everything runs on a ~WORK_M grid (about 1 s per
ZL18 texture) and is scaled back to the mask.
"""

from __future__ import annotations

import zlib

import cv2
import numpy as np

STAND_M = 25.0
SHAPES = ("watershed", "warped")
WORK_M = 1.0
# Watershed image: a height step of EDGE_STEP_M weighs like the full noise swing.
EDGE_STEP_M = 4.0
NOISE = 0.35
# Warp: displacement as a share of the stand size.
WARP = 0.45
# Stand pieces under this share of a full stand join a neighbouring stand.
MIN_FRAC = 0.25
SIGNATURE = "forest-stands-v2"


def _seeds(shape, spacing_px, rng):
    """One seed per spacing cell of a grid, jittered up to +-0.45 cell."""
    h, w = shape
    ys, xs = np.mgrid[0:h:spacing_px, 0:w:spacing_px].astype(np.float64)
    ys += (rng.random(ys.shape) - 0.5) * 0.9 * spacing_px + spacing_px / 2
    xs += (rng.random(xs.shape) - 0.5) * 0.9 * spacing_px + spacing_px / 2
    ys = np.clip(ys, 0, h - 1).astype(np.int64).ravel()
    xs = np.clip(xs, 0, w - 1).astype(np.int64).ravel()
    return ys, xs


def _fractal_noise(shape, scale_px, rng, octaves=3):
    """Smooth noise, about -1..1, features ~scale_px with two finer octaves at half weight."""
    h, w = shape
    out = np.zeros(shape, np.float32)
    amp, s, total = 1.0, float(scale_px), 0.0
    for _ in range(octaves):
        s = max(s, 2.0)
        gh, gw = int(h / s) + 3, int(w / s) + 3
        g = rng.standard_normal((gh, gw)).astype(np.float32)
        up = cv2.resize(g, (int(np.ceil(gw * s)), int(np.ceil(gh * s))), interpolation=cv2.INTER_CUBIC)
        out += amp * up[:h, :w]
        total += amp
        amp *= 0.5
        s /= 2.0
    return out / total


def _warped_labels(shape, spacing_px, rng):
    ys, xs = _seeds(shape, spacing_px, rng)
    img = np.ones(shape, np.uint8)
    img[ys, xs] = 0
    _d, lab = cv2.distanceTransformWithLabels(img, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    h, w = shape
    dx = _fractal_noise(shape, spacing_px * 0.6, rng) * WARP * spacing_px
    dy = _fractal_noise(shape, spacing_px * 0.6, rng) * WARP * spacing_px
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(lab.astype(np.float32), gx + dx, gy + dy, cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_REFLECT).astype(np.int32)


def _watershed_labels(shape, spacing_px, rng, tops, measured, canopy, px_m):
    ys, xs = _seeds(shape, spacing_px, rng)
    markers = np.zeros(shape, np.int32)
    markers[ys, xs] = np.arange(1, len(ys) + 1, dtype=np.int32)
    # Unmeasured canopy is filled from the measured canopy around it, so it
    # draws no height edges of its own.
    blur = lambda a, m: cv2.GaussianBlur(a.astype(np.float32), (0, 0), m / px_m)
    w = blur(measured, 6.0)
    fill = blur(np.where(measured, tops, 0), 6.0)
    hs = blur(np.where(measured, tops, fill / np.maximum(w, 1e-3)), 2.0)
    # The flood settles on ridges: height steps (a step of EDGE_STEP_M spread
    # over the 2 m blur is a full ridge), the line between measured and
    # unmeasured canopy, and the edge of the canopy. Noise lets the edges
    # wander through flat canopy.
    per_m = 1.0 / px_m
    step = np.hypot(cv2.Sobel(hs, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(hs, cv2.CV_32F, 0, 1, ksize=3))
    step *= per_m / 8.0 * 5.0 / EDGE_STEP_M
    mb = blur(measured, 2.0)
    known = np.hypot(cv2.Sobel(mb, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(mb, cv2.CV_32F, 0, 1, ksize=3))
    known *= per_m / 8.0 * 5.0
    ridge = np.clip(np.maximum(step, known), 0.0, 1.0)
    noise = (_fractal_noise(shape, spacing_px * 0.5, rng) + 1.0) * 0.5
    v = 150.0 * ridge + 100.0 * NOISE * noise
    v = np.where(canopy, v, 250.0)
    lab = cv2.watershed(cv2.merge([np.clip(v, 0, 255).astype(np.uint8)] * 3), markers)
    edge = lab <= 0
    if edge.any():
        grown = cv2.dilate(np.where(edge, 0, lab).astype(np.float32), np.ones((3, 3), np.uint8))
        lab = np.where(edge, grown.astype(np.int32), lab)
    return lab


def _split_disconnected(lab, on):
    """Pieces of one stand cut apart by open ground are stands of their own."""
    n, cc = cv2.connectedComponents(on.astype(np.uint8), connectivity=8)
    out = np.zeros(lab.shape, np.int64)
    if n <= 1:
        return out
    key = cc.astype(np.int64) * (int(lab.max()) + 2) + lab
    _u, inv = np.unique(key[on], return_inverse=True)
    out[on] = inv.ravel() + 1
    return out


def _absorb_small(lab, on, min_px):
    """Stand pieces under min_px join the stand next to them (the nearest
    pixel of a bigger stand); a canopy patch too small for any stand stays one."""
    n = int(lab.max()) + 1
    area = np.bincount(lab[on], minlength=n)
    small = (area < min_px) & (area > 0)
    small[0] = False
    if not small.any():
        return lab
    keep = on & ~small[lab]
    if not keep.any():
        return lab
    _d, near = cv2.distanceTransformWithLabels((~keep).astype(np.uint8), cv2.DIST_L2, 3,
                                               labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(keep)
    lut = np.zeros(int(near.max()) + 1, np.int64)
    lut[near[ys, xs]] = lab[ys, xs]
    out = lab.copy()
    sel = on & small[lab]
    out[sel] = lut[near[sel]]
    # A canopy patch with no big stand of its own keeps its own label.
    _n, cc = cv2.connectedComponents(on.astype(np.uint8), connectivity=8)
    has_keep = np.zeros(int(cc.max()) + 1, bool)
    has_keep[cc[keep]] = True
    alone = sel & ~has_keep[cc]
    out[alone] = lab[alone]
    return out


def stand_labels(tops, canopy, measured, m_per_px, stand_m=STAND_M, shape="watershed", seed=0,
                 min_frac=MIN_FRAC):
    """Stand index per canopy pixel (0 elsewhere), on the arrays' own grid.
    Stand pieces under ``min_frac`` of a full stand's area join a neighbour."""
    rng = np.random.default_rng(seed)
    spacing = max(2, int(round(stand_m / m_per_px)))
    if shape == "warped":
        lab = _warped_labels(canopy.shape, spacing, rng)
    else:
        lab = _watershed_labels(canopy.shape, spacing, rng, tops, measured, canopy, m_per_px)
    lab = _split_disconnected(lab, canopy)
    if min_frac > 0:
        lab = _absorb_small(lab, canopy, min_frac * spacing * spacing)
    return lab


def _medians(lab, values, sel, n):
    """Median of ``values[sel]`` per label 0..n-1 (nan where a label has none)."""
    out = np.full(n, np.nan, np.float32)
    labs = lab[sel]
    if not labs.size:
        return out
    order = np.argsort(labs, kind="stable")
    labs = labs[order]
    vals = values[sel][order]
    idx = np.arange(n)
    starts = np.searchsorted(labs, idx)
    ends = np.searchsorted(labs, idx, side="right")
    for i in np.nonzero(ends > starts)[0]:
        out[i] = np.median(vals[starts[i]:ends[i]])
    return out


def texture_seed(name):
    """Stable seed from a texture name."""
    return zlib.crc32(str(name).encode("utf-8")) & 0x7FFFFFFF


def stand_height_map(tops, canopy, measured, m_per_px, min_share, stand_m=STAND_M,
                     shape="watershed", seed=0):
    """Per pixel of the mask: its stand's median measured crown-top height,
    NaN where under ``min_share`` of the stand's canopy is measured (or off canopy).

    ``tops`` crown-top heights (m), ``canopy`` the tree mask, ``measured`` the
    canopy pixels whose height counts, all the mask's size at ``m_per_px``.
    Returns (heights, labels) on the mask's grid.
    """
    h, w = canopy.shape
    k = max(1, int(round(WORK_M / m_per_px)))
    size = (max(1, w // k), max(1, h // k))
    px_m = m_per_px * w / size[0]
    small = lambda a, interp: cv2.resize(a, size, interpolation=interp)
    tops_s = small(np.nan_to_num(tops).astype(np.float32), cv2.INTER_AREA)
    canopy_s = small(canopy.astype(np.uint8), cv2.INTER_NEAREST) > 0
    meas_s = small(measured.astype(np.uint8), cv2.INTER_NEAREST) > 0
    meas_s &= canopy_s
    lab_s = stand_labels(tops_s, canopy_s, meas_s, px_m, stand_m, shape, seed)
    # Labels back on the mask's grid; mask pixels the small grid missed take the nearest stand.
    lab = cv2.resize(lab_s.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int64)
    gap = canopy & (lab == 0)
    if gap.any() and (lab > 0).any():
        _d, near = cv2.distanceTransformWithLabels((lab == 0).astype(np.uint8), cv2.DIST_L2, 3,
                                                   labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(lab > 0)
        lut = np.zeros(int(near.max()) + 1, np.int64)
        lut[near[ys, xs]] = lab[ys, xs]
        lab[gap] = lut[near[gap]]
    lab[~canopy] = 0
    n = int(lab.max()) + 1
    on = canopy & (lab > 0)
    count = np.bincount(lab[on], minlength=n)
    sel = on & measured
    mcount = np.bincount(lab[sel], minlength=n)
    med = _medians(lab, tops.astype(np.float32), sel, n)
    med[(mcount == 0) | (mcount < min_share * count)] = np.nan
    med[0] = np.nan
    return med[lab], lab
