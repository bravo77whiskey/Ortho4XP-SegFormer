"""Forest density from the texture: how much of the tree mask is really canopy.

SegFormer's P(tree) follows LiDAR cover closely (r 0.76-0.96 per 64 m cell inside
the mask, ZL16-18, Brandenburg) but reads near zero on some forests it does not
recognise (Bahia, Brazil), so the rest of the estimate comes from brightness:
each pixel unmixed between the masked canopy and the open ground nearby. That
needs no resolved crowns and caps at UNMIX_WEIGHT, so a lawn the RGB mask took
for trees stays thin. Deep inside a solid block of mask the cap lifts to 1: closed
forest the model misses (Shenzhen hills) is not a lawn or a grove. Models trained
for tree cover did worse (OAM-TCD reads ~0 on real ZL16 textures). Scores and
sheets: experiments/veg_density_20261102.
"""
import os

import cv2
import numpy as np

# cover_map works on a grid HALF_RES_SCALE times coarser than the mask, then
# scales back up: the cover is smoothed over COVER_SIGMA_M (32 m) anyway. At 2x
# it is ~3x faster and 97-99 % of tree pixels keep their band, none moves two
# bands (hotspots, 2026-11-04). Setting veg_density_half_res, passed in as
# O4_VEG_DENSITY_HALF; off computes at the mask's size.
HALF_RES_SCALE = 2
BASE_SIGNATURE = "density-v2"


def work_scale():
    value = os.environ.get("O4_VEG_DENSITY_HALF", "").strip().lower()
    return 1 if value in {"0", "false", "no", "off"} else HALF_RES_SCALE


def signature():
    """Cache signature: changes with the working resolution."""
    k = work_scale()
    return BASE_SIGNATURE if k == 1 else f"{BASE_SIGNATURE}-scale{k}"

UNMIX_WEIGHT = 0.5
SOLID_RADIUS_M = 200.0
SOLID_LOW = 0.7           # share of the land around a pixel that is mask: cap starts to lift
SOLID_HIGH = 0.9          # cap gone
UNMIX_CELL_M = 600.0
CANOPY_PERCENTILE = 30
GROUND_PERCENTILE = 50
MIN_SPAN = 0.08           # luminance; little ground contrast must not exaggerate cover noise
LUM_BLUR_M = 2.0
COVER_SIGMA_M = 32.0

# A polygon has one density, so the mask is split where cover crosses an edge.
# The top band is the full forest density (veg_density when set, else FULL_DENSITY,
# auto's densest forest): never denser than without this.
FULL_DENSITY = 230
BAND_EDGES = (0.25, 0.40, 0.55, 0.70, 0.85)
BAND_COVER = (0.18, 0.33, 0.47, 0.62, 0.77, 1.0)

_LUM = np.array([0.299, 0.587, 0.114], np.float32)


def band_density(band, full=FULL_DENSITY):
    """DSF density of a band from density_bands (1 = thinnest)."""
    return int(round(full * BAND_COVER[int(band) - 1]))


def _cell_percentile(lum, on, cell_px, q):
    h, w = lum.shape
    ny, nx = max(1, h // cell_px), max(1, w // cell_px)
    grid = np.full((ny, nx), np.nan, np.float32)
    for j in range(ny):
        for i in range(nx):
            cell = (slice(j * h // ny, (j + 1) * h // ny), slice(i * w // nx, (i + 1) * w // nx))
            values = lum[cell][on[cell]]
            if values.size >= 0.03 * lum[cell].size:
                grid[j, i] = np.percentile(values[::3], q)
    return grid


def _smooth_grid(grid, fallback, shape):
    """Cells without pixels take their neighbours', the texture's value if none."""
    missing = np.isnan(grid)
    if missing.all():
        return np.full(shape, fallback, np.float32)
    weight = cv2.GaussianBlur((~missing).astype(np.float32), (0, 0), 1.5)
    value = cv2.GaussianBlur(np.where(missing, 0.0, grid).astype(np.float32), (0, 0), 1.5)
    smooth = np.where(weight > 0.02, value / np.maximum(weight, 1e-3), fallback).astype(np.float32)
    return cv2.resize(smooth, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def unmix(img, mask, m_per_px, not_ground=None):
    """Per pixel 0..1 from local ground to local canopy brightness.

    ``not_ground`` marks pixels unfit as the bright endmember (water).
    """
    lum = (img.astype(np.float32) / 255.0) @ _LUM
    if LUM_BLUR_M / m_per_px > 0.4:
        lum = cv2.GaussianBlur(lum, (0, 0), LUM_BLUR_M / m_per_px)
    on = mask > 0
    open_ground = ~on if not_ground is None else ~on & ~not_ground
    canopy_all = float(np.percentile(lum[on], CANOPY_PERCENTILE)) if on.any() else 0.15
    ground_all = (float(np.percentile(lum[open_ground], GROUND_PERCENTILE))
                  if open_ground.any() else canopy_all + MIN_SPAN)
    cell = max(16, int(round(UNMIX_CELL_M / m_per_px)))
    canopy = _smooth_grid(_cell_percentile(lum, on, cell, CANOPY_PERCENTILE), canopy_all, lum.shape)
    ground = _smooth_grid(_cell_percentile(lum, open_ground, cell, GROUND_PERCENTILE), ground_all, lum.shape)
    span = np.maximum(ground - canopy, MIN_SPAN)
    return np.clip((canopy + span - lum) / span, 0.0, 1.0)


def _blur_ratio(num, den, sigma_m, m_per_px):
    """Gaussian blur of num / blur of den at the arrays' size.

    On a grid ~1/8 sigma per cell: same result, a fraction of the time.
    """
    h, w = den.shape
    k = max(1, int(sigma_m / m_per_px / 8))
    size = (max(1, w // k), max(1, h // k))
    sigma = sigma_m / (m_per_px * w / size[0])
    n = cv2.GaussianBlur(cv2.resize(num, size, interpolation=cv2.INTER_AREA), (0, 0), sigma)
    d = cv2.GaussianBlur(cv2.resize(den, size, interpolation=cv2.INTER_AREA), (0, 0), sigma)
    return cv2.resize(n / np.maximum(d, 1e-3), (w, h), interpolation=cv2.INTER_LINEAR)


def solidity(mask, m_per_px, not_ground=None):
    """Share of the land around each pixel that is mask; water counts for neither side."""
    m = (mask > 0).astype(np.float32)
    land = np.ones_like(m) if not_ground is None else (~not_ground).astype(np.float32)
    return _blur_ratio(m * land, land, SOLID_RADIUS_M, m_per_px)


def cover_map(img, tree_prob, mask, m_per_px, not_ground=None):
    """Canopy cover 0..1 at the mask's size, smoothed over the mask's pixels only.

    ``tree_prob`` is P(tree) as uint8 at any size; ``img`` the RGB texture.
    """
    full_h, full_w = mask.shape
    k = work_scale()
    if k > 1 and min(full_h, full_w) >= 64 * k:
        size = (full_w // k, full_h // k)
        mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
        if not_ground is not None:
            not_ground = cv2.resize(not_ground.astype(np.uint8), size,
                                    interpolation=cv2.INTER_NEAREST) > 0
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        m_per_px = m_per_px * full_w / size[0]
    h, w = mask.shape
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    prob = cv2.resize(tree_prob.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_LINEAR)
    t = np.clip((solidity(mask, m_per_px, not_ground) - SOLID_LOW) / (SOLID_HIGH - SOLID_LOW), 0.0, 1.0)
    weight = UNMIX_WEIGHT + (1.0 - UNMIX_WEIGHT) * t * t * (3.0 - 2.0 * t)
    raw = prob + (1.0 - prob) * weight * unmix(img, mask, m_per_px, not_ground)
    m = (mask > 0).astype(np.float32)
    cover = _blur_ratio(raw * m, m, COVER_SIGMA_M, m_per_px)
    if (h, w) != (full_h, full_w):
        cover = cv2.resize(cover, (full_w, full_h), interpolation=cv2.INTER_LINEAR)
    return cover


def density_bands(cover, mask):
    """Band per masked pixel (1 = thinnest), 0 elsewhere."""
    return np.where(mask > 0, np.digitize(cover, BAND_EDGES) + 1, 0).astype(np.uint8)
