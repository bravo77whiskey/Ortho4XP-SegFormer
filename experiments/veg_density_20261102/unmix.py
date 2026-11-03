"""Sub-pixel tree cover by linear unmixing: each pixel lies between two local endmembers.

Canopy = dark percentile of the masked pixels, ground = bright percentile of the rest.
Unlike crown counting this keeps working when crowns are not resolved (ZL16).
"""
import cv2
import numpy as np

WEIGHTS = np.array([0.299, 0.587, 0.114], np.float32)


def _grid(lum, on, cell_px, q, fallback):
    h, w = lum.shape
    ny, nx = max(1, h // cell_px), max(1, w // cell_px)
    out = np.full((ny, nx), np.nan, np.float32)
    for j in range(ny):
        for i in range(nx):
            s = (slice(j * h // ny, (j + 1) * h // ny), slice(i * w // nx, (i + 1) * w // nx))
            v = lum[s][on[s]]
            if v.size >= 0.03 * lum[s].size:
                out[j, i] = np.percentile(v[::3], q)
    return out, fallback


def _fill_smooth(grid, fallback, shape):
    bad = np.isnan(grid)
    if bad.all():
        return np.full(shape, fallback, np.float32)
    weight = cv2.GaussianBlur((~bad).astype(np.float32), (0, 0), 1.5)
    value = cv2.GaussianBlur(np.where(bad, 0, grid), (0, 0), 1.5)
    smooth = np.where(weight > 0.02, value / np.maximum(weight, 1e-3), fallback)
    return cv2.resize(smooth.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def unmix(rgb, mask, mpp, cell_m=600.0, canopy_q=15, ground_q=60, min_span=0.08, blur_m=2.0):
    """Cover 0..1 per pixel. Without visible ground nearby the span is small and cover stays high."""
    lum = (rgb.astype(np.float32) / 255) @ WEIGHTS
    if blur_m / mpp > 0.4:
        lum = cv2.GaussianBlur(lum, (0, 0), blur_m / mpp)
    on = mask > 0
    cell = max(16, int(round(cell_m / mpp)))
    c_fallback = np.percentile(lum[on], canopy_q) if on.any() else 0.15
    g_fallback = np.percentile(lum[~on], ground_q) if (~on).any() else c_fallback + min_span
    cg, _ = _grid(lum, on, cell, canopy_q, c_fallback)
    gg, _ = _grid(lum, ~on, cell, ground_q, g_fallback)
    canopy = _fill_smooth(cg, c_fallback, lum.shape)
    ground = _fill_smooth(gg, g_fallback, lum.shape)
    span = np.maximum(ground - canopy, min_span)
    return np.clip((canopy + span - lum) / span, 0, 1)
