"""Image-only canopy closure: uniform canopy is closed, crowns over bright ground are open."""
import cv2
import numpy as np


def _blur(a, size):
    return cv2.blur(a.astype(np.float32), (size, size), borderType=cv2.BORDER_REFLECT)


def lum_of(rgb, mpp):
    f = rgb.astype(np.float32) / 255
    lum = f @ np.array([0.299, 0.587, 0.114], np.float32)
    return cv2.GaussianBlur(lum, (0, 0), max(0.4, 0.5 / mpp))


def local_quantiles(lum, mask, cell_px, qs=(10, 90)):
    """Percentiles of lum on a coarse grid (all pixels: the ground matters), bilinear."""
    h, w = lum.shape
    ny, nx = max(1, h // cell_px), max(1, w // cell_px)
    out = [np.zeros((ny, nx), np.float32) for _ in qs]
    for j in range(ny):
        for i in range(nx):
            v = lum[j * h // ny:(j + 1) * h // ny, i * w // nx:(i + 1) * w // nx].ravel()[::3]
            for k, value in enumerate(np.percentile(v, qs)):
                out[k][j, i] = value
    return [cv2.resize(cv2.GaussianBlur(o, (0, 0), 0.7), (w, h), interpolation=cv2.INTER_LINEAR) for o in out]


def closure(rgb, mask, mpp, window_m=128.0, c0=0.18, c1=0.40, mid=0.5, ground_lum=0.0):
    """Closure 0..1 per pixel (before averaging over the mask).

    Over ``window_m``: contrast = (p90 - p10) / (p90 + p10). A low contrast
    means one surface, which inside the tree mask is closed canopy. With a high
    contrast the dark mode is crowns and the bright mode ground, so a pixel
    counts as crown if darker than the point ``mid`` of the way from p10 to p90.
    """
    lum = lum_of(rgb, mpp)
    cell = max(8, int(round(window_m / mpp)))
    p10, p90 = local_quantiles(lum, mask, cell)
    contrast = (p90 - p10) / np.maximum(p90 + p10, 0.02)
    weight = np.clip((contrast - c0) / max(c1 - c0, 1e-3), 0, 1)
    crown = (lum < p10 + mid * (p90 - p10)).astype(np.float32)
    return crown * weight + (1 - weight), contrast
