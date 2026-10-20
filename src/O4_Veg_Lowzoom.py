"""Option G for ZL16/17: RGB option F within SegFormer forest/rangeland.

Uses native RGB and the existing land-cover map, with no extra inference or
downloads. Thresholds match the reviewed low-zoom experiment. They are image
heuristics, not calibrated tree probabilities.
"""
import math

import cv2
import numpy as np

# Bump when the algorithm changes so polygon caches cannot replay older masks.
SIGNATURE = "segformer-forest-rangeland-rgb-f-v1"


def applies(zl):
    return zl in (16, 17)


def metres_per_pixel(lat_n, lat_s, lon_w, lon_e, height, width):
    """Use the same ground scale as the reviewed RGB experiment."""
    radius = 6378137.0
    y = math.radians(lat_n - lat_s) * radius / height
    x = (math.radians(lon_e - lon_w) * radius
         * math.cos(math.radians((lat_n + lat_s) / 2)) / width)
    return (x + y) / 2


def _mean(a, metres, mpp):
    size = max(3, int(round(metres / mpp)) | 1)
    return cv2.blur(a.astype(np.float32), (size, size), borderType=cv2.BORDER_REFLECT)


def _remove_small(mask, mpp):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] * mpp * mpp >= 60
    keep[0] = False
    return keep[labels]


def _coherence(gx, gy, metres, mpp):
    xx, yy, xy = [_mean(v, metres, mpp) for v in (gx * gx, gy * gy, gx * gy)]
    return np.sqrt((xx - yy) ** 2 + 4 * xy * xy) / (xx + yy + 1e-7)


def tree_mask(rgb, classes, mpp, *, forest_class=5, rangeland_class=2):
    """Return native uint8 0/1 G mask; do not expand it with morphology."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected HxWx3 uint8 RGB")
    if classes.shape != rgb.shape[:2]:
        raise ValueError("RGB and SegFormer must use the same native image grid")
    if not math.isfinite(mpp) or mpp <= 0:
        raise ValueError("metres/pixel must be positive and finite")
    zones = (classes == forest_class) | (classes == rangeland_class)
    if not zones.any():
        return np.zeros(classes.shape, np.uint8)

    f = rgb.astype(np.float32) / 255
    r, g, b = f.transpose(2, 0, 1)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    exg = (2 * g - r - b) / np.maximum(r + g + b, 0.06)
    green = (exg > 0.065) & (g > r * 1.025) & (g > b * 1.08) & (lum > 0.055)
    green_share = _mean(green, 12, mpp)
    smooth = cv2.GaussianBlur(lum, (0, 0), max(0.45, 0.7 / mpp))
    local_mean = _mean(smooth, 18, mpp)
    local_std = np.sqrt(np.maximum(
        0, _mean(smooth * smooth, 18, mpp) - local_mean * local_mean))
    contrast = local_std / np.maximum(local_mean, 0.08)
    coarse = cv2.GaussianBlur(smooth, (0, 0), max(0.7, 3.0 / mpp))
    fine_energy = np.sqrt(np.maximum(0, _mean((smooth - coarse) ** 2, 18, mpp)))
    detail = fine_energy / np.maximum(local_std, 0.004)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    eligible = green & (green_share > 0.55) & (local_mean < 0.50)
    textured = (eligible & (local_std > 0.012) & (contrast > 0.065)
                & (detail > 0.28) & (_coherence(gx, gy, 22, mpp) < 0.80))
    texture = _remove_small(textured, mpp)

    # Preserve the experiment's separate F pass, including its float32 luma.
    lum = f @ np.array([0.299, 0.587, 0.114], np.float32)
    smooth = cv2.GaussianBlur(lum, (0, 0), max(0.45, 0.7 / mpp))
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    aligned = np.zeros(classes.shape, bool)
    for metres in (50, 80):
        aligned |= _coherence(gx, gy, metres, mpp) > 0.55
    fill = _remove_small(texture & ~aligned, mpp)
    return (fill & zones).astype(np.uint8)
