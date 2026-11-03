"""Candidate tree-cover estimators from ortho RGB. Experiment code, not production."""
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, r"G:\Dev\Ortho4XP-SegFormer\src")

MODELS = r"E:\Ortho4XP_Review\veg_density\models"
_nets = {}


def _net(name):
    if name not in _nets:
        from transformers import SegformerForSemanticSegmentation
        net = SegformerForSemanticSegmentation.from_pretrained(os.path.join(MODELS, name))
        net = net.half().cuda().eval() if torch.cuda.is_available() else net.eval()
        _nets[name] = net
    return _nets[name]


MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def tcd_prob(rgb, name="tcd_b0", upscale=1.0, tile=1024, stride=768, batch=4):
    """P(tree) per pixel of ``rgb`` from a Restor TCD SegFormer."""
    h, w = rgb.shape[:2]
    if upscale != 1.0:
        rgb = cv2.resize(rgb, (int(w * upscale), int(h * upscale)),
                         interpolation=cv2.INTER_CUBIC if upscale > 1 else cv2.INTER_AREA)
    H, W = rgb.shape[:2]
    x = ((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)
    xs = sorted({*range(0, max(1, W - tile), stride), max(0, W - tile)})
    ys = sorted({*range(0, max(1, H - tile), stride), max(0, H - tile)})
    num = np.zeros((H, W), np.float32)
    den = np.zeros((H, W), np.float32)
    # Feather so that patch edges do not show
    win = np.minimum(np.linspace(0.05, 1, tile // 2), 1)
    win = np.outer(np.r_[win, win[::-1]], np.r_[win, win[::-1]]).astype(np.float32)
    net = _net(name)
    coords = [(y, x0) for y in ys for x0 in xs]
    dtype = next(net.parameters()).dtype
    device = next(net.parameters()).device
    for i in range(0, len(coords), batch):
        part = coords[i:i + batch]
        t = torch.from_numpy(np.stack([x[:, y:y + tile, x0:x0 + tile] for y, x0 in part])).to(device, dtype)
        with torch.inference_mode():
            logits = net(pixel_values=t).logits
            logits = torch.nn.functional.interpolate(logits.float(), size=t.shape[-2:], mode="bilinear", align_corners=False)
            p = torch.softmax(logits, 1)[:, 1].cpu().numpy()
        for (y, x0), q in zip(part, p):
            wy = win[:q.shape[0], :q.shape[1]]
            num[y:y + q.shape[0], x0:x0 + q.shape[1]] += q * wy
            den[y:y + q.shape[0], x0:x0 + q.shape[1]] += wy
    out = num / np.maximum(den, 1e-5)
    if (H, W) != (h, w):
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)
    return out


def _mean(a, metres, mpp):
    size = max(3, int(round(metres / mpp)) | 1)
    return cv2.blur(a.astype(np.float32), (size, size), borderType=cv2.BORDER_REFLECT)


def rgb_mask(rgb, mpp):
    """Production RGB option F with no SegFormer."""
    import O4_Veg_Lowzoom as LOWZOOM
    return LOWZOOM.tree_mask(rgb, np.zeros(rgb.shape[:2], np.uint8), mpp).astype(np.float32)


def green_dark(rgb, mpp, window_m=48.0):
    """Crown pixels: green, and darker than the brighter ground around them.

    The ground reference is the local 80th percentile (approximated by mean +
    0.85 std over a wide window), so hazy or dim images still separate.
    """
    f = rgb.astype(np.float32) / 255
    r, g, b = f.transpose(2, 0, 1)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    lum = cv2.GaussianBlur(lum, (0, 0), max(0.5, 0.6 / mpp))
    m = _mean(lum, window_m, mpp)
    s = np.sqrt(np.maximum(0, _mean(lum * lum, window_m, mpp) - m * m))
    exg = (2 * g - r - b) / np.maximum(r + g + b, 0.06)
    return ((lum < m + 0.15 * s) & (exg > 0.02) & (lum < 0.42)).astype(np.float32)


def shadow_gap(rgb, mpp):
    """Bright or non-green pixels among trees: ground seen between crowns."""
    f = rgb.astype(np.float32) / 255
    r, g, b = f.transpose(2, 0, 1)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    exg = (2 * g - r - b) / np.maximum(r + g + b, 0.06)
    return ((lum > 0.42) | (exg < 0.0)).astype(np.float32)


def rough(rgb, mpp, crown_m=6.0):
    """Crown-scale luminance texture over a wide window, relative to brightness."""
    lum = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    fine = cv2.GaussianBlur(lum, (0, 0), max(0.5, crown_m / 4 / mpp))
    coarse = cv2.GaussianBlur(lum, (0, 0), max(1.0, crown_m / mpp))
    band = fine - coarse
    return np.sqrt(_mean(band * band, 32, mpp)) / np.maximum(_mean(lum, 32, mpp), 0.05)


def cell_means(a, cell_px):
    """Mean over non-overlapping square cells."""
    h, w = a.shape
    n_y, n_x = h // cell_px, w // cell_px
    a = a[:n_y * cell_px, :n_x * cell_px].astype(np.float32)
    return cv2.resize(a, (n_x, n_y), interpolation=cv2.INTER_AREA)


def score(est, truth, keep):
    e, t = est[keep], truth[keep]
    if e.size < 10 or e.std() < 1e-6:
        return float("nan"), float("nan"), int(e.size), float("nan")
    r = float(np.corrcoef(e, t)[0, 1])
    # Best monotone-free linear map, so a scale error is reported apart
    a, b = np.polyfit(e, t, 1)
    mae_fit = float(np.abs(a * e + b - t).mean())
    return r, float(np.abs(e - t).mean()), int(e.size), mae_fit


def timed(fn, *a, **k):
    t0 = time.time()
    out = fn(*a, **k)
    return out, time.time() - t0
