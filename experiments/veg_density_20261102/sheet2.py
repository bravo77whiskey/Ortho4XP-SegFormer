"""Texture, mask, cover estimates and a simulated planting, for looking at.

Cover = estimate averaged over the mask with a Gaussian of SIGMA_M. Planting: X-Plane's
grid (12 m), each cell kept with probability density/255, drawn at 8 m crown size.
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

import density_lib as D
import eval_in_mask_lib as L
import O4_Veg_Lowzoom as LOWZOOM

Image.MAX_IMAGE_PIXELS = None
BIOMES = r"E:\Ortho4XP_Review\veg_density\biomes"
OUT = r"E:\Ortho4XP_Review\veg_density\sheets"
SIGMA_M = 32.0


def mpp_of(lat, zl):
    return 40075016.7 * np.cos(np.radians(lat)) / 2 ** zl / 256


def smooth_in_mask(a, mask, mpp, sigma_m=SIGMA_M):
    s = max(0.5, sigma_m / mpp)
    num = cv2.GaussianBlur(a * mask, (0, 0), s)
    den = cv2.GaussianBlur(mask, (0, 0), s)
    return num / np.maximum(den, 1e-3)


def heat(a, mask=None):
    out = cv2.cvtColor(cv2.applyColorMap((np.clip(a, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
    if mask is not None:
        out[mask <= 0] = (25, 25, 25)
    return out


def plant(rgb, mask, density, mpp, seed=1, spacing_m=12.0, crown_m=8.0):
    """rgb with trees drawn where the mask is, density in 0..255 (scalar or map)."""
    h, w = mask.shape
    out = rgb.copy()
    rng = np.random.default_rng(seed)
    step = spacing_m / mpp
    r = max(1, int(round(crown_m / 2 / mpp)))
    dmap = density if isinstance(density, np.ndarray) else None
    count = 0
    for y in np.arange(0, h, step):
        for x in np.arange(0, w, step):
            px = int(x + rng.uniform(-0.5, 0.5) * step)
            py = int(y + rng.uniform(-0.5, 0.5) * step)
            if not (0 <= px < w and 0 <= py < h) or mask[py, px] <= 0:
                continue
            d = dmap[py, px] if dmap is not None else density
            if rng.uniform() * 255 < d:
                cv2.circle(out, (px, py), r, (255, 60, 200), 1, cv2.LINE_AA)
                count += 1
    return out, count


def panel(items, path, size=620):
    n = len(items)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * size, rows * (size + 18)), (18, 18, 18))
    d = ImageDraw.Draw(sheet)
    for i, (label, im) in enumerate(items):
        im = Image.fromarray(im).resize((size, size), Image.LANCZOS)
        x, y = (i % cols) * size, (i // cols) * (size + 18)
        sheet.paste(im, (x, y + 18))
        d.text((x + 4, y + 3), label, fill=(240, 240, 180))
    sheet.save(path, quality=90)


def run(path, lat, zl, name, crop=None):
    rgb = np.array(Image.open(path).convert("RGB"))
    mpp = mpp_of(lat, zl)
    classes, ptree = L.seg_probs(rgb)
    mask = LOWZOOM.tree_mask(rgb, classes, mpp).astype(np.float32)
    gap = L.gap_family(rgb, mask, mpp)["gap_600m_k1.25"]
    tcd = D.tcd_prob(rgb, "tcd_b2", 2.0)
    c_seg = smooth_in_mask(ptree, mask, mpp)
    c_gap = smooth_in_mask(gap, mask, mpp)
    c_tcd = smooth_in_mask(tcd, mask, mpp)
    H = rgb.shape[0]
    if crop is None:
        crop = (H // 4, H // 4, H // 2)
    y, x, s = crop
    v = lambda a: a[y:y + s, x:x + s]
    today, n0 = plant(v(rgb), v(mask), 230, mpp)
    new, n1 = plant(v(rgb), v(mask), v(np.clip(c_seg, 0, 1) * 230), mpp)
    items = [
        (f"{name} z{zl} mpp={mpp:.2f}", v(rgb)),
        (f"production mask share={mask.mean():.2f}", heat(v(mask))),
        ("SegFormer P(tree) in mask", heat(v(ptree), v(mask))),
        ("cover: smoothed P(tree)", heat(v(c_seg), v(mask))),
        ("cover: canopy-relative gap", heat(v(c_gap), v(mask))),
        ("cover: TCD b2 x2", heat(v(c_tcd), v(mask))),
        (f"planted at flat 230: {n0} trees", today),
        (f"planted from P(tree): {n1} trees", new),
    ]
    panel(items, os.path.join(OUT, f"b_{name}_z{zl}.jpg"))
    on = mask > 0
    print(f"{name} z{zl}: mask {on.mean():.2f}, mean in mask seg {c_seg[on].mean() if on.any() else 0:.2f} gap {c_gap[on].mean() if on.any() else 0:.2f} tcd {c_tcd[on].mean() if on.any() else 0:.2f}; trees {n0}->{n1}", flush=True)


if __name__ == "__main__":
    want = set(sys.argv[1:])
    rows = json.load(open(os.path.join(BIOMES, "samples.json")))
    for row in rows:
        if want and row["key"] not in want:
            continue
        run(os.path.join(BIOMES, row["file"]), row["lat"], row["zl"], row["key"])
