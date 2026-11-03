"""Hybrid cover per biome: texture, P(tree), unmix, cover, then flat vs inferred planting."""
import json
import os
import sys

import numpy as np
from PIL import Image

import eval_in_mask_lib as L
import O4_Veg_Lowzoom as LOWZOOM
import sheet2 as S
import unmix as U

Image.MAX_IMAGE_PIXELS = None
CACHE = os.path.join(S.BIOMES, "seg_cache")
os.makedirs(CACHE, exist_ok=True)
ALPHA = 0.5
FLOOR = 0.12
CEILING = 230


def seg_cached(path, rgb, mpp):
    out = os.path.join(CACHE, os.path.basename(path) + ".npz")
    if os.path.exists(out):
        z = np.load(out)
        return z["ptree"].astype(np.float32), z["mask"].astype(np.float32)
    classes, ptree = L.seg_probs(rgb)
    mask = LOWZOOM.tree_mask(rgb, classes, mpp)
    np.savez_compressed(out, ptree=ptree.astype(np.float16), mask=mask.astype(np.uint8))
    return ptree.astype(np.float32), mask.astype(np.float32)


def cover_of(rgb, ptree, mask, mpp):
    u = U.unmix(rgb, mask, mpp, 600, 30, 50)
    return np.clip(S.smooth_in_mask(ptree + (1 - ptree) * ALPHA * u, mask, mpp), FLOOR, 1), u


def run(path, lat, zl, name):
    rgb = np.array(Image.open(path).convert("RGB"))
    mpp = S.mpp_of(lat, zl)
    ptree, mask = seg_cached(path, rgb, mpp)
    cover, u = cover_of(rgb, ptree, mask, mpp)
    H = rgb.shape[0]
    y, x, s = H // 4, H // 4, H // 2
    v = lambda a: a[y:y + s, x:x + s]
    flat, n0 = S.plant(v(rgb), v(mask), CEILING, mpp)
    new, n1 = S.plant(v(rgb), v(mask), v(cover * CEILING), mpp)
    S.panel([
        (f"{name} z{zl} mpp={mpp:.2f}", v(rgb)),
        ("P(tree)", S.heat(v(ptree), v(mask))),
        ("unmix", S.heat(v(u), v(mask))),
        ("cover = P + (1-P)*0.5*unmix, smoothed", S.heat(v(cover), v(mask))),
        (f"flat {CEILING}: {n0} trees", flat),
        (f"inferred: {n1} trees", new),
    ], os.path.join(S.OUT, f"h_{name}_z{zl}.jpg"), size=660)
    on = mask > 0
    print(f"{name} z{zl}: mask {on.mean():.2f} P {ptree[on].mean() if on.any() else 0:.2f} unmix {u[on].mean() if on.any() else 0:.2f} cover {cover[on].mean() if on.any() else 0:.2f} trees {n0}->{n1}", flush=True)


if __name__ == "__main__":
    want = set(sys.argv[1:])
    for row in json.load(open(os.path.join(S.BIOMES, "samples.json"))):
        if not want or row["key"] in want:
            run(os.path.join(S.BIOMES, row["file"]), row["lat"], row["zl"], row["key"])
