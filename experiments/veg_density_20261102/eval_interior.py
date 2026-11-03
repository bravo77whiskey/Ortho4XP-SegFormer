"""Closed forest SegFormer misses: let unmixing reach full cover deep inside a solid mask.

szx_e planted a uniform dark forest at half density because P(tree) was low there and
unmixing caps at 0.5. A lawn or an olive grove is not a solid block hundreds of metres
across, so the cap can lift with how solid the mask is around the pixel.
Usage: eval_interior.py lidar | sheets
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

import sheet2 as S
import sheet3 as H
import unmix as U

Image.MAX_IMAGE_PIXELS = None
SETS = "E:/Ortho4XP_Review/sets/hotspots"
RUNS = "E:/Ortho4XP_Review/runs/veg-density-texture/BI"
SAMPLES = {"szx_e": 22.7, "wnz_sw": 27.8, "wnz_z18": 27.8, "shouguang": 36.9, "khh_nw": 22.7, "szx_n": 22.7}
BIOMES = ("brazil_bahia", "olives_jaen", "dehesa_spain", "serengeti", "cerrado", "finland")


def solidity(mask, mpp, radius_m):
    k = max(1, int(radius_m / mpp / 8))
    w, h = mask.shape[1], mask.shape[0]
    small = cv2.resize(mask.astype(np.float32), (max(1, w // k), max(1, h // k)), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (0, 0), radius_m / (mpp * w / small.shape[1]))
    return cv2.resize(blur, (w, h), interpolation=cv2.INTER_LINEAR)


def weight(mask, mpp, radius_m, lo, hi, base=0.5):
    t = np.clip((solidity(mask, mpp, radius_m) - lo) / (hi - lo), 0, 1)
    return base + (1 - base) * t * t * (3 - 2 * t)


def cover(rgb, p, mask, mpp, radius_m=None, lo=0.8, hi=0.95):
    u = U.unmix(rgb, mask, mpp, 600, 30, 50)
    w = 0.5 if radius_m is None else weight(mask, mpp, radius_m, lo, hi)
    return p + (1 - p) * w * u


VARIANTS = [("current", dict())] + [
    (f"solid r{r} {lo}-{hi}", dict(radius_m=r, lo=lo, hi=hi))
    for r in (100, 200, 400) for lo, hi in ((0.7, 0.9), (0.8, 0.95), (0.9, 0.98))
]


def lidar():
    import eval_unmix_data as E
    print("site+scale = r / raw MAE / mean estimate")
    for name, kw in VARIANTS:
        E.report(name, lambda rgb, p, m, mpp, k=kw: cover(rgb, p, m, mpp, **k))


def sheet(name, rgb, p, mask, mpp, kw):
    c0 = np.clip(S.smooth_in_mask(cover(rgb, p, mask, mpp), mask, mpp), 0, 1)
    c1 = np.clip(S.smooth_in_mask(cover(rgb, p, mask, mpp, **kw), mask, mpp), 0, 1)
    S.panel([
        (name, rgb),
        ("P(tree)", S.heat(p, mask)),
        ("cover now", S.heat(c0, mask)),
        ("cover with solid mask", S.heat(c1, mask)),
    ], os.path.join(S.OUT, f"solid_{name}.jpg"), size=800)
    on = mask > 0
    print(f"{name}: mask {on.mean():.2f} P {p[on].mean() if on.any() else 0:.2f} "
          f"cover {c0[on].mean() if on.any() else 0:.2f} -> {c1[on].mean() if on.any() else 0:.2f}", flush=True)


def sheets(kw):
    for key, lat in SAMPLES.items():
        with open(f"{RUNS}/{key}.js", encoding="utf-8") as f:
            head = f.read(400)
        name = head.split('"file":"')[1].split('"')[0]
        zl = int(head.split('"zl":')[1].split(",")[0])
        path = glob.glob(f"{SETS}/**/{name}", recursive=True)[0]
        rgb = np.array(Image.open(path).convert("RGB"))
        mpp = S.mpp_of(lat, zl)
        p, mask = H.seg_cached(path, rgb, mpp)
        sheet(key, rgb, p, mask, mpp, kw)
    for row in json.load(open(os.path.join(S.BIOMES, "samples.json"))):
        if row["key"] in BIOMES and row["zl"] == 16:
            path = os.path.join(S.BIOMES, row["file"])
            rgb = np.array(Image.open(path).convert("RGB"))
            mpp = S.mpp_of(row["lat"], row["zl"])
            p, mask = H.seg_cached(path, rgb, mpp)
            sheet(row["key"], rgb, p, mask, mpp, kw)


if __name__ == "__main__":
    if sys.argv[1] == "lidar":
        lidar()
    else:
        sheets(dict(radius_m=200, lo=0.8, hi=0.95))
