"""Solid-mask thresholds: closed forest the model misses vs the places that must stay thin."""
import glob
import json
import os

import numpy as np
from PIL import Image

import eval_interior as I
import O4_Veg_Density as DENSITY
import sheet2 as S
import sheet3 as H

SETTINGS = [(0.8, 0.95), (0.7, 0.9), (0.65, 0.85), (0.6, 0.8)]
# szx_e region in texture pixels: the south-west of the dark forest block.
BLOCK = (1300, 2100, 700, 1500)


def samples():
    for key, lat in I.SAMPLES.items():
        head = open(f"{I.RUNS}/{key}.js", encoding="utf-8").read(400)
        name = head.split('"file":"')[1].split('"')[0]
        zl = int(head.split('"zl":')[1].split(",")[0])
        yield key, glob.glob(f"{I.SETS}/**/{name}", recursive=True)[0], S.mpp_of(lat, zl)
    for row in json.load(open(os.path.join(S.BIOMES, "samples.json"))):
        if row["key"] in I.BIOMES and row["zl"] == 16:
            yield row["key"], os.path.join(S.BIOMES, row["file"]), S.mpp_of(row["lat"], row["zl"])


for key, path, mpp in samples():
    rgb = np.array(Image.open(path).convert("RGB"))
    p, mask = H.seg_cached(path, rgb, mpp)
    on = mask > 0
    prob = (p * 255).astype(np.uint8)
    line = f"{key:14s} mask {on.mean():.2f}"
    for lo, hi in SETTINGS:
        DENSITY.SOLID_LOW, DENSITY.SOLID_HIGH = lo, hi
        c = DENSITY.cover_map(rgb, prob, mask.astype(np.uint8), mpp)
        line += f" | {lo}-{hi}: {c[on].mean():.2f} top {(c[on] >= 0.85).mean():.2f}"
        if key == "szx_e":
            y0, y1, x0, x1 = BLOCK
            b = mask[y0:y1, x0:x1] > 0
            sol = DENSITY.solidity(mask.astype(np.uint8), mpp)[y0:y1, x0:x1][b]
            print(f"  block {lo}-{hi}: cover p10/50/90 {np.percentile(c[y0:y1, x0:x1][b], [10, 50, 90]).round(2)} solid p10/50/90 {np.percentile(sol, [10, 50, 90]).round(2)}")
    print(line, flush=True)
