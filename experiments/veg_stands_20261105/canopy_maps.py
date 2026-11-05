"""Canopy maps for the hotspots ZL18/19 textures, twice: as the overlay runs the
model, and with the texture shifted by half a window stride.

The two runs read the same imagery through differently placed windows, so
where they disagree is the model's own noise. stands.py measures stand
heights on run A and scores them against run B: stands small enough to fit
that noise score worse, which is what bounds the stand size from below.

Run with the CUDA venv:
  G:/Dev/Ortho4XP/.venv/Scripts/python.exe experiments/veg_stands_20261105/canopy_maps.py
"""
import json
import os
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import O4_Canopy_Height as CANOPY  # noqa: E402

DATA = os.environ.get("O4_REVIEW_DATA", "G:/Ortho4XP_Review")
SET = os.path.join(DATA, "sets", "hotspots", "BI")
OUT = os.path.join(DATA, "_scratch", "veg_stands", "maps")
MODEL_DIR = os.path.join(DATA, "_cache", "w0", "_canopy_height")
SHIFT_INF = CANOPY.STRIDE // 2   # pixels at the model's scale


def main():
    os.makedirs(OUT, exist_ok=True)
    samples = [s for s in json.load(open(os.path.join(SET, "samples.json"))) if s["zl"] >= 18]
    model = CANOPY.CanopyModel.load(MODEL_DIR)
    if model is None:
        sys.exit("no canopy model")
    for s in samples:
        dst = os.path.join(OUT, s["key"] + ".npz")
        if os.path.exists(dst):
            continue
        img = np.asarray(Image.open(os.path.join(SET, s["file"])).convert("RGB"))
        factor = 2 ** (s["zl"] - CANOPY.INFER_ZL)
        t0 = time.time()
        a = model._maps(img, s["zl"])
        o = SHIFT_INF * factor
        shifted = np.pad(img, ((o, 0), (o, 0), (0, 0)), mode="reflect")
        b = model._maps(shifted, s["zl"])[:, SHIFT_INF:, SHIFT_INF:][:, :a.shape[1], :a.shape[2]]
        np.savez_compressed(dst, a=a[0], b=b[0], pa=a[1 + CANOPY.LEVELS_M.index(2.0)])
        print(f"{s['key']}: {a.shape} in {time.time() - t0:.1f}s, "
              f"mean |a-b| {np.abs(a[0].astype(np.float32) - b[0]).mean():.2f} m", flush=True)


if __name__ == "__main__":
    main()
