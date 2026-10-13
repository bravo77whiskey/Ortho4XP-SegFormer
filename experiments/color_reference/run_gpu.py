"""Run the GPU pipeline over the colour set, write outputs like the CPU runs, report timing.

Usage: python run_gpu.py RUN [key ...]      (CUDA venv)
  clear blocks -> runs/<RUN>/<key>/<prov>/, cloud blocks -> runs/<RUN>fill/<key>/<prov>/
"""
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refnorm as RN  # noqa: E402
import refnorm_gpu as RG  # noqa: E402
import refnorm_gpu_clouds as RC  # noqa: E402

RUN = sys.argv[1]
ALL = sorted(k for k in os.listdir(RN.SET) if os.path.isfile(os.path.join(RN.SET, k, "meta.json")))
KEYS = sys.argv[2:] or ALL
CLOUDY = {"colombia", "bioko"}
RUNS = os.path.join(RN.DATA, "color", "runs")
look = json.load(open(os.path.join(RUNS, "v1", "look.json")))["look"]


def save(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img.permute(1, 2, 0).cpu().numpy()).save(path, quality=92)


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


# Warm-up (CUDA context and kernels) so the first timing is honest.
RG.run_block("houston", "BI", look)
for key in KEYS:
    meta = json.load(open(os.path.join(RN.SET, key, "meta.json")))
    if key in CLOUDY:
        t0 = sync()
        res = RC.run_cloud_block(key, look)
        t1 = sync()
        for prov, out, alpha, b in res:
            m = b.margin * RN.COARSE // RN.MED
            w = RN.TEX // RN.MED * b.n
            for ij, img in out.items():
                save(img, os.path.join(RUNS, RUN.replace("v13", "v14") + "fill", key, prov, os.path.basename(b.paths[ij])))
            Image.fromarray((RG._np(alpha[m:m + w, m:m + w]) * 255).astype(np.uint8)).save(
                os.path.join(RUNS, RUN.replace("v13", "v14") + "fill", key, f"{prov}_alpha.png"))
        print(f"{key}: cloud block, 2 providers x {len(b.paths)} textures  GPU {t1 - t0:.2f}s", flush=True)
    else:
        for prov in meta.get("providers", ["BI", "GO2"]):
            t0 = sync()
            out, _, b = RG.run_block(key, prov, look)
            t1 = sync()
            for ij, img in out.items():
                save(img, os.path.join(RUNS, RUN, key, prov, os.path.basename(b.paths[ij])))
            print(f"{key} {prov}: {len(out)} textures  GPU {t1 - t0:.2f}s", flush=True)
