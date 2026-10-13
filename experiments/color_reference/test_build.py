"""End-to-end test of the Step 3 colour path (O4_Color_Build.run -> .venv worker -> DDS).

Copies the colour test set into a scratch Orthophotos tree on E:, builds into
a scratch tile folder on E:, and compares the resulting DDS with the reviewed
GPU runs (runs/v13gpu, runs/v14gpufill).  Never touches real tiles.

Usage: python test_build.py key [key ...]        (repo .venv; the worker uses
       O4_CC_WORKER_PYTHON, default G:/Dev/Ortho4XP/.venv/Scripts/python.exe)
"""
import json
import os
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
DATA = os.environ.get("O4_REVIEW_DATA", r"E:\Ortho4XP_Review")

import O4_File_Names as FNAMES  # noqa: E402
FNAMES.Imagery_dir = os.path.join(DATA, "color", "imagery")
sys.path.append(FNAMES.Provider_dir)
import O4_Imagery_Utils as IMG  # noqa: E402
import O4_Overlay_Pipeline as PIPELINE  # noqa: E402
import O4_Color_Build as CCB  # noqa: E402

worker = os.environ.get("O4_CC_WORKER_PYTHON", r"G:\Dev\Ortho4XP\.venv\Scripts\python.exe")
PIPELINE._venv_python = lambda: worker
IMG.initialize_extents_dict()
IMG.initialize_color_filters_dict()
IMG.initialize_providers_dict()
IMG.local_combined_providers_dict = {}

SET = os.path.join(DATA, "color", "set")
RUNS = os.path.join(DATA, "color", "runs")

for key in sys.argv[1:]:
    meta = json.load(open(os.path.join(SET, key, "meta.json")))
    x0, y0, n, provs = meta["x0"], meta["y0"], meta["n"], meta["providers"]
    lat, lon = int(np.floor(meta["lat"])), int(np.floor(meta["lon"]))
    build = os.path.join(DATA, "color", "_tilebuild", key)
    shutil.rmtree(build, ignore_errors=True)
    os.makedirs(os.path.join(build, "textures"))
    tile = SimpleNamespace(lat=lat, lon=lon, build_dir=build, imprint_masks_to_dds=False,
                           color_correction=True, color_fill_clouds=True, mask_zl=14)
    tex = [(x0 + 16 * i, y0 + 16 * j) for j in range(n) for i in range(n)]
    for prov in provs:                      # stage the imagery like Ortho4XP would have it
        d = FNAMES.jpeg_file_dir_from_attributes(lat, lon, 16, IMG.providers_dict[prov])
        os.makedirs(d, exist_ok=True)
        for tx, ty in tex:
            dst = os.path.join(d, FNAMES.jpeg_file_name_from_attributes(tx, ty, 16, prov))
            if not os.path.isfile(dst):
                shutil.copy(os.path.join(SET, key, prov, f"{ty}_{tx}.jpg"), dst)
    attrs = [(tx, ty, 16, provs[0]) for tx, ty in tex]
    t0 = time.time()
    done = CCB.run(tile, attrs)
    print(f"{key}: {len(done)}/{len(attrs)} converted in {time.time() - t0:.0f}s", flush=True)
    ref_run = "v14gpufill" if key in ("colombia", "bioko") else "v13gpu"
    diffs = []
    for tx, ty in tex:
        dds = os.path.join(build, "textures", FNAMES.dds_file_name_from_attributes(tx, ty, 16, provs[0]))
        if not os.path.isfile(dds):
            print("  missing", dds)
            continue
        a = np.asarray(Image.open(dds).convert("RGB")).astype(int)
        b = np.asarray(Image.open(os.path.join(RUNS, ref_run, key, provs[0], f"{ty}_{tx}.jpg"))
                       .convert("RGB")).astype(int)
        diffs.append(np.abs(a - b).max(-1).ravel()[::16])
    if diffs:
        d = np.concatenate(diffs)
        print(f"  DDS vs {ref_run}: mean {d.mean():.2f}  p99 {np.percentile(d, 99):.0f}  "
              f">20 levels {np.mean(d > 20):.3%}")
