"""Existing tile, Orthophotos cleared: colour-correct from the tile's own DDS.

Builds a scratch tile folder with plain (uncorrected) DDS of the Houston block
and no orthophotos for it, then runs Step 3's decision (rebuild_reason) and the
colour phase (O4_Color_Build.run), and compares with the clean JPEG path
(_tilebuild/houston from test_build.py).  Everything on E:.
"""
import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
DATA = os.environ.get("O4_REVIEW_DATA", r"E:\Ortho4XP_Review")

import O4_File_Names as FNAMES  # noqa: E402
FNAMES.Imagery_dir = os.path.join(DATA, "color", "imagery_cleared")
sys.path.append(FNAMES.Provider_dir)
import O4_Imagery_Utils as IMG  # noqa: E402
import O4_Overlay_Pipeline as PIPELINE  # noqa: E402
import O4_Color_Build as CCB  # noqa: E402
import O4_Color_Correction as CC  # noqa: E402

PIPELINE._venv_python = lambda: os.environ.get("O4_CC_WORKER_PYTHON", r"G:\Dev\Ortho4XP\.venv\Scripts\python.exe")
IMG.initialize_extents_dict()
IMG.initialize_color_filters_dict()
IMG.initialize_providers_dict()
IMG.local_combined_providers_dict = {}

key = "houston"
meta = json.load(open(os.path.join(DATA, "color", "set", key, "meta.json")))
x0, y0, prov = meta["x0"], meta["y0"], "BI"
lat, lon = int(np.floor(meta["lat"])), int(np.floor(meta["lon"]))
build = os.path.join(DATA, "color", "_tilebuild", key + "_fromdds")
shutil.rmtree(build, ignore_errors=True)
os.makedirs(os.path.join(build, "textures"))
# Reference already downloaded once: reuse it so only the colour path is tested.
src_ref = FNAMES.jpeg_file_dir_from_attributes(lat, lon, 14, IMG.providers_dict["EOX"]).replace(
    "imagery_cleared", "imagery")
dst_ref = FNAMES.jpeg_file_dir_from_attributes(lat, lon, 14, IMG.providers_dict["EOX"])
if not os.path.isdir(dst_ref):
    shutil.copytree(src_ref, dst_ref)
nv = IMG.dds_convert_cmd
tile = SimpleNamespace(lat=lat, lon=lon, build_dir=build, imprint_masks_to_dds=False,
                       color_correction=True, color_fill_clouds=True)
attrs = []
for j in range(3):
    for i in range(3):
        tx, ty = x0 + 16 * i, y0 + 16 * j
        dds = os.path.join(build, "textures", FNAMES.dds_file_name_from_attributes(tx, ty, 16, prov))
        subprocess.call([nv, "-bc1", "-fast", os.path.join(DATA, "color", "set", key, prov, f"{ty}_{tx}.jpg"), dds],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        attrs.append((tx, ty, 16, prov))
reasons = {CCB.rebuild_reason(tile, a, os.path.join(build, "textures",
                                                    FNAMES.dds_file_name_from_attributes(*a))) for a in attrs}
print("rebuild reasons:", reasons)
print("corrected from existing DDS:", sum(CCB.uses_existing_dds(tile, a) for a in attrs), "/", len(attrs))
done = CCB.run(tile, attrs)
print("converted:", len(done), "/", len(attrs))
d = []
for tx, ty, zl, pv in attrs:
    name = FNAMES.dds_file_name_from_attributes(tx, ty, zl, pv)
    a = np.asarray(Image.open(os.path.join(build, "textures", name)).convert("RGB")).astype(np.float32)
    b = np.asarray(Image.open(os.path.join(DATA, "color", "_tilebuild", key, "textures", name)).convert("RGB")).astype(np.float32)
    d.append(np.abs(cv2.GaussianBlur(a, (0, 0), 8) - cv2.GaussianBlur(b, (0, 0), 8)).max(-1).ravel()[::64])
    assert CCB.dds_mark(os.path.join(build, "textures", name)) is not None, "missing stamp"
d = np.concatenate(d)
print(f"vs clean JPEG path (blurred colour): mean {d.mean():.2f}  p99 {np.percentile(d, 99):.1f}  "
      f">10 levels {np.mean(d > 10):.2%};  all stamped")
again = {CCB.rebuild_reason(tile, a, os.path.join(build, "textures", FNAMES.dds_file_name_from_attributes(*a)))
         for a in attrs}
print("second build would redo:", again)
