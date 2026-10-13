"""Run src/O4_Color_Correction.py on the colour test set like a tile build would.

The reference comes through Ortho4XP's EOX provider (build_jpeg_ortho) into a
scratch imagery folder on E:, the textures from the test set.  Writes
runs/<RUN>/<key>/<prov>/ (for the viewer) and reports timings.

Usage: python test_module.py RUN key [key ...]     (CUDA venv)
"""
import json
import os
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
import O4_Color_Correction as CC  # noqa: E402

IMG.initialize_extents_dict()
IMG.initialize_color_filters_dict()
IMG.initialize_providers_dict()

RUN = sys.argv[1]
SET = os.path.join(DATA, "color", "set")
RUNS = os.path.join(DATA, "color", "runs")

for key in sys.argv[2:]:
    meta = json.load(open(os.path.join(SET, key, "meta.json")))
    provs = meta["providers"]
    x0, y0, n = meta["x0"], meta["y0"], meta["n"]
    tile = SimpleNamespace(lat=int(np.floor(meta["lat"])), lon=int(np.floor(meta["lon"])),
                           build_dir=os.path.join(DATA, "color", "_tilebuild"),
                           color_correction=True, color_fill_clouds=True)
    path = lambda prov, tx, ty: os.path.join(SET, key, prov, f"{ty}_{tx}.jpg")

    def loader(prov, tx, ty):
        p = path(prov, tx, ty)
        return np.asarray(Image.open(p).convert("RGB")) if os.path.isfile(p) else None

    def fetch_reference(tx, ty):
        IMG.build_jpeg_ortho(tile, tx, ty, CC.REF_ZL, CC.REF_PROVIDER)
        d = FNAMES.jpeg_file_dir_from_attributes(tile.lat, tile.lon, CC.REF_ZL,
                                                  IMG.providers_dict[CC.REF_PROVIDER])
        p = os.path.join(d, FNAMES.jpeg_file_name_from_attributes(tx, ty, CC.REF_ZL, CC.REF_PROVIDER))
        return np.asarray(Image.open(p).convert("RGB")) if os.path.isfile(p) else None

    def fetch_fill(prov, tx, ty, zl):
        return os.path.isfile(path(prov, tx, ty))
    fetch_fill.providers = provs

    prim = provs[0]
    tex = [(x0 + 16 * i, y0 + 16 * j, 16, prim) for j in range(n) for i in range(n)]
    t0 = time.perf_counter()
    CC.prepare(tile, tex, loader, fetch_reference, fetch_fill)
    t1 = time.perf_counter()
    for tx, ty, zl, prov in tex:
        out = CC.corrected_texture(tile, tx, ty, zl, prov, loader)
        d = os.path.join(RUNS, RUN, key, prov)
        os.makedirs(d, exist_ok=True)
        Image.fromarray(out).save(os.path.join(d, f"{ty}_{tx}.jpg"), quality=92)
    t2 = time.perf_counter()
    plan = CC._plans[CC._tile_key(tile)]
    filled = sum(len(v) for v in plan.fill_tex.values())
    print(f"{key} {prim}: prepare {t1 - t0:.1f}s (incl. reference fetch), apply {t2 - t1:.1f}s "
          f"for {len(tex)} textures, {filled} filled from {provs[1]}", flush=True)
    CC.finish(tile)
