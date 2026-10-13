"""Is a provider's low-ZL imagery the same imagery as its ZL16, just smaller?

Downloads the ZL14 (and ZL13) textures covering a 3x3 ZL16 block, reduces both
to the fit's coarse grid (the ZL12 pixel) and compares them.  If they agree,
the colour fit can run on a cheap low-ZL prefetch instead of waiting for every
ZL16 texture.

Usage: python lowzl_check.py key [key ...]
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
import refnorm as RN  # noqa: E402
import O4_File_Names as FNAMES  # noqa: E402
sys.path.append(FNAMES.Provider_dir)
import O4_Imagery_Utils as IMG  # noqa: E402

IMG.initialize_extents_dict()
IMG.initialize_color_filters_dict()
IMG.initialize_providers_dict()
OUT = os.path.join(RN.DATA, "color", "lowzl")


def lab(u8):
    return cv2.cvtColor(u8.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


for key in sys.argv[1:]:
    meta = json.load(open(os.path.join(RN.SET, key, "meta.json")))
    x0, y0, n = meta["x0"], meta["y0"], meta["n"]
    for prov in meta["providers"]:
        b = RN.Block(key, prov)
        # ZL16 block reduced to the ZL12 grid (16x).
        c = RN.TEX // RN.COARSE
        hi = np.zeros((n * c, n * c, 3), np.uint8)
        for (i, j), pth in b.paths.items():
            t = np.asarray(Image.open(pth).convert("RGB"))
            hi[j * c:(j + 1) * c, i * c:(i + 1) * c] = cv2.resize(t, (c, c), interpolation=cv2.INTER_AREA)
        sheet = [hi]
        for zl in (14, 13):
            f = 2 ** (16 - zl)
            # ZL-zl textures covering the block (ortho grid: multiples of 16 tiles).
            tx0, ty0 = (x0 // f) // 16 * 16, (y0 // f) // 16 * 16
            tx1, ty1 = ((x0 + 16 * n - 1) // f) // 16 * 16, ((y0 + 16 * n - 1) // f) // 16 * 16
            d = os.path.join(OUT, key, prov)
            os.makedirs(d, exist_ok=True)
            big = {}
            for ty in range(ty0, ty1 + 1, 16):
                for tx in range(tx0, tx1 + 1, 16):
                    name = f"{ty}_{tx}_{prov}{zl}.jpg"
                    if not os.path.isfile(os.path.join(d, name)):
                        IMG.download_jpeg_ortho(d, name, tx, ty, zl, prov)
                    big[(tx, ty)] = np.asarray(Image.open(os.path.join(d, name)).convert("RGB"))
            # Mosaic, crop the block, reduce to the ZL12 grid.
            W = (tx1 - tx0) // 16 + 1
            Hh = (ty1 - ty0) // 16 + 1
            mos = np.zeros((Hh * 4096, W * 4096, 3), np.uint8)
            for (tx, ty), a in big.items():
                mos[(ty - ty0) // 16 * 4096:(ty - ty0) // 16 * 4096 + 4096,
                    (tx - tx0) // 16 * 4096:(tx - tx0) // 16 * 4096 + 4096] = a
            px = 4096 // 16                      # px per ZL-zl tile
            ox, oy = (x0 - tx0 * f) * px // f, (y0 - ty0 * f) * px // f
            size = 16 * n * px // f
            crop = mos[oy:oy + size, ox:ox + size]
            lo = cv2.resize(crop, (n * c, n * c), interpolation=cv2.INTER_AREA)
            sheet.append(lo)
            d_e = np.linalg.norm(lab(hi) - lab(lo), axis=-1)
            dL = lab(lo)[..., 0] - lab(hi)[..., 0]
            print(f"{key} {prov} ZL{zl} vs ZL16 on the ZL12 grid: dE p50 {np.median(d_e):.1f}  "
                  f"p90 {np.percentile(d_e, 90):.1f}  mean dL {dL.mean():+.1f}", flush=True)
        Image.fromarray(np.concatenate(sheet, 1)).save(os.path.join(OUT, f"{key}_{prov}.jpg"), quality=90)
