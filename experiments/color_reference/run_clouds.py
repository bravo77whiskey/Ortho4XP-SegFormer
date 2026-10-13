"""Colour correction with cloud handling on the cloudy blocks.

Usage: python run_clouds.py RUN_PREFIX key [key ...]

Per block (primary provider = first in meta, alternate = second):
  1. fit with water excluded, detect cloud against the reference, refit with
     cloud excluded too, detect again with the cleaner fit;
  2. correct both providers (same constant look);
  3. <prefix>fill   : primary, cloud filled from the alternate where it is clearer
     <prefix>dehaze : primary only, thin cloud inverted with the haze model
Also writes alpha previews (<run>/<key>/<prov>_alpha.png) for inspection.
"""
import json
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refnorm as RN  # noqa: E402

PREFIX = sys.argv[1]
KEYS = sys.argv[2:]
RUNS = os.path.join(RN.DATA, "color", "runs")
NEAR_M = 400.0     # Gaussian reach of the near-cloud blend to the TV fit
FILL_GROW = 4      # medium px (~30 m) the fill reaches beyond detected cloud
FILL_FULL = 0.2    # cloud alpha at which the fill is complete
look = json.load(open(os.path.join(RUNS, "v1", "look.json")))["look"]


class Wide(RN.P):
    """Plain wide fit used only to *predict clear sky* for cloud detection.

    The edge-aware fit adapts to a hazy area as if it were ground, so thin
    cloud measured against it looks clear; a wide fit does not.
    """
    edge_aware = False
    red_cap = 0.0
    win_L_m = 1000.0
    win_ab_m = 1500.0


class AltTV(RN.P):
    """Correction for the fill-source provider when ALT_TV=1: the TV fit
    ignores haze in the reference, which matters most exactly where we fill."""
    mode = "tv"
    tv_lambda = 12.0


def fit_with_clouds(block, p=RN.P):
    O, R, valid = block.coarse()
    water = RN.water_mask(R, block.rdet, block.odet, valid)
    Om, Rm, vm = RN.medium_mosaic(block)
    cand = RN.coarse_from_medium(RN.cloud_candidates(Om, Rm, vm), block) > 0.15
    cloud_c = cand
    for _ in range(2):
        wg, wo, ws, wt, _ = RN.fit_fields(O, R, valid, block.px_m, p=Wide, exclude=water | cloud_c)
        alpha, rc = RN.cloud_alpha(Om, Rm, vm, (wg, wo, ws, wt))
        cloud_c = cand | (RN.coarse_from_medium(alpha, block) > RN.CP.coarse_alpha) | \
                  (RN.coarse_from_medium(rc, block) > 0.3)
    # The correction itself: the edge-aware fit, with cloud and water excluded.
    g, o, s, t, w = RN.fit_fields(O, R, valid, block.px_m, p=p, exclude=water | cloud_c)
    return (g, o, s, t), alpha, water | cloud_c


for key in KEYS:
    meta = json.load(open(os.path.join(RN.SET, key, "meta.json")))
    prim, alt = meta["providers"][0], meta["providers"][1]
    t0 = time.time()
    res = {}
    for pv in (prim, alt):
        b = RN.Block(key, pv)
        alt_tv = os.environ.get("ALT_TV") == "1" and pv == alt
        fields, alpha, excl = fit_with_clouds(b, AltTV if alt_tv else RN.P)
        if os.environ.get("NEAR_CLOUD_TV") == "1" and pv == prim:
            # Around clouds the edge-aware fit copies whatever the reference
            # has under the haze; the TV fit does not.  Patch fixing matters
            # least there (it gets filled anyway), so blend to TV near cloud.
            O, R, valid = b.coarse()
            tv = RN.fit_fields(O, R, valid, b.px_m, p=AltTV, exclude=excl)[:4]
            cl = RN.coarse_from_medium(alpha, b)
            near = np.clip(RN.gblur((cl > 0.05).astype(np.float32), NEAR_M / b.px_m) * 4.0, 0, 1)
            fields = tuple(near[..., None] * a + (1 - near[..., None]) * c if c.ndim == 3
                           else near * a + (1 - near) * c for a, c in zip(tv, fields))
        res[pv] = (b, RN.compose_look(fields, look), alpha)
        m = b.margin * RN.COARSE // RN.MED
        n = RN.TEX // RN.MED * b.n
        for run in (PREFIX + "fill", PREFIX + "dehaze"):
            os.makedirs(os.path.join(RUNS, run, key), exist_ok=True)
        Image.fromarray((alpha[m:m + n, m:m + n] * 255).astype(np.uint8)).save(
            os.path.join(RUNS, PREFIX + "fill", key, f"{pv}_alpha.png"))
        print(f"{key} {pv}: cloud fraction {alpha[m:m + n, m:m + n].mean():.1%}  "
              f"excluded coarse {excl.mean():.1%}", flush=True)

    bP, fP, aP = res[prim]
    bA, fA, aA = res[alt]
    # Fill weight: cloud edges are thinner than the detector can see, and
    # filling clear ground from the colour-matched alternate is harmless, so
    # grow the primary's cloud a little and saturate it early; never fill from
    # where the alternate is itself cloudy.
    grow = np.ones((2 * FILL_GROW + 1,) * 2, np.uint8)
    aPf = np.clip(RN.gblur(cv2.dilate(aP, grow), 1.5) / FILL_FULL, 0, 1)
    aAf = np.clip(RN.gblur(cv2.dilate(aA, grow), 1.5) / FILL_FULL, 0, 1)
    wfill_m = aPf * (1 - aAf)
    for run in (PREFIX + "fill", PREFIX + "dehaze"):
        for pv in (prim, alt):
            os.makedirs(os.path.join(RUNS, run, key, pv), exist_ok=True)
    for ij in bP.paths:
        name = os.path.basename(bP.paths[ij])
        rgbP, rgbA = bP.texture(ij), bA.texture(ij)
        missP = RN.missing_mask(rgbP)
        alP = np.maximum(RN.texture_medium_field(aP, bP, ij), missP)
        wfill = np.maximum(RN.texture_medium_field(wfill_m, bP, ij), missP)
        corP = RN.correct_texture(rgbP, fP, bP, ij)
        corA = RN.correct_texture(rgbA, fA, bA, ij)
        # Fill only where the alternate is clearer, and never from its white fill.
        wfill = np.clip(wfill, 0, 1) * ~RN.missing_mask(rgbA)
        fill = (corP * (1 - wfill[..., None]) + corA * wfill[..., None] + 0.5).astype(np.uint8)
        Image.fromarray(fill).save(os.path.join(RUNS, PREFIX + "fill", key, prim, name), quality=92)
        Image.fromarray(corA).save(os.path.join(RUNS, PREFIX + "fill", key, alt, name), quality=92)
        dh = RN.correct_texture(RN.dehaze(rgbP, alP), fP, bP, ij)
        Image.fromarray(dh).save(os.path.join(RUNS, PREFIX + "dehaze", key, prim, name), quality=92)
        Image.fromarray(corA).save(os.path.join(RUNS, PREFIX + "dehaze", key, alt, name), quality=92)
    print(f"{key}: {time.time() - t0:.0f}s", flush=True)
