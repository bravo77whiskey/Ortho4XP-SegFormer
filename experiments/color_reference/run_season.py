"""Summer foliage: recolour autumn canopy toward the reference's summer green.

Usage: python run_season.py SRC_RUN DST_RUN key [key ...]

Reuses SRC_RUN's fitted fields (fields.npz, pre-look) and look.json, writes
DST_RUN.  The general correction deliberately does not chase reference detail
(that copies reference artefacts), so autumn foliage keeps its tree-by-tree
red/orange speckle.  This step targets exactly that case:

  * the reference (a summer composite) says green vegetation,
  * the corrected ortho is clearly redder/browner than that,
  * over a forest-sized area (an opening of ~100 m), so a red roof in a green
    suburb never qualifies,
  * and it is not cloud.

There the local mean chroma (~15 m) is moved onto the reference's and the
chroma variation around it is compressed.  Lightness and texture stay the
ortho's - like pan-sharpening in reverse.
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refnorm as RN  # noqa: E402

SRC, DST, KEYS = sys.argv[1], sys.argv[2], sys.argv[3:]
RUNS = os.path.join(RN.DATA, "color", "runs")
look = json.load(open(os.path.join(RUNS, SRC, "look.json")))["look"] \
    if os.path.isfile(os.path.join(RUNS, SRC, "look.json")) else \
    json.load(open(os.path.join(RUNS, "v1", "look.json")))["look"]
os.makedirs(os.path.join(RUNS, DST), exist_ok=True)
json.dump({"look": look}, open(os.path.join(RUNS, DST, "look.json"), "w"))

VEG_A = -3.0       # reference a* below this (green) ...
VEG_C = 6.0        # ... with at least this chroma = summer vegetation
REDDER = 6.0       # a pixel is "red" when its a* is this far above the reference's
RED_FRAC = 0.2     # fraction of red pixels in the window for full weight
WIN_M = 60.0       # Gaussian window for the local statistics
EXCESS0 = 1.3      # ortho/reference chroma spread ratio where the weight starts
EXCESS1 = 0.7      # ... and how much more it takes to reach full weight
OPEN_M = 100.0     # opening size: only forest-scale areas qualify
MEAN_PX = 1.5      # medium px, local-mean scale (~15 m)
COMPRESS = 0.35    # chroma variation kept around the new mean


def season_shift(block, fields):
    """Chroma shift (ref space, medium res) and its weight.

    Autumn canopy is a speckle of red/orange trees among green ones: its local
    *mean* is already pulled onto the reference by the main fit, but its
    chroma *variation* is far larger than the summer reference's and contains
    red.  That signature, over forest-sized areas where the reference shows
    vegetation, is what gets the weight.
    """
    Om, Rm, vm = RN.medium_mosaic(block)
    H = Om.shape[0]
    up = lambda f: cv2.resize(f.astype(np.float32), (H, H), interpolation=cv2.INTER_LINEAR)
    g, o, s, t = up(fields["g"]), up(fields["o"]), up(fields["s"]), up(fields["t"])
    C = RN.apply(Om, g, o, s, t)
    win = WIN_M / (block.px_m / (RN.COARSE // RN.MED))
    vegf = RN.gblur(((Rm[..., 1] < VEG_A) & (np.hypot(Rm[..., 1], Rm[..., 2]) > VEG_C)).astype(np.float32), win)

    def cstd(x):
        m = RN.gblur(x, win)
        return np.sqrt(np.maximum(RN.gblur(x * x, win) - m * m, 0).sum(-1))
    excess = np.clip((cstd(C[..., 1:]) / (cstd(Rm[..., 1:]) + 1.0) - EXCESS0) / EXCESS1, 0, 1)
    redf = RN.gblur((C[..., 1] > Rm[..., 1] + REDDER).astype(np.float32), win)
    cand, _ = RN.cloud_alpha(Om, Rm, vm, (fields["g"], fields["o"], fields["s"], fields["t"]))
    w = np.clip((vegf - 0.5) / 0.3, 0, 1) * excess * np.clip(redf / RED_FRAC, 0, 1) * vm * (cand < 0.1)
    k = int(round(OPEN_M / (block.px_m / (RN.COARSE // RN.MED)))) | 1
    keep = cv2.morphologyEx((w > 0.25).astype(np.uint8), cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    keep = cv2.dilate(keep, np.ones((5, 5), np.uint8)).astype(np.float32)
    w = RN.gblur(np.clip(w / 0.6, 0, 1) * keep, 1.5)
    shift = (RN.gblur(Rm[..., 1:], MEAN_PX) - RN.gblur(C[..., 1:], MEAN_PX)) * w[..., None]
    return shift, w


for key in KEYS:
    meta = json.load(open(os.path.join(RN.SET, key, "meta.json")))
    for pv in meta.get("providers", ["BI", "GO2"]):
        src_dir = os.path.join(RUNS, SRC, key, pv)
        if not os.path.isfile(os.path.join(src_dir, "fields.npz")):
            continue
        b = RN.Block(key, pv)
        b.coarse()
        f = np.load(os.path.join(src_dir, "fields.npz"))
        shift, w = season_shift(b, f)
        fields = RN.compose_look((f["g"], f["o"], f["s"], f["t"]), look)
        S = look[2]
        d = os.path.join(RUNS, DST, key, pv)
        os.makedirs(d, exist_ok=True)
        for ij in b.paths:
            rgb = b.texture(ij)
            g, o, s, t = (RN.texture_field(x, b, ij) for x in fields)
            lab = RN.apply(RN.srgb_to_lab(rgb), g, o, s, t)
            wf = RN.texture_medium_field(w, b, ij)[..., None]
            sh = np.stack([RN.texture_medium_field(shift[..., c], b, ij) for c in range(2)], -1)
            # Compress variation around the local mean, then move the mean.
            mean = cv2.GaussianBlur(lab[..., 1:], (0, 0), MEAN_PX * RN.MED)
            lab[..., 1:] = lab[..., 1:] + wf * ((COMPRESS - 1) * (lab[..., 1:] - mean)) + S * sh
            out = RN.lab_to_srgb(lab)
            out[RN.missing_mask(rgb)] = 255
            Image.fromarray(out).save(os.path.join(d, os.path.basename(b.paths[ij])), quality=92)
        m = b.margin * RN.COARSE // RN.MED
        n = RN.TEX // RN.MED * b.n
        Image.fromarray((w[m:m + n, m:m + n] * 255).astype(np.uint8)).save(os.path.join(d, "season_w.png"))
        print(f"{key} {pv}: summerised {w[m:m + n, m:m + n].mean():.1%}", flush=True)
