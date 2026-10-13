"""Run the prototype on the colour set; write corrected textures and previews.

Usage: python run_refnorm.py RUN [key ...]
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

RUN = sys.argv[1] if len(sys.argv) > 1 else "v1"
OUT = os.path.join(RN.DATA, "color", "runs", RUN)
os.makedirs(OUT, exist_ok=True)
ALL = sorted(k for k in os.listdir(RN.SET) if os.path.isdir(os.path.join(RN.SET, k)))
KEYS = sys.argv[2:] or ALL


def providers(key):
    meta = json.load(open(os.path.join(RN.SET, key, "meta.json")))
    return meta.get("providers", ["BI", "GO2"])


def pooled_look():
    """Constant ref->ortho map making reference space look like the average provider.

    Pooled over every location and provider, so it is one worldwide constant.
    """
    Ls, Lr, Cs, Cr = [], [], [], []
    for k in ALL:
        for pv in providers(k):
            O, R, valid = RN.Block(k, pv).coarse()
            Ls.append(O[valid][:, 0]); Lr.append(R[valid][:, 0])
            Cs.append(O[valid][:, 1:]); Cr.append(R[valid][:, 1:])
    Ls, Lr, Cs, Cr = map(np.concatenate, (Ls, Lr, Cs, Cr))
    G = Ls.std() / Lr.std(); O_ = Ls.mean() - G * Lr.mean()
    S = np.sqrt(Cs.var(0).sum() / Cr.var(0).sum()); T = Cs.mean(0) - S * Cr.mean(0)
    return float(G), float(O_), float(S), [float(x) for x in T]


look_path = os.path.join(OUT, "look.json")
if os.path.isfile(look_path):
    look = json.load(open(look_path))["look"]
else:
    look = pooled_look()
    json.dump({"look": look}, open(look_path, "w"))
print("look", look, flush=True)

PREV = 1024   # preview px per texture
for k in KEYS:
    for pv in providers(k):
        t0 = time.time()
        b = RN.Block(k, pv)
        O, R, valid = b.coarse()
        excl = RN.water_mask(R, b.rdet, b.odet, valid) if RN.P.water else None
        g, o, s, t, w = RN.fit_fields(O, R, valid, b.px_m, exclude=excl)
        fields = RN.compose_look((g, o, s, t), look)
        n = b.n
        before = Image.new("RGB", (PREV * n, PREV * n))
        after = Image.new("RGB", (PREV * n, PREV * n))
        d = os.path.join(OUT, k, pv)
        os.makedirs(d, exist_ok=True)
        for ij in b.paths:
            rgb = b.texture(ij)
            out = RN.correct_texture(rgb, fields, b, ij)
            Image.fromarray(out).save(os.path.join(d, os.path.basename(b.paths[ij])), quality=92)
            pos = (ij[0] * PREV, ij[1] * PREV)
            before.paste(Image.fromarray(rgb).resize((PREV, PREV), Image.LANCZOS), pos)
            after.paste(Image.fromarray(out).resize((PREV, PREV), Image.LANCZOS), pos)
        before.save(os.path.join(OUT, f"{k}_{pv}_before.jpg"), quality=90)
        after.save(os.path.join(OUT, f"{k}_{pv}_after.jpg"), quality=90)
        np.savez_compressed(os.path.join(d, "fields.npz"), g=g, o=o, s=s, t=t, w=w,
                            margin=b.margin, exclude=excl if excl is not None else 0)
        print(f"{k} {pv}: g {np.percentile(g, 2):.2f}-{np.percentile(g, 98):.2f} "
              f"o {np.percentile(o, 2):.0f}..{np.percentile(o, 98):.0f} "
              f"s {np.percentile(s, 2):.2f}-{np.percentile(s, 98):.2f}  {time.time() - t0:.0f}s",
              flush=True)
