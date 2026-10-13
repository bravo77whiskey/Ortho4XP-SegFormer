"""Per-texture cloud diagnostics of the production worker (CUDA venv).

Usage: python diag_clouds.py key   (uses the scratch imagery staged by test_build.py)
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import O4_Color_Correction as CC  # noqa: E402
from PIL import Image  # noqa: E402

DATA = os.environ.get("O4_REVIEW_DATA", r"E:\Ortho4XP_Review")
key = sys.argv[1]
meta = json.load(open(os.path.join(DATA, "color", "set", key, "meta.json")))
x0, y0, n, prov = meta["x0"], meta["y0"], meta["n"], meta["providers"][0]
paths = {(x0 + 16 * i, y0 + 16 * j): os.path.join(DATA, "color", "set", key, prov, f"{y0 + 16 * j}_{x0 + 16 * i}.jpg")
         for j in range(n) for i in range(n)}
refs = {}
for p in glob.glob(os.path.join(DATA, "color", "imagery", "**", "*_EOX14.jpg"), recursive=True):
    ty, tx = os.path.basename(p).split("_")[:2]
    refs[(int(tx), int(ty))] = p
g = CC.Group(meta["lat"], 16, prov, paths.keys(),
             lambda pv, tx, ty: np.asarray(Image.open(paths[(tx, ty)]).convert("RGB")))
with CC.torch.no_grad():
    g.fit(lambda tx, ty: np.asarray(Image.open(refs[(tx, ty)]).convert("RGB")) if (tx, ty) in refs else None)
print("has_missing:", sorted(g.has_missing))
for k in sorted(g.alpha):
    a = g.alpha[k].astype(np.float32)
    print(k, "alpha max %.2f  mean %.4f  >0.2 frac %.4f" % (a.max(), a.mean(), (a > 0.2).mean()))
out = np.zeros((n * g.cm, n * g.cm), np.float32)
for (tx, ty), a in g.alpha.items():
    i, j = g.ij(tx, ty)
    out[j * g.cm:(j + 1) * g.cm, i * g.cm:(i + 1) * g.cm] = a
Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).resize((900, 900)).save(
    os.path.join(DATA, "color", "runs", f"_diag_alpha_{key}.png"))
