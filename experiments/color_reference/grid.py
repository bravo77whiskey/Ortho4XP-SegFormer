"""Side-by-side block mosaics for quick comparison.

Usage: python grid.py OUT.jpg key prov src [src ...]
  src = "raw" | run name (e.g. v5l12fill) | "alpha:<run>" (that run's cloud alpha)
"""
import json
import os
import sys

from PIL import Image

DATA = os.environ.get("O4_REVIEW_DATA", r"E:\Ortho4XP_Review")
C = os.path.join(DATA, "color")
out, key, prov, srcs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
meta = json.load(open(os.path.join(C, "set", key, "meta.json")))
n = meta.get("n", 1)
S = 3000 // len(srcs) if len(srcs) > 2 else 1000


def mosaic(base):
    m = Image.new("RGB", (S, S))
    for j in range(n):
        for i in range(n):
            name = f"{meta['y0'] + 16 * j}_{meta['x0'] + 16 * i}.jpg" if n > 1 else f"{prov}16.jpg"
            m.paste(Image.open(os.path.join(base, name)).resize((S // n, S // n), Image.LANCZOS),
                    (i * S // n, j * S // n))
    return m


ims = []
for s in srcs:
    if s == "raw":
        ims.append(mosaic(os.path.join(C, "set", key, prov) if n > 1 else os.path.join(C, "set", key)))
    elif s.startswith("alpha:"):
        ims.append(Image.open(os.path.join(C, "runs", s[6:], key, f"{prov}_alpha.png")).convert("RGB").resize((S, S)))
    else:
        ims.append(mosaic(os.path.join(C, "runs", s, key, prov)))
sheet = Image.new("RGB", (len(ims) * (S + 8) - 8, S), (40, 40, 40))
for i, im in enumerate(ims):
    sheet.paste(im, (i * (S + 8), 0))
sheet.save(out, quality=88)
