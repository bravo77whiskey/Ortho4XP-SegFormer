"""Rewrite a built grass pack's per-colour PNGs from its saved palette.

Usage: python regen_textures.py PACK_DIR   (PACK_DIR = .../O4XP_Grass_Spike)
Same ramp as make_region_pack.build_texture (no --tint-tiers): centred on the
palette colour so the flat far level (texture middle) is exactly that colour.
"""
import glob, json, os, sys
import numpy as np, cv2
from PIL import Image

TEX_W, TEX_H = 16, 64
pack = sys.argv[1]
info = json.load(open(os.path.join(pack, "spike_info.json")))
assert not info["params"].get("tint_tiers"), "tinted pack: rebuild instead"
palette = info["palette_lab"]
yy, xx = np.mgrid[0:TEX_H, 0:TEX_W].astype(np.float32)
up = 1.0 - yy / (TEX_H - 1)
side = xx / (TEX_W - 1) * 2 - 1
n = 0
for png in glob.glob(os.path.join(pack, "grass", "*", "c*.png")):
    k = int(os.path.basename(png)[1:3])
    L, a, b = palette[k]
    lab = np.stack([L - 7 + 14 * up + 4 * side, a + 0.75 - 1.5 * up, b - 1.5 + 3 * up], -1)
    rgb = (np.clip(cv2.cvtColor(lab.astype(np.float32).reshape(-1, 1, 3), cv2.COLOR_LAB2RGB), 0, 1) * 255)
    Image.fromarray(rgb.round().astype(np.uint8).reshape(TEX_H, TEX_W, 3), "RGB").save(png)
    n += 1
print("rewrote", n, "textures")
