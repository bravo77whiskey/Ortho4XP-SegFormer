"""Contact sheet of finished Cycles regions (natural orientation, scaled)."""
import sys, os, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import textures_cycles as TC, layout as L
work, out = sys.argv[1], sys.argv[2]
names = sys.argv[3].split(",")
tiles = []
for nm in names:
    page, reg = nm.split(":")
    region = (L.TUNNELS if page == "tunnels" else L.SOLAR)[reg]
    d = np.load(os.path.join(work, f"{page}_{reg}.npz")); sh = d["shaded"].astype(np.float32)
    sc = 0.85 / max(1e-3, float(np.quantile(sh.max(-1), 0.995)))
    im = Image.fromarray((TC._to_srgb(sh * min(sc, TC.EXPOSURE_MAX)) * 255).astype(np.uint8))
    h = 700; im = im.resize((max(1, int(im.width * h / im.height)), h), Image.LANCZOS)
    tiles.append((nm, im))
W = sum(t.width for _, t in tiles) + 10 * len(tiles)
S = Image.new("RGB", (W, 720), (30, 30, 30)); x = 0; d = ImageDraw.Draw(S)
for nm, t in tiles:
    S.paste(t, (x, 20)); d.text((x + 3, 3), nm, fill=(255, 255, 255)); x += t.width + 10
S.save(out)
