"""Contact sheet: one crop of X-Plane's summer 3-d tree atlas in each leaf colour bucket
(rows: hue step, columns: chroma scale), on mid grey."""
import io, os, sys, numpy as np, cv2
from PIL import Image, ImageDraw
sys.path.insert(0, r"G:\Dev\Ortho4XP-SegFormer\src")
import O4_Forest_Tint as T
XP = r"D:\SteamLibrary\steamapps\common\X-Plane 12\Resources\default scenery\1200 forests\textures"
src = open(os.path.join(XP, "trees_3D1_ALB.dds"), "rb").read()
crop = (0, 0, 1024, 1024)
cells = []
for k in range(len(T.BUCKETS)):
    im = Image.open(io.BytesIO(T.recolour_dxt5(src, k))).convert("RGBA").crop(crop).resize((320, 320), Image.LANCZOS)
    bg = Image.new("RGBA", im.size, (128, 128, 128, 255)); bg.alpha_composite(im)
    d = ImageDraw.Draw(bg); dh, s = T.BUCKETS[k]
    d.text((6, 6), f"t{k}  hue {dh:+g}  x{s:g}" + ("  (X-Plane)" if k == T.STOCK else ""), fill=(255, 255, 255))
    cells.append(bg.convert("RGB"))
n = len(T.CHROMA_SCALES)
sheet = Image.new("RGB", (320 * n, 320 * len(T.HUE_STEPS_DEG)))
for k, c in enumerate(cells):
    sheet.paste(c, ((k % n) * 320, (k // n) * 320))
out = sys.argv[1] if len(sys.argv) > 1 else "tint_sheet.jpg"
sheet.save(out, quality=88); print(out)
