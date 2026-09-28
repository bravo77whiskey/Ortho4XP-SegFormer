"""Debug pages: each region a colour with a grid and its name, arrows = along."""
import os, sys
from PIL import Image, ImageDraw
import layout as L
COLS = [(200, 80, 80), (80, 160, 80), (80, 100, 200), (200, 170, 60), (160, 80, 180), (60, 170, 170), (150, 150, 150)]
def page(regions, out):
    im = Image.new("RGB", (1024, 1024), (40, 40, 40)); d = ImageDraw.Draw(im)
    for k, (name, r) in enumerate(regions.items()):
        c = COLS[k % len(COLS)]
        d.rectangle((r.x0, r.y0, r.x1 - 1, r.y1 - 1), fill=c)
        for i in range(0, 1024, 32):
            d.line((r.x0, i, r.x1, i), fill=tuple(x // 2 for x in c)) if r.y0 <= i < r.y1 else None
        d.text((r.x0 + 4, r.y0 + 4), name, fill=(255, 255, 255))
        # mark "along bottom" end: source along=0 -> page bottom (plain) / left (rotated)
        if r.rotated:
            d.rectangle((r.x0, r.y0, r.x0 + 6, r.y1 - 1), fill=(0, 0, 0))
        else:
            d.rectangle((r.x0, r.y1 - 6, r.x1 - 1, r.y1 - 1), fill=(0, 0, 0))
    im.save(out)
    Image.new("RGBA", (64, 64), (128, 128, 10, 120)).save(out.replace(".png", "_nml.png"))
if __name__ == "__main__":
    o = sys.argv[1]; os.makedirs(o, exist_ok=True)
    page(L.TUNNELS, os.path.join(o, "tunnels.png")); page(L.SOLAR, os.path.join(o, "solar.png"))
