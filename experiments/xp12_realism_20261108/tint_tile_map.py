"""Whole-tile map of the leaf colour bucket each tree polygon got (from a veg DSF), to spot
colour seams along texture edges. Distinct false colours per bucket, texture grid overlaid.

Usage: python tint_tile_map.py <veg.dsf> <out.png> [zl_grid=16]"""
import os, re, subprocess, sys, tempfile, colorsys
import numpy as np, cv2
DSFTOOL = r"G:\Dev\Ortho4XP-SegFormer\Utils\win\DSFTool.exe"
dsf, out = sys.argv[1], sys.argv[2]
zl = int(sys.argv[3]) if len(sys.argv) > 3 else 16
SIZE = 6000
txt = os.path.join(tempfile.gettempdir() if False else os.path.dirname(out), "_tint_map.txt")
subprocess.run([DSFTOOL, "--dsf2text", dsf, txt], check=True, stdout=subprocess.DEVNULL)
defs, west, south = [], None, None
img = np.full((SIZE, SIZE, 3), 30, np.uint8)
TRUE = len(sys.argv) > 4 and sys.argv[4] == "true"
if TRUE:   # the bucket's real summer leaf colour (L* 45), not exaggerated
    sys.path.insert(0, r"G:\Dev\Ortho4XP-SegFormer\src")
    import O4_Forest_Tint as T
    def _rgb(k):
        a, b = T.bucket_ab(k)
        lab = np.array([[[45.0, a, b]]], np.float32)
        return tuple(int(v) for v in np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)[0, 0] * 255, 0, 255))
    pal = {k: _rgb(k) for k in range(16)}
else:
    pal = {k: tuple(int(255 * c) for c in colorsys.hsv_to_rgb((k * 0.61) % 1, 0.85, 0.95)) for k in range(16)}
cur, ring, rings = None, [], []
def flush():
    if cur is None or not rings: return
    m = re.search(r"(?:^|[/-])t(\d+)/", defs[cur])
    col = pal[int(m.group(1))] if m else (200, 200, 200)
    pts = [np.array(r, np.int32) for r in rings if len(r) >= 3]
    if pts: cv2.fillPoly(img, pts, col[::-1])
    for r in rings:
        if len(r) == 1 or len(r) == 2:
            cv2.circle(img, tuple(r[0]), 1, col[::-1], -1)
for ln in open(txt):
    p = ln.split()
    if not p: continue
    if p[0] == "PROPERTY" and p[1] == "sim/west": west = int(p[2])
    elif p[0] == "PROPERTY" and p[1] == "sim/south": south = int(p[2])
    elif p[0] == "POLYGON_DEF": defs.append(p[1])
    elif p[0] == "BEGIN_POLYGON": cur, rings = int(p[1]), []
    elif p[0] == "BEGIN_WINDING": ring = []
    elif p[0] == "POLYGON_POINT":
        lon, lat = float(p[1]), float(p[2])
        ring.append((int((lon - west) * SIZE), int((south + 1 - lat) * SIZE)))
    elif p[0] == "END_WINDING": rings.append(ring)
    elif p[0] == "END_POLYGON": flush(); cur = None
os.remove(txt)
# ZL texture grid (Web Mercator tiles of 16 px at ZL+4 = one texture at ZL)
import math
n = 2 ** zl
def ytile(lat): return (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
for xt in range(int((west + 180) / 360 * n), int((west + 181) / 360 * n) + 2):
    if xt % 16: continue
    x = int(((xt / n * 360 - 180) - west) * SIZE); cv2.line(img, (x, 0), (x, SIZE), (90, 90, 90), 1)
for yt in range(int(ytile(south + 1)), int(ytile(south)) + 2):
    if yt % 16: continue
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yt / n))))
    y = int((south + 1 - lat) * SIZE); cv2.line(img, (0, y), (SIZE, y), (90, 90, 90), 1)
cv2.imwrite(out, cv2.resize(img, (3000, 3000), interpolation=cv2.INTER_AREA))
print(out)
