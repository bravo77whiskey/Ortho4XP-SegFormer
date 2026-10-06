"""Make a whole-tile review set from finished Ortho4XP tiles.

Usage (repo .venv):
    python scripts/visual_review/make_tile_set.py SET TILE [TILE ...]
        [--tiles-root DIR] [--provider BI]

Copies each tile's textures, cfg, apt and mesh (read-only on the source) to
DATA/sets/SET/tiles/<tile folder>/, which run_review.py builds as whole tiles
(all zoom levels together, neighbouring tiles side by side), and writes a JPG
of every texture the tile renders plus samples.json for the viewer under
DATA/sets/SET/PROVIDER/. The viewer's Location is the tile.
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
DATA = os.environ.get("O4_REVIEW_DATA", r"E:\Ortho4XP_Review")
TILES_ROOT = r"H:\XP12 Addons\Applications\Ortho4XP 1.4\_internal\Ortho4XP_Data\Tiles"

import O4_Texture_Selection as TEXSEL  # noqa: E402
from PIL import Image  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("set")
ap.add_argument("tiles", nargs="+", help="tile folder names or +LL+LLL")
ap.add_argument("--tiles-root", default=TILES_ROOT)
ap.add_argument("--provider", default="BI")
args = ap.parse_args()

set_root = os.path.join(DATA, "sets", args.set)
jpg_dir = os.path.join(set_root, args.provider)
os.makedirs(jpg_dir, exist_ok=True)
samples = []
for name in args.tiles:
    folder = name if name.startswith("zOrtho4XP_") else "zOrtho4XP_" + name
    short = folder[len("zOrtho4XP_"):]
    lat, lon = int(short[:3]), int(short[3:])
    src = os.path.join(args.tiles_root, folder)
    dst = os.path.join(set_root, "tiles", folder)
    os.makedirs(os.path.join(dst, "textures"), exist_ok=True)
    for f in (f"Ortho4XP_{short}.cfg", f"Data{short}.apt", f"Data{short}.mesh"):
        if os.path.exists(os.path.join(src, f)) and not os.path.exists(os.path.join(dst, f)):
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    names = [f for f in os.listdir(os.path.join(src, "textures")) if TEXSEL.TEXTURE_NAME_RE.match(f)]
    for f in names:
        if not os.path.exists(os.path.join(dst, "textures", f)):
            shutil.copy2(os.path.join(src, "textures", f), os.path.join(dst, "textures", f))
    files, covered, _ = TEXSEL.select_textures(names, os.path.join(dst, "textures"), lat, lon)
    print(f"{folder}: {len(names)} textures copied, {len(files)} rendered")
    for f in files:
        m = TEXSEL.TEXTURE_NAME_RE.match(f)
        til_y, til_x, zl = int(m.group(1)), int(m.group(2)), int(m.group(4))
        stem = f[:-4]
        jpg = f"{short}_{stem}.jpg"
        if not os.path.exists(os.path.join(jpg_dir, jpg)):
            with Image.open(os.path.join(dst, "textures", f)) as im:
                im.convert("RGB").save(os.path.join(jpg_dir, jpg), quality=90)
        lat_n, lon_w = TEXSEL._gtile_to_wgs84(til_x, til_y, zl)
        lat_s, lon_e = TEXSEL._gtile_to_wgs84(til_x + 16, til_y + 16, zl)
        samples.append({
            "key": f"{short}_{stem}", "file": jpg, "zl": zl,
            "label": f"{stem} (ZL{zl})", "location": folder,
            "bbox": [lat_s, lon_w, lat_n, lon_e],
            "tile": [lat, lon], "tile_dir": f"tiles/{folder}", "dds": f,
        })
samples.sort(key=lambda s: (s["location"], -s["zl"], s["key"]))
with open(os.path.join(jpg_dir, "samples.json"), "w") as f:
    json.dump(samples, f, indent=1)
print(f"{len(samples)} textures in set {args.set}")
