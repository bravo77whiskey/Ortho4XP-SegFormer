"""Add XP12 terrain material maps to already-built tiles, in place.

    python scripts/add_terrain_materials.py <tile dir> [<tile dir> ...]
        [--relief 0.15] [--workers 4] [--nvcompress PATH] [--force]

For every land .ter of each tile (water / sea / overlay .ter are left alone)
it makes <texture>_mat.dds from the tile's own DDS texture and inserts after
BASE_TEX_NOWRAP:

    TEXTURE_NORMAL 1 ../textures/<texture>_mat.dds
    GLOBAL_specular 1.0
    NORMAL_METALNESS

Idempotent: a .ter that already has these lines is only checked (its map is
remade when missing, older than the texture or made with another relief); a
.ter whose TEXTURE_NORMAL points at something else is left alone.  Same code
as Step 3 (src/O4_Terrain_Materials.py), which does this by itself on the
next build of a tile.  Quit X-Plane first.
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import O4_Terrain_Materials as TMAT  # noqa: E402


def default_converter():
    if sys.platform.startswith("win"):
        return os.path.join(ROOT, "Utils", "win", "nvcompress", "nvcompress.exe")
    if sys.platform == "darwin":
        return os.path.join(ROOT, "Utils", "mac", "DDSTool")
    return os.path.join(ROOT, "Utils", "lin", "nvcompress")


def xplane_running():
    if not sys.platform.startswith("win"):
        return False
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq X-Plane.exe"],
                         capture_output=True, text=True).stdout
    return "X-Plane.exe" in out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tiles", nargs="+", help="tile directories (zOrtho4XP_+NN+EEE)")
    ap.add_argument("--relief", type=float, default=TMAT.RELIEF_DEFAULT,
                    help="micro-relief strength (cfg terrain_materials_relief)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nvcompress", default=default_converter())
    ap.add_argument("--force", action="store_true", help="run even while X-Plane is running")
    args = ap.parse_args()
    if not os.path.isfile(args.nvcompress):
        sys.exit(f"nvcompress not found: {args.nvcompress}")
    if xplane_running() and not args.force:
        sys.exit("X-Plane is running - quit it first (or --force)")
    for tile in args.tiles:
        if not os.path.isdir(os.path.join(tile, "terrain")):
            print(f"{tile}: no terrain directory, skipped")
            continue
        t0 = time.time()
        s = TMAT.retrofit_tile(tile, args.relief, args.nvcompress, workers=args.workers, log=print)
        print(f"{tile}: {s['made']} maps made, {s['kept']} up to date, {s['added']} .ter patched"
              + (f", {s['removed']} .ter lines dropped (no texture)" if s["removed"] else "")
              + (f", {s['failed']} failed" if s["failed"] else "")
              + (f", {s['skipped']} .ter with another normal map left alone" if s["skipped"] else "")
              + f" ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
