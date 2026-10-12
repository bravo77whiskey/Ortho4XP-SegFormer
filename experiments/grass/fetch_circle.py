"""Download the textures (Ortho4XP 16x16-tile blocks) covering a circle.

Usage (repo .venv, needs pyproj):
  python fetch_circle.py --center LAT LON --radius-nm R --zl 17 --provider BI --out DIR
Files are named <til_y>_<til_x>_<PROVIDER><ZL>.jpg as Ortho4XP does.
"""
import argparse, math, os, sys

REPO = r"G:\Dev\Ortho4XP-SegFormer"
ap = argparse.ArgumentParser()
ap.add_argument("--center", nargs=2, type=float, required=True)
ap.add_argument("--radius-nm", type=float, default=2.0)
ap.add_argument("--zl", type=int, default=17)
ap.add_argument("--provider", default="BI")
ap.add_argument("--out", required=True)
args = ap.parse_args()
out = os.path.abspath(args.out)
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
import O4_File_Names as FNAMES  # noqa: E402
sys.path.append(FNAMES.Provider_dir)
import O4_Imagery_Utils as IMG  # noqa: E402
import O4_Geo_Utils as GEO  # noqa: E402

IMG.initialize_extents_dict()
IMG.initialize_color_filters_dict()
IMG.initialize_providers_dict()
os.makedirs(out, exist_ok=True)
lat0, lon0 = args.center
r = args.radius_nm * 1852
dlat, dlon = r / 110574, r / (111320 * math.cos(math.radians(lat0)))
x_w, y_n = GEO.wgs84_to_orthogrid(lat0 + dlat, lon0 - dlon, args.zl)
x_e, y_s = GEO.wgs84_to_orthogrid(lat0 - dlat, lon0 + dlon, args.zl)
n = 0
for ty in range(min(y_n, y_s), max(y_n, y_s) + 1, 16):
    for tx in range(min(x_w, x_e), max(x_w, x_e) + 1, 16):
        name = f"{ty}_{tx}_{args.provider}{args.zl}.jpg"
        if not os.path.isfile(os.path.join(out, name)):
            ok = IMG.download_jpeg_ortho(out, name, tx, ty, args.zl, args.provider)
            print(name, "ok" if ok else "FAILED", flush=True)
        n += 1
print(n, "textures in", out)
