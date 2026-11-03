"""Bing ZL16 + ZL18 textures across tree-density extremes, into E:. Touches no tile folder."""
import json
import os
import sys

REPO = r"G:\Dev\Ortho4XP-SegFormer"
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
import O4_File_Names as FNAMES
sys.path.append(FNAMES.Provider_dir)
import O4_Imagery_Utils as IMG
import O4_Geo_Utils as GEO

OUT = r"E:\Ortho4XP_Review\veg_density\biomes"
os.makedirs(OUT, exist_ok=True)
PLACES = {
    "brazil_bahia": (-12.5, -41.5),     # tile -13-042, the square-hole forests
    "amazon": (-3.05, -60.5),
    "cerrado": (-14.15, -47.75),
    "serengeti": (-2.4, 34.8),
    "dehesa_spain": (39.0, -6.3),
    "nsw_woodland": (-33.0, 147.0),
    "finland": (62.0, 26.0),
    "oregon_pine": (44.0, -121.4),
    "olives_jaen": (37.8, -3.8),
}
IMG.initialize_extents_dict()
IMG.initialize_color_filters_dict()
IMG.initialize_providers_dict()
rows = []
for key, (lat, lon) in PLACES.items():
    for zl in (16, 18):
        til_x, til_y = GEO.wgs84_to_orthogrid(lat, lon, zl)
        name = f"{key}_{til_y}_{til_x}_BI{zl}.jpg"
        if not os.path.isfile(os.path.join(OUT, name)):
            ok = IMG.download_jpeg_ortho(OUT, name, til_x, til_y, zl, "BI")
            print(name, "ok" if ok else "FAILED", flush=True)
        rows.append({"key": key, "zl": zl, "file": name, "lat": lat, "lon": lon})
json.dump(rows, open(os.path.join(OUT, "samples.json"), "w"), indent=1)
