"""Download the review test set for a provider and write its samples.json.

Usage: python scripts/visual_review/fetch_set.py [PROVIDER ...]   (default: BI)

The test set (the user's choice, 2026-10-05): Kaohsiung NW->SW of KHH and
Shenzhen N->E of SZX (OSM-poor areas), Wenzhou N->SW of WNZ and Shouguang
(greenhouses). ZL16 sector textures (~9 km each) plus one nested
ZL17/18/19 texture per location, saved as
<DATA>/sets/hotspots/<PROVIDER>/<til_y>_<til_x>_<PROVIDER><ZL>.jpg like Ortho4XP
does. DATA defaults to E:\\Ortho4XP_Review (override with O4_REVIEW_DATA).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
# Review data (textures, runs, caches, logs, the served viewer) lives outside
# the repo and off C:. Override with the O4_REVIEW_DATA environment variable.
DATA = os.environ.get("O4_REVIEW_DATA", r"E:\Ortho4XP_Review")
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import O4_File_Names as FNAMES  # noqa: E402
sys.path.append(FNAMES.Provider_dir)
import O4_Imagery_Utils as IMG  # noqa: E402
import O4_Geo_Utils as GEO  # noqa: E402

ZL = 16
LOCATIONS = {"khh": "Kaohsiung", "szx": "Shenzhen", "wnz": "Wenzhou", "shouguang": "Shouguang"}
SET = [
    ("khh_nw", "Kaohsiung NW of KHH (Gushan, harbour city)", 28528, 54656),
    ("khh_n", "Kaohsiung N/NW of KHH (Sanmin, Zuoying)", 28528, 54672),
    ("khh_airport", "Kaohsiung W/SW of KHH (Xiaogang, steel, port)", 28544, 54672),
    ("szx_n", "Shenzhen N of SZX (Shajing, Fuyong)", 28512, 53472),
    ("szx_ne", "Shenzhen NE of SZX (Songgang, Shiyan)", 28512, 53488),
    ("szx_e", "Shenzhen E of SZX (Xixiang, Hangcheng)", 28528, 53488),
    ("wnz_n", "Wenzhou N of WNZ (Lingkun island)", 27456, 54768),
    ("wnz_airport", "Wenzhou E/S of WNZ (fields, greenhouses)", 27472, 54768),
    ("wnz_w", "Wenzhou W/SW of WNZ (runway, Longwan)", 27472, 54752),
    ("wnz_sw", "Wenzhou SW of WNZ (industry, reclaimed land)", 27488, 54752),
    ("shouguang", "Shouguang, Shandong (greenhouse fields)", 25536, 54384),
]
# One point per location inside its ZL16 sector; the ZL17/18/19 textures that
# contain it are added so every location is also tested at higher zoom levels.
ANCHORS = [
    ("khh", "Kaohsiung dense city grid (Lingya/Yancheng)", 22.63024, 120.29238),
    ("szx", "Shenzhen urban villages (Shajing)", 22.71134, 113.78848),
    ("wnz", "Wenzhou greenhouse rows by the runway", 27.89657, 120.86191),
    ("shouguang", "Shouguang greenhouse fields", 36.82336, 118.80352),
]
HIGH_ZLS = (17, 18, 19)


def set_entries():
    """(key, label, til_y, til_x, zl) for every texture in the set."""
    entries = [(key, label, til_y, til_x, ZL) for key, label, til_y, til_x in SET]
    for loc, label, lat, lon in ANCHORS:
        for zl in HIGH_ZLS:
            til_x, til_y = GEO.wgs84_to_orthogrid(lat, lon, zl)
            entries.append((f"{loc}_z{zl}", f"{label} - ZL{zl}", til_y, til_x, zl))
    return entries


def fetch(provider):
    out_dir = os.path.join(DATA, "sets", "hotspots", provider)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for key, label, til_y, til_x, zl in set_entries():
        name = f"{til_y}_{til_x}_{provider}{zl}.jpg"
        if not os.path.isfile(os.path.join(out_dir, name)):
            ok = IMG.download_jpeg_ortho(out_dir, name, til_x, til_y, zl, provider)
            print(f"{key:14} {name}  {'ok' if ok else 'FAILED'}", flush=True)
        lat_n, lon_w = GEO.gtile_to_wgs84(til_x, til_y, zl)
        lat_s, lon_e = GEO.gtile_to_wgs84(til_x + 16, til_y + 16, zl)
        rows.append({"key": key, "location": LOCATIONS[key.split("_")[0]],
                     "label": label, "zl": zl, "file": name,
                     "bbox": [lat_s, lon_w, lat_n, lon_e]})
    with open(os.path.join(out_dir, "samples.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(len(rows), "samples in", out_dir)


if __name__ == "__main__":
    IMG.initialize_extents_dict()
    IMG.initialize_color_filters_dict()
    IMG.initialize_providers_dict()
    for prov in sys.argv[1:] or ["BI"]:
        fetch(prov)
