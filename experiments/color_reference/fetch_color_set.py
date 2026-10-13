"""Download the colour-correction test set.

For each location: the ZL16 Ortho4XP texture containing it from every provider
asked for (default BI and GO2), plus the EOX Sentinel-2 cloudless mosaic at
ZL14 over the same footprint with a one-tile margin, as the radiometric
reference.  Output: <DATA>/color/set/<key>/{BI16.jpg, GO216.jpg, ref.png,
meta.json}.  DATA defaults to E:\\Ortho4XP_Review.

Usage: python experiments/color_reference/fetch_color_set.py [PROVIDER ... | blocks]
"""
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.environ.get("O4_REVIEW_DATA", r"E:\Ortho4XP_Review")
OUT = os.path.join(DATA, "color", "set")
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import O4_File_Names as FNAMES  # noqa: E402
sys.path.append(FNAMES.Provider_dir)
import O4_Imagery_Utils as IMG  # noqa: E402
import O4_Geo_Utils as GEO  # noqa: E402

ZL = 16
REF_ZL = 14
# Climates / seasons / known provider defects.
# Multi-texture blocks: (key, label, lat, lon, providers, n) -> n x n ZL16
# textures centred on the point, so a defect can be judged against its
# surroundings and texture borders inside the block test consistency.
BLOCKS = [
    ("houston", "Houston (Bing: washed-out centre vs surroundings)", 29.758, -95.367, ["BI", "GO2"], 3),
    ("boston_sw", "SW of Boston (Google patches)", 42.280, -71.160, ["GO2", "BI"], 3),
    ("colombia", "Colombia, Valle del Cauca (Bing: heavy clouds, near worst case)", 4.651761, -76.440048, ["BI", "GO2"], 3),
    ("bioko", "Bioko, Equatorial Guinea (Bing: some clouds)", 3.567126, 8.733982, ["BI", "GO2"], 3),
]
LOCATIONS = [
    ("khh", "Kaohsiung city centre (humid subtropical, dense city)", 22.630, 120.300),
    ("paris", "Paris centre (temperate, Bing city haze)", 48.860, 2.340),
    ("vegas", "Las Vegas Strip edge (hot desert)", 36.120, -115.170),
    ("boston", "Boston harbour (temperate coast, water patches)", 42.355, -71.030),
    ("nagasaki", "Nagasaki bay (humid, Google water patches)", 32.745, 129.865),
    ("anchorage", "Anchorage (subarctic, haze/cloud)", 61.195, -149.900),
    ("bangkok", "Bangkok (tropical monsoon)", 13.745, 100.520),
]
EOX_URL = ("https://tiles.maps.eox.at/wmts/?layer=s2cloudless-2023_3857&style=default"
           "&tilematrixset=GoogleMapsCompatible&Service=WMTS&Request=GetTile&Version=1.0.0"
           "&Format=image%2Fjpeg&TileMatrix={z}&TileCol={x}&TileRow={y}")
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "Ortho4XP colour-reference experiment"


def eox_tile(x, y):
    for _ in range(4):
        try:
            r = SESSION.get(EOX_URL.format(z=REF_ZL, x=x, y=y), timeout=30)
            if r.ok:
                return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))
        except requests.RequestException:
            pass
    raise RuntimeError(f"EOX tile {x},{y} failed")


def fetch_reference(til_x, til_y, path):
    """ZL14 mosaic covering the ZL16 texture plus one ZL14 tile on each side."""
    f = 2 ** (ZL - REF_ZL)
    x0, y0 = til_x // f - 1, til_y // f - 1
    n = 16 // f + 2
    coords = [(x0 + i, y0 + j) for j in range(n) for i in range(n)]
    with ThreadPoolExecutor(16) as ex:
        tiles = list(ex.map(lambda c: eox_tile(*c), coords))
    mosaic = np.zeros((n * 256, n * 256, 3), np.uint8)
    for (x, y), t in zip(coords, tiles):
        mosaic[(y - y0) * 256:(y - y0 + 1) * 256, (x - x0) * 256:(x - x0 + 1) * 256] = t
    Image.fromarray(mosaic).save(path)
    # Offset of the texture inside the mosaic, in reference pixels.
    return {"ref_zl": REF_ZL, "ref_origin_px": [256, 256], "ref_scale": f}


def fetch_block(key, label, lat, lon, providers, n):
    d = os.path.join(OUT, key)
    cx, cy = GEO.wgs84_to_orthogrid(lat, lon, ZL)
    x0, y0 = cx - 16 * (n // 2), cy - 16 * (n // 2)
    for prov in providers:
        pd = os.path.join(d, prov)
        os.makedirs(pd, exist_ok=True)
        for j in range(n):
            for i in range(n):
                tx, ty = x0 + 16 * i, y0 + 16 * j
                name = f"{ty}_{tx}.jpg"
                if not os.path.isfile(os.path.join(pd, name)):
                    ok = IMG.download_jpeg_ortho(pd, name, tx, ty, ZL, prov)
                    print(key, prov, name, "ok" if ok else "FAILED", flush=True)
    f = 2 ** (ZL - REF_ZL)
    ref = os.path.join(d, "ref.png")
    if not os.path.isfile(ref):
        rx0, ry0 = x0 // f - 1, y0 // f - 1
        m = 16 * n // f + 2
        coords = [(rx0 + i, ry0 + j) for j in range(m) for i in range(m)]
        with ThreadPoolExecutor(16) as ex:
            tiles = list(ex.map(lambda c: eox_tile(*c), coords))
        mosaic = np.zeros((m * 256, m * 256, 3), np.uint8)
        for (x, y), t in zip(coords, tiles):
            mosaic[(y - ry0) * 256:(y - ry0 + 1) * 256, (x - rx0) * 256:(x - rx0 + 1) * 256] = t
        Image.fromarray(mosaic).save(ref)
    meta = {"key": key, "label": label, "lat": lat, "lon": lon, "zl": ZL, "n": n,
            "x0": x0, "y0": y0, "providers": providers, "ref_zl": REF_ZL,
            "ref_scale": f, "ref_origin_px": [256, 256]}
    with open(os.path.join(d, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(key, "done", flush=True)


def main(providers):
    for key, label, lat, lon in LOCATIONS:
        d = os.path.join(OUT, key)
        os.makedirs(d, exist_ok=True)
        til_x, til_y = GEO.wgs84_to_orthogrid(lat, lon, ZL)
        meta = {"key": key, "label": label, "lat": lat, "lon": lon,
                "til_x": til_x, "til_y": til_y, "zl": ZL}
        for prov in providers:
            name = f"{prov}{ZL}.jpg"
            if not os.path.isfile(os.path.join(d, name)):
                ok = IMG.download_jpeg_ortho(d, name, til_x, til_y, ZL, prov)
                print(key, name, "ok" if ok else "FAILED", flush=True)
        ref = os.path.join(d, "ref.png")
        meta.update(fetch_reference(til_x, til_y, ref) if not os.path.isfile(ref)
                    else {"ref_zl": REF_ZL, "ref_origin_px": [256, 256],
                          "ref_scale": 2 ** (ZL - REF_ZL)})
        with open(os.path.join(d, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        print(key, "done", flush=True)


if __name__ == "__main__":
    IMG.initialize_extents_dict()
    IMG.initialize_color_filters_dict()
    IMG.initialize_providers_dict()
    if sys.argv[1:2] == ["blocks"]:
        for b in BLOCKS:
            if not sys.argv[2:] or b[0] in sys.argv[2:]:
                fetch_block(*b)
    else:
        main(sys.argv[1:] or ["BI", "GO2"])
