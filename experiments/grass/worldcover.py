"""ESA WorldCover 2021 v200 (10 m, CC-BY 4.0) windows, read by HTTP range.

The global map is published as 3x3 degree cloud-optimised GeoTIFFs on AWS
(deflate, 1024 px tiles, no predictor), which O4_Leaf_Type's GeoTiffTiles
reads directly, so only the tiles under a window are fetched.  Windows are
cached as .npz.

Classes: 10 tree cover, 20 shrubland, 30 grassland, 40 cropland, 50 built-up,
60 bare / sparse vegetation, 70 snow and ice, 80 permanent water,
90 herbaceous wetland, 95 mangroves, 100 moss and lichen, 0 no data.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, r"G:\Dev\Ortho4XP-SegFormer\src")
import O4_Leaf_Type as LT  # noqa: E402

URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
       "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
DEG_PX = 1.0 / 12000.0


def _tile_name(lat, lon):
    la, lo = 3 * math.floor(lat / 3), 3 * math.floor(lon / 3)
    return f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"


def window(south, north, west, east, cache_dir):
    """Codes on a regular grid covering the box: (codes, west, north, dx, dy)."""
    os.makedirs(cache_dir, exist_ok=True)
    key = f"wc_{south:.4f}_{north:.4f}_{west:.4f}_{east:.4f}.npz"
    path = os.path.join(cache_dir, key)
    if os.path.isfile(path):
        z = np.load(path)
        return z["codes"], float(z["west"]), float(z["north"]), DEG_PX, DEG_PX
    rows = int(math.ceil((north - south) / DEG_PX))
    cols = int(math.ceil((east - west) / DEG_PX))
    codes = np.zeros((rows, cols), np.uint8)
    lat_tiles = range(3 * math.floor(south / 3), 3 * math.floor((north - 1e-9) / 3) + 1, 3)
    lon_tiles = range(3 * math.floor(west / 3), 3 * math.floor((east - 1e-9) / 3) + 1, 3)
    for la in lat_tiles:
        for lo in lon_tiles:
            try:
                tif = LT.GeoTiffTiles(LT._HttpBytes(URL.format(tile=_tile_name(la, lo))))
            except Exception as exc:          # ocean tiles are not published
                print(f"  [WorldCover] no tile {_tile_name(la, lo)}: {exc}")
                continue
            r0 = int(round((tif.north - north) / tif.dy))
            c0 = int(round((west - tif.west) / tif.dx))
            part = tif.read_window(r0, c0, rows, cols)
            codes = np.where(part > 0, part, codes)
    np.savez_compressed(path, codes=codes, west=west, north=north)
    return codes, west, north, DEG_PX, DEG_PX


def sample(codes, west, north, dx, dy, lats, lons):
    """Nearest-neighbour codes for 1-D arrays of row latitudes and column longitudes."""
    r = np.clip(((north - np.asarray(lats)) / dy).astype(int), 0, codes.shape[0] - 1)
    c = np.clip(((np.asarray(lons) - west) / dx).astype(int), 0, codes.shape[1] - 1)
    return codes[r[:, None], c[None, :]]
