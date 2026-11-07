"""Greenhouse maps from multispectral satellite data, as a placement prior.

RGB alone cannot tell film from grey concrete or furrowed soil reliably;
plastic has its own signature in the near and shortwave infrared (Koc-San &
Sonmez, ISPRS 2016).  Published maps built on those bands can tell the
overlay where greenhouse land is:

* Global-PCG-10 (Niu et al., ESSD 2025): 10 m, Sentinel-2 2020, CC BY 4.0,
  1x1 degree GeoTIFFs (``*_PCG_Result.tif``), precision 99.8 %, recall 85 %.
* Tong et al. (Nature Food 2024): 3 m, PlanetScope + Sentinel-2 2019,
  non-commercial use, Planet attribution.

Point the ``greenhouse_map_dir`` setting (env O4_GREENHOUSE_MAP_DIR) at a
folder of such GeoTIFFs in geographic coordinates (EPSG:4326); any non-zero,
non-nodata pixel counts as greenhouse.  Files are found by their bounds,
cached in ``_o4_index.json`` in that folder.
"""

from __future__ import annotations

import json
import os

import numpy as np

INDEX_NAME = "_o4_index.json"
_INDEX_CACHE = {}


def map_dir():
    path = os.environ.get("O4_GREENHOUSE_MAP_DIR", "").strip()
    return path if path and os.path.isdir(path) else None


def _tif_paths(root):
    out = []
    for base, _dirs, files in os.walk(root):
        for name in files:
            if name.lower().endswith((".tif", ".tiff")):
                out.append(os.path.relpath(os.path.join(base, name), root))
    return out


def _scan(root, known=None):
    """Bounds of every EPSG:4326 GeoTIFF under root; entries in ``known``
    whose file is unchanged are reused."""
    import rasterio

    known = {e["path"]: e for e in (known or [])}
    entries = []
    for rel in _tif_paths(root):
        path = os.path.join(root, rel)
        mtime = os.path.getmtime(path)
        old = known.get(rel)
        if old is not None and old.get("mtime") == mtime:
            entries.append(old)
            continue
        bounds = None  # unreadable or not EPSG:4326: listed, never used
        try:
            with rasterio.open(path) as ds:
                crs = ds.crs.to_epsg() if ds.crs is not None else None
                b = ds.bounds
            if crs in (None, 4326):
                bounds = [b.left, b.bottom, b.right, b.top]
        except Exception:
            pass
        entries.append({"path": rel, "bounds": bounds, "mtime": mtime})
    return entries


def _index(root):
    if root in _INDEX_CACHE:
        return _INDEX_CACHE[root]
    idx_path = os.path.join(root, INDEX_NAME)
    try:
        with open(idx_path, encoding="utf-8") as f:
            known = json.load(f)
    except (OSError, ValueError):
        known = []
    # maps get added to: rescan whenever the set of files differs
    if sorted(e["path"] for e in known) == sorted(_tif_paths(root)):
        entries = known
    else:
        entries = _scan(root, known)
        try:
            with open(idx_path, "w", encoding="utf-8") as f:
                json.dump(entries, f)
        except OSError:
            pass
    _INDEX_CACHE[root] = entries
    return entries


def texture_mask(lat_n, lat_s, lon_w, lon_e, img_h, img_w, root=None):
    """uint8 greenhouse mask (1 = mapped greenhouse) on the texture's pixel
    grid, or None when no map covers the texture.

    Rows and columns map linearly to latitude and longitude, like the
    overlay's own px_to_latlon.
    """
    root = root or map_dir()
    if root is None:
        return None
    hits = [e for e in _index(root)
            if e["bounds"] and e["bounds"][0] < lon_e and e["bounds"][2] > lon_w and
            e["bounds"][1] < lat_n and e["bounds"][3] > lat_s]
    if not hits:
        return None
    import rasterio
    from rasterio.windows import from_bounds

    lons = lon_w + (np.arange(img_w) + 0.5) / img_w * (lon_e - lon_w)
    lats = lat_n - (np.arange(img_h) + 0.5) / img_h * (lat_n - lat_s)
    out = np.zeros((img_h, img_w), np.uint8)
    covered = False
    for e in hits:
        with rasterio.open(os.path.join(root, e["path"])) as ds:
            win = from_bounds(lon_w, lat_s, lon_e, lat_n, ds.transform).round_offsets().round_lengths()
            r0, c0 = int(win.row_off) - 1, int(win.col_off) - 1
            r0, c0 = max(0, r0), max(0, c0)
            r1 = min(ds.height, int(win.row_off + win.height) + 2)
            c1 = min(ds.width, int(win.col_off + win.width) + 2)
            if r1 <= r0 or c1 <= c0:
                continue
            data = ds.read(1, window=((r0, r1), (c0, c1)))
            nodata = ds.nodata
            t = ds.transform
            # pixel indices of each texture row / column in the window
            cols = np.floor((lons - t.c) / t.a).astype(np.int64) - c0
            rows = np.floor((lats - t.f) / t.e).astype(np.int64) - r0
            okc = (cols >= 0) & (cols < data.shape[1])
            okr = (rows >= 0) & (rows < data.shape[0])
            if not okc.any() or not okr.any():
                continue
            covered = True
            good = data > 0
            if nodata is not None:
                good &= data != nodata
            sub = good[np.clip(rows, 0, data.shape[0] - 1)][:, np.clip(cols, 0, data.shape[1] - 1)]
            sub &= okr[:, None] & okc[None, :]
            out |= sub.astype(np.uint8)
    return out if covered else None
