"""Leaf-type maps that pick broadleaf, conifer or mixed vegetation assets.

The SegFormer tree mask says *where* trees are; this module says *which* of the
three simHeaven forest families belongs there.  Per one-degree tile it prepares
open leaf-type maps (downloaded once, cached under ``Overlay_cache/_leaf_type``):

1. Copernicus HRL Dominant Leaf Type 2018, 10 m, Europe (EEA image service)
2. explicit OSM ``leaf_type`` — simHeaven X-World broad/coni forest polygons
   (``mixed`` is skipped: it is the untagged default for most OSM forests)
3. GLC_FCS30D 2022, 30 m, global (Zenodo, 5x5 degree blocks)
4. Copernicus CGLS-LC100 forest type 2019, 100 m, 80N-60S (Zenodo, HTTP ranges)

Per texture, every pixel takes the first source that types it, then the nearest
typed pixel within ``FILL_LIMIT_M`` (sources in the same order).  Shares are
smoothed over ``SMOOTH_M`` and thresholded into broad / conifer / mixed zones,
so one large SegFormer polygon spanning a conifer slope and a broadleaf valley
is split rather than voted into a single family.

``XPForestMapper`` then plants each family as X-Plane 12's own forest
(``lib/g8/<family>_<temp>_<moist>.for``) for the climate band the default
global scenery uses at that spot (Köppen-Geiger where it is not installed).

Only numpy, zlib, OpenCV, Pillow and requests are used: no GDAL.
"""

import io
import math
import os
import struct
import time
import zipfile
import zlib

import cv2
import numpy as np

import O4_Forest_Assets as FOREST_ASSETS
import O4_Climate_Regions as CLIMATE

VERSION = 1

NONE, BROAD, CONI, MIXED = 0, 1, 2, 3
CODE_PATHS = (
    (BROAD, "simheaven/forests/broad.for"),
    (CONI, "simheaven/forests/coni.for"),
    (MIXED, "simheaven/forests/mixed.for"),
)
PATH_CODES = {path: code for code, path in CODE_PATHS}

ANALYSIS_M = 10.0        # zone raster resolution
MAX_ANALYSIS_PX = 1024
SMOOTH_M = 60.0          # half-width of the share smoothing window
FILL_LIMIT_M = 1500.0    # nearest typed pixel reach for untyped tree pixels
BROAD_HI, BROAD_LO = 0.65, 0.35
TILE_MARGIN_DEG = 0.03

USER_AGENT = "Ortho4XP-SegFormer leaf-type (+https://github.com/bravo77whiskey/Ortho4XP-SegFormer)"
HTTP_TIMEOUT = 120
# A big download that stops delivering for this long is resumed from the byte
# it reached; it gives up after this many resumes in a row that got nothing.
STREAM_STALL_S = 30
STREAM_RESUMES = 5

# ── Unified code look-up tables ──────────────────────────────────────────────
# GLC_FCS30D fine classes: 51/52 evergreen broadleaf, 61/62 deciduous broadleaf,
# 71/72 evergreen needleleaf, 81/82 deciduous needleleaf, 91/92 mixed leaf.
GLC_LUT = np.zeros(256, np.uint8)
GLC_LUT[[51, 52, 61, 62]] = BROAD
GLC_LUT[[71, 72, 81, 82]] = CONI
GLC_LUT[[91, 92]] = MIXED
# CGLS forest type: 0 unknown, 1 ENF, 2 EBF, 3 DNF, 4 DBF, 5 mixed, 255 nodata.
CGLS_LUT = np.zeros(256, np.uint8)
CGLS_LUT[[2, 4]] = BROAD
CGLS_LUT[[1, 3]] = CONI
CGLS_LUT[5] = MIXED
# HRL DLT: 0 non-tree, 1 broadleaved, 2 coniferous, 255 outside coverage.
HRL_LUT = np.zeros(256, np.uint8)
HRL_LUT[1] = BROAD
HRL_LUT[2] = CONI

GLC_URL = "https://zenodo.org/records/8239305/files/GLC_FCS30D_19852022maps_{zip}.zip?download=1"
GLC_PX_DEG = 0.0002694945852358564
GLC_BAND_2022 = 22       # 0-based: annual bands run 2000..2022
CGLS_URL = ("https://zenodo.org/records/3939050/files/"
            "PROBAV_LC100_global_v3.0.1_2019-nrt_Forest-Type-layer_EPSG-4326.tif")
HRL_URL = ("https://image.discomap.eea.europa.eu/arcgis/rest/services/GioLandPublic/"
           "HRL_DominantLeafType2018/ImageServer/exportImage")
HRL_ENVELOPE = (34.0, 72.0, -32.0, 45.0)   # south, north, west, east (EEA38 + UK)
HRL_MAX_PX = 4000
HRL_PX_DEG = 1.0e-4      # ~11 m north-south, 5-8 m east-west across Europe


def _log_default(*args):
    print(*args, flush=True)


def _session():
    import requests

    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def _http_get(session, url, retries=4, **kwargs):
    kwargs.setdefault("timeout", HTTP_TIMEOUT)
    for attempt in range(retries):
        try:
            r = session.get(url, **kwargs)
            r.raise_for_status()
            return r
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("unreachable")


def _atomic_save_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _tile_key(lat, lon):
    return f"{int(lat):+03d}{int(lon):+04d}"


# ── Minimal GeoTIFF tile reader (classic + BigTIFF, none/deflate/LZW) ────────
_TYPE_FMT = {1: "B", 2: "c", 3: "H", 4: "I", 7: "B", 11: "f", 12: "d", 16: "Q", 17: "q"}


class _FileBytes:
    def __init__(self, path):
        self._f = open(path, "rb")

    def read(self, offset, size):
        self._f.seek(offset)
        return self._f.read(size)

    def close(self):
        self._f.close()


class _HttpBytes:
    def __init__(self, url, session=None):
        self.url = url
        self.session = session or _session()

    def read(self, offset, size):
        if size <= 0:
            return b""
        r = _http_get(self.session, self.url,
                      headers={"Range": f"bytes={offset}-{offset + size - 1}"})
        data = r.content
        if len(data) != size:
            raise IOError(f"short range read from {self.url}")
        return data

    def close(self):
        pass


def _lzw_decode(data, width, height):
    """Decode one LZW tile through OpenCV's libtiff (no extra dependency).

    The tile is wrapped as a single-strip, single-sample TIFF whose width is
    ``tile_width * samples``: with no predictor the byte stream is identical.
    """
    entries = ((256, 4, width), (257, 4, height), (258, 3, 8), (259, 3, 5),
               (262, 3, 1), (273, 4, 0), (277, 3, 1), (278, 4, height),
               (279, 4, len(data)))
    data_off = 8 + 2 + len(entries) * 12 + 4
    out = bytearray(b"II*\x00" + struct.pack("<I", 8) + struct.pack("<H", len(entries)))
    for tag, typ, value in entries:
        if tag == 273:
            value = data_off
        packed = struct.pack("<HH", value, 0) if typ == 3 else struct.pack("<I", value)
        out += struct.pack("<HHI", tag, typ, 1) + packed
    out += struct.pack("<I", 0) + data
    img = cv2.imdecode(np.frombuffer(bytes(out), np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None or img.shape[:2] != (height, width):
        raise ValueError("LZW tile decode failed")
    return img


class GeoTiffTiles:
    """Windowed reads from a tiled, 8-bit, pixel-interleaved GeoTIFF."""

    def __init__(self, source):
        self.src = source
        head = source.read(0, 16)
        self.bo = "<" if head[:2] == b"II" else ">"
        magic = struct.unpack(self.bo + "H", head[2:4])[0]
        self.big = magic == 43
        if self.big:
            ifd = struct.unpack(self.bo + "Q", head[8:16])[0]
            n = struct.unpack(self.bo + "Q", source.read(ifd, 8))[0]
            raw, esz, val_len = source.read(ifd + 8, n * 20), 20, 8
        elif magic == 42:
            ifd = struct.unpack(self.bo + "I", head[4:8])[0]
            n = struct.unpack(self.bo + "H", source.read(ifd, 2))[0]
            raw, esz, val_len = source.read(ifd + 2, n * 12), 12, 4
        else:
            raise ValueError("not a TIFF")
        self._tags = {}
        for i in range(n):
            e = raw[i * esz:(i + 1) * esz]
            tag, typ = struct.unpack(self.bo + "HH", e[:4])
            count = struct.unpack(self.bo + ("Q" if self.big else "I"), e[4:12] if self.big else e[4:8])[0]
            self._tags[tag] = (typ, count, e[esz - val_len:])
        self.width = self._scalar(256)
        self.height = self._scalar(257)
        self.tile_w = self._scalar(322)
        self.tile_h = self._scalar(323)
        self.spp = self._scalar(277, 1)
        self.compression = self._scalar(259, 1)
        if self._scalar(317, 1) != 1:
            raise ValueError("TIFF predictors are not supported")
        if self._scalar(284, 1) != 1 and self.spp != 1:
            raise ValueError("planar TIFF layouts are not supported")
        if any(b != 8 for b in self._values(258)):
            raise ValueError("only 8-bit TIFFs are supported")
        if self.compression not in (1, 5, 8, 32946):
            raise ValueError(f"unsupported TIFF compression {self.compression}")
        sx, sy = self._values(33550)[:2]
        tp = self._values(33922)
        self.dx, self.dy = float(sx), float(sy)
        self.west = float(tp[3]) - float(tp[0]) * self.dx
        self.north = float(tp[4]) + float(tp[1]) * self.dy
        self.tiles_across = -(-self.width // self.tile_w)
        self.tiles_down = -(-self.height // self.tile_h)

    def _values(self, tag, start=0, count=None):
        typ, n, field = self._tags[tag]
        fmt = _TYPE_FMT[typ]
        size = struct.calcsize(fmt)
        count = n - start if count is None else count
        if n * size <= len(field):
            data = field[start * size:(start + count) * size]
        else:
            base = struct.unpack(self.bo + ("Q" if self.big else "I"), field[:8 if self.big else 4])[0]
            data = self.src.read(base + start * size, count * size)
        return struct.unpack(self.bo + fmt * count, data)

    def _scalar(self, tag, default=None):
        if tag not in self._tags:
            if default is None:
                raise KeyError(tag)
            return default
        return self._values(tag)[0]

    def _decode(self, data):
        if self.compression == 1:
            raw = data
        elif self.compression in (8, 32946):
            raw = zlib.decompress(data)
        else:
            return _lzw_decode(data, self.tile_w * self.spp, self.tile_h).reshape(
                self.tile_h, self.tile_w, self.spp)
        return np.frombuffer(raw, np.uint8)[: self.tile_h * self.tile_w * self.spp].reshape(
            self.tile_h, self.tile_w, self.spp)

    def tile_row(self, tile_row, col0=0, col1=None, sample=0):
        """Decode tiles ``col0..col1`` of one tile row; returns a list of 2-D arrays."""
        col1 = self.tiles_across if col1 is None else col1
        first = tile_row * self.tiles_across + col0
        count = col1 - col0
        offs = self._values(324, first, count)
        cnts = self._values(325, first, count)
        lo = min(o for o, c in zip(offs, cnts) if c) if any(cnts) else 0
        hi = max(o + c for o, c in zip(offs, cnts)) if any(cnts) else 0
        blob = self.src.read(lo, hi - lo) if hi > lo else b""
        out = []
        for o, c in zip(offs, cnts):
            if not c:
                out.append(np.zeros((self.tile_h, self.tile_w), np.uint8))
                continue
            out.append(self._decode(blob[o - lo:o - lo + c])[:, :, sample])
        return out

    def read_window(self, row0, col0, rows, cols, sample=0):
        """Read pixels ``[row0:row0+rows, col0:col0+cols]``; outside the raster is 0."""
        out = np.zeros((rows, cols), np.uint8)
        r_lo, r_hi = max(0, row0), min(self.height, row0 + rows)
        c_lo, c_hi = max(0, col0), min(self.width, col0 + cols)
        if r_lo >= r_hi or c_lo >= c_hi:
            return out
        tc0, tc1 = c_lo // self.tile_w, (c_hi - 1) // self.tile_w + 1
        for tr in range(r_lo // self.tile_h, (r_hi - 1) // self.tile_h + 1):
            tiles = self.tile_row(tr, tc0, tc1, sample)
            strip = np.concatenate(tiles, axis=1)
            y0 = tr * self.tile_h
            x0 = tc0 * self.tile_w
            ya, yb = max(r_lo, y0), min(r_hi, y0 + self.tile_h)
            out[ya - row0:yb - row0, c_lo - col0:c_hi - col0] = strip[ya - y0:yb - y0, c_lo - x0:c_hi - x0]
        return out


# ── Gridded leaf-type layer ──────────────────────────────────────────────────
class LeafGrid:
    """Unified codes on a regular lat/lon grid (``west``/``north`` = outer edges)."""

    def __init__(self, name, codes, west, north, dx, dy):
        self.name = name
        self.codes = codes
        self.west, self.north = float(west), float(north)
        self.dx, self.dy = float(dx), float(dy)

    def index(self, lat, lon):
        col = np.floor((np.asarray(lon) - self.west) / self.dx).astype(np.int64)
        row = np.floor((self.north - np.asarray(lat)) / self.dy).astype(np.int64)
        h, w = self.codes.shape
        ok = (row >= 0) & (row < h) & (col >= 0) & (col < w)
        return row.clip(0, h - 1), col.clip(0, w - 1), ok

    def sample(self, lat, lon):
        r, c, ok = self.index(lat, lon)
        return np.where(ok, self.codes[r, c], NONE).astype(np.uint8)

    def window(self, south, north, west, east):
        """Return the sub-grid covering the bounds (clipped to this grid)."""
        h, w = self.codes.shape
        r0 = max(0, int(math.floor((self.north - north) / self.dy)))
        r1 = min(h, int(math.ceil((self.north - south) / self.dy)))
        c0 = max(0, int(math.floor((west - self.west) / self.dx)))
        c1 = min(w, int(math.ceil((east - self.west) / self.dx)))
        if r0 >= r1 or c0 >= c1:
            return None
        return LeafGrid(self.name, self.codes[r0:r1, c0:c1],
                        self.west + c0 * self.dx, self.north - r0 * self.dy, self.dx, self.dy)

    def filled(self, limit_m):
        """Nearest typed code for every pixel within ``limit_m`` (0 beyond)."""
        valid = self.codes > 0
        if not valid.any():
            return None
        if valid.all():
            return self
        dist, labels = cv2.distanceTransformWithLabels(
            (~valid).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        # DIST_LABEL_PIXEL numbers the zero (= typed) pixels in raster order from 1.
        valid_codes = self.codes[valid]
        filled = valid_codes[np.clip(labels - 1, 0, len(valid_codes) - 1)]
        lat_c = math.radians(self.north - self.codes.shape[0] * self.dy / 2)
        px_m = 0.5 * (self.dy * 110540.0 + self.dx * 111320.0 * max(0.05, math.cos(lat_c)))
        filled = np.where(dist * px_m <= limit_m, filled, NONE).astype(np.uint8)
        return LeafGrid(self.name, filled, self.west, self.north, self.dx, self.dy)

    def typed_fraction(self):
        return float((self.codes > 0).mean()) if self.codes.size else 0.0


def _load_grid(path, name):
    with np.load(path) as d:
        return LeafGrid(name, d["codes"], float(d["west"]), float(d["north"]),
                        float(d["dx"]), float(d["dy"]))


def _save_grid(path, grid):
    _atomic_save_npz(path, codes=grid.codes, west=grid.west, north=grid.north,
                     dx=grid.dx, dy=grid.dy)


def _mosaic(name, parts, south, north, west, east, dx, dy):
    """Paste grids sharing one pixel size into a window covering the bounds.

    The window origin snaps to the first part's pixel grid so pixels are
    copied, not shifted by up to half a pixel.
    """
    ref = parts[0]
    west = ref.west + math.floor((west - ref.west) / dx) * dx
    north = ref.north - math.floor((ref.north - north) / dy) * dy
    W = int(math.ceil((east - west) / dx))
    H = int(math.ceil((north - south) / dy))
    out = np.zeros((H, W), np.uint8)
    for g in parts:
        r0 = int(round((north - g.north) / dy))
        c0 = int(round((g.west - west) / dx))
        h, w = g.codes.shape
        ya, yb = max(0, r0), min(H, r0 + h)
        xa, xb = max(0, c0), min(W, c0 + w)
        if ya < yb and xa < xb:
            sub = g.codes[ya - r0:yb - r0, xa - c0:xb - c0]
            tgt = out[ya:yb, xa:xb]
            np.copyto(tgt, sub, where=sub > 0)
    return LeafGrid(name, out, west, north, dx, dy)


# ── GLC_FCS30D ───────────────────────────────────────────────────────────────
def glc_block_for(lat, lon):
    """Return (tile label, zip name, west, north) of the 5-degree GLC block."""
    lon_left = int(math.floor(lon / 5.0) * 5)
    lat_top = int(math.floor(lat / 5.0) * 5 + 5)
    lon_lbl = f"E{lon_left}" if lon_left >= 0 else f"W{-lon_left}"
    lat_lbl = f"N{lat_top}" if lat_top >= 0 else f"S{-lat_top}"
    if lon_left >= 0:
        a = lon_left - lon_left % 10
        zip_name = f"E{a}-E{a + 5}"
    else:
        w = -lon_left
        zip_name = f"W{w}-W{w + 5}" if w % 10 == 5 else f"W{w - 5}-W{w}"
    return f"{lon_lbl}{lat_lbl}", zip_name, lon_left, lat_top


class _HttpRangeFile(io.RawIOBase):
    """Seekable read-only HTTP file, enough for zipfile's central directory."""

    def __init__(self, url, session):
        self.url, self.s = url, session
        r = self.s.head(url, allow_redirects=True, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        self.size = int(r.headers["content-length"])
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        r = _http_get(self.s, self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        d = r.content
        b[:len(d)] = d
        self.pos += len(d)
        return len(d)


def _stream_zip_member(url, info, dst_path, session, log=_log_default):
    """Download one deflated zip member with ranged GETs, resuming after a stall.

    The inflater keeps its state across requests, so a resumed request just
    feeds it the compressed bytes from where the previous one stopped.
    """
    import requests

    hdr = _http_get(session, url, headers={
        "Range": f"bytes={info.header_offset}-{info.header_offset + 29}"}).content
    name_len, extra_len = struct.unpack("<HH", hdr[26:30])
    start = info.header_offset + 30 + name_len + extra_len
    end = start + info.compress_size - 1
    dec = zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None
    pos = start
    fruitless = 0
    with open(dst_path, "wb") as dst:
        while pos <= end:
            before = pos
            problem = None
            try:
                with session.get(url, headers={"Range": f"bytes={pos}-{end}"}, stream=True,
                                 timeout=(HTTP_TIMEOUT, STREAM_STALL_S)) as r:
                    r.raise_for_status()
                    if r.status_code != 206:
                        raise IOError(f"server ignored the byte range (HTTP {r.status_code})")
                    for chunk in r.iter_content(1 << 20):
                        dst.write(dec.decompress(chunk) if dec else chunk)
                        pos += len(chunk)
            except (requests.RequestException, ConnectionError, TimeoutError) as exc:
                problem = exc.__class__.__name__
            if pos > end:
                break
            if problem is None:
                problem = "connection closed early"
            fruitless = 0 if pos > before else fruitless + 1
            if fruitless > STREAM_RESUMES:
                raise IOError(f"download of {info.filename} stalled at "
                              f"{(pos - start) / 1e6:.0f} of {info.compress_size / 1e6:.0f} MB")
            log(f"  [LeafType] download stopped at {(pos - start) / 1e6:.0f} of "
                f"{info.compress_size / 1e6:.0f} MB ({problem}); resuming")
            if fruitless:
                time.sleep(min(30, 2 ** fruitless))
        if dec:
            dst.write(dec.flush())
    if os.path.getsize(dst_path) != info.file_size:
        raise IOError(f"incomplete download of {info.filename}")


def _glc_slice_path(cache_dir, lat, lon):
    return os.path.join(cache_dir, "glc", f"glc2022_{_tile_key(lat, lon)}.npz")


def _prepare_glc_block(cache_dir, lat, lon, log):
    """Download one 5-degree GLC block and cache its 25 one-degree slices."""
    label, zip_name, lon_left, lat_top = glc_block_for(lat + 0.5, lon + 0.5)
    glc_dir = os.path.join(cache_dir, "glc")
    done = os.path.join(glc_dir, f"block_{label}.done")
    missing = os.path.join(glc_dir, f"block_{label}.missing")
    if os.path.exists(done) or os.path.exists(missing):
        return
    os.makedirs(glc_dir, exist_ok=True)
    session = _session()
    url = GLC_URL.format(zip=zip_name)
    member = f"GLC_FCS30D_20002022_{label}_Annual.tif"
    z = zipfile.ZipFile(_HttpRangeFile(url, session))
    try:
        info = z.getinfo(member)
    except KeyError:
        open(missing, "w").close()   # all-ocean block: nothing published
        log(f"  [LeafType] GLC_FCS30D: no block {label} (ocean)")
        return
    raw = os.path.join(glc_dir, f"{label}.{os.getpid()}.download.tif")
    t0 = time.time()
    log(f"  [LeafType] GLC_FCS30D: downloading block {label} "
        f"({info.compress_size / 1e6:.0f} MB, once per 5x5 degrees) ...")
    try:
        _stream_zip_member(url, info, raw, session, log)
        src = _FileBytes(raw)
        try:
            tif = GeoTiffTiles(src)
            block = np.zeros((tif.height, tif.width), np.uint8)
            for tr in range(tif.tiles_down):
                strip = np.concatenate(tif.tile_row(tr, sample=GLC_BAND_2022), axis=1)
                y0 = tr * tif.tile_h
                rows = min(tif.tile_h, tif.height - y0)
                block[y0:y0 + rows] = GLC_LUT[strip[:rows, :tif.width]]
            west, north, dx, dy = tif.west, tif.north, tif.dx, tif.dy
        finally:
            src.close()
    finally:
        if os.path.exists(raw):
            os.remove(raw)
    full = LeafGrid("GLC_FCS30D", block, west, north, dx, dy)
    for la in range(lat_top - 5, lat_top):
        for lo in range(lon_left, lon_left + 5):
            part = full.window(la, la + 1, lo, lo + 1)
            if part is not None:
                _save_grid(_glc_slice_path(cache_dir, la, lo), part)
    open(done, "w").close()
    log(f"  [LeafType] GLC_FCS30D: block {label} cached in {time.time() - t0:.0f}s")


def glc_grid(cache_dir, lat, lon, log=_log_default, online=True):
    """GLC_FCS30D 2022 codes for the tile plus margin (from cached slices)."""
    if online:
        _prepare_glc_block(cache_dir, lat, lon, log)
    south, north = lat - TILE_MARGIN_DEG, lat + 1 + TILE_MARGIN_DEG
    west, east = lon - TILE_MARGIN_DEG, lon + 1 + TILE_MARGIN_DEG
    parts = []
    for la in (lat - 1, lat, lat + 1):
        for lo in (lon - 1, lon, lon + 1):
            path = _glc_slice_path(cache_dir, la, lo)
            if os.path.exists(path):
                g = _load_grid(path, "GLC_FCS30D").window(south, north, west, east)
                if g is not None:
                    parts.append(g)
    if not parts:
        return None
    return _mosaic("GLC_FCS30D", parts, south, north, west, east, GLC_PX_DEG, GLC_PX_DEG)


# ── CGLS-LC100 forest type ───────────────────────────────────────────────────
def cgls_grid(cache_dir, lat, lon, log=_log_default, online=True):
    path = os.path.join(cache_dir, "cgls", f"cgls2019_{_tile_key(lat, lon)}.npz")
    if os.path.exists(path):
        return _load_grid(path, "CGLS-LC100")
    if not online or lat >= 80 or lat + 1 <= -60:
        return None
    tif = GeoTiffTiles(_HttpBytes(CGLS_URL))
    south, north = lat - TILE_MARGIN_DEG, lat + 1 + TILE_MARGIN_DEG
    west, east = lon - TILE_MARGIN_DEG, lon + 1 + TILE_MARGIN_DEG
    row0 = int(math.floor((tif.north - north) / tif.dy))
    col0 = int(math.floor((west - tif.west) / tif.dx))
    rows = int(math.ceil((north - south) / tif.dy)) + 1
    cols = int(math.ceil((east - west) / tif.dx)) + 1
    raw = tif.read_window(row0, col0, rows, cols)
    grid = LeafGrid("CGLS-LC100", CGLS_LUT[raw], tif.west + col0 * tif.dx,
                    tif.north - row0 * tif.dy, tif.dx, tif.dy)
    _save_grid(path, grid)
    return grid


# ── Copernicus HRL Dominant Leaf Type (Europe) ───────────────────────────────
def hrl_grid(cache_dir, lat, lon, log=_log_default, online=True):
    path = os.path.join(cache_dir, "hrl", f"hrl2018_{_tile_key(lat, lon)}.npz")
    none_marker = path + ".none"
    if os.path.exists(path):
        return _load_grid(path, "HRL-DLT-2018")
    if os.path.exists(none_marker) or not online:
        return None
    es, en, ew, ee = HRL_ENVELOPE
    if lat + 1 <= es or lat >= en or lon + 1 <= ew or lon >= ee:
        return None
    from PIL import Image

    south, north = lat - TILE_MARGIN_DEG, lat + 1 + TILE_MARGIN_DEG
    west, east = lon - TILE_MARGIN_DEG, lon + 1 + TILE_MARGIN_DEG
    # The image service keeps pixels square in degrees (it widens the extent
    # otherwise), so request square pixels and still georeference each chunk
    # from the extent it reports back.
    dx = dy = HRL_PX_DEG
    W = int(math.ceil((east - west) / dx))
    H = int(math.ceil((north - south) / dy))
    codes = np.zeros((H, W), np.uint8)
    session = _session()
    nx, ny = -(-W // HRL_MAX_PX), -(-H // HRL_MAX_PX)
    for iy in range(ny):
        for ix in range(nx):
            c0, c1 = ix * W // nx, (ix + 1) * W // nx
            r0, r1 = iy * H // ny, (iy + 1) * H // ny
            bbox = (west + c0 * dx, north - r1 * dy, west + c1 * dx, north - r0 * dy)
            meta = _http_get(session, HRL_URL, params=dict(
                bbox=",".join(f"{v:.9f}" for v in bbox), bboxSR=4326, imageSR=4326,
                size=f"{c1 - c0},{r1 - r0}", format="tiff", pixelType="U8",
                compression="LZ77", interpolation="RSP_NearestNeighbor",
                noData=255, f="json")).json()
            if "href" not in meta or "extent" not in meta:
                raise IOError(f"HRL export failed: {meta.get('error')}")
            arr = np.array(Image.open(io.BytesIO(_http_get(session, meta["href"]).content)))
            ext = meta["extent"]
            chunk = LeafGrid("HRL", HRL_LUT[arr], ext["xmin"], ext["ymax"],
                             (ext["xmax"] - ext["xmin"]) / arr.shape[1],
                             (ext["ymax"] - ext["ymin"]) / arr.shape[0])
            lats = north - (np.arange(r0, r1) + 0.5) * dy
            lons = west + (np.arange(c0, c1) + 0.5) * dx
            codes[r0:r1, c0:c1] = chunk.sample(lats[:, None], lons[None, :])
    if not codes.any():
        os.makedirs(os.path.dirname(none_marker), exist_ok=True)
        open(none_marker, "w").close()
        return None
    grid = LeafGrid("HRL-DLT-2018", codes, west, north, dx, dy)
    _save_grid(path, grid)
    return grid


GRID_SOURCES = (
    ("HRL-DLT-2018", hrl_grid),
    ("GLC_FCS30D", glc_grid),
    ("CGLS-LC100", cgls_grid),
)
# Direct-coverage priority; OSM-explicit sits after the 10 m European map and
# ahead of the global 30 m / 100 m products, which call exotic pine plantations
# broadleaf where a mapper has tagged them needleleaved.
CHAIN = ("HRL-DLT-2018", "OSM-explicit", "GLC_FCS30D", "CGLS-LC100")


# ── Per-tile resolver ────────────────────────────────────────────────────────
def _mercator_row_lats(lat_n, lat_s, rows):
    y_n = math.asinh(math.tan(math.radians(lat_n)))
    y_s = math.asinh(math.tan(math.radians(lat_s)))
    y = y_n + (np.arange(rows) + 0.5) / rows * (y_s - y_n)
    return np.degrees(np.arctan(np.sinh(y)))


def _explicit_osm_records(forest_records):
    """simHeaven polygons whose family is explicitly broad or coni."""
    out = []
    for rec in forest_records or ():
        if not isinstance(rec, dict):
            continue
        path = (rec.get("path") or "").replace("\\", "/").strip().lower().lstrip("/")
        code = PATH_CODES.get(path)
        if code not in (BROAD, CONI) or len(rec.get("pts") or ()) < 3:
            continue
        bounds = rec.get("_bounds")
        if bounds is None:
            pts = np.asarray(rec["pts"], np.float64)
            bounds = (pts[:, 1].min(), pts[:, 1].max(), pts[:, 0].min(), pts[:, 0].max())
        out.append((code, bounds, rec["pts"]))
    return out


def _signature_part(grid):
    return f"{grid.name}:{zlib.crc32(grid.codes.tobytes()) & 0xFFFFFFFF:08x}"


class TileLeafTypes:
    """Leaf-type zones for the textures of one one-degree tile.

    ``available`` is False when no source covers the tile (offline with an
    empty cache, open ocean, ...); callers then keep their previous behaviour.
    """

    def __init__(self, lat, lon, cache_dir, forest_records=(), log=_log_default,
                 online=True, grid_sources=GRID_SOURCES, warning_log=None):
        self.lat, self.lon = int(math.floor(lat)), int(math.floor(lon))
        self.log = log
        self.grids = {}
        # A source's own messages are about downloads: rare, and slow enough
        # that a quiet log would look hung, so they print like warnings.
        fetch_log = warning_log or log
        t0 = time.time()
        for name, fn in grid_sources:
            t = time.time()
            try:
                grid = fn(cache_dir, self.lat, self.lon, log=fetch_log, online=online)
            except Exception as exc:
                (warning_log or log)(f"  [LeafType] {name}: unavailable ({exc})")
                grid = None
            if grid is not None and grid.codes.any():
                self.grids[name] = grid
                log(f"  [LeafType] {name}: {100 * grid.typed_fraction():.0f}% of tile typed "
                    f"({time.time() - t:.1f}s)")
            else:
                log(f"  [LeafType] {name}: no coverage")
        self.osm = _explicit_osm_records(forest_records)
        if self.osm:
            self._osm_bounds = np.array([b for _, b, _ in self.osm], np.float64)
            log(f"  [LeafType] OSM-explicit: {len(self.osm)} broad/coni polygons")
        self.chain = [n for n in CHAIN if n in self.grids or (n == "OSM-explicit" and self.osm)]
        self.available = bool(self.grids)
        self.fallback_code = self._dominant_code()
        parts = [f"v{VERSION}"] + [_signature_part(self.grids[n]) for n in CHAIN if n in self.grids]
        if self.osm:
            parts.append(f"osm:{len(self.osm)}:{zlib.crc32(self._osm_bounds.round(7).tobytes()) & 0xFFFFFFFF:08x}")
        self.signature = "|".join(parts)
        log(f"  [LeafType] chain: {' > '.join(self.chain) or 'none'} "
            f"({time.time() - t0:.1f}s)")

    def _dominant_code(self):
        for name in CHAIN:
            g = self.grids.get(name)
            if g is None:
                continue
            counts = np.bincount(g.codes.ravel(), minlength=4)[1:4]
            if counts.sum():
                return int(np.argmax(counts)) + 1
        return MIXED

    def _osm_image(self, lon_w, lon_e, lat_n, lat_s, rows, cols):
        img = np.zeros((rows, cols), np.uint8)
        if not self.osm:
            return img
        b = self._osm_bounds
        hit = np.where((b[:, 1] >= lat_s) & (b[:, 0] <= lat_n) & (b[:, 3] >= lon_w) & (b[:, 2] <= lon_e))[0]
        if not len(hit):
            return img
        y_n = math.asinh(math.tan(math.radians(lat_n)))
        y_s = math.asinh(math.tan(math.radians(lat_s)))
        for code in (BROAD, CONI):          # conifer drawn last wins overlaps
            polys = []
            for i in hit:
                c, _, pts = self.osm[i]
                if c != code:
                    continue
                a = np.asarray(pts, np.float64)
                x = (a[:, 0] - lon_w) / (lon_e - lon_w) * cols
                yl = np.arcsinh(np.tan(np.radians(np.clip(a[:, 1], -85.0, 85.0))))
                y = (y_n - yl) / (y_n - y_s) * rows
                polys.append(np.round(np.stack([x, y], 1) * 8).astype(np.int32))
            if polys:
                cv2.fillPoly(img, polys, int(code), lineType=cv2.LINE_8, shift=3)
        return img

    def zone_for_texture(self, lat_n, lat_s, lon_w, lon_e, out_h, out_w):
        """uint8 (out_h, out_w) zone codes 1..3 for a Web Mercator texture."""
        if not self.available:
            return None
        width_m = (lon_e - lon_w) * 111320.0 * max(0.05, math.cos(math.radians(0.5 * (lat_n + lat_s))))
        height_m = (lat_n - lat_s) * 110540.0
        cols = int(min(MAX_ANALYSIS_PX, max(16, round(width_m / ANALYSIS_M))))
        rows = int(min(MAX_ANALYSIS_PX, max(16, round(height_m / ANALYSIS_M))))
        lat_rows = _mercator_row_lats(lat_n, lat_s, rows)
        lon_cols = lon_w + (np.arange(cols) + 0.5) / cols * (lon_e - lon_w)
        LATS = np.repeat(lat_rows[:, None], cols, 1)
        LONS = np.repeat(lon_cols[None, :], rows, 0)

        px = np.zeros((rows, cols), np.uint8)
        for name in self.chain:
            img = (self._osm_image(lon_w, lon_e, lat_n, lat_s, rows, cols)
                   if name == "OSM-explicit" else self.grids[name].sample(LATS, LONS))
            np.copyto(px, img, where=(px == 0) & (img > 0))
        if (px == 0).any():
            m_lat = FILL_LIMIT_M / 110540.0
            m_lon = FILL_LIMIT_M / (111320.0 * max(0.05, math.cos(math.radians(lat_n))))
            for name in self.chain:
                if name == "OSM-explicit":
                    continue
                win = self.grids[name].window(lat_s - m_lat, lat_n + m_lat, lon_w - m_lon, lon_e + m_lon)
                filled = win.filled(FILL_LIMIT_M) if win is not None else None
                if filled is None:
                    continue
                img = filled.sample(LATS, LONS)
                np.copyto(px, img, where=(px == 0) & (img > 0))
                if not (px == 0).any():
                    break
        px[px == 0] = self.fallback_code

        lab_m = max(1e-3, 0.5 * (width_m / cols + height_m / rows))
        k = max(1, int(round(SMOOTH_M / lab_m)))
        ksize = (2 * k + 1, 2 * k + 1)
        b = cv2.boxFilter((px == BROAD).astype(np.float32), -1, ksize)
        c = cv2.boxFilter((px == CONI).astype(np.float32), -1, ksize)
        m = cv2.boxFilter((px == MIXED).astype(np.float32), -1, ksize)
        bf = (b + 0.5 * m) / np.maximum(1e-6, b + c + m)
        zone = np.where(bf >= BROAD_HI, BROAD, np.where(bf <= BROAD_LO, CONI, MIXED)).astype(np.uint8)
        return cv2.resize(zone, (int(out_w), int(out_h)), interpolation=cv2.INTER_NEAREST)

    def summary(self):
        return {"chain": list(self.chain), "signature": self.signature,
                "fallback": {BROAD: "broad", CONI: "coni", MIXED: "mixed"}[self.fallback_code]}


# ── X-Plane 12 forests: climate band per location ───────────────────────────
# X-Plane's own forests are split by temperature x moisture band.  The band
# X-Plane itself uses at a spot is read from the default global scenery DSF
# (nearest g8 polygon); Köppen-Geiger is the fallback where it is missing.
BAND_PX_DEG = 0.005      # ~500 m: bands change over tens of kilometres
_G8_BAND_RE = None


def band_id(temp, moist):
    return 1 + FOREST_ASSETS.G8_TEMPS.index(temp) * 4 + FOREST_ASSETS.G8_MOISTS.index(moist)


def band_name(bid):
    t, m = divmod(int(bid) - 1, 4)
    return FOREST_ASSETS.G8_TEMPS[t], FOREST_ASSETS.G8_MOISTS[m]


def g8_band_of_path(path):
    """Band id of any lib/g8 polygon definition carrying a climate suffix, else 0."""
    global _G8_BAND_RE
    if _G8_BAND_RE is None:
        import re

        _G8_BAND_RE = re.compile(
            r"^lib/g8/[a-z_]+?_(vcld|cld|tmp|wrm|hot|vhot)_(dry|sdry|wet|rain)\.for$")
    m = _G8_BAND_RE.match((path or "").strip().lower())
    return band_id(m.group(1), m.group(2)) if m else 0


# Köppen-Geiger class id (O4_Climate_Regions) -> (temperature, moisture).
KOPPEN_BANDS = {
    1: ("vhot", "rain"), 2: ("vhot", "wet"), 3: ("hot", "sdry"),
    4: ("hot", "dry"), 5: ("tmp", "dry"), 6: ("hot", "sdry"), 7: ("tmp", "sdry"),
    8: ("wrm", "sdry"), 9: ("tmp", "sdry"), 10: ("cld", "sdry"),
    11: ("wrm", "sdry"), 12: ("tmp", "sdry"), 13: ("cld", "sdry"),
    14: ("wrm", "wet"), 15: ("tmp", "wet"), 16: ("cld", "wet"),
    17: ("tmp", "sdry"), 18: ("cld", "sdry"), 19: ("vcld", "sdry"), 20: ("vcld", "sdry"),
    21: ("tmp", "sdry"), 22: ("cld", "sdry"), 23: ("vcld", "sdry"), 24: ("vcld", "sdry"),
    25: ("tmp", "wet"), 26: ("tmp", "wet"), 27: ("vcld", "wet"), 28: ("vcld", "wet"),
    29: ("vcld", "dry"), 30: ("vcld", "dry"),
}


def koppen_band(lat, lon):
    try:
        class_id = CLIMATE.koppen_class_id(lat, lon)
    except Exception:
        class_id = None
    return band_id(*KOPPEN_BANDS.get(class_id, ("tmp", "wet")))


def xp_global_dsf_path(custom_scenery_dir, lat, lon):
    """The default X-Plane global scenery DSF for a tile, when installed."""
    if not custom_scenery_dir:
        return None
    lat, lon = int(math.floor(lat)), int(math.floor(lon))
    grid = _tile_key(lat // 10 * 10, lon // 10 * 10)
    gs_root = os.path.join(os.path.dirname(os.path.normpath(custom_scenery_dir)), "Global Scenery")
    try:
        packs = sorted(os.listdir(gs_root))
    except OSError:
        return None
    # The full global scenery first, demo areas after.
    packs.sort(key=lambda name: (0 if "global scenery" in name.lower() else 1, name))
    for pack in packs:
        path = os.path.join(gs_root, pack, "Earth nav data", grid, _tile_key(lat, lon) + ".dsf")
        if os.path.isfile(path):
            return path
    return None


def rasterize_g8_bands(lines, west, north, dx, dy, width, height):
    """Rasterize the g8 forest polygons of DSFTool text ``lines`` by band id."""
    img = np.zeros((height, width), np.uint8)
    bands = []
    cur = 0
    pts = None
    for raw in lines:
        if not raw or raw[0] not in (66, 69, 80):   # B, E, P
            continue
        if raw.startswith(b"POLYGON_POINT"):
            if pts is not None:
                parts = raw.split()
                pts.append((float(parts[1]), float(parts[2])))
        elif raw.startswith(b"POLYGON_DEF"):
            bands.append(g8_band_of_path(raw.split(None, 1)[1].decode("utf-8", "ignore")))
        elif raw.startswith(b"BEGIN_POLYGON"):
            idx = int(raw.split()[1])
            cur = bands[idx] if idx < len(bands) else 0
        elif raw.startswith(b"BEGIN_WINDING"):
            pts = [] if cur else None
        elif raw.startswith(b"END_WINDING"):
            if pts and len(pts) >= 3:
                a = np.asarray(pts, np.float64)
                xy = np.stack([(a[:, 0] - west) / dx, (north - a[:, 1]) / dy], 1)
                cv2.fillPoly(img, [np.round(xy * 8).astype(np.int32)], int(cur), shift=3)
            pts = None
        elif raw.startswith(b"END_POLYGON"):
            cur = 0
            pts = None
    return img


def xp_band_grid(cache_dir, lat, lon, custom_scenery_dir, dsftool_path, log=_log_default):
    """Band ids of X-Plane's own forests over the tile (+margin), or None."""
    dsf = xp_global_dsf_path(custom_scenery_dir, lat, lon)
    if not dsf or not dsftool_path or not os.path.exists(dsftool_path):
        return None
    st = os.stat(dsf)
    stamp = f"{st.st_size}:{int(st.st_mtime)}"
    path = os.path.join(cache_dir, "xpbands", f"xpbands_{_tile_key(lat, lon)}.npz")
    if os.path.exists(path):
        with np.load(path) as d:
            if str(d["stamp"]) == stamp:
                return LeafGrid("XP-global", d["codes"], float(d["west"]), float(d["north"]),
                                float(d["dx"]), float(d["dy"]))
    import subprocess
    import sys

    t0 = time.time()
    west, north = lon - TILE_MARGIN_DEG, lat + 1 + TILE_MARGIN_DEG
    size = int(round((1 + 2 * TILE_MARGIN_DEG) / BAND_PX_DEG))
    flags = 0x08000000 if sys.platform == "win32" else 0   # CREATE_NO_WINDOW
    proc = subprocess.Popen([dsftool_path, "--dsf2text", dsf, "-"], stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, creationflags=flags)
    try:
        codes = rasterize_g8_bands(proc.stdout, west, north, BAND_PX_DEG, BAND_PX_DEG, size, size)
    finally:
        proc.stdout.close()
        proc.wait()
    grid = LeafGrid("XP-global", codes, west, north, BAND_PX_DEG, BAND_PX_DEG)
    _atomic_save_npz(path, codes=codes, west=west, north=north, dx=BAND_PX_DEG,
                     dy=BAND_PX_DEG, stamp=stamp)
    log(f"  [XPForests] climate bands from {os.path.basename(dsf)} "
        f"({100 * grid.typed_fraction():.0f}% forest, {time.time() - t0:.1f}s)")
    return grid


class XPForestMapper:
    """Turn a tree family into X-Plane 12's forest for the local climate band."""

    def __init__(self, lat, lon, cache_dir, custom_scenery_dir, dsftool_path,
                 log=_log_default, warning_log=None):
        self.lat, self.lon = int(math.floor(lat)), int(math.floor(lon))
        self.fallback_band = koppen_band(self.lat + 0.5, self.lon + 0.5)
        grid = None
        try:
            grid = xp_band_grid(cache_dir, self.lat, self.lon, custom_scenery_dir,
                                dsftool_path, log=log)
        except Exception as exc:
            (warning_log or log)(f"  [XPForests] global scenery unreadable ({exc})")
        self.grid = grid.filled(float("inf")) if grid is not None else None
        if self.grid is None:
            t, m = band_name(self.fallback_band)
            log(f"  [XPForests] no global scenery forests — Köppen band {t}_{m}")
        sig = _signature_part(self.grid) if self.grid is not None else f"koppen:{self.fallback_band}"
        self.signature = f"xpg8-v1|{sig}"

    def band_at(self, lon, lat):
        if self.grid is not None:
            bid = int(self.grid.sample(lat, lon))
            if bid:
                return band_name(bid)
        return band_name(self.fallback_band)

    def map_path(self, path, lon, lat, veg_type=None):
        family = FOREST_ASSETS.tree_family(path)
        if family is None:
            return path
        temp, moist = self.band_at(lon, lat)
        return FOREST_ASSETS.g8_forest_path(
            family, temp, moist, orchard=(veg_type == "orchard_or_plantation"))


def default_cache_dir(tile_cache_dir):
    """``<cache root>/_leaf_type`` derived from a per-tile ``<cache root>/<grid>/<tile>`` dir."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(tile_cache_dir)))
    return os.path.join(root, "_leaf_type")
