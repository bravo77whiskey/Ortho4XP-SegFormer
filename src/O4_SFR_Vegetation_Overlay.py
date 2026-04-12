"""
O4_SFR_Vegetation_Overlay.py — Per-DDS vegetation DSF overlay from SegFormer inference.

Reuses the inference cache written by the SFR building overlay.

Pipeline
--------
1. Discover ZL16 DDS files in tex_dir
2. Load cached SegFormer veg_map per DDS at native resolution (~2.4 m/px at ZL16)
3. Per DDS: extract CLASS_TREE / CLASS_RANGELAND masks
4. Morphological close → open (kernel sizes in metres)
5. Vectorise (OpenCV contours) → Douglas-Peucker → lat/lon polygons
6. Per polygon: shape ratio → area-fill | treeline
7. Pixel fill fraction → Global Forests v2 .for file + DSF density
8. Write DSF text, compile with DSFTool

Usage:
    python src/scripts/generate_veg_overlay.py <tex_dir> <lat> <lon> [options]

Options:
    --cache-dir DIR   Inference cache (default: <o4xp_root>/SFR_cache/<tile>)
    --close-m  M      Morphological close radius in metres (default 10)
    --open-m   M      Morphological open  radius in metres (default 3)
    --min-area M2     Minimum polygon area in m² (default 50)
    --simplify M      Douglas-Peucker tolerance in metres (default 3)
    --density  D      Override DSF density for all polygons (0.0–1.0)
    --no-viz          Skip overview image

Example:
    python src/scripts/generate_veg_overlay.py \\
        "dist/Ortho4XP/Tiles/zOrtho4XP_+36+101/textures" 36 101 \\
        --cache-dir "cache_zl16_36_101"
"""

import sys, os, argparse, warnings, time, math
warnings.filterwarnings('ignore')

import numpy as np
import cv2
from math import pi, atan, exp
from PIL import Image

import O4_SegFormer_Overlay as SEGFORMER
from O4_SFR_DSF_Utils import ensure_cached_dsf_text, find_simheaven_network_dsfs

# ── Defaults (all spatial params in metres) ───────────────────────────────────
CLOSE_M       = 10.0   # close kernel radius — fill gaps within a patch
OPEN_M        = 3.0    # open  kernel radius — remove sub-pixel noise
MIN_AREA_M2   = 50.0   # minimum polygon area (~7×7 m)
SIMPLIFY_M    = 3.0    # Douglas-Peucker tolerance
TREELINE_RATIO= 5.0    # perimeter² / (4π × area) ≥ this → treeline mode
MAX_RING_PTS  = 8000   # hard vertex cap per DSF winding
EXCL_BUFFER_M = 5.0    # dilation buffer around SegFormer buildings/roads before exclusion

# Per-type road exclusion half-widths in metres (one side from centre).
# These represent road surface + verge/shoulder so trees don't overlap tarmac.
ROAD_EXCL_HALF_WIDTH_M = {
    'motorway':       18, 'motorway_link':  10,
    'trunk':          14, 'trunk_link':      8,
    'primary':        10, 'primary_link':    6,
    'secondary':       8, 'secondary_link':  5,
    'tertiary':        6, 'tertiary_link':   4,
    'residential':     5, 'living_street':   4,
    'service':         4, 'unclassified':    5, 'road': 5,
    'network':         8,   # simHeaven full street network
    'rail':            7, 'light_rail':       5,
    'subway':          5, 'tram':             4,
    'narrow_gauge':    4, 'monorail':         4,
}
_ROAD_DEFAULT_HALF_M = 5   # fallback for unrecognised highway types

# DSF density (0–255 integer) + fill-mode bits (treeline +256, point +512)
_BASE_DENSITY = {
    'tree':     {100: 230, 75: 190, 50: 150, 25: 110},
    'woodland': {100: 150, 75: 110, 50:  80, 25:  50},
    'shrub':    {100:  80, 75:  60, 50:  45, 25:  30},
}

# ── Coordinate helpers ────────────────────────────────────────────────────────
def _gtile_to_wgs84(til_x, til_y, zl):
    rat_x = til_x / (2 ** (zl - 1)) - 1
    rat_y = 1 - til_y / (2 ** (zl - 1))
    return 360 / pi * atan(exp(pi * rat_y)) - 90, rat_x * 180


def dds_bounds(til_y_top, til_x_left, zl=16):
    lat_n, lon_w = _gtile_to_wgs84(til_x_left,      til_y_top,      zl)
    lat_s, lon_e = _gtile_to_wgs84(til_x_left + 16, til_y_top + 16, zl)
    return lat_n, lat_s, lon_w, lon_e


def dds_m_per_px(lat_n, lat_s, lon_w, lon_e, img_h, img_w):
    """Approximate metres per pixel for a DDS tile."""
    mid_lat  = math.radians((lat_n + lat_s) / 2)
    lon_m    = (lon_e - lon_w) * 111320 * math.cos(mid_lat)
    lat_m    = (lat_n - lat_s) * 110540
    return (lon_m / img_w + lat_m / img_h) / 2


def px_to_latlon(px, py, img_w, img_h, lat_n, lat_s, lon_w, lon_e):
    """Convert DDS pixel (px, py) → (lon, lat)."""
    lon = lon_w + px / img_w * (lon_e - lon_w)
    lat = lat_n - py / img_h * (lat_n - lat_s)
    return lon, lat


# ── Road / network data loading ──────────────────────────────────────────────
def _load_osm_roads(osm_bz2_path):
    """Parse an Ortho4XP *_big_roads.osm.bz2 file.
    Returns list of {'pts': [(lat,lon),...], 'type': highway_value}.
    """
    import bz2, xml.etree.ElementTree as ET
    if not os.path.exists(osm_bz2_path):
        return []
    with bz2.open(osm_bz2_path, 'rb') as f:
        root = ET.parse(f).getroot()
    nodes = {}
    for node in root.iter('node'):
        nodes[node.get('id')] = (float(node.get('lat')), float(node.get('lon')))
    roads = []
    for way in root.iter('way'):
        hw = None
        for tag in way.iter('tag'):
            if tag.get('k') == 'highway':
                hw = tag.get('v'); break
        if hw is None:
            continue
        pts = [nodes[nd.get('ref')] for nd in way.iter('nd') if nd.get('ref') in nodes]
        if len(pts) >= 2:
            roads.append({'pts': pts, 'type': hw})
    return roads


def _load_excl_railways(cache_path):
    """Extract railway open-way records from an Overpass exclusion .osm.bz2.
    Returns list of {'pts': [(lat,lon),...], 'type': railway_value}.
    """
    import bz2, xml.etree.ElementTree as ET
    if not os.path.exists(cache_path):
        return []
    with bz2.open(cache_path, 'rb') as f:
        root = ET.parse(f).getroot()
    nodes = {}
    for node in root.iter('node'):
        nodes[node.get('id')] = (float(node.get('lat')), float(node.get('lon')))
    rails = []
    for way in root.iter('way'):
        rw = None
        for tag in way.iter('tag'):
            if tag.get('k') == 'railway':
                rw = tag.get('v'); break
        if rw is None:
            continue
        pts = [nodes[nd.get('ref')] for nd in way.iter('nd') if nd.get('ref') in nodes]
        is_closed = len(pts) >= 4 and pts[0] == pts[-1]
        if len(pts) >= 2 and not is_closed:
            rails.append({'pts': pts, 'type': rw})
    return rails


def _load_simheaven_network(custom_scenery_dir, tile_lat, tile_lon, dsftool_path, cache_dir):
    """Parse simHeaven X-World network DSFs for road exclusion."""
    road_ways = []
    for folder_name, dsf_path in find_simheaven_network_dsfs(custom_scenery_dir, tile_lat, tile_lon):
        segment_count_before = len(road_ways)
        try:
            cached_text_path = ensure_cached_dsf_text(
                dsf_path,
                dsftool_path,
                cache_dir,
                create_no_window=SEGFORMER._CREATE_NO_WINDOW,
            )
            current_points = None
            with open(cached_text_path, "r", encoding="utf-8", errors="ignore") as text_file:
                for raw_line in text_file:
                    line = raw_line.strip()
                    if line.startswith("BEGIN_SEGMENT "):
                        parts = line.split()
                        try:
                            current_points = [(float(parts[5]), float(parts[4]))]
                        except (IndexError, ValueError):
                            current_points = None
                    elif line.startswith("SHAPE_POINT ") and current_points is not None:
                        parts = line.split()
                        try:
                            current_points.append((float(parts[2]), float(parts[1])))
                        except (IndexError, ValueError):
                            pass
                    elif line.startswith("END_SEGMENT ") and current_points is not None:
                        parts = line.split()
                        try:
                            current_points.append((float(parts[3]), float(parts[2])))
                        except (IndexError, ValueError):
                            pass
                        if len(current_points) >= 2:
                            road_ways.append({"pts": current_points, "type": "network"})
                        current_points = None
            print(f"  [simHeaven] {folder_name}: +{len(road_ways) - segment_count_before} segments")
        except Exception as exc:
            print(f"  [simHeaven] failed {dsf_path}: {exc}")
    return road_ways


def _rasterize_roads_typed(roads, lat_n, lat_s, lon_w, lon_e,
                            img_h, img_w, mpp):
    """Rasterize road/railway polylines with per-type half-widths (in metres).

    Each road segment is drawn at 2 × half_width_m / mpp pixels wide, giving
    granular exclusion zones that match the actual road surface + verge.
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    def ll_to_px(lat, lon):
        x = int((lon - lon_w) / (lon_e - lon_w) * img_w)
        y = int((lat_n - lat) / (lat_n - lat_s) * img_h)
        return (max(0, min(img_w - 1, x)),
                max(0, min(img_h - 1, y)))

    for road in roads:
        half_m   = ROAD_EXCL_HALF_WIDTH_M.get(road['type'], _ROAD_DEFAULT_HALF_M)
        thickness = max(1, int(2 * half_m / mpp))
        pts_px    = [ll_to_px(lat, lon) for lat, lon in road['pts']]
        for i in range(len(pts_px) - 1):
            cv2.line(mask, pts_px[i], pts_px[i + 1], 1, thickness=thickness)
    return mask


def _road_bounds(road):
    pts = road.get('pts', ())
    if not pts:
        return (0.0, 0.0, 0.0, 0.0)
    lats = [p[0] for p in pts]
    lons = [p[1] for p in pts]
    return min(lats), max(lats), min(lons), max(lons)


def _prepare_roads(roads):
    prepared = []
    for road in roads or []:
        pts = road.get('pts', ())
        if len(pts) < 2:
            continue
        item = dict(road)
        item['_bounds'] = _road_bounds(item)
        prepared.append(item)
    return prepared


def _roads_for_bounds(roads, lat_n, lat_s, lon_w, lon_e, pad_deg=0.0):
    if not roads:
        return []
    s = lat_s - pad_deg
    n = lat_n + pad_deg
    w = lon_w - pad_deg
    e = lon_e + pad_deg
    result = []
    for road in roads:
        r_s, r_n, r_w, r_e = road.get('_bounds') or _road_bounds(road)
        if r_n >= s and r_s <= n and r_e >= w and r_w <= e:
            result.append(road)
    return result


# ── Climate region ────────────────────────────────────────────────────────────
def _climate_region(lat):
    a = abs(lat + 0.5)
    if a < 15:  return 'tropical'
    if a < 25:  return 'subtropical'
    if a < 35:  return 'northsouth'
    if a < 55:  return 'northmiddle'
    return 'northnorth'


# ── .for file selection ───────────────────────────────────────────────────────
def _density_level(frac):
    if frac >= 0.72: return 100
    if frac >= 0.50: return 75
    if frac >= 0.28: return 50
    return 25


def _for_density_level(override):
    v = override * 100
    if v <= 37.5: return 25
    if v <= 62.5: return 50
    if v <= 87.5: return 75
    return 100


def _gfv2_path(region, ftype, dlevel, variant):
    fname = f"{region}_{ftype}_{dlevel}_y{variant}.for"
    return f"forests/{region}/{ftype}/{fname}"


def _for_entry(veg_cls, frac, shape, region, rng, density_override=None):
    """Return (for_path, dsf_density) for a polygon."""
    variant = int(rng.integers(1, 4))
    dlevel  = _for_density_level(density_override) if density_override is not None \
              else _density_level(frac)

    if veg_cls == SEGFORMER.CLASS_TREE:
        ftype    = 'mixed'    if dlevel >= 50 else 'woodland'
        base_key = 'tree'     if dlevel >= 50 else 'woodland'
        base     = _BASE_DENSITY[base_key][dlevel]
    elif veg_cls == SEGFORMER.CLASS_RANGELAND:
        ftype    = 'woodland'
        base_key = 'woodland' if dlevel >= 50 else 'shrub'
        base     = _BASE_DENSITY[base_key][dlevel]
    else:   # AGRICULTURE treelines only
        ftype  = 'cropland'
        base   = _BASE_DENSITY['shrub'][25]
        dlevel = 25

    if density_override is not None:
        base = int(round(density_override * 255))

    path = _gfv2_path(region, ftype, dlevel, variant)
    dsf_density = base + 256 if shape == 'treeline' else base
    return path, dsf_density


# ── Polygon helpers ───────────────────────────────────────────────────────────
def _polygon_fill_frac(mask, cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    if w == 0 or h == 0:
        return 0.0
    roi  = mask[y:y+h, x:x+w]
    fill = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(fill, [cnt - np.array([[x, y]])], 0, 1, cv2.FILLED)
    total = fill.sum()
    if total == 0:
        return 0.0
    return float((roi * fill).sum()) / float(total)


def _contour_shape(cnt, min_area_px):
    area = cv2.contourArea(cnt)
    if area < min_area_px:
        return 'tiny', area
    perim = cv2.arcLength(cnt, closed=True)
    if perim == 0:
        return 'tiny', area
    ratio = (perim ** 2) / (4 * pi * area)
    return ('treeline' if ratio >= TREELINE_RATIO else 'area'), area



def _contour_to_latlon(cnt, img_w, img_h, lat_n, lat_s, lon_w, lon_e):
    pts = cnt[:, 0, :]
    result = []
    for x, y in pts:
        o_lon, o_lat = px_to_latlon(float(x), float(y),
                                    img_w, img_h, lat_n, lat_s, lon_w, lon_e)
        result.append((o_lon, o_lat))
    if result and result[0] != result[-1]:
        result.append(result[0])
    return result


def _write_winding(f, ring_pts):
    pts = ring_pts
    if len(pts) > MAX_RING_PTS + 1:
        step = max(1, len(pts) // MAX_RING_PTS)
        pts  = pts[::step]
        if pts[0] != pts[-1]:
            pts.append(pts[0])
    f.write("BEGIN_WINDING\n")
    for lo, la in pts:
        f.write(f"POLYGON_POINT {lo:.7f} {la:.7f}\n")
    f.write("END_WINDING\n")


# ── Per-DDS mask processing ───────────────────────────────────────────────────
def _process_dds_mask(mask, veg_cls, img_w, img_h,
                      lat_n, lat_s, lon_w, lon_e, tile_lat, tile_lon,
                      m_per_px, min_area_px, simplify_px,
                      region, rng, density_override):
    """Extract polygons from one DDS class mask. Returns list of (path, density, ring)."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys   = []
    for cnt in cnts:
        shape, _ = _contour_shape(cnt, min_area_px)
        if shape == 'tiny':
            continue
        if veg_cls == SEGFORMER.CLASS_AGRICULTURE and shape != 'treeline':
            continue

        cnt_s = cv2.approxPolyDP(cnt, simplify_px, closed=True)
        if len(cnt_s) < 3:
            continue

        ring = _contour_to_latlon(cnt_s, img_w, img_h,
                                  lat_n, lat_s, lon_w, lon_e)
        if len(ring) < 4:
            continue

        # Skip polygons outside the degree tile (DDS tiles overlap tile boundary)
        lons = [p[0] for p in ring]
        lats = [p[1] for p in ring]
        if min(lons) >= tile_lon + 1 or max(lons) <= tile_lon: continue
        if min(lats) >= tile_lat + 1 or max(lats) <= tile_lat: continue

        frac = _polygon_fill_frac(mask, cnt)
        fpath, dsf_den = _for_entry(veg_cls, frac, shape, region, rng, density_override)
        polys.append((fpath, dsf_den, ring))
    return polys


# ── Core pipeline ─────────────────────────────────────────────────────────────
def run(tex_dir, lat, lon, out_dsf, cache_dir,
        close_m, open_m, min_area_m2, simplify_m,
        make_viz, density_override=None, res_m=None, excl_buffer_m=EXCL_BUFFER_M,
        bld_excl_m=10.0,
        osm_roads_path=None, use_simheaven=True, dsftool_path=None,
        custom_scenery_dir=None):

    import re as _re
    STD_RE = _re.compile(r"^(\d+)_(\d+)_([A-Za-z][A-Za-z0-9_]*)(\d{2})\.dds$",
                         _re.IGNORECASE)

    def _orthophoto_tile_dir():
        o4xp_root = os.path.dirname(os.path.dirname(os.path.dirname(tex_dir)))
        lat_g = int(math.floor(lat / 10)) * 10
        lon_g = int(math.floor(lon / 10)) * 10
        lat_s_str = f"{'+' if int(lat) >= 0 else '-'}{abs(int(lat)):02d}"
        lon_s_str = f"{'+' if int(lon) >= 0 else '-'}{abs(int(lon)):03d}"
        lat_g_str = f"{'+' if lat_g >= 0 else '-'}{abs(lat_g):02d}"
        lon_g_str = f"{'+' if lon_g >= 0 else '-'}{abs(lon_g):03d}"
        return os.path.join(
            o4xp_root, 'Orthophotos', f'{lat_g_str}{lon_g_str}',
            f'{lat_s_str}{lon_s_str}')

    def _collect_source_files():
        ortho_dir = _orthophoto_tile_dir()
        jpg_files = []
        if os.path.isdir(ortho_dir):
            for root, _, names in os.walk(ortho_dir):
                for name in names:
                    if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        candidate = os.path.splitext(name)[0] + '.dds'
                        if STD_RE.match(candidate):
                            jpg_files.append(candidate)
            if jpg_files:
                print(
                    f"Using {len(jpg_files)} original cached orthophotos from {ortho_dir}",
                    flush=True)
                return jpg_files, 'orthophoto', ortho_dir

        if os.path.isdir(tex_dir):
            dds_files = [f for f in os.listdir(tex_dir) if STD_RE.match(f)]
            if dds_files:
                print(
                    f"Original cached orthophotos missing; using {len(dds_files)} DDS textures from {tex_dir}",
                    flush=True)
                return dds_files, 'dds', None

        raise FileNotFoundError(
            f"No DDS textures found at {tex_dir!r} and no cached orthophotos found at {ortho_dir!r}")

    def _orthophoto_path(fname, ortho_dir):
        m = STD_RE.match(fname)
        if not m:
            return None
        provider = m.group(3)
        zl = int(m.group(4))
        stem = os.path.splitext(fname)[0]
        subdir = os.path.join(ortho_dir, f"{provider}_{zl}")
        for ext in ('.jpg', '.jpeg', '.png'):
            p = os.path.join(subdir, stem + ext)
            if os.path.exists(p):
                return p
        return None

    def _load_source_image(fname, source_mode, ortho_dir):
        if source_mode == 'dds':
            return SEGFORMER._load_dds(os.path.join(tex_dir, fname))
        p = _orthophoto_path(fname, ortho_dir)
        if not p:
            return None
        try:
            return np.asarray(Image.open(p).convert('RGB'))
        except Exception:
            return None

    # Collect all DDS files; resolve overlapping zoom levels.
    # Each ZL tile maps to exactly 4 children at ZL+1, 16 at ZL+2, etc.
    # A lower-ZL tile is skipped only when ALL 4 of its ZL+1 children are present
    # or themselves fully covered — otherwise it is kept to fill the missing area.
    _by_zl = {}
    _source_files, _source_mode, _orthophoto_dir = _collect_source_files()
    for _f in _source_files:
        _m = STD_RE.match(_f)
        if not _m: continue
        _by_zl.setdefault(int(_m.group(4)), []).append(
            (int(_m.group(1)), int(_m.group(2)), _f))
    if not _by_zl:
        print("No DDS files found."); return 0
    _tiles_at_zl = {zl: {(y, x) for y, x, _ in tiles} for zl, tiles in _by_zl.items()}
    _all_zls = sorted(_by_zl)
    _max_zl   = _all_zls[-1]
    _fc_memo  = {}
    def _fully_covered(y, x, zl):
        """True iff all 4 children of (y,x,zl) exist or are themselves fully covered."""
        key = (y, x, zl)
        if key in _fc_memo: return _fc_memo[key]
        if zl >= _max_zl:
            _fc_memo[key] = False; return False
        result = all(
            (cy, cx) in _tiles_at_zl.get(zl + 1, set()) or _fully_covered(cy, cx, zl + 1)
            for cy, cx in ((2*y, 2*x), (2*y, 2*x+1), (2*y+1, 2*x), (2*y+1, 2*x+1))
        )
        _fc_memo[key] = result; return result
    files = sorted(
        _f for _zl in _all_zls
        for _y, _x, _f in _by_zl[_zl]
        if not _fully_covered(_y, _x, _zl)
    )
    if not files:
        print("No DDS files found."); return 0
    _used_zls = sorted(set(int(STD_RE.match(_f).group(4)) for _f in files))
    _zl_str = f"ZL{_used_zls[0]}" if len(_used_zls) == 1 else f"mixed ZL {_used_zls}"

    dens_str = f"{density_override:.2f}" if density_override is not None else "auto"
    print(f"Tile: lat={lat} lon={lon}  DDS: {len(files)} ({_zl_str})")
    print(f"close={close_m}m  open={open_m}m  min_area={min_area_m2}m²  "
          f"simplify={simplify_m}m  density={dens_str}  excl_buffer={excl_buffer_m}m")

    device = __import__('torch').device('cuda' if __import__('torch').cuda.is_available() else 'cpu')
    model = proc = None

    region  = _climate_region(lat)
    rng     = np.random.default_rng(7)
    polygons = []   # (for_path, dsf_density, ring)
    timings = {
        'simheaven_parse': 0.0,
        'cache_load': 0.0,
        'dds_load': 0.0,
        'inference': 0.0,
        'road_excl': 0.0,
        'bld_excl': 0.0,
        'contours': 0.0,
        'dsf_text': 0.0,
        'dsf_compile': 0.0,
    }

    # ── Pre-load road / network exclusion sources ────────────────────────────
    # Auto-discover Ortho4XP OSM_data path for this tile
    if osm_roads_path is None:
        o4xp_root = os.path.dirname(os.path.dirname(os.path.dirname(tex_dir)))
        lat_i = int(lat); lon_i = int(lon)
        lat_g = int(math.floor(lat / 10)) * 10
        lon_g = int(math.floor(lon / 10)) * 10
        lat_s_str  = f"{'+' if lat_i >= 0 else '-'}{abs(lat_i):02d}"
        lon_s_str  = f"{'+' if lon_i >= 0 else '-'}{abs(lon_i):03d}"
        lat_g_str  = f"{'+' if lat_g >= 0 else '-'}{abs(lat_g):02d}"
        lon_g_str  = f"{'+' if lon_g >= 0 else '-'}{abs(lon_g):03d}"
        osm_roads_path = os.path.join(
            o4xp_root, 'OSM_data',
            f'{lat_g_str}{lon_g_str}',
            f'{lat_s_str}{lon_s_str}',
            f'{lat_s_str}{lon_s_str}_big_roads.osm.bz2')

    osm_roads = _load_osm_roads(osm_roads_path)
    print(f"OSM roads: {len(osm_roads)} ways  ({osm_roads_path})")

    # Railways from the exclusion cache written by building overlay (if present)
    excl_cache = osm_roads_path.replace('_big_roads.osm.bz2', '_excl_bld_rail.osm.bz2')
    excl_rails = _load_excl_railways(excl_cache)
    print(f"OSM railways: {len(excl_rails)} ways")

    # simHeaven full street network
    if dsftool_path is None:
        dsftool_path = SEGFORMER._dsftool
    sh_network = []
    if use_simheaven and dsftool_path and os.path.exists(dsftool_path):
        _t = time.perf_counter()
        sh_network = _load_simheaven_network(custom_scenery_dir, lat, lon, dsftool_path, cache_dir)
        timings['simheaven_parse'] += time.perf_counter() - _t
        print(f"simHeaven network: {len(sh_network)} road segments")
    elif use_simheaven:
        print(f"  [simHeaven] DSFTool not found at {dsftool_path} — skipping")

    all_road_ways = _prepare_roads(osm_roads + excl_rails + sh_network)

    t_inf = time.time()
    n_tree = n_range = n_agri = 0
    n_files = len(files)
    n_bld_excl_used = 0
    n_bld_excl_missing = 0

    for idx, fname in enumerate(files, 1):
        m = STD_RE.match(fname)
        if not m: continue
        til_y_top  = int(m.group(1))
        til_x_left = int(m.group(2))
        zl         = int(m.group(4))

        lat_n, lat_s, lon_w, lon_e = dds_bounds(til_y_top, til_x_left, zl)

        cache_path = os.path.join(cache_dir, fname.replace('.dds', '_veg.npy'))
        if os.path.exists(cache_path):
            _t = time.perf_counter()
            veg_map = np.load(cache_path)
            timings['cache_load'] += time.perf_counter() - _t
            print(f"  [{idx}/{n_files}] {fname}  (cached)", flush=True)
        else:
            print(f"  [{idx}/{n_files}] {fname}  (inferring…)", flush=True)
            _t = time.perf_counter()
            img = _load_source_image(fname, _source_mode, _orthophoto_dir)
            if img is None: continue
            timings['dds_load'] += time.perf_counter() - _t
            if model is None:
                model, proc, device = SEGFORMER.load_vegetation_model(device)
            _t = time.perf_counter()
            veg_map = SEGFORMER.run_inference(model, device, img, proc)
            timings['inference'] += time.perf_counter() - _t
            np.save(cache_path, veg_map)

        img_h, img_w = veg_map.shape
        mpp = dds_m_per_px(lat_n, lat_s, lon_w, lon_e, img_h, img_w)

        # Optionally downsample to a coarser resolution to reduce polygon count
        if res_m is not None and res_m > mpp:
            scale   = mpp / res_m
            new_w   = max(64, int(img_w * scale))
            new_h   = max(64, int(img_h * scale))
            veg_map = cv2.resize(veg_map.astype(np.uint8), (new_w, new_h),
                                 interpolation=cv2.INTER_NEAREST).astype(veg_map.dtype)
            img_h, img_w = veg_map.shape
            mpp = dds_m_per_px(lat_n, lat_s, lon_w, lon_e, img_h, img_w)

        # Convert metre params to pixels at this DDS resolution
        close_px    = max(1, int(close_m   / mpp))
        open_px     = max(1, int(open_m    / mpp))
        min_area_px = max(1, min_area_m2   / (mpp * mpp))
        simplify_px = max(1.0, simplify_m  / mpp)

        kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px*2+1,)*2)
        ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px*2+1, )*2)

        def _morph(raw):
            m = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kc)
            m = cv2.morphologyEx(m,   cv2.MORPH_OPEN,  ko)
            return m

        tree_raw  = (veg_map == SEGFORMER.CLASS_TREE).astype(np.uint8)
        tree_mask = _morph(tree_raw)

        # ── Exclusion layer 1: SegFormer CLASS_BUILDING / CLASS_ROAD / CLASS_DEVELOPED
        # Dilate by excl_buffer_m to catch morphological bleed from close operation
        excl_px = max(1, int(excl_buffer_m / mpp))
        ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (excl_px*2+1,)*2)
        excl_raw  = ((veg_map == SEGFORMER.CLASS_BUILDING) |
                     (veg_map == SEGFORMER.CLASS_ROAD)     |
                     (veg_map == SEGFORMER.CLASS_DEVELOPED)).astype(np.uint8)
        excl_mask = cv2.dilate(excl_raw, ke)

        # ── Exclusion layer 2: OSM roads + SimHeaven network + railways
        # Each way is drawn at 2×half_width_m thickness for granular, type-aware clearance
        if all_road_ways:
            _t = time.perf_counter()
            local_road_ways = _roads_for_bounds(
                all_road_ways, lat_n, lat_s, lon_w, lon_e, pad_deg=0.001)
            road_excl = _rasterize_roads_typed(
                local_road_ways, lat_n, lat_s, lon_w, lon_e, img_h, img_w, mpp)
            excl_mask = cv2.bitwise_or(excl_mask, road_excl)
            timings['road_excl'] += time.perf_counter() - _t

        # ── Exclusion layer 3: placed building objects from bld overlay ──────
        # Read the per-DDS placement cache written by src/scripts/generate_bld_overlay.
        # Use a loose circular buffer (bld_excl_m) — much larger than the
        # building-to-building 3 m margin — so trees stay clear of structures
        # without being pushed too far back from the building edge.
        if bld_excl_m > 0:
            _t = time.perf_counter()
            import pickle as _pickle
            _bld_pkl = os.path.join(cache_dir, fname.replace('.dds', '_bld.pkl'))
            if os.path.exists(_bld_pkl):
                try:
                    with open(_bld_pkl, 'rb') as _f:
                        _bld_data = _pickle.load(_f)
                    _bld_objs = _bld_data.get('objects', [])
                    _bld_r = max(1, int(bld_excl_m / mpp))
                    _bld_excl = np.zeros((img_h, img_w), dtype=np.uint8)
                    for _o_lon, _o_lat, _heading, _obj_path in _bld_objs:
                        _bpx = int((_o_lon - lon_w) / (lon_e - lon_w) * img_w)
                        _bpy = int((lat_n - _o_lat) / (lat_n - lat_s) * img_h)
                        if 0 <= _bpx < img_w and 0 <= _bpy < img_h:
                            cv2.circle(_bld_excl, (_bpx, _bpy), _bld_r, 1, -1)
                    excl_mask = cv2.bitwise_or(excl_mask, _bld_excl)
                    n_bld_excl_used += 1
                except Exception as _e:
                    print(f"    bld exclusion: failed to load pkl — {_e}", flush=True)
            else:
                n_bld_excl_missing += 1
            timings['bld_excl'] += time.perf_counter() - _t

        tree_mask = cv2.bitwise_and(tree_mask, cv2.bitwise_not(excl_mask))

        kwargs = dict(img_w=img_w, img_h=img_h,
                      lat_n=lat_n, lat_s=lat_s, lon_w=lon_w, lon_e=lon_e,
                      tile_lat=lat, tile_lon=lon,
                      m_per_px=mpp, min_area_px=min_area_px,
                      simplify_px=simplify_px,
                      region=region, rng=rng,
                      density_override=density_override)

        _t = time.perf_counter()
        p = _process_dds_mask(tree_mask, SEGFORMER.CLASS_TREE, **kwargs)
        timings['contours'] += time.perf_counter() - _t
        n_tree += len(p); polygons.extend(p)

    print(f"Loaded {len(files)} DDS  ({time.time()-t_inf:.1f}s)  "
          f"climate={region}  "
          f"bld_excl: {n_bld_excl_used} used / {n_bld_excl_missing} missing")
    print(f"Polygons: tree={n_tree}  total={len(polygons)}")

    if not polygons:
        print("No vegetation polygons — skipping DSF write."); return 0

    # ── Write DSF ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(out_dsf)), exist_ok=True)
    txt_path = out_dsf.replace('.dsf', '_veg.txt')

    for_paths = sorted(set(p for p, _, _ in polygons))
    for_idx   = {p: i for i, p in enumerate(for_paths)}
    lat_i = int(lat); lon_i = int(lon)

    _t = time.perf_counter()
    with open(txt_path, 'w') as f:
        f.write("PROPERTY sim/planet earth\n")
        f.write("PROPERTY sim/overlay 1\n")
        f.write(f"PROPERTY sim/west  {lon_i}\n")
        f.write(f"PROPERTY sim/east  {lon_i + 1}\n")
        f.write(f"PROPERTY sim/south {lat_i}\n")
        f.write(f"PROPERTY sim/north {lat_i + 1}\n\n")
        for p in for_paths:
            f.write(f"POLYGON_DEF {p}\n")
        f.write("\n")
        for fpath, dsf_den, ring in polygons:
            f.write(f"BEGIN_POLYGON {for_idx[fpath]} {dsf_den} 2\n")
            _write_winding(f, ring)
            f.write("END_POLYGON\n")
    timings['dsf_text'] += time.perf_counter() - _t

    print(f"DSF text → {txt_path}  ({len(for_paths)} unique .for defs)")

    _t = time.perf_counter()
    ok = SEGFORMER.compile_dsf(txt_path, out_dsf)
    timings['dsf_compile'] += time.perf_counter() - _t
    if ok:
        print(f"DSF compiled → {out_dsf}")
        try: os.remove(txt_path)
        except: pass
    else:
        print(f"DSFTool failed — text file kept at {txt_path}")

    print(
        "[Veg timing] "
        f"simHeaven={timings['simheaven_parse']:.1f}s  "
        f"cache_load={timings['cache_load']:.1f}s  "
        f"dds_load={timings['dds_load']:.1f}s  "
        f"inference={timings['inference']:.1f}s  "
        f"road_excl={timings['road_excl']:.1f}s  "
        f"bld_excl={timings['bld_excl']:.1f}s  "
        f"contours={timings['contours']:.1f}s  "
        f"dsf_text={timings['dsf_text']:.1f}s  "
        f"dsf_compile={timings['dsf_compile']:.1f}s"
    )

    return len(polygons)


# ── Entry point ───────────────────────────────────────────────────────────────
def parse_args():
    """Parse CLI arguments for vegetation overlay generation."""
    ap = argparse.ArgumentParser(description='SegFormer vegetation overlay generator')
    ap.add_argument('tex_dir')
    ap.add_argument('lat',  type=float)
    ap.add_argument('lon',  type=float)
    ap.add_argument('out_dsf', nargs='?', default=None)
    ap.add_argument('--cache-dir',  default=None)
    ap.add_argument('--close-m',    type=float, default=CLOSE_M,     dest='close_m')
    ap.add_argument('--open-m',     type=float, default=OPEN_M,      dest='open_m')
    ap.add_argument('--min-area',   type=float, default=MIN_AREA_M2, dest='min_area')
    ap.add_argument('--simplify',   type=float, default=SIMPLIFY_M,  dest='simplify')
    ap.add_argument('--density',    type=float, default=None,
                    help='Override DSF density 0.0–1.0 for all polygons')
    ap.add_argument('--res-m',      type=float, default=None, dest='res_m',
                    help='Downsample veg_map to this resolution in m/px before processing '
                         '(default: native DDS resolution ~2m). Use 5–10 to reduce polygon count.')
    ap.add_argument('--excl-buffer-m', type=float, default=EXCL_BUFFER_M, dest='excl_buffer_m',
                    help='Dilation buffer in metres around SegFormer buildings/roads before '
                         'excluding from tree mask (default: %(default)sm)')
    ap.add_argument('--osm-roads', default=None, dest='osm_roads',
                    help='Path to *_big_roads.osm.bz2 (auto-discovered from tex_dir if omitted)')
    ap.add_argument('--custom-scenery-dir', default=None,
                    help='Configured X-Plane Custom Scenery directory used to locate simHeaven.')
    ap.add_argument('--no-simheaven', action='store_true', dest='no_simheaven',
                    help='Skip simHeaven X-World network road exclusion')
    ap.add_argument('--no-viz',     action='store_true')
    return ap.parse_args()


def main():
    """Run the vegetation overlay CLI."""
    args = parse_args()
    o4xp_root = os.path.dirname(os.path.dirname(os.path.dirname(args.tex_dir)))
    lat_i = int(args.lat); lon_i = int(args.lon)
    lat_g = int(math.floor(args.lat / 10)) * 10
    lon_g = int(math.floor(args.lon / 10)) * 10
    lat_s = f"{'+' if lat_i >= 0 else '-'}{abs(lat_i):02d}"
    lon_s = f"{'+' if lon_i >= 0 else '-'}{abs(lon_i):03d}"
    lat_gs = f"{'+' if lat_g >= 0 else '-'}{abs(lat_g):02d}"
    lon_gs = f"{'+' if lon_g >= 0 else '-'}{abs(lon_g):03d}"

    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = os.path.join(
            o4xp_root, 'SFR_cache', f'{lat_gs}{lon_gs}', f'{lat_s}{lon_s}'
        )
    os.makedirs(cache_dir, exist_ok=True)

    out_dsf = args.out_dsf
    if out_dsf is None:
        out_dsf = os.path.join(o4xp_root, 'yOrtho4XP_Veg_Overlays',
                               'Earth nav data',
                               f'{lat_gs}{lon_gs}',
                               f'{lat_s}{lon_s}.dsf')
    print(f"Output DSF: {out_dsf}")

    t0 = time.time()
    n = run(
        tex_dir          = args.tex_dir,
        lat              = args.lat,
        lon              = args.lon,
        out_dsf          = out_dsf,
        cache_dir        = cache_dir,
        close_m          = args.close_m,
        open_m           = args.open_m,
        min_area_m2      = args.min_area,
        simplify_m       = args.simplify,
        make_viz         = not args.no_viz,
        density_override = args.density,
        res_m            = args.res_m,
        excl_buffer_m    = args.excl_buffer_m,
        osm_roads_path   = args.osm_roads,
        use_simheaven    = not args.no_simheaven,
        custom_scenery_dir = args.custom_scenery_dir,
    )


if __name__ == '__main__':
    main()
    print(f"\nDone: {n} vegetation polygons in {(time.time()-t0)/60:.1f}min")

