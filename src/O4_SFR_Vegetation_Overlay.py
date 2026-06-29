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

import sys, os, argparse, warnings, time, math, urllib.request, urllib.parse
warnings.filterwarnings('ignore')

import hashlib
import numpy as np
import cv2
from math import pi, atan, exp
from PIL import Image

import O4_Forest_Assets as FOREST_ASSETS
import O4_SFR_Climate_Regions as CLIMATE_REGIONS
import O4_SFR_Bounds_Index as BBOX
import O4_SFR_Persistent_Cache as PCACHE
import O4_SFR_Inference as SEGFORMER
from O4_SFR_Building_Overlay import (
    _load_simheaven_building_exclusions,
    _load_mesh_water_index,
    _mesh_file_for_tile,
    _mesh_water_signature,
    _prepare_simheaven_objects,
    _rasterize_mesh_water_mask,
    _rasterize_simheaven_objects,
    _simheaven_objects_for_bounds,
    _transient_cache_peer_path,
)
from O4_SFR_DSF_Utils import (
    ensure_cached_dsf_text,
    find_default_overlay_dsfs,
    find_global_forests_dsfs,
    find_simheaven_network_dsfs,
    find_simheaven_vegetation_dsfs,
)
from O4_SFR_Region_Boundaries import asset_region_for_latlon

# ── Defaults (all spatial params in metres) ───────────────────────────────────
CLOSE_M       = 10.0   # close kernel radius — fill gaps within a patch
OPEN_M        = 3.0    # open  kernel radius — remove sub-pixel noise
MIN_AREA_M2   = 50.0   # minimum polygon area (~7×7 m)
SIMPLIFY_M    = 3.0    # Douglas-Peucker tolerance
TREELINE_RATIO= 5.0    # perimeter² / (4π × area) ≥ this → candidate treeline
TREELINE_MAX_WIDTH_M = 30.0  # broad irregular blobs should stay filled forest
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

RESIDENTIAL_HIGHWAY_TYPES = {
    'living_street',
    'residential',
    'service',
    'unclassified',
}
RESIDENTIAL_FALLBACK_BUFFER_M = 45.0
RESIDENTIAL_FALLBACK_BUFFER_PX_MIN = 16
TREE_ROW_WIDTH_M = 8.0
CONTEXT_RING_M = 18.0

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
def _load_osm_roads(osm_bz2_path, cache_dir=None):
    """Parse an Ortho4XP *_big_roads.osm.bz2 file.
    Returns list of {'pts': [(lat,lon),...], 'type': highway_value}.
    """
    import bz2, xml.etree.ElementTree as ET
    if not os.path.exists(osm_bz2_path):
        return []
    def _parse():
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
                    hw = tag.get('v')
                    break
            if hw is None:
                continue
            pts = [nodes[nd.get('ref')] for nd in way.iter('nd') if nd.get('ref') in nodes]
            if len(pts) >= 2:
                roads.append({'pts': pts, 'type': hw})
        return roads

    return PCACHE.load_or_build(
        osm_bz2_path,
        cache_dir,
        "osm_parse",
        _parse,
        version="roads-v1",
    )


def _parse_excl_context_cache(cache_path, cache_dir=None):
    """Parse exclusion cache into railway ways and residential polygons."""
    import bz2, xml.etree.ElementTree as ET
    if not os.path.exists(cache_path):
        return [], []
    def _parse():
        with bz2.open(cache_path, 'rb') as f:
            root = ET.parse(f).getroot()
        nodes = {}
        for node in root.iter('node'):
            nodes[node.get('id')] = (float(node.get('lat')), float(node.get('lon')))
        rails = []
        residential_polys = []
        for way in root.iter('way'):
            pts = [nodes[nd.get('ref')] for nd in way.iter('nd') if nd.get('ref') in nodes]
            if len(pts) < 2:
                continue
            is_closed = len(pts) >= 4 and pts[0] == pts[-1]
            tags = {tag.get('k'): tag.get('v') for tag in way.iter('tag')}
            rw = tags.get('railway')
            if rw and not is_closed:
                rails.append({'pts': pts, 'type': rw})
            elif is_closed and tags.get('landuse') == 'residential':
                residential_polys.append([(lon, lat) for lat, lon in pts])
        return rails, residential_polys

    return PCACHE.load_or_build(
        cache_path,
        cache_dir,
        "osm_parse",
        _parse,
        version="veg-excl-context-v1",
    )


def _download_and_cache_veg_context_osm(lat, lon, cache_path, timeout=45):
    """Download OSM semantics used to classify tree cover into real-world types."""
    if os.path.exists(cache_path):
        return True

    bbox = f"{int(lat)},{int(lon)},{int(lat)+1},{int(lon)+1}"
    query = (
        f'[out:xml][timeout:{timeout}];'
        f'('
        f'  way["natural"~"^(wood|tree_row|water|wetland|scrub)$"]({bbox});'
        f'  way["waterway"="riverbank"]({bbox});'
        f'  way["landuse"~"^(forest|orchard|farmland|farmyard|residential|allotments|village_green|recreation_ground)$"]({bbox});'
        f'  way["leisure"~"^(park|garden|golf_course|pitch)$"]({bbox});'
        f');'
        f'(._;>;);out body;'
    )
    servers = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
        "https://overpass.osm.jp/api/interpreter",
    ]
    for server in servers:
        try:
            url = server + '?data=' + urllib.parse.quote(query)
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = resp.read()
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            import bz2 as _bz2
            with _bz2.open(cache_path, 'wb') as f:
                f.write(data)
            print(f"  [OSM veg] Downloaded {len(data)//1024} KB → {os.path.basename(cache_path)}")
            return True
        except Exception as exc:
            print(f"  [OSM veg] {server} failed: {exc}")
    return False


def _parse_veg_context_osm(cache_path, cache_dir=None):
    """Parse vegetation semantic polygons and tree-row hints from OSM XML cache."""
    import bz2, xml.etree.ElementTree as ET

    context = {
        'forest_polys': [],
        'woodland_hint_polys': [],
        'orchard_polys': [],
        'managed_polys': [],
        'residential_polys': [],
        'farmland_polys': [],
        'water_polys': [],
        'tree_rows': [],
    }
    if not os.path.exists(cache_path):
        return context

    def _parse():
        parsed_context = {
            'forest_polys': [],
            'woodland_hint_polys': [],
            'orchard_polys': [],
            'managed_polys': [],
            'residential_polys': [],
            'farmland_polys': [],
            'water_polys': [],
            'tree_rows': [],
        }

        with bz2.open(cache_path, 'rb') as f:
            root = ET.parse(f).getroot()

        nodes = {}
        for node in root.iter('node'):
            nodes[node.get('id')] = (float(node.get('lat')), float(node.get('lon')))

        for way in root.iter('way'):
            pts_latlon = [nodes[nd.get('ref')] for nd in way.iter('nd') if nd.get('ref') in nodes]
            if len(pts_latlon) < 2:
                continue

            is_closed = len(pts_latlon) >= 4 and pts_latlon[0] == pts_latlon[-1]
            tags = {tag.get('k'): tag.get('v') for tag in way.iter('tag')}
            landuse = tags.get('landuse')
            natural = tags.get('natural')
            leisure = tags.get('leisure')
            waterway = tags.get('waterway')

            if natural == 'tree_row' and len(pts_latlon) >= 2:
                parsed_context['tree_rows'].append({'pts': pts_latlon, 'type': 'tree_row'})
                continue

            if not is_closed:
                continue

            pts_lonlat = [(lon, lat) for lat, lon in pts_latlon]
            if natural == 'wood' or landuse == 'forest':
                parsed_context['forest_polys'].append(pts_lonlat)
            elif natural == 'scrub':
                parsed_context['woodland_hint_polys'].append(pts_lonlat)
            elif landuse == 'orchard':
                parsed_context['orchard_polys'].append(pts_lonlat)
            elif landuse in {'residential'}:
                parsed_context['residential_polys'].append(pts_lonlat)
            elif landuse in {'farmland', 'farmyard', 'allotments'}:
                parsed_context['farmland_polys'].append(pts_lonlat)
            elif leisure in {'park', 'garden', 'golf_course', 'pitch'} or landuse in {
                'village_green', 'recreation_ground'
            }:
                parsed_context['managed_polys'].append(pts_lonlat)
            elif natural in {'water', 'wetland'} or waterway == 'riverbank':
                parsed_context['water_polys'].append(pts_lonlat)

        return parsed_context

    return PCACHE.load_or_build(
        cache_path,
        cache_dir,
        "osm_parse",
        _parse,
        version="veg-context-v1",
    )


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
            def _parse():
                parsed_ways = []
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
                                parsed_ways.append({"pts": current_points, "type": "network"})
                            current_points = None
                return parsed_ways

            road_ways.extend(
                PCACHE.load_or_build(
                    cached_text_path,
                    cache_dir,
                    "dsf_parse",
                    _parse,
                    version="simheaven-network-v1",
                )
            )
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


def _roads_signature(roads):
    """Small deterministic signature for cache invalidation."""
    n_roads = 0
    n_pts = 0
    checksum = 0.0
    for road in roads or []:
        pts = road.get('pts', ())
        n_roads += 1
        n_pts += len(pts)
        r_s, r_n, r_w, r_e = road.get('_bounds') or _road_bounds(road)
        checksum += (r_s * 3.0) + (r_n * 5.0) + (r_w * 7.0) + (r_e * 11.0) + len(pts)
    return (n_roads, n_pts, round(checksum, 6))


def _roads_for_bounds(roads, lat_n, lat_s, lon_w, lon_e, pad_deg=0.0):
    if not roads:
        return []
    s = lat_s - pad_deg
    n = lat_n + pad_deg
    w = lon_w - pad_deg
    e = lon_e + pad_deg
    if isinstance(roads, dict) and {'items', 'south', 'north', 'west', 'east'} <= set(roads):
        return BBOX.query_bounds(roads, s, n, w, e)
    result = []
    for road in roads:
        r_s, r_n, r_w, r_e = road.get('_bounds') or _road_bounds(road)
        if r_n >= s and r_s <= n and r_e >= w and r_w <= e:
            result.append(road)
    return result


def _poly_bounds(poly):
    if not poly:
        return (0.0, 0.0, 0.0, 0.0)
    lons = [pt[0] for pt in poly]
    lats = [pt[1] for pt in poly]
    return min(lats), max(lats), min(lons), max(lons)


def _prepare_polygons(polys):
    prepared = []
    for poly in polys or []:
        if isinstance(poly, dict):
            pts = poly.get('pts', ())
            source_path = poly.get('path')
        else:
            pts = poly
            source_path = None
        if len(pts) < 3:
            continue
        record = {'pts': pts, '_bounds': _poly_bounds(pts)}
        if source_path:
            record['path'] = source_path
        prepared.append(record)
    return prepared


def _polys_signature(polys):
    """Small deterministic signature for polygon-mask cache invalidation."""
    n_polys = 0
    n_pts = 0
    checksum = 0.0
    path_checksum = 0
    for poly in polys or []:
        if isinstance(poly, dict):
            pts = poly.get('pts', ())
            bounds = poly.get('_bounds')
            path = poly.get('path')
        else:
            pts = poly
            bounds = None
            path = None
        if not pts:
            continue
        n_polys += 1
        n_pts += len(pts)
        if bounds is None:
            bounds = _poly_bounds(pts)
        south, north, west, east = bounds
        checksum += (
            south * 3.0 + north * 5.0 +
            west * 7.0 + east * 11.0 +
            len(pts)
        )
        if path:
            for char in str(path):
                path_checksum = ((path_checksum * 33) + ord(char)) & 0xFFFFFFFF
    return (n_polys, n_pts, round(checksum, 6), path_checksum)


def _polys_for_bounds(polys, lat_n, lat_s, lon_w, lon_e, pad_deg=0.0):
    if not polys:
        return []
    s = lat_s - pad_deg
    n = lat_n + pad_deg
    w = lon_w - pad_deg
    e = lon_e + pad_deg
    if isinstance(polys, dict) and {'items', 'south', 'north', 'west', 'east'} <= set(polys):
        return [poly.get('pts', poly) for poly in BBOX.query_bounds(polys, s, n, w, e)]
    result = []
    for poly in polys:
        p_s, p_n, p_w, p_e = poly.get('_bounds') or _poly_bounds(poly.get('pts', ()))
        if p_n >= s and p_s <= n and p_e >= w and p_w <= e:
            result.append(poly.get('pts', poly))
    return result


def _poly_records_for_bounds(polys, lat_n, lat_s, lon_w, lon_e, pad_deg=0.0):
    if not polys:
        return []
    s = lat_s - pad_deg
    n = lat_n + pad_deg
    w = lon_w - pad_deg
    e = lon_e + pad_deg
    if isinstance(polys, dict) and {'items', 'south', 'north', 'west', 'east'} <= set(polys):
        return BBOX.query_bounds(polys, s, n, w, e)
    result = []
    for poly in polys:
        p_s, p_n, p_w, p_e = poly.get('_bounds') or _poly_bounds(poly.get('pts', ()))
        if p_n >= s and p_s <= n and p_e >= w and p_w <= e:
            result.append(poly)
    return result


def _rasterize_polygons(polys, lat_n, lat_s, lon_w, lon_e, img_h, img_w):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not polys:
        return mask

    def ll_to_px(lat, lon):
        x = int((lon - lon_w) / (lon_e - lon_w) * img_w)
        y = int((lat_n - lat) / (lat_n - lat_s) * img_h)
        return x, y

    for poly in polys:
        pts_px = np.array([ll_to_px(lat, lon) for lon, lat in poly], dtype=np.int32)
        if len(pts_px) >= 3:
            cv2.fillPoly(mask, [pts_px], 1)
    return mask


def _rasterize_ways_constant_width(ways, lat_n, lat_s, lon_w, lon_e,
                                   img_h, img_w, width_px):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not ways:
        return mask

    def ll_to_px(lat, lon):
        x = int((lon - lon_w) / (lon_e - lon_w) * img_w)
        y = int((lat_n - lat) / (lat_n - lat_s) * img_h)
        return (max(0, min(img_w - 1, x)),
                max(0, min(img_h - 1, y)))

    for way in ways:
        pts = way.get('pts', ())
        if len(pts) < 2:
            continue
        pts_px = [ll_to_px(lat, lon) for lat, lon in pts]
        for i in range(len(pts_px) - 1):
            cv2.line(mask, pts_px[i], pts_px[i + 1], 1, thickness=width_px)
    return mask


def _residential_roads(roads):
    return [road for road in (roads or []) if road.get('type') in RESIDENTIAL_HIGHWAY_TYPES]


def _build_residential_context_mask(residential_polys, residential_roads,
                                    lat_n, lat_s, lon_w, lon_e,
                                    img_h, img_w, mpp):
    if residential_polys:
        return _rasterize_polygons(
            residential_polys, lat_n, lat_s, lon_w, lon_e, img_h, img_w
        ), 'OSM residential landuse'

    if residential_roads:
        road_mask = _rasterize_ways_constant_width(
            residential_roads, lat_n, lat_s, lon_w, lon_e, img_h, img_w, width_px=1
        )
        road_buffer_px = max(
            RESIDENTIAL_FALLBACK_BUFFER_PX_MIN,
            int(round(RESIDENTIAL_FALLBACK_BUFFER_M / max(mpp, 1e-6))),
        )
        road_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (road_buffer_px * 2 + 1, road_buffer_px * 2 + 1)
        )
        return cv2.dilate(road_mask, road_kernel), 'OSM neighborhood roads'

    return None, 'unavailable'


def _dds_mask_cache_key(fname, lat_n, lat_s, lon_w, lon_e, img_h, img_w, mpp,
                        road_sig, res_road_sig, tree_row_sig, context_poly_sigs,
                        forest_layer_sigs, sh_bld_poly_sig, sh_bld_obj_sig,
                        bld_excl_m, bld_cache_stat,
                        simheaven_building_buffer_m, mesh_water_sig=None):
    return {
        'version': 2,
        'fname': fname,
        'bounds': tuple(round(v, 8) for v in (lat_n, lat_s, lon_w, lon_e)),
        'shape': (int(img_h), int(img_w)),
        'mpp': round(float(mpp), 6),
        'road_sig': road_sig,
        'res_road_sig': res_road_sig,
        'tree_row_sig': tree_row_sig,
        'context_poly_sigs': tuple(sorted(context_poly_sigs.items())),
        'forest_layer_sigs': forest_layer_sigs,
        'sh_bld_poly_sig': sh_bld_poly_sig,
        'sh_bld_obj_sig': sh_bld_obj_sig,
        'bld_excl_m': round(float(bld_excl_m), 4),
        'bld_cache_stat': bld_cache_stat,
        'simheaven_building_buffer_m': round(float(simheaven_building_buffer_m), 4),
        'mesh_water_sig': mesh_water_sig,
        'residential_buffer_m': float(RESIDENTIAL_FALLBACK_BUFFER_M),
        'residential_buffer_px_min': int(RESIDENTIAL_FALLBACK_BUFFER_PX_MIN),
        'tree_row_width_m': float(TREE_ROW_WIDTH_M),
    }


def _or_optional_masks(lhs, rhs):
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    return cv2.bitwise_or(lhs, rhs)


def _load_dds_mask_cache(cache_path, key):
    import pickle as _pickle
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, 'rb') as f:
            data = _pickle.load(f)
        if data.get('key') == key:
            return data
    except Exception:
        return None
    return None


def _save_dds_mask_cache(cache_path, key, payload):
    import pickle as _pickle
    try:
        with open(cache_path, 'wb') as f:
            _pickle.dump({'key': key, **payload}, f, protocol=_pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass


def _dds_polygon_cache_key(
    fname,
    lat_n,
    lat_s,
    lon_w,
    lon_e,
    img_h,
    img_w,
    veg_cache_stat,
    mask_cache_key,
    close_px,
    open_px,
    min_area_px,
    simplify_px,
    density_override,
    excl_buffer_m,
    region,
    climate_code=None,
    asset_selection_mode="climate",
    gfv2_type_source_path=None,
):
    return {
        'version': 4,
        'fname': fname,
        'bounds': tuple(round(v, 8) for v in (lat_n, lat_s, lon_w, lon_e)),
        'shape': (int(img_h), int(img_w)),
        'veg_cache_stat': veg_cache_stat,
        'mask_cache_key': mask_cache_key,
        'close_px': int(close_px),
        'open_px': int(open_px),
        'min_area_px': round(float(min_area_px), 4),
        'simplify_px': round(float(simplify_px), 4),
        'density_override': None if density_override is None else round(float(density_override), 6),
        'excl_buffer_m': round(float(excl_buffer_m), 4),
        'region': region,
        'climate_code': climate_code,
        'asset_selection_mode': asset_selection_mode,
        'gfv2_type_source_path': gfv2_type_source_path,
    }


def _load_dds_polygon_cache(cache_path, key):
    import pickle as _pickle
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, 'rb') as f:
            data = _pickle.load(f)
        if data.get('key') == key:
            return data
    except Exception:
        return None
    return None


def _save_dds_polygon_cache(cache_path, key, polygons, type_counts):
    import pickle as _pickle
    try:
        with open(cache_path, 'wb') as f:
            _pickle.dump(
                {
                    'key': key,
                    'polygons': polygons,
                    'type_counts': dict(type_counts or {}),
                },
                f,
                protocol=_pickle.HIGHEST_PROTOCOL,
            )
    except Exception:
        pass


def _is_forest_polygon_path(path):
    p = (path or '').replace('\\', '/').lower()
    return p.endswith('.for') or '/forest' in p or 'forest/' in p


def _load_forest_polygons(layer_name, dsf_matches, dsftool_path, cache_dir):
    """Parse forest polygons from a list of DSFs."""
    polys = []
    seen_layers = set()

    for folder_name, dsf_path in dsf_matches:
        layer_key = folder_name.lower()
        if layer_key in seen_layers:
            continue
        seen_layers.add(layer_key)

        n_poly0 = len(polys)
        try:
            cached_text_path = ensure_cached_dsf_text(
                dsf_path,
                dsftool_path,
                cache_dir,
                create_no_window=SEGFORMER._CREATE_NO_WINDOW,
            )
            def _parse():
                parsed_polys = []
                polygon_defs = []
                current_polygon_is_forest = False
                current_winding = None
                current_polygon_path = None

                with open(cached_text_path, "r", encoding="utf-8", errors="ignore") as text_file:
                    for raw_line in text_file:
                        line = raw_line.strip()
                        if line.startswith("POLYGON_DEF "):
                            polygon_defs.append(line.split(" ", 1)[1])
                        elif line.startswith("BEGIN_POLYGON "):
                            parts = line.split()
                            current_polygon_is_forest = False
                            current_winding = None
                            current_polygon_path = None
                            try:
                                polygon_index = int(parts[1])
                                current_polygon_path = polygon_defs[polygon_index]
                                current_polygon_is_forest = _is_forest_polygon_path(current_polygon_path)
                            except (IndexError, ValueError):
                                current_polygon_is_forest = False
                        elif line == "BEGIN_WINDING" and current_polygon_is_forest:
                            current_winding = []
                        elif line.startswith("POLYGON_POINT ") and current_winding is not None:
                            parts = line.split()
                            try:
                                current_winding.append((float(parts[1]), float(parts[2])))
                            except (IndexError, ValueError):
                                pass
                        elif line == "END_WINDING" and current_winding is not None:
                            if len(current_winding) >= 3:
                                parsed_polys.append({
                                    'pts': current_winding,
                                    'path': current_polygon_path,
                                })
                            current_winding = None
                        elif line == "END_POLYGON":
                            current_polygon_is_forest = False
                            current_winding = None
                            current_polygon_path = None
                return parsed_polys

            polys.extend(
                PCACHE.load_or_build(
                    cached_text_path,
                    cache_dir,
                    "dsf_parse",
                    _parse,
                    version="forest-polygons-v2",
                )
            )

            print(
                f"  [{layer_name}] {folder_name}: +{len(polys) - n_poly0} forest polys"
            )
        except Exception as exc:
            print(f"  [{layer_name}] failed {dsf_path}: {exc}")

    return polys


def _dominant_acceptable_gfv2_path(records):
    """Return the most common acceptable GFv2 source path in a tile."""
    counts = {}
    for record in records or ():
        path = record.get('path') if isinstance(record, dict) else None
        metadata = FOREST_ASSETS.parse_gfv2_path(path)
        if metadata is None or not FOREST_ASSETS.is_acceptable_gfv2_type_source(path):
            continue
        normalized = metadata['path']
        counts[normalized] = counts.get(normalized, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


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


def _short_tree_candidates(region, dlevel):
    """Combine the measured short-tree GFv2 and default asset pools."""
    return FOREST_ASSETS.short_tree_candidates(region, dlevel)


def _for_entry(veg_cls, frac, shape, region, rng,
               density_override=None, veg_type=None, gfv2_type_path=None):
    """Return (for_path, dsf_density) for a polygon."""
    dlevel  = _for_density_level(density_override) if density_override is not None \
              else _density_level(frac)

    if veg_cls == SEGFORMER.CLASS_TREE:
        if veg_type in {'natural_woodland_open', 'uncertain_tree_cover'}:
            ftype = 'woodland'
            base_key = 'woodland'
        elif veg_type == 'riparian_trees':
            ftype = 'woodland'
            base_key = 'woodland'
            dlevel = min(dlevel, 50 if shape == 'treeline' else 75)
        elif veg_type in {'settlement_trees', 'park_or_managed_green'}:
            # Keep coverage, but bias managed/settlement canopy toward the safer
            # woodland assets instead of full mixed-forest sets.
            ftype = 'woodland'
            base_key = 'woodland'
            dlevel = min(dlevel, 50)
        elif veg_type in {'orchard_or_plantation', 'tree_row_linear'}:
            ftype = 'woodland'
            base_key = 'woodland'
            dlevel = 25 if shape == 'treeline' else min(dlevel, 50)
        else:
            # Future generated overlays stay under the Brazilian-nut height cap.
            # Dense forest gets denser placement, not taller asset families.
            ftype = 'woodland'
            base_key = 'tree' if shape == 'area' and dlevel >= 75 else 'woodland'
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

    if veg_cls == SEGFORMER.CLASS_TREE:
        tree_context = 'bulk'
        if shape == 'treeline':
            tree_context = 'treeline'
        elif veg_type in {'settlement_trees', 'park_or_managed_green'}:
            tree_context = 'managed'
        if gfv2_type_path:
            candidates = FOREST_ASSETS.gfv2_type_hint_candidates(
                gfv2_type_path,
                region,
                dlevel,
            )
            path = FOREST_ASSETS.choose_path(candidates, rng)
        else:
            path = FOREST_ASSETS.choose_tree_path(
                region,
                dlevel,
                rng,
                context=tree_context,
            )
    else:
        candidates = FOREST_ASSETS.short_gfv2_candidates(region, ftype, dlevel)
        path = FOREST_ASSETS.choose_path(candidates, rng)
    dsf_density = base + 256 if shape == 'treeline' else base
    return path, dsf_density


# ── Polygon helpers ───────────────────────────────────────────────────────────
def _polygon_fill_frac(mask, cnt, prepared=None):
    prepared = prepared or _contour_fill_stats(cnt, m_per_px=1.0, include_ring=False)
    if prepared is None:
        return 0.0
    x = prepared['x']
    y = prepared['y']
    w = prepared['w']
    h = prepared['h']
    fill = prepared['fill']
    roi = mask[y:y+h, x:x+w]
    total = prepared['area_px']
    if total == 0:
        return 0.0
    return float((roi * fill).sum()) / float(total)


def _contour_fill_stats(cnt, m_per_px, include_ring=True):
    """Return reusable rasterized contour stats for fill/context queries."""
    x, y, w, h = cv2.boundingRect(cnt)
    if w == 0 or h == 0:
        return None

    fill = np.zeros((h, w), dtype=np.uint8)
    offset = np.array([[x, y]], dtype=np.int32)
    cv2.drawContours(fill, [cnt - offset], 0, 1, cv2.FILLED)
    area_px = int(fill.sum())
    if area_px <= 0:
        return None

    result = {
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'fill': fill,
        'area_px': area_px,
    }
    if include_ring:
        ring_px = max(2, int(round(CONTEXT_RING_M / max(m_per_px, 1e-6))))
        ring_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (ring_px * 2 + 1, ring_px * 2 + 1)
        )
        outer = cv2.dilate(fill, ring_kernel)
        ring = cv2.subtract(outer, fill)
        result['ring'] = ring
        result['ring_area_px'] = int(ring.sum())
    return result


def _contour_context_stats(cnt, masks, m_per_px, prepared=None):
    prepared = prepared or _contour_fill_stats(cnt, m_per_px)
    if prepared is None:
        return {}
    x = prepared['x']
    y = prepared['y']
    w = prepared['w']
    h = prepared['h']
    fill = prepared['fill']
    area_px = prepared['area_px']
    ring = prepared['ring']
    ring_area_px = prepared['ring_area_px']
    stats = {'area_px': float(area_px), 'ring_area_px': float(ring_area_px)}
    for name, mask in (masks or {}).items():
        if mask is None:
            stats[name] = 0.0
            stats[f'{name}_ring'] = 0.0
            continue
        roi = mask[y:y+h, x:x+w]
        if roi.shape != fill.shape:
            stats[name] = 0.0
            stats[f'{name}_ring'] = 0.0
            continue
        stats[name] = float((roi * fill).sum()) / float(area_px)
        stats[f'{name}_ring'] = (
            float((roi * ring).sum()) / float(ring_area_px)
            if ring_area_px > 0 else 0.0
        )
    return stats


def _contour_shape(cnt, min_area_px, m_per_px):
    area = cv2.contourArea(cnt)
    if area < min_area_px:
        return 'tiny', area
    perim = cv2.arcLength(cnt, closed=True)
    if perim == 0:
        return 'tiny', area
    ratio = (perim ** 2) / (4 * pi * area)
    width_m = (2.0 * area / perim) * m_per_px
    is_treeline = ratio >= TREELINE_RATIO and width_m <= TREELINE_MAX_WIDTH_M
    return ('treeline' if is_treeline else 'area'), area


def _classify_tree_cover(cnt, frac, shape, m_per_px, context_masks, prepared_stats=None):
    stats = prepared_stats or _contour_context_stats(cnt, context_masks, m_per_px)
    if not stats:
        return 'natural_woodland_open'

    area_m2 = stats['area_px'] * m_per_px * m_per_px
    forest_in = stats.get('forest', 0.0)
    forest = max(forest_in, 0.7 * stats.get('forest_ring', 0.0))
    woodland_hint = max(
        stats.get('woodland_hint', 0.0),
        0.7 * stats.get('woodland_hint_ring', 0.0),
    )
    orchard_in = stats.get('orchard', 0.0)
    managed_in = stats.get('managed', 0.0)
    residential_in = stats.get('residential', 0.0)
    developed = max(stats.get('developed', 0.0), stats.get('developed_ring', 0.0))
    water = max(stats.get('water', 0.0), stats.get('water_ring', 0.0))
    tree_row = max(stats.get('tree_row', 0.0), stats.get('tree_row_ring', 0.0))

    # Only explicit semantic evidence should suppress canopy from forest output.
    if orchard_in >= 0.30:
        return 'orchard_or_plantation'
    if managed_in >= 0.30 and forest < 0.12:
        return 'park_or_managed_green'
    if shape == 'treeline' and tree_row >= 0.20 and forest < 0.10:
        return 'tree_row_linear'
    if residential_in >= 0.42:
        return 'settlement_trees'
    if (
        shape == 'treeline' and area_m2 <= 1600.0 and
        residential_in >= 0.20 and forest < 0.10
    ):
        return 'settlement_trees'
    if water >= 0.12 and developed < 0.12 and residential_in < 0.15 and managed_in < 0.15:
        return 'riparian_trees'
    if forest >= 0.12 or woodland_hint >= 0.20:
        if shape == 'area' and frac >= 0.58 and area_m2 >= 600.0:
            return 'natural_forest_closed'
        return 'natural_woodland_open'
    # Preserve broad coverage: when semantics are inconclusive, prefer a natural
    # fallback rather than dropping vegetation entirely.
    if shape == 'area' and frac >= 0.66 and area_m2 >= 900.0 and residential_in < 0.20:
        return 'natural_forest_closed'
    if area_m2 >= 200.0:
        return 'natural_woodland_open'
    return 'uncertain_tree_cover'



def _contour_to_latlon(cnt, img_w, img_h, lat_n, lat_s, lon_w, lon_e):
    pts = cnt[:, 0, :].astype(np.float64, copy=False)
    lons = lon_w + pts[:, 0] / img_w * (lon_e - lon_w)
    lats = lat_n - pts[:, 1] / img_h * (lat_n - lat_s)
    result = list(zip(lons.tolist(), lats.tolist()))
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
                      region, rng, density_override,
                      context_masks=None, type_counts=None,
                      gfv2_type_path=None):
    """Extract polygons from one DDS class mask. Returns list of (path, density, ring)."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys   = []
    for cnt in cnts:
        shape, _ = _contour_shape(cnt, min_area_px, m_per_px)
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

        prepared_stats = None
        frac = 0.0
        veg_type = None
        if veg_cls == SEGFORMER.CLASS_TREE:
            prepared_fill = _contour_fill_stats(cnt, m_per_px, include_ring=True)
            frac = _polygon_fill_frac(mask, cnt, prepared=prepared_fill)
            prepared_stats = _contour_context_stats(
                cnt, context_masks, m_per_px, prepared=prepared_fill
            )
            veg_type = _classify_tree_cover(
                cnt, frac, shape, m_per_px, context_masks, prepared_stats=prepared_stats
            )
            if type_counts is not None:
                type_counts[veg_type] = type_counts.get(veg_type, 0) + 1
        else:
            frac = _polygon_fill_frac(mask, cnt)

        fpath, dsf_den = _for_entry(
            veg_cls, frac, shape, region, rng, density_override,
            veg_type=veg_type,
            gfv2_type_path=gfv2_type_path,
        )
        if not fpath:
            continue
        polys.append((fpath, dsf_den, ring))
    return polys


# ── Core pipeline ─────────────────────────────────────────────────────────────
def run(tex_dir, lat, lon, out_dsf, cache_dir,
        close_m, open_m, min_area_m2, simplify_m,
        make_viz, density_override=None, res_m=None, disable_cache=False,
        excl_buffer_m=EXCL_BUFFER_M,
        bld_excl_m=10.0,
        osm_roads_path=None, use_simheaven=True, dsftool_path=None,
        download_veg_context=True,
        custom_scenery_dir=None, custom_overlay_src=None,
        custom_overlay_src_alternate=None,
        avoid_simheaven_buildings=True, simheaven_building_buffer_m=10.0,
        avoid_gfv2=True, gfv2_buffer_m=0.0,
        use_gfv2_asset_proximity=False,
        avoid_simheaven_forests=True, simheaven_buffer_m=0.0,
        avoid_default_forests=True, default_buffer_m=0.0):

    import re as _re
    STD_RE = _re.compile(r"^(\d+)_(\d+)_([A-Za-z][A-Za-z0-9_]*)(\d{2})\.dds$",
                         _re.IGNORECASE)
    _transient_cache_ctx = None
    sidecar_cache_dir = cache_dir
    if disable_cache:
        import tempfile as _tempfile
        _transient_cache_ctx = _tempfile.TemporaryDirectory(
            prefix="sfr_veg_nocache_"
        )
        sidecar_cache_dir = _transient_cache_ctx.name

    def _collect_source_files():
        return SEGFORMER.collect_source_texture_files(tex_dir, lat, lon)

    def _orthophoto_path(fname, ortho_dir):
        m = STD_RE.match(fname)
        if not m or SEGFORMER.is_mask_texture_name(fname):
            return None
        provider = m.group(3)
        zl = int(m.group(4))
        stem = os.path.splitext(fname)[0]
        subdir = os.path.join(ortho_dir, f"{provider}_{zl}")
        for ext in ('.jpg', '.jpeg', '.png'):
            p = os.path.join(subdir, stem + ext)
            if os.path.exists(p) and not SEGFORMER.is_mask_texture_name(os.path.basename(p)):
                return p
        return None

    def _load_source_image(fname, source_mode, ortho_dir):
        if source_mode == 'dds':
            return SEGFORMER.load_dds_or_none(
                os.path.join(tex_dir, fname),
                log_prefix='[SFR Veg]',
                display_name=fname,
            )
        p = _orthophoto_path(fname, ortho_dir)
        if not p:
            return None
        try:
            return np.asarray(Image.open(p).convert('RGB'))
        except Exception:
            return None

    def _dds_cache_paths(fname):
        if not cache_dir:
            return ()
        stem = fname.replace('.dds', '')
        return tuple(
            os.path.join(cache_dir, stem + suffix)
            for suffix in (
                '_veg.npy',
                '_road.pkl',
                '_bld.pkl',
                '_vegaux.pkl',
                '_vegpoly.pkl',
            )
        )

    def _remove_cache_files(paths):
        for path in paths:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

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
    print(
        "forest overlap avoid:"
        f" GFv2={'on' if avoid_gfv2 else 'off'} ({gfv2_buffer_m}m),"
        f" simHeaven={'on' if avoid_simheaven_forests else 'off'} ({simheaven_buffer_m}m),"
        f" default={'on' if avoid_default_forests else 'off'} ({default_buffer_m}m)"
    )
    print(
        "vegetation asset selection:"
        f" {'GFv2 tile-dominant' if use_gfv2_asset_proximity else 'climate default'}"
    )
    print(
        f"building overlap avoid: SFR cache={'on' if bld_excl_m > 0 else 'off'} ({bld_excl_m}m)"
        f" simHeaven={'on' if avoid_simheaven_buildings else 'off'} ({simheaven_building_buffer_m}m)"
    )

    device = __import__('torch').device('cuda' if __import__('torch').cuda.is_available() else 'cpu')
    model = proc = None

    tile_center_lat = lat + 0.5
    tile_center_lon = lon + 0.5
    asset_region = asset_region_for_latlon(tile_center_lat, tile_center_lon)
    koppen_code = CLIMATE_REGIONS.koppen_code(tile_center_lat, tile_center_lon)
    region = FOREST_ASSETS.climate_region(tile_center_lat, tile_center_lon)
    print(
        f"vegetation context: climate={region}"
        f" koppen={koppen_code or 'ocean/unknown'} asset_region={asset_region}"
    )
    polygons = []   # (for_path, dsf_density, ring)
    timings = {
        'scenery_parse': 0.0,
        'cache_load': 0.0,
        'mask_cache': 0.0,
        'poly_cache': 0.0,
        'dds_load': 0.0,
        'inference': 0.0,
        'mesh_water': 0.0,
        'road_excl': 0.0,
        'bld_excl': 0.0,
        'forest_layer_excl': 0.0,
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

    transient_peer_dir = sidecar_cache_dir if disable_cache else None
    osm_roads = _load_osm_roads(osm_roads_path, cache_dir=sidecar_cache_dir)
    print(f"OSM roads: {len(osm_roads)} ways  ({osm_roads_path})")
    osm_roads = _prepare_roads(osm_roads)

    # Railways + residential hints from the exclusion cache written by building overlay.
    excl_cache_candidates = [
        osm_roads_path.replace('_big_roads.osm.bz2', '_excl_bld_rail_res.osm.bz2'),
        osm_roads_path.replace('_big_roads.osm.bz2', '_excl_bld_rail.osm.bz2'),
    ]
    excl_cache = next((p for p in excl_cache_candidates if os.path.exists(p)), excl_cache_candidates[0])
    excl_rails, excl_residential = _parse_excl_context_cache(
        excl_cache, cache_dir=sidecar_cache_dir
    )
    print(f"OSM railways: {len(excl_rails)} ways")
    excl_rails = _prepare_roads(excl_rails)

    veg_context_osm_path = _transient_cache_peer_path(
        osm_roads_path,
        '_veg_context.osm.bz2',
        transient_peer_dir,
    ) or osm_roads_path.replace('_big_roads.osm.bz2', '_veg_context.osm.bz2')
    if download_veg_context and not os.path.exists(veg_context_osm_path):
        _download_and_cache_veg_context_osm(lat, lon, veg_context_osm_path)
    veg_context = _parse_veg_context_osm(veg_context_osm_path, cache_dir=sidecar_cache_dir)
    if excl_residential and not veg_context['residential_polys']:
        veg_context['residential_polys'] = excl_residential
    print(
        "OSM veg context: "
        f"forest={len(veg_context['forest_polys'])} "
        f"woodland={len(veg_context['woodland_hint_polys'])} "
        f"orchard={len(veg_context['orchard_polys'])} "
        f"managed={len(veg_context['managed_polys'])} "
        f"residential={len(veg_context['residential_polys'])} "
        f"tree_rows={len(veg_context['tree_rows'])}"
    )
    veg_context_prepared = {
        key: _prepare_polygons(value)
        for key, value in veg_context.items()
        if key.endswith('_polys')
    }
    context_poly_sigs = {
        key: _polys_signature(value)
        for key, value in veg_context_prepared.items()
    }
    veg_context_indexes = {
        key: BBOX.build_bounds_index(value)
        for key, value in veg_context_prepared.items()
    }
    osm_residential_roads = _prepare_roads(_residential_roads(osm_roads))
    osm_residential_index = BBOX.build_bounds_index(osm_residential_roads)
    veg_context_tree_rows = _prepare_roads(veg_context.get('tree_rows', []))
    tree_row_index = BBOX.build_bounds_index(veg_context_tree_rows)

    # simHeaven full street network
    if dsftool_path is None:
        dsftool_path = SEGFORMER._dsftool
    sh_network = []
    if use_simheaven and dsftool_path and os.path.exists(dsftool_path):
        _t = time.perf_counter()
        sh_network = _load_simheaven_network(custom_scenery_dir, lat, lon, dsftool_path, sidecar_cache_dir)
        timings['scenery_parse'] += time.perf_counter() - _t
        print(f"simHeaven network: {len(sh_network)} road segments")
    elif use_simheaven:
        print(f"  [simHeaven] DSFTool not found at {dsftool_path} — skipping")

    sh_network = _prepare_roads(sh_network)
    all_road_ways = osm_roads + excl_rails + sh_network
    all_road_index = BBOX.build_bounds_index(all_road_ways)
    all_road_sig = _roads_signature(all_road_ways)
    osm_residential_sig = _roads_signature(osm_residential_roads)
    tree_row_sig = _roads_signature(veg_context_tree_rows)

    forest_layers = []
    gfv2_type_path = None
    if dsftool_path and os.path.exists(dsftool_path):
        layer_specs = [
            (
                "Global Forests v2",
                avoid_gfv2,
                find_global_forests_dsfs(custom_scenery_dir, lat, lon),
                gfv2_buffer_m,
            ),
            (
                "simHeaven",
                avoid_simheaven_forests,
                find_simheaven_vegetation_dsfs(custom_scenery_dir, lat, lon),
                simheaven_buffer_m,
            ),
            (
                "default",
                avoid_default_forests,
                find_default_overlay_dsfs(
                    custom_overlay_src, lat, lon, custom_overlay_src_alternate
                ),
                default_buffer_m,
            ),
        ]
        for layer_name, enabled, dsf_matches, buffer_m in layer_specs:
            if not enabled:
                print(f"{layer_name} forests: disabled")
                continue
            _t = time.perf_counter()
            polys = _load_forest_polygons(layer_name, dsf_matches, dsftool_path, sidecar_cache_dir)
            timings['scenery_parse'] += time.perf_counter() - _t
            prepared = _prepare_polygons(polys)
            if layer_name == "Global Forests v2":
                if use_gfv2_asset_proximity:
                    gfv2_type_path = _dominant_acceptable_gfv2_path(prepared)
                    if gfv2_type_path:
                        print(f"{layer_name} tile-dominant type source: {gfv2_type_path}")
                    else:
                        print(f"{layer_name} type sources: no acceptable polygons")
                else:
                    print(f"{layer_name} type sources: disabled (climate asset selection)")
            forest_layers.append(
                {
                    'name': layer_name,
                    'buffer_m': max(0.0, float(buffer_m)),
                    'polys': prepared,
                    'index': BBOX.build_bounds_index(prepared),
                }
            )
            print(f"{layer_name} forests: {len(prepared)} polygons")
    else:
        if any((avoid_gfv2, avoid_simheaven_forests, avoid_default_forests)):
            print(f"Forest overlap layers: skipped (DSFTool unavailable at {dsftool_path})")
    forest_layer_sigs = tuple(
        (layer['name'], round(float(layer['buffer_m']), 4), _polys_signature(layer['polys']))
        for layer in forest_layers
    )
    gfv2_type_source_path = gfv2_type_path if use_gfv2_asset_proximity else None

    t_inf = time.time()
    n_tree = n_range = n_agri = 0
    n_files = len(files)
    tree_type_counts = {}
    n_bld_excl_used = 0
    n_bld_excl_missing = 0
    n_sh_bld_polys = 0
    n_sh_bld_objs = 0

    sh_bld_polys = []
    sh_bld_poly_index = None
    sh_bld_index = None
    sh_bld_poly_sig = (0, 0, 0.0)
    sh_bld_obj_sig = (0, 0.0)
    if avoid_simheaven_buildings and dsftool_path and os.path.exists(dsftool_path):
        _t = time.perf_counter()
        sh_bld_polys, sh_bld_objects = _load_simheaven_building_exclusions(
            custom_scenery_dir,
            lat,
            lon,
            dsftool_path,
            sidecar_cache_dir,
        )
        timings['scenery_parse'] += time.perf_counter() - _t
        n_sh_bld_polys = len(sh_bld_polys)
        n_sh_bld_objs = len(sh_bld_objects)
        sh_bld_polys = _prepare_polygons(sh_bld_polys)
        sh_bld_poly_index = BBOX.build_bounds_index(sh_bld_polys)
        sh_bld_index = _prepare_simheaven_objects(sh_bld_objects)
        sh_bld_poly_sig = _polys_signature(sh_bld_polys)
        if sh_bld_objects:
            arr = np.asarray(
                [
                    (obj['lat'], obj['lon'], obj['heading'], obj['w_m'], obj['h_m'])
                    for obj in sh_bld_objects
                ],
                dtype=np.float32,
            )
            checksum = float(np.sum(arr[:, 0] * 3.0 + arr[:, 1] * 5.0 +
                                    arr[:, 2] * 0.01 + arr[:, 3] + arr[:, 4]))
            sh_bld_obj_sig = (int(arr.shape[0]), round(checksum, 3))
        print(f"simHeaven buildings: {n_sh_bld_objs} objects  {n_sh_bld_polys} facade polys")
    elif avoid_simheaven_buildings:
        print(f"simHeaven buildings: skipped (DSFTool unavailable at {dsftool_path})")

    _t = time.perf_counter()
    mesh_water_path = _mesh_file_for_tile(tex_dir, lat, lon)
    mesh_water_sig = _mesh_water_signature(mesh_water_path)
    mesh_water_index = _load_mesh_water_index(mesh_water_path, sidecar_cache_dir)
    timings['mesh_water'] += time.perf_counter() - _t
    if mesh_water_index:
        print(
            f"Mesh water: {len(mesh_water_index['tris'])} water triangles from "
            f"{mesh_water_path}"
        )
    else:
        print(f"Mesh water: unavailable from {mesh_water_path}")

    for idx, fname in enumerate(files, 1):
        m = STD_RE.match(fname)
        if not m: continue
        _dds_cache_files = _dds_cache_paths(fname)
        if disable_cache:
            # Delete stale per-DDS cache before this DDS starts. Keep the
            # building placement cache only until we've consumed it below.
            _pre_dds_cleanup = tuple(
                path for path in _dds_cache_files
                if not (bld_excl_m > 0 and path.endswith('_bld.pkl'))
            )
            _remove_cache_files(
                _pre_dds_cleanup
            )
        til_y_top  = int(m.group(1))
        til_x_left = int(m.group(2))
        zl         = int(m.group(4))

        lat_n, lat_s, lon_w, lon_e = dds_bounds(til_y_top, til_x_left, zl)

        cache_path = os.path.join(cache_dir, fname.replace('.dds', '_veg.npy'))
        img = None
        veg_map = None
        mesh_water_mask = None
        mesh_water_full = False
        if not disable_cache and os.path.exists(cache_path):
            _t = time.perf_counter()
            veg_map = np.load(cache_path)
            timings['cache_load'] += time.perf_counter() - _t
            img_h, img_w = veg_map.shape[:2]
            _t = time.perf_counter()
            mesh_water_mask = _rasterize_mesh_water_mask(
                mesh_water_index, lat_n, lat_s, lon_w, lon_e, img_h, img_w
            )
            timings['mesh_water'] += time.perf_counter() - _t
            mesh_water_px = int(np.count_nonzero(mesh_water_mask)) if mesh_water_mask is not None else 0
            mesh_water_full = mesh_water_px == int(img_h * img_w)
            if (
                not mesh_water_full and
                bool(np.all(veg_map == SEGFORMER.CLASS_WATER))
            ):
                veg_map = None
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
                print(f"  [{idx}/{n_files}] {fname}  (stale all-water cache ignored)", flush=True)
            else:
                print(f"  [{idx}/{n_files}] {fname}  (cached)", flush=True)
        if veg_map is None:
            print(f"  [{idx}/{n_files}] {fname}  (inferring…)", flush=True)
            _t = time.perf_counter()
            img = _load_source_image(fname, _source_mode, _orthophoto_dir)
            if img is None:
                if disable_cache:
                    _remove_cache_files(_dds_cache_files)
                continue
            timings['dds_load'] += time.perf_counter() - _t
            img_h, img_w = img.shape[:2]
            _t = time.perf_counter()
            mesh_water_mask = _rasterize_mesh_water_mask(
                mesh_water_index, lat_n, lat_s, lon_w, lon_e, img_h, img_w
            )
            timings['mesh_water'] += time.perf_counter() - _t
            mesh_water_full = (
                mesh_water_mask is not None and
                int(np.count_nonzero(mesh_water_mask)) == int(img_h * img_w)
            )
            if mesh_water_full:
                veg_map = np.full((img_h, img_w), SEGFORMER.CLASS_WATER, dtype=np.int8)
                print(f"    {fname}: mesh water full; inference skipped", flush=True)
            else:
                if model is None:
                    model, proc, device = SEGFORMER.load_vegetation_model(device)
                _t = time.perf_counter()
                veg_map = SEGFORMER.run_inference(model, device, img, proc)
                timings['inference'] += time.perf_counter() - _t
            if not disable_cache and not mesh_water_full:
                np.save(cache_path, veg_map)

        if disable_cache:
            veg_cache_stat = None
        else:
            try:
                _veg_st = os.stat(cache_path)
                veg_cache_stat = (_veg_st.st_mtime_ns, _veg_st.st_size)
            except OSError:
                veg_cache_stat = None

        img_h, img_w = veg_map.shape
        mpp = dds_m_per_px(lat_n, lat_s, lon_w, lon_e, img_h, img_w)

        # Optionally downsample to a coarser resolution to reduce polygon count
        if res_m is not None and res_m > mpp:
            scale   = mpp / res_m
            new_w   = max(64, int(img_w * scale))
            new_h   = max(64, int(img_h * scale))
            veg_map = cv2.resize(veg_map.astype(np.uint8), (new_w, new_h),
                                 interpolation=cv2.INTER_NEAREST).astype(veg_map.dtype)
            if mesh_water_mask is not None:
                mesh_water_mask = cv2.resize(
                    mesh_water_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST
                ).astype(np.uint8, copy=False)
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
        segformer_excl_mask = cv2.dilate(excl_raw, ke)
        excl_mask = segformer_excl_mask.copy()
        if mesh_water_mask is not None and mesh_water_mask.any():
            excl_mask = cv2.bitwise_or(excl_mask, mesh_water_mask)

        _bld_pkl = os.path.join(cache_dir, fname.replace('.dds', '_bld.pkl'))
        _bld_cache_stat = None
        if bld_excl_m > 0 and os.path.exists(_bld_pkl):
            try:
                _st = os.stat(_bld_pkl)
                _bld_cache_stat = (_st.st_mtime_ns, _st.st_size)
            except OSError:
                _bld_cache_stat = None
        _mask_cache_file = os.path.join(cache_dir, fname.replace('.dds', '_vegaux.pkl'))
        _mask_cache_key = _dds_mask_cache_key(
            fname, lat_n, lat_s, lon_w, lon_e, img_h, img_w, mpp,
            all_road_sig, osm_residential_sig, tree_row_sig, context_poly_sigs,
            forest_layer_sigs, sh_bld_poly_sig, sh_bld_obj_sig,
            bld_excl_m, _bld_cache_stat, simheaven_building_buffer_m,
            mesh_water_sig,
        )
        _mask_cached = None
        if not disable_cache:
            _t = time.perf_counter()
            _mask_cached = _load_dds_mask_cache(_mask_cache_file, _mask_cache_key)
            timings['mask_cache'] += time.perf_counter() - _t
        if _mask_cached is not None:
            road_excl = _mask_cached.get('road_excl')
            bld_excl_mask = _mask_cached.get('bld_excl_mask')
            sh_bld_poly_mask = _mask_cached.get('sh_bld_poly_mask')
            sh_bld_obj_mask = _mask_cached.get('sh_bld_obj_mask')
            forest_excl_mask = _mask_cached.get('forest_excl_mask')
            cached_context_masks = _mask_cached.get('context_masks', {})
            if _bld_cache_stat is not None:
                n_bld_excl_used += 1
            elif bld_excl_m > 0:
                n_bld_excl_missing += 1
        else:
            road_excl = None
            bld_excl_mask = None
            sh_bld_poly_mask = None
            sh_bld_obj_mask = None
            forest_excl_mask = None
            cached_context_masks = {}

        # ── Exclusion layer 2: OSM roads + SimHeaven network + railways
        # Each way is drawn at 2×half_width_m thickness for granular, type-aware clearance
        if road_excl is None and all_road_ways:
            _t = time.perf_counter()
            local_road_ways = _roads_for_bounds(
                all_road_index, lat_n, lat_s, lon_w, lon_e, pad_deg=0.001)
            road_excl = _rasterize_roads_typed(
                local_road_ways, lat_n, lat_s, lon_w, lon_e, img_h, img_w, mpp)
            timings['road_excl'] += time.perf_counter() - _t
        if road_excl is not None:
            excl_mask = cv2.bitwise_or(excl_mask, road_excl)

        # ── Exclusion layer 3: placed building objects from bld overlay ──────
        # Read the per-DDS placement cache written by src/scripts/generate_bld_overlay.
        # Use a loose circular buffer (bld_excl_m) — much larger than the
        # building-to-building 3 m margin — so trees stay clear of structures
        # without being pushed too far back from the building edge.
        if bld_excl_m > 0 and bld_excl_mask is None:
            _t = time.perf_counter()
            import pickle as _pickle
            if os.path.exists(_bld_pkl):
                try:
                    with open(_bld_pkl, 'rb') as _f:
                        _bld_data = _pickle.load(_f)
                    _bld_objs = _bld_data.get('objects', [])
                    _bld_r = max(1, int(bld_excl_m / mpp))
                    bld_excl_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                    for _o_lon, _o_lat, _heading, _obj_path in _bld_objs:
                        _bpx = int((_o_lon - lon_w) / (lon_e - lon_w) * img_w)
                        _bpy = int((lat_n - _o_lat) / (lat_n - lat_s) * img_h)
                        if 0 <= _bpx < img_w and 0 <= _bpy < img_h:
                            cv2.circle(bld_excl_mask, (_bpx, _bpy), _bld_r, 1, -1)
                    n_bld_excl_used += 1
                except Exception as _e:
                    print(f"    bld exclusion: failed to load pkl — {_e}", flush=True)
            else:
                n_bld_excl_missing += 1
            timings['bld_excl'] += time.perf_counter() - _t
        if bld_excl_mask is not None:
            excl_mask = cv2.bitwise_or(excl_mask, bld_excl_mask)
        if disable_cache and bld_excl_m > 0:
            _remove_cache_files((_bld_pkl,))

        # ── Exclusion layer 3b: existing simHeaven buildings ─────────────────
        if avoid_simheaven_buildings and (sh_bld_polys or sh_bld_index) and (
            sh_bld_poly_mask is None or sh_bld_obj_mask is None
        ):
            _t = time.perf_counter()
            if sh_bld_poly_mask is None:
                local_sh_bld_polys = _polys_for_bounds(
                    sh_bld_poly_index, lat_n, lat_s, lon_w, lon_e, pad_deg=0.002
                )
                if local_sh_bld_polys:
                    sh_bld_poly_mask = _rasterize_polygons(
                        local_sh_bld_polys, lat_n, lat_s, lon_w, lon_e, img_h, img_w
                    )
                    if simheaven_building_buffer_m > 0 and sh_bld_poly_mask.any():
                        sh_bld_poly_px = max(1, int(simheaven_building_buffer_m / mpp))
                        k_sh_poly = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE, (sh_bld_poly_px * 2 + 1, sh_bld_poly_px * 2 + 1)
                        )
                        sh_bld_poly_mask = cv2.dilate(sh_bld_poly_mask, k_sh_poly)
            if sh_bld_obj_mask is None:
                local_sh_bld_objects = _simheaven_objects_for_bounds(
                    sh_bld_index, lat_n, lat_s, lon_w, lon_e, pad_deg=0.002
                )
                if local_sh_bld_objects:
                    sh_bld_obj_mask = _rasterize_simheaven_objects(
                        local_sh_bld_objects, lat_n, lat_s, lon_w, lon_e,
                        img_h, img_w, mpp, margin_m=max(0.0, float(simheaven_building_buffer_m))
                    )
            timings['bld_excl'] += time.perf_counter() - _t
        if sh_bld_poly_mask is not None:
            excl_mask = cv2.bitwise_or(excl_mask, sh_bld_poly_mask)
        if sh_bld_obj_mask is not None:
            excl_mask = cv2.bitwise_or(excl_mask, sh_bld_obj_mask)

        # ── Exclusion layer 4: existing forest overlays in scenery packages ──
        if forest_excl_mask is None and forest_layers:
            _t = time.perf_counter()
            forest_excl_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            for layer in forest_layers:
                local_polys = _polys_for_bounds(
                    layer['index'], lat_n, lat_s, lon_w, lon_e, pad_deg=0.001
                )
                if not local_polys:
                    continue
                layer_mask = _rasterize_polygons(
                    local_polys, lat_n, lat_s, lon_w, lon_e, img_h, img_w
                )
                buffer_m = layer['buffer_m']
                if buffer_m > 0 and layer_mask.any():
                    buffer_px = max(1, int(buffer_m / mpp))
                    k_layer = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (buffer_px * 2 + 1, buffer_px * 2 + 1)
                    )
                    layer_mask = cv2.dilate(layer_mask, k_layer)
                forest_excl_mask = cv2.bitwise_or(forest_excl_mask, layer_mask)
            timings['forest_layer_excl'] += time.perf_counter() - _t
        if forest_excl_mask is not None:
            excl_mask = cv2.bitwise_or(excl_mask, forest_excl_mask)

        local_context_masks = {
            'developed': segformer_excl_mask,
            'agriculture': (veg_map == SEGFORMER.CLASS_AGRICULTURE).astype(np.uint8),
        }

        if cached_context_masks:
            local_context_masks.update(cached_context_masks)
        else:
            for key in (
                'forest_polys',
                'woodland_hint_polys',
                'orchard_polys',
                'managed_polys',
                'farmland_polys',
                'water_polys',
                'residential_polys',
            ):
                local_polys = _polys_for_bounds(
                    veg_context_indexes.get(key), lat_n, lat_s, lon_w, lon_e, pad_deg=0.001
                )
                mask_name = key.replace('_polys', '')
                if local_polys:
                    local_context_masks[mask_name] = _rasterize_polygons(
                        local_polys, lat_n, lat_s, lon_w, lon_e, img_h, img_w
                    )
                else:
                    local_context_masks[mask_name] = None

            if local_context_masks['residential'] is None:
                local_residential_roads = _roads_for_bounds(
                    osm_residential_index, lat_n, lat_s, lon_w, lon_e, pad_deg=0.001
                )
                residential_mask, _res_source = _build_residential_context_mask(
                    None, local_residential_roads, lat_n, lat_s, lon_w, lon_e, img_h, img_w, mpp
                )
                local_context_masks['residential'] = residential_mask

            local_tree_rows = _roads_for_bounds(
                tree_row_index,
                lat_n, lat_s, lon_w, lon_e, pad_deg=0.001,
            )
            if local_tree_rows:
                tree_row_px = max(1, int(round(TREE_ROW_WIDTH_M / max(mpp, 1e-6))))
                local_context_masks['tree_row'] = _rasterize_ways_constant_width(
                    local_tree_rows, lat_n, lat_s, lon_w, lon_e, img_h, img_w, tree_row_px
                )
            else:
                local_context_masks['tree_row'] = None

            if not disable_cache:
                _save_dds_mask_cache(
                    _mask_cache_file,
                    _mask_cache_key,
                    {
                        'road_excl': road_excl,
                        'bld_excl_mask': bld_excl_mask,
                        'sh_bld_poly_mask': sh_bld_poly_mask,
                        'sh_bld_obj_mask': sh_bld_obj_mask,
                        'forest_excl_mask': forest_excl_mask,
                        'context_masks': {
                            key: local_context_masks.get(key)
                            for key in (
                                'forest',
                                'woodland_hint',
                                'orchard',
                                'managed',
                                'farmland',
                                'water',
                                'residential',
                                'tree_row',
                            )
                        },
                    },
                )

        if local_context_masks.get('farmland') is not None:
            if local_context_masks['agriculture'] is None:
                local_context_masks['agriculture'] = local_context_masks['farmland']
            else:
                local_context_masks['agriculture'] = cv2.bitwise_or(
                    local_context_masks['agriculture'], local_context_masks['farmland']
                )
        if mesh_water_mask is not None and mesh_water_mask.any():
            local_context_masks['water'] = _or_optional_masks(
                local_context_masks.get('water'), mesh_water_mask
            )

        tree_mask = cv2.bitwise_and(tree_mask, cv2.bitwise_not(excl_mask))

        _poly_cache_file = os.path.join(cache_dir, fname.replace('.dds', '_vegpoly.pkl'))
        _poly_cache_key = _dds_polygon_cache_key(
            fname,
            lat_n,
            lat_s,
            lon_w,
            lon_e,
            img_h,
            img_w,
            veg_cache_stat,
            _mask_cache_key,
            close_px,
            open_px,
            min_area_px,
            simplify_px,
            density_override,
            excl_buffer_m,
            region,
            koppen_code,
            'gfv2_tile_dominant' if use_gfv2_asset_proximity else 'climate',
            gfv2_type_source_path,
        )
        _poly_cached = None
        if not disable_cache:
            _t = time.perf_counter()
            _poly_cached = _load_dds_polygon_cache(_poly_cache_file, _poly_cache_key)
            timings['poly_cache'] += time.perf_counter() - _t
        if _poly_cached is not None:
            p = list(_poly_cached.get('polygons', ()))
            for veg_type, count in (_poly_cached.get('type_counts', {}) or {}).items():
                tree_type_counts[veg_type] = tree_type_counts.get(veg_type, 0) + int(count)
            n_tree += len(p)
            polygons.extend(p)
            if disable_cache:
                _remove_cache_files(_dds_cache_files)
            continue

        if not tree_mask.any():
            if not disable_cache:
                _save_dds_polygon_cache(_poly_cache_file, _poly_cache_key, (), {})
            if disable_cache:
                _remove_cache_files(_dds_cache_files)
            continue

        kwargs = dict(img_w=img_w, img_h=img_h,
                      lat_n=lat_n, lat_s=lat_s, lon_w=lon_w, lon_e=lon_e,
                      tile_lat=lat, tile_lon=lon,
                      m_per_px=mpp, min_area_px=min_area_px,
                      simplify_px=simplify_px,
                      region=region,
                      density_override=density_override,
                      context_masks=local_context_masks,
                      type_counts=None)
        if gfv2_type_source_path:
            kwargs['gfv2_type_path'] = gfv2_type_source_path

        dds_seed = int.from_bytes(
            hashlib.sha1(f"veg-poly:{fname}".encode("utf-8")).digest()[:8],
            "big",
        )
        dds_rng = np.random.default_rng(dds_seed)
        kwargs['rng'] = dds_rng
        dds_type_counts = {}
        kwargs['type_counts'] = dds_type_counts

        _t = time.perf_counter()
        p = _process_dds_mask(tree_mask, SEGFORMER.CLASS_TREE, **kwargs)
        timings['contours'] += time.perf_counter() - _t
        n_tree += len(p); polygons.extend(p)
        for veg_type, count in dds_type_counts.items():
            tree_type_counts[veg_type] = tree_type_counts.get(veg_type, 0) + int(count)
        if not disable_cache:
            _save_dds_polygon_cache(_poly_cache_file, _poly_cache_key, p, dds_type_counts)
        if disable_cache:
            _remove_cache_files(_dds_cache_files)

    print(f"Loaded {len(files)} DDS  ({time.time()-t_inf:.1f}s)  "
          f"climate={region}  "
          f"bld_excl: {n_bld_excl_used} used / {n_bld_excl_missing} missing  "
          f"simh_bld: {n_sh_bld_objs} obj / {n_sh_bld_polys} poly")
    print(f"Polygons: tree={n_tree}  total={len(polygons)}")
    if tree_type_counts:
        ordered = sorted(tree_type_counts.items(), key=lambda item: (-item[1], item[0]))
        summary = ", ".join(f"{name}={count}" for name, count in ordered)
        print(f"Tree types: {summary}")

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
        f"scenery_parse={timings['scenery_parse']:.1f}s  "
        f"cache_load={timings['cache_load']:.1f}s  "
        f"mask_cache={timings['mask_cache']:.1f}s  "
        f"poly_cache={timings['poly_cache']:.1f}s  "
        f"dds_load={timings['dds_load']:.1f}s  "
        f"inference={timings['inference']:.1f}s  "
        f"mesh_water={timings['mesh_water']:.1f}s  "
        f"road_excl={timings['road_excl']:.1f}s  "
        f"bld_excl={timings['bld_excl']:.1f}s  "
        f"forest_excl={timings['forest_layer_excl']:.1f}s  "
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
    ap.add_argument('--no-download-veg-context', action='store_true',
                    dest='no_download_veg_context',
                    help='Do not download OSM vegetation context when the cache is missing.')
    ap.add_argument('--custom-scenery-dir', default=None,
                    help='Configured X-Plane Custom Scenery directory used to locate simHeaven.')
    ap.add_argument('--custom-overlay-src', default=None,
                    help='Configured overlay source root used to locate default forest DSFs.')
    ap.add_argument('--custom-overlay-src-alternate', default=None,
                    help='Alternate overlay source root used if the main default overlay source is missing.')
    ap.add_argument('--no-simheaven', action='store_true', dest='no_simheaven',
                    help='Skip simHeaven X-World network road exclusion')
    ap.add_argument('--no-avoid-simheaven-buildings', action='store_true',
                    dest='no_avoid_simheaven_buildings',
                    help='Do not exclude simHeaven building footprints/objects from generated vegetation.')
    ap.add_argument('--simheaven-building-buffer-m', type=float, default=10.0,
                    dest='simheaven_building_buffer_m',
                    help='Extra exclusion buffer in metres around simHeaven building footprints/objects.')
    ap.add_argument('--no-avoid-gfv2', action='store_true', dest='no_avoid_gfv2',
                    help='Do not exclude Global Forests v2 polygons from generated vegetation.')
    ap.add_argument('--gfv2-buffer-m', type=float, default=0.0, dest='gfv2_buffer_m',
                    help='Extra exclusion buffer in metres around Global Forests v2 polygons.')
    ap.add_argument('--gfv2-asset-proximity', action='store_true',
                    dest='gfv2_asset_proximity',
                    help='Use the tile-dominant Global Forests v2 polygon type to choose generated vegetation asset types.')
    ap.add_argument('--no-avoid-simheaven-forests', action='store_true',
                    dest='no_avoid_simheaven_forests',
                    help='Do not exclude simHeaven forest polygons from generated vegetation.')
    ap.add_argument('--simheaven-buffer-m', type=float, default=0.0, dest='simheaven_buffer_m',
                    help='Extra exclusion buffer in metres around simHeaven forest polygons.')
    ap.add_argument('--no-avoid-default-forests', action='store_true',
                    dest='no_avoid_default_forests',
                    help='Do not exclude default-overlay forest polygons from generated vegetation.')
    ap.add_argument('--default-buffer-m', type=float, default=0.0, dest='default_buffer_m',
                    help='Extra exclusion buffer in metres around default-overlay forest polygons.')
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
        download_veg_context = not args.no_download_veg_context,
        custom_scenery_dir = args.custom_scenery_dir,
        custom_overlay_src = args.custom_overlay_src,
        custom_overlay_src_alternate = args.custom_overlay_src_alternate,
        avoid_simheaven_buildings = not args.no_avoid_simheaven_buildings,
        simheaven_building_buffer_m = args.simheaven_building_buffer_m,
        avoid_gfv2       = not args.no_avoid_gfv2,
        gfv2_buffer_m    = args.gfv2_buffer_m,
        use_gfv2_asset_proximity = args.gfv2_asset_proximity,
        avoid_simheaven_forests = not args.no_avoid_simheaven_forests,
        simheaven_buffer_m = args.simheaven_buffer_m,
        avoid_default_forests = not args.no_avoid_default_forests,
        default_buffer_m = args.default_buffer_m,
    )
    print(f"\nDone: {n} vegetation polygons in {(time.time()-t0)/60:.1f}min")
    return n


if __name__ == '__main__':
    main()

