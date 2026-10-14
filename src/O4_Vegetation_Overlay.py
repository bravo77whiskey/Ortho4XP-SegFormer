"""
O4_Vegetation_Overlay.py — Per-DDS vegetation DSF overlay from SegFormer inference.

Reuses the inference cache written by the building overlay.

Pipeline
--------
1. Discover ZL16 DDS files in tex_dir
2. Load cached SegFormer veg_map per DDS at native resolution (~2.4 m/px at ZL16)
3. Per DDS: extract CLASS_TREE / CLASS_RANGELAND masks
4. Morphological close → open (kernel sizes in metres)
5. Vectorise (OpenCV contours) → Douglas-Peucker → lat/lon polygons
6. Per polygon: shape ratio → area-fill | treeline
7. Leaf-type map zones (O4_Leaf_Type) split each tree mask into broad /
   conifer / mixed; pixel fill fraction → simHeaven .for path + density
8. Write DSF text, compile with DSFTool

Usage:
    python src/scripts/generate_veg_overlay.py <tex_dir> <lat> <lon> [options]

Options:
    --cache-dir DIR   Inference cache (default: <o4xp_root>/Overlay_cache/<tile>)
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

from O4_Overlay_Logging import verbose_print, with_verbose_logging
import os, argparse, warnings, time, math, json
warnings.filterwarnings('ignore')

import hashlib
import numpy as np
import cv2
from math import pi, atan, exp
from PIL import Image

import O4_Forest_Assets as FOREST_ASSETS
import O4_Climate_Regions as CLIMATE_REGIONS
import O4_Leaf_Type as LEAF_TYPE
import O4_Bounds_Index as BBOX
import O4_Persistent_Cache as PCACHE
import O4_Progress as PROG
import O4_Landcover_Inference as SEGFORMER
import O4_Texture_Selection as TEXSEL
from O4_Building_Overlay import (
    _load_simheaven_building_exclusions,
    _load_simheaven_network as _load_simheaven_network_typed,
    _load_mesh_water_index,
    _mesh_file_for_tile,
    _mesh_water_signature,
    _prepare_simheaven_objects,
    _rasterize_mesh_water_mask,
    _rasterize_simheaven_objects,
    _simheaven_objects_for_bounds,
    covered_rects_to_px,
    mask_covered_regions,
)
from O4_Scenery_Packs import (
    ensure_cached_dsf_text,
    find_active_custom_scenery_dsfs,
    find_simheaven_forest_dsfs,
    find_simheaven_network_dsfs,
)
from O4_Region_Boundaries import asset_region_for_latlon

# ── Defaults (all spatial params in metres) ───────────────────────────────────
CLOSE_M       = 10.0   # close kernel radius — fill gaps within a patch
OPEN_M        = 3.0    # open  kernel radius — remove sub-pixel noise
MIN_AREA_M2   = 50.0   # minimum polygon area (~7×7 m)
SIMPLIFY_M    = 3.0    # Douglas-Peucker tolerance
TREELINE_RATIO= 5.0    # perimeter² / (4π × area) ≥ this → candidate treeline
TREELINE_MAX_WIDTH_M = 30.0  # max inscribed width — wider cores stay filled forest
TREELINE_MIN_ASPECT  = 3.0   # min length/width — compact patches stay filled forest
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

# Exclusion half-widths for typed simHeaven network segments (road.net subtype
# → OSM class, see O4_Building_Overlay._XP_ROAD_SUBTYPE_RANGES): the carriageway
# X-Plane draws plus about a metre, so kerbside street trees survive. Untyped
# 'network' segments are simHeaven's powerlines and cableways.
SIMHEAVEN_ROAD_HALF_WIDTH_M = {
    'motorway': 13, 'trunk': 10, 'motorway_link': 6,
    'primary': 8, 'secondary': 6, 'residential': 4, 'service': 3,
    'rail': 4, 'network': 3,
}

# Small canopy blobs become point-mode forests: X-Plane's forests plant on a
# SPACING grid (12 m for lib/g8), so a blob under a few grid cells usually gets
# no tree at all. Instead one tree goes on each crown, picked greedily from the
# deepest canopy pixel outwards, at least POINT_TREE_SPACING_M apart.
POINT_TREE_MAX_AREA_M2 = 600.0
POINT_TREE_SPACING_M = 8.0
POINT_TREE_MIN_DEPTH_M = 1.5
POINT_MODE = 512
LINE_MODE = 256


def _veg_flag(name):
    """Vegetation experiment switches: on unless the env var says 0/off."""
    value = os.environ.get(name)
    return value is None or value.strip().lower() not in {"0", "false", "no", "off"}


def _road_widths_enabled():
    """O4_VEG_ROAD_WIDTHS=0: every simHeaven segment excludes a flat 16 m."""
    return _veg_flag("O4_VEG_ROAD_WIDTHS")


def _road_width_scale():
    """O4_VEG_ROAD_WIDTH_SCALE: multiplies every road clearance.

    Default 0: roads remove no canopy, so street trees and canopy over the
    carriageway are planted (user choice after the 2026-10-07 review). 1 uses
    the road-type widths in SIMHEAVEN_ROAD_HALF_WIDTH_M / ROAD_EXCL_HALF_WIDTH_M.
    """
    try:
        return max(0.0, float(os.environ.get("O4_VEG_ROAD_WIDTH_SCALE", "0")))
    except ValueError:
        return 0.0


def _road_clip_enabled():
    """O4_VEG_ROAD_CLIP=0: clamp road points to the texture edge (old bug)."""
    return _veg_flag("O4_VEG_ROAD_CLIP")


def _point_trees_enabled():
    """O4_VEG_POINT_TREES=0: small canopy blobs stay area-fill forests."""
    return _veg_flag("O4_VEG_POINT_TREES")


def _veg_feature_sig():
    """Cache-key part for the switches above and the tree-support rule."""
    return (
        _road_widths_enabled(),
        _road_clip_enabled(),
        _road_width_scale(),
        _point_trees_enabled(),
        SEGFORMER.tree_support_enabled(),
        tuple(sorted(SIMHEAVEN_ROAD_HALF_WIDTH_M.items())),
        POINT_TREE_MAX_AREA_M2, POINT_TREE_SPACING_M, POINT_TREE_MIN_DEPTH_M,
        SEGFORMER.TREE_SUPPORT_MIN_PROB,
    )


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


def _veg_context_from_simheaven(forest_records):
    """Forest-asset context from the installed simHeaven 7-forests polygons.

    SegFormer alone decides where trees go; this context only steers which
    forest assets a tree patch gets (see ``_classify_tree_cover``).
    """
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
    for record in forest_records or ():
        path = (record.get('path') or '').replace('\\', '/').lower()
        if 'orchard' in path:
            key = 'orchard_polys'
        elif 'wetland' in path:
            key = 'water_polys'
        elif 'shrub' in path or 'scrub' in path:
            key = 'woodland_hint_polys'
        elif 'hedge' in path or 'random_tree' in path:
            continue
        else:
            key = 'forest_polys'
        context[key].append(record['pts'])
    return context


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
            verbose_print(f"  [simHeaven] {folder_name}: +{len(road_ways) - segment_count_before} segments")
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
    scale = _road_width_scale()
    if scale <= 0:
        return mask

    clip = _road_clip_enabled()

    def ll_to_px(lat, lon):
        x = int((lon - lon_w) / (lon_e - lon_w) * img_w)
        y = int((lat_n - lat) / (lat_n - lat_s) * img_h)
        if clip:
            # cv2.line clips to the image itself; clamping each end to the
            # edge separately bends lines into phantom wedges along it.
            return (x, y)
        return (max(0, min(img_w - 1, x)),
                max(0, min(img_h - 1, y)))

    for road in roads:
        half_m   = road.get('half_m') or ROAD_EXCL_HALF_WIDTH_M.get(road['type'], _ROAD_DEFAULT_HALF_M)
        half_m  *= scale
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

    clip = _road_clip_enabled()

    def ll_to_px(lat, lon):
        x = int((lon - lon_w) / (lon_e - lon_w) * img_w)
        y = int((lat_n - lat) / (lat_n - lat_s) * img_h)
        if clip:
            # cv2.line clips to the image itself; clamping each end to the
            # edge separately bends lines into phantom wedges along it.
            return (x, y)
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
                        simheaven_building_buffer_m, mesh_water_sig=None,
                        excl_zone_sig=None):
    return {
        # 4: 0/1 masks stored bit-packed (see _pack_cache_mask)
        'version': 4,
        'fname': fname,
        'bounds': tuple(round(v, 8) for v in (lat_n, lat_s, lon_w, lon_e)),
        'shape': (int(img_h), int(img_w)),
        'mpp': round(float(mpp), 6),
        'road_sig': road_sig,
        'res_road_sig': res_road_sig,
        'tree_row_sig': tree_row_sig,
        'context_poly_sigs': tuple(sorted(context_poly_sigs.items())),
        'forest_layer_sigs': forest_layer_sigs,
        'excl_zone_sig': excl_zone_sig,
        'sh_bld_poly_sig': sh_bld_poly_sig,
        'sh_bld_obj_sig': sh_bld_obj_sig,
        'bld_excl_m': round(float(bld_excl_m), 4),
        'bld_cache_stat': bld_cache_stat,
        'simheaven_building_buffer_m': round(float(simheaven_building_buffer_m), 4),
        'mesh_water_sig': mesh_water_sig,
        'residential_buffer_m': float(RESIDENTIAL_FALLBACK_BUFFER_M),
        'residential_buffer_px_min': int(RESIDENTIAL_FALLBACK_BUFFER_PX_MIN),
        'tree_row_width_m': float(TREE_ROW_WIDTH_M),
        'veg_features': _veg_feature_sig(),
    }


def _or_optional_masks(lhs, rhs):
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    return cv2.bitwise_or(lhs, rhs)


def _bld_cache_objects(bld_data):
    """Placed objects ``(lon, lat, heading, path)`` from a ``*_bld.pkl``.

    The building pass nests them under ``'placements'``; older caches kept
    ``'objects'`` at the top level.
    """
    if not isinstance(bld_data, dict):
        return ()
    placements = bld_data.get('placements')
    if isinstance(placements, dict) and placements.get('objects'):
        return placements['objects']
    return bld_data.get('objects') or ()


def _pack_cache_mask(mask):
    """Bit-pack a 2-D 0/1 uint8 mask for the mask cache; anything else as is.

    The cached masks are full-texture rasters (16 MB each at 4096 px), a
    tile's worth was ~10 GB of writes per run; packed they are 1/8 of that
    and round-trip exactly.
    """
    if (
        isinstance(mask, np.ndarray) and mask.dtype == np.uint8 and
        mask.ndim == 2 and not np.any(mask > 1)
    ):
        return {'__bits__': np.packbits(mask, axis=None), 'shape': mask.shape}
    return mask


def _unpack_cache_mask(value):
    if isinstance(value, dict) and '__bits__' in value:
        h, w = value['shape']
        return np.unpackbits(value['__bits__'], count=h * w).reshape(h, w)
    return value


def _load_dds_mask_cache(cache_path, key):
    import pickle as _pickle
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, 'rb') as f:
            data = _pickle.load(f)
        if data.get('key') == key:
            for name, value in list(data.items()):
                if name == 'context_masks' and isinstance(value, dict):
                    data[name] = {
                        k: _unpack_cache_mask(v) for k, v in value.items()
                    }
                elif name != 'key':
                    data[name] = _unpack_cache_mask(value)
            return data
    except Exception:
        return None
    return None


def _save_dds_mask_cache(cache_path, key, payload):
    import pickle as _pickle
    packed = {}
    for name, value in payload.items():
        if name == 'context_masks' and isinstance(value, dict):
            packed[name] = {k: _pack_cache_mask(v) for k, v in value.items()}
        else:
            packed[name] = _pack_cache_mask(value)
    try:
        with open(cache_path, 'wb') as f:
            _pickle.dump({'key': key, **packed}, f, protocol=_pickle.HIGHEST_PROTOCOL)
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
    asset_selection_mode="simheaven_climate",
    forest_type_source_path=None,
    forest_type_sig=None,
    covered_fracs=None,
):
    return {
        'version': 7,
        'forest_type_sig': forest_type_sig,
        'covered_fracs': covered_fracs,
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
        'forest_type_source_path': forest_type_source_path,
        'veg_features': _veg_feature_sig(),
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

            verbose_print(
                f"  [{layer_name}] {folder_name}: +{len(polys) - n_poly0} forest polys"
            )
        except Exception as exc:
            print(f"  [{layer_name}] failed {dsf_path}: {exc}")

    return polys


# ── Custom-scenery exclusion zones ───────────────────────────────────────────
# A DSF exclusion zone is a header property, one per excluded type:
#
#     PROPERTY sim/exclude_for  lon_west/lat_south/lon_east/lat_north
#
# WED writes one such line per type ticked on a single exclusion rectangle, so
# one zone that excludes forests, objects and facades appears here as three
# lines sharing the same rectangle under three different keys. Only the keys in
# `FOREST_EXCLUSION_PROPERTY_KEYS` suppress vegetation: this overlay emits
# nothing but `.for` polygons, so an object- or facade-only zone must be left
# alone, and a mixed zone counts once.
FOREST_EXCLUSION_PROPERTY_KEYS = ("sim/exclude_for",)
_EXCLUSION_PROPERTY_KEY_PREFIX = "sim/exclude_"

# Only a custom scenery *airport* may veto vegetation. Regional and global packs
# use a tile-sized `sim/exclude_for` as a "my forests replace the stock ones"
# marker rather than as an authored keep-out zone — Global Forests v2 stamps
# `38/11/39/12` on every one of its 37k tiles — and honouring that erases the
# whole generated overlay instead of a clearing inside it.
#
# A pack counts as an airport when it defines airports (`Earth nav data/apt.dat`)
# and stays inside a handful of degree tiles. Both halves are needed: regional
# VFR packs ship an apt.dat too, and at least one of them (Bhutan_VFR, 15 tiles)
# declares a whole-tile forest exclusion.
AIRPORT_PACK_MAX_DSFS = 8


def _is_airport_scenery_pack(package_dir):
    """True when a scenery package is a custom airport rather than area scenery."""
    earth_nav_data = os.path.join(package_dir, "Earth nav data")
    if not os.path.isfile(os.path.join(earth_nav_data, "apt.dat")):
        return False
    n_dsf = 0
    for _root, _dirs, files in os.walk(earth_nav_data):
        for name in files:
            if name.lower().endswith(".dsf"):
                n_dsf += 1
                if n_dsf > AIRPORT_PACK_MAX_DSFS:
                    return False
    return True


def _parse_exclusion_rect(value):
    """Parse an exclusion property value into a lon/lat ring, or None."""
    parts = (value or "").split("/")
    if len(parts) != 4:
        return None
    try:
        west, south, east, north = (float(part) for part in parts)
    except ValueError:
        return None
    west, east = min(west, east), max(west, east)
    south, north = min(south, north), max(south, north)
    if east <= west or north <= south:
        return None
    return [(west, south), (east, south), (east, north), (west, north)]


def _zones_from_property_pairs(pairs):
    """Group ``(key, value)`` header properties into exclusion rings by type."""
    zones = {}
    for key, value in pairs:
        if not key.startswith(_EXCLUSION_PROPERTY_KEY_PREFIX):
            continue
        ring = _parse_exclusion_rect(value)
        if ring is None:
            continue
        zones.setdefault(key, []).append(ring)
    return zones


def effective_avoid_simheaven_forests(avoid_simheaven_forests,
                                      exclude_lower_forests):
    """Resolve the overlap/replacement conflict for one generation run.

    Avoidance keeps generated trees out of the existing simHeaven forests;
    replacement then excludes those same forests. Honouring both would leave a
    bare hole exactly where the old forest was, so replacement wins. Only the
    value used for this run changes; the user's saved setting is untouched and
    resumes its normal behaviour as soon as replacement is off.
    """
    if exclude_lower_forests:
        return False
    return bool(avoid_simheaven_forests)


def tile_exclusion_property_value(lat, lon):
    """Return the ``west/south/east/north`` value covering one whole 1x1 tile.

    Floor, not truncation, so the tile south/west of the equator or prime
    meridian still gets the degree square it actually occupies: lat -34.2 is
    tile -35..-34, and ``int()`` would have produced -34..-33.
    """
    lat_i = int(math.floor(float(lat)))
    lon_i = int(math.floor(float(lon)))
    return f"{lon_i}/{lat_i}/{lon_i + 1}/{lat_i + 1}"


def validate_compiled_overlay(dsf_path, expected_exclusion_value=None):
    """Check a freshly compiled overlay before it replaces the installed one.

    Returns None when the file is usable, otherwise a short reason. DSFTool
    reports success on some inputs it has not actually turned into a loadable
    tile, and a tile-wide exclusion is worth proving rather than assuming: an
    overlay that silently lost its `sim/exclude_for` would leave the old
    forests and the new trees stacked on top of each other.
    """
    try:
        size = os.path.getsize(dsf_path)
    except OSError as exc:
        return f"{dsf_path} is missing ({exc})"
    if size < 64:
        return f"{dsf_path} is only {size} bytes"
    try:
        pairs = _read_dsf_binary_properties(dsf_path)
    except Exception as exc:
        return f"header unreadable ({exc})"
    if pairs is None:
        return "not a plain uncompressed DSF"
    properties = dict(pairs)
    if properties.get("sim/overlay") != "1":
        return "sim/overlay is not 1"
    key = FOREST_EXCLUSION_PROPERTY_KEYS[0]
    found = [value for name, value in pairs if name == key]
    if expected_exclusion_value is None:
        if found:
            return f"unexpected {key} {found[0]}"
        return None
    if not found:
        return f"{key} missing"
    if len(found) != 1:
        return f"{len(found)} {key} properties, expected 1"
    if _parse_exclusion_rect(found[0]) != _parse_exclusion_rect(expected_exclusion_value):
        return f"{key} is {found[0]}, expected {expected_exclusion_value}"
    return None


def write_generation_status(status_path, payload):
    """Record how this generation run ended, for the caller across a subprocess.

    The overlay runs inside the .venv subprocess, so an exit code is all the
    pipeline would otherwise see. It needs more than that: whether the tile was
    really processed, whether a tile-wide exclusion was installed, and whether
    the compile succeeded, because embedded-tree cleanup must not start after a
    partial or failed run.
    """
    if not status_path:
        return
    import json as _json
    try:
        os.makedirs(os.path.dirname(os.path.abspath(status_path)), exist_ok=True)
        tmp = status_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            _json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, status_path)
    except Exception as exc:  # a status file is never worth failing a build for
        print(f"[Veg] could not write status file {status_path}: {exc}")


def _read_dsf_binary_properties(dsf_path):
    """Read the header key/value properties straight out of a binary DSF.

    Exclusion zones live in the HEAD/PROP atom at the very start of the file,
    so this reads a few kilobytes instead of asking DSFTool to disassemble a
    whole mesh tile. Returns None when the file is not a plain uncompressed
    DSF (7z-packed global scenery, for instance) so the caller can fall back
    to a text disassembly.
    """
    import struct

    with open(dsf_path, "rb") as handle:
        if handle.read(8) != b"XPLNEDSF":
            return None
        handle.read(4)  # format version
        while True:
            atom_header = handle.read(8)
            if len(atom_header) < 8:
                return None
            atom_id = atom_header[:4]
            atom_len = struct.unpack("<I", atom_header[4:])[0]
            if atom_len < 8:
                return None
            if atom_id != b"DAEH":  # 'HEAD', stored little-endian
                handle.seek(atom_len - 8, os.SEEK_CUR)
                continue
            head = handle.read(atom_len - 8)
            break

    offset = 0
    while offset + 8 <= len(head):
        sub_id = head[offset:offset + 4]
        sub_len = struct.unpack("<I", head[offset + 4:offset + 8])[0]
        if sub_len < 8:
            return None
        payload = head[offset + 8:offset + sub_len]
        offset += sub_len
        if sub_id != b"PORP":  # 'PROP', stored little-endian
            continue
        strings = payload.split(b"\0")
        if strings and strings[-1] == b"":
            strings.pop()
        decoded = [item.decode("utf-8", "replace") for item in strings]
        return list(zip(decoded[0::2], decoded[1::2]))
    return []


def _read_dsf_text_properties(text_path):
    """Read header key/value properties from a DSFTool text disassembly."""
    pairs = []
    with open(text_path, "r", encoding="utf-8", errors="ignore") as text_file:
        for raw_line in text_file:
            line = raw_line.strip()
            if not line.startswith("PROPERTY "):
                continue
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            pairs.append((parts[1], parts[2].strip()))
    return pairs


def _load_custom_scenery_forest_exclusions(custom_scenery_dir, tile_lat, tile_lon,
                                           out_dsf, dsftool_path, cache_dir):
    """Collect forest exclusion rectangles from the active airport scenery packs.

    Returns ``(rings, type_counts)`` where ``rings`` are the zones that exclude
    forests and ``type_counts`` tallies every exclusion type seen, so the caller
    can report how much of a mixed zone set actually applies to vegetation.

    Only custom airports are consulted (see ``_is_airport_scenery_pack``); the
    zones of regional and global packs describe layer ownership, not ground the
    author cleared, and cover whole tiles.
    """
    rings = []
    type_counts = {}
    seen_rects = set()
    skipped_packs = []
    dsf_name = os.path.basename(out_dsf or "")
    if not dsf_name:
        lat_i, lon_i = int(tile_lat), int(tile_lon)
        dsf_name = (
            f"{'+' if lat_i >= 0 else '-'}{abs(lat_i):02d}"
            f"{'+' if lon_i >= 0 else '-'}{abs(lon_i):03d}.dsf"
        )

    dsf_matches = find_active_custom_scenery_dsfs(
        custom_scenery_dir,
        dsf_name,
        skip_dsf_path=out_dsf,
    )
    for folder_name, dsf_path, package_dir in dsf_matches:
        # Checked before the header is read: it also spares us a DSFTool
        # disassembly of the 7z-packed global packs, which is the slow path.
        if not _is_airport_scenery_pack(package_dir):
            skipped_packs.append(folder_name)
            continue
        try:
            pairs = _read_dsf_binary_properties(dsf_path)
            if pairs is None:
                # Not a plain DSF (7z-packed, or a format we don't recognise) —
                # pay for a full disassembly only in that case.
                if not dsftool_path or not os.path.exists(dsftool_path):
                    print(
                        f"  [excl zones] {folder_name}: unreadable DSF and no "
                        f"DSFTool at {dsftool_path} — zones ignored"
                    )
                    continue
                cached_text_path = ensure_cached_dsf_text(
                    dsf_path,
                    dsftool_path,
                    cache_dir,
                    create_no_window=SEGFORMER._CREATE_NO_WINDOW,
                )
                pairs = _read_dsf_text_properties(cached_text_path)
            zones = _zones_from_property_pairs(pairs)
        except Exception as exc:
            print(f"  [excl zones] failed {dsf_path}: {exc}")
            continue

        n_pack_forest = 0
        for key, key_rings in (zones or {}).items():
            type_counts[key] = type_counts.get(key, 0) + len(key_rings)
            if key not in FOREST_EXCLUSION_PROPERTY_KEYS:
                continue
            for ring in key_rings:
                # The same rectangle can arrive from several packs (and, for a
                # mixed zone, under several keys); one copy is enough.
                rect_key = tuple(round(coord, 9) for pt in ring for coord in pt)
                if rect_key in seen_rects:
                    continue
                seen_rects.add(rect_key)
                rings.append(ring)
                n_pack_forest += 1
        if zones:
            verbose_print(
                f"  [excl zones] {folder_name}: {n_pack_forest} forest zones "
                f"of {sum(len(v) for v in zones.values())} exclusion properties"
            )
    if skipped_packs:
        verbose_print(
            f"  [excl zones] {len(skipped_packs)} non-airport pack(s) not "
            f"consulted: {', '.join(sorted(skipped_packs))}"
        )
    return rings, type_counts


def _dominant_simheaven_tree_path(records):
    """Return the most common selectable X-World tree path in a tile."""
    counts = {}
    for record in records or ():
        path = record.get('path') if isinstance(record, dict) else None
        normalized = FOREST_ASSETS.normalize_simheaven_tree_path(path)
        if normalized is None:
            continue
        counts[normalized] = counts.get(normalized, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


class _SimHeavenTypeResolver:
    """Resolve the nearest selectable X-World forest family per polygon.

    ``resolve(lon, lat)`` returns the source path of the nearest acceptable
    X-World tree polygon within ``max_radius_m`` by bounding-box distance. It
    falls back to the tile-dominant path when none is close enough. Lookups are
    vectorised over all candidate bounds and memoised on a roughly 250 m grid.
    """

    def __init__(self, records, dominant_path, tile_lat,
                 max_radius_m=3000.0, cell_deg=0.0025):
        self.dominant_path = dominant_path
        self.max_radius_m = float(max_radius_m)
        self.cell_deg = float(cell_deg)
        self._memo = {}
        self._m_lat = 110540.0
        self._m_lon = 111320.0 * max(0.05, math.cos(math.radians(tile_lat + 0.5)))
        souths, norths, wests, easts, paths = [], [], [], [], []
        for record in records or ():
            if not isinstance(record, dict):
                continue
            path = record.get('path')
            bounds = record.get('_bounds')
            normalized = FOREST_ASSETS.normalize_simheaven_tree_path(path)
            if bounds is None or normalized is None:
                continue
            south, north, west, east = bounds
            souths.append(float(south)); norths.append(float(north))
            wests.append(float(west)); easts.append(float(east))
            paths.append(normalized)
        self.n = len(paths)
        if self.n:
            self._south = np.asarray(souths)
            self._north = np.asarray(norths)
            self._west = np.asarray(wests)
            self._east = np.asarray(easts)
            self._paths = paths

    def resolve(self, lon, lat):
        if not self.n:
            return self.dominant_path
        key = (int(lon / self.cell_deg), int(lat / self.cell_deg))
        cached = self._memo.get(key)
        if cached is not None:
            return cached[0]
        dx = np.maximum(0.0, np.maximum(self._west - lon, lon - self._east))
        dy = np.maximum(0.0, np.maximum(self._south - lat, lat - self._north))
        dist2 = (dx * self._m_lon) ** 2 + (dy * self._m_lat) ** 2
        idx = int(np.argmin(dist2))
        if dist2[idx] <= self.max_radius_m ** 2:
            result = self._paths[idx]
        else:
            result = self.dominant_path
        self._memo[key] = (result,)
        return result


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
    """Return stable simHeaven forest library paths."""
    return FOREST_ASSETS.short_tree_candidates(region, dlevel)


def _for_entry(veg_cls, frac, shape, region, rng,
               density_override=None, veg_type=None,
               simheaven_type_path=None):
    """Return (for_path, dsf_density) for a polygon."""
    dlevel  = _for_density_level(density_override) if density_override is not None \
              else _density_level(frac)

    if veg_cls == SEGFORMER.CLASS_TREE:
        if veg_type in {'natural_woodland_open', 'uncertain_tree_cover'}:
            base_key = 'woodland'
        elif veg_type == 'riparian_trees':
            base_key = 'woodland'
            dlevel = min(dlevel, 50 if shape == 'treeline' else 75)
        elif veg_type in {'settlement_trees', 'park_or_managed_green'}:
            # Keep coverage, but bias managed/settlement canopy toward the safer
            # woodland assets instead of full mixed-forest sets.
            base_key = 'woodland'
            dlevel = min(dlevel, 50)
        elif veg_type in {'orchard_or_plantation', 'tree_row_linear'}:
            base_key = 'woodland'
            dlevel = 25 if shape == 'treeline' else min(dlevel, 50)
        else:
            # Future generated overlays stay under the Brazilian-nut height cap.
            # Dense forest gets denser placement, not taller asset families.
            base_key = 'tree' if shape == 'area' and dlevel >= 75 else 'woodland'
        base     = _BASE_DENSITY[base_key][dlevel]
    elif veg_cls == SEGFORMER.CLASS_RANGELAND:
        base_key = 'woodland' if dlevel >= 50 else 'shrub'
        base     = _BASE_DENSITY[base_key][dlevel]
    else:   # AGRICULTURE treelines only
        base   = _BASE_DENSITY['shrub'][25]
        dlevel = 25

    if density_override is not None:
        base = int(round(density_override * 255))

    tree_context = 'treeline' if shape == 'treeline' else 'bulk'
    if veg_type in {'settlement_trees', 'park_or_managed_green'}:
        tree_context = 'managed'
    if veg_cls == SEGFORMER.CLASS_TREE and simheaven_type_path:
        candidates = FOREST_ASSETS.simheaven_type_hint_candidates(
            simheaven_type_path,
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
    dsf_density = base + 256 if shape == 'treeline' else base
    return path, dsf_density


# ── Visual-review dump ────────────────────────────────────────────────────────
def _review_dump_enabled():
    """O4_BLD_REVIEW_DUMP=1 (set by the review harness) also dumps tree cuts."""
    value = os.environ.get("O4_BLD_REVIEW_DUMP", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _write_veg_review(path, img_w, img_h, layers):
    """Write canopy the exclusion layers removed, as rings in 4096-px texture space."""
    sx, sy = 4096.0 / img_w, 4096.0 / img_h
    out = {}
    for name, mask in layers.items():
        rings = []
        if mask is not None and mask.any():
            cnts, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in cnts:
                if cv2.contourArea(cnt) < 2.0:
                    continue
                cnt = cv2.approxPolyDP(cnt, 1.0, closed=True)
                if len(cnt) < 3:
                    continue
                rings.append([[round(float(x) * sx, 1), round(float(y) * sy, 1)]
                              for x, y in cnt[:, 0, :]])
        out[name] = rings
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, separators=(",", ":"))
    except OSError as exc:
        print(f"    veg review dump failed: {exc}", flush=True)


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
    if ratio < TREELINE_RATIO:
        return 'area', area
    # The raw pixel perimeter overstates elongation for jagged or dendritic
    # blobs (treelines render edge-only via the +256 DSF param, hollowing out
    # any misclassified patch), so candidates must also have a genuinely
    # narrow core: max inscribed width via distance transform, plus a minimum
    # length/width aspect.
    stats = _contour_fill_stats(cnt, m_per_px, include_ring=False)
    if stats is None:
        return 'tiny', area
    padded = cv2.copyMakeBorder(stats['fill'], 1, 1, 1, 1,
                                cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    core_width_m = 2.0 * float(dist.max()) * m_per_px
    if core_width_m > TREELINE_MAX_WIDTH_M or core_width_m <= 0:
        return 'area', area
    area_m2 = stats['area_px'] * m_per_px * m_per_px
    length_m = area_m2 / core_width_m
    if length_m / core_width_m < TREELINE_MIN_ASPECT:
        return 'area', area
    return 'treeline', area


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


def _crown_points(mask, prepared, m_per_px):
    """Crown centres in a small canopy blob, as image pixel (x, y) pairs.

    Greedy: the deepest canopy pixel (distance transform) takes a tree, a
    POINT_TREE_SPACING_M disc around it is cleared, repeat. Later trees must
    sit at least POINT_TREE_MIN_DEPTH_M inside the canopy; the first always
    goes in, so every blob that survives the filters gets one tree.
    """
    x, y, fill = prepared['x'], prepared['y'], prepared['fill']
    h, w = fill.shape
    canopy = (mask[y:y + h, x:x + w] & fill).astype(np.uint8)
    if not canopy.any():
        canopy = fill
    padded = cv2.copyMakeBorder(canopy, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    spacing_px = max(1, int(round(POINT_TREE_SPACING_M / max(m_per_px, 1e-6))))
    min_depth_px = POINT_TREE_MIN_DEPTH_M / max(m_per_px, 1e-6)
    pts = []
    while True:
        _, depth, _, (px, py) = cv2.minMaxLoc(dist)
        if depth <= 0 or (pts and depth < min_depth_px):
            break
        pts.append((x + px - 1, y + py - 1))
        cv2.circle(dist, (px, py), spacing_px, 0, -1)
    return pts


def _write_point_windings(f, pts):
    """Point-mode forest: one tree per point, so the ring is never closed."""
    for start in range(0, len(pts), MAX_RING_PTS):
        f.write("BEGIN_WINDING\n")
        for lo, la in pts[start:start + MAX_RING_PTS]:
            f.write(f"POLYGON_POINT {lo:.7f} {la:.7f}\n")
        f.write("END_WINDING\n")


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
                      simheaven_type_path=None,
                      simheaven_type_resolver=None,
                      forest_path_mapper=None):
    """Extract polygons from one DDS class mask. Returns list of (path, density, ring).

    ``forest_path_mapper.map_path(path, lon, lat, veg_type)`` may swap the
    chosen tree family for a climate-specific X-Plane forest at the polygon.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys   = []
    point_trees = _point_trees_enabled()
    for cnt in cnts:
        shape, area_px = _contour_shape(cnt, min_area_px, m_per_px)
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
        prepared_fill = None
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

        poly_simheaven_type_path = simheaven_type_path
        if (
            veg_cls == SEGFORMER.CLASS_TREE
            and simheaven_type_resolver is not None
        ):
            c_lon = sum(p[0] for p in ring) / len(ring)
            c_lat = sum(p[1] for p in ring) / len(ring)
            poly_simheaven_type_path = (
                simheaven_type_resolver.resolve(c_lon, c_lat)
                or simheaven_type_path
            )
        fpath, dsf_den = _for_entry(
            veg_cls, frac, shape, region, rng, density_override,
            veg_type=veg_type,
            simheaven_type_path=poly_simheaven_type_path,
        )
        if not fpath:
            continue
        if forest_path_mapper is not None and veg_cls == SEGFORMER.CLASS_TREE:
            c_lon = sum(p[0] for p in ring) / len(ring)
            c_lat = sum(p[1] for p in ring) / len(ring)
            fpath = forest_path_mapper.map_path(fpath, c_lon, c_lat, veg_type)
        if (
            point_trees and prepared_fill is not None and shape == 'area'
            and area_px * m_per_px * m_per_px < POINT_TREE_MAX_AREA_M2
        ):
            pts = []
            for px, py in _crown_points(mask, prepared_fill, m_per_px):
                lo = lon_w + px / img_w * (lon_e - lon_w)
                la = lat_n - py / img_h * (lat_n - lat_s)
                if tile_lon <= lo < tile_lon + 1 and tile_lat <= la < tile_lat + 1:
                    pts.append((lo, la))
            if pts:
                polys.append((fpath, (dsf_den % LINE_MODE) + POINT_MODE, pts))
            continue
        polys.append((fpath, dsf_den, ring))
    return polys


# ── Core pipeline ─────────────────────────────────────────────────────────────
@with_verbose_logging
def run(tex_dir, lat, lon, out_dsf, cache_dir,
        close_m, open_m, min_area_m2, simplify_m,
        make_viz, density_override=None, res_m=None, disable_cache=False,
        excl_buffer_m=EXCL_BUFFER_M,
        bld_excl_m=10.0,
        osm_roads_path=None, use_simheaven=True, dsftool_path=None,
        custom_scenery_dir=None,
        avoid_simheaven_buildings=True, simheaven_building_buffer_m=10.0,
        avoid_simheaven_forests=True, simheaven_forest_buffer_m=0.0,
        use_simheaven_asset_proximity=True,
        exclude_lower_forests=False,
        status_path=None,
        avoid_gfv2=False, gfv2_buffer_m=0.0,
        use_gfv2_asset_proximity=False,
        use_leaf_type_maps=True, leaf_type_cache_dir=None,
        use_xp_forests=True, grass_settings=None, verbose_log=False):

    started = time.perf_counter()

    def _print_tile_summary(tree_polygons=0, grass_polygons=0):
        tile = f"{int(math.floor(lat)):+03d}{int(math.floor(lon)):+04d}"
        print(
            f"[Veg] Tile {tile}: tree polygons={tree_polygons:,}; "
            f"grass polygons={grass_polygons:,}; "
            f"total generation time={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    import re as _re
    STD_RE = _re.compile(r"^(\d+)_(\d+)_([A-Za-z][A-Za-z0-9_]*)(\d{2})\.dds$",
                         _re.IGNORECASE)

    # Tile replacement and forest-overlap avoidance are mutually exclusive.
    # Avoidance keeps generated trees out of existing simHeaven forests;
    # replacement then excludes those same forests, so honouring both would
    # leave a bare hole exactly where the old forest used to be. Replacement
    # wins for this run only — the user's saved setting is never rewritten.
    exclude_lower_forests = bool(exclude_lower_forests)
    requested_avoid_simheaven_forests = bool(avoid_simheaven_forests)
    avoid_simheaven_forests = effective_avoid_simheaven_forests(
        requested_avoid_simheaven_forests, exclude_lower_forests
    )
    if requested_avoid_simheaven_forests and not avoid_simheaven_forests:
        verbose_print(
            "[Veg] tile forest replacement on: forest-overlap avoidance "
            "forced off for this run (saved setting unchanged) so replaced "
            "forests do not leave holes"
        )
    exclusion_value = tile_exclusion_property_value(lat, lon)
    status = {
        'tile': f"{'+' if int(math.floor(lat)) >= 0 else '-'}"
                f"{abs(int(math.floor(lat))):02d}"
                f"{'+' if int(math.floor(lon)) >= 0 else '-'}"
                f"{abs(int(math.floor(lon))):03d}",
        'lat': int(math.floor(lat)),
        'lon': int(math.floor(lon)),
        'out_dsf': os.path.abspath(out_dsf) if out_dsf else None,
        'exclude_lower_forests': exclude_lower_forests,
        'exclusion_property': FOREST_EXCLUSION_PROPERTY_KEYS[0],
        'exclusion_value': exclusion_value if exclude_lower_forests else None,
        'requested_avoid_simheaven_forests': requested_avoid_simheaven_forests,
        'effective_avoid_simheaven_forests': bool(avoid_simheaven_forests),
        'outcome': 'failed',
        'polygons': 0,
        'compiled': False,
        'installed': False,
        'embedded_tree_cleanup_allowed': False,
    }
    _transient_cache_ctx = None
    sidecar_cache_dir = cache_dir
    if disable_cache:
        import tempfile as _tempfile
        _transient_cache_ctx = _tempfile.TemporaryDirectory(
            prefix="veg_nocache_"
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
                log_prefix='[Veg]',
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
                '_vegtree.npy',
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

    # Pick the textures that actually render, and the part of each one they own.
    # The tile cfg is the source of truth: it is what Ortho4XP turned into the
    # per-mesh-cell texture assignment at build time, so replaying it drops
    # leftovers from earlier builds with a different provider and resolves both
    # kinds of overlap — a zone boundary splitting one footprint between two
    # providers, and a higher-ZL texture covering part of a lower-ZL one. The
    # regions another texture owns generate their forest polygons from that
    # texture instead, so they are blanked out of this one's mask; otherwise
    # the forests there are emitted twice (double density).
    _source_files, _source_mode, _orthophoto_dir = _collect_source_files()
    if _source_mode == 'dds':
        _stale, _n_stale = SEGFORMER.stale_dds_caches(tex_dir, _source_files, _dds_cache_paths)
        if _n_stale:
            _remove_cache_files(_stale)
            print(f"Dropped the cached results of {_n_stale} texture(s) rebuilt since they were cached.")
    files, _covered_fracs_by_file, _selection_report = TEXSEL.select_textures(
        _source_files, tex_dir, lat, lon
    )
    if not files:
        print("No DDS files found.")
        _print_tile_summary()
        return 0
    for _line in TEXSEL.format_selection_report(
        _selection_report, region_label="vegetation masks", include_details=verbose_log
    ):
        print(_line)
    # Grass around airports (O4_Grass_Overlay) rides along: it collects the
    # class maps of the textures its airport zones reach as this loop goes,
    # and its polygons are merged into this DSF below. A grass failure never
    # costs the trees.
    grass_pack_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(out_dsf))))
    grass_hook = None
    if grass_settings is not None:
        try:
            import O4_Grass_Overlay as GRASS
            grass_hook = GRASS.GrassHook(
                tex_dir, lat, lon, cache_dir, custom_scenery_dir, grass_pack_dir,
                dsftool_path, settings=grass_settings,
            )
            if not grass_hook.sources:
                grass_hook = None
                GRASS.remove_tile_assets(grass_pack_dir, lat, lon)
        except Exception as exc:
            print(f"[Grass] disabled for this tile: {exc!r}")
            grass_hook = None
    else:
        try:
            import O4_Grass_Overlay as GRASS
            GRASS.remove_tile_assets(grass_pack_dir, lat, lon)
        except Exception:
            pass
    _used_zls = sorted(set(int(STD_RE.match(_f).group(4)) for _f in files))
    _zl_str = f"ZL{_used_zls[0]}" if len(_used_zls) == 1 else f"mixed ZL {_used_zls}"

    dens_str = f"{density_override:.2f}" if density_override is not None else "auto"
    print(f"Tile: lat={lat} lon={lon}  DDS: {len(files)} ({_zl_str})")
    verbose_print(f"close={close_m}m  open={open_m}m  min_area={min_area_m2}m²  "
          f"simplify={simplify_m}m  density={dens_str}  excl_buffer={excl_buffer_m}m")
    if avoid_gfv2 or use_gfv2_asset_proximity or gfv2_buffer_m:
        verbose_print("Legacy Global Forests settings ignored; the pipeline is simHeaven-only")
    verbose_print(
        "vegetation asset selection: "
        + ("leaf-type maps, simHeaven X-World fallback" if use_leaf_type_maps
           else "simHeaven X-World forests")
        + ("; planted as X-Plane 12 climate forests (lib/g8)" if use_xp_forests
           else "; planted as simHeaven forest aliases")
    )
    verbose_print(
        "simHeaven forest overlap avoid:"
        f" {'on' if avoid_simheaven_forests else 'off'}"
        f" (requested {'on' if requested_avoid_simheaven_forests else 'off'},"
        f" {simheaven_forest_buffer_m}m); closest asset:"
        f" {'on' if use_simheaven_asset_proximity else 'off'}"
    )
    verbose_print(
        "tile forest replacement:"
        f" {'on' if exclude_lower_forests else 'off'}"
        + (f" — {FOREST_EXCLUSION_PROPERTY_KEYS[0]} {exclusion_value}"
           " over the whole tile" if exclude_lower_forests else "")
    )
    verbose_print(
        f"building overlap avoid: overlay cache={'on' if bld_excl_m > 0 else 'off'} ({bld_excl_m}m)"
        f" simHeaven={'on' if avoid_simheaven_buildings else 'off'} ({simheaven_building_buffer_m}m)"
    )

    device = __import__('torch').device('cuda' if __import__('torch').cuda.is_available() else 'cpu')
    model = proc = None

    tile_center_lat = lat + 0.5
    tile_center_lon = lon + 0.5
    asset_region = asset_region_for_latlon(tile_center_lat, tile_center_lon)
    koppen_code = CLIMATE_REGIONS.koppen_code(tile_center_lat, tile_center_lon)
    region = FOREST_ASSETS.climate_region(tile_center_lat, tile_center_lon)
    verbose_print(
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
        'excl_zones': 0.0,
        'leaf_type': 0.0,
        'contours': 0.0,
        'dsf_text': 0.0,
        'dsf_compile': 0.0,
        'bld_wait': 0.0,
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

    osm_roads = _load_osm_roads(osm_roads_path, cache_dir=sidecar_cache_dir)
    verbose_print(f"OSM roads: {len(osm_roads)} ways  ({osm_roads_path})")
    osm_roads = _prepare_roads(osm_roads)

    osm_residential_roads = _prepare_roads(_residential_roads(osm_roads))
    osm_residential_index = BBOX.build_bounds_index(osm_residential_roads)

    # simHeaven full street network
    if dsftool_path is None:
        dsftool_path = SEGFORMER._dsftool
    sh_network = []
    if use_simheaven and dsftool_path and os.path.exists(dsftool_path):
        _t = time.perf_counter()
        if _road_widths_enabled():
            sh_network = _load_simheaven_network_typed(
                custom_scenery_dir, lat, lon, dsftool_path, sidecar_cache_dir
            )
            for way in sh_network:
                way['half_m'] = SIMHEAVEN_ROAD_HALF_WIDTH_M.get(
                    way['type'], SIMHEAVEN_ROAD_HALF_WIDTH_M['network']
                )
        else:
            sh_network = _load_simheaven_network(custom_scenery_dir, lat, lon, dsftool_path, sidecar_cache_dir)
        timings['scenery_parse'] += time.perf_counter() - _t
        verbose_print(f"simHeaven network: {len(sh_network)} road segments"
              + (" (exclusion width by road type)" if _road_widths_enabled() else ""))
    elif use_simheaven:
        print(f"  [simHeaven] DSFTool not found at {dsftool_path} — skipping")

    sh_network = _prepare_roads(sh_network)
    all_road_ways = osm_roads + sh_network
    all_road_index = BBOX.build_bounds_index(all_road_ways)
    all_road_sig = _roads_signature(all_road_ways)
    osm_residential_sig = _roads_signature(osm_residential_roads)

    forest_layers = []
    forest_records = []
    simheaven_type_source_path = None
    simheaven_type_resolver = None
    simheaven_type_sig = None
    # Always read: the 7-forests polygons also feed the forest-asset context.
    if dsftool_path and os.path.exists(dsftool_path):
        forest_matches = find_simheaven_forest_dsfs(
            custom_scenery_dir,
            lat,
            lon,
        )
        if forest_matches:
            _t = time.perf_counter()
            forest_records = _prepare_polygons(
                _load_forest_polygons(
                    "simHeaven X-World",
                    forest_matches,
                    dsftool_path,
                    sidecar_cache_dir,
                )
            )
            timings['scenery_parse'] += time.perf_counter() - _t
            if use_simheaven_asset_proximity:
                simheaven_type_source_path = (
                    _dominant_simheaven_tree_path(forest_records)
                )
                simheaven_type_sig = _polys_signature(forest_records)
                simheaven_type_resolver = _SimHeavenTypeResolver(
                    forest_records,
                    simheaven_type_source_path,
                    lat,
                )
                verbose_print(
                    "simHeaven closest-asset sources:"
                    f" {simheaven_type_resolver.n} tree polygons;"
                    f" tile fallback={simheaven_type_source_path or 'climate'}"
                )
            if avoid_simheaven_forests:
                forest_layers.append(
                    {
                        'name': 'simHeaven X-World',
                        'buffer_m': max(
                            0.0,
                            float(simheaven_forest_buffer_m),
                        ),
                        'polys': forest_records,
                        'index': BBOX.build_bounds_index(forest_records),
                    }
                )
                verbose_print(
                    f"simHeaven forest overlap: {len(forest_records)} polygons"
                )
        else:
            verbose_print("simHeaven forests: no enabled 7-forests DSF for this tile")
    else:
        print(
            "simHeaven forests: skipped "
            f"(DSFTool unavailable at {dsftool_path})"
        )
    forest_layer_sigs = tuple(
        (
            layer['name'],
            round(float(layer['buffer_m']), 4),
            _polys_signature(layer['polys']),
        )
        for layer in forest_layers
    )

    # ── Leaf-type maps: broadleaf / conifer / mixed zones ───────────────────
    # Downloaded once per tile (GLC once per 5x5 degree block) into a shared
    # cache; without any coverage the simHeaven closest-asset choice stays.
    leaf_types = None
    if use_leaf_type_maps:
        _t = time.perf_counter()
        verbose_print("Leaf-type maps:")
        try:
            leaf_types = LEAF_TYPE.TileLeafTypes(
                lat,
                lon,
                leaf_type_cache_dir or LEAF_TYPE.default_cache_dir(cache_dir),
                forest_records=forest_records,
                log=verbose_print,
                warning_log=print,
            )
        except Exception as exc:
            print(f"  [LeafType] failed: {exc}")
            leaf_types = None
        if leaf_types is not None and not leaf_types.available:
            verbose_print("  [LeafType] no map coverage — using simHeaven closest asset")
            leaf_types = None
        timings['leaf_type'] += time.perf_counter() - _t

    # ── X-Plane 12 climate forests ──────────────────────────────────────────
    # The family (broad / coni / mixed) chosen above becomes X-Plane's own
    # forest for the climate band its global scenery uses at the polygon.
    xp_forest_mapper = None
    if use_xp_forests:
        _t = time.perf_counter()
        try:
            xp_forest_mapper = LEAF_TYPE.XPForestMapper(
                lat,
                lon,
                leaf_type_cache_dir or LEAF_TYPE.default_cache_dir(cache_dir),
                custom_scenery_dir,
                dsftool_path,
                log=verbose_print,
                warning_log=print,
            )
        except Exception as exc:
            print(f"  [XPForests] failed, keeping simHeaven aliases: {exc}")
            xp_forest_mapper = None
        timings['leaf_type'] += time.perf_counter() - _t

    # ── Forest-asset context ────────────────────────────────────────────────
    veg_context = _veg_context_from_simheaven(forest_records)
    verbose_print(
        "Forest-asset context (simHeaven 7-forests): "
        f"forest={len(veg_context['forest_polys'])} "
        f"woodland={len(veg_context['woodland_hint_polys'])} "
        f"orchard={len(veg_context['orchard_polys'])} "
        f"wetland={len(veg_context['water_polys'])}"
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
    veg_context_tree_rows = _prepare_roads(veg_context.get('tree_rows', []))
    tree_row_index = BBOX.build_bounds_index(veg_context_tree_rows)
    tree_row_sig = _roads_signature(veg_context_tree_rows)

    # ── Custom-scenery forest exclusion zones ───────────────────────────────
    # An airport's `sim/exclude_for` zone is honoured with no buffer and no
    # opt-out: it is the scenery author stating that nothing may grow forests
    # there, so it outranks every heuristic layer above. Zones that exclude only
    # other types (objects, facades, roads, …) leave vegetation untouched, and
    # regional/global packs are not consulted at all.
    _t = time.perf_counter()
    excl_zone_rings, excl_zone_types = _load_custom_scenery_forest_exclusions(
        custom_scenery_dir, lat, lon, out_dsf, dsftool_path, sidecar_cache_dir
    )
    timings['scenery_parse'] += time.perf_counter() - _t
    excl_zone_polys = _prepare_polygons(excl_zone_rings)
    excl_zone_index = BBOX.build_bounds_index(excl_zone_polys)
    excl_zone_sig = _polys_signature(excl_zone_polys)
    if excl_zone_types:
        type_summary = ", ".join(
            f"{key.rsplit('_', 1)[-1]}={count}"
            for key, count in sorted(excl_zone_types.items())
        )
        verbose_print(
            f"Custom scenery exclusion zones: {len(excl_zone_polys)} forest "
            f"({type_summary})"
        )
    else:
        verbose_print("Custom scenery exclusion zones: none")

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
        verbose_print(f"simHeaven buildings: {n_sh_bld_objs} objects  {n_sh_bld_polys} facade polys")
    elif avoid_simheaven_buildings:
        print(f"simHeaven buildings: skipped (DSFTool unavailable at {dsftool_path})")

    _t = time.perf_counter()
    mesh_water_path = _mesh_file_for_tile(tex_dir, lat, lon)
    mesh_water_sig = _mesh_water_signature(mesh_water_path)
    mesh_water_index = _load_mesh_water_index(mesh_water_path, sidecar_cache_dir)
    timings['mesh_water'] += time.perf_counter() - _t
    if mesh_water_index:
        verbose_print(
            f"Mesh water: {len(mesh_water_index['tris'])} water triangles from "
            f"{mesh_water_path}"
        )
    else:
        print(f"Mesh water: unavailable from {mesh_water_path}")

    # With caches off, read the SegFormer class map the building pass of this
    # build left behind instead of inferring it again; it is deleted once this
    # DDS is done. O4_VEG_REUSE_BLD_SEG=0 always infers.
    reuse_bld_seg = os.environ.get("O4_VEG_REUSE_BLD_SEG", "").strip().lower() not in ("0", "false", "no", "off")
    # Set by the pipeline when the building overlay runs alongside: the path
    # of the flag it writes when it is done. Until then, wait for each
    # texture's class map (or its "no class map" marker) to appear.
    bld_done_flag = os.environ.get("O4_VEG_WAIT_FOR_BLD_SEG", "").strip() if reuse_bld_seg else ""

    def _wait_for_bld_seg(cache_path):
        if not bld_done_flag:
            return
        skip_path = cache_path[:-len('_veg.npy')] + '_bldseg.skip'
        _t = time.perf_counter()
        while not (
            os.path.exists(cache_path) or os.path.exists(skip_path) or os.path.exists(bld_done_flag)
        ):
            time.sleep(0.2)
        timings['bld_wait'] += time.perf_counter() - _t
    n_textures = sum(1 for f in files if STD_RE.match(f))
    n_started = 0
    PROG.emit(ev="items", done=0, total=n_textures)
    for idx, fname in enumerate(files, 1):
        m = STD_RE.match(fname)
        if not m: continue
        # Starting this one means the ones before it are done.
        PROG.emit(ev="items", done=n_started)
        n_started += 1
        _dds_cache_files = _dds_cache_paths(fname)
        if disable_cache:
            # Delete stale per-DDS cache before this DDS starts. Keep the
            # building placement cache only until we've consumed it below.
            _pre_dds_cleanup = tuple(
                path for path in _dds_cache_files
                if not (bld_excl_m > 0 and path.endswith('_bld.pkl'))
                and not (reuse_bld_seg and path.endswith(('_veg.npy', '_vegtree.npy')))
            )
            _remove_cache_files(
                _pre_dds_cleanup
            )
        til_y_top  = int(m.group(1))
        til_x_left = int(m.group(2))
        zl         = int(m.group(4))

        lat_n, lat_s, lon_w, lon_e = dds_bounds(til_y_top, til_x_left, zl)

        cache_path = os.path.join(cache_dir, fname.replace('.dds', '_veg.npy'))
        tree_support_path = SEGFORMER.tree_support_cache_path(cache_path)
        want_tree_support = SEGFORMER.tree_support_enabled()
        tree_support = None
        img = None
        veg_map = None
        mesh_water_mask = None
        mesh_water_full = False
        if reuse_bld_seg:
            _wait_for_bld_seg(cache_path)
        if (not disable_cache or reuse_bld_seg) and os.path.exists(cache_path):
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
                verbose_print(f"  [{idx}/{n_files}] {fname}  (stale all-water cache ignored)", flush=True)
            elif want_tree_support and not mesh_water_full:
                tree_support = SEGFORMER.load_tree_support(tree_support_path, veg_map.shape)
                if tree_support is None:
                    # Class map cached without its tree-support mask: infer again.
                    veg_map = None
                    verbose_print(f"  [{idx}/{n_files}] {fname}  (no tree-support cache; re-inferring)", flush=True)
                else:
                    print(f"  [{idx}/{n_files}] {fname}  (cached)", flush=True)
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
                verbose_print(f"    {fname}: mesh water full; inference skipped", flush=True)
            else:
                if model is None:
                    model, proc, device = SEGFORMER.load_vegetation_model(device)
                _t = time.perf_counter()
                if want_tree_support:
                    veg_map, tree_support = SEGFORMER.run_inference(
                        model, device, img, proc, tree_support=True
                    )
                else:
                    veg_map = SEGFORMER.run_inference(model, device, img, proc)
                timings['inference'] += time.perf_counter() - _t
            if not disable_cache and not mesh_water_full:
                np.save(cache_path, veg_map)
                if tree_support is not None:
                    SEGFORMER.save_tree_support(tree_support_path, tree_support)

        # Grass reuses this class map instead of inferring again: with caches
        # off it is deleted once this DDS is done.
        if grass_hook is not None and grass_hook.wants(fname):
            grass_hook.take_class_map(fname, veg_map)

        if disable_cache:
            veg_cache_stat = None
        else:
            try:
                _veg_st = os.stat(cache_path)
                veg_cache_stat = (_veg_st.st_mtime_ns, _veg_st.st_size)
                if tree_support is not None:
                    _ts_st = os.stat(tree_support_path)
                    veg_cache_stat += (_ts_st.st_mtime_ns, _ts_st.st_size)
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
            if tree_support is not None:
                tree_support = cv2.resize(
                    tree_support.astype(np.uint8), (new_w, new_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
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

        tree_raw  = (veg_map == SEGFORMER.CLASS_TREE)
        if tree_support is not None:
            tree_raw = tree_raw | tree_support
        tree_raw = tree_raw.astype(np.uint8)
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
            excl_zone_sig=excl_zone_sig,
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
            excl_zone_mask = _mask_cached.get('excl_zone_mask')
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
            excl_zone_mask = None
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
                    _bld_objs = _bld_cache_objects(_bld_data)
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

        # ── Exclusion layer 5: custom-scenery forest exclusion zones ────────
        if excl_zone_mask is None and excl_zone_polys:
            _t = time.perf_counter()
            local_excl_zones = _polys_for_bounds(
                excl_zone_index, lat_n, lat_s, lon_w, lon_e, pad_deg=0.0
            )
            if local_excl_zones:
                excl_zone_mask = _rasterize_polygons(
                    local_excl_zones, lat_n, lat_s, lon_w, lon_e, img_h, img_w
                )
            timings['excl_zones'] += time.perf_counter() - _t
        if excl_zone_mask is not None:
            excl_mask = cv2.bitwise_or(excl_mask, excl_zone_mask)

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
                        'excl_zone_mask': excl_zone_mask,
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

        review_cuts = None
        if _review_dump_enabled():
            cut = cv2.bitwise_and(tree_mask, excl_mask)
            cut_road = (
                cv2.bitwise_and(cut, road_excl) if road_excl is not None
                else np.zeros_like(cut)
            )
            review_cuts = {
                'road': cut_road,
                'other': cv2.bitwise_and(cut, cv2.bitwise_not(cut_road)),
            }

        tree_mask = cv2.bitwise_and(tree_mask, cv2.bitwise_not(excl_mask))

        # Regions covered by kept higher-ZL textures emit their forests at the
        # higher ZL; blank them here so trees are not generated twice.
        _covered_fracs = _covered_fracs_by_file.get(fname)
        if _covered_fracs:
            mask_covered_regions(
                tree_mask, covered_rects_to_px(_covered_fracs, img_w, img_h)
            )
        if review_cuts is not None:
            for _cut in review_cuts.values():
                if _covered_fracs:
                    mask_covered_regions(
                        _cut, covered_rects_to_px(_covered_fracs, img_w, img_h)
                    )
            _write_veg_review(
                os.path.join(cache_dir, fname.replace('.dds', '_vegreview.json')),
                img_w, img_h, review_cuts,
            )

        _poly_cache_file = os.path.join(cache_dir, fname.replace('.dds', '_vegpoly.pkl'))
        if leaf_types is not None:
            _asset_mode = 'leaf_type_maps_v1'
        elif simheaven_type_resolver is not None:
            _asset_mode = 'simheaven_closest_v1'
        else:
            _asset_mode = 'simheaven_climate_v1'
        if xp_forest_mapper is not None:
            _asset_mode += '+' + xp_forest_mapper.signature
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
            _asset_mode,
            None if leaf_types is not None else simheaven_type_source_path,
            forest_type_sig=(
                leaf_types.signature if leaf_types is not None
                else simheaven_type_sig
            ),
            covered_fracs=_covered_fracs,
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
                      type_counts=None,
                      forest_path_mapper=xp_forest_mapper)
        if simheaven_type_source_path:
            kwargs['simheaven_type_path'] = simheaven_type_source_path
        if simheaven_type_resolver is not None:
            kwargs['simheaven_type_resolver'] = simheaven_type_resolver
        dds_seed = int.from_bytes(
            hashlib.sha1(f"veg-poly:{fname}".encode("utf-8")).digest()[:8],
            "big",
        )
        dds_rng = np.random.default_rng(dds_seed)
        kwargs['rng'] = dds_rng
        dds_type_counts = {}
        kwargs['type_counts'] = dds_type_counts

        if leaf_types is not None:
            _t = time.perf_counter()
            zone = leaf_types.zone_for_texture(lat_n, lat_s, lon_w, lon_e, img_h, img_w)
            timings['leaf_type'] += time.perf_counter() - _t
            _t = time.perf_counter()
            p = []
            for code, for_path in LEAF_TYPE.CODE_PATHS:
                part = np.where(zone == code, tree_mask, 0).astype(np.uint8)
                if part.any():
                    p.extend(_process_dds_mask(
                        part, SEGFORMER.CLASS_TREE,
                        **dict(kwargs, simheaven_type_path=for_path,
                               simheaven_type_resolver=None),
                    ))
            timings['contours'] += time.perf_counter() - _t
        else:
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

    PROG.emit(ev="items", done=n_textures)
    verbose_print(f"Loaded {len(files)} DDS  ({time.time()-t_inf:.1f}s)  "
          f"climate={region}  "
          f"bld_excl: {n_bld_excl_used} used / {n_bld_excl_missing} missing  "
          f"simh_bld: {n_sh_bld_objs} obj / {n_sh_bld_polys} poly")
    verbose_print(f"Polygons: tree={n_tree}  total={len(polygons)}")
    if tree_type_counts:
        ordered = sorted(tree_type_counts.items(), key=lambda item: (-item[1], item[0]))
        summary = ", ".join(f"{name}={count}" for name, count in ordered)
        verbose_print(f"Tree types: {summary}")

    # Grass polygons for this same DSF, their POLYGON_DEFs numbered after the
    # tree ones.
    grass_result = None
    if grass_hook is not None:
        try:
            grass_result = grass_hook.build(len(set(p for p, _, _ in polygons)))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[Grass] failed, writing trees only: {exc!r}")
            try:
                GRASS.remove_tile_assets(grass_pack_dir, lat, lon)
            except Exception:
                pass
        finally:
            grass_hook.cleanup()
    if grass_result:
        status['grass'] = grass_result[2]
    n_grass = int(grass_result[2]['polygons']) if grass_result else 0
    if not polygons and grass_result:
        # Grass but no trees of our own: the DSF carries the grass, but never
        # the tile-wide exclusion -- nothing of ours would replace the lower
        # forests it removes.
        exclude_lower_forests = False

    if not polygons and not grass_result:
        # A processed tile with nothing to plant must not keep a tile-wide
        # exclusion alive: with no generated trees of our own, excluding the
        # lower layers would leave the tile bare. Drop any managed overlay we
        # wrote earlier and tell the caller to undo the rest of the workflow.
        print("No vegetation polygons — skipping DSF write.")
        status['outcome'] = 'empty'
        status['needs_reconcile'] = True
        if out_dsf and os.path.exists(out_dsf):
            try:
                os.remove(out_dsf)
                print(f"Removed stale generated overlay {out_dsf}")
                status['removed_stale_overlay'] = True
            except OSError as exc:
                print(f"Could not remove stale generated overlay {out_dsf}: {exc}")
                status['outcome'] = 'failed'
                status['error'] = f"stale overlay not removable: {exc}"
        write_generation_status(status_path, status)
        _print_tile_summary(n_tree, n_grass)
        return 0

    # ── Write DSF ────────────────────────────────────────────────────────────
    # Compile to a temporary file beside the target and swap it in only once it
    # has been validated, so a DSFTool failure leaves the tile's previous
    # overlay exactly as it was instead of destroying it.
    os.makedirs(os.path.dirname(os.path.abspath(out_dsf)), exist_ok=True)
    txt_path = out_dsf.replace('.dsf', '_veg.txt')
    tmp_dsf = out_dsf + '.tmp'

    for_paths = sorted(set(p for p, _, _ in polygons))
    for_idx   = {p: i for i, p in enumerate(for_paths)}
    lat_i = int(math.floor(lat)); lon_i = int(math.floor(lon))

    _t = time.perf_counter()
    with open(txt_path, 'w') as f:
        f.write("PROPERTY sim/planet earth\n")
        f.write("PROPERTY sim/overlay 1\n")
        f.write(f"PROPERTY sim/west  {lon_i}\n")
        f.write(f"PROPERTY sim/east  {lon_i + 1}\n")
        f.write(f"PROPERTY sim/south {lat_i}\n")
        f.write(f"PROPERTY sim/north {lat_i + 1}\n")
        if exclude_lower_forests:
            # One rectangle over the whole degree square. X-Plane applies an
            # exclusion to the DSFs below this overlay in scenery_packs.ini and
            # never to the DSF carrying it, so our own trees survive it.
            f.write(
                f"PROPERTY {FOREST_EXCLUSION_PROPERTY_KEYS[0]} {exclusion_value}\n"
            )
        f.write("\n")
        for p in for_paths:
            f.write(f"POLYGON_DEF {p}\n")
        if grass_result:
            for p in grass_result[0]:
                f.write(f"POLYGON_DEF {p}\n")
        f.write("\n")
        for fpath, dsf_den, ring in polygons:
            f.write(f"BEGIN_POLYGON {for_idx[fpath]} {dsf_den} 2\n")
            if dsf_den >= POINT_MODE:
                _write_point_windings(f, ring)
            else:
                _write_winding(f, ring)
            f.write("END_POLYGON\n")
        if grass_result:
            f.write(grass_result[1])
    timings['dsf_text'] += time.perf_counter() - _t

    verbose_print(f"DSF text → {txt_path}  ({len(for_paths)} unique .for defs"
          + (f" + {len(grass_result[0])} grass" if grass_result else "") + ")")

    _t = time.perf_counter()
    try:
        if os.path.exists(tmp_dsf):
            os.remove(tmp_dsf)
    except OSError:
        pass
    ok = SEGFORMER.compile_dsf(txt_path, tmp_dsf)
    if ok:
        problem = validate_compiled_overlay(
            tmp_dsf, exclusion_value if exclude_lower_forests else None
        )
        if problem:
            ok = False
            print(f"Compiled DSF rejected: {problem}")
    timings['dsf_compile'] += time.perf_counter() - _t
    if ok:
        os.replace(tmp_dsf, out_dsf)
        print(f"DSF compiled → {out_dsf}")
        if exclude_lower_forests:
            verbose_print(f"Tile forest replacement installed: "
                  f"{FOREST_EXCLUSION_PROPERTY_KEYS[0]} {exclusion_value}")
        status['outcome'] = 'ok'
        status['compiled'] = True
        status['installed'] = True
        status['embedded_tree_cleanup_allowed'] = exclude_lower_forests
        status['needs_reconcile'] = not exclude_lower_forests
        try: os.remove(txt_path)
        except: pass
    else:
        print(f"DSFTool failed — text file kept at {txt_path}")
        print("Previous generated overlay left untouched.")
        try:
            if os.path.exists(tmp_dsf):
                os.remove(tmp_dsf)
        except OSError:
            pass
        status['outcome'] = 'compile_failed'
        status['error'] = 'DSFTool could not produce a valid overlay'
        status['text_path'] = txt_path
    status['polygons'] = len(polygons)
    write_generation_status(status_path, status)

    verbose_print(
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
        f"excl_zones={timings['excl_zones']:.1f}s  "
        f"leaf_type={timings['leaf_type']:.1f}s  "
        f"contours={timings['contours']:.1f}s  "
        f"dsf_text={timings['dsf_text']:.1f}s  "
        f"dsf_compile={timings['dsf_compile']:.1f}s  "
        f"bld_wait={timings['bld_wait']:.1f}s"
    )

    _print_tile_summary(n_tree, n_grass)
    return len(polygons)


# ── Entry point ───────────────────────────────────────────────────────────────
def parse_args():
    """Parse CLI arguments for vegetation overlay generation."""
    ap = argparse.ArgumentParser(description='Vegetation overlay generator')
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
    ap.add_argument('--no-avoid-simheaven-buildings', action='store_true',
                    dest='no_avoid_simheaven_buildings',
                    help='Do not exclude simHeaven building footprints/objects from generated vegetation.')
    ap.add_argument('--simheaven-building-buffer-m', type=float, default=10.0,
                    dest='simheaven_building_buffer_m',
                    help='Extra exclusion buffer in metres around simHeaven building footprints/objects.')
    ap.add_argument('--no-avoid-simheaven-forests', action='store_true',
                    dest='no_avoid_simheaven_forests',
                    help='Do not exclude enabled X-World 7-forests polygons.')
    ap.add_argument('--simheaven-forest-buffer-m', type=float, default=0.0,
                    dest='simheaven_forest_buffer_m',
                    help='Extra exclusion buffer in metres around X-World forests.')
    ap.add_argument('--no-simheaven-asset-proximity', action='store_true',
                    dest='no_simheaven_asset_proximity',
                    help='Use climate assets instead of the nearest X-World tree family.')
    ap.add_argument('--exclude-lower-forests', action='store_true',
                    dest='exclude_lower_forests',
                    help='Replace lower-priority forests across the WHOLE one-degree '
                         'tile by writing sim/exclude_for into the generated DSF, '
                         'including where no trees are generated. Forces forest-overlap '
                         'avoidance off for this run.')
    ap.add_argument('--status-path', default=None, dest='status_path',
                    help='Write a JSON summary of how this run ended to this path.')
    ap.add_argument('--no-avoid-gfv2', action='store_true', dest='no_avoid_gfv2',
                    help='Legacy no-op retained for command-line compatibility.')
    ap.add_argument('--gfv2-buffer-m', type=float, default=0.0, dest='gfv2_buffer_m',
                    help='Legacy no-op retained for command-line compatibility.')
    ap.add_argument('--gfv2-asset-proximity', action='store_true',
                    dest='gfv2_asset_proximity',
                    help='Legacy no-op retained for command-line compatibility.')
    ap.add_argument('--no-leaf-type-maps', action='store_true', dest='no_leaf_type_maps',
                    help='Choose broadleaf/conifer/mixed from the nearest simHeaven forest '
                         'instead of the HRL / GLC_FCS30D / CGLS leaf-type maps.')
    ap.add_argument('--leaf-type-cache-dir', default=None, dest='leaf_type_cache_dir',
                    help='Shared leaf-type map cache (default: Overlay_cache/_leaf_type).')
    ap.add_argument('--no-xp-forests', action='store_true', dest='no_xp_forests',
                    help='Emit the simHeaven forest aliases instead of X-Plane 12 '
                         'climate forests (lib/g8).')
    ap.add_argument('--verbose-log', action='store_true',
                    help='Show setup, scenery, cache and timing diagnostics')
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
            PCACHE.cache_root(o4xp_root), f'{lat_gs}{lon_gs}', f'{lat_s}{lon_s}'
        )
    os.makedirs(cache_dir, exist_ok=True)

    out_dsf = args.out_dsf
    if out_dsf is None:
        out_dsf = os.path.join(o4xp_root, 'yOrtho4XP_Veg_Overlays',
                               'Earth nav data',
                               f'{lat_gs}{lon_gs}',
                               f'{lat_s}{lon_s}.dsf')
    print(f"Output DSF: {out_dsf}")

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
        verbose_log      = args.verbose_log,
        density_override = args.density,
        res_m            = args.res_m,
        excl_buffer_m    = args.excl_buffer_m,
        osm_roads_path   = args.osm_roads,
        use_simheaven    = not args.no_simheaven,
        custom_scenery_dir = args.custom_scenery_dir,
        avoid_simheaven_buildings = not args.no_avoid_simheaven_buildings,
        simheaven_building_buffer_m = args.simheaven_building_buffer_m,
        avoid_simheaven_forests = not args.no_avoid_simheaven_forests,
        simheaven_forest_buffer_m = args.simheaven_forest_buffer_m,
        use_simheaven_asset_proximity = not args.no_simheaven_asset_proximity,
        exclude_lower_forests = args.exclude_lower_forests,
        status_path      = args.status_path,
        avoid_gfv2       = False,
        gfv2_buffer_m    = 0.0,
        use_gfv2_asset_proximity = False,
        use_leaf_type_maps = not args.no_leaf_type_maps,
        leaf_type_cache_dir = args.leaf_type_cache_dir,
        use_xp_forests = not args.no_xp_forests,
    )
    return n


if __name__ == '__main__':
    main()
