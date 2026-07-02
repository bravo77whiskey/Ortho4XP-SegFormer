"""
O4_SFR_Building_Overlay.py — SegFormer-assisted building overlay generation.

Usage:
    python src/scripts/generate_bld_overlay.py <tex_dir> <lat> <lon> <out_dsf> [options]

Options:
    --spacing   METRES   Object spacing in metres (default 0)
    --close     PIXELS   Morphological close kernel radius (default 15)
    --open      PIXELS   Morphological open  kernel radius (default 5)
    --min-zone-m2 M2     Min zone area to bother filling   (default 200)
    --no-viz             Skip overview image generation
    --cache-dir DIR      Where to store per-DDS inference caches
                          (default: <o4xp_root>/SFR_cache/<tile>)
    --custom-scenery-dir DIR
                         X-Plane Custom Scenery root used to discover simHeaven
                         network and building DSFs.

Inference is cached per-DDS under cache-dir — changing spacing/kernel params
does NOT re-run inference, only the fast fill + write steps.

Example:
    python src/scripts/generate_bld_overlay.py ^
        "H:/XP12/Tiles/zOrtho4XP_-02+037/textures" -2 37 ^
        "H:/XP12/Custom Scenery/yOrtho4XP_Bld_Overlays/-02+037.dsf" ^
        --spacing 15 --close 15 --open 5
"""
import sys, os, json, argparse, warnings, time, math, re, urllib.request, urllib.parse, hashlib, fnmatch
from collections import Counter, defaultdict
warnings.filterwarnings('ignore')

import numpy as np
import cv2
import torch
from math import pi, atan, exp, log, tan
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None

import O4_SFR_Bounds_Index as BBOX
import O4_SFR_Persistent_Cache as PCACHE
import O4_SFR_Inference as SEGFORMER
import O4_SFR_Stock_Yolo_Objects as STOCKYOLO
import O4_SFR_Asset_Inventory as ASSETINV
from O4_SFR_DSF_Utils import (
    active_scenery_pack_dirs,
    ensure_cached_dsf_text,
    find_active_custom_scenery_dsfs,
    find_simheaven_building_dsfs,
    find_simheaven_network_dsfs,
    find_simheaven_package_region_for_tile,
    resolve_custom_scenery_dir,
)
from O4_SFR_Region_Boundaries import asset_region_for_latlon


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _stock_yolo_batch_default():
    try:
        return max(1, int(getattr(STOCKYOLO, "DEFAULT_STOCK_YOLO_BATCH", 1)))
    except (TypeError, ValueError):
        return 1


def _stock_yolo_batch_metadata(stock_res, requested_batch_size):
    """Return stock-YOLO batch metadata, tolerating legacy result objects."""
    requested = max(1, int(requested_batch_size or 1))
    try:
        effective = max(1, int(getattr(stock_res, "effective_batch_size")))
    except (TypeError, ValueError, AttributeError):
        effective = requested
    batch_fell_back = bool(getattr(stock_res, "batch_fell_back", effective != requested))
    return effective, batch_fell_back


def _run_stock_yolo_pass_compat(image, **kwargs):
    """Call stock-YOLO while tolerating older helper modules in deployed copies."""
    try:
        return STOCKYOLO.run_stock_yolo_pass(image, **kwargs)
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        legacy_kwargs = dict(kwargs)
        removed = False
        for key in ("asset_map", "static_classes"):
            if key in legacy_kwargs:
                legacy_kwargs.pop(key)
                removed = True
        if not removed:
            raise
        return STOCKYOLO.run_stock_yolo_pass(image, **legacy_kwargs)


def _cuda_memory_counts_mb():
    if not torch.cuda.is_available():
        return None
    try:
        return (
            int(round(torch.cuda.memory_allocated() / (1024 * 1024))),
            int(round(torch.cuda.memory_reserved() / (1024 * 1024))),
        )
    except Exception:
        return None


def _record_elapsed(timings, file_timings, key, start):
    elapsed = time.perf_counter() - start
    timings[key] += elapsed
    if file_timings is not None:
        file_timings[key] = file_timings.get(key, 0.0) + elapsed
    return elapsed


def _print_dds_timing(fname, file_timings, total_elapsed, file_counts=None):
    ordered = (
        ("load", "dds_load"),
        ("cache", "cache_load"),
        ("segformer", "segformer_inference"),
        ("trained_yolo", "trained_yolo_inference"),
        ("yolo_suppress", "yolo_suppress"),
        ("stock_yolo", "stock_yolo_inference"),
        ("zone", "zone_cleanup"),
        ("lookup", "lookup"),
        ("road", "road_raster"),
        ("road_cache", "road_cache"),
        ("meshwater", "mesh_water"),
        ("heading", "heading_grid"),
        ("excl", "existing_bld_excl"),
        ("mask", "mask_apply"),
        ("cc", "connected_components"),
        ("candidates", "candidate_grid"),
        ("place", "fit_loop"),
        # O4_SFR_BLD_PLACE_PROFILE sub-phase breakdown (silent when 0).
        ("place_prep", "place_prep"),
        ("place_fits", "place_fits"),
        ("place_select", "place_select"),
        ("place_blockers", "place_blockers"),
        ("place_maskfill", "place_maskfill"),
        ("save", "cache_save"),
        ("viz", "viz"),
    )
    parts = [f"total={total_elapsed:.2f}s"]
    parts.extend(
        f"{label}={file_timings.get(key, 0.0):.2f}s"
        for label, key in ordered
        if file_timings.get(key, 0.0) >= 0.005
    )
    if file_counts:
        parts.extend(
            f"{label}={int(file_counts.get(key, 0))}"
            for label, key in (
                ("cand", "candidates"),
                ("yolo_raw", "yolo_raw_detections"),
                ("yolo_det", "yolo_detections"),
                ("yolo_supp", "yolo_suppressed_overlap"),
                ("yolo_placed", "yolo_placed"),
                ("yolo_obj", "yolo_object_placed"),
                ("yolo_fac", "yolo_facade_placed"),
                ("yolo_obj_no_cand", "yolo_object_no_candidates"),
                ("yolo_obj_nodim", "yolo_object_no_dimensions"),
                ("yolo_obj_size", "yolo_object_size_reject"),
                ("yolo_obj_cov", "yolo_object_coverage_reject"),
                ("yolo_obj_outline", "yolo_object_outline_reject"),
                ("yolo_obj_context", "yolo_object_context_skipped"),
                ("yolo_obj_occ", "yolo_object_occupancy_reject"),
                ("yolo_blocked", "yolo_blocked"),
                ("yolo_overlap_blocked", "yolo_overlap_blocked"),
                ("yolo_tpl", "yolo_template_placed"),
                ("zone_yolo_tpl", "same_zone_yolo_template_placed"),
                ("gap_yolo_tpl", "gap_yolo_template_placed"),
                ("initial_blocked", "initial_center_blocked"),
                ("static_clearance_blocked", "static_clearance_blocked"),
                ("dynamic_blocked", "dynamic_center_blocked"),
                ("dynamic_clearance_blocked", "dynamic_clearance_blocked"),
                ("cap", "candidate_cap"),
                ("gap_cand", "gap_candidates"),
                ("gap_static_blocked", "gap_static_blocked"),
                ("gap_dynamic_blocked", "gap_dynamic_blocked"),
                ("gap_comp", "gap_components"),
                ("gap_cap", "gap_candidate_cap"),
                ("gap_placed", "gap_placed"),
                ("res_asset_skip", "residential_asset_skipped"),
                ("side_head", "side_heading_zones"),
                ("sh_head", "simheaven_heading_zones"),
                ("fit_checks", "fit_checks"),
                # O4_SFR_BLD_PLACE_PROFILE candidate-scan accounting.
                ("place_select_calls", "place_select_calls"),
                ("place_cands_offered", "place_cands_offered"),
                ("place_cands_viable", "place_cands_viable"),
                ("place_cands_scanned", "place_cands_scanned"),
                ("placed", "placed"),
                ("cuda_alloc_before_mb", "yolo_cuda_alloc_before_mb"),
                ("cuda_alloc_after_mb", "yolo_cuda_alloc_after_mb"),
                ("cuda_reserved_before_mb", "yolo_cuda_reserved_before_mb"),
                ("cuda_reserved_after_mb", "yolo_cuda_reserved_after_mb"),
            )
            if file_counts.get(key, 0)
        )
    print(f"    [Bld DDS timing] {fname}  " + "  ".join(parts), flush=True)


def _env_patterns(name):
    value = os.environ.get(name, "")
    return [part.strip() for part in value.split(",") if part.strip()]


def _limit_candidates_by_component(cand_x, cand_y, cand_cls, cand_labels, max_candidates, rng):
    """Keep a deterministic, component-balanced sample of placement candidates."""
    n_candidates = int(cand_x.size)
    if max_candidates <= 0 or n_candidates <= max_candidates:
        return cand_x, cand_y, cand_cls, cand_labels, 0

    labels, inverse, counts = np.unique(cand_labels, return_inverse=True, return_counts=True)
    n_labels = labels.size
    if n_labels == 0:
        return cand_x[:0], cand_y[:0], cand_cls[:0], cand_labels[:0], n_candidates

    reserve = np.minimum(counts, 64)
    if int(reserve.sum()) > max_candidates:
        reserve = np.zeros_like(counts)

    remaining = max_candidates - int(reserve.sum())
    alloc = reserve.astype(np.int64, copy=True)
    needs = counts - alloc
    if remaining > 0 and np.any(needs > 0):
        weights = np.sqrt(needs.astype(np.float64))
        raw = remaining * (weights / weights.sum())
        extra = np.minimum(needs, np.floor(raw).astype(np.int64))
        alloc += extra
        leftover = max_candidates - int(alloc.sum())
        if leftover > 0:
            fractional_order = np.argsort(raw - np.floor(raw))[::-1]
            for idx in fractional_order:
                if leftover <= 0:
                    break
                if alloc[idx] < counts[idx]:
                    alloc[idx] += 1
                    leftover -= 1

    selected = []
    for label_idx in range(n_labels):
        label_candidates = np.flatnonzero(inverse == label_idx)
        keep_count = int(min(alloc[label_idx], label_candidates.size))
        if keep_count <= 0:
            continue
        if keep_count == label_candidates.size:
            selected.append(label_candidates)
        else:
            selected.append(
                label_candidates[
                    np.linspace(0, label_candidates.size - 1, keep_count, dtype=np.int64)
                ]
            )

    if not selected:
        return cand_x[:0], cand_y[:0], cand_cls[:0], cand_labels[:0], n_candidates

    keep_idx = np.sort(np.concatenate(selected))
    return cand_x[keep_idx], cand_y[keep_idx], cand_cls[keep_idx], cand_labels[keep_idx], n_candidates - keep_idx.size


def _leftover_gap_candidates(leftover_mask, zone_class, max_candidates, rng,
                             max_per_component=24):
    """Return high-value gap-fill centers from leftover connected components."""
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        leftover_mask.astype(np.uint8), connectivity=8
    )
    if n_labels <= 1:
        empty_i = np.empty(0, dtype=np.int32)
        empty_c = np.empty(0, dtype=np.uint8)
        return empty_i, empty_i, empty_c, empty_i, 0, 0

    rows = []
    min_area_px = 4
    peak_kernel = np.ones((3, 3), dtype=np.uint8)
    for label in range(1, n_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area_px or w <= 0 or h <= 0:
            continue

        local_labels = labels[y:y + h, x:x + w]
        component = (local_labels == label).astype(np.uint8)
        dist = cv2.distanceTransform(component, cv2.DIST_L2, 3)
        dilated = cv2.dilate(dist, peak_kernel)
        peak_mask = (component != 0) & (dist >= dilated - 1e-6) & (dist >= 1.0)
        py, px = np.nonzero(peak_mask)
        if px.size == 0:
            py = np.array([int(round(float(centroids[label, 1]) - y))], dtype=np.int32)
            px = np.array([int(round(float(centroids[label, 0]) - x))], dtype=np.int32)
            px = np.clip(px, 0, w - 1)
            py = np.clip(py, 0, h - 1)

        scores = dist[py, px]
        order = np.argsort(scores)[::-1]
        component_budget = max(
            int(max_per_component),
            min(512, int(math.sqrt(float(area)) * 0.5)),
        )
        keep = order[:max(1, min(component_budget, order.size))]
        gx = (px[keep] + x).astype(np.int32)
        gy = (py[keep] + y).astype(np.int32)
        gcls = zone_class[gy, gx].astype(np.uint8)
        valid = gcls != 0
        gx = gx[valid]
        gy = gy[valid]
        gcls = gcls[valid]
        if gx.size:
            glabels = np.full(gx.shape, label, dtype=np.int32)
            rows.append((gx, gy, gcls, glabels))

    if not rows:
        empty_i = np.empty(0, dtype=np.int32)
        empty_c = np.empty(0, dtype=np.uint8)
        return empty_i, empty_i, empty_c, empty_i, max(0, n_labels - 1), 0

    gap_x = np.concatenate([row[0] for row in rows])
    gap_y = np.concatenate([row[1] for row in rows])
    gap_cls = np.concatenate([row[2] for row in rows])
    gap_labels = np.concatenate([row[3] for row in rows])
    if max_candidates and gap_x.size > max_candidates:
        gap_x, gap_y, gap_cls, gap_labels, n_dropped = _limit_candidates_by_component(
            gap_x, gap_y, gap_cls, gap_labels, max_candidates, rng
        )
    else:
        n_dropped = 0

    order = rng.permutation(gap_x.size) if gap_x.size else np.empty(0, dtype=np.int64)
    return (
        gap_x[order],
        gap_y[order],
        gap_cls[order],
        gap_labels[order] if gap_labels.size == gap_x.size else gap_labels,
        max(0, n_labels - 1),
        int(n_dropped),
    )


# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    """Parse CLI arguments for building overlay generation."""
    ap = argparse.ArgumentParser(description='SegFormer building overlay generator')
    ap.add_argument('tex_dir')
    ap.add_argument('lat',     type=float)
    ap.add_argument('lon',     type=float)
    ap.add_argument('out_dsf', nargs='?', default=None,
                    help='Output DSF path (auto-derived under yOrtho4XP_Bld_Overlays if omitted)')
    ap.add_argument(
        '--spacing',
        type=float,
        default=0.0,
        help='Target edge gap in metres between generated building footprints',
    )
    ap.add_argument('--close',     type=int,   default=15)
    ap.add_argument('--open-k',    type=int,   default=5,  dest='open_k')
    ap.add_argument('--min-zone', '--min-zone-m2', type=float, default=200.0, dest='min_zone_m2')
    ap.add_argument('--no-viz',    action='store_true')
    ap.add_argument('--debug-image-only', action='store_true', dest='debug_image_only',
                    help='Generate overview/footprint PNGs then exit — no DSF written.')
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--grid-n',    type=int,   default=HEADING_GRID_N, dest='grid_n')
    ap.add_argument('--osm-roads', default=None,
                    help='Path to *_big_roads.osm.bz2 (auto-discovered if omitted)')
    ap.add_argument('--custom-scenery-dir', default=None,
                    help='Configured X-Plane root or Custom Scenery directory used to locate building libraries.')
    ap.add_argument('--no-custom-scenery-avoidance', action='store_true',
                    help='Disable active custom scenery object/facade overlap avoidance.')
    ap.add_argument('--allow-road-overlap', action='store_true',
                    help='Allow generated building footprints to overlap road masks.')
    ap.add_argument('--no-yolo', action='store_true',
                    help='Disable YOLO OBB direct building placements.')
    ap.add_argument('--yolo-checkpoint', default=None,
                    help='YOLO OBB checkpoint used for direct building placements.')
    ap.add_argument('--yolo-conf', type=float, default=None,
                    help='YOLO OBB confidence threshold.')
    ap.add_argument('--yolo-iou', type=float, default=None,
                    help='YOLO OBB NMS IoU threshold.')
    ap.add_argument('--yolo-stride', type=int, default=None,
                    help='YOLO OBB crop stride in pixels.')
    ap.add_argument('--yolo-max-det', type=int, default=None,
                    help='YOLO OBB max detections per crop.')
    ap.add_argument('--yolo-suppress-coverage', type=float, default=0.0,
                    help='Drop lower-confidence YOLO OBBs whose overlap coverage exceeds this threshold. 0 disables.')
    ap.add_argument('--yolo-suppress-min-overlap-m2', type=float, default=0.0,
                    help='Minimum absolute YOLO OBB overlap area in m² before drop suppression applies. 0 = remove overlaps with no minimum-area limit.')
    ap.add_argument('--yolo-keep-mode', choices=('drop', 'marginal'), default='drop',
                    help='Overlap removal strategy: "drop" (legacy greedy NMS) or "marginal" (set-cover keep that preserves coverage).')
    ap.add_argument('--yolo-keep-min-new-frac', type=float, default=0.25,
                    help='marginal keep_mode: keep a detection only if it adds at least this fraction of new ground area.')
    ap.add_argument('--yolo-freearea-downsize', action='store_true',
                    help='Allow smaller assets into partially-occupied detections by gating coverage against the free area.')
    ap.add_argument('--yolo-facade-clip', action='store_true',
                    help='Clip fallback facades to the occupancy-free region instead of stamping the full detection polygon.')
    return ap.parse_args()


# ── Coordinate helpers ────────────────────────────────────────────────────────
def _gtile_to_wgs84(til_x, til_y, zl):
    """Top-left (lat_n, lon_w) of Google-tile (til_x, til_y) at zoom zl."""
    rat_x = til_x / (2 ** (zl - 1)) - 1
    rat_y = 1 - til_y / (2 ** (zl - 1))
    lon = rat_x * 180
    lat = 360 / pi * atan(exp(pi * rat_y)) - 90
    return lat, lon


def dds_bounds(til_y_top, til_x_left, zl=16):
    """Return (lat_n, lat_s, lon_w, lon_e) for a DDS tile."""
    lat_n, lon_w = _gtile_to_wgs84(til_x_left,      til_y_top,      zl)
    lat_s, lon_e = _gtile_to_wgs84(til_x_left + 16, til_y_top + 16, zl)
    return lat_n, lat_s, lon_w, lon_e


def px_to_latlon(px, py, img_w, img_h, lat_n, lat_s, lon_w, lon_e):
    """Convert pixel (px, py) in a DDS to (lon, lat)."""
    lon = lon_w + px / img_w * (lon_e - lon_w)
    lat = lat_n - py / img_h * (lat_n - lat_s)
    return lon, lat


# ── OSM road helpers ─────────────────────────────────────────────────────────
def _load_osm_roads(osm_bz2_path, cache_dir=None):
    """Parse an OSM highway extract, such as Ortho4XP *_roads.osm.bz2.

    Returns a list of road records, each a dict with:
      'pts'  — list of (lat, lon) tuples
      'type' — highway tag value  (e.g. 'primary')
    """
    import bz2
    import xml.etree.ElementTree as ET

    if not os.path.exists(osm_bz2_path):
        return []

    def _parse():
        with bz2.open(osm_bz2_path, 'rb') as f:
            tree = ET.parse(f)
        root = tree.getroot()

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
            pts = [nodes[nd.get('ref')] for nd in way.iter('nd')
                   if nd.get('ref') in nodes]
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


def _osm_tile_peer_path(osm_roads_path, suffix):
    """Return a sibling OSM cache path for the same tile and a new suffix."""
    if not osm_roads_path:
        return None
    folder = os.path.dirname(os.path.abspath(osm_roads_path))
    base = os.path.basename(osm_roads_path)
    for old_suffix in (
        "_big_roads.osm.bz2",
        "_small_roads.osm.bz2",
        "_all_roads.osm.bz2",
        "_excl_bld_rail_res.osm.bz2",
    ):
        if base.endswith(old_suffix):
            return os.path.join(folder, base[:-len(old_suffix)] + suffix)
    return None


def _transient_cache_peer_path(osm_roads_path, suffix, transient_cache_dir=None):
    """Return an OSM peer path, using transient storage for new no-cache peers."""
    peer_path = _osm_tile_peer_path(osm_roads_path, suffix)
    if not peer_path:
        return None
    if not transient_cache_dir or os.path.exists(peer_path):
        return peer_path
    return os.path.join(transient_cache_dir, os.path.basename(peer_path))


def _rasterize_roads(roads, lat_n, lat_s, lon_w, lon_e, img_h, img_w,
                     road_width_px=12):
    """Rasterize road polylines onto a uint8 mask in DDS-tile pixel space."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    def ll_to_px(lat, lon):
        x = int((lon - lon_w) / (lon_e - lon_w) * img_w)
        y = int((lat_n - lat) / (lat_n - lat_s) * img_h)
        return x, y

    for road in roads:
        pts_px = [ll_to_px(lat, lon) for lat, lon in road['pts']]
        for i in range(len(pts_px) - 1):
            cv2.line(mask, pts_px[i], pts_px[i + 1], 1, thickness=road_width_px)
    return mask


def _road_bounds(road):
    pts = road.get('pts', ())
    if not pts:
        return (0.0, 0.0, 0.0, 0.0)
    lats = [p[0] for p in pts]
    lons = [p[1] for p in pts]
    return min(lats), max(lats), min(lons), max(lons)


def _roads_for_bounds(roads, lat_n, lat_s, lon_w, lon_e, pad_deg=0.0):
    """Return ways whose lat/lon bbox intersects the DDS bounds."""
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


def _prepare_roads(roads):
    """Attach bbox metadata once so per-DDS queries do not rescan vertices."""
    prepared = []
    for road in roads or []:
        pts = road.get('pts', ())
        if len(pts) < 2:
            continue
        item = dict(road)
        item['_bounds'] = _road_bounds(item)
        prepared.append(item)
    return prepared


def _prepare_segment_arrays(roads):
    """Flatten road ways into arrays used for fast per-DDS heading lookup."""
    rows = []
    for road in roads or []:
        pts = road.get('pts', ())
        for i in range(len(pts) - 1):
            lat1, lon1 = pts[i]
            lat2, lon2 = pts[i + 1]
            mid_lat = (lat1 + lat2) / 2
            mid_lon = (lon1 + lon2) / 2
            dlat = lat2 - lat1
            dlon = (lon2 - lon1) * math.cos(math.radians(mid_lat))
            raw_deg = math.degrees(math.atan2(dlat, dlon))
            heading = (90.0 - (raw_deg % 180.0)) % 360.0
            seg_len = math.hypot(dlat, dlon * 111320 / 110540)
            rows.append((mid_lat, mid_lon, raw_deg, heading, seg_len))
    if not rows:
        return None
    arr = np.asarray(rows, dtype=np.float32)
    return {
        'lat': arr[:, 0],
        'lon': arr[:, 1],
        'raw_deg': arr[:, 2],
        'heading': arr[:, 3],
        'len': arr[:, 4],
    }


def _segments_for_bounds(seg_index, lat_n, lat_s, lon_w, lon_e, pad_deg=0.0):
    """Return a lightweight view of flattened segments inside DDS bounds."""
    if not seg_index:
        return None
    keep = (
        (seg_index['lat'] >= lat_s - pad_deg) &
        (seg_index['lat'] <= lat_n + pad_deg) &
        (seg_index['lon'] >= lon_w - pad_deg) &
        (seg_index['lon'] <= lon_e + pad_deg)
    )
    if not bool(np.any(keep)):
        return None
    return {k: v[keep] for k, v in seg_index.items()}


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


def _polys_signature(polys):
    """Small deterministic signature for polygon exclusion cache invalidation."""
    n_polys = 0
    n_pts = 0
    checksum = 0.0
    for poly in polys or []:
        if isinstance(poly, dict):
            pts = poly.get('pts', ())
            bounds = poly.get('_bounds')
        else:
            pts = poly
            bounds = None
        if not pts:
            continue
        n_polys += 1
        n_pts += len(pts)
        if bounds is None:
            lats = [pt[0] for pt in pts]
            lons = [pt[1] for pt in pts]
            bounds = (min(lats), max(lats), min(lons), max(lons))
        south, north, west, east = bounds
        checksum += (
            south * 3.0 + north * 5.0 +
            west * 7.0 + east * 11.0 +
            len(pts)
        )
    return (n_polys, n_pts, round(checksum, 6))


def _dds_road_cache_key(fname, lat_n, lat_s, lon_w, lon_e, img_h, img_w,
                        grid_n, road_width_px, road_dilate_px,
                        separator_sig, rail_sig, heading_sig,
                        residential_poly_sig, excl_poly_sig,
                        existing_bld_poly_sig, sh_bld_sig,
                        custom_bld_sig=None):
    return {
        'version': 7,
        'fname': fname,
        'bounds': tuple(round(v, 8) for v in (lat_n, lat_s, lon_w, lon_e)),
        'shape': (int(img_h), int(img_w)),
        'grid_n': int(grid_n),
        'road_width_px': int(road_width_px),
        'road_dilate_px': int(road_dilate_px),
        'separator_sig': separator_sig,
        'rail_sig': rail_sig,
        'heading_sig': heading_sig,
        'residential_poly_sig': residential_poly_sig,
        'excl_poly_sig': excl_poly_sig,
        'existing_bld_poly_sig': existing_bld_poly_sig,
        'sh_bld_sig': sh_bld_sig,
        'custom_bld_sig': custom_bld_sig,
        'residential_road_types': tuple(sorted(RESIDENTIAL_HIGHWAY_TYPES)),
        'residential_buffer_m': float(RESIDENTIAL_FALLBACK_BUFFER_M),
        'residential_buffer_px_min': int(RESIDENTIAL_FALLBACK_BUFFER_PX_MIN),
    }


def _load_dds_road_cache(cache_path, key):
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


def _save_dds_road_cache(cache_path, key, road_mask, rail_mask, hgrid, n_osm_cells,
                         residential_area_mask, residential_area_source,
                         poly_mask, existing_bld_mask, sh_bld_mask,
                         custom_bld_mask=None):
    import pickle as _pickle
    try:
        with open(cache_path, 'wb') as f:
            _pickle.dump({
                'key': key,
                'road_mask': road_mask,
                'rail_mask': rail_mask,
                'hgrid': hgrid,
                'n_osm_cells': int(n_osm_cells),
                'residential_area_mask': residential_area_mask,
                'residential_area_source': residential_area_source,
                'poly_mask': poly_mask,
                'existing_bld_mask': existing_bld_mask,
                'sh_bld_mask': sh_bld_mask,
                'custom_bld_mask': custom_bld_mask,
            }, f, protocol=_pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass


def _cell_edge_hist(patch, bin_deg=5.0, resize_to=128):
    """Edge direction histogram in [0°, 180°) from an image patch via Sobel.

    Returns a float array of length int(180/bin_deg), or None if featureless.
    Convention: 0° = E-W edge, 90° = N-S edge (matches road angle space).
    Keeping the full histogram (rather than a single peak) means callers can
    score multiple candidate orientations against it independently — important
    when a cell contains blocks at different orientations.
    """
    if patch is None or patch.size == 0:
        return None
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY) if patch.ndim == 3 else patch
    if gray.shape[0] > resize_to or gray.shape[1] > resize_to:
        gray = cv2.resize(gray, (resize_to, resize_to))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy).ravel()
    if mag.sum() == 0:
        return None
    edge_deg = (np.degrees(np.arctan2(gy.ravel(), gx.ravel())) + 90.0) % 180.0
    n_bins = int(180 / bin_deg)
    hist, _ = np.histogram(edge_deg, bins=n_bins, range=(0.0, 180.0), weights=mag)
    return hist


def _contact_patch_straightness(xs, ys, band_px):
    """Return (is_straight, length_px) for one road-contact patch."""
    if len(xs) < 2:
        return False, 0.0
    pts = np.column_stack([
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
    ])
    centred = pts - pts.mean(axis=0, keepdims=True)
    if float(np.sum(centred * centred)) <= 0.0:
        return False, 0.0
    _, singular_values, vh = np.linalg.svd(centred, full_matrices=False)
    axis = vh[0]
    along = centred @ axis
    length_px = float(along.max() - along.min())
    if length_px <= 0.0:
        return False, 0.0
    if singular_values.size < 2 or float(singular_values[1]) <= 1e-6:
        rms_off_axis = 0.0
    else:
        off_axis = centred @ vh[1]
        rms_off_axis = float(np.sqrt(np.mean(off_axis * off_axis)))
    min_len = max(10.0, float(band_px) * 2.0)
    max_rms = max(2.0, length_px * 0.12)
    return length_px >= min_len and rms_off_axis <= max_rms, length_px


def _component_side_touch_headings(labels, stats, valid_labels, roads,
                                   lat_n, lat_s, lon_w, lon_e, img_h, img_w,
                                   band_px=8, img=None, bin_deg=5.0,
                                   contact_px=None, road_mask=None):
    """Estimate component headings from distinct road contacts on its outline.

    The contact search follows the actual component boundary rather than an
    axis-aligned bounding box, so rotated or irregular blocks are handled the
    same way as rectangular blocks. When a road mask is available, only road
    mask pixels touching the component boundary can vote; the wider band is kept
    for grouping and straightness checks.
    Each connected contact patch contributes one vote, so short local streets
    framing a block are not overwhelmed by a longer nearby segment.
    """
    headings = np.full(labels.max() + 1 if labels.size else 0, np.nan, dtype=np.float32)
    contact_counts = np.zeros(headings.shape, dtype=np.uint8)
    if headings.size == 0 or len(valid_labels) == 0:
        return headings, contact_counts
    if not roads:
        return headings, contact_counts

    n_bins90 = int(90 / bin_deg)
    n_bins180 = int(180 / bin_deg)
    band_px = max(2, int(band_px))
    contact_px = max(
        1,
        min(band_px, 6 if contact_px is None else int(contact_px)),
    )
    ring_labels = np.zeros(labels.shape, dtype=np.int32)
    contact_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (contact_px * 2 + 1, contact_px * 2 + 1)
    )
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for label in valid_labels:
        label = int(label)
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if w <= 0 or h <= 0:
            continue
        x0 = max(0, x - band_px - 1)
        y0 = max(0, y - band_px - 1)
        x1 = min(img_w, x + w + band_px + 1)
        y1 = min(img_h, y + h + band_px + 1)
        local_component = (labels[y0:y1, x0:x1] == label).astype(np.uint8)
        if not local_component.any():
            continue
        eroded = cv2.erode(local_component, erode_kernel)
        contact_dilated = cv2.dilate(local_component, contact_kernel)
        if road_mask is not None:
            local_roads = road_mask[y0:y1, x0:x1] != 0
            touch_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            touch_seed = (
                (cv2.dilate(local_component, touch_kernel) != 0) &
                (local_component == 0) &
                local_roads
            )
            boundary_ring = (
                (cv2.dilate(touch_seed.astype(np.uint8), contact_kernel) != 0) &
                local_roads
            )
        else:
            boundary_ring = (contact_dilated != 0) & (eroded == 0)
        target = ring_labels[y0:y1, x0:x1]
        target[(target == 0) & boundary_ring] = label

    def ll_to_px(lat, lon):
        x = (float(lon) - lon_w) / (lon_e - lon_w) * img_w
        y = (lat_n - float(lat)) / (lat_n - lat_s) * img_h
        return x, y

    mid_lat_rad = math.radians((lat_n + lat_s) * 0.5)
    m_per_lon = 111320.0 * math.cos(mid_lat_rad)
    m_per_lat = 110540.0
    margin = float(band_px + 2)
    contact_points = {}
    for road in roads:
        pts = road.get('pts', ())
        if len(pts) < 2:
            continue
        for (lat1, lon1), (lat2, lon2) in zip(pts[:-1], pts[1:]):
            x1, y1 = ll_to_px(lat1, lon1)
            x2, y2 = ll_to_px(lat2, lon2)
            if (
                max(x1, x2) < -margin or min(x1, x2) >= img_w + margin or
                max(y1, y2) < -margin or min(y1, y2) >= img_h + margin
            ):
                continue
            px_len = math.hypot(x2 - x1, y2 - y1)
            if px_len <= 0.0:
                continue
            dx_m = (float(lon2) - float(lon1)) * m_per_lon
            dy_m = (float(lat2) - float(lat1)) * m_per_lat
            if dx_m == 0.0 and dy_m == 0.0:
                continue
            raw_deg = (math.degrees(math.atan2(dy_m, dx_m)) % 180.0)
            b180 = int((raw_deg % 180.0) / bin_deg) % n_bins180
            n_samples = max(2, min(192, int(px_len / max(1.0, band_px * 0.5)) + 1))
            xs = np.rint(np.linspace(x1, x2, n_samples)).astype(np.int32)
            ys = np.rint(np.linspace(y1, y2, n_samples)).astype(np.int32)
            valid = (xs >= 0) & (xs < img_w) & (ys >= 0) & (ys < img_h)
            if not np.any(valid):
                continue
            xs = xs[valid]
            ys = ys[valid]
            touched_labels = ring_labels[ys, xs]
            touched = touched_labels != 0
            if not np.any(touched):
                continue
            for label, px, py in zip(touched_labels[touched], xs[touched], ys[touched]):
                contact_points.setdefault(int(label), []).append((int(px), int(py), b180))

    for label in valid_labels:
        label = int(label)
        points = contact_points.get(label)
        if not points:
            continue

        pts_arr = np.asarray(points, dtype=np.int32)
        pxs = pts_arr[:, 0]
        pys = pts_arr[:, 1]
        bins = pts_arr[:, 2]
        x0 = max(0, int(pxs.min()) - band_px)
        x1 = min(img_w, int(pxs.max()) + band_px + 1)
        y0 = max(0, int(pys.min()) - band_px)
        y1 = min(img_h, int(pys.max()) + band_px + 1)
        if x1 <= x0 or y1 <= y0:
            continue

        connect_px = max(1, min(3, band_px // 2))
        connect_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (connect_px * 2 + 1, connect_px * 2 + 1)
        )

        votes90 = np.zeros(n_bins90, dtype=np.float32)
        votes180 = np.zeros(n_bins180, dtype=np.float32)
        contact_votes = []
        remaining = np.ones(bins.shape, dtype=bool)
        while np.any(remaining):
            orient_hist = np.bincount(bins[remaining], minlength=n_bins180)
            peak_bin = int(np.argmax(orient_hist))
            circular_dist = np.abs(
                ((bins - peak_bin + n_bins180 // 2) % n_bins180) - n_bins180 // 2
            )
            orient_sel = remaining & (circular_dist <= 1)
            remaining[orient_sel] = False
            if not np.any(orient_sel):
                continue

            sel_x = pxs[orient_sel]
            sel_y = pys[orient_sel]
            sel_bins = bins[orient_sel]
            local_contact = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            local_contact[sel_y - y0, sel_x - x0] = 1
            local_contact = cv2.dilate(local_contact, connect_kernel)
            n_contact_labels, contact_labels = cv2.connectedComponents(
                local_contact, connectivity=8
            )
            sample_contact_labels = contact_labels[sel_y - y0, sel_x - x0]
            for contact_label in range(1, n_contact_labels):
                patch_mask = sample_contact_labels == contact_label
                contact_bins = sel_bins[patch_mask]
                if contact_bins.size == 0:
                    continue
                straight, length_px = _contact_patch_straightness(
                    sel_x[patch_mask],
                    sel_y[patch_mask],
                    band_px,
                )
                side_hist = np.bincount(contact_bins, minlength=n_bins180).astype(np.float32)
                smooth = (
                    np.roll(side_hist, 1) +
                    side_hist +
                    np.roll(side_hist, -1)
                ) / 3.0
                peak180 = int(np.argmax(smooth))
                contact_votes.append((peak180, bool(straight), length_px))

        if not contact_votes:
            continue

        straight_votes = [vote for vote in contact_votes if vote[1]]
        usable_votes = straight_votes or contact_votes
        contact_counts[label] = min(len(usable_votes), 255)
        for peak180, _straight, _length_px in usable_votes:
            raw_angle = peak180 * bin_deg + bin_deg * 0.5
            votes180[peak180] += 1.0
            votes90[int((raw_angle % 90.0) / bin_deg) % n_bins90] += 1.0
        v90s = (np.roll(votes90, 1) + votes90 + np.roll(votes90, -1)) / 3.0
        dom_mod90 = int(np.argmax(v90s)) * bin_deg + bin_deg * 0.5
        opt1 = dom_mod90
        opt2 = dom_mod90 + 90.0

        dom_angle = None
        if img is not None:
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            x0 = max(0, x - band_px)
            y0 = max(0, y - band_px)
            x1 = min(img_w, x + w + band_px)
            y1 = min(img_h, y + h + band_px)
            edge_hist = _cell_edge_hist(img[y0:y1, x0:x1], bin_deg=bin_deg)
            if edge_hist is not None:
                def _img_score(angle):
                    b = int(angle / bin_deg) % n_bins180
                    return (
                        edge_hist[(b - 1) % n_bins180] +
                        edge_hist[b] +
                        edge_hist[(b + 1) % n_bins180]
                    )
                dom_angle = opt1 if _img_score(opt1) >= _img_score(opt2) else opt2

        if dom_angle is None:
            def _road_score(angle):
                b = int(angle / bin_deg) % n_bins180
                return (
                    votes180[(b - 1) % n_bins180] +
                    votes180[b] +
                    votes180[(b + 1) % n_bins180]
                )
            dom_angle = opt1 if _road_score(opt1) >= _road_score(opt2) else opt2

        headings[label] = (90.0 - dom_angle) % 360.0

    return headings, contact_counts


def _simheaven_building_zone_headings(objects, labels, valid_labels,
                                      lat_n, lat_s, lon_w, lon_e,
                                      img_h, img_w, bin_deg=5.0):
    """Return per-component headings from simHeaven buildings inside it."""
    headings = np.full(labels.max() + 1 if labels.size else 0, np.nan, dtype=np.float32)
    counts = np.zeros(headings.shape, dtype=np.uint16)
    if headings.size == 0 or len(valid_labels) == 0 or not objects:
        return headings, counts

    px = np.rint((objects['lon'] - lon_w) / (lon_e - lon_w) * img_w).astype(np.int32)
    py = np.rint((lat_n - objects['lat']) / (lat_n - lat_s) * img_h).astype(np.int32)
    in_img = (px >= 0) & (px < img_w) & (py >= 0) & (py < img_h)
    if not bool(np.any(in_img)):
        return headings, counts

    px = px[in_img]
    py = py[in_img]
    obj_heading = objects['heading'][in_img].astype(np.float32)
    obj_w = objects['w_m'][in_img].astype(np.float32)
    obj_h = objects['h_m'][in_img].astype(np.float32)
    obj_labels = labels[py, px]
    valid_label_set = set(int(label) for label in valid_labels)

    n_bins = int(180 / bin_deg)
    per_label_votes = {}
    for label, heading, width_m, height_m in zip(obj_labels, obj_heading, obj_w, obj_h):
        label = int(label)
        if label == 0 or label not in valid_label_set:
            continue
        long_side = max(float(width_m), float(height_m))
        short_side = max(1e-6, min(float(width_m), float(height_m)))
        aspect = long_side / short_side
        if aspect < 1.15:
            continue
        b = int((float(heading) % 180.0) / bin_deg) % n_bins
        weight = min(4.0, max(1.0, aspect - 0.15))
        per_label_votes.setdefault(label, np.zeros(n_bins, dtype=np.float32))[b] += weight
        counts[label] += 1

    for label, hist in per_label_votes.items():
        if counts[label] <= 0 or float(hist.sum()) <= 0.0:
            continue
        smooth = (np.roll(hist, 1) + hist + np.roll(hist, -1)) / 3.0
        peak = int(np.argmax(smooth))
        headings[label] = (peak * bin_deg + bin_deg * 0.5) % 180.0

    return headings, counts


def _road_heading_grid(roads, lat_n, lat_s, lon_w, lon_e,
                       img_h, img_w, grid_n, img=None, segments=None):
    """Compute dominant road bearing per grid cell.

    Three-stage approach:
      1. Mod-90° histogram — parallel AND perpendicular roads reinforce the
         same bin, so the dominant block orientation wins over outlier roads.
      2. Imagery tie-break — use Sobel edge histogram on the DDS patch to
         pick which of the two perpendicular options matches actual building
         edges visible in the imagery.
      3. Mod-180° road fallback — if imagery is unavailable or featureless,
         fall back to whichever perpendicular option has more road length.

    Returns an (grid_n, grid_n) float array of X-Plane compass headings
    in [0, 360°), with NaN where no road segments fall in that cell.
    """
    cell_h = img_h / grid_n
    cell_w = img_w / grid_n

    BIN_DEG  = 5.0
    N_BINS90 = int(90  / BIN_DEG)   # 18 bins covering [0°,  90°)
    N_BINS180= int(180 / BIN_DEG)   # 36 bins covering [0°, 180°)

    votes90  = np.zeros((grid_n, grid_n, N_BINS90))
    votes180 = np.zeros((grid_n, grid_n, N_BINS180))

    if segments is None:
        segments = _prepare_segment_arrays(roads)

    if segments:
        px = (segments['lon'] - lon_w) / (lon_e - lon_w) * img_w
        py = (lat_n - segments['lat']) / (lat_n - lat_s) * img_h
        gi = (py // cell_h).astype(np.int32)
        gj = (px // cell_w).astype(np.int32)
        valid = (gi >= 0) & (gi < grid_n) & (gj >= 0) & (gj < grid_n)
        if bool(np.any(valid)):
            b90 = ((segments['raw_deg'][valid] % 90.0) / BIN_DEG).astype(np.int32) % N_BINS90
            b180 = ((segments['raw_deg'][valid] % 180.0) / BIN_DEG).astype(np.int32) % N_BINS180
            np.add.at(votes90, (gi[valid], gj[valid], b90), segments['len'][valid])
            np.add.at(votes180, (gi[valid], gj[valid], b180), segments['len'][valid])

    hgrid = np.full((grid_n, grid_n), np.nan)
    for gi in range(grid_n):
        for gj in range(grid_n):
            if np.sum(votes90[gi, gj]) == 0:
                continue

            # Stage 1: dominant grid orientation from mod-90° peak
            v90  = votes90[gi, gj]
            v90s = (np.roll(v90, 1) + v90 + np.roll(v90, -1)) / 3.0
            peak_bin  = int(np.argmax(v90s))
            dom_mod90 = peak_bin * BIN_DEG + BIN_DEG / 2.0

            opt1 = dom_mod90          # one perpendicular option
            opt2 = dom_mod90 + 90.0  # the other

            # Stage 2: imagery tie-break — score each option against the full
            # edge histogram.  Avoids collapsing the histogram to a single peak,
            # which would break when a cell contains blocks at different orientations.
            dom_angle = None
            if img is not None:
                y0 = int(gi * cell_h);  y1 = min(img_h, int((gi + 1) * cell_h))
                x0 = int(gj * cell_w);  x1 = min(img_w, int((gj + 1) * cell_w))
                eh = _cell_edge_hist(img[y0:y1, x0:x1], bin_deg=BIN_DEG)
                if eh is not None:
                    def _img_score(angle):
                        b = int(angle / BIN_DEG) % N_BINS180
                        return (eh[(b - 1) % N_BINS180] +
                                eh[ b                 ] +
                                eh[(b + 1) % N_BINS180])
                    dom_angle = opt1 if _img_score(opt1) >= _img_score(opt2) else opt2

            # Stage 3: mod-180° road fallback
            if dom_angle is None:
                def _score(angle):
                    b = int(angle / BIN_DEG) % N_BINS180
                    return (votes180[gi, gj, (b - 1) % N_BINS180] +
                            votes180[gi, gj,  b                  ] +
                            votes180[gi, gj, (b + 1) % N_BINS180])
                dom_angle = opt1 if _score(opt1) >= _score(opt2) else opt2

            hgrid[gi, gj] = (90.0 - dom_angle) % 360.0
    return hgrid


def _building_zone_cell_mask(bld_zone, img_h, img_w, grid_n):
    """Return grid cells that contain building-zone pixels."""
    cell_mask = np.zeros((grid_n, grid_n), dtype=bool)
    ys, xs = np.nonzero(bld_zone)
    if xs.size == 0:
        return cell_mask
    cell_h = img_h / grid_n
    cell_w = img_w / grid_n
    gi = np.minimum(grid_n - 1, (ys / cell_h).astype(np.int32))
    gj = np.minimum(grid_n - 1, (xs / cell_w).astype(np.int32))
    cell_mask[gi, gj] = True
    return cell_mask


def _fill_heading_grid_nearest(hgrid, segments, lat_n, lat_s, lon_w, lon_e,
                               img_h, img_w, grid_n, cell_mask=None):
    """Fill NaN heading cells from the nearest precomputed segment midpoint."""
    if not segments or not np.any(np.isnan(hgrid)):
        return hgrid

    px = (segments['lon'] - lon_w) / (lon_e - lon_w) * img_w
    py = (lat_n - segments['lat']) / (lat_n - lat_s) * img_h
    if len(px) == 0:
        return hgrid

    cell_h = img_h / grid_n
    cell_w = img_w / grid_n
    missing_mask = np.isnan(hgrid)
    if cell_mask is not None:
        missing_mask &= cell_mask
    missing = np.argwhere(missing_mask)
    for gi, gj in missing:
        cy = (gi + 0.5) * cell_h
        cx = (gj + 0.5) * cell_w
        d2 = (py - cy) ** 2 + (px - cx) ** 2
        hgrid[gi, gj] = float(segments['heading'][int(np.argmin(d2))])
    return hgrid


def _image_heading_grid(img, img_h, img_w, grid_n, cell_mask=None):
    """Estimate per-cell heading from imagery edges when road vectors are absent."""
    hgrid = np.full((grid_n, grid_n), np.nan)
    if img is None:
        return hgrid
    bin_deg = 5.0
    cell_h = img_h / grid_n
    cell_w = img_w / grid_n
    cells = (
        np.argwhere(cell_mask)
        if cell_mask is not None else
        np.indices((grid_n, grid_n)).reshape(2, -1).T
    )
    for gi, gj in cells:
        gi = int(gi)
        gj = int(gj)
        y0 = int(gi * cell_h)
        y1 = min(img_h, int((gi + 1) * cell_h))
        x0 = int(gj * cell_w)
        x1 = min(img_w, int((gj + 1) * cell_w))
        hist = _cell_edge_hist(img[y0:y1, x0:x1], bin_deg=bin_deg)
        if hist is None:
            continue
        smooth = np.convolve(np.r_[hist[-1], hist, hist[0]], [1, 2, 1], mode='same')[1:-1]
        if float(smooth.max()) <= 0.0:
            continue
        edge_angle = float(np.argmax(smooth) * bin_deg + bin_deg / 2.0)
        hgrid[gi, gj] = (90.0 - edge_angle) % 360.0
    return hgrid


# ── OSM polygon / exclusion helpers ──────────────────────────────────────────
def _load_osm_closed_ways(osm_bz2_path, cache_dir=None):
    """Return list of closed-way polygons as [(lat,lon),...] from an .osm.bz2 file."""
    import bz2, xml.etree.ElementTree as ET
    if not os.path.exists(osm_bz2_path):
        return []
    def _parse():
        with bz2.open(osm_bz2_path, 'rb') as f:
            root = ET.parse(f).getroot()
        nodes = {}
        for node in root.iter('node'):
            nodes[node.get('id')] = (float(node.get('lat')), float(node.get('lon')))
        polys = []
        for way in root.iter('way'):
            pts = [nodes[nd.get('ref')] for nd in way.iter('nd') if nd.get('ref') in nodes]
            if len(pts) >= 4 and pts[0] == pts[-1]:
                polys.append(pts)
        return polys

    return PCACHE.load_or_build(
        osm_bz2_path,
        cache_dir,
        "osm_parse",
        _parse,
        version="closed-ways-v1",
    )


def _clip_polygon_to_latlon_bounds(poly, south, north, west, east):
    """Clip a lat/lon polygon to a rectangular lat/lon bounds box."""
    def _clip(points, inside, intersect):
        if not points:
            return []
        output = []
        prev = points[-1]
        prev_inside = inside(prev)
        for curr in points:
            curr_inside = inside(curr)
            if curr_inside:
                if not prev_inside:
                    output.append(intersect(prev, curr))
                output.append(curr)
            elif prev_inside:
                output.append(intersect(prev, curr))
            prev = curr
            prev_inside = curr_inside
        return output

    def _intersect_lat(a, b, lat_value):
        a_lat, a_lon = a
        b_lat, b_lon = b
        denom = b_lat - a_lat
        if abs(denom) < 1e-12:
            return lat_value, a_lon
        t = (lat_value - a_lat) / denom
        return lat_value, a_lon + t * (b_lon - a_lon)

    def _intersect_lon(a, b, lon_value):
        a_lat, a_lon = a
        b_lat, b_lon = b
        denom = b_lon - a_lon
        if abs(denom) < 1e-12:
            return a_lat, lon_value
        t = (lon_value - a_lon) / denom
        return a_lat + t * (b_lat - a_lat), lon_value

    pts = list(poly)
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    pts = _clip(pts, lambda p: p[0] >= south, lambda a, b: _intersect_lat(a, b, south))
    pts = _clip(pts, lambda p: p[0] <= north, lambda a, b: _intersect_lat(a, b, north))
    pts = _clip(pts, lambda p: p[1] >= west, lambda a, b: _intersect_lon(a, b, west))
    pts = _clip(pts, lambda p: p[1] <= east, lambda a, b: _intersect_lon(a, b, east))
    return pts if len(pts) >= 3 else []


def _rasterize_polygons(polys, lat_n, lat_s, lon_w, lon_e, img_h, img_w):
    """Rasterize filled closed-way polygons onto a uint8 mask."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    def ll_to_px(lat, lon):
        x = int((lon - lon_w) / (lon_e - lon_w) * img_w)
        y = int((lat_n - lat) / (lat_n - lat_s) * img_h)
        return x, y

    for poly in polys:
        pts_px = np.array([ll_to_px(lat, lon) for lat, lon in poly], dtype=np.int32)
        min_x = int(pts_px[:, 0].min())
        max_x = int(pts_px[:, 0].max())
        min_y = int(pts_px[:, 1].min())
        max_y = int(pts_px[:, 1].max())
        # Only draw polygons whose bbox intersects this DDS tile.  Most coastal
        # polygons are faster through OpenCV directly; clip only pathological
        # huge offscreen rings that make fillPoly do unnecessary work.
        if max_x >= 0 and min_x < img_w and max_y >= 0 and min_y < img_h:
            bbox_w = max_x - min_x + 1
            bbox_h = max_y - min_y + 1
            if (
                bbox_w > img_w * 4 or bbox_h > img_h * 4 or
                max(abs(min_x), abs(max_x), abs(min_y), abs(max_y)) > 32767
            ):
                clipped = _clip_polygon_to_latlon_bounds(
                    poly, lat_s, lat_n, lon_w, lon_e
                )
                if not clipped:
                    continue
                pts_px = np.array(
                    [ll_to_px(lat, lon) for lat, lon in clipped], dtype=np.int32
                )
            cv2.fillPoly(mask, [pts_px], 1)
    return mask


def _mesh_file_for_tile(tex_dir, lat, lon):
    """Return the tile mesh path next to an Ortho4XP textures directory."""
    lat_i = int(lat)
    lon_i = int(lon)
    short = f"{lat_i:+03d}{lon_i:+04d}"
    candidates = []
    tex_parent = os.path.dirname(os.path.abspath(tex_dir))
    if os.path.basename(os.path.abspath(tex_dir)).lower() == "textures":
        candidates.append(os.path.join(tex_parent, f"Data{short}.mesh"))
    candidates.append(os.path.join(os.path.abspath(tex_dir), f"Data{short}.mesh"))
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0] if candidates else None


def _mesh_water_signature(mesh_path):
    """Return a cache-key-safe signature for the mesh water artifact."""
    if not mesh_path or not os.path.exists(mesh_path):
        return None
    stat = os.stat(mesh_path)
    return (
        os.path.realpath(mesh_path),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


def _read_mesh_water_triangles(mesh_path):
    """Read only water triangle lat/lon vertices from an Ortho4XP .mesh file."""
    with open(mesh_path, "r", encoding="utf-8", errors="ignore") as mesh_file:
        mesh_version = float(mesh_file.readline().strip().split()[-1])
        for _ in range(3):
            mesh_file.readline()
        nbr_nodes = int(mesh_file.readline())
        node_lons = np.empty(nbr_nodes, dtype=np.float64)
        node_lats = np.empty(nbr_nodes, dtype=np.float64)
        for idx in range(nbr_nodes):
            parts = mesh_file.readline().split()
            node_lons[idx] = float(parts[0])
            node_lats[idx] = float(parts[1])
        for _ in range(3):
            mesh_file.readline()
        for _ in range(nbr_nodes):
            mesh_file.readline()
        for _ in range(2):
            mesh_file.readline()
        nbr_tris = int(mesh_file.readline())
        has_water = 7 if mesh_version >= 1.3 else 3
        water_tris = []
        for _ in range(nbr_tris):
            parts = mesh_file.readline().split()
            if len(parts) < 4:
                continue
            tri_type = int(parts[3])
            if not (tri_type & has_water):
                continue
            idx = [int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2]) - 1]
            water_tris.append(
                [
                    (node_lats[idx[0]], node_lons[idx[0]]),
                    (node_lats[idx[1]], node_lons[idx[1]]),
                    (node_lats[idx[2]], node_lons[idx[2]]),
                ]
            )
    if not water_tris:
        return np.empty((0, 3, 2), dtype=np.float64)
    return np.asarray(water_tris, dtype=np.float64)


def _load_mesh_water_index(mesh_path, cache_dir):
    """Load water triangles from the Ortho4XP mesh produced by Step 2.

    The mesh is downstream of Ortho4XP's normal water acquisition path, so this
    covers both OSM water and the default-scenery fallback used when OSM fails.
    """
    if not mesh_path or not os.path.exists(mesh_path):
        return None

    def _build():
        tris = _read_mesh_water_triangles(mesh_path)
        if not tris.size:
            return None
        return {
            "tris": tris,
            "south": tris[:, :, 0].min(axis=1).astype(np.float64, copy=False),
            "north": tris[:, :, 0].max(axis=1).astype(np.float64, copy=False),
            "west": tris[:, :, 1].min(axis=1).astype(np.float64, copy=False),
            "east": tris[:, :, 1].max(axis=1).astype(np.float64, copy=False),
        }

    return PCACHE.load_or_build(
        mesh_path,
        cache_dir,
        "mesh_water",
        _build,
        version="mesh-water-v2",
    )


def _rasterize_mesh_water_mask(mesh_water_index, lat_n, lat_s, lon_w, lon_e,
                               img_h, img_w, pad_deg=0.0):
    """Rasterize mesh water triangles intersecting one DDS bounds."""
    if not mesh_water_index:
        return None
    keep = (
        (mesh_water_index["north"] >= lat_s - pad_deg) &
        (mesh_water_index["south"] <= lat_n + pad_deg) &
        (mesh_water_index["east"] >= lon_w - pad_deg) &
        (mesh_water_index["west"] <= lon_e + pad_deg)
    )
    if not bool(np.any(keep)):
        return None
    tris = mesh_water_index["tris"][keep]
    xs = np.rint((tris[:, :, 1] - lon_w) / (lon_e - lon_w) * img_w).astype(np.int32)
    ys = np.rint((lat_n - tris[:, :, 0]) / (lat_n - lat_s) * img_h).astype(np.int32)
    pts = np.stack((xs, ys), axis=2)
    intersects = (
        (pts[:, :, 0].max(axis=1) >= 0) &
        (pts[:, :, 0].min(axis=1) < img_w) &
        (pts[:, :, 1].max(axis=1) >= 0) &
        (pts[:, :, 1].min(axis=1) < img_h)
    )
    if not bool(np.any(intersects)):
        return None
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.ascontiguousarray(poly) for poly in pts[intersects]], 1)
    return mask


def _poly_bounds(poly):
    """Return ``(south, north, west, east)`` for one polygon."""
    if not poly:
        return (0.0, 0.0, 0.0, 0.0)
    lats = [pt[0] for pt in poly]
    lons = [pt[1] for pt in poly]
    return min(lats), max(lats), min(lons), max(lons)


def _prepare_polygons(polys):
    """Attach cached bounds so per-DDS polygon filtering stays cheap."""
    prepared = []
    for poly in polys or []:
        if len(poly) < 3:
            continue
        prepared.append({'pts': poly, '_bounds': _poly_bounds(poly)})
    return prepared


def _polys_for_bounds(polys, lat_n, lat_s, lon_w, lon_e, pad_deg=0.0):
    """Return polygons whose bounding boxes intersect the DDS bounds."""
    if not polys:
        return []
    south = lat_s - pad_deg
    north = lat_n + pad_deg
    west = lon_w - pad_deg
    east = lon_e + pad_deg
    if isinstance(polys, dict) and {'items', 'south', 'north', 'west', 'east'} <= set(polys):
        return [poly.get('pts', poly) for poly in BBOX.query_bounds(polys, south, north, west, east)]
    filtered = []
    for poly in polys:
        if isinstance(poly, dict):
            pts = poly.get('pts', ())
            bounds = poly.get('_bounds')
        else:
            pts = poly
            bounds = None
        if not pts:
            continue
        if bounds is None:
            bounds = _poly_bounds(pts)
        poly_south, poly_north, poly_west, poly_east = bounds
        if (
            poly_north >= south and
            poly_south <= north and
            poly_east >= west and
            poly_west <= east
        ):
            filtered.append(pts)
    return filtered


def _download_and_cache_osm(lat, lon, cache_path, timeout=45):
    """Download OSM building, transport, and residential-area data.

    Fetches one combined Overpass query for the 1 degree tile and caches it as
    bz2-compressed OSM XML. The cached file feeds three downstream uses:
    building footprint exclusions, railway exclusions, and residential-area
    guidance for small-house placement.
    """
    if os.path.exists(cache_path):
        return True

    import bz2 as _bz2
    bbox = f"{int(lat)},{int(lon)},{int(lat)+1},{int(lon)+1}"
    query = (
        f'[out:xml][timeout:{timeout}];'
        f'('
        f'  way["building"]({bbox});'
        f'  way["railway"~"^(rail|light_rail|subway|tram|narrow_gauge|monorail)$"]({bbox});'
        f'  way["aeroway"~"^(aerodrome|apron|taxiway|runway|terminal|hangar)$"]({bbox});'
        f'  way["landuse"~"^(railway|aeroway)$"]({bbox});'
        f'  way["landuse"="residential"]({bbox});'
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
            with _bz2.open(cache_path, 'wb') as f:
                f.write(data)
            print(f"  [OSM excl] Downloaded {len(data)//1024} KB → {os.path.basename(cache_path)}")
            return True
        except Exception as e:
            print(f"  [OSM excl] {server} failed: {e}")
    return False


def _download_and_cache_osm_roads(lat, lon, cache_path, timeout=60):
    """Download all OSM highway ways for zone splitting and road context."""
    if os.path.exists(cache_path):
        return True

    import bz2 as _bz2
    bbox = f"{int(lat)},{int(lon)},{int(lat)+1},{int(lon)+1}"
    query = (
        f'[out:xml][timeout:{timeout}];'
        f'('
        f'  way["highway"]({bbox});'
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
            with _bz2.open(cache_path, 'wb') as f:
                f.write(data)
            print(f"  [OSM roads] Downloaded {len(data)//1024} KB -> {os.path.basename(cache_path)}")
            return True
        except Exception as e:
            print(f"  [OSM roads] {server} failed: {e}")
    return False


def _load_simheaven_network(custom_scenery_dir, tile_lat, tile_lon, dsftool_path, cache_dir):
    """Parse simHeaven X-World network DSFs into road-style dicts."""
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
            print(f"  [simHeaven net] {folder_name}: +{len(road_ways) - segment_count_before} segments")
        except Exception as exc:
            print(f"  [simHeaven net] failed {dsf_path}: {exc}")
    return road_ways


def _is_simheaven_building_object(path):
    """Return True for simHeaven object defs that represent physical buildings."""
    p = (path or '').replace('\\', '/').lower()
    if not p.startswith('simheaven/'):
        return False
    if any(token in p for token in (
        '/houses/', '/industrial/', '/commercial/', '/residential/',
        '/farms/', '/sheds/',
    )):
        return True
    if '/landmarks/' in p:
        return any(name in p for name in ('church', 'chapel', 'mosque'))
    return False


def _is_simheaven_building_polygon(path):
    """Return True for simHeaven facade defs that occupy building footprints."""
    p = (path or '').replace('\\', '/').lower()
    if not p.startswith('simheaven/'):
        return False
    if '/ground/' in p or 'parking' in p or 'pier' in p:
        return False
    return '/facades/' in p or '/landmarks/' in p


_SIMHEAVEN_DIMS_RE = re.compile(r'_(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)(?:x\d+(?:\.\d+)?)?')
_SIMHEAVEN_FLOORS_RE = re.compile(
    r'_(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)(?:\.obj)?$'
)
_DEFAULT_OBJECT_DIMS_RE = re.compile(
    r'(?:^|/)(?:feat_Building|(?:hill|in|ind|out)_sq)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)'
)
_DEFAULT_OBJECT_HEIGHT_RE = re.compile(
    r'(?:^|/)feat_Building_\d+(?:\.\d+)?_\d+(?:\.\d+)?_(\d+(?:\.\d+)?)'
)
_SFD_HEIGHT_TOKEN_RE = re.compile(r'_(\d+(?:\.\d+)?)m(?:_|\.|$)', re.IGNORECASE)
VERY_TALL_BUILDING_TOKENS = (
    "skyscraper",
    "highrise",
    "high_rise",
    "high-rise",
    "/tower",
    "tower_",
    "_tower",
)


def _simheaven_object_dims(path):
    """Infer a simHeaven object's footprint dimensions from its filename."""
    name = os.path.basename((path or '').replace('\\', '/')).lower()
    match = _SIMHEAVEN_DIMS_RE.search(name)
    if match:
        return float(match.group(1)), float(match.group(2))
    if any(token in name for token in ('church', 'chapel', 'mosque')):
        return 26.0, 26.0
    return 14.0, 14.0


def _simheaven_object_floor_count(path):
    """Infer a simHeaven object's floor count from names like house_09x12x2."""
    name = os.path.basename((path or '').replace('\\', '/')).lower()
    match = _SIMHEAVEN_FLOORS_RE.search(name)
    if not match:
        return None
    return float(match.group(3))


def _default_object_dims(path):
    """Infer a default-library object footprint from its virtual path."""
    p = (path or '').replace('\\', '/')
    match = _DEFAULT_OBJECT_DIMS_RE.search(p)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def _default_object_height_m(path):
    """Infer default-library object height when encoded in the virtual path."""
    p = (path or '').replace('\\', '/')
    match = _DEFAULT_OBJECT_HEIGHT_RE.search(p)
    if not match:
        return None
    return float(match.group(1)) * 0.3048


def _sfd_object_height_m(path):
    name = os.path.basename((path or '').replace('\\', '/'))
    if "apartment" not in name.lower():
        return None
    match = _SFD_HEIGHT_TOKEN_RE.search(name)
    if not match:
        return None
    return float(match.group(1))


def _object_estimated_height_m(path):
    p = (path or '').replace('\\', '/').lower()
    floors = _simheaven_object_floor_count(p)
    if floors is not None:
        return floors * 3.2
    default_height = _default_object_height_m(path)
    if default_height is not None:
        return default_height
    return _sfd_object_height_m(path)


def _is_very_tall_building_asset(path=None, height_m=None):
    if height_m is not None and float(height_m) > MAX_GENERATED_BUILDING_HEIGHT_M:
        return True
    p = (path or '').replace('\\', '/').lower()
    return any(token in p for token in VERY_TALL_BUILDING_TOKENS)


def _norm_library_path(path):
    return (path or '').replace('\\', '/').strip().lower().lstrip('/')


def _read_obj8_bounds(obj_path, cache_dir=None):
    """Return OBJ8 X/Z footprint bounds as ``(xmin, xmax, zmin, zmax)``."""
    if not obj_path or not os.path.isfile(obj_path):
        return None

    def _parse():
        min_x = min_z = float('inf')
        max_x = max_z = float('-inf')
        n_vertices = 0
        with open(obj_path, 'r', encoding='utf-8', errors='ignore') as handle:
            for raw_line in handle:
                line = raw_line.lstrip()
                if not line.startswith('VT'):
                    continue
                parts = line.split()
                if len(parts) < 4 or parts[0] != 'VT':
                    continue
                try:
                    x = float(parts[1])
                    z = float(parts[3])
                except ValueError:
                    continue
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_z = min(min_z, z)
                max_z = max(max_z, z)
                n_vertices += 1
        if n_vertices <= 0 or max_x <= min_x or max_z <= min_z:
            return None
        return (float(min_x), float(max_x), float(min_z), float(max_z))

    return PCACHE.load_or_build(
        obj_path,
        cache_dir,
        "obj8_bounds",
        _parse,
        version="obj8-xz-bounds-v1",
    )


def _active_custom_library_index(custom_scenery_dir):
    """Build a best-effort active library virtual-path to physical OBJ map."""
    index = {}
    for _, package_dir in active_scenery_pack_dirs(custom_scenery_dir):
        library_txt = os.path.join(package_dir, 'library.txt')
        if not os.path.isfile(library_txt):
            continue
        try:
            with open(library_txt, 'r', encoding='utf-8', errors='ignore') as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line or line.startswith(('#', '//')):
                        continue
                    parts = line.split()
                    if not parts:
                        continue
                    command = parts[0].upper()
                    virtual_path = physical_path = None
                    if command in {'EXPORT', 'EXPORT_EXCLUDE', 'EXPORT_BACKUP'} and len(parts) >= 3:
                        virtual_path, physical_path = parts[1], parts[2]
                    elif command == 'EXPORT_RATIO' and len(parts) >= 4:
                        virtual_path, physical_path = parts[2], parts[3]
                    if not virtual_path or not physical_path:
                        continue
                    if not physical_path.lower().replace('\\', '/').endswith('.obj'):
                        continue
                    resolved = os.path.abspath(os.path.join(package_dir, physical_path))
                    if not os.path.isfile(resolved):
                        continue
                    index.setdefault(_norm_library_path(virtual_path), resolved)
        except Exception:
            continue
    return index


def _resolve_custom_object_path(object_path, package_dir, library_index):
    """Resolve a DSF object definition to a readable local OBJ8 path when possible."""
    if not object_path:
        return None
    object_path = object_path.replace('\\', '/')
    if os.path.isabs(object_path) and os.path.isfile(object_path):
        return os.path.abspath(object_path)
    if package_dir:
        local_path = os.path.abspath(os.path.join(package_dir, object_path))
        if os.path.isfile(local_path):
            return local_path
    return library_index.get(_norm_library_path(object_path))


def _custom_object_dims(object_path, package_dir, library_index, cache_dir):
    """Return existing custom object footprint dimensions in metres, or None."""
    p = (object_path or '').replace('\\', '/')
    p_lower = p.lower()
    if p_lower.startswith('simheaven/') and _is_simheaven_building_object(p):
        return _simheaven_object_dims(p)
    bounds_m = _bounds_for_object_path(p)
    if bounds_m is not None:
        xmin, xmax, zmin, zmax = bounds_m
        return float(xmax) - float(xmin), float(zmax) - float(zmin)
    dims = _default_object_dims(p)
    if dims is not None:
        return dims
    resolved_obj = _resolve_custom_object_path(p, package_dir, library_index)
    bounds_m = _read_obj8_bounds(resolved_obj, cache_dir)
    if bounds_m is None:
        return None
    xmin, xmax, zmin, zmax = bounds_m
    return float(xmax) - float(xmin), float(zmax) - float(zmin)


def _load_simheaven_building_exclusions(custom_scenery_dir, tile_lat, tile_lon, dsftool_path, cache_dir):
    """Parse simHeaven building objects/facades into exclusion geometry."""
    objects = []
    polys = []
    seen_layers = set()

    for folder_name, dsf_path in find_simheaven_building_dsfs(custom_scenery_dir, tile_lat, tile_lon):
        folder_key = folder_name.lower()
        if folder_key in seen_layers:
            continue
        seen_layers.add(folder_key)

        n_obj0 = len(objects)
        n_poly0 = len(polys)
        try:
            cached_text_path = ensure_cached_dsf_text(
                dsf_path,
                dsftool_path,
                cache_dir,
                create_no_window=SEGFORMER._CREATE_NO_WINDOW,
            )
            def _parse():
                parsed_objects = []
                parsed_polys = []
                object_defs = []
                polygon_defs = []
                current_polygon_is_building = False
                current_winding = None

                with open(cached_text_path, "r", encoding="utf-8", errors="ignore") as text_file:
                    for raw_line in text_file:
                        line = raw_line.strip()
                        if line.startswith("OBJECT_DEF "):
                            object_defs.append(line.split(" ", 1)[1])
                        elif line.startswith("POLYGON_DEF "):
                            polygon_defs.append(line.split(" ", 1)[1])
                        elif line.startswith("OBJECT "):
                            parts = line.split()
                            try:
                                object_index = int(parts[1])
                                object_path = object_defs[object_index]
                                if not _is_simheaven_building_object(object_path):
                                    continue
                                object_lon = float(parts[2])
                                object_lat = float(parts[3])
                                object_heading = float(parts[4]) if len(parts) > 4 else 0.0
                                object_width_m, object_height_m = _simheaven_object_dims(object_path)
                            except (IndexError, ValueError):
                                continue
                            parsed_objects.append(
                                {
                                    'lat': object_lat,
                                    'lon': object_lon,
                                    'heading': object_heading,
                                    'w_m': object_width_m,
                                    'h_m': object_height_m,
                                    'path': object_path,
                                }
                            )
                        elif line.startswith("BEGIN_POLYGON "):
                            parts = line.split()
                            current_polygon_is_building = False
                            current_winding = None
                            try:
                                polygon_index = int(parts[1])
                                current_polygon_is_building = _is_simheaven_building_polygon(
                                    polygon_defs[polygon_index]
                                )
                            except (IndexError, ValueError):
                                current_polygon_is_building = False
                        elif line == "BEGIN_WINDING" and current_polygon_is_building:
                            current_winding = []
                        elif line.startswith("POLYGON_POINT ") and current_winding is not None:
                            parts = line.split()
                            try:
                                current_winding.append((float(parts[2]), float(parts[1])))
                            except (IndexError, ValueError):
                                pass
                        elif line == "END_WINDING" and current_winding is not None:
                            if len(current_winding) >= 3:
                                parsed_polys.append(current_winding)
                            current_winding = None
                        elif line == "END_POLYGON":
                            current_polygon_is_building = False
                            current_winding = None
                return {"objects": parsed_objects, "polys": parsed_polys}

            parsed = PCACHE.load_or_build(
                cached_text_path,
                cache_dir,
                "dsf_parse",
                _parse,
                version="simheaven-buildings-v1",
            )
            objects.extend(parsed.get("objects", ()))
            polys.extend(parsed.get("polys", ()))

            print(
                f"  [simHeaven bld] {folder_name}: "
                f"+{len(objects) - n_obj0} objects  +{len(polys) - n_poly0} facade polys"
            )
        except Exception as exc:
            print(f"  [simHeaven bld] failed {dsf_path}: {exc}")

    return polys, objects


def _load_custom_scenery_building_exclusions(
    custom_scenery_dir,
    tile_lat,
    tile_lon,
    out_dsf,
    dsftool_path,
    cache_dir,
):
    """Parse active custom scenery objects/facades into exclusion geometry."""
    objects = []
    polys = []
    skipped_objects = 0
    dsf_name = os.path.basename(out_dsf or _tile_name_for_latlon(tile_lat, tile_lon))
    library_index = _active_custom_library_index(custom_scenery_dir)
    dsf_matches = find_active_custom_scenery_dsfs(
        custom_scenery_dir,
        dsf_name,
        skip_dsf_path=out_dsf,
    )

    for folder_name, dsf_path, package_dir in dsf_matches:
        n_obj0 = len(objects)
        n_poly0 = len(polys)
        skipped0 = skipped_objects
        try:
            cached_text_path = ensure_cached_dsf_text(
                dsf_path,
                dsftool_path,
                cache_dir,
                create_no_window=SEGFORMER._CREATE_NO_WINDOW,
            )
            object_defs = []
            polygon_defs = []
            current_polygon_is_facade = False
            current_winding = None

            with open(cached_text_path, "r", encoding="utf-8", errors="ignore") as text_file:
                for raw_line in text_file:
                    line = raw_line.strip()
                    if line.startswith("OBJECT_DEF "):
                        object_defs.append(line.split(" ", 1)[1])
                    elif line.startswith("POLYGON_DEF "):
                        polygon_defs.append(line.split(" ", 1)[1])
                    elif line.startswith("OBJECT "):
                        parts = line.split()
                        try:
                            object_index = int(parts[1])
                            object_path = object_defs[object_index]
                            object_lon = float(parts[2])
                            object_lat = float(parts[3])
                            object_heading = float(parts[4]) if len(parts) > 4 else 0.0
                        except (IndexError, ValueError):
                            skipped_objects += 1
                            continue
                        dims = _custom_object_dims(
                            object_path,
                            package_dir,
                            library_index,
                            cache_dir,
                        )
                        if dims is None:
                            skipped_objects += 1
                            continue
                        object_width_m, object_height_m = dims
                        objects.append(
                            {
                                'lat': object_lat,
                                'lon': object_lon,
                                'heading': object_heading,
                                'w_m': object_width_m,
                                'h_m': object_height_m,
                                'path': object_path,
                            }
                        )
                    elif line.startswith("BEGIN_POLYGON "):
                        parts = line.split()
                        current_polygon_is_facade = False
                        current_winding = None
                        try:
                            polygon_index = int(parts[1])
                            polygon_path = polygon_defs[polygon_index]
                            current_polygon_is_facade = (
                                polygon_path.replace('\\', '/').lower().endswith('.fac')
                            )
                        except (IndexError, ValueError):
                            current_polygon_is_facade = False
                    elif line == "BEGIN_WINDING" and current_polygon_is_facade:
                        current_winding = []
                    elif line.startswith("POLYGON_POINT ") and current_winding is not None:
                        parts = line.split()
                        try:
                            current_winding.append((float(parts[2]), float(parts[1])))
                        except (IndexError, ValueError):
                            pass
                    elif line == "END_WINDING" and current_winding is not None:
                        if len(current_winding) >= 3:
                            polys.append(current_winding)
                        current_winding = None
                    elif line == "END_POLYGON":
                        current_polygon_is_facade = False
                        current_winding = None

            print(
                f"  [custom scenery bld] {folder_name}: "
                f"+{len(objects) - n_obj0} objects  "
                f"+{len(polys) - n_poly0} facade polys  "
                f"skipped_objects={skipped_objects - skipped0}"
            )
        except Exception as exc:
            print(f"  [custom scenery bld] failed {dsf_path}: {exc}")

    return polys, objects, skipped_objects, len(dsf_matches)


def _tile_name_for_latlon(lat, lon):
    lat_i = int(lat)
    lon_i = int(lon)
    lat_s = f"{'+' if lat_i >= 0 else '-'}{abs(lat_i):02d}"
    lon_s = f"{'+' if lon_i >= 0 else '-'}{abs(lon_i):03d}"
    return f"{lat_s}{lon_s}.dsf"


def _prepare_simheaven_objects(objects):
    if not objects:
        return None
    arr = np.asarray(
        [
            (
                obj['lat'],
                obj['lon'],
                obj['heading'],
                obj['w_m'],
                obj['h_m'],
            )
            for obj in objects
        ],
        dtype=np.float32,
    )
    return {
        'lat': arr[:, 0],
        'lon': arr[:, 1],
        'heading': arr[:, 2],
        'w_m': arr[:, 3],
        'h_m': arr[:, 4],
    }


def _simheaven_objects_signature(objects):
    if not objects:
        return (0, 0.0)
    arr = np.asarray(
        [
            (
                obj['lat'],
                obj['lon'],
                obj['heading'],
                obj['w_m'],
                obj['h_m'],
            )
            for obj in objects
        ],
        dtype=np.float32,
    )
    checksum = float(np.sum(arr[:, 0] * 3.0 + arr[:, 1] * 5.0 +
                            arr[:, 2] * 0.01 + arr[:, 3] + arr[:, 4]))
    return (int(arr.shape[0]), round(checksum, 3))


def _simheaven_objects_for_bounds(obj_index, lat_n, lat_s, lon_w, lon_e, pad_deg=0.002):
    if not obj_index:
        return None
    keep = (
        (obj_index['lat'] >= lat_s - pad_deg) &
        (obj_index['lat'] <= lat_n + pad_deg) &
        (obj_index['lon'] >= lon_w - pad_deg) &
        (obj_index['lon'] <= lon_e + pad_deg)
    )
    if not bool(np.any(keep)):
        return None
    return {k: v[keep] for k, v in obj_index.items()}


def _rasterize_simheaven_objects(objects, lat_n, lat_s, lon_w, lon_e,
                                 img_h, img_w, m_per_px, margin_m=3.0):
    """Rasterize existing simHeaven building object footprints onto a mask."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not objects:
        return mask

    px = (objects['lon'] - lon_w) / (lon_e - lon_w) * img_w
    py = (lat_n - objects['lat']) / (lat_n - lat_s) * img_h
    margin_px = max(1.0, margin_m / max(m_per_px, 1e-6))

    for cx, cy, heading, w_m, h_m in zip(
            px, py, objects['heading'], objects['w_m'], objects['h_m']):
        w_px = max(2.0, float(w_m) / max(m_per_px, 1e-6)) + 2.0 * margin_px
        h_px = max(2.0, float(h_m) / max(m_per_px, 1e-6)) + 2.0 * margin_px
        pts = cv2.boxPoints(((float(cx), float(cy)), (w_px, h_px), float(heading)))
        pts = np.int32(pts)
        if (pts[:, 0].max() >= 0 and pts[:, 0].min() < img_w and
                pts[:, 1].max() >= 0 and pts[:, 1].min() < img_h):
            cv2.fillPoly(mask, [pts], 1)
    return mask


def _parse_excl_osm(cache_path, cache_dir=None):
    """Parse downloaded OSM into exclusions and residential guidance.

    Returns:
      exclusion_polys: closed ways that should block building placement
      rail_ways:       open railway ways rasterized like roads
      residential_polys:
                       closed ``landuse=residential`` polygons used to keep
                       small residential objects inside residential areas
    """
    import bz2, xml.etree.ElementTree as ET
    exclusion_polys = []
    rail_ways = []
    residential_polys = []
    if not os.path.exists(cache_path):
        return exclusion_polys, rail_ways, residential_polys
    def _parse():
        parsed_exclusion_polys = []
        parsed_rail_ways = []
        parsed_residential_polys = []
        with bz2.open(cache_path, 'rb') as f:
            root = ET.parse(f).getroot()
        nodes = {}
        for node in root.iter('node'):
            nodes[node.get('id')] = (float(node.get('lat')), float(node.get('lon')))
        for way in root.iter('way'):
            pts = [nodes[nd.get('ref')] for nd in way.iter('nd') if nd.get('ref') in nodes]
            if len(pts) < 2:
                continue
            is_closed = len(pts) >= 4 and pts[0] == pts[-1]
            tags = {tag.get('k'): tag.get('v') for tag in way.iter('tag')}
            railway_type = tags.get('railway')
            landuse_type = tags.get('landuse')

            if railway_type and not is_closed:
                parsed_rail_ways.append({'pts': pts, 'type': railway_type})
            elif is_closed and landuse_type == 'residential':
                parsed_residential_polys.append(pts)
            elif is_closed and (
                'building' in tags or
                'aeroway' in tags or
                landuse_type in {'railway', 'aeroway'}
            ):
                parsed_exclusion_polys.append(pts)
        return parsed_exclusion_polys, parsed_rail_ways, parsed_residential_polys

    return PCACHE.load_or_build(
        cache_path,
        cache_dir,
        "osm_parse",
        _parse,
        version="building-exclusions-v1",
    )


# ── Per-zone heading detection (Hough fallback) ────────────────────────────
def _zone_heading(img, bx, by, bw, bh, pad=80):
    """Detect dominant road/block heading for a single zone.

    Runs Canny + HoughLinesP on a padded subregion of the full DDS image
    centred on the zone's bounding box.  Padding captures nearby roads that
    bound the zone but lie outside it.

    Uses a 1° histogram with a weighted-mean peak — no bin snapping, so the
    result is continuous (e.g. 291.4° stays 291°, not 300° or 275°).

    Returns a heading in [0, 360°), or None if too few lines are detected.
    """
    h_img, w_img = img.shape[:2]
    x0 = max(0, bx - pad);  y0 = max(0, by - pad)
    x1 = min(w_img, bx + bw + pad);  y1 = min(h_img, by + bh + pad)
    region = img[y0:y1, x0:x1]

    rh, rw = region.shape[:2]
    if rh < 32 or rw < 32:
        return None

    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY) if region.ndim == 3 else region

    # Downsample only when the patch is large
    max_dim = max(gray.shape)
    if max_dim > HOUGH_PREVIEW_PX:
        sc = HOUGH_PREVIEW_PX / max_dim
        gray = cv2.resize(gray, (max(1, int(rw * sc)), max(1, int(rh * sc))))

    edges = cv2.Canny(gray, 40, 120, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                             threshold=20, minLineLength=10, maxLineGap=6)

    if lines is None or len(lines) < HOUGH_MIN_LINES:
        return None

    # Length-weighted angle histogram — 1° bins for continuous output
    angles  = np.empty(len(lines))
    weights = np.empty(len(lines))
    for i, seg in enumerate(lines):
        x1l, y1l, x2l, y2l = seg[0]
        angles[i]  = math.degrees(math.atan2(y2l - y1l, x2l - x1l)) % 180.0
        weights[i] = math.hypot(x2l - x1l, y2l - y1l)

    hist, edges_h = np.histogram(angles, bins=180, range=(0.0, 180.0),
                                 weights=weights)
    hist = np.convolve(hist, [0.2, 0.6, 0.2], mode='same')

    peak = int(np.argmax(hist))
    # Weighted mean over ±6 bins around peak → sub-degree precision, no snapping
    lo = max(0, peak - 6);  hi = min(180, peak + 7)
    bin_centres = (edges_h[:-1] + edges_h[1:]) / 2.0
    dominant_img_angle = float(np.average(bin_centres[lo:hi],
                                          weights=hist[lo:hi]))

    # Image-math angle → X-Plane compass heading (0 = N, CW)
    return (90.0 - dominant_img_angle) % 360.0


# ── Building object/facade pools by footprint size ────────────────────────────
BLD_CLASS_TINY_RESIDENTIAL = 1
BLD_CLASS_SMALL_RESIDENTIAL = 2
BLD_CLASS_COMPACT_RESIDENTIAL = 3
BLD_CLASS_MEDIUM = 4
BLD_CLASS_SMALL_APARTMENT = 5
BLD_CLASS_APARTMENT_BLOCK = 6
BLD_CLASS_LARGE = 7
BLD_CLASS_EXTRA_LARGE = 8
# Legacy internal name retained for compatibility with any existing callers.
BLD_CLASS_STANDARD_RESIDENTIAL = BLD_CLASS_MEDIUM
BLD_PLACEMENT_CLASSES = (
    BLD_CLASS_TINY_RESIDENTIAL,
    BLD_CLASS_SMALL_RESIDENTIAL,
    BLD_CLASS_COMPACT_RESIDENTIAL,
    BLD_CLASS_MEDIUM,
    BLD_CLASS_SMALL_APARTMENT,
    BLD_CLASS_APARTMENT_BLOCK,
    BLD_CLASS_LARGE,
    BLD_CLASS_EXTRA_LARGE,
)
BLD_CLASS_LABELS = {
    BLD_CLASS_TINY_RESIDENTIAL: "tiny residential",
    BLD_CLASS_SMALL_RESIDENTIAL: "small residential",
    BLD_CLASS_COMPACT_RESIDENTIAL: "compact residential",
    BLD_CLASS_MEDIUM: "medium footprint",
    BLD_CLASS_SMALL_APARTMENT: "small apartment",
    BLD_CLASS_APARTMENT_BLOCK: "apartment block",
    BLD_CLASS_LARGE: "large footprint",
    BLD_CLASS_EXTRA_LARGE: "extra-large footprint",
}

# Zone size thresholds (pixels²) after morphological cleanup at ZL16 native res.
# These classify building-zone clusters into progressively larger footprint
# candidates; the per-asset footprint fit remains the final placement gate.
ZONE_COMPACT_PX = 1_500
ZONE_MEDIUM_PX = 3_000
ZONE_SMALL_APARTMENT_PX = 10_000
ZONE_APARTMENT_BLOCK_PX = 30_000
ZL16_SMALL_ZONE_M2 = 7_500.0
ZL16_COMPACT_ZONE_M2 = 18_000.0
ZL16_MEDIUM_ZONE_M2 = 38_000.0
ZL16_SMALL_APARTMENT_ZONE_M2 = 80_000.0
ZL16_APARTMENT_BLOCK_ZONE_M2 = 180_000.0
ZL16_LARGE_ROOF_M2 = 2_200.0
ZL16_EXTRA_LARGE_ROOF_M2 = 7_000.0
ZL16_APARTMENT_ROOF_M2 = 900.0
ZL16_FINE_GRAIN_FRAGS_PER_HA = 14.0
ZL16_LOCAL_HEADING_ZONE_M2 = 55_000.0

# Asset footprint thresholds in square metres. Height is intentionally ignored:
# tall objects are acceptable when the footprint is modest.
ASSET_TINY_RESIDENTIAL_MAX_M2 = 90.0
ASSET_SMALL_RESIDENTIAL_MAX_M2 = 170.0
ASSET_COMPACT_RESIDENTIAL_MAX_M2 = 270.0
ASSET_MEDIUM_MAX_M2 = 450.0
ASSET_MEDIUM_MAX_SIDE_M = 28.0
ASSET_SMALL_APARTMENT_MAX_M2 = 850.0
ASSET_SMALL_APARTMENT_MAX_SIDE_M = 45.0
ASSET_APARTMENT_BLOCK_MAX_M2 = 1_650.0
ASSET_APARTMENT_BLOCK_MAX_SIDE_M = 55.0
ASSET_LARGE_MAX_M2 = 7_000.0
ASSET_LARGE_MAX_SIDE_M = 100.0

# Heading alignment — grid-based Hough.
# Each DDS is divided into HEADING_GRID_N × HEADING_GRID_N cells.  Hough line
# detection runs on each cell independently, so headings vary at city-block
# granularity (~200 m per cell at ZL16 with N=8) rather than per large blob.
# NaN cells (too few lines) are filled from neighbours; remaining NaN fall back
# to the tile-wide median, then random.
# Jitter: small per-building random offset so the result isn't perfectly rigid.
HEADING_GRID_N       = 16     # NxN heading grid per DDS  (16 → ~350 m cells at ZL16)
HEADING_JITTER_DEG   = 2.0
HOUGH_MIN_LINES      = 8      # minimum Hough line segments to trust the result
HOUGH_PREVIEW_PX     = 512    # resize cell to this before Hough (speed vs precision)

# Minimum clearance radius (metres) between building centres, by class.
# After placing a building its centre is marked with a filled circle on an
# occupancy mask; future candidates within that radius are rejected.
# This prevents geometric overlap without probabilistic thinning:
#   compact/medium: suburban houses, fine packing
#   apartments: larger clearance but still footprint-fit constrained
#   large: conservative fallback for broad warehouse/industrial objects
OBJ_CLEARANCE_M = {
    BLD_CLASS_TINY_RESIDENTIAL: 7.0,
    BLD_CLASS_SMALL_RESIDENTIAL: 10.0,
    BLD_CLASS_COMPACT_RESIDENTIAL: 14.0,
    BLD_CLASS_MEDIUM: 20.0,
    BLD_CLASS_SMALL_APARTMENT: 28.0,
    BLD_CLASS_APARTMENT_BLOCK: 36.0,
    BLD_CLASS_LARGE: 55.0,
    BLD_CLASS_EXTRA_LARGE: 80.0,
}  # legacy fallback for unknown footprints
PLACE_UNKNOWN_OBJECTS = False

# Per-object footprint dimensions (width × depth in metres).
# Width = dimension perpendicular to facing direction; depth = along facing.
# Used for rectangular collision detection and 90° rotation fallback.
# These values were measured from the actual SFD Global OBJ plan footprints.
# Buildings not listed here fall back to OBJ_CLEARANCE_M circular exclusion.
OBJ_DIMS: dict = {
    # ── Asia suburban (measured from Asia/Suburban_N.obj VT vertices) ──────────
    "SFD_Global/Asia/Suburban_1.obj":  (10.06, 15.54),
    "SFD_Global/Asia/Suburban_2.obj":  (11.6,  9.4),
    "SFD_Global/Asia/Suburban_3.obj":  (11.13, 9.36),
    "SFD_Global/Asia/Suburban_4.obj":  (10.79, 8.86),
    "SFD_Global/Asia/Suburban_5.obj":  (11.96, 9.93),
    "SFD_Global/Asia/Suburban_6.obj":  (12.74, 9.87),
    "SFD_Global/Asia/Suburban_7.obj":  (11.47, 9.12),
    "SFD_Global/Asia/Suburban_8.obj":  (9.89,  9.19),
    "SFD_Global/Asia/Suburban_9.obj":  (11.0,  9.7),
    "SFD_Global/Asia/Suburban_10.obj": (9.67,  9.03),
    # ── Asia suburban south (measured from Asia/Suburban_South_N.obj) ────────
    "SFD_Global/Asia/Suburban_South_1.obj": (13.56, 12.29),
    "SFD_Global/Asia/Suburban_South_3.obj": (19.42, 12.29),
    "SFD_Global/Asia/Suburban_South_7.obj": (20.49, 13.09),
    "SFD_Global/Asia/Suburban_South_8.obj": (19.76, 10.31),
    # ── Asia apartment blocks (library aliases Apartment_N.obj → Apartment_N_1/_2) ──
    "SFD_Global/Asia/Apartment_1.obj": (26.66, 9.5),
    "SFD_Global/Asia/Apartment_2.obj": (60.0,  10.37),
    "SFD_Global/Asia/Apartment_3.obj": (35.0,  9.12),
    "SFD_Global/Buildings/Apartment_30m_1.obj": (34.5, 16.1),
    "SFD_Global/Buildings/Apartment_30m_2.obj": (18.0, 27.0),
    "SFD_Global/Buildings/Apartment_30m_3.obj": (28.9, 16.7),
    "SFD_Global/Buildings/Apartment_30m_4.obj": (28.0, 18.1),
    # ── Asia industrial (library aliases Industry_NxM.obj → colour variants) ──
    "SFD_Global/Asia/Industry_20x40.obj": (20.48, 38.89),
    "SFD_Global/Asia/Industry_30x40.obj": (32.06, 40.66),
    "SFD_Global/Asia/Industry_50x30.obj": (51.56, 31.87),
    "SFD_Global/Asia/Industry_40x60.obj": (56.91, 68.2),
    "SFD_Global/Asia/Industry_60x50.obj": (31.91, 31.42),
    "SFD_Global/Asia/Industry_70x90.obj": (87.2, 16.52),
    "SFD_Global/Asia/Industry_150x80.obj": (81.42, 56.23),
    # ── Scandinavia ───────────────────────────────────────────────────────────
    **{f"SFD_Global/Scandinavia/Residential/Suburban_{i}.obj": d
       for i, d in enumerate([
           (15.37, 10.0), (9.66, 10.2),  (12.4,  12.33), (12.88, 9.93),
           (16.89,  9.03), (15.65,  9.03), (14.8,  11.6),  (9.77,  14.5),
       ], 1)},
    # ── Africa ───────────────────────────────────────────────────────────────
    **{f"SFD_Global/Africa/Residential/Suburban_{i}.obj": d
       for i, d in enumerate([
           (12.59, 13.64), (14.59, 14.11), (13.42, 16.52), (13.55, 19.73),
           (12.14, 13.83), (17.87, 13.36), (15.79, 13.28), (16.75, 17.75),
       ], 1)},
    # ── Mediterranean suburban ────────────────────────────────────────────────
    **{f"SFD_Global/Med/Residential/Suburban_{i}.obj": d
       for i, d in enumerate([
           (12.47, 15.52), (24.62, 13.87), (15.97, 22.93), (15.0,  13.02),
           (10.34, 14.55), (9.91,  12.0),  (13.01, 12.62), (13.06,  9.37),
       ], 1)},
    # ── Mediterranean apartments ──────────────────────────────────────────────
    **{f"SFD_Global/Med/Residential/Apartment_North_{i}.obj": d
       for i, d in enumerate([
           (20.93, 12.51), (24.08, 11.64), (18.33, 11.07), (16.19, 11.44),
           (16.79, 11.6),  (13.0,  15.6),  (23.66, 13.19), (12.0,  18.07),
       ], 1)},
    # ── Mediterranean urban residential row blocks ───────────────────────────
    "SFD_Global/Med/Residential/Urban_Mid_7m.obj": (7.0, 12.0),
    "SFD_Global/Med/Residential/Urban_Mid_9m.obj": (9.0, 12.0),
    "SFD_Global/Med/Residential/Urban_Mid_12m.obj": (12.0, 12.0),
    "SFD_Global/Med/Residential/Urban_Mid_15m.obj": (15.0, 12.0),
    "SFD_Global/Med/Residential/Urban_Mid_18m.obj": (18.0, 12.0),
    "SFD_Global/Med/Residential/Urban_Mid_22m.obj": (22.0, 12.0),
    "SFD_Global/Med/Residential/Urban_Mid_25m.obj": (25.0, 12.0),
    "SFD_Global/Med/Residential/Urban_Mid_30m.obj": (30.0, 12.0),
    "SFD_Global/Med/Residential/Urban_Mid_90.obj": (13.0, 13.0),
    "SFD_Global/Med/Residential/Urban_Mid_-90.obj": (13.0, 13.0),
    "SFD_Global/Med/Residential/Urban_Mid_Corner_1L.obj": (17.0, 12.0),
    "SFD_Global/Med/Residential/Urban_Mid_Corner_1R.obj": (12.0, 11.9),
    "SFD_Global/Med/Residential/Urban_Mid_Corner_2L.obj": (14.0, 12.0),
    "SFD_Global/Med/Residential/Urban_Mid_Corner_2R.obj": (12.0, 8.0),
    "SFD_Global/Med/Residential/Urban_Mid_Corner_3L.obj": (19.9, 12.0),
    "SFD_Global/Med/Residential/Urban_Mid_Corner_3R.obj": (10.5, 14.7),
    # ── New England ──────────────────────────────────────────────────────────
    **{f"SFD_Global/New_England/Residential/Suburban_{i}.obj": d
       for i, d in enumerate([
           (9.69,  14.13), (8.98, 12.79), (8.0,  16.63), (10.09, 12.22),
           (10.76,  7.29), (9.21, 16.12), (9.56, 14.8),  (8.12,  12.8),
       ], 1)},
    "SFD_Global/New_England/Residential/Garage.obj": (4.6, 6.6),
    # ── Small accessory buildings ────────────────────────────────────────────
    "SFD_Global/Asia/Carport_1.obj": (2.5, 4.9),
    "SFD_Global/Asia/Carport_2.obj": (3.0, 5.1),
    "SFD_Global/Asia/Shed_1.obj": (1.5, 2.5),
    "SFD_Global/Australia/Shed.obj": (8.9, 6.0),
    "SFD_Global/Australia/Carport.obj": (4.3, 5.9),
    # ── US West Coast ─────────────────────────────────────────────────────────
    **{f"SFD_Global/US_West_Coast/Suburban_{i}.obj": d
       for i, d in enumerate([
           (15.17, 15.83), (13.69, 16.03), (14.24, 17.67), (14.38, 13.4),
           (14.59, 17.65), (16.54, 17.9),  (14.89, 15.8),  (16.34, 16.14),
       ], 1)},
    # ── South America suburban ────────────────────────────────────────────────
    **{f"SFD_Global/South_America/Suburban_{i}.obj": d
       for i, d in enumerate([
           (7.56, 16.55), (6.42, 12.08), (7.98, 16.6),  (7.6,  17.35),
           (7.6,  18.34), (7.6,  20.73), (7.96, 12.39), (7.6,  14.2),
           (6.36, 11.33), (8.47, 20.19),
       ], 1)},
    # ── South America medium ──────────────────────────────────────────────────
    **{f"SFD_Global/South_America/Med_{i}.obj": d
       for i, d in enumerate([
           (12.0, 21.0), (12.0, 16.5),  (12.0,  16.5),  (12.08, 15.28),
           (17.7, 14.92), (10.32, 16.59), (14.06, 21.94), (12.0,  16.5),
       ], 1)},
}

# Actual local footprint bounds (xmin, xmax, zmin, zmax in metres) measured
# from SFD Global OBJ vertices. These are relative to the object origin used
# by X-Plane when placing the object, so they capture off-centre footprints
# that a width/height-only collision test misses.
OBJ_FOOTPRINTS: dict = {
    "SFD_Global/Africa/Residential/Suburban_4.obj": (-6.77, 6.77, -9.20, 10.52),
    "SFD_Global/Africa/Residential/Suburban_6.obj": (-8.93, 8.93, -6.68, 6.68),
    "SFD_Global/Asia/Apartment_3.obj": (-17.50, 17.50, -4.21, 4.91),
    "SFD_Global/Asia/Industry_20x40.obj": (-10.13, 10.34, -19.44, 19.44),
    "SFD_Global/Asia/Industry_30x40.obj": (-15.78, 16.28, -20.66, 19.99),
    "SFD_Global/Asia/Industry_60x50.obj": (-15.35, 47.74, -33.44, 15.73),
    "SFD_Global/Asia/Industry_70x90.obj": (-36.76, 36.76, -43.60, 43.60),
    "SFD_Global/Asia/Industry_150x80.obj": (-75.30, 74.65, -32.41, 48.96),
    "SFD_Global/Asia/Suburban_3.obj": (-5.57, 5.57, -4.68, 4.68),
    "SFD_Global/Asia/Suburban_10.obj": (-4.83, 4.83, -4.52, 4.52),
    "SFD_Global/Asia/Suburban_South_1.obj": (-6.78, 6.78, -5.27, 7.02),
    "SFD_Global/Asia/Suburban_South_3.obj": (-7.50, 4.79, -8.38, 11.04),
    "SFD_Global/Asia/Suburban_South_7.obj": (-7.33, 5.78, -8.42, 12.08),
    "SFD_Global/Asia/Suburban_South_8.obj": (-9.88, 9.88, -5.16, 5.16),
    "SFD_Global/Med/Residential/Apartment_North_1.obj": (-10.47, 10.47, -6.00, 6.51),
    "SFD_Global/Med/Residential/Apartment_North_2.obj": (-12.04, 12.04, -5.82, 5.82),
    "SFD_Global/Med/Residential/Apartment_North_3.obj": (-9.16, 9.16, -5.25, 5.82),
    "SFD_Global/Med/Residential/Apartment_North_4.obj": (-8.09, 8.09, -5.50, 5.94),
    "SFD_Global/Med/Residential/Apartment_North_5.obj": (-8.39, 8.39, -5.80, 5.80),
    "SFD_Global/Med/Residential/Suburban_3.obj": (-7.99, 7.99, -14.00, 8.93),
    "SFD_Global/Med/Residential/Suburban_5.obj": (-5.17, 5.17, -7.27, 7.27),
    "SFD_Global/Med/Residential/Suburban_6.obj": (-4.95, 4.95, -6.00, 6.00),
    "SFD_Global/Med/Residential/Suburban_8.obj": (-6.53, 6.53, -4.69, 4.69),
    "SFD_Global/New_England/Residential/Suburban_1.obj": (-4.85, 4.85, -6.90, 7.23),
    "SFD_Global/New_England/Residential/Suburban_2.obj": (-4.49, 4.49, -6.39, 6.39),
    "SFD_Global/New_England/Residential/Suburban_5.obj": (-5.38, 5.38, -3.65, 3.65),
    "SFD_Global/New_England/Residential/Suburban_6.obj": (-4.61, 4.61, -6.66, 9.47),
    "SFD_Global/Scandinavia/Residential/Suburban_3.obj": (-6.20, 6.20, -6.53, 5.80),
    "SFD_Global/Scandinavia/Residential/Suburban_5.obj": (-8.45, 8.45, -4.52, 4.52),
    "SFD_Global/Scandinavia/Residential/Suburban_7.obj": (-7.40, 7.40, -5.80, 5.80),
    "SFD_Global/South_America/Suburban_1.obj": (-3.78, 3.78, -8.28, 8.28),
    "SFD_Global/South_America/Suburban_2.obj": (-3.21, 3.21, -6.04, 6.04),
    "SFD_Global/South_America/Suburban_3.obj": (-3.99, 3.99, -5.10, 11.50),
    "SFD_Global/South_America/Suburban_6.obj": (-3.80, 3.80, -10.36, 10.36),
    "SFD_Global/South_America/Suburban_7.obj": (-3.99, 3.97, -6.08, 6.31),
    "SFD_Global/South_America/Suburban_8.obj": (-3.80, 3.80, -7.10, 7.10),
    "SFD_Global/South_America/Suburban_9.obj": (-3.19, 3.17, -5.67, 5.67),
    "SFD_Global/South_America/Suburban_10.obj": (-4.00, 4.47, -8.80, 11.39),
    "SFD_Global/South_America/Med_1.obj": (-6.00, 6.00, -12.88, 8.12),
    "SFD_Global/South_America/Med_2.obj": (-6.00, 6.00, -8.25, 8.25),
    "SFD_Global/South_America/Med_3.obj": (-6.00, 6.00, -8.16, 8.34),
    "SFD_Global/South_America/Med_6.obj": (-5.16, 5.16, -8.30, 8.30),
    "SFD_Global/South_America/Med_7.obj": (-7.71, 6.35, -10.97, 10.97),
    "SFD_Global/South_America/Med_8.obj": (-6.00, 6.00, -8.25, 8.25),
    "SFD_Global/US_West_Coast/Suburban_4.obj": (-7.19, 7.18, -5.87, 7.53),
}
PLACEMENT_MARGIN_M = 6.0   # legacy fallback margin if mark bounds are missing
FOOTPRINT_PAD_M = 4.0      # expand known footprints before fit/mark to reduce overlaps
MAX_GENERATED_BUILDING_HEIGHT_M = 24.0
BLD_PLACEMENT_CACHE_VERSION = 60
BLD_PLACEMENT_FAST_CACHE_VERSION = 62
BLD_MAX_CANDIDATES_PER_DDS = 180_000  # 0 = exhaustive search; override with O4_SFR_BLD_MAX_CANDIDATES.
BLD_SMART_GAP_FILL_ENABLED = False

CURATED_EXTRA_BUILDING_LIBRARIES = {
    "opensceneryx": {
        "label": "OpenSceneryX",
        "package_patterns": ("opensceneryx",),
        "virtual_prefixes": ("opensceneryx/",),
    },
    "world-models": {
        "label": "world-models",
        "package_patterns": ("world-models",),
        "virtual_prefixes": ("objects/", "world-models/"),
    },
    "cdb-library": {
        "label": "CDB-Library",
        "package_patterns": ("cdb-library",),
        "virtual_prefixes": ("cdb-library/",),
    },
    "handyobjects": {
        "label": "The Handy Objects Library",
        "package_patterns": ("the_handy_objects_library", "handy objects"),
        "virtual_prefixes": ("handyobjects/",),
    },
    "r2-library": {
        "label": "R2_Library",
        "package_patterns": ("r2_library", "r2-library"),
        "virtual_prefixes": ("r2_library/",),
    },
    "ff-library": {
        "label": "FFLibrary",
        "package_patterns": ("ff_library", "fflibrary", "ff library"),
        "virtual_prefixes": ("ff_library/", "fflibrary/"),
        # German/DACH residential & VFR facades.
        "regions": ("europe",),
    },
    "ruscenery": {
        "label": "RuScenery",
        "package_patterns": ("ruscenery", "ru_scenery", "ru-scenery"),
        "virtual_prefixes": ("ruscenery/", "ru_scenery/"),
        # Russian / Eastern European blocks and houses.
        "regions": ("europe",),
    },
    "bs2001": {
        "label": "BS2001 Object Library",
        "package_patterns": ("bs2001",),
        "virtual_prefixes": ("bs2001/",),
        # European airport/road objects with some residential clutter.
        "regions": ("europe",),
    },
    "ar-library": {
        "label": "AR_Library",
        "package_patterns": ("ar_library", "ar-library"),
        "virtual_prefixes": ("ar_library/",),
        # Argentine residential & commercial assets.
        "regions": ("south_america",),
    },
    "ob-library": {
        "label": "OB_Library",
        "package_patterns": ("ob_library", "ob-library"),
        "virtual_prefixes": ("ob_library/",),
        # Generic houses, hangars, towers — global fallback bucket.
        "regions": ("generic",),
    },
    "zdp-library": {
        "label": "ZDP Library",
        "package_patterns": ("zdp_library", "zdp-library", "zdp"),
        "virtual_prefixes": ("zdp/", "zdp_library/"),
        # Mixed Americas / Europe shared assets used with MisterX sceneries.
        "regions": ("north_america", "europe"),
    },
    "misterx": {
        "label": "MisterX Library",
        "package_patterns": ("misterx_library", "misterx", "mister_x"),
        "virtual_prefixes": ("misterx/", "misterx_library/"),
        # Mostly airport/aircraft; only residential subset survives include/exclude tokens.
        "regions": ("generic",),
    },
    "vectors-to-final": {
        "label": "Vectors to Final",
        "package_patterns": ("vectors_to_final", "vectors-to-final"),
        "virtual_prefixes": ("vectors_to_final/", "v2f/"),
        # Niche Cold-War / vintage thematic — primarily European bases.
        "regions": ("europe",),
    },
    "o4sfr": {
        "label": "O4SFR_Library",
        "package_patterns": ("o4sfr_library", "o4sfr"),
        "virtual_prefixes": ("o4sfr/",),
        # Our shipped library — per-asset region encoded in path token /<region>/.
    },
}

OPTIONAL_LIBRARY_INCLUDE_TOKENS = (
    "/house",
    "/houses/",
    "/residential/",
    "/commercial/",
    "/industrial/",
    "/industry",
    "/warehouse",
    "/buildings/houses/",
    "/buildings/residential/",
    "/buildings/commercial/",
    "/buildings/industrial/",
    "/terraced/",
    "/apartments/",
    "/apartment",
)

GENERIC_BUILDING_EXCLUDE_TOKENS = (
    "/airport",
    "aircraft",
    "antenna",
    "barrier",
    "billboard",
    "boat",
    "bridge",
    "car_",
    "/cars/",
    "cathedral",
    "chapel",
    "chimney",
    "church",
    "clinic",
    "container",
    "cooling_tower",
    "chladici_vez",
    "crane",
    "dock",
    "dumpster",
    "fence",
    "fuel",
    "gas_",
    "hangar",
    "hospital",
    "hotel",
    "jetty",
    "jetway",
    "komin",
    "/kominy/",
    "lighthouse",
    "mast",
    "marine",
    "monument",
    "nadrz",
    "/nadrze/",
    "oil_",
    "parking",
    "petrol",
    "pipeline",
    "propane",
    "radio_",
    "reactor",
    "reaktor",
    "recreational",
    "school",
    "ship",
    "sign",
    "silo",
    "smokestack",
    "smoke_stack",
    "smoke-stack",
    "sports",
    "stadium",
    "storage_tank",
    "/store",
    "store_",
    "supermarket",
    "/tank",
    "tank_",
    "terminal",
    "tower",
    "train",
    "trash",
    "utility",
    "vehicle",
    "windmill",
)

OPTIONAL_LIBRARY_SPECIAL_LANDMARK_TOKENS = (
    "/wind_turbine",
    "wind_turbine",
    "wind-turbine",
    "windmill",
    "velektrarna",
    "solar_panel",
    "solar-panel",
    "/solar/",
    "/hydro/",
    "vodojem",
    "/bts/",
    "btska",
    "antenna",
    "mast",
    "power_station",
    "power-station",
    "powerplant",
    "power_plant",
    "gasometer",
    "smokestack",
    "smoke_stack",
    "smoke-stack",
    "/reklamy/",
    "billboard",
    "advert",
)

OPTIONAL_ASSET_REGION_ALIASES = {
    "generic": {"generic"},
    "europe": {"europe", "scandinavia", "mediterranean"},
    "scandinavia": {"europe", "scandinavia"},
    "mediterranean": {"europe", "mediterranean"},
    "north_america": {"north_america", "north_america_ne", "north_america_west"},
    "north_america_ne": {"north_america", "north_america_ne"},
    "north_america_west": {"north_america", "north_america_west"},
    "south_america": {"south_america"},
    "asia": {"asia", "se_asia"},
    "se_asia": {"asia", "se_asia"},
    "africa": {"africa"},
    "australia_oceania": {"australia_oceania"},
}

YOLO_OBB_CACHE_VERSION = 3
YOLO_ANALYSIS_CACHE_VERSION = 1
YOLO_ANALYSIS_TARGET_ZL = 16
DEFAULT_YOLO_OBB_CHECKPOINT = (
    r"H:\model_training\runs\yolo_obb_v1\weights\visual_candidate_step_12000.pt"
)
DEFAULT_YOLO_OBB_IMGSZ = 512
DEFAULT_YOLO_OBB_STRIDE = 512
DEFAULT_YOLO_OBB_CONF = 0.18
DEFAULT_YOLO_OBB_IOU = 0.5
DEFAULT_YOLO_OBB_MAX_DET = 100000
DEFAULT_YOLO_OBB_BATCH = 1
YOLO_GUIDANCE_MAX_DISTANCE_M = 70.0
YOLO_TEMPLATE_MAX_CANDIDATES_PER_ZONE = 5000
YOLO_TEMPLATE_HEADING_TOL_DEG = 10.0
YOLO_TEMPLATE_SHAPE_REL_TOL = 0.20
YOLO_TEMPLATE_SHAPE_ABS_TOL_PX = 4.0
YOLO_OBJECT_MIN_COVERAGE = 0.80
YOLO_OBJECT_MIN_COVERAGE_BY_CLASS = {
    BLD_CLASS_TINY_RESIDENTIAL: 0.50,
    BLD_CLASS_SMALL_RESIDENTIAL: 0.50,
    BLD_CLASS_COMPACT_RESIDENTIAL: 0.65,
}
# Object footprints must lie inside the YOLO detection polygon. Containment is
# strict apart from a small real-world quantisation margin: a thin band of width
# ``YOLO_OBJECT_OUTLINE_MARGIN_M`` *metres* around the footprint perimeter may fall
# outside the polygon, absorbing integer pixel/anchor rounding. The margin is in
# metres (not pixels) so it is resolution-independent -- a fixed pixel band would
# be several metres of overhang at coarse analysis scales (~2 m/px at ZL16). An
# object can never stick out by a real fraction of its area regardless of size.
# The legitimate "fit a *smaller* object when none matches the polygon exactly"
# behaviour is governed by ``YOLO_OBJECT_MIN_COVERAGE`` (see above), not by
# relaxing containment.
YOLO_OBJECT_OUTLINE_MARGIN_M = 0.5


def _building_fill_modes(smart_gap_fill):
    allow_inferred_fill = bool(smart_gap_fill)
    run_legacy_gap_fill = bool(allow_inferred_fill and BLD_SMART_GAP_FILL_ENABLED)
    return allow_inferred_fill, run_legacy_gap_fill


# Road exclusion is metre-based with a modest pixel floor so higher-ZL tiles
# don't get an overly aggressive street buffer.
ROAD_CENTERLINE_WIDTH_M = 3.0
ROAD_EXTRA_BUFFER_M = 0.0
ROAD_WIDTH_PX_MIN = 6
ROAD_DILATE_PX_MIN = 0

# Small residential objects should stay inside residential-looking fabric
# instead of spilling into industrial/commercial zones. Prefer explicit OSM
# ``landuse=residential`` polygons and fall back to a buffered mask around
# neighborhood street classes when landuse mapping is sparse.
RESIDENTIAL_HIGHWAY_TYPES = {
    'living_street',
    'residential',
    'service',
    'unclassified',
}
RESIDENTIAL_FALLBACK_BUFFER_M = 45.0
RESIDENTIAL_FALLBACK_BUFFER_PX_MIN = 16

DEFAULT_FACADE_BOUNDS = {
    BLD_CLASS_TINY_RESIDENTIAL: (-4.0, 4.0, -4.0, 4.0),
    BLD_CLASS_SMALL_RESIDENTIAL: (-5.5, 5.5, -6.0, 6.0),
    BLD_CLASS_COMPACT_RESIDENTIAL: (-7.0, 7.0, -7.0, 7.0),
    BLD_CLASS_MEDIUM: (-10.0, 10.0, -8.0, 8.0),
    BLD_CLASS_SMALL_APARTMENT: (-14.0, 14.0, -9.0, 9.0),
    BLD_CLASS_APARTMENT_BLOCK: (-18.0, 18.0, -12.0, 12.0),
    BLD_CLASS_LARGE: (-45.0, 45.0, -30.0, 30.0),
    BLD_CLASS_EXTRA_LARGE: (-75.0, 75.0, -45.0, 45.0),
}
DEFAULT_FACADE_PATHS = {
    BLD_CLASS_TINY_RESIDENTIAL: SEGFORMER._FAC_DEFS["medium"],
    BLD_CLASS_SMALL_RESIDENTIAL: SEGFORMER._FAC_DEFS["medium"],
    BLD_CLASS_COMPACT_RESIDENTIAL: SEGFORMER._FAC_DEFS["medium"],
    BLD_CLASS_MEDIUM: SEGFORMER._FAC_DEFS["medium"],
    BLD_CLASS_SMALL_APARTMENT: SEGFORMER._FAC_DEFS["medium"],
    BLD_CLASS_APARTMENT_BLOCK: SEGFORMER._FAC_DEFS["medium"],
    BLD_CLASS_LARGE: SEGFORMER._FAC_DEFS["large"],
    BLD_CLASS_EXTRA_LARGE: SEGFORMER._FAC_DEFS["large"],
}
DEFAULT_FACADE_VARIANTS_BY_CLASS = {
    BLD_CLASS_TINY_RESIDENTIAL: (
        "lib/buildings/facades/generic/low_modern_01.fac",
        "lib/buildings/facades/commercial/low_commercial_01.fac",
        "lib/buildings/facades/commercial/low_commercial_02.fac",
    ),
    BLD_CLASS_SMALL_RESIDENTIAL: (
        "lib/buildings/facades/generic/low_modern_01.fac",
        "lib/buildings/facades/commercial/low_commercial_01.fac",
        "lib/buildings/facades/commercial/low_commercial_02.fac",
        "lib/buildings/facades/commercial/low_commercial_04.fac",
    ),
    BLD_CLASS_COMPACT_RESIDENTIAL: (
        "lib/buildings/facades/generic/low_modern_01.fac",
        "lib/buildings/facades/commercial/low_commercial_01.fac",
        "lib/buildings/facades/commercial/low_commercial_02.fac",
        "lib/buildings/facades/commercial/low_commercial_03.fac",
        "lib/buildings/facades/commercial/low_commercial_04.fac",
        "lib/buildings/facades/commercial/low_commercial_05.fac",
    ),
    BLD_CLASS_MEDIUM: (
        "lib/buildings/facades/generic/mid_classic_01.fac",
        "lib/buildings/facades/generic/mid_classic_02.fac",
        "lib/buildings/facades/generic/mid_modern_01.fac",
        "lib/buildings/facades/generic/mid_modern_02.fac",
        "lib/buildings/facades/generic/mid_modern_03.fac",
        "lib/buildings/facades/generic/mid_modern_04.fac",
        "lib/buildings/facades/generic/mid_modern_05.fac",
        "lib/buildings/facades/commercial/low_commercial_03.fac",
        "lib/buildings/facades/commercial/low_commercial_04.fac",
        "lib/buildings/facades/commercial/low_commercial_05.fac",
        "lib/buildings/facades/commercial/low_commercial_06.fac",
        "lib/buildings/facades/commercial/low_commercial_07.fac",
        "lib/buildings/facades/commercial/low_commercial_08.fac",
    ),
    BLD_CLASS_SMALL_APARTMENT: (
        "lib/buildings/facades/generic/mid_classic_01.fac",
        "lib/buildings/facades/generic/mid_classic_02.fac",
        "lib/buildings/facades/generic/mid_modern_01.fac",
        "lib/buildings/facades/generic/mid_modern_02.fac",
        "lib/buildings/facades/generic/mid_modern_03.fac",
        "lib/buildings/facades/generic/mid_modern_04.fac",
        "lib/buildings/facades/generic/mid_modern_05.fac",
        "lib/buildings/facades/generic/high_classic_01.fac",
        "lib/buildings/facades/generic/high_classic_02.fac",
        "lib/buildings/facades/generic/high_glass_01.fac",
        "lib/buildings/facades/generic/high_glass_02.fac",
        "lib/buildings/facades/generic/high_modern_01.fac",
        "lib/buildings/facades/generic/high_modern_02.fac",
        "lib/buildings/facades/generic/high_modern_03.fac",
    ),
    BLD_CLASS_APARTMENT_BLOCK: (
        "lib/buildings/facades/generic/high_classic_01.fac",
        "lib/buildings/facades/generic/high_classic_02.fac",
        "lib/buildings/facades/generic/high_glass_01.fac",
        "lib/buildings/facades/generic/high_glass_02.fac",
        "lib/buildings/facades/generic/high_glass_03.fac",
        "lib/buildings/facades/generic/high_metallic_01.fac",
        "lib/buildings/facades/generic/high_modern_01.fac",
        "lib/buildings/facades/generic/high_modern_02.fac",
        "lib/buildings/facades/generic/high_modern_03.fac",
        "lib/buildings/facades/generic/high_modern_04.fac",
        "lib/buildings/facades/generic/high_modern_05.fac",
        "lib/buildings/facades/generic/high_modern_06.fac",
        "lib/buildings/facades/generic/high_modern_07.fac",
        "lib/buildings/facades/generic/high_universal_01.fac",
        "lib/buildings/facades/generic/high_universal_02.fac",
    ),
    BLD_CLASS_LARGE: (
        "lib/buildings/facades/commercial/low_commercial_06.fac",
        "lib/buildings/facades/commercial/low_commercial_07.fac",
        "lib/buildings/facades/commercial/low_commercial_08.fac",
        "lib/buildings/facades/industrial/warehouse_01_45x45.fac",
        "lib/buildings/facades/industrial/warehouse_02_45x45.fac",
        "lib/buildings/facades/industrial/warehouse_03_60x60.fac",
        "lib/buildings/facades/industrial/warehouse_04_60x60.fac",
        "lib/buildings/facades/industrial/warehouse_05_60x60.fac",
        "lib/buildings/facades/industrial/warehouse_06_90x40.fac",
        "lib/buildings/facades/industrial/warehouse_07_90x40.fac",
        "lib/buildings/facades/industrial/warehouse_08_90x90.fac",
        "lib/buildings/facades/industrial/warehouse_09_90x90.fac",
        "lib/buildings/facades/industrial/warehouse_10_90x90.fac",
    ),
    BLD_CLASS_EXTRA_LARGE: (
        "lib/buildings/facades/industrial/warehouse_06_90x40.fac",
        "lib/buildings/facades/industrial/warehouse_07_90x40.fac",
        "lib/buildings/facades/industrial/warehouse_08_90x90.fac",
        "lib/buildings/facades/industrial/warehouse_09_90x90.fac",
        "lib/buildings/facades/industrial/warehouse_10_90x90.fac",
    ),
}
DEFAULT_FACADE_HEIGHT_M = {
    BLD_CLASS_TINY_RESIDENTIAL: 3.5,
    BLD_CLASS_SMALL_RESIDENTIAL: 4.0,
    BLD_CLASS_COMPACT_RESIDENTIAL: 4.0,
    BLD_CLASS_MEDIUM: 7.0,
    BLD_CLASS_SMALL_APARTMENT: 9.0,
    BLD_CLASS_APARTMENT_BLOCK: 12.0,
    # LARGE/EXTRA_LARGE are dominantly warehouses/distribution centers in the
    # training data — big footprint but single-story / low-rise. Heights here
    # reflect typical warehouse ceilings, NOT a linear scale from footprint.
    BLD_CLASS_LARGE: 8.0,
    BLD_CLASS_EXTRA_LARGE: 10.0,
}
YOLO_HEIGHT_BINS_M = (
    1.0, 2.0, 3.0, 4.0, 5.0,
    6.0, 7.0, 8.0, 9.0, 10.0,
    12.0, 15.0, 18.0, 21.0, 24.0,
    30.0, 40.0, 60.0, 90.0, 120.0,
)
YOLO_HEIGHT_BIN_COUNT = len(YOLO_HEIGHT_BINS_M)

# SegFormer landcover class IDs used by the variant picker (must match SEGFORMER constants):
#   0=background, 1=bareland, 2=rangeland, 3=developed, 4=road,
#   5=tree,       6=water,    7=agriculture, 8=buildings
_SF_BARELAND    = 1
_SF_RANGELAND   = 2
_SF_DEVELOPED   = 3
_SF_ROAD        = 4
_SF_TREE        = 5
_SF_WATER       = 6
_SF_AGRICULTURE = 7
_SF_BUILDING    = 8

# simHeaven facade library group exports (resolve to a pool of variants at sim load time)
_SH_RESIDENTIAL = "simheaven/facades/residential.fac"
_SH_BUILDING    = "simheaven/facades/building.fac"
_SH_COMMERCIAL  = "simheaven/facades/commercial.fac"
_SH_INDUSTRIAL  = "simheaven/facades/industrial.fac"
_SH_HIGH_RES    = "simheaven/facades/bld-high-res.fac"
_SH_HIGH_COM    = "simheaven/facades/bld-high-com.fac"
_SH_RETAIL      = "simheaven/facades/retail.fac"
_SH_FARM        = "simheaven/facades/farm.fac"
_SH_GARAGE      = "simheaven/facades/garage.fac"
_SH_HOTEL       = "simheaven/facades/hotel.fac"
_SH_SCHOOL      = "simheaven/facades/school.fac"
_SH_COLLEGE     = "simheaven/facades/college.fac"
_SH_UNIVERSITY  = "simheaven/facades/university.fac"

_XP12_LOW_FACADES = (
    "lib/buildings/facades/generic/low_modern_01.fac",
    "lib/buildings/facades/commercial/low_commercial_01.fac",
    "lib/buildings/facades/commercial/low_commercial_02.fac",
    "lib/buildings/facades/commercial/low_commercial_03.fac",
    "lib/buildings/facades/commercial/low_commercial_04.fac",
    "lib/buildings/facades/commercial/low_commercial_05.fac",
)
_XP12_STRIP_COMMERCIAL_FACADES = (
    "lib/buildings/facades/commercial/low_commercial_01.fac",
    "lib/buildings/facades/commercial/low_commercial_02.fac",
    "lib/buildings/facades/commercial/low_commercial_03.fac",
    "lib/buildings/facades/commercial/low_commercial_04.fac",
    "lib/buildings/facades/commercial/low_commercial_05.fac",
    "lib/buildings/facades/commercial/low_commercial_06.fac",
    "lib/buildings/facades/commercial/low_commercial_07.fac",
    "lib/buildings/facades/commercial/low_commercial_08.fac",
)
_XP12_MID_FACADES = (
    "lib/buildings/facades/generic/mid_classic_01.fac",
    "lib/buildings/facades/generic/mid_classic_02.fac",
    "lib/buildings/facades/generic/mid_modern_01.fac",
    "lib/buildings/facades/generic/mid_modern_02.fac",
    "lib/buildings/facades/generic/mid_modern_03.fac",
    "lib/buildings/facades/generic/mid_modern_04.fac",
    "lib/buildings/facades/generic/mid_modern_05.fac",
)
_XP12_HIGH_FACADES = (
    "lib/buildings/facades/generic/high_classic_01.fac",
    "lib/buildings/facades/generic/high_classic_02.fac",
    "lib/buildings/facades/generic/high_glass_01.fac",
    "lib/buildings/facades/generic/high_glass_02.fac",
    "lib/buildings/facades/generic/high_glass_03.fac",
    "lib/buildings/facades/generic/high_metallic_01.fac",
    "lib/buildings/facades/generic/high_modern_01.fac",
    "lib/buildings/facades/generic/high_modern_02.fac",
    "lib/buildings/facades/generic/high_modern_03.fac",
    "lib/buildings/facades/generic/high_modern_04.fac",
    "lib/buildings/facades/generic/high_modern_05.fac",
    "lib/buildings/facades/generic/high_modern_06.fac",
    "lib/buildings/facades/generic/high_modern_07.fac",
    "lib/buildings/facades/generic/high_universal_01.fac",
    "lib/buildings/facades/generic/high_universal_02.fac",
)
_XP12_WAREHOUSE_FACADES = (
    "lib/buildings/facades/industrial/warehouse_01_45x45.fac",
    "lib/buildings/facades/industrial/warehouse_02_45x45.fac",
    "lib/buildings/facades/industrial/warehouse_03_60x60.fac",
    "lib/buildings/facades/industrial/warehouse_04_60x60.fac",
    "lib/buildings/facades/industrial/warehouse_05_60x60.fac",
    "lib/buildings/facades/industrial/warehouse_06_90x40.fac",
    "lib/buildings/facades/industrial/warehouse_07_90x40.fac",
    "lib/buildings/facades/industrial/warehouse_08_90x90.fac",
    "lib/buildings/facades/industrial/warehouse_09_90x90.fac",
    "lib/buildings/facades/industrial/warehouse_10_90x90.fac",
)
# NOTE: _SH_BUILDING / _SH_RESIDENTIAL / _SH_FARM resolve to simHeaven's
# house_default_*.fac with ROOF_SLOPE SLANT. X-Plane constructs that roof
# as concentric inset rings around the footprint — fine for the ~10-20 m
# houses they were authored for, but on class-4+ megablock detections
# (footprints up to 250 m) the rings degenerate into rainbow-banded roofs.
# Keep them OUT of every class >= BLD_CLASS_MEDIUM variant pool.
_SH_URBAN_FACADES = (
    _SH_COMMERCIAL, _SH_RETAIL, _SH_HOTEL,
    _SH_HIGH_RES, _SH_HIGH_COM, _SH_SCHOOL, _SH_COLLEGE, _SH_UNIVERSITY,
)
_SH_ROAD_FACADES = (_SH_COMMERCIAL, _SH_RETAIL, _SH_GARAGE, _SH_HOTEL)
_SH_RURAL_FACADES = (_SH_RESIDENTIAL, _SH_BUILDING, _SH_FARM, _SH_GARAGE)
_SH_RURAL_FACADES_LG = (_SH_GARAGE,)  # slant-roof house facades excluded
_SH_INDUSTRIAL_FACADES = (_SH_INDUSTRIAL, _SH_COMMERCIAL, _SH_GARAGE)
_SH_WATERFRONT_FACADES = (_SH_HOTEL,)

# CONTEXT_FACADE_VARIANTS[(placement_class, dominant_landcover_class)] = tuple of facade lib paths.
# Picker uses the dominant non-building landcover class around the detection to pick a variant pool;
# falls back to DEFAULT_FACADE_VARIANTS_BY_CLASS if no entry matches.
CONTEXT_FACADE_VARIANTS = {
    # Dense or mixed developed areas: XP12 mid/high plus broad urban simHeaven groups.
    (BLD_CLASS_TINY_RESIDENTIAL, _SF_DEVELOPED): _XP12_LOW_FACADES[:4] + (_SH_RESIDENTIAL, _SH_BUILDING),
    (BLD_CLASS_SMALL_RESIDENTIAL, _SF_DEVELOPED): _XP12_LOW_FACADES + (_SH_RESIDENTIAL, _SH_BUILDING, _SH_RETAIL),
    (BLD_CLASS_COMPACT_RESIDENTIAL, _SF_DEVELOPED): _XP12_LOW_FACADES + _XP12_MID_FACADES[:3] + (_SH_RESIDENTIAL, _SH_BUILDING, _SH_COMMERCIAL, _SH_RETAIL),
    (BLD_CLASS_MEDIUM, _SF_DEVELOPED): _XP12_MID_FACADES + _XP12_STRIP_COMMERCIAL_FACADES[3:] + _SH_URBAN_FACADES,
    (BLD_CLASS_SMALL_APARTMENT, _SF_DEVELOPED): _XP12_MID_FACADES + _XP12_HIGH_FACADES[:8] + _SH_URBAN_FACADES,
    (BLD_CLASS_APARTMENT_BLOCK, _SF_DEVELOPED): _XP12_HIGH_FACADES + _SH_URBAN_FACADES,
    (BLD_CLASS_LARGE, _SF_DEVELOPED): _XP12_STRIP_COMMERCIAL_FACADES[5:] + _XP12_WAREHOUSE_FACADES[2:] + (_SH_COMMERCIAL, _SH_RETAIL, _SH_HIGH_COM),
    (BLD_CLASS_EXTRA_LARGE, _SF_DEVELOPED): _XP12_WAREHOUSE_FACADES[5:] + (_SH_COMMERCIAL, _SH_HIGH_COM),

    # Road-adjacent commercial strips.
    (BLD_CLASS_TINY_RESIDENTIAL, _SF_ROAD): _XP12_STRIP_COMMERCIAL_FACADES[:5] + _SH_ROAD_FACADES,
    (BLD_CLASS_SMALL_RESIDENTIAL, _SF_ROAD): _XP12_STRIP_COMMERCIAL_FACADES[:6] + _SH_ROAD_FACADES,
    (BLD_CLASS_COMPACT_RESIDENTIAL, _SF_ROAD): _XP12_STRIP_COMMERCIAL_FACADES + _SH_ROAD_FACADES,
    (BLD_CLASS_MEDIUM, _SF_ROAD): _XP12_STRIP_COMMERCIAL_FACADES + _XP12_MID_FACADES[2:] + _SH_ROAD_FACADES,
    (BLD_CLASS_SMALL_APARTMENT, _SF_ROAD): _XP12_MID_FACADES + _XP12_STRIP_COMMERCIAL_FACADES[5:] + _SH_ROAD_FACADES,
    (BLD_CLASS_APARTMENT_BLOCK, _SF_ROAD): _XP12_HIGH_FACADES[:10] + _SH_ROAD_FACADES + (_SH_HIGH_COM,),
    (BLD_CLASS_LARGE, _SF_ROAD): _XP12_STRIP_COMMERCIAL_FACADES[5:] + _XP12_WAREHOUSE_FACADES[:5] + _SH_ROAD_FACADES,
    (BLD_CLASS_EXTRA_LARGE, _SF_ROAD): _XP12_WAREHOUSE_FACADES[5:] + (_SH_COMMERCIAL, _SH_RETAIL),

    # Rural and green contexts.
    (BLD_CLASS_TINY_RESIDENTIAL, _SF_AGRICULTURE): _XP12_LOW_FACADES[:3] + _SH_RURAL_FACADES,
    (BLD_CLASS_SMALL_RESIDENTIAL, _SF_AGRICULTURE): _XP12_LOW_FACADES[:4] + _SH_RURAL_FACADES,
    (BLD_CLASS_COMPACT_RESIDENTIAL, _SF_AGRICULTURE): _XP12_LOW_FACADES[:5] + _XP12_MID_FACADES[:2] + _SH_RURAL_FACADES,
    (BLD_CLASS_MEDIUM, _SF_AGRICULTURE): _XP12_LOW_FACADES + _XP12_MID_FACADES[:3] + _SH_RURAL_FACADES_LG,
    (BLD_CLASS_LARGE, _SF_AGRICULTURE): _XP12_WAREHOUSE_FACADES[:5] + (_SH_INDUSTRIAL,),
    (BLD_CLASS_EXTRA_LARGE, _SF_AGRICULTURE): _XP12_WAREHOUSE_FACADES[5:] + (_SH_INDUSTRIAL,),
    (BLD_CLASS_TINY_RESIDENTIAL, _SF_TREE): _XP12_LOW_FACADES[:3] + _SH_RURAL_FACADES,
    (BLD_CLASS_SMALL_RESIDENTIAL, _SF_TREE): _XP12_LOW_FACADES[:4] + _SH_RURAL_FACADES,
    (BLD_CLASS_COMPACT_RESIDENTIAL, _SF_TREE): _XP12_LOW_FACADES + _SH_RURAL_FACADES,
    (BLD_CLASS_MEDIUM, _SF_TREE): _XP12_LOW_FACADES + _XP12_MID_FACADES[:3] + _SH_RURAL_FACADES_LG,
    (BLD_CLASS_TINY_RESIDENTIAL, _SF_RANGELAND): _XP12_LOW_FACADES[:3] + _SH_RURAL_FACADES,
    (BLD_CLASS_SMALL_RESIDENTIAL, _SF_RANGELAND): _XP12_LOW_FACADES[:4] + _SH_RURAL_FACADES,
    (BLD_CLASS_COMPACT_RESIDENTIAL, _SF_RANGELAND): _XP12_LOW_FACADES + _SH_RURAL_FACADES,
    (BLD_CLASS_MEDIUM, _SF_RANGELAND): _XP12_LOW_FACADES + _XP12_MID_FACADES[:3] + _SH_RURAL_FACADES_LG,

    # Bareland reads best as industrial, warehouse, or big-lot commercial.
    (BLD_CLASS_MEDIUM, _SF_BARELAND): _XP12_WAREHOUSE_FACADES[:5] + _XP12_STRIP_COMMERCIAL_FACADES[5:] + _SH_INDUSTRIAL_FACADES,
    (BLD_CLASS_SMALL_APARTMENT, _SF_BARELAND): _XP12_WAREHOUSE_FACADES[:6] + _SH_INDUSTRIAL_FACADES,
    (BLD_CLASS_APARTMENT_BLOCK, _SF_BARELAND): _XP12_WAREHOUSE_FACADES[2:8] + _SH_INDUSTRIAL_FACADES,
    (BLD_CLASS_LARGE, _SF_BARELAND): _XP12_WAREHOUSE_FACADES + _SH_INDUSTRIAL_FACADES,
    (BLD_CLASS_EXTRA_LARGE, _SF_BARELAND): _XP12_WAREHOUSE_FACADES[5:] + _SH_INDUSTRIAL_FACADES,

    # Waterfront: keep it conservative, avoiding rural or industrial-only groups.
    (BLD_CLASS_MEDIUM, _SF_WATER): _XP12_MID_FACADES[:3] + ("lib/buildings/facades/generic/high_glass_01.fac",) + _SH_WATERFRONT_FACADES,
    (BLD_CLASS_SMALL_APARTMENT, _SF_WATER): _XP12_MID_FACADES[:4] + _XP12_HIGH_FACADES[:4] + _SH_WATERFRONT_FACADES,
    (BLD_CLASS_APARTMENT_BLOCK, _SF_WATER): _XP12_HIGH_FACADES[:8] + _SH_WATERFRONT_FACADES,
}


def _facade_for_detection(facade_cls, veg_map, jx, jy, m_per_px,
                          lat=0.0, lon=0.0, include_simheaven_assets=True):
    """Pick a facade lib path for a YOLO detection using SegFormer landcover context.

    Samples a ~50 m radius window around the detection center, takes the
    dominant non-building landcover class, and looks up a variant pool in
    CONTEXT_FACADE_VARIANTS. Falls back to DEFAULT_FACADE_VARIANTS_BY_CLASS.
    Variant within the pool is chosen by a stable hash so identical detections
    always pick the same path."""
    variants = None
    if veg_map is not None and veg_map.size:
        h, w = veg_map.shape[:2]
        if 0 <= jx < w and 0 <= jy < h:
            r = max(1, int(50.0 / max(float(m_per_px), 0.1)))
            y0 = max(0, jy - r); y1 = min(h, jy + r + 1)
            x0 = max(0, jx - r); x1 = min(w, jx + r + 1)
            win = veg_map[y0:y1, x0:x1]
            if win.size:
                # SegFormer can emit negative ignore-indices (-100, -1) and
                # out-of-range labels at padded edges; drop them before
                # bincount, which requires non-negative inputs.
                arr = np.asarray(win, dtype=np.int64).ravel()
                arr = arr[(arr >= 0) & (arr < 9)]
                if arr.size:
                    counts = np.bincount(arr, minlength=9)
                    counts[_SF_BUILDING] = 0
                    counts[0] = 0  # ignore background
                    if counts.sum() > 0:
                        dominant = int(counts.argmax())
                        variants = CONTEXT_FACADE_VARIANTS.get((facade_cls, dominant))
    if variants and not include_simheaven_assets:
        variants = tuple(
            path for path in variants
            if not str(path).lower().startswith("simheaven/")
        )
    if not variants:
        variants = DEFAULT_FACADE_VARIANTS_BY_CLASS.get(
            facade_cls, (DEFAULT_FACADE_PATHS[facade_cls],)
        )
    # Deterministic variant pick per (lat, lon, jx, jy), independent of
    # Python's process-randomized hash seed.
    key = (int(round(lat * 1e6)), int(round(lon * 1e6)), int(jx), int(jy))
    digest = hashlib.sha1(
        f"{key[0]}:{key[1]}:{key[2]}:{key[3]}".encode("ascii")
    ).digest()
    idx = int.from_bytes(digest[:8], "big") % len(variants)
    return variants[idx]


DEFAULT_OBJECT_CATALOG_OLD_WORLD = (
    # Default library aliases with simple rectangular footprints. These aliases
    # randomize among multiple physical building variants in X-Plane's library.
    "/lib/global8/us/hill_sq_30_30r.obj",
    "/lib/global8/us/hill_sq_30_30f.obj",
    "/lib/global8/us/in_sq_30_30r.obj",
    "/lib/global8/us/out_sq_30_30r.obj",
    "/lib/global8/us/out_sq_30_30f.obj",
    "/lib/global8/us/ind_sq_30_30r.obj",
    "/lib/global8/us/hill_sq_60_60f.obj",
    "/lib/global8/us/ind_sq_60_60r.obj",
    "/lib/global8/us/out_sq_60_60f.obj",
)
DEFAULT_OBJECT_CATALOG_NORTH_AMERICA = (
    "/lib/global8/us/feat_Building_50_40_600r20.obj",
    "/lib/global8/us/feat_Building_50_40_600r30.obj",
    "/lib/global8/us/feat_Building_50_40_600r40.obj",
    "/lib/global8/us/feat_Building_50_40_600r50.obj",
    "/lib/global8/us/feat_Building_50_40_600r80.obj",
    "/lib/global8/us/feat_Building_50_40_600r90.obj",
    "/lib/global8/us/feat_Building_50_40_600r100.obj",
    "/lib/global8/us/feat_Building_50_40_600r120.obj",
    "/lib/global8/us/feat_Building_50_40_600r160.obj",
    "/lib/global8/us/feat_Building_50_40_600r200.obj",
)

SIMHEAVEN_SMALL_BUILDING_CATALOG = (
    "simheaven/sheds/shed_02x03x1.obj",
    "simheaven/houses/house_06x12x1.obj",
    "simheaven/houses/house_09x09x1.obj",
    "simheaven/houses/house_09x12x1.obj",
    "simheaven/houses/house_09x12x2.obj",
    "simheaven/houses/house_12x12x1.obj",
    "simheaven/houses/house_12x12x2.obj",
    "simheaven/houses/house_12x15x2.obj",
    "simheaven/houses/house_15x15x2.obj",
    "simheaven/houses/house_15x20x2.obj",
    "simheaven/houses/house_20x25x2.obj",
)
SIMHEAVEN_RESIDENTIAL_CATALOG = (
    "simheaven/residential/residential_10x10x3.obj",
    "simheaven/residential/residential_10x15x3.obj",
    "simheaven/residential/residential_10x20x3.obj",
    "simheaven/residential/residential_15x15x5.obj",
    "simheaven/residential/residential_15x20x4.obj",
    "simheaven/residential/residential_20x20x3.obj",
    "simheaven/residential/residential_20x25x3.obj",
    "simheaven/residential/residential_20x30x3.obj",
)
SIMHEAVEN_COMMERCIAL_CATALOG = (
    "simheaven/commercial/commercial_18x42.obj",
    "simheaven/commercial/commercial_24x30.obj",
)
SIMHEAVEN_INDUSTRIAL_CATALOG = (
    "simheaven/industrial/industrial_18x30.obj",
    "simheaven/industrial/industrial_18x36.obj",
    "simheaven/industrial/industrial_24x36.obj",
    "simheaven/industrial/industrial_30x54.obj",
    "simheaven/industrial/industrial_30x60.obj",
    "simheaven/industrial/industrial_36x36.obj",
    "simheaven/industrial/industrial_45x30.obj",
    "simheaven/industrial/industrial_60x30.obj",
    "simheaven/industrial/industrial_60x60.obj",
)

EXCLUDED_BUILDING_FILLER_ASSETS = {
    "sfd_global/asia/shed_1.obj",
    "sfd_global/asia/carport_1.obj",
    "sfd_global/asia/carport_2.obj",
    "sfd_global/australia/carport.obj",
    "simheaven/sheds/shed_02x03x1.obj",
}

EXCLUDED_BUILDING_ASSETS = {
    # 60 m apartment slab: too wide for apartment-block placement and visually
    # unsuitable as a generated large-footprint fallback.
    "sfd_global/asia/apartment_2.obj",
    # The user's "SFD Global Autogen" library.txt aliases many small simHeaven
    # house/residential virtual paths to tall SFD Asia apartment meshes
    # (Asia/Apartment_*.obj) — a library-author mistake. The overlay places these
    # simHeaven OBJ paths, so X-Plane renders an oversized apartment dwarfing its
    # spot. Blacklist every simHeaven virtual that resolves to an Asia apartment.
    # (The 137 lib/g10/autogen/*.ags|.agb aliases to Asia/Apartments*.ags are
    # native autogen strings the overlay never places, so they are not listed.)
    "simheaven/houses/house_06x15x2.obj",
    "simheaven/houses/house_12x30x2.obj",
    "simheaven/houses/house_12x30x3.obj",
    "simheaven/houses/house_15x20x3.obj",
    "simheaven/houses/house_15x30x2.obj",
    "simheaven/houses/house_15x30x3.obj",
    "simheaven/houses/house_15x35x2.obj",
    "simheaven/houses/house_15x35x3.obj",
    "simheaven/residential/residential_10x20x4.obj",
    "simheaven/residential/residential_10x20x5.obj",
    "simheaven/residential/residential_12x30x2.obj",
    "simheaven/residential/residential_12x30x3.obj",
    "simheaven/residential/residential_15x20x3.obj",
    "simheaven/residential/residential_15x20x5.obj",
    "simheaven/residential/residential_15x20x6.obj",
    "simheaven/residential/residential_15x30x3.obj",
    "simheaven/residential/residential_15x30x5.obj",
    "simheaven/residential/residential_15x30x6.obj",
    "simheaven/residential/residential_15x50x5.obj",
    "simheaven/residential/residential_16x12x4.obj",
    "simheaven/residential/residential_18x12x3.obj",
    "simheaven/residential/residential_20x12x3.obj",
    "simheaven/residential/residential_20x20x3.obj",
    "simheaven/residential/residential_20x20x5.obj",
    "simheaven/residential/residential_20x50x5.obj",
}

SIMHEAVEN_REPEATABLE_ASSET_DIRS = (
    "/houses/",
    "/residential/",
    "/sheds/",
    "/industrial/",
    "/commercial/",
)
SIMHEAVEN_SPECIAL_ASSET_TOKENS = (
    "/landmarks/",
    "/church",
    "church",
    "chapel",
    "cathedral",
    "basilica",
    "abbey",
    "monastery",
    "shrine",
    "relig",
    "mosque",
    "synagogue",
    "temple",
    "monument",
    "memorial",
    "castle",
    "palace",
    "stadium",
    "arena",
    "sports",
    "school",
    "kindergarten",
    "college",
    "university",
    "hospital",
    "clinic",
    "nursing",
    "fire_station",
    "police",
    "museum",
    "library",
    "courthouse",
    "prison",
    "townhall",
    "town_hall",
    "city_hall",
    "government",
    "community",
    "train_station",
    "railway_station",
    "airport",
    "terminal",
    "petrol",
    "fuel",
    "gas_station",
    "supermarket",
    "market",
    "mall",
    "retail",
    "restaurant",
    "fast_food",
    "hotel",
    "motel",
    "bank",
    "post_office",
    "postoffice",
    "pharmacy",
    "drugstore",
    "cinema",
    "theatre",
    "theater",
    "tower",
    "windmill",
    "lighthouse",
)

# Viz colours per zone class (BGR→RGB in numpy overlay)
ZONE_COLOURS = {
    BLD_CLASS_TINY_RESIDENTIAL: np.array([ 80, 220,  80]),
    BLD_CLASS_SMALL_RESIDENTIAL: np.array([150, 230,  60]),
    BLD_CLASS_COMPACT_RESIDENTIAL: np.array([245, 220,  70]),
    BLD_CLASS_MEDIUM: np.array([255, 165,  50]),
    BLD_CLASS_SMALL_APARTMENT: np.array([255,  90,  70]),
    BLD_CLASS_APARTMENT_BLOCK: np.array([ 80, 200, 255]),
    BLD_CLASS_LARGE: np.array([ 70, 130, 255]),
    BLD_CLASS_EXTRA_LARGE: np.array([135,  90, 255]),
}

def _natural_asset_region(tile_lat, tile_lon):
    """Return the asset region key from Natural Earth boundary polygons."""
    return asset_region_for_latlon(tile_lat, tile_lon)


def _asset_region(tile_lat, tile_lon, simheaven_package_region=None):
    """Return the effective asset region, optionally steered by X-World package."""
    natural_region = _natural_asset_region(tile_lat, tile_lon)
    package_region = (simheaven_package_region or "").strip().lower()
    if not package_region:
        return natural_region

    if package_region == "europe":
        if natural_region in ("scandinavia", "mediterranean", "europe"):
            return natural_region
        return "europe"

    if package_region == "america":
        if natural_region in (
            "north_america",
            "north_america_ne",
            "north_america_west",
            "south_america",
        ):
            return natural_region
        if -90.0 <= float(tile_lon) <= -30.0 and -60.0 <= float(tile_lat) <= 15.0:
            return "south_america"
        return "north_america"

    if package_region == "asia":
        if natural_region in ("asia", "se_asia"):
            return natural_region
        if 90.0 <= float(tile_lon) <= 145.0 and -12.0 <= float(tile_lat) <= 25.0:
            return "se_asia"
        return "asia"

    if package_region == "africa":
        return "africa"
    if package_region == "australia_oceania":
        return "australia_oceania"
    if package_region == "antarctica":
        return "generic"
    return natural_region


def _default_object_catalog_paths(tile_lat, tile_lon, asset_region=None):
    """Return default object aliases that fit the regional context."""
    region = asset_region or _asset_region(tile_lat, tile_lon)
    if region in ("north_america", "north_america_ne", "north_america_west"):
        return DEFAULT_OBJECT_CATALOG_NORTH_AMERICA
    if region in ("scandinavia", "mediterranean", "europe", "generic"):
        return DEFAULT_OBJECT_CATALOG_OLD_WORLD
    return ()


def _simheaven_catalog_paths(tile_lat, tile_lon, asset_region=None):
    """Return simHeaven virtual objects suited to the tile's region."""
    region = asset_region or _asset_region(tile_lat, tile_lon)
    if region in ("asia", "se_asia", "africa", "australia_oceania", "south_america"):
        return (
            SIMHEAVEN_SMALL_BUILDING_CATALOG +
            SIMHEAVEN_RESIDENTIAL_CATALOG +
            SIMHEAVEN_COMMERCIAL_CATALOG +
            SIMHEAVEN_INDUSTRIAL_CATALOG
        )
    if region in ("scandinavia", "mediterranean", "europe", "generic"):
        return (
            SIMHEAVEN_SMALL_BUILDING_CATALOG +
            SIMHEAVEN_RESIDENTIAL_CATALOG +
            SIMHEAVEN_COMMERCIAL_CATALOG[:8] +
            SIMHEAVEN_INDUSTRIAL_CATALOG
        )
    return (
        SIMHEAVEN_SMALL_BUILDING_CATALOG +
        SIMHEAVEN_RESIDENTIAL_CATALOG +
        SIMHEAVEN_COMMERCIAL_CATALOG +
        SIMHEAVEN_INDUSTRIAL_CATALOG
    )


def _is_repeatable_simheaven_asset(path):
    """Return True for generic simHeaven assets safe for non-factual placement."""
    p = (path or '').replace('\\', '/').lower()
    if not p.startswith('simheaven/'):
        return False
    if _is_excluded_building_filler_asset(p):
        return False
    if any(token in p for token in SIMHEAVEN_SPECIAL_ASSET_TOKENS):
        return False
    return any(token in p for token in SIMHEAVEN_REPEATABLE_ASSET_DIRS)


def _is_excluded_building_filler_asset(path):
    """Return True for object assets that should not stand in for buildings."""
    p = (path or '').replace('\\', '/').lower()
    return (
        p in EXCLUDED_BUILDING_ASSETS or
        p in EXCLUDED_BUILDING_FILLER_ASSETS or
        'carport' in p
    )


def _sfd_catalog_paths(tile_lat, tile_lon, asset_region=None):
    """Return the strict audited SFD object allowlist for this location."""
    region = asset_region or _asset_region(tile_lat, tile_lon)

    if region == "scandinavia":
        return [
            f"SFD_Global/Scandinavia/Residential/Suburban_{i}.obj"
            for i in range(1, 9)
        ]

    if region == "australia_oceania":
        return (
            [f"SFD_Global/Asia/Suburban_South_{i}.obj" for i in range(1, 9)] +
            [
                "SFD_Global/Australia/Shed.obj",
                "SFD_Global/Australia/Carport.obj",
            ]
        )

    if region == "asia":
        return (
            [f"SFD_Global/Asia/Suburban_{i}.obj" for i in range(1, 11)] +
            [
                "SFD_Global/Asia/Carport_1.obj",
                "SFD_Global/Asia/Carport_2.obj",
                "SFD_Global/Asia/Shed_1.obj",
            ] +
            ["SFD_Global/Asia/Apartment_1.obj", "SFD_Global/Asia/Apartment_3.obj"] +
            [f"SFD_Global/Buildings/Apartment_30m_{i}.obj" for i in range(1, 5)] +
            [
                "SFD_Global/Asia/Industry_20x40.obj",
                "SFD_Global/Asia/Industry_30x40.obj",
                "SFD_Global/Asia/Industry_60x50.obj",
                "SFD_Global/Asia/Industry_70x90.obj",
                "SFD_Global/Asia/Industry_150x80.obj",
            ]
        )

    if region == "se_asia":
        return (
            [f"SFD_Global/Asia/Suburban_South_{i}.obj" for i in range(1, 9)] +
            [
                "SFD_Global/Asia/Carport_1.obj",
                "SFD_Global/Asia/Carport_2.obj",
                "SFD_Global/Asia/Shed_1.obj",
            ]
        )

    if region == "africa":
        return [
            f"SFD_Global/Africa/Residential/Suburban_{i}.obj"
            for i in range(1, 9)
        ]

    if region == "mediterranean":
        return (
            [f"SFD_Global/Med/Residential/Suburban_{i}.obj" for i in range(1, 9)] +
            [f"SFD_Global/Med/Residential/Apartment_North_{i}.obj" for i in range(1, 9)] +
            [
                "SFD_Global/Med/Residential/Urban_Mid_7m.obj",
                "SFD_Global/Med/Residential/Urban_Mid_9m.obj",
                "SFD_Global/Med/Residential/Urban_Mid_12m.obj",
                "SFD_Global/Med/Residential/Urban_Mid_15m.obj",
                "SFD_Global/Med/Residential/Urban_Mid_18m.obj",
                "SFD_Global/Med/Residential/Urban_Mid_22m.obj",
                "SFD_Global/Med/Residential/Urban_Mid_25m.obj",
                "SFD_Global/Med/Residential/Urban_Mid_30m.obj",
                "SFD_Global/Med/Residential/Urban_Mid_90.obj",
                "SFD_Global/Med/Residential/Urban_Mid_-90.obj",
                "SFD_Global/Med/Residential/Urban_Mid_Corner_1L.obj",
                "SFD_Global/Med/Residential/Urban_Mid_Corner_1R.obj",
                "SFD_Global/Med/Residential/Urban_Mid_Corner_2L.obj",
                "SFD_Global/Med/Residential/Urban_Mid_Corner_2R.obj",
                "SFD_Global/Med/Residential/Urban_Mid_Corner_3L.obj",
                "SFD_Global/Med/Residential/Urban_Mid_Corner_3R.obj",
            ]
        )

    if region == "north_america_ne":
        return (
            [
                f"SFD_Global/New_England/Residential/Suburban_{i}.obj"
                for i in range(1, 9)
            ] +
            ["SFD_Global/New_England/Residential/Garage.obj"]
        )

    if region in ("north_america", "north_america_west"):
        return (
            [f"SFD_Global/US_West_Coast/Suburban_{i}.obj" for i in range(1, 9)] +
            [f"SFD_Global/New_England/Residential/Suburban_{i}.obj" for i in range(1, 9)] +
            ["SFD_Global/New_England/Residential/Garage.obj"]
        )

    if region == "south_america":
        return (
            [f"SFD_Global/South_America/Suburban_{i}.obj" for i in range(1, 11)] +
            [f"SFD_Global/South_America/Med_{i}.obj" for i in range(1, 9)]
        )

    # Default — Mediterranean (same exclusions as Med region above)
    return (
        [f"SFD_Global/Med/Residential/Suburban_{i}.obj" for i in range(1, 9)] +
        [f"SFD_Global/Med/Residential/Apartment_North_{i}.obj" for i in range(1, 9)] +
        [
            "SFD_Global/Med/Residential/Urban_Mid_7m.obj",
            "SFD_Global/Med/Residential/Urban_Mid_9m.obj",
            "SFD_Global/Med/Residential/Urban_Mid_12m.obj",
            "SFD_Global/Med/Residential/Urban_Mid_15m.obj",
            "SFD_Global/Med/Residential/Urban_Mid_18m.obj",
            "SFD_Global/Med/Residential/Urban_Mid_22m.obj",
            "SFD_Global/Med/Residential/Urban_Mid_25m.obj",
            "SFD_Global/Med/Residential/Urban_Mid_30m.obj",
            "SFD_Global/Med/Residential/Urban_Mid_90.obj",
            "SFD_Global/Med/Residential/Urban_Mid_-90.obj",
            "SFD_Global/Med/Residential/Urban_Mid_Corner_1L.obj",
            "SFD_Global/Med/Residential/Urban_Mid_Corner_1R.obj",
            "SFD_Global/Med/Residential/Urban_Mid_Corner_2L.obj",
            "SFD_Global/Med/Residential/Urban_Mid_Corner_2R.obj",
            "SFD_Global/Med/Residential/Urban_Mid_Corner_3L.obj",
            "SFD_Global/Med/Residential/Urban_Mid_Corner_3R.obj",
        ]
    )


def _bounds_from_dimensions(width_m, depth_m):
    """Return centred local object bounds from width/depth dimensions in metres."""
    return (-0.5 * width_m, 0.5 * width_m, -0.5 * depth_m, 0.5 * depth_m)


def _bounds_for_object_path(obj_path):
    """Return footprint bounds for an object path when dimensions are known."""
    bounds_m = OBJ_FOOTPRINTS.get(obj_path)
    if bounds_m is not None:
        return bounds_m
    dims = OBJ_DIMS.get(obj_path)
    if dims is None:
        return None
    return _bounds_from_dimensions(dims[0], dims[1])


def _footprint_metrics(bounds_m):
    """Return (area_m2, max_side_m) for local footprint bounds."""
    xmin, xmax, zmin, zmax = bounds_m
    width_m = float(xmax) - float(xmin)
    depth_m = float(zmax) - float(zmin)
    return width_m * depth_m, max(width_m, depth_m)


def _class_for_footprint(bounds_m):
    """Classify an asset by footprint only; height does not affect placement."""
    area_m2, max_side_m = _footprint_metrics(bounds_m)
    if area_m2 <= ASSET_TINY_RESIDENTIAL_MAX_M2:
        return BLD_CLASS_TINY_RESIDENTIAL
    if area_m2 <= ASSET_SMALL_RESIDENTIAL_MAX_M2:
        return BLD_CLASS_SMALL_RESIDENTIAL
    if area_m2 <= ASSET_COMPACT_RESIDENTIAL_MAX_M2:
        return BLD_CLASS_COMPACT_RESIDENTIAL
    if area_m2 <= ASSET_MEDIUM_MAX_M2 and max_side_m <= ASSET_MEDIUM_MAX_SIDE_M:
        return BLD_CLASS_MEDIUM
    if area_m2 <= ASSET_SMALL_APARTMENT_MAX_M2 and max_side_m <= ASSET_SMALL_APARTMENT_MAX_SIDE_M:
        return BLD_CLASS_SMALL_APARTMENT
    if (
        area_m2 <= ASSET_APARTMENT_BLOCK_MAX_M2 and
        max_side_m <= ASSET_APARTMENT_BLOCK_MAX_SIDE_M
    ):
        return BLD_CLASS_APARTMENT_BLOCK
    if area_m2 <= ASSET_LARGE_MAX_M2 and max_side_m <= ASSET_LARGE_MAX_SIDE_M:
        return BLD_CLASS_LARGE
    return BLD_CLASS_EXTRA_LARGE


def _nearest_available_class(zone_class, asset_pools):
    """Return the nearest placement class with at least one building asset."""
    zone_class = int(zone_class)
    if asset_pools.get(zone_class):
        return zone_class
    ordered = sorted(
        BLD_PLACEMENT_CLASSES,
        key=lambda cls: (abs(int(cls) - zone_class), int(cls)),
    )
    for cls in ordered:
        if asset_pools.get(cls):
            return int(cls)
    return BLD_CLASS_LARGE


def _minimum_class_for_object_path(obj_path):
    """Return the smallest placement class allowed by an asset's visual scale."""
    p = (obj_path or '').replace('\\', '/').lower()
    if '/industry' in p or '/industrial/' in p or 'warehouse' in p:
        return BLD_CLASS_LARGE
    if '/apartment' in p or '/commercial/' in p:
        return BLD_CLASS_SMALL_APARTMENT
    if '/urban_mid_' in p:
        return BLD_CLASS_MEDIUM
    floors = _simheaven_object_floor_count(p)
    if floors is not None and floors > 2.0:
        return BLD_CLASS_MEDIUM
    return BLD_CLASS_TINY_RESIDENTIAL


def _class_for_object_asset(obj_path, bounds_m):
    """Classify an object by footprint, with compact kept to low-rise assets."""
    return max(
        _class_for_footprint(bounds_m),
        _minimum_class_for_object_path(obj_path),
    )


def _append_object_asset(asset_pools, obj_path, bounds_m, source,
                         region_priority=None):
    """Append one rectangular object asset to the footprint-classed pool map.

    ``region_priority`` is an integer where lower = preferred:
        0 = region-specific match (e.g. asset regions {'europe'} on an EU tile)
        1 = generic wildcard match
        2 = unspecified (falls back like 1 today)
    Pools sort by this key first when picking placement candidates.
    """
    if _is_excluded_building_filler_asset(obj_path):
        return False
    if bounds_m is None:
        return False
    height_m = _object_estimated_height_m(obj_path)
    if _is_very_tall_building_asset(obj_path, height_m):
        return False
    zone_class = _class_for_object_asset(obj_path, bounds_m)
    area_m2, max_side_m = _footprint_metrics(bounds_m)
    asset_pools[zone_class].append({
        'kind': 'object',
        'path': obj_path,
        'bounds_m': bounds_m,
        'height_m': height_m,
        'footprint_area_m2': area_m2,
        'footprint_max_side_m': max_side_m,
        'footprint_class': zone_class,
        'source': source,
        'region_priority': 1 if region_priority is None else int(region_priority),
        'requires_residential_context': _asset_requires_residential_context(
            {'kind': 'object', 'path': obj_path}
        ),
    })
    return True


def _region_priority_for_asset_regions(asset_regions, asset_region):
    """Return 0 when the asset is region-specific for the tile, 1 if generic."""
    if not asset_regions:
        return 1
    if "generic" in asset_regions and len(asset_regions) == 1:
        # Pure-generic wildcard.
        return 1
    region = (asset_region or "generic").lower()
    allowed = OPTIONAL_ASSET_REGION_ALIASES.get(region, {region})
    if set(asset_regions).intersection(allowed):
        return 0
    # Mixed regional + generic where the regional doesn't match the tile.
    # Fall back to generic priority since "generic" is still a wildcard.
    return 1 if "generic" in asset_regions else 1


def _asset_retry_sort_key(asset):
    """Sort assets so exhaustive retries test tighter footprints first.

    ``region_priority`` comes FIRST: regional matches (0) win over generic
    wildcards (1) for any tile that has classified regional candidates.
    Inside a priority tier, the existing footprint-tight-first order applies.
    """
    area_m2 = asset.get('footprint_area_m2')
    max_side_m = asset.get('footprint_max_side_m')
    if area_m2 is None or max_side_m is None:
        bounds_m = asset.get('bounds_m')
        if bounds_m is not None:
            area_m2, max_side_m = _footprint_metrics(bounds_m)
    region_priority = asset.get('region_priority')
    if region_priority is None:
        region_priority = 1
    return (
        int(region_priority),
        float('inf') if area_m2 is None else float(area_m2),
        float('inf') if max_side_m is None else float(max_side_m),
        asset.get('source', ''),
        asset.get('path', ''),
    )


def _sort_asset_pools_for_retry(asset_pools):
    """Order each regional/class pool for fast exhaustive fit retry."""
    for zone_class in BLD_PLACEMENT_CLASSES:
        asset_pools[zone_class].sort(key=_asset_retry_sort_key)


def _keep_smallest_asset_per_class(asset_pools):
    """Reduce each placement class to its smallest available footprint asset."""
    for zone_class in BLD_PLACEMENT_CLASSES:
        pool = asset_pools.get(zone_class, [])
        if not pool:
            continue
        pool.sort(key=_asset_retry_sort_key)
        asset_pools[zone_class] = pool[:1]
    return asset_pools


def _height_priors_by_class(asset_pools):
    """Return per-placement-class facade height ranges from available assets."""
    priors = {}
    for zone_class in BLD_PLACEMENT_CLASSES:
        heights = []
        for asset in (asset_pools or {}).get(zone_class, ()):
            try:
                height = float(asset.get('height_m'))
            except (TypeError, ValueError):
                continue
            if (
                math.isfinite(height) and
                height > 0.0 and
                height <= float(MAX_GENERATED_BUILDING_HEIGHT_M)
            ):
                heights.append(height)

        if heights:
            min_h = float(min(heights))
            max_h = float(max(heights))
            default_h = float(DEFAULT_FACADE_HEIGHT_M.get(zone_class, min_h))
            mode_h = float(np.clip(default_h, min_h, max_h))
        else:
            fallback_h = float(DEFAULT_FACADE_HEIGHT_M.get(zone_class, 8.0))
            fallback_h = min(fallback_h, float(MAX_GENERATED_BUILDING_HEIGHT_M))
            min_h = max_h = mode_h = fallback_h
        priors[int(zone_class)] = (min_h, max_h, mode_h)
    return priors


def _height_priors_signature(height_priors_by_class):
    return tuple(
        (
            int(zone_class),
            round(float(bounds[0]), 3),
            round(float(bounds[1]), 3),
            round(float(bounds[2]), 3),
        )
        for zone_class, bounds in sorted((height_priors_by_class or {}).items())
    )


def _randomized_facade_height_m(rng, height_priors_by_class, placement_cls):
    """Sample a deterministic-runtime facade height for a placement class."""
    try:
        zone_class = int(placement_cls)
    except (TypeError, ValueError):
        zone_class = BLD_CLASS_MEDIUM
    bounds = (height_priors_by_class or {}).get(zone_class)
    if bounds is None:
        fallback_h = float(DEFAULT_FACADE_HEIGHT_M.get(zone_class, 8.0))
        return min(fallback_h, float(MAX_GENERATED_BUILDING_HEIGHT_M))

    min_h, max_h, mode_h = (float(v) for v in bounds)
    min_h = max(0.1, min(min_h, float(MAX_GENERATED_BUILDING_HEIGHT_M)))
    max_h = max(min_h, min(max_h, float(MAX_GENERATED_BUILDING_HEIGHT_M)))
    mode_h = float(np.clip(mode_h, min_h, max_h))
    if max_h <= min_h + 1e-6:
        return min_h
    return float(rng.triangular(min_h, mode_h, max_h))


def _asset_footprint_span_m(asset):
    """Return the larger raw footprint side for centre-spacing estimates."""
    max_side_m = asset.get('footprint_max_side_m')
    if max_side_m is not None:
        return float(max_side_m)
    bounds_m = asset.get('bounds_m')
    if bounds_m is None:
        return None
    _, max_side_m = _footprint_metrics(bounds_m)
    return float(max_side_m)


def _class_min_footprint_span_m(asset_pools):
    """Return the smallest raw footprint span available for each class."""
    min_by_class = {}
    for zone_class in BLD_PLACEMENT_CLASSES:
        spans = [
            span
            for span in (_asset_footprint_span_m(asset) for asset in asset_pools[zone_class])
            if span is not None
        ]
        min_by_class[zone_class] = min(spans) if spans else 0.0
    return min_by_class


def _mark_pad_for_edge_spacing_m(edge_spacing_m, footprint_pad_m=FOOTPRINT_PAD_M):
    """Return mark expansion that yields the requested raw-footprint edge gap."""
    return max(0.0, float(edge_spacing_m) - float(footprint_pad_m))


def _asset_retry_sequence(pool, rng, prefer_small=False):
    """Yield every asset once, randomized while optionally biasing smaller assets."""
    n_assets = len(pool)
    if n_assets <= 0:
        return
    if prefer_small:
        # Pools are sorted by footprint. Shuffle small windows to keep visual
        # variety while still trying compact assets first in leftover gaps.
        window = 4
        for start in range(0, n_assets, window):
            idx = np.arange(start, min(start + window, n_assets), dtype=np.int32)
            rng.shuffle(idx)
            for i in idx:
                yield pool[int(i)]
        return
    start = 0 if n_assets == 1 else int(rng.integers(0, n_assets))
    for offset in range(n_assets):
        yield pool[(start + offset) % n_assets]


def _asset_retry_context(pool):
    """Return cached context-filter metadata for one sorted asset pool."""
    requires = tuple(bool(_asset_requires_residential_context(asset)) for asset in pool)
    prefix = [0]
    for flag in requires:
        prefix.append(prefix[-1] + int(flag))
    eligible = tuple(i for i, flag in enumerate(requires) if not flag)
    return {
        'requires_residential': requires,
        'residential_prefix': tuple(prefix),
        'nonresidential_indices': eligible,
    }


def _residential_skip_count(retry_context, start, end):
    """Count residential-only assets in cyclic [start, end)."""
    prefix = retry_context['residential_prefix']
    n_assets = len(prefix) - 1
    if n_assets <= 0 or start == end:
        return 0
    if start < end:
        return int(prefix[end] - prefix[start])
    return int((prefix[n_assets] - prefix[start]) + prefix[end])


def _asset_retry_sequence_for_context(pool, rng, prefer_small, residential_context,
                                      retry_context=None):
    """Yield ``(asset, skipped)`` while avoiding known-ineligible assets.

    In nonresidential context, house-like assets can never be selected.  This
    preserves the original random start/shuffle order for eligible assets and
    reports the same skipped count without walking every skipped asset in the
    hot fit loop.
    """
    if residential_context or retry_context is None:
        for asset in _asset_retry_sequence(pool, rng, prefer_small=prefer_small):
            yield asset, 0
        return

    n_assets = len(pool)
    if n_assets <= 0:
        return

    requires = retry_context['requires_residential']
    if prefer_small:
        skipped = 0
        window = 4
        for start in range(0, n_assets, window):
            idx = np.arange(start, min(start + window, n_assets), dtype=np.int32)
            rng.shuffle(idx)
            for i in idx:
                i = int(i)
                if requires[i]:
                    skipped += 1
                    continue
                yield pool[i], skipped
                skipped = 0
        if skipped:
            yield None, skipped
        return

    start = 0 if n_assets == 1 else int(rng.integers(0, n_assets))
    cursor = start
    eligible = retry_context['nonresidential_indices']
    yielded = False
    for i in eligible:
        if i >= start:
            yield pool[i], _residential_skip_count(retry_context, cursor, i)
            cursor = (i + 1) % n_assets
            yielded = True
    for i in eligible:
        if i < start:
            yield pool[i], _residential_skip_count(retry_context, cursor, i)
            cursor = (i + 1) % n_assets
            yielded = True
    if yielded:
        trailing = _residential_skip_count(retry_context, cursor, start)
    else:
        trailing = int(retry_context['residential_prefix'][-1])
    if trailing:
        yield None, trailing


def _asset_requires_residential_context(asset):
    """Return True for house-like assets that should stay in residential areas."""
    if 'requires_residential_context' in asset:
        return bool(asset['requires_residential_context'])
    path = (asset.get('path') or '').replace('\\', '/').lower()
    if asset.get('kind') == 'facade':
        return False
    if path.startswith('simheaven/'):
        return '/houses/' in path or '/residential/' in path
    if path.startswith('sfd_global/'):
        return (
            '/suburban' in path or
            '/residential/' in path or
            '/new_england/' in path or
            '/scandinavia/' in path or
            '/south_america/suburban' in path
        )
    return (
        '/house' in path or
        '/houses/' in path or
        '/residential/' in path or
        '/buildings/houses/' in path or
        '/buildings/residential/' in path or
        '/terraced/' in path
    )


def _asset_fit_inradius_m(asset):
    """Return the guaranteed occupied centre radius for an asset fit footprint."""
    bounds_m = asset.get('fit_bounds_m')
    if bounds_m is None:
        base_bounds_m = asset.get('bounds_m')
        if base_bounds_m is None:
            return None
        bounds_m = _expand_bounds(base_bounds_m, FOOTPRINT_PAD_M)
    xmin, xmax, zmin, zmax = bounds_m
    return 0.5 * min(float(xmax) - float(xmin), float(zmax) - float(zmin))


def _class_min_fit_inradius_m(asset_pools):
    """Return the smallest centre-blocker clearance needed by each class."""
    min_by_class = {}
    for zone_class in BLD_PLACEMENT_CLASSES:
        radii = [
            radius
            for radius in (_asset_fit_inradius_m(asset) for asset in asset_pools[zone_class])
            if radius is not None
        ]
        min_by_class[zone_class] = min(radii) if radii else 0.0
    return min_by_class


def _extra_library_policy():
    """Return optional building-library enrichment policy from the environment."""
    return os.environ.get("O4_SFR_BLD_EXTRA_LIBRARIES", "auto").strip() or "auto"


def _enabled_extra_library_ids(policy=None):
    """Return curated optional-library ids enabled by policy."""
    policy = (policy if policy is not None else _extra_library_policy()).strip().lower()
    if policy in {"", "auto", "1", "true", "yes", "on"}:
        return tuple(CURATED_EXTRA_BUILDING_LIBRARIES)
    if policy in {"off", "0", "false", "no", "none"}:
        return ()

    enabled = []
    requested = [part.strip().lower() for part in policy.split(",") if part.strip()]
    for requested_name in requested:
        for lib_id, meta in CURATED_EXTRA_BUILDING_LIBRARIES.items():
            names = {
                lib_id,
                meta["label"].lower(),
                meta["label"].lower().replace(" ", "-"),
                meta["label"].lower().replace(" ", "_"),
            }
            names.update(pattern.lower() for pattern in meta["package_patterns"])
            if requested_name in names and lib_id not in enabled:
                enabled.append(lib_id)
    return tuple(enabled)


def _runtime_library_package_patterns(include_sfd=False, include_simheaven=False,
                                      extra_library_ids=()):
    patterns = []
    if include_sfd:
        patterns.append("sfd global autogen")
    if include_simheaven:
        patterns.append("simheaven")
    for lib_id in extra_library_ids or ():
        meta = CURATED_EXTRA_BUILDING_LIBRARIES.get(lib_id)
        if meta:
            patterns.extend(meta["package_patterns"])
    return tuple(dict.fromkeys(patterns))


def _scan_runtime_library_exports(custom_scenery_dir, *, include_sfd=False,
                                  include_simheaven=False,
                                  extra_library_ids=(),
                                  suffixes=(".obj",)):
    """Scan only packages that may contribute runtime building assets."""
    patterns = _runtime_library_package_patterns(
        include_sfd=include_sfd,
        include_simheaven=include_simheaven,
        extra_library_ids=extra_library_ids,
    )
    if not patterns:
        return []
    return ASSETINV.scan_library_exports(
        custom_scenery_dir=custom_scenery_dir,
        package_name_patterns=patterns,
        suffixes=suffixes,
    )


def _library_exports_by_virtual_path(library_exports):
    return ASSETINV.unique_virtual_exports(library_exports or (), suffix=".obj")


# library.txt ``REGION`` block names (pack-author vocabulary, e.g. SFD's
# "scandinavia"/"asia_LHD", simHeaven's "asia") -> the overlay regions where
# the export is assumed active. Names are matched by substring so variants
# like asia_south or america_north resolve; unknown names match nothing
# (conservative: better a smaller pool than a DSF reference X-Plane cannot
# resolve -- see "Failed to find resource simheaven/houses/...").
_LIBRARY_REGION_NAME_TOKENS = (
    ("scandinavia", {"scandinavia"}),
    ("mediterranean", {"mediterranean"}),
    ("med", {"mediterranean"}),
    ("europe", {"europe", "scandinavia", "mediterranean"}),
    ("se_asia", {"se_asia", "asia"}),
    ("asia", {"asia", "se_asia"}),
    ("africa", {"africa"}),
    ("australia", {"australia_oceania"}),
    ("oceania", {"australia_oceania"}),
    ("pacific", {"australia_oceania"}),
    ("caribbean", {"south_america"}),
    ("south_america", {"south_america"}),
    ("america_south", {"south_america"}),
    ("latam", {"south_america"}),
    ("brazil", {"south_america"}),
    ("north_america", {"north_america", "north_america_ne", "north_america_west"}),
    ("america_north", {"north_america", "north_america_ne", "north_america_west"}),
    ("canada", {"north_america", "north_america_ne", "north_america_west"}),
    ("usa", {"north_america", "north_america_ne", "north_america_west"}),
    ("us", {"north_america", "north_america_ne", "north_america_west"}),
    ("america", {"north_america", "north_america_ne", "north_america_west",
                 "south_america"}),
)


def _library_region_matches_tile(region_name, tile_region):
    """Return True when a library REGION block is assumed active on the tile."""
    if not region_name:
        return True  # global export
    name = str(region_name).lower()
    tile = (tile_region or "generic").lower()
    for token, regions in _LIBRARY_REGION_NAME_TOKENS:
        if token in name:
            return tile in regions
    return False  # unknown region vocabulary: do not risk a dangling ref


def _filter_exports_for_tile_region(library_exports, tile_region):
    """Drop region-gated exports that X-Plane will not resolve on this tile."""
    if not library_exports:
        return library_exports
    kept = []
    dropped = 0
    for export in library_exports:
        if _library_region_matches_tile(
            getattr(export, "region", None), tile_region
        ):
            kept.append(export)
        else:
            dropped += 1
    if dropped:
        print(
            f"Library exports: dropped {dropped} region-gated exports "
            f"inactive for tile region '{tile_region}' "
            f"({len(kept)} remain)"
        )
    return kept


def _exported_virtual_library_paths(library_exports, suffixes=(".obj", ".fac")):
    """Return normalized virtual paths exported by installed libraries."""
    suffixes = tuple(str(suffix).lower() for suffix in suffixes)
    exported = set()
    for export in library_exports or ():
        virtual_path = getattr(export, "virtual_path", "") or ""
        norm_path = _norm_library_path(virtual_path)
        lower_path = norm_path.lower()
        if suffixes and not lower_path.endswith(suffixes):
            continue
        exported.add(norm_path)
    return exported


def _stock_yolo_asset_path_available(path, exported_paths):
    """Return True when a stock-YOLO library path can be emitted safely."""
    norm_path = _norm_library_path(path)
    lower_path = norm_path.lower()
    if lower_path.startswith("lib/"):
        return True
    if lower_path.startswith("simheaven/") or lower_path.startswith("sfd_global/"):
        return norm_path in exported_paths
    return False


def _filter_stock_yolo_asset_map(asset_map, exported_paths):
    """Drop stock-YOLO asset variants that are not exported by installed libs."""
    filtered = {}
    dropped_paths = []
    dropped_classes = []
    for cls, (placement_type, paths, default_height_m) in (asset_map or {}).items():
        kept_paths = tuple(
            path for path in paths
            if _stock_yolo_asset_path_available(path, exported_paths)
        )
        dropped_paths.extend(path for path in paths if path not in kept_paths)
        if not kept_paths:
            dropped_classes.append(cls)
            continue
        filtered[cls] = (placement_type, kept_paths, default_height_m)
    return filtered, dropped_paths, dropped_classes


def _bounds_union(bounds_iter):
    bounds = [tuple(float(v) for v in item) for item in bounds_iter if item is not None]
    if not bounds:
        return None
    return (
        min(item[0] for item in bounds),
        max(item[1] for item in bounds),
        min(item[2] for item in bounds),
        max(item[3] for item in bounds),
    )


def _measured_bounds_for_exports(exports, cache_dir=None):
    """Return unioned OBJ8 bounds for all physical variants of one virtual path."""
    return _bounds_union(
        _read_obj8_bounds(export.resolved_path, cache_dir)
        for export in exports
        if getattr(export, "resolved_path", None)
    )


# Reject a library alias when the mesh it actually resolves to has a much larger
# FOOTPRINT than the footprint its virtual-path name declares -- i.e. the library
# author exported a small footprint to a big mesh (e.g. simHeaven residential ->
# SFD Asia apartment slab). Height is intentionally ignored: a legit mid-rise can
# be tall while its footprint still matches its name. The test is PER-VARIANT
# (one virtual path can export many physical variants and X-Plane picks one at
# random) and RELATIVE to the declared footprint, so a declared-large asset
# (industrial) whose mesh is correspondingly large is kept, and a multi-variant
# house with one oversized variant among many is kept. Only a *majority* of
# variants being much larger than declared drops the alias.
# A small-declared alias whose mesh max-side reaches this is an apartment-slab
# scale footprint (SFD Asia Apartment_2 ~60 m). A relative ratio cannot be used:
# tiny house names (5x5, 10x10) resolve to normal ~16-20 m suburban meshes and
# would trip any ratio, while a legit large-declared asset (industrial) whose mesh
# is correspondingly large must be kept. So the test is ABSOLUTE on the resolved
# side, gated by a small DECLARED footprint.
ALIAS_OVERSIZE_ABS_SIDE_M = 30.0   # resolved footprint max-side = apartment-slab scale
ALIAS_OVERSIZE_VARIANT_FRAC = 0.5  # majority of variants must exceed to drop


def _alias_oversized_footprint_fraction(declared_bounds, exports, cache_dir=None):
    """Return ``(frac_oversized, n)``: variants resolving to an apartment-slab footprint.

    Only small-declared aliases are considered (a legitimately large declared asset
    whose mesh is correspondingly large is fine). Footprint-only; height ignored.
    """
    if declared_bounds is None:
        return 0.0, 0
    try:
        _dec_area, dec_side = _footprint_metrics(declared_bounds)
    except Exception:
        return 0.0, 0
    if dec_side <= 0.0 or dec_side >= ALIAS_OVERSIZE_ABS_SIDE_M:
        return 0.0, 0
    n = 0
    over = 0
    for export in exports:
        rp = getattr(export, "resolved_path", None)
        if not rp:
            continue
        bounds = _read_obj8_bounds(rp, cache_dir)
        if bounds is None:
            continue
        n += 1
        try:
            _real_area, real_side = _footprint_metrics(bounds)
        except Exception:
            continue
        if real_side > ALIAS_OVERSIZE_ABS_SIDE_M:
            over += 1
    if n == 0:
        return 0.0, 0
    return over / n, n


def _drop_oversized_aliased_assets(asset_pools, library_exports, cache_dir=None):
    """Drop object assets whose library alias resolves (mostly) to a bigger footprint.

    Auto-catches third-party library mistakes (a small declared virtual footprint
    exported to a large mesh) so they don't render as oversized buildings, without
    hand-listing each one. Footprint-only: height is ignored. Assets with no library
    export are kept; SFD scanned assets already carry measured bounds (declared ==
    real) so they pass. Returns ``(asset_pools, dropped_count, dropped_paths)``.
    """
    grouped = _library_exports_by_virtual_path(library_exports)
    if not grouped:
        return asset_pools, 0, []
    dropped = 0
    dropped_paths = set()
    for zone_class, pool in asset_pools.items():
        kept = []
        for asset in pool:
            if asset.get('kind') != 'object':
                kept.append(asset)
                continue
            path = asset.get('path')
            exports = grouped.get(_norm_library_path(path or ""))
            if not exports:
                kept.append(asset)
                continue
            frac, n = _alias_oversized_footprint_fraction(
                asset.get('bounds_m'), exports, cache_dir
            )
            if n > 0 and frac >= ALIAS_OVERSIZE_VARIANT_FRAC:
                dropped += 1
                dropped_paths.add(path)
                continue
            kept.append(asset)
        asset_pools[zone_class] = kept
    return asset_pools, dropped, sorted(dropped_paths)


def _footprint_within_limits(bounds_m, *, max_area_m2=12_000.0, max_side_m=170.0):
    if bounds_m is None:
        return False
    try:
        area_m2, side_m = _footprint_metrics(bounds_m)
    except Exception:
        return False
    return 12.0 <= float(area_m2) <= max_area_m2 and 2.0 <= float(side_m) <= max_side_m


def _path_has_any(path, tokens):
    p = (path or "").replace("\\", "/").lower()
    return any(token in p for token in tokens)


def _is_sfd_building_asset_candidate(obj_path):
    p = (obj_path or "").replace("\\", "/").lower()
    if not p.startswith("sfd_global/"):
        return False
    if _is_excluded_building_filler_asset(p):
        return False
    if _path_has_any(p, GENERIC_BUILDING_EXCLUDE_TOKENS):
        return False
    return _path_has_any(
        p,
        (
            "/suburban",
            "/apartment",
            "/apartments_",
            "/industry",
            "/industrial",
            "/urban_mid",
            "/garage",
            "/shed",
        ),
    )


def _sfd_export_matches_region(obj_path, asset_region):
    p = (obj_path or "").replace("\\", "/").lower()
    region = (asset_region or "generic").lower()
    if region == "scandinavia":
        return p.startswith("sfd_global/scandinavia/")
    if region == "australia_oceania":
        return p.startswith("sfd_global/australia/") or "/asia/suburban_south" in p
    if region == "asia":
        return p.startswith("sfd_global/asia/") or p.startswith("sfd_global/buildings/")
    if region == "se_asia":
        return "/asia/suburban_south" in p or p.startswith("sfd_global/asia/shed")
    if region == "africa":
        return p.startswith("sfd_global/africa/")
    if region == "mediterranean":
        return p.startswith("sfd_global/med/")
    if region == "north_america_ne":
        return p.startswith("sfd_global/new_england/")
    if region in ("north_america", "north_america_west"):
        return p.startswith("sfd_global/us_west_coast/") or p.startswith("sfd_global/new_england/")
    if region == "south_america":
        return p.startswith("sfd_global/south_america/")
    if region in ("europe", "generic"):
        return p.startswith("sfd_global/med/")
    return False


def _scanned_simheaven_catalog_paths(library_exports=None, custom_scenery_dir=None):
    """Return installed repeatable simHeaven aliases with encoded dimensions."""
    if library_exports is None and custom_scenery_dir:
        library_exports = _scan_runtime_library_exports(
            custom_scenery_dir,
            include_simheaven=True,
        )
    paths = []
    for key, exports in _library_exports_by_virtual_path(library_exports).items():
        if not key.startswith("simheaven/"):
            continue
        obj_path = exports[0].virtual_path
        if not _is_repeatable_simheaven_asset(obj_path):
            continue
        dims = _simheaven_object_dims(obj_path)
        if dims is None:
            continue
        height_m = _object_estimated_height_m(obj_path)
        if _is_very_tall_building_asset(obj_path, height_m):
            continue
        paths.append(obj_path)
    return tuple(sorted(set(paths), key=lambda item: item.lower()))


def _exported_simheaven_catalog_paths(library_exports):
    """Return repeatable simHeaven aliases safe to reference from this overlay."""
    if library_exports is None:
        return None
    return set(_scanned_simheaven_catalog_paths(library_exports=library_exports))


def _exported_simheaven_object_paths(library_exports):
    """Return all installed simHeaven OBJ aliases safe to reference in output."""
    if library_exports is None:
        return None
    paths = set()
    for key, exports in _library_exports_by_virtual_path(library_exports).items():
        if key.startswith("simheaven/") and exports:
            paths.add(exports[0].virtual_path)
    return paths


def _filter_unexported_simheaven_object_placements(placements, exported_paths):
    """Drop simHeaven OBJ placements that are not exported library aliases."""
    if exported_paths is None:
        return list(placements or ()), 0
    exported_norm = {_norm_library_path(path) for path in exported_paths}
    kept = []
    dropped = 0
    for placement in placements or ():
        try:
            obj_path = placement[3]
        except (TypeError, IndexError):
            kept.append(placement)
            continue
        norm_path = _norm_library_path(obj_path)
        if (
            norm_path.startswith("simheaven/") and
            norm_path.endswith(".obj") and
            norm_path not in exported_norm
        ):
            dropped += 1
            continue
        kept.append(placement)
    return kept, dropped


def _count_unexported_simheaven_object_placements(placements, exported_paths):
    """Count cached placements that reference unavailable simHeaven aliases."""
    if exported_paths is None:
        return 0
    _kept, dropped = _filter_unexported_simheaven_object_placements(
        placements,
        exported_paths,
    )
    return dropped


def _unexported_simheaven_object_path_counts(placements, exported_paths):
    """Return unavailable simHeaven OBJ aliases referenced by placements."""
    if exported_paths is None:
        return Counter()
    exported_norm = {_norm_library_path(path) for path in exported_paths}
    bad_paths = Counter()
    for placement in placements or ():
        try:
            obj_path = placement[3]
        except (TypeError, IndexError):
            continue
        norm_path = _norm_library_path(obj_path)
        if (
            norm_path.startswith("simheaven/")
            and norm_path.endswith(".obj")
            and norm_path not in exported_norm
        ):
            bad_paths[obj_path] += 1
    return bad_paths


def _assert_no_unexported_simheaven_object_placements(placements, exported_paths):
    """Fail loudly before writing scenery refs that X-Plane cannot resolve."""
    bad_paths = _unexported_simheaven_object_path_counts(placements, exported_paths)
    if not bad_paths:
        return
    examples = ", ".join(
        f"{path} x{count}" for path, count in bad_paths.most_common(5)
    )
    raise RuntimeError(
        "Building overlay contains unexported simHeaven OBJ reference(s); "
        "refusing to write a DSF that X-Plane will load with missing scenery. "
        f"Bad refs: {examples}"
    )


def _scanned_sfd_catalog_entries(asset_region, library_exports=None,
                                 custom_scenery_dir=None, cache_dir=None):
    """Return ``(virtual_path, measured_bounds)`` SFD entries for a region."""
    if library_exports is None and custom_scenery_dir:
        library_exports = _scan_runtime_library_exports(
            custom_scenery_dir,
            include_sfd=True,
        )
    entries = []
    for key, exports in _library_exports_by_virtual_path(library_exports).items():
        obj_path = exports[0].virtual_path
        if not _is_sfd_building_asset_candidate(obj_path):
            continue
        if not _sfd_export_matches_region(obj_path, asset_region):
            continue
        height_m = _object_estimated_height_m(obj_path)
        if _is_very_tall_building_asset(obj_path, height_m):
            continue
        bounds_m = _measured_bounds_for_exports(exports, cache_dir)
        if not _footprint_within_limits(bounds_m):
            continue
        entries.append((obj_path, bounds_m))
    return tuple(sorted(entries, key=lambda item: item[0].lower()))


def _is_optional_library_export_enabled(export, enabled_library_ids):
    package = (getattr(export, "package_name", "") or "").lower()
    virtual_path = (getattr(export, "virtual_path", "") or "").replace("\\", "/").lower()
    for lib_id in enabled_library_ids or ():
        meta = CURATED_EXTRA_BUILDING_LIBRARIES.get(lib_id)
        if not meta:
            continue
        if not any(pattern in package for pattern in meta["package_patterns"]):
            continue
        if any(virtual_path.startswith(prefix.lower()) for prefix in meta["virtual_prefixes"]):
            return True
        if lib_id == "world-models" and package == "world-models":
            return True
    return False


def _optional_library_id_for_export(export, enabled_library_ids):
    """Return the curated optional-library id for one library export."""
    for lib_id in enabled_library_ids or ():
        if _is_optional_library_export_enabled(export, (lib_id,)):
            return lib_id
    return None


def _optional_asset_region_match(asset_regions, asset_region):
    """Return True when a classified optional asset is valid for a tile region.

    The token ``"generic"`` on an asset acts as a wildcard so global libraries
    (OB_Library, MisterX residual residential subset, the o4sfr generic bucket)
    can serve tiles in any natural-earth region.
    """
    if not asset_regions:
        return False
    if "generic" in asset_regions:
        return True
    region = (asset_region or "generic").lower()
    allowed = OPTIONAL_ASSET_REGION_ALIASES.get(region, {region})
    return bool(set(asset_regions).intersection(allowed))


_O4SFR_PATH_REGION_TOKENS = {
    "/north_america/": ("north_america",),
    "/north-america/": ("north_america",),
    "/america/": ("north_america", "south_america"),
    "/americas/": ("north_america", "south_america"),
    "/south_america/": ("south_america",),
    "/south-america/": ("south_america",),
    "/latam/": ("south_america",),
    "/europe/": ("europe",),
    "/scandinavia/": ("scandinavia",),
    "/mediterranean/": ("mediterranean",),
    "/asia/": ("asia",),
    "/se_asia/": ("se_asia",),
    "/southeast_asia/": ("se_asia",),
    "/africa/": ("africa",),
    "/australia_oceania/": ("australia_oceania",),
    "/oceania/": ("australia_oceania",),
}


def _load_region_overrides():
    """Load Track A visual-triage decisions from disk.

    Returns a tuple ``((lib_id, family_prefix, regions), ...)`` sorted by
    descending ``len(family_prefix)`` so the longest (most specific) match
    wins when a path is a prefix of another override.

    Missing file is treated as "no overrides" rather than an error -- the
    file is generated by ``scripts/asset_pipeline/compile_region_overrides.py``
    and may not exist in a fresh checkout.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(here, "O4_SFR_Region_Overrides.json")
    if not os.path.isfile(json_path):
        return ()
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return ()
    entries = []
    for row in (data.get("overrides") or ()):
        lib_id = (row.get("lib_id") or "").lower()
        family = (row.get("family_prefix") or "").replace("\\", "/").lower()
        regions = tuple(
            r.lower() for r in (row.get("regions") or ()) if r
        )
        if not lib_id or not family or not regions:
            continue
        entries.append((lib_id, family, regions))
    entries.sort(key=lambda item: (-len(item[1]), item[0]))
    return tuple(entries)


_REGION_OVERRIDES = _load_region_overrides()


def _optional_library_asset_regions(obj_path, lib_id=None):
    """Return explicit region tags for optional-library building assets.

    Empty tuple means the path is intentionally treated as unclassified and is
    not eligible for generated building placement.

    Resolution order, first match wins:

    1. ``O4_SFR_Region_Overrides.json`` (visual-triage decisions).
    2. Per-library hand-coded path rules (world-models, cdb-library, o4sfr).
    3. The library's static ``regions`` registration in
       ``CURATED_EXTRA_BUILDING_LIBRARIES``.
    """
    p = (obj_path or "").replace("\\", "/").lower()
    lib_id = (lib_id or "").lower()

    # (1) Triage overrides win first.  The list is sorted longest-prefix
    # first; an entry whose ``family_prefix`` is a prefix of ``p`` matches.
    if lib_id and _REGION_OVERRIDES:
        for ov_lib, ov_prefix, ov_regions in _REGION_OVERRIDES:
            if ov_lib != lib_id:
                continue
            if p == ov_prefix or p.startswith(ov_prefix):
                return ov_regions

    if p.startswith("world-models/"):
        p = p[len("world-models/"):]

    if lib_id == "world-models" or p.startswith("objects/"):
        if (
            p.startswith("objects/houses/us/") or
            p.startswith("objects/world-models/houses/us/")
        ):
            return ("north_america",)
        if (
            p.startswith("objects/houses/nz/") or
            p.startswith("objects/world-models/houses/nz/")
        ):
            return ("australia_oceania",)
        if (
            "/eu/med/" in p or
            p.startswith("objects/houses/eu/med/") or
            p.startswith("objects/commercial/eu/med/")
        ):
            return ("mediterranean",)
        if (
            p.startswith("objects/houses/eu/") or
            p.startswith("objects/commercial/eu/") or
            p.startswith("objects/industrial/eu/") or
            p.startswith("objects/world-models/houses/eu/") or
            p.startswith("objects/world-models/commercial/eu/") or
            p.startswith("objects/world-models/industrial/eu/")
        ):
            return ("europe",)

    if lib_id == "cdb-library" or p.startswith("cdb-library/"):
        if (
            p.startswith("cdb-library/buildings/samoa/") or
            "/papua_" in p or
            "/hihifo_" in p
        ):
            return ("australia_oceania",)

    # Our shipped library carries the region in a path token: o4sfr/<region>/...
    if lib_id == "o4sfr" or p.startswith("o4sfr/"):
        for token, regions in _O4SFR_PATH_REGION_TOKENS.items():
            if token in p:
                return regions
        return ("generic",)

    # Fall back to the library's static `regions` registration.  Libraries that
    # leave this empty stay "unclassified" and won't enter the placement pool
    # unless a path branch above promotes them.
    entry = CURATED_EXTRA_BUILDING_LIBRARIES.get(lib_id) if lib_id else None
    if entry:
        static_regions = entry.get("regions")
        if static_regions:
            return tuple(static_regions)
        if "regions" in entry:
            # Explicit empty tuple => intentionally global / generic fallback.
            return ("generic",)

    return ()


def _optional_library_rejection_reason(obj_path, lib_id, asset_region):
    """Return None for accepted optional assets, otherwise a compact reason."""
    p = (obj_path or "").replace("\\", "/").lower()
    if _path_has_any(p, OPTIONAL_LIBRARY_SPECIAL_LANDMARK_TOKENS):
        return "special-landmark"
    regions = _optional_library_asset_regions(obj_path, lib_id)
    if not regions:
        return "unclassified"
    if not _optional_asset_region_match(regions, asset_region):
        return "region-mismatch"
    return None


def _optional_library_audit_counters(library_exports, enabled_library_ids, asset_region):
    """Return accepted/rejected optional-library candidate counts for diagnostics."""
    counters = {
        "exports": Counter(),
        "building_candidates": Counter(),
        "accepted": Counter(),
        "rejected_unclassified": Counter(),
        "rejected_region_mismatch": Counter(),
        "rejected_special_landmark": Counter(),
        "regions": Counter(),
        "samples": defaultdict(list),
    }
    for _key, exports in _library_exports_by_virtual_path(library_exports).items():
        export = exports[0]
        lib_id = _optional_library_id_for_export(export, enabled_library_ids)
        if not lib_id:
            continue
        counters["exports"][lib_id] += 1
        obj_path = export.virtual_path
        if not _is_optional_library_building_candidate(obj_path):
            continue
        counters["building_candidates"][lib_id] += 1
        reason = _optional_library_rejection_reason(obj_path, lib_id, asset_region)
        if reason is None:
            counters["accepted"][lib_id] += 1
            regions = _optional_library_asset_regions(obj_path, lib_id)
            for region in regions:
                counters["regions"][(lib_id, region)] += 1
            sample_key = (lib_id, "accepted")
        else:
            key = "rejected_" + reason.replace("-", "_")
            counters[key][lib_id] += 1
            sample_key = (lib_id, reason)
        if len(counters["samples"][sample_key]) < 5:
            counters["samples"][sample_key].append(obj_path)
    return counters


def _describe_optional_library_diagnostics(library_exports, enabled_library_ids,
                                           asset_region, extra_asset_pools):
    """Return a compact optional-library scan/filter summary for logs."""
    enabled_library_ids = tuple(enabled_library_ids or ())
    if not enabled_library_ids:
        return "disabled"
    counters = _optional_library_audit_counters(
        library_exports,
        enabled_library_ids,
        asset_region,
    )
    exports = sum(counters["exports"].values())
    candidates = sum(counters["building_candidates"].values())
    accepted = sum(counters["accepted"].values())
    pooled_by_source = Counter(
        asset.get("source") or "unknown"
        for pool in (extra_asset_pools or {}).values()
        for asset in pool
    )
    pooled = sum(pooled_by_source.values())
    if not exports:
        return (
            f"enabled={','.join(enabled_library_ids)}  "
            "exports=0  packages=not-found-or-no-library-exports"
        )
    rejected_bits = []
    for key, label in (
        ("rejected_unclassified", "unclassified"),
        ("rejected_region_mismatch", "region-mismatch"),
        ("rejected_special_landmark", "special-landmark"),
    ):
        value = sum(counters[key].values())
        if value:
            rejected_bits.append(f"{label}:{value}")
    pooled_bits = ",".join(
        f"{source}:{count}"
        for source, count in sorted(pooled_by_source.items())
    ) or "none"
    return (
        f"enabled={','.join(enabled_library_ids)}  "
        f"exports={exports}  candidates={candidates}  "
        f"accepted={accepted}  pooled={pooled}({pooled_bits})  "
        f"rejected={','.join(rejected_bits) or 'none'}"
    )


def _is_optional_library_building_candidate(obj_path):
    p = (obj_path or "").replace("\\", "/").lower()
    if not p.endswith(".obj"):
        return False
    if _is_excluded_building_filler_asset(p):
        return False
    if _path_has_any(p, GENERIC_BUILDING_EXCLUDE_TOKENS):
        return False
    return _path_has_any(p, OPTIONAL_LIBRARY_INCLUDE_TOKENS)


def _build_optional_library_asset_pools(custom_scenery_dir=None, library_exports=None,
                                        enabled_library_ids=None, cache_dir=None,
                                        asset_region=None):
    """Return optional curated installed-library building assets."""
    asset_pools = {cls: [] for cls in BLD_PLACEMENT_CLASSES}
    enabled_library_ids = tuple(
        enabled_library_ids
        if enabled_library_ids is not None else _enabled_extra_library_ids()
    )
    if not enabled_library_ids:
        return asset_pools
    if library_exports is None and custom_scenery_dir:
        library_exports = _scan_runtime_library_exports(
            custom_scenery_dir,
            extra_library_ids=enabled_library_ids,
        )
    grouped = _library_exports_by_virtual_path(library_exports)
    for key, exports in grouped.items():
        export = exports[0]
        obj_path = export.virtual_path
        lib_id = _optional_library_id_for_export(export, enabled_library_ids)
        if not lib_id:
            continue
        if not _is_optional_library_building_candidate(obj_path):
            continue
        if _optional_library_rejection_reason(obj_path, lib_id, asset_region):
            continue
        height_m = _object_estimated_height_m(obj_path)
        if _is_very_tall_building_asset(obj_path, height_m):
            continue
        bounds_m = _measured_bounds_for_exports(exports, cache_dir)
        if not _footprint_within_limits(bounds_m, max_area_m2=7_000.0, max_side_m=110.0):
            continue
        regions = _optional_library_asset_regions(obj_path, lib_id)
        region_priority = _region_priority_for_asset_regions(
            regions, asset_region
        )
        _append_object_asset(
            asset_pools,
            obj_path,
            bounds_m,
            CURATED_EXTRA_BUILDING_LIBRARIES.get(lib_id, {}).get(
                "label", export.package_name
            ),
            region_priority=region_priority,
        )
    return asset_pools


def _asset_pools_signature(asset_pools):
    """Return a compact signature of active placement assets for cache keys."""
    rows = []
    for zone_class in BLD_PLACEMENT_CLASSES:
        for asset in asset_pools.get(zone_class, ()):
            bounds = asset.get("bounds_m")
            rows.append((
                int(zone_class),
                asset.get("kind", ""),
                asset.get("path", ""),
                asset.get("source", ""),
                tuple(round(float(v), 3) for v in bounds) if bounds is not None else (),
                round(float(asset.get("height_m") or 0.0), 3),
            ))
    digest = hashlib.sha1(repr(sorted(rows)).encode("utf-8")).hexdigest()
    return digest, len(rows)


def _describe_asset_pool_counts(asset_pools):
    """Return compact per-class object asset counts for diagnostics."""
    parts = []
    for zone_class in BLD_PLACEMENT_CLASSES:
        pool = [
            asset for asset in asset_pools.get(zone_class, ())
            if asset.get('kind') == 'object'
        ]
        if not pool:
            continue
        sources = {}
        for asset in pool:
            source = asset.get('source') or 'unknown'
            sources[source] = sources.get(source, 0) + 1
        source_bits = ",".join(
            f"{source}:{count}"
            for source, count in sorted(sources.items())
        )
        parts.append(
            f"{BLD_CLASS_LABELS.get(zone_class, zone_class)}={len(pool)}({source_bits})"
        )
    return "  ".join(parts)


def _build_sfd_asset_pools(
    tile_lat,
    tile_lon,
    asset_region=None,
    custom_scenery_dir=None,
    library_exports=None,
    cache_dir=None,
):
    """Return strict tree-free SFD candidates grouped by footprint class."""
    asset_pools = {cls: [] for cls in BLD_PLACEMENT_CLASSES}
    seen_paths = set()
    region = asset_region or _asset_region(tile_lat, tile_lon)
    known_entries = [
        (obj_path, _bounds_for_object_path(obj_path))
        for obj_path in _sfd_catalog_paths(tile_lat, tile_lon, region)
    ]
    scanned_entries = _scanned_sfd_catalog_entries(
        region,
        library_exports=library_exports,
        custom_scenery_dir=custom_scenery_dir,
        cache_dir=cache_dir,
    )
    for obj_path, bounds_m in known_entries + list(scanned_entries):
        if obj_path in seen_paths:
            continue
        seen_paths.add(obj_path)
        if bounds_m is None:
            continue
        _append_object_asset(asset_pools, obj_path, bounds_m, 'SFD Global')
    return asset_pools


def _build_default_asset_pools(tile_lat=45.0, tile_lon=7.0, asset_region=None):
    """Return default X-Plane candidates grouped by footprint placement class."""
    asset_pools = {cls: [] for cls in BLD_PLACEMENT_CLASSES}
    for zone_class in BLD_PLACEMENT_CLASSES:
        for facade_path in DEFAULT_FACADE_VARIANTS_BY_CLASS.get(
            zone_class, (DEFAULT_FACADE_PATHS[zone_class],)
        ):
            height_m = DEFAULT_FACADE_HEIGHT_M[zone_class]
            if _is_very_tall_building_asset(facade_path, height_m):
                continue
            asset_pools[zone_class].append({
                'kind': 'facade',
                'path': facade_path,
                'bounds_m': DEFAULT_FACADE_BOUNDS[zone_class],
                'height_m': height_m,
                'source': 'Default X-Plane',
                'requires_residential_context': False,
            })
    seen_paths = set()
    for obj_path in _default_object_catalog_paths(tile_lat, tile_lon, asset_region):
        if obj_path in seen_paths:
            continue
        seen_paths.add(obj_path)
        dims = _default_object_dims(obj_path)
        if dims is None:
            continue
        _append_object_asset(
            asset_pools,
            obj_path,
            _bounds_from_dimensions(dims[0], dims[1]),
            'Default X-Plane',
        )
    return asset_pools


def _simheaven_zone_class(width_m, depth_m):
    """Classify a simHeaven object into footprint placement buckets."""
    return _class_for_footprint(_bounds_from_dimensions(width_m, depth_m))


def _build_simheaven_asset_pools(
    simheaven_objects=None,
    tile_lat=45.0,
    tile_lon=7.0,
    asset_region=None,
    custom_scenery_dir=None,
    library_exports=None,
):
    """Return simHeaven object candidates grouped by placement size class."""
    asset_pools = {cls: [] for cls in BLD_PLACEMENT_CLASSES}
    seen_paths = set()
    if library_exports is None and custom_scenery_dir:
        library_exports = _scan_runtime_library_exports(
            custom_scenery_dir,
            include_simheaven=True,
        )
    exported_paths = _exported_simheaven_catalog_paths(library_exports)
    catalog_paths = (
        tuple(_simheaven_catalog_paths(tile_lat, tile_lon, asset_region)) +
        _scanned_simheaven_catalog_paths(
            library_exports=library_exports,
            custom_scenery_dir=custom_scenery_dir,
        )
    )
    for obj_path in catalog_paths:
        if obj_path in seen_paths:
            continue
        if not _is_repeatable_simheaven_asset(obj_path):
            continue
        if exported_paths is not None and obj_path not in exported_paths:
            continue
        seen_paths.add(obj_path)
        dims = _simheaven_object_dims(obj_path)
        _append_object_asset(
            asset_pools,
            obj_path,
            _bounds_from_dimensions(dims[0], dims[1]),
            'simHeaven',
        )
    for obj in simheaven_objects or ():
        obj_path = obj.get('path')
        if not obj_path or obj_path in seen_paths:
            continue
        if not _is_repeatable_simheaven_asset(obj_path):
            continue
        if exported_paths is not None and obj_path not in exported_paths:
            continue
        seen_paths.add(obj_path)
        _append_object_asset(
            asset_pools,
            obj_path,
            _bounds_from_dimensions(obj['w_m'], obj['h_m']),
            'simHeaven',
        )
    return asset_pools


def _merge_asset_pools(*pool_maps):
    """Merge multiple ``{class: [asset, ...]}`` mappings into one pool map."""
    merged = {cls: [] for cls in BLD_PLACEMENT_CLASSES}
    for pool_map in pool_maps:
        if not pool_map:
            continue
        for zone_class in BLD_PLACEMENT_CLASSES:
            merged[zone_class].extend(pool_map.get(zone_class, ()))
    return merged


def _find_library_export(custom_scenery_dir, library_prefix):
    """Return True if any installed scenery package exports the requested prefix."""
    custom_scenery_dir = resolve_custom_scenery_dir(custom_scenery_dir)
    if not custom_scenery_dir or not os.path.isdir(custom_scenery_dir):
        return False
    library_prefix = library_prefix.lower()
    for entry_name in os.listdir(custom_scenery_dir):
        library_txt = os.path.join(custom_scenery_dir, entry_name, 'library.txt')
        if not os.path.isfile(library_txt):
            continue
        try:
            with open(library_txt, 'r', encoding='utf-8', errors='ignore') as handle:
                if library_prefix in handle.read().lower():
                    return True
        except Exception:
            continue
    return False


def _describe_asset_sources(default_available, sfd_available, simheaven_available,
                            extra_library_labels=()):
    """Return a short user-facing summary of building asset source selection."""
    parts = []
    if default_available:
        parts.append('default')
    if sfd_available:
        parts.append('SFD')
    if simheaven_available:
        parts.append('simHeaven')
    parts.extend(label for label in (extra_library_labels or ()) if label)
    if not parts:
        return 'none'
    return ', '.join(parts)


def _residential_roads(roads):
    """Return local-street OSM roads that best match residential fabric."""
    return [
        road for road in (roads or [])
        if road.get('type') in RESIDENTIAL_HIGHWAY_TYPES
    ]


def _build_residential_area_mask(residential_polys, residential_roads,
                                 lat_n, lat_s, lon_w, lon_e,
                                 img_h, img_w, m_per_px):
    """Build a DDS-local mask that marks where small houses may be placed.

    Returns ``(mask, source_label)`` where ``mask`` is a uint8 array or ``None``
    when no residential guidance is available for this DDS.
    """
    if residential_polys:
        return (
            _rasterize_polygons(
                residential_polys, lat_n, lat_s, lon_w, lon_e, img_h, img_w
            ),
            'OSM residential landuse',
        )

    if residential_roads:
        road_mask = _rasterize_roads(
            residential_roads, lat_n, lat_s, lon_w, lon_e, img_h, img_w,
            road_width_px=1,
        )
        road_buffer_px = max(
            RESIDENTIAL_FALLBACK_BUFFER_PX_MIN,
            int(round(RESIDENTIAL_FALLBACK_BUFFER_M / max(m_per_px, 1e-6))),
        )
        dist = cv2.distanceTransform(
            (1 - road_mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        return (dist <= road_buffer_px).astype(np.uint8), 'OSM neighborhood roads'

    return None, 'unavailable'


def _zone_class_from_area_m2(area_m2):
    if area_m2 < ZL16_SMALL_ZONE_M2:
        return BLD_CLASS_SMALL_RESIDENTIAL
    if area_m2 < ZL16_COMPACT_ZONE_M2:
        return BLD_CLASS_COMPACT_RESIDENTIAL
    if area_m2 < ZL16_MEDIUM_ZONE_M2:
        return BLD_CLASS_MEDIUM
    if area_m2 < ZL16_SMALL_APARTMENT_ZONE_M2:
        return BLD_CLASS_SMALL_APARTMENT
    if area_m2 < ZL16_APARTMENT_BLOCK_ZONE_M2:
        return BLD_CLASS_APARTMENT_BLOCK
    return BLD_CLASS_LARGE


def _image_edge_mask_for_zone_classification(img, cc_labels):
    if img is None:
        return None
    try:
        if img.ndim == 3:
            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
        else:
            gray = img.astype(np.uint8, copy=False)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(grad_x, grad_y)
        zone_grad = grad[cc_labels != 0]
        if zone_grad.size == 0:
            return None
        threshold = max(24.0, float(np.percentile(zone_grad, 72.0)))
        return grad >= threshold
    except Exception:
        return None


def _classify_building_zones_zl16(
    labels,
    stats,
    valid_labels,
    bld_raw,
    img,
    residential_area_mask,
    m_per_px,
):
    """Classify broad ZL16 building zones by urban fabric, not just blob area."""
    n_labels = int(stats.shape[0])
    label_class = np.zeros(n_labels, dtype=np.uint8)
    feature_counts = {
        'fine_grain_zones': 0,
        'large_roof_zones': 0,
        'apartment_roof_zones': 0,
        'low_density_zones': 0,
    }
    if len(valid_labels) == 0:
        return label_class, feature_counts

    edge_mask = _image_edge_mask_for_zone_classification(img, labels)
    px_area_m2 = max(float(m_per_px) * float(m_per_px), 1e-6)
    min_fragment_px = max(2, int(round(18.0 / px_area_m2)))

    for label in valid_labels:
        label = int(label)
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area_px = int(stats[label, cv2.CC_STAT_AREA])
        if area_px <= 0 or w <= 0 or h <= 0:
            continue

        area_m2 = float(area_px) * px_area_m2
        component = labels[y:y + h, x:x + w] == label
        raw_component = (bld_raw[y:y + h, x:x + w] != 0) & component
        raw_px = int(np.count_nonzero(raw_component))
        raw_density = raw_px / max(float(area_px), 1.0)
        bbox_fill = area_px / max(float(w * h), 1.0)

        fragment_count = 0
        mean_fragment_m2 = 0.0
        max_fragment_m2 = 0.0
        if raw_px:
            n_frag, _, frag_stats, _ = cv2.connectedComponentsWithStats(
                raw_component.astype(np.uint8), connectivity=8
            )
            if n_frag > 1:
                frag_areas = frag_stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
                frag_areas = frag_areas[frag_areas >= min_fragment_px]
                fragment_count = int(frag_areas.size)
                if fragment_count:
                    mean_fragment_m2 = float(np.mean(frag_areas) * px_area_m2)
                    max_fragment_m2 = float(np.max(frag_areas) * px_area_m2)

        frag_density_ha = fragment_count * 10_000.0 / max(area_m2, 1.0)
        edge_density = 0.0
        if edge_mask is not None:
            edge_density = (
                np.count_nonzero(edge_mask[y:y + h, x:x + w] & component) /
                max(float(area_px), 1.0)
            )
        residential_fraction = 0.0
        if residential_area_mask is not None:
            residential_fraction = (
                np.count_nonzero((residential_area_mask[y:y + h, x:x + w] != 0) & component) /
                max(float(area_px), 1.0)
            )

        cls = _zone_class_from_area_m2(area_m2)
        extra_large_roof = max_fragment_m2 >= ZL16_EXTRA_LARGE_ROOF_M2
        large_roof_candidate = (
            extra_large_roof or
            max_fragment_m2 >= ZL16_LARGE_ROOF_M2 or
            mean_fragment_m2 >= ZL16_APARTMENT_ROOF_M2 * 1.4
        )
        large_roof = large_roof_candidate and (
            edge_mask is None or edge_density < 0.22
        )
        apartment_roof = (
            large_roof_candidate or
            max_fragment_m2 >= ZL16_APARTMENT_ROOF_M2 or
            mean_fragment_m2 >= 420.0
        )
        fine_grain = (
            frag_density_ha >= ZL16_FINE_GRAIN_FRAGS_PER_HA and
            max_fragment_m2 < ZL16_APARTMENT_ROOF_M2
        )

        if large_roof:
            feature_counts['large_roof_zones'] += 1
            if residential_fraction >= 0.45:
                cls = BLD_CLASS_APARTMENT_BLOCK
            elif extra_large_roof:
                cls = BLD_CLASS_EXTRA_LARGE
            else:
                cls = BLD_CLASS_LARGE
        elif apartment_roof:
            feature_counts['apartment_roof_zones'] += 1
            cls = max(cls, BLD_CLASS_SMALL_APARTMENT)
        elif fine_grain:
            feature_counts['fine_grain_zones'] += 1
            if residential_fraction >= 0.40 or raw_density >= 0.35:
                if mean_fragment_m2 <= 95.0:
                    cls = BLD_CLASS_TINY_RESIDENTIAL
                elif mean_fragment_m2 <= 180.0:
                    cls = BLD_CLASS_SMALL_RESIDENTIAL
                elif mean_fragment_m2 <= 300.0:
                    cls = BLD_CLASS_COMPACT_RESIDENTIAL
                elif mean_fragment_m2 <= 520.0:
                    cls = BLD_CLASS_MEDIUM
                else:
                    cls = BLD_CLASS_SMALL_APARTMENT
                if raw_density > 0.62 and cls < BLD_CLASS_MEDIUM:
                    cls += 1
            else:
                cls = max(cls, BLD_CLASS_MEDIUM)

        if raw_density < 0.28 and not large_roof and edge_density < 0.20:
            feature_counts['low_density_zones'] += 1
            cls = min(cls, BLD_CLASS_COMPACT_RESIDENTIAL)
        elif raw_density > 0.62 and area_m2 >= ZL16_SMALL_APARTMENT_ZONE_M2:
            cls = max(cls, BLD_CLASS_SMALL_APARTMENT)

        if bbox_fill < 0.18 and not large_roof:
            cls = min(cls, BLD_CLASS_MEDIUM)

        label_class[label] = int(cls)

    return label_class, feature_counts


def _roof_fragment_class(area_m2, max_side_m, fill_ratio):
    """Classify one raw roof fragment before zone cleanup merges it into a blob."""
    area_m2 = float(area_m2)
    max_side_m = float(max_side_m)
    if area_m2 <= 100.0 and max_side_m <= 16.0:
        return BLD_CLASS_TINY_RESIDENTIAL
    if area_m2 <= 190.0 and max_side_m <= 22.0:
        return BLD_CLASS_SMALL_RESIDENTIAL
    if area_m2 <= 320.0 and max_side_m <= 30.0:
        return BLD_CLASS_COMPACT_RESIDENTIAL
    if area_m2 <= 650.0 and max_side_m <= 38.0:
        return BLD_CLASS_MEDIUM
    if area_m2 <= 1_100.0 and max_side_m <= 48.0:
        return BLD_CLASS_SMALL_APARTMENT
    if area_m2 <= 2_200.0 and max_side_m <= 65.0:
        return BLD_CLASS_APARTMENT_BLOCK
    if area_m2 <= 7_000.0 and max_side_m <= 115.0:
        return BLD_CLASS_LARGE
    return BLD_CLASS_EXTRA_LARGE


def _checkpoint_signature(path):
    """Return a small cache signature for a checkpoint file."""
    if not path:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    return {
        'path': os.path.abspath(path),
        'size': int(st.st_size),
        'mtime_ns': int(getattr(st, 'st_mtime_ns', int(st.st_mtime * 1_000_000_000))),
    }


def _yolo_obb_cache_key(
    fname, img_w, img_h, checkpoint, imgsz, stride, conf, iou, max_det,
    batch_size=1, fused=False, analysis_signature=None, analysis_target_zl=None,
):
    return {
        'version': YOLO_OBB_CACHE_VERSION,
        'fname': str(fname),
        'image_size': (int(img_w), int(img_h)),
        'checkpoint': _checkpoint_signature(checkpoint),
        'imgsz': int(imgsz),
        'stride': int(stride),
        'conf': round(float(conf), 6),
        'iou': round(float(iou), 6),
        'max_det': int(max_det),
        'batch_size': int(batch_size),
        'fused': bool(fused),
        'analysis_target_zl': (
            None if analysis_target_zl is None else int(analysis_target_zl)
        ),
        'analysis': analysis_signature,
    }


def _load_yolo_obb_cache(cache_path, key):
    if not cache_path or not os.path.exists(cache_path):
        return None
    try:
        import pickle
        with open(cache_path, 'rb') as handle:
            data = pickle.load(handle)
        if data.get('key') == key:
            return data.get('detections')
    except Exception:
        return None
    return None


def _save_yolo_obb_cache(cache_path, key, detections):
    if not cache_path:
        return
    try:
        import pickle
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, 'wb') as handle:
            pickle.dump({'key': key, 'detections': detections}, handle)
    except Exception:
        pass


def _load_yolo_obb_model(checkpoint, *, fuse=False):
    from ultralytics import YOLO
    model = YOLO(str(checkpoint))
    if fuse and hasattr(model, 'fuse'):
        model.fuse()
    return model


def _image_resampling_lanczos():
    resampling = getattr(Image, "Resampling", Image)
    return getattr(resampling, "LANCZOS")


def _safe_cache_token(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "unknown")).strip("_") or "unknown"


def _yolo_analysis_size(width, height, source_zl, target_zl=YOLO_ANALYSIS_TARGET_ZL):
    """Return the RGB image size that presents this footprint at target-ZL scale."""
    width = int(width)
    height = int(height)
    source_zl = int(source_zl)
    target_zl = int(target_zl)
    if source_zl <= target_zl:
        return max(1, width), max(1, height)
    scale = 2 ** (source_zl - target_zl)
    return (
        max(1, int(round(width / scale))),
        max(1, int(round(height / scale))),
    )


def _yolo_analysis_signature(fname, source_path, img_w, img_h, source_zl, target_zl):
    analysis_w, analysis_h = _yolo_analysis_size(
        img_w, img_h, source_zl, target_zl=target_zl
    )
    return {
        'source_fname': str(fname),
        'source_path': None if source_path is None else os.path.abspath(os.fspath(source_path)),
        'source_signature': _checkpoint_signature(source_path) if source_path else None,
        'source_size': [int(img_w), int(img_h)],
        'source_zl': int(source_zl),
        'target_zl': int(target_zl),
        'analysis_size': [int(analysis_w), int(analysis_h)],
    }


def build_yolo_zl16_analysis_image(
    tex_dir,
    til_y_top,
    til_x_left,
    provider,
    source_zl,
    target_zl=YOLO_ANALYSIS_TARGET_ZL,
    cache_dir=None,
    source_image=None,
    source_path=None,
    bounds=None,
):
    """Create or reuse a target-ZL RGB PNG for trained YOLO inference.

    The returned image covers exactly the same geographic footprint as the
    source texture. Higher-ZL inputs are downsampled so the detector sees the
    same approximate ground scale as a ZL16 texture, while the original scenery
    DDS remains untouched.
    """
    if source_image is None:
        if source_path is None:
            source_path = os.path.join(
                os.fspath(tex_dir),
                f"{int(til_y_top)}_{int(til_x_left)}_{provider}{int(source_zl):02d}.dds",
            )
        try:
            with Image.open(source_path) as image:
                source_image = np.asarray(image.convert("RGB"), dtype=np.uint8)
        except Exception as exc:
            raise FileNotFoundError(f"Could not load YOLO analysis source {source_path}: {exc}") from exc

    source_arr = np.asarray(source_image, dtype=np.uint8)
    if source_arr.ndim != 3 or source_arr.shape[2] < 3:
        raise ValueError("YOLO analysis source image must be an RGB-like array")
    source_arr = source_arr[:, :, :3]
    source_h, source_w = source_arr.shape[:2]
    analysis_w, analysis_h = _yolo_analysis_size(
        source_w, source_h, source_zl, target_zl=target_zl
    )

    root = (
        os.path.join(os.getcwd(), "tmp", "sfr_yolo_zl16_cache")
        if cache_dir is None
        else os.path.join(os.fspath(cache_dir), "yolo_zl16_analysis")
    )
    os.makedirs(root, exist_ok=True)
    provider_token = _safe_cache_token(provider)
    stem = (
        f"{int(til_y_top)}_{int(til_x_left)}_{provider_token}{int(source_zl):02d}"
        f"_to_zl{int(target_zl):02d}_{analysis_w}x{analysis_h}"
    )
    png_path = os.path.join(root, stem + ".png")
    meta_path = os.path.join(root, stem + ".json")
    source_sig = _checkpoint_signature(source_path) if source_path else None
    expected_meta = {
        'version': YOLO_ANALYSIS_CACHE_VERSION,
        'source_path': None if source_path is None else os.path.abspath(os.fspath(source_path)),
        'source_signature': source_sig,
        'source_zl': int(source_zl),
        'target_zl': int(target_zl),
        'til_y_top': int(til_y_top),
        'til_x_left': int(til_x_left),
        'provider': str(provider),
        'source_size': [int(source_w), int(source_h)],
        'analysis_size': [int(analysis_w), int(analysis_h)],
        'bounds': None if bounds is None else [float(v) for v in bounds],
    }
    try:
        if os.path.exists(png_path) and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as handle:
                if json.load(handle) == expected_meta:
                    return png_path
    except Exception:
        pass

    image = Image.fromarray(np.ascontiguousarray(source_arr), mode="RGB")
    if image.size != (analysis_w, analysis_h):
        image = image.resize((analysis_w, analysis_h), _image_resampling_lanczos())
    image.save(png_path)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(expected_meta, handle, indent=2, sort_keys=True)
    return png_path


def _load_yolo_analysis_image(
    tex_dir,
    fname,
    til_y_top,
    til_x_left,
    provider,
    source_zl,
    target_zl,
    cache_dir,
    source_image,
    source_path,
    bounds,
):
    analysis_path = build_yolo_zl16_analysis_image(
        tex_dir,
        til_y_top,
        til_x_left,
        provider,
        source_zl,
        target_zl=target_zl,
        cache_dir=cache_dir,
        source_image=source_image,
        source_path=source_path,
        bounds=bounds,
    )
    with Image.open(analysis_path) as image:
        analysis_image = np.asarray(image.convert("RGB"), dtype=np.uint8)
    source_h, source_w = np.asarray(source_image).shape[:2]
    analysis_h, analysis_w = analysis_image.shape[:2]
    scale_x = float(source_w) / max(1.0, float(analysis_w))
    scale_y = float(source_h) / max(1.0, float(analysis_h))
    return {
        'path': analysis_path,
        'image': analysis_image,
        'scale_x': scale_x,
        'scale_y': scale_y,
        'signature': _yolo_analysis_signature(
            fname, source_path, source_w, source_h, source_zl, target_zl
        ),
    }


def _scale_yolo_detections_to_image(detections, scale_x, scale_y, img_w, img_h):
    if not detections or (abs(scale_x - 1.0) < 1e-9 and abs(scale_y - 1.0) < 1e-9):
        return detections
    scaled = []
    for detection in detections:
        item = dict(detection)
        points = np.asarray(item.get('points', ()), dtype=np.float32).reshape(-1, 2)
        if points.size:
            points[:, 0] = np.clip(points[:, 0] * float(scale_x), 0, max(0, img_w - 1))
            points[:, 1] = np.clip(points[:, 1] * float(scale_y), 0, max(0, img_h - 1))
            item['points'] = points.tolist()
        center = np.asarray(item.get('center', ()), dtype=np.float32).reshape(-1)
        if center.size >= 2:
            item['center'] = [
                float(np.clip(center[0] * float(scale_x), 0, max(0, img_w - 1))),
                float(np.clip(center[1] * float(scale_y), 0, max(0, img_h - 1))),
            ]
        scaled.append(item)
    return scaled


def _iter_yolo_crops(image, stride):
    img_h, img_w = image.shape[:2]
    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            crop = image[y:min(y + stride, img_h), x:min(x + stride, img_w)]
            if crop.shape[0] != stride or crop.shape[1] != stride:
                padded = np.zeros((stride, stride, 3), dtype=image.dtype)
                padded[:crop.shape[0], :crop.shape[1]] = crop
                crop = padded
            yield x, y, crop


def _iter_yolo_crop_batches(image, stride, batch_size):
    batch = []
    for item in _iter_yolo_crops(image, stride):
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _yolo_model_class_count(model):
    names = getattr(model, 'names', None)
    if names is None and getattr(model, 'model', None) is not None:
        names = getattr(model.model, 'names', None)
    if isinstance(names, dict):
        return len(names)
    if isinstance(names, (list, tuple)):
        return len(names)
    return None


def _decode_yolo_obb_detection_class(model_cls_int, model_class_count=None):
    """Decode legacy 8-class or height-expanded YOLO OBB classes.

    Height-expanded checkpoints are treated as placement-class models at
    runtime; facade heights are randomized from regional/class asset priors.
    """
    height_model_class_count = len(BLD_PLACEMENT_CLASSES) * YOLO_HEIGHT_BIN_COUNT
    if int(model_class_count or 0) == height_model_class_count:
        placement_index, height_bin = divmod(int(model_cls_int), YOLO_HEIGHT_BIN_COUNT)
        if 0 <= placement_index < len(BLD_PLACEMENT_CLASSES):
            return (
                int(BLD_PLACEMENT_CLASSES[placement_index]),
                None,
            )

    mapped_cls = int(model_cls_int) + 1
    if mapped_cls in BLD_PLACEMENT_CLASSES:
        return mapped_cls, None
    return None, None


def _yolo_obb_detection_from_points(
    points,
    confidence,
    cls,
    img_w,
    img_h,
    m_per_px,
    xywhr=None,
    model_class_count=None,
):
    """Convert one YOLO OBB polygon into overlay placement evidence."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    valid = (
        (pts[:, 0] >= 0) & (pts[:, 0] < img_w) &
        (pts[:, 1] >= 0) & (pts[:, 1] < img_h)
    )
    center = pts.mean(axis=0)
    if not np.any(valid) and not (0 <= center[0] < img_w and 0 <= center[1] < img_h):
        return None

    clipped = pts.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, max(0, img_w - 1))
    clipped[:, 1] = np.clip(clipped[:, 1], 0, max(0, img_h - 1))
    area_px = abs(float(cv2.contourArea(clipped.astype(np.float32))))
    if area_px <= 1.0:
        return None

    # Use the polygon corners for heading rather than Ultralytics ``xywhr``.
    # ``xywhr`` uses a normalized OBB representation whose angle can flip with
    # width/height canonicalization; facades were aligned because they used the
    # polygon directly, so object headings must follow the same geometry.
    edges = np.roll(clipped, -1, axis=0) - clipped
    edge_lengths = np.linalg.norm(edges, axis=1)
    long_edge_index = int(np.argmax(edge_lengths))
    long_vec = edges[long_edge_index]
    img_angle = math.degrees(math.atan2(float(long_vec[1]), float(long_vec[0])))
    max_side_px = float(edge_lengths[long_edge_index])
    min_side_px = float(edge_lengths[(long_edge_index + 1) % 4])
    if min_side_px > max_side_px:
        max_side_px, min_side_px = min_side_px, max_side_px
    # Image pixels use +Y down, while X-Plane OBJ headings are compass-style
    # clockwise from north. Horizontal/vertical axes are unchanged by the sign
    # error, but diagonals must use image_angle + 90 so object headings follow
    # the YOLO OBB rather than its mirror image.
    heading = (img_angle + 90.0) % 180.0
    max_side_m = float(max_side_px) * float(m_per_px)
    min_side_m = float(min_side_px) * float(m_per_px)
    area_m2 = area_px * float(m_per_px) * float(m_per_px)
    model_cls_int = int(cls)
    placement_cls, height_m = _decode_yolo_obb_detection_class(
        model_cls_int,
        model_class_count=model_class_count,
    )
    if placement_cls is not None:
        placement_cls = int(placement_cls)
    else:
        placement_cls = _roof_fragment_class(area_m2, max_side_m, 1.0)
    height_m = float(DEFAULT_FACADE_HEIGHT_M.get(placement_cls, 8.0))

    return {
        'points': clipped.tolist(),
        'center': [
            float(np.clip(center[0], 0, max(0, img_w - 1))),
            float(np.clip(center[1], 0, max(0, img_h - 1))),
        ],
        'heading': float(heading),
        'confidence': float(confidence),
        'model_class': model_cls_int,
        'area_m2': float(area_m2),
        'max_side_m': float(max_side_m),
        'length_m': float(max_side_m),
        'width_m': float(min_side_m),
        'placement_class': int(placement_cls),
        'height_m': float(height_m),
    }


def _append_yolo_result_detections(
    detections,
    result,
    ox,
    oy,
    *,
    img_w,
    img_h,
    m_per_px,
    model_class_count,
):
    obb = corners_tensor = corners = xywhr_tensor = xywhr = confs = classes = None
    try:
        obb = getattr(result, 'obb', None)
        corners_tensor = None if obb is None else getattr(obb, 'xyxyxyxy', None)
        if corners_tensor is None:
            return
        corners = corners_tensor.detach().cpu().numpy()
        xywhr_tensor = getattr(obb, 'xywhr', None)
        xywhr = (
            None if xywhr_tensor is None
            else xywhr_tensor.detach().cpu().numpy()
        )
        confs = (
            obb.conf.detach().cpu().numpy()
            if obb.conf is not None else np.ones((len(corners),), dtype=float)
        )
        classes = (
            obb.cls.detach().cpu().numpy()
            if obb.cls is not None else np.zeros((len(corners),), dtype=float)
        )
        if xywhr is not None and len(xywhr) != len(corners):
            xywhr = None
        for det_idx, (points, score, cls) in enumerate(zip(corners, confs, classes)):
            shifted = np.asarray(points, dtype=np.float32).reshape(4, 2)
            shifted[:, 0] += float(ox)
            shifted[:, 1] += float(oy)
            shifted_xywhr = None
            if xywhr is not None:
                shifted_xywhr = np.asarray(xywhr[det_idx], dtype=np.float32).copy()
                shifted_xywhr[0] += float(ox)
                shifted_xywhr[1] += float(oy)
            detection = _yolo_obb_detection_from_points(
                shifted, score, cls, img_w, img_h, m_per_px,
                xywhr=shifted_xywhr,
                model_class_count=model_class_count,
            )
            if detection is not None:
                detections.append(detection)
    finally:
        del corners, xywhr, confs, classes, corners_tensor, xywhr_tensor, obb, result


def _run_yolo_obb_inference(model, image, *, imgsz, stride, conf, iou, max_det,
                            device=None, m_per_px=1.0, batch_size=1,
                            return_metadata=False):
    img_h, img_w = image.shape[:2]
    model_class_count = _yolo_model_class_count(model)

    def _detect(effective_batch):
        detections = []
        if effective_batch <= 1:
            for ox, oy, crop in _iter_yolo_crops(image, int(stride)):
                crop_bgr = np.ascontiguousarray(crop[..., ::-1])
                with torch.inference_mode():
                    results = model.predict(
                        source=crop_bgr,
                        imgsz=int(imgsz),
                        conf=float(conf),
                        iou=float(iou),
                        max_det=int(max_det),
                        device=device,
                        verbose=False,
                        stream=True,
                    )
                    for result in results:
                        _append_yolo_result_detections(
                            detections, result, ox, oy,
                            img_w=img_w, img_h=img_h, m_per_px=m_per_px,
                            model_class_count=model_class_count,
                        )
                    del results
                del crop_bgr
        else:
            for batch in _iter_yolo_crop_batches(image, int(stride), effective_batch):
                offsets = [(ox, oy) for ox, oy, _ in batch]
                crops = [np.ascontiguousarray(crop[..., ::-1]) for _, _, crop in batch]
                with torch.inference_mode():
                    results = model.predict(
                        source=crops,
                        imgsz=int(imgsz),
                        conf=float(conf),
                        iou=float(iou),
                        max_det=int(max_det),
                        device=device,
                        verbose=False,
                        stream=True,
                        batch=effective_batch,
                    )
                    for (ox, oy), result in zip(offsets, results):
                        _append_yolo_result_detections(
                            detections, result, ox, oy,
                            img_w=img_w, img_h=img_h, m_per_px=m_per_px,
                            model_class_count=model_class_count,
                        )
                    del results, crops, offsets
        return detections

    batch_size = max(1, int(batch_size or 1))
    effective_batch_size = batch_size
    fell_back = False
    try:
        detections = _detect(batch_size)
    except RuntimeError as exc:
        if batch_size > 1 and SEGFORMER.is_cuda_oom(exc):
            # Discard any partial batched results and restart from scratch so
            # the fallback output is identical to a pure batch-1 run.
            print(
                f"    YOLO OBB batch={batch_size} hit CUDA OOM; "
                "retrying with batch=1",
                flush=True,
            )
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            detections = _detect(1)
            effective_batch_size = 1
            fell_back = True
        else:
            raise

    detections.sort(key=lambda item: (float(item['area_m2']), -float(item['confidence'])))
    if return_metadata:
        return {
            'detections': detections,
            'requested_batch': batch_size,
            'effective_batch': effective_batch_size,
            'fell_back': fell_back,
        }
    return detections


def _build_yolo_guidance(detections, img_h, img_w, m_per_px):
    """Build nearest-YOLO-center guidance for fallback class and heading."""
    if not detections:
        return None

    center_mask = np.ones((img_h, img_w), dtype=np.uint8)
    center_to_det = {}
    for det_idx, det in enumerate(detections, 1):
        cx = int(round(float(det['center'][0])))
        cy = int(round(float(det['center'][1])))
        if 0 <= cx < img_w and 0 <= cy < img_h:
            center_mask[cy, cx] = 0
            center_to_det[(cx, cy)] = det_idx

    if not center_to_det:
        return None

    n_centers, center_components = cv2.connectedComponents(
        (center_mask == 0).astype(np.uint8), connectivity=8
    )
    class_by_label = np.zeros(n_centers, dtype=np.uint8)
    heading_by_label = np.full(n_centers, np.nan, dtype=np.float32)
    height_by_label = np.full(n_centers, np.nan, dtype=np.float32)
    confidence_by_label = np.zeros(n_centers, dtype=np.float32)
    area_by_label_m2 = np.zeros(n_centers, dtype=np.float32)
    center_by_label = np.full((n_centers, 2), np.nan, dtype=np.float32)
    points_by_label = [None] * n_centers
    for (cx, cy), det_idx in center_to_det.items():
        label = int(center_components[cy, cx])
        if label <= 0:
            continue
        det = detections[det_idx - 1]
        if float(det['confidence']) < float(confidence_by_label[label]):
            continue
        class_by_label[label] = int(det['placement_class'])
        heading_by_label[label] = float(det['heading'])
        height_by_label[label] = float(det.get(
            'height_m',
            DEFAULT_FACADE_HEIGHT_M.get(int(det['placement_class']), 8.0),
        ))
        confidence_by_label[label] = float(det['confidence'])
        area_by_label_m2[label] = float(det['area_m2'])
        center_by_label[label] = np.asarray(det['center'], dtype=np.float32)
        points_by_label[label] = np.asarray(det['points'], dtype=np.float32)

    dist_px, nearest_label = cv2.distanceTransformWithLabels(
        center_mask,
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_CCOMP,
    )
    return {
        'distance_px': dist_px.astype(np.float32, copy=False),
        'nearest_label': nearest_label.astype(np.int32, copy=False),
        'class_by_label': class_by_label,
        'heading_by_label': heading_by_label,
        'height_by_label': height_by_label,
        'confidence_by_label': confidence_by_label,
        'area_by_label_m2': area_by_label_m2,
        'center_by_label': center_by_label,
        'points_by_label': points_by_label,
        'max_distance_m': float(YOLO_GUIDANCE_MAX_DISTANCE_M),
        'm_per_px': float(m_per_px),
        'image_size': (int(img_h), int(img_w)),
    }


def _yolo_guidance_label(yolo_guidance, jx, jy, max_distance_m=None):
    if yolo_guidance is None:
        return 0
    label = int(yolo_guidance['nearest_label'][jy, jx])
    class_by_label = yolo_guidance['class_by_label']
    if label <= 0 or label >= class_by_label.shape[0]:
        return 0
    max_dist = (
        float(yolo_guidance.get('max_distance_m', YOLO_GUIDANCE_MAX_DISTANCE_M))
        if max_distance_m is None else float(max_distance_m)
    )
    dist_m = (
        float(yolo_guidance['distance_px'][jy, jx]) *
        float(yolo_guidance.get('m_per_px', 1.0))
    )
    if dist_m > max_dist:
        return 0
    return label


def _yolo_heading_for_candidate(yolo_guidance, jx, jy, max_distance_m=None):
    label = _yolo_guidance_label(yolo_guidance, jx, jy, max_distance_m)
    if label <= 0:
        return float('nan')
    heading = float(yolo_guidance['heading_by_label'][label])
    return heading if not np.isnan(heading) else float('nan')


def _yolo_template_for_candidate(yolo_guidance, jx, jy, max_distance_m=None):
    """Return nearest YOLO OBB footprint translated to a candidate center."""
    label = _yolo_guidance_label(yolo_guidance, jx, jy, max_distance_m)
    if label <= 0:
        return None
    points = yolo_guidance.get('points_by_label', ())
    centers = yolo_guidance.get('center_by_label')
    class_by_label = yolo_guidance.get('class_by_label')
    heading_by_label = yolo_guidance.get('heading_by_label')
    if (
        label >= len(points) or points[label] is None or
        centers is None or label >= centers.shape[0] or
        class_by_label is None or label >= class_by_label.shape[0] or
        heading_by_label is None or label >= heading_by_label.shape[0]
    ):
        return None
    source_center = centers[label]
    if np.any(np.isnan(source_center)):
        return None
    translated = np.asarray(points[label], dtype=np.float32).copy()
    translated[:, 0] += float(jx) - float(source_center[0])
    translated[:, 1] += float(jy) - float(source_center[1])
    img_h, img_w = yolo_guidance.get('image_size', (0, 0))
    if (
        translated[:, 0].min() < 0 or translated[:, 0].max() >= int(img_w) or
        translated[:, 1].min() < 0 or translated[:, 1].max() >= int(img_h)
    ):
        return None
    placement_cls = int(class_by_label[label])
    if placement_cls not in BLD_PLACEMENT_CLASSES:
        return None
    heading = float(heading_by_label[label])
    if np.isnan(heading):
        heading = 0.0
    return np.rint(translated).astype(np.int32), placement_cls, heading


def _angle_delta_180(a_deg, b_deg):
    """Return the unsigned difference between 180-periodic headings."""
    return abs(((float(a_deg) - float(b_deg) + 90.0) % 180.0) - 90.0)


def _yolo_template_metrics(template):
    """Return long/short side lengths and edge direction for a YOLO template."""
    pts = np.asarray(template.get('points', ()), dtype=np.float32)
    if pts.shape != (4, 2):
        return None
    edges = np.roll(pts, -1, axis=0) - pts
    edge_lengths = np.linalg.norm(edges, axis=1)
    long_idx = int(np.argmax(edge_lengths))
    long_len = float(edge_lengths[long_idx])
    short_len = float(edge_lengths[(long_idx + 1) % 4])
    if long_len < 1.0 or short_len < 1.0:
        return None
    if short_len > long_len:
        long_len, short_len = short_len, long_len
    return {
        'long_len': long_len,
        'short_len': short_len,
        'area_px': long_len * short_len,
    }


def _points_from_yolo_heading(center, long_len, short_len, heading):
    """Build an image-space rotated rectangle from YOLO heading and dimensions."""
    cx, cy = np.asarray(center, dtype=np.float32)
    img_angle = math.radians(float(heading) - 90.0)
    u_axis = np.asarray([math.cos(img_angle), math.sin(img_angle)], dtype=np.float32)
    v_axis = np.asarray([-u_axis[1], u_axis[0]], dtype=np.float32)
    hu = u_axis * (float(long_len) * 0.5)
    hv = v_axis * (float(short_len) * 0.5)
    return np.asarray(
        (
            (cx - hu[0] - hv[0], cy - hu[1] - hv[1]),
            (cx + hu[0] - hv[0], cy + hu[1] - hv[1]),
            (cx + hu[0] + hv[0], cy + hu[1] + hv[1]),
            (cx - hu[0] + hv[0], cy - hu[1] + hv[1]),
        ),
        dtype=np.float32,
    )


def _weighted_heading_mean_180(headings, weights):
    headings = np.asarray(headings, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if headings.size == 0:
        return float('nan')
    if not np.any(weights > 0):
        weights = np.ones_like(headings, dtype=np.float64)
    angles = np.deg2rad(headings * 2.0)
    mean = math.atan2(
        float(np.sum(np.sin(angles) * weights)),
        float(np.sum(np.cos(angles) * weights)),
    )
    return (math.degrees(mean) * 0.5) % 180.0


def _consensus_yolo_template(
    zone_templates,
    heading_tol_deg=YOLO_TEMPLATE_HEADING_TOL_DEG,
    shape_rel_tol=YOLO_TEMPLATE_SHAPE_REL_TOL,
    shape_abs_tol_px=YOLO_TEMPLATE_SHAPE_ABS_TOL_PX,
):
    """Return a synthetic zone template from fuzzy heading and shape majorities."""
    records = []
    for template in zone_templates or ():
        metrics = _yolo_template_metrics(template)
        if metrics is None:
            continue
        records.append({
            'template': template,
            'heading': float(template.get('heading', 0.0)) % 180.0,
            'class': int(template.get('class', BLD_CLASS_MEDIUM)),
            'height_m': float(template.get(
                'height_m',
                DEFAULT_FACADE_HEIGHT_M.get(int(template.get('class', BLD_CLASS_MEDIUM)), 8.0),
            )),
            'confidence': max(0.0, float(template.get('confidence', 0.0))),
            **metrics,
        })
    if not records:
        return None

    headings = np.asarray([rec['heading'] for rec in records], dtype=np.float32)
    weights = np.asarray(
        [max(0.01, rec['confidence']) for rec in records], dtype=np.float32
    )
    best_heading_idx = []
    best_heading_key = (-1, -1.0, -1.0)
    for idx, heading in enumerate(headings):
        cluster = [
            cand_idx for cand_idx, cand_heading in enumerate(headings)
            if _angle_delta_180(cand_heading, heading) <= float(heading_tol_deg)
        ]
        key = (
            len(cluster),
            float(np.sum(weights[cluster])),
            float(np.sum([records[cand_idx]['area_px'] for cand_idx in cluster])),
        )
        if key > best_heading_key:
            best_heading_key = key
            best_heading_idx = cluster
    consensus_heading = _weighted_heading_mean_180(
        headings[best_heading_idx], weights[best_heading_idx]
    )

    best_shape_idx = []
    best_shape_key = (-1, -1.0, -1.0)
    for idx, rec in enumerate(records):
        long_tol = max(float(shape_abs_tol_px), float(shape_rel_tol) * rec['long_len'])
        short_tol = max(float(shape_abs_tol_px), float(shape_rel_tol) * rec['short_len'])
        cluster = [
            cand_idx for cand_idx, cand in enumerate(records)
            if (
                cand['class'] == rec['class'] and
                abs(cand['long_len'] - rec['long_len']) <= long_tol and
                abs(cand['short_len'] - rec['short_len']) <= short_tol
            )
        ]
        key = (
            len(cluster),
            float(np.sum(weights[cluster])),
            float(np.sum([records[cand_idx]['area_px'] for cand_idx in cluster])),
        )
        if key > best_shape_key:
            best_shape_key = key
            best_shape_idx = cluster

    shape_records = [records[idx] for idx in best_shape_idx]
    long_len = float(np.median([rec['long_len'] for rec in shape_records]))
    short_len = float(np.median([rec['short_len'] for rec in shape_records]))
    placement_cls = int(shape_records[0]['class'])
    height_m = float(np.median([rec['height_m'] for rec in shape_records]))
    centers = np.asarray(
        [records[idx]['template']['center'] for idx in best_shape_idx],
        dtype=np.float32,
    )
    center = np.mean(centers, axis=0)
    points = _points_from_yolo_heading(center, long_len, short_len, consensus_heading)
    return {
        'center': center.astype(np.float32),
        'points': points,
        'class': placement_cls,
        'height_m': height_m,
        'heading': float(consensus_heading),
        'confidence': float(np.mean([rec['confidence'] for rec in shape_records])),
        'area_px': float(long_len * short_len),
        'consensus_heading_votes': int(len(best_heading_idx)),
        'consensus_shape_votes': int(len(best_shape_idx)),
    }


def _retarget_yolo_template_heading(template, heading):
    """Return the same YOLO shape/class centered at template center with a new heading."""
    metrics = _yolo_template_metrics(template)
    if metrics is None:
        return None
    center = np.asarray(template.get('center', ()), dtype=np.float32)
    if center.shape != (2,) or np.any(np.isnan(center)):
        return None
    retargeted = dict(template)
    retargeted['center'] = center.copy()
    retargeted['heading'] = float(heading) % 180.0
    retargeted['points'] = _points_from_yolo_heading(
        center,
        float(metrics['long_len']),
        float(metrics['short_len']),
        retargeted['heading'],
    )
    return retargeted


def _scaled_template_points(template, jx, jy, long_scale=1.0, short_scale=1.0):
    """Translate a YOLO template and scale long/short axes independently."""
    metrics = _yolo_template_metrics(template)
    center = np.asarray(template.get('center', ()), dtype=np.float32)
    if metrics is None or center.shape != (2,):
        return None
    long_scale = float(np.clip(long_scale, 0.0, 1.0))
    short_scale = float(np.clip(short_scale, 0.0, 1.0))
    target = np.asarray([float(jx), float(jy)], dtype=np.float32)
    return _points_from_yolo_heading(
        target,
        float(metrics['long_len']) * long_scale,
        float(metrics['short_len']) * short_scale,
        float(template.get('heading', 0.0)),
    )


def _template_poly_in_bounds(poly, img_w, img_h):
    return (
        poly is not None and
        poly.shape == (4, 2) and
        float(poly[:, 0].min()) >= 0.0 and
        float(poly[:, 0].max()) < float(img_w) and
        float(poly[:, 1].min()) >= 0.0 and
        float(poly[:, 1].max()) < float(img_h)
    )


def _largest_fitting_yolo_template_poly(
    template,
    jx,
    jy,
    img_w,
    img_h,
    static_occ_mask,
    building_spacing_mask,
    scratch_mask,
    static_occ_integral=None,
    min_scale=0.25,
    iterations=6,
):
    """Return largest-area <= original YOLO footprint that fits current blockers."""
    min_scale = float(np.clip(min_scale, 0.05, 1.0))

    def _fits(long_scale, short_scale):
        poly_f = _scaled_template_points(template, jx, jy, long_scale, short_scale)
        if not _template_poly_in_bounds(poly_f, img_w, img_h):
            return None
        poly_i = np.rint(poly_f).astype(np.int32)
        if abs(float(cv2.contourArea(poly_i.astype(np.float32)))) <= 1.0:
            return None
        if not _poly_fits_with_integral(
            static_occ_mask,
            poly_i,
            scratch_mask=scratch_mask,
            occ_integral=static_occ_integral,
        ):
            return None
        if not _poly_fits(building_spacing_mask, poly_i, scratch_mask):
            return None
        return poly_i

    full = _fits(1.0, 1.0)
    if full is not None:
        return full, (1.0, 1.0)

    def _best_long_for_short(short_scale):
        low_poly = _fits(min_scale, short_scale)
        if low_poly is None:
            return None, 0.0, short_scale
        lo = min_scale
        hi = 1.0
        best_poly = low_poly
        best_long = lo
        for _ in range(max(1, int(iterations))):
            mid = (lo + hi) * 0.5
            mid_poly = _fits(mid, short_scale)
            if mid_poly is None:
                hi = mid
            else:
                lo = mid
                best_long = mid
                best_poly = mid_poly
        return best_poly, float(best_long), float(short_scale)

    def _best_short_for_long(long_scale):
        low_poly = _fits(long_scale, min_scale)
        if low_poly is None:
            return None, long_scale, 0.0
        lo = min_scale
        hi = 1.0
        best_poly = low_poly
        best_short = lo
        for _ in range(max(1, int(iterations))):
            mid = (lo + hi) * 0.5
            mid_poly = _fits(long_scale, mid)
            if mid_poly is None:
                hi = mid
            else:
                lo = mid
                best_short = mid
                best_poly = mid_poly
        return best_poly, float(long_scale), float(best_short)

    # Three targeted binary searches replace the 10-sample grid:
    # (a) shrink long only  (b) shrink short only  (c) uniform shrink.
    # Worst case: 3 × (1 + iterations) = 21 poly fits vs 140 in the grid.
    def _best_uniform():
        low_poly = _fits(min_scale, min_scale)
        if low_poly is None:
            return None, 0.0, 0.0
        lo, hi = min_scale, 1.0
        best_p, best_s = low_poly, lo
        for _ in range(max(1, int(iterations))):
            mid = (lo + hi) * 0.5
            mid_poly = _fits(mid, mid)
            if mid_poly is None:
                hi = mid
            else:
                lo = mid
                best_s = mid
                best_p = mid_poly
        return best_p, float(best_s), float(best_s)

    best_poly = None
    best_scales = (0.0, 0.0)
    best_key = (-1.0, -1.0, -1.0)
    for poly, long_scale, short_scale in (
        _best_long_for_short(1.0),
        _best_short_for_long(1.0),
        _best_uniform(),
    ):
        if poly is not None:
            key = (long_scale * short_scale, max(long_scale, short_scale), min(long_scale, short_scale))
            if key > best_key:
                best_key = key
                best_poly = poly
                best_scales = (long_scale, short_scale)
    return best_poly, best_scales


def _neighbor_yolo_templates_by_zone(cc_labels, cc_stats, valid_labels,
                                     yolo_templates_by_zone, radius_px):
    """Collect original OBB-zone templates for neighboring labels without OBBs.

    This is intentionally one-hop only: zones filled from a neighbor template do
    not become template sources for further zones.
    """
    if not yolo_templates_by_zone or cc_labels is None or cc_labels.size == 0:
        return {}
    source_labels = {int(label) for label in yolo_templates_by_zone if int(label) > 0}
    if not source_labels:
        return {}

    radius_px = max(1, int(radius_px))

    # Pre-compute source-zone bboxes (x0, y0, x1, y1) for fast proximity test.
    src_bboxes = {}
    for sl in source_labels:
        if sl < cc_stats.shape[0]:
            sx0 = int(cc_stats[sl, cv2.CC_STAT_LEFT])
            sy0 = int(cc_stats[sl, cv2.CC_STAT_TOP])
            src_bboxes[sl] = (
                sx0,
                sy0,
                sx0 + int(cc_stats[sl, cv2.CC_STAT_WIDTH]),
                sy0 + int(cc_stats[sl, cv2.CC_STAT_HEIGHT]),
            )

    result = {}
    for label in (int(v) for v in valid_labels):
        if label <= 0 or label in source_labels or label >= cc_stats.shape[0]:
            continue
        x0 = int(cc_stats[label, cv2.CC_STAT_LEFT])
        y0 = int(cc_stats[label, cv2.CC_STAT_TOP])
        zone_w = int(cc_stats[label, cv2.CC_STAT_WIDTH])
        zone_h = int(cc_stats[label, cv2.CC_STAT_HEIGHT])
        if zone_w <= 0 or zone_h <= 0:
            continue
        ex0 = x0 - radius_px
        ey0 = y0 - radius_px
        ex1 = x0 + zone_w + radius_px
        ey1 = y0 + zone_h + radius_px
        zone_cx = x0 + zone_w * 0.5
        zone_cy = y0 + zone_h * 0.5
        best_key = None
        best_templates = ()
        for sl, (sx0, sy0, sx1, sy1) in src_bboxes.items():
            if sx1 >= ex0 and sx0 <= ex1 and sy1 >= ey0 and sy0 <= ey1:
                src_cx = (sx0 + sx1) * 0.5
                src_cy = (sy0 + sy1) * 0.5
                dist2 = (src_cx - zone_cx) ** 2 + (src_cy - zone_cy) ** 2
                key = (float(dist2), int(sl))
                if best_key is None or key < best_key:
                    best_key = key
                    best_templates = yolo_templates_by_zone.get(sl, ())
        if best_templates:
            result[label] = list(best_templates)
    return result


def _nearest_obb_zone_headings(cc_stats, cc_centroids, valid_labels,
                               yolo_templates_by_zone,
                               heading_tol_deg=YOLO_TEMPLATE_HEADING_TOL_DEG,
                               shape_rel_tol=YOLO_TEMPLATE_SHAPE_REL_TOL,
                               shape_abs_tol_px=YOLO_TEMPLATE_SHAPE_ABS_TOL_PX):
    """Return heading evidence from the nearest zone with same-zone OBBs.

    This intentionally ignores placement class.  The OBB-neighbor path is only
    about borrowing local orientation, while spacing/class fit remains on the
    normal candidate-placement path.
    """
    n_labels = cc_stats.shape[0] if cc_stats is not None else 0
    headings = np.full(n_labels, np.nan, dtype=np.float32)
    counts = np.zeros(n_labels, dtype=np.uint16)
    source_labels = np.zeros(n_labels, dtype=np.int32)
    distances = np.full(n_labels, np.nan, dtype=np.float32)
    if n_labels == 0 or not yolo_templates_by_zone:
        return headings, counts, source_labels, distances

    sources = []
    for source_label, templates in yolo_templates_by_zone.items():
        source_label = int(source_label)
        if source_label <= 0 or source_label >= n_labels:
            continue
        consensus = _consensus_yolo_template(
            templates,
            heading_tol_deg=heading_tol_deg,
            shape_rel_tol=shape_rel_tol,
            shape_abs_tol_px=shape_abs_tol_px,
        )
        if consensus is None:
            continue
        left = float(cc_stats[source_label, cv2.CC_STAT_LEFT])
        top = float(cc_stats[source_label, cv2.CC_STAT_TOP])
        width = float(cc_stats[source_label, cv2.CC_STAT_WIDTH])
        height = float(cc_stats[source_label, cv2.CC_STAT_HEIGHT])
        if width <= 0.0 or height <= 0.0:
            continue
        sources.append({
            'label': source_label,
            'heading': float(consensus['heading']) % 360.0,
            'votes': int(consensus.get('consensus_heading_votes', len(templates))),
            'left': left,
            'top': top,
            'right': left + width - 1.0,
            'bottom': top + height - 1.0,
            'cx': float(cc_centroids[source_label, 0]),
            'cy': float(cc_centroids[source_label, 1]),
        })
    if not sources:
        return headings, counts, source_labels, distances

    source_label_set = {item['label'] for item in sources}
    for zone_label in valid_labels:
        zone_label = int(zone_label)
        if zone_label <= 0 or zone_label >= n_labels or zone_label in source_label_set:
            continue
        left = float(cc_stats[zone_label, cv2.CC_STAT_LEFT])
        top = float(cc_stats[zone_label, cv2.CC_STAT_TOP])
        width = float(cc_stats[zone_label, cv2.CC_STAT_WIDTH])
        height = float(cc_stats[zone_label, cv2.CC_STAT_HEIGHT])
        if width <= 0.0 or height <= 0.0:
            continue
        right = left + width - 1.0
        bottom = top + height - 1.0
        cx = float(cc_centroids[zone_label, 0])
        cy = float(cc_centroids[zone_label, 1])
        best = None
        best_key = None
        for source in sources:
            dx = max(source['left'] - right, left - source['right'], 0.0)
            dy = max(source['top'] - bottom, top - source['bottom'], 0.0)
            bbox_dist = math.hypot(dx, dy)
            center_dist = math.hypot(cx - source['cx'], cy - source['cy'])
            key = (bbox_dist, center_dist, -source['votes'])
            if best_key is None or key < best_key:
                best_key = key
                best = source
        if best is None:
            continue
        headings[zone_label] = best['heading']
        counts[zone_label] = min(best['votes'], 65535)
        source_labels[zone_label] = best['label']
        distances[zone_label] = float(best_key[0])
    return headings, counts, source_labels, distances


def _refine_candidate_classes_from_yolo(cand_x, cand_y, cand_cls, yolo_guidance):
    if yolo_guidance is None or cand_x.size == 0:
        return cand_cls, 0
    labels = yolo_guidance['nearest_label'][cand_y, cand_x]
    dist_m = (
        yolo_guidance['distance_px'][cand_y, cand_x] *
        float(yolo_guidance.get('m_per_px', 1.0))
    )
    class_by_label = yolo_guidance['class_by_label']
    valid = (
        (labels > 0) &
        (labels < class_by_label.shape[0]) &
        (dist_m <= float(yolo_guidance.get('max_distance_m', YOLO_GUIDANCE_MAX_DISTANCE_M)))
    )
    if not np.any(valid):
        return cand_cls, 0
    refined = cand_cls.copy()
    local_cls = np.zeros_like(cand_cls)
    local_cls[valid] = class_by_label[labels[valid]]
    apply = valid & (local_cls != 0)
    refined[apply] = local_cls[apply]
    return refined, int(np.count_nonzero(apply))


def _pca_long_axis_heading(xs, ys):
    """Return compass heading of a pixel cloud's long axis, or NaN if weak."""
    if xs.size < 6:
        return float('nan'), 1.0
    coords = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    coords -= np.mean(coords, axis=0)
    cov = coords.T @ coords / max(float(coords.shape[0] - 1), 1.0)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)
    major = eigvecs[:, order[-1]]
    minor_val = max(float(eigvals[order[-2]]), 1e-6)
    aspect = math.sqrt(max(float(eigvals[order[-1]]), 1e-6) / minor_val)
    if aspect < 1.35:
        return float('nan'), aspect
    img_angle = math.degrees(math.atan2(float(major[1]), float(major[0])))
    return (90.0 - img_angle) % 180.0, aspect


def _build_local_roof_evidence(bld_raw, bld_zone, m_per_px):
    """Build nearest-raw-roof class and heading evidence for candidate placement."""
    raw_mask = ((bld_raw != 0) & (bld_zone != 0)).astype(np.uint8)
    if not raw_mask.any():
        return None

    n_frag, frag_labels, frag_stats, _ = cv2.connectedComponentsWithStats(
        raw_mask, connectivity=8
    )
    if n_frag <= 1:
        return None

    px_area_m2 = max(float(m_per_px) * float(m_per_px), 1e-6)
    min_fragment_px = max(2, int(round(18.0 / px_area_m2)))
    class_by_label = np.zeros(n_frag, dtype=np.uint8)
    heading_by_label = np.full(n_frag, np.nan, dtype=np.float32)
    aspect_by_label = np.ones(n_frag, dtype=np.float32)
    area_by_label_m2 = np.zeros(n_frag, dtype=np.float32)

    for label in range(1, n_frag):
        area_px = int(frag_stats[label, cv2.CC_STAT_AREA])
        if area_px < min_fragment_px:
            continue
        x = int(frag_stats[label, cv2.CC_STAT_LEFT])
        y = int(frag_stats[label, cv2.CC_STAT_TOP])
        w = int(frag_stats[label, cv2.CC_STAT_WIDTH])
        h = int(frag_stats[label, cv2.CC_STAT_HEIGHT])
        area_m2 = float(area_px) * px_area_m2
        max_side_m = max(float(w), float(h)) * float(m_per_px)
        fill_ratio = area_px / max(float(w * h), 1.0)
        class_by_label[label] = _roof_fragment_class(area_m2, max_side_m, fill_ratio)
        area_by_label_m2[label] = area_m2

        local = frag_labels[y:y + h, x:x + w] == label
        ys, xs = np.nonzero(local)
        heading, aspect = _pca_long_axis_heading(xs + x, ys + y)
        heading_by_label[label] = heading
        aspect_by_label[label] = aspect

    if not np.any(class_by_label):
        return None

    dist_src = np.where(raw_mask != 0, 0, 1).astype(np.uint8)
    dist_px, nearest_label = cv2.distanceTransformWithLabels(
        dist_src,
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_CCOMP,
    )
    return {
        'distance_px': dist_px.astype(np.float32, copy=False),
        'nearest_label': nearest_label.astype(np.int32, copy=False),
        'class_by_label': class_by_label,
        'heading_by_label': heading_by_label,
        'aspect_by_label': aspect_by_label,
        'area_by_label_m2': area_by_label_m2,
    }


def _refine_candidate_classes_from_roofs(cand_x, cand_y, cand_cls, roof_evidence,
                                         m_per_px, max_distance_m=30.0):
    """Use nearby raw roof fragments to fix classes inside broad cleaned zones."""
    if roof_evidence is None or cand_x.size == 0:
        return cand_cls, 0
    labels = roof_evidence['nearest_label'][cand_y, cand_x]
    dist_m = roof_evidence['distance_px'][cand_y, cand_x] * float(m_per_px)
    class_by_label = roof_evidence['class_by_label']
    valid = (
        (labels > 0) &
        (labels < class_by_label.shape[0]) &
        (dist_m <= float(max_distance_m))
    )
    if not np.any(valid):
        return cand_cls, 0

    local_cls = np.zeros_like(cand_cls)
    local_cls[valid] = class_by_label[labels[valid]]
    # Raw SegFormer building islands can still be over-merged at ZL16, so use
    # local evidence as a conservative correction for oversized zone classes.
    apply = valid & (local_cls != 0) & (local_cls < cand_cls)
    if not np.any(apply):
        return cand_cls, 0
    refined = cand_cls.copy()
    refined[apply] = local_cls[apply]
    return refined, int(np.count_nonzero(apply))


def _roof_heading_for_candidate(roof_evidence, jx, jy, m_per_px,
                                max_distance_m=18.0, min_area_m2=140.0):
    """Return local raw-roof long-axis heading when the evidence is rectangular."""
    if roof_evidence is None:
        return float('nan')
    label = int(roof_evidence['nearest_label'][jy, jx])
    if label <= 0 or label >= roof_evidence['heading_by_label'].shape[0]:
        return float('nan')
    if float(roof_evidence['distance_px'][jy, jx]) * float(m_per_px) > max_distance_m:
        return float('nan')
    if float(roof_evidence['area_by_label_m2'][label]) < float(min_area_m2):
        return float('nan')
    if float(roof_evidence['aspect_by_label'][label]) < 1.45:
        return float('nan')
    return float(roof_evidence['heading_by_label'][label])


def _spacing_for_zone_class_m(edge_spacing_m, zone_class,
                              class_min_footprint_span_m=None):
    """Return candidate centre spacing from class footprint span + edge gap."""
    class_span_m = 0.0
    if class_min_footprint_span_m is not None:
        class_span_m = float(
            class_min_footprint_span_m.get(int(zone_class), 0.0)
        )
    return max(0.0, float(edge_spacing_m)) + max(0.0, class_span_m)


def _class_spacing_px(edge_spacing_m, m_per_px, class_min_footprint_span_m=None):
    """Return per-zone-class candidate spacing in native image pixels."""
    return {
        cls: max(
            3,
            int(_spacing_for_zone_class_m(
                edge_spacing_m, cls, class_min_footprint_span_m
            ) / max(m_per_px, 1e-6)),
        )
        for cls in BLD_PLACEMENT_CLASSES
    }


def _gap_fill_class_sequence(zone_class):
    """Return the only placement class allowed for this gap-fill center."""
    zc = int(zone_class)
    return (zc,) if zc in BLD_PLACEMENT_CLASSES else ()


def _yolo_direct_class_sequence(zone_class):
    """Try YOLO's inferred class first, then smaller compatible classes."""
    zc = int(zone_class)
    if zc not in BLD_PLACEMENT_CLASSES:
        return ()
    return tuple(
        cls for cls in reversed(BLD_PLACEMENT_CLASSES)
        if cls <= zc
    )


def _residential_infill_class(zone_class):
    """Return a smaller house-like class for residential subareas in coarse zones."""
    zone_class = int(zone_class)
    if zone_class <= BLD_CLASS_MEDIUM:
        return zone_class
    if zone_class == BLD_CLASS_SMALL_APARTMENT:
        return BLD_CLASS_SMALL_RESIDENTIAL
    if zone_class == BLD_CLASS_APARTMENT_BLOCK:
        return BLD_CLASS_COMPACT_RESIDENTIAL
    return BLD_CLASS_MEDIUM


def _format_class_spacing(spacing_px_by_class, m_per_px):
    """Return compact spacing summary for footprint classes."""
    metres = "/".join(
        f"{spacing_px_by_class[cls] * m_per_px:.1f}" for cls in BLD_PLACEMENT_CLASSES
    )
    pixels = "/".join(str(int(spacing_px_by_class[cls])) for cls in BLD_PLACEMENT_CLASSES)
    return f"spacing≈{metres}m ({pixels}px tiny/small/compact/medium/apt/block/large/xl)"


def _describe_placement_summary(class_counts, building_coverage_pct, osm_cell_count,
                                grid_n, spacing_label, yolo_counts=None):
    """Return a user-facing summary for one DDS building-placement pass."""
    total_count = sum(class_counts.values())
    class_bits = "  ".join(
        f"{BLD_CLASS_LABELS[cls]}={class_counts.get(cls, 0)}"
        for cls in BLD_PLACEMENT_CLASSES
    )
    yolo_bits = ""
    if yolo_counts:
        yolo_bits = (
            "  yolo "
            f"det={int(yolo_counts.get('yolo_detections', 0))} "
            f"accepted={int(yolo_counts.get('yolo_placed', 0))} "
            f"obj={int(yolo_counts.get('yolo_object_placed', 0))} "
            f"facade={int(yolo_counts.get('yolo_facade_placed', 0))} "
            f"blocked={int(yolo_counts.get('yolo_blocked', 0))} "
            f"overlap={int(yolo_counts.get('yolo_overlap_blocked', 0))}  "
        )
        if yolo_counts.get('yolo_suppressed_overlap', 0):
            yolo_bits += (
                f"suppressed={int(yolo_counts.get('yolo_suppressed_overlap', 0))}  "
            )
    return (
        f"placed={total_count:4d}  "
        f"{class_bits}  "
        f"{yolo_bits}"
        f"building cover={building_coverage_pct:.1f}%  "
        f"street-guided cells={osm_cell_count}/{grid_n * grid_n}  "
        f"{spacing_label}"
    )


def _footprint_poly_with_bbox(cx: int, cy: int, bounds_m, heading_deg: float, m_per_px: float):
    """Return rotated footprint polygon and its pixel bbox."""
    rad = math.radians(float(heading_deg))
    return _footprint_poly_with_bbox_basis(
        cx,
        cy,
        bounds_m,
        1.0 / float(m_per_px),
        math.cos(rad),
        math.sin(rad),
        math.sin(rad),
        -math.cos(rad),
    )


def _footprint_poly_with_bbox_basis(
    cx: int,
    cy: int,
    bounds_m,
    inv_m_per_px: float,
    right_x: float,
    right_y: float,
    fwd_x: float,
    fwd_y: float,
):
    """Return rotated footprint polygon and bbox using a precomputed basis."""
    xmin, xmax, zmin, zmax = bounds_m
    cx = float(cx)
    cy = float(cy)
    pts = np.empty((4, 2), dtype=np.int32)
    min_x = min_y = 2 ** 31 - 1
    max_x = max_y = -(2 ** 31)
    for idx, (lx, lz) in enumerate(((xmin, zmin), (xmax, zmin), (xmax, zmax), (xmin, zmax))):
        px = int(round(cx + (lx * right_x + lz * fwd_x) * inv_m_per_px))
        py = int(round(cy + (lx * right_y + lz * fwd_y) * inv_m_per_px))
        pts[idx, 0] = px
        pts[idx, 1] = py
        if px < min_x: min_x = px
        if px > max_x: max_x = px
        if py < min_y: min_y = py
        if py > max_y: max_y = py
    return pts, (min_x, min_y, max_x + 1, max_y + 1)


def _footprint_poly(cx: int, cy: int, bounds_m, heading_deg: float, m_per_px: float):
    """Return the rotated local footprint polygon in image pixel space.

    bounds_m are local SFD object bounds relative to the object origin:
    (xmin, xmax, zmin, zmax) in metres.
    """
    pts, _ = _footprint_poly_with_bbox(cx, cy, bounds_m, heading_deg, m_per_px)
    return pts


def _orientation_angles_for_bounds(bounds_m, desired_long_axis_heading):
    """Return headings ordered to align an asset's longer footprint side first."""
    xmin, xmax, zmin, zmax = bounds_m
    width_m = abs(float(xmax) - float(xmin))
    depth_m = abs(float(zmax) - float(zmin))
    heading = float(desired_long_axis_heading) % 360.0
    if width_m > depth_m * 1.10:
        return ((heading - 90.0) % 360.0, heading)
    return (heading, (heading - 90.0) % 360.0)


def _fit_bbox_for_poly(pts: np.ndarray, img_w: int, img_h: int):
    """Return clipped [x1, y1, x2, y2] bbox for polygon fit checks."""
    if pts.shape[0] == 0:
        return (0, 0, 0, 0)
    return (
        max(0, int(pts[:, 0].min())),
        max(0, int(pts[:, 1].min())),
        min(int(img_w), int(pts[:, 0].max()) + 1),
        min(int(img_h), int(pts[:, 1].max()) + 1),
    )


def _prepare_direct_yolo_detection(detection, img_w: int, img_h: int):
    """Return cached polygon geometry for the direct-YOLO placement loop."""
    yolo_poly = np.rint(
        np.asarray(detection.get('points', ()), dtype=np.float32)
    ).astype(np.int32)
    if yolo_poly.shape != (4, 2):
        return {
            'valid': False,
            'poly': yolo_poly,
            'bbox': (0, 0, 0, 0),
            'area_px': 0.0,
            'jx': 0,
            'jy': 0,
        }
    try:
        jx = int(round(float(detection['center'][0])))
        jy = int(round(float(detection['center'][1])))
    except (KeyError, IndexError, TypeError, ValueError):
        jx = jy = -1
    return {
        'valid': True,
        'poly': yolo_poly,
        'bbox': _fit_bbox_for_poly(yolo_poly, img_w, img_h),
        'area_px': abs(float(cv2.contourArea(yolo_poly.astype(np.float32)))),
        'jx': jx,
        'jy': jy,
    }


def _poly_fits(
    occ_mask: np.ndarray,
    pts: np.ndarray,
    scratch_mask: np.ndarray | None = None,
    bbox=None,
) -> bool:
    """Return True if polygon pts have no overlap with any set pixel in occ_mask."""
    return _poly_fits_with_integral(occ_mask, pts, scratch_mask=scratch_mask, bbox=bbox)


def _spacing_poly_fits(
    spacing_mask: np.ndarray,
    yolo_poly: np.ndarray,
    scratch_mask: np.ndarray | None = None,
    spacing_integral: np.ndarray | None = None,
    recent_spacing_mask: np.ndarray | None = None,
    bbox=None,
) -> bool:
    """Return True if poly does not overlap building_spacing_mask. Uses an
    integral image plus an incremental ``recent_spacing_mask`` to avoid a full
    bbox ``countNonZero`` when the area is sparse."""
    if bbox is None:
        bbox = _fit_bbox_for_poly(yolo_poly, spacing_mask.shape[1], spacing_mask.shape[0])
    x1, y1, x2, y2 = bbox
    if x1 >= x2 or y1 >= y2:
        return True
    if spacing_integral is not None:
        integral_has_occupancy = _integral_bbox_sum(spacing_integral, x1, y1, x2, y2) != 0
    else:
        integral_has_occupancy = cv2.countNonZero(spacing_mask[y1:y2, x1:x2]) != 0
    recent_has_occupancy = (
        recent_spacing_mask is not None and
        cv2.countNonZero(recent_spacing_mask[y1:y2, x1:x2]) != 0
    )
    if not integral_has_occupancy and not recent_has_occupancy:
        return True
    return _poly_fits(spacing_mask, yolo_poly, scratch_mask, bbox=bbox)


def _direct_yolo_poly_fits(
    static_occ_mask: np.ndarray,
    building_spacing_mask: np.ndarray,
    yolo_poly: np.ndarray,
    scratch_mask: np.ndarray | None = None,
    static_occ_integral: np.ndarray | None = None,
    bbox=None,
    spacing_occ_integral: np.ndarray | None = None,
    recent_spacing_mask: np.ndarray | None = None,
) -> bool:
    """Return True when a direct YOLO footprint clears static and dynamic blockers."""
    if not _poly_fits_with_integral(
        static_occ_mask,
        yolo_poly,
        scratch_mask=scratch_mask,
        occ_integral=static_occ_integral,
        bbox=bbox,
    ):
        return False
    return _spacing_poly_fits(
        building_spacing_mask,
        yolo_poly,
        scratch_mask=scratch_mask,
        spacing_integral=spacing_occ_integral,
        recent_spacing_mask=recent_spacing_mask,
        bbox=bbox,
    )


def _placed_yolo_poly_fits(
    placed_yolo_mask: np.ndarray,
    yolo_poly: np.ndarray,
    scratch_mask: np.ndarray | None = None,
    placed_yolo_integral: np.ndarray | None = None,
    recent_yolo_mask: np.ndarray | None = None,
    bbox=None,
) -> bool:
    """Return True if a YOLO polygon does not overlap already placed YOLOs."""
    if bbox is None:
        bbox = _fit_bbox_for_poly(yolo_poly, placed_yolo_mask.shape[1], placed_yolo_mask.shape[0])
    x1, y1, x2, y2 = bbox
    if x1 >= x2 or y1 >= y2:
        return True
    if placed_yolo_integral is not None:
        integral_has_occupancy = _integral_bbox_sum(placed_yolo_integral, x1, y1, x2, y2) != 0
    else:
        integral_has_occupancy = cv2.countNonZero(placed_yolo_mask[y1:y2, x1:x2]) != 0
    recent_has_occupancy = (
        recent_yolo_mask is not None and
        cv2.countNonZero(recent_yolo_mask[y1:y2, x1:x2]) != 0
    )
    if not integral_has_occupancy and not recent_has_occupancy:
        return True
    return _poly_fits(placed_yolo_mask, yolo_poly, scratch_mask, bbox=bbox)


def _poly_inside_poly(inner_poly: np.ndarray, outer_poly: np.ndarray) -> bool:
    """Return True when every inner vertex lies inside or on the outer polygon."""
    outer = np.asarray(outer_poly, dtype=np.float32)
    inner = np.asarray(inner_poly, dtype=np.float32)
    if outer.shape[0] < 3 or inner.shape[0] < 3:
        return False
    for px, py in inner:
        if cv2.pointPolygonTest(outer, (float(px), float(py)), False) < -1e-6:
            return False
    center = inner.mean(axis=0)
    return cv2.pointPolygonTest(outer, (float(center[0]), float(center[1])), False) >= -1e-6


def _footprint_inside_detection(inner_poly: np.ndarray,
                                outer_poly: np.ndarray,
                                m_per_px: float = 1.0) -> bool:
    """Containment test for object footprints inside YOLO detection polys.

    The footprint must lie inside the detection polygon except for a thin
    perimeter sliver: at most a band of width ``YOLO_OBJECT_OUTLINE_MARGIN_M``
    *metres* around the footprint edge may fall outside (absorbing integer
    pixel/anchor rounding). The allowance is a fixed real-world band scaled by
    the footprint perimeter -- not a fraction of its area and not a fixed pixel
    count -- so it is resolution-independent and an object can never stick out
    by a real proportion of its size, regardless of how large it is. Both
    polygons are convex (rotated footprint rectangles and crop-clipped OBBs).
    """
    if _poly_inside_poly(inner_poly, outer_poly):
        return True
    inner = np.asarray(inner_poly, dtype=np.float32)
    outer = np.asarray(outer_poly, dtype=np.float32)
    if inner.shape[0] < 3 or outer.shape[0] < 3:
        return False
    inner_area = abs(float(cv2.contourArea(inner)))
    if inner_area <= 0.0:
        return False
    try:
        inter_area, _ = cv2.intersectConvexConvex(inner, outer)
    except cv2.error:
        return False
    outside_area = inner_area - float(inter_area)
    if outside_area <= 1e-6:
        return True
    # Allowed overhang: a fixed real-world band (metres) along the footprint
    # perimeter. outside_area and perimeter are in pixels; a band of width
    # MARGIN_M metres is (MARGIN_M / m_per_px) pixels wide, so the band area in
    # px^2 is perimeter_px * (MARGIN_M / m_per_px).
    perimeter_px = float(cv2.arcLength(inner, True))
    mpp = max(float(m_per_px), 1e-6)
    margin_area = perimeter_px * (float(YOLO_OBJECT_OUTLINE_MARGIN_M) / mpp)
    return outside_area <= margin_area + 1e-6


def _yolo_facade_class(detection_class):
    try_cls = int(detection_class)
    if try_cls in BLD_PLACEMENT_CLASSES:
        height_m = DEFAULT_FACADE_HEIGHT_M.get(try_cls, 8.0)
        if not _is_very_tall_building_asset(DEFAULT_FACADE_PATHS.get(try_cls), height_m):
            return try_cls
    allowed = [
        cls for cls in BLD_PLACEMENT_CLASSES
        if not _is_very_tall_building_asset(
            DEFAULT_FACADE_PATHS.get(cls),
            DEFAULT_FACADE_HEIGHT_M.get(cls, 8.0),
        )
    ]
    if not allowed:
        return None
    return min(allowed, key=lambda cls: abs(int(cls) - try_cls))


def _all_object_assets_by_size(asset_pools):
    assets = []
    for pool in (asset_pools or {}).values():
        for asset in pool:
            if asset.get('kind') != 'object':
                continue
            if _is_very_tall_building_asset(asset.get('path'), asset.get('height_m')):
                continue
            area_m2 = asset.get('footprint_area_m2')
            if area_m2 is None and asset.get('bounds_m') is not None:
                area_m2, _ = _footprint_metrics(asset['bounds_m'])
            if area_m2 is None:
                continue
            assets.append((float(area_m2), asset))
    assets.sort(key=lambda item: (item[0], item[1].get('path', '')), reverse=True)
    return [asset for _area, asset in assets]


def _required_centered_dimensions_for_bounds(bounds_m):
    """Return dimensions required to keep off-centre bounds inside a centred OBB."""
    xmin, xmax, zmin, zmax = (float(v) for v in bounds_m)
    width_m = 2.0 * max(abs(xmin), abs(xmax))
    depth_m = 2.0 * max(abs(zmin), abs(zmax))
    return width_m, depth_m


def _yolo_object_dimension_key(detection):
    """Return floored integer-metre (length, width) lookup key for a detection."""
    try:
        length_m = float(detection.get('length_m'))
        width_m = float(detection.get('width_m'))
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(length_m) and math.isfinite(width_m)):
        return None
    length_key = int(math.floor(max(length_m, width_m) + 1e-4))
    width_key = int(math.floor(min(length_m, width_m) + 1e-4))
    if length_key <= 0 or width_key <= 0:
        return None
    return length_key, width_key


def _build_yolo_object_fit_table(
    asset_pools,
    min_coverage=YOLO_OBJECT_MIN_COVERAGE,
):
    """Return a full integer-metre detection-dimension lookup for object assets."""
    table = {}
    min_coverage = max(1e-6, float(min_coverage))
    seen_orientations = set()
    for asset in _all_object_assets_by_size(asset_pools):
        bounds_m = asset.get('bounds_m')
        if bounds_m is None:
            continue
        try:
            footprint_area_m2, _ = _footprint_metrics(bounds_m)
            req_width_m, req_depth_m = _required_centered_dimensions_for_bounds(bounds_m)
        except Exception:
            continue
        if footprint_area_m2 <= 0.0 or req_width_m <= 0.0 or req_depth_m <= 0.0:
            continue

        orientations = (
            (req_depth_m, req_width_m, 0.0),
            (req_width_m, req_depth_m, -90.0),
        )
        for required_length_m, required_width_m, heading_delta in orientations:
            min_length_key = int(math.ceil(required_length_m - 1e-6))
            min_width_key = int(math.ceil(required_width_m - 1e-6))
            if min_length_key <= 0 or min_width_key <= 0:
                continue
            max_detection_area_m2 = footprint_area_m2 / min_coverage
            max_length_key = int(math.floor(max_detection_area_m2 / min_width_key))
            orientation_id = (
                asset.get('path', ''),
                round(float(heading_delta), 6),
                min_length_key,
                min_width_key,
            )
            if orientation_id in seen_orientations:
                continue
            seen_orientations.add(orientation_id)
            for length_key in range(min_length_key, max_length_key + 1):
                max_width_key = int(math.floor(max_detection_area_m2 / length_key))
                if max_width_key < min_width_key:
                    continue
                for width_key in range(min_width_key, max_width_key + 1):
                    key = (int(length_key), int(width_key))
                    coverage = footprint_area_m2 / max(1.0, float(length_key * width_key))
                    table.setdefault(key, []).append({
                        'asset': asset,
                        'heading_delta': float(heading_delta),
                        'required_length_m': float(required_length_m),
                        'required_width_m': float(required_width_m),
                        'coverage': float(coverage),
                        'footprint_area_m2': float(footprint_area_m2),
                    })

    for key, candidates in table.items():
        candidates.sort(
            key=lambda item: (
                -float(item['coverage']),
                -float(item['footprint_area_m2']),
                item['asset'].get('source', ''),
                item['asset'].get('path', ''),
                float(item['heading_delta']),
            )
        )
    return table


def _yolo_object_min_coverage_for_class(zone_class, default=YOLO_OBJECT_MIN_COVERAGE):
    """Return the minimum object/YOLO footprint coverage for a detection class."""
    try:
        zone_class = int(zone_class)
    except (TypeError, ValueError):
        return float(default)
    return float(YOLO_OBJECT_MIN_COVERAGE_BY_CLASS.get(zone_class, default))


def _yolo_object_candidate_class_sequence(zone_class):
    """Return object classes to try for a direct YOLO detection."""
    try:
        zone_class = int(zone_class)
    except (TypeError, ValueError):
        zone_class = BLD_CLASS_MEDIUM
    if zone_class not in BLD_PLACEMENT_CLASSES:
        zone_class = _yolo_facade_class(zone_class) or BLD_CLASS_MEDIUM

    residential = (
        BLD_CLASS_TINY_RESIDENTIAL,
        BLD_CLASS_SMALL_RESIDENTIAL,
        BLD_CLASS_COMPACT_RESIDENTIAL,
    )
    if zone_class in residential:
        ordered = sorted(
            residential,
            key=lambda cls: (abs(int(cls) - zone_class), int(cls)),
        )
    else:
        ordered = [zone_class]

    ordered.extend(
        cls for cls in BLD_PLACEMENT_CLASSES
        if cls not in ordered
    )
    return tuple(ordered)


def _build_yolo_object_candidate_index(asset_pools):
    """Return object candidates grouped for scored direct-YOLO selection."""
    by_class = {cls: [] for cls in BLD_PLACEMENT_CLASSES}
    seen_orientations = set()
    for asset in _all_object_assets_by_size(asset_pools):
        bounds_m = asset.get('bounds_m')
        if bounds_m is None:
            continue
        try:
            footprint_area_m2, _ = _footprint_metrics(bounds_m)
            req_width_m, req_depth_m = _required_centered_dimensions_for_bounds(bounds_m)
        except Exception:
            continue
        if footprint_area_m2 <= 0.0 or req_width_m <= 0.0 or req_depth_m <= 0.0:
            continue
        try:
            asset_class = int(asset.get('footprint_class') or _class_for_object_asset(
                asset.get('path'), bounds_m
            ))
        except Exception:
            asset_class = _class_for_footprint(bounds_m)
        if asset_class not in by_class:
            asset_class = _nearest_available_class(asset_class, by_class)

        orientations = (
            (max(req_depth_m, req_width_m), min(req_depth_m, req_width_m), 0.0),
            (max(req_width_m, req_depth_m), min(req_width_m, req_depth_m), -90.0),
        )
        for required_length_m, required_width_m, heading_delta in orientations:
            orientation_id = (
                asset.get('path', ''),
                round(float(heading_delta), 6),
                round(float(required_length_m), 4),
                round(float(required_width_m), 4),
            )
            if orientation_id in seen_orientations:
                continue
            seen_orientations.add(orientation_id)
            by_class[asset_class].append({
                'asset': asset,
                'heading_delta': float(heading_delta),
                'required_length_m': float(required_length_m),
                'required_width_m': float(required_width_m),
                'coverage_area_m2': float(footprint_area_m2),
            })

    for candidates in by_class.values():
        candidates.sort(
            key=lambda item: (
                -float(item['coverage_area_m2']),
                item['asset'].get('source', ''),
                item['asset'].get('path', ''),
                float(item['heading_delta']),
            )
        )
    return {
        'kind': 'scored_yolo_object_candidates',
        'assets_by_class': by_class,
        'merged_by_zone': {},
    }


def _merged_yolo_candidates_by_zone(fit_table, zone_class):
    """Return candidates across fallback classes sorted by -coverage_area_m2.

    Returns a dict with ``entries`` (list of ``(class_rank, asset_class, candidate)``)
    aligned with three numpy arrays — ``cov_area``, ``req_l``, ``req_w`` — used to
    vectorise the size/coverage pre-filter at call time. Cached on
    ``fit_table['merged_by_zone']`` so the merge runs once per zone_class.
    """
    cache = fit_table.setdefault('merged_by_zone', {})
    try:
        cache_key = int(zone_class)
    except (TypeError, ValueError):
        cache_key = zone_class
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    assets_by_class = fit_table.get('assets_by_class', {}) or {}
    entries = []
    class_sequence = _yolo_object_candidate_class_sequence(zone_class)
    for class_rank, asset_class in enumerate(class_sequence):
        for candidate in assets_by_class.get(asset_class, ()):  # already coverage-desc within class
            entries.append((class_rank, int(asset_class), candidate))
    entries.sort(
        key=lambda item: (
            -float(item[2]['coverage_area_m2']),
            int(item[0]),
            int(item[1]),
            item[2]['asset'].get('source', ''),
            item[2]['asset'].get('path', ''),
            float(item[2]['heading_delta']),
        )
    )
    n = len(entries)
    cov_area = np.empty(n, dtype=np.float64)
    req_l = np.empty(n, dtype=np.float64)
    req_w = np.empty(n, dtype=np.float64)
    for i, (_rank, _cls, cand) in enumerate(entries):
        cov_area[i] = float(cand['coverage_area_m2'])
        req_l[i] = float(cand['required_length_m'])
        req_w[i] = float(cand['required_width_m'])
    bundle = {
        'entries': entries,
        'cov_area': cov_area,
        'req_l': req_l,
        'req_w': req_w,
    }
    cache[cache_key] = bundle
    return bundle


_GLOBAL_YOLO_OBJECT_FIT_TABLE = None


def _all_known_yolo_object_asset_pools():
    """Return object pools for every static asset catalog entry we know about."""
    asset_pools = {cls: [] for cls in BLD_PLACEMENT_CLASSES}
    seen_paths = set()

    default_paths = set(DEFAULT_OBJECT_CATALOG_OLD_WORLD) | set(DEFAULT_OBJECT_CATALOG_NORTH_AMERICA)
    for obj_path in sorted(default_paths):
        if obj_path in seen_paths:
            continue
        seen_paths.add(obj_path)
        dims = _default_object_dims(obj_path)
        if dims is None:
            continue
        _append_object_asset(
            asset_pools,
            obj_path,
            _bounds_from_dimensions(dims[0], dims[1]),
            'Default X-Plane',
        )

    asset_regions = (
        'generic',
        'europe',
        'scandinavia',
        'mediterranean',
        'north_america',
        'north_america_ne',
        'north_america_west',
        'south_america',
        'asia',
        'se_asia',
        'africa',
        'australia_oceania',
    )
    for region in asset_regions:
        for obj_path in _sfd_catalog_paths(0.0, 0.0, region):
            if obj_path in seen_paths:
                continue
            seen_paths.add(obj_path)
            bounds_m = _bounds_for_object_path(obj_path)
            if bounds_m is None:
                continue
            _append_object_asset(asset_pools, obj_path, bounds_m, 'SFD Global')

    simheaven_paths = (
        SIMHEAVEN_SMALL_BUILDING_CATALOG +
        SIMHEAVEN_RESIDENTIAL_CATALOG +
        SIMHEAVEN_COMMERCIAL_CATALOG +
        SIMHEAVEN_INDUSTRIAL_CATALOG
    )
    for obj_path in simheaven_paths:
        if obj_path in seen_paths:
            continue
        if not _is_repeatable_simheaven_asset(obj_path):
            continue
        seen_paths.add(obj_path)
        dims = _simheaven_object_dims(obj_path)
        _append_object_asset(
            asset_pools,
            obj_path,
            _bounds_from_dimensions(dims[0], dims[1]),
            'simHeaven',
        )
    return asset_pools


def _global_yolo_object_fit_table():
    """Return the static object-dimension lookup, building it once per process."""
    global _GLOBAL_YOLO_OBJECT_FIT_TABLE
    if _GLOBAL_YOLO_OBJECT_FIT_TABLE is None:
        _GLOBAL_YOLO_OBJECT_FIT_TABLE = _build_yolo_object_fit_table(
            _all_known_yolo_object_asset_pools()
        )
    return _GLOBAL_YOLO_OBJECT_FIT_TABLE


def _select_yolo_object_candidate(
    fit_table,
    detection,
    yolo_poly,
    jx,
    jy,
    heading,
    m_per_px,
    residential_context=True,
    enabled_assets_by_path=None,
    min_coverage=YOLO_OBJECT_MIN_COVERAGE,
    static_occ_mask=None,
    building_spacing_mask=None,
    scratch_mask=None,
    static_occ_integral=None,
    spacing_occ_integral=None,
    recent_spacing_mask=None,
    freearea_downsize=False,
    skip_occupancy=False,
    profile_out=None,
):
    """Return a mapped object candidate that fits inside the YOLO polygon.

    When ``profile_out`` is a dict it is used as an accumulator to attribute
    placement cost: ``calls`` (selector invocations), ``entries`` (candidate
    pool size offered), ``viable`` (survivors of the vectorised size/coverage
    prefilter) and ``seen`` (candidates actually scanned in Python). Used by the
    ``O4_SFR_BLD_PLACE_PROFILE`` diagnostic; ``None`` keeps the hot path free of
    bookkeeping.
    """
    if profile_out is not None:
        profile_out['calls'] = profile_out.get('calls', 0) + 1
    if not isinstance(fit_table, dict) or fit_table.get('kind') != 'scored_yolo_object_candidates':
        key = _yolo_object_dimension_key(detection)
        if key is None:
            return None, 'miss'
        candidates = (fit_table or {}).get(key)
        if not candidates:
            return None, 'miss'
        if profile_out is not None:
            profile_out['entries'] = profile_out.get('entries', 0) + len(candidates)

        try:
            detection_area_m2 = float(detection.get('area_m2') or 0.0)
        except (TypeError, ValueError):
            detection_area_m2 = 0.0
        if detection_area_m2 <= 0.0:
            detection_area_m2 = (
                abs(float(cv2.contourArea(np.asarray(yolo_poly, dtype=np.float32)))) *
                float(m_per_px) * float(m_per_px)
            )
        if detection_area_m2 <= 0.0:
            return None, 'miss'

        context_skipped = 0
        for candidate in candidates:
            if profile_out is not None:
                profile_out['seen'] = profile_out.get('seen', 0) + 1
            candidate_asset = candidate['asset']
            path = candidate_asset.get('path')
            if enabled_assets_by_path is not None:
                asset = enabled_assets_by_path.get(path)
                if asset is None:
                    continue
            else:
                asset = candidate_asset
            if not residential_context and _asset_requires_residential_context(asset):
                context_skipped += 1
                continue
            footprint_area_m2 = float(candidate['footprint_area_m2'])
            if footprint_area_m2 / detection_area_m2 < float(min_coverage):
                continue
            final_heading = (float(heading) + float(candidate['heading_delta'])) % 360.0
            footprint_poly = _footprint_poly(
                int(jx), int(jy), asset['bounds_m'], final_heading, m_per_px
            )
            if not _footprint_inside_detection(footprint_poly, yolo_poly, m_per_px):
                continue
            return {
                'asset': asset,
                'heading': final_heading,
                'footprint_poly': footprint_poly,
            }, 'selected'

        if context_skipped:
            return None, 'context_skipped'
        return None, 'miss'

    def _detection_dimensions_m():
        try:
            length_m = float(detection.get('length_m'))
            width_m = float(detection.get('width_m'))
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(length_m) and math.isfinite(width_m)):
            return None
        length_m, width_m = max(length_m, width_m), min(length_m, width_m)
        if length_m <= 0.0 or width_m <= 0.0:
            return None
        return length_m, width_m

    dims_m = _detection_dimensions_m()
    if dims_m is None:
        return None, 'no_dimensions'
    detection_length_m, detection_width_m = dims_m

    try:
        detection_area_m2 = float(detection.get('area_m2') or 0.0)
    except (TypeError, ValueError):
        detection_area_m2 = 0.0
    if detection_area_m2 <= 0.0:
        detection_area_m2 = (
            abs(float(cv2.contourArea(np.asarray(yolo_poly, dtype=np.float32)))) *
            float(m_per_px) * float(m_per_px)
        )
    if detection_area_m2 <= 0.0:
        return None, 'no_dimensions'

    zone_class = int(detection.get('placement_class') or BLD_CLASS_MEDIUM)
    min_coverage = _yolo_object_min_coverage_for_class(zone_class, min_coverage)
    det_aspect = detection_length_m / max(detection_width_m, 1e-6)
    best = None
    counters = {
        'context_skipped': 0,
        'size_reject': 0,
        'coverage_reject': 0,
        'outline_reject': 0,
        'occupancy_reject': 0,
        'seen': 0,
    }

    best_cov_area_m2 = -1.0
    merged_bundle = _merged_yolo_candidates_by_zone(fit_table, zone_class)
    entries = merged_bundle['entries']
    if not entries:
        return None, 'no_candidates'
    # Vectorised size + coverage pre-filter: avoids per-candidate Python checks
    # for every clearly-too-big or below-min-coverage asset. The min_coverage
    # bound also enforces the down-stream cov_area / det_area comparison.
    #
    # Free-area downsizing: when most of the detection is already occupied (e.g.
    # a larger neighbour was placed first), gate min-coverage against the
    # remaining FREE area rather than the full detection area, so a smaller asset
    # that fills the genuine gap is admitted instead of leaving the ground empty.
    effective_cov_area_m2 = detection_area_m2
    if freearea_downsize and float(m_per_px) > 0.0 and (
        static_occ_integral is not None or spacing_occ_integral is not None
    ):
        ref = (
            static_occ_integral if static_occ_integral is not None
            else spacing_occ_integral
        )
        int_h = int(ref.shape[0]) - 1
        int_w = int(ref.shape[1]) - 1
        bx1, by1, bx2, by2 = _yolo_poly_bbox(yolo_poly)
        bx1 = max(0, min(int(bx1), int_w))
        bx2 = max(0, min(int(bx2), int_w))
        by1 = max(0, min(int(by1), int_h))
        by2 = max(0, min(int(by2), int_h))
        if bx2 > bx1 and by2 > by1:
            bbox_px = (bx2 - bx1) * (by2 - by1)
            # Union of occupancy is unknown; max() under-estimates it, which
            # over-estimates free area -> relaxation only triggers when the
            # detection is clearly occupied (conservative).
            occ_px = 0
            if static_occ_integral is not None:
                occ_px = max(occ_px, _integral_bbox_sum(
                    static_occ_integral, bx1, by1, bx2, by2))
            if spacing_occ_integral is not None:
                occ_px = max(occ_px, _integral_bbox_sum(
                    spacing_occ_integral, bx1, by1, bx2, by2))
            free_area_m2 = max(0, bbox_px - occ_px) * (float(m_per_px) ** 2)
            if free_area_m2 < detection_area_m2 * (1.0 - 1e-3):
                effective_cov_area_m2 = max(0.0, free_area_m2)
    min_cov_area_m2 = float(min_coverage) * effective_cov_area_m2
    size_ok = (
        (merged_bundle['req_l'] <= detection_length_m + 1e-6) &
        (merged_bundle['req_w'] <= detection_width_m + 1e-6)
    )
    cov_ok = merged_bundle['cov_area'] >= min_cov_area_m2 - 1e-12
    viable_mask = size_ok & cov_ok
    viable_idx = np.flatnonzero(viable_mask)
    jxi = int(jx)
    jyi = int(jy)
    _check_occupancy = (
        not skip_occupancy and
        static_occ_mask is not None and building_spacing_mask is not None
    )
    # The footprint polygon, its inside-detection test and the occupancy test
    # depend only on (bounds_m, heading_delta) for this call (jx/jy/heading/
    # m_per_px and the masks are fixed). Many ProcGen assets share dimensions
    # and sit adjacent in the coverage-sorted scan (notably the equal-coverage
    # ties the loop must walk), so memoise the geometry to avoid recomputing the
    # same cv2 footprint/intersection dozens of times per detection. Pure
    # function of the key -> result is byte-identical to recomputing.
    _geom_cache = {}
    for idx in viable_idx:
        idx = int(idx)
        class_rank, asset_class, candidate = entries[idx]
        candidate_cov_area_m2 = float(merged_bundle['cov_area'][idx])
        if best is not None and candidate_cov_area_m2 < best_cov_area_m2 - 1e-9:
            break  # remaining candidates have strictly smaller coverage; cannot beat best on score
        counters['seen'] += 1
        candidate_asset = candidate['asset']
        path = candidate_asset.get('path')
        if enabled_assets_by_path is not None:
            asset = enabled_assets_by_path.get(path)
            if asset is None:
                continue
        else:
            asset = candidate_asset
        if not residential_context and _asset_requires_residential_context(asset):
            counters['context_skipped'] += 1
            continue

        coverage = candidate_cov_area_m2 / detection_area_m2

        heading_delta = float(candidate['heading_delta'])
        bounds_m = asset['bounds_m']
        geom_key = (
            bounds_m if type(bounds_m) is tuple else tuple(bounds_m),
            heading_delta,
        )
        cached = _geom_cache.get(geom_key)
        if cached is None:
            final_heading = (float(heading) + heading_delta) % 360.0
            footprint_poly = _footprint_poly(jxi, jyi, bounds_m, final_heading, m_per_px)
            inside_ok = _footprint_inside_detection(footprint_poly, yolo_poly, m_per_px)
            occ_ok = True
            if inside_ok and _check_occupancy:
                occ_ok = _direct_yolo_poly_fits(
                    static_occ_mask,
                    building_spacing_mask,
                    footprint_poly,
                    scratch_mask=scratch_mask,
                    static_occ_integral=static_occ_integral,
                    spacing_occ_integral=spacing_occ_integral,
                    recent_spacing_mask=recent_spacing_mask,
                )
            cached = (footprint_poly, final_heading, inside_ok, occ_ok)
            _geom_cache[geom_key] = cached
        footprint_poly, final_heading, inside_ok, occ_ok = cached
        if not inside_ok:
            counters['outline_reject'] += 1
            continue
        if not occ_ok:
            counters['occupancy_reject'] += 1
            continue

        cand_aspect = (
            float(candidate['required_length_m']) /
            max(float(candidate['required_width_m']), 1e-6)
        )
        aspect_error = abs(math.log(max(cand_aspect, 1e-6) / max(det_aspect, 1e-6)))
        score = (
            -float(coverage),
            float(aspect_error),
            abs(int(asset_class) - zone_class),
            int(class_rank),
            asset.get('source', ''),
            asset.get('path', ''),
            float(candidate['heading_delta']),
        )
        selected = {
            'asset': asset,
            'heading': final_heading,
            'footprint_poly': footprint_poly,
        }
        if best is None or score < best[0]:
            best = (score, selected)
            best_cov_area_m2 = candidate_cov_area_m2

    if profile_out is not None:
        profile_out['entries'] = profile_out.get('entries', 0) + len(entries)
        profile_out['viable'] = profile_out.get('viable', 0) + int(viable_idx.size)
        profile_out['seen'] = profile_out.get('seen', 0) + int(counters['seen'])

    if best is not None:
        return best[1], 'selected'
    # No placement: compute the size/coverage reject tallies now (only needed to
    # pick the no-result status; deferred out of the hot path above).
    if not counters['size_reject'] and not counters['coverage_reject']:
        counters['size_reject'] = int((~size_ok).sum())
        counters['coverage_reject'] = int(np.count_nonzero(size_ok & ~cov_ok))
    for status in (
        'context_skipped',
        'occupancy_reject',
        'outline_reject',
        'coverage_reject',
        'size_reject',
    ):
        if counters[status]:
            return None, status
    return None, 'no_candidates'



def _yolo_poly_bbox(points):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return (
        int(math.floor(float(pts[:, 0].min()))),
        int(math.floor(float(pts[:, 1].min()))),
        int(math.ceil(float(pts[:, 0].max()))),
        int(math.ceil(float(pts[:, 1].max()))),
    )


def _bbox_grid_cells(bbox, cell_size_px):
    x1, y1, x2, y2 = bbox
    cell_size_px = max(1, int(cell_size_px))
    cx1 = int(math.floor(x1 / cell_size_px))
    cy1 = int(math.floor(y1 / cell_size_px))
    cx2 = int(math.floor(max(x1, x2) / cell_size_px))
    cy2 = int(math.floor(max(y1, y2) / cell_size_px))
    for cy in range(cy1, cy2 + 1):
        for cx in range(cx1, cx2 + 1):
            yield (cx, cy)


def _convex_intersection_area(poly_a, poly_b):
    try:
        area, _points = cv2.intersectConvexConvex(
            np.asarray(poly_a, dtype=np.float32),
            np.asarray(poly_b, dtype=np.float32),
            handleNested=True,
        )
    except cv2.error:
        return 0.0
    return max(0.0, float(area))


def _suppress_overlapping_yolo_marginal(
    detections,
    keep_min_new_frac=0.25,
    m_per_px=1.0,
    img_w=None,
    img_h=None,
    target_cell_m=2.0,
    max_grid_dim=4096,
):
    """Set-cover style keep that preserves ground coverage.

    Detections are processed smallest-first; one is kept only when it adds at
    least ``keep_min_new_frac`` of NEW ground area beyond the union of the
    already-kept detections. This removes redundant overlap (which would just
    stack buildings) while retaining detections that fill genuine gaps, so a
    densely-covered block stays covered instead of being thinned out.

    Returns ``(kept, dropped_count)``. Kept detections are returned in
    smallest-first order, so the smaller detection wins an inter-detection
    overlap conflict and the placement path packs larger assets into whatever
    remains.
    """
    keep_min_new_frac = min(1.0, max(0.0, float(keep_min_new_frac or 0.0)))
    detections = list(detections)
    if not detections or keep_min_new_frac <= 0.0:
        return detections, 0

    polys = []
    max_x = float(img_w) if img_w else 0.0
    max_y = float(img_h) if img_h else 0.0
    for det in detections:
        pts = np.asarray(det.get('points', ()), dtype=np.float32)
        if pts.shape != (4, 2):
            polys.append(None)
            continue
        polys.append(pts)
        if not img_w:
            max_x = max(max_x, float(pts[:, 0].max()))
        if not img_h:
            max_y = max(max_y, float(pts[:, 1].max()))

    grid_w = int(math.ceil(max(1.0, max_x)))
    grid_h = int(math.ceil(max(1.0, max_y)))
    # Coarse cells ~target_cell_m metres across; capped so the accumulator
    # raster stays small on large tiles.
    scale = max(1, int(round(float(target_cell_m) / max(float(m_per_px), 1e-6))))
    scale = max(scale, int(math.ceil(max(grid_w, grid_h) / float(max_grid_dim))))
    acc_h = max(1, grid_h // scale)
    acc_w = max(1, grid_w // scale)
    accumulator = np.zeros((acc_h, acc_w), dtype=np.uint8)
    scratch = np.zeros((acc_h, acc_w), dtype=np.uint8)
    inv_scale = 1.0 / float(scale)

    order = sorted(
        range(len(detections)),
        key=lambda i: (
            float(detections[i].get('area_m2', 0.0)),
            -float(detections[i].get('confidence', 0.0)),
            i,
        ),
    )
    kept = []
    dropped = 0
    # Per-detection cost is restricted to the detection's own bbox window of the
    # accumulator. A detection polygon only ever touches its bbox, so the cleared
    # scratch, the cell counts and the accumulator overlap are byte-identical to
    # scanning the whole grid — but O(bbox) instead of O(acc_w*acc_h), which
    # matters when there are tens of thousands of detections (conf<=0.05).
    for i in order:
        pts = polys[i]
        if pts is None:
            dropped += 1
            continue
        scaled = np.round(pts * inv_scale).astype(np.int32)
        bx1 = max(0, int(scaled[:, 0].min()))
        bx2 = min(acc_w, int(scaled[:, 0].max()) + 1)
        by1 = max(0, int(scaled[:, 1].min()))
        by2 = min(acc_h, int(scaled[:, 1].max()) + 1)
        if bx2 <= bx1 or by2 <= by1:
            dropped += 1
            continue
        # Clear only this bbox window, then stamp the polygon (fillPoly clips to
        # the array and only writes inside the polygon's bbox).
        scratch[by1:by2, bx1:bx2] = 0
        cv2.fillPoly(scratch, [scaled], 1)
        region = scratch[by1:by2, bx1:bx2]
        poly_cells = int(np.count_nonzero(region))
        if poly_cells <= 0:
            dropped += 1
            continue
        overlap_cells = int(np.count_nonzero(region & accumulator[by1:by2, bx1:bx2]))
        new_cells = poly_cells - overlap_cells
        if new_cells >= keep_min_new_frac * poly_cells:
            kept.append(i)
            cv2.fillPoly(accumulator, [scaled], 1)
        else:
            dropped += 1
    return [detections[i] for i in kept], dropped


def _suppress_overlapping_yolo_detections(
    detections,
    coverage_threshold=0.35,
    min_overlap_m2=25.0,
    m_per_px=1.0,
    cell_size_px=128,
    keep_mode="drop",
    keep_min_new_frac=0.25,
    img_w=None,
    img_h=None,
    pair_rule=False,
    containment_frac=0.35,
    explain_frac=0.60,
):
    """Greedily remove overlapping YOLO OBBs.

    ``keep_mode="drop"``: keep smaller detections first and drop any later
    detection that overlaps an already-kept one. The overlap must exceed
    ``min_overlap_m2`` square metres (default 0 -> any positive overlap) and,
    when ``coverage_threshold`` > 0, also cover more than that fraction of the
    smaller footprint. With both at 0 (the default) any overlap at all drops the
    larger detection.

    ``pair_rule=True`` refines the drop decision per overlap instead of the
    blanket smaller-always-wins: when the already-kept detections overlapping a
    bigger detection are substantially INSIDE it (intersection >=
    ``containment_frac`` of the smaller footprint) the bigger one survives
    unless those contained detections collectively explain >= ``explain_frac``
    of its ground area (then they are the real per-building boxes and the big
    merged box is dropped, as before). A surviving bigger detection evicts the
    contained detections whose confidence does not beat its own -- they are
    roof furniture / sub-structure boxes of the same building. Mere sliver
    contact (below ``containment_frac``) keeps the smallest-first drop.

    ``keep_mode="marginal"``: delegate to :func:`_suppress_overlapping_yolo_marginal`,
    which preserves coverage by keeping detections that add new ground area.
    """
    keep_mode = str(keep_mode or "drop").strip().lower()
    if keep_mode == "marginal":
        return _suppress_overlapping_yolo_marginal(
            detections,
            keep_min_new_frac=keep_min_new_frac,
            m_per_px=m_per_px,
            img_w=img_w,
            img_h=img_h,
        )
    coverage_threshold = float(coverage_threshold or 0.0)
    if not detections:
        return list(detections), 0
    min_overlap_px = float(min_overlap_m2 or 0.0) / max(float(m_per_px) ** 2, 1e-6)
    detections = list(detections)
    n = len(detections)
    # Vectorised prep: one stacked min/max pass replaces four numpy reductions
    # per detection, and the tight float bboxes feed the reject test below.
    polys = [None] * n
    for i, det in enumerate(detections):
        pts = np.asarray(det.get('points', ()), dtype=np.float32)
        if pts.shape == (4, 2):
            polys[i] = pts
    valid = [i for i in range(n) if polys[i] is not None]
    fmins = np.empty((n, 2), dtype=np.float64)
    fmaxs = np.empty((n, 2), dtype=np.float64)
    if valid:
        stacked = np.stack([polys[i] for i in valid]).astype(np.float64)
        fmins[valid] = stacked.min(axis=1)
        fmaxs[valid] = stacked.max(axis=1)
    order = sorted(
        range(n),
        key=lambda i: (
            float(detections[i].get('area_m2', 0.0)),
            -float(detections[i].get('confidence', 0.0)),
        ),
    )
    if pair_rule:
        containment_frac = min(1.0, max(0.0, float(containment_frac)))
        explain_frac = min(1.0, max(0.0, float(explain_frac)))
        confs = [float(det.get('confidence', 0.0)) for det in detections]
        entries = []        # detection index per kept slot
        alive = []          # eviction tombstones, parallel to entries
        entry_areas = []
        entry_bboxes = []
        grid = {}
        for i in order:
            poly = polys[i]
            if poly is None:
                continue
            area = abs(float(cv2.contourArea(poly)))
            if area <= 1.0:
                continue
            fx1 = float(fmins[i, 0])
            fy1 = float(fmins[i, 1])
            fx2 = float(fmaxs[i, 0])
            fy2 = float(fmaxs[i, 1])
            gb = (
                int(math.floor(fx1)), int(math.floor(fy1)),
                int(math.ceil(fx2)), int(math.ceil(fy2)),
            )
            candidate_slots = set()
            for cell in _bbox_grid_cells(gb, cell_size_px):
                candidate_slots.update(grid.get(cell, ()))
            contained = []
            blocked = False
            for slot in candidate_slots:
                if not alive[slot]:
                    continue
                kbx1, kby1, kbx2, kby2 = entry_bboxes[slot]
                if fx1 > kbx2 or fx2 < kbx1 or fy1 > kby2 or fy2 < kby1:
                    continue
                inter = _convex_intersection_area(poly, polys[entries[slot]])
                if inter <= min_overlap_px:
                    continue
                if inter / max(1e-6, min(area, entry_areas[slot])) >= containment_frac:
                    contained.append(slot)
                else:
                    # Sliver contact: two distinct buildings whose loose OBBs
                    # touch -- the earlier (smaller) one wins, as before.
                    blocked = True
                    break
            if not blocked and contained:
                # Containment conflict: do the kept detections inside me
                # collectively explain my ground? Rasterise them in my bbox.
                ox, oy = gb[0], gb[1]
                bw = max(1, gb[2] - ox)
                bh = max(1, gb[3] - oy)
                scale = max(1, int(math.ceil(max(bw, bh) / 512.0)))
                acc_w = max(1, bw // scale)
                acc_h = max(1, bh // scale)
                own = np.zeros((acc_h, acc_w), dtype=np.uint8)
                cv2.fillPoly(own, [np.round(
                    (poly - (ox, oy)) / scale).astype(np.int32)], 1)
                keep_mask = np.zeros_like(own)
                for slot in contained:
                    cv2.fillPoly(keep_mask, [np.round(
                        (polys[entries[slot]] - (ox, oy)) / scale
                    ).astype(np.int32)], 1)
                own_cells = int(np.count_nonzero(own))
                explained = (
                    int(np.count_nonzero(own & keep_mask)) / own_cells
                    if own_cells else 1.0
                )
                if explained >= explain_frac:
                    blocked = True  # the smalls ARE the buildings; I am a merge
                else:
                    # I am the real building: evict contained boxes that do not
                    # beat my confidence (roof furniture / sub-structures).
                    conf_i = confs[i]
                    for slot in contained:
                        if confs[entries[slot]] <= conf_i:
                            alive[slot] = False
            if blocked:
                continue
            slot = len(entries)
            entries.append(i)
            alive.append(True)
            entry_areas.append(area)
            entry_bboxes.append((fx1, fy1, fx2, fy2))
            for cell in _bbox_grid_cells(gb, cell_size_px):
                grid.setdefault(cell, []).append(slot)
        kept = [
            detections[entries[s]] for s in range(len(entries)) if alive[s]
        ]
        return kept, len(detections) - len(kept)

    kept = []
    kept_polys = []
    kept_areas = []
    kept_bboxes = []
    grid = {}
    dropped = 0
    for i in order:
        poly = polys[i]
        if poly is None:
            dropped += 1
            continue
        area = abs(float(cv2.contourArea(poly)))
        if area <= 1.0:
            dropped += 1
            continue
        fx1 = float(fmins[i, 0])
        fy1 = float(fmins[i, 1])
        fx2 = float(fmaxs[i, 0])
        fy2 = float(fmaxs[i, 1])
        bbox = (
            int(math.floor(fx1)), int(math.floor(fy1)),
            int(math.ceil(fx2)), int(math.ceil(fy2)),
        )
        candidate_ids = set()
        for cell in _bbox_grid_cells(bbox, cell_size_px):
            candidate_ids.update(grid.get(cell, ()))
        blocked = False
        for kept_idx in candidate_ids:
            kbx1, kby1, kbx2, kby2 = kept_bboxes[kept_idx]
            if fx1 > kbx2 or fx2 < kbx1 or fy1 > kby2 or fy2 < kby1:
                # Strictly disjoint tight bboxes -> zero polygon intersection,
                # which can never exceed min_overlap_px (>= 0). Skipping the
                # exact convex-intersection call here removes ~90% of them on
                # dense tiles while keeping the outcome identical.
                continue
            kept_poly = kept_polys[kept_idx]
            kept_area = kept_areas[kept_idx]
            intersection = _convex_intersection_area(poly, kept_poly)
            if intersection <= min_overlap_px:
                continue
            if coverage_threshold > 0.0:
                coverage = intersection / max(1e-6, min(area, kept_area))
                if coverage <= coverage_threshold:
                    continue
            blocked = True
            break
        if blocked:
            dropped += 1
            continue
        kept_idx = len(kept)
        kept.append(detections[i])
        kept_polys.append(poly)
        kept_areas.append(area)
        kept_bboxes.append((fx1, fy1, fx2, fy2))
        for cell in _bbox_grid_cells(bbox, cell_size_px):
            grid.setdefault(cell, []).append(kept_idx)
    return kept, dropped


def _footprint_poly_metres(x_m, y_m, bounds_m, heading_deg):
    """Return an object's 4-point footprint polygon in local metres.

    Matches the basis used by :func:`_footprint_poly_with_bbox` (right=(cos,sin),
    forward=(sin,-cos)) but in metric space, so cross-tile placements can be
    overlap-tested without a raster mask. ``bounds_m`` are local object bounds
    (xmin, xmax, zmin, zmax) relative to the object origin placed at (x_m, y_m).
    """
    rad = math.radians(float(heading_deg))
    c = math.cos(rad)
    s = math.sin(rad)
    xmin, xmax, zmin, zmax = (float(v) for v in bounds_m)
    pts = np.empty((4, 2), dtype=np.float32)
    for idx, (lx, lz) in enumerate(((xmin, zmin), (xmax, zmin), (xmax, zmax), (xmin, zmax))):
        pts[idx, 0] = x_m + (lx * c + lz * s)
        pts[idx, 1] = y_m + (lx * s - lz * c)
    return pts


def _dedupe_overlapping_placements(placements, lat, lon, cell_m=64.0):
    """Drop object placements whose footprints overlap an already-kept one.

    Overlap removal during placement is scoped per DDS texture (each texture
    rebuilds its own occupancy mask), so a building straddling the seam between
    two textures is placed twice and never cross-deduplicated -- the dominant
    source of overlaps on high-ZL tiles, which are split into many textures.
    This final tile-wide pass works in vector (metre) space: smallest footprint
    first, drop any later placement whose footprint overlaps a kept one (any
    overlap, no minimum). It also removes cross-pass overlaps (stock vs trained).

    ``placements`` is a list of ``(lon, lat, heading, path)`` tuples; returns the
    kept list (original order preserved) and the number dropped.
    """
    if not placements:
        return list(placements), 0
    mlat = 110540.0
    mlon = 111320.0 * math.cos(math.radians(float(lat) + 0.5))
    bounds_cache = {}

    def _bounds(path):
        if path in bounds_cache:
            return bounds_cache[path]
        b = _bounds_for_object_path(path)
        if b is None:
            dims = _default_object_dims(path) or _simheaven_object_dims(path)
            b = _bounds_from_dimensions(dims[0], dims[1]) if dims else None
        bounds_cache[path] = b
        return b

    n = len(placements)
    polys = [None] * n
    bboxes = [None] * n
    areas = [0.0] * n
    for i, (o_lon, o_lat, heading, path) in enumerate(placements):
        b = _bounds(path)
        if b is None:
            continue  # unmeasurable -> always keep (area 0 sorts first)
        x = (float(o_lon) - float(lon)) * mlon
        y = (float(o_lat) - float(lat)) * mlat
        poly = _footprint_poly_metres(x, y, b, heading)
        polys[i] = poly
        bboxes[i] = (
            float(poly[:, 0].min()), float(poly[:, 1].min()),
            float(poly[:, 0].max()), float(poly[:, 1].max()),
        )
        areas[i] = (float(b[1]) - float(b[0])) * (float(b[3]) - float(b[2]))

    order = sorted(range(n), key=lambda i: areas[i])
    grid = {}
    keep = [False] * n
    dropped = 0
    for i in order:
        poly = polys[i]
        if poly is None:
            keep[i] = True
            continue
        x1, y1, x2, y2 = bboxes[i]
        cx0 = int(math.floor(x1 / cell_m)); cx1 = int(math.floor(x2 / cell_m))
        cy0 = int(math.floor(y1 / cell_m)); cy1 = int(math.floor(y2 / cell_m))
        blocked = False
        seen = set()
        for cxx in range(cx0, cx1 + 1):
            for cyy in range(cy0, cy1 + 1):
                for j in grid.get((cxx, cyy), ()):
                    if j in seen:
                        continue
                    seen.add(j)
                    bj = bboxes[j]
                    if x2 < bj[0] or bj[2] < x1 or y2 < bj[1] or bj[3] < y1:
                        continue  # bbox quick reject
                    if _convex_intersection_area(poly, polys[j]) > 0.0:
                        blocked = True
                        break
                if blocked:
                    break
            if blocked:
                break
        if blocked:
            dropped += 1
            continue
        keep[i] = True
        for cxx in range(cx0, cx1 + 1):
            for cyy in range(cy0, cy1 + 1):
                grid.setdefault((cxx, cyy), []).append(i)
    return [placements[i] for i in range(n) if keep[i]], dropped


def _integral_bbox_sum(integral: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> int:
    """Return the sum in [x1:x2, y1:y2] from a cv2-style integral image."""
    return int(
        integral[y2, x2]
        - integral[y1, x2]
        - integral[y2, x1]
        + integral[y1, x1]
    )


def _poly_fits_with_integral(
    occ_mask: np.ndarray,
    pts: np.ndarray,
    scratch_mask: np.ndarray | None = None,
    occ_integral: np.ndarray | None = None,
    bbox=None,
) -> bool:
    """Return True if polygon pts have no overlap with any set pixel in occ_mask."""
    if bbox is None:
        x1 = max(0, int(pts[:, 0].min()))
        x2 = min(occ_mask.shape[1], int(pts[:, 0].max()) + 1)
        y1 = max(0, int(pts[:, 1].min()))
        y2 = min(occ_mask.shape[0], int(pts[:, 1].max()) + 1)
    else:
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(occ_mask.shape[1], int(bbox[2]))
        y2 = min(occ_mask.shape[0], int(bbox[3]))
    if x1 >= x2 or y1 >= y2:
        return True
    if occ_integral is not None:
        bbox_has_occupancy = _integral_bbox_sum(occ_integral, x1, y1, x2, y2) != 0
    else:
        bbox_has_occupancy = cv2.countNonZero(occ_mask[y1:y2, x1:x2]) != 0
    if not bbox_has_occupancy:
        return True
    if scratch_mask is None:
        tmp = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    else:
        tmp = scratch_mask[y1:y2, x1:x2]
        tmp.fill(0)
    cv2.fillPoly(tmp, [pts - np.int32([x1, y1])], 1)
    cv2.bitwise_and(occ_mask[y1:y2, x1:x2], tmp, dst=tmp)
    return cv2.countNonZero(tmp) == 0


def _find_fitting_asset(pool, rng, jx, jy, heading, m_per_px,
                        static_occ_mask, building_spacing_mask, fit_scratch,
                        file_counts, mark_pad_m=None, prefer_small=False,
                        residential_context=True, footprint_pad_m=FOOTPRINT_PAD_M,
                        static_occ_integral=None, fit_cache_enabled=True,
                        retry_context=None, prefer_largest_fit=False):
    """Return the selected asset when it fits this candidate."""
    unknown_skipped = 0
    candidate_fit_cache = {}
    angle_basis_cache = {}

    def _bounds_key(bounds):
        return tuple(float(v) for v in bounds)

    def _angle_basis(angle):
        angle_norm = float(angle) % 360.0
        cached = angle_basis_cache.get(angle_norm)
        if cached is None:
            rad = math.radians(float(angle))
            sin_a = math.sin(rad)
            cos_a = math.cos(rad)
            cached = (1.0 / float(m_per_px), cos_a, sin_a, sin_a, -cos_a)
            angle_basis_cache[angle_norm] = cached
        return angle_norm, cached

    def _orientation_fit(bounds_m, fit_bounds, mark_bounds, bounds_cache_key, angle):
        if bounds_cache_key is None:
            raw_key = _bounds_key(bounds_m)
            fit_key = _bounds_key(fit_bounds)
            mark_key = _bounds_key(mark_bounds)
            bounds_cache_key = (raw_key, fit_key, mark_key)
        else:
            raw_key, fit_key, mark_key = bounds_cache_key
        angle_norm, basis = _angle_basis(angle)
        cache_key = (bounds_cache_key, angle_norm)
        cached = candidate_fit_cache.get(cache_key)
        if cached is not None:
            return cached

        static_poly, static_bbox = _footprint_poly_with_bbox_basis(
            jx, jy, bounds_m, *basis
        )
        dynamic_poly, dynamic_bbox = _footprint_poly_with_bbox_basis(
            jx, jy, fit_bounds, *basis
        )
        if (
            _poly_fits_with_integral(
                static_occ_mask, static_poly, fit_scratch, static_occ_integral,
                bbox=static_bbox,
            ) and
            _poly_fits_with_integral(
                building_spacing_mask, dynamic_poly, fit_scratch,
                bbox=dynamic_bbox,
            )
        ):
            mark_poly, _ = _footprint_poly_with_bbox_basis(
                jx, jy, mark_bounds, *basis
            )
            result = (
                True,
                static_poly,
                mark_poly,
            )
        else:
            result = (False, None, None)
        candidate_fit_cache[cache_key] = result
        return result

    def _fit_asset(asset):
        bounds_m = asset.get('bounds_m')
        if bounds_m is None:
            return None, None, None, None, 1

        fit_bounds = asset.get('fit_bounds_m')
        mark_bounds = asset.get('mark_bounds_m')
        bounds_cache_key = asset.get('fit_cache_key')
        if fit_bounds is None or mark_bounds is None:
            fit_bounds = _expand_bounds(bounds_m, footprint_pad_m)
            mark_pad = (
                footprint_pad_m + PLACEMENT_MARGIN_M
                if mark_pad_m is None else float(mark_pad_m)
            )
            mark_bounds = _expand_bounds(bounds_m, mark_pad)
            bounds_cache_key = None

        heading_options = _orientation_angles_for_bounds(bounds_m, heading)
        if not fit_cache_enabled:
            for fit_heading in heading_options:
                static_poly = _footprint_poly(jx, jy, bounds_m, fit_heading, m_per_px)
                dynamic_poly = _footprint_poly(jx, jy, fit_bounds, fit_heading, m_per_px)
                file_counts['fit_checks'] = file_counts.get('fit_checks', 0) + 1
                if (
                    _poly_fits(static_occ_mask, static_poly, fit_scratch) and
                    _poly_fits(building_spacing_mask, dynamic_poly, fit_scratch)
                ):
                    return (
                        asset,
                        fit_heading,
                        static_poly,
                        _footprint_poly(jx, jy, mark_bounds, fit_heading, m_per_px),
                        0,
                    )
            return None, None, None, None, 0

        for fit_heading in heading_options:
            file_counts['fit_checks'] = file_counts.get('fit_checks', 0) + 1
            fits, static_poly, mark_poly = _orientation_fit(
                bounds_m, fit_bounds, mark_bounds, bounds_cache_key, fit_heading
            )
            if fits:
                return asset, fit_heading, static_poly, mark_poly, 0
        return None, None, None, None, 0

    if prefer_largest_fit:
        skipped_residential = 0
        for asset in sorted(pool, key=_asset_retry_sort_key, reverse=True):
            if not residential_context and _asset_requires_residential_context(asset):
                skipped_residential += 1
                continue
            fitted_asset, final_h, footprint_poly, spacing_poly, skipped_unknown = _fit_asset(asset)
            unknown_skipped += skipped_unknown
            if fitted_asset is not None:
                if skipped_residential:
                    file_counts['residential_asset_skipped'] = (
                        file_counts.get('residential_asset_skipped', 0) + skipped_residential
                    )
                file_counts['largest_fit_asset_selected'] = (
                    file_counts.get('largest_fit_asset_selected', 0) + 1
                )
                return fitted_asset, final_h, footprint_poly, spacing_poly, unknown_skipped
        if skipped_residential:
            file_counts['residential_asset_skipped'] = (
                file_counts.get('residential_asset_skipped', 0) + skipped_residential
            )
        return None, None, None, None, unknown_skipped

    selected_asset = None
    for asset, skipped_before in _asset_retry_sequence_for_context(
        pool,
        rng,
        prefer_small=prefer_small,
        residential_context=residential_context,
        retry_context=retry_context,
    ):
        if skipped_before:
            file_counts['residential_asset_skipped'] = (
                file_counts.get('residential_asset_skipped', 0) + int(skipped_before)
            )
        if asset is None:
            break
        if not residential_context and _asset_requires_residential_context(asset):
            file_counts['residential_asset_skipped'] = (
                file_counts.get('residential_asset_skipped', 0) + 1
            )
            continue
        selected_asset = asset
        break

    if selected_asset is None:
        return None, None, None, None, unknown_skipped

    fitted_asset, final_h, footprint_poly, spacing_poly, skipped_unknown = _fit_asset(selected_asset)
    unknown_skipped += skipped_unknown
    if fitted_asset is not None:
        return fitted_asset, final_h, footprint_poly, spacing_poly, unknown_skipped
    return None, None, None, None, unknown_skipped


def _expand_bounds(bounds_m, pad_m: float):
    xmin, xmax, zmin, zmax = bounds_m
    return (xmin - pad_m, xmax + pad_m, zmin - pad_m, zmax + pad_m)


def _mark_poly(occ_mask: np.ndarray, pts: np.ndarray) -> None:
    """Fill polygon footprint into occ_mask (in-place)."""
    cv2.fillPoly(occ_mask, [np.int32(pts)], 1)


def _mark_dynamic_center_blockers(center_block_masks, cx, cy, heading, mark_bounds,
                                  m_per_px, class_min_fit_inradius_m):
    """Mark centers where no future class-minimum footprint can fit."""
    if mark_bounds is None:
        return
    rad = math.radians(float(heading))
    sin_a = math.sin(rad)
    cos_a = math.cos(rad)
    basis = (1.0 / float(m_per_px), cos_a, sin_a, sin_a, -cos_a)
    for zone_class, center_mask in center_block_masks.items():
        radius_m = class_min_fit_inradius_m.get(zone_class, 0.0)
        if radius_m <= 0.0:
            continue
        block_bounds = _expand_bounds(mark_bounds, radius_m)
        block_poly, _ = _footprint_poly_with_bbox_basis(cx, cy, block_bounds, *basis)
        cv2.fillPoly(center_mask, [np.int32(block_poly)], 1)


def _pixel_ring_to_latlon(points_px, img_w, img_h, lat_n, lat_s, lon_w, lon_e):
    """Convert a closed or open polygon ring from image pixels to lon/lat pairs."""
    ring = []
    for px, py in points_px:
        lon_pt, lat_pt = px_to_latlon(float(px), float(py), img_w, img_h, lat_n, lat_s, lon_w, lon_e)
        ring.append((lon_pt, lat_pt))
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def _clip_facade_to_free(yolo_poly, occ_masks, min_free_px=1.0):
    """Return the largest occupancy-free sub-polygon of ``yolo_poly``, or None.

    Subtracts the union of ``occ_masks`` (already-placed buildings / static
    scenery) from the detection polygon and returns the largest remaining free
    region as a pixel ring. This lets a fallback facade fill only the genuine
    gap inside a kept overlapping detection instead of stacking on top of a
    neighbour that already occupies part of it. Returns None when the free area
    is below ``min_free_px`` (nothing worth placing).
    """
    pts = np.asarray(yolo_poly, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 3:
        return None
    mask_shape = None
    for m in occ_masks:
        if m is not None:
            mask_shape = m.shape[:2]
            break
    if mask_shape is None:
        return None
    mh, mw = int(mask_shape[0]), int(mask_shape[1])
    x1 = max(0, min(int(math.floor(float(pts[:, 0].min()))), mw))
    x2 = max(0, min(int(math.ceil(float(pts[:, 0].max()))), mw))
    y1 = max(0, min(int(math.floor(float(pts[:, 1].min()))), mh))
    y2 = max(0, min(int(math.ceil(float(pts[:, 1].max()))), mh))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    # Fast path: nothing occupied anywhere in the detection bbox -> no clipping
    # needed, return the detection polygon unchanged (avoids findContours).
    any_occ = False
    for m in occ_masks:
        if m is not None and int(cv2.countNonZero(m[y1:y2, x1:x2])) > 0:
            any_occ = True
            break
    if not any_occ:
        return pts
    free = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.fillPoly(free, [np.round(pts - np.float32([x1, y1])).astype(np.int32)], 1)
    for m in occ_masks:
        if m is None:
            continue
        free[m[y1:y2, x1:x2] != 0] = 0
    if int(cv2.countNonZero(free)) < float(min_free_px):
        return None
    contours, _ = cv2.findContours(free, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if float(cv2.contourArea(largest)) < float(min_free_px):
        return None
    eps = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, eps, True)
    ring = approx.reshape(-1, 2).astype(np.float32) + np.float32([x1, y1])
    if ring.shape[0] < 3:
        return None
    return ring


def _write_polygon_winding(handle, lonlat_ring):
    """Write a DSF polygon winding from a sequence of ``(lon, lat)`` tuples."""
    handle.write("BEGIN_WINDING\n")
    for lon_pt, lat_pt in lonlat_ring:
        handle.write(f"POLYGON_POINT {lon_pt:.7f} {lat_pt:.7f}\n")
    handle.write("END_WINDING\n")


def _merge_exclusion_rects(rects, eps=1e-6):
    """Merge overlapping/touching exclusion rectangles.

    Rects are (west, south, east, north).
    """
    rects = [tuple(map(float, r)) for r in rects if r[0] < r[2] and r[1] < r[3]]
    merged = []
    for rect in rects:
        w, s, e, n = rect
        changed = True
        while changed:
            changed = False
            new_merged = []
            for mw, ms, me, mn in merged:
                if not (e < mw - eps or me < w - eps or n < ms - eps or mn < s - eps):
                    w = min(w, mw)
                    s = min(s, ms)
                    e = max(e, me)
                    n = max(n, mn)
                    changed = True
                else:
                    new_merged.append((mw, ms, me, mn))
            merged = new_merged
        merged.append((w, s, e, n))
    return merged


# ── Core pipeline ─────────────────────────────────────────────────────────────
def run(
    tex_dir,
    lat,
    lon,
    out_dsf,
    spacing_m,
    close_k,
    open_k,
    min_zone_m2=None,
    make_viz=False,
    cache_dir=None,
    disable_cache=False,
    grid_n=HEADING_GRID_N,
    osm_roads_path=None,
    dsftool_path=None,
    skip_osm_excl_download=False,
    custom_scenery_dir=None,
    avoid_custom_scenery=True,
    smart_gap_fill=None,
    debug_image_only=False,
    allow_road_overlap=False,
    dds_filter=None,
    ignore_placement_cache=False,
    yolo_enabled=True,
    yolo_checkpoint=None,
    yolo_conf=None,
    yolo_iou=None,
    yolo_stride=None,
    yolo_max_det=None,
    yolo_imgsz=DEFAULT_YOLO_OBB_IMGSZ,
    yolo_suppress_coverage=0.0,
    yolo_suppress_min_overlap_m2=0.0,
    yolo_keep_mode="drop",
    yolo_keep_min_new_frac=0.25,
    yolo_freearea_downsize=False,
    yolo_facade_clip=False,
    yolo_no_overlap_removal=False,
    yolo_outline_tolerance=None,
    yolo_min_coverage=YOLO_OBJECT_MIN_COVERAGE,
    yolo_facade_fallback=False,
    **legacy_kwargs,
):
    legacy_min_zone_px = legacy_kwargs.pop('min_zone_px', None)
    legacy_kwargs.pop('include_default_assets', None)
    legacy_kwargs.pop('include_sfd_assets', None)
    legacy_kwargs.pop('include_simheaven_assets', None)
    if legacy_kwargs:
        bad_keys = ", ".join(sorted(legacy_kwargs))
        raise TypeError(f"run() got unexpected keyword argument(s): {bad_keys}")

    if min_zone_m2 is None:
        if legacy_min_zone_px is None:
            raise TypeError("run() missing required argument: 'min_zone_m2'")
        # Older callers passed native-ZL16 pixel area; convert back to the
        # canonical square-metre config value used by the UI.
        min_zone_m2 = float(legacy_min_zone_px) * (2.0 ** 2)

    import re as _re
    STD_RE = _re.compile(r"^(\d+)_(\d+)_([A-Za-z][A-Za-z0-9_]*)(\d{2})\.dds$", _re.IGNORECASE)
    _transient_cache_ctx = None
    sidecar_cache_dir = cache_dir
    if disable_cache:
        import tempfile as _tempfile
        _transient_cache_ctx = _tempfile.TemporaryDirectory(
            prefix="sfr_bld_nocache_"
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

    def _source_image_path(fname, source_mode, ortho_dir):
        if source_mode == 'dds':
            return os.path.join(tex_dir, fname)
        return _orthophoto_path(fname, ortho_dir)

    def _load_source_image(fname, source_mode, ortho_dir):
        p = _source_image_path(fname, source_mode, ortho_dir)
        if not p:
            return None
        if source_mode == 'dds':
            return SEGFORMER.load_dds_or_none(
                p,
                log_prefix='[SFR Bld]',
                display_name=fname,
            )
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
                '_yolo_obb.pkl',
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
        print("No DDS files found."); return
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
    file_filter = _env_patterns("O4_SFR_FILE_FILTER")
    if dds_filter:
        dds_filter = [os.path.basename(str(name)) for name in dds_filter]
        files = [name for name in files if name in set(dds_filter)]
        print(f"DDS debug filter: {', '.join(dds_filter)} -> {len(files)} files")
    if file_filter:
        files = [
            name for name in files
            if any(fnmatch.fnmatchcase(name, pattern) for pattern in file_filter)
        ]
        print(f"DDS filter: {', '.join(file_filter)} -> {len(files)} files")
    if not files:
        print("No DDS files found."); return
    _used_zls = sorted(set(int(STD_RE.match(_f).group(4)) for _f in files))
    _zl_str = f"ZL{_used_zls[0]}" if len(_used_zls) == 1 else f"mixed ZL {_used_zls}"

    # Grid geometry for visualisation
    xs = sorted(set(int(f.split('_')[1]) for f in files))
    ys = sorted(set(int(f.split('_')[0]) for f in files))
    x_step = xs[1]-xs[0] if len(xs)>1 else 16
    y_step = ys[1]-ys[0] if len(ys)>1 else 16
    x_min, y_min = xs[0], ys[0]
    n_cols = (xs[-1]-xs[0])//x_step + 1
    n_rows = (ys[-1]-ys[0])//y_step + 1

    print(f"Tile: lat={lat} lon={lon}  DDS: {len(files)} ({_zl_str})  Grid: {n_cols}×{n_rows}")
    print(
        f"Params: edge_spacing={spacing_m}m  close={close_k}px  open={open_k}px  "
        f"min_zone={min_zone_m2}m²"
    )

    # ── Load OSM roads ────────────────────────────────────────────────────────
    if osm_roads_path is None:
        # Auto-discover: <tex_dir>/../../.. = Ortho4XP root
        o4xp_root  = os.path.dirname(os.path.dirname(os.path.dirname(tex_dir)))
        lat_g = int(math.floor(lat / 10)) * 10
        lon_g = int(math.floor(lon / 10)) * 10
        lat_s_str = f"{'+' if lat >= 0 else '-'}{abs(int(lat)):02d}"
        lon_s_str = f"{'+' if lon >= 0 else '-'}{abs(int(lon)):03d}"
        lat_g_str = f"{'+' if lat_g >= 0 else '-'}{abs(lat_g):02d}"
        lon_g_str = f"{'+' if lon_g >= 0 else '-'}{abs(lon_g):03d}"
        osm_roads_path = os.path.join(
            o4xp_root, 'OSM_data',
            f'{lat_g_str}{lon_g_str}',
            f'{lat_s_str}{lon_s_str}',
            f'{lat_s_str}{lon_s_str}_big_roads.osm.bz2')
    transient_peer_dir = sidecar_cache_dir if disable_cache else None
    osm_big_roads = _load_osm_roads(osm_roads_path, cache_dir=sidecar_cache_dir)
    osm_small_roads_path = _osm_tile_peer_path(osm_roads_path, "_small_roads.osm.bz2")
    osm_all_roads_path = _transient_cache_peer_path(
        osm_roads_path,
        "_all_roads.osm.bz2",
        transient_peer_dir,
    )
    osm_small_roads = (
        _load_osm_roads(osm_small_roads_path, cache_dir=sidecar_cache_dir)
        if osm_small_roads_path else []
    )
    osm_all_roads = []
    if osm_all_roads_path:
        if not os.path.exists(osm_all_roads_path) and not skip_osm_excl_download:
            _download_and_cache_osm_roads(lat, lon, osm_all_roads_path)
        osm_all_roads = _load_osm_roads(osm_all_roads_path, cache_dir=sidecar_cache_dir)

    if osm_all_roads:
        osm_roads = osm_all_roads
        print(f"OSM all roads: {len(osm_roads)} ways from {osm_all_roads_path}")
    else:
        osm_roads = (osm_big_roads or []) + (osm_small_roads or [])
        print(
            f"OSM roads: {len(osm_roads)} ways "
            f"(big={len(osm_big_roads)} small={len(osm_small_roads)})"
        )

    # ── Exclusion data: water, airports, simHeaven buildings, railways ───────
    # OSM building polygons are parsed for diagnostics only. They describe where
    # buildings should exist, not what is already visible in the simulator.
    excl_polys = []           # hard exclusions: water / airports
    existing_bld_polys = []   # occupancy-only blockers: simHeaven facade footprints
    excl_rails = []           # road-style dicts for railways

    for suffix in ('_water.osm.bz2', '_airports.osm.bz2'):
        p = _osm_tile_peer_path(osm_roads_path, suffix) or osm_roads_path.replace(
            '_big_roads.osm.bz2', suffix
        )
        excl_polys.extend(_load_osm_closed_ways(p, cache_dir=sidecar_cache_dir))
    print(f"Exclusion polygons (water+airports): {len(excl_polys)}")

    excl_cache = _transient_cache_peer_path(
        osm_roads_path,
        '_excl_bld_rail_res.osm.bz2',
        transient_peer_dir,
    ) or osm_roads_path.replace('_big_roads.osm.bz2', '_excl_bld_rail_res.osm.bz2')
    if skip_osm_excl_download:
        ok = os.path.exists(excl_cache)
    else:
        ok = _download_and_cache_osm(lat, lon, excl_cache)
    residential_polys = []
    if ok:
        bld_polys, rail_ways, residential_polys = _parse_excl_osm(
            excl_cache, cache_dir=sidecar_cache_dir
        )
        excl_rails.extend(rail_ways)
        print(
            f"OSM buildings: {len(bld_polys)} polys (not used as blockers)  "
            f"railways: {len(rail_ways)} ways  "
            f"residential areas: {len(residential_polys)} polys"
        )
    else:
        print("OSM building/rail/residential data: unavailable (OSM cache/download failed)")

    timings = {
        'simheaven_parse': 0.0,
        'custom_scenery_parse': 0.0,
        'cache_load': 0.0,
        'dds_load': 0.0,
        'segformer_inference': 0.0,
        'trained_yolo_inference': 0.0,
        'yolo_suppress': 0.0,
        'stock_yolo_inference': 0.0,
        'zone_cleanup': 0.0,
        'lookup': 0.0,
        'road_cache': 0.0,
        'road_raster': 0.0,
        'mesh_water': 0.0,
        'existing_bld_excl': 0.0,
        'heading_grid': 0.0,
        'mask_apply': 0.0,
        'connected_components': 0.0,
        'candidate_grid': 0.0,
        'fit_loop': 0.0,
        'cache_save': 0.0,
        'viz': 0.0,
        'placement': 0.0,
        'dsf_text': 0.0,
        'dsf_compile': 0.0,
        # O4_SFR_BLD_PLACE_PROFILE sub-phase breakdown of the fit_loop.
        'place_prep': 0.0,
        'place_fits': 0.0,
        'place_select': 0.0,
        'place_blockers': 0.0,
        'place_maskfill': 0.0,
    }
    detail_timing = _env_flag("O4_SFR_TIMING_DETAIL")
    place_profile = _env_flag("O4_SFR_BLD_PLACE_PROFILE")
    slow_timing_s = _env_float("O4_SFR_TIMING_SLOW", 3.0)
    max_candidates_per_dds = max(
        0, int(_env_float("O4_SFR_BLD_MAX_CANDIDATES", BLD_MAX_CANDIDATES_PER_DDS))
    )
    if smart_gap_fill is None:
        smart_gap_fill = _env_flag("O4_SFR_BLD_SMART_GAP_FILL")
    else:
        smart_gap_fill = bool(smart_gap_fill)
    allow_inferred_fill, run_legacy_gap_fill = _building_fill_modes(smart_gap_fill)
    # Pipeline is permanently in direct-YOLO mode; detected footprints try
    # mapped objects first and fall back to facades when no object fits.
    allow_inferred_fill = False
    run_legacy_gap_fill = False
    yolo_cuda_cleanup_every = max(
        0, int(_env_float("O4_SFR_BLD_YOLO_CUDA_CLEANUP_EVERY", 0))
    )
    footprint_pad_m = max(0.0, _env_float("O4_SFR_BLD_FOOTPRINT_PAD_M", FOOTPRINT_PAD_M))
    disable_center_blockers = _env_flag("O4_SFR_BLD_DISABLE_CENTER_BLOCKERS")
    max_gap_candidates_per_dds = max(
        0, int(_env_float("O4_SFR_BLD_GAP_MAX_CANDIDATES", max_candidates_per_dds))
    )
    max_gap_candidates_per_component = max(
        1, int(_env_float("O4_SFR_BLD_GAP_MAX_PER_COMPONENT", 24))
    )
    yolo_template_gap_m = max(
        0.0, _env_float("O4_SFR_BLD_YOLO_TEMPLATE_GAP_M", spacing_m)
    )
    yolo_template_max_candidates_per_zone = max(
        1,
        int(_env_float(
            "O4_SFR_BLD_YOLO_TEMPLATE_MAX_CANDIDATES_PER_ZONE",
            YOLO_TEMPLATE_MAX_CANDIDATES_PER_ZONE,
        )),
    )
    yolo_template_heading_tol_deg = max(
        0.0,
        _env_float(
            "O4_SFR_BLD_YOLO_TEMPLATE_HEADING_TOL_DEG",
            YOLO_TEMPLATE_HEADING_TOL_DEG,
        ),
    )
    yolo_template_shape_rel_tol = max(
        0.0,
        _env_float(
            "O4_SFR_BLD_YOLO_TEMPLATE_SHAPE_REL_TOL",
            YOLO_TEMPLATE_SHAPE_REL_TOL,
        ),
    )
    yolo_template_shape_abs_tol_px = max(
        0.0,
        _env_float(
            "O4_SFR_BLD_YOLO_TEMPLATE_SHAPE_ABS_TOL_PX",
            YOLO_TEMPLATE_SHAPE_ABS_TOL_PX,
        ),
    )
    strict_fit = _env_flag("O4_SFR_BLD_STRICT_FIT")
    n_unknown_skipped = 0
    _t = time.perf_counter()

    if yolo_checkpoint is None:
        yolo_checkpoint = os.environ.get(
            "O4_SFR_BLD_YOLO_CHECKPOINT",
            DEFAULT_YOLO_OBB_CHECKPOINT,
        )
    yolo_conf = float(
        DEFAULT_YOLO_OBB_CONF if yolo_conf is None
        else yolo_conf
    )
    yolo_iou = float(
        DEFAULT_YOLO_OBB_IOU if yolo_iou is None
        else yolo_iou
    )
    yolo_stride = int(
        DEFAULT_YOLO_OBB_STRIDE if yolo_stride is None
        else yolo_stride
    )
    yolo_max_det = int(
        DEFAULT_YOLO_OBB_MAX_DET if yolo_max_det is None
        else yolo_max_det
    )
    yolo_imgsz = int(yolo_imgsz or DEFAULT_YOLO_OBB_IMGSZ)
    yolo_analysis_target_zl = max(
        1,
        _env_int("O4_SFR_BLD_YOLO_ANALYSIS_ZL", YOLO_ANALYSIS_TARGET_ZL),
    )
    # Batch stays 1 by default: the scripts/*_batch_autotune.py harnesses
    # showed batch>1 changes borderline detections on dense tiles (not
    # placement-exact), for ~0.3s/DDS saved on an RTX 4080. Opting in via the
    # env vars is OOM-safe — the inference loops retry at batch=1 on CUDA OOM.
    yolo_batch_size = max(
        1,
        _env_int("O4_SFR_BLD_YOLO_BATCH", DEFAULT_YOLO_OBB_BATCH),
    )
    stock_yolo_batch_size = max(
        1,
        _env_int("O4_SFR_STOCK_YOLO_BATCH", _stock_yolo_batch_default()),
    )
    yolo_fuse_model = _env_flag("O4_SFR_BLD_YOLO_FUSE")
    yolo_suppress_coverage = max(0.0, float(yolo_suppress_coverage or 0.0))
    yolo_suppress_min_overlap_m2 = max(
        0.0, float(yolo_suppress_min_overlap_m2 or 0.0)
    )
    yolo_keep_mode = str(yolo_keep_mode or "drop").strip().lower()
    if yolo_keep_mode not in ("drop", "marginal"):
        yolo_keep_mode = "drop"
    yolo_keep_min_new_frac = min(
        1.0, max(0.0, float(yolo_keep_min_new_frac or 0.0))
    )
    yolo_freearea_downsize = bool(yolo_freearea_downsize)
    yolo_facade_clip = bool(yolo_facade_clip)
    yolo_no_overlap_removal = bool(yolo_no_overlap_removal)
    # Overlap-free placement is the product behaviour: no two objects may overlap.
    # The main-app config modules (O4_Cfg_Vars/O4_Config_Utils/O4_GUI_Utils/
    # O4_Tile_Utils) are frozen in Ortho4XP.exe and still default this True
    # (legacy max coverage), which can't be changed without rebuilding the exe.
    # This module is loose, so force overlap removal on here regardless of the
    # frozen default. Opt back into max coverage explicitly via the env var.
    if yolo_no_overlap_removal and not _env_flag("O4_SFR_BLD_MAX_COVERAGE", False):
        yolo_no_overlap_removal = False
    if not yolo_no_overlap_removal:
        # Remove overlaps with NO minimum limit: drop on ANY overlap, smallest-first.
        # The frozen main-app config still passes the legacy thresholds (25 m^2 /
        # 0.35 coverage), which let small overlaps survive -- visible mainly on
        # higher-ZL tiles where detections (and thus their overlaps) are finer than
        # those thresholds, while coarse ZL16 overlaps exceeded them and were
        # dropped. Force the zero-threshold drop here in the loose module so the
        # behaviour is consistent at every zoom level.
        yolo_keep_mode = "drop"
        yolo_suppress_coverage = 0.0
        yolo_suppress_min_overlap_m2 = 0.0
    # Per-overlap containment rule for the zero-threshold drop: a bigger
    # detection survives when the smaller kept detections inside it explain
    # less than yolo_suppress_explain_frac of its ground area (it then evicts
    # the contained ones that do not beat its confidence); sliver contacts
    # keep the smallest-first drop. Chosen from a 9-variant overlay A/B on
    # +22+113 28512_53472_BI16 (sliver=smaller, evict=conf). Disable with
    # O4_SFR_BLD_PAIR_OVERLAP_RULE=0 to restore the blanket smaller-wins drop.
    yolo_pair_overlap_rule = _env_flag("O4_SFR_BLD_PAIR_OVERLAP_RULE", True)
    yolo_suppress_containment_frac = _env_float(
        "O4_SFR_BLD_SUPPRESS_CONTAINMENT_FRAC", 0.35
    )
    yolo_suppress_explain_frac = _env_float(
        "O4_SFR_BLD_SUPPRESS_EXPLAIN_FRAC", 0.60
    )
    # Roads/railways do NOT block building placement by default. The footprint
    # gate rejects a detection when ANY pixel of its OBB touches the static
    # mask, and the road mask is a >=6px-wide lattice through every dense
    # block, so road blocking killed most of a dense city texture (measured
    # +22+120 57072_109328_Arc17: 8,544 of 10,900 kept detections blocked
    # while full custom-scenery avoidance added only ~100 of that). Roads
    # still street-divide the gap-fill zone; scenery/water/OSM avoidance is
    # unaffected. Restore road+rail blocking with O4_SFR_BLD_ROAD_AVOIDANCE=1.
    yolo_road_block = _env_flag("O4_SFR_BLD_ROAD_AVOIDANCE", False)
    # Per-candidate object self-avoidance (selector occupancy check + OBB dedup +
    # spacing-mask marking + incremental integral rebuilds) is redundant now that
    # inter-object overlaps are removed tile-wide after placement
    # (_dedupe_overlapping_placements). Skip that per-texture work by default for
    # speed: static obstacles (roads/water/OSM/scenery) are still enforced by the
    # detection center gate + footprint containment, and detection suppression
    # still drops overlapping detections. ``_obj_avoid`` therefore gates only the
    # inter-object machinery; road avoidance and suppression stay on
    # ``yolo_no_overlap_removal``. Restore per-candidate self-avoidance (which can
    # substitute a smaller fitting object instead of dropping) via the env var.
    yolo_skip_object_self_avoid = not _env_flag("O4_SFR_BLD_OBJECT_SELF_AVOID", False)
    _obj_avoid = (not yolo_no_overlap_removal) and (not yolo_skip_object_self_avoid)
    # Minimum fraction of a detection polygon an object must cover to be placed.
    # Lower it to admit smaller objects when no near-detection-size object fits
    # (the per-class floors in YOLO_OBJECT_MIN_COVERAGE_BY_CLASS still apply).
    if yolo_min_coverage is None:
        yolo_min_coverage = YOLO_OBJECT_MIN_COVERAGE
    yolo_min_coverage = min(1.0, max(1e-6, float(yolo_min_coverage)))
    # Object-only placement by default: when no object fully fits a detection,
    # leave it empty instead of stamping a facade over the remaining polygon.
    yolo_facade_fallback = bool(yolo_facade_fallback)
    # When inter-detection overlap avoidance is on, make dedup coverage-preserving:
    # clip every overlapping detection to the free area so each keeps its
    # non-overlapping ground, instead of dropping whole detections (which would
    # leave a hole where the dropped detection extended beyond its overlap). This
    # reuses the purpose-built facade-clip machinery, so force it on. The
    # avoidance-off (max-coverage) path is untouched.
    if not yolo_no_overlap_removal:
        yolo_facade_clip = True
    # Deprecated: the fractional outline tolerance (run(yolo_outline_tolerance=...)
    # / O4_SFR_BLD_YOLO_OUTLINE_TOL) permitted objects to overhang their detection
    # polygon by a percentage of their area. Containment is now strict apart from a
    # fixed quantisation margin (YOLO_OBJECT_OUTLINE_MARGIN_M), so this knob no
    # longer has any effect. It is still accepted so existing configs do not error;
    # use yolo_min_coverage to admit smaller objects instead.
    del yolo_outline_tolerance  # accepted for back-compat, intentionally ignored
    if "O4_SFR_BLD_YOLO_ENABLED" in os.environ:
        yolo_enabled = _env_flag("O4_SFR_BLD_YOLO_ENABLED", bool(yolo_enabled))
    else:
        yolo_enabled = bool(yolo_enabled)
    yolo_model = None
    yolo_available = False
    yolo_allow_missing = _env_flag("O4_SFR_BLD_YOLO_ALLOW_MISSING")
    yolo_signature = _checkpoint_signature(yolo_checkpoint) if yolo_enabled else None
    if yolo_enabled and yolo_signature is None:
        if yolo_allow_missing:
            print(f"YOLO OBB placement: checkpoint unavailable ({yolo_checkpoint}); using SegFormer-only fallback")
            yolo_enabled = False
        else:
            raise RuntimeError(
                f"YOLO OBB checkpoint not found: {yolo_checkpoint}. "
                "Set sfr_bld_yolo_enabled=False (or O4_SFR_BLD_YOLO_ENABLED=0) to opt out, "
                "or set O4_SFR_BLD_YOLO_ALLOW_MISSING=1 to fall back to SegFormer-only placement."
            )
    elif yolo_enabled:
        print(
            f"YOLO OBB placement: enabled checkpoint={yolo_checkpoint} "
            f"conf={yolo_conf} iou={yolo_iou} stride={yolo_stride} "
            f"max_det={yolo_max_det} batch={yolo_batch_size} "
            f"analysis_zl={yolo_analysis_target_zl} fuse={yolo_fuse_model}"
        )
        print(
            "YOLO OBB placement: direct detections only "
            f"(max asset height {MAX_GENERATED_BUILDING_HEIGHT_M:.0f}m)"
        )
        print(
            "YOLO OBB placement: "
            + ("object-first mode with facade fallback" if yolo_facade_fallback
               else "object-only mode (no facade fallback)")
            + f" (strict containment, +{YOLO_OBJECT_OUTLINE_MARGIN_M:.2f}m quantisation margin)"
        )
        try:
            yolo_model = _load_yolo_obb_model(
                yolo_checkpoint,
                fuse=yolo_fuse_model,
            )
            yolo_available = True
        except Exception as exc:
            if yolo_allow_missing:
                print(
                    f"YOLO OBB placement unavailable ({exc}); using SegFormer-only fallback"
                )
                yolo_enabled = False
                yolo_model = None
            else:
                raise RuntimeError(
                    f"YOLO OBB model failed to load from {yolo_checkpoint}: {exc}. "
                    "Set O4_SFR_BLD_YOLO_ALLOW_MISSING=1 to fall back to SegFormer-only placement."
                ) from exc

    # ── Stock YOLO-OBB (DOTAv1) for static objects pre-step ───────────────────
    # Detects storage tanks, sports fields, harbor cranes, pools etc. BEFORE
    # the trained-YOLO facade pass so their footprints can occupy static_occ_mask.
    # The loader downloads the checkpoint from ultralytics/assets when it is
    # absent locally, so first-run installs don't need a manual sync step.
    stock_yolo_model = None
    try:
        stock_yolo_model = STOCKYOLO.load_stock_yolo_model(
            STOCKYOLO.DEFAULT_STOCK_YOLO_CHECKPOINT
        )
        print(
            f"Stock YOLO OBB (DOTAv1): loaded "
            f"{STOCKYOLO.DEFAULT_STOCK_YOLO_CHECKPOINT}; "
            f"batch={stock_yolo_batch_size}",
            flush=True,
        )
    except Exception as exc:
        print(f"Stock YOLO OBB unavailable ({exc}); skipping pre-step", flush=True)
        stock_yolo_model = None

    # simHeaven network roads provide the local street grid when available.
    if dsftool_path is None:
        dsftool_path = SEGFORMER._dsftool

    if dsftool_path and os.path.exists(dsftool_path):
        sh_network = _load_simheaven_network(custom_scenery_dir, lat, lon, dsftool_path, sidecar_cache_dir)
        timings['simheaven_parse'] += time.perf_counter() - _t
        print(f"simHeaven network: {len(sh_network)} road segments")
    else:
        sh_network = []
        timings['simheaven_parse'] += time.perf_counter() - _t
        print("simHeaven network: skipped (DSFTool unavailable)")

    _t = time.perf_counter()
    if dsftool_path and os.path.exists(dsftool_path):
        sh_bld_polys, sh_bld_objects = _load_simheaven_building_exclusions(
            custom_scenery_dir,
            lat,
            lon,
            dsftool_path,
            sidecar_cache_dir,
        )
    else:
        sh_bld_polys, sh_bld_objects = [], []
    timings['simheaven_parse'] += time.perf_counter() - _t
    if sh_bld_polys or sh_bld_objects:
        existing_bld_polys.extend(sh_bld_polys)
    sh_bld_index = _prepare_simheaven_objects(sh_bld_objects)
    sh_bld_sig = (_polys_signature(sh_bld_polys),
                  _simheaven_objects_signature(sh_bld_objects))
    print(f"simHeaven buildings: {len(sh_bld_objects)} objects  {len(sh_bld_polys)} facade polys")

    _t = time.perf_counter()
    if avoid_custom_scenery and dsftool_path and os.path.exists(dsftool_path):
        custom_bld_polys, custom_bld_objects, custom_bld_skipped, custom_bld_layers = (
            _load_custom_scenery_building_exclusions(
                custom_scenery_dir,
                lat,
                lon,
                out_dsf,
                dsftool_path,
                sidecar_cache_dir,
            )
        )
    else:
        custom_bld_polys, custom_bld_objects = [], []
        custom_bld_skipped = 0
        custom_bld_layers = 0
    timings['custom_scenery_parse'] += time.perf_counter() - _t
    if custom_bld_polys or custom_bld_objects:
        existing_bld_polys.extend(custom_bld_polys)
    custom_bld_index = _prepare_simheaven_objects(custom_bld_objects)
    custom_bld_sig = (
        _polys_signature(custom_bld_polys),
        _simheaven_objects_signature(custom_bld_objects),
    )
    if avoid_custom_scenery:
        print(
            "Custom scenery buildings: "
            f"{len(custom_bld_objects)} objects  "
            f"{len(custom_bld_polys)} facade polys  "
            f"layers={custom_bld_layers}  skipped_objects={custom_bld_skipped}"
        )
    else:
        print("Custom scenery buildings: disabled")

    default_assets_available = True
    sfd_assets_available = _find_library_export(custom_scenery_dir, 'sfd_global/')
    simheaven_assets_available = (
        bool(sh_bld_objects) or _find_library_export(custom_scenery_dir, 'simheaven/')
    )
    enabled_extra_library_ids = _enabled_extra_library_ids()
    # Filter every scanned export down to the ones X-Plane will actually
    # resolve on this tile: packs like SFD Global ship simHeaven-alias shims
    # inside REGION blocks (e.g. REGION scandinavia), and referencing those
    # aliases on other tiles produces "Failed to find resource" at sim load.
    _tile_natural_region = _natural_asset_region(lat + 0.5, lon + 0.5)
    runtime_library_exports = _filter_exports_for_tile_region(
        _scan_runtime_library_exports(
            custom_scenery_dir,
            include_sfd=sfd_assets_available,
            include_simheaven=simheaven_assets_available,
            extra_library_ids=enabled_extra_library_ids,
        ),
        _tile_natural_region,
    )
    stock_yolo_library_exports = _filter_exports_for_tile_region(
        _scan_runtime_library_exports(
            custom_scenery_dir,
            include_sfd=sfd_assets_available,
            include_simheaven=simheaven_assets_available,
            suffixes=(".obj", ".fac"),
        ),
        _tile_natural_region,
    )
    stock_yolo_exported_paths = _exported_virtual_library_paths(
        stock_yolo_library_exports,
        suffixes=(".obj", ".fac"),
    )
    stock_yolo_asset_map, dropped_stock_paths, dropped_stock_classes = (
        _filter_stock_yolo_asset_map(
            STOCKYOLO.STOCK_YOLO_ASSET_MAP,
            stock_yolo_exported_paths,
        )
    )
    stock_yolo_static_classes = tuple(
        cls for cls in STOCKYOLO.STATIC_DOTA_CLASSES
        if cls in stock_yolo_asset_map
    )
    if stock_yolo_model is not None:
        if stock_yolo_static_classes:
            print(
                "Stock YOLO OBB (DOTAv1): "
                f"keeping classes {stock_yolo_static_classes}",
                flush=True,
            )
        else:
            print(
                "Stock YOLO OBB (DOTAv1): no resolvable stock assets; "
                "skipping pre-step",
                flush=True,
            )
            stock_yolo_model = None
        if dropped_stock_paths:
            sample = ", ".join(dropped_stock_paths[:5])
            print(
                "Stock YOLO OBB (DOTAv1): disabled unavailable asset refs "
                f"{len(dropped_stock_paths)}"
                + (f" ({sample})" if sample else ""),
                flush=True,
            )
        if dropped_stock_classes:
            labels = ", ".join(
                f"{cls}:{STOCKYOLO.DOTA_CLASS_NAMES.get(cls, cls)}"
                for cls in dropped_stock_classes
            )
            print(
                "Stock YOLO OBB (DOTAv1): disabled classes with no assets "
                f"({labels})",
                flush=True,
            )
    exported_simheaven_object_paths = _exported_simheaven_object_paths(
        runtime_library_exports
    )
    asset_lat = lat + 0.5
    asset_lon = lon + 0.5
    natural_asset_region = _natural_asset_region(asset_lat, asset_lon)
    simheaven_package_region, simheaven_package_folder = (
        find_simheaven_package_region_for_tile(custom_scenery_dir, lat, lon)
    )
    effective_asset_region = _asset_region(
        asset_lat, asset_lon, simheaven_package_region
    )
    if simheaven_package_region:
        print(
            "Asset region: "
            f"natural={natural_asset_region}  "
            f"simHeaven={simheaven_package_region} "
            f"({simheaven_package_folder})  "
            f"effective={effective_asset_region}"
        )
    else:
        print(
            "Asset region: "
            f"natural={natural_asset_region}  "
            f"simHeaven=none  effective={effective_asset_region}"
        )

    enabled_asset_pools = [
        _build_default_asset_pools(asset_lat, asset_lon, effective_asset_region)
    ]
    if sfd_assets_available:
        enabled_asset_pools.append(
            _build_sfd_asset_pools(
                asset_lat,
                asset_lon,
                effective_asset_region,
                library_exports=runtime_library_exports,
                cache_dir=sidecar_cache_dir,
            )
        )
    if simheaven_assets_available:
        enabled_asset_pools.append(_build_simheaven_asset_pools(
            sh_bld_objects,
            asset_lat,
            asset_lon,
            effective_asset_region,
            library_exports=runtime_library_exports,
        ))
    extra_asset_pools = _build_optional_library_asset_pools(
        custom_scenery_dir=custom_scenery_dir,
        library_exports=runtime_library_exports,
        enabled_library_ids=enabled_extra_library_ids,
        cache_dir=sidecar_cache_dir,
        asset_region=effective_asset_region,
    )
    if any(extra_asset_pools.values()):
        enabled_asset_pools.append(extra_asset_pools)
    asset_pools = _merge_asset_pools(*enabled_asset_pools)
    # Auto-reject library aliases whose resolved mesh is far larger/taller than the
    # footprint their virtual-path name declares (third-party library_txt mistakes,
    # e.g. a small simHeaven residential exported to a tall SFD apartment). This
    # generalises the hand-maintained EXCLUDED_BUILDING_ASSETS list.
    if not _env_flag("O4_SFR_BLD_ALLOW_OVERSIZED_ALIASES", False):
        asset_pools, _alias_dropped, _alias_paths = _drop_oversized_aliased_assets(
            asset_pools, runtime_library_exports, sidecar_cache_dir
        )
        if _alias_dropped:
            print(
                f"[SFR Bld] Dropped {_alias_dropped} oversized library alias(es) "
                f"(resolved mesh >> declared footprint): "
                + ", ".join(_alias_paths[:8])
                + (" ..." if len(_alias_paths) > 8 else "")
            )
    smallest_asset_only = _env_flag("O4_SFR_BLD_SMALLEST_ASSET_ONLY")
    if smallest_asset_only:
        _keep_smallest_asset_per_class(asset_pools)
    asset_sources_label = _describe_asset_sources(
        default_assets_available,
        sfd_assets_available,
        simheaven_assets_available,
        sorted({
            asset.get('source', '')
            for pool in extra_asset_pools.values()
            for asset in pool
            if asset.get('source')
        }),
    )
    if smallest_asset_only:
        asset_sources_label += " (smallest asset per class)"
    print(f"Building assets: {asset_sources_label}")
    print(
        "Optional building libraries: "
        + _describe_optional_library_diagnostics(
            runtime_library_exports,
            enabled_extra_library_ids,
            effective_asset_region,
            extra_asset_pools,
        )
    )
    asset_pool_counts = _describe_asset_pool_counts(asset_pools)
    if asset_pool_counts:
        print(f"Building asset pool counts: {asset_pool_counts}")
    if not any(asset_pools.values()):
        print("Building assets: none available for placement")
        return 0
    _sort_asset_pools_for_retry(asset_pools)
    asset_retry_context = {
        cls: _asset_retry_context(asset_pools[cls])
        for cls in BLD_PLACEMENT_CLASSES
    }
    class_min_footprint_span_m = _class_min_footprint_span_m(asset_pools)
    mark_pad_m = _mark_pad_for_edge_spacing_m(spacing_m, footprint_pad_m)
    for pool in asset_pools.values():
        for asset in pool:
            asset['requires_residential_context'] = _asset_requires_residential_context(asset)
            bounds_m = asset.get('bounds_m')
            if bounds_m is None:
                continue
            asset['fit_bounds_m'] = _expand_bounds(bounds_m, footprint_pad_m)
            asset['mark_bounds_m'] = _expand_bounds(
                bounds_m, mark_pad_m
            )
            asset['fit_cache_key'] = (
                tuple(float(v) for v in bounds_m),
                tuple(float(v) for v in asset['fit_bounds_m']),
                tuple(float(v) for v in asset['mark_bounds_m']),
            )
    class_min_fit_inradius_m = _class_min_fit_inradius_m(asset_pools)
    if disable_center_blockers:
        class_min_fit_inradius_m = {cls: 0.0 for cls in BLD_PLACEMENT_CLASSES}
    yolo_object_fit_table = _build_yolo_object_candidate_index(asset_pools)
    asset_catalog_signature = _asset_pools_signature(asset_pools)
    height_priors_by_class = _height_priors_by_class(asset_pools)
    height_priors_signature = _height_priors_signature(height_priors_by_class)
    enabled_yolo_object_assets_by_path = {
        asset['path']: asset
        for pool in asset_pools.values()
        for asset in pool
        if asset.get('kind') == 'object' and asset.get('path')
    }

    osm_roads = _prepare_roads(osm_roads)
    osm_roads_index = BBOX.build_bounds_index(osm_roads)
    excl_rails = _prepare_roads(excl_rails)
    excl_rails_index = BBOX.build_bounds_index(excl_rails)
    sh_network = _prepare_roads(sh_network)
    excl_polys = _prepare_polygons(excl_polys)
    excl_polys_index = BBOX.build_bounds_index(excl_polys)
    existing_bld_polys = _prepare_polygons(existing_bld_polys)
    existing_bld_polys_index = BBOX.build_bounds_index(existing_bld_polys)
    residential_polys = _prepare_polygons(residential_polys)
    residential_polys_index = BBOX.build_bounds_index(residential_polys)
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

    # Split zones with every road source we have: cached OSM all-highway ways
    # (or Ortho4XP big+small extracts) plus simHeaven network roads.
    separator_roads = (osm_roads or []) + (sh_network or [])
    # For heading, prefer simHeaven because it is the local street grid. If it
    # is unavailable, fall back to the all-road OSM source instead of leaving
    # the road heading grid empty.
    all_roads = sh_network if sh_network else osm_roads
    separator_roads_index = BBOX.build_bounds_index(separator_roads)
    heading_seg_index = _prepare_segment_arrays(all_roads)
    separator_sig = _roads_signature(separator_roads)
    rail_sig = _roads_signature(excl_rails)
    heading_sig = _roads_signature(all_roads)
    excl_poly_sig = _polys_signature(excl_polys)
    existing_bld_poly_sig = _polys_signature(existing_bld_polys)
    residential_poly_sig = _polys_signature(residential_polys)
    n_road_cache_hits = 0
    n_road_cache_misses = 0

    # Load model lazily only when a DDS has no shared vegetation class-map cache.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = proc = None

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k*2+1,)*2)
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k*2+1,)*2)

    TILE_VIZ = max(256, int(_env_float("O4_SFR_BLD_VIZ_SIZE", 512)))
    composite = np.zeros((n_rows*TILE_VIZ, n_cols*TILE_VIZ, 3), dtype=np.uint8) if make_viz else None
    footprint_composite = np.zeros_like(composite) if composite is not None else None

    placed_stock_objects = []   # list of (lon, lat, heading, obj_path) — from stock YOLO
    placed_facades = []   # list of (lonlat_ring, facade_path, height_m)
    placed_draped = []    # list of (lonlat_ring, pol_path) — stock YOLO ground polys (.pol)
    direct_yolo_facade_placements = 0
    candidate_grid_cache = {}
    def _building_cache_params(effective_yolo_batch, effective_stock_yolo_batch):
        return (
            (
                BLD_PLACEMENT_CACHE_VERSION
                if strict_fit else BLD_PLACEMENT_FAST_CACHE_VERSION
            ),
            spacing_m, close_k, open_k, min_zone_m2,
            max_candidates_per_dds,
            PLACE_UNKNOWN_OBJECTS,
            footprint_pad_m,
            mark_pad_m,
            smallest_asset_only,
            allow_inferred_fill,
            run_legacy_gap_fill,
            disable_center_blockers,
            max_gap_candidates_per_dds,
            max_gap_candidates_per_component,
            yolo_template_gap_m,
            yolo_template_max_candidates_per_zone,
            yolo_template_heading_tol_deg,
            yolo_template_shape_rel_tol,
            yolo_template_shape_abs_tol_px,
            tuple(sorted(class_min_footprint_span_m.items())),
            ROAD_CENTERLINE_WIDTH_M, ROAD_EXTRA_BUFFER_M,
            ROAD_WIDTH_PX_MIN, ROAD_DILATE_PX_MIN,
            bool(allow_road_overlap),
            separator_sig, heading_sig,
            excl_poly_sig, existing_bld_poly_sig, rail_sig, sh_bld_sig,
            custom_bld_sig, bool(avoid_custom_scenery),
            mesh_water_sig,
            residential_poly_sig,
            default_assets_available, sfd_assets_available, simheaven_assets_available,
            tuple(enabled_extra_library_ids),
            asset_catalog_signature,
            height_priors_signature,
            natural_asset_region,
            simheaven_package_region,
            effective_asset_region,
            bool(yolo_enabled), yolo_signature,
            yolo_imgsz, yolo_stride, round(float(yolo_conf), 6),
            round(float(yolo_iou), 6), yolo_max_det,
            int(yolo_analysis_target_zl),
            int(effective_yolo_batch), int(effective_stock_yolo_batch),
            bool(yolo_fuse_model),
            round(float(yolo_suppress_coverage), 6),
            round(float(yolo_suppress_min_overlap_m2), 4),
            str(yolo_keep_mode),
            round(float(yolo_keep_min_new_frac), 6),
            bool(yolo_freearea_downsize),
            bool(yolo_facade_clip),
            bool(yolo_no_overlap_removal),
            round(float(YOLO_OBJECT_OUTLINE_MARGIN_M), 6),
            round(float(yolo_min_coverage), 6),
            bool(yolo_facade_fallback),
            float(MAX_GENERATED_BUILDING_HEIGHT_M),
            # Bump on schema-breaking changes to per-DDS cache contents.
            # v14: invalidate v13 caches with native-ZL/RGB-interpreted YOLO detections.
            # v15: coverage-preserving overlap removal knobs (keep_mode/freearea/facade_clip).
            # v16: no_overlap_removal default on (roads/rail/self-overlap not avoided).
            # v17: bounded-margin containment + configurable yolo_min_coverage.
            # v18: metre-based outline margin + facade-fallback default off (objects only).
            # v19: overlap removal default on (drop, smallest-first, no min-overlap area).
            # v20: blacklist update + tile-wide cross-texture object dedup.
            # v21: auto-drop library aliases whose resolved mesh >> declared footprint.
            # v22: per-overlap containment rule (big survives when contained
            #      smalls explain < explain_frac; conf-gated eviction).
            bool(yolo_pair_overlap_rule),
            round(float(yolo_suppress_containment_frac), 6),
            round(float(yolo_suppress_explain_frac), 6),
            # v23: roads/railways no longer in static_occ_mask by default.
            bool(yolo_road_block),
            "schema=v23-roads-dont-block",
        )

    _requested_bld_params = _building_cache_params(
        yolo_batch_size, stock_yolo_batch_size
    )

    def _prepare_dds_inference(fi, fname, m):
        """Inference phase for one DDS: veg class map, trained YOLO, stock YOLO.

        Runs one DDS ahead on the prefetch worker thread so GPU inference
        overlaps the CPU placement stages of the previous DDS. Only touches
        this DDS's own cache files and thread-local accumulators; all global
        state merges happen on the main thread in file order, so the output
        is identical to the sequential path.
        """
        nonlocal model, proc, device
        import pickle as _pickle
        local_timings = {}
        file_timings = {}
        file_counts = {}

        def _rec(key, start):
            elapsed = time.perf_counter() - start
            local_timings[key] = local_timings.get(key, 0.0) + elapsed
            file_timings[key] = file_timings.get(key, 0.0) + elapsed

        prep = {
            'fi': fi,
            'fname': fname,
            'cached_bld': None,
            'no_image': False,
            'img': None,
            'veg_map': None,
            'mesh_water_mask': None,
            'mesh_water_full': False,
            'img_h': 0,
            'img_w': 0,
            'm_per_px': 1.0,
            'yolo_detections': [],
            'yolo_guidance': None,
            'effective_yolo_batch_size': yolo_batch_size,
            'yolo_batch_fell_back': False,
            'effective_stock_yolo_batch_size': stock_yolo_batch_size,
            'stock_yolo_batch_fell_back': False,
            'stock_res': None,
            'file_timings': file_timings,
            'file_counts': file_counts,
            'local_timings': local_timings,
        }
        if disable_cache:
            # Drop any stale per-DDS cache before starting this DDS.
            _remove_cache_files(_dds_cache_paths(fname))

        # ── Building placement cache ──────────────────────────────────────
        # Cache is keyed by DDS filename (encodes tile position+ZL) + params.
        _bld_cache_file = os.path.join(cache_dir, fname.replace('.dds', '_bld.pkl'))
        if (
            not ignore_placement_cache and
            not disable_cache and
            os.path.exists(_bld_cache_file)
        ):
            try:
                with open(_bld_cache_file, 'rb') as _f:
                    _cd = _pickle.load(_f)
                if _cd.get('params') == _requested_bld_params:
                    if _cd.get('source') == 'direct_yolo':
                        _placements = _cd.get('placements') or {}
                        _bad_simheaven = _count_unexported_simheaven_object_placements(
                            _placements.get('objects', ()),
                            exported_simheaven_object_paths,
                        )
                        if _bad_simheaven:
                            file_counts['stale_bld_cache_unexported_simheaven'] = (
                                int(_bad_simheaven)
                            )
                            try:
                                os.remove(_bld_cache_file)
                            except OSError:
                                pass
                            print(
                                f"  [{fi:3d}/{n_files}] {fname}  "
                                "stale bld cache ignored: "
                                f"{_bad_simheaven} unexported simHeaven object refs",
                                flush=True,
                            )
                        else:
                            prep['cached_bld'] = _placements
                            return prep
            except Exception:
                pass

        til_y_top  = int(m.group(1))
        til_x_left = int(m.group(2))
        zl         = int(m.group(4))

        # Geographic bounds
        lat_n, lat_s, lon_w, lon_e = dds_bounds(til_y_top, til_x_left, zl)

        # Inference (cached per DDS filename — filename encodes tile coords + ZL)
        cache_path = os.path.join(cache_dir, fname.replace('.dds', '_veg.npy'))
        img = None
        veg_map = None
        mesh_water_mask = None
        mesh_water_full = False
        img_h = img_w = 0
        if not disable_cache and os.path.exists(cache_path):
            _t = time.perf_counter()
            veg_map = np.load(cache_path)
            _rec('cache_load', _t)
            img_h, img_w = veg_map.shape[:2]
            _t = time.perf_counter()
            mesh_water_mask = _rasterize_mesh_water_mask(
                mesh_water_index, lat_n, lat_s, lon_w, lon_e, img_h, img_w
            )
            _rec('mesh_water', _t)
            mesh_water_px = int(np.count_nonzero(mesh_water_mask)) if mesh_water_mask is not None else 0
            mesh_water_full = mesh_water_px == int(img_h * img_w)
            if (
                not mesh_water_full and
                bool(np.all(veg_map == SEGFORMER.CLASS_WATER))
            ):
                # Earlier mesh-water shortcut builds could write a synthetic
                # all-water map into the shared inference cache.  Treat that
                # shape as stale when the fixed mesh reader says this DDS is
                # not fully water.
                veg_map = None
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
                if detail_timing:
                    print(
                        f"    [Bld stage] {fname} stale all-water class-map ignored",
                        flush=True,
                    )
            if detail_timing:
                if veg_map is not None:
                    print(f"    [Bld stage] {fname} class-map cached", flush=True)
        if veg_map is None:
            _t = time.perf_counter()
            img = _load_source_image(fname, _source_mode, _orthophoto_dir)
            if img is None:
                prep['no_image'] = True
                return prep
            img_h, img_w = img.shape[:2]
            _rec('dds_load', _t)
            _t = time.perf_counter()
            mesh_water_mask = _rasterize_mesh_water_mask(
                mesh_water_index, lat_n, lat_s, lon_w, lon_e, img_h, img_w
            )
            _rec('mesh_water', _t)
            mesh_water_full = (
                mesh_water_mask is not None and
                int(np.count_nonzero(mesh_water_mask)) == int(img_h * img_w)
            )
            if mesh_water_full:
                veg_map = np.full(
                    (img_h, img_w), SEGFORMER.CLASS_WATER, dtype=np.int8
                )
                if detail_timing:
                    print(
                        f"    [Bld stage] {fname} mesh water full; inference skipped",
                        flush=True,
                    )
            else:
                if model is None:
                    model, proc, device = SEGFORMER.load_vegetation_model(device)
                _t = time.perf_counter()
                veg_map = SEGFORMER.run_inference(model, device, img, proc)
                _rec('segformer_inference', _t)
            if not disable_cache and not mesh_water_full:
                _t = time.perf_counter()
                np.save(cache_path, veg_map)
                _rec('cache_save', _t)
            if detail_timing:
                if not mesh_water_full:
                    print(f"    [Bld stage] {fname} inference complete", flush=True)
        if mesh_water_mask is not None:
            mesh_water_px = int(np.count_nonzero(mesh_water_mask))
            file_counts['mesh_water_px'] = mesh_water_px
            mesh_water_full = mesh_water_px == int(img_h * img_w)

        # Pixel size in metres (approximate, using mid-latitude)
        mid_lat_rad = math.radians((lat_n + lat_s) / 2)
        lon_span_m  = (lon_e - lon_w) * 111320 * math.cos(mid_lat_rad)
        lat_span_m  = (lat_n - lat_s) * 110540
        m_per_px_x = lon_span_m / img_w
        m_per_px_y = lat_span_m / img_h
        m_per_px   = (m_per_px_x + m_per_px_y) / 2

        yolo_detections = []
        yolo_guidance = None
        if yolo_enabled:
            _source_path = _source_image_path(fname, _source_mode, _orthophoto_dir)
            _analysis_signature = _yolo_analysis_signature(
                fname, _source_path, img_w, img_h, zl, yolo_analysis_target_zl
            )
            _yolo_cache_file = os.path.join(cache_dir, fname.replace('.dds', '_yolo_obb.pkl'))
            _yolo_key = _yolo_obb_cache_key(
                fname, img_w, img_h, yolo_checkpoint, yolo_imgsz,
                yolo_stride, yolo_conf, yolo_iou, yolo_max_det,
                batch_size=yolo_batch_size,
                fused=yolo_fuse_model,
                analysis_signature=_analysis_signature,
                analysis_target_zl=yolo_analysis_target_zl,
            )
            if not disable_cache:
                _t = time.perf_counter()
                cached_yolo = _load_yolo_obb_cache(_yolo_cache_file, _yolo_key)
                _rec('cache_load', _t)
            else:
                cached_yolo = None
            if cached_yolo is not None:
                yolo_detections = cached_yolo
            else:
                if img is None:
                    _img_t = time.perf_counter()
                    img = _load_source_image(fname, _source_mode, _orthophoto_dir)
                    if img is not None:
                        _rec('dds_load', _img_t)
                if img is not None:
                    try:
                        if detail_timing:
                            _cuda_counts = _cuda_memory_counts_mb()
                            if _cuda_counts is not None:
                                (
                                    file_counts['yolo_cuda_alloc_before_mb'],
                                    file_counts['yolo_cuda_reserved_before_mb'],
                                ) = _cuda_counts
                        _t = time.perf_counter()
                        yolo_analysis = _load_yolo_analysis_image(
                            tex_dir,
                            fname,
                            til_y_top,
                            til_x_left,
                            m.group(3),
                            zl,
                            yolo_analysis_target_zl,
                            sidecar_cache_dir,
                            img,
                            _source_path,
                            (lat_n, lat_s, lon_w, lon_e),
                        )
                        analysis_scale_x = float(yolo_analysis['scale_x'])
                        analysis_scale_y = float(yolo_analysis['scale_y'])
                        analysis_m_per_px = (
                            float(m_per_px) * (analysis_scale_x + analysis_scale_y) / 2.0
                        )
                        file_counts['yolo_analysis_zl'] = int(yolo_analysis_target_zl)
                        file_counts['yolo_analysis_scale_x1000'] = int(
                            round(((analysis_scale_x + analysis_scale_y) / 2.0) * 1000)
                        )
                        yolo_result = _run_yolo_obb_inference(
                            yolo_model,
                            yolo_analysis['image'],
                            imgsz=yolo_imgsz,
                            stride=yolo_stride,
                            conf=yolo_conf,
                            iou=yolo_iou,
                            max_det=yolo_max_det,
                            device=("0" if torch.cuda.is_available() else "cpu"),
                            m_per_px=analysis_m_per_px,
                            batch_size=yolo_batch_size,
                            return_metadata=True,
                        )
                        yolo_detections = _scale_yolo_detections_to_image(
                            yolo_result['detections'],
                            analysis_scale_x,
                            analysis_scale_y,
                            img_w,
                            img_h,
                        )
                        prep['effective_yolo_batch_size'] = yolo_result['effective_batch']
                        prep['yolo_batch_fell_back'] = yolo_result['fell_back']
                        if yolo_result['fell_back']:
                            file_counts['yolo_batch_fallback'] = 1
                        _rec('trained_yolo_inference', _t)
                        if detail_timing:
                            _cuda_counts = _cuda_memory_counts_mb()
                            if _cuda_counts is not None:
                                (
                                    file_counts['yolo_cuda_alloc_after_mb'],
                                    file_counts['yolo_cuda_reserved_after_mb'],
                                ) = _cuda_counts
                        if not disable_cache:
                            _save_yolo_obb_cache(
                                _yolo_cache_file,
                                _yolo_obb_cache_key(
                                    fname, img_w, img_h, yolo_checkpoint, yolo_imgsz,
                                    yolo_stride, yolo_conf, yolo_iou, yolo_max_det,
                                    batch_size=yolo_result['effective_batch'],
                                    fused=yolo_fuse_model,
                                    analysis_signature=_analysis_signature,
                                    analysis_target_zl=yolo_analysis_target_zl,
                                ),
                                yolo_detections,
                            )
                    except Exception as exc:
                        print(f"    [Bld stage] {fname} YOLO OBB failed: {exc}")
                        yolo_detections = []
        # ── Stock YOLO-OBB pre-step (DOTAv1 static objects) ──────────────
        # Runs on the same loaded image; the main thread merges the returned
        # placements and OBB pixel quads in file order.
        if stock_yolo_model is not None and img is not None:
            try:
                _t = time.perf_counter()
                stock_res = _run_stock_yolo_pass_compat(
                    img,
                    model=stock_yolo_model,
                    img_w=img_w, img_h=img_h,
                    lat=lat, lon=lon,
                    lat_n=lat_n, lat_s=lat_s, lon_w=lon_w, lon_e=lon_e,
                    m_per_px=m_per_px,
                    device=("0" if torch.cuda.is_available() else "cpu"),
                    batch_size=stock_yolo_batch_size,
                    asset_map=stock_yolo_asset_map,
                    static_classes=stock_yolo_static_classes,
                )
                _rec('stock_yolo_inference', _t)
                prep['stock_res'] = stock_res
                _stock_effective_batch, _stock_batch_fell_back = (
                    _stock_yolo_batch_metadata(stock_res, stock_yolo_batch_size)
                )
                prep['effective_stock_yolo_batch_size'] = _stock_effective_batch
                prep['stock_yolo_batch_fell_back'] = _stock_batch_fell_back
                if _stock_batch_fell_back:
                    file_counts['stock_yolo_batch_fallback'] = 1
                if stock_res.counts_by_class:
                    _summary = " ".join(
                        f"{STOCKYOLO.DOTA_CLASS_NAMES.get(c, c)}={n}"
                        for c, n in sorted(stock_res.counts_by_class.items())
                    )
                    file_counts['stock_yolo_detections'] = sum(stock_res.counts_by_class.values())
                    print(f"    [Bld stage] {fname} stock YOLO: {_summary}", flush=True)
            except Exception as exc:
                print(f"    [Bld stage] {fname} stock YOLO failed: {exc}", flush=True)
        raw_yolo_count = len(yolo_detections)
        # Under avoidance, remove overlapping detections so each building area
        # keeps a single object. Honour the configured keep mode (default 'drop':
        # smallest-first, drop any detection that overlaps an already-kept one,
        # with no minimum-overlap area). 'marginal' remains available for the
        # coverage-preserving keep. Suppression runs whenever avoidance is on.
        _keep_mode = yolo_keep_mode
        _suppress_active = not yolo_no_overlap_removal
        if _suppress_active and yolo_detections:
            _t = time.perf_counter()
            yolo_detections, yolo_suppressed = _suppress_overlapping_yolo_detections(
                yolo_detections,
                coverage_threshold=yolo_suppress_coverage,
                min_overlap_m2=yolo_suppress_min_overlap_m2,
                m_per_px=m_per_px,
                keep_mode=_keep_mode,
                keep_min_new_frac=yolo_keep_min_new_frac,
                img_w=img_w,
                img_h=img_h,
                pair_rule=yolo_pair_overlap_rule,
                containment_frac=yolo_suppress_containment_frac,
                explain_frac=yolo_suppress_explain_frac,
            )
            _rec('yolo_suppress', _t)
            file_counts['yolo_raw_detections'] = raw_yolo_count
            file_counts['yolo_suppressed_overlap'] = yolo_suppressed
        file_counts['yolo_detections'] = len(yolo_detections)
        if yolo_detections:
            yolo_guidance = _build_yolo_guidance(yolo_detections, img_h, img_w, m_per_px)

        prep['img'] = img
        prep['veg_map'] = veg_map
        prep['mesh_water_mask'] = mesh_water_mask
        prep['mesh_water_full'] = mesh_water_full
        prep['img_h'] = img_h
        prep['img_w'] = img_w
        prep['m_per_px'] = m_per_px
        prep['yolo_detections'] = yolo_detections
        prep['yolo_guidance'] = yolo_guidance
        return prep

    # One-ahead inference prefetch: a single worker thread keeps the GPU busy
    # with the next DDS while the main thread runs the CPU placement stages.
    # Disable with O4_SFR_BLD_PREFETCH=0 to get the fully sequential path.
    prefetch_enabled = os.environ.get("O4_SFR_BLD_PREFETCH", "1").strip() != "0"
    n_files = len(files)
    prefetch_executor = None
    _prefetch_futures = {}
    if prefetch_enabled and n_files > 1:
        from concurrent.futures import ThreadPoolExecutor
        prefetch_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sfr-bld-prefetch"
        )

    def _submit_prefetch(start_idx):
        if prefetch_executor is None or _prefetch_futures:
            return
        for nfi in range(start_idx, n_files + 1):
            nfname = files[nfi - 1]
            nm = STD_RE.match(nfname)
            if not nm:
                continue
            _prefetch_futures[nfi] = prefetch_executor.submit(
                _prepare_dds_inference, nfi, nfname, nm
            )
            return

    t_start = time.time()
    for fi, fname in enumerate(files, 1):
        m = STD_RE.match(fname)
        if not m: continue
        _dds_cache_files = _dds_cache_paths(fname)
        file_timings = {}
        file_counts = {}
        file_t0 = time.perf_counter()

        try:
            print(f"  [{fi:3d}/{n_files}] {fname}  (starting)", flush=True)
            _prep_future = _prefetch_futures.pop(fi, None)
            if _prep_future is None and prefetch_executor is not None:
                # Not prefetched yet (first DDS): still run it on the worker
                # thread so ALL inference happens on one thread — cuDNN
                # handles are per-thread and re-warm on a new thread.
                _prep_future = prefetch_executor.submit(
                    _prepare_dds_inference, fi, fname, m
                )
            if _prep_future is not None:
                _prep = _prep_future.result()
            else:
                _prep = _prepare_dds_inference(fi, fname, m)
            _submit_prefetch(fi + 1)
            file_timings = _prep['file_timings']
            file_counts = _prep['file_counts']
            for _tk, _tv in _prep['local_timings'].items():
                timings[_tk] = timings.get(_tk, 0.0) + _tv

            # Per-tile deterministic rng so cached and non-cached tiles both reproduce.
            import pickle as _pickle
            rng_seed = int.from_bytes(
                hashlib.sha1(f"bld-place:{fname}".encode("utf-8")).digest()[:8],
                "big",
            )
            rng = np.random.default_rng(rng_seed)
            height_rng_seed = int.from_bytes(
                hashlib.sha1(f"bld-height:{fname}".encode("utf-8")).digest()[:8],
                "big",
            )
            height_rng = np.random.default_rng(height_rng_seed)
            _bld_cache_file = os.path.join(cache_dir, fname.replace('.dds', '_bld.pkl'))
            _cached_bld = _prep['cached_bld']
            if _cached_bld is not None:
                placed_stock_objects.extend(_cached_bld.get('objects', ()))
                placed_facades.extend(_cached_bld.get('facades', ()))
                cached_count = (
                    len(_cached_bld.get('objects', ())) +
                    len(_cached_bld.get('facades', ()))
                )
                direct_yolo_facade_placements += len(_cached_bld.get('facades', ()))
                print(f"  [{fi:3d}/{n_files}] {fname}  (bld cached — {cached_count} placements)", flush=True)
                continue
            if _prep['no_image']:
                continue
            _start_object_idx = len(placed_stock_objects)
            _start_facade_idx = len(placed_facades)
            _start_draped_idx = len(placed_draped)
            stock_yolo_occupied_polys = []  # OBB quads in image-pixel space for this DDS
            til_y_top  = int(m.group(1))
            til_x_left = int(m.group(2))
            zl         = int(m.group(4))

            # Geographic bounds
            lat_n, lat_s, lon_w, lon_e = dds_bounds(til_y_top, til_x_left, zl)

            img = _prep['img']
            veg_map = _prep['veg_map']
            mesh_water_mask = _prep['mesh_water_mask']
            mesh_water_full = _prep['mesh_water_full']
            img_h = _prep['img_h']
            img_w = _prep['img_w']
            m_per_px = _prep['m_per_px']
            yolo_detections = _prep['yolo_detections']
            yolo_guidance = _prep['yolo_guidance']
            _effective_bld_params = _building_cache_params(
                _prep['effective_yolo_batch_size'],
                _prep['effective_stock_yolo_batch_size'],
            )

            # Spacing in pixels at this tile's native resolution
            spacing_px_by_class = _class_spacing_px(
                spacing_m, m_per_px, class_min_footprint_span_m
            )
            road_width_px = max(
                ROAD_WIDTH_PX_MIN,
                int(round(ROAD_CENTERLINE_WIDTH_M / max(m_per_px, 1e-6))),
            )
            road_dilate_px = max(
                ROAD_DILATE_PX_MIN,
                int(round(ROAD_EXTRA_BUFFER_M / max(m_per_px, 1e-6))),
            )
            k_road = cv2.getStructuringElement(
                cv2.MORPH_RECT, (road_dilate_px * 2 + 1, road_dilate_px * 2 + 1))

            # Merge the stock YOLO pre-step placements in file order.
            stock_res = _prep['stock_res']
            if stock_res is not None:
                placed_stock_objects.extend(stock_res.placed_objects)
                placed_facades.extend(stock_res.placed_facades)
                placed_draped.extend(stock_res.placed_draped)
                stock_yolo_occupied_polys = list(stock_res.occupied_px_polys)

            # Zone cleanup
            _t = time.perf_counter()
            bld_raw = np.zeros((img_h, img_w), dtype=np.uint8)
            bld_zone = bld_raw
            sfr_road_dilated = None
            _record_elapsed(timings, file_timings, 'zone_cleanup', _t)

            if not bld_zone.any() and not yolo_detections:
                file_counts['candidates'] = 0
                file_counts['placed'] = 0
                bld_pct = 100 * np.sum(bld_raw) / (img_w * img_h)
                spacing_label = _format_class_spacing(spacing_px_by_class, m_per_px)
                class_counts = {cls: 0 for cls in BLD_PLACEMENT_CLASSES}
                print(
                    f"  [{fi:3d}/{n_files}] {fname}  "
                    f"{_describe_placement_summary(class_counts, bld_pct, 0, grid_n, spacing_label, file_counts)}"
                    f"  small-house areas=unavailable"
                )
                if not disable_cache and not ignore_placement_cache:
                    try:
                        _t = time.perf_counter()
                        with open(_bld_cache_file, 'wb') as _f:
                            _pickle.dump({
                                'params': _effective_bld_params,
                                'source': 'direct_yolo',
                                'placements': {'objects': (), 'facades': ()},
                            }, _f)
                        _record_elapsed(timings, file_timings, 'cache_save', _t)
                    except Exception:
                        pass
                continue

            TILE_EDGE_MARGIN_M = 20.0
            edge_px = max(2, int(TILE_EDGE_MARGIN_M / m_per_px))

            _t = time.perf_counter()
            local_separator_roads = _roads_for_bounds(
                separator_roads_index, lat_n, lat_s, lon_w, lon_e, pad_deg=0.001)
            local_rails = _roads_for_bounds(
                excl_rails_index, lat_n, lat_s, lon_w, lon_e, pad_deg=0.001)
            local_excl_polys = _polys_for_bounds(
                excl_polys_index, lat_n, lat_s, lon_w, lon_e, pad_deg=0.001)
            local_existing_bld_polys = _polys_for_bounds(
                existing_bld_polys_index, lat_n, lat_s, lon_w, lon_e, pad_deg=0.001)
            local_sh_bld_objects = _simheaven_objects_for_bounds(
                sh_bld_index, lat_n, lat_s, lon_w, lon_e, pad_deg=0.002)
            local_custom_bld_objects = _simheaven_objects_for_bounds(
                custom_bld_index, lat_n, lat_s, lon_w, lon_e, pad_deg=0.002)
            local_residential_roads = []
            local_residential_polys = []
            local_heading_segments = []
            nearest_heading_segments = []
            _record_elapsed(timings, file_timings, 'lookup', _t)

            _road_cache_file = os.path.join(cache_dir, fname.replace('.dds', '_road.pkl'))
            _road_key = _dds_road_cache_key(
                fname, lat_n, lat_s, lon_w, lon_e, img_h, img_w,
                grid_n, road_width_px, road_dilate_px,
                separator_sig, rail_sig, heading_sig, residential_poly_sig,
                excl_poly_sig, existing_bld_poly_sig, sh_bld_sig,
                custom_bld_sig,
            )
            _road_cached = None
            if not disable_cache:
                _t = time.perf_counter()
                _road_cached = _load_dds_road_cache(_road_cache_file, _road_key)
                _record_elapsed(timings, file_timings, 'road_cache', _t)

            if _road_cached is not None:
                road_mask = _road_cached['road_mask']
                rail_mask = _road_cached['rail_mask']
                hgrid = _road_cached['hgrid'].copy()
                n_osm_cells = int(_road_cached['n_osm_cells'])
                residential_area_mask = _road_cached.get('residential_area_mask')
                residential_area_source = _road_cached.get(
                    'residential_area_source', 'unknown'
                )
                poly_mask = _road_cached.get('poly_mask')
                existing_bld_mask = _road_cached.get('existing_bld_mask')
                sh_bld_mask = _road_cached.get('sh_bld_mask')
                custom_bld_mask = _road_cached.get('custom_bld_mask')
                n_road_cache_hits += 1
            else:
                n_road_cache_misses += 1
                _t = time.perf_counter()
                if local_separator_roads:
                    road_mask = _rasterize_roads(
                        local_separator_roads, lat_n, lat_s, lon_w, lon_e,
                        img_h, img_w, road_width_px=road_width_px)
                    road_mask = cv2.dilate(road_mask, k_road)
                else:
                    road_mask = np.zeros((img_h, img_w), dtype=np.uint8)

                if local_rails:
                    rail_mask = _rasterize_roads(
                        local_rails, lat_n, lat_s, lon_w, lon_e,
                        img_h, img_w, road_width_px=road_width_px)
                    rail_mask = cv2.dilate(rail_mask, k_road)
                else:
                    rail_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                _record_elapsed(timings, file_timings, 'road_raster', _t)

                hgrid = np.full((grid_n, grid_n), np.nan)
                n_osm_cells = 0
                residential_area_mask = None
                residential_area_source = 'none'

                poly_mask = None
                if local_excl_polys:
                    _t = time.perf_counter()
                    poly_mask = _rasterize_polygons(
                        local_excl_polys, lat_n, lat_s, lon_w, lon_e, img_h, img_w
                    )
                    _record_elapsed(timings, file_timings, 'road_raster', _t)

                existing_bld_mask = None
                if local_existing_bld_polys:
                    _t = time.perf_counter()
                    existing_bld_mask = _rasterize_polygons(
                        local_existing_bld_polys, lat_n, lat_s, lon_w, lon_e, img_h, img_w
                    )
                    _record_elapsed(timings, file_timings, 'existing_bld_excl', _t)

                sh_bld_mask = None
                if local_sh_bld_objects:
                    _t = time.perf_counter()
                    sh_bld_mask = _rasterize_simheaven_objects(
                        local_sh_bld_objects, lat_n, lat_s, lon_w, lon_e,
                        img_h, img_w, m_per_px
                    )
                    _record_elapsed(timings, file_timings, 'existing_bld_excl', _t)

                custom_bld_mask = None
                if local_custom_bld_objects:
                    _t = time.perf_counter()
                    custom_bld_mask = _rasterize_simheaven_objects(
                        local_custom_bld_objects, lat_n, lat_s, lon_w, lon_e,
                        img_h, img_w, m_per_px
                    )
                    _record_elapsed(timings, file_timings, 'existing_bld_excl', _t)

            _t = time.perf_counter()
            # static_occ_mask collects everything a trained-YOLO building must
            # avoid. With overlap removal disabled (the default), roads and
            # railways are intentionally NOT included (buildings may sit over
            # them) and trained-YOLO detections do not avoid each other; water,
            # OSM building footprints, scenery objects and stock-YOLO placements
            # are still avoided. `allow_road_overlap` drops road avoidance only.
            static_occ_mask = np.zeros_like(road_mask)
            if allow_road_overlap:
                file_counts['road_overlap_allowed'] = 1
            else:
                # Roads always carve the gap-fill zone so it stays street-divided,
                # but only block trained-YOLO placement when road avoidance is
                # explicitly restored (O4_SFR_BLD_ROAD_AVOIDANCE=1; see the
                # yolo_road_block comment in run()).
                bld_zone = bld_zone & (~road_mask)
                if yolo_road_block and not yolo_no_overlap_removal:
                    static_occ_mask = road_mask.copy()
                    if sfr_road_dilated is not None and sfr_road_dilated.any():
                        static_occ_mask = static_occ_mask | sfr_road_dilated

            if poly_mask is not None and poly_mask.any():
                bld_zone  = bld_zone & (~poly_mask)
                static_occ_mask  = static_occ_mask | poly_mask

            if mesh_water_mask is not None and mesh_water_mask.any():
                bld_zone = bld_zone & (~mesh_water_mask)
                static_occ_mask = static_occ_mask | mesh_water_mask

            if existing_bld_mask is not None and existing_bld_mask.any():
                static_occ_mask = static_occ_mask | existing_bld_mask

            if rail_mask.any():
                bld_zone  = bld_zone & (~rail_mask)
                if yolo_road_block and not (
                    yolo_no_overlap_removal or allow_road_overlap
                ):
                    static_occ_mask  = static_occ_mask | rail_mask

            if sh_bld_mask is not None and sh_bld_mask.any():
                static_occ_mask = static_occ_mask | sh_bld_mask

            if custom_bld_mask is not None and custom_bld_mask.any():
                static_occ_mask = static_occ_mask | custom_bld_mask

            DEGREE_TOL = 1e-4
            if lat_n >= lat + 1 - DEGREE_TOL:   static_occ_mask[:edge_px,  :]  = 1
            if lat_s <= lat     + DEGREE_TOL:   static_occ_mask[-edge_px:, :]  = 1
            if lon_w <= lon     + DEGREE_TOL:   static_occ_mask[:,  :edge_px]  = 1
            if lon_e >= lon + 1 - DEGREE_TOL:   static_occ_mask[:, -edge_px:]  = 1
            # Mark stock-YOLO OBBs (storage tanks, sports fields, pools, harbor
            # cranes) into both static_occ_mask and building_spacing_mask BEFORE
            # the trained-YOLO facade loop runs, so building facades don't
            # overlap the static objects we just placed.
            if stock_yolo_occupied_polys:
                for _quad in stock_yolo_occupied_polys:
                    cv2.fillPoly(static_occ_mask, [_quad], 1)
            static_occ_integral = (
                None if strict_fit else cv2.integral(static_occ_mask, sdepth=cv2.CV_32S)
            )
            building_spacing_mask = np.zeros_like(static_occ_mask)
            if stock_yolo_occupied_polys:
                for _quad in stock_yolo_occupied_polys:
                    cv2.fillPoly(building_spacing_mask, [_quad], 1)
            placed_yolo_mask = np.zeros_like(static_occ_mask)
            fit_scratch = np.zeros_like(static_occ_mask)
            pts_this = []
            dynamic_center_block_masks = {
                cls: np.zeros_like(static_occ_mask)
                for cls in BLD_PLACEMENT_CLASSES
            }
            placed_viz_polys = []
            yolo_viz_polys = []
            road_divided_zone = (
                (bld_zone != 0) &
                (static_occ_mask == 0)
            ).astype(np.uint8)
            min_zone_px = max(1, int(min_zone_m2 / max(m_per_px * m_per_px, 1e-6)))
            n_cc, cc_labels, cc_stats, cc_centroids = cv2.connectedComponentsWithStats(
                road_divided_zone, connectivity=8
            )
            cc_area = cc_stats[:, cv2.CC_STAT_AREA]
            cc_area_m2 = cc_area.astype(np.float32) * float(m_per_px * m_per_px)
            valid_labels = np.flatnonzero((np.arange(n_cc) != 0) & (cc_area >= min_zone_px))
            yolo_templates_by_zone = {}
            _record_elapsed(timings, file_timings, 'mask_apply', _t)

            if yolo_detections:
                _t_yolo_place = time.perf_counter()
                # O4_SFR_BLD_PLACE_PROFILE: per-phase wall-time + candidate-scan
                # accounting for the placement loop. perf_counter is only called
                # when the flag is on, so the default path is unaffected.
                _pf = place_profile
                _pf_prep = _pf_fits = _pf_select = _pf_blockers = _pf_maskfill = 0.0
                _pf_select_acc = {} if _pf else None
                placed_yolo_integral = cv2.integral(placed_yolo_mask, sdepth=cv2.CV_32S)
                recent_yolo_mask = np.zeros_like(placed_yolo_mask)
                recent_yolo_marks = 0
                recent_yolo_rebuild_threshold = 256
                building_spacing_integral = cv2.integral(
                    building_spacing_mask, sdepth=cv2.CV_32S
                )
                recent_spacing_mask = np.zeros_like(building_spacing_mask)
                recent_spacing_marks = 0
                recent_spacing_rebuild_threshold = 256
                # When inter-detection overlap avoidance is on, place smallest
                # detections first so the smaller one wins an overlap conflict
                # (OBB-dedup / object self-spacing). Under no_overlap_removal the
                # order is irrelevant (everything places), so leave it untouched
                # to keep the default max-coverage path byte-identical.
                if not yolo_no_overlap_removal:
                    yolo_detections = sorted(
                        yolo_detections,
                        key=lambda d: (
                            float(d.get('area_m2', 0.0)),
                            -float(d.get('confidence', 0.0)),
                        ),
                    )
                for detection in yolo_detections:
                    if _pf:
                        _t_pf = time.perf_counter()
                        prepared_yolo = _prepare_direct_yolo_detection(detection, img_w, img_h)
                        _pf_prep += time.perf_counter() - _t_pf
                    else:
                        prepared_yolo = _prepare_direct_yolo_detection(detection, img_w, img_h)
                    yolo_poly = prepared_yolo['poly']
                    yolo_bbox = prepared_yolo['bbox']
                    if not prepared_yolo['valid']:
                        file_counts['yolo_blocked'] = file_counts.get('yolo_blocked', 0) + 1
                        continue
                    yolo_viz_polys.append((yolo_poly.copy(), False))
                    jx = int(prepared_yolo['jx'])
                    jy = int(prepared_yolo['jy'])
                    if not (0 <= jx < img_w and 0 <= jy < img_h):
                        file_counts['yolo_blocked'] = file_counts.get('yolo_blocked', 0) + 1
                        continue
                    if static_occ_mask[jy, jx]:
                        file_counts['yolo_blocked'] = file_counts.get('yolo_blocked', 0) + 1
                        continue
                    if _obj_avoid and not _placed_yolo_poly_fits(
                        placed_yolo_mask,
                        yolo_poly,
                        fit_scratch,
                        placed_yolo_integral=placed_yolo_integral,
                        recent_yolo_mask=recent_yolo_mask,
                        bbox=yolo_bbox,
                    ):
                        file_counts['yolo_overlap_blocked'] = (
                            file_counts.get('yolo_overlap_blocked', 0) + 1
                        )
                        # A loose OBB overlapping an already-placed OBB does NOT
                        # mean the footprints overlap, so dropping here removes
                        # close-but-separate neighbours. With facade clipping (or
                        # the no-overlap-removal diagnostic) on, keep the detection
                        # and let the object/clip path fill its free area.
                        if not (yolo_facade_clip or yolo_no_overlap_removal):
                            file_counts['yolo_blocked'] = file_counts.get('yolo_blocked', 0) + 1
                            continue
                    _t_pf = time.perf_counter() if _pf else 0.0
                    if yolo_facade_clip or yolo_no_overlap_removal:
                        # Hard-gate only on static scenery; footprint overlap with
                        # other placed buildings is resolved by clipping below (or
                        # intentionally ignored under no-overlap-removal).
                        footprint_clear = _poly_fits_with_integral(
                            static_occ_mask, yolo_poly, fit_scratch,
                            static_occ_integral, bbox=yolo_bbox,
                        )
                    else:
                        footprint_clear = _direct_yolo_poly_fits(
                            static_occ_mask,
                            building_spacing_mask,
                            yolo_poly,
                            scratch_mask=fit_scratch,
                            static_occ_integral=static_occ_integral,
                            bbox=yolo_bbox,
                            spacing_occ_integral=building_spacing_integral,
                            recent_spacing_mask=recent_spacing_mask,
                        )
                    if _pf:
                        _pf_fits += time.perf_counter() - _t_pf
                    if not footprint_clear:
                        file_counts['yolo_blocked'] = file_counts.get('yolo_blocked', 0) + 1
                        continue

                    heading = float(detection['heading'])
                    placed_direct = False
                    try_cls = int(detection['placement_class'])
                    facade_cls = _yolo_facade_class(try_cls)
                    if facade_cls is None:
                        # Guarantee a placement for any detection that cleared the
                        # overlap/static steps: fall back to its own class (or a
                        # medium default) so the object/facade path always runs.
                        facade_cls = (
                            try_cls if try_cls in BLD_PLACEMENT_CLASSES
                            else BLD_CLASS_MEDIUM
                        )
                    # The center-block disc enforces minimum object spacing; under
                    # facade clipping it would also drop close non-overlapping
                    # neighbours, so bypass it and rely on clipping for spacing.
                    center_clear = (
                        yolo_facade_clip or yolo_no_overlap_removal or
                        not dynamic_center_block_masks[facade_cls][jy, jx]
                    )
                    if center_clear:
                        o_lon, o_lat = px_to_latlon(jx, jy, img_w, img_h,
                                                    lat_n, lat_s, lon_w, lon_e)
                        if not (lon <= o_lon < lon + 1 and lat <= o_lat < lat + 1):
                            file_counts['yolo_blocked'] = file_counts.get('yolo_blocked', 0) + 1
                            continue

                        residential_context = (
                            residential_area_mask is None or
                            bool(residential_area_mask[jy, jx])
                        )
                        _t_pf = time.perf_counter() if _pf else 0.0
                        object_candidate, object_status = _select_yolo_object_candidate(
                            yolo_object_fit_table,
                            detection,
                            yolo_poly,
                            jx,
                            jy,
                            heading,
                            m_per_px,
                            residential_context=residential_context,
                            min_coverage=yolo_min_coverage,
                            enabled_assets_by_path=enabled_yolo_object_assets_by_path,
                            static_occ_mask=static_occ_mask,
                            building_spacing_mask=building_spacing_mask,
                            scratch_mask=fit_scratch,
                            static_occ_integral=static_occ_integral,
                            spacing_occ_integral=building_spacing_integral,
                            recent_spacing_mask=recent_spacing_mask,
                            freearea_downsize=yolo_freearea_downsize,
                            skip_occupancy=not _obj_avoid,
                            profile_out=_pf_select_acc,
                        )
                        if _pf:
                            _pf_select += time.perf_counter() - _t_pf
                        if object_candidate is not None:
                            asset = object_candidate['asset']
                            final_h = float(object_candidate['heading'])
                            footprint_poly = object_candidate['footprint_poly']
                            placed_stock_objects.append((o_lon, o_lat, final_h, asset['path']))
                            placed_viz_polys.append((footprint_poly.copy(), facade_cls))
                            # Spacing/center-block marks only feed per-candidate
                            # self-avoidance; skip them when relying on tile-wide dedup.
                            if _obj_avoid:
                                _t_pf = time.perf_counter() if _pf else 0.0
                                _mark_poly(building_spacing_mask, footprint_poly)
                                cv2.fillPoly(recent_spacing_mask, [np.int32(footprint_poly)], 1)
                                if _pf:
                                    _pf_maskfill += time.perf_counter() - _t_pf
                                recent_spacing_marks += 1
                                _t_pf = time.perf_counter() if _pf else 0.0
                                _mark_dynamic_center_blockers(
                                    dynamic_center_block_masks,
                                    jx,
                                    jy,
                                    final_h,
                                    asset.get('mark_bounds_m'),
                                    m_per_px,
                                    class_min_fit_inradius_m,
                                )
                                if _pf:
                                    _pf_blockers += time.perf_counter() - _t_pf
                            pts_this.append((jx, jy, final_h, facade_cls))
                            file_counts['yolo_object_placed'] = (
                                file_counts.get('yolo_object_placed', 0) + 1
                            )
                        else:
                            status_key = {
                                'no_candidates': 'yolo_object_no_candidates',
                                'size_reject': 'yolo_object_size_reject',
                                'coverage_reject': 'yolo_object_coverage_reject',
                                'outline_reject': 'yolo_object_outline_reject',
                                'context_skipped': 'yolo_object_context_skipped',
                                'occupancy_reject': 'yolo_object_occupancy_reject',
                                'no_dimensions': 'yolo_object_no_dimensions',
                                'miss': 'yolo_object_lookup_miss',
                            }.get(object_status, 'yolo_object_lookup_miss')
                            file_counts[status_key] = file_counts.get(status_key, 0) + 1
                            final_h = heading
                            # Facade fallback: when no object fits, optionally stamp
                            # a facade over the detection polygon. Disabled by default
                            # (object-only placement) so detections that no object
                            # fully fits are left empty rather than backfilled with a
                            # facade covering the remaining polygon space.
                            facade_path = (
                                _facade_for_detection(
                                    facade_cls, veg_map, jx, jy, m_per_px,
                                    lat=float(o_lat), lon=float(o_lon),
                                    include_simheaven_assets=simheaven_assets_available,
                                )
                                if yolo_facade_fallback else None
                            )
                            footprint_poly = yolo_poly
                            place_facade = bool(yolo_facade_fallback)
                            if yolo_facade_clip and not yolo_no_overlap_removal:
                                _min_free_px = max(
                                    1.0, 10.0 / max(float(m_per_px) ** 2, 1e-6)
                                )
                                clipped_ring = _clip_facade_to_free(
                                    yolo_poly,
                                    (static_occ_mask, building_spacing_mask),
                                    min_free_px=_min_free_px,
                                )
                                if clipped_ring is None:
                                    place_facade = False
                                    file_counts['yolo_facade_clip_skipped'] = (
                                        file_counts.get('yolo_facade_clip_skipped', 0) + 1
                                    )
                                else:
                                    footprint_poly = clipped_ring
                            if place_facade:
                                placed_facades.append((
                                    _pixel_ring_to_latlon(
                                        footprint_poly, img_w, img_h,
                                        lat_n, lat_s, lon_w, lon_e,
                                    ),
                                    facade_path,
                                    _randomized_facade_height_m(
                                        height_rng, height_priors_by_class, facade_cls
                                    ),
                                ))
                                direct_yolo_facade_placements += 1
                                placed_viz_polys.append(
                                    (np.asarray(footprint_poly).copy(), facade_cls)
                                )
                                _t_pf = time.perf_counter() if _pf else 0.0
                                _mark_poly(building_spacing_mask, footprint_poly)
                                cv2.fillPoly(
                                    recent_spacing_mask, [np.int32(footprint_poly)], 1
                                )
                                if _pf:
                                    _pf_maskfill += time.perf_counter() - _t_pf
                                recent_spacing_marks += 1
                                _t_pf = time.perf_counter() if _pf else 0.0
                                _mark_dynamic_center_blockers(
                                    dynamic_center_block_masks,
                                    jx,
                                    jy,
                                    final_h,
                                    DEFAULT_FACADE_BOUNDS.get(facade_cls),
                                    m_per_px,
                                    class_min_fit_inradius_m,
                                )
                                if _pf:
                                    _pf_blockers += time.perf_counter() - _t_pf
                                pts_this.append((jx, jy, final_h, facade_cls))
                                file_counts['yolo_facade_placed'] = (
                                    file_counts.get('yolo_facade_placed', 0) + 1
                                )

                        yolo_viz_polys[-1] = (yolo_poly.copy(), True)
                        # The placed-OBB mask and the incremental occupancy
                        # integrals are only consumed by the overlap-avoidance
                        # paths (OBB dedup + selector occupancy), both skipped
                        # unless per-candidate self-avoidance is on — so don't pay
                        # to maintain them when relying on tile-wide dedup.
                        if _obj_avoid:
                            cv2.fillPoly(placed_yolo_mask, [np.int32(yolo_poly)], 1)
                            cv2.fillPoly(recent_yolo_mask, [np.int32(yolo_poly)], 1)
                            recent_yolo_marks += 1
                            if recent_yolo_marks >= recent_yolo_rebuild_threshold:
                                placed_yolo_integral = cv2.integral(
                                    placed_yolo_mask, sdepth=cv2.CV_32S
                                )
                                recent_yolo_mask.fill(0)
                                recent_yolo_marks = 0
                            if recent_spacing_marks >= recent_spacing_rebuild_threshold:
                                building_spacing_integral = cv2.integral(
                                    building_spacing_mask, sdepth=cv2.CV_32S
                                )
                                recent_spacing_mask.fill(0)
                                recent_spacing_marks = 0
                        file_counts['yolo_placed'] = file_counts.get('yolo_placed', 0) + 1
                        placed_direct = True
                    if not placed_direct:
                        file_counts['yolo_blocked'] = file_counts.get('yolo_blocked', 0) + 1
                yolo_place_elapsed = time.perf_counter() - _t_yolo_place
                timings['placement'] += yolo_place_elapsed
                file_timings['fit_loop'] = file_timings.get('fit_loop', 0.0) + yolo_place_elapsed
                if _pf:
                    for _k, _v in (
                        ('place_prep', _pf_prep),
                        ('place_fits', _pf_fits),
                        ('place_select', _pf_select),
                        ('place_blockers', _pf_blockers),
                        ('place_maskfill', _pf_maskfill),
                    ):
                        timings[_k] += _v
                        file_timings[_k] = file_timings.get(_k, 0.0) + _v
                    _acc = _pf_select_acc or {}
                    file_counts['place_select_calls'] = (
                        file_counts.get('place_select_calls', 0) + int(_acc.get('calls', 0))
                    )
                    file_counts['place_cands_scanned'] = (
                        file_counts.get('place_cands_scanned', 0) + int(_acc.get('seen', 0))
                    )
                    file_counts['place_cands_viable'] = (
                        file_counts.get('place_cands_viable', 0) + int(_acc.get('viable', 0))
                    )
                    file_counts['place_cands_offered'] = (
                        file_counts.get('place_cands_offered', 0) + int(_acc.get('entries', 0))
                    )

            cell_h = img_h // grid_n
            cell_w = img_w // grid_n

            if np.any(np.isnan(hgrid)):
                fallback_heading = float(rng.integers(0, 360))
                hgrid = np.where(np.isnan(hgrid), fallback_heading, hgrid)

            _t = time.perf_counter()
            fallback_zone = np.zeros((img_h, img_w), dtype=np.uint8)
            label_class = np.zeros(n_cc, dtype=np.uint8)
            zone_class = label_class[cc_labels]
            roof_evidence = None
            zone_heading = np.full(n_cc, np.nan, dtype=np.float32)
            side_band_px = max(
                8,
                road_width_px + road_dilate_px + 2,
                int(round(14.0 / max(m_per_px, 1e-6))),
            )
            neighbor_yolo_templates_by_zone = {}
            nearest_obb_heading = np.full(n_cc, np.nan, dtype=np.float32)
            nearest_obb_counts = np.zeros(n_cc, dtype=np.int32)
            nearest_obb_source_labels = np.full(n_cc, -1, dtype=np.int32)
            nearest_obb_distances = np.full(n_cc, np.inf, dtype=np.float32)
            cc_elapsed = _record_elapsed(timings, file_timings, 'connected_components', _t)
            timings['placement'] += cc_elapsed

            _t = time.perf_counter()
            cand_x_parts = []
            cand_y_parts = []
            cand_cls_parts = []
            cand_label_parts = []
            n_initial_blocked = 0
            n_candidates_total = 0
            candidate_available = fallback_zone != 0
            for target_cls in BLD_PLACEMENT_CLASSES:
                spacing_passes = []
                coarse_sp_px = spacing_px_by_class[target_cls]
                if target_cls > BLD_CLASS_MEDIUM and residential_area_mask is not None:
                    residential_cls = _residential_infill_class(target_cls)
                    residential_sp_px = spacing_px_by_class[residential_cls]
                    spacing_passes.append((residential_sp_px, True, residential_cls))
                    spacing_passes.append((coarse_sp_px, False, target_cls))
                else:
                    spacing_passes.append((coarse_sp_px, None, target_cls))

                for cls_sp_px, residential_only, placement_cls in spacing_passes:
                    half = cls_sp_px // 2
                    candidate_grid_key = (img_w, img_h, cls_sp_px)
                    base_candidates = candidate_grid_cache.get(candidate_grid_key)
                    if base_candidates is None:
                        xs = np.arange(half, img_w, cls_sp_px, dtype=np.int32)
                        ys = np.arange(half, img_h, cls_sp_px, dtype=np.int32)
                        if xs.size and ys.size:
                            grid_x, grid_y = np.meshgrid(xs, ys)
                            base_candidates = (grid_x.ravel(), grid_y.ravel())
                        else:
                            base_candidates = (
                                np.empty(0, dtype=np.int32),
                                np.empty(0, dtype=np.int32),
                            )
                        candidate_grid_cache[candidate_grid_key] = base_candidates

                    base_x, base_y = base_candidates
                    if not (base_x.size and base_y.size):
                        continue

                    n_candidates = base_x.size
                    jitter = rng.integers(
                        -half, half + 1, size=(n_candidates, 2), dtype=np.int32
                    )
                    cls_cand_x = np.clip(base_x + jitter[:, 0], 0, img_w - 1)
                    cls_cand_y = np.clip(base_y + jitter[:, 1], 0, img_h - 1)
                    keep = (
                        candidate_available[cls_cand_y, cls_cand_x] &
                        (zone_class[cls_cand_y, cls_cand_x] == target_cls)
                    )
                    if residential_only is True:
                        keep &= residential_area_mask[cls_cand_y, cls_cand_x] != 0
                    elif residential_only is False:
                        keep &= residential_area_mask[cls_cand_y, cls_cand_x] == 0
                    cls_cand_x = cls_cand_x[keep]
                    cls_cand_y = cls_cand_y[keep]
                    n_candidates_total += int(cls_cand_x.size)
                    if not cls_cand_x.size:
                        continue

                    cls_labels = cc_labels[cls_cand_y, cls_cand_x]
                    open_center = static_occ_mask[cls_cand_y, cls_cand_x] == 0
                    n_initial_blocked += int(
                        cls_cand_x.size - np.count_nonzero(open_center)
                    )
                    cls_cand_x = cls_cand_x[open_center]
                    cls_cand_y = cls_cand_y[open_center]
                    cls_labels = cls_labels[open_center]
                    if not cls_cand_x.size:
                        continue

                    cand_x_parts.append(cls_cand_x)
                    cand_y_parts.append(cls_cand_y)
                    cand_cls_parts.append(
                        np.full(cls_cand_x.shape, placement_cls, dtype=np.uint8)
                    )
                    cand_label_parts.append(cls_labels)

            file_counts['candidates'] = n_candidates_total
            file_counts['initial_center_blocked'] = n_initial_blocked
            if cand_x_parts:
                cand_x = np.concatenate(cand_x_parts)
                cand_y = np.concatenate(cand_y_parts)
                cand_cls = np.concatenate(cand_cls_parts)
                cand_labels = np.concatenate(cand_label_parts)
                cand_cls, n_local_class_refined = _refine_candidate_classes_from_roofs(
                    cand_x, cand_y, cand_cls, roof_evidence, m_per_px
                )
                if n_local_class_refined:
                    file_counts['local_roof_class_refined'] = int(n_local_class_refined)
                if max_candidates_per_dds and cand_x.size > max_candidates_per_dds:
                    cand_x, cand_y, cand_cls, cand_labels, n_dropped = _limit_candidates_by_component(
                        cand_x, cand_y, cand_cls, cand_labels, max_candidates_per_dds, rng
                    )
                    file_counts['candidate_cap'] = int(n_dropped)
            else:
                cand_x = np.empty(0, dtype=np.int32)
                cand_y = np.empty(0, dtype=np.int32)
                cand_cls = np.empty(0, dtype=np.uint8)
            if not allow_inferred_fill:
                if cand_x.size:
                    file_counts['inferred_fill_disabled_candidates'] = int(cand_x.size)
                cand_x = np.empty(0, dtype=np.int32)
                cand_y = np.empty(0, dtype=np.int32)
                cand_cls = np.empty(0, dtype=np.uint8)
                cand_labels = np.empty(0, dtype=np.int32)
            candidate_elapsed = _record_elapsed(timings, file_timings, 'candidate_grid', _t)
            timings['placement'] += candidate_elapsed
            if detail_timing:
                fit_input_count = int(cand_x.size)
                print(
                    f"    [Bld stage] {fname} placement start  candidates={fit_input_count}",
                    flush=True,
                )

            _t = time.perf_counter()

            def _zone_yolo_template(jx, jy, zone_label):
                zone_templates = yolo_templates_by_zone.get(int(zone_label), ())
                if not zone_templates:
                    return None
                return min(
                    zone_templates,
                    key=lambda item: (
                        float(item['center'][0] - jx) ** 2 +
                        float(item['center'][1] - jy) ** 2,
                        -float(item.get('confidence', 0.0)),
                    ),
                )

            def _dominant_zone_yolo_template(zone_label):
                zone_templates = yolo_templates_by_zone.get(int(zone_label), ())
                if not zone_templates:
                    return None
                return _consensus_yolo_template(
                    zone_templates,
                    heading_tol_deg=yolo_template_heading_tol_deg,
                    shape_rel_tol=yolo_template_shape_rel_tol,
                    shape_abs_tol_px=yolo_template_shape_abs_tol_px,
                )

            def _zone_road_heading(zone_label):
                zone_label = int(zone_label)
                if (
                    0 <= zone_label < zone_heading.shape[0] and
                    not np.isnan(zone_heading[zone_label])
                ):
                    return float(zone_heading[zone_label])
                if 0 <= zone_label < cc_centroids.shape[0]:
                    cx = int(np.clip(round(float(cc_centroids[zone_label, 0])), 0, img_w - 1))
                    cy = int(np.clip(round(float(cc_centroids[zone_label, 1])), 0, img_h - 1))
                    return float(hgrid[
                        min(grid_n - 1, cy // cell_h),
                        min(grid_n - 1, cx // cell_w),
                    ])
                return 0.0

            _neighbor_tpl_cache: dict = {}

            def _dominant_neighbor_yolo_template(zone_label):
                zone_int = int(zone_label)
                if zone_int in _neighbor_tpl_cache:
                    return _neighbor_tpl_cache[zone_int]
                zone_templates = neighbor_yolo_templates_by_zone.get(zone_int, ())
                if not zone_templates:
                    _neighbor_tpl_cache[zone_int] = None
                    return None
                consensus = _consensus_yolo_template(
                    zone_templates,
                    heading_tol_deg=yolo_template_heading_tol_deg,
                    shape_rel_tol=yolo_template_shape_rel_tol,
                    shape_abs_tol_px=yolo_template_shape_abs_tol_px,
                )
                if consensus is not None:
                    consensus['template_source'] = 'neighbor_zone'
                _neighbor_tpl_cache[zone_int] = consensus
                return consensus

            def _old_heading_for_candidate(jx, jy, zone_label):
                roof_h = _roof_heading_for_candidate(roof_evidence, jx, jy, m_per_px)
                if not np.isnan(roof_h):
                    file_counts['local_roof_heading'] = (
                        file_counts.get('local_roof_heading', 0) + 1
                    )
                    return roof_h
                local_h = float(hgrid[
                    min(grid_n - 1, jy // cell_h),
                    min(grid_n - 1, jx // cell_w),
                ])
                if (
                    0 <= zone_label < zone_heading.shape[0] and
                    not np.isnan(zone_heading[zone_label]) and
                    cc_area_m2[zone_label] < ZL16_LOCAL_HEADING_ZONE_M2
                ):
                    return float(zone_heading[zone_label])
                return local_h

            def _heading_for_candidate(jx, jy, zone_label, zone_template=None):
                if zone_template is None:
                    zone_template = _zone_yolo_template(jx, jy, zone_label)
                if zone_template is not None:
                    file_counts['yolo_heading'] = (
                        file_counts.get('yolo_heading', 0) + 1
                    )
                    return float(zone_template['heading'])
                if int(zone_label) not in yolo_templates_by_zone:
                    if (
                        0 <= int(zone_label) < nearest_obb_heading.shape[0] and
                        not np.isnan(nearest_obb_heading[int(zone_label)])
                    ):
                        file_counts['non_obb_zone_nearest_obb_heading'] = (
                            file_counts.get('non_obb_zone_nearest_obb_heading', 0) + 1
                        )
                        return float(nearest_obb_heading[int(zone_label)])
                    file_counts['non_obb_zone_heading_fallback'] = (
                        file_counts.get('non_obb_zone_heading_fallback', 0) + 1
                    )
                    return _old_heading_for_candidate(jx, jy, zone_label)
                return _old_heading_for_candidate(jx, jy, zone_label)

            def _try_place_yolo_template(
                jx, jy, zone_label, count_key, template=None, use_center_blockers=True
            ):
                best = template if template is not None else _zone_yolo_template(jx, jy, zone_label)
                if best is None:
                    best = _dominant_neighbor_yolo_template(zone_label)
                if best is None:
                    return False
                try_cls = int(best['class'])
                final_h = float(best['heading'])
                if try_cls not in BLD_PLACEMENT_CLASSES:
                    return False
                if use_center_blockers and dynamic_center_block_masks[try_cls][jy, jx]:
                    return False
                footprint_poly, footprint_scales = _largest_fitting_yolo_template_poly(
                    best,
                    jx,
                    jy,
                    img_w,
                    img_h,
                    static_occ_mask,
                    building_spacing_mask,
                    fit_scratch,
                    static_occ_integral=static_occ_integral,
                )
                if footprint_poly is None:
                    file_counts['yolo_template_blocked'] = (
                        file_counts.get('yolo_template_blocked', 0) + 1
                    )
                    return False
                long_scale, short_scale = footprint_scales
                if long_scale < 0.999 or short_scale < 0.999:
                    file_counts['yolo_template_shrunk'] = (
                        file_counts.get('yolo_template_shrunk', 0) + 1
                    )
                    file_counts['yolo_template_long_scale_sum_x1000'] = (
                        file_counts.get('yolo_template_long_scale_sum_x1000', 0) +
                        int(round(float(long_scale) * 1000))
                    )
                    file_counts['yolo_template_short_scale_sum_x1000'] = (
                        file_counts.get('yolo_template_short_scale_sum_x1000', 0) +
                        int(round(float(short_scale) * 1000))
                    )

                o_lon, o_lat = px_to_latlon(jx, jy, img_w, img_h,
                                            lat_n, lat_s, lon_w, lon_e)
                if not (lon <= o_lon < lon + 1 and lat <= o_lat < lat + 1):
                    return False

                facade_path = _facade_for_detection(
                    try_cls, veg_map, jx, jy, m_per_px,
                    lat=float(o_lat), lon=float(o_lon),
                    include_simheaven_assets=simheaven_assets_available,
                )
                pts_this.append((jx, jy, final_h, try_cls))
                placed_facades.append((
                    _pixel_ring_to_latlon(
                        footprint_poly, img_w, img_h, lat_n, lat_s, lon_w, lon_e
                    ),
                    facade_path,
                    _randomized_facade_height_m(
                        height_rng, height_priors_by_class, try_cls
                    ),
                ))
                placed_viz_polys.append((footprint_poly.copy(), try_cls))
                _mark_poly(building_spacing_mask, footprint_poly)
                if use_center_blockers:
                    _mark_dynamic_center_blockers(
                        dynamic_center_block_masks,
                        jx,
                        jy,
                        final_h,
                        DEFAULT_FACADE_BOUNDS.get(try_cls),
                        m_per_px,
                        class_min_fit_inradius_m,
                    )
                actual_count_key = count_key
                if (
                    count_key == 'yolo_template_placed' and
                    best.get('template_source') == 'neighbor_zone'
                ):
                    actual_count_key = 'neighbor_yolo_template_placed'
                file_counts[actual_count_key] = file_counts.get(actual_count_key, 0) + 1
                return True

            def _tile_same_zone_yolo_templates():
                if not yolo_templates_by_zone:
                    return set()
                tiled_labels = set()
                gap_px = int(round(yolo_template_gap_m / max(m_per_px, 1e-6)))
                gap_px = max(0, gap_px)
                for zone_label in sorted(int(label) for label in yolo_templates_by_zone):
                    if zone_label <= 0 or zone_label >= cc_stats.shape[0]:
                        continue
                    template = _dominant_zone_yolo_template(zone_label)
                    if template is None:
                        continue
                    pts = np.asarray(template['points'], dtype=np.float32)
                    if pts.shape != (4, 2):
                        continue
                    metrics = _yolo_template_metrics(template)
                    if metrics is None:
                        continue
                    long_len = float(metrics['long_len'])
                    short_len = float(metrics['short_len'])
                    img_angle = math.radians(float(template['heading']) - 90.0)
                    u_axis = np.asarray(
                        [math.cos(img_angle), math.sin(img_angle)], dtype=np.float32
                    )
                    v_axis = np.asarray([-u_axis[1], u_axis[0]], dtype=np.float32)
                    step_u = max(3.0, long_len + float(gap_px))
                    step_v = max(3.0, short_len + float(gap_px))
                    x0 = int(cc_stats[zone_label, cv2.CC_STAT_LEFT])
                    y0 = int(cc_stats[zone_label, cv2.CC_STAT_TOP])
                    zone_w = int(cc_stats[zone_label, cv2.CC_STAT_WIDTH])
                    zone_h = int(cc_stats[zone_label, cv2.CC_STAT_HEIGHT])
                    if zone_w <= 0 or zone_h <= 0:
                        continue
                    x1 = min(img_w, x0 + zone_w)
                    y1 = min(img_h, y0 + zone_h)
                    anchor = np.asarray(template['center'], dtype=np.float32)
                    bbox_corners = np.asarray(
                        ((x0, y0), (x1 - 1, y0), (x1 - 1, y1 - 1), (x0, y1 - 1)),
                        dtype=np.float32,
                    )
                    rel_corners = bbox_corners - anchor
                    proj_u = rel_corners @ u_axis
                    proj_v = rel_corners @ v_axis
                    iu0 = int(math.floor(float(proj_u.min()) / step_u)) - 1
                    iu1 = int(math.ceil(float(proj_u.max()) / step_u)) + 1
                    iv0 = int(math.floor(float(proj_v.min()) / step_v)) - 1
                    iv1 = int(math.ceil(float(proj_v.max()) / step_v)) + 1

                    attempted = 0
                    placed = 0
                    stop_zone = False
                    for iv in range(iv0, iv1 + 1):
                        if stop_zone:
                            break
                        for iu in range(iu0, iu1 + 1):
                            if attempted >= yolo_template_max_candidates_per_zone:
                                stop_zone = True
                                break
                            center = anchor + u_axis * (iu * step_u) + v_axis * (iv * step_v)
                            jx_t = int(round(float(center[0])))
                            jy_t = int(round(float(center[1])))
                            if not (0 <= jx_t < img_w and 0 <= jy_t < img_h):
                                continue
                            if cc_labels[jy_t, jx_t] != zone_label:
                                continue
                            if fallback_zone[jy_t, jx_t] == 0:
                                continue
                            attempted += 1
                            if _try_place_yolo_template(
                                int(jx_t),
                                int(jy_t),
                                zone_label,
                                'same_zone_yolo_template_placed',
                                template=template,
                                use_center_blockers=False,
                            ):
                                placed += 1
                    if attempted:
                        file_counts['same_zone_yolo_template_candidates'] = (
                            file_counts.get('same_zone_yolo_template_candidates', 0) + attempted
                        )
                        file_counts['same_zone_yolo_heading_votes'] = (
                            file_counts.get('same_zone_yolo_heading_votes', 0) +
                            int(template.get('consensus_heading_votes', 1))
                        )
                        file_counts['same_zone_yolo_shape_votes'] = (
                            file_counts.get('same_zone_yolo_shape_votes', 0) +
                            int(template.get('consensus_shape_votes', 1))
                        )
                    if placed:
                        tiled_labels.add(zone_label)
                return tiled_labels

            def _tile_neighbor_zone_yolo_templates():
                if not neighbor_yolo_templates_by_zone:
                    return set()
                tiled_labels = set()
                gap_px = int(round(yolo_template_gap_m / max(m_per_px, 1e-6)))
                gap_px = max(0, gap_px)
                for zone_label in sorted(int(label) for label in neighbor_yolo_templates_by_zone):
                    if zone_label <= 0 or zone_label >= cc_stats.shape[0]:
                        continue
                    if zone_label in yolo_templates_by_zone:
                        continue
                    template = _dominant_neighbor_yolo_template(zone_label)
                    if template is None:
                        continue
                    pts = np.asarray(template['points'], dtype=np.float32)
                    if pts.shape != (4, 2):
                        continue
                    metrics = _yolo_template_metrics(template)
                    if metrics is None:
                        continue
                    long_len = float(metrics['long_len'])
                    short_len = float(metrics['short_len'])
                    img_angle = math.radians(float(template['heading']) - 90.0)
                    u_axis = np.asarray(
                        [math.cos(img_angle), math.sin(img_angle)], dtype=np.float32
                    )
                    v_axis = np.asarray([-u_axis[1], u_axis[0]], dtype=np.float32)
                    step_u = max(3.0, long_len + float(gap_px))
                    step_v = max(3.0, short_len + float(gap_px))
                    x0 = int(cc_stats[zone_label, cv2.CC_STAT_LEFT])
                    y0 = int(cc_stats[zone_label, cv2.CC_STAT_TOP])
                    zone_w = int(cc_stats[zone_label, cv2.CC_STAT_WIDTH])
                    zone_h = int(cc_stats[zone_label, cv2.CC_STAT_HEIGHT])
                    if zone_w <= 0 or zone_h <= 0:
                        continue
                    x1 = min(img_w, x0 + zone_w)
                    y1 = min(img_h, y0 + zone_h)
                    anchor = np.asarray(template['center'], dtype=np.float32)
                    bbox_corners = np.asarray(
                        ((x0, y0), (x1 - 1, y0), (x1 - 1, y1 - 1), (x0, y1 - 1)),
                        dtype=np.float32,
                    )
                    rel_corners = bbox_corners - anchor
                    proj_u = rel_corners @ u_axis
                    proj_v = rel_corners @ v_axis
                    iu0 = int(math.floor(float(proj_u.min()) / step_u)) - 1
                    iu1 = int(math.ceil(float(proj_u.max()) / step_u)) + 1
                    iv0 = int(math.floor(float(proj_v.min()) / step_v)) - 1
                    iv1 = int(math.ceil(float(proj_v.max()) / step_v)) + 1

                    attempted = 0
                    placed = 0
                    stop_zone = False
                    for iv in range(iv0, iv1 + 1):
                        if stop_zone:
                            break
                        for iu in range(iu0, iu1 + 1):
                            if attempted >= yolo_template_max_candidates_per_zone:
                                stop_zone = True
                                break
                            center = anchor + u_axis * (iu * step_u) + v_axis * (iv * step_v)
                            jx_t = int(round(float(center[0])))
                            jy_t = int(round(float(center[1])))
                            if not (0 <= jx_t < img_w and 0 <= jy_t < img_h):
                                continue
                            if cc_labels[jy_t, jx_t] != zone_label:
                                continue
                            if fallback_zone[jy_t, jx_t] == 0:
                                continue
                            attempted += 1
                            if _try_place_yolo_template(
                                int(jx_t),
                                int(jy_t),
                                zone_label,
                                'neighbor_zone_yolo_template_placed',
                                template=template,
                                use_center_blockers=False,
                            ):
                                placed += 1
                    if attempted:
                        file_counts['neighbor_yolo_template_candidates'] = (
                            file_counts.get('neighbor_yolo_template_candidates', 0) + attempted
                        )
                        file_counts['neighbor_yolo_heading_votes'] = (
                            file_counts.get('neighbor_yolo_heading_votes', 0) +
                            int(template.get('consensus_heading_votes', 1))
                        )
                        file_counts['neighbor_yolo_shape_votes'] = (
                            file_counts.get('neighbor_yolo_shape_votes', 0) +
                            int(template.get('consensus_shape_votes', 1))
                        )
                    if placed:
                        tiled_labels.add(zone_label)
                return tiled_labels

            tiled_yolo_zone_labels = set()
            if allow_inferred_fill:
                tiled_yolo_zone_labels = _tile_same_zone_yolo_templates()
                tiled_yolo_zone_labels.update(_tile_neighbor_zone_yolo_templates())
            if allow_inferred_fill and cand_x.size and tiled_yolo_zone_labels:
                keep_non_templated = ~np.isin(
                    cand_labels,
                    np.fromiter(tiled_yolo_zone_labels, dtype=cand_labels.dtype),
                )
                file_counts['procedural_yolo_zone_skipped'] = int(
                    cand_x.size - np.count_nonzero(keep_non_templated)
                )
                cand_x = cand_x[keep_non_templated]
                cand_y = cand_y[keep_non_templated]
                cand_cls = cand_cls[keep_non_templated]
                cand_labels = cand_labels[keep_non_templated]

            for jx, jy, zone_cls in zip(cand_x, cand_y, cand_cls):
                jx = int(jx)
                jy = int(jy)
                zone_cls = int(zone_cls)
                if static_occ_mask[jy, jx]:
                    file_counts['dynamic_center_blocked'] = (
                        file_counts.get('dynamic_center_blocked', 0) + 1
                    )
                    continue
                if dynamic_center_block_masks[zone_cls][jy, jx]:
                    file_counts['dynamic_clearance_blocked'] = (
                        file_counts.get('dynamic_clearance_blocked', 0) + 1
                    )
                    continue

                zone_label = int(cc_labels[jy, jx])
                zone_tpl   = _zone_yolo_template(jx, jy, zone_label)
                dom_h = _heading_for_candidate(jx, jy, zone_label, zone_template=zone_tpl)
                jitter  = rng.uniform(-HEADING_JITTER_DEG, HEADING_JITTER_DEG)
                heading = (dom_h + jitter) % 360.0

                residential_context = (
                    residential_area_mask is None or
                    bool(residential_area_mask[jy, jx])
                )
                if _try_place_yolo_template(jx, jy, zone_label, 'yolo_template_placed', template=zone_tpl):
                    continue
                for try_cls in (zone_cls,):
                    pool = asset_pools[try_cls]
                    if not pool:
                        continue

                    asset, final_h, footprint_poly, spacing_poly, skipped = _find_fitting_asset(
                        pool, rng, jx, jy, heading, m_per_px,
                        static_occ_mask, building_spacing_mask, fit_scratch,
                        file_counts, mark_pad_m,
                        prefer_small=(try_cls != zone_cls),
                        residential_context=residential_context,
                        footprint_pad_m=footprint_pad_m,
                        static_occ_integral=static_occ_integral,
                        fit_cache_enabled=not strict_fit,
                        retry_context=asset_retry_context.get(try_cls),
                        prefer_largest_fit=(zone_label not in yolo_templates_by_zone),
                    )
                    n_unknown_skipped += skipped
                    if asset is None:
                        continue

                    o_lon, o_lat = px_to_latlon(jx, jy, img_w, img_h,
                                                lat_n, lat_s, lon_w, lon_e)
                    if not (lon <= o_lon < lon + 1 and lat <= o_lat < lat + 1):
                        break
                    pts_this.append((jx, jy, final_h, try_cls))
                    if asset['kind'] == 'object':
                        placed_stock_objects.append((o_lon, o_lat, final_h, asset['path']))
                    else:
                        placed_facades.append((
                            _pixel_ring_to_latlon(footprint_poly, img_w, img_h, lat_n, lat_s, lon_w, lon_e),
                            asset['path'],
                            _randomized_facade_height_m(
                                height_rng, height_priors_by_class, try_cls
                            ),
                        ))
                    placed_viz_polys.append((footprint_poly.copy(), try_cls))
                    _mark_poly(building_spacing_mask, spacing_poly)
                    _mark_dynamic_center_blockers(
                        dynamic_center_block_masks,
                        jx,
                        jy,
                        final_h,
                        asset.get('mark_bounds_m'),
                        m_per_px,
                        class_min_fit_inradius_m,
                    )
                    break

            if run_legacy_gap_fill:
                leftover_mask = (
                    (road_divided_zone != 0) &
                    (static_occ_mask == 0) &
                    (building_spacing_mask == 0)
                ).astype(np.uint8)
                gap_x, gap_y, gap_cls, _, n_gap_components, n_gap_dropped = (
                    _leftover_gap_candidates(
                        leftover_mask,
                        zone_class,
                        max_gap_candidates_per_dds,
                        rng,
                        max_gap_candidates_per_component,
                    )
                )
                file_counts['gap_components'] = int(n_gap_components)
                file_counts['gap_candidates'] = int(gap_x.size + n_gap_dropped)
                if n_gap_dropped:
                    file_counts['gap_candidate_cap'] = int(n_gap_dropped)
                if detail_timing:
                    print(
                        f"    [Bld stage] {fname} gap fill start  "
                        f"gap_candidates={int(gap_x.size)}  "
                        f"components={int(n_gap_components)}",
                        flush=True,
                    )

                for jx, jy, zone_cls in zip(gap_x, gap_y, gap_cls):
                    jx = int(jx)
                    jy = int(jy)
                    zone_cls = int(zone_cls)
                    if static_occ_mask[jy, jx]:
                        file_counts['gap_static_blocked'] = (
                            file_counts.get('gap_static_blocked', 0) + 1
                        )
                        continue

                    zone_label = int(cc_labels[jy, jx])
                    zone_tpl   = _zone_yolo_template(jx, jy, zone_label)
                    dom_h = _heading_for_candidate(jx, jy, zone_label, zone_template=zone_tpl)
                    jitter_h = rng.uniform(-HEADING_JITTER_DEG, HEADING_JITTER_DEG)
                    heading = (dom_h + jitter_h) % 360.0

                    placed_gap = False
                    dynamic_blocked = False
                    residential_context = (
                        residential_area_mask is None or
                        bool(residential_area_mask[jy, jx])
                    )
                    if _try_place_yolo_template(jx, jy, zone_label, 'gap_yolo_template_placed', template=zone_tpl):
                        placed_gap = True
                        continue
                    for try_cls in _gap_fill_class_sequence(zone_cls):
                        if dynamic_center_block_masks[try_cls][jy, jx]:
                            dynamic_blocked = True
                            continue

                        pool = asset_pools[try_cls]
                        if not pool:
                            continue

                        asset, final_h, footprint_poly, spacing_poly, skipped = _find_fitting_asset(
                            pool, rng, jx, jy, heading, m_per_px,
                            static_occ_mask, building_spacing_mask, fit_scratch,
                            file_counts, mark_pad_m, prefer_small=True,
                            residential_context=residential_context,
                            footprint_pad_m=footprint_pad_m,
                            static_occ_integral=static_occ_integral,
                            fit_cache_enabled=not strict_fit,
                            retry_context=asset_retry_context.get(try_cls),
                            prefer_largest_fit=(zone_label not in yolo_templates_by_zone),
                        )
                        n_unknown_skipped += skipped
                        if asset is None:
                            continue

                        o_lon, o_lat = px_to_latlon(jx, jy, img_w, img_h,
                                                    lat_n, lat_s, lon_w, lon_e)
                        if not (lon <= o_lon < lon + 1 and lat <= o_lat < lat + 1):
                            break
                        pts_this.append((jx, jy, final_h, try_cls))
                        if asset['kind'] == 'object':
                            placed_stock_objects.append((o_lon, o_lat, final_h, asset['path']))
                        else:
                            placed_facades.append((
                                _pixel_ring_to_latlon(
                                    footprint_poly, img_w, img_h,
                                    lat_n, lat_s, lon_w, lon_e,
                                ),
                                asset['path'],
                                _randomized_facade_height_m(
                                    height_rng, height_priors_by_class, try_cls
                                ),
                            ))
                        placed_viz_polys.append((footprint_poly.copy(), try_cls))
                        _mark_poly(building_spacing_mask, spacing_poly)
                        _mark_dynamic_center_blockers(
                            dynamic_center_block_masks,
                            jx,
                            jy,
                            final_h,
                            asset.get('mark_bounds_m'),
                            m_per_px,
                            class_min_fit_inradius_m,
                        )
                        file_counts['gap_placed'] = (
                            file_counts.get('gap_placed', 0) + 1
                        )
                        placed_gap = True
                        break

                    if not placed_gap and dynamic_blocked:
                        file_counts['gap_dynamic_blocked'] = (
                            file_counts.get('gap_dynamic_blocked', 0) + 1
                        )

            fit_elapsed = _record_elapsed(timings, file_timings, 'fit_loop', _t)
            timings['placement'] += fit_elapsed
            file_counts['placed'] = len(pts_this)

            col = (til_x_left - x_min) // x_step
            row = (til_y_top  - y_min) // y_step
            bld_pct = 100 * np.sum(bld_raw) / (img_w * img_h)
            # Placed-footprint coverage: fraction of the tile actually covered by
            # placed building footprints (objects + facades). Unlike bld_pct
            # (segmentation coverage) this reflects packing density, so it is the
            # metric to watch when tuning overlap removal vs. coverage.
            if placed_viz_polys:
                _placed_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                cv2.fillPoly(
                    _placed_mask,
                    [np.int32(np.asarray(_p)) for _p, _cls in placed_viz_polys],
                    1,
                )
                placed_pct = 100.0 * float(cv2.countNonZero(_placed_mask)) / float(img_w * img_h)
            else:
                placed_pct = 0.0
            file_counts['placed_footprint_pct'] = round(placed_pct, 2)
            class_counts = {
                cls: sum(1 for p in pts_this if p[3] == cls)
                for cls in BLD_PLACEMENT_CLASSES
            }
            spacing_label = _format_class_spacing(spacing_px_by_class, m_per_px)
            print(
                f"  [{fi:3d}/{n_files}] {fname}  "
                f"{_describe_placement_summary(class_counts, bld_pct, n_osm_cells, grid_n, spacing_label, file_counts)}"
                f"  placed%={placed_pct:.1f}"
                f"  small-house areas={residential_area_source}"
            )

            if not disable_cache and not ignore_placement_cache:
                try:
                    _t = time.perf_counter()
                    with open(_bld_cache_file, 'wb') as _f:
                        _pickle.dump({
                            'params': _effective_bld_params,
                            'source': 'direct_yolo',
                            'placements': {
                                'objects': placed_stock_objects[_start_object_idx:],
                                'facades': placed_facades[_start_facade_idx:],
                            },
                        }, _f)
                    _record_elapsed(timings, file_timings, 'cache_save', _t)
                except Exception:
                    pass

            if make_viz and composite is not None:
                _t_viz = time.perf_counter()
                if img is None:
                    _img_t = time.perf_counter()
                    img = _load_source_image(fname, _source_mode, _orthophoto_dir)
                    if img is None:
                        continue
                    _record_elapsed(timings, file_timings, 'dds_load', _img_t)
                scale = TILE_VIZ / img_w
                panel = np.array(Image.fromarray(img).resize((TILE_VIZ, TILE_VIZ), Image.LANCZOS))
                for cls, colour in ZONE_COLOURS.items():
                    mask = np.array(
                        Image.fromarray((zone_class == cls).astype(np.uint8) * 255)
                        .resize((TILE_VIZ, TILE_VIZ), Image.NEAREST)
                    ) > 127
                    panel[mask] = (panel[mask] * 0.45 + colour * 0.55).astype(np.uint8)
                def _blend_viz_mask(src_mask, colour, alpha):
                    if src_mask is None or not np.any(src_mask):
                        return
                    viz_mask = np.array(
                        Image.fromarray((src_mask != 0).astype(np.uint8) * 255)
                        .resize((TILE_VIZ, TILE_VIZ), Image.NEAREST)
                    ) > 127
                    panel[viz_mask] = (
                        panel[viz_mask] * (1.0 - alpha) +
                        np.asarray(colour, dtype=np.float32) * alpha
                    ).astype(np.uint8)

                _blend_viz_mask(sfr_road_dilated, (255, 150, 0), 0.70)
                _blend_viz_mask(mesh_water_mask, (0, 60, 255), 0.65)
                _blend_viz_mask(road_mask, (255, 0, 0), 0.75)
                _blend_viz_mask(rail_mask, (255, 0, 255), 0.75)
                _blend_viz_mask(poly_mask, (120, 0, 255), 0.75)
                _blend_viz_mask(existing_bld_mask, (0, 0, 0), 0.70)
                _blend_viz_mask(sh_bld_mask, (255, 255, 255), 0.65)
                _blend_viz_mask(custom_bld_mask, (0, 255, 255), 0.65)
                pil = Image.fromarray(panel); draw = ImageDraw.Draw(pil, "RGBA")
                dot_colours = {
                    BLD_CLASS_TINY_RESIDENTIAL: (80, 220, 80),
                    BLD_CLASS_SMALL_RESIDENTIAL: (150, 230, 60),
                    BLD_CLASS_COMPACT_RESIDENTIAL: (245, 220, 70),
                    BLD_CLASS_MEDIUM: (255, 165, 50),
                    BLD_CLASS_SMALL_APARTMENT: (255, 90, 70),
                    BLD_CLASS_APARTMENT_BLOCK: (80, 200, 255),
                    BLD_CLASS_LARGE: (70, 130, 255),
                    BLD_CLASS_EXTRA_LARGE: (135, 90, 255),
                }
                footprint_fill = (255, 255, 255)
                yolo_line_w = max(1, int(round(TILE_VIZ / 2048)))
                for poly, placed in yolo_viz_polys:
                    pts = [
                        (int(round(float(px2) * scale)), int(round(float(py2) * scale)))
                        for px2, py2 in poly
                    ]
                    if len(pts) >= 3:
                        fill_alpha = 34 if placed else 18
                        draw.polygon(pts, fill=(40, 180, 255, fill_alpha), outline=(40, 180, 255, 230))
                        draw.line(pts + [pts[0]], fill=(255, 245, 80, 235), width=yolo_line_w)
                for poly, cls2 in placed_viz_polys:
                    pts = [
                        (int(round(float(px2) * scale)), int(round(float(py2) * scale)))
                        for px2, py2 in poly
                    ]
                    if len(pts) >= 3:
                        draw.polygon(pts, outline=dot_colours.get(cls2, (0, 220, 0)))
                        cx = sum(p[0] for p in pts) / len(pts)
                        cy = sum(p[1] for p in pts) / len(pts)
                        inner = [
                            (
                                int(round(cx + (p[0] - cx) * 0.88)),
                                int(round(cy + (p[1] - cy) * 0.88)),
                            )
                            for p in pts
                        ]
                        draw.polygon(inner, outline=footprint_fill)
                for px2, py2, _, cls2 in pts_this:
                    sx, sy = int(px2*scale), int(py2*scale)
                    draw.ellipse([sx-1,sy-1,sx+1,sy+1], fill=dot_colours.get(cls2, (0,220,0)))
                for poly, placed in yolo_viz_polys:
                    pts = [
                        (int(round(float(px2) * scale)), int(round(float(py2) * scale)))
                        for px2, py2 in poly
                    ]
                    if len(pts) >= 3:
                        draw.line(
                            pts + [pts[0]],
                            fill=(255, 245, 80, 255 if placed else 190),
                            width=yolo_line_w,
                        )
                composite[row*TILE_VIZ:(row+1)*TILE_VIZ, col*TILE_VIZ:(col+1)*TILE_VIZ] = np.array(pil)

                if footprint_composite is not None:
                    fp_base = Image.fromarray(
                        np.array(
                            Image.fromarray(img).resize((TILE_VIZ, TILE_VIZ), Image.LANCZOS)
                        )
                    ).convert('RGBA')
                    fp_layer = Image.new('RGBA', (TILE_VIZ, TILE_VIZ), (0, 0, 0, 0))
                    fp_draw = ImageDraw.Draw(fp_layer)
                    yolo_line_w = max(1, int(round(TILE_VIZ / 2048)))
                    for poly, placed in yolo_viz_polys:
                        if not placed:
                            continue
                        pts = [
                            (
                                int(round(float(px2) * scale)),
                                int(round(float(py2) * scale)),
                            )
                            for px2, py2 in poly
                        ]
                        if len(pts) >= 3:
                            alpha = 44 if placed else 24
                            fp_draw.polygon(pts, fill=(40, 180, 255, alpha), outline=(40, 180, 255, 230))
                            fp_draw.line(pts + [pts[0]], fill=(255, 245, 80, 240), width=yolo_line_w)
                    for poly, cls2 in placed_viz_polys:
                        pts = [
                            (
                                int(round(float(px2) * scale)),
                                int(round(float(py2) * scale)),
                            )
                            for px2, py2 in poly
                        ]
                        if len(pts) >= 3:
                            outline = dot_colours.get(cls2, (0, 220, 0))
                            fp_draw.polygon(pts, fill=(245, 242, 232, 170), outline=outline + (255,))
                    for poly, placed in yolo_viz_polys:
                        if not placed:
                            continue
                        pts = [
                            (
                                int(round(float(px2) * scale)),
                                int(round(float(py2) * scale)),
                            )
                            for px2, py2 in poly
                        ]
                        if len(pts) >= 3:
                            fp_draw.line(
                                pts + [pts[0]],
                                fill=(255, 245, 80, 255 if placed else 190),
                                width=yolo_line_w,
                            )
                    fp_img = Image.alpha_composite(fp_base, fp_layer).convert('RGB')
                    footprint_composite[
                        row*TILE_VIZ:(row+1)*TILE_VIZ,
                        col*TILE_VIZ:(col+1)*TILE_VIZ,
                    ] = np.array(fp_img)
                _record_elapsed(timings, file_timings, 'viz', _t_viz)
        except BaseException:
            if prefetch_executor is not None:
                prefetch_executor.shutdown(wait=True, cancel_futures=True)
                prefetch_executor = None
            raise
        finally:
            file_elapsed = time.perf_counter() - file_t0
            if detail_timing or (slow_timing_s > 0 and file_elapsed >= slow_timing_s):
                _print_dds_timing(fname, file_timings, file_elapsed, file_counts)
            img = veg_map = mesh_water_mask = bld_raw = bld_zone = sfr_road_dilated = None
            road_mask = rail_mask = poly_mask = existing_bld_mask = sh_bld_mask = custom_bld_mask = None
            static_occ_mask = static_occ_integral = building_spacing_mask = None
            placed_yolo_mask = fit_scratch = None
            road_divided_zone = fallback_zone = zone_class = roof_evidence = None
            yolo_detections = yolo_guidance = yolo_viz_polys = placed_viz_polys = None
            cc_labels = cc_stats = cc_centroids = valid_labels = None
            local_separator_roads = local_rails = local_excl_polys = None
            local_existing_bld_polys = local_sh_bld_objects = local_custom_bld_objects = None
            local_residential_roads = local_residential_polys = local_heading_segments = None
            residential_area_mask = dynamic_center_block_masks = None
            if (
                yolo_cuda_cleanup_every > 0 and
                fi % yolo_cuda_cleanup_every == 0 and
                torch.cuda.is_available()
            ):
                try:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            if disable_cache:
                _remove_cache_files(_dds_cache_files)

    if prefetch_executor is not None:
        prefetch_executor.shutdown(wait=True, cancel_futures=True)

    # Tile-wide cross-texture / cross-pass object dedup. Per-DDS placement only
    # avoids overlaps within a single texture; buildings on texture seams (far
    # more numerous at high ZL) and stock-vs-trained coincidences slip through.
    if not yolo_no_overlap_removal and placed_stock_objects:
        _t = time.perf_counter()
        _before = len(placed_stock_objects)
        placed_stock_objects, _seam_dropped = _dedupe_overlapping_placements(
            placed_stock_objects, lat, lon
        )
        timings['global_dedup'] = timings.get('global_dedup', 0.0) + (
            time.perf_counter() - _t
        )
        if _seam_dropped:
            print(
                f"[SFR Bld] Cross-texture object dedup: dropped {_seam_dropped:,} "
                f"overlapping objects ({_before:,} -> {len(placed_stock_objects):,})"
            )

    total_time = time.time() - t_start
    total_placements = (
        len(placed_stock_objects) + len(placed_facades) + len(placed_draped)
    )
    # NOTE: legacy direct-YOLO total-count assertion removed — the trained-YOLO
    # facade pass and the stock-YOLO pre-step now both contribute placements,
    # and the building-overlay tracks only the trained-YOLO subtotals
    # (direct_yolo_facade_placements). Stock-YOLO counts live in
    # file_counts['stock_yolo_detections'] per DDS file.
    print(
        f"\nTotal: {total_placements:,} placements  "
        f"({len(placed_stock_objects):,} stock-yolo objects, "
        f"{len(placed_facades):,} facades, "
        f"{len(placed_draped):,} draped polys)  "
        f"trained_yolo_facades={direct_yolo_facade_placements:,}  "
        f"({total_time/60:.1f}min)"
    )

    if debug_image_only:
        if make_viz and composite is not None:
            os.makedirs(os.path.dirname(os.path.abspath(out_dsf)), exist_ok=True)
            viz_path = out_dsf.replace('.dsf', '_overview.png')
            Image.fromarray(composite).save(viz_path)
            print(f"Overview  → {viz_path}  ({n_cols*TILE_VIZ}×{n_rows*TILE_VIZ}px)")
            if footprint_composite is not None:
                footprint_viz_path = out_dsf.replace('.dsf', '_footprints.png')
                Image.fromarray(footprint_composite).save(footprint_viz_path)
                print(
                    f"Footprints → {footprint_viz_path}  "
                    f"({n_cols*TILE_VIZ}×{n_rows*TILE_VIZ}px)"
                )
        print("[SFR Bld] Debug image-only mode: skipped DSF text write and compile.")
        return total_placements

    # ── Write DSF text ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(out_dsf)), exist_ok=True)
    txt_path = out_dsf.replace('.dsf', '_bld.txt')
    _assert_no_unexported_simheaven_object_placements(
        placed_stock_objects,
        exported_simheaven_object_paths,
    )

    obj_paths = sorted(set(obj_path for _, _, _, obj_path in placed_stock_objects))
    obj_idx = {path: i for i, path in enumerate(obj_paths)}
    # POLYGON_DEFs are shared between facade (.fac, extruded) and draped (.pol, ground)
    # entries — both use BEGIN_POLYGON, distinguished only by param semantics at sim load.
    facade_paths_set = set(facade_path for _, facade_path, _ in placed_facades)
    draped_paths_set = set(pol_path for _, pol_path in placed_draped)
    poly_paths = sorted(facade_paths_set | draped_paths_set)
    poly_idx = {path: i for i, path in enumerate(poly_paths)}
    _t = time.perf_counter()
    with open(txt_path, 'w') as f:
        f.write("PROPERTY sim/planet earth\n")
        f.write("PROPERTY sim/overlay 1\n")
        f.write(f"PROPERTY sim/west  {int(lon)}\n")
        f.write(f"PROPERTY sim/east  {int(lon)+1}\n")
        f.write(f"PROPERTY sim/south {int(lat)}\n")
        f.write(f"PROPERTY sim/north {int(lat)+1}\n")
        f.write("\n")
        for p in poly_paths:
            f.write(f"POLYGON_DEF {p}\n")
        for p in obj_paths:
            f.write(f"OBJECT_DEF {p}\n")
        f.write("\n")
        for lonlat_ring, facade_path, height_m in placed_facades:
            idx = poly_idx[facade_path]
            f.write(f"BEGIN_POLYGON {idx} {int(round(height_m))} 2\n")
            _write_polygon_winding(f, lonlat_ring)
            f.write("END_POLYGON\n")
        for lonlat_ring, pol_path in placed_draped:
            idx = poly_idx[pol_path]
            # Draped polygons (.pol) use param=0 — sim treats geometry as ground-pinned.
            f.write(f"BEGIN_POLYGON {idx} 0 2\n")
            _write_polygon_winding(f, lonlat_ring)
            f.write("END_POLYGON\n")
        for o_lon, o_lat, heading, obj_path in placed_stock_objects:
            idx = obj_idx[obj_path]
            f.write(f"OBJECT {idx} {o_lon:.7f} {o_lat:.7f} {heading:.1f}\n")
    timings['dsf_text'] += time.perf_counter() - _t

    print(f"DSF text → {txt_path}")

    # Compile
    dsf_dir = os.path.dirname(os.path.abspath(out_dsf))
    _t = time.perf_counter()
    ok = SEGFORMER.compile_dsf(txt_path, out_dsf)
    timings['dsf_compile'] += time.perf_counter() - _t
    if ok:
        print(f"DSF compiled → {out_dsf}")
        print(f"[debug] DSF text preserved at {txt_path}")
        # try: os.remove(txt_path)
        # except: pass
    else:
        print(f"DSFTool failed — text file kept at {txt_path}")

    # ── Save overview image ───────────────────────────────────────────────────
    if make_viz and composite is not None:
        viz_path = out_dsf.replace('.dsf', '_overview.png')
        Image.fromarray(composite).save(viz_path)
        print(f"Overview  → {viz_path}  ({n_cols*TILE_VIZ}×{n_rows*TILE_VIZ}px)")
        if footprint_composite is not None:
            footprint_viz_path = out_dsf.replace('.dsf', '_footprints.png')
            Image.fromarray(footprint_composite).save(footprint_viz_path)
            print(
                f"Footprints → {footprint_viz_path}  "
                f"({n_cols*TILE_VIZ}×{n_rows*TILE_VIZ}px)"
            )

    print(
        "[Bld timing] "
        f"simHeaven={timings['simheaven_parse']:.1f}s  "
        f"customScenery={timings['custom_scenery_parse']:.1f}s  "
        f"cache_load={timings['cache_load']:.1f}s  "
        f"dds_load={timings['dds_load']:.1f}s  "
        f"segformer={timings['segformer_inference']:.1f}s  "
        f"trained_yolo={timings['trained_yolo_inference']:.1f}s  "
        f"stock_yolo={timings['stock_yolo_inference']:.1f}s  "
        f"road_cache={timings['road_cache']:.1f}s  "
        f"road_raster={timings['road_raster']:.1f}s  "
        f"mesh_water={timings['mesh_water']:.1f}s  "
        f"existing_bld_excl={timings['existing_bld_excl']:.1f}s  "
        f"heading={timings['heading_grid']:.1f}s  "
        f"placement={timings['placement']:.1f}s  "
        f"dsf_text={timings['dsf_text']:.1f}s  "
        f"dsf_compile={timings['dsf_compile']:.1f}s  "
        f"road_cache_hits={n_road_cache_hits}/{n_files}  "
        f"unknown_skipped={n_unknown_skipped}"
    )
    if detail_timing:
        print(
            "[Bld timing detail] "
            f"zone={timings['zone_cleanup']:.1f}s  "
            f"lookup={timings['lookup']:.1f}s  "
            f"mask={timings['mask_apply']:.1f}s  "
            f"cc={timings['connected_components']:.1f}s  "
            f"candidates={timings['candidate_grid']:.1f}s  "
            f"place={timings['fit_loop']:.1f}s  "
            f"cache_save={timings['cache_save']:.1f}s  "
            f"viz={timings['viz']:.1f}s"
        )

    return total_placements


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    """Run the building overlay CLI."""
    args = parse_args()
    cache_dir = args.cache_dir
    o4xp_root = os.path.dirname(os.path.dirname(os.path.dirname(args.tex_dir)))
    lat_i = int(args.lat);  lon_i = int(args.lon)
    lat_g = int(math.floor(args.lat / 10)) * 10
    lon_g = int(math.floor(args.lon / 10)) * 10
    lat_s = f"{'+' if lat_i >= 0 else '-'}{abs(lat_i):02d}"
    lon_s = f"{'+' if lon_i >= 0 else '-'}{abs(lon_i):03d}"
    lat_gs = f"{'+' if lat_g >= 0 else '-'}{abs(lat_g):02d}"
    lon_gs = f"{'+' if lon_g >= 0 else '-'}{abs(lon_g):03d}"

    if cache_dir is None:
        cache_dir = os.path.join(
            o4xp_root, 'SFR_cache', f'{lat_gs}{lon_gs}', f'{lat_s}{lon_s}'
        )
    os.makedirs(cache_dir, exist_ok=True)

    out_dsf = args.out_dsf
    if out_dsf is None:
        # Auto-derive: <o4xp_root>/yOrtho4XP_Bld_Overlays/Earth nav data/<lat_g><lon_g>/<lat><lon>.dsf
        out_dsf = os.path.join(
            o4xp_root,
            'yOrtho4XP_Bld_Overlays',
            'Earth nav data',
            f'{lat_gs}{lon_gs}',
            f'{lat_s}{lon_s}.dsf',
        )
        print(f"Output DSF: {out_dsf}")

    run(
        tex_dir    = args.tex_dir,
        lat        = args.lat,
        lon        = args.lon,
        out_dsf    = out_dsf,
        spacing_m  = args.spacing,
        close_k    = args.close,
        open_k     = args.open_k,
        min_zone_m2= args.min_zone_m2,
        make_viz   = not args.no_viz,
        debug_image_only = args.debug_image_only,
        cache_dir  = cache_dir,
        grid_n          = args.grid_n,
        osm_roads_path  = args.osm_roads,
        custom_scenery_dir = args.custom_scenery_dir,
        avoid_custom_scenery = not args.no_custom_scenery_avoidance,
        allow_road_overlap = args.allow_road_overlap,
        yolo_enabled = not args.no_yolo,
        yolo_checkpoint = args.yolo_checkpoint,
        yolo_conf = args.yolo_conf,
        yolo_iou = args.yolo_iou,
        yolo_stride = args.yolo_stride,
        yolo_max_det = args.yolo_max_det,
        yolo_suppress_coverage = args.yolo_suppress_coverage,
        yolo_suppress_min_overlap_m2 = args.yolo_suppress_min_overlap_m2,
        yolo_keep_mode = args.yolo_keep_mode,
        yolo_keep_min_new_frac = args.yolo_keep_min_new_frac,
        yolo_freearea_downsize = args.yolo_freearea_downsize,
        yolo_facade_clip = args.yolo_facade_clip,
    )


if __name__ == '__main__':
    main()

