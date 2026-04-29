"""
O4_SFR_Building_Overlay.py — SegFormer-assisted building overlay generation.

Usage:
    python src/scripts/generate_bld_overlay.py <tex_dir> <lat> <lon> <out_dsf> [options]

Options:
    --spacing   METRES   Object spacing in metres (default 20)
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
import sys, os, argparse, warnings, time, math, re, urllib.request, urllib.parse, hashlib, fnmatch
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
from O4_SFR_DSF_Utils import (
    ensure_cached_dsf_text,
    find_simheaven_building_dsfs,
    find_simheaven_network_dsfs,
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
        ("infer", "inference"),
        ("zone", "zone_cleanup"),
        ("lookup", "lookup"),
        ("road", "road_raster"),
        ("road_cache", "road_cache"),
        ("heading", "heading_grid"),
        ("excl", "existing_bld_excl"),
        ("mask", "mask_apply"),
        ("cc", "connected_components"),
        ("candidates", "candidate_grid"),
        ("fit", "fit_loop"),
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
                ("placed", "placed"),
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
        return cand_x, cand_y, cand_cls, 0

    labels, inverse, counts = np.unique(cand_labels, return_inverse=True, return_counts=True)
    n_labels = labels.size
    if n_labels == 0:
        return cand_x[:0], cand_y[:0], cand_cls[:0], n_candidates

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
        return cand_x[:0], cand_y[:0], cand_cls[:0], n_candidates

    keep_idx = np.sort(np.concatenate(selected))
    return cand_x[keep_idx], cand_y[keep_idx], cand_cls[keep_idx], n_candidates - keep_idx.size


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
    n_before_cap = int(gap_x.size)
    if max_candidates and gap_x.size > max_candidates:
        gap_x, gap_y, gap_cls, n_dropped = _limit_candidates_by_component(
            gap_x, gap_y, gap_cls, gap_labels, max_candidates, rng
        )
        kept = max(0, n_before_cap - int(n_dropped))
        if kept != gap_x.size:
            gap_labels = gap_labels[:0]
        else:
            gap_labels = gap_labels[:gap_x.size]
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
        default=20.0,
        help='Target edge gap in metres between generated building footprints',
    )
    ap.add_argument('--close',     type=int,   default=15)
    ap.add_argument('--open-k',    type=int,   default=5,  dest='open_k')
    ap.add_argument('--min-zone', '--min-zone-m2', type=float, default=200.0, dest='min_zone_m2')
    ap.add_argument('--no-viz',    action='store_true')
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--grid-n',    type=int,   default=HEADING_GRID_N, dest='grid_n')
    ap.add_argument('--osm-roads', default=None,
                    help='Path to *_big_roads.osm.bz2 (auto-discovered if omitted)')
    ap.add_argument('--custom-scenery-dir', default=None,
                    help='Configured X-Plane root or Custom Scenery directory used to locate building libraries.')
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
    """Parse an Ortho4XP *_big_roads.osm.bz2 file.

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
                        existing_bld_poly_sig, sh_bld_sig):
    return {
        'version': 4,
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
                         poly_mask, existing_bld_mask, sh_bld_mask):
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
                                   band_px=8, img=None, bin_deg=5.0):
    """Estimate component headings from distinct road contacts on its outline.

    The contact search follows the actual component boundary rather than an
    axis-aligned bounding box, so rotated or irregular blocks are handled the
    same way as rectangular blocks. Each connected contact patch contributes
    one vote, so short local streets framing a block are not overwhelmed by a
    longer nearby segment.
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
    ring_labels = np.zeros(labels.shape, dtype=np.int32)
    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (band_px * 2 + 1, band_px * 2 + 1)
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
        dilated = cv2.dilate(local_component, ring_kernel)
        eroded = cv2.erode(local_component, erode_kernel)
        boundary_ring = (dilated != 0) & (eroded == 0)
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


def _fill_heading_grid_nearest(hgrid, segments, lat_n, lat_s, lon_w, lon_e,
                               img_h, img_w, grid_n):
    """Fill NaN heading cells from the nearest precomputed segment midpoint."""
    if not segments or not np.any(np.isnan(hgrid)):
        return hgrid

    px = (segments['lon'] - lon_w) / (lon_e - lon_w) * img_w
    py = (lat_n - segments['lat']) / (lat_n - lat_s) * img_h
    if len(px) == 0:
        return hgrid

    cell_h = img_h / grid_n
    cell_w = img_w / grid_n
    missing = np.argwhere(np.isnan(hgrid))
    for gi, gj in missing:
        cy = (gi + 0.5) * cell_h
        cx = (gj + 0.5) * cell_w
        d2 = (py - cy) ** 2 + (px - cx) ** 2
        hgrid[gi, gj] = float(segments['heading'][int(np.argmin(d2))])
    return hgrid


def _image_heading_grid(img, img_h, img_w, grid_n):
    """Estimate per-cell heading from imagery edges when road vectors are absent."""
    hgrid = np.full((grid_n, grid_n), np.nan)
    if img is None:
        return hgrid
    bin_deg = 5.0
    cell_h = img_h / grid_n
    cell_w = img_w / grid_n
    for gi in range(grid_n):
        y0 = int(gi * cell_h)
        y1 = min(img_h, int((gi + 1) * cell_h))
        for gj in range(grid_n):
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


def _rasterize_polygons(polys, lat_n, lat_s, lon_w, lon_e, img_h, img_w):
    """Rasterize filled closed-way polygons onto a uint8 mask."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    def ll_to_px(lat, lon):
        x = int((lon - lon_w) / (lon_e - lon_w) * img_w)
        y = int((lat_n - lat) / (lat_n - lat_s) * img_h)
        return x, y

    for poly in polys:
        pts_px = np.array([ll_to_px(lat, lon) for lat, lon in poly], dtype=np.int32)
        # Only draw if at least 1 vertex falls inside this DDS tile
        if (pts_px[:, 0].max() >= 0 and pts_px[:, 0].min() < img_w and
                pts_px[:, 1].max() >= 0 and pts_px[:, 1].min() < img_h):
            cv2.fillPoly(mask, [pts_px], 1)
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


# ── SFD object pools by footprint size ────────────────────────────────────────
BLD_CLASS_COMPACT_RESIDENTIAL = 1
BLD_CLASS_MEDIUM = 2
BLD_CLASS_SMALL_APARTMENT = 3
BLD_CLASS_APARTMENT_BLOCK = 4
BLD_CLASS_LARGE = 5
# Legacy internal name retained for compatibility with any existing callers.
BLD_CLASS_STANDARD_RESIDENTIAL = BLD_CLASS_MEDIUM
BLD_PLACEMENT_CLASSES = (
    BLD_CLASS_COMPACT_RESIDENTIAL,
    BLD_CLASS_MEDIUM,
    BLD_CLASS_SMALL_APARTMENT,
    BLD_CLASS_APARTMENT_BLOCK,
    BLD_CLASS_LARGE,
)
BLD_CLASS_LABELS = {
    BLD_CLASS_COMPACT_RESIDENTIAL: "compact residential",
    BLD_CLASS_MEDIUM: "medium footprint",
    BLD_CLASS_SMALL_APARTMENT: "small apartment",
    BLD_CLASS_APARTMENT_BLOCK: "apartment block",
    BLD_CLASS_LARGE: "large footprint",
}

# Zone size thresholds (pixels²) after morphological cleanup at ZL16 native res.
# These classify building-zone clusters into progressively larger footprint
# candidates; the per-asset footprint fit remains the final placement gate.
ZONE_COMPACT_PX = 1_500
ZONE_MEDIUM_PX = 3_000
ZONE_SMALL_APARTMENT_PX = 10_000
ZONE_APARTMENT_BLOCK_PX = 30_000

# SFD footprint thresholds in square metres. Height is intentionally ignored:
# tall objects are acceptable when the footprint is modest.
SFD_COMPACT_RESIDENTIAL_MAX_M2 = 180.0
SFD_MEDIUM_MAX_M2 = 325.0
SFD_SMALL_APARTMENT_MAX_M2 = 500.0
SFD_APARTMENT_BLOCK_MAX_M2 = 650.0
SFD_APARTMENT_BLOCK_MAX_SIDE_M = 50.0

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
    BLD_CLASS_COMPACT_RESIDENTIAL: 14.0,
    BLD_CLASS_MEDIUM: 20.0,
    BLD_CLASS_SMALL_APARTMENT: 28.0,
    BLD_CLASS_APARTMENT_BLOCK: 36.0,
    BLD_CLASS_LARGE: 55.0,
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
BLD_PLACEMENT_CACHE_VERSION = 37
BLD_MAX_CANDIDATES_PER_DDS = 180_000  # 0 = exhaustive search; override with O4_SFR_BLD_MAX_CANDIDATES.

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
    BLD_CLASS_COMPACT_RESIDENTIAL: (-7.0, 7.0, -7.0, 7.0),
    BLD_CLASS_MEDIUM: (-10.0, 10.0, -8.0, 8.0),
    BLD_CLASS_SMALL_APARTMENT: (-14.0, 14.0, -9.0, 9.0),
    BLD_CLASS_APARTMENT_BLOCK: (-18.0, 18.0, -12.0, 12.0),
    BLD_CLASS_LARGE: (-24.0, 24.0, -24.0, 24.0),
}
DEFAULT_FACADE_PATHS = {
    BLD_CLASS_COMPACT_RESIDENTIAL: SEGFORMER._FAC_DEFS["medium"],
    BLD_CLASS_MEDIUM: SEGFORMER._FAC_DEFS["medium"],
    BLD_CLASS_SMALL_APARTMENT: SEGFORMER._FAC_DEFS["medium"],
    BLD_CLASS_APARTMENT_BLOCK: SEGFORMER._FAC_DEFS["medium"],
    BLD_CLASS_LARGE: SEGFORMER._FAC_DEFS["large"],
}
DEFAULT_FACADE_VARIANTS_BY_CLASS = {
    BLD_CLASS_COMPACT_RESIDENTIAL: (
        "lib/buildings/facades/generic/low_modern_01.fac",
        "lib/buildings/facades/commercial/low_commercial_01.fac",
        "lib/buildings/facades/commercial/low_commercial_02.fac",
        "lib/buildings/facades/commercial/low_commercial_03.fac",
    ),
    BLD_CLASS_MEDIUM: (
        "lib/buildings/facades/generic/mid_classic_01.fac",
        "lib/buildings/facades/generic/mid_classic_02.fac",
        "lib/buildings/facades/generic/mid_modern_01.fac",
        "lib/buildings/facades/generic/mid_modern_02.fac",
        "lib/buildings/facades/generic/mid_modern_03.fac",
        "lib/buildings/facades/generic/mid_modern_04.fac",
        "lib/buildings/facades/generic/mid_modern_05.fac",
    ),
    BLD_CLASS_SMALL_APARTMENT: (
        "lib/buildings/facades/generic/mid_classic_01.fac",
        "lib/buildings/facades/generic/mid_classic_02.fac",
        "lib/buildings/facades/generic/high_classic_01.fac",
        "lib/buildings/facades/generic/high_glass_01.fac",
        "lib/buildings/facades/generic/high_modern_02.fac",
    ),
    BLD_CLASS_APARTMENT_BLOCK: (
        "lib/buildings/facades/generic/high_classic_01.fac",
        "lib/buildings/facades/generic/high_glass_01.fac",
        "lib/buildings/facades/generic/high_modern_01.fac",
        "lib/buildings/facades/generic/high_modern_02.fac",
        "lib/buildings/facades/generic/high_universal_01.fac",
    ),
    BLD_CLASS_LARGE: (
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
}
DEFAULT_FACADE_HEIGHT_M = {
    BLD_CLASS_COMPACT_RESIDENTIAL: 4.0,
    BLD_CLASS_MEDIUM: 7.0,
    BLD_CLASS_SMALL_APARTMENT: 9.0,
    BLD_CLASS_APARTMENT_BLOCK: 12.0,
    BLD_CLASS_LARGE: 14.0,
}

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
    "simheaven/sheds/shed_02x03x1.obj",
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
    BLD_CLASS_COMPACT_RESIDENTIAL: np.array([255,  80,  80]),
    BLD_CLASS_MEDIUM: np.array([255, 150,  50]),
    BLD_CLASS_SMALL_APARTMENT: np.array([245, 220,  70]),
    BLD_CLASS_APARTMENT_BLOCK: np.array([ 80, 180, 255]),
    BLD_CLASS_LARGE: np.array([ 80, 120, 255]),
}

def _asset_region(tile_lat, tile_lon):
    """Return the asset region key from Natural Earth boundary polygons."""
    return asset_region_for_latlon(tile_lat, tile_lon)


def _default_object_catalog_paths(tile_lat, tile_lon):
    """Return default object aliases that fit the regional context."""
    region = _asset_region(tile_lat, tile_lon)
    if region in ("north_america", "north_america_ne", "north_america_west"):
        return DEFAULT_OBJECT_CATALOG_NORTH_AMERICA
    if region in ("scandinavia", "mediterranean", "europe", "generic"):
        return DEFAULT_OBJECT_CATALOG_OLD_WORLD
    return ()


def _simheaven_catalog_paths(tile_lat, tile_lon):
    """Return simHeaven virtual objects suited to the tile's region."""
    region = _asset_region(tile_lat, tile_lon)
    if region in ("asia", "se_asia", "africa", "australia_oceania", "south_america"):
        return (
            SIMHEAVEN_SMALL_BUILDING_CATALOG +
            SIMHEAVEN_RESIDENTIAL_CATALOG[:5] +
            SIMHEAVEN_COMMERCIAL_CATALOG[:3]
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
    if p in EXCLUDED_BUILDING_FILLER_ASSETS:
        return False
    if any(token in p for token in SIMHEAVEN_SPECIAL_ASSET_TOKENS):
        return False
    return any(token in p for token in SIMHEAVEN_REPEATABLE_ASSET_DIRS)


def _sfd_catalog_paths(tile_lat, tile_lon):
    """Return the strict audited SFD object allowlist for this location."""
    region = _asset_region(tile_lat, tile_lon)

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
    if area_m2 <= SFD_COMPACT_RESIDENTIAL_MAX_M2:
        return BLD_CLASS_COMPACT_RESIDENTIAL
    if area_m2 <= SFD_MEDIUM_MAX_M2:
        return BLD_CLASS_MEDIUM
    if area_m2 <= SFD_SMALL_APARTMENT_MAX_M2:
        return BLD_CLASS_SMALL_APARTMENT
    if (
        area_m2 <= SFD_APARTMENT_BLOCK_MAX_M2 and
        max_side_m <= SFD_APARTMENT_BLOCK_MAX_SIDE_M
    ):
        return BLD_CLASS_APARTMENT_BLOCK
    return BLD_CLASS_LARGE


def _minimum_class_for_object_path(obj_path):
    """Return the smallest placement class allowed by an asset's visual scale."""
    p = (obj_path or '').replace('\\', '/').lower()
    if '/urban_mid_' in p:
        return BLD_CLASS_MEDIUM
    floors = _simheaven_object_floor_count(p)
    if floors is not None and floors > 2.0:
        return BLD_CLASS_MEDIUM
    return BLD_CLASS_COMPACT_RESIDENTIAL


def _class_for_object_asset(obj_path, bounds_m):
    """Classify an object by footprint, with compact kept to low-rise assets."""
    return max(
        _class_for_footprint(bounds_m),
        _minimum_class_for_object_path(obj_path),
    )


def _append_object_asset(asset_pools, obj_path, bounds_m, source):
    """Append one rectangular object asset to the footprint-classed pool map."""
    if (obj_path or '').replace('\\', '/').lower() in EXCLUDED_BUILDING_FILLER_ASSETS:
        return False
    if bounds_m is None:
        return False
    zone_class = _class_for_object_asset(obj_path, bounds_m)
    area_m2, max_side_m = _footprint_metrics(bounds_m)
    asset_pools[zone_class].append({
        'kind': 'object',
        'path': obj_path,
        'bounds_m': bounds_m,
        'footprint_area_m2': area_m2,
        'footprint_max_side_m': max_side_m,
        'footprint_class': zone_class,
        'source': source,
    })
    return True


def _asset_retry_sort_key(asset):
    """Sort assets so exhaustive retries test tighter footprints first."""
    area_m2 = asset.get('footprint_area_m2')
    max_side_m = asset.get('footprint_max_side_m')
    if area_m2 is None or max_side_m is None:
        bounds_m = asset.get('bounds_m')
        if bounds_m is not None:
            area_m2, max_side_m = _footprint_metrics(bounds_m)
    return (
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


def _asset_requires_residential_context(asset):
    """Return True for house-like assets that should stay in residential areas."""
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
    return False


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


def _build_sfd_asset_pools(tile_lat, tile_lon):
    """Return strict tree-free SFD candidates grouped by footprint class."""
    asset_pools = {cls: [] for cls in BLD_PLACEMENT_CLASSES}
    seen_paths = set()
    for obj_path in _sfd_catalog_paths(tile_lat, tile_lon):
        if obj_path in seen_paths:
            continue
        seen_paths.add(obj_path)
        bounds_m = _bounds_for_object_path(obj_path)
        if bounds_m is None:
            continue
        _append_object_asset(asset_pools, obj_path, bounds_m, 'SFD Global')
    return asset_pools


def _build_default_asset_pools(tile_lat=45.0, tile_lon=7.0):
    """Return default X-Plane candidates grouped by footprint placement class."""
    asset_pools = {cls: [] for cls in BLD_PLACEMENT_CLASSES}
    for zone_class in BLD_PLACEMENT_CLASSES:
        for facade_path in DEFAULT_FACADE_VARIANTS_BY_CLASS.get(
            zone_class, (DEFAULT_FACADE_PATHS[zone_class],)
        ):
            asset_pools[zone_class].append({
                'kind': 'facade',
                'path': facade_path,
                'bounds_m': DEFAULT_FACADE_BOUNDS[zone_class],
                'height_m': DEFAULT_FACADE_HEIGHT_M[zone_class],
                'source': 'Default X-Plane',
            })
    seen_paths = set()
    for obj_path in _default_object_catalog_paths(tile_lat, tile_lon):
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


def _build_simheaven_asset_pools(simheaven_objects=None, tile_lat=45.0, tile_lon=7.0):
    """Return simHeaven object candidates grouped by placement size class."""
    asset_pools = {cls: [] for cls in BLD_PLACEMENT_CLASSES}
    seen_paths = set()
    for obj_path in _simheaven_catalog_paths(tile_lat, tile_lon):
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
    for obj in simheaven_objects or ():
        obj_path = obj.get('path')
        if not obj_path or obj_path in seen_paths:
            continue
        if not _is_repeatable_simheaven_asset(obj_path):
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


def _describe_asset_sources(default_available, sfd_available, simheaven_available):
    """Return a short user-facing summary of building asset source selection."""
    parts = []
    if default_available:
        parts.append('default')
    if sfd_available:
        parts.append('SFD')
    if simheaven_available:
        parts.append('simHeaven')
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
        road_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (road_buffer_px * 2 + 1, road_buffer_px * 2 + 1)
        )
        return cv2.dilate(road_mask, road_kernel), 'OSM neighborhood roads'

    return None, 'unavailable'


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


def _format_class_spacing(spacing_px_by_class, m_per_px):
    """Return compact spacing summary for footprint classes."""
    metres = "/".join(
        f"{spacing_px_by_class[cls] * m_per_px:.1f}" for cls in BLD_PLACEMENT_CLASSES
    )
    pixels = "/".join(str(int(spacing_px_by_class[cls])) for cls in BLD_PLACEMENT_CLASSES)
    return f"spacing≈{metres}m ({pixels}px c/s/a/b/l)"


def _describe_placement_summary(class_counts, building_coverage_pct, osm_cell_count,
                                grid_n, spacing_label):
    """Return a user-facing summary for one DDS building-placement pass."""
    total_count = sum(class_counts.values())
    class_bits = "  ".join(
        f"{BLD_CLASS_LABELS[cls]}={class_counts.get(cls, 0)}"
        for cls in BLD_PLACEMENT_CLASSES
    )
    return (
        f"placed={total_count:4d}  "
        f"{class_bits}  "
        f"building cover={building_coverage_pct:.1f}%  "
        f"street-guided cells={osm_cell_count}/{grid_n * grid_n}  "
        f"{spacing_label}"
    )


def _footprint_poly(cx: int, cy: int, bounds_m, heading_deg: float, m_per_px: float):
    """Return the rotated local footprint polygon in image pixel space.

    bounds_m are local SFD object bounds relative to the object origin:
    (xmin, xmax, zmin, zmax) in metres.
    """
    xmin, xmax, zmin, zmax = bounds_m
    rad = math.radians(float(heading_deg))
    right_x = math.cos(rad)
    right_y = math.sin(rad)
    fwd_x = math.sin(rad)
    fwd_y = -math.cos(rad)
    pts = []
    for lx, lz in ((xmin, zmin), (xmax, zmin), (xmax, zmax), (xmin, zmax)):
        px = float(cx) + (lx * right_x + lz * fwd_x) / m_per_px
        py = float(cy) + (lx * right_y + lz * fwd_y) / m_per_px
        pts.append((px, py))
    return np.round(np.asarray(pts, dtype=np.float32)).astype(np.int32)


def _poly_fits(occ_mask: np.ndarray, pts: np.ndarray, scratch_mask: np.ndarray | None = None) -> bool:
    """Return True if polygon pts have no overlap with any set pixel in occ_mask."""
    x1 = max(0, int(pts[:, 0].min()))
    x2 = min(occ_mask.shape[1], int(pts[:, 0].max()) + 1)
    y1 = max(0, int(pts[:, 1].min()))
    y2 = min(occ_mask.shape[0], int(pts[:, 1].max()) + 1)
    if x1 >= x2 or y1 >= y2:
        return True
    if cv2.countNonZero(occ_mask[y1:y2, x1:x2]) == 0:
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
                        residential_context=True, footprint_pad_m=FOOTPRINT_PAD_M):
    """Return the first asset fitting this candidate after exhausting retries."""
    unknown_skipped = 0
    for asset in _asset_retry_sequence(pool, rng, prefer_small=prefer_small):
        if not residential_context and _asset_requires_residential_context(asset):
            file_counts['residential_asset_skipped'] = (
                file_counts.get('residential_asset_skipped', 0) + 1
            )
            continue
        bounds_m = asset.get('bounds_m')
        if bounds_m is None:
            unknown_skipped += 1
            continue

        fit_bounds = asset.get('fit_bounds_m')
        mark_bounds = asset.get('mark_bounds_m')
        if fit_bounds is None or mark_bounds is None:
            fit_bounds = _expand_bounds(bounds_m, footprint_pad_m)
            mark_pad = (
                footprint_pad_m + PLACEMENT_MARGIN_M
                if mark_pad_m is None else float(mark_pad_m)
            )
            mark_bounds = _expand_bounds(
                bounds_m, mark_pad
            )

        final_h = footprint_poly = spacing_poly = None
        static_poly = _footprint_poly(jx, jy, bounds_m, heading, m_per_px)
        dynamic_poly = _footprint_poly(jx, jy, fit_bounds, heading, m_per_px)
        file_counts['fit_checks'] = file_counts.get('fit_checks', 0) + 1
        if (
            _poly_fits(static_occ_mask, static_poly, fit_scratch) and
            _poly_fits(building_spacing_mask, dynamic_poly, fit_scratch)
        ):
            final_h = heading
            footprint_poly = static_poly
            spacing_poly = _footprint_poly(jx, jy, mark_bounds, heading, m_per_px)
        else:
            h90 = (heading + 90.0) % 360.0
            static_poly90 = _footprint_poly(jx, jy, bounds_m, h90, m_per_px)
            dynamic_poly90 = _footprint_poly(jx, jy, fit_bounds, h90, m_per_px)
            file_counts['fit_checks'] = file_counts.get('fit_checks', 0) + 1
            if (
                _poly_fits(static_occ_mask, static_poly90, fit_scratch) and
                _poly_fits(building_spacing_mask, dynamic_poly90, fit_scratch)
            ):
                final_h = h90
                footprint_poly = static_poly90
                spacing_poly = _footprint_poly(jx, jy, mark_bounds, h90, m_per_px)

        if final_h is not None:
            return asset, final_h, footprint_poly, spacing_poly, unknown_skipped

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
    for zone_class, center_mask in center_block_masks.items():
        radius_m = class_min_fit_inradius_m.get(zone_class, 0.0)
        if radius_m <= 0.0:
            continue
        block_bounds = _expand_bounds(mark_bounds, radius_m)
        block_poly = _footprint_poly(cx, cy, block_bounds, heading, m_per_px)
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
    smart_gap_fill=None,
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
                        if SEGFORMER.is_mask_texture_name(name):
                            continue
                        candidate = os.path.splitext(name)[0] + '.dds'
                        if STD_RE.match(candidate) and not SEGFORMER.is_mask_texture_name(candidate):
                            jpg_files.append(candidate)
            if jpg_files:
                print(
                    f"Using {len(jpg_files)} original cached orthophotos from {ortho_dir}",
                    flush=True)
                return jpg_files, 'orthophoto', ortho_dir

        if os.path.isdir(tex_dir):
            dds_files = [
                f for f in os.listdir(tex_dir)
                if STD_RE.match(f) and not SEGFORMER.is_mask_texture_name(f)
            ]
            if dds_files:
                print(
                    f"Original cached orthophotos missing; using {len(dds_files)} DDS textures from {tex_dir}",
                    flush=True)
                return dds_files, 'dds', None

        raise FileNotFoundError(
            f"No DDS textures found at {tex_dir!r} and no cached orthophotos found at {ortho_dir!r}")

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
                log_prefix='[SFR Bld]',
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
    osm_roads = _load_osm_roads(osm_roads_path, cache_dir=cache_dir)
    print(f"OSM roads: {len(osm_roads)} ways from {osm_roads_path}")

    # ── Exclusion data: water, airports, simHeaven buildings, railways ───────
    # OSM building polygons are parsed for diagnostics only. They describe where
    # buildings should exist, not what is already visible in the simulator.
    excl_polys = []           # hard exclusions: water / airports
    existing_bld_polys = []   # occupancy-only blockers: simHeaven facade footprints
    excl_rails = []           # road-style dicts for railways

    for suffix in ('_water.osm.bz2', '_airports.osm.bz2'):
        p = osm_roads_path.replace('_big_roads.osm.bz2', suffix)
        excl_polys.extend(_load_osm_closed_ways(p, cache_dir=cache_dir))
    print(f"Exclusion polygons (water+airports): {len(excl_polys)}")

    excl_cache = osm_roads_path.replace('_big_roads.osm.bz2', '_excl_bld_rail_res.osm.bz2')
    if skip_osm_excl_download:
        ok = os.path.exists(excl_cache)
    else:
        ok = _download_and_cache_osm(lat, lon, excl_cache)
    residential_polys = []
    if ok:
        bld_polys, rail_ways, residential_polys = _parse_excl_osm(
            excl_cache, cache_dir=cache_dir
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
        'cache_load': 0.0,
        'dds_load': 0.0,
        'inference': 0.0,
        'zone_cleanup': 0.0,
        'lookup': 0.0,
        'road_cache': 0.0,
        'road_raster': 0.0,
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
    }
    detail_timing = _env_flag("O4_SFR_TIMING_DETAIL")
    slow_timing_s = _env_float("O4_SFR_TIMING_SLOW", 3.0)
    max_candidates_per_dds = max(
        0, int(_env_float("O4_SFR_BLD_MAX_CANDIDATES", BLD_MAX_CANDIDATES_PER_DDS))
    )
    if smart_gap_fill is None:
        smart_gap_fill = _env_flag("O4_SFR_BLD_SMART_GAP_FILL")
    else:
        smart_gap_fill = bool(smart_gap_fill)
    footprint_pad_m = max(0.0, _env_float("O4_SFR_BLD_FOOTPRINT_PAD_M", FOOTPRINT_PAD_M))
    disable_center_blockers = _env_flag("O4_SFR_BLD_DISABLE_CENTER_BLOCKERS")
    max_gap_candidates_per_dds = max(
        0, int(_env_float("O4_SFR_BLD_GAP_MAX_CANDIDATES", max_candidates_per_dds))
    )
    max_gap_candidates_per_component = max(
        1, int(_env_float("O4_SFR_BLD_GAP_MAX_PER_COMPONENT", 24))
    )
    n_unknown_skipped = 0
    _t = time.perf_counter()

    # simHeaven network roads — used for heading grid only (not zone separation).
    # Provides much better heading coverage than OSM major roads alone.
    if dsftool_path is None:
        dsftool_path = SEGFORMER._dsftool

    if dsftool_path and os.path.exists(dsftool_path):
        sh_network = _load_simheaven_network(custom_scenery_dir, lat, lon, dsftool_path, cache_dir)
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
            cache_dir,
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

    default_assets_available = True
    sfd_assets_available = _find_library_export(custom_scenery_dir, 'sfd_global/')
    simheaven_assets_available = (
        bool(sh_bld_objects) or _find_library_export(custom_scenery_dir, 'simheaven/')
    )
    asset_lat = lat + 0.5
    asset_lon = lon + 0.5
    enabled_asset_pools = [_build_default_asset_pools(asset_lat, asset_lon)]
    if sfd_assets_available:
        enabled_asset_pools.append(_build_sfd_asset_pools(asset_lat, asset_lon))
    if simheaven_assets_available:
        enabled_asset_pools.append(_build_simheaven_asset_pools(
            sh_bld_objects, asset_lat, asset_lon
        ))
    asset_pools = _merge_asset_pools(*enabled_asset_pools)
    smallest_asset_only = _env_flag("O4_SFR_BLD_SMALLEST_ASSET_ONLY")
    if smallest_asset_only:
        _keep_smallest_asset_per_class(asset_pools)
    asset_sources_label = _describe_asset_sources(
        default_assets_available,
        sfd_assets_available,
        simheaven_assets_available,
    )
    if smallest_asset_only:
        asset_sources_label += " (smallest asset per class)"
    print(f"Building assets: {asset_sources_label}")
    if not any(asset_pools.values()):
        print("Building assets: none available for placement")
        return 0
    _sort_asset_pools_for_retry(asset_pools)
    class_min_footprint_span_m = _class_min_footprint_span_m(asset_pools)
    mark_pad_m = _mark_pad_for_edge_spacing_m(spacing_m, footprint_pad_m)
    for pool in asset_pools.values():
        for asset in pool:
            bounds_m = asset.get('bounds_m')
            if bounds_m is None:
                continue
            asset['fit_bounds_m'] = _expand_bounds(bounds_m, footprint_pad_m)
            asset['mark_bounds_m'] = _expand_bounds(
                bounds_m, mark_pad_m
            )
    class_min_fit_inradius_m = _class_min_fit_inradius_m(asset_pools)
    if disable_center_blockers:
        class_min_fit_inradius_m = {cls: 0.0 for cls in BLD_PLACEMENT_CLASSES}

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

    # Heading grid uses simHeaven network only — it represents the full local
    # street grid (minor roads, blocks) which defines actual building alignment.
    # OSM major roads (motorways, primaries) are long segments that dominate the
    # length-weighted mean and misalign buildings with the local block grid.
    all_roads       = sh_network                          # heading grid source (local street grid)
    separator_roads = (osm_roads or []) + (sh_network or [])  # zone separator: major roads + local streets
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

    placed_objects = []   # list of (lon, lat, heading, obj_path)
    placed_facades = []   # list of (lonlat_ring, facade_path, height_m)
    candidate_grid_cache = {}
    _bld_params = (
        BLD_PLACEMENT_CACHE_VERSION, spacing_m, close_k, open_k, min_zone_m2,
        max_candidates_per_dds,
        PLACE_UNKNOWN_OBJECTS,
        footprint_pad_m,
        mark_pad_m,
        smallest_asset_only,
        smart_gap_fill,
        disable_center_blockers,
        max_gap_candidates_per_dds,
        max_gap_candidates_per_component,
        tuple(sorted(class_min_footprint_span_m.items())),
        ROAD_CENTERLINE_WIDTH_M, ROAD_EXTRA_BUFFER_M,
        ROAD_WIDTH_PX_MIN, ROAD_DILATE_PX_MIN,
        excl_poly_sig, existing_bld_poly_sig, rail_sig, sh_bld_sig,
        residential_poly_sig,
        default_assets_available, sfd_assets_available, simheaven_assets_available,
        _asset_region(asset_lat, asset_lon),
    )

    t_start = time.time()
    n_files = len(files)
    for fi, fname in enumerate(files, 1):
        m = STD_RE.match(fname)
        if not m: continue
        _dds_cache_files = _dds_cache_paths(fname)
        file_timings = {}
        file_counts = {}
        file_t0 = time.perf_counter()
        if disable_cache:
            # Drop any stale per-DDS cache before starting this DDS.
            _remove_cache_files(_dds_cache_files)

        try:
            # ── Building placement cache ──────────────────────────────────────
            # Cache is keyed by DDS filename (encodes tile position+ZL) + params.
            # Per-tile deterministic rng so cached and non-cached tiles both reproduce.
            import pickle as _pickle
            rng_seed = int.from_bytes(
                hashlib.sha1(f"bld-place:{fname}".encode("utf-8")).digest()[:8],
                "big",
            )
            rng = np.random.default_rng(rng_seed)
            _bld_cache_file = os.path.join(cache_dir, fname.replace('.dds', '_bld.pkl'))
            _cached_bld = None
            if not disable_cache and os.path.exists(_bld_cache_file):
                try:
                    with open(_bld_cache_file, 'rb') as _f:
                        _cd = _pickle.load(_f)
                    if _cd.get('params') == _bld_params:
                        _cached_bld = _cd['placements']
                except Exception:
                    pass
            if _cached_bld is not None:
                placed_objects.extend(_cached_bld.get('objects', ()))
                placed_facades.extend(_cached_bld.get('facades', ()))
                cached_count = len(_cached_bld.get('objects', ())) + len(_cached_bld.get('facades', ()))
                print(f"  [{fi:3d}/{n_files}] {fname}  (bld cached — {cached_count} placements)", flush=True)
                continue
            _start_object_idx = len(placed_objects)
            _start_facade_idx = len(placed_facades)
            til_y_top  = int(m.group(1))
            til_x_left = int(m.group(2))
            zl         = int(m.group(4))

            # Geographic bounds
            lat_n, lat_s, lon_w, lon_e = dds_bounds(til_y_top, til_x_left, zl)

            # Inference (cached per DDS filename — filename encodes tile coords + ZL)
            cache_path = os.path.join(cache_dir, fname.replace('.dds', '_veg.npy'))
            img = None
            if not disable_cache and os.path.exists(cache_path):
                _t = time.perf_counter()
                veg_map = np.load(cache_path)
                _record_elapsed(timings, file_timings, 'cache_load', _t)
            else:
                _t = time.perf_counter()
                img = _load_source_image(fname, _source_mode, _orthophoto_dir)
                if img is None:
                    continue
                _record_elapsed(timings, file_timings, 'dds_load', _t)
                if model is None:
                    model, proc, device = SEGFORMER.load_vegetation_model(device)
                _t = time.perf_counter()
                veg_map = SEGFORMER.run_inference(model, device, img, proc)
                _record_elapsed(timings, file_timings, 'inference', _t)
                if not disable_cache:
                    _t = time.perf_counter()
                    np.save(cache_path, veg_map)
                    _record_elapsed(timings, file_timings, 'cache_save', _t)

            img_h, img_w = veg_map.shape[:2]

            # Pixel size in metres (approximate, using mid-latitude)
            mid_lat_rad = math.radians((lat_n + lat_s) / 2)
            lon_span_m  = (lon_e - lon_w) * 111320 * math.cos(mid_lat_rad)
            lat_span_m  = (lat_n - lat_s) * 110540

            # Spacing in pixels at this tile's native resolution
            m_per_px_x = lon_span_m / img_w
            m_per_px_y = lat_span_m / img_h
            m_per_px   = (m_per_px_x + m_per_px_y) / 2
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

            # Zone cleanup
            _t = time.perf_counter()
            bld_raw  = (veg_map == SEGFORMER.CLASS_BUILDING).astype(np.uint8)
            bld_zone = cv2.morphologyEx(bld_raw,  cv2.MORPH_CLOSE, k_close)
            bld_zone = cv2.morphologyEx(bld_zone, cv2.MORPH_OPEN,  k_open)

            # Re-subtract SegFormer road/developed pixels that were filled over by
            # the morphological close.  This restores street gaps between building
            # blocks that close_k would otherwise bridge.
            sfr_road_raw = (
                (veg_map == SEGFORMER.CLASS_ROAD) | (veg_map == SEGFORMER.CLASS_DEVELOPED)
            ).astype(np.uint8)
            sfr_road_dilated = None
            if sfr_road_raw.any():
                _sfr_r  = road_dilate_px
                _sfr_ks = _sfr_r * 2 + 1
                _ksfr   = cv2.getStructuringElement(cv2.MORPH_RECT, (_sfr_ks, _sfr_ks))
                sfr_road_dilated = cv2.dilate(sfr_road_raw, _ksfr)
                bld_zone = bld_zone & (~sfr_road_dilated)
            _record_elapsed(timings, file_timings, 'zone_cleanup', _t)

            TILE_EDGE_MARGIN_M = 20.0
            edge_px = max(2, int(TILE_EDGE_MARGIN_M / m_per_px))

            _t = time.perf_counter()
            local_separator_roads = _roads_for_bounds(
                separator_roads_index, lat_n, lat_s, lon_w, lon_e, pad_deg=0.001)
            local_residential_roads = _residential_roads(
                _roads_for_bounds(osm_roads_index, lat_n, lat_s, lon_w, lon_e, pad_deg=0.001)
            )
            local_rails = _roads_for_bounds(
                excl_rails_index, lat_n, lat_s, lon_w, lon_e, pad_deg=0.001)
            local_residential_polys = _polys_for_bounds(
                residential_polys_index, lat_n, lat_s, lon_w, lon_e, pad_deg=0.001)
            local_excl_polys = _polys_for_bounds(
                excl_polys_index, lat_n, lat_s, lon_w, lon_e, pad_deg=0.001)
            local_existing_bld_polys = _polys_for_bounds(
                existing_bld_polys_index, lat_n, lat_s, lon_w, lon_e, pad_deg=0.001)
            local_heading_segments = _segments_for_bounds(
                heading_seg_index, lat_n, lat_s, lon_w, lon_e, pad_deg=0.001)
            local_sh_bld_objects = _simheaven_objects_for_bounds(
                sh_bld_index, lat_n, lat_s, lon_w, lon_e, pad_deg=0.002)
            nearest_heading_segments = (
                local_heading_segments or
                _segments_for_bounds(heading_seg_index, lat_n, lat_s, lon_w, lon_e, pad_deg=0.02) or
                heading_seg_index
            )
            _record_elapsed(timings, file_timings, 'lookup', _t)

            _road_cache_file = os.path.join(cache_dir, fname.replace('.dds', '_road.pkl'))
            _road_key = _dds_road_cache_key(
                fname, lat_n, lat_s, lon_w, lon_e, img_h, img_w,
                grid_n, road_width_px, road_dilate_px,
                separator_sig, rail_sig, heading_sig, residential_poly_sig,
                excl_poly_sig, existing_bld_poly_sig, sh_bld_sig,
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

                # ── Heading grid ─────────────────────────────────────────────
                _t = time.perf_counter()
                if local_heading_segments:
                    if img is None:
                        _img_t = time.perf_counter()
                        img = _load_source_image(fname, _source_mode, _orthophoto_dir)
                        if img is None:
                            continue
                        _record_elapsed(timings, file_timings, 'dds_load', _img_t)
                    hgrid = _road_heading_grid(
                        None, lat_n, lat_s, lon_w, lon_e,
                        img_h, img_w, grid_n, img=img,
                        segments=local_heading_segments,
                    )
                else:
                    hgrid = np.full((grid_n, grid_n), np.nan)
                n_osm_cells = int(np.sum(~np.isnan(hgrid)))
                hgrid = _fill_heading_grid_nearest(
                    hgrid, nearest_heading_segments, lat_n, lat_s, lon_w, lon_e,
                    img_h, img_w, grid_n,
                )
                if np.any(np.isnan(hgrid)):
                    if img is None:
                        _img_t = time.perf_counter()
                        img = _load_source_image(fname, _source_mode, _orthophoto_dir)
                        if img is None:
                            continue
                        _record_elapsed(timings, file_timings, 'dds_load', _img_t)
                    image_hgrid = _image_heading_grid(img, img_h, img_w, grid_n)
                    hgrid = np.where(np.isnan(hgrid), image_hgrid, hgrid)
                residential_area_mask, residential_area_source = _build_residential_area_mask(
                    local_residential_polys,
                    local_residential_roads,
                    lat_n,
                    lat_s,
                    lon_w,
                    lon_e,
                    img_h,
                    img_w,
                    m_per_px,
                )
                _record_elapsed(timings, file_timings, 'heading_grid', _t)

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

                if not disable_cache:
                    _t = time.perf_counter()
                    _save_dds_road_cache(
                        _road_cache_file,
                        _road_key,
                        road_mask,
                        rail_mask,
                        hgrid,
                        n_osm_cells,
                        residential_area_mask,
                        residential_area_source,
                        poly_mask,
                        existing_bld_mask,
                        sh_bld_mask,
                    )
                    _record_elapsed(timings, file_timings, 'cache_save', _t)

            _t = time.perf_counter()
            bld_zone = bld_zone & (~road_mask)
            static_occ_mask = road_mask.copy()

            if poly_mask is not None and poly_mask.any():
                bld_zone  = bld_zone & (~poly_mask)
                static_occ_mask  = static_occ_mask | poly_mask

            if existing_bld_mask is not None and existing_bld_mask.any():
                static_occ_mask = static_occ_mask | existing_bld_mask

            if rail_mask.any():
                bld_zone  = bld_zone & (~rail_mask)
                static_occ_mask  = static_occ_mask | rail_mask

            if sh_bld_mask is not None and sh_bld_mask.any():
                static_occ_mask = static_occ_mask | sh_bld_mask

            DEGREE_TOL = 1e-4
            if lat_n >= lat + 1 - DEGREE_TOL:   static_occ_mask[:edge_px,  :]  = 1
            if lat_s <= lat     + DEGREE_TOL:   static_occ_mask[-edge_px:, :]  = 1
            if lon_w <= lon     + DEGREE_TOL:   static_occ_mask[:,  :edge_px]  = 1
            if lon_e >= lon + 1 - DEGREE_TOL:   static_occ_mask[:, -edge_px:]  = 1
            building_spacing_mask = np.zeros_like(static_occ_mask)
            _record_elapsed(timings, file_timings, 'mask_apply', _t)

            cell_h = img_h // grid_n
            cell_w = img_w // grid_n

            if np.any(np.isnan(hgrid)):
                fallback_heading = float(rng.integers(0, 360))
                hgrid = np.where(np.isnan(hgrid), fallback_heading, hgrid)

            _t = time.perf_counter()
            min_zone_px = max(1, int(min_zone_m2 / max(m_per_px * m_per_px, 1e-6)))
            n_cc, cc_labels, cc_stats, cc_centroids = cv2.connectedComponentsWithStats(bld_zone, connectivity=8)
            cc_area = cc_stats[:, cv2.CC_STAT_AREA]
            label_class = np.zeros(n_cc, dtype=np.uint8)
            valid_labels = np.flatnonzero((np.arange(n_cc) != 0) & (cc_area >= min_zone_px))
            if valid_labels.size:
                valid_area = cc_area[valid_labels]
                label_class[valid_labels[valid_area < ZONE_COMPACT_PX]] = (
                    BLD_CLASS_COMPACT_RESIDENTIAL
                )
                label_class[valid_labels[
                    (valid_area >= ZONE_COMPACT_PX) &
                    (valid_area < ZONE_MEDIUM_PX)
                ]] = BLD_CLASS_MEDIUM
                label_class[valid_labels[
                    (valid_area >= ZONE_MEDIUM_PX) &
                    (valid_area < ZONE_SMALL_APARTMENT_PX)
                ]] = BLD_CLASS_SMALL_APARTMENT
                label_class[valid_labels[
                    (valid_area >= ZONE_SMALL_APARTMENT_PX) &
                    (valid_area < ZONE_APARTMENT_BLOCK_PX)
                ]] = BLD_CLASS_APARTMENT_BLOCK
                label_class[valid_labels[valid_area >= ZONE_APARTMENT_BLOCK_PX]] = (
                    BLD_CLASS_LARGE
                )
            zone_class = label_class[cc_labels]

            zone_heading = np.full(n_cc, np.nan, dtype=np.float32)
            if valid_labels.size:
                centroid_x = np.clip(
                    np.rint(cc_centroids[valid_labels, 0]).astype(np.int32), 0, img_w - 1
                )
                centroid_y = np.clip(
                    np.rint(cc_centroids[valid_labels, 1]).astype(np.int32), 0, img_h - 1
                )
                zone_heading[valid_labels] = hgrid[
                    np.minimum(grid_n - 1, centroid_y // cell_h),
                    np.minimum(grid_n - 1, centroid_x // cell_w),
                ]
                side_band_px = max(
                    8,
                    road_width_px + road_dilate_px + 2,
                    int(round(14.0 / max(m_per_px, 1e-6))),
                )
                side_heading, side_heading_counts = _component_side_touch_headings(
                    cc_labels,
                    cc_stats,
                    valid_labels,
                    local_separator_roads,
                    lat_n,
                    lat_s,
                    lon_w,
                    lon_e,
                    img_h,
                    img_w,
                    band_px=side_band_px,
                    img=img,
                )
                side_valid = ~np.isnan(side_heading[valid_labels])
                if np.any(side_valid):
                    side_labels = valid_labels[side_valid]
                    zone_heading[side_labels] = side_heading[side_labels]
                    file_counts['side_heading_zones'] = int(side_valid.sum())
                    file_counts['side_heading_touches'] = int(
                        np.sum(side_heading_counts[side_labels])
                    )
                sh_heading, sh_heading_counts = _simheaven_building_zone_headings(
                    local_sh_bld_objects,
                    cc_labels,
                    valid_labels,
                    lat_n,
                    lat_s,
                    lon_w,
                    lon_e,
                    img_h,
                    img_w,
                )
                sh_valid = ~np.isnan(sh_heading[valid_labels])
                if np.any(sh_valid):
                    sh_labels = valid_labels[sh_valid]
                    zone_heading[sh_labels] = sh_heading[sh_labels]
                    file_counts['simheaven_heading_zones'] = int(sh_valid.sum())
                    file_counts['simheaven_heading_objects'] = int(
                        np.sum(sh_heading_counts[sh_labels])
                    )
            cc_elapsed = _record_elapsed(timings, file_timings, 'connected_components', _t)
            timings['placement'] += cc_elapsed

            _t = time.perf_counter()
            pts_this = []
            cand_x_parts = []
            cand_y_parts = []
            cand_cls_parts = []
            cand_label_parts = []
            n_initial_blocked = 0
            n_candidates_total = 0
            for target_cls in BLD_PLACEMENT_CLASSES:
                spacing_passes = []
                fine_sp_px = spacing_px_by_class[BLD_CLASS_COMPACT_RESIDENTIAL]
                coarse_sp_px = spacing_px_by_class[target_cls]
                if target_cls > BLD_CLASS_MEDIUM and residential_area_mask is not None:
                    residential_cls = min(target_cls, BLD_CLASS_SMALL_APARTMENT)
                    spacing_passes.append((fine_sp_px, True, residential_cls))
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
                    keep = zone_class[cls_cand_y, cls_cand_x] == target_cls
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
                if max_candidates_per_dds and cand_x.size > max_candidates_per_dds:
                    cand_x, cand_y, cand_cls, n_dropped = _limit_candidates_by_component(
                        cand_x, cand_y, cand_cls, cand_labels, max_candidates_per_dds, rng
                    )
                    file_counts['candidate_cap'] = int(n_dropped)
            else:
                cand_x = np.empty(0, dtype=np.int32)
                cand_y = np.empty(0, dtype=np.int32)
                cand_cls = np.empty(0, dtype=np.uint8)
            candidate_elapsed = _record_elapsed(timings, file_timings, 'candidate_grid', _t)
            timings['placement'] += candidate_elapsed

            _t = time.perf_counter()
            fit_scratch = np.zeros_like(static_occ_mask)
            dynamic_center_block_masks = {
                cls: np.zeros_like(static_occ_mask)
                for cls in BLD_PLACEMENT_CLASSES
            }
            placed_viz_polys = []

            def _heading_for_candidate(jx, jy, zone_label):
                return float(zone_heading[zone_label]) if (
                    0 <= zone_label < zone_heading.shape[0]
                    and not np.isnan(zone_heading[zone_label])
                ) else float(hgrid[
                    min(grid_n - 1, jy // cell_h),
                    min(grid_n - 1, jx // cell_w),
                ])

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
                dom_h = _heading_for_candidate(jx, jy, zone_label)
                jitter  = rng.uniform(-HEADING_JITTER_DEG, HEADING_JITTER_DEG)
                heading = (dom_h + jitter) % 360.0

                residential_context = (
                    residential_area_mask is None or
                    bool(residential_area_mask[jy, jx])
                )
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
                        placed_objects.append((o_lon, o_lat, final_h, asset['path']))
                    else:
                        placed_facades.append((
                            _pixel_ring_to_latlon(footprint_poly, img_w, img_h, lat_n, lat_s, lon_w, lon_e),
                            asset['path'],
                            float(asset.get('height_m', DEFAULT_FACADE_HEIGHT_M.get(try_cls, 8.0))),
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

            if smart_gap_fill:
                leftover_mask = (
                    (bld_zone != 0) &
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
                    dom_h = _heading_for_candidate(jx, jy, zone_label)
                    jitter_h = rng.uniform(-HEADING_JITTER_DEG, HEADING_JITTER_DEG)
                    heading = (dom_h + jitter_h) % 360.0

                    placed_gap = False
                    dynamic_blocked = False
                    residential_context = (
                        residential_area_mask is None or
                        bool(residential_area_mask[jy, jx])
                    )
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
                            placed_objects.append((o_lon, o_lat, final_h, asset['path']))
                        else:
                            placed_facades.append((
                                _pixel_ring_to_latlon(
                                    footprint_poly, img_w, img_h,
                                    lat_n, lat_s, lon_w, lon_e,
                                ),
                                asset['path'],
                                float(asset.get(
                                    'height_m',
                                    DEFAULT_FACADE_HEIGHT_M.get(try_cls, 8.0),
                                )),
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
            class_counts = {
                cls: sum(1 for p in pts_this if p[3] == cls)
                for cls in BLD_PLACEMENT_CLASSES
            }
            spacing_label = _format_class_spacing(spacing_px_by_class, m_per_px)
            print(
                f"  [{fi:3d}/{n_files}] {fname}  "
                f"{_describe_placement_summary(class_counts, bld_pct, n_osm_cells, grid_n, spacing_label)}"
                f"  small-house areas={residential_area_source}"
            )

            if not disable_cache:
                try:
                    _t = time.perf_counter()
                    with open(_bld_cache_file, 'wb') as _f:
                        _pickle.dump({
                            'params': _bld_params,
                            'placements': {
                                'objects': placed_objects[_start_object_idx:],
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
                _blend_viz_mask(road_mask, (255, 0, 0), 0.75)
                _blend_viz_mask(rail_mask, (255, 0, 255), 0.75)
                _blend_viz_mask(poly_mask, (120, 0, 255), 0.75)
                _blend_viz_mask(existing_bld_mask, (0, 0, 0), 0.70)
                _blend_viz_mask(sh_bld_mask, (255, 255, 255), 0.65)
                pil = Image.fromarray(panel); draw = ImageDraw.Draw(pil)
                dot_colours = {
                    BLD_CLASS_COMPACT_RESIDENTIAL: (0, 220, 0),
                    BLD_CLASS_MEDIUM: (160, 220, 0),
                    BLD_CLASS_SMALL_APARTMENT: (255, 220, 0),
                    BLD_CLASS_APARTMENT_BLOCK: (0, 190, 255),
                    BLD_CLASS_LARGE: (0, 120, 255),
                }
                footprint_fill = (255, 255, 255)
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
                composite[row*TILE_VIZ:(row+1)*TILE_VIZ, col*TILE_VIZ:(col+1)*TILE_VIZ] = np.array(pil)

                if footprint_composite is not None:
                    fp_base = Image.fromarray(
                        np.array(
                            Image.fromarray(img).resize((TILE_VIZ, TILE_VIZ), Image.LANCZOS)
                        )
                    ).convert('RGBA')
                    fp_layer = Image.new('RGBA', (TILE_VIZ, TILE_VIZ), (0, 0, 0, 0))
                    fp_draw = ImageDraw.Draw(fp_layer)
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
                    fp_img = Image.alpha_composite(fp_base, fp_layer).convert('RGB')
                    footprint_composite[
                        row*TILE_VIZ:(row+1)*TILE_VIZ,
                        col*TILE_VIZ:(col+1)*TILE_VIZ,
                    ] = np.array(fp_img)
                _record_elapsed(timings, file_timings, 'viz', _t_viz)
        finally:
            file_elapsed = time.perf_counter() - file_t0
            if detail_timing or (slow_timing_s > 0 and file_elapsed >= slow_timing_s):
                _print_dds_timing(fname, file_timings, file_elapsed, file_counts)
            if disable_cache:
                _remove_cache_files(_dds_cache_files)

    total_time = time.time() - t_start
    total_placements = len(placed_objects) + len(placed_facades)
    print(
        f"\nTotal: {total_placements:,} placements  "
        f"({len(placed_objects):,} objects, {len(placed_facades):,} facades)  "
        f"({total_time/60:.1f}min)"
    )

    # ── Write DSF text ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(out_dsf)), exist_ok=True)
    txt_path = out_dsf.replace('.dsf', '_bld.txt')

    obj_paths = sorted(set(obj_path for _, _, _, obj_path in placed_objects))
    obj_idx = {path: i for i, path in enumerate(obj_paths)}
    facade_paths = sorted(set(facade_path for _, facade_path, _ in placed_facades))
    facade_idx = {path: i for i, path in enumerate(facade_paths)}
    _t = time.perf_counter()
    with open(txt_path, 'w') as f:
        f.write("PROPERTY sim/planet earth\n")
        f.write("PROPERTY sim/overlay 1\n")
        f.write(f"PROPERTY sim/west  {int(lon)}\n")
        f.write(f"PROPERTY sim/east  {int(lon)+1}\n")
        f.write(f"PROPERTY sim/south {int(lat)}\n")
        f.write(f"PROPERTY sim/north {int(lat)+1}\n")
        f.write("\n")
        for p in facade_paths:
            f.write(f"POLYGON_DEF {p}\n")
        for p in obj_paths:
            f.write(f"OBJECT_DEF {p}\n")
        f.write("\n")
        for lonlat_ring, facade_path, height_m in placed_facades:
            idx = facade_idx[facade_path]
            f.write(f"BEGIN_POLYGON {idx} {height_m:.1f} 2\n")
            _write_polygon_winding(f, lonlat_ring)
            f.write("END_POLYGON\n")
        for o_lon, o_lat, heading, obj_path in placed_objects:
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
        try: os.remove(txt_path)
        except: pass
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
        f"cache_load={timings['cache_load']:.1f}s  "
        f"dds_load={timings['dds_load']:.1f}s  "
        f"inference={timings['inference']:.1f}s  "
        f"road_cache={timings['road_cache']:.1f}s  "
        f"road_raster={timings['road_raster']:.1f}s  "
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
            f"fit={timings['fit_loop']:.1f}s  "
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
        cache_dir  = cache_dir,
        grid_n          = args.grid_n,
        osm_roads_path  = args.osm_roads,
        custom_scenery_dir = args.custom_scenery_dir,
    )


if __name__ == '__main__':
    main()

