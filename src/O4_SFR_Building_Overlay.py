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
)


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
                ("dynamic_blocked", "dynamic_center_blocked"),
                ("cap", "candidate_cap"),
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


# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    """Parse CLI arguments for building overlay generation."""
    ap = argparse.ArgumentParser(description='SegFormer building overlay generator')
    ap.add_argument('tex_dir')
    ap.add_argument('lat',     type=float)
    ap.add_argument('lon',     type=float)
    ap.add_argument('out_dsf', nargs='?', default=None,
                    help='Output DSF path (auto-derived under yOrtho4XP_Bld_Overlays if omitted)')
    ap.add_argument('--spacing',   type=float, default=20.0)
    ap.add_argument('--close',     type=int,   default=15)
    ap.add_argument('--open-k',    type=int,   default=5,  dest='open_k')
    ap.add_argument('--min-zone', '--min-zone-m2', type=float, default=200.0, dest='min_zone_m2')
    ap.add_argument('--no-viz',    action='store_true')
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--grid-n',    type=int,   default=HEADING_GRID_N, dest='grid_n')
    ap.add_argument('--osm-roads', default=None,
                    help='Path to *_big_roads.osm.bz2 (auto-discovered if omitted)')
    ap.add_argument('--custom-scenery-dir', default=None,
                    help='Configured X-Plane Custom Scenery directory used to locate simHeaven.')
    ap.add_argument('--default-assets', dest='default_assets', action='store_true',
                    help='Allow default X-Plane facade assets.')
    ap.add_argument('--no-default-assets', dest='default_assets', action='store_false',
                    help='Disable default X-Plane facade assets.')
    ap.add_argument('--sfd-assets', dest='sfd_assets', action='store_true',
                    help='Allow SFD Global object assets.')
    ap.add_argument('--no-sfd-assets', dest='sfd_assets', action='store_false',
                    help='Disable SFD Global object assets.')
    ap.add_argument('--simheaven-assets', dest='simheaven_assets', action='store_true',
                    help='Allow simHeaven object assets when simHeaven scenery is available.')
    ap.add_argument('--no-simheaven-assets', dest='simheaven_assets', action='store_false',
                    help='Disable simHeaven object assets.')
    ap.set_defaults(default_assets=False, sfd_assets=True, simheaven_assets=False)
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
        'version': 3,
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


def _simheaven_object_dims(path):
    """Infer a simHeaven object's footprint dimensions from its filename."""
    name = os.path.basename((path or '').replace('\\', '/')).lower()
    match = _SIMHEAVEN_DIMS_RE.search(name)
    if match:
        return float(match.group(1)), float(match.group(2))
    if any(token in name for token in ('church', 'chapel', 'mosque')):
        return 26.0, 26.0
    return 14.0, 14.0


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


# ── SFD object pools by zone size ─────────────────────────────────────────────
# Zone size thresholds (pixels²) after morphological cleanup at ZL16 native res.
# At ~2.4 m/px:  SMALL < 3 000 px²  ≈ < 55×55 m cluster  → houses
#                MEDIUM 3 000-30 000 px²  ≈ 55-175 m      → apartments
#                LARGE  > 30 000 px²  ≈ > 175 m           → industrial / hi-rise
ZONE_SMALL_PX  =  3_000
ZONE_LARGE_PX  = 30_000

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
#   SMALL:  7 m  → ~14 m exclusion diameter  (suburban houses, fine packing)
#   MEDIUM: 22 m → ~44 m exclusion diameter  (apartment slabs, small industry)
#   LARGE:  42 m → ~84 m exclusion diameter  (large warehouse/industry boxes)
OBJ_CLEARANCE_M = {1: 14.0, 2: 32.0, 3: 55.0}  # legacy fallback for unknown footprints
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
    # ── New England ──────────────────────────────────────────────────────────
    **{f"SFD_Global/New_England/Residential/Suburban_{i}.obj": d
       for i, d in enumerate([
           (9.69,  14.13), (8.98, 12.79), (8.0,  16.63), (10.09, 12.22),
           (10.76,  7.29), (9.21, 16.12), (9.56, 14.8),  (8.12,  12.8),
       ], 1)},
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
PLACEMENT_MARGIN_M = 6.0   # clearance gap (metres) added around each footprint
FOOTPRINT_PAD_M = 4.0      # expand known footprints before fit/mark to reduce overlaps
BLD_PLACEMENT_CACHE_VERSION = 13
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
    1: (-7.0, 7.0, -7.0, 7.0),
    2: (-12.0, 12.0, -9.0, 9.0),
    3: (-24.0, 24.0, -24.0, 24.0),
}
DEFAULT_FACADE_PATHS = {
    1: SEGFORMER._FAC_DEFS["medium"],
    2: SEGFORMER._FAC_DEFS["medium"],
    3: SEGFORMER._FAC_DEFS["large"],
}
DEFAULT_FACADE_HEIGHT_M = {
    1: 4.0,
    2: 9.0,
    3: 14.0,
}

# Viz colours per zone class (BGR→RGB in numpy overlay)
ZONE_COLOURS = {
    1: np.array([255,  80,  80]),   # red    — small/residential
    2: np.array([255, 160,  40]),   # orange — medium/apartments
    3: np.array([ 80, 120, 255]),   # blue   — large/industrial
}


def _sfd_pools_by_size(tile_lat, tile_lon):
    """Return {1: small_pool, 2: medium_pool, 3: large_pool} for this location."""

    if tile_lat >= 55:  # Scandinavia
        # Global audit rule: keep only measured 4-sided footprints with strong
        # rectangular fill. This avoids T/L/U-shaped models contributing large
        # overlaps even when centre-to-centre spacing looks acceptable.
        s = [f"SFD_Global/Scandinavia/Residential/Suburban_{i}.obj" for i in (3, 5, 7)]
        return {1: s, 2: s, 3: s}

    if tile_lon >= 60 and tile_lon <= 150 and tile_lat >= 10:  # Asia
        suburban = [f"SFD_Global/Asia/Suburban_{i}.obj" for i in (3, 10)]
        apts     = ["SFD_Global/Asia/Apartment_3.obj"]
        ind_med  = [
            "SFD_Global/Asia/Industry_20x40.obj",
            "SFD_Global/Asia/Industry_30x40.obj",
        ]
        ind_large = [
            "SFD_Global/Asia/Industry_60x50.obj",
            "SFD_Global/Asia/Industry_70x90.obj",
            "SFD_Global/Asia/Industry_150x80.obj",
        ]
        return {
            1: suburban,
            2: suburban + apts + ind_med,
            3: apts + ind_med + ind_large,
        }

    if tile_lon >= 60 and tile_lat < 10:  # SE Asia south
        s = [f"SFD_Global/Asia/Suburban_South_{i}.obj" for i in (1, 3, 7, 8)]
        return {1: s, 2: s, 3: s}

    if -20 <= tile_lon <= 55 and tile_lat < 20:  # Africa tropical
        s = [f"SFD_Global/Africa/Residential/Suburban_{i}.obj" for i in (4, 6)]
        return {1: s, 2: s, 3: s}

    if -20 <= tile_lon <= 55 and tile_lat >= 20:  # Med / N Africa / Middle East
        sub  = [f"SFD_Global/Med/Residential/Suburban_{i}.obj" for i in (3, 5, 6, 8)]
        apts = [f"SFD_Global/Med/Residential/Apartment_North_{i}.obj" for i in (1, 2, 3, 4, 5)]
        return {1: sub, 2: sub + apts, 3: apts}

    if tile_lon < -30 and tile_lat >= 40:  # N America north-east
        s = [f"SFD_Global/New_England/Residential/Suburban_{i}.obj" for i in (1, 2, 5, 6)]
        return {1: s, 2: s, 3: s}

    if tile_lon < -30 and tile_lat >= 15:  # N America west / south
        wc = ["SFD_Global/US_West_Coast/Suburban_4.obj"]
        ne = [f"SFD_Global/New_England/Residential/Suburban_{i}.obj" for i in (1, 2, 5, 6)]
        s = wc + ne
        return {1: s, 2: s, 3: s}

    if tile_lon < -30 and tile_lat < 15:  # S America
        sub = [f"SFD_Global/South_America/Suburban_{i}.obj" for i in (1, 2, 3, 6, 7, 8, 9, 10)]
        med = [f"SFD_Global/South_America/Med_{i}.obj" for i in (1, 2, 3, 6, 7, 8)]
        return {1: sub, 2: sub + med, 3: med}

    # Default — Mediterranean (same exclusions as Med region above)
    sub  = [f"SFD_Global/Med/Residential/Suburban_{i}.obj" for i in (3, 5, 6, 8)]
    apts = [f"SFD_Global/Med/Residential/Apartment_North_{i}.obj" for i in (1, 2, 3, 4, 5)]
    return {1: sub, 2: sub + apts, 3: apts}


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


def _build_sfd_asset_pools(tile_lat, tile_lon):
    """Return SFD object candidates grouped by placement size class."""
    asset_pools = {1: [], 2: [], 3: []}
    for zone_class, obj_paths in _sfd_pools_by_size(tile_lat, tile_lon).items():
        for obj_path in obj_paths:
            bounds_m = _bounds_for_object_path(obj_path)
            if bounds_m is None:
                continue
            asset_pools[zone_class].append({
                'kind': 'object',
                'path': obj_path,
                'bounds_m': bounds_m,
                'source': 'SFD Global',
            })
    return asset_pools


def _build_default_asset_pools():
    """Return default X-Plane facade candidates grouped by placement size class."""
    asset_pools = {1: [], 2: [], 3: []}
    for zone_class in (1, 2, 3):
        asset_pools[zone_class].append({
            'kind': 'facade',
            'path': DEFAULT_FACADE_PATHS[zone_class],
            'bounds_m': DEFAULT_FACADE_BOUNDS[zone_class],
            'height_m': DEFAULT_FACADE_HEIGHT_M[zone_class],
            'source': 'Default X-Plane',
        })
    return asset_pools


def _simheaven_zone_class(width_m, depth_m):
    """Classify a simHeaven object into the overlay's small/medium/large buckets."""
    area_m2 = float(width_m) * float(depth_m)
    if area_m2 < 350.0:
        return 1
    if area_m2 < 2500.0:
        return 2
    return 3


def _build_simheaven_asset_pools(simheaven_objects):
    """Return simHeaven object candidates grouped by placement size class."""
    asset_pools = {1: [], 2: [], 3: []}
    seen_paths = set()
    for obj in simheaven_objects or ():
        obj_path = obj.get('path')
        if not obj_path or obj_path in seen_paths:
            continue
        seen_paths.add(obj_path)
        zone_class = _simheaven_zone_class(obj['w_m'], obj['h_m'])
        asset_pools[zone_class].append({
            'kind': 'object',
            'path': obj_path,
            'bounds_m': _bounds_from_dimensions(obj['w_m'], obj['h_m']),
            'source': 'simHeaven',
        })
    return asset_pools


def _merge_asset_pools(*pool_maps):
    """Merge multiple ``{class: [asset, ...]}`` mappings into one pool map."""
    merged = {1: [], 2: [], 3: []}
    for pool_map in pool_maps:
        if not pool_map:
            continue
        for zone_class in merged:
            merged[zone_class].extend(pool_map.get(zone_class, ()))
    return merged


def _find_library_export(custom_scenery_dir, library_prefix):
    """Return True if any installed scenery package exports the requested prefix."""
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


def _describe_asset_sources(enabled_default, enabled_sfd, enabled_simheaven,
                            sfd_available, simheaven_available):
    """Return a short user-facing summary of building asset source selection."""
    parts = []
    if enabled_default:
        parts.append('default')
    if enabled_sfd:
        parts.append('SFD' if sfd_available else 'SFD unavailable')
    if enabled_simheaven:
        parts.append('simHeaven' if simheaven_available else 'simHeaven unavailable')
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


def _describe_placement_summary(small_count, medium_count, large_count,
                                building_coverage_pct, osm_cell_count,
                                grid_n, spacing_px, spacing_m):
    """Return a user-facing summary for one DDS building-placement pass."""
    return (
        f"placed={small_count + medium_count + large_count:4d}  "
        f"small homes={small_count}  "
        f"medium blocks={medium_count}  "
        f"large buildings={large_count}  "
        f"building cover={building_coverage_pct:.1f}%  "
        f"street-guided cells={osm_cell_count}/{grid_n * grid_n}  "
        f"spacing≈{spacing_m:.1f}m ({spacing_px}px)"
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


def _expand_bounds(bounds_m, pad_m: float):
    xmin, xmax, zmin, zmax = bounds_m
    return (xmin - pad_m, xmax + pad_m, zmin - pad_m, zmax + pad_m)


def _mark_poly(occ_mask: np.ndarray, pts: np.ndarray) -> None:
    """Fill polygon footprint into occ_mask (in-place)."""
    cv2.fillPoly(occ_mask, [np.int32(pts)], 1)


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
    include_default_assets=False,
    include_sfd_assets=True,
    include_simheaven_assets=False,
    **legacy_kwargs,
):
    legacy_min_zone_px = legacy_kwargs.pop('min_zone_px', None)
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
        f"Params: spacing={spacing_m}m  close={close_k}px  open={open_k}px  "
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

    # ── Exclusion data: water, airports, buildings, railways ─────────────────
    # Closed-way polygons (buildings, water bodies, aeroway areas) are rasterized
    # as filled masks per-DDS.  Railway lines are rasterized like roads.
    excl_polys = []           # hard exclusions: water / airports
    existing_bld_polys = []   # occupancy-only blockers: OSM / simHeaven building footprints
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
        existing_bld_polys.extend(bld_polys)
        excl_rails.extend(rail_ways)
        print(
            f"Exclusion buildings: {len(bld_polys)} polys  "
            f"railways: {len(rail_ways)} ways  "
            f"residential areas: {len(residential_polys)} polys"
        )
    else:
        print("Exclusion buildings: unavailable (OSM cache/download failed)")

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

    sfd_assets_available = _find_library_export(custom_scenery_dir, 'sfd_global/')
    simheaven_assets_available = bool(sh_bld_objects)
    enabled_asset_pools = []
    if include_default_assets:
        enabled_asset_pools.append(_build_default_asset_pools())
    if include_sfd_assets and sfd_assets_available:
        enabled_asset_pools.append(_build_sfd_asset_pools(lat + 0.5, lon + 0.5))
    if include_simheaven_assets and simheaven_assets_available:
        enabled_asset_pools.append(_build_simheaven_asset_pools(sh_bld_objects))
    asset_pools = _merge_asset_pools(*enabled_asset_pools)
    asset_sources_label = _describe_asset_sources(
        include_default_assets,
        include_sfd_assets,
        include_simheaven_assets,
        sfd_assets_available,
        simheaven_assets_available,
    )
    print(f"Building assets: {asset_sources_label}")
    if not any(asset_pools.values()):
        print("Building assets: none available for placement")
        return 0
    for pool in asset_pools.values():
        for asset in pool:
            bounds_m = asset.get('bounds_m')
            if bounds_m is None:
                continue
            asset['fit_bounds_m'] = _expand_bounds(bounds_m, FOOTPRINT_PAD_M)
            asset['mark_bounds_m'] = _expand_bounds(
                bounds_m, FOOTPRINT_PAD_M + PLACEMENT_MARGIN_M
            )

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

    TILE_VIZ = 512
    composite = np.zeros((n_rows*TILE_VIZ, n_cols*TILE_VIZ, 3), dtype=np.uint8) if make_viz else None

    placed_objects = []   # list of (lon, lat, heading, obj_path)
    placed_facades = []   # list of (lonlat_ring, facade_path, height_m)
    candidate_grid_cache = {}
    _bld_params = (
        BLD_PLACEMENT_CACHE_VERSION, spacing_m, close_k, open_k, min_zone_m2,
        max_candidates_per_dds,
        PLACE_UNKNOWN_OBJECTS,
        FOOTPRINT_PAD_M,
        ROAD_CENTERLINE_WIDTH_M, ROAD_EXTRA_BUFFER_M,
        ROAD_WIDTH_PX_MIN, ROAD_DILATE_PX_MIN,
        excl_poly_sig, existing_bld_poly_sig, rail_sig, sh_bld_sig,
        residential_poly_sig,
        include_default_assets, include_sfd_assets, include_simheaven_assets,
        sfd_assets_available, simheaven_assets_available,
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
            sp_px = max(3, int(spacing_m / m_per_px))
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
            occ_mask = road_mask.copy()

            if poly_mask is not None and poly_mask.any():
                bld_zone  = bld_zone & (~poly_mask)
                occ_mask  = occ_mask | poly_mask

            if existing_bld_mask is not None and existing_bld_mask.any():
                occ_mask = occ_mask | existing_bld_mask

            if rail_mask.any():
                bld_zone  = bld_zone & (~rail_mask)
                occ_mask  = occ_mask | rail_mask

            if sh_bld_mask is not None and sh_bld_mask.any():
                occ_mask = occ_mask | sh_bld_mask

            DEGREE_TOL = 1e-4
            if lat_n >= lat + 1 - DEGREE_TOL:   occ_mask[:edge_px,  :]  = 1
            if lat_s <= lat     + DEGREE_TOL:   occ_mask[-edge_px:, :]  = 1
            if lon_w <= lon     + DEGREE_TOL:   occ_mask[:,  :edge_px]  = 1
            if lon_e >= lon + 1 - DEGREE_TOL:   occ_mask[:, -edge_px:]  = 1
            _record_elapsed(timings, file_timings, 'mask_apply', _t)

            cell_h = img_h // grid_n
            cell_w = img_w // grid_n

            if np.any(np.isnan(hgrid)):
                hgrid = np.where(np.isnan(hgrid), float(rng.integers(0, 360)), hgrid)

            _t = time.perf_counter()
            min_zone_px = max(1, int(min_zone_m2 / max(m_per_px * m_per_px, 1e-6)))
            n_cc, cc_labels, cc_stats, cc_centroids = cv2.connectedComponentsWithStats(bld_zone, connectivity=8)
            cc_area = cc_stats[:, cv2.CC_STAT_AREA]
            label_class = np.zeros(n_cc, dtype=np.uint8)
            valid_labels = np.flatnonzero((np.arange(n_cc) != 0) & (cc_area >= min_zone_px))
            if valid_labels.size:
                valid_area = cc_area[valid_labels]
                label_class[valid_labels[valid_area < ZONE_SMALL_PX]] = 1
                label_class[valid_labels[(valid_area >= ZONE_SMALL_PX)
                                         & (valid_area < ZONE_LARGE_PX)]] = 2
                label_class[valid_labels[valid_area >= ZONE_LARGE_PX]] = 3
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
            cc_elapsed = _record_elapsed(timings, file_timings, 'connected_components', _t)
            timings['placement'] += cc_elapsed

            _t = time.perf_counter()
            half = sp_px // 2
            pts_this = []
            candidate_grid_key = (img_w, img_h, sp_px)
            base_candidates = candidate_grid_cache.get(candidate_grid_key)
            if base_candidates is None:
                xs = np.arange(half, img_w, sp_px, dtype=np.int32)
                ys = np.arange(half, img_h, sp_px, dtype=np.int32)
                if xs.size and ys.size:
                    grid_x, grid_y = np.meshgrid(xs, ys)
                    base_candidates = (grid_x.ravel(), grid_y.ravel())
                else:
                    base_candidates = (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32))
                candidate_grid_cache[candidate_grid_key] = base_candidates
            base_x, base_y = base_candidates
            if base_x.size and base_y.size:
                n_candidates = base_x.size
                jitter = rng.integers(-half, half + 1, size=(n_candidates, 2),
                                      dtype=np.int32)
                cand_x = np.clip(base_x + jitter[:, 0], 0, img_w - 1)
                cand_y = np.clip(base_y + jitter[:, 1], 0, img_h - 1)
                cand_cls = zone_class[cand_y, cand_x]
                keep = cand_cls != 0
                cand_x = cand_x[keep]
                cand_y = cand_y[keep]
                cand_cls = cand_cls[keep]
                file_counts['candidates'] = int(cand_x.size)
                if cand_x.size:
                    cand_labels = cc_labels[cand_y, cand_x]
                    open_center = occ_mask[cand_y, cand_x] == 0
                    file_counts['initial_center_blocked'] = int(
                        cand_x.size - np.count_nonzero(open_center)
                    )
                    cand_x = cand_x[open_center]
                    cand_y = cand_y[open_center]
                    cand_cls = cand_cls[open_center]
                    cand_labels = cand_labels[open_center]
                if max_candidates_per_dds and cand_x.size > max_candidates_per_dds:
                    cand_x, cand_y, cand_cls, n_dropped = _limit_candidates_by_component(
                        cand_x, cand_y, cand_cls, cand_labels, max_candidates_per_dds, rng
                    )
                    file_counts['candidate_cap'] = int(n_dropped)
            else:
                cand_x = cand_y = cand_cls = ()
            candidate_elapsed = _record_elapsed(timings, file_timings, 'candidate_grid', _t)
            timings['placement'] += candidate_elapsed

            _t = time.perf_counter()
            fit_scratch = np.zeros_like(occ_mask)
            for jx, jy, zone_cls in zip(cand_x, cand_y, cand_cls):
                jx = int(jx)
                jy = int(jy)
                zone_cls = int(zone_cls)
                if occ_mask[jy, jx]:
                    file_counts['dynamic_center_blocked'] = (
                        file_counts.get('dynamic_center_blocked', 0) + 1
                    )
                    continue

                zone_label = int(cc_labels[jy, jx])
                dom_h = float(zone_heading[zone_label]) if (
                    0 <= zone_label < zone_heading.shape[0]
                    and not np.isnan(zone_heading[zone_label])
                ) else float(hgrid[
                    min(grid_n - 1, jy // cell_h),
                    min(grid_n - 1, jx // cell_w),
                ])
                jitter  = rng.uniform(-HEADING_JITTER_DEG, HEADING_JITTER_DEG)
                heading = (dom_h + jitter) % 360.0

                for try_cls in [zone_cls]:
                    if (
                        try_cls == 1 and
                        residential_area_mask is not None and
                        not bool(residential_area_mask[jy, jx])
                    ):
                        continue

                    pool = asset_pools[try_cls]
                    if not pool:
                        continue
                    asset = pool[int(rng.integers(0, len(pool)))]
                    bounds_m = asset.get('bounds_m')
                    if bounds_m is None:
                        n_unknown_skipped += 1
                        continue

                    fit_bounds = asset.get('fit_bounds_m')
                    mark_bounds = asset.get('mark_bounds_m')
                    if fit_bounds is None or mark_bounds is None:
                        fit_bounds = _expand_bounds(bounds_m, FOOTPRINT_PAD_M)
                        mark_bounds = _expand_bounds(
                            bounds_m, FOOTPRINT_PAD_M + PLACEMENT_MARGIN_M
                        )

                    final_h = final_poly = None
                    poly = _footprint_poly(jx, jy, fit_bounds, heading, m_per_px)
                    file_counts['fit_checks'] = file_counts.get('fit_checks', 0) + 1
                    if _poly_fits(occ_mask, poly, fit_scratch):
                        final_h = heading
                        final_poly = _footprint_poly(jx, jy, mark_bounds, heading, m_per_px)
                    else:
                        h90 = (heading + 90.0) % 360.0
                        poly90 = _footprint_poly(jx, jy, fit_bounds, h90, m_per_px)
                        file_counts['fit_checks'] = file_counts.get('fit_checks', 0) + 1
                        if _poly_fits(occ_mask, poly90, fit_scratch):
                            final_h = h90
                            final_poly = _footprint_poly(jx, jy, mark_bounds, h90, m_per_px)

                    if final_h is None:
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
                            _pixel_ring_to_latlon(final_poly, img_w, img_h, lat_n, lat_s, lon_w, lon_e),
                            asset['path'],
                            float(asset.get('height_m', DEFAULT_FACADE_HEIGHT_M.get(try_cls, 8.0))),
                        ))
                    _mark_poly(occ_mask, final_poly)
                    break

            fit_elapsed = _record_elapsed(timings, file_timings, 'fit_loop', _t)
            timings['placement'] += fit_elapsed
            file_counts['placed'] = len(pts_this)

            col = (til_x_left - x_min) // x_step
            row = (til_y_top  - y_min) // y_step
            bld_pct = 100 * np.sum(bld_raw) / (img_w * img_h)
            n_s = sum(1 for p in pts_this if p[3] == 1)
            n_m = sum(1 for p in pts_this if p[3] == 2)
            n_l = sum(1 for p in pts_this if p[3] == 3)
            print(
                f"  [{fi:3d}/{n_files}] {fname}  "
                f"{_describe_placement_summary(n_s, n_m, n_l, bld_pct, n_osm_cells, grid_n, sp_px, sp_px * m_per_px)}"
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
                zc_small = np.array(Image.fromarray((zone_class == 1).astype(np.uint8)*255)
                                    .resize((TILE_VIZ, TILE_VIZ), Image.NEAREST)) > 127
                zc_med   = np.array(Image.fromarray((zone_class == 2).astype(np.uint8)*255)
                                    .resize((TILE_VIZ, TILE_VIZ), Image.NEAREST)) > 127
                zc_large = np.array(Image.fromarray((zone_class == 3).astype(np.uint8)*255)
                                    .resize((TILE_VIZ, TILE_VIZ), Image.NEAREST)) > 127
                for mask, colour in ((zc_small, ZONE_COLOURS[1]),
                                     (zc_med,   ZONE_COLOURS[2]),
                                     (zc_large, ZONE_COLOURS[3])):
                    panel[mask] = (panel[mask] * 0.45 + colour * 0.55).astype(np.uint8)
                pil = Image.fromarray(panel); draw = ImageDraw.Draw(pil)
                dot_colours = {1: (0, 220, 0), 2: (255, 220, 0), 3: (0, 160, 255)}
                for px2, py2, _, cls2 in pts_this:
                    sx, sy = int(px2*scale), int(py2*scale)
                    draw.ellipse([sx-1,sy-1,sx+1,sy+1], fill=dot_colours.get(cls2, (0,220,0)))
                composite[row*TILE_VIZ:(row+1)*TILE_VIZ, col*TILE_VIZ:(col+1)*TILE_VIZ] = np.array(pil)
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
        include_default_assets = args.default_assets,
        include_sfd_assets = args.sfd_assets,
        include_simheaven_assets = args.simheaven_assets,
    )


if __name__ == '__main__':
    main()

