"""Stock YOLO-OBB pre-step: detect static real-world objects (sports fields,
tanks, pools) using the DOTAv1-trained `yolo26x-obb.pt` model,
map them to SFD Global / simHeaven / X-Plane default asset library paths, and
produce DSF placements that the building-overlay facade pass treats as overlap
blockers.

This module is self-contained: it does NOT modify the building overlay.
It exposes `run_stock_yolo_pass()` which returns a `StockYoloResults` dataclass:

    @dataclass
    class StockYoloResults:
        placed_objects:   list[tuple[float, float, float, str]]   # (lon, lat, heading_deg, obj_path)
        placed_facades:   list[tuple[list, str, float]]           # (lonlat_ring, facade_path, height_m)
        placed_draped:    list[tuple[list, str]]                  # (lonlat_ring, pol_path)
        occupied_px_polys: list[np.ndarray]                       # OBB quads in image-pixel space (for masking)

The building-overlay integration site is expected to:
  1. Call `run_stock_yolo_pass(image, lat, lon, lat_bounds, lon_bounds, m_per_px)`.
  2. For each polygon in `occupied_px_polys`, mark it into `static_occ_mask` and
     `building_spacing_mask` so the trained-YOLO facade loop excludes those pixels.
  3. Pass `placed_objects`, `placed_facades`, `placed_draped` to the DSF text writer.

The pre-step runs on every DDS regardless of zoom level. At ZL16 the DOTAv1
model finds little because feature size collapses to a few pixels, but the
inference cost is bounded and skipping by ZL is not worth the conditional.

Only "static" DOTA classes are kept (filters out plane/ship/vehicle/etc.).
Bridge and roundabout are excluded because X-Plane mesh / road network already
renders those.

Asset mapping is in `STOCK_YOLO_ASSET_MAP` — easy to edit. Each entry is
`(placement_type, asset_paths, default_height_m_or_None)`:
  - placement_type ∈ {'object', 'facade'}.
  - asset_paths is a tuple of library-virtual path strings. When more than one
    path is given, the picker selects one deterministically per detection via
    a stable hash of (lat, lon, jx, jy) — same approach as the building
    overlay's context-facade picker. Paths may reference any installed
    library (simHeaven, SFD Global, OpenSceneryX, Handy Objects, MisterX,
    world-models, XPC LuiMor, …): the building overlay filters the map down to
    paths the installed library set actually exports before the pass runs, so
    missing libraries simply shrink the pool instead of producing dangling
    DSF refs.
  - Sports classes (3/6/13) instead pick by real-world size via
    `_SPORT_SIZE_LADDER_M`, and place nothing when no surviving asset matches
    the detected footprint closely enough. Their models are fixed-size, so a
    hash pick would drop a 145 m stadium onto an 80 m pitch.
  - 'object' placements emit DSF `OBJECT path lon lat heading` at the OBB
    center. Heading is taken from the OBB only when the class is marked
    directional in `_USE_OBB_HEADING_PER_CLASS`.
  - 'facade' placements emit DSF `BEGIN_POLYGON path height_m 2` using the OBB
    as the polygon ring. Only the storage tank uses this: a facade wraps a
    wall around the whole OBB, which is right for a tank but produced a
    warehouse ring around every pitch when the sports classes used it.
  - Draped `.pol` polygons are NOT supported by design: the project disabled
    ground polygons globally and the disable should not be re-introduced here.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── Stock YOLO checkpoint location ───────────────────────────────────────────
# Repository root contains yolo26x-obb.pt (DOTAv1-trained Ultralytics OBB).
# Override via environment if needed.
DEFAULT_STOCK_YOLO_CHECKPOINT = os.environ.get(
    "O4_SFR_STOCK_YOLO_CHECKPOINT",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "yolo26x-obb.pt",
    ),
)
DEFAULT_STOCK_YOLO_BATCH = 1
STORAGE_TANK_DOTA_CLASS = 2
STORAGE_TANK_CIRCLE_SEGMENTS = 24


def _is_cuda_oom(exc):
    # Local copy (module is deliberately self-contained; see O4_AI_Overlay).
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text

# ── DOTAv1 class taxonomy and static-class filter ────────────────────────────
DOTA_CLASS_NAMES = {
    0: 'plane',                1: 'ship',
    2: 'storage tank',         3: 'baseball diamond',
    4: 'tennis court',         5: 'basketball court',
    6: 'ground track field',   7: 'harbor',
    8: 'bridge',               9: 'large vehicle',
    10: 'small vehicle',       11: 'helicopter',
    12: 'roundabout',          13: 'soccer ball field',
    14: 'swimming pool',
}

# Classes we KEEP — static / doesn't-move in real life, AND not already rendered
# by X-Plane mesh or road network, AND can be placed without draped polygons.
#   Excluded as moving:           plane (0), ship (1), large vehicle (9),
#                                 small vehicle (10), helicopter (11)
#   Excluded as already-rendered: bridge (8), roundabout (12)
#   Excluded as asset mismatch:   harbor (7) — DOTA 'harbor' mostly fires on
#                                 small-boat marinas, but every available
#                                 crane asset is big-port scale, so the
#                                 placements looked wrong far more often
#                                 than right.
STATIC_DOTA_CLASSES = (2, 3, 4, 5, 6, 13, 14)

# ── Asset mapping ────────────────────────────────────────────────────────────
# Each entry: (placement_type, asset_paths, default_height_m_or_None)
#   placement_type ∈ {'object', 'facade'}
#   asset_paths    = tuple of X-Plane library-virtual paths (resolved by sim
#                    at load time). When more than one entry is given, the
#                    picker chooses one deterministically per detection via a
#                    stable hash of (lat, lon, jx, jy) — EXCEPT for classes
#                    listed in `_SPORT_SIZE_LADDER_M`, where the variant is
#                    chosen by real-world size instead (see below).
#   default_height = facade height in metres (only used for 'facade'); ignored
#                    for 'object' (height is asset-intrinsic).
#
# Sports classes (3/6/13) are deliberately OBJECT placements, never facades.
# A facade extrudes a wall around the whole OBB, which produced a warehouse
# ring around every pitch: simHeaven aliases `grandstand.fac`, `stadium_01.fac`
# and `stadium_02.fac` all to one `objects/facades/stadium.fac` textured with
# `commercial_1.dds`, so the "variant pool" was a single commercial shell.
#
# Paths verified present in the user's installed library set:
#   - simHeaven X-World region libraries (`simheaven/...`)
#   - X-Plane 12 default scenery (`lib/constructions/grandstands/...`,
#     `lib/public_area/sports/...`, `lib/garden/pools/...`).
#   - L_XPC_LuiMor_free (`lib_XPC_LuiMor_free/...`) — pitch objects; the
#     library exports globally (no REGION blocks). When it is not installed
#     the overlay's availability filter drops these paths and the class falls
#     back to whatever else survives, or stops placing entirely.
STOCK_YOLO_ASSET_MAP: dict[int, tuple[str, tuple[str, ...], Optional[float]]] = {
    # Storage tank — vertical extruded structure → use facade with tank texture.
    # simHeaven's `tank.fac` provides a cylindrical-tank visual.
    2:  ('facade', ('simheaven/facades/tank.fac',), 10.0),

    # Baseball diamond — ballpark object sized to the detection. The XP12
    # grandstand bowls carry their own draped infield, so the stands read as
    # real geometry rather than a wall wrapped round the OBB. No plain-object
    # asset exists for a *bare* diamond, so small detections place nothing
    # (the size ladder below rejects them).
    3:  ('object', (
        'lib/constructions/grandstands/baseball_1.obj',
        'lib/constructions/grandstands/baseball_2.obj',
        'lib/constructions/grandstands/baseball_3.obj',
    ), None),

    # Tennis court — small (~24×11 m). Full-court OBJs from Handy Objects /
    # OpenSceneryX / MisterX when those libraries are installed; XP-default
    # net / judge-chair accessories keep the class alive without them.
    4:  ('object', (
        'lib/public_area/sports/tennis_net.obj',
        'lib/public_area/sports/tennis_judge_chair.obj',
        'handyobjects/sports/tennis_court.obj',
        'handyobjects/sports/tennis_court_no_surround.obj',
        'opensceneryx/objects/buildings/recreational/tennis_courts/1.obj',
        'MisterX_Library/Objects/General/Tennis_Court_Double.obj',
    ), None),

    # Basketball court — small (~28×15 m). Full-court OBJs from Handy
    # Objects when installed; XP-default hoop accessories as fallback.
    5:  ('object', (
        'lib/public_area/sports/basketball_hoop.obj',
        'lib/public_area/sports/basketball_hoop_01.obj',
        'lib/public_area/sports/basketball_hoop_02.obj',
        'handyobjects/sports/basketball_court.obj',
        'handyobjects/sports/basketball_court_asphalt.obj',
        'handyobjects/sports/basketball_hoop_with_stand.obj',
    ), None),

    # Ground track field — a running track almost always rings a football
    # pitch, so the pitch object goes in the middle. Surround-free variant:
    # the track itself is the surround. Stadium-scale tracks get the bowl.
    6:  ('object', (
        'lib_XPC_LuiMor_free/comunes/objects/deportes/cancha_football.obj',
        'lib/constructions/grandstands/soccer_1.obj',
    ), None),

    # Soccer ball field — the pitch itself, placed as a pitch object with its
    # surround. Genuinely stadium-scale detections get the grandstand bowl.
    13: ('object', (
        'lib_XPC_LuiMor_free/comunes/objects/deportes/cancha_football_borde.obj',
        'lib/constructions/grandstands/soccer_1.obj',
    ), None),

    # Swimming pool — small OBB, varied size. Use a residential pool OBJ at
    # center; SFD Global has an Australia-flavoured variant too.
    14: ('object', (
        'lib/garden/pools/pool_Small_7x10.obj',
        'SFD_Global/Australia/Pool.obj',
    ), None),
}

# Heading clamp: for 'object' placements, OBB rotation is meaningful only for
# directional assets. Full-surface courts (tennis, basketball), pitches and
# ballparks (3/6/13) and rectangular pools should follow the OBB long axis so
# the 3D asset lines up with the feature painted in the ortho (180° ambiguity
# is harmless — the assets are end-symmetric). Only the tank stays at 0: it is
# a rotationally symmetric facade, so a heading would be meaningless.
_USE_OBB_HEADING_PER_CLASS = {
    2: False, 3: True, 4: True, 5: True,
    6: True, 13: True, 14: True,
}

# Per-class footprint sanity limits (metres), long-side. Filters detections
# whose physical dimensions don't match the class (e.g., a 200 m "storage
# tank" is almost certainly a misclassification). Bounds are deliberately
# generous to cover real-world variation.
_MAX_LONG_SIDE_M = {
    2:  50.0,    # storage tank
    3:  150.0,   # baseball diamond (incl. outfield)
    4:  50.0,    # tennis court
    5:  50.0,    # basketball court
    6:  200.0,   # ground track field (full 400 m track is ~150 m long axis)
    13: 150.0,   # soccer ball field
    14: 60.0,    # swimming pool (Olympic ≤ 50 m)
}
# Per-class minimum (metres). Filters out tiny noise/false positives.
_MIN_LONG_SIDE_M = {
    2:  5.0,
    3:  30.0,
    4:  15.0,
    5:  15.0,
    6:  80.0,
    13: 60.0,
    14: 5.0,
}

# ── Size-matched variant ladders (sports classes) ───────────────────────────
# Sports assets are fixed-size models, so the variant is chosen by real-world
# size rather than by the location hash: an 80 m detection must not receive a
# 145 m stadium bowl. Values are each OBJ's long-side bounding-box extent in
# metres, measured from the shipped models.
#
# When no surviving variant lands within `_SPORT_SIZE_TOLERANCE_FRAC` of the
# detection's long side, the detection places NOTHING. That is deliberate: no
# plain-object asset exists for a bare pitch or diamond at every scale, and a
# wrong-scale model is worse than leaving the pitch the ortho already shows.
_SPORT_SIZE_LADDER_M: dict[int, dict[str, float]] = {
    3: {
        'lib/constructions/grandstands/baseball_1.obj':   77.0,
        'lib/constructions/grandstands/baseball_2.obj':   94.0,
        'lib/constructions/grandstands/baseball_3.obj':  161.6,
    },
    6: {
        'lib_XPC_LuiMor_free/comunes/objects/deportes/cancha_football.obj': 110.0,
        'lib/constructions/grandstands/soccer_1.obj':                       145.2,
    },
    13: {
        'lib_XPC_LuiMor_free/comunes/objects/deportes/cancha_football_borde.obj': 110.0,
        'lib/constructions/grandstands/soccer_1.obj':                             145.2,
    },
}
# Accept an asset whose long side is within ±35% of the detected long side.
_SPORT_SIZE_TOLERANCE_FRAC = 0.35


# ── Variant picker ──────────────────────────────────────────────────────────
def _pick_variant(asset_paths: tuple[str, ...], lat: float, lon: float,
                  jx: float, jy: float) -> str:
    """Deterministically choose one asset path from a pool, keyed by location.

    The same DDS coordinate always picks the same asset on every rerun. Uses
    SHA-1 of the rounded location tuple so the choice is stable across Python
    versions (no hash-seed dependency)."""
    if not asset_paths:
        raise ValueError("asset_paths is empty")
    if len(asset_paths) == 1:
        return asset_paths[0]
    key = f"{round(float(lat), 4)}:{round(float(lon), 4)}:{int(jx)}:{int(jy)}"
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    idx = int.from_bytes(digest[:8], "big") % len(asset_paths)
    return asset_paths[idx]


def _pick_variant_by_size(asset_paths: tuple[str, ...],
                          ladder: dict[str, float],
                          long_m: float) -> Optional[str]:
    """Choose the asset whose real-world long side best matches `long_m`.

    Only paths that survived the caller's availability filter are considered,
    so an uninstalled library simply removes its rungs from the ladder.

    Returns None when nothing lands within `_SPORT_SIZE_TOLERANCE_FRAC`; the
    caller must then skip the detection rather than place a wrong-scale model.
    """
    long_m = float(long_m)
    if long_m <= 0.0:
        return None
    best_path = None
    best_err = None
    for path in asset_paths:
        nominal = ladder.get(path)
        if nominal is None:
            continue
        err = abs(float(nominal) - long_m) / long_m
        if best_err is None or err < best_err:
            best_err, best_path = err, path
    if best_path is None or best_err > _SPORT_SIZE_TOLERANCE_FRAC:
        return None
    return best_path


@dataclass
class StockYoloResults:
    placed_objects:   list = field(default_factory=list)
    placed_facades:   list = field(default_factory=list)
    placed_draped:    list = field(default_factory=list)
    occupied_px_polys: list = field(default_factory=list)
    # Parallel to occupied_px_polys: ('object'|'facade', index-into-that-list)
    # so downstream filters can drop a detection's placement and occupancy
    # quad together without guessing correspondence by centroid.
    occupied_owner:   list = field(default_factory=list)
    counts_by_class:  dict = field(default_factory=dict)
    inference_time_s: float = 0.0
    requested_batch_size: int = 1
    effective_batch_size: int = 1
    batch_fell_back: bool = False


# ── Model loading & inference ────────────────────────────────────────────────

def ensure_stock_yolo_checkpoint(checkpoint: str = DEFAULT_STOCK_YOLO_CHECKPOINT) -> str:
    """Ensure the stock DOTAv1 OBB checkpoint exists on disk, downloading from
    `ultralytics/assets` if it's missing. Returns the absolute path that the
    file is guaranteed to live at on success."""
    target = os.path.abspath(str(checkpoint))
    if os.path.exists(target):
        return target
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    # `attempt_download_asset` resolves the name against `ultralytics/assets`
    # GitHub releases and downloads next to the requested path when missing.
    from ultralytics.utils.downloads import attempt_download_asset
    print(
        f"[stock-yolo] checkpoint missing at {target}; downloading "
        f"{os.path.basename(target)} from ultralytics/assets …",
        flush=True,
    )
    resolved = attempt_download_asset(target)
    resolved_abs = os.path.abspath(str(resolved))
    if resolved_abs != target and os.path.exists(resolved_abs):
        # Ultralytics dropped the asset under SETTINGS['weights_dir'] or CWD;
        # move/copy it to the location our pipeline expects so subsequent
        # runs find it without re-resolving.
        import shutil
        shutil.copy2(resolved_abs, target)
    if not os.path.exists(target):
        raise FileNotFoundError(
            f"Stock-YOLO checkpoint download failed: expected {target}"
        )
    return target


def load_stock_yolo_model(checkpoint: str = DEFAULT_STOCK_YOLO_CHECKPOINT):
    """Load the stock DOTAv1 YOLO-OBB model, fetching the checkpoint from
    `ultralytics/assets` if it's not already on disk. Caller is responsible
    for caching the returned model across DDS files within a tile."""
    path = ensure_stock_yolo_checkpoint(checkpoint)
    try:
        import O4_SFR_Remote as REMOTE
        remote = REMOTE.active_client()
    except Exception:
        remote = None
    if remote is not None:
        return remote.remote_yolo(path)
    from ultralytics import YOLO  # local import keeps module-load light
    return YOLO(path)


def _iter_yolo_crops(image: np.ndarray, stride: int):
    """Tile the image into overlapping crops for stride-based YOLO inference.
    Identical pattern to the trained-YOLO loop in O4_SFR_Building_Overlay."""
    img_h, img_w = image.shape[:2]
    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            crop = image[y:min(y + stride, img_h), x:min(x + stride, img_w)]
            if crop.shape[0] != stride or crop.shape[1] != stride:
                padded = np.zeros((stride, stride, 3), dtype=image.dtype)
                padded[:crop.shape[0], :crop.shape[1]] = crop
                crop = padded
            yield x, y, crop


def _iter_yolo_crop_batches(image: np.ndarray, stride: int, batch_size: int):
    batch = []
    for item in _iter_yolo_crops(image, stride):
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _pixel_quad_to_lonlat(quad_px: np.ndarray, img_w: int, img_h: int,
                          lat_n: float, lat_s: float,
                          lon_w: float, lon_e: float) -> list[tuple[float, float]]:
    """Convert a pixel-space polygon to a closed, CCW lon/lat ring."""
    ring = []
    for px, py in quad_px:
        lon = lon_w + float(px) / float(img_w) * (lon_e - lon_w)
        lat = lat_n - float(py) / float(img_h) * (lat_n - lat_s)
        ring.append((float(lon), float(lat)))
    if _signed_lonlat_ring_area(ring) < 0.0:
        ring.reverse()
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def _signed_lonlat_ring_area(ring: list[tuple[float, float]]) -> float:
    """Return positive area for counter-clockwise lon/lat rings."""
    if len(ring) < 3:
        return 0.0
    pts = ring[:-1] if ring[0] == ring[-1] else ring
    area = 0.0
    for (lon_a, lat_a), (lon_b, lat_b) in zip(pts, pts[1:] + pts[:1]):
        area += lon_a * lat_b - lon_b * lat_a
    return 0.5 * area


def _pixel_circle_polygon(cx: float, cy: float, radius_px: float,
                          segments: int = STORAGE_TANK_CIRCLE_SEGMENTS) -> np.ndarray:
    """Return a regular closed polygon approximating a circle in image pixels."""
    radius_px = max(0.5, float(radius_px))
    segments = max(8, int(segments))
    angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False, dtype=np.float32)
    pts = np.column_stack((
        float(cx) + np.cos(angles) * radius_px,
        float(cy) + np.sin(angles) * radius_px,
    )).astype(np.float32)
    return np.vstack((pts, pts[:1]))


def _circle_radius_within_image(cx: float, cy: float, radius_px: float,
                                img_w: int, img_h: int) -> float:
    """Largest radius keeping a circle centred on (cx, cy) inside the image.

    Clamping a circle's vertices per coordinate (the rectangle bug in circular
    form) flattens whichever arc crosses the border into a straight edge, so a
    storage tank at a texture seam renders as a squashed blob instead of a
    cylinder. Shrinking the radius keeps the tank round and keeps it centred on
    the detection, which also keeps the two halves seen from neighbouring
    textures concentric rather than offset.
    """
    fit = min(
        float(cx),
        float(cy),
        float(max(0, int(img_w) - 1)) - float(cx),
        float(max(0, int(img_h) - 1)) - float(cy),
    )
    return max(0.0, min(float(radius_px), fit))


def _rect_clip_obb_to_image(points, img_w: int, img_h: int, min_extent_px: float = 0.5):
    """Clip an oriented box to the image while keeping it a true rectangle.

    keep in sync with O4_SFR_Building_Overlay._rect_clip_obb_to_image: clamping
    the four corners independently shears a rotated rectangle into an irregular
    quad, and facade rings are written straight from these corners. Trim along
    the box's own axes instead. Returns clipped 4x2 corners in the input
    winding order, or None when nothing of usable size survives.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(4, 2)
    x_max = float(max(0, int(img_w) - 1))
    y_max = float(max(0, int(img_h) - 1))
    if (
        pts[:, 0].min() >= 0.0 and pts[:, 0].max() <= x_max and
        pts[:, 1].min() >= 0.0 and pts[:, 1].max() <= y_max
    ):
        return pts.astype(np.float32)

    center = pts.mean(axis=0)
    u = pts[1] - pts[0]
    u_len = float(np.linalg.norm(u))
    if u_len < 1e-6:
        return None
    u = u / u_len
    v = pts[3] - pts[0]
    v = v - float(np.dot(v, u)) * u
    v_len = float(np.linalg.norm(v))
    if v_len < 1e-6:
        return None
    v = v / v_len

    rel = pts - center
    s = rel @ u
    t = rel @ v
    s_lo, s_hi = float(s.min()), float(s.max())
    t_lo, t_hi = float(t.min()), float(t.max())
    min_extent = float(min_extent_px)

    constraints = (
        (0, 1.0, x_max), (0, -1.0, 0.0),
        (1, 1.0, y_max), (1, -1.0, 0.0),
    )
    for _ in range(16):
        worst = None
        for axis, sign, bound in constraints:
            a = sign * float(u[axis])
            b = sign * float(v[axis])
            reach = (
                sign * float(center[axis]) +
                max(s_lo * a, s_hi * a) +
                max(t_lo * b, t_hi * b)
            )
            violation = reach - bound
            if violation > 1e-9 and (worst is None or violation > worst[0]):
                worst = (violation, a, b)
        if worst is None:
            break
        violation, a, b = worst
        s_span = s_hi - s_lo
        t_span = t_hi - t_lo
        s_shrink = violation / abs(a) if abs(a) > 1e-9 else math.inf
        t_shrink = violation / abs(b) if abs(b) > 1e-9 else math.inf
        if s_shrink > s_span - min_extent:
            s_shrink = math.inf
        if t_shrink > t_span - min_extent:
            t_shrink = math.inf
        if not math.isfinite(s_shrink) and not math.isfinite(t_shrink):
            return None
        if s_shrink * t_span <= t_shrink * s_span:
            if a > 0.0:
                s_hi -= s_shrink
            else:
                s_lo += s_shrink
        else:
            if b > 0.0:
                t_hi -= t_shrink
            else:
                t_lo += t_shrink
    else:
        return None

    s_mid = (s_lo + s_hi) * 0.5
    t_mid = (t_lo + t_hi) * 0.5
    clipped = np.empty((4, 2), dtype=np.float64)
    for idx in range(4):
        s_idx = s_hi if s[idx] > s_mid else s_lo
        t_idx = t_hi if t[idx] > t_mid else t_lo
        clipped[idx] = center + s_idx * u + t_idx * v
    clipped[:, 0] = np.clip(clipped[:, 0], 0.0, x_max)
    clipped[:, 1] = np.clip(clipped[:, 1], 0.0, y_max)
    return clipped.astype(np.float32)


def _quad_geometry(quad_px: np.ndarray, m_per_px: float) -> tuple[float, float, float, float, float]:
    """Return (center_x, center_y, long_side_m, short_side_m, heading_deg)."""
    pts = np.asarray(quad_px, dtype=np.float32).reshape(4, 2)
    cx = float(pts[:, 0].mean())
    cy = float(pts[:, 1].mean())
    edges = np.roll(pts, -1, axis=0) - pts
    edge_lens_px = np.linalg.norm(edges, axis=1)
    long_idx = int(np.argmax(edge_lens_px))
    long_len_px = float(edge_lens_px[long_idx])
    short_len_px = float(edge_lens_px[(long_idx + 1) % 4])
    long_vec = edges[long_idx]
    img_angle = math.degrees(math.atan2(float(long_vec[1]), float(long_vec[0])))
    heading_deg = (90.0 - img_angle) % 360.0
    return cx, cy, long_len_px * m_per_px, short_len_px * m_per_px, heading_deg


def run_stock_yolo_pass(
    image: np.ndarray,
    *,
    model,
    img_w: int, img_h: int,
    lat: int, lon: int,
    lat_n: float, lat_s: float, lon_w: float, lon_e: float,
    m_per_px: float,
    static_occ_mask: Optional[np.ndarray] = None,
    conf: float = 0.25,
    iou: float = 0.5,
    stride: int = 1024,
    imgsz: int = 1024,
    max_det: int = 500,
    device: Optional[str] = None,
    batch_size: int = 1,
    asset_map: Optional[dict[int, tuple[str, tuple[str, ...], Optional[float]]]] = None,
    static_classes: Optional[tuple[int, ...]] = None,
) -> StockYoloResults:
    """Run the stock DOTAv1 YOLO-OBB on `image`, map detections to assets, and
    return placements + occupancy polygons.

    If `static_occ_mask` is provided, detections whose center pixel is already
    occupied are filtered out (cheap sanity check; doesn't replace full
    polygon-overlap checks done by the caller).
    """
    import torch  # local import — heavy

    t0 = time.perf_counter()
    active_asset_map = asset_map if asset_map is not None else STOCK_YOLO_ASSET_MAP
    active_static_classes = (
        tuple(static_classes)
        if static_classes is not None
        else STATIC_DOTA_CLASSES
    )

    def _consume_result(res, r, ox, oy):
        obb = getattr(r, 'obb', None)
        if obb is None:
            return
        corners_t = getattr(obb, 'xyxyxyxy', None)
        if corners_t is None:
            return
        corners = corners_t.detach().cpu().numpy()
        confs = (
            obb.conf.detach().cpu().numpy()
            if obb.conf is not None else np.ones((len(corners),), dtype=float)
        )
        classes = (
            obb.cls.detach().cpu().numpy().astype(int)
            if obb.cls is not None else np.zeros((len(corners),), dtype=int)
        )
        for points, score, cls in zip(corners, confs, classes):
            cls_i = int(cls)
            if cls_i not in active_static_classes:
                continue
            if cls_i not in active_asset_map:
                continue
            quad = np.asarray(points, dtype=np.float32).reshape(4, 2)
            quad[:, 0] += float(ox)
            quad[:, 1] += float(oy)
            # Clip to image bounds, keeping the box rectangular (a per-corner
            # clamp would shear it — see _rect_clip_obb_to_image).
            quad = _rect_clip_obb_to_image(quad, img_w, img_h)
            if quad is None:
                continue
            cx, cy, long_m, short_m, heading_deg = _quad_geometry(quad, m_per_px)
            if not (0 <= cx < img_w and 0 <= cy < img_h):
                continue
            # Class-specific size filter
            if long_m < _MIN_LONG_SIDE_M.get(cls_i, 0.0):
                continue
            if long_m > _MAX_LONG_SIDE_M.get(cls_i, float('inf')):
                continue
            # Optional static-occupancy fast reject
            if static_occ_mask is not None:
                ix = int(round(cx)); iy = int(round(cy))
                if (0 <= iy < static_occ_mask.shape[0]
                        and 0 <= ix < static_occ_mask.shape[1]
                        and static_occ_mask[iy, ix]):
                    continue
            placement_type, asset_paths, default_height_m = active_asset_map[cls_i]
            o_lon = lon_w + cx / float(img_w) * (lon_e - lon_w)
            o_lat = lat_n - cy / float(img_h) * (lat_n - lat_s)
            # Bounds check
            if not (lon <= o_lon < lon + 1 and lat <= o_lat < lat + 1):
                continue
            size_ladder = _SPORT_SIZE_LADDER_M.get(cls_i)
            if size_ladder is not None:
                asset_path = _pick_variant_by_size(
                    asset_paths, size_ladder, long_m
                )
                if asset_path is None:
                    # No sports asset within tolerance of this footprint —
                    # placing a wrong-scale pitch/ballpark is worse than none.
                    continue
            else:
                asset_path = _pick_variant(asset_paths, o_lat, o_lon, cx, cy)
            use_heading = _USE_OBB_HEADING_PER_CLASS.get(cls_i, True)
            placement_heading = float(heading_deg) if use_heading else 0.0
            occupied_poly = quad
            if placement_type == 'object':
                owner = ('object', len(res.placed_objects))
                res.placed_objects.append(
                    (float(o_lon), float(o_lat), placement_heading, asset_path)
                )
            elif placement_type == 'facade':
                facade_poly = quad
                if cls_i == STORAGE_TANK_DOTA_CLASS:
                    radius_px = 0.5 * min(long_m, short_m) / max(float(m_per_px), 1e-6)
                    # Shrink rather than clamp the vertices, which would flatten
                    # the arc crossing the border (see the docstring).
                    radius_px = _circle_radius_within_image(
                        cx, cy, radius_px, img_w, img_h
                    )
                    facade_poly = _pixel_circle_polygon(cx, cy, radius_px)
                    occupied_poly = facade_poly
                ring = _pixel_quad_to_lonlat(facade_poly, img_w, img_h,
                                             lat_n, lat_s, lon_w, lon_e)
                h_m = float(default_height_m if default_height_m else 6.0)
                owner = ('facade', len(res.placed_facades))
                res.placed_facades.append((ring, asset_path, h_m))
            else:
                # Draped .pol polygons are not supported by design.
                raise ValueError(
                    f"Unsupported placement_type {placement_type!r} for DOTA class {cls_i}"
                )
            res.occupied_px_polys.append(occupied_poly.astype(np.int32))
            res.occupied_owner.append(owner)
            res.counts_by_class[cls_i] = res.counts_by_class.get(cls_i, 0) + 1

    def _detect(effective_batch):
        res = StockYoloResults(
            requested_batch_size=batch_size,
            effective_batch_size=effective_batch,
            batch_fell_back=(effective_batch != batch_size),
        )
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
                    for r in results:
                        _consume_result(res, r, ox, oy)
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
                    for (ox, oy), r in zip(offsets, results):
                        _consume_result(res, r, ox, oy)
                    del results, crops, offsets
        return res

    batch_size = max(1, int(batch_size or 1))
    try:
        res = _detect(batch_size)
    except RuntimeError as exc:
        if batch_size > 1 and _is_cuda_oom(exc):
            # Discard the partial batched results and restart from scratch so
            # the fallback output is identical to a pure batch-1 run.
            print(
                f"    Stock YOLO batch={batch_size} hit CUDA OOM; "
                "retrying with batch=1",
                flush=True,
            )
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            res = _detect(1)
        else:
            raise

    # Cumulative across both attempts when OOM fallback triggers — that is the
    # honest wall cost of the pass.
    res.inference_time_s = time.perf_counter() - t0
    return res


# ── DSF text emission helpers ────────────────────────────────────────────────
# Same writing conventions as O4_SFR_Building_Overlay.py:7361+
#   - Facade heights MUST be integers — DSFTool 2.4.0-b1 silently drops .fac
#     polygons with float heights (bug confirmed earlier).
#   - Draped .pol entries use param = 0 (no extrusion, ground-pinned).

def emit_dsf_text_block(handle, results: StockYoloResults,
                        obj_def_offset: int = 0,
                        poly_def_offset: int = 0) -> tuple[int, int]:
    """Append POLYGON_DEF / OBJECT_DEF and instance entries to an open DSF
    text-file handle. Returns (n_poly_defs_added, n_obj_defs_added) so the
    caller can keep its own index counters aligned.

    NOTE: this writes ONLY the stock-YOLO portion. The main building overlay
    is expected to merge defs+instances with its own facade output. A
    self-contained writer is provided here for diagnostic / standalone use.
    """
    # Unique asset paths in order of first appearance
    poly_paths: list[str] = []
    poly_idx: dict[str, int] = {}
    for ring, path, _height in results.placed_facades:
        if path not in poly_idx:
            poly_idx[path] = len(poly_paths) + poly_def_offset
            poly_paths.append(path)
    for ring, path in results.placed_draped:
        if path not in poly_idx:
            poly_idx[path] = len(poly_paths) + poly_def_offset
            poly_paths.append(path)
    obj_paths: list[str] = []
    obj_idx: dict[str, int] = {}
    for _lon, _lat, _h, path in results.placed_objects:
        if path not in obj_idx:
            obj_idx[path] = len(obj_paths) + obj_def_offset
            obj_paths.append(path)

    for p in poly_paths:
        handle.write(f"POLYGON_DEF {p}\n")
    for p in obj_paths:
        handle.write(f"OBJECT_DEF {p}\n")
    handle.write("\n")

    for ring, path, height_m in results.placed_facades:
        idx = poly_idx[path]
        handle.write(f"BEGIN_POLYGON {idx} {int(round(float(height_m)))} 2\n")
        handle.write("BEGIN_WINDING\n")
        for lon_pt, lat_pt in ring:
            handle.write(f"POLYGON_POINT {lon_pt:.7f} {lat_pt:.7f}\n")
        handle.write("END_WINDING\n")
        handle.write("END_POLYGON\n")
    for ring, path in results.placed_draped:
        idx = poly_idx[path]
        # Draped polys use param=0 (no extrusion).
        handle.write(f"BEGIN_POLYGON {idx} 0 2\n")
        handle.write("BEGIN_WINDING\n")
        for lon_pt, lat_pt in ring:
            handle.write(f"POLYGON_POINT {lon_pt:.7f} {lat_pt:.7f}\n")
        handle.write("END_WINDING\n")
        handle.write("END_POLYGON\n")
    for o_lon, o_lat, heading, path in results.placed_objects:
        idx = obj_idx[path]
        handle.write(f"OBJECT {idx} {o_lon:.7f} {o_lat:.7f} {heading:.1f}\n")

    return len(poly_paths), len(obj_paths)


# ── Diagnostic standalone runner ─────────────────────────────────────────────
def _summarize(results: StockYoloResults) -> str:
    lines = [f"[stock-yolo] inference: {results.inference_time_s:.1f}s",
             f"[stock-yolo]   objects: {len(results.placed_objects)}",
             f"[stock-yolo]   facades: {len(results.placed_facades)}",
             f"[stock-yolo]   draped : {len(results.placed_draped)}"]
    if results.counts_by_class:
        lines.append("[stock-yolo] by class:")
        for cls in sorted(results.counts_by_class):
            label = DOTA_CLASS_NAMES.get(cls, str(cls))
            lines.append(f"[stock-yolo]   {label:<22s} {results.counts_by_class[cls]:>5d}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Diagnostic standalone runner for the stock YOLO pre-step.")
    ap.add_argument("image", help="Aerial image (PNG/JPG/DDS).")
    ap.add_argument("--lat", type=int, required=True, help="Tile lat (south edge).")
    ap.add_argument("--lon", type=int, required=True, help="Tile lon (west edge).")
    ap.add_argument("--lat-n", type=float, default=None, help="North bound (defaults lat+1).")
    ap.add_argument("--lat-s", type=float, default=None, help="South bound (defaults lat).")
    ap.add_argument("--lon-w", type=float, default=None, help="West bound (defaults lon).")
    ap.add_argument("--lon-e", type=float, default=None, help="East bound (defaults lon+1).")
    ap.add_argument("--m-per-px", type=float, default=2.4, help="Approx ground sample distance.")
    ap.add_argument("--checkpoint", default=DEFAULT_STOCK_YOLO_CHECKPOINT)
    ap.add_argument("--device", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=DEFAULT_STOCK_YOLO_BATCH)
    ap.add_argument("--out-dsf", default=None, help="Optional path; if given, write a "
                                                     "self-contained .txt DSF.")
    args = ap.parse_args()

    from PIL import Image as PILImage
    pil = PILImage.open(args.image)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    arr = np.asarray(pil, dtype=np.uint8)
    img_h, img_w = arr.shape[:2]
    lat_n = args.lat_n if args.lat_n is not None else (args.lat + 1)
    lat_s = args.lat_s if args.lat_s is not None else args.lat
    lon_w = args.lon_w if args.lon_w is not None else args.lon
    lon_e = args.lon_e if args.lon_e is not None else (args.lon + 1)

    model = load_stock_yolo_model(args.checkpoint)
    res = run_stock_yolo_pass(
        arr, model=model,
        img_w=img_w, img_h=img_h,
        lat=args.lat, lon=args.lon,
        lat_n=lat_n, lat_s=lat_s, lon_w=lon_w, lon_e=lon_e,
        m_per_px=args.m_per_px,
        conf=args.conf,
        device=args.device,
        batch_size=args.batch,
    )
    print(_summarize(res))

    if args.out_dsf:
        with open(args.out_dsf, "w") as f:
            f.write("PROPERTY sim/planet earth\n")
            f.write("PROPERTY sim/overlay 1\n")
            f.write(f"PROPERTY sim/west  {args.lon}\n")
            f.write(f"PROPERTY sim/east  {args.lon+1}\n")
            f.write(f"PROPERTY sim/south {args.lat}\n")
            f.write(f"PROPERTY sim/north {args.lat+1}\n\n")
            emit_dsf_text_block(f, res)
        print(f"[stock-yolo] wrote {args.out_dsf}")
