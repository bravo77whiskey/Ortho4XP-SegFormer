"""Stock YOLO-OBB pre-step: detect static real-world objects (storage
tanks, swimming pools, moored ships) using the DOTAv1-trained `yolo26x-obb.pt`
model, and parked vehicles with a DIOR-trained YOLOv8s-OBB (second pass,
class ids offset by DIOR_CLASS_OFFSET),
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

Only "static" classes are kept (see STATIC_DOTA_CLASSES /
STATIC_DIOR_CLASSES). Bridge and roundabout are excluded because X-Plane mesh /
road network already renders those. Ships and vehicles use size ladders
(_SIZE_LADDERS_M): the OBB long side picks the nearest measured model, and
nothing is placed when no model is within tolerance.

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
  - 'object' placements emit DSF `OBJECT path lon lat heading` at the OBB
    center. Heading is taken from the OBB only when the class is marked
    directional in `_USE_OBB_HEADING_PER_CLASS`.
  - 'facade' placements emit DSF `BEGIN_POLYGON path height_m 2` using the OBB
    as the polygon ring. Only the storage tank uses this: a facade wraps a
    wall around the whole OBB, which is right for a tank and wrong for
    anything flat.
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

import O4_GPU_Budget as GPUB


# ── Stock YOLO checkpoint location ───────────────────────────────────────────
# yolo26x-obb.pt (DOTAv1-trained Ultralytics OBB) lives in the repository root,
# or in a packaged build in _internal/Ortho4XP_Data: overlay_scripts/ is
# replaced on every update, which would throw the downloaded file away.
# Override via environment if needed.
STOCK_YOLO_CHECKPOINT_NAME = "yolo26x-obb.pt"
_MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_stock_yolo_checkpoint(module_root=_MODULE_ROOT):
    data_dir = os.path.join(os.path.dirname(module_root), "Ortho4XP_Data")
    if os.path.basename(module_root) == "overlay_scripts" and os.path.isdir(data_dir):
        return os.path.join(data_dir, STOCK_YOLO_CHECKPOINT_NAME)
    return os.path.join(module_root, STOCK_YOLO_CHECKPOINT_NAME)


DEFAULT_STOCK_YOLO_CHECKPOINT = os.environ.get(
    "O4_STOCK_YOLO_CHECKPOINT", _default_stock_yolo_checkpoint()
)
DEFAULT_STOCK_YOLO_BATCH = 8
STORAGE_TANK_DOTA_CLASS = 2
STORAGE_TANK_CIRCLE_SEGMENTS = 24


def _is_cuda_oom(exc):
    # Local copy: this module deliberately imports no other O4_* module.
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

# Classes we KEEP — things a still photo shows parked or moored, that X-Plane
# mesh / road network does not already render, and that can be placed without
# draped polygons.
#   Kept with placement rules:    ship (1) only when moored (on water, land
#                                 within SHIP_MOORED_M) and not cut by a crop
#                                 seam; it picks a size-matched boat or ship
#                                 from a ladder and places nothing when none
#                                 fits. DOTA's vehicle classes (9, 10) are NOT
#                                 used: they find almost no cars on these
#                                 orthos. Vehicles come from the DIOR model
#                                 (VEHICLE_CLASS, below).
#   Excluded as airport traffic:  plane (0), helicopter (11) — X-Plane puts
#                                 static aircraft on ramp starts itself.
#   Excluded as already-rendered: bridge (8), roundabout (12)
#   Excluded as asset mismatch:   harbor (7) — DOTA 'harbor' mostly fires on
#                                 small-boat marinas, but every available
#                                 crane asset is big-port scale, so the
#                                 placements looked wrong far more often
#                                 than right.
#   Excluded by decision:         every sports class — baseball diamond (3),
#                                 tennis court (4), basketball court (5),
#                                 ground track field (6) and soccer ball
#                                 field (13). The available assets are
#                                 fixed-size models that never matched the
#                                 pitch the ortho already shows, so nothing
#                                 sports-related is placed at all. Do not
#                                 re-add these without being asked.
SHIP_DOTA_CLASS = 1
STATIC_DOTA_CLASSES = (1, 2, 14)

# ── Second stock model: DIOR (20 classes) ────────────────────────────────────
# DOTA's own vehicle classes find almost no cars on Bing/Google orthos (2 on a
# ZL19 Shenzhen texture where the DIOR model finds ~300), so vehicles come
# from a YOLOv8s-OBB trained on DIOR. Its class ids are offset by
# DIOR_CLASS_OFFSET so both models share one int-keyed asset map. Only
# 'vehicle' is used: on the review textures 'windmill' fired on parked
# aircraft and 'chimney' on domes and tanks, and every other DIOR class is
# sports, infrastructure the mesh already draws, or a blacklisted DOTA twin.
DIOR_CLASS_OFFSET = 100
DIOR_CLASS_NAMES = {
    0: 'airplane', 1: 'airport', 2: 'baseballfield', 3: 'basketballcourt',
    4: 'bridge', 5: 'chimney', 6: 'dam', 7: 'Expressway-Service-area',
    8: 'Expressway-toll-station', 9: 'golffield', 10: 'groundtrackfield',
    11: 'harbor', 12: 'overpass', 13: 'ship', 14: 'stadium',
    15: 'storagetank', 16: 'tenniscourt', 17: 'trainstation', 18: 'vehicle',
    19: 'windmill',
}
VEHICLE_CLASS = DIOR_CLASS_OFFSET + 18
STATIC_DIOR_CLASSES = (VEHICLE_CLASS,)
STOCK_CLASS_NAMES = dict(DOTA_CLASS_NAMES)
STOCK_CLASS_NAMES.update({DIOR_CLASS_OFFSET + k: v for k, v in DIOR_CLASS_NAMES.items()})

# Size ladders: (length_m, paths[, kind]) rungs sorted by the measured OBJ8
# length of the models (scripts measured every file, not the names). A
# detection takes the rung nearest its OBB long side, within
# _SIZE_TOLERANCE_FRAC; when no rung is that close nothing is placed, so a
# 90 m ferry never becomes a 30 m fishing boat.
_XP_TRUCKS = "lib/vehicles/static/trucks/"
_XP_BOATS = "lib/ships/powered/"
_SIZE_LADDERS_M = {
    VEHICLE_CLASS: (
        # X-Plane's own parked-car pool (cars and vans, 5.0 / 5.9 m models),
        # then lorries, buses and articulated lorries by length.
        (4.8, ("lib/cars/car_static.obj",)),
        (8.7, ("lib/cars/truck_static.obj",)),
        (10.0, tuple(f"{_XP_TRUCKS}cargo_delivery_0{i}.obj" for i in range(1, 5))
               + tuple(f"{_XP_TRUCKS}cargo_delivery_cabover_0{i}.obj" for i in range(1, 6))),
        (12.8, ("lib/cars/bus_static.obj",)),
        (13.8, tuple(f"{_XP_TRUCKS}semiTrailer_13.6m_0{i}.obj" for i in range(1, 9))),
        (16.5, tuple(f"{_XP_TRUCKS}semiTrailer_truck_cabover_0{i}.obj" for i in range(1, 5))),
        (18.3, tuple(f"{_XP_TRUCKS}semiTrailer_truck_0{i}.obj" for i in range(1, 7))
               + (f"{_XP_TRUCKS}tankerTrailer_truck_01.obj",
                  f"{_XP_TRUCKS}tankerTrailer_truck_02.obj")),
    ),
    SHIP_DOTA_CLASS: (
        # XP12 default static boats (the *_any exports pick a colour variant).
        (3.6, (f"{_XP_BOATS}0_to_5mt/Dinghy_360_any.obj",), "any"),
        (4.0, (f"{_XP_BOATS}0_to_5mt/Dinghy_400_any.obj",), "any"),
        (4.8, (f"{_XP_BOATS}0_to_5mt/Dinghy_470_any.obj",
               f"{_XP_BOATS}5_to_10mt/Whaler_470_any.obj"), "any"),
        (5.2, (f"{_XP_BOATS}5_to_10mt/Whaler_518_any.obj",), "any"),
        (6.5, (f"{_XP_BOATS}5_to_10mt/Whaler_650_any.obj",), "any"),
        (7.8, (f"{_XP_BOATS}5_to_10mt/Runabout_750_any.obj",), "any"),
        (11.1, ("opensceneryx/objects/vehicles/boats_ships/power/4.obj",), "any"),
        (12.5, (f"{_XP_BOATS}over_10mt/Cruiser_1200_any.obj",
                "opensceneryx/objects/vehicles/boats_ships/fishing/1.obj",
                "opensceneryx/objects/vehicles/boats_ships/fishing/3.obj",
                "opensceneryx/objects/vehicles/boats_ships/fishing/4.obj"), "any"),
        (14.5, ("opensceneryx/objects/vehicles/boats_ships/fishing/2.obj",
                "opensceneryx/objects/vehicles/boats_ships/power/3.obj",
                "opensceneryx/objects/vehicles/boats_ships/power/5.obj"), "any"),
        (19.6, ("opensceneryx/objects/vehicles/boats_ships/power/1.obj",
                "opensceneryx/objects/vehicles/boats_ships/power/2.obj"), "any"),
        (20.9, (f"{_XP_BOATS}over_10mt/Cruiser_1900_any.obj",), "any"),
        (30.0, ("MisterX_Library/Vehicles/Boats/Fishing_Boat_1.obj",
                "opensceneryx/objects/vehicles/boats_ships/tour/1.obj",
                "opensceneryx/objects/vehicles/boats_ships/tour/3.obj"), "any"),
        # Harbour tug and a coastal freighter: moored 35-80 m hulls are working
        # boats in most ports (the 65 m XP12 luxury yacht filled whole fishing
        # harbours at Kaohsiung). Both are off-centre, +/-X models: _MODEL_FRAME.
        (35.9, ("CDB-Library/ship/ship_trawler1_senza_scia.obj",), "any"),
        (65.0, ("CDB-Library/ship/ship_pesca.obj",), "any"),
        # XP12 default merchant ships: AG_POINT .agp files, which DSFs place
        # with plain OBJECT_DEF / OBJECT lines (Laminar's Global Airports do).
        (156.0, ("lib/ships/bulk_carriers/BulkCarrier_155A.agp",
                 "lib/ships/bulk_carriers/BulkCarrier_155B.agp"), "cargo"),
        (156.0, ("lib/ships/container_carriers/ContainerCarrier_155A.agp",
                 "lib/ships/container_carriers/ContainerCarrier_155B.agp",
                 "lib/ships/container_carriers/ContainerCarrier_155C.agp"), "container"),
        (166.3, ("MisterX_Library/Vehicles/Ships/Container_Ship_2.obj",), "container"),
        (186.0, ("lib/ships/bulk_carriers/BulkCarrier_190A.agp",
                 "lib/ships/bulk_carriers/BulkCarrier_190C.agp",
                 "lib/ships/lng_carriers/LNGCarrier_190A.agp",
                 "lib/ships/lng_carriers/LNGCarrier_190C.agp",
                 "lib/ships/oil_tankers/OilTanker_183A.agp",
                 "lib/ships/oil_tankers/OilTanker_183B.agp",
                 "lib/ships/oil_tankers/OilTanker_190A.agp"), "cargo"),
        (193.4, ("opensceneryx/objects/vehicles/boats_ships/container/1.obj",
                 "opensceneryx/objects/vehicles/boats_ships/container/2.obj"), "container"),
        (249.3, ("lib/ships/oil_tankers/OilTanker_250A.agp",
                 "lib/ships/oil_tankers/OilTanker_250B.agp"), "cargo"),
        (249.2, ("MisterX_Library/Vehicles/Ships/Container_Ship_1.obj",), "container"),
        (340.0, ("lib/ships/bulk_carriers/BulkCarrier_342B.agp",), "cargo"),
        (402.5, ("lib/ships/container_carriers/ContainerCarrier_399A.agp",
                 "lib/ships/container_carriers/ContainerCarrier_399B.agp",
                 "lib/ships/container_carriers/ContainerCarrier_399C.agp",
                 "lib/ships/container_carriers/ContainerCarrier_399D.agp",
                 "lib/ships/container_carriers/ContainerCarrier_399E.agp",
                 "lib/ships/container_carriers/ContainerCarrier_399F.agp"), "container"),
    ),
}
# Boat models face -Z (bow forward) with the hull centred on the origin,
# except these: path -> (yaw added to the bow heading, hull centre x, z in
# model metres). OpenSceneryX fishing/1-4 taper at +Z (0.1 m wide at the +Z
# tip vs 1.4-1.6 m at -Z). The CDB ships lie along X with the origin near one
# end: the tug's high bow and wheelhouse are at -X (flag at +X), the
# freighter's raked stem at +X; centres from their above-water vertex bounds.
_MODEL_FRAME = {
    **{f"opensceneryx/objects/vehicles/boats_ships/fishing/{i}.obj": (180.0, 0.0, 0.0)
       for i in range(1, 5)},
    "CDB-Library/ship/ship_trawler1_senza_scia.obj": (90.0, 22.37, -5.93),
    "CDB-Library/ship/ship_pesca.obj": (-90.0, 31.71, -0.03),
}


# Beam (m) of ladder models whose width is not ~0.4 / 0.3 / 0.16 of their
# length (small boats / boats / merchant ships); measured from the OBJs.
_BEAM_M = {
    "opensceneryx/objects/vehicles/boats_ships/power/4.obj": 3.3,
    "opensceneryx/objects/vehicles/boats_ships/power/3.obj": 3.5,
    "opensceneryx/objects/vehicles/boats_ships/power/5.obj": 4.4,
    "opensceneryx/objects/vehicles/boats_ships/power/1.obj": 4.9,
    "opensceneryx/objects/vehicles/boats_ships/power/2.obj": 4.8,
    "opensceneryx/objects/vehicles/boats_ships/fishing/2.obj": 4.9,
    "opensceneryx/objects/vehicles/boats_ships/tour/1.obj": 9.7,
    "opensceneryx/objects/vehicles/boats_ships/tour/3.obj": 9.3,
    "MisterX_Library/Vehicles/Boats/Fishing_Boat_1.obj": 9.2,
    "MisterX_Library/Vehicles/Ships/Container_Ship_2.obj": 22.3,
    "MisterX_Library/Vehicles/Ships/Container_Ship_1.obj": 36.6,
    "opensceneryx/objects/vehicles/boats_ships/container/1.obj": 33.9,
    "opensceneryx/objects/vehicles/boats_ships/container/2.obj": 33.9,
}
# Off-centre models: raw OBJ X/Z bounds above the waterline (the convention
# of O4_Building_Overlay.OBJ_FOOTPRINTS).
_RAW_BOUNDS_M = {
    "CDB-Library/ship/ship_trawler1_senza_scia.obj": (4.43, 40.32, -10.52, -1.35),
    "CDB-Library/ship/ship_pesca.obj": (-0.79, 64.21, -5.38, 5.32),
}
_STOCK_BOUNDS = None


def stock_asset_bounds(asset_path):
    """Footprint (xmin, xmax, zmin, zmax) in metres of a sized stock asset
    (vehicles, boats, ships), or None. The building overlay's tile de-dup and
    the review viewer otherwise fall back to a 14 m square: a car next to a
    house tripped the overlap rule and a 65 m ship counted as 14 m."""
    global _STOCK_BOUNDS
    if _STOCK_BOUNDS is None:
        table = {}
        for cls_i, ladder in _SIZE_LADDERS_M.items():
            for rung in ladder:
                length, paths = float(rung[0]), rung[1]
                for path in paths:
                    if path in table:
                        continue
                    if path in _RAW_BOUNDS_M:
                        table[path] = _RAW_BOUNDS_M[path]
                        continue
                    if cls_i == VEHICLE_CLASS:
                        beam = 1.9 if length < 6.0 else 2.55
                    else:
                        beam = _BEAM_M.get(path) or length * (
                            0.4 if length < 10.0 else 0.3 if length < 60.0 else 0.16)
                    table[path] = (-0.5 * beam, 0.5 * beam, -0.5 * length, 0.5 * length)
        _STOCK_BOUNDS = table
    return _STOCK_BOUNDS.get(asset_path)


def model_origin_for_centre(lon, lat, heading_deg, asset_path):
    """(lon, lat, heading) to place asset_path so its hull centre lands on
    (lon, lat) with its bow at heading_deg."""
    yaw, cx, cz = _MODEL_FRAME.get(asset_path, (0.0, 0.0, 0.0))
    h = (float(heading_deg) + yaw) % 360.0
    if cx or cz:
        r = math.radians(h)
        # Model +X is east and +Z south at heading 0; X-Plane turns clockwise.
        east = cx * math.cos(r) - cz * math.sin(r)
        north = -cx * math.sin(r) - cz * math.cos(r)
        lat = lat - north / 111320.0
        lon = lon - east / (111320.0 * max(math.cos(math.radians(lat)), 1e-6))
    return lon, lat, h
# Pixels from an inner crop edge at which a ship box counts as cut by the
# crop seam (crops do not overlap, so a cut hull would be sized too small).
SHIP_SEAM_MARGIN_PX = 3.0
_SIZE_TOLERANCE_FRAC = {
    VEHICLE_CLASS: 0.3,
    SHIP_DOTA_CLASS: 0.25,
}


def _ladder_paths(cls):
    paths = []
    for rung in _SIZE_LADDERS_M.get(cls, ()):
        for path in rung[1]:
            if path not in paths:
                paths.append(path)
    return tuple(paths)


# Per-class confidence floors above the pass-wide ``conf``: weak vehicle hits
# are mostly shed roofs and weak ship hits mostly quayside clutter.
_MIN_CONF = {
    SHIP_DOTA_CLASS: 0.35,
    VEHICLE_CLASS: 0.35,
}
# Coarsest texture (metres per pixel) a class is placed from: below this a
# car is a couple of pixels and the hits are noise.
_MAX_M_PER_PX = {
    VEHICLE_CLASS: 0.7,
}
# A ship counts as moored when land lies within this distance of its box;
# ships further out are under way (X-Plane runs its own traffic there).
SHIP_MOORED_M = 25.0
# Width of the ring around the hull that must be mostly water (when the mesh
# has no water under the centre) for a ship to count as afloat.
SHIP_HULL_RING_M = 5.0

# ── Asset mapping ────────────────────────────────────────────────────────────
# Each entry: (placement_type, asset_paths, default_height_m_or_None)
#   placement_type ∈ {'object', 'facade'}
#   asset_paths    = tuple of X-Plane library-virtual paths (resolved by sim
#                    at load time). When more than one entry is given, the
#                    picker chooses one deterministically per detection via a
#                    stable hash of (lat, lon, jx, jy).
#   default_height = facade height in metres (only used for 'facade'); ignored
#                    for 'object' (height is asset-intrinsic).
#
# Paths verified present in the user's installed library set:
#   - simHeaven X-World region libraries (`simheaven/...`)
#   - X-Plane 12 default scenery (`lib/garden/pools/...`)
#   - SFD Global (`SFD_Global/...`)
STOCK_YOLO_ASSET_MAP: dict[int, tuple[str, tuple[str, ...], Optional[float]]] = {
    # Storage tank — vertical extruded structure → use facade with tank texture.
    # simHeaven's `tank.fac` provides a cylindrical-tank visual.
    2:  ('facade', ('simheaven/facades/tank.fac',), 10.0),

    # Swimming pool — small OBB, varied size. Use a residential pool OBJ at
    # center; SFD Global has an Australia-flavoured variant too.
    14: ('object', (
        'lib/garden/pools/pool_Small_7x10.obj',
        'SFD_Global/Australia/Pool.obj',
    ), None),

    # Ships and vehicles: every rung of the class's size ladder, so the
    # availability filter can drop uninstalled libraries; the picker then
    # chooses the rung by size (see _SIZE_LADDERS_M).
    SHIP_DOTA_CLASS: ('object', _ladder_paths(SHIP_DOTA_CLASS), None),
    VEHICLE_CLASS: ('object', _ladder_paths(VEHICLE_CLASS), None),
}

# Heading clamp: for 'object' placements, OBB rotation is meaningful only for
# directional assets. Rectangular pools follow the OBB long axis so the 3D
# asset lines up with the feature painted in the ortho (180° ambiguity is
# harmless — the assets are end-symmetric). The tank stays at 0: it is a
# rotationally symmetric facade, so a heading would be meaningless.
_USE_OBB_HEADING_PER_CLASS = {
    1: True, 2: False, 14: True, VEHICLE_CLASS: True,
}

# Per-class footprint sanity limits (metres), long-side. Filters detections
# whose physical dimensions don't match the class (e.g., a 200 m "storage
# tank" is almost certainly a misclassification). Bounds are deliberately
# generous to cover real-world variation.
_MAX_LONG_SIDE_M = {
    1:  420.0,   # ship (largest container carriers ~400 m)
    2:  50.0,    # storage tank
    VEHICLE_CLASS: 22.0,  # cars to articulated lorries (~18.5 m)
    14: 60.0,    # swimming pool (Olympic ≤ 50 m)
}
# Per-class minimum (metres). Filters out tiny noise/false positives.
_MIN_LONG_SIDE_M = {
    1:  3.0,
    2:  5.0,
    VEHICLE_CLASS: 3.0,
    14: 5.0,
}


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


def _pick_sized_variant(cls: int, long_m: float, available_paths, lat: float,
                        lon: float, jx: float, jy: float,
                        kind: Optional[str] = None) -> Optional[str]:
    """Asset for a size-laddered class, or None when no rung fits.

    Picks the rung whose length is nearest ``long_m`` (by ratio) among rungs
    within the class tolerance that still have an installed path. ``kind``
    ('container' / 'cargo') restricts ship rungs to that kind first and falls
    back to any kind, since the deck test is only a colour heuristic.
    """
    ladder = _SIZE_LADDERS_M.get(cls)
    if not ladder or long_m <= 0.0:
        return None
    tol = float(_SIZE_TOLERANCE_FRAC.get(cls, 0.25))
    available = set(available_paths)

    def _best(want_kind):
        best = None
        for rung in ladder:
            length, paths = float(rung[0]), rung[1]
            rung_kind = rung[2] if len(rung) > 2 else "any"
            if want_kind is not None and rung_kind not in ("any", want_kind):
                continue
            if abs(long_m / length - 1.0) > tol:
                continue
            kept = tuple(p for p in paths if p in available)
            if not kept:
                continue
            err = abs(math.log(long_m / length))
            if best is None or err < best[0]:
                best = (err, kept)
        return best

    best = _best(kind) if kind is not None else None
    if best is None:
        best = _best(None)
    if best is None:
        return None
    return _pick_variant(best[1], lat, lon, jx, jy)


# ── Ship placement rules ─────────────────────────────────────────────────────
def _quad_axes(quad):
    """(centre, unit long axis, unit short axis, half long, half short) in px."""
    pts = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    centre = pts.mean(axis=0)
    edges = np.roll(pts, -1, axis=0) - pts
    lens = np.linalg.norm(edges, axis=1)
    i = int(np.argmax(lens))
    u = edges[i] / max(float(lens[i]), 1e-9)
    v = np.array([-u[1], u[0]])
    return centre, u, v, 0.5 * float(lens[i]), 0.5 * float(lens[(i + 1) % 4])


def _local_window(quad, margin_px, img_w, img_h):
    pts = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    x0 = max(0, int(math.floor(pts[:, 0].min() - margin_px)))
    y0 = max(0, int(math.floor(pts[:, 1].min() - margin_px)))
    x1 = min(int(img_w), int(math.ceil(pts[:, 0].max() + margin_px)) + 1)
    y1 = min(int(img_h), int(math.ceil(pts[:, 1].max() + margin_px)) + 1)
    return x0, y0, x1, y1


def ship_surroundings(image, water, mesh_water, quad, m_per_px):
    """Judge a ship detection against the water around it.

    ``water`` marks open water (mesh and SegFormer agree), ``mesh_water`` the
    mesh's water. Returns ``(on_water, moored, bow_vec)``: on_water when the
    mesh has water under the centre or most of the ring around the box is
    water; moored when land lies within SHIP_MOORED_M of the box; bow_vec a
    unit image vector pointing at the bow, or None when the ends look alike.
    """
    import cv2

    img_h, img_w = water.shape[:2]
    centre, u, v, hl, hs = _quad_axes(quad)
    ring_px = max(2.0, SHIP_MOORED_M / max(float(m_per_px), 1e-6))
    x0, y0, x1, y1 = _local_window(quad, ring_px + 1, img_w, img_h)
    if x1 <= x0 or y1 <= y0:
        return False, False, None
    off = np.array([x0, y0], dtype=np.float64)

    def _box(half_long, half_short):
        corners = np.array([
            centre + sl * half_long * u + ss * half_short * v
            for sl, ss in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]) - off
        m = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillPoly(m, [np.round(corners).astype(np.int32)], 1)
        return m.astype(bool)

    inner = _box(hl, hs)
    ring = _box(hl + ring_px, hs + ring_px) & ~inner
    # On-water is judged from a thin ring hugging the hull; the wide mooring
    # ring would call a boat on a quay "on water" from the harbour nearby.
    near_px = min(ring_px, max(3.0, SHIP_HULL_RING_M / max(float(m_per_px), 1e-6)))
    near = _box(hl + near_px, hs + near_px) & ~inner
    local_water = water[y0:y1, x0:x1].astype(bool)
    n_ring = int(ring.sum())
    n_near = int(near.sum())
    if n_ring == 0 or n_near == 0:
        return False, False, None
    ring_water = float((local_water & ring).sum()) / n_ring
    near_water = float((local_water & near).sum()) / n_near
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    centre_mesh_water = bool(
        mesh_water is not None and 0 <= cy < img_h and 0 <= cx < img_w
        and mesh_water[cy, cx]
    )
    on_water = centre_mesh_water or near_water >= 0.5
    moored = (1.0 - ring_water) >= 0.03

    bow_vec = None
    if hl * 2.0 >= 12.0 and image is not None:
        pix = image[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
        water_px = (local_water & ring).reshape(-1)
        if water_px.sum() >= 8:
            water_rgb = np.median(pix[water_px], axis=0)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            rel = np.stack([xx - centre[0], yy - centre[1]], axis=-1).reshape(-1, 2)
            t = rel @ u
            s = np.abs(rel @ v)
            hull = np.linalg.norm(pix - water_rgb, axis=1) > 40.0
            fracs = []
            for sign in (1.0, -1.0):
                end = (sign * t >= 0.6 * hl) & (sign * t <= hl) & (s <= hs)
                fracs.append(float(hull[end].mean()) if end.any() else math.nan)
            if all(math.isfinite(f) for f in fracs) and abs(fracs[0] - fracs[1]) >= 0.12:
                # The bow tapers, so its end shows more water inside the box.
                bow_vec = u if fracs[0] < fracs[1] else -u
    return on_water, moored, bow_vec


def touches_inner_crop_seam(points, ox, oy, crop_w, crop_h, img_w, img_h,
                            margin_px=SHIP_SEAM_MARGIN_PX):
    """True when a crop-local box reaches a crop edge that is not the image
    edge, i.e. the object continues in the neighbouring crop."""
    pts = np.asarray(points, dtype=np.float64).reshape(4, 2)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    return bool(
        (ox > 0 and x_min <= margin_px)
        or (oy > 0 and y_min <= margin_px)
        or (ox + crop_w < img_w and x_max >= crop_w - margin_px)
        or (oy + crop_h < img_h and y_max >= crop_h - margin_px)
    )


def deck_looks_like_containers(image, quad):
    """True when the deck is a patchwork of saturated colours (container
    stacks) rather than a bulk/tanker deck of one or two colours."""
    import cv2

    if image is None:
        return False
    img_h, img_w = image.shape[:2]
    centre, u, v, hl, hs = _quad_axes(quad)
    x0, y0, x1, y1 = _local_window(quad, 0, img_w, img_h)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return False
    corners = np.array([
        centre + sl * 0.75 * hl * u + ss * 0.7 * hs * v
        for sl, ss in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]) - np.array([x0, y0])
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(corners).astype(np.int32)], 1)
    if mask.sum() < 20:
        return False
    hsv = cv2.cvtColor(np.ascontiguousarray(image[y0:y1, x0:x1]), cv2.COLOR_RGB2HSV)
    sel = mask.astype(bool)
    h = hsv[..., 0][sel].astype(np.int32)
    s = hsv[..., 1][sel] / 255.0
    val = hsv[..., 2][sel] / 255.0
    colourful = (s > 0.35) & (val > 0.25)
    if colourful.mean() < 0.2:
        return False
    hist = np.bincount(h[colourful] * 12 // 180, minlength=12) / max(1, int(colourful.sum()))
    return int((hist >= 0.06).sum()) >= 3


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
    # Detections dropped by a placement rule, by reason (diagnostics only).
    rejected:         dict = field(default_factory=dict)
    inference_time_s: float = 0.0
    requested_batch_size: int = 1
    effective_batch_size: int = 1
    batch_fell_back: bool = False


# ── Model loading & inference ────────────────────────────────────────────────

def merge_stock_results(into: StockYoloResults, other: Optional[StockYoloResults]) -> StockYoloResults:
    """Append ``other``'s placements to ``into`` (owner indices re-based)."""
    if other is None:
        return into
    base = {
        'object': len(into.placed_objects),
        'facade': len(into.placed_facades),
        'draped': len(into.placed_draped),
    }
    into.placed_objects.extend(other.placed_objects)
    into.placed_facades.extend(other.placed_facades)
    into.placed_draped.extend(other.placed_draped)
    into.occupied_px_polys.extend(other.occupied_px_polys)
    into.occupied_owner.extend((kind, base[kind] + idx) for kind, idx in other.occupied_owner)
    for table, src in ((into.counts_by_class, other.counts_by_class),
                       (into.rejected, other.rejected)):
        for key, n in src.items():
            table[key] = table.get(key, 0) + n
    into.inference_time_s += other.inference_time_s
    return into


def ensure_stock_yolo_checkpoint(checkpoint: str = DEFAULT_STOCK_YOLO_CHECKPOINT) -> str:
    """Ensure the stock DOTAv1 OBB checkpoint exists on disk, downloading from
    `ultralytics/assets` if it's missing. Returns the absolute path that the
    file is guaranteed to live at on success."""
    target = os.path.abspath(str(checkpoint))
    if os.path.exists(target):
        return target
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    # Packaged builds before the move kept the download in overlay_scripts/.
    legacy = os.path.join(_MODULE_ROOT, os.path.basename(target))
    if (os.path.basename(_MODULE_ROOT) == "overlay_scripts"
            and os.path.abspath(legacy) != target and os.path.exists(legacy)):
        import shutil
        shutil.move(legacy, target)
        print(f"[stock-yolo] moved checkpoint {legacy} -> {target}", flush=True)
        return target
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
        import O4_Remote_GPU as REMOTE
        remote = REMOTE.active_client()
    except Exception:
        remote = None
    if remote is not None:
        return remote.remote_yolo(path)
    from ultralytics import YOLO  # local import keeps module-load light
    return YOLO(path)


# ── DIOR vehicle model ──────────────────────────────────────────────────────
# A community YOLOv8s-OBB fine-tuned on DIOR (Hugging Face
# pauhidalgoo/yolov8-DIOR, pinned commit). Its .pt is a pickle written by a
# third party, so it is never unpickled as-is: rebuild_checkpoint_safely()
# reads only its tensors (every class it references becomes an inert stub) and
# loads them into a model built by the installed ultralytics, then saves that.
DIOR_CHECKPOINT_NAME = "dior-yolov8s-obb.pt"
DIOR_SOURCE_URL = (
    "https://huggingface.co/pauhidalgoo/yolov8-DIOR/resolve/"
    "b96bc27ef17251c87659020270c7f7abad640381/DIOR_yolov8s_backbone.pt"
)
DIOR_SOURCE_SHA256 = "065c2508eea1f99d172bbe83f11fe14ff0b19ec2ba8da9e62db18bcc2732fa2e"
DEFAULT_DIOR_CHECKPOINT = os.environ.get(
    "O4_STOCK_DIOR_CHECKPOINT",
    os.path.join(os.path.dirname(DEFAULT_STOCK_YOLO_CHECKPOINT), DIOR_CHECKPOINT_NAME),
)
DIOR_IMGSZ = 1024
DIOR_STRIDE = 1024


def stock_dior_enabled() -> bool:
    """O4_STOCK_DIOR=0 turns the DIOR vehicle pass off."""
    return os.environ.get("O4_STOCK_DIOR", "1").strip().lower() not in ("0", "false", "no", "off")


def rebuild_checkpoint_safely(src: str, dst: str) -> dict:
    """Rebuild an Ultralytics OBB checkpoint without executing its pickle.

    Only torch's tensor/storage rebuild helpers and plain containers are
    resolved; every other global becomes an inert stub. The tensors are then
    loaded strictly into an OBBModel built from the checkpoint's own layer
    config by the installed ultralytics. Returns the class-name dict.
    """
    import collections
    import pickle
    import types
    import torch

    allowed = {
        ("torch._utils", "_rebuild_tensor_v2"), ("torch._utils", "_rebuild_parameter"),
        ("torch._utils", "_rebuild_parameter_with_state"),
        ("collections", "OrderedDict"), ("builtins", "set"), ("__builtin__", "set"),
        ("torch", "Size"), ("torch", "device"),
    }
    storages = {
        "FloatStorage", "HalfStorage", "DoubleStorage", "LongStorage",
        "IntStorage", "BoolStorage", "ByteStorage", "BFloat16Storage",
    }

    class _Stub:
        def __new__(cls, *args, **kwargs):
            return object.__new__(cls)

        def __init__(self, *args, **kwargs):
            pass

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)

        def __call__(self, *args, **kwargs):
            return _Stub()

    stubs = {}

    def _reconstructor(cls, base, state):
        if isinstance(cls, type) and issubclass(cls, _Stub):
            return object.__new__(cls)
        return _Stub()

    class _SafeUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) in allowed or (module == "torch" and name in storages):
                return super().find_class(module, name)
            if (module, name) == ("copyreg", "_reconstructor"):
                return _reconstructor
            key = f"{module}.{name}"
            if key not in stubs:
                stubs[key] = type(name, (_Stub,), {})
            return stubs[key]

    safe_pickle = types.ModuleType("o4_safe_pickle")
    safe_pickle.Unpickler = _SafeUnpickler
    safe_pickle.load = pickle.load

    ckpt = torch.load(src, map_location="cpu", pickle_module=safe_pickle, weights_only=False)
    tree = ckpt.get("ema") or ckpt.get("model")
    layer_cfg = dict(tree.__dict__.get("yaml") or {})
    names = {int(k): str(v) for k, v in dict(tree.__dict__.get("names") or {}).items()}
    if not layer_cfg or not names:
        raise ValueError(f"{src}: no model config / class names")

    def _tensors(node, prefix=""):
        out = collections.OrderedDict()
        state = node.__dict__
        for group in ("_parameters", "_buffers"):
            for key, value in (state.get(group) or {}).items():
                if value is not None:
                    out[prefix + key] = value
        for key, child in (state.get("_modules") or {}).items():
            if child is not None:
                out.update(_tensors(child, prefix + key + "."))
        return out

    from ultralytics.nn.tasks import OBBModel

    net = OBBModel(cfg=layer_cfg, nc=len(names), verbose=False)
    net.load_state_dict({k: v.float() for k, v in _tensors(tree).items()}, strict=True)
    net.names = names
    net.args = {"imgsz": DIOR_IMGSZ, "task": "obb"}
    net = net.half().eval()
    tmp = f"{dst}.{os.getpid()}.tmp"
    torch.save({"model": net, "train_args": net.args, "date": None, "version": "o4-rebuilt"}, tmp)
    os.replace(tmp, dst)
    return names


def ensure_dior_checkpoint(checkpoint: str = DEFAULT_DIOR_CHECKPOINT) -> str:
    """Download (pinned, hash-checked) and safely rebuild the DIOR model once."""
    import hashlib as _hashlib
    import urllib.request

    target = os.path.abspath(str(checkpoint))
    if os.path.exists(target):
        return target
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    # Per-process name: parallel overlay workers may all find the model missing.
    src = f"{target}.{os.getpid()}.download"
    print(f"[stock-yolo] DIOR vehicle model missing; downloading {DIOR_SOURCE_URL} …", flush=True)
    digest = _hashlib.sha256()
    with urllib.request.urlopen(DIOR_SOURCE_URL, timeout=60) as resp, open(src, "wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            out.write(chunk)
    try:
        if digest.hexdigest() != DIOR_SOURCE_SHA256:
            raise ValueError("DIOR checkpoint download does not match its pinned SHA-256")
        rebuild_checkpoint_safely(src, target)
    finally:
        try:
            os.remove(src)
        except OSError:
            pass
    return target


def load_dior_model(checkpoint: str = DEFAULT_DIOR_CHECKPOINT):
    """The DIOR vehicle model (local GPU/CPU; never sent to a remote GPU)."""
    path = ensure_dior_checkpoint(checkpoint)
    from ultralytics import YOLO
    return YOLO(path, task="obb")


def yolo_predict_options() -> dict:
    """Extra ultralytics predict() options for both YOLO passes.

    Inference runs in fp16; O4_BLD_YOLO_HALF=0 runs it in fp32.
    """
    if os.environ.get("O4_BLD_YOLO_HALF", "").strip().lower() not in ("0", "false", "no", "off"):
        return {"half": True}
    return {}


def _crop_skipped(x: int, y: int, stride: int, img_w: int, img_h: int, skip_rects_px) -> bool:
    """True when the crop window lies entirely inside one skip rect."""
    cx1 = min(x + stride, img_w)
    cy1 = min(y + stride, img_h)
    for x0, y0, x1, y1 in skip_rects_px:
        if x >= x0 and y >= y0 and cx1 <= x1 and cy1 <= y1:
            return True
    return False


def _iter_yolo_crops(image: np.ndarray, stride: int, skip_rects_px=()):
    """Tile the image into overlapping crops for stride-based YOLO inference.
    Identical pattern to the trained-YOLO loop in O4_Building_Overlay."""
    img_h, img_w = image.shape[:2]
    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            if skip_rects_px and _crop_skipped(x, y, stride, img_w, img_h, skip_rects_px):
                continue
            crop = image[y:min(y + stride, img_h), x:min(x + stride, img_w)]
            if crop.shape[0] != stride or crop.shape[1] != stride:
                padded = np.zeros((stride, stride, 3), dtype=image.dtype)
                padded[:crop.shape[0], :crop.shape[1]] = crop
                crop = padded
            yield x, y, crop


def _iter_yolo_crop_batches(image: np.ndarray, stride: int, batch_size: int, skip_rects_px=()):
    batch = []
    for item in _iter_yolo_crops(image, stride, skip_rects_px):
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

    keep in sync with O4_Building_Overlay._rect_clip_obb_to_image: clamping
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
    return (cx, cy, long_len_px * m_per_px, short_len_px * m_per_px,
            _image_vector_heading(long_vec[0], long_vec[1]))


def _image_vector_heading(dx: float, dy: float) -> float:
    """Compass heading (clockwise from north) of an image-pixel vector.

    Image pixels run +Y down, so east = dx and north = -dy, giving
    atan2(dy, dx) + 90. keep in sync with O4_Building_Overlay's
    ``heading = img_angle + 90``: the old ``90 - img_angle`` here was the
    mirror image, which only agreed on axis-aligned boxes.
    """
    return (math.degrees(math.atan2(float(dy), float(dx))) + 90.0) % 360.0


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
    skip_rects_px=(),
    class_offset: int = 0,
    mesh_water_mask: Optional[np.ndarray] = None,
    class_map: Optional[np.ndarray] = None,
    water_class: Optional[int] = None,
    road_class: Optional[int] = None,
) -> StockYoloResults:
    """Run a stock YOLO-OBB (DOTAv1, or DIOR with ``class_offset`` =
    DIOR_CLASS_OFFSET) on `image`, map detections to assets, and return
    placements + occupancy polygons.

    Crops lying wholly inside one of `skip_rects_px` (x0, y0, x1, y1) are not
    inferred.

    If `static_occ_mask` is provided, detections whose center pixel is already
    occupied are filtered out (cheap sanity check; doesn't replace full
    polygon-overlap checks done by the caller).

    ``mesh_water_mask`` and the SegFormer ``class_map`` (with its
    ``water_class`` / ``road_class`` ids), when given, gate ships (moored on
    water only) and vehicles (never on water or a SegFormer road). Without
    them ships are dropped, since nothing proves they are moored.
    """
    import torch  # local import — heavy

    t0 = time.perf_counter()
    active_asset_map = asset_map if asset_map is not None else STOCK_YOLO_ASSET_MAP
    active_static_classes = (
        tuple(static_classes)
        if static_classes is not None
        else (STATIC_DIOR_CLASSES if class_offset == DIOR_CLASS_OFFSET else STATIC_DOTA_CLASSES)
    )
    seg_water = seg_road = None
    if class_map is not None and class_map.shape[:2] == (img_h, img_w):
        if water_class is not None:
            seg_water = class_map == int(water_class)
        if road_class is not None:
            seg_road = class_map == int(road_class)
    if mesh_water_mask is not None and mesh_water_mask.shape[:2] != (img_h, img_w):
        mesh_water_mask = None
    open_water = None
    if mesh_water_mask is not None or seg_water is not None:
        # Open water = where the mesh and SegFormer agree (piers and pontoons
        # the mesh lacks are land to SegFormer); either alone when that is
        # all there is.
        if mesh_water_mask is not None and seg_water is not None:
            open_water = mesh_water_mask.astype(bool) & seg_water
        else:
            open_water = (mesh_water_mask if mesh_water_mask is not None else seg_water).astype(bool)

    def _reject(res, reason):
        res.rejected[reason] = res.rejected.get(reason, 0) + 1

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
        orig_shape = getattr(r, 'orig_shape', None)
        crop_h, crop_w = (
            (int(orig_shape[0]), int(orig_shape[1])) if orig_shape is not None
            else (min(int(stride), img_h - int(oy)), min(int(stride), img_w - int(ox)))
        )
        for points, score, cls in zip(corners, confs, classes):
            cls_i = int(cls) + int(class_offset)
            if cls_i not in active_static_classes:
                continue
            if cls_i not in active_asset_map:
                continue
            if float(score) < _MIN_CONF.get(cls_i, 0.0):
                continue
            if float(m_per_px) > _MAX_M_PER_PX.get(cls_i, float('inf')):
                continue
            if cls_i == SHIP_DOTA_CLASS and touches_inner_crop_seam(
                    points, ox, oy, crop_w, crop_h, img_w, img_h):
                _reject(res, 'ship_cut_by_crop')
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
            ix = min(max(int(round(cx)), 0), img_w - 1)
            iy = min(max(int(round(cy)), 0), img_h - 1)
            kind = None
            if cls_i == SHIP_DOTA_CLASS:
                if open_water is None:
                    _reject(res, 'ship_no_water_data')
                    continue
                on_water, moored, bow_vec = ship_surroundings(
                    image, open_water, mesh_water_mask, quad, m_per_px
                )
                if not on_water:
                    _reject(res, 'ship_on_land')
                    continue
                if not moored:
                    _reject(res, 'ship_under_way')
                    continue
                if bow_vec is not None:
                    heading_deg = _image_vector_heading(bow_vec[0], bow_vec[1])
                if long_m >= 120.0:
                    kind = 'container' if deck_looks_like_containers(image, quad) else 'cargo'
            elif cls_i == VEHICLE_CLASS:
                if (mesh_water_mask is not None and mesh_water_mask[iy, ix]) or (
                        seg_water is not None and seg_water[iy, ix]):
                    _reject(res, 'vehicle_on_water')
                    continue
                if seg_road is not None and seg_road[iy, ix]:
                    _reject(res, 'vehicle_on_road')
                    continue
            if cls_i in _SIZE_LADDERS_M:
                asset_path = _pick_sized_variant(
                    cls_i, long_m, asset_paths, o_lat, o_lon, cx, cy, kind=kind
                )
                if asset_path is None:
                    _reject(res, 'no_asset_fits')
                    continue
            else:
                asset_path = _pick_variant(asset_paths, o_lat, o_lon, cx, cy)
            use_heading = _USE_OBB_HEADING_PER_CLASS.get(cls_i, True)
            placement_heading = float(heading_deg) if use_heading else 0.0
            occupied_poly = quad
            if placement_type == 'object':
                p_lon, p_lat = o_lon, o_lat
                if use_heading:
                    p_lon, p_lat, placement_heading = model_origin_for_centre(
                        o_lon, o_lat, placement_heading, asset_path)
                owner = ('object', len(res.placed_objects))
                res.placed_objects.append(
                    (float(p_lon), float(p_lat), placement_heading, asset_path)
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

    predict_options = yolo_predict_options()

    def _detect(effective_batch):
        res = StockYoloResults(
            requested_batch_size=batch_size,
            effective_batch_size=effective_batch,
            batch_fell_back=(effective_batch != batch_size),
        )
        if effective_batch <= 1:
            for ox, oy, crop in _iter_yolo_crops(image, int(stride), skip_rects_px):
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
                        **predict_options,
                    )
                    for r in results:
                        _consume_result(res, r, ox, oy)
                    del results
                del crop_bgr
        else:
            for batch in _iter_yolo_crop_batches(image, int(stride), effective_batch, skip_rects_px):
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
                        **predict_options,
                        batch=effective_batch,
                    )
                    for (ox, oy), r in zip(offsets, results):
                        _consume_result(res, r, ox, oy)
                    del results, crops, offsets
        return res

    def _detect_admitted(effective_batch):
        if str(device) == "cpu" or not torch.cuda.is_available():
            return _detect(effective_batch)
        return GPUB.call(f"stock-yolo b{effective_batch}", _detect, effective_batch)

    batch_size = max(1, int(batch_size or 1))
    try:
        res = _detect_admitted(batch_size)
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
            res = _detect_admitted(1)
        else:
            raise

    # Cumulative across both attempts when OOM fallback triggers — that is the
    # honest wall cost of the pass.
    res.inference_time_s = time.perf_counter() - t0
    return res


# ── DSF text emission helpers ────────────────────────────────────────────────
# Same writing conventions as O4_Building_Overlay.py:7361+
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
