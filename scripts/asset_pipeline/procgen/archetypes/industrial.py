"""Industrial archetypes: low-pitch warehouse and big-box retail.

warehouse: concrete walls with roller doors at grade, a 6-degree ridged
metal roof along the long axis, rooftop vents and a small office annex.
bigbox: windowless block with a storefront entrance band on the front
facade only, tall parapet and large rooftop HVAC plant.
"""

from __future__ import annotations

import math
import random

from .common import MeshSpec, StripUV, box, parapet_flat_roof, \
    set_shell_meta, validate_spec, wall_quad
from .styles import profile_for_asset, style_for_profile, weighted_choice

FLOOR_H = 3.2
WAREHOUSE_PITCH_DEG = 6.0
FASCIA_DROP_M = 0.25


def build_warehouse(length_m: float, width_m: float, floors: int, seed: int,
                    layout: dict, flavor: str = "generic") -> MeshSpec:
    rng = random.Random(seed)
    shade = rng.choice(("a", "b"))
    profile = profile_for_asset(
        length_m, width_m, bucket="industrial", archetype="warehouse"
    )
    weights = style_for_profile(flavor, profile, industrial_bias=True)
    family = weighted_choice(
        rng,
        tuple(
            (candidate, weight)
            for candidate, weight in weights["families"]
            if candidate in {"concrete", "brick", "stucco"}
        ) or (("concrete", 1),),
    )
    wall = StripUV(layout, f"wall_{family}_{shade}")
    plain = StripUV(layout, f"plain_{family}")
    roller = StripUV(layout, "ground_roller")
    roof = StripUV(layout, "roof_metal")
    trim = StripUV(layout, "trim_dark")

    hx, hy = length_m / 2.0, width_m / 2.0
    overhang = min(0.35, 0.03 * min(length_m, width_m))
    wx, wy = hx - overhang, hy - overhang
    eave_z_wall = floors * FLOOR_H
    rise = max(0.4, math.tan(math.radians(WAREHOUSE_PITCH_DEG)) * wy)
    ridge_z = eave_z_wall + rise
    slope = rise / wy
    eave_z = eave_z_wall - slope * overhang

    spec = MeshSpec()
    ring = ((-wx, -wy), (wx, -wy), (wx, wy), (-wx, wy))
    for i in range(4):
        a, b = ring[i], ring[(i + 1) % 4]
        # Roller doors at grade on the long facades, blank panels between,
        # and a high window band (real warehouses light from the eaves) on
        # the top floor only -- no stretched window rows.
        ground = roller if i in (0, 2) else plain
        wall_quad(spec, a, b, 0.0, FLOOR_H, ground)
        for floor in range(1, floors):
            strip = wall if floor == floors - 1 else plain
            wall_quad(spec, a, b, floor * FLOOR_H, (floor + 1) * FLOOR_H,
                      strip)

    # Gable ends (shallow triangles) + ridged roof spanning the footprint.
    gable_uvs = (
        (plain.u(0.0), plain.v(0.0)), (plain.u(width_m), plain.v(0.0)),
        (plain.u(hy), plain.v(min(1.0, rise / plain.world_h_m))),
    )
    spec.add_tri((wx, -wy, eave_z_wall), (wx, wy, eave_z_wall),
                 (wx, 0.0, ridge_z), *gable_uvs)
    spec.add_tri((-wx, wy, eave_z_wall), (-wx, -wy, eave_z_wall),
                 (-wx, 0.0, ridge_z), *gable_uvs)
    v_top = min(1.0, math.hypot(hy, rise) / roof.world_h_m)
    # Cap roof tiling at 2 repeats (deep-mip rainbow guard).
    u_ridge = min(roof.u(length_m), 2.0)
    spec.add_quad(
        (-hx, -hy, eave_z), (hx, -hy, eave_z),
        (hx, 0.0, ridge_z), (-hx, 0.0, ridge_z),
        (roof.u(0.0), roof.v(0.0)), (u_ridge, roof.v(0.0)),
        (u_ridge, roof.v(v_top)), (roof.u(0.0), roof.v(v_top)))
    spec.add_quad(
        (hx, hy, eave_z), (-hx, hy, eave_z),
        (-hx, 0.0, ridge_z), (hx, 0.0, ridge_z),
        (roof.u(0.0), roof.v(0.0)), (u_ridge, roof.v(0.0)),
        (u_ridge, roof.v(v_top)), (roof.u(0.0), roof.v(v_top)))
    wall_quad(spec, (-hx, -hy), (hx, -hy), eave_z - FASCIA_DROP_M, eave_z, trim)
    wall_quad(spec, (hx, hy), (-hx, hy), eave_z - FASCIA_DROP_M, eave_z, trim)

    # Ridge vents removed at user request: rooflines stay clean.

    # Office annex at a seeded corner of the long facade.
    if length_m >= 20.0 and width_m >= 12.0:
        annex_l = min(8.0, length_m / 4.0)
        annex_w = min(4.0, width_m / 4.0)
        sx = rng.choice((-1.0, 1.0))
        sy = rng.choice((-1.0, 1.0))
        x1 = sx * (wx - 1.0)
        x0 = x1 - sx * annex_l
        y1 = sy * (wy - 1.0)
        y0 = y1 - sy * annex_w
        storefront = StripUV(layout, "ground_storefront")
        box(spec, min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1),
            0.0, FLOOR_H + 0.4, storefront, top_strip=plain)

    set_shell_meta(spec, "pitched", ridge_z, eave_z_wall, wall, roof)
    return validate_spec(spec, length_m, width_m)


def build_bigbox(length_m: float, width_m: float, floors: int, seed: int,
                 layout: dict, flavor: str = "generic") -> MeshSpec:
    rng = random.Random(seed)
    shade = rng.choice(("a", "b"))
    profile = profile_for_asset(
        length_m, width_m, bucket="commercial", archetype="bigbox"
    )
    weights = style_for_profile(flavor, profile, industrial_bias=True)
    family = weighted_choice(
        rng,
        tuple(
            (candidate, weight)
            for candidate, weight in weights["families"]
            if candidate in {"concrete", "brick", "stucco"}
        ) or (("concrete", 1),),
    )
    wall = StripUV(layout, f"wall_{family}_{shade}")
    plain = StripUV(layout, f"plain_{family}")
    storefront = StripUV(layout, "ground_storefront")
    roof = StripUV(layout, "roof_flat")
    trim = StripUV(layout, "trim_dark")

    hx, hy = length_m / 2.0, width_m / 2.0
    top_z = floors * FLOOR_H

    spec = MeshSpec()
    ring = ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))
    for i in range(4):
        a, b = ring[i], ring[(i + 1) % 4]
        # Entrance glazing only on the front facade; everything else blank.
        ground = storefront if i == 0 else plain
        wall_quad(spec, a, b, 0.0, FLOOR_H, ground)
        for floor in range(1, floors):
            # Windowless upper bands: big boxes read as solid slabs.
            wall_quad(spec, a, b, floor * FLOOR_H, (floor + 1) * FLOOR_H,
                      plain)
    parapet_flat_roof(
        spec, hx, hy, top_z, 1.1, 0.35, plain, trim, roof,
    )

    # Rooftop HVAC plant removed at user request: flat roofs stay clean.
    set_shell_meta(spec, "flat", top_z + 1.1, top_z, plain, roof)
    return validate_spec(spec, length_m, width_m)
