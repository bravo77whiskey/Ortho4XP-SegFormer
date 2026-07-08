"""Industrial archetypes: low-pitch warehouse and big-box retail.

warehouse: per-combo cladding (ribbed metal / tilt-up concrete / brick)
with roller or dock doors at grade, a high window band under the eaves,
a shallow ridged metal roof and a small office annex.
bigbox: windowless block with an office/storefront entrance band on the
front facade only and a tall per-combo parapet.

Both draw from the ``ind`` combo: window bands, door widths and ground
heights differ per region and never share the residential rhythm.
"""

from __future__ import annotations

import math
import random

from .common import MeshSpec, StripUV, box, floor_zs, parapet_flat_roof, \
    set_shell_meta, validate_spec, wall_quad
from .styles import flavor_massing, profile_for_asset, style_weights, \
    weighted_choice
from .combo_styles import GROUP_SHADES

FLOOR_H = 3.2
FASCIA_DROP_M = 0.25


def _ind_style(rng: random.Random, layout: dict, flavor: str,
               profile: str | None):
    weights = style_weights(flavor, "ind", profile)
    family = weighted_choice(rng, weights["families"])
    shade = rng.choice(GROUP_SHADES["ind"])
    plain_name = f"plain_{family}" if f"plain_{family}" in layout["strips"] \
        else "plain_concrete"
    return {
        "family": family,
        "wall": StripUV(layout, f"wall_{family}_{shade}"),
        "plain": StripUV(layout, plain_name),
        "band": StripUV(layout, "band_window"),
        "trim": StripUV(layout, "trim_dark"),
    }


def build_warehouse(length_m: float, width_m: float, floors: int, seed: int,
                    layout: dict, flavor: str = "generic") -> MeshSpec:
    rng = random.Random(seed)
    profile = profile_for_asset(
        length_m, width_m, bucket="industrial", archetype="warehouse"
    )
    style = _ind_style(rng, layout, flavor, profile)
    wall, plain = style["wall"], style["plain"]
    # Dock doors are the North-American default; simple rollers elsewhere.
    dock_prob = 0.7 if flavor == "north_america" else 0.25
    roller = StripUV(layout, "ground_dock" if rng.random() < dock_prob
                     else "ground_roller")
    roof = StripUV(layout, "roof_metal")
    trim = StripUV(layout, "trim_dark")
    massing = flavor_massing(flavor, profile)
    ground_h = float((layout.get("dims") or {}).get("ground_h", FLOOR_H))
    spans = floor_zs(floors, FLOOR_H, ground_h)

    hx, hy = length_m / 2.0, width_m / 2.0
    overhang = min(0.35, 0.03 * min(length_m, width_m))
    wx, wy = hx - overhang, hy - overhang
    eave_z_wall = floors * FLOOR_H
    shed_lo, shed_hi = massing.get("shed_pitch", (4.0, 8.0))
    pitch = rng.uniform(shed_lo, shed_hi)
    rise = max(0.4, math.tan(math.radians(pitch)) * wy)
    ridge_z = eave_z_wall + rise
    slope = rise / wy
    eave_z = eave_z_wall - slope * overhang

    spec = MeshSpec()
    ring = ((-wx, -wy), (wx, -wy), (wx, wy), (-wx, wy))
    for i in range(4):
        a, b = ring[i], ring[(i + 1) % 4]
        # Roller/dock doors at grade on the long facades, blank panels
        # between, and a high window band under the eaves (real warehouses
        # light from the top of the wall) -- no stretched window rows.
        ground = roller if i in (0, 2) else plain
        wall_quad(spec, a, b, spans[0][0], spans[0][1], ground)
        for floor in range(1, floors):
            strip = style["band"] if floor == floors - 1 else plain
            wall_quad(spec, a, b, spans[floor][0], spans[floor][1], strip)

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
        office = StripUV(layout, "ground_office")
        box(spec, min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1),
            0.0, min(ground_h, FLOOR_H) + 0.4, office, top_strip=plain)

    set_shell_meta(spec, "pitched", ridge_z, eave_z_wall, wall, roof)
    return validate_spec(spec, length_m, width_m)


def build_bigbox(length_m: float, width_m: float, floors: int, seed: int,
                 layout: dict, flavor: str = "generic") -> MeshSpec:
    rng = random.Random(seed)
    profile = profile_for_asset(
        length_m, width_m, bucket="commercial", archetype="bigbox"
    )
    style = _ind_style(rng, layout, flavor, profile)
    wall, plain = style["wall"], style["plain"]
    office = StripUV(layout, "ground_office")
    roof = StripUV(layout, "roof_flat")
    trim = StripUV(layout, "trim_dark")
    massing = flavor_massing(flavor, profile)
    parapet_h = rng.uniform(*massing["parapet"])
    ground_h = float((layout.get("dims") or {}).get("ground_h", FLOOR_H))
    spans = floor_zs(floors, FLOOR_H, ground_h)

    hx, hy = length_m / 2.0, width_m / 2.0
    top_z = floors * FLOOR_H

    spec = MeshSpec()
    ring = ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))
    for i in range(4):
        a, b = ring[i], ring[(i + 1) % 4]
        # Entrance glazing only on the front facade; everything else blank.
        ground = office if i == 0 else plain
        wall_quad(spec, a, b, spans[0][0], spans[0][1], ground)
        for floor in range(1, floors):
            # Windowless upper bands: big boxes read as solid slabs.
            wall_quad(spec, a, b, spans[floor][0], spans[floor][1], plain)
    parapet_flat_roof(
        spec, hx, hy, top_z, parapet_h, 0.35, plain, trim, roof,
    )

    # Rooftop HVAC plant removed at user request: flat roofs stay clean.
    set_shell_meta(spec, "flat", top_z + parapet_h, top_z, plain, roof)
    return validate_spec(spec, length_m, width_m)
