"""Flat-roofed regional archetypes: flatres and shophouse.

flatres -- the parapet flat-roof house that dominates residential fabric
across the Mediterranean, Africa, South America and much of Asia: walls to
the footprint edge, low parapet, seeded rooftop water tank and stair bump.

shophouse -- the narrow mixed-use row building of Asian streets: bay-divided
facades (storefront glazing at grade, shuttered windows above), flat parapet
roof. Geometry stays deliberately lean; both archetypes undercut the pitched
houses' vertex budgets.
"""

from __future__ import annotations

import random

from .common import MeshSpec, StripUV, parapet_flat_roof, \
    set_shell_meta, validate_spec, wall_quad, walls_with_floors
from .styles import flavor_style, weighted_choice

FLOOR_H = 3.2
WALL_SHADES = ("a", "b")
BAY_TARGET_W_M = 6.5  # wider bays = fewer facade quads (vertex budget)


def _flat_style(rng: random.Random, layout: dict, flavor: str):
    weights = flavor_style(flavor)
    family = weighted_choice(rng, weights["families"])
    shade = rng.choice(WALL_SHADES)
    return {
        "wall": StripUV(layout, f"wall_{family}_{shade}"),
        "ground": StripUV(layout, f"ground_{family}"),
        "plain": StripUV(layout, f"plain_{family}"),
        "trim": StripUV(layout, "trim_dark"),
        "roof_flat": StripUV(layout, "roof_flat"),
    }


# Rooftop clutter (water tanks / stair bumps) was removed at user request:
# the small cubes read poorly from the air; flat roofs stay clean.


def build_flatres(length_m: float, width_m: float, floors: int, seed: int,
                  layout: dict, flavor: str = "generic") -> MeshSpec:
    rng = random.Random(seed)
    style = _flat_style(rng, layout, flavor)
    hx, hy = length_m / 2.0, width_m / 2.0
    top_z = floors * FLOOR_H

    spec = MeshSpec()
    ring = ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))
    walls_with_floors(spec, ring, 0.0, floors, FLOOR_H,
                      style["ground"], style["wall"])
    parapet_flat_roof(
        spec, hx, hy, top_z, 0.6, 0.25,
        style["plain"], style["trim"], style["roof_flat"],
    )
    set_shell_meta(spec, "flat", top_z + 0.6, top_z,
                   style["wall"], style["roof_flat"])
    return validate_spec(spec, length_m, width_m)


def build_shophouse(length_m: float, width_m: float, floors: int, seed: int,
                    layout: dict, flavor: str = "generic") -> MeshSpec:
    rng = random.Random(seed)
    style = _flat_style(rng, layout, flavor)
    storefront = StripUV(layout, "ground_storefront")
    wall_b = style["wall"]
    weights = flavor_style(flavor)
    alt_family = weighted_choice(rng, weights["families"])
    wall_a = StripUV(layout, f"wall_{alt_family}_a")

    hx, hy = length_m / 2.0, width_m / 2.0
    top_z = floors * FLOOR_H
    n_bays = max(1, int(round(length_m / BAY_TARGET_W_M)))
    bay_w = (2 * hx) / n_bays

    spec = MeshSpec()
    # Long facades bay by bay: storefronts at grade, alternating upper walls.
    for bay in range(n_bays):
        x0 = -hx + bay * bay_w
        x1 = x0 + bay_w
        upper = wall_a if bay % 2 == 0 else wall_b
        for y_edge, flip in ((-hy, False), (hy, True)):
            a, b = ((x1, y_edge), (x0, y_edge)) if flip else \
                ((x0, y_edge), (x1, y_edge))
            wall_quad(spec, a, b, 0.0, FLOOR_H, storefront)
            for floor in range(1, floors):
                wall_quad(spec, a, b, floor * FLOOR_H, (floor + 1) * FLOOR_H,
                          upper)
    # End walls in one piece (plain at grade: party-wall ends).
    for x_edge, flip in ((hx, False), (-hx, True)):
        a, b = ((x_edge, -hy), (x_edge, hy)) if not flip else \
            ((x_edge, hy), (x_edge, -hy))
        for floor in range(floors):
            strip = style["plain"] if floor == 0 else wall_b
            wall_quad(spec, a, b, floor * FLOOR_H, (floor + 1) * FLOOR_H,
                      strip)

    parapet_flat_roof(
        spec, hx, hy, top_z, 0.8, 0.3,
        style["plain"], style["trim"], style["roof_flat"],
    )
    set_shell_meta(spec, "flat", top_z + 0.8, top_z,
                   wall_b, style["roof_flat"])
    return validate_spec(spec, length_m, width_m)
