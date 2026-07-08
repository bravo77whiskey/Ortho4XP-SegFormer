"""Flat-roofed regional archetypes: flatres and shophouse.

flatres -- the parapet flat-roof house that dominates residential fabric
across the Mediterranean, Africa, South America and much of Asia: walls to
the footprint edge, per-combo parapet height, res-combo materials.

shophouse -- the narrow mixed-use row building of Asian streets: bay-
divided facades (storefront glazing at grade with the com combo's taller
ground floor, regional upper walls), flat parapet roof.  Draws from the
``com`` combo so its windows/doors/storefronts differ from houses in the
same region.
"""

from __future__ import annotations

import random

from .common import MeshSpec, StripUV, floor_zs, parapet_flat_roof, \
    set_shell_meta, validate_spec, wall_quad, walls_with_floors
from .styles import flavor_massing, profile_for_asset, style_weights, \
    weighted_choice
from .combo_styles import GROUP_SHADES

FLOOR_H = 3.2


def _res_style(rng: random.Random, layout: dict, flavor: str,
               profile: str | None = None):
    weights = style_weights(flavor, "res", profile)
    family = weighted_choice(rng, weights["families"])
    shade = rng.choice(GROUP_SHADES["res"])
    return {
        "family": family,
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
    profile = profile_for_asset(length_m, width_m, archetype="flatres")
    style = _res_style(rng, layout, flavor, profile)
    massing = flavor_massing(flavor, profile)
    parapet_h = rng.uniform(*massing["parapet"])
    ground_h = float((layout.get("dims") or {}).get("ground_h", FLOOR_H))
    hx, hy = length_m / 2.0, width_m / 2.0
    top_z = floors * FLOOR_H

    spec = MeshSpec()
    ring = ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))
    walls_with_floors(spec, ring, 0.0, floors, FLOOR_H,
                      style["ground"], style["wall"], ground_h=ground_h)
    parapet_flat_roof(
        spec, hx, hy, top_z, parapet_h, 0.25,
        style["plain"], style["trim"], style["roof_flat"],
    )
    set_shell_meta(spec, "flat", top_z + parapet_h, top_z,
                   style["wall"], style["roof_flat"])
    return validate_spec(spec, length_m, width_m)


def build_shophouse(length_m: float, width_m: float, floors: int, seed: int,
                    layout: dict, flavor: str = "generic") -> MeshSpec:
    rng = random.Random(seed)
    profile = profile_for_asset(
        length_m, width_m, bucket="commercial", archetype="shophouse"
    )
    weights = style_weights(flavor, "com", profile)
    fam_pairs = tuple((f, w) for f, w in weights["families"]
                      if f != "curtain") or weights["families"]
    family_a = weighted_choice(rng, fam_pairs)
    family_b = weighted_choice(rng, fam_pairs)
    shades = GROUP_SHADES["com"]
    wall_a = StripUV(layout, f"wall_{family_a}_{rng.choice(shades)}")
    wall_b = StripUV(layout, f"wall_{family_b}_{rng.choice(shades)}")
    plain_name = f"plain_{family_b}" if f"plain_{family_b}" \
        in layout["strips"] else "plain_concrete"
    plain = StripUV(layout, plain_name)
    storefront = StripUV(layout, rng.choice(("ground_storefront",
                                             "ground_storefront_b")))
    roof_flat = StripUV(layout, "roof_flat")
    trim = StripUV(layout, "trim_dark")

    massing = flavor_massing(flavor, profile)
    parapet_h = rng.uniform(*massing["parapet"])
    ground_h = float((layout.get("dims") or {}).get("ground_h", FLOOR_H))
    spans = floor_zs(floors, FLOOR_H, ground_h)
    hx, hy = length_m / 2.0, width_m / 2.0
    top_z = floors * FLOOR_H
    n_bays = max(1, int(round(length_m / float(massing.get("bay_w", 6.5)))))
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
            wall_quad(spec, a, b, spans[0][0], spans[0][1], storefront)
            for floor in range(1, floors):
                wall_quad(spec, a, b, spans[floor][0], spans[floor][1],
                          upper)
    # End walls in one piece (plain at grade: party-wall ends).
    for x_edge, flip in ((hx, False), (-hx, True)):
        a, b = ((x_edge, -hy), (x_edge, hy)) if not flip else \
            ((x_edge, hy), (x_edge, -hy))
        for floor in range(floors):
            strip = plain if floor == 0 else wall_b
            wall_quad(spec, a, b, spans[floor][0], spans[floor][1], strip)

    parapet_flat_roof(
        spec, hx, hy, top_z, parapet_h, 0.3,
        plain, trim, roof_flat,
    )
    set_shell_meta(spec, "flat", top_z + parapet_h, top_z,
                   wall_b, roof_flat)
    return validate_spec(spec, length_m, width_m)
