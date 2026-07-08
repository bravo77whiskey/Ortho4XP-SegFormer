"""Commercial archetypes: flat-roof block with parapet.

flatcom massing: walls sit ON the declared footprint edge (the parapet line
IS the roof outline aerial imagery sees), a storefront / lobby band with
the com combo's taller ground floor, per-combo parapet height, and wall
families that include curtain glazing (office towers) alongside masonry.
"""

from __future__ import annotations

import random

from .common import MeshSpec, StripUV, parapet_flat_roof, \
    set_shell_meta, validate_spec, walls_with_floors
from .styles import flavor_massing, profile_for_asset, style_weights, \
    weighted_choice
from .combo_styles import GROUP_SHADES

FLOOR_H = 3.2
ROOF_DROP_M = 0.35  # roof slab sits this far below the parapet lip


def build_flatcom(length_m: float, width_m: float, floors: int, seed: int,
                  layout: dict, flavor: str = "generic") -> MeshSpec:
    rng = random.Random(seed)
    profile = profile_for_asset(
        length_m, width_m, bucket="commercial", archetype="flatcom"
    )
    weights = style_weights(flavor, "com", profile)
    family = weighted_choice(rng, weights["families"])
    shade = rng.choice(GROUP_SHADES["com"])
    wall = StripUV(layout, f"wall_{family}_{shade}")
    plain_name = f"plain_{family}" if f"plain_{family}" in layout["strips"] \
        else "plain_concrete"
    plain = StripUV(layout, plain_name)
    # Curtain-wall towers get a lobby base; masonry blocks a storefront
    # band (two designs in the combo layout).
    if family == "curtain" or rng.random() < 0.25:
        ground = StripUV(layout, "ground_lobby")
    else:
        ground = StripUV(layout, rng.choice(("ground_storefront",
                                             "ground_storefront_b")))
    roof = StripUV(layout, "roof_flat")
    trim = StripUV(layout, "trim_dark")

    massing = flavor_massing(flavor, profile)
    parapet_h = rng.uniform(*massing["parapet"])
    ground_h = float((layout.get("dims") or {}).get("ground_h", FLOOR_H))
    hx, hy = length_m / 2.0, width_m / 2.0
    top_z = floors * FLOOR_H

    spec = MeshSpec()
    ring = ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))
    walls_with_floors(spec, ring, 0.0, floors, FLOOR_H, ground, wall,
                      ground_h=ground_h)

    parapet_flat_roof(
        spec, hx, hy, top_z, parapet_h, 0.3, plain, trim, roof,
        roof_drop=ROOF_DROP_M,
    )

    # Rooftop HVAC boxes removed at user request: flat roofs stay clean.
    set_shell_meta(spec, "flat", top_z + parapet_h, top_z, wall, roof)
    return validate_spec(spec, length_m, width_m)
