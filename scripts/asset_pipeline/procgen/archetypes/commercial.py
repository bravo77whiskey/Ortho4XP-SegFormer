"""Commercial archetypes: flat-roof block with parapet and rooftop units.

flatcom massing: walls sit ON the declared footprint edge (the parapet line
IS the roof outline aerial imagery sees), storefront glazing on the ground
floor, a parapet band above the top floor, a flat roof slab tucked below the
parapet lip, and 1-3 seeded HVAC boxes.
"""

from __future__ import annotations

import random

from .common import MeshSpec, StripUV, parapet_flat_roof, \
    set_shell_meta, validate_spec, walls_with_floors

FLOOR_H = 3.2
PARAPET_H = 0.8
ROOF_DROP_M = 0.35  # roof slab sits this far below the parapet top

WALL_FAMILIES = ("brick", "stucco", "concrete")
WALL_SHADES = ("a", "b")


def build_flatcom(length_m: float, width_m: float, floors: int, seed: int,
                  layout: dict, flavor: str = "generic") -> MeshSpec:
    rng = random.Random(seed)
    family = rng.choice(WALL_FAMILIES)
    shade = rng.choice(WALL_SHADES)
    wall = StripUV(layout, f"wall_{family}_{shade}")
    plain = StripUV(layout, f"plain_{family}")
    storefront = StripUV(layout, "ground_storefront")
    roof = StripUV(layout, "roof_flat")
    trim = StripUV(layout, "trim_dark")

    hx, hy = length_m / 2.0, width_m / 2.0
    top_z = floors * FLOOR_H

    spec = MeshSpec()
    ring = ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))
    walls_with_floors(spec, ring, 0.0, floors, FLOOR_H, storefront, wall)

    parapet_flat_roof(
        spec, hx, hy, top_z, PARAPET_H, 0.3, plain, trim, roof,
        roof_drop=ROOF_DROP_M,
    )

    # Rooftop HVAC boxes removed at user request: flat roofs stay clean.
    set_shell_meta(spec, "flat", top_z + PARAPET_H, top_z, wall, roof)
    return validate_spec(spec, length_m, width_m)
