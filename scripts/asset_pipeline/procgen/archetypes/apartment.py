"""Apartment archetypes: flat-roof slab (with balconies) and hipped block.

aptslab: long facades are inset so balconies can protrude back out to the
declared footprint edge (bounds stay exact); parapet flat roof with a
stairwell bump.  aptblock: compact massing with a hip roof, no balconies.
"""

from __future__ import annotations

import random

from .common import MeshSpec, StripUV, parapet_flat_roof, \
    set_shell_meta, validate_spec, walls_with_floors
from .house import _hip_roof
from .styles import flavor_massing, flavor_style, weighted_choice

FLOOR_H = 3.2
PARAPET_H = 0.7
BALCONY_DEPTH_M = 0.8
BALCONY_W_M = 2.2
MAX_BALCONIES = 6  # vertex-budget cap; balconies read as texture at range

WALL_SHADES = ("a", "b")


def _apt_style(rng: random.Random, layout: dict, flavor: str = "generic"):
    # Apartments skew to concrete everywhere; bias the regional weights.
    weights = flavor_style(flavor)
    families = tuple(weights["families"]) + (("concrete", 3),)
    family = weighted_choice(rng, families)
    shade = rng.choice(WALL_SHADES)
    return {
        "wall": StripUV(layout, f"wall_{family}_{shade}"),
        "ground": StripUV(layout, f"ground_{family}"),
        "plain": StripUV(layout, f"plain_{family}"),
        "trim": StripUV(layout, "trim_dark"),
        "roof_flat": StripUV(layout, "roof_flat"),
    }


def build_aptslab(length_m: float, width_m: float, floors: int, seed: int,
                  layout: dict, flavor: str = "generic") -> MeshSpec:
    rng = random.Random(seed)
    style = _apt_style(rng, layout, flavor)
    hx, hy = length_m / 2.0, width_m / 2.0
    inset = min(BALCONY_DEPTH_M, 0.08 * width_m)
    wy = hy - inset
    top_z = floors * FLOOR_H

    spec = MeshSpec()
    ring = ((-hx, -wy), (hx, -wy), (hx, wy), (-hx, wy))
    walls_with_floors(spec, ring, 0.0, floors, FLOOR_H,
                      style["ground"], style["wall"])
    # End walls flush to the footprint corners are covered by the parapet
    # band; close the gap strips between wall plane and footprint edge with
    # the roof overhanging via the parapet block below.
    roof_z = parapet_flat_roof(
        spec, hx, hy, top_z, PARAPET_H, 0.3,
        style["plain"], style["trim"], style["roof_flat"],
    )
    # Soffits: close the underside of the roof band overhanging the inset
    # facades (downward-facing, same winding as a box bottom).
    for y0, y1 in ((-hy, -wy), (wy, hy)):
        spec.add_quad(
            (-hx, y0, top_z), (-hx, y1, top_z),
            (hx, y1, top_z), (hx, y0, top_z),
            *style["plain"].quad_uvs(0.0, 2 * hx, 0.0, 0.3))

    # Balconies on both long facades, floors 1+, protruding from the wall
    # plane back out to the footprint edge.
    if floors >= 2 and inset >= 0.5:
        per_floor = max(1, min(int(length_m // (BALCONY_W_M * 2.2)),
                               MAX_BALCONIES // max(floors - 1, 1) // 2))
        for floor in range(1, floors):
            z0 = floor * FLOOR_H
            for k in range(per_floor):
                cx = -hx + (k + 0.5) * (2 * hx) / per_floor \
                    + rng.uniform(-0.3, 0.3)
                x0 = max(-hx + 0.3, cx - BALCONY_W_M / 2)
                x1 = min(hx - 0.3, cx + BALCONY_W_M / 2)
                for sy in (-1.0, 1.0):
                    y_out = sy * hy
                    # Front rail only -- the slab quad was dropped in the
                    # vertex-budget pass (invisible from above the rail).
                    a, b = ((x0, y_out), (x1, y_out)) if sy < 0 else \
                        ((x1, y_out), (x0, y_out))
                    rail_uvs = style["plain"].quad_uvs(0.0, x1 - x0, 0.0, 0.5)
                    spec.add_quad(
                        (a[0], a[1], z0), (b[0], b[1], z0),
                        (b[0], b[1], z0 + 1.05), (a[0], a[1], z0 + 1.05),
                        *rail_uvs)

    # Rooftop stairwell bump removed at user request: flat roofs stay clean.
    set_shell_meta(spec, "flat", top_z + PARAPET_H, top_z,
                   style["wall"], style["roof_flat"])
    return validate_spec(spec, length_m, width_m)


def build_aptblock(length_m: float, width_m: float, floors: int, seed: int,
                   layout: dict, flavor: str = "generic") -> MeshSpec:
    rng = random.Random(seed)
    style = _apt_style(rng, layout, flavor)
    roof = StripUV(
        layout, weighted_choice(rng, flavor_style(flavor)["roofs"])
    )
    hx, hy = length_m / 2.0, width_m / 2.0
    overhang = min(0.4, 0.05 * min(length_m, width_m))
    wx, wy = hx - overhang, hy - overhang
    top_z = floors * FLOOR_H

    spec = MeshSpec()
    ring = ((-wx, -wy), (wx, -wy), (wx, wy), (-wx, wy))
    walls_with_floors(spec, ring, 0.0, floors, FLOOR_H,
                      style["ground"], style["wall"])
    hip_lo, hip_hi = flavor_massing(flavor)["hip_pitch"]
    geo = _hip_roof(spec, hx, hy, overhang, top_z, roof, style["trim"],
                    pitch_deg=rng.uniform(min(hip_lo, 24.0), min(hip_hi, 28.0)),
                    max_rise=3.2)
    set_shell_meta(spec, "pitched", geo["ridge_z"], top_z,
                   style["wall"], roof)
    return validate_spec(spec, length_m, width_m)
