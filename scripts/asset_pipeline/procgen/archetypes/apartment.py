"""Apartment archetypes: flat-roof slab (with balconies) and hipped block.

aptslab: long facades are inset so balconies can protrude back out to the
declared footprint edge (bounds stay exact); parapet flat roof with per-
combo parapet height and regional balcony fronts.  aptblock: compact
massing whose roof form (hip vs parapet flat) is a per-combo probability.

Style comes from the ``apt`` combo: wall families include prefab panel and
curtain glazing; the ground floor is a lobby / shop / plain band with the
combo's own (taller) ground height, squeezed against the F*3.2 total.
"""

from __future__ import annotations

import random

from .common import MeshSpec, StripUV, floor_zs, parapet_flat_roof, \
    set_shell_meta, validate_spec, walls_with_floors
from .house import _hip_roof
from .styles import flavor_massing, profile_for_asset, style_weights, \
    weighted_choice
from .combo_styles import GROUP_SHADES

FLOOR_H = 3.2
BALCONY_W_M = 2.2
MAX_BALCONIES = 6  # vertex-budget cap; balconies read as texture at range


def _apt_style(rng: random.Random, layout: dict, flavor: str = "generic",
               profile: str | None = None):
    weights = style_weights(flavor, "apt", profile)
    family = weighted_choice(rng, weights["families"])
    shade = rng.choice(GROUP_SHADES["apt"])
    strips = layout["strips"]
    wall_name = f"wall_{family}_{shade}"
    plain_name = f"plain_{family}" if f"plain_{family}" in strips \
        else "plain_concrete"
    # Ground band is semantic, not per-family: entrance lobby, ground-floor
    # shops (common in dense fabrics) or a plain concrete base.
    roll = rng.random()
    shop_prob = 0.45 if flavor in ("asia", "mediterranean", "south_america",
                                   "africa") else 0.2
    if family == "curtain" or roll < 0.35:
        ground_name = "ground_lobby"
    elif roll < 0.35 + shop_prob:
        ground_name = "ground_shop"
    else:
        ground_name = "ground_concrete"
    return {
        "family": family,
        "wall": StripUV(layout, wall_name),
        "ground": StripUV(layout, ground_name),
        "plain": StripUV(layout, plain_name),
        "trim": StripUV(layout, "trim_dark"),
        "roof_flat": StripUV(layout, "roof_flat"),
        "balcony": StripUV(layout, "balcony_band"),
        "weights": weights,
    }


def _ground_h(layout: dict) -> float:
    return float((layout.get("dims") or {}).get("ground_h", FLOOR_H))


def build_aptslab(length_m: float, width_m: float, floors: int, seed: int,
                  layout: dict, flavor: str = "generic") -> MeshSpec:
    rng = random.Random(seed)
    profile = profile_for_asset(
        length_m, width_m, bucket="apartments", archetype="aptslab"
    )
    style = _apt_style(rng, layout, flavor, profile)
    massing = flavor_massing(flavor, profile)
    parapet_h = rng.uniform(*massing["parapet"])
    hx, hy = length_m / 2.0, width_m / 2.0
    balcony_depth = float(massing["balcony_depth"])
    inset = min(balcony_depth, 0.08 * width_m)
    wy = hy - inset
    top_z = floors * FLOOR_H
    ground_h = _ground_h(layout)

    spec = MeshSpec()
    ring = ((-hx, -wy), (hx, -wy), (hx, wy), (-hx, wy))
    walls_with_floors(spec, ring, 0.0, floors, FLOOR_H,
                      style["ground"], style["wall"], ground_h=ground_h)
    roof_z = parapet_flat_roof(
        spec, hx, hy, top_z, parapet_h, 0.3,
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
    # plane back out to the footprint edge.  Regional: probability, width
    # and front style (balcony_band strip) come from the combo.
    balcony_w = float(massing.get("balcony_w", BALCONY_W_M))
    if (floors >= 2 and inset >= 0.5
            and rng.random() < float(massing["balcony_prob"])):
        spans = floor_zs(floors, FLOOR_H, ground_h)
        per_floor = max(1, min(int(length_m // (balcony_w * 2.2)),
                               MAX_BALCONIES // max(floors - 1, 1) // 2))
        for floor in range(1, floors):
            z0 = spans[floor][0]
            for k in range(per_floor):
                cx = -hx + (k + 0.5) * (2 * hx) / per_floor \
                    + rng.uniform(-0.3, 0.3)
                x0 = max(-hx + 0.3, cx - balcony_w / 2)
                x1 = min(hx - 0.3, cx + balcony_w / 2)
                for sy in (-1.0, 1.0):
                    y_out = sy * hy
                    # Front rail only -- the slab quad was dropped in the
                    # vertex-budget pass (invisible from above the rail).
                    a, b = ((x0, y_out), (x1, y_out)) if sy < 0 else \
                        ((x1, y_out), (x0, y_out))
                    rail_uvs = style["balcony"].quad_uvs(0.0, x1 - x0,
                                                         0.0, 1.0)
                    spec.add_quad(
                        (a[0], a[1], z0), (b[0], b[1], z0),
                        (b[0], b[1], z0 + 1.05), (a[0], a[1], z0 + 1.05),
                        *rail_uvs)

    set_shell_meta(spec, "flat", top_z + parapet_h, top_z,
                   style["wall"], style["roof_flat"])
    return validate_spec(spec, length_m, width_m)


def build_aptblock(length_m: float, width_m: float, floors: int, seed: int,
                   layout: dict, flavor: str = "generic") -> MeshSpec:
    rng = random.Random(seed)
    profile = profile_for_asset(
        length_m, width_m, bucket="apartments", archetype="aptblock"
    )
    style = _apt_style(rng, layout, flavor, profile)
    massing = flavor_massing(flavor, profile)
    ground_h = _ground_h(layout)
    flat = bool(massing.get("apt_flat")) \
        or rng.random() < float(massing.get("flat_prob", 0.5)) \
        or style["family"] == "curtain"
    if flat:
        parapet_h = rng.uniform(*massing["parapet"])
        hx, hy = length_m / 2.0, width_m / 2.0
        top_z = floors * FLOOR_H
        spec = MeshSpec()
        ring = ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))
        walls_with_floors(spec, ring, 0.0, floors, FLOOR_H,
                          style["ground"], style["wall"], ground_h=ground_h)
        parapet_flat_roof(
            spec, hx, hy, top_z, parapet_h, 0.3,
            style["plain"], style["trim"], style["roof_flat"],
        )
        set_shell_meta(spec, "flat", top_z + parapet_h, top_z,
                       style["wall"], style["roof_flat"])
        return validate_spec(spec, length_m, width_m)
    roof_choices = tuple(
        (name, weight) for name, weight in style["weights"]["roofs"]
        if name != "roof_flat"
    ) or style["weights"]["roofs"]
    roof = StripUV(layout, weighted_choice(rng, roof_choices))
    hx, hy = length_m / 2.0, width_m / 2.0
    overhang = min(0.4, 0.05 * min(length_m, width_m))
    wx, wy = hx - overhang, hy - overhang
    top_z = floors * FLOOR_H

    spec = MeshSpec()
    ring = ((-wx, -wy), (wx, -wy), (wx, wy), (-wx, wy))
    walls_with_floors(spec, ring, 0.0, floors, FLOOR_H,
                      style["ground"], style["wall"], ground_h=ground_h)
    hip_lo, hip_hi = massing["hip_pitch"]
    geo = _hip_roof(spec, hx, hy, overhang, top_z, roof, style["trim"],
                    pitch_deg=rng.uniform(min(hip_lo, 24.0), min(hip_hi, 28.0)),
                    max_rise=3.2)
    set_shell_meta(spec, "pitched", geo["ridge_z"], top_z,
                   style["wall"], roof)
    return validate_spec(spec, length_m, width_m)
