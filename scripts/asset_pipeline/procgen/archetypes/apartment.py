"""Apartment archetypes: flat-roof slab and hipped block.

aptslab: full-footprint walls with a parapet flat roof at the combo's
parapet height.  Protruding balcony quads were REMOVED at user request
(2026-07-08): the single vertex-budget bay stacked into an odd centered
column in-sim; the balcony_band strip stays in the apt layout so UVs of
other apartment objects are unaffected.  aptblock: compact massing whose
roof form (hip vs parapet flat) is a per-combo probability.

Style comes from the ``apt`` combo: wall families include prefab panel and
curtain glazing; the ground floor is a lobby / shop / plain band with the
combo's own (taller) ground height, squeezed against the F*3.2 total.
"""

from __future__ import annotations

import random

from .common import MeshSpec, StripUV, parapet_flat_roof, \
    set_shell_meta, validate_spec, walls_with_floors
from .house import _hip_roof
from .styles import flavor_massing, profile_for_asset, style_weights, \
    weighted_choice
from .combo_styles import GROUP_SHADES

FLOOR_H = 3.2


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
    top_z = floors * FLOOR_H
    ground_h = _ground_h(layout)

    spec = MeshSpec()
    ring = ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))
    walls_with_floors(spec, ring, 0.0, floors, FLOOR_H,
                      style["ground"], style["wall"], ground_h=ground_h)
    parapet_flat_roof(
        spec, hx, hy, top_z, parapet_h, 0.3,
        style["plain"], style["trim"], style["roof_flat"],
    )

    # Balconies removed at user request (2026-07-08); see module docstring.
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
