"""Detached-house archetypes: gable, hip and L-shaped (two gable wings).

Massing rules shared by the family:
  * The DECLARED footprint (length x width) is the ROOF outline -- that is
    what aerial imagery (and therefore the YOLO OBB) sees.  Walls are inset
    by the eave overhang so every vertex stays inside +-L/2 x +-W/2 and the
    measured OBJ8 bounds equal the declared dimensions exactly.
  * The floors token F gives wall (eave) height F * 3.2 m, matching the
    height the asset pool parses from the ``_LxWxF`` filename token.  The
    ridge rises above that, which mirrors how real roofs exceed eave height.
"""

from __future__ import annotations

import math
import random

from .common import MeshSpec, StripUV, box, set_shell_meta, validate_spec, \
    wall_quad, walls_with_floors
from .styles import flavor_massing, profile_for_asset, style_weights, \
    weighted_choice
from .combo_styles import GROUP_SHADES

FLOOR_H = 3.2
ROOF_PITCH_DEG = 40.0
MAX_ROOF_RISE_M = 4.2
HIP_PITCH_DEG = 35.0
HIP_MAX_RISE_M = 3.8
FASCIA_DROP_M = 0.18


def _ground_h(layout: dict) -> float:
    return float((layout.get("dims") or {}).get("ground_h", FLOOR_H))


def _house_style(rng: random.Random, layout: dict, flavor: str = "generic",
                 profile: str | None = None):
    """Pick combo-weighted wall/ground/plain strips plus a roof strip."""
    weights = style_weights(flavor, "res", profile)
    family = weighted_choice(rng, weights["families"])
    shade = rng.choice(GROUP_SHADES["res"])
    return {
        "family": family,
        "wall": StripUV(layout, f"wall_{family}_{shade}"),
        "ground": StripUV(layout, f"ground_{family}"),
        "plain": StripUV(layout, f"plain_{family}"),
        "roof": StripUV(layout, weighted_choice(
            rng,
            tuple((name, weight) for name, weight in weights["roofs"]
                  if name != "roof_flat") or weights["roofs"],
        )),
        "trim": StripUV(layout, "trim_dark"),
    }


def _gable_volume(spec: MeshSpec, cx: float, cy: float, lu: float, lv: float,
                  floors: int, style: dict, axis: str = "x",
                  pitch_deg: float = ROOF_PITCH_DEG,
                  max_rise: float = MAX_ROOF_RISE_M,
                  overhang_scale: float = 1.0, ground_h: float = None):
    """One gabled volume: walls, gable ends, roof, fascia.

    ``lu`` is the extent along the ridge, ``lv`` across it; ``axis`` maps
    the ridge to world X or Y.  The ROOF spans the full lu x lv rectangle
    centered at (cx, cy); walls are inset by the eave overhang.
    """
    flip = axis != "x"

    def world(u, v, z):
        return (cx + v, cy + u, z) if flip else (cx + u, cy + v, z)

    def emit(points, uvs):
        if flip:  # axis swap mirrors handedness; reverse to stay outward
            points = points[::-1]
            uvs = uvs[::-1]
        spec.add_face(points, uvs)

    wall, ground = style["wall"], style["ground"]
    plain, roof, trim = style["plain"], style["roof"], style["trim"]

    hu, hv = lu / 2.0, lv / 2.0
    overhang = min(0.45 * max(overhang_scale, 0.5), 0.12 * min(lu, lv))
    wu, wv = hu - overhang, hv - overhang
    eave_wall_z = floors * FLOOR_H
    rise = max(0.8, min(math.tan(math.radians(pitch_deg)) * wv, max_rise))
    ridge_z = eave_wall_z + rise
    slope = rise / wv
    eave_z = eave_wall_z - slope * overhang

    # Walls (world-space CCW ring; walls_with_floors handles winding).
    ring_local = ((-wu, -wv), (wu, -wv), (wu, wv), (-wu, wv))
    ring = [world(u, v, 0.0)[:2] for u, v in ring_local]
    if flip:
        ring = ring[::-1]
    walls_with_floors(spec, ring, 0.0, floors, FLOOR_H, ground, wall,
                      ground_h=ground_h)

    def _gable_uvs():
        return (
            (plain.u(0.0), plain.v(0.0)),
            (plain.u(lv), plain.v(0.0)),
            (plain.u(hv), plain.v(min(1.0, rise / plain.world_h_m))),
        )

    emit((world(wu, -wv, eave_wall_z), world(wu, wv, eave_wall_z),
          world(wu, 0.0, ridge_z)), _gable_uvs())
    emit((world(-wu, wv, eave_wall_z), world(-wu, -wv, eave_wall_z),
          world(-wu, 0.0, ridge_z)), _gable_uvs())

    slope_len = math.hypot(hv, ridge_z - eave_z)
    v_top = min(1.0, slope_len / roof.world_h_m)
    # Cap roof tiling at 2 repeats: long roofs otherwise hit deep mip
    # levels at distance where atlas strips bleed (rainbow artifact).
    u_ridge = min(roof.u(lu), 2.0)
    emit((world(-hu, -hv, eave_z), world(hu, -hv, eave_z),
          world(hu, 0.0, ridge_z), world(-hu, 0.0, ridge_z)),
         ((roof.u(0.0), roof.v(0.0)), (u_ridge, roof.v(0.0)),
          (u_ridge, roof.v(v_top)), (roof.u(0.0), roof.v(v_top))))
    emit((world(hu, hv, eave_z), world(-hu, hv, eave_z),
          world(-hu, 0.0, ridge_z), world(hu, 0.0, ridge_z)),
         ((roof.u(0.0), roof.v(0.0)), (u_ridge, roof.v(0.0)),
          (u_ridge, roof.v(v_top)), (roof.u(0.0), roof.v(v_top))))

    # Eave fascia only -- rake fascia along the gable slopes was dropped in
    # the vertex-budget pass; it is invisible from aerial viewing distances.
    emit((world(-hu, -hv, eave_z - FASCIA_DROP_M),
          world(hu, -hv, eave_z - FASCIA_DROP_M),
          world(hu, -hv, eave_z), world(-hu, -hv, eave_z)),
         trim.quad_uvs(0.0, lu))
    emit((world(hu, hv, eave_z - FASCIA_DROP_M),
          world(-hu, hv, eave_z - FASCIA_DROP_M),
          world(-hu, hv, eave_z), world(hu, hv, eave_z)),
         trim.quad_uvs(0.0, lu))

    return {"ridge_z": ridge_z, "eave_wall_z": eave_wall_z, "slope": slope,
            "wu": wu, "wv": wv}


def _roof_chimney(spec, rng, style, cx, cy, geo, axis="x"):
    """Seeded chimney near the ridge of a gable volume."""
    half = 0.30
    u = rng.uniform(-0.35, 0.35) * max(geo["wu"] - 2 * half, 0.0)
    v = rng.uniform(0.15, 0.45) * geo["wv"] * rng.choice((-1.0, 1.0))
    base_z = geo["ridge_z"] - geo["slope"] * abs(v) - 0.4
    wx, wy = (cx + v, cy + u) if axis != "x" else (cx + u, cy + v)
    box(spec, wx - half, wx + half, wy - half, wy + half,
        base_z, geo["ridge_z"] + 0.7, style["plain"], top_strip=style["trim"])


def build_gable(length_m: float, width_m: float, floors: int, seed: int,
                layout: dict, flavor: str = "generic") -> MeshSpec:
    rng = random.Random(seed)
    profile = profile_for_asset(length_m, width_m, archetype="gable")
    style = _house_style(rng, layout, flavor, profile)
    massing = flavor_massing(flavor, profile)
    pitch = rng.uniform(*massing["pitch"])
    spec = MeshSpec()
    geo = _gable_volume(
        spec, 0.0, 0.0, length_m, width_m, floors, style,
        pitch_deg=pitch, overhang_scale=massing["overhang"],
        ground_h=_ground_h(layout),
    )
    if (rng.random() < massing["chimney_prob"]
            and length_m >= 5.0 and floors <= 3):
        _roof_chimney(spec, rng, style, 0.0, 0.0, geo)
    set_shell_meta(spec, "pitched", geo["ridge_z"], geo["eave_wall_z"],
                   style["wall"], style["roof"])
    return validate_spec(spec, length_m, width_m)


def _hip_roof(spec: MeshSpec, hx: float, hy: float, overhang: float,
              eave_wall_z: float, roof: StripUV, trim: StripUV,
              pitch_deg: float = HIP_PITCH_DEG,
              max_rise: float = HIP_MAX_RISE_M):
    """Hip roof spanning the full +-hx x +-hy rectangle (ridge along X)."""
    wy = hy - overhang
    slope = min(math.tan(math.radians(pitch_deg)), max_rise / hy)
    ridge_z = eave_wall_z + slope * wy
    eave_z = eave_wall_z - slope * overhang
    rx = hx - hy

    rise = ridge_z - eave_z
    slope_len = math.hypot(hy, rise)
    v_top = min(1.0, slope_len / roof.world_h_m)
    # Cap roof tiling at 2 repeats (deep-mip rainbow guard, see gable).
    u_scale = min(1.0, 2.0 / max(roof.u(2 * hx), 1e-6))

    def ru(distance_m):
        return roof.u(distance_m) * u_scale

    if rx < 0.05:
        rx = 0.0
        spec.add_tri(
            (-hx, -hy, eave_z), (hx, -hy, eave_z), (0.0, 0.0, ridge_z),
            (ru(0.0), roof.v(0.0)), (ru(2 * hx), roof.v(0.0)),
            (ru(hx), roof.v(v_top)))
        spec.add_tri(
            (hx, hy, eave_z), (-hx, hy, eave_z), (0.0, 0.0, ridge_z),
            (ru(0.0), roof.v(0.0)), (ru(2 * hx), roof.v(0.0)),
            (ru(hx), roof.v(v_top)))
    else:
        spec.add_quad(
            (-hx, -hy, eave_z), (hx, -hy, eave_z),
            (rx, 0.0, ridge_z), (-rx, 0.0, ridge_z),
            (ru(0.0), roof.v(0.0)), (ru(2 * hx), roof.v(0.0)),
            (ru(hx + rx), roof.v(v_top)), (ru(hx - rx), roof.v(v_top)))
        spec.add_quad(
            (hx, hy, eave_z), (-hx, hy, eave_z),
            (-rx, 0.0, ridge_z), (rx, 0.0, ridge_z),
            (ru(0.0), roof.v(0.0)), (ru(2 * hx), roof.v(0.0)),
            (ru(hx + rx), roof.v(v_top)), (ru(hx - rx), roof.v(v_top)))
    spec.add_tri(
        (hx, -hy, eave_z), (hx, hy, eave_z), (rx, 0.0, ridge_z),
        (ru(0.0), roof.v(0.0)), (ru(2 * hy), roof.v(0.0)),
        (ru(hy), roof.v(v_top)))
    spec.add_tri(
        (-hx, hy, eave_z), (-hx, -hy, eave_z), (-rx, 0.0, ridge_z),
        (ru(0.0), roof.v(0.0)), (ru(2 * hy), roof.v(0.0)),
        (ru(hy), roof.v(v_top)))

    eave_ring = ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))
    for i in range(4):
        a, b = eave_ring[i], eave_ring[(i + 1) % 4]
        wall_quad(spec, a, b, eave_z - FASCIA_DROP_M, eave_z, trim)
    return {"ridge_z": ridge_z, "rx": rx, "slope": slope}


def build_hip(length_m: float, width_m: float, floors: int, seed: int,
              layout: dict, flavor: str = "generic") -> MeshSpec:
    rng = random.Random(seed)
    profile = profile_for_asset(length_m, width_m, archetype="hip")
    style = _house_style(rng, layout, flavor, profile)
    massing = flavor_massing(flavor, profile)
    hx, hy = length_m / 2.0, width_m / 2.0
    overhang = min(0.45 * max(massing["overhang"], 0.5),
                   0.12 * min(length_m, width_m))
    wx, wy = hx - overhang, hy - overhang
    eave_wall_z = floors * FLOOR_H

    spec = MeshSpec()
    ring = ((-wx, -wy), (wx, -wy), (wx, wy), (-wx, wy))
    walls_with_floors(spec, ring, 0.0, floors, FLOOR_H,
                      style["ground"], style["wall"],
                      ground_h=_ground_h(layout))
    geo = _hip_roof(spec, hx, hy, overhang, eave_wall_z,
                    style["roof"], style["trim"],
                    pitch_deg=rng.uniform(*massing["hip_pitch"]))
    if (rng.random() < massing["chimney_prob"]
            and geo["rx"] >= 1.0 and floors <= 3):
        half = 0.30
        cx = rng.uniform(-0.6, 0.6) * max(geo["rx"] - 2 * half, 0.0)
        box(spec, cx - half, cx + half, -half, half,
            geo["ridge_z"] - 0.5, geo["ridge_z"] + 0.6,
            style["plain"], top_strip=style["trim"])
    set_shell_meta(spec, "pitched", geo["ridge_z"], eave_wall_z,
                   style["wall"], style["roof"])
    return validate_spec(spec, length_m, width_m)


def build_lshape(length_m: float, width_m: float, floors: int, seed: int,
                 layout: dict, flavor: str = "generic") -> MeshSpec:
    """L-shaped house: a main gable wing along X plus a cross wing along Y.

    The two gabled volumes interpenetrate at the corner (standard low-poly
    practice; the junction is hidden inside).  Together their roofs span the
    full declared footprint.
    """
    rng = random.Random(seed)
    profile = profile_for_asset(length_m, width_m, archetype="lshape")
    style = _house_style(rng, layout, flavor, profile)
    massing = flavor_massing(flavor, profile)
    pitch = rng.uniform(*massing["pitch"])
    hx, hy = length_m / 2.0, width_m / 2.0

    main_depth = max(min(0.58 * width_m, width_m - 2.2), 0.45 * width_m)
    wing_len = max(min(0.5 * length_m, length_m - 2.2), 0.38 * length_m)
    sx = rng.choice((-1.0, 1.0))  # which end hosts the cross wing
    sy = rng.choice((-1.0, 1.0))  # which side the main wing hugs

    spec = MeshSpec()
    # Main wing: full length, hugging the +-sy edge.
    main_cy = sy * (hy - main_depth / 2.0)
    geo_main = _gable_volume(
        spec, 0.0, main_cy, length_m, main_depth, floors, style, axis="x",
        pitch_deg=pitch, overhang_scale=massing["overhang"],
        ground_h=_ground_h(layout))
    # Cross wing: full width, hugging the +-sx end.
    wing_cx = sx * (hx - wing_len / 2.0)
    _gable_volume(
        spec, wing_cx, 0.0, width_m, wing_len, floors, style, axis="y",
        pitch_deg=pitch, overhang_scale=massing["overhang"],
        ground_h=_ground_h(layout))

    if rng.random() < massing["chimney_prob"] and floors <= 3:
        _roof_chimney(spec, rng, style, 0.0, main_cy, geo_main)
    set_shell_meta(spec, "pitched", geo_main["ridge_z"],
                   geo_main["eave_wall_z"], style["wall"], style["roof"])
    return validate_spec(spec, length_m, width_m)
