"""Terraced / rowhouse strip: one shallow gable volume split into bays.

Front and back facades are built bay by bay with alternating wall shades and
a door per bay; thin party-wall stubs and per-2-bay chimneys break up the
roofline so the strip reads as several joined houses from the air.
"""

from __future__ import annotations

import math
import random

from .common import MeshSpec, StripUV, box, set_shell_meta, validate_spec, \
    wall_quad
from .styles import flavor_massing, flavor_style, weighted_choice

FLOOR_H = 3.2
PITCH_DEG = 33.0
MAX_RISE_M = 3.4
FASCIA_DROP_M = 0.18
BAY_TARGET_W_M = 7.0  # wider bays = fewer facade quads (vertex budget)


def build_rowhouse(length_m: float, width_m: float, floors: int, seed: int,
                   layout: dict, flavor: str = "generic") -> MeshSpec:
    rng = random.Random(seed)
    weights = flavor_style(flavor)
    family = weighted_choice(rng, weights["families"])
    wall_a = StripUV(layout, f"wall_{family}_a")
    wall_b = StripUV(layout, f"wall_{family}_b")
    ground = StripUV(layout, f"ground_{family}")
    plain = StripUV(layout, f"plain_{family}")
    roof = StripUV(layout, weighted_choice(rng, weights["roofs"]))
    trim = StripUV(layout, "trim_dark")

    massing = flavor_massing(flavor)
    hx, hy = length_m / 2.0, width_m / 2.0
    overhang = min(0.40 * max(massing["overhang"], 0.5),
                   0.10 * min(length_m, width_m))
    wx, wy = hx - overhang, hy - overhang
    eave_wall_z = floors * FLOOR_H
    pitch = rng.uniform(*massing["pitch"])
    rise = max(0.7, min(math.tan(math.radians(pitch)) * wy, MAX_RISE_M))
    ridge_z = eave_wall_z + rise
    slope = rise / wy
    eave_z = eave_wall_z - slope * overhang

    n_bays = max(2, int(round(length_m / BAY_TARGET_W_M)))
    bay_w = (2 * wx) / n_bays

    spec = MeshSpec()

    # Front/back facades bay by bay (alternating shades, door per bay via
    # the ground strip), end walls in one piece.
    for bay in range(n_bays):
        x0 = -wx + bay * bay_w
        x1 = x0 + bay_w
        upper = wall_a if bay % 2 == 0 else wall_b
        for y_edge, flip in ((-wy, False), (wy, True)):
            a, b = ((x1, y_edge), (x0, y_edge)) if flip else \
                ((x0, y_edge), (x1, y_edge))
            for floor in range(floors):
                strip = ground if floor == 0 else upper
                wall_quad(spec, a, b, floor * FLOOR_H, (floor + 1) * FLOOR_H,
                          strip)
    for x_edge, flip in ((wx, False), (-wx, True)):
        a, b = ((x_edge, -wy), (x_edge, wy)) if not flip else \
            ((x_edge, wy), (x_edge, -wy))
        for floor in range(floors):
            strip = ground if floor == 0 else wall_a
            wall_quad(spec, a, b, floor * FLOOR_H, (floor + 1) * FLOOR_H,
                      strip)

    # Gable ends + roof (full footprint), as in the gable house.
    gable_uvs = (
        (plain.u(0.0), plain.v(0.0)), (plain.u(width_m), plain.v(0.0)),
        (plain.u(hy), plain.v(min(1.0, rise / plain.world_h_m))),
    )
    spec.add_tri((wx, -wy, eave_wall_z), (wx, wy, eave_wall_z),
                 (wx, 0.0, ridge_z), *gable_uvs)
    spec.add_tri((-wx, wy, eave_wall_z), (-wx, -wy, eave_wall_z),
                 (-wx, 0.0, ridge_z), *gable_uvs)
    slope_len = math.hypot(hy, ridge_z - eave_z)
    v_top = min(1.0, slope_len / roof.world_h_m)
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

    # Party-wall stub boxes were dropped in the vertex-budget pass (up to
    # 120 tris on long rows for a detail invisible from the air); bay shade
    # alternation carries the terraced read on its own.

    # One chimney per ~2 bays along the ridge (region-gated).
    for bay in range(0, n_bays, 2):
        if rng.random() < massing["chimney_prob"] and floors <= 4:
            half = 0.28
            cx = -wx + (bay + 0.5 + rng.uniform(0.0, 1.0)) * bay_w
            cx = max(-wx + half, min(wx - half, cx))
            box(spec, cx - half, cx + half, -half, half,
                ridge_z - 0.45, ridge_z + 0.65, plain, top_strip=trim)

    set_shell_meta(spec, "pitched", ridge_z, eave_wall_z, wall_a, roof)
    return validate_spec(spec, length_m, width_m)
