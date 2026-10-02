"""High-poly strip recipes: facade floors, plain walls, roof coverings, trims.

Each builder returns ``(Geo, materials)`` for ONE atlas strip modelled in
strip space (see kitgeo): x = u in metres over [0, W), z = v over [0, H).
Geometry is built for x in [-margin, W + margin] from a per-bay table that
wraps modulo the bay count, so the rendered strip tiles seamlessly in u and
ambient occlusion at the frame edges sees the same neighbours it will see
after wrapping.

Nothing here imports bpy: recipes are testable on the host and the Blender
bake script only turns the Geo/material dicts into meshes and node trees.
"""

from __future__ import annotations

import math
import random

from kitgeo import Geo, circle_section, reveal, wall_with_openings

# Material classes written to the "cls" AOV; the weathering pass keys on them.
CLS_WALL = 1
CLS_TRIM = 2       # frames, surrounds, fascia boards, doors
CLS_GLASS = 3      # everything seen through a window (interior, curtains)
CLS_ROOF = 4
CLS_METAL = 5      # bars, rails, AC units, gutters
CLS_SILL = 6
CLS_PLINTH = 7
CLS_SHUTTER = 8
CLS_SIGN = 9
CLS_OBJECT = 10    # rooftop equipment baked into roof strips


def mat(base, *, noise=(0.6, 0.06), fine=(9.0, 0.035), pattern=None,
        bump=0.0, rough=0.85, metal=0.0, drip=0.0, cls=CLS_WALL,
        mode="rnd", emit=None, **extra):
    """Material spec consumed by bake_materials.build_material()."""
    if isinstance(base, str):
        base = [base]
    spec = {"base": list(base), "noise": list(noise), "fine": list(fine),
            "pattern": pattern, "bump": bump, "rough": rough, "metal": metal,
            "drip": drip, "cls": cls, "mode": mode, "emit": emit}
    spec.update(extra)
    return spec


# ---------------------------------------------------------------- windows

def _interior_volume(geo, W, H, margin, depth=3.2, z_floor=0.0,
                     z_ceiling=None):
    """Closed dark volume behind the wall so windows read as rooms.

    Light only enters through the openings, which gives the natural
    bright-sill / dark-back gradient of a real window in daylight.
    """
    zc = H if z_ceiling is None else z_ceiling
    x0, x1 = -margin - 1.0, W + margin + 1.0
    geo.face("interior", [(x0, depth, z_floor), (x1, depth, z_floor),
                          (x1, depth, zc), (x0, depth, zc)],
             facing=(0, -1, 0), rnd=0.3)
    geo.face("interior_floor", [(x0, 0.0, z_floor), (x1, 0.0, z_floor),
                                (x1, depth, z_floor), (x0, depth, z_floor)],
             facing=(0, 0, 1))
    geo.face("interior_floor", [(x0, 0.0, zc), (x1, 0.0, zc),
                                (x1, depth, zc), (x0, depth, zc)],
             facing=(0, 0, -1))
    for xe, nx in ((x0, 1), (x1, -1)):
        geo.face("interior", [(xe, 0.0, z_floor), (xe, depth, z_floor),
                              (xe, depth, zc), (xe, 0.0, zc)],
                 facing=(nx, 0, 0))


def _frame_rect(geo, m, x0, x1, z0, z1, y0, y1, fw, rnd=0.0):
    """Four frame members around an opening (outer frame of a window)."""
    geo.box(m, x0, x1, y0, y1, z0, z0 + fw, rnd)
    geo.box(m, x0, x1, y0, y1, z1 - fw, z1, rnd)
    geo.box(m, x0, x0 + fw, y0, y1, z0 + fw, z1 - fw, rnd)
    geo.box(m, x1 - fw, x1, y0, y1, z0 + fw, z1 - fw, rnd)


def _curtains(geo, rng, x0, x1, z0, z1, y, style):
    """Per-window interior dressing: curtains, blinds or nothing."""
    r = rng.random()
    tint = rng.random()
    w = x1 - x0
    if style == "clean_blinds" and r < 0.55:
        drop = rng.uniform(0.15, 0.95)
        geo.box("blind", x0, x1, y, y + 0.01, z1 - drop * (z1 - z0), z1,
                rnd=tint)
        return
    if r < 0.30:
        return  # bare window, see the room
    if r < 0.72:
        # two side curtains, partly drawn
        a = rng.uniform(0.12, 0.45) * w
        b = rng.uniform(0.12, 0.45) * w
        geo.box("curtain", x0, x0 + a, y, y + 0.02, z0 - 0.05, z1, rnd=tint)
        geo.box("curtain", x1 - b, x1, y, y + 0.02, z0 - 0.05, z1, rnd=tint)
        if rng.random() < 0.5:
            # net curtain across the lower part
            geo.box("net", x0 + a, x1 - b, y + 0.03, y + 0.04, z0,
                    z0 + rng.uniform(0.4, 0.8) * (z1 - z0), rnd=tint)
        return
    if r < 0.9:
        drop = rng.uniform(0.2, 0.9)
        geo.box("blind", x0, x1, y, y + 0.01, z1 - drop * (z1 - z0), z1,
                rnd=tint)
        return
    geo.box("curtain", x0, x1, y, y + 0.02, z0 - 0.05, z1, rnd=tint)


def window(geo, rng, x0, x1, z0, z1, spec, *, sill=True):
    """One window assembly in an opening already cut into the wall.

    spec keys: type, depth, frame_w, muntins (cols, rows), shutter,
    shutter_state, bars, sill_proud, sill_mat, surround, grille, style.
    """
    d = spec.get("depth", 0.14)
    fw = spec.get("frame_w", 0.07)
    kind = spec.get("type", "casement")
    reveal(geo, spec.get("reveal_mat", "reveal"), x0, x1, z0, z1, d)
    fy0, fy1 = d - 0.02, d + 0.06
    # glazing dressing sits behind the frame
    _curtains(geo, rng, x0 + fw, x1 - fw, z0 + fw, z1 - fw, d + 0.12,
              spec.get("style", "mixed"))
    frame = spec.get("frame_mat", "frame")
    _frame_rect(geo, frame, x0, x1, z0, z1, fy0, fy1, fw)
    w = x1 - x0
    h = z1 - z0
    if kind in ("casement", "steel", "modern"):
        cols, rows = spec.get("muntins", (2 if w > 1.0 else 1, 1))
        if kind == "casement" and spec.get("transom", False):
            tz = z1 - 0.32 * h
            geo.box(frame, x0 + fw, x1 - fw, fy0, fy1, tz - fw / 2,
                    tz + fw / 2)
        mw = fw * (0.55 if kind == "steel" else 1.0)
        for c in range(1, cols):
            mx = x0 + w * c / cols
            geo.box(frame, mx - mw / 2, mx + mw / 2, fy0 - 0.01, fy1, z0 + fw,
                    z1 - fw)
        for r in range(1, rows):
            mz = z0 + h * r / rows
            geo.box(frame, x0 + fw, x1 - fw, fy0 - 0.01, fy1, mz - mw / 2,
                    mz + mw / 2)
        if kind == "casement" and cols >= 2:
            # opening sashes read as a second, thinner frame per leaf
            for c in range(cols):
                lx0 = x0 + w * c / cols + fw * 0.5
                lx1 = x0 + w * (c + 1) / cols - fw * 0.5
                _frame_rect(geo, frame, lx0, lx1, z0 + fw * 0.6,
                            z1 - fw * 0.6, fy0 - 0.02, fy0 + 0.02, fw * 0.55)
    elif kind == "double_hung":
        mid = z0 + h * 0.5
        # lower sash sits in front of the upper one: a meeting rail
        geo.box(frame, x0 + fw, x1 - fw, fy0 - 0.03, fy1, mid - fw * 0.6,
                mid + fw * 0.6)
        cols, rows = spec.get("muntins", (3, 2))
        mw = 0.025
        for half_z0, half_z1 in ((z0 + fw, mid), (mid, z1 - fw)):
            for c in range(1, cols):
                mx = x0 + w * c / cols
                geo.box(frame, mx - mw / 2, mx + mw / 2, fy0 - 0.015, fy0,
                        half_z0, half_z1)
            for r in range(1, rows):
                mz = half_z0 + (half_z1 - half_z0) * r / rows
                geo.box(frame, x0 + fw, x1 - fw, fy0 - 0.015, fy0,
                        mz - mw / 2, mz + mw / 2)
    elif kind == "slider":
        mx = x0 + w * rng.uniform(0.45, 0.55)
        geo.box(frame, mx - fw * 0.6, mx + fw * 0.6, fy0 - 0.03, fy1,
                z0 + fw, z1 - fw)
        # the two sashes sit in different tracks
        _frame_rect(geo, frame, x0 + fw * 0.5, mx + fw * 0.6, z0 + fw * 0.5,
                    z1 - fw * 0.5, fy0 - 0.03, fy0, fw * 0.5)
        if spec.get("transom", False):
            tz = z1 - 0.28 * h
            geo.box(frame, x0 + fw, x1 - fw, fy0 - 0.01, fy1, tz - fw / 2,
                    tz + fw / 2)

    # ---- shutters
    shutter = spec.get("shutter")
    if shutter == "roller":
        box_h = spec.get("roller_box", 0.0)
        state = rng.random()
        if state < 0.55:
            frac = rng.uniform(0.0, 0.08)
        elif state < 0.88:
            frac = rng.uniform(0.2, 0.65)
        else:
            frac = 1.0
        # guide rails
        geo.box("rail", x0, x0 + 0.035, 0.02, 0.07, z0, z1)
        geo.box("rail", x1 - 0.035, x1, 0.02, 0.07, z0, z1)
        if frac > 0.02:
            geo.box("roller", x0 + 0.03, x1 - 0.03, 0.04, 0.06,
                    z1 - frac * h, z1, rnd=rng.random())
        if box_h > 0:
            geo.box("roller", x0 - 0.02, x1 + 0.02, -0.005, 0.03, z1,
                    z1 + box_h, rnd=0.5)
    elif shutter in ("louvre", "faux", "panel"):
        state = spec.get("shutter_state")
        r = rng.random()
        if state is None:
            state = "open" if r < 0.62 else ("closed" if r < 0.85 else "half")
        if shutter == "faux":
            state = "open"
        sw = w / 2.0 if shutter != "faux" else min(0.5, w * 0.42)
        sm = "shutter"
        tint = rng.random() * 0.25 + spec.get("shutter_tint", 0.0)
        left_open = state in ("open", "half")
        right_open = state == "open"
        for side, is_open in (("l", left_open), ("r", right_open)):
            if is_open:
                sx0 = x0 - sw - 0.03 if side == "l" else x1 + 0.03
                geo.box(sm, sx0, sx0 + sw, -0.045, -0.005, z0 - 0.02,
                        z1 + 0.02, rnd=tint, rnd2=0.0)
            else:
                sx0 = x0 if side == "l" else x0 + w / 2
                geo.box(sm, sx0 + 0.005, sx0 + w / 2 - 0.005, 0.02, 0.05,
                        z0 + 0.005, z1 - 0.005, rnd=tint, rnd2=1.0)
    # ---- security bars / grilles
    bars = spec.get("bars")
    if bars:
        by0, by1 = (-0.10, -0.08) if bars == "cage" else (0.015, 0.03)
        n = max(3, int(round(w / 0.12)))
        for i in range(1, n):
            bx = x0 + w * i / n
            geo.box("bar", bx - 0.008, bx + 0.008, by0, by1, z0, z1)
        rows_b = 2 if bars != "cage" else max(2, int(h / 0.45))
        for i in range(1, rows_b + 1):
            bz = z0 + h * i / (rows_b + 1)
            geo.box("bar", x0, x1, by0 - 0.004, by1, bz - 0.01, bz + 0.01)
        if bars == "cage":
            geo.box("bar", x0 - 0.03, x0, -0.10, 0.0, z0, z1)
            geo.box("bar", x1, x1 + 0.03, -0.10, 0.0, z0, z1)
            geo.box("bar", x0 - 0.03, x1 + 0.03, -0.10, 0.0, z0 - 0.03, z0)
            geo.box("bar", x0 - 0.03, x1 + 0.03, -0.10, 0.0, z1, z1 + 0.03)
    # ---- sill
    if sill:
        proud = spec.get("sill_proud", 0.05)
        sm = spec.get("sill_mat", "sill")
        geo.box(sm, x0 - 0.05, x1 + 0.05, -proud, d * 0.5, z0 - 0.055, z0,
                bottom=True)
    surround = spec.get("surround")
    if surround:
        sw_ = surround
        geo.box("surround", x0 - sw_, x1 + sw_, -0.018, 0.0, z1, z1 + sw_)
        geo.box("surround", x0 - sw_, x0, -0.018, 0.0, z0, z1)
        geo.box("surround", x1, x1 + sw_, -0.018, 0.0, z0, z1)
    lintel = spec.get("lintel")
    if lintel:
        geo.box("lintel", x0 - 0.08, x1 + 0.08, -0.012, 0.0, z1,
                z1 + lintel)


def ac_unit(geo, rng, x0, z0, *, proud=0.32, w=0.82, h=0.56):
    """Split-system outdoor unit on a bracket (fan face is a material)."""
    geo.box("ac", x0, x0 + w, -proud, -0.02, z0, z0 + h, rnd=rng.random())
    geo.box("ac_fan", x0 + 0.08, x0 + 0.08 + h * 0.8, -proud - 0.004,
            -proud, z0 + 0.1 * h, z0 + 0.9 * h, rnd=rng.random())
    geo.box("bar", x0 + 0.05, x0 + 0.08, -proud, 0.0, z0 - 0.05, z0)
    geo.box("bar", x0 + w - 0.08, x0 + w - 0.05, -proud, 0.0, z0 - 0.05, z0)


def door(geo, rng, x0, x1, z0, z1, spec):
    d = spec.get("door_depth", 0.18)
    reveal(geo, spec.get("reveal_mat", "reveal"), x0, x1, z0, z1, d,
           bottom=False)
    kind = spec.get("door", "panel")
    m = "door"
    tint = rng.random()
    if kind == "glass":
        _curtains(geo, rng, x0 + 0.06, x1 - 0.06, z0 + 0.06, z1 - 0.06,
                  d + 0.12, "mixed")
        _frame_rect(geo, "frame", x0, x1, z0, z1, d - 0.02, d + 0.05, 0.07)
        geo.box("frame", x0, x1, d - 0.03, d + 0.05, z0 + 0.85, z0 + 0.95)
    else:
        geo.rect_front(m, x0, x1, z0, z1, y=d, rnd=tint)
        # raised panels
        w = x1 - x0
        h = z1 - z0
        fan = spec.get("fanlight", True) and h > 2.2
        top = z1 - (0.35 if fan else 0.0)
        for (px0, px1) in ((x0 + 0.1, x0 + w / 2 - 0.05),
                           (x0 + w / 2 + 0.05, x1 - 0.1)) if w > 0.8 else \
                ((x0 + 0.1, x1 - 0.1),):
            geo.box(m, px0, px1, d - 0.015, d, z0 + 0.15, z0 + 0.45 * h,
                    rnd=tint)
            geo.box(m, px0, px1, d - 0.015, d, z0 + 0.52 * h, top - 0.12,
                    rnd=tint)
        if fan:
            _curtains(geo, rng, x0, x1, top, z1, d + 0.1, "bare")
            geo.box("frame", x0, x1, d - 0.03, d + 0.02, top - 0.05, top)
            geo.face("interior_floor", [(x0, d + 0.02, top), (x1, d + 0.02, top),
                                        (x1, d + 0.02, z1), (x0, d + 0.02, z1)],
                     facing=(0, -1, 0))
        geo.box("metal", x1 - 0.14, x1 - 0.1, d - 0.06, d, z0 + 1.0,
                z0 + 1.08)
    # threshold step
    geo.box("step", x0 - 0.15, x1 + 0.15, -0.28, d, z0 - 0.18, z0)
    if spec.get("canopy"):
        geo.box("canopy", x0 - 0.35, x1 + 0.35, -0.75, 0.0, z1 + 0.18,
                z1 + 0.28, rnd=0.5)


def garage(geo, rng, x0, x1, z0, z1, spec):
    d = 0.12
    reveal(geo, "reveal", x0, x1, z0, z1, d, bottom=False)
    tint = rng.random()
    n = 4
    h = z1 - z0
    for i in range(n):
        a = z0 + h * i / n
        b = z0 + h * (i + 1) / n
        geo.box("garage", x0, x1, d - 0.03, d, a + 0.012, b - 0.012, rnd=tint)
        geo.rect_front("garage", x0, x1, a, b, y=d, rnd=tint)
    geo.box("frame", x0 - 0.1, x1 + 0.1, -0.02, 0.0, z1, z1 + 0.1)
    geo.box("step", x0, x1, -0.6, d, z0 - 0.12, z0)


def shopfront(geo, rng, x0, x1, z0, z1, spec, *, sign_rnd=0.0):
    """Glazed shop bay with fascia board; roller shutter on some."""
    d = 0.22
    reveal(geo, "reveal", x0, x1, z0, z1, d, bottom=False)
    _curtains(geo, rng, x0, x1, z0 + 0.4, z1, d + 0.3, "bare")
    fr = "shopframe"
    _frame_rect(geo, fr, x0, x1, z0, z1, d - 0.02, d + 0.05, 0.06)
    geo.box(fr, x0 + 0.05, x1 - 0.05, d - 0.03, d + 0.05, z0 + 0.35,
            z0 + 0.45)
    geo.rect_front("stallriser", x0, x1, z0, z0 + 0.4, y=d - 0.03)
    n = max(1, int(round((x1 - x0) / 1.4)))
    for i in range(1, n):
        mx = x0 + (x1 - x0) * i / n
        geo.box(fr, mx - 0.03, mx + 0.03, d - 0.03, d + 0.05, z0 + 0.4, z1)
    if spec.get("shop_shutter") and rng.random() < spec["shop_shutter"]:
        frac = rng.choice((1.0, 1.0, rng.uniform(0.3, 0.7)))
        geo.box("roller", x0 + 0.02, x1 - 0.02, 0.05, 0.07,
                z1 - frac * (z1 - z0), z1, rnd=rng.random())
        geo.box("rail", x0, x0 + 0.05, 0.02, 0.08, z0, z1)
        geo.box("rail", x1 - 0.05, x1, 0.02, 0.08, z0, z1)
    # fascia sign board (colour picked per shop via rnd)
    geo.box("sign", x0 - 0.05, x1 + 0.05, -0.12, 0.0, z1 + 0.08,
            z1 + spec.get("fascia_h", 0.55), rnd=sign_rnd)
    geo.box("step", x0, x1, -0.25, d, z0 - 0.15, z0)


def loggia(geo, rng, x0, x1, z0, z1, spec):
    """Recessed balcony: deep reveal, back wall with door + window, and a
    balustrade (solid, railing or glazed enclosure) at the wall plane."""
    d = spec.get("loggia_depth", 1.3)
    reveal(geo, "reveal", x0, x1, z0, z1, d)
    # back wall with a glazed door and a window into the interior
    w = x1 - x0
    dw = min(0.95, 0.4 * w)
    dx0 = x0 + 0.25 * w - dw / 2
    wx0, wx1 = x0 + 0.55 * w, x1 - 0.2
    ops = [(dx0, dx0 + dw, z0, z0 + 2.15), (wx0, wx1, z0 + 0.9, z0 + 2.15)]
    wall_with_openings(geo, "loggia_back", x0, x1, z0, z1, ops, y=d)
    for ox0, ox1, oz0, oz1 in ops:
        _curtains(geo, rng, ox0 + 0.06, ox1 - 0.06, oz0 + 0.06, oz1 - 0.06,
                  d + 0.15, "mixed")
        _frame_rect(geo, "frame", ox0, ox1, oz0, oz1, d + 0.02, d + 0.08,
                    0.06)
    # a few things people keep on balconies
    if rng.random() < 0.5:
        px = rng.uniform(x0 + 0.2, x1 - 0.8)
        geo.box("clutter", px, px + rng.uniform(0.3, 0.6), d - 0.5, d - 0.1,
                z0, z0 + rng.uniform(0.4, 0.9), rnd=rng.random())
    kind = spec.get("balustrade", "solid")
    top = z0 + 1.05
    if kind == "solid":
        geo.box("wall", x0, x1, -0.02, 0.12, z0, top, bevel=0.0)
        geo.box("sill", x0 - 0.02, x1 + 0.02, -0.06, 0.14, top, top + 0.06)
    elif kind == "rail":
        geo.box("bar", x0, x1, -0.04, 0.0, top - 0.05, top)
        n = max(3, int(w / 0.12))
        for i in range(1, n):
            bx = x0 + w * i / n
            geo.box("bar", bx - 0.01, bx + 0.01, -0.03, -0.01, z0 + 0.05,
                    top - 0.05)
        geo.box("sill", x0, x1, -0.08, 0.1, z0 - 0.05, z0 + 0.05)
    elif kind == "glass":
        geo.box("rail_glass", x0 + 0.02, x1 - 0.02, -0.02, 0.0, z0 + 0.1,
                top, rnd=rng.random())
        geo.box("bar", x0, x1, -0.04, 0.0, top, top + 0.04)
        geo.box("sill", x0, x1, -0.08, 0.1, z0 - 0.05, z0 + 0.05)
    elif kind == "enclosed":
        # enclosed balcony (common in East Asia): a glazed slider band
        # sits on a solid parapet at the wall plane
        geo.box("wall", x0, x1, -0.02, 0.12, z0, z0 + 0.9, bevel=0.0)
        fx0, fx1, fz0, fz1 = x0, x1, z0 + 0.9, z1
        _frame_rect(geo, "frame", fx0, fx1, fz0, fz1, -0.01, 0.05, 0.06)
        n = max(2, int(round(w / 0.9)))
        for i in range(1, n):
            bx = fx0 + w * i / n
            geo.box("frame", bx - 0.03, bx + 0.03, -0.03, 0.05, fz0, fz1)
        geo.box("sill", x0 - 0.02, x1 + 0.02, -0.06, 0.14, z0 + 0.86,
                z0 + 0.92)


def dock_door(geo, rng, x0, x1, z0, z1, spec):
    """Industrial sectional / roller door with a bump-rail apron."""
    d = 0.15
    reveal(geo, "reveal", x0, x1, z0, z1, d, bottom=False)
    tint = rng.random()
    state = rng.random()
    frac = 1.0 if state < 0.75 else rng.uniform(0.25, 0.8)
    top_open = z1 - (z1 - z0) * frac
    if frac < 1.0:
        _curtains(geo, rng, x0, x1, z0, top_open, d + 0.5, "bare")
    n = max(4, int((z1 - z0) / 0.5))
    for i in range(n):
        a = z1 - (z1 - z0) * (i + 1) / n
        b = z1 - (z1 - z0) * i / n
        if b <= top_open - 1e-6:
            continue
        a = max(a, top_open)
        geo.box("dock", x0, x1, d - 0.03, d, a + 0.01, b - 0.01, rnd=tint,
                bevel=0.008)
    geo.box("rail", x0 - 0.12, x0, -0.1, 0.0, z0, z1)
    geo.box("rail", x1, x1 + 0.12, -0.1, 0.0, z0, z1)
    geo.box("dark", x0 - 0.15, x0 + 0.1, -0.35, 0.0, z0, z0 + 0.5)
    geo.box("dark", x1 - 0.1, x1 + 0.15, -0.35, 0.0, z0, z0 + 0.5)
    if spec.get("dock_canopy", True):
        geo.box("canopy", x0 - 0.4, x1 + 0.4, -1.2, 0.0, z1 + 0.15,
                z1 + 0.3)


# ------------------------------------------------------------- facades

def motif_len(scheme):
    """Upper floors repeat a short bay motif (real facades are regular).

    Stacked floors shift by whole motifs, so windows stay aligned while
    each floor shows a different stretch of the strip's wear.
    """
    return int(scheme.get("motif", 3))


def bay_count(W, bay, motif):
    m = max(1, int(motif))
    return max(4 if m == 1 else m, m * int(round(W / (bay * m))))


def _upper_motif(rng, scheme):
    m = motif_len(scheme)
    blank_p = scheme.get("blank_p", 0.06)
    small_p = scheme.get("small_p", 0.10)
    kinds = []
    for _ in range(m):
        if rng.random() < scheme.get("loggia_p", 0.0):
            kinds.append(("loggia", 1))
            continue
        r = rng.random()
        kinds.append(("blank", 1) if r < blank_p else
                     ("small", 1) if r < blank_p + small_p else ("win", 1))
    if not any(k in ("win", "loggia") for k, _ in kinds):
        kinds[0] = ("win", 1)
    return kinds


def _bay_table(rng, n, level, scheme):
    """Per-bay content for one strip (wraps modulo n)."""
    if level == "upper":
        motif = _upper_motif(rng, scheme)
        return [motif[i % len(motif)] for i in range(n)]
    kinds = []
    ground = scheme.get("ground", "house")
    door_every = scheme.get("door_every", 4)
    blank_p = scheme.get("blank_p", 0.06)
    small_p = scheme.get("small_p", 0.10)
    i = 0
    next_door = rng.randrange(1, max(2, door_every))
    while i < n:
        if level == "ground" and ground == "shop":
            kinds.append(("shop", 1))
            i += 1
            continue
        if level == "upper" and rng.random() < scheme.get("loggia_p", 0.0):
            kinds.append(("loggia", 1))
            i += 1
            continue
        if level == "ground" and i == next_door:
            if ground == "industrial" and i + 1 < n:
                kinds.append(("dock", 2))
                i += 2
            elif ground == "garage" and i + 1 < n and rng.random() < 0.55:
                kinds.append(("garage", 2))
                i += 2
            else:
                kinds.append(("door", 1))
                i += 1
            next_door = i + rng.randrange(max(2, door_every - 1),
                                          door_every + 2)
            continue
        r = rng.random()
        if r < blank_p:
            kinds.append(("blank", 1))
        elif r < blank_p + small_p:
            kinds.append(("small", 1))
        else:
            kinds.append(("win", 1))
        i += 1
    return kinds


def facade_strip(scheme, level, W, H, seed, margin_bays=1):
    """One floor of a facade (``level`` = "upper" or "ground")."""
    rng = random.Random(seed)
    geo = Geo()
    n = bay_count(W, scheme["bay"], motif_len(scheme))
    p = W / n
    table = _bay_table(rng, n, level, scheme)
    # expand multi-bay entries into a per-slot list of (kind, span, start)
    slots = []
    for kind, span in table:
        slots.append((kind, span))
    # positions of each entry
    entries = []
    x = 0.0
    for kind, span in slots:
        entries.append((x, kind, span))
        x += span * p
    win = scheme["window"]
    sill_z = win.get("sill_z", 0.9) + (scheme.get("ground_raise", 0.0)
                                       if level == "ground" else 0.0)
    wh = win["h"]
    ww = win["w"]
    openings = []
    jobs = []
    table_openings = []     # (kind, (x0, x1, z0, z1), entry) of the strip itself
    per_bay_rng = [random.Random(seed * 7919 + k) for k in range(len(entries))]
    shop_sign = 0.0
    for rep in range(-margin_bays, 1 + margin_bays):
        # rep = -1 copies the tail of the strip to the left of x = 0 etc.
        for k, (x0e, kind, span) in enumerate(entries):
            xo = x0e + rep * W
            if xo + span * p < -margin_bays * p or xo > W + margin_bays * p:
                continue
            brng = random.Random(seed * 7919 + k)  # identical across reps
            cx = xo + span * p / 2.0
            n_open = len(openings)
            if kind == "win" or (kind == "small"):
                w_ = ww if kind == "win" else ww * 0.55
                h_ = wh if kind == "win" else wh * 0.55
                z0 = sill_z if kind == "win" else sill_z + wh - h_
                o = (cx - w_ / 2, cx + w_ / 2, z0, z0 + h_)
                openings.append(o)
                jobs.append(("window", o, brng, kind))
            elif kind == "door":
                dw = scheme.get("door_w", 1.0)
                dh = scheme.get("door_h", 2.25)
                dz = scheme.get("door_z", 0.15)
                o = (cx - dw / 2, cx + dw / 2, dz, dz + dh)
                openings.append(o)
                jobs.append(("door", o, brng, kind))
            elif kind == "garage":
                gw = min(span * p - 0.6, 2.6 * (span - 0.5) + 0.3)
                o = (cx - gw / 2, cx + gw / 2, 0.02, 2.2)
                openings.append(o)
                jobs.append(("garage", o, brng, kind))
            elif kind == "loggia":
                pier = scheme.get("loggia_pier", 0.35)
                o = (xo + pier / 2, xo + span * p - pier / 2, 0.12, H - 0.42)
                openings.append(o)
                jobs.append(("loggia", o, brng, kind))
            elif kind == "dock":
                dw = min(span * p - 0.8, scheme.get("dock_w", 3.6))
                o = (cx - dw / 2, cx + dw / 2, 0.02, H - 0.35)
                openings.append(o)
                jobs.append(("dock", o, brng, kind))
            elif kind == "shop":
                pier = scheme.get("shop_pier", 0.35)
                top = scheme.get("shop_top", 2.75)
                o = (xo + pier / 2, xo + span * p - pier / 2, 0.15, top)
                openings.append(o)
                if k % scheme.get("shop_run", 2) == 0:
                    shop_sign = brng.random()
                jobs.append(("shop", o, brng, shop_sign))
            if rep == 0 and len(openings) > n_open:
                table_openings.append((kind, openings[-1], k))
    x_lo, x_hi = -margin_bays * p - 0.5, W + margin_bays * p + 0.5
    wall_with_openings(geo, "wall", x_lo, x_hi, 0.0, H, openings)
    _interior_volume(geo, W, H, margin_bays * p + 0.5)
    ac_p = scheme.get("ac_p", 0.0)
    for job, o, brng, extra in jobs:
        x0, x1, z0, z1 = o
        if job == "window":
            spec = dict(win)
            if extra == "small":
                spec["muntins"] = (1, 1)
                spec["shutter"] = None if win.get("shutter") == "faux" else \
                    win.get("shutter")
            window(geo, brng, x0, x1, z0, z1, spec)
            if ac_p and brng.random() < ac_p:
                side = brng.choice((-1, 1))
                ax = (x1 + 0.12) if side > 0 else (x0 - 0.94)
                az = z0 + brng.uniform(-0.4, 0.2) if level == "upper" else \
                    z0 + 0.5
                ac_unit(geo, brng, ax, max(0.2, az))
        elif job == "door":
            door(geo, brng, x0, x1, z0, z1, scheme)
        elif job == "garage":
            garage(geo, brng, x0, x1, z0, z1, scheme)
        elif job == "shop":
            shopfront(geo, brng, x0, x1, z0, z1, scheme, sign_rnd=extra)
        elif job == "loggia":
            loggia(geo, brng, x0, x1, z0, z1, scheme)
        elif job == "dock":
            dock_door(geo, brng, x0, x1, z0, z1, scheme)
    # plinth / base course on ground floors, string course on uppers
    if level == "ground" and scheme.get("plinth"):
        ph = scheme["plinth"]
        cut = sorted(o for o in openings if o[2] < ph)
        x = x_lo
        for o in cut + [(x_hi, x_hi, 0, 0)]:
            if o[0] > x:
                geo.box("plinth", x, o[0], -0.025, 0.0, 0.0, ph, top=True)
            x = max(x, o[1])
    if level == "upper" and scheme.get("string_course"):
        sc = scheme["string_course"]
        geo.box("surround", x_lo, x_hi, -0.03, 0.0, 0.0, sc)
    return geo, {"bays": n, "bay_m": p, "openings": table_openings}


def plain_strip(scheme, W, H, seed, vents=True):
    """Windowless wall of a scheme: gable ends, far-LOD boxes, chimneys."""
    rng = random.Random(seed)
    geo = Geo()
    openings = []
    jobs = []
    if vents and scheme.get("gable_vent", True):
        n = max(1, int(W / 7.0))
        for i in range(n):
            cx = (i + 0.5) * W / n
            for rep in (-1, 0, 1):
                x = cx + rep * W
                o = (x - 0.25, x + 0.25, H * 0.55, H * 0.55 + 0.35)
                openings.append(o)
                jobs.append(o)
    wall_with_openings(geo, "wall", -2.0, W + 2.0, 0.0, H, openings)
    for (x0, x1, z0, z1) in jobs:
        reveal(geo, "reveal", x0, x1, z0, z1, 0.06)
        for k in range(5):
            z = z0 + (k + 0.5) * (z1 - z0) / 5
            geo.box("rail", x0, x1, 0.03, 0.06, z - 0.02, z + 0.02)
        geo.rect_front("interior", x0, x1, z0, z1, y=0.07)
    return geo, {}


# ---------------------------------------------------------------- roofs

def _periodic_cols(W, target):
    n = max(1, int(round(W / target)))
    return n, W / n


def roof_strip(roof, W, H, seed):
    """Roof covering laid in strip space (z = up-slope)."""
    rng = random.Random(seed)
    geo = Geo()
    kind = roof["kind"]
    margin = 1.0
    if "course" in roof or kind in ("clay_interlock", "concrete_tile",
                                    "barrel", "slate", "shingle", "kawara"):
        # whole courses per strip height so split roof rows line up
        default = {"slate": 0.22, "shingle": 0.14, "kawara": 0.24,
                   "barrel": 0.36}.get(kind, 0.34)
        c = roof.get("course", default)
        roof = dict(roof, course=H / max(1, round(H / c)))
    # an underlay so gaps read dark rather than as holes
    geo.rect_front("underlay", -margin - 1, W + margin + 1, -0.5, H + 0.5,
                   y=0.05)
    if (roof.get("mat") or {}).get("photo"):
        # a photoscanned covering carries its own tiles: one flat sheet
        geo.rect_front("tile", -margin - 1, W + margin + 1, -0.5, H + 0.5,
                       y=0.0, rnd=0.5)
    elif kind in ("clay_interlock", "concrete_tile"):
        n, tw = _periodic_cols(W, roof.get("tile_w", 0.30))
        course = roof.get("course", 0.34)
        rows = int(math.ceil((H + 0.6) / course)) + 1
        roll = roof.get("roll", 0.035)
        # cross-section: flat pan + one roll on the right
        sec = [(0.0, 0.0), (tw * 0.55, 0.0), (tw * 0.62, -roll * 0.5),
               (tw * 0.75, -roll), (tw * 0.88, -roll * 0.5), (tw, 0.0)]
        for r in range(rows):
            z0 = -0.4 + r * course
            for c in range(-3, n + 3):
                ci = c % n
                trng = random.Random(seed * 1_000_003 + r * 7919 + ci)
                x = c * tw
                lift = -0.018 - trng.uniform(0.0, 0.008)
                geo.profile_sweep("tile", sec, x, z0, z0 + course + 0.08,
                                  lift, 0.0, rnd=trng.random(),
                                  rnd2=trng.random(), lift=0.03, nose=0.035)
    elif kind in ("barrel",):
        n, tw = _periodic_cols(W, roof.get("tile_w", 0.36))
        course = roof.get("course", 0.36)
        rows = int(math.ceil((H + 0.6) / course)) + 1
        rad = tw * 0.27
        conv = circle_section(rad, 0, 180, 7)
        conc = [(x, -y * 0.65 + 0.0) for x, y in
                circle_section(rad * 1.25, 180, 360, 7)]
        for r in range(rows):
            z0 = -0.4 + r * course
            for c in range(-3, n + 3):
                ci = c % n
                trng = random.Random(seed * 1_000_003 + r * 7919 + ci)
                x = c * tw
                j = trng.uniform(-0.015, 0.015)
                # channel (concave) tile under, cap (convex) tile over
                geo.profile_sweep("tile", [(p[0], p[1] + 0.012) for p in conc],
                                  x + tw * 0.25 + j, z0, z0 + course + 0.07,
                                  -0.01, 0.0, rnd=trng.random(),
                                  rnd2=trng.random(), lift=0.02, nose=0.02)
                geo.profile_sweep("tile", conv, x + tw * 0.75 + j,
                                  z0 + trng.uniform(-0.02, 0.02),
                                  z0 + course + 0.07, -0.02, -0.005,
                                  rnd=trng.random(), rnd2=trng.random(),
                                  lift=0.035, nose=0.035)
    elif kind in ("slate", "shingle"):
        is_sh = kind == "shingle"
        course = roof.get("course", 0.14 if is_sh else 0.22)
        piece = roof.get("piece_w", 0.33 if is_sh else 0.28)
        rows = int(math.ceil((H + 0.6) / course)) + 1
        n, pw = _periodic_cols(W, piece)
        for r in range(rows):
            z0 = -0.4 + r * course
            off = (pw * 0.5) if r % 2 else 0.0
            if is_sh:
                off = pw * (0.5 if r % 2 else 0.0) + pw * 0.25 * ((r // 2) % 2)
            for c in range(-3, n + 3):
                ci = c % n
                trng = random.Random(seed * 1_000_003 + r * 7919 + ci)
                x0 = c * pw + off + trng.uniform(-0.004, 0.004)
                gap = 0.006 if is_sh else 0.004
                zj = trng.uniform(-0.01, 0.01) if not is_sh else 0.0
                geo.tilted_box("tile", x0 + gap, x0 + pw - gap, z0 + zj,
                               z0 + course * 2.2, -0.012 - trng.uniform(
                                   0, 0.004), 0.0, 0.01,
                               rnd=trng.random(), rnd2=trng.random())
    elif kind in ("corrugated", "trapezoid", "fibre_cement", "standing_seam"):
        sheet_w = roof.get("sheet_w", 0.76)
        n, sw = _periodic_cols(W, sheet_w)
        sheet_len = roof.get("sheet_len", 2.4)
        amp = roof.get("amp", 0.018)
        period = roof.get("period", 0.076)
        seg = 4 if kind != "standing_seam" else 1
        for c in range(-2, n + 2):
            ci = c % n
            z = -0.5
            row = 0
            while z < H + 0.5:
                trng = random.Random(seed * 1_000_003 + row * 7919 + ci)
                x0 = c * sw
                waves = max(1, int(round(sw / period)))
                sec = []
                for i in range(waves * seg + 1):
                    t = i / (waves * seg)
                    xx = t * (sw + period * 0.5)
                    if kind == "corrugated" or kind == "fibre_cement":
                        yy = -amp * 0.5 * (1 - math.cos(2 * math.pi * t * waves))
                    elif kind == "trapezoid":
                        ph = (t * waves) % 1.0
                        yy = -amp if 0.35 < ph < 0.6 else (
                            -amp * (ph - 0.25) / 0.1 if 0.25 < ph <= 0.35 else
                            (-amp * (0.7 - ph) / 0.1 if 0.6 <= ph < 0.7 else 0))
                    else:
                        yy = 0.0
                    sec.append((xx, yy))
                ln = sheet_len * trng.uniform(0.9, 1.1)
                geo.profile_sweep("sheet", sec, x0, z, z + ln + 0.12,
                                  -0.004 * (row % 2), 0.0,
                                  rnd=trng.random(), rnd2=trng.random(),
                                  lift=0.004)
                if kind == "standing_seam":
                    geo.box("sheet", x0 - 0.012, x0 + 0.012, -0.035, 0.0, z,
                            z + ln + 0.12, rnd=trng.random())
                z += ln
                row += 1
    elif kind == "kawara":
        n, tw = _periodic_cols(W, roof.get("tile_w", 0.30))
        course = roof.get("course", 0.24)
        rows = int(math.ceil((H + 0.6) / course)) + 1
        # S-profile: a trough then a roll, one continuous section per tile
        sec = []
        for i in range(13):
            t = i / 12
            sec.append((t * tw, -0.028 * math.sin(math.pi * 2 * t - math.pi / 2)
                        - 0.028))
        for r in range(rows):
            z0 = -0.4 + r * course
            for c in range(-3, n + 3):
                ci = c % n
                trng = random.Random(seed * 1_000_003 + r * 7919 + ci)
                geo.profile_sweep("tile", sec, c * tw, z0,
                                  z0 + course + 0.06, -0.022, -0.002,
                                  rnd=trng.random(), rnd2=trng.random(),
                                  lift=0.03, nose=0.03)
    elif kind == "membrane":
        # flat roof: membrane sheets with lap seams, a few patches
        n, sw = _periodic_cols(W, roof.get("sheet_w", 1.0))
        geo.rect_front("tile", -margin - 1, W + margin + 1, -0.5, H + 0.5,
                       y=0.0, rnd=0.5)
        for c in range(-2, n + 2):
            x = c * sw
            geo.box("seam", x - 0.03, x + 0.03, -0.006, 0.0, -0.5, H + 0.5,
                    rnd=0.5)
        for k in range(int(W * H / 40) if roof.get("patches", True) else 0):
            prng = random.Random(seed * 31 + k)
            px = prng.uniform(0, W)
            pz = prng.uniform(0, H)
            s = prng.uniform(0.4, 1.4)
            for rep in (-W, 0, W):
                geo.box("patch", px + rep, px + rep + s, -0.004, 0.0, pz,
                        pz + s * prng.uniform(0.5, 1.2), rnd=prng.random())
    elif kind == "gravel":
        geo.rect_front("tile", -margin - 1, W + margin + 1, -0.5, H + 0.5,
                       y=0.0, rnd=0.5)
    elif kind == "concrete_slab":
        n, sw = _periodic_cols(W, roof.get("panel", 3.0))
        geo.rect_front("tile", -margin - 1, W + margin + 1, -0.5, H + 0.5,
                       y=0.0, rnd=0.5)
        for c in range(-2, n + 2):
            x = c * sw
            geo.box("seam", x - 0.01, x + 0.01, 0.0, 0.01, -0.5, H + 0.5)
        zr = 0.0
        while zr < H + 0.5:
            geo.box("seam", -margin - 1, W + margin + 1, 0.0, 0.01, zr - 0.01,
                    zr + 0.01)
            zr += roof.get("panel", 3.0)
    else:
        raise ValueError(f"unknown roof kind {kind}")
    if roof.get("objects"):
        roof_objects(geo, random.Random(seed * 17 + 3), W, H,
                     roof["objects"], flat=kind in ("membrane", "gravel",
                                                    "concrete_slab"))
    return geo, {}


# ------------------------------------------------------ rooftop objects
# Seen from above and baked FLAT into the roof strip: no geometry reaches
# the OBJ.  In strip space a roof is a relief facing the camera, so an
# object's height above the roof is -y and "north" (up-slope) is +z.

OBJECT_SIZES = {
    "tank_steel": (1.4, 1.4), "tank_black": (1.3, 1.3),
    "tank_blue": (1.3, 1.3), "tank_green": (1.4, 1.4),
    "ac": (1.0, 0.8), "ac_bank": (4.0, 1.2), "hatch": (2.6, 3.2),
    "solar_heater": (2.1, 2.4), "vent": (0.3, 0.3), "dish": (0.9, 0.9),
    "skylight": (0.8, 1.2), "shed": (3.5, 2.6),
}


def _disc(geo, mat, cx, cz, r, y, seg=18, rnd=0.0, facing=(0, -1, 0)):
    pts = [(cx + r * math.cos(2 * math.pi * i / seg), y,
            cz + r * math.sin(2 * math.pi * i / seg)) for i in range(seg)]
    geo.face(mat, pts, rnd, facing=facing)


def _cylinder_up(geo, mat, cx, cz, r, h, rnd=0.0, lid=None):
    """Vertical cylinder: top disc with a rounded rim (for the normal map)."""
    seg = 20
    rim = min(0.06, r * 0.2)
    ring_o = [(cx + r * math.cos(2 * math.pi * i / seg), -h + rim,
               cz + r * math.sin(2 * math.pi * i / seg)) for i in range(seg)]
    ring_i = [(cx + (r - rim) * math.cos(2 * math.pi * i / seg), -h,
               cz + (r - rim) * math.sin(2 * math.pi * i / seg))
              for i in range(seg)]
    for i in range(seg):
        j = (i + 1) % seg
        out = (ring_o[i][0] - cx, 0.0, ring_o[i][2] - cz)
        geo.face(mat, [ring_o[i], ring_o[j], ring_i[j], ring_i[i]], rnd,
                 facing=(out[0], -1.0, out[2]))
        geo.face(mat, [ring_o[i], ring_o[j], (ring_o[j][0], 0.0, ring_o[j][2]),
                       (ring_o[i][0], 0.0, ring_o[i][2])], rnd, facing=out)
    geo.face(mat, ring_i, rnd, facing=(0, -1, 0))
    if lid:
        _disc(geo, lid, cx, cz, (r - rim) * 0.45, -h - 0.02, rnd=rnd)


def _box_up(geo, mat, x0, x1, z0, z1, h, rnd=0.0, bevel=None):
    geo.box(mat, x0, x1, -h, 0.0, z0, z1, rnd=rnd, bevel=bevel)


def _place(occupied, x0, x1, z0, z1, pad=0.4):
    for a0, a1, b0, b1 in occupied:
        if x0 < a1 + pad and x1 > a0 - pad and z0 < b1 + pad and z1 > b0 - pad:
            return False
    occupied.append((x0, x1, z0, z1))
    return True


def roof_objects(geo, rng, W, H, spec, flat):
    """Scatter region-typical rooftop equipment over a roof strip.

    ``spec`` = {"per_100m2": density, "kinds": {kind: weight},
    "zmin"/"zmax": metres from the eave}.  Objects are placed on [0, W) and
    copied one strip-width left and right so the strip still wraps in u.
    Clusters: objects gather around a few seeds per strip, like real roofs
    where equipment sits together and whole roofs stay bare.
    """
    kinds = [(k, w) for k, w in spec["kinds"].items() if w > 0]
    if not kinds:
        return
    total = sum(w for _, w in kinds)
    zlo = spec.get("zmin", 0.8 if flat else 0.6)
    zhi = min(H - 0.8, spec.get("zmax", H - 0.8))
    area = W * max(0.5, zhi - zlo)
    n = int(round(area / 100.0 * spec.get("per_100m2", 1.0)))
    n_clusters = max(1, int(round(W / spec.get("cluster_every_m", 14.0))))
    centres = [rng.uniform(0.0, W) for _ in range(n_clusters)]
    spread = spec.get("cluster_spread_m", 6.0)
    occupied = []
    placed = []
    for _ in range(n * 4):
        if len(placed) >= n:
            break
        r = rng.random() * total
        kind = kinds[-1][0]
        for k, w in kinds:
            r -= w
            if r <= 0:
                kind = k
                break
        if kind == "panels":
            sx = rng.choice((3.6, 5.4, 7.2))
            sz = rng.choice((1.8, 2.8, 3.6))
        else:
            sx, sz = OBJECT_SIZES[kind]
        x0 = rng.choice(centres) + rng.gauss(0.0, spread)
        x0 %= W
        z0 = rng.uniform(zlo, max(zlo + 0.01, zhi - sz))
        if not _place(occupied, x0, x0 + sx, z0, z0 + sz):
            continue
        placed.append((kind, x0, z0, sx, sz, rng.random(), rng.random()))
    for kind, x0, z0, sx, sz, a, b in placed:
        for rep in (-W, 0.0, W):
            x = x0 + rep
            if x + sx < -2.0 or x > W + 2.0:
                continue
            _roof_object(geo, kind, x, z0, sx, sz, a, b, flat)


def _roof_object(geo, kind, x0, z0, sx, sz, a, b, flat):
    cx, cz = x0 + sx / 2, z0 + sz / 2
    if kind.startswith("tank"):
        if flat:
            # tanks sit on a small steel stand
            _box_up(geo, "obj_frame", x0 - 0.1, x0 + sx + 0.1, z0 - 0.1,
                    z0 + sz + 0.1, 0.35, rnd=a)
        _cylinder_up(geo, "obj_" + kind, cx, cz, sx / 2, 1.5 + 0.3 * b,
                     rnd=a, lid="obj_frame")
    elif kind == "ac":
        _box_up(geo, "obj_ac", x0, x0 + sx, z0, z0 + sz, 0.9, rnd=a)
        _disc(geo, "obj_fan", cx, cz, 0.3, -0.905, rnd=a)
    elif kind == "ac_bank":
        n = 3
        for i in range(n):
            xx = x0 + i * sx / n
            _box_up(geo, "obj_ac", xx + 0.05, xx + sx / n - 0.05, z0,
                    z0 + sz, 1.1, rnd=a)
            _disc(geo, "obj_fan", xx + sx / n / 2, cz, 0.35, -1.105, rnd=a)
    elif kind in ("hatch", "shed"):
        mat = "obj_hatch" if kind == "hatch" else "obj_shed"
        _box_up(geo, mat, x0, x0 + sx, z0, z0 + sz, 2.4, rnd=a, bevel=0.06)
        if kind == "shed":
            # corrugated lean-to roof: ribs across the top
            for i in range(int(sx / 0.2)):
                xx = x0 + i * 0.2 + 0.05
                geo.box("obj_shed", xx, xx + 0.06, -2.44, -2.4, z0, z0 + sz,
                        rnd=a, bevel=0.01)
    elif kind == "solar_heater":
        # tilted collector plus a horizontal tank along its top edge
        geo.face("obj_collector", [(x0, -0.3, z0), (x0 + sx, -0.3, z0),
                                   (x0 + sx, -1.4, z0 + sz * 0.7),
                                   (x0, -1.4, z0 + sz * 0.7)], a,
                 facing=(0, -1, -0.5))
        geo.box("obj_heater_tank", x0 - 0.05, x0 + sx + 0.05, -1.95, -1.3,
                z0 + sz * 0.7, z0 + sz, rnd=a, bevel=0.25)
        _box_up(geo, "obj_frame", x0, x0 + 0.08, z0, z0 + sz, 0.3, rnd=a)
        _box_up(geo, "obj_frame", x0 + sx - 0.08, x0 + sx, z0, z0 + sz, 0.3,
                rnd=a)
    elif kind == "vent":
        _cylinder_up(geo, "obj_vent", cx, cz, sx / 2, 0.45, rnd=a)
    elif kind == "dish":
        _box_up(geo, "obj_frame", cx - 0.03, cx + 0.03, cz - 0.3, cz, 0.6)
        _disc(geo, "obj_dish", cx, cz, sx / 2, -0.62, rnd=a,
              facing=(0.3, -1, 0.2))
    elif kind == "panels":
        nx = max(1, int(round(sx / 1.8)))
        nz = max(1, int(round(sz / 1.0)))
        for i in range(nx):
            for j in range(nz):
                px0 = x0 + i * sx / nx + 0.02
                pz0 = z0 + j * sz / nz + 0.02
                px1 = px0 + sx / nx - 0.04
                pz1 = pz0 + sz / nz - 0.04
                geo.box("obj_panel_frame", px0, px1, -0.08, 0.0, pz0, pz1,
                        rnd=a, bevel=0.01)
                geo.rect_front("obj_panel", px0 + 0.03, px1 - 0.03,
                               pz0 + 0.03, pz1 - 0.03, y=-0.081, rnd=a,
                               rnd2=b)
    elif kind == "skylight":
        geo.box("obj_panel_frame", x0, x0 + sx, -0.12, 0.0, z0, z0 + sz,
                rnd=a, bevel=0.02)
        geo.rect_front("obj_skylight", x0 + 0.07, x0 + sx - 0.07, z0 + 0.07,
                       z0 + sz - 0.07, y=-0.121, rnd=a)

# ---------------------------------------------------------------- trims

def trim_band(bands, W, H, seed):
    """Stacked 0.4 m trim bands (fascia, soffit, ridge, chimney, ...).

    Band i occupies the i-th 0.4 m from the TOP of the strip so it lands
    on the same rows layout.page_layout() assigns it.
    """
    rng = random.Random(seed)
    geo = Geo()
    bh = H / len(bands)
    for i, name in enumerate(bands):
        z1 = H - i * bh
        z0 = z1 - bh
        m = f"trim_{name}"
        # keep every band's relief inside its own rows
        geo.rect_front(m, -2.0, W + 2.0, z0, z1, y=0.0, rnd=0.5)
        if name == "fascia":
            # gutter along the bottom of the board
            geo.box("trim_metal", -2.0, W + 2.0, -0.12, 0.0, z0 + 0.01,
                    z0 + 0.11)
            n = max(1, int(W / 9.0))
            for k in range(n):
                x = (k + 0.5) * W / n
                geo.box("trim_metal", x - 0.04, x + 0.04, -0.13, -0.11,
                        z0 + 0.02, z0 + 0.1)
        elif name == "ridge":
            n = max(1, int(round(W / 0.42)))
            p = W / n
            for k in range(-2, n + 2):
                geo.box(m, k * p + 0.01, (k + 1) * p + 0.02, -0.05, 0.0,
                        z0 + 0.06, z1 - 0.06, rnd=rng.random(),
                        rnd2=rng.random(), bevel=0.05)
        elif name == "coping":
            n = max(1, int(round(W / 1.0)))
            p = W / n
            for k in range(-2, n + 2):
                geo.box(m, k * p + 0.004, (k + 1) * p - 0.004, -0.03, 0.0,
                        z0 + 0.04, z1 - 0.04, rnd=rng.random(), bevel=0.02)
        elif name == "soffit":
            n = max(1, int(W / 6.0))
            for k in range(n):
                x = (k + 0.5) * W / n
                geo.box("trim_dark", x - 0.2, x + 0.2, -0.004, 0.0,
                        z0 + 0.12, z1 - 0.12, bevel=0.003)
    return geo
