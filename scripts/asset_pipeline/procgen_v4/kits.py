"""Regional architecture kits: materials, facade schemes, roofs, massing.

Pure data + small helpers.  A *scheme* describes one facade family (wall
material, window design, ground-floor treatment, bay rhythm); a *roof* one
covering.  ``materials_for(scheme)`` resolves every material name the strip
recipes use into a spec for bake_materials.build_material().

Weathering is NOT here: the bake produces clean, physically lit relief and
weather.py adds climate-driven wear from the masks afterwards (per region,
with asia forced clean -- see weather.CLEAN_REGIONS).
"""

from __future__ import annotations

import copy

from strips import (CLS_GLASS, CLS_METAL, CLS_OBJECT, CLS_PLINTH, CLS_ROOF,
                    CLS_SHUTTER, CLS_SIGN, CLS_SILL, CLS_TRIM, CLS_WALL, mat)

# ------------------------------------------------------------ materials

def plaster(colors, *, var=0.06, fine=0.04, drift=None, bump=0.6,
            rough=0.9):
    return mat(colors, noise=(0.35, var), fine=(7.0, fine), bump=bump,
               rough=rough, cls=CLS_WALL, bump_scale=22.0, bump_dist=0.008,
               **({"drift": drift} if drift else {}))


def brick(colors, mortar="#9b958c", *, w=0.225, h=0.075, var=0.22, alt=None,
          mortar_w=0.011, rough=0.9):
    spec = mat(colors, noise=(0.5, 0.05), fine=(12.0, 0.03),
               pattern={"type": "brick", "w": w, "h": h, "mortar": mortar_w,
                        "mortar_color": mortar, "var": var,
                        **({"alt": alt} if alt else {})},
               bump=0.6, rough=rough, cls=CLS_WALL)
    return spec


def siding(colors, *, board=0.2, shadow=0.28, rough=0.6):
    return mat(colors, noise=(0.4, 0.035), fine=(20.0, 0.02),
               pattern={"type": "siding", "h": board, "shadow": shadow},
               bump=0.5, rough=rough, cls=CLS_WALL)


def tiles(colors, grout="#b9b6ae", *, w=0.1, h=0.1, rough=0.35):
    return mat(colors, noise=(0.4, 0.03), fine=(30.0, 0.015),
               pattern={"type": "tiles", "w": w, "h": h, "mortar": 0.006,
                        "mortar_color": grout, "var": 0.06},
               bump=0.4, rough=rough, cls=CLS_WALL)


def concrete(colors, *, panel=None, rough=0.9, var=0.07):
    pattern = None
    if panel:
        pattern = {"type": "panels", "w": panel[0], "h": panel[1],
                   "mortar": 0.018, "mortar_color": "#6d6a66", "var": 0.04,
                   "offset": 0.0}
    return mat(colors, noise=(0.45, var), fine=(10.0, 0.05), pattern=pattern,
               bump=0.5, rough=rough, cls=CLS_WALL)


def paint(colors, *, rough=0.45, cls=CLS_TRIM, var=0.02, metal=0.0,
          pattern=None, piece_var=0.0, drip=0.0):
    spec = mat(colors, noise=(0.8, var), fine=(15.0, 0.012), rough=rough,
               cls=cls, metal=metal, pattern=pattern, drip=drip)
    spec["piece_var"] = piece_var
    return spec


BASE_MATS = {
    "reveal": None,                 # = wall (resolved in materials_for)
    "interior": mat(["#b9b3a8", "#a8a397", "#c4beb2", "#9ea3a6", "#b3a490"],
                    noise=(0.9, 0.2), fine=(3.0, 0.05), rough=0.1,
                    cls=CLS_GLASS),
    "interior_floor": mat(["#6b5a48"], noise=(1.0, 0.1), rough=0.1,
                          cls=CLS_GLASS),
    "curtain": mat(["#d8d2c4", "#c9bca3", "#9a8f7c", "#b3b8bc", "#7d6552",
                    "#e6e1d6", "#8c9a8c", "#a88c7a"], noise=(3.0, 0.08),
                   fine=(25.0, 0.04),
                   pattern={"type": "boards", "h": 0.09, "shadow": 0.25},
                   rough=0.15, cls=CLS_GLASS),
    "net": mat(["#e8e6df", "#dcd8cc"], noise=(3.0, 0.05), rough=0.15,
               cls=CLS_GLASS),
    "blind": mat(["#e4e2dc", "#cfc8b8", "#b5b2aa", "#e9e4d5"],
                 pattern={"type": "slats", "h": 0.035, "shadow": 0.35},
                 rough=0.15, cls=CLS_GLASS),
    "frame": paint(["#eeeeea"], rough=0.35),
    "shopframe": paint(["#3c3f42"], rough=0.35, metal=0.5),
    "rail": paint(["#9a9c9c"], rough=0.4, metal=0.6, cls=CLS_METAL),
    "roller": paint(["#d8d8d2", "#b8b8b2", "#8f9296", "#6f6a62"],
                    pattern={"type": "slats", "h": 0.045, "shadow": 0.3},
                    rough=0.5, cls=CLS_SHUTTER),
    "shutter": paint(["#355c3f"], pattern={"type": "slats", "h": 0.055,
                                           "shadow": 0.45},
                     rough=0.6, cls=CLS_SHUTTER, piece_var=0.12),
    "bar": paint(["#2b2b2b"], rough=0.5, metal=0.4, cls=CLS_METAL),
    "sill": paint(["#b9b8b2"], rough=0.6, cls=CLS_SILL, drip=1.0),
    "surround": None,               # = lighter wall
    "lintel": paint(["#a9a59c"], rough=0.8, cls=CLS_TRIM),
    "ac": paint(["#e6e6e1", "#dcdcd6", "#d2d3cf"], rough=0.4, cls=CLS_METAL,
                drip=1.0),
    "ac_fan": paint(["#3d3f41"], rough=0.6, cls=CLS_METAL,
                    pattern={"type": "slats", "h": 0.03, "shadow": 0.4}),
    "door": paint(["#6b4a33", "#3d4e5c", "#7a2f2a", "#e8e6df", "#2f3f33",
                   "#4a4a4a"], rough=0.5,
                  pattern={"type": "grain", "scale": 4.0, "amount": 0.06}),
    "metal": paint(["#b8a36a"], rough=0.3, metal=1.0, cls=CLS_METAL),
    "step": concrete(["#9d9a94"]),
    "canopy": paint(["#e6e6e0"], rough=0.5, drip=1.0),
    "garage": paint(["#ecebe6", "#d9d4c7", "#b9b3a6"],
                    pattern={"type": "siding", "h": 0.13, "shadow": 0.2},
                    rough=0.45),
    "stallriser": paint(["#4a4b4d"], rough=0.6),
    "sign": paint(["#b23a2e", "#2d5a8c", "#e0b830", "#2f7a4a", "#e8e4d8",
                   "#40404a", "#c7602c", "#8a2f55"], rough=0.35,
                  cls=CLS_SIGN),
    "plinth": concrete(["#6f6c68"], rough=0.9),
    "loggia_back": None,            # = wall
    "clutter": paint(["#7a6a55", "#4d5a4a", "#b0a590", "#5a4a44", "#8a8a88"],
                     rough=0.8, cls=CLS_TRIM),
    "rail_glass": mat(["#5d6a70"], noise=(1.0, 0.05), rough=0.1,
                      cls=CLS_GLASS),
    "dock": paint(["#c9ccce", "#3d5a7a", "#b8b2a2", "#7a7d80"],
                  pattern={"type": "slats", "h": 0.1, "shadow": 0.2},
                  rough=0.45, metal=0.5, cls=CLS_METAL),
    "dark": paint(["#262626"], rough=0.8, cls=CLS_TRIM),
    # rooftop equipment (baked flat into roof strips)
    "obj_tank_steel": paint(["#c9ccce", "#b9bcbe"], rough=0.3, metal=0.9,
                            cls=CLS_OBJECT, pattern={"type": "grain",
                                                     "scale": 20.0,
                                                     "amount": 0.08}),
    "obj_tank_black": paint(["#232323", "#2b2b2a"], rough=0.55,
                            cls=CLS_OBJECT),
    "obj_tank_blue": paint(["#2f5f9a", "#35679e"], rough=0.5, cls=CLS_OBJECT),
    "obj_tank_green": paint(["#3c5a3a", "#44613f"], rough=0.55,
                            cls=CLS_OBJECT),
    "obj_frame": paint(["#6a6a68"], rough=0.5, metal=0.6, cls=CLS_OBJECT),
    "obj_ac": paint(["#dcdcd6", "#d0d1cc", "#e4e3dd"], rough=0.45,
                    cls=CLS_OBJECT),
    "obj_fan": paint(["#2a2a2a"], rough=0.6, cls=CLS_OBJECT,
                     pattern={"type": "slats", "h": 0.03, "shadow": 0.5}),
    "obj_hatch": concrete(["#b0aca4", "#a39f97"]),
    "obj_shed": paint(["#8a9094", "#7d8488", "#9aa0a3", "#3d5f8a"],
                      rough=0.5, metal=0.7, cls=CLS_OBJECT),
    "obj_collector": paint(["#1d2a3a"], rough=0.15, metal=0.3,
                           cls=CLS_OBJECT, pattern={"type": "boards",
                                                    "h": 0.1,
                                                    "shadow": 0.4}),
    "obj_heater_tank": paint(["#e8e8e4", "#d8d8d2"], rough=0.35,
                             cls=CLS_OBJECT),
    "obj_vent": paint(["#8a8a88", "#9c9c98"], rough=0.45, metal=0.7,
                      cls=CLS_OBJECT),
    "obj_dish": paint(["#e6e6e2"], rough=0.4, cls=CLS_OBJECT),
    "obj_panel": paint(["#1c2a44", "#1a2640", "#20304a"], rough=0.12,
                       metal=0.2, cls=CLS_OBJECT,
                       pattern={"type": "tiles", "w": 0.16, "h": 0.16,
                                "mortar": 0.004, "mortar_color": "#8a96a8",
                                "var": 0.03}),
    "obj_panel_frame": paint(["#b8bcc0"], rough=0.3, metal=0.9,
                             cls=CLS_OBJECT),
    "obj_skylight": mat(["#3c4a58"], noise=(1.0, 0.05), rough=0.08,
                        cls=CLS_GLASS),
    "underlay": mat(["#1e1c1a"], noise=(1, 0.0), fine=(1, 0.0), rough=0.9,
                    cls=CLS_ROOF),
    "seam": paint(["#5a5a58"], rough=0.6, cls=CLS_ROOF),
    "patch": paint(["#3a3a3a", "#55524e", "#2d2d2d"], rough=0.6,
                   cls=CLS_ROOF),
}


def materials_for(scheme):
    """Resolve the full material table for a facade scheme."""
    mats = copy.deepcopy(BASE_MATS)
    for key, value in (scheme.get("mats") or {}).items():
        mats[key] = copy.deepcopy(value)
    wall = mats["wall"]
    if mats.get("reveal") is None:
        mats["reveal"] = copy.deepcopy(wall)
    if mats.get("loggia_back") is None:
        mats["loggia_back"] = copy.deepcopy(wall)
    if mats.get("surround") is None:
        sur = copy.deepcopy(wall)
        sur["base"] = [_lighten(c, 1.1) for c in sur["base"]]
        sur["pattern"] = None
        sur["cls"] = CLS_TRIM
        mats["surround"] = sur
    return mats


def roof_materials(roof):
    mats = copy.deepcopy(BASE_MATS)
    mats["tile"] = copy.deepcopy(roof["mat"])
    mats["sheet"] = copy.deepcopy(roof["mat"])
    if "seam" in roof:
        mats["seam"] = copy.deepcopy(roof["seam"])
    return mats


def _lighten(hex_color, factor):
    h = hex_color.lstrip("#")
    rgb = [min(255, int(int(h[i:i + 2], 16) * factor)) for i in (0, 2, 4)]
    return "#" + "".join(f"{c:02x}" for c in rgb)


def roof_mat(colors, *, piece_var=0.1, noise=0.08, fine=0.05, rough=0.7,
             metal=0.0, pattern=None, bump=0.3, drift=None):
    spec = mat(colors, noise=(0.25, noise), fine=(6.0, fine), rough=rough,
               metal=metal, cls=CLS_ROOF, pattern=pattern, bump=bump)
    spec["piece_var"] = piece_var
    spec["gloss_var"] = 0.1
    if drift:
        spec["drift"] = drift
    return spec
