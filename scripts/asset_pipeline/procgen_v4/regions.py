"""Regional architecture: facade schemes, roofs, trims and massing rules.

Each region owns two residential pages (colourways A/B), one apartment +
commercial page and draws one of three shared industrial pages by wear
grade.  The massing block drives the geometry builders (buildings.py):
roof forms and pitches, eave depth, chimneys, dormers, porches/verandas,
flat-roof share -- the silhouette cues that distinguish regions from the
air before any texture resolves.
"""

from __future__ import annotations

import copy

from kits import (brick, concrete, paint, plaster, roof_mat, siding, tiles)
from strips import CLS_ROOF, CLS_SHUTTER

# ------------------------------------------------------------- windows

W_EU = {"type": "casement", "w": 1.25, "h": 1.4, "sill_z": 0.95,
        "depth": 0.16, "frame_w": 0.075, "shutter": "roller",
        "muntins": (2, 1)}
W_EU_OLD = {"type": "casement", "w": 1.1, "h": 1.55, "sill_z": 0.85,
            "depth": 0.2, "frame_w": 0.075, "muntins": (2, 1),
            "transom": True, "lintel": 0.22}
W_NA = {"type": "double_hung", "w": 0.95, "h": 1.5, "sill_z": 0.8,
        "depth": 0.1, "frame_w": 0.09, "shutter": "faux",
        "muntins": (3, 2)}
W_NA_PLAIN = {"type": "double_hung", "w": 1.0, "h": 1.45, "sill_z": 0.85,
              "depth": 0.1, "frame_w": 0.09, "muntins": (1, 1)}
W_NA_WEST = {"type": "slider", "w": 1.5, "h": 1.2, "sill_z": 1.0,
             "depth": 0.12, "frame_w": 0.05}
W_MED = {"type": "casement", "w": 1.0, "h": 1.45, "sill_z": 0.9,
         "depth": 0.22, "frame_w": 0.07, "shutter": "louvre",
         "muntins": (2, 1), "sill_proud": 0.06}
W_ASIA = {"type": "slider", "w": 1.5, "h": 1.3, "sill_z": 0.9,
          "depth": 0.12, "frame_w": 0.05, "style": "clean_blinds"}
W_ASIA_GRILLE = dict(W_ASIA, bars="cage")
W_AF = {"type": "steel", "w": 1.2, "h": 1.3, "sill_z": 0.9, "depth": 0.14,
        "frame_w": 0.05, "muntins": (3, 3), "bars": "flat"}
W_SA = {"type": "casement", "w": 1.2, "h": 1.3, "sill_z": 0.95,
        "depth": 0.14, "frame_w": 0.06, "muntins": (2, 1), "bars": "flat"}
W_AU = {"type": "slider", "w": 1.6, "h": 1.2, "sill_z": 1.0, "depth": 0.12,
        "frame_w": 0.05}
W_AU_QLD = {"type": "casement", "w": 0.9, "h": 1.5, "sill_z": 0.85,
            "depth": 0.08, "frame_w": 0.07, "muntins": (1, 3)}
W_MODERN = {"type": "modern", "w": 1.7, "h": 1.55, "sill_z": 0.75,
            "depth": 0.18, "frame_w": 0.06, "muntins": (2, 1)}
W_OFFICE = {"type": "modern", "w": 2.6, "h": 1.9, "sill_z": 0.8,
            "depth": 0.12, "frame_w": 0.05, "muntins": (2, 1),
            "style": "clean_blinds"}
W_IND_HIGH = {"type": "modern", "w": 2.4, "h": 0.8, "sill_z": 2.0,
              "depth": 0.08, "frame_w": 0.05, "muntins": (3, 1)}


def frame(colour, rough=0.35, metal=0.0):
    return paint([colour], rough=rough, metal=metal)


def shutters(colours):
    return paint(colours if isinstance(colours, list) else [colours],
                 pattern={"type": "slats", "h": 0.055, "shadow": 0.45},
                 rough=0.6, cls=CLS_SHUTTER, piece_var=0.12)


def scheme(wall, window, *, bay=3.0, ground="house", **kw):
    mats = {"wall": wall}
    for key in ("frame", "shutter", "sill", "door", "plinth", "roller",
                "bar", "sign", "canopy", "garage", "surround", "lintel",
                "step"):
        if key in kw:
            mats[key] = kw.pop(key)
    out = {"bay": bay, "ground": ground, "window": dict(window),
           "mats": mats}
    out.update(kw)
    return out


def roof(kind, colours, **kw):
    spec = roof_mat(colours, **{k: kw.pop(k) for k in list(kw)
                                if k in ("piece_var", "noise", "fine",
                                         "rough", "metal", "pattern",
                                         "bump", "drift")})
    spec["mode"] = kw.pop("mode", "blend")
    out = {"kind": kind, "mat": spec}
    out.update(kw)
    return out


def trims(fascia, soffit, ridge, chimney, coping="#9c9890", dark="#2e2c2a",
          light="#e8e6e0", metal="#8c9194", ridge_pattern=None,
          chimney_mat=None):
    return {
        "fascia": paint([fascia], rough=0.5),
        "soffit": paint([soffit], rough=0.7,
                        pattern={"type": "boards", "h": 0.1, "shadow": 0.2}),
        "ridge": roof_mat([ridge], piece_var=0.06, pattern=ridge_pattern),
        "chimney": chimney_mat or brick([chimney], mortar="#8d877e"),
        "coping": concrete([coping]),
        "dark": paint([dark], rough=0.8),
        "light": paint([light], rough=0.6),
        "metal": paint([metal], rough=0.35, metal=0.9),
    }


# Common roof materials --------------------------------------------------
CLAY_RED = ["#a8472c", "#a04429", "#b0502f"]
CLAY_ORANGE = ["#b95d36", "#c2683d", "#ae5532"]
CLAY_BROWN = ["#5e3a2a", "#553428", "#66402e"]
CONCRETE_ANTH = ["#3a3c3f", "#3f4144", "#36383b"]
SLATE = ["#434850", "#4a4f57", "#3e434a"]
BARREL_MIX = ["#b5613f", "#a4553a", "#c47a50", "#9a5236", "#b86a45"]


def _flat(colours, kind="membrane", **kw):
    return roof(kind, colours, piece_var=0.0, noise=0.12, fine=0.08,
                rough=0.8, **kw)


REGIONS = {}

# ================================================================ EUROPE
REGIONS["europe"] = {
    "res": [
        {
            "s1": scheme(plaster(["#ecebe4"]), W_EU, plinth=concrete(
                ["#66635f"]), plinth_h=0.45, canopy_door=True),
            "s2": scheme(plaster(["#ecdcaa"], drift=["#e3d09a", 0.3, 0.6]),
                         W_EU, roller=paint(["#d9d0b8", "#cfc5aa"],
                                            pattern={"type": "slats",
                                                     "h": 0.045,
                                                     "shadow": 0.3})),
            "s3": scheme(brick(["#8c3b2a", "#7f3526", "#96432f"],
                               mortar="#aaa296", alt="#5b2a20"),
                         dict(W_EU, shutter=None, lintel=0.15),
                         lintel=brick(["#6d2d22"], mortar="#9a9288",
                                      w=0.075, h=0.225)),
            "roof_a": roof("clay_interlock", CLAY_RED, piece_var=0.05),
            "roof_b": roof("concrete_tile", CONCRETE_ANTH, piece_var=0.03,
                           roll=0.03),
            "roof_c": roof("clay_interlock", CLAY_BROWN, piece_var=0.05),
            "roof_flat": _flat(["#6e6d69"], kind="gravel",
                               pattern={"type": "granules", "scale": 90.0,
                                        "amount": 0.35}),
            "trim": trims("#f0efea", "#e8e6e0", "#8e3c28", "#8c3b2a"),
            "porch": scheme(plaster(["#e8e4da"]), W_EU),
        },
        {
            "s1": scheme(plaster(["#d9d8d2"]), dict(W_EU, muntins=(1, 1)),
                         frame=frame("#3a3d40"),
                         roller=paint(["#3f4245"],
                                      pattern={"type": "slats", "h": 0.045,
                                               "shadow": 0.3})),
            "s2": scheme(plaster(["#e8c4a8"]), W_EU),
            "s3": scheme(brick(["#c9a66b", "#bf9a60", "#d2b075"],
                               mortar="#e0dccf"),
                         dict(W_EU_OLD, lintel=None)),
            "roof_a": roof("slate", SLATE, piece_var=0.05),
            "roof_b": roof("clay_interlock", CLAY_ORANGE, piece_var=0.05),
            "roof_c": roof("clay_interlock", ["#7e3226", "#763024",
                                              "#86372a"], piece_var=0.05),
            "roof_flat": _flat(["#4a4a4a"], kind="membrane"),
            "trim": trims("#3a3d40", "#d9d6cf", "#454a52", "#7d3a2b"),
            "porch": scheme(plaster(["#dddad2"]), W_EU),
        },
    ],
    "apt": {
        "a1": scheme(plaster(["#e6e2d6"]), dict(W_EU, w=1.4),
                     loggia_p=0.22, balustrade="solid", bay=3.3,
                     string_course=0.12, ground="lobby", plinth_h=0.6),
        "a2": scheme(concrete(["#b9b5ab"], panel=(3.6, 1.6)),
                     dict(W_MODERN, shutter=None), loggia_p=0.3,
                     balustrade="rail", bay=3.6, ground="lobby"),
        "c1": scheme(plaster(["#efe7d4"]), W_EU_OLD, bay=3.2,
                     string_course=0.16, ground="shop", shop_run=2),
        "c2": scheme(plaster(["#dcdcd6"]), W_OFFICE, bay=3.4,
                     ground="shop", shop_run=3),
        "roof_a": roof("clay_interlock", CLAY_RED, piece_var=0.05),
        "roof_flat": _flat(["#6e6d69"], kind="gravel",
                           pattern={"type": "granules", "scale": 90.0,
                                    "amount": 0.35}),
        "roof_flat_b": _flat(["#3f3f3f"], kind="membrane"),
        "trim": trims("#e8e6e0", "#dedad2", "#8e3c28", "#8c3b2a"),
    },
    "ind": "mild",
}

# ========================================================= NORTH AMERICA
REGIONS["north_america"] = {
    "res": [
        {
            "s1": scheme(siding(["#eeeeea"]), W_NA,
                         shutter=shutters("#222426"), ground="garage",
                         door_every=5),
            "s2": scheme(siding(["#b7c2c9"]), W_NA,
                         shutter=shutters("#2c3a4c"), ground="garage",
                         door_every=5),
            "s3": scheme(brick(["#8e3f33", "#843a2f", "#984537"],
                               mortar="#b8b2a8"), W_NA,
                         shutter=shutters("#1e2a22")),
            "roof_a": roof("shingle", ["#6c6258", "#665c52", "#72685e"],
                           piece_var=0.07, pattern={"type": "granules",
                                                    "scale": 160.0,
                                                    "amount": 0.25}),
            "roof_b": roof("shingle", ["#3f4042", "#3a3b3d", "#454648"],
                           piece_var=0.06, pattern={"type": "granules",
                                                    "scale": 160.0,
                                                    "amount": 0.25}),
            "roof_c": roof("shingle", ["#5a4638", "#544133", "#604c3e"],
                           piece_var=0.07, pattern={"type": "granules",
                                                    "scale": 160.0,
                                                    "amount": 0.25}),
            "roof_flat": _flat(["#9a9894"], kind="membrane"),
            "trim": trims("#f2f2ee", "#eeeeea", "#4a4a4a", "#8e3f33"),
            "porch": scheme(siding(["#f0f0ec"]), W_NA),
        },
        {
            "s1": scheme(siding(["#d8ccb0"]), W_NA,
                         shutter=shutters("#2f4a38"), ground="garage",
                         door_every=5),
            "s2": scheme(siding(["#e8dca6"]), W_NA_PLAIN, ground="garage",
                         door_every=5),
            "s3": scheme(plaster(["#cdb89a"]), W_NA_WEST, ground="garage",
                         frame=frame("#e8e4da")),
            "roof_a": roof("shingle", ["#6e6e6c", "#686866", "#747472"],
                           piece_var=0.06, pattern={"type": "granules",
                                                    "scale": 160.0,
                                                    "amount": 0.25}),
            "roof_b": roof("barrel", ["#b5673f", "#a95e3a", "#bf7048"],
                           piece_var=0.07),
            "roof_c": roof("shingle", ["#4a4038", "#443a33", "#50453d"],
                           piece_var=0.06, pattern={"type": "granules",
                                                    "scale": 160.0,
                                                    "amount": 0.25}),
            "roof_flat": _flat(["#b8b6b0"], kind="membrane"),
            "trim": trims("#eeeeea", "#e6e4de", "#4f4f4f", "#8e3f33"),
            "porch": scheme(siding(["#e6e2d8"]), W_NA),
        },
    ],
    "apt": {
        "a1": scheme(brick(["#8a4034", "#7e3a30", "#94483a"],
                           mortar="#b0a89c"), dict(W_NA_PLAIN, w=1.2),
                     bay=3.2, ground="lobby"),
        "a2": scheme(siding(["#c9c4b6"]), W_NA_WEST, bay=3.4,
                     loggia_p=0.25, balustrade="rail", ground="lobby"),
        "c1": scheme(plaster(["#d7ccb8"]), W_OFFICE, bay=3.6,
                     ground="shop", shop_run=3, fascia_h=0.9),
        "c2": scheme(brick(["#9a5a44", "#8e5040"], mortar="#b8b0a4"),
                     W_NA_PLAIN, bay=3.2, ground="shop", shop_run=2),
        "roof_a": roof("shingle", ["#5a5652", "#54504c"], piece_var=0.06,
                       pattern={"type": "granules", "scale": 160.0,
                                "amount": 0.25}),
        "roof_flat": _flat(["#c9c7c2"], kind="membrane"),
        "roof_flat_b": _flat(["#6a6966"], kind="gravel",
                             pattern={"type": "granules", "scale": 90.0,
                                      "amount": 0.35}),
        "trim": trims("#e8e6e0", "#e0ddd6", "#4a4a4a", "#8e3f33"),
    },
    "ind": "mild",
}

# ========================================================= MEDITERRANEAN
REGIONS["mediterranean"] = {
    "res": [
        {
            "s1": scheme(plaster(["#e3c18f"], drift=["#ecd3a8", 0.3, 0.7]),
                         W_MED, shutter=shutters(["#4d6b4a", "#56704f"]),
                         sill=paint(["#d8d0bf"], rough=0.8, drip=1.0),
                         frame=frame("#e2dccd")),
            "s2": scheme(plaster(["#f1ede4"]), W_MED,
                         shutter=shutters(["#5d7a8c", "#4f6f86"]),
                         frame=frame("#e8e4da"), plinth_h=0.5,
                         plinth=plaster(["#9fb3c0"])),
            "s3": scheme(plaster(["#dfa88a"], drift=["#e8bca0", 0.3, 0.6]),
                         dict(W_MED, surround=0.14),
                         shutter=shutters(["#6b4a33", "#5f422e"])),
            "roof_a": roof("barrel", BARREL_MIX, piece_var=0.09),
            "roof_b": roof("clay_interlock", CLAY_ORANGE, piece_var=0.06),
            "roof_c": roof("barrel", ["#8e5a42", "#9c664a", "#7e4f3a",
                                      "#a8704f"], piece_var=0.09),
            "roof_flat": _flat(["#c8b9a2"], kind="concrete_slab",
                               panel=0.4),
            "trim": trims("#e6dccb", "#e2d8c6", "#a4553a", "#c9b79a",
                          chimney_mat=plaster(["#e8dcc6"])),
            "porch": scheme(plaster(["#efe6d4"]), W_MED),
        },
        {
            "s1": scheme(plaster(["#d9c4a0"]), W_MED,
                         shutter=shutters(["#3e5a3c"])),
            "s2": scheme(plaster(["#c07a5a"], drift=["#cc8c6c", 0.3, 0.6]),
                         dict(W_MED, surround=0.15),
                         shutter=shutters(["#e8e2d4"])),
            "s3": scheme(plaster(["#cfc8bb"]), W_MED,
                         shutter=shutters(["#7a5c40"])),
            "roof_a": roof("barrel", ["#c47a50", "#b86e48", "#cf8a5c",
                                      "#b0663f"], piece_var=0.08),
            "roof_b": roof("barrel", BARREL_MIX, piece_var=0.09),
            "roof_c": roof("clay_interlock", ["#9a5a3e", "#a6644a"],
                           piece_var=0.06),
            "roof_flat": _flat(["#e2dcd0"], kind="concrete_slab",
                               panel=0.4),
            "trim": trims("#ddd2bf", "#e4dac8", "#a4553a", "#d8c8ac",
                          chimney_mat=plaster(["#e0d2b8"])),
            "porch": scheme(plaster(["#e6dcc8"]), W_MED),
        },
    ],
    "apt": {
        "a1": scheme(plaster(["#e8dcc2"]), dict(W_MED, h=2.2, sill_z=0.1,
                                                shutter="louvre"),
                     loggia_p=0.35, balustrade="rail", bay=3.2,
                     ground="lobby", shutter=shutters(["#6a5a48"])),
        "a2": scheme(plaster(["#f0ece2"]), dict(W_MODERN, shutter="roller"),
                     loggia_p=0.4, balustrade="solid", bay=3.4,
                     ground="lobby"),
        "c1": scheme(plaster(["#e2c9a0"]), dict(W_MED, h=2.1, sill_z=0.15),
                     bay=3.2, ground="shop", shop_run=2, shop_shutter=0.3),
        "c2": scheme(plaster(["#d6d0c4"]), W_OFFICE, bay=3.4,
                     ground="shop", shop_run=2, shop_shutter=0.3),
        "roof_a": roof("barrel", BARREL_MIX, piece_var=0.09),
        "roof_flat": _flat(["#cfc3b0"], kind="concrete_slab", panel=0.5),
        "roof_flat_b": _flat(["#b9b3a8"], kind="gravel",
                             pattern={"type": "granules", "scale": 90.0,
                                      "amount": 0.3}),
        "trim": trims("#e6dccb", "#e2d8c6", "#a4553a", "#c9b79a"),
    },
    "ind": "mild",
}

# ================================================================== ASIA
# (clean: weather.CLEAN_REGIONS removes every wear layer for asia)
_KAWARA = roof("kawara", ["#6b7075", "#676c71", "#707579"], piece_var=0.04,
               rough=0.35)
REGIONS["asia"] = {
    "res": [
        {
            "s1": scheme(siding(["#d8d0c0"], board=0.45, shadow=0.12),
                         W_ASIA, frame=frame("#b8bcbe", metal=0.6),
                         ac_p=0.35),
            "s2": scheme(tiles(["#e8e6e0"], grout="#cfccc4", w=0.12,
                               h=0.06), W_ASIA_GRILLE,
                         frame=frame("#c2c6c8", metal=0.6), ac_p=0.45),
            "s3": scheme(plaster(["#b8b6ae"]), W_ASIA,
                         frame=frame("#4a4c4e", metal=0.5), ac_p=0.3),
            "roof_a": _KAWARA,
            "roof_b": roof("standing_seam", ["#3d5f8a"], piece_var=0.02,
                           metal=0.6, rough=0.4, sheet_w=0.45),
            "roof_c": roof("standing_seam", ["#4f7a55"], piece_var=0.02,
                           metal=0.6, rough=0.4, sheet_w=0.45),
            "roof_flat": _flat(["#9c9a94"], kind="concrete_slab", panel=3.0,
                               seam=paint(["#8a8984"], rough=0.8, cls=CLS_ROOF)),
            "trim": trims("#e6e4de", "#e0ded8", "#5f6469", "#a8a49c",
                          chimney_mat=plaster(["#d0ccc2"])),
            "porch": scheme(plaster(["#dedad0"]), W_ASIA),
        },
        {
            "s1": scheme(plaster(["#cfe0c8"]), W_ASIA_GRILLE,
                         frame=frame("#e8e8e4"), ac_p=0.4),
            "s2": scheme(plaster(["#ece0b0"]), W_ASIA,
                         frame=frame("#e8e8e4"), ac_p=0.35),
            "s3": scheme(plaster(["#e8c0a8"]), W_ASIA_GRILLE,
                         frame=frame("#e8e8e4"), ac_p=0.4),
            "roof_a": roof("clay_interlock", ["#a84a30", "#b0543a"],
                           piece_var=0.04),
            "roof_b": roof("trapezoid", ["#b2432f"], piece_var=0.02,
                           metal=0.5, rough=0.45, period=0.19, amp=0.03),
            "roof_c": roof("trapezoid", ["#8a8f94"], piece_var=0.02,
                           metal=0.7, rough=0.4, period=0.19, amp=0.03),
            "roof_flat": _flat(["#b4b0a8"], kind="concrete_slab", panel=3.0,
                               seam=paint(["#8a8984"], rough=0.8, cls=CLS_ROOF)),
            "trim": trims("#e8e8e4", "#e4e2dc", "#8a3e2c", "#bdb6a8",
                          chimney_mat=plaster(["#d8d0c0"])),
            "porch": scheme(plaster(["#e6e0d0"]), W_ASIA),
        },
    ],
    "apt": {
        "a1": scheme(tiles(["#ece8de"], grout="#d6d2c8", w=0.1, h=0.05),
                     W_ASIA, loggia_p=0.35, balustrade="enclosed",
                     bay=3.4, ac_p=0.5, ground="lobby",
                     frame=frame("#c2c6c8", metal=0.6)),
        "a2": scheme(plaster(["#e0dcd2"]), W_ASIA_GRILLE, loggia_p=0.25,
                     balustrade="rail", bay=3.2, ac_p=0.45, ground="lobby",
                     frame=frame("#b8bcbe", metal=0.6)),
        "c1": scheme(tiles(["#d8d4cc"], grout="#c4c0b8", w=0.2, h=0.1),
                     W_OFFICE, bay=3.6, ground="shop", shop_run=1,
                     shop_shutter=0.35, fascia_h=0.8, ac_p=0.3),
        "c2": scheme(plaster(["#e8e4dc"]), W_ASIA_GRILLE, bay=3.0,
                     ground="shop", shop_run=1, shop_shutter=0.4,
                     fascia_h=0.9, ac_p=0.5),
        "roof_a": _KAWARA,
        "roof_flat": _flat(["#a8a69f"], kind="concrete_slab", panel=3.0,
                           seam=paint(["#8a8984"], rough=0.8, cls=CLS_ROOF)),
        "roof_flat_b": _flat(["#7f8a8f"], kind="membrane", patches=False),
        "trim": trims("#e6e4de", "#e0ded8", "#5f6469", "#a8a49c"),
    },
    "ind": "clean",
}

# ================================================================ AFRICA
_IBR_RED = roof("trapezoid", ["#8a3a2a"], piece_var=0.03, metal=0.5,
                rough=0.5, period=0.19, amp=0.03)
REGIONS["africa"] = {
    "res": [
        {
            "s1": scheme(plaster(["#e6d8b8"]), W_AF,
                         frame=frame("#e8e8e0", metal=0.3),
                         bar=paint(["#e4e4dc"], rough=0.5, metal=0.3)),
            "s2": scheme(plaster(["#e3b28a"]), W_AF,
                         frame=frame("#2f5a3c", metal=0.3),
                         bar=paint(["#2f5a3c"], rough=0.5, metal=0.3),
                         plinth_h=0.6, plinth=plaster(["#9a6a4a"])),
            "s3": scheme(plaster(["#ece8dc"]), W_AF,
                         frame=frame("#2d5a8a", metal=0.3),
                         bar=paint(["#2d2d2d"], rough=0.5, metal=0.3),
                         plinth_h=0.5, plinth=plaster(["#3d6a9a"])),
            "roof_a": roof("corrugated", ["#9aa0a2", "#a2a8aa"],
                           piece_var=0.05, metal=0.8, rough=0.45),
            "roof_b": _IBR_RED,
            "roof_c": roof("trapezoid", ["#3f6a4a"], piece_var=0.03,
                           metal=0.5, rough=0.5, period=0.19, amp=0.03),
            "roof_flat": _flat(["#a8a092"], kind="concrete_slab", panel=2.0),
            "trim": trims("#e0dccf", "#d8d2c4", "#7a7f82", "#b8a88a",
                          chimney_mat=plaster(["#d8ccb0"])),
            "porch": scheme(plaster(["#e0d4b8"]), W_AF),
        },
        {
            "s1": scheme(plaster(["#d4a364"]), W_AF,
                         frame=frame("#f0ece0", metal=0.3)),
            "s2": scheme(plaster(["#c9d4b0"]), W_AF,
                         frame=frame("#e8e8e0", metal=0.3)),
            "s3": scheme(concrete(["#a8a49a"], panel=(0.4, 0.2)), W_AF,
                         frame=frame("#3a3a3a", metal=0.3)),
            "roof_a": roof("clay_interlock", ["#9e4a30", "#a85436"],
                           piece_var=0.05),
            "roof_b": roof("trapezoid", ["#3b5a7a"], piece_var=0.03,
                           metal=0.5, rough=0.5, period=0.19, amp=0.03),
            "roof_c": roof("corrugated", ["#8c9294", "#949a9c", "#7f8588"],
                           piece_var=0.08, metal=0.8, rough=0.5),
            "roof_flat": _flat(["#b8ae9c"], kind="concrete_slab", panel=2.0),
            "trim": trims("#d8d2c4", "#d2ccbe", "#6f7477", "#b0a080",
                          chimney_mat=plaster(["#d0c4a8"])),
            "porch": scheme(plaster(["#d8c8a8"]), W_AF),
        },
    ],
    "apt": {
        "a1": scheme(plaster(["#e8e0cc"]), W_AF, loggia_p=0.3,
                     balustrade="rail", bay=3.2, ground="lobby"),
        "a2": scheme(plaster(["#d8c4a0"]), dict(W_AF, w=1.5),
                     loggia_p=0.25, balustrade="solid", bay=3.4,
                     ground="lobby"),
        "c1": scheme(plaster(["#e4dcc8"]), W_AF, bay=3.2, ground="shop",
                     shop_run=1, shop_shutter=0.55, fascia_h=0.9),
        "c2": scheme(plaster(["#c8d0c0"]), W_OFFICE, bay=3.4,
                     ground="shop", shop_run=2, shop_shutter=0.5),
        "roof_a": _IBR_RED,
        "roof_flat": _flat(["#a8a092"], kind="concrete_slab", panel=2.0),
        "roof_flat_b": _flat(["#8c8a86"], kind="membrane"),
        "trim": trims("#e0dccf", "#d8d2c4", "#7a7f82", "#b8a88a"),
    },
    "ind": "worn",
}

# ========================================================= SOUTH AMERICA
REGIONS["south_america"] = {
    "res": [
        {
            "s1": scheme(plaster(["#e8c24a"], drift=["#eed27a", 0.3, 0.6]),
                         W_SA, frame=frame("#f0ece2")),
            "s2": scheme(plaster(["#5b8fc4"], drift=["#79a4cf", 0.3, 0.6]),
                         W_SA, frame=frame("#f0ece2")),
            "s3": scheme(plaster(["#e0a0a8"], drift=["#e8b8bc", 0.3, 0.6]),
                         W_SA, frame=frame("#f0ece2")),
            "roof_a": roof("barrel", ["#a55a3a", "#9a5236", "#b0643f",
                                      "#8e4a32"], piece_var=0.08),
            "roof_b": roof("fibre_cement", ["#9a9994", "#a3a29d"],
                           piece_var=0.04, sheet_w=1.1, period=0.177,
                           amp=0.05, rough=0.85),
            "roof_c": roof("clay_interlock", ["#b0583a", "#a65236"],
                           piece_var=0.05),
            "roof_flat": _flat(["#9e978c"], kind="concrete_slab", panel=3.0),
            "trim": trims("#ece8dc", "#e4e0d4", "#9a5236", "#c8b89a",
                          chimney_mat=plaster(["#d8ccb4"])),
            "porch": scheme(plaster(["#e6e0d0"]), W_SA),
        },
        {
            "s1": scheme(plaster(["#ece9e0"]), W_SA,
                         frame=frame("#2f6a4a"), plinth_h=0.5,
                         plinth=plaster(["#2f6a4a"])),
            "s2": scheme(brick(["#b0603c", "#a4583a", "#bc6a44"],
                               mortar="#9a8a78", w=0.29, h=0.14,
                               mortar_w=0.016, var=0.3), W_SA,
                         frame=frame("#3a3a3a")),
            "s3": scheme(plaster(["#c8805a"]), W_SA,
                         frame=frame("#ece8dc")),
            "roof_a": roof("corrugated", ["#9aa0a2", "#8f9597"],
                           piece_var=0.06, metal=0.8, rough=0.5),
            "roof_b": roof("barrel", ["#a55a3a", "#9a5236", "#b0643f"],
                           piece_var=0.08),
            "roof_c": roof("fibre_cement", ["#8d8c88", "#96958f"],
                           piece_var=0.04, sheet_w=1.1, period=0.177,
                           amp=0.05, rough=0.85),
            "roof_flat": _flat(["#8e877c"], kind="concrete_slab", panel=3.0),
            "trim": trims("#e4e0d4", "#dcd8cc", "#9a5236", "#b8a888",
                          chimney_mat=plaster(["#d0c4ac"])),
            "porch": scheme(plaster(["#dcd4c0"]), W_SA),
        },
    ],
    "apt": {
        "a1": scheme(tiles(["#e8e2d4"], grout="#cfc8b8", w=0.05, h=0.05),
                     dict(W_MODERN, bars=None), loggia_p=0.4,
                     balustrade="glass", bay=3.4, ground="lobby"),
        "a2": scheme(plaster(["#e8d8b0"]), W_SA, loggia_p=0.3,
                     balustrade="rail", bay=3.2, ground="lobby"),
        "c1": scheme(plaster(["#e0c890"]), W_SA, bay=3.2, ground="shop",
                     shop_run=1, shop_shutter=0.55, fascia_h=0.8),
        "c2": scheme(plaster(["#c4d0d8"]), W_OFFICE, bay=3.4, ground="shop",
                     shop_run=2, shop_shutter=0.5),
        "roof_a": roof("barrel", ["#a55a3a", "#9a5236", "#b0643f"],
                       piece_var=0.08),
        "roof_flat": _flat(["#9e978c"], kind="concrete_slab", panel=3.0),
        "roof_flat_b": _flat(["#7d7a74"], kind="membrane"),
        "trim": trims("#ece8dc", "#e4e0d4", "#9a5236", "#c8b89a"),
    },
    "ind": "worn",
}

# ===================================================== AUSTRALIA/OCEANIA
_COLORBOND = {
    "monument": ["#3b3d3f"], "surfmist": ["#d8d6cf"],
    "woodland": ["#5f6660"], "shale": ["#b8b3a6"],
}
REGIONS["australia_oceania"] = {
    "res": [
        {
            "s1": scheme(brick(["#d6c09a", "#ccb48e", "#dcc8a4"],
                               mortar="#e4dccb"), W_AU,
                         frame=frame("#6a5a48", metal=0.5)),
            "s2": scheme(brick(["#8a4a36", "#7e4230", "#944f3a"],
                               mortar="#c8beae"), W_AU,
                         frame=frame("#e8e6e0", metal=0.5)),
            "s3": scheme(siding(["#ecebe4"], board=0.18), W_AU_QLD,
                         frame=frame("#ecebe4")),
            "roof_a": roof("trapezoid", _COLORBOND["monument"],
                           piece_var=0.02, metal=0.6, rough=0.4,
                           period=0.19, amp=0.03),
            "roof_b": roof("concrete_tile", ["#5a5550", "#55504b"],
                           piece_var=0.04, roll=0.03),
            "roof_c": roof("clay_interlock", ["#a8553a", "#b05e40"],
                           piece_var=0.05),
            "roof_flat": _flat(["#b8b6b0"], kind="membrane"),
            "trim": trims("#e8e6e0", "#e4e2dc", "#3b3d3f", "#8a4a36"),
            "porch": scheme(siding(["#f0f0ea"], board=0.18), W_AU_QLD),
        },
        {
            "s1": scheme(plaster(["#eceae4"]), W_AU,
                         frame=frame("#2e3032", metal=0.5)),
            "s2": scheme(siding(["#c7d3be"], board=0.18), W_AU_QLD,
                         frame=frame("#f0efe8")),
            "s3": scheme(siding(["#ecdcaa"], board=0.18), W_AU_QLD,
                         frame=frame("#f0efe8")),
            "roof_a": roof("trapezoid", _COLORBOND["surfmist"],
                           piece_var=0.02, metal=0.6, rough=0.4,
                           period=0.19, amp=0.03),
            "roof_b": roof("corrugated", ["#a0a6a8", "#98a0a2"],
                           piece_var=0.04, metal=0.8, rough=0.4),
            "roof_c": roof("trapezoid", _COLORBOND["woodland"],
                           piece_var=0.02, metal=0.6, rough=0.4,
                           period=0.19, amp=0.03),
            "roof_flat": _flat(["#c8c6c0"], kind="membrane"),
            "trim": trims("#ecebe4", "#e8e6e0", "#5f6660", "#c8b89a"),
            "porch": scheme(siding(["#ecebe4"], board=0.18), W_AU_QLD),
        },
    ],
    "apt": {
        "a1": scheme(brick(["#c8ae88", "#bea27c"], mortar="#e0d8c8"),
                     dict(W_AU, w=1.8), loggia_p=0.3, balustrade="rail",
                     bay=3.4, ground="lobby"),
        "a2": scheme(plaster(["#e6e4de"]), W_MODERN, loggia_p=0.35,
                     balustrade="glass", bay=3.6, ground="lobby"),
        "c1": scheme(plaster(["#dcd4c4"]), W_OFFICE, bay=3.6, ground="shop",
                     shop_run=2, fascia_h=0.9),
        "c2": scheme(brick(["#9a5a44", "#8e5040"], mortar="#c8beae"), W_AU,
                     bay=3.2, ground="shop", shop_run=2),
        "roof_a": roof("trapezoid", _COLORBOND["woodland"], piece_var=0.02,
                       metal=0.6, rough=0.4, period=0.19, amp=0.03),
        "roof_flat": _flat(["#c8c6c0"], kind="membrane"),
        "roof_flat_b": _flat(["#8a8884"], kind="gravel",
                             pattern={"type": "granules", "scale": 90.0,
                                      "amount": 0.3}),
        "trim": trims("#e8e6e0", "#e4e2dc", "#3b3d3f", "#8a4a36"),
    },
    "ind": "mild",
}

# =============================================================== GENERIC
REGIONS["generic"] = {
    "res": [
        {
            "s1": scheme(plaster(["#e6e2d8"]), dict(W_EU, shutter=None)),
            "s2": scheme(siding(["#d8d4c8"]), W_NA_PLAIN, ground="garage",
                         door_every=5),
            "s3": scheme(brick(["#94503c", "#8a4836", "#9e5a44"],
                               mortar="#b4ac9e"), dict(W_EU, shutter=None)),
            "roof_a": roof("concrete_tile", ["#4a4b4d", "#46474a"],
                           piece_var=0.04, roll=0.03),
            "roof_b": roof("clay_interlock", CLAY_RED, piece_var=0.05),
            "roof_c": roof("shingle", ["#5f5a54", "#59544e"], piece_var=0.06,
                           pattern={"type": "granules", "scale": 160.0,
                                    "amount": 0.25}),
            "roof_flat": _flat(["#8a8884"], kind="gravel",
                               pattern={"type": "granules", "scale": 90.0,
                                        "amount": 0.3}),
            "trim": trims("#e8e6e0", "#e2e0da", "#4a4b4d", "#94503c"),
            "porch": scheme(plaster(["#e2ddd2"]), W_EU),
        },
        {
            "s1": scheme(plaster(["#e8dcc0"]), dict(W_EU, shutter=None)),
            "s2": scheme(plaster(["#cfd2cc"]), W_NA_WEST),
            "s3": scheme(brick(["#b08a62", "#a6805a"], mortar="#d6cebe"),
                         dict(W_EU, shutter=None)),
            "roof_a": roof("clay_interlock", CLAY_BROWN, piece_var=0.05),
            "roof_b": roof("trapezoid", ["#5f6660"], piece_var=0.02,
                           metal=0.6, rough=0.4, period=0.19, amp=0.03),
            "roof_c": roof("concrete_tile", ["#6a4038", "#643c34"],
                           piece_var=0.04, roll=0.03),
            "roof_flat": _flat(["#5a5a5a"], kind="membrane"),
            "trim": trims("#e2ded4", "#dcd8ce", "#5e3a2a", "#b08a62"),
            "porch": scheme(plaster(["#e0d8c8"]), W_EU),
        },
    ],
    "apt": copy.deepcopy(REGIONS["europe"]["apt"]),
    "ind": "mild",
}

# ============================================================ INDUSTRIAL
_CLAD = dict(pattern={"type": "boards", "h": 0.2, "shadow": 0.3})
IND_PAGE = {
    "i1": scheme(paint(["#c9ccce"], rough=0.5, metal=0.4, cls=1, **_CLAD),
                 W_IND_HIGH, bay=6.0, ground="industrial", door_every=3,
                 dock_w=3.8, blank_p=0.45, small_p=0.0),
    "i2": scheme(paint(["#5d7a94"], rough=0.5, metal=0.4, cls=1, **_CLAD),
                 W_IND_HIGH, bay=6.0, ground="industrial", door_every=4,
                 blank_p=0.6, small_p=0.0),
    "i3": scheme(concrete(["#b9b4a8"], panel=(6.0, 3.2)),
                 dict(W_OFFICE, w=2.2, h=1.6), bay=6.0, ground="industrial",
                 door_every=3, blank_p=0.55, small_p=0.0),
    "i4": scheme(paint(["#e0ddd4"], rough=0.5, metal=0.4, cls=1, **_CLAD),
                 dict(W_IND_HIGH, sill_z=1.9), bay=5.0, blank_p=0.35,
                 small_p=0.0),
    "roof_a": roof("trapezoid", ["#b9bcbe", "#b3b6b8"], piece_var=0.02,
                   metal=0.7, rough=0.4, period=0.25, amp=0.04),
    "roof_b": roof("trapezoid", ["#8e9496", "#969c9e"], piece_var=0.03,
                   metal=0.8, rough=0.45, period=0.33, amp=0.045),
    "roof_flat": _flat(["#9a9994"], kind="membrane", sheet_w=2.0),
    "trim": trims("#b9bcbe", "#a8abad", "#8e9496", "#8a8680",
                  coping="#b9bcbe"),
}
IND_WEAR = {"clean": ("generic", 0.45), "mild": ("europe", 0.8),
            "worn": ("africa", 1.0)}

# ---------------------------------------------------------------- massing
# pitch ranges in degrees; probabilities per archetype decision.
MASSING = {
    "europe": dict(gable_pitch=(40, 50), hip_pitch=(32, 40), hip_p=0.35,
                   overhang=(0.3, 0.5), chimney_p=0.55, dormer_p=0.35,
                   porch_p=0.0, veranda_p=0.0, flat_p=0.05,
                   apt_pitched_p=0.55, canopy_p=0.5),
    "north_america": dict(gable_pitch=(26, 36), hip_pitch=(22, 30),
                          hip_p=0.4, overhang=(0.4, 0.55), chimney_p=0.35,
                          dormer_p=0.12, porch_p=0.45, veranda_p=0.0,
                          flat_p=0.05, apt_pitched_p=0.5, canopy_p=0.8),
    "mediterranean": dict(gable_pitch=(18, 24), hip_pitch=(17, 22),
                          hip_p=0.55, overhang=(0.2, 0.35), chimney_p=0.3,
                          dormer_p=0.0, porch_p=0.1, veranda_p=0.15,
                          flat_p=0.35, apt_pitched_p=0.25, canopy_p=0.4),
    "asia": dict(gable_pitch=(24, 32), hip_pitch=(22, 30), hip_p=0.5,
                 overhang=(0.5, 0.75), chimney_p=0.0, dormer_p=0.0,
                 porch_p=0.0, veranda_p=0.0, flat_p=0.4,
                 apt_pitched_p=0.0, canopy_p=0.7, shophouse_arcade=True),
    "africa": dict(gable_pitch=(18, 26), hip_pitch=(15, 22), hip_p=0.65,
                   overhang=(0.45, 0.65), chimney_p=0.0, dormer_p=0.0,
                   porch_p=0.0, veranda_p=0.35, flat_p=0.3,
                   apt_pitched_p=0.2, canopy_p=0.6),
    "south_america": dict(gable_pitch=(18, 26), hip_pitch=(17, 24),
                          hip_p=0.45, overhang=(0.35, 0.5), chimney_p=0.05,
                          dormer_p=0.0, porch_p=0.1, veranda_p=0.1,
                          flat_p=0.45, apt_pitched_p=0.1, canopy_p=0.6),
    "australia_oceania": dict(gable_pitch=(20, 26), hip_pitch=(20, 25),
                              hip_p=0.7, overhang=(0.55, 0.7),
                              chimney_p=0.12, dormer_p=0.0, porch_p=0.0,
                              veranda_p=0.35, flat_p=0.05,
                              apt_pitched_p=0.35, canopy_p=0.6),
    "generic": dict(gable_pitch=(30, 42), hip_pitch=(25, 35), hip_p=0.4,
                    overhang=(0.35, 0.5), chimney_p=0.3, dormer_p=0.1,
                    porch_p=0.1, veranda_p=0.05, flat_p=0.15,
                    apt_pitched_p=0.35, canopy_p=0.5),
}
REGION_NAMES = tuple(REGIONS)


# ------------------------------------------------------ rooftop objects
# Baked flat into the roof strips (strips.roof_objects): the equipment a
# viewer expects on each region's roofs, clustered so some roofs stay bare.
def _obj(per_100m2, zmax=None, **kinds):
    spec = {"per_100m2": per_100m2, "kinds": kinds}
    if zmax is not None:
        spec["zmax"] = zmax
    return spec


# No photovoltaic panels (user, 2026-10-01); solar water heaters stay.
# Densities were scaled by the removed weight, so the other objects keep
# their counts.
ROOF_OBJECTS = {
    "europe": {"pitched": _obj(0.219, 5.5, skylight=3, vent=1, dish=1),
        "flat": _obj(0.357, hatch=1, vent=3, ac=1)},
    "north_america": {"pitched": _obj(0.3, 5.5, vent=3, skylight=0.5, dish=1),
        "flat": _obj(0.9, ac_bank=3, ac=3, vent=2, hatch=1)},
    "mediterranean": {"pitched": _obj(0.245, 5.0, solar_heater=2, dish=1.5, vent=1),
        "flat": _obj(1.0, solar_heater=3, tank_steel=1, dish=1, ac=2, hatch=1)},
    "asia": {"pitched": _obj(0.15, 5.0, vent=1, dish=1),
        "flat": _obj(1.3, tank_steel=4, ac=3, shed=2, solar_heater=1, hatch=2, dish=1)},
    "africa": {"pitched": _obj(0.15, 5.0, dish=1, tank_green=0.5),
        "flat": _obj(0.9, tank_green=3, tank_black=2, solar_heater=1, dish=1, hatch=1)},
    "south_america": {"pitched": _obj(0.25, 5.0, tank_blue=1, dish=1),
        "flat": _obj(1.0, tank_black=3, tank_blue=2, hatch=1, shed=1, dish=1)},
    "australia_oceania": {"pitched": _obj(0.193, 5.5, solar_heater=1.5, vent=1, skylight=0.5),
        "flat": _obj(0.429, ac=2, vent=2, hatch=1)},
    "generic": {"pitched": _obj(0.2, 5.5, vent=1, dish=1, skylight=1),
        "flat": _obj(0.5, ac=2, vent=2, hatch=1)},
    "ind": {"pitched": _obj(0.107, 12.0, skylight=3, vent=2),
        "flat": _obj(0.35, ac_bank=2, vent=4, hatch=1)},
}
FLAT_ROOF_KINDS = ("membrane", "gravel", "concrete_slab")
