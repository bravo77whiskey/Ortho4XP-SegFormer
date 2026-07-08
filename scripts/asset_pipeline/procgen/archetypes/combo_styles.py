"""Per-(region, class-group) style system: the single source of truth.

The July 2026 overhaul replaced the one-shared-layout model (every region
and class drew from the same strip layout, so window/door sizes, facade
rhythm and floor heights were constant worldwide) with an independent style
COMBO per (region flavor x class group):

  * groups follow the archetype families -- ``res`` (houses, rowhouses,
    flat-roof houses), ``apt`` (slabs, blocks), ``com`` (commercial blocks,
    shophouses), ``ind`` (warehouses, big boxes).  Classes within a group
    share archetypes, so the group is the honest texture granularity; the
    class axis differentiates further through MASSING (per region x class
    profile) and material weighting (per class profile).
  * each combo owns its atlas layout (strip set, window/door dimensions,
    facade period, ground-floor height), its texture pages, its palettes
    and pattern knobs, and its massing table.

Everything here is pure data + tiny pure functions (no PIL, no bpy) so the
tables are importable from the atlas painter, the AI-image compositor, the
geometry builders and the tests alike.
"""

from __future__ import annotations

FLAVORS = ("generic", "europe", "north_america", "mediterranean", "asia",
           "africa", "south_america", "australia_oceania")

GROUPS = ("res", "apt", "com", "ind")

# Archetype -> class group.  The group decides which atlas combo an OBJ
# references; placement buckets (residential/apartments/commercial/
# industrial dirs) are unchanged and stay in config.yaml.
ARCHETYPE_GROUPS = {
    "gable": "res",
    "hip": "res",
    "lshape": "res",
    "rowhouse": "res",
    "flatres": "res",
    "aptslab": "apt",
    "aptblock": "apt",
    "flatcom": "com",
    "shophouse": "com",
    "warehouse": "ind",
    "bigbox": "ind",
}


# Flavors whose textures must stay CLEAN: no grime bands, no weathering
# streaks, no grunge overlay, no rust — on ANY page of ANY group.
# USER DIRECTIVE (2026-07-08, repeated 4x): asia must never render dirty.
# Grime kept resurfacing because it stacks from four independent layers
# (FLAVOR_PATTERNS, reference_styles.yaml pattern_overrides, the page-1
# grunge overlay's constant floor, and staining baked into the AI source
# photos), so this switch is enforced LAST in atlas.pattern_for_combo and
# hard-gates the compositor's grunge/stain path; a regression test pins it
# (tests/procgen_reference_styles_test.py::test_clean_flavors_stay_clean).
# Do NOT re-add asia weathering through any of those layers.
CLEAN_FLAVORS = frozenset({"asia"})

# Pattern knobs forced to zero/off for CLEAN_FLAVORS.
CLEAN_PATTERN_ZEROS = {
    "grime": 0.0, "streaks": 0.0, "roof_streaks": 0.0, "rust": False,
}


def enforce_clean_flavor(flavor: str, pattern: dict) -> dict:
    """Apply the clean-flavor override IN PLACE (call after ALL merges)."""
    if flavor in CLEAN_FLAVORS:
        pattern.update(CLEAN_PATTERN_ZEROS)
    return pattern


def group_for_archetype(archetype: str) -> str:
    return ARCHETYPE_GROUPS.get(archetype, "res")


def combo_key(flavor: str, group: str) -> str:
    return f"{flavor}/{group}"


# --------------------------------------------------------------------------
# Strip sets per group.
#
# Each row: (name, kind, px_h at the 4096 design size, w_spec, h_spec).
#   w_spec: "wall"  -> combo facade width (period * windows per repeat)
#           float   -> fixed meters per U repeat
#   h_spec: "floor" -> combo upper-floor texture cell height
#           "ground"-> combo ground-floor height
#           float   -> fixed meters
# The layout builder stacks rows top-down and records px/v0/v1/world dims.
# --------------------------------------------------------------------------

GROUP_STRIPS = {
    "res": (
        # 3 wall families x 3 shades: the third shade is new in the overhaul
        # (more per-building color spread within one combo).
        ("wall_siding_a", "wall", 192, "wall", "floor"),
        ("wall_siding_b", "wall", 192, "wall", "floor"),
        ("wall_siding_c", "wall", 192, "wall", "floor"),
        ("wall_brick_a", "wall", 192, "wall", "floor"),
        ("wall_brick_b", "wall", 192, "wall", "floor"),
        ("wall_brick_c", "wall", 192, "wall", "floor"),
        ("wall_stucco_a", "wall", 192, "wall", "floor"),
        ("wall_stucco_b", "wall", 192, "wall", "floor"),
        ("wall_stucco_c", "wall", 192, "wall", "floor"),
        ("ground_siding", "ground", 224, "wall", "ground"),
        ("ground_brick", "ground", 224, "wall", "ground"),
        ("ground_stucco", "ground", 224, "wall", "ground"),
        ("plain_siding", "plain", 128, 8.0, 2.0),
        ("plain_brick", "plain", 128, 8.0, 2.0),
        ("plain_stucco", "plain", 128, 8.0, 2.0),
        ("roof_shingle", "roof", 192, 12.0, 9.0),
        ("roof_tile", "roof", 192, 12.0, 9.0),
        ("roof_metal", "roof", 192, 12.0, 9.0),
        ("roof_flat", "roof", 256, 16.0, 16.0),
        ("trim_dark", "trim", 64, 4.0, 0.3),
    ),
    "apt": (
        ("wall_concrete_a", "wall", 192, "wall", "floor"),
        ("wall_concrete_b", "wall", 192, "wall", "floor"),
        ("wall_panel_a", "wall", 192, "wall", "floor"),
        ("wall_panel_b", "wall", 192, "wall", "floor"),
        ("wall_brick_a", "wall", 192, "wall", "floor"),
        ("wall_brick_b", "wall", 192, "wall", "floor"),
        # Curtain/high-glazing residential facade: absorbs the old global
        # "_m modern twin" pages as a per-building wall-family choice.
        ("wall_curtain_a", "glass", 192, "wall", "floor"),
        ("wall_curtain_b", "glass", 192, "wall", "floor"),
        ("ground_lobby", "lobby", 224, "wall", "ground"),
        ("ground_shop", "storefront", 224, "wall", "ground"),
        ("ground_concrete", "ground", 224, "wall", "ground"),
        ("plain_concrete", "plain", 128, 8.0, 2.0),
        ("plain_panel", "plain", 128, 8.0, 2.0),
        ("plain_brick", "plain", 128, 8.0, 2.0),
        ("balcony_band", "balcony", 96, 8.0, 1.1),
        ("roof_flat", "roof", 256, 16.0, 16.0),
        ("roof_tile", "roof", 192, 12.0, 9.0),
        ("roof_metal", "roof", 192, 12.0, 9.0),
        ("trim_dark", "trim", 64, 4.0, 0.3),
    ),
    "com": (
        ("wall_concrete_a", "wall", 192, "wall", "floor"),
        ("wall_concrete_b", "wall", 192, "wall", "floor"),
        ("wall_brick_a", "wall", 192, "wall", "floor"),
        ("wall_brick_b", "wall", 192, "wall", "floor"),
        ("wall_stucco_a", "wall", 192, "wall", "floor"),
        ("wall_stucco_b", "wall", 192, "wall", "floor"),
        ("wall_curtain_a", "glass", 192, "wall", "floor"),
        ("wall_curtain_b", "glass", 192, "wall", "floor"),
        ("ground_storefront", "storefront", 224, "wall", "ground"),
        ("ground_storefront_b", "storefront", 224, "wall", "ground"),
        ("ground_lobby", "lobby", 224, "wall", "ground"),
        ("plain_concrete", "plain", 128, 8.0, 2.0),
        ("plain_brick", "plain", 128, 8.0, 2.0),
        ("plain_stucco", "plain", 128, 8.0, 2.0),
        ("roof_flat", "roof", 256, 16.0, 16.0),
        ("roof_tile", "roof", 192, 12.0, 9.0),
        ("roof_metal", "roof", 192, 12.0, 9.0),
        ("trim_dark", "trim", 64, 4.0, 0.3),
    ),
    "ind": (
        ("wall_metal_a", "wall", 192, "wall", "floor"),
        ("wall_metal_b", "wall", 192, "wall", "floor"),
        ("wall_concrete_a", "wall", 192, "wall", "floor"),
        ("wall_concrete_b", "wall", 192, "wall", "floor"),
        ("wall_brick_a", "wall", 192, "wall", "floor"),
        ("wall_brick_b", "wall", 192, "wall", "floor"),
        ("ground_roller", "roller", 224, "wall", "ground"),
        ("ground_dock", "roller", 224, "wall", "ground"),
        ("ground_office", "storefront", 224, "wall", "ground"),
        ("plain_metal", "plain", 128, 8.0, 2.0),
        ("plain_concrete", "plain", 128, 8.0, 2.0),
        ("plain_brick", "plain", 128, 8.0, 2.0),
        ("band_window", "highband", 96, "wall", 1.4),
        ("roof_metal", "roof", 192, 12.0, 9.0),
        ("roof_flat", "roof", 256, 16.0, 16.0),
        ("trim_dark", "trim", 64, 4.0, 0.3),
    ),
}

# Wall families available per group (strip base names above).
GROUP_FAMILIES = {
    "res": ("siding", "brick", "stucco"),
    "apt": ("concrete", "panel", "brick", "curtain"),
    "com": ("concrete", "brick", "stucco", "curtain"),
    "ind": ("metal", "concrete", "brick"),
}

GROUP_SHADES = {
    "res": ("a", "b", "c"),
    "apt": ("a", "b"),
    "com": ("a", "b"),
    "ind": ("a", "b"),
}


# --------------------------------------------------------------------------
# Per-combo dimensions: windows, doors, rhythm, floor heights.
#
# All in meters.  ``period`` is the facade window rhythm; the layout builder
# snaps the wall strip's U repeat to ``period * windows_per_repeat`` so the
# texture tiles seamlessly.  ``ground_h`` is the GEOMETRIC ground-floor
# height: builders keep the F*3.2 total-height filename contract by
# squeezing upper floors to (F*3.2 - ground_h)/(F-1).
# ``floor_tex_h`` is the texture-space cell height for upper floors.
# --------------------------------------------------------------------------

_DIM_DEFAULTS = {
    "windows_per_repeat": 8,
    "floor_tex_h": 3.2,
    "sill": 0.9,
    "door_w": 1.0,
    "door_h": 2.1,
    "ground_h": 3.2,
}

COMBO_DIMS = {
    # ---- residential -----------------------------------------------------
    ("generic", "res"): {
        "window_w": 1.25, "window_h": 1.45, "period": 2.5, "sill": 0.9,
        "door_w": 1.0, "door_h": 2.05,
    },
    ("europe", "res"): {
        "window_w": 1.05, "window_h": 1.7, "period": 2.3, "sill": 0.85,
        "door_w": 0.95, "door_h": 2.05, "floor_tex_h": 3.0,
    },
    ("north_america", "res"): {
        "window_w": 1.5, "window_h": 1.4, "period": 2.7, "sill": 0.85,
        "door_w": 1.0, "door_h": 2.05,
    },
    ("mediterranean", "res"): {
        "window_w": 0.95, "window_h": 1.6, "period": 2.6, "sill": 0.8,
        "door_w": 1.0, "door_h": 2.2,
    },
    ("asia", "res"): {
        "window_w": 1.8, "window_h": 1.25, "period": 2.6, "sill": 1.0,
        "door_w": 0.9, "door_h": 2.0, "ground_h": 3.0, "floor_tex_h": 3.0,
    },
    ("africa", "res"): {
        "window_w": 0.9, "window_h": 1.05, "period": 3.0, "sill": 1.1,
        "door_w": 0.95, "door_h": 2.0, "ground_h": 3.0, "floor_tex_h": 3.0,
    },
    ("south_america", "res"): {
        "window_w": 1.1, "window_h": 1.3, "period": 2.8, "sill": 0.95,
        "door_w": 0.95, "door_h": 2.1,
    },
    ("australia_oceania", "res"): {
        "window_w": 1.6, "window_h": 1.2, "period": 2.9, "sill": 0.9,
        "door_w": 0.95, "door_h": 2.05,
    },
    # ---- apartments --------------------------------------------------------
    ("generic", "apt"): {
        "window_w": 1.8, "window_h": 1.6, "period": 3.0, "sill": 0.75,
        "door_w": 1.6, "door_h": 2.4, "ground_h": 3.6,
    },
    ("europe", "apt"): {
        "window_w": 1.6, "window_h": 1.55, "period": 3.0, "sill": 0.8,
        "door_w": 1.5, "door_h": 2.3, "ground_h": 3.4, "floor_tex_h": 3.0,
    },
    ("north_america", "apt"): {
        "window_w": 1.9, "window_h": 1.5, "period": 3.2, "sill": 0.8,
        "door_w": 1.7, "door_h": 2.4, "ground_h": 3.8,
    },
    ("mediterranean", "apt"): {
        "window_w": 1.5, "window_h": 1.6, "period": 2.9, "sill": 0.7,
        "door_w": 1.5, "door_h": 2.3, "ground_h": 3.5,
    },
    ("asia", "apt"): {
        "window_w": 2.2, "window_h": 1.6, "period": 2.8, "sill": 0.85,
        "door_w": 1.6, "door_h": 2.4, "ground_h": 3.6, "floor_tex_h": 3.0,
    },
    ("africa", "apt"): {
        "window_w": 1.4, "window_h": 1.3, "period": 3.0, "sill": 1.0,
        "door_w": 1.5, "door_h": 2.3, "ground_h": 3.3,
    },
    ("south_america", "apt"): {
        "window_w": 1.7, "window_h": 1.5, "period": 2.9, "sill": 0.9,
        "door_w": 1.5, "door_h": 2.3, "ground_h": 3.4,
    },
    ("australia_oceania", "apt"): {
        "window_w": 2.0, "window_h": 1.6, "period": 3.1, "sill": 0.8,
        "door_w": 1.7, "door_h": 2.4, "ground_h": 3.7,
    },
    # ---- commercial --------------------------------------------------------
    ("generic", "com"): {
        "window_w": 1.9, "window_h": 1.5, "period": 3.0, "sill": 0.9,
        "door_w": 1.8, "door_h": 2.6, "ground_h": 4.0,
    },
    ("europe", "com"): {
        "window_w": 1.4, "window_h": 1.7, "period": 2.7, "sill": 0.85,
        "door_w": 1.6, "door_h": 2.5, "ground_h": 3.8, "floor_tex_h": 3.1,
    },
    ("north_america", "com"): {
        "window_w": 2.3, "window_h": 1.5, "period": 3.4, "sill": 0.95,
        "door_w": 1.9, "door_h": 2.7, "ground_h": 4.2,
    },
    ("mediterranean", "com"): {
        "window_w": 1.3, "window_h": 1.6, "period": 2.8, "sill": 0.8,
        "door_w": 1.6, "door_h": 2.5, "ground_h": 3.6,
    },
    ("asia", "com"): {
        "window_w": 2.1, "window_h": 1.5, "period": 2.7, "sill": 0.9,
        "door_w": 1.6, "door_h": 2.5, "ground_h": 3.8, "floor_tex_h": 3.1,
    },
    ("africa", "com"): {
        "window_w": 1.4, "window_h": 1.3, "period": 2.9, "sill": 1.0,
        "door_w": 1.6, "door_h": 2.4, "ground_h": 3.4,
    },
    ("south_america", "com"): {
        "window_w": 1.6, "window_h": 1.4, "period": 2.8, "sill": 0.95,
        "door_w": 1.6, "door_h": 2.4, "ground_h": 3.6,
    },
    ("australia_oceania", "com"): {
        "window_w": 2.1, "window_h": 1.5, "period": 3.2, "sill": 0.9,
        "door_w": 1.8, "door_h": 2.6, "ground_h": 4.0,
    },
    # ---- industrial --------------------------------------------------------
    ("generic", "ind"): {
        "window_w": 2.4, "window_h": 1.0, "period": 4.0, "sill": 1.8,
        "door_w": 3.6, "door_h": 3.4, "ground_h": 4.2,
    },
    ("europe", "ind"): {
        "window_w": 2.2, "window_h": 1.1, "period": 3.8, "sill": 1.7,
        "door_w": 3.4, "door_h": 3.4, "ground_h": 4.0,
    },
    ("north_america", "ind"): {
        "window_w": 2.6, "window_h": 1.0, "period": 4.4, "sill": 1.9,
        "door_w": 4.0, "door_h": 3.6, "ground_h": 4.4,
    },
    ("mediterranean", "ind"): {
        "window_w": 2.0, "window_h": 1.0, "period": 3.8, "sill": 1.7,
        "door_w": 3.2, "door_h": 3.2, "ground_h": 3.9,
    },
    ("asia", "ind"): {
        "window_w": 2.4, "window_h": 1.1, "period": 3.6, "sill": 1.6,
        "door_w": 3.2, "door_h": 3.2, "ground_h": 4.0,
    },
    ("africa", "ind"): {
        "window_w": 1.8, "window_h": 0.9, "period": 3.6, "sill": 1.8,
        "door_w": 3.0, "door_h": 3.0, "ground_h": 3.8,
    },
    ("south_america", "ind"): {
        "window_w": 2.0, "window_h": 1.0, "period": 3.7, "sill": 1.7,
        "door_w": 3.2, "door_h": 3.2, "ground_h": 3.9,
    },
    ("australia_oceania", "ind"): {
        "window_w": 2.4, "window_h": 1.0, "period": 4.2, "sill": 1.8,
        "door_w": 3.8, "door_h": 3.4, "ground_h": 4.2,
    },
}


def combo_dims(flavor: str, group: str) -> dict:
    dims = dict(_DIM_DEFAULTS)
    dims.update(COMBO_DIMS.get((flavor, group))
                or COMBO_DIMS[("generic", group)])
    # Snap the wall strip's world width to a whole number of window periods
    # so windows tile seamlessly across the U wrap.
    n = int(dims.get("windows_per_repeat", 8))
    dims["wall_strip_w"] = round(float(dims["period"]) * n, 2)
    return dims


# --------------------------------------------------------------------------
# Per-combo wall-family and roof weights.  These replace the old global
# FLAVOR_STYLES families/roofs for the group axis; class profiles from
# reference_styles.yaml still merge on top (per-class differentiation
# inside a group).
# --------------------------------------------------------------------------

COMBO_FAMILY_WEIGHTS = {
    ("generic", "res"): (("siding", 2), ("brick", 2), ("stucco", 2)),
    ("europe", "res"): (("stucco", 3), ("brick", 3), ("siding", 1)),
    ("north_america", "res"): (("siding", 4), ("brick", 2), ("stucco", 1)),
    ("mediterranean", "res"): (("stucco", 5), ("brick", 1), ("siding", 0)),
    ("asia", "res"): (("stucco", 3), ("brick", 2), ("siding", 1)),
    ("africa", "res"): (("stucco", 5), ("brick", 1), ("siding", 1)),
    ("south_america", "res"): (("stucco", 4), ("brick", 2), ("siding", 0)),
    ("australia_oceania", "res"): (("brick", 3), ("siding", 3), ("stucco", 1)),

    ("generic", "apt"): (("concrete", 3), ("panel", 2), ("brick", 2),
                         ("curtain", 2)),
    ("europe", "apt"): (("panel", 3), ("brick", 2), ("concrete", 2),
                        ("curtain", 1)),
    ("north_america", "apt"): (("brick", 3), ("concrete", 2), ("panel", 1),
                               ("curtain", 2)),
    ("mediterranean", "apt"): (("panel", 3), ("concrete", 3), ("brick", 1),
                               ("curtain", 1)),
    ("asia", "apt"): (("panel", 3), ("concrete", 3), ("curtain", 2),
                      ("brick", 0)),
    ("africa", "apt"): (("concrete", 3), ("panel", 3), ("brick", 1),
                        ("curtain", 0)),
    ("south_america", "apt"): (("concrete", 3), ("panel", 2), ("brick", 2),
                               ("curtain", 1)),
    ("australia_oceania", "apt"): (("brick", 2), ("concrete", 2),
                                   ("panel", 2), ("curtain", 2)),

    ("generic", "com"): (("concrete", 2), ("brick", 2), ("stucco", 2),
                         ("curtain", 2)),
    ("europe", "com"): (("stucco", 3), ("brick", 3), ("concrete", 1),
                        ("curtain", 2)),
    ("north_america", "com"): (("brick", 3), ("concrete", 2), ("stucco", 1),
                               ("curtain", 3)),
    ("mediterranean", "com"): (("stucco", 4), ("brick", 1), ("concrete", 1),
                               ("curtain", 1)),
    ("asia", "com"): (("concrete", 3), ("stucco", 2), ("curtain", 3),
                      ("brick", 1)),
    ("africa", "com"): (("stucco", 3), ("concrete", 3), ("brick", 1),
                        ("curtain", 1)),
    ("south_america", "com"): (("stucco", 3), ("concrete", 2), ("brick", 2),
                               ("curtain", 1)),
    ("australia_oceania", "com"): (("brick", 2), ("concrete", 2),
                                   ("stucco", 1), ("curtain", 3)),

    ("generic", "ind"): (("metal", 3), ("concrete", 3), ("brick", 1)),
    ("europe", "ind"): (("metal", 3), ("concrete", 2), ("brick", 2)),
    ("north_america", "ind"): (("metal", 3), ("concrete", 3), ("brick", 1)),
    ("mediterranean", "ind"): (("concrete", 3), ("metal", 2), ("brick", 1)),
    ("asia", "ind"): (("metal", 3), ("concrete", 3), ("brick", 1)),
    ("africa", "ind"): (("metal", 4), ("concrete", 2), ("brick", 1)),
    ("south_america", "ind"): (("metal", 3), ("concrete", 2), ("brick", 2)),
    ("australia_oceania", "ind"): (("metal", 4), ("concrete", 2),
                                   ("brick", 1)),
}

COMBO_ROOF_WEIGHTS = {
    ("generic", "res"): (("roof_shingle", 2), ("roof_tile", 2),
                         ("roof_metal", 1), ("roof_flat", 1)),
    ("europe", "res"): (("roof_tile", 4), ("roof_shingle", 1),
                        ("roof_metal", 0), ("roof_flat", 0)),
    ("north_america", "res"): (("roof_shingle", 4), ("roof_metal", 1),
                               ("roof_tile", 1), ("roof_flat", 0)),
    ("mediterranean", "res"): (("roof_tile", 5), ("roof_metal", 0),
                               ("roof_flat", 2)),
    ("asia", "res"): (("roof_metal", 2), ("roof_tile", 2), ("roof_flat", 2)),
    ("africa", "res"): (("roof_metal", 4), ("roof_tile", 1),
                        ("roof_flat", 2)),
    ("south_america", "res"): (("roof_tile", 3), ("roof_metal", 2),
                               ("roof_flat", 1)),
    ("australia_oceania", "res"): (("roof_metal", 4), ("roof_tile", 2),
                                   ("roof_shingle", 0), ("roof_flat", 0)),

    ("generic", "apt"): (("roof_flat", 3), ("roof_tile", 1),
                         ("roof_metal", 1)),
    ("europe", "apt"): (("roof_flat", 2), ("roof_tile", 3),
                        ("roof_metal", 0)),
    ("north_america", "apt"): (("roof_flat", 3), ("roof_tile", 1),
                               ("roof_metal", 1)),
    ("mediterranean", "apt"): (("roof_flat", 3), ("roof_tile", 2)),
    ("asia", "apt"): (("roof_flat", 5),),
    ("africa", "apt"): (("roof_flat", 4), ("roof_metal", 1)),
    ("south_america", "apt"): (("roof_flat", 4), ("roof_tile", 1)),
    ("australia_oceania", "apt"): (("roof_flat", 3), ("roof_metal", 2)),

    ("generic", "com"): (("roof_flat", 4), ("roof_metal", 1),
                         ("roof_tile", 1)),
    ("europe", "com"): (("roof_flat", 3), ("roof_tile", 2)),
    ("north_america", "com"): (("roof_flat", 5), ("roof_metal", 1)),
    ("mediterranean", "com"): (("roof_flat", 3), ("roof_tile", 2)),
    ("asia", "com"): (("roof_flat", 5),),
    ("africa", "com"): (("roof_flat", 4), ("roof_metal", 1)),
    ("south_america", "com"): (("roof_flat", 4), ("roof_tile", 1)),
    ("australia_oceania", "com"): (("roof_flat", 4), ("roof_metal", 1)),

    ("generic", "ind"): (("roof_metal", 3), ("roof_flat", 2)),
    ("europe", "ind"): (("roof_metal", 3), ("roof_flat", 2)),
    ("north_america", "ind"): (("roof_flat", 3), ("roof_metal", 2)),
    ("mediterranean", "ind"): (("roof_metal", 2), ("roof_flat", 3)),
    ("asia", "ind"): (("roof_metal", 3), ("roof_flat", 2)),
    ("africa", "ind"): (("roof_metal", 4), ("roof_flat", 1)),
    ("south_america", "ind"): (("roof_metal", 3), ("roof_flat", 2)),
    ("australia_oceania", "ind"): (("roof_metal", 4), ("roof_flat", 1)),
}


# --------------------------------------------------------------------------
# Massing per (flavor, class profile).
#
# ``_MASSING_BASE`` seeds every profile of a flavor; ``MASSING_PROFILES``
# overrides per class profile so class 1 and class 6 of the same region no
# longer share their silhouette parameters.  Consumed by the builders via
# ``combo_massing(flavor, profile)``.
#
# Keys: pitch/hip_pitch (deg ranges), overhang (multiplier), chimney_prob,
# tank_prob, parapet (m range), balcony_prob, balcony_depth (m),
# balcony_w (m), bay_w (m), flat_prob (aptblock flat-vs-hip), apt_flat
# (hard regional rule), ground_h duplicated from dims at lookup time.
# --------------------------------------------------------------------------

_MASSING_BASE = {
    "generic": {
        "pitch": (32.0, 42.0), "hip_pitch": (26.0, 34.0), "overhang": 1.0,
        "chimney_prob": 0.55, "tank_prob": 0.25, "parapet": (0.6, 0.9),
        "balcony_prob": 0.6, "balcony_depth": 0.8, "balcony_w": 2.2,
        "bay_w": 7.0, "flat_prob": 0.45,
    },
    "europe": {
        "pitch": (38.0, 50.0), "hip_pitch": (30.0, 38.0), "overhang": 0.9,
        "chimney_prob": 0.8, "tank_prob": 0.0, "parapet": (0.4, 0.7),
        "balcony_prob": 0.5, "balcony_depth": 0.7, "balcony_w": 2.0,
        "bay_w": 5.6, "flat_prob": 0.35,
    },
    "north_america": {
        "pitch": (33.0, 45.0), "hip_pitch": (28.0, 36.0), "overhang": 1.0,
        "chimney_prob": 0.6, "tank_prob": 0.05, "parapet": (0.7, 1.1),
        "balcony_prob": 0.45, "balcony_depth": 0.9, "balcony_w": 2.4,
        "bay_w": 7.5, "flat_prob": 0.55,
    },
    "mediterranean": {
        "pitch": (16.0, 24.0), "hip_pitch": (14.0, 21.0), "overhang": 1.1,
        "chimney_prob": 0.2, "tank_prob": 0.35, "parapet": (0.5, 0.8),
        "balcony_prob": 0.8, "balcony_depth": 1.0, "balcony_w": 2.2,
        "bay_w": 5.8, "flat_prob": 0.6,
    },
    "asia": {
        "pitch": (20.0, 29.0), "hip_pitch": (17.0, 25.0), "overhang": 1.5,
        "chimney_prob": 0.05, "tank_prob": 0.55, "parapet": (0.9, 1.3),
        "balcony_prob": 0.75, "balcony_depth": 0.9, "balcony_w": 2.6,
        "bay_w": 5.0, "flat_prob": 1.0, "apt_flat": True,
    },
    "africa": {
        "pitch": (12.0, 21.0), "hip_pitch": (11.0, 18.0), "overhang": 1.2,
        "chimney_prob": 0.05, "tank_prob": 0.6, "parapet": (0.35, 0.65),
        "balcony_prob": 0.35, "balcony_depth": 0.7, "balcony_w": 2.0,
        "bay_w": 5.4, "flat_prob": 0.85,
    },
    "south_america": {
        "pitch": (16.0, 26.0), "hip_pitch": (14.0, 22.0), "overhang": 1.1,
        "chimney_prob": 0.1, "tank_prob": 0.5, "parapet": (0.5, 0.85),
        "balcony_prob": 0.7, "balcony_depth": 0.9, "balcony_w": 2.2,
        "bay_w": 5.6, "flat_prob": 0.7,
    },
    "australia_oceania": {
        "pitch": (18.0, 28.0), "hip_pitch": (16.0, 24.0), "overhang": 1.35,
        "chimney_prob": 0.3, "tank_prob": 0.1, "parapet": (0.6, 0.95),
        "balcony_prob": 0.55, "balcony_depth": 0.9, "balcony_w": 2.4,
        "bay_w": 7.2, "flat_prob": 0.5,
    },
}

# Class-profile overrides per flavor: {flavor: {profile: {key: value}}}.
# Only deltas are listed; everything else inherits _MASSING_BASE[flavor].
MASSING_PROFILES = {
    "generic": {
        "tiny_residential": {"pitch": (30.0, 40.0), "chimney_prob": 0.45},
        "compact_residential": {"pitch": (34.0, 44.0), "chimney_prob": 0.65},
        "medium": {"balcony_prob": 0.5, "parapet": (0.6, 0.9)},
        "small_apartment": {"balcony_prob": 0.65, "parapet": (0.7, 1.0)},
        "apartment_block": {"balcony_prob": 0.7, "parapet": (0.8, 1.1),
                            "flat_prob": 0.7},
        "large": {"parapet": (0.9, 1.2)},
        "extra_large": {"parapet": (1.0, 1.4)},
    },
    "europe": {
        "tiny_residential": {"pitch": (40.0, 52.0), "chimney_prob": 0.85},
        "small_residential": {"pitch": (38.0, 50.0)},
        "compact_residential": {"pitch": (36.0, 46.0), "bay_w": 5.2},
        "medium": {"balcony_prob": 0.45, "flat_prob": 0.3,
                   "hip_pitch": (26.0, 34.0)},
        "small_apartment": {"balcony_prob": 0.55, "parapet": (0.5, 0.8),
                            "flat_prob": 0.45},
        "apartment_block": {"balcony_prob": 0.65, "parapet": (0.6, 0.9),
                            "flat_prob": 0.7, "hip_pitch": (24.0, 30.0)},
        "large": {"parapet": (0.8, 1.1)},
        "extra_large": {"parapet": (0.9, 1.2)},
    },
    "north_america": {
        "tiny_residential": {"pitch": (30.0, 42.0)},
        "compact_residential": {"pitch": (34.0, 46.0), "chimney_prob": 0.7},
        "medium": {"flat_prob": 0.5, "balcony_prob": 0.4},
        "small_apartment": {"flat_prob": 0.6, "balcony_prob": 0.5},
        "apartment_block": {"flat_prob": 0.8, "balcony_prob": 0.55,
                            "parapet": (0.9, 1.2)},
        "large": {"parapet": (1.0, 1.3)},
        "extra_large": {"parapet": (1.1, 1.5)},
    },
    "mediterranean": {
        "tiny_residential": {"tank_prob": 0.4, "parapet": (0.45, 0.7)},
        "compact_residential": {"balcony_prob": 0.85},
        "medium": {"balcony_prob": 0.85, "flat_prob": 0.65},
        "small_apartment": {"balcony_prob": 0.9, "flat_prob": 0.75},
        "apartment_block": {"balcony_prob": 0.9, "flat_prob": 0.85,
                            "parapet": (0.6, 0.9)},
        "large": {"parapet": (0.7, 1.0)},
        "extra_large": {"parapet": (0.8, 1.1)},
    },
    "asia": {
        "tiny_residential": {"pitch": (22.0, 32.0), "tank_prob": 0.45},
        "small_residential": {"pitch": (20.0, 30.0)},
        "compact_residential": {"bay_w": 4.6, "tank_prob": 0.6},
        "medium": {"balcony_prob": 0.8, "parapet": (0.9, 1.3)},
        "small_apartment": {"balcony_prob": 0.85, "parapet": (1.0, 1.4)},
        "apartment_block": {"balcony_prob": 0.85, "parapet": (1.0, 1.5)},
        "large": {"parapet": (0.9, 1.3)},
        "extra_large": {"parapet": (1.0, 1.4)},
    },
    "africa": {
        "tiny_residential": {"pitch": (10.0, 18.0), "tank_prob": 0.65},
        "compact_residential": {"tank_prob": 0.7},
        "medium": {"balcony_prob": 0.4},
        "small_apartment": {"balcony_prob": 0.45},
        "apartment_block": {"balcony_prob": 0.5, "parapet": (0.5, 0.8)},
        "large": {"parapet": (0.6, 0.9)},
        "extra_large": {"parapet": (0.7, 1.0)},
    },
    "south_america": {
        "tiny_residential": {"pitch": (14.0, 24.0), "tank_prob": 0.55},
        "compact_residential": {"balcony_prob": 0.75, "bay_w": 5.2},
        "medium": {"balcony_prob": 0.75},
        "small_apartment": {"balcony_prob": 0.8, "parapet": (0.6, 0.95)},
        "apartment_block": {"balcony_prob": 0.8, "parapet": (0.7, 1.05)},
        "large": {"parapet": (0.7, 1.0)},
        "extra_large": {"parapet": (0.8, 1.1)},
    },
    "australia_oceania": {
        "tiny_residential": {"pitch": (16.0, 26.0), "overhang": 1.45},
        "compact_residential": {"pitch": (20.0, 30.0)},
        "medium": {"balcony_prob": 0.6, "flat_prob": 0.45},
        "small_apartment": {"balcony_prob": 0.65, "flat_prob": 0.55},
        "apartment_block": {"balcony_prob": 0.7, "flat_prob": 0.7,
                            "parapet": (0.7, 1.0)},
        "large": {"parapet": (0.9, 1.2)},
        "extra_large": {"parapet": (1.0, 1.3)},
    },
}


def combo_massing(flavor: str, profile: str | None = None) -> dict:
    base = dict(_MASSING_BASE.get(flavor) or _MASSING_BASE["generic"])
    overrides = (MASSING_PROFILES.get(flavor)
                 or MASSING_PROFILES["generic"]).get(profile or "", {})
    base.update(overrides)
    return base


def combo_family_weights(flavor: str, group: str) -> tuple:
    return (COMBO_FAMILY_WEIGHTS.get((flavor, group))
            or COMBO_FAMILY_WEIGHTS[("generic", group)])


def combo_roof_weights(flavor: str, group: str) -> tuple:
    weights = (COMBO_ROOF_WEIGHTS.get((flavor, group))
               or COMBO_ROOF_WEIGHTS[("generic", group)])
    return tuple((name, w) for name, w in weights if w > 0)


# --------------------------------------------------------------------------
# Per-combo palette overrides.
#
# The atlas painter/compositor starts from atlas.PALETTES[flavor] (regional
# hue identity) and deep-merges these on top, so an apartment combo can own
# panel/glass/lobby colors and an industrial combo its cladding pools
# without duplicating the whole regional table.  Colors are (r, g, b).
# --------------------------------------------------------------------------

COMBO_PALETTES = {
    ("generic", "apt"): {
        "panel": ((172, 168, 160), (150, 148, 144)),
        "curtain": ((88, 104, 118), (66, 84, 100)),
        "lobby": (70, 78, 84),
        "balcony": (150, 148, 145),
    },
    ("europe", "apt"): {
        "panel": ((196, 186, 170), (168, 158, 150)),
        "curtain": ((92, 106, 116), (70, 86, 98)),
        "lobby": (76, 82, 88),
        "balcony": (120, 118, 116),
    },
    ("north_america", "apt"): {
        "panel": ((188, 176, 158), (160, 148, 134)),
        "curtain": ((84, 100, 114), (62, 80, 96)),
        "lobby": (66, 74, 82),
        "balcony": (140, 138, 134),
    },
    ("mediterranean", "apt"): {
        "panel": ((228, 208, 184), (214, 188, 158)),
        "curtain": ((96, 112, 120), (74, 92, 102)),
        "lobby": (88, 92, 94),
        "balcony": (206, 196, 182),
    },
    ("asia", "apt"): {
        "panel": ((208, 200, 190), (186, 178, 168)),
        "curtain": ((78, 96, 104), (56, 76, 86)),
        "lobby": (64, 72, 78),
        "balcony": (164, 160, 154),
    },
    ("africa", "apt"): {
        "panel": ((198, 176, 148), (176, 156, 128)),
        "curtain": ((86, 100, 106), (64, 80, 88)),
        "lobby": (72, 76, 78),
        "balcony": (170, 158, 140),
    },
    ("south_america", "apt"): {
        "panel": ((202, 158, 138), (178, 168, 156)),
        "curtain": ((82, 98, 108), (60, 78, 90)),
        "lobby": (70, 76, 80),
        "balcony": (172, 162, 150),
    },
    ("australia_oceania", "apt"): {
        "panel": ((212, 206, 196), (186, 182, 174)),
        "curtain": ((90, 106, 118), (68, 86, 100)),
        "lobby": (72, 80, 86),
        "balcony": (158, 156, 150),
    },

    ("generic", "com"): {
        "curtain": ((84, 102, 116), (62, 82, 98)),
        "lobby": (62, 72, 80),
    },
    ("europe", "com"): {
        "curtain": ((90, 104, 114), (68, 84, 96)),
        "lobby": (70, 78, 84),
    },
    ("north_america", "com"): {
        "curtain": ((78, 98, 114), (56, 78, 96)),
        "lobby": (58, 70, 80),
    },
    ("mediterranean", "com"): {
        "curtain": ((98, 112, 120), (76, 92, 102)),
        "lobby": (84, 90, 92),
    },
    ("asia", "com"): {
        "curtain": ((72, 92, 102), (50, 72, 84)),
        "lobby": (58, 68, 74),
    },
    ("africa", "com"): {
        "curtain": ((88, 100, 104), (66, 80, 86)),
        "lobby": (74, 78, 78),
    },
    ("south_america", "com"): {
        "curtain": ((82, 98, 106), (60, 78, 88)),
        "lobby": (68, 74, 78),
    },
    ("australia_oceania", "com"): {
        "curtain": ((86, 104, 118), (64, 84, 100)),
        "lobby": (66, 76, 84),
    },

    ("generic", "ind"): {
        "metal": ((176, 178, 180), (150, 154, 158)),
        "tiltup": (168, 164, 158),
    },
    ("europe", "ind"): {
        "metal": ((162, 172, 166), (140, 148, 146)),
        "tiltup": (162, 158, 152),
    },
    ("north_america", "ind"): {
        "metal": ((214, 214, 210), (188, 184, 176)),
        "tiltup": (178, 172, 162),
    },
    ("mediterranean", "ind"): {
        "metal": ((206, 198, 182), (182, 174, 158)),
        "tiltup": (186, 178, 164),
    },
    ("asia", "ind"): {
        "metal": ((128, 142, 156), (108, 120, 132)),
        "tiltup": (158, 156, 152),
    },
    ("africa", "ind"): {
        "metal": ((168, 162, 150), (138, 126, 112)),
        "tiltup": (166, 158, 144),
    },
    ("south_america", "ind"): {
        "metal": ((172, 158, 142), (146, 134, 122)),
        "tiltup": (164, 158, 148),
    },
    ("australia_oceania", "ind"): {
        "metal": ((222, 216, 202), (192, 188, 178)),
        "tiltup": (182, 178, 168),
    },
}


def combo_palette_overrides(flavor: str, group: str) -> dict:
    return dict(COMBO_PALETTES.get((flavor, group)) or {})


# --------------------------------------------------------------------------
# Per-combo pattern overrides (paint-side knobs on top of the regional
# FLAVOR_PATTERNS baseline).  Window/door dims come from combo_dims and are
# merged in by atlas.pattern_for_combo; entries here carry the qualitative
# knobs that differ per group.
# --------------------------------------------------------------------------

COMBO_PATTERNS = {
    # Apartments: sliders, rails and AC in the east/south, roller shutters
    # in Europe; more banding on painted concrete fabrics.
    ("generic", "apt"): {"mullion": "slider", "shutters": False,
                         "rail": True, "unit_banding": 0.2},
    ("europe", "apt"): {"mullion": "single", "shutters": False,
                        "rail": True, "unit_banding": 0.25,
                        "roller_shutter": True},
    ("north_america", "apt"): {"mullion": "slider", "shutters": False,
                               "rail": False, "unit_banding": 0.1},
    ("mediterranean", "apt"): {"mullion": "single", "shutters": True,
                               "rail": True, "unit_banding": 0.35,
                               "louvered": True},
    ("asia", "apt"): {"mullion": "slider", "rail": True, "ac_prob": 0.6,
                      "unit_banding": 0.7, "streaks": 0.0, "grime": 0.0},
    ("africa", "apt"): {"mullion": "single", "bars_prob": 0.6, "rail": False,
                        "unit_banding": 0.5, "streaks": 0.55, "grime": 0.5},
    ("south_america", "apt"): {"mullion": "slider", "bars_prob": 0.4,
                               "rail": True, "unit_banding": 0.6,
                               "streaks": 0.45, "grime": 0.35},
    ("australia_oceania", "apt"): {"mullion": "slider", "rail": True,
                                   "unit_banding": 0.15},

    # Commercial: office glass, storefront design pairs, awnings.
    ("generic", "com"): {"mullion": "single", "shutters": False,
                         "unit_banding": 0.15},
    ("europe", "com"): {"mullion": "cross", "transom": True,
                        "unit_banding": 0.3, "awning_stripes": True},
    ("north_america", "com"): {"mullion": "single", "unit_banding": 0.05},
    ("mediterranean", "com"): {"mullion": "single", "arch": True,
                               "awning_stripes": True, "unit_banding": 0.4},
    ("asia", "com"): {"mullion": "slider", "ac_prob": 0.5,
                      "unit_banding": 0.85, "streaks": 0.0, "grime": 0.0},
    ("africa", "com"): {"mullion": "single", "bars_prob": 0.55,
                        "unit_banding": 0.5, "streaks": 0.5, "grime": 0.5},
    ("south_america", "com"): {"mullion": "single", "bars_prob": 0.35,
                               "unit_banding": 0.6, "streaks": 0.45},
    ("australia_oceania", "com"): {"mullion": "slider",
                                   "unit_banding": 0.1},

    # Industrial: sparse high glazing, no residential cues anywhere.
    ("generic", "ind"): {"mullion": "single", "shutters": False,
                         "rail": False, "arch": False, "transom": False,
                         "unit_banding": 0.0, "grime": 0.3, "streaks": 0.4},
    ("europe", "ind"): {"grime": 0.25, "streaks": 0.3},
    ("north_america", "ind"): {"grime": 0.15, "streaks": 0.2},
    ("mediterranean", "ind"): {"grime": 0.3, "streaks": 0.35},
    ("asia", "ind"): {"grime": 0.0, "streaks": 0.0, "rust": False},
    ("africa", "ind"): {"grime": 0.5, "streaks": 0.6, "rust": True},
    ("south_america", "ind"): {"grime": 0.4, "streaks": 0.5, "rust": True},
    ("australia_oceania", "ind"): {"grime": 0.1, "streaks": 0.15},
}


def combo_pattern_overrides(flavor: str, group: str) -> dict:
    if group == "res":
        return {}
    merged = dict(COMBO_PATTERNS.get(("generic", group)) or {})
    merged.update(COMBO_PATTERNS.get((flavor, group)) or {})
    return merged
