"""Generate the shared procedural facade/roof texture atlases (PIL).

One 2048x2048 PNG per region flavor; every generated OBJ references exactly
one atlas, so the whole library costs eight textures.  The atlas is a stack
of full-width horizontal strips: U tiles freely (GL wrap) while V stays
inside a strip's band, so a wall of any length is a single quad whose
windows never stretch.

Paint rules (learned the hard way -- see the rainbow/banding history):
  * NOTHING may be painted as a full-width line in a per-line random color.
    Pitched roofs magnify each strip row into a ~40 cm band across the whole
    roof, so per-row randomness renders as giant color bands in-sim.  All
    randomness is per CELL (shingle tab, tile pan, metal sheet, brick);
    structural lines (course shadows, seams, joints) use a deterministic
    offset of the base color via _adjust().
  * Every periodic feature must complete an integer number of cells across
    the strip width (walls tile up to ~5 U repeats, roofs are capped at 2),
    so cells are laid out with _row_cells() which wraps the split cell.
  * Keep 1px detail at moderate contrast: X-Plane box-filters PNG mips and
    high-contrast pixels shimmer / tint the deep mips.

Outputs:
  <output>/textures/o4sfr_procgen_atlas_<flavor>.png   (one per flavor)
  atlas_layout.json (next to this script) -- strip name -> V band + world
  meters per U repeat / per band height, consumed by the archetypes.

Run:  python atlas.py --output <Custom Scenery>/O4SFR_ProcGen_Library
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import yaml  # noqa: E402

# V inset per band edge (2048px units, scaled with atlas size) so mipmaps
# don't bleed across strips. 8px protects the first ~3 mip levels; combined
# with the untiled-UV caps on roofs/far meshes this keeps sampling out of
# the deep mips where the strip stack collapses into rainbow bands.
GUTTER_PX = 8

# (name, height_px, world_w_m, world_h_m, painter-kind)
STRIPS = (
    ("wall_siding_a",   96, 20.0, 3.2, "wall"),
    ("wall_siding_b",   96, 20.0, 3.2, "wall"),
    ("wall_brick_a",    96, 20.0, 3.2, "wall"),
    ("wall_brick_b",    96, 20.0, 3.2, "wall"),
    ("wall_stucco_a",   96, 20.0, 3.2, "wall"),
    ("wall_stucco_b",   96, 20.0, 3.2, "wall"),
    ("wall_concrete_a", 96, 20.0, 3.2, "wall"),
    ("wall_concrete_b", 96, 20.0, 3.2, "wall"),
    ("ground_siding",   96, 20.0, 3.2, "ground"),
    ("ground_brick",    96, 20.0, 3.2, "ground"),
    ("ground_stucco",   96, 20.0, 3.2, "ground"),
    ("ground_concrete", 96, 20.0, 3.2, "ground"),
    ("ground_storefront", 96, 20.0, 3.2, "storefront"),
    ("ground_roller",   96, 24.0, 3.2, "roller"),
    ("plain_siding",    64,  8.0, 2.0, "plain"),
    ("plain_brick",     64,  8.0, 2.0, "plain"),
    ("plain_stucco",    64,  8.0, 2.0, "plain"),
    ("plain_concrete",  64,  8.0, 2.0, "plain"),
    ("trim_dark",       32,  4.0, 0.3, "trim"),
    ("roof_shingle",    96, 12.0, 9.0, "roof"),
    ("roof_tile",       96, 12.0, 9.0, "roof"),
    ("roof_metal",      96, 12.0, 9.0, "roof"),
    ("roof_flat",      128, 16.0, 16.0, "roof"),
)

# Per-flavor material palettes (RGB).  Flavors only swap colors; geometry
# UVs stay identical because the layout is shared.  Besides the material
# bases, each flavor carries: window "frame"/"glass" colors, brick "mortar",
# and color pools for doors / awnings / shop signs / shutters.
PALETTES = {
    "generic": {
        "siding":   ((188, 182, 170), (170, 164, 152)),
        "brick":    ((150, 96, 78),   (132, 84, 70)),
        "stucco":   ((205, 198, 182), (186, 178, 164)),
        "concrete": ((168, 168, 166), (150, 150, 148)),
        "roof_shingle": (88, 86, 84),
        "roof_tile":    (158, 88, 64),
        "roof_metal":   (130, 134, 138),
        "roof_flat":    (142, 140, 136),
        "trim": (62, 58, 54),
        "frame": (225, 222, 214),
        "glass": (96, 112, 126),
        "mortar": (184, 172, 160),
        "door_pool": ((84, 62, 48), (110, 86, 64), (70, 72, 76),
                      (96, 40, 34)),
        "awning_pool": ((150, 140, 124), (110, 116, 122), (96, 104, 96)),
        "sign_pool": ((150, 60, 52), (62, 86, 110), (190, 170, 120),
                      (90, 110, 90)),
        "shutter_pool": ((62, 58, 54),),
    },
    "europe": {
        "siding":   ((196, 188, 168), (176, 168, 150)),
        "brick":    ((146, 88, 68),   (126, 76, 60)),
        "stucco":   ((222, 210, 184), (206, 192, 168)),
        "concrete": ((176, 174, 168), (158, 156, 152)),
        "roof_shingle": (94, 88, 82),
        "roof_tile":    (172, 92, 58),
        "roof_metal":   (122, 126, 130),
        "roof_flat":    (148, 146, 140),
        "trim": (70, 62, 54),
        "frame": (235, 232, 224),
        "glass": (92, 106, 120),
        "mortar": (190, 178, 160),
        "door_pool": ((96, 44, 40), (44, 68, 52), (40, 52, 76),
                      (38, 38, 40), (92, 66, 46)),
        "awning_pool": ((52, 78, 58), (110, 48, 52), (46, 58, 84)),
        "sign_pool": ((120, 96, 72), (60, 72, 96), (140, 52, 48)),
        "shutter_pool": ((90, 96, 88), (74, 68, 60)),
    },
    "north_america": {
        "siding":   ((202, 198, 188), (172, 178, 184)),
        "brick":    ((142, 84, 66),   (120, 74, 62)),
        "stucco":   ((212, 200, 178), (192, 182, 162)),
        "concrete": ((170, 170, 168), (152, 152, 150)),
        "roof_shingle": (78, 76, 74),
        "roof_tile":    (150, 94, 72),
        "roof_metal":   (134, 138, 142),
        "roof_flat":    (138, 136, 132),
        "trim": (58, 54, 50),
        "frame": (238, 236, 230),
        "glass": (100, 116, 130),
        "mortar": (186, 176, 166),
        "door_pool": ((224, 220, 212), (140, 44, 38), (160, 134, 100),
                      (42, 42, 44)),
        "awning_pool": ((48, 74, 54), (44, 56, 82), (48, 48, 50)),
        "sign_pool": ((148, 52, 46), (52, 70, 98), (170, 150, 110)),
        "shutter_pool": ((52, 60, 52), (40, 44, 52), (60, 50, 42)),
    },
    "mediterranean": {
        "siding":   ((216, 208, 192), (200, 190, 172)),
        "brick":    ((176, 124, 92),  (158, 110, 82)),
        "stucco":   ((238, 230, 212), (224, 210, 186)),  # whitewash / cream
        "concrete": ((196, 190, 178), (180, 174, 162)),
        "roof_shingle": (110, 96, 86),
        "roof_tile":    (188, 102, 62),  # bright clay
        "roof_metal":   (150, 148, 142),
        "roof_flat":    (188, 182, 170),  # pale terraces
        "trim": (88, 78, 66),
        "frame": (232, 226, 210),
        "glass": (88, 104, 116),
        "mortar": (200, 186, 164),
        "door_pool": ((108, 84, 60), (54, 76, 56), (76, 98, 116)),
        "awning_pool": ((176, 98, 66), (224, 216, 200), (160, 76, 60)),
        "sign_pool": ((170, 90, 60), (96, 110, 120), (180, 160, 130)),
        "shutter_pool": ((74, 96, 72), (88, 118, 134), (104, 78, 56)),
    },
    "asia": {
        "siding":   ((184, 180, 172), (164, 162, 156)),
        "brick":    ((148, 104, 88),  (130, 92, 78)),
        "stucco":   ((208, 204, 194), (190, 186, 176)),
        "concrete": ((178, 178, 174), (158, 158, 154)),  # weathered gray
        "roof_shingle": (72, 70, 68),
        "roof_tile":    (96, 84, 92),    # dark glazed tile
        "roof_metal":   (104, 118, 128), # blue-gray corrugated
        "roof_flat":    (150, 148, 144),
        "trim": (60, 58, 56),
        "frame": (70, 74, 78),           # dark aluminum
        "glass": (78, 92, 104),
        "mortar": (170, 164, 156),
        "door_pool": ((110, 112, 116), (134, 48, 42), (140, 144, 148)),
        "awning_pool": ((150, 60, 54), (60, 98, 130), (170, 140, 70)),
        "sign_pool": ((168, 52, 44), (196, 164, 60), (54, 88, 140),
                      (60, 118, 84)),
        "shutter_pool": ((88, 90, 92),),
    },
    "africa": {
        "siding":   ((198, 184, 162), (182, 168, 146)),
        "brick":    ((164, 112, 82),  (146, 100, 74)),
        "stucco":   ((226, 208, 178), (208, 188, 156)),  # warm render / adobe
        "concrete": ((186, 178, 164), (168, 160, 146)),
        "roof_shingle": (96, 88, 78),
        "roof_tile":    (166, 96, 64),
        "roof_metal":   (146, 134, 118),  # sun-bleached, rusty corrugated
        "roof_flat":    (172, 162, 144),
        "trim": (78, 68, 56),
        "frame": (104, 100, 92),
        "glass": (90, 102, 110),
        "mortar": (192, 176, 150),
        "door_pool": ((62, 96, 130), (66, 110, 76), (140, 56, 44),
                      (130, 128, 124)),
        "awning_pool": ((150, 110, 70), (96, 118, 126)),
        "sign_pool": ((170, 70, 50), (70, 110, 140), (190, 160, 80),
                      (90, 130, 90)),
        "shutter_pool": ((98, 92, 82),),
    },
    "south_america": {
        "siding":   ((204, 192, 174), (186, 174, 156)),
        "brick":    ((158, 96, 70),   (140, 86, 64)),   # exposed ladrillo
        "stucco":   ((222, 206, 182), (204, 188, 164)),
        "concrete": ((180, 176, 168), (162, 158, 150)),
        "roof_shingle": (90, 84, 78),
        "roof_tile":    (172, 90, 56),
        "roof_metal":   (138, 132, 122),
        "roof_flat":    (160, 154, 142),
        "trim": (72, 64, 54),
        "frame": (220, 214, 202),
        "glass": (92, 106, 118),
        "mortar": (188, 172, 152),
        "door_pool": ((120, 80, 52), (90, 56, 46), (70, 96, 120),
                      (84, 110, 82), (150, 120, 80)),
        "awning_pool": ((158, 72, 54), (70, 104, 128), (180, 150, 90)),
        "sign_pool": ((172, 64, 48), (64, 96, 134), (196, 168, 84),
                      (84, 128, 92)),
        "shutter_pool": ((86, 76, 64),),
    },
    "australia_oceania": {
        "siding":   ((206, 200, 186), (188, 184, 174)),
        "brick":    ((152, 92, 72),   (134, 100, 86)),
        "stucco":   ((214, 204, 184), (196, 188, 170)),
        "concrete": ((176, 174, 168), (158, 156, 150)),
        "roof_shingle": (84, 80, 76),
        "roof_tile":    (140, 82, 64),
        "roof_metal":   (158, 162, 166),  # Colorbond-style light metal
        "roof_flat":    (150, 146, 138),
        "trim": (64, 60, 56),
        "frame": (212, 212, 208),
        "glass": (100, 114, 126),
        "mortar": (204, 196, 182),        # cream mortar
        "door_pool": ((222, 220, 214), (84, 92, 98), (130, 62, 52)),
        "awning_pool": ((220, 218, 212), (70, 72, 74)),
        "sign_pool": ((96, 104, 110), (150, 70, 56), (70, 96, 120)),
        "shutter_pool": ((110, 114, 116),),
    },
}

WINDOW_PERIOD_M = 2.5

# Per-flavor PATTERN parameters (paint-side only: the strip layout and the
# UV contract are shared, so geometry never changes). Windows in metres;
# row/pitch values in pixels on the 2048px strip.
#   mullion: single | cross | sash | slider     (window divider style)
#   glass_mix: weights for sky/dark/curtain/warm glass per window
#   bars_prob/ac_prob: per-window security grille / AC unit probability
#   streaks: weathering streak probability under sills (concrete/stucco)
#   grime: 0..1 strength of the floor-line grime band (skipped for siding)
#   door_panels: panel2x2 | planks | flush
FLAVOR_PATTERNS = {
    "generic": {
        "window_w": 1.25, "window_h": 1.45, "period": 2.5, "shutters": False,
        "rail": False, "brick_row": 20, "siding_step": 16,
        "tile_row": 6, "shingle_row": 5, "corrugation": 14, "rust": False,
        "tile_gloss": False,
        "mullion": "single", "glass_mix": (3, 2, 1, 1),
        "bars_prob": 0.0, "ac_prob": 0.0, "arch": False, "transom": False,
        "louvered": False, "streaks": 0.2, "grime": 0.2,
        "awning_stripes": False, "door_panels": "panel2x2",
    },
    "europe": {
        "window_w": 1.1, "window_h": 1.7, "period": 2.4, "shutters": False,
        "rail": False, "brick_row": 22, "siding_step": 14,
        "tile_row": 6, "shingle_row": 5, "corrugation": 14, "rust": False,
        "tile_gloss": False,
        "mullion": "cross", "glass_mix": (3, 2, 2, 1),
        "bars_prob": 0.0, "ac_prob": 0.0, "arch": False, "transom": True,
        "louvered": False, "streaks": 0.3, "grime": 0.3,
        "awning_stripes": False, "door_panels": "panel2x2",
    },
    "north_america": {
        "window_w": 1.4, "window_h": 1.5, "period": 2.6, "shutters": True,
        "rail": False, "brick_row": 18, "siding_step": 18,
        "tile_row": 6, "shingle_row": 4, "corrugation": 16, "rust": False,
        "tile_gloss": False,
        "mullion": "sash", "glass_mix": (3, 2, 2, 1),
        "bars_prob": 0.0, "ac_prob": 0.0, "arch": False, "transom": False,
        "louvered": False, "streaks": 0.1, "grime": 0.15,
        "awning_stripes": False, "door_panels": "panel2x2",
    },
    "mediterranean": {
        "window_w": 1.0, "window_h": 1.5, "period": 2.7, "shutters": True,
        "rail": False, "brick_row": 20, "siding_step": 16,
        "tile_row": 9, "shingle_row": 5, "corrugation": 14, "rust": False,
        "tile_gloss": False,
        "mullion": "single", "glass_mix": (3, 2, 1, 1),
        "bars_prob": 0.0, "ac_prob": 0.0, "arch": True, "transom": False,
        "louvered": True, "streaks": 0.3, "grime": 0.25,
        "awning_stripes": True, "door_panels": "planks",
    },
    "asia": {
        "window_w": 1.9, "window_h": 1.25, "period": 2.8, "shutters": False,
        "rail": True, "brick_row": 18, "siding_step": 14,
        "tile_row": 5, "shingle_row": 5, "corrugation": 10, "rust": False,
        "tile_gloss": True,
        "mullion": "slider", "glass_mix": (2, 4, 1, 1),
        "bars_prob": 0.0, "ac_prob": 0.45, "arch": False, "transom": False,
        "louvered": False, "streaks": 0.8, "grime": 0.5,
        "awning_stripes": False, "door_panels": "flush",
    },
    "africa": {
        "window_w": 1.0, "window_h": 1.1, "period": 3.2, "shutters": False,
        "rail": False, "brick_row": 16, "siding_step": 16,
        "tile_row": 7, "shingle_row": 5, "corrugation": 10, "rust": True,
        "tile_gloss": False,
        "mullion": "single", "glass_mix": (2, 3, 1, 0),
        "bars_prob": 0.8, "ac_prob": 0.15, "arch": False, "transom": False,
        "louvered": False, "streaks": 0.6, "grime": 0.6,
        "awning_stripes": False, "door_panels": "planks",
    },
    "south_america": {
        "window_w": 1.15, "window_h": 1.35, "period": 2.8, "shutters": False,
        "rail": True, "brick_row": 16, "siding_step": 16,
        "tile_row": 8, "shingle_row": 5, "corrugation": 12, "rust": True,
        "tile_gloss": False,
        "mullion": "single", "glass_mix": (3, 3, 1, 1),
        "bars_prob": 0.5, "ac_prob": 0.15, "arch": False, "transom": False,
        "louvered": False, "streaks": 0.5, "grime": 0.4,
        "awning_stripes": False, "door_panels": "planks",
    },
    "australia_oceania": {
        "window_w": 1.5, "window_h": 1.3, "period": 2.7, "shutters": False,
        "rail": False, "brick_row": 19, "siding_step": 18,
        "tile_row": 7, "shingle_row": 5, "corrugation": 18, "rust": False,
        "tile_gloss": False,
        "mullion": "slider", "glass_mix": (3, 2, 1, 1),
        "bars_prob": 0.0, "ac_prob": 0.0, "arch": False, "transom": False,
        "louvered": False, "streaks": 0.1, "grime": 0.1,
        "awning_stripes": False, "door_panels": "flush",
    },
}


def _load_reference_styles(path=None):
    path = path or os.path.join(HERE, "reference_styles.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _pattern_for_flavor(flavor: str, references: dict) -> dict:
    pattern = dict(FLAVOR_PATTERNS.get(flavor, FLAVOR_PATTERNS["generic"]))
    region = (references.get("regions") or {}).get(flavor, {})
    pattern.update(region.get("pattern_overrides") or {})
    return pattern


def _S(px_scale, px):
    """Scale a 2048px-authored pixel constant to the actual atlas size."""
    return max(1, int(round(px * px_scale)))


def _jitter(rng, color, amount):
    """Independent per-channel jitter -- ONLY for glass/detail accents.

    On near-neutral bases, independent channels shift the HUE (teal, pink,
    olive...). Used at small scales that is rainbow noise: roof rows painted
    with this at 1024px caused the rainbow-roof artifact. Surface shading
    must use _shade() instead.
    """
    return tuple(
        max(0, min(255, c + rng.randint(-amount, amount))) for c in color
    )


def _shade(rng, color, amount):
    """Luminance-correlated jitter: one delta applied to all channels."""
    delta = rng.randint(-amount, amount)
    return tuple(max(0, min(255, c + delta)) for c in color)


def _adjust(color, delta):
    """DETERMINISTIC luminance offset -- for structural lines (course
    shadows, seams, joints).  Full-width lines must never roll the RNG:
    that is exactly the roof-banding bug."""
    return tuple(max(0, min(255, c + delta)) for c in color)


def _speckle(draw, rng, box, base, amount, count):
    x0, y0, x1, y1 = box
    for _ in range(count):
        x = rng.randint(x0, x1 - 1)
        y = rng.randint(y0, y1 - 1)
        draw.point((x, y), fill=_shade(rng, base, amount))


def _area_count(box, divisor):
    """Pixel-count helper so speckle density survives resolution changes."""
    x0, y0, x1, y1 = box
    return max(1, ((x1 - x0) * (y1 - y0)) // divisor)


def _row_cells(draw, x0, x1, ya, yb, n, off_frac, color_fn):
    """Paint one course of n cells exactly tiling x0..x1 (wrap-continuous).

    Cells sit on float edges of period (x1-x0)/n, shifted by ``off_frac``
    cells; a cell crossing the right edge is split and its remainder drawn
    at the left edge in the SAME color, so the strip tiles seamlessly in U.
    """
    w = x1 - x0
    cw = w / float(n)
    for i in range(n):
        e0 = x0 + ((i + off_frac) * cw) % w
        e1 = e0 + cw
        color = color_fn(i)
        if e1 <= x1 + 0.001:
            xa, xb = int(round(e0)), int(round(e1)) - 1
            if xb >= xa:
                draw.rectangle((xa, ya, xb, yb), fill=color)
        else:
            draw.rectangle((int(round(e0)), ya, x1 - 1, yb), fill=color)
            xb = int(round(e1 - w)) - 1
            if xb >= x0:
                draw.rectangle((x0, ya, xb, yb), fill=color)


def _cell_edges(x0, x1, n, off_frac):
    """X coordinates of the n cell edges used by _row_cells (wrapped)."""
    w = x1 - x0
    cw = w / float(n)
    return [x0 + int(round(((i + off_frac) * cw) % w)) for i in range(n)]


def _vgrad(draw, x0, y0, x1, y1, top, bottom):
    """Vertical gradient fill (glass sky reflections, awning fades)."""
    h = max(1, y1 - y0)
    for i, y in enumerate(range(y0, y1 + 1)):
        t = i / float(h)
        draw.line((x0, y, x1, y), fill=tuple(
            int(round(a + (b - a) * t)) for a, b in zip(top, bottom)))


def _pick_weighted(rng, choices, weights):
    total = float(sum(weights))
    roll = rng.uniform(0.0, total)
    acc = 0.0
    for choice, weight in zip(choices, weights):
        acc += weight
        if roll <= acc:
            return choice
    return choices[-1]


def _material_base(draw, rng, box, family, palette, shade_index,
                   pattern=None, px_scale=1.0, world_w=20.0):
    pattern = pattern or FLAVOR_PATTERNS["generic"]
    x0, y0, x1, y1 = box
    base = palette[family][shade_index]
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=base)
    h = y1 - y0
    if family == "siding":
        step = max(4, h // int(pattern["siding_step"]))
        for y in range(y0, y1, step):
            yb = min(y + step - 1, y1 - 1)
            # One clapboard course: per-board uniform color is realistic
            # for siding; keep deltas small so courses read subtly.
            roll = rng.random()
            board = _shade(rng, base, 12) if roll < 0.04 \
                else _shade(rng, base, 5)
            draw.rectangle((x0, y, x1 - 1, yb), fill=board)
            draw.line((x0, yb, x1, yb), fill=_adjust(base, -12), width=1)
    elif family == "brick":
        row_h = max(4, h // int(pattern["brick_row"]))
        n_bricks = max(8, int(round(world_w / 0.21)))  # ~21 cm stretchers
        mortar = palette.get("mortar") or _adjust(base, 26)
        for row, y in enumerate(range(y0, y1, row_h)):
            yb = min(y + row_h - 1, y1 - 1)
            off = 0.5 * (row % 2)
            draw.rectangle((x0, y, x1 - 1, yb), fill=base)
            shades = [
                _shade(rng, base, 8) if rng.random() < 0.35 else base
                for _ in range(n_bricks)
            ]
            _row_cells(draw, x0, x1, y, yb, n_bricks, off,
                       lambda i: shades[i])
            draw.line((x0, y, x1, y), fill=mortar, width=1)
            for ex in _cell_edges(x0, x1, n_bricks, off):
                draw.line((ex, y, ex, yb), fill=mortar, width=1)
    elif family == "stucco":
        # Large soft tonal blotches under the fine grain: render/whitewash.
        w = x1 - x0
        for _ in range(16):
            bw = rng.randint(max(8, w // 40), max(12, w // 12))
            bh = rng.randint(max(4, h // 6), max(6, h // 2))
            bx = rng.randint(x0 - bw, x1)
            by = rng.randint(y0, y1 - 1)
            draw.ellipse((bx, by, bx + bw, by + bh),
                         fill=_shade(rng, base, 5))
        _speckle(draw, rng, box, base, 10, _area_count(box, 48))
    elif family == "concrete":
        _speckle(draw, rng, box, base, 8, _area_count(box, 96))
        joint = _adjust(base, -16)
        n_panels = max(2, int(round(world_w / 3.0)))  # ~3 m panel joints
        for ex in _cell_edges(x0, x1, n_panels, 0.0):
            draw.line((ex, y0, ex, y1 - 1), fill=joint, width=1)
        for frac in (1.0 / 3.0, 2.0 / 3.0):  # faint horizontal tie lines
            ty = y0 + int(h * frac)
            draw.line((x0, ty, x1, ty), fill=_adjust(base, -6), width=1)


GLASS_KINDS = ("sky", "dark", "curtain", "warm")


def _draw_window(draw, rng, cx, sill_y, win_w, win_h, palette, pattern,
                 wall_base, px_scale, ppm_x, ppm_y):
    frame = palette.get("frame", (225, 222, 214))
    glass_base = palette.get("glass", (96, 112, 126))
    bw = _S(px_scale, 2)
    x0 = int(cx - win_w / 2)
    x1 = int(cx + win_w / 2)
    y1 = int(sill_y)
    y0 = int(sill_y - win_h)

    # Lintel hint over the frame; arch flavors get a light arc instead.
    if pattern.get("arch"):
        ah = max(3, int(0.22 * ppm_y))
        draw.arc((x0 - bw, y0 - bw - ah, x1 + bw, y0 - bw + ah),
                 180, 360, fill=_adjust(wall_base, +10), width=bw)
    else:
        draw.line((x0 - bw, y0 - bw - 1, x1 + bw, y0 - bw - 1),
                  fill=_adjust(wall_base, -12), width=1)

    # Frame, then glass.
    draw.rectangle((x0 - bw, y0 - bw, x1 + bw, y1 + bw), fill=frame)
    tone = rng.randint(-8, 8)
    kind = _pick_weighted(rng, GLASS_KINDS,
                          pattern.get("glass_mix", (3, 2, 1, 1)))
    if kind == "sky":
        _vgrad(draw, x0, y0, x1, y1,
               _adjust(glass_base, 20 + tone), _adjust(glass_base, -12 + tone))
    elif kind == "dark":
        draw.rectangle((x0, y0, x1, y1), fill=_adjust((58, 66, 72), tone))
    elif kind == "curtain":
        draw.rectangle((x0, y0, x1, y1), fill=_adjust((168, 162, 150), tone))
        gap = x0 + rng.randint((x1 - x0) // 4, 3 * (x1 - x0) // 4)
        draw.line((gap, y0, gap, y1), fill=_adjust((58, 66, 72), tone),
                  width=bw)
    else:  # warm interior light
        draw.rectangle((x0, y0, x1, y1), fill=_adjust((150, 126, 92), tone))

    # Mullions (frame color) per regional style.
    mullion = pattern.get("mullion", "single")
    mx = (x0 + x1) // 2
    my = y0 + int((y1 - y0) * 0.45)
    if mullion == "single":
        draw.line((mx, y0, mx, y1), fill=frame, width=bw)
    elif mullion == "cross":
        draw.line((mx, y0, mx, y1), fill=frame, width=bw)
        draw.line((x0, my, x1, my), fill=frame, width=bw)
    elif mullion == "sash":
        draw.line((x0, my, x1, my), fill=frame, width=bw)
    elif mullion == "slider":
        for frac in (1.0 / 3.0, 2.0 / 3.0):
            sx = x0 + int((x1 - x0) * frac)
            draw.line((sx, y0, sx, y1), fill=frame, width=bw)

    # Security bars (painted over the glass).
    if rng.random() < float(pattern.get("bars_prob", 0.0)):
        bar = (40, 42, 44)
        for frac in (0.25, 0.5, 0.75):
            sx = x0 + int((x1 - x0) * frac)
            draw.line((sx, y0, sx, y1), fill=bar, width=1)
        draw.line((x0, my, x1, my), fill=bar, width=1)

    # Sill: darker ledge with a light top edge, extending past the jambs.
    ext = _S(px_scale, 3)
    sill_h = _S(px_scale, 2)
    draw.rectangle((x0 - bw - ext, y1 + bw + 1, x1 + bw + ext,
                    y1 + bw + sill_h), fill=_adjust(frame, -30))
    draw.line((x0 - bw - ext, y1 + bw + 1, x1 + bw + ext, y1 + bw + 1),
              fill=_adjust(frame, 15), width=1)

    # Shutters.
    if pattern.get("shutters"):
        sw = max(2, int(win_w * 0.35))
        pool = palette.get("shutter_pool") or (palette["trim"],)
        scolor = _shade(rng, rng.choice(pool), 8)
        for sx0, sx1 in ((x0 - bw - sw, x0 - bw - 1),
                         (x1 + bw + 1, x1 + bw + sw)):
            draw.rectangle((sx0, y0, sx1, y1), fill=scolor)
            if pattern.get("louvered"):
                for ly in range(y0 + 2, y1 - 1, _S(px_scale, 4)):
                    draw.line((sx0 + 1, ly, sx1 - 1, ly),
                              fill=_adjust(scolor, -10), width=1)

    # AC unit hung under the window's right half.
    if rng.random() < float(pattern.get("ac_prob", 0.0)):
        acw = int(0.55 * ppm_x)
        ach = int(0.4 * ppm_y)
        ax1 = x1 + bw
        ax0 = ax1 - acw
        ay0 = y1 + bw + sill_h + 1
        draw.rectangle((ax0, ay0, ax1, ay0 + ach), fill=(180, 182, 184))
        draw.rectangle((ax0, ay0, ax1, ay0 + ach), outline=(120, 122, 124))
        for frac in (0.4, 0.7):
            gy = ay0 + int(ach * frac)
            draw.line((ax0 + 1, gy, ax1 - 1, gy), fill=(150, 152, 154),
                      width=1)


def _paint_door(draw, rng, cx, base_y, y_gutter_end, palette, pattern,
                glass_base, px_scale, ppm_x, ppm_y):
    frame = palette.get("frame", (225, 222, 214))
    door_w = 1.0 * ppm_x
    door_h = 2.15 * ppm_y
    bw = _S(px_scale, 2)
    dx0 = int(cx - door_w / 2)
    dx1 = int(cx + door_w / 2)
    dy1 = base_y
    dy0 = int(dy1 - door_h)
    door = _shade(rng, rng.choice(palette.get("door_pool",
                                              ((84, 62, 48),))), 10)
    draw.rectangle((dx0 - bw, dy0 - bw, dx1 + bw, dy1), fill=frame)
    # Fill through the bottom gutter so mip bleed matches the door.
    draw.rectangle((dx0, dy0, dx1, y_gutter_end), fill=door)
    panels = pattern.get("door_panels", "panel2x2")
    if panels == "panel2x2":
        mx = max(2, int((dx1 - dx0) * 0.18))
        mid_x = (dx0 + dx1) // 2
        mid_y = (dy0 + dy1) // 2
        for px0, px1 in ((dx0 + mx, mid_x - mx // 2),
                         (mid_x + mx // 2, dx1 - mx)):
            for py0, py1 in ((dy0 + mx, mid_y - mx // 2),
                             (mid_y + mx // 2, dy1 - mx)):
                if px1 > px0 and py1 > py0:
                    draw.rectangle((px0, py0, px1, py1),
                                   fill=_adjust(door, -14))
    elif panels == "planks":
        for frac in (0.25, 0.5, 0.75):
            px = dx0 + int((dx1 - dx0) * frac)
            draw.line((px, dy0 + 1, px, dy1 - 1), fill=_adjust(door, -16),
                      width=1)
    draw.line((dx0 - bw, dy1, dx1 + bw, dy1),
              fill=_adjust(door, -20), width=1)  # threshold
    if pattern.get("transom"):
        ty1 = dy0 - bw - 1
        ty0 = ty1 - int(0.35 * ppm_y)
        draw.rectangle((dx0 - bw, ty0 - bw, dx1 + bw, ty1 + bw), fill=frame)
        _vgrad(draw, dx0, ty0, dx1, ty1,
               _adjust(glass_base, 18), _adjust(glass_base, -8))
    return dx0, dx1


def _paint_wall(draw, rng, box, family, palette, shade_index, world_w,
                world_h, ground=False, pattern=None, px_scale=1.0):
    pattern = pattern or FLAVOR_PATTERNS["generic"]
    x0, y0, x1, y1 = box
    _material_base(draw, rng, box, family, palette, shade_index, pattern,
                   px_scale, world_w)
    gutter = max(1, int(round(GUTTER_PX * px_scale)))
    px_per_m_x = (x1 - x0) / world_w
    # The OBJ maps V frac 0..1 onto the gutter-inset band, so vertical
    # meters are measured against the inset height and features anchor to
    # the visible floor line at y1 - gutter.
    px_per_m_y = (y1 - y0 - 2 * gutter) / world_h
    base_y = y1 - gutter
    glass_base = palette.get("glass", (96, 112, 126))

    # Grime / floor-slab shadow at the bottom edge (before openings, so
    # doors and windows paint over it).  Deterministic gradient: full-width
    # lines must never roll the RNG.
    grime = float(pattern.get("grime", 0.0))
    if grime > 0 and family != "siding":
        gh = max(2, int(round(0.18 * px_per_m_y)))
        for k in range(gh):
            frac = (k + 1) / float(gh)
            yy = base_y - gh + k
            draw.line((x0, yy, x1, yy),
                      fill=_adjust(palette[family][shade_index],
                                   -int(round(16 * grime * frac))), width=1)
        draw.rectangle((x0, base_y, x1 - 1, y1 - 1),
                       fill=_adjust(palette[family][shade_index],
                                    -int(round(16 * grime))))

    n_windows = max(2, int(round(world_w / float(pattern["period"]))))
    door_slot = rng.randrange(n_windows) if ground else -1
    sill_y = base_y - int(0.9 * px_per_m_y)
    win_w = float(pattern["window_w"]) * px_per_m_x
    win_h = float(pattern["window_h"]) * px_per_m_y
    base = palette[family][shade_index]
    sill_xs = []
    for i in range(n_windows):
        cx = x0 + (i + 0.5) * (x1 - x0) / n_windows
        if i == door_slot:
            _paint_door(draw, rng, cx, base_y, y1 - 1, palette, pattern,
                        glass_base, px_scale, px_per_m_x, px_per_m_y)
        else:
            _draw_window(draw, rng, cx, sill_y, win_w, win_h, palette,
                         pattern, base, px_scale, px_per_m_x, px_per_m_y)
            sill_xs.append((int(cx - win_w / 2), int(cx + win_w / 2)))

    # Weathering streaks running down from sill corners.
    streaks = float(pattern.get("streaks", 0.0))
    if streaks > 0 and family in ("concrete", "stucco"):
        for wx0, wx1 in sill_xs:
            for sx in (wx0, wx1):
                if rng.random() < streaks:
                    length = int(rng.uniform(0.5, 1.2) * px_per_m_y)
                    draw.line((sx, sill_y + _S(px_scale, 4), sx,
                               min(base_y, sill_y + length)),
                              fill=_adjust(base, -8), width=1)

    if pattern["rail"] and not ground:
        # Balcony railing band under the window row: top rail + pickets.
        rail_color = _adjust(palette["trim"], 20)
        rail_y = int(sill_y + 0.15 * px_per_m_y)
        picket_bot = min(base_y - 1, rail_y + int(0.5 * px_per_m_y))
        n_pickets = max(8, int(round(world_w / 0.18)))
        for ex in _cell_edges(x0, x1, n_pickets, 0.0):
            draw.line((ex, rail_y, ex, picket_bot), fill=rail_color, width=1)
        draw.line((x0, rail_y, x1, rail_y), fill=rail_color,
                  width=_S(px_scale, 3))


def _paint_storefront(draw, rng, box, palette, world_w, world_h,
                      pattern=None, px_scale=1.0):
    pattern = pattern or FLAVOR_PATTERNS["generic"]
    x0, y0, x1, y1 = box
    _material_base(draw, rng, box, "concrete", palette, 0, pattern,
                   px_scale, world_w)
    gutter = max(1, int(round(GUTTER_PX * px_scale)))
    ppm_x = (x1 - x0) / world_w
    ppm_y = (y1 - y0 - 2 * gutter) / world_h
    base = palette["concrete"][0]
    base_y = y1 - gutter
    glass_base = palette.get("glass", (96, 112, 126))
    frame = (50, 54, 58)  # dark storefront aluminum

    n_bays = 4  # 5 m bays over the 20 m repeat (wrap-clean)
    sign_top = y0 + gutter
    sign_bot = sign_top + int(0.7 * ppm_y)
    awning_bot = sign_bot + int(0.3 * ppm_y)
    bulkhead_top = base_y - int(0.35 * ppm_y)
    bay_edges = _cell_edges(x0, x1, n_bays, 0.0) + [x1]
    door_bay = rng.randrange(n_bays)
    for b in range(n_bays):
        bx0, bx1 = bay_edges[b], bay_edges[b + 1]
        # Sign band with low-contrast "lettering" dashes.
        sign = _shade(rng, rng.choice(palette.get("sign_pool",
                                                  ((120, 110, 100),))), 8)
        draw.rectangle((bx0, sign_top, bx1 - 1, sign_bot), fill=sign)
        letter = _adjust(sign, 28)
        ly = (sign_top + sign_bot) // 2
        lx = bx0 + rng.randint(2, max(3, (bx1 - bx0) // 4))
        while lx < bx1 - 4:
            seg = rng.randint(_S(px_scale, 8), _S(px_scale, 26))
            draw.line((lx, ly, min(lx + seg, bx1 - 3), ly), fill=letter,
                      width=_S(px_scale, 3))
            lx += seg + rng.randint(_S(px_scale, 6), _S(px_scale, 14))
        # Awning on ~60% of bays.
        glaze_top = sign_bot + 2
        if rng.random() < 0.6:
            awn = _shade(rng, rng.choice(palette.get("awning_pool",
                                                     ((120, 110, 100),))), 8)
            draw.rectangle((bx0, sign_bot + 1, bx1 - 1, awning_bot),
                           fill=awn)
            if pattern.get("awning_stripes"):
                stripe_n = max(4, int(round((bx1 - bx0) /
                                            (0.4 * ppm_x))))
                for ex in _cell_edges(bx0, bx1, stripe_n, 0.0)[::2]:
                    ew = max(1, int(0.2 * ppm_x))
                    draw.rectangle((ex, sign_bot + 1,
                                    min(ex + ew, bx1 - 1), awning_bot),
                                   fill=_adjust(awn, 25))
            draw.line((bx0, awning_bot, bx1, awning_bot),
                      fill=_adjust(awn, -20), width=1)
            glaze_top = awning_bot + 1
        # Glazing panels (~1 m) with gradient glass; one bay gets a door.
        n_panels = max(3, int(round((bx1 - bx0) / ppm_x)))
        panel_edges = _cell_edges(bx0, bx1, n_panels, 0.0) + [bx1]
        door_panel = rng.randrange(n_panels) if b == door_bay else -1
        for p in range(n_panels):
            px0, px1 = panel_edges[p] + 1, panel_edges[p + 1] - 1
            if px1 <= px0:
                continue
            if p == door_panel:
                draw.rectangle((px0, glaze_top, px1, base_y - 1),
                               fill=_adjust(glass_base, -22))
                bar_y = glaze_top + int((base_y - glaze_top) * 0.55)
                draw.line((px0 + 1, bar_y, px1 - 1, bar_y),
                          fill=(170, 172, 174), width=_S(px_scale, 2))
            else:
                _vgrad(draw, px0, glaze_top, px1, bulkhead_top - 1,
                       _jitter(rng, _adjust(glass_base, 14), 8),
                       _adjust(glass_base, -16))
                # Bulkhead under the display glass.
                draw.rectangle((px0, bulkhead_top, px1, base_y - 1),
                               fill=_adjust(base, -18))
            draw.line((panel_edges[p], glaze_top, panel_edges[p], base_y - 1),
                      fill=frame, width=_S(px_scale, 2))
        # Pilaster between bays.
        draw.rectangle((bx0, sign_top, bx0 + _S(px_scale, 3), base_y - 1),
                       fill=_adjust(base, -18))


def _paint_roller(draw, rng, box, palette, world_w, world_h,
                  pattern=None, px_scale=1.0):
    x0, y0, x1, y1 = box
    _material_base(draw, rng, box, "concrete", palette, 1,
                   pattern, px_scale, world_w)
    gutter = max(1, int(round(GUTTER_PX * px_scale)))
    ppm_x = (x1 - x0) / world_w
    ppm_y = (y1 - y0 - 2 * gutter) / world_h
    base = palette["concrete"][1]
    base_y = y1 - gutter
    door_pool = ((150, 152, 154), (166, 160, 148),
                 (120, 134, 146), (128, 138, 128))
    n_doors = 6  # 4 m cells over the 24 m repeat (wrap-clean)
    door_w = int(3.4 * ppm_x)
    door_top = y0 + int((y1 - y0) * 0.15)
    slat_h = max(3, int(0.18 * ppm_y))
    cell_edges = _cell_edges(x0, x1, n_doors, 0.0)
    # Lintel beam running over the doors.
    draw.line((x0, door_top - 2, x1, door_top - 2),
              fill=_adjust(base, -20), width=_S(px_scale, 3))
    for ex in cell_edges:
        gap = int(0.3 * ppm_x)
        dx0 = ex + gap
        dx1 = dx0 + door_w
        door = _shade(rng, rng.choice(door_pool), 6)
        draw.rectangle((dx0, door_top, dx1, y1 - 1), fill=door)
        for y in range(door_top + slat_h, base_y, slat_h):
            draw.line((dx0, y, dx1, y), fill=_adjust(door, -12), width=1)
            draw.line((dx0, y + 1, dx1, y + 1), fill=_adjust(door, 10),
                      width=1)
        # Oil/tyre grime on the bottom slats.
        draw.rectangle((dx0, base_y - max(2, int(0.12 * ppm_y)), dx1, y1 - 1),
                       fill=_adjust(door, -16))
        for px in (dx0 - 1, dx1 + 1):  # frame posts
            draw.line((px, door_top - 1, px, base_y), fill=_adjust(base, -24),
                      width=_S(px_scale, 3))


def _paint_roof_shingle(draw, rng, box, base, pattern):
    x0, y0, x1, y1 = box
    row_h = max(3, int(pattern["shingle_row"]))
    n_tabs = 48  # 0.25 m tabs over the 12 m repeat

    def tab_color(_i):
        if rng.random() < 0.06:
            return _shade(rng, base, 14)  # rare odd tab
        return _shade(rng, base, 6)

    for row, y in enumerate(range(y0, y1, row_h)):
        yb = min(y + row_h - 1, y1 - 1)
        _row_cells(draw, x0, x1, y, yb, n_tabs, 0.5 * (row % 2), tab_color)
        draw.line((x0, y, x1, y), fill=_adjust(base, -10), width=1)
    _speckle(draw, rng, box, base, 8, _area_count(box, 600))


def _paint_roof_tile(draw, rng, box, base, pattern):
    x0, y0, x1, y1 = box
    row_h = max(3, int(pattern["tile_row"]))
    n_pans = 64  # ~0.19 m pan width over the 12 m repeat
    for row, y in enumerate(range(y0, y1, row_h)):
        yb = min(y + row_h - 1, y1 - 1)
        off = 0.5 * (row % 2)
        pan_shades = [_shade(rng, base, 7) for _ in range(n_pans)]
        _row_cells(draw, x0, x1, y, yb, n_pans, off, lambda i: pan_shades[i])
        edges = _cell_edges(x0, x1, n_pans, off)
        for i, ex in enumerate(edges):  # pan-gap shadow
            draw.line((ex, y, ex, yb), fill=_adjust(pan_shades[i], -14),
                      width=1)
        if pattern["tile_gloss"]:  # glazed crest highlight at pan centers
            for i, ex in enumerate(_cell_edges(x0, x1, n_pans, off + 0.5)):
                draw.line((ex, y + 1, ex, yb), fill=_adjust(pan_shades[i], 16),
                          width=1)
        draw.line((x0, y, x1, y), fill=_adjust(base, -12), width=1)


def _paint_roof_metal(draw, rng, box, base, pattern, px_scale):
    x0, y0, x1, y1 = box
    n_sheets = 13  # ~0.92 m sheets over the 12 m repeat
    sheet_shades = [_shade(rng, base, 4) for _ in range(n_sheets)]
    _row_cells(draw, x0, x1, y0, y1 - 1, n_sheets, 0.0,
               lambda i: sheet_shades[i])
    corr = max(2, int(pattern["corrugation"]))
    n_corr = max(8, int(round((x1 - x0) / float(corr))))  # snap to wrap
    for ex in _cell_edges(x0, x1, n_corr, 0.0):
        draw.line((ex, y0, ex, y1 - 1), fill=_adjust(base, -10), width=1)
        draw.line((ex + 1, y0, ex + 1, y1 - 1), fill=_adjust(base, 6),
                  width=1)
    for ex in _cell_edges(x0, x1, n_sheets, 0.0):  # sheet seams
        draw.line((ex, y0, ex, y1 - 1), fill=_adjust(base, -14), width=1)
    if pattern["rust"]:
        # Weathered rust blotches, spread evenly (LOD far quads sample a
        # small window of the strip -- keep the mean representative).
        for _ in range(int(round(28 * px_scale * px_scale))):
            bw = rng.randint(_S(px_scale, 8), _S(px_scale, 24))
            bh = rng.randint(_S(px_scale, 3), _S(px_scale, 8))
            bx = rng.randint(x0, max(x0, x1 - bw - 1))
            by = rng.randint(y0, max(y0, y1 - bh - 1))
            rust = _shade(rng, (120, 74, 48), 18)
            draw.ellipse((bx, by, bx + bw, by + bh), fill=rust)
            # Gravity streak toward the eave (image-down = strip bottom).
            sx = bx + bw // 2
            draw.line((sx, by + bh, sx, min(y1 - 1, by + bh + _S(px_scale, 14))),
                      fill=_adjust(base, -8), width=1)


def _paint_roof_flat(draw, rng, box, base, px_scale):
    x0, y0, x1, y1 = box
    n_sheets = 8  # 2 m membrane sheets over the 16 m repeat
    sheet_shades = [_shade(rng, base, 4) for _ in range(n_sheets)]
    _row_cells(draw, x0, x1, y0, y1 - 1, n_sheets, 0.0,
               lambda i: sheet_shades[i])
    for ex in _cell_edges(x0, x1, n_sheets, 0.0):
        draw.line((ex, y0, ex, y1 - 1), fill=_adjust(base, -10), width=1)
    _speckle(draw, rng, box, base, 12, _area_count(box, 60))
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=_adjust(base, -12),
                   width=_S(px_scale, 3))


def _paint_roof(draw, rng, box, kind, palette, pattern=None, px_scale=1.0):
    """Roof strips.  Banding rule: pitched roofs magnify each strip row to
    ~40 cm across the whole roof, so randomness is per CELL via _row_cells
    and every full-width/full-height line is a deterministic _adjust()."""
    pattern = pattern or FLAVOR_PATTERNS["generic"]
    x0, y0, x1, y1 = box
    base = palette[kind]
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=base)
    if kind == "roof_shingle":
        _paint_roof_shingle(draw, rng, box, base, pattern)
    elif kind == "roof_tile":
        _paint_roof_tile(draw, rng, box, base, pattern)
    elif kind == "roof_metal":
        _paint_roof_metal(draw, rng, box, base, pattern, px_scale)
    else:  # roof_flat: membrane sheets + gravel grain
        _paint_roof_flat(draw, rng, box, base, px_scale)


def build_layout(size: int) -> dict:
    """Compute strip pixel rows and the V bands archetypes will use.

    STRIPS heights are authored for a 2048px atlas and scale proportionally
    for other sizes (e.g. 1024 halves every band), so the V bands -- and
    therefore the archetypes' UVs -- are identical at any resolution.
    """
    scale = size / 2048.0
    gutter = max(1, int(round(GUTTER_PX * scale)))
    strips = {}
    y = 0
    for name, height_px, world_w, world_h, kind in STRIPS:
        scaled_h = max(8, int(round(height_px * scale)))
        y0, y1 = y, y + scaled_h
        if y1 > size:
            raise SystemExit(
                f"scaled strip heights exceed atlas size {size}px at {name}"
            )
        strips[name] = {
            "px": [y0, y1],
            "kind": kind,
            "v0": round(1.0 - (y1 - gutter) / size, 6),
            "v1": round(1.0 - (y0 + gutter) / size, 6),
            "world_w_m": world_w,
            "world_h_m": world_h,
        }
        y = y1
    return {"size": size, "window_period_m": WINDOW_PERIOD_M, "strips": strips}


def paint_atlas(flavor: str, layout: dict, seed: int,
                references: dict | None = None) -> Image.Image:
    palette = PALETTES[flavor]
    references = references or _load_reference_styles()
    pattern = _pattern_for_flavor(flavor, references)
    size = layout["size"]
    # Roof row/pitch constants are authored in 2048px units; rescale so the
    # painted feature size in world metres is resolution-independent.
    px_scale = size / 2048.0
    for key in ("tile_row", "shingle_row", "corrugation"):
        pattern[key] = max(2, int(round(pattern[key] * px_scale)))
    img = Image.new("RGB", (size, size), (96, 96, 96))
    draw = ImageDraw.Draw(img)
    for index, (name, _h, world_w, world_h, kind) in enumerate(STRIPS):
        flavor_salt = int(hashlib.sha1(flavor.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed * 7919 + index * 104729 + flavor_salt)
        y0, y1 = layout["strips"][name]["px"]
        # Paint the FULL band including the gutter so bleed shows the same
        # material, then UVs stay inside the inset V range.
        strip_box = (0, y0, size, y1)
        if kind == "wall":
            family, shade = name.split("_")[1:3]
            _paint_wall(draw, rng, strip_box, family, palette,
                        0 if shade == "a" else 1, world_w, world_h,
                        pattern=pattern, px_scale=px_scale)
        elif kind == "ground":
            family = name.split("_")[1]
            _paint_wall(draw, rng, strip_box, family, palette, 0,
                        world_w, world_h, ground=True, pattern=pattern,
                        px_scale=px_scale)
        elif kind == "storefront":
            _paint_storefront(draw, rng, strip_box, palette, world_w,
                              world_h, pattern=pattern, px_scale=px_scale)
        elif kind == "roller":
            _paint_roller(draw, rng, strip_box, palette, world_w, world_h,
                          pattern=pattern, px_scale=px_scale)
        elif kind == "plain":
            family = name.split("_")[1]
            _material_base(draw, rng, strip_box, family, palette, 0,
                           pattern, px_scale, world_w)
        elif kind == "trim":
            draw.rectangle((0, y0, size - 1, y1 - 1), fill=palette["trim"])
            _speckle(draw, rng, strip_box, palette["trim"], 8,
                     _area_count(strip_box, 64))
        elif kind == "roof":
            _paint_roof(draw, rng, strip_box, name, palette, pattern,
                        px_scale)
    return img


def texture_name(flavor: str) -> str:
    return f"o4sfr_procgen_atlas_{flavor}.png"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    parser.add_argument("--output", required=True,
                        help="Library package folder; PNGs land in textures/.")
    parser.add_argument("--layout-out",
                        default=os.path.join(HERE, "atlas_layout.json"))
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    references = _load_reference_styles()
    atlas_cfg = config["atlas"]
    layout = build_layout(int(atlas_cfg["size"]))
    layout["flavors"] = {
        flavor: f"textures/{texture_name(flavor)}"
        for flavor in atlas_cfg["flavors"]
    }
    layout["reference_styles"] = {
        "version": references.get("version"),
        "regions": sorted((references.get("regions") or {}).keys()),
        "class_profiles": sorted((references.get("class_profiles") or {}).keys()),
    }
    with open(args.layout_out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(layout, fh, indent=1, sort_keys=True)
        fh.write("\n")

    textures_dir = os.path.join(args.output, "textures")
    os.makedirs(textures_dir, exist_ok=True)
    # NOTE: PNG only. atlas_dds.py (strip-clamped mips) exists as an
    # experiment but X-Plane's DDS loader wants DXT compression; the
    # uncompressed variant is unvalidated in-sim. The rainbow-roof artifact
    # turned out to be hue-jittered roof rows in the PNG itself (_shade vs
    # _jitter), not mip bleed.
    for flavor in atlas_cfg["flavors"]:
        img = paint_atlas(flavor, layout, int(atlas_cfg["seed"]), references)
        path = os.path.join(textures_dir, texture_name(flavor))
        img.save(path, optimize=True)
        print(f"wrote {path}")
    print(f"wrote {args.layout_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
