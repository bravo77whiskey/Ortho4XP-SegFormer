"""Generate the shared procedural facade/roof texture atlases (PIL).

One 2048x2048 PNG per region flavor; every generated OBJ references exactly
one atlas, so the whole library costs three textures.  The atlas is a stack
of full-width horizontal strips: U tiles freely (GL wrap) while V stays
inside a strip's band, so a wall of any length is a single quad whose
windows never stretch.

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
# UVs stay identical because the layout is shared.
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
    },
}

WINDOW_PERIOD_M = 2.5

# Per-flavor PATTERN parameters (paint-side only: the strip layout and the
# UV contract are shared, so geometry never changes). Windows in metres;
# row/pitch values in pixels on the 2048px strip.
FLAVOR_PATTERNS = {
    "generic": {
        "window_w": 1.25, "window_h": 1.45, "period": 2.5, "shutters": False,
        "rail": False, "brick_row": 20, "siding_step": 16,
        "tile_row": 6, "shingle_row": 5, "corrugation": 14, "rust": False,
        "tile_gloss": False,
    },
    "europe": {
        "window_w": 1.1, "window_h": 1.7, "period": 2.4, "shutters": False,
        "rail": False, "brick_row": 22, "siding_step": 14,
        "tile_row": 6, "shingle_row": 5, "corrugation": 14, "rust": False,
        "tile_gloss": False,
    },
    "north_america": {
        "window_w": 1.4, "window_h": 1.5, "period": 2.6, "shutters": True,
        "rail": False, "brick_row": 18, "siding_step": 18,
        "tile_row": 6, "shingle_row": 4, "corrugation": 16, "rust": False,
        "tile_gloss": False,
    },
    "mediterranean": {
        "window_w": 1.0, "window_h": 1.5, "period": 2.7, "shutters": True,
        "rail": False, "brick_row": 20, "siding_step": 16,
        "tile_row": 9, "shingle_row": 5, "corrugation": 14, "rust": False,
        "tile_gloss": False,
    },
    "asia": {
        "window_w": 1.9, "window_h": 1.25, "period": 2.8, "shutters": False,
        "rail": True, "brick_row": 18, "siding_step": 14,
        "tile_row": 5, "shingle_row": 5, "corrugation": 10, "rust": False,
        "tile_gloss": True,
    },
    "africa": {
        "window_w": 1.0, "window_h": 1.1, "period": 3.2, "shutters": False,
        "rail": False, "brick_row": 16, "siding_step": 16,
        "tile_row": 7, "shingle_row": 5, "corrugation": 10, "rust": True,
        "tile_gloss": False,
    },
    "south_america": {
        "window_w": 1.15, "window_h": 1.35, "period": 2.8, "shutters": False,
        "rail": True, "brick_row": 16, "siding_step": 16,
        "tile_row": 8, "shingle_row": 5, "corrugation": 12, "rust": True,
        "tile_gloss": False,
    },
    "australia_oceania": {
        "window_w": 1.5, "window_h": 1.3, "period": 2.7, "shutters": False,
        "rail": False, "brick_row": 19, "siding_step": 18,
        "tile_row": 7, "shingle_row": 5, "corrugation": 18, "rust": False,
        "tile_gloss": False,
    },
}


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


def _speckle(draw, rng, box, base, amount, count):
    x0, y0, x1, y1 = box
    for _ in range(count):
        x = rng.randint(x0, x1 - 1)
        y = rng.randint(y0, y1 - 1)
        draw.point((x, y), fill=_shade(rng, base, amount))


def _material_base(draw, rng, box, family, palette, shade_index,
                   pattern=None):
    pattern = pattern or FLAVOR_PATTERNS["generic"]
    x0, y0, x1, y1 = box
    base = palette[family][shade_index]
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=base)
    h = y1 - y0
    if family == "siding":
        step = max(4, h // int(pattern["siding_step"]))
        for y in range(y0 + step, y1, step):
            draw.line((x0, y, x1, y), fill=_shade(rng, base, 18), width=1)
    elif family == "brick":
        row_h = max(4, h // int(pattern["brick_row"]))
        brick_w = 22
        mortar = _shade(rng, base, 30)
        for row, y in enumerate(range(y0, y1, row_h)):
            draw.line((x0, y, x1, y), fill=mortar, width=1)
            offset = (row % 2) * (brick_w // 2)
            for x in range(x0 - offset, x1, brick_w):
                draw.line((x, y, x, min(y + row_h, y1)), fill=mortar, width=1)
        _speckle(draw, rng, box, base, 12, (x1 - x0) // 2)
    elif family == "stucco":
        _speckle(draw, rng, box, base, 10, (x1 - x0) * 2)
    elif family == "concrete":
        joint = _shade(rng, base, 26)
        _speckle(draw, rng, box, base, 8, (x1 - x0))
        panel_px = max(8, int((x1 - x0) * 3.0 / 20.0))  # ~3 m panel joints
        for x in range(x0 + panel_px, x1, panel_px):
            draw.line((x, y0, x, y1 - 1), fill=joint, width=1)


def _draw_window(draw, rng, cx, sill_y, win_w, win_h, trim_color,
                 shutters=False, shutter_color=None):
    glass = _jitter(rng, (96, 112, 126), 16)
    x0 = int(cx - win_w / 2)
    x1 = int(cx + win_w / 2)
    y1 = int(sill_y)
    y0 = int(sill_y - win_h)
    draw.rectangle((x0 - 1, y0 - 1, x1 + 1, y1 + 1), fill=trim_color)
    draw.rectangle((x0, y0, x1, y1), fill=glass)
    # center mullion
    draw.line(((x0 + x1) // 2, y0, (x0 + x1) // 2, y1), fill=trim_color)
    if shutters:
        sw = max(2, int(win_w * 0.35))
        color = shutter_color or trim_color
        draw.rectangle((x0 - sw - 1, y0, x0 - 2, y1),
                       fill=_jitter(rng, color, 14))
        draw.rectangle((x1 + 2, y0, x1 + sw + 1, y1),
                       fill=_jitter(rng, color, 14))


def _paint_wall(draw, rng, box, family, palette, shade_index, world_w, world_h,
                ground=False, pattern=None):
    pattern = pattern or FLAVOR_PATTERNS["generic"]
    x0, y0, x1, y1 = box
    _material_base(draw, rng, box, family, palette, shade_index, pattern)
    px_per_m_x = (x1 - x0) / world_w
    px_per_m_y = (y1 - y0) / world_h
    trim_color = palette["trim"]
    n_windows = max(2, int(round(world_w / float(pattern["period"]))))
    door_slot = rng.randrange(n_windows) if ground else -1
    sill_y = y1 - int(0.9 * px_per_m_y)
    win_w = float(pattern["window_w"]) * px_per_m_x
    win_h = float(pattern["window_h"]) * px_per_m_y
    for i in range(n_windows):
        cx = x0 + (i + 0.5) * (x1 - x0) / n_windows
        if i == door_slot:
            door_w = 1.0 * px_per_m_x
            door_h = 2.15 * px_per_m_y
            dx0 = int(cx - door_w / 2)
            dy1 = y1 - 1
            draw.rectangle(
                (dx0 - 1, int(dy1 - door_h) - 1, int(dx0 + door_w) + 1, dy1),
                fill=trim_color,
            )
            draw.rectangle(
                (dx0, int(dy1 - door_h), int(dx0 + door_w), dy1),
                fill=_jitter(rng, (84, 62, 48), 18),
            )
        else:
            _draw_window(draw, rng, cx, sill_y, win_w, win_h, trim_color,
                         shutters=bool(pattern["shutters"]))
    if pattern["rail"] and not ground:
        # Continuous balcony/railing line under the window row.
        rail_y = int(sill_y + 0.15 * px_per_m_y)
        draw.line((x0, rail_y, x1, rail_y),
                  fill=_jitter(rng, trim_color, 20), width=2)


def _paint_storefront(draw, rng, box, palette, world_w, world_h):
    x0, y0, x1, y1 = box
    _material_base(draw, rng, box, "concrete", palette, 0)
    px_per_m_x = (x1 - x0) / world_w
    px_per_m_y = (y1 - y0) / world_h
    trim_color = palette["trim"]
    glaze_top = y0 + int(0.55 * px_per_m_y)
    glaze_bottom = y1 - int(0.25 * px_per_m_y)
    draw.rectangle((x0, glaze_top, x1 - 1, glaze_bottom), fill=trim_color)
    panel_w = int(2.0 * px_per_m_x)
    for x in range(x0 + 2, x1 - 2, panel_w):
        draw.rectangle(
            (x + 2, glaze_top + 2, min(x + panel_w - 2, x1 - 2), glaze_bottom - 2),
            fill=_jitter(rng, (88, 104, 118), 14),
        )
    # awning band
    draw.rectangle((x0, y0 + 2, x1 - 1, glaze_top - 2),
                   fill=_jitter(rng, (120, 110, 100), 10))


def _paint_roller(draw, rng, box, palette, world_w, world_h):
    x0, y0, x1, y1 = box
    _material_base(draw, rng, box, "concrete", palette, 1)
    px_per_m_x = (x1 - x0) / world_w
    door_w = int(3.4 * px_per_m_x)
    gap_w = int(0.6 * px_per_m_x)
    door_top = y0 + int((y1 - y0) * 0.12)
    for x in range(x0 + gap_w, x1 - door_w, door_w + gap_w):
        door = _jitter(rng, (148, 150, 152), 12)
        draw.rectangle((x, door_top, x + door_w, y1 - 2), fill=door)
        slat = _jitter(rng, door, 18)
        for y in range(door_top + 3, y1 - 2, 5):
            draw.line((x, y, x + door_w, y), fill=slat, width=1)


def _paint_roof(draw, rng, box, kind, palette, pattern=None):
    pattern = pattern or FLAVOR_PATTERNS["generic"]
    x0, y0, x1, y1 = box
    base = palette[kind]
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=base)
    if kind == "roof_shingle":
        row = max(3, int(pattern["shingle_row"]))
        for y in range(y0, y1, row):
            draw.line((x0, y, x1, y), fill=_shade(rng, base, 14), width=2)
        _speckle(draw, rng, box, base, 12, (x1 - x0) * 2)
    elif kind == "roof_tile":
        row_h = max(3, int(pattern["tile_row"]))
        tile_w = row_h * 2
        for row, y in enumerate(range(y0, y1, row_h)):
            shade = _shade(rng, base, 12)
            draw.line((x0, y, x1, y), fill=_shade(rng, base, 26), width=1)
            offset = (row % 2) * (tile_w // 2)
            for x in range(x0 - offset, x1, tile_w):
                draw.line((x, y, x, min(y + row_h, y1 - 1)),
                          fill=shade, width=1)
            if pattern["tile_gloss"]:
                # Glazed-tile highlight along each row crest.
                draw.line((x0, y + 1, x1, y + 1),
                          fill=_shade(rng, base, 40), width=1)
    elif kind == "roof_metal":
        seam = _shade(rng, base, 24)
        for x in range(x0, x1, int(pattern["corrugation"])):
            draw.line((x, y0, x, y1 - 1), fill=seam, width=1)
        _speckle(draw, rng, box, base, 6, x1 - x0)
        if pattern["rust"]:
            # Weathered rust blotches on corrugated sheets.
            for _ in range(28):
                bx = rng.randint(x0, x1 - 24)
                by = rng.randint(y0, y1 - 8)
                bw = rng.randint(8, 24)
                bh = rng.randint(3, 8)
                rust = _shade(rng, (120, 74, 48), 22)
                draw.ellipse((bx, by, bx + bw, by + bh), fill=rust)
    else:  # roof_flat: membrane/gravel with gentle vignette
        _speckle(draw, rng, box, base, 14, (x1 - x0) * 4)
        edge = _shade(rng, base, 20)
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=edge, width=3)


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


def paint_atlas(flavor: str, layout: dict, seed: int) -> Image.Image:
    palette = PALETTES[flavor]
    pattern = dict(FLAVOR_PATTERNS.get(flavor, FLAVOR_PATTERNS["generic"]))
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
                        pattern=pattern)
        elif kind == "ground":
            family = name.split("_")[1]
            _paint_wall(draw, rng, strip_box, family, palette, 0,
                        world_w, world_h, ground=True, pattern=pattern)
        elif kind == "storefront":
            _paint_storefront(draw, rng, strip_box, palette, world_w, world_h)
        elif kind == "roller":
            _paint_roller(draw, rng, strip_box, palette, world_w, world_h)
        elif kind == "plain":
            family = name.split("_")[1]
            _material_base(draw, rng, strip_box, family, palette, 0, pattern)
        elif kind == "trim":
            draw.rectangle((0, y0, size - 1, y1 - 1), fill=palette["trim"])
            _speckle(draw, rng, strip_box, palette["trim"], 8, size // 2)
        elif kind == "roof":
            _paint_roof(draw, rng, strip_box, name, palette, pattern)
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
    atlas_cfg = config["atlas"]
    layout = build_layout(int(atlas_cfg["size"]))
    layout["flavors"] = {
        flavor: f"textures/{texture_name(flavor)}"
        for flavor in atlas_cfg["flavors"]
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
        img = paint_atlas(flavor, layout, int(atlas_cfg["seed"]))
        path = os.path.join(textures_dir, texture_name(flavor))
        img.save(path, optimize=True)
        print(f"wrote {path}")
    print(f"wrote {args.layout_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
