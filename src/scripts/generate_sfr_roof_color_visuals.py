"""Generate human-readable previews for the synthetic roof-colour fixtures.

This is a visual companion to ``tests/sfr_roof_color_test.py``.  It exercises
the production OBJ analyser, rooftop sampler, and two-pass object selector,
then writes a contact sheet and a JSON manifest under ``test_output``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SRC_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import O4_SFR_Building_Overlay as BLD
import O4_SFR_Roof_Color as ROOF


PANEL_SIZE = (620, 430)
BACKGROUND = (25, 29, 36)
PANEL_BACKGROUND = (35, 41, 50)
TEXT = (235, 238, 242)
MUTED = (166, 176, 190)
GREEN = (75, 210, 135)
AMBER = (245, 180, 70)
BLUE = (55, 105, 220)
WARM = (200, 70, 45)


def _font(size=18, bold=False):
    candidates = (
        "C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _write_obj(path, texture_name="roof.png", wrapped=False):
    roof_uv = "1.25 -0.75" if wrapped else "0.25 0.25"
    lines = [
        "I",
        "800",
        "OBJ",
        "",
        f"TEXTURE {texture_name}",
        "POINT_COUNTS 12 0 0 18",
        f"VT -2 5 -2 0 1 0 {roof_uv}",
        f"VT  2 5 -2 0 1 0 {roof_uv}",
        f"VT  2 5  2 0 1 0 {roof_uv}",
        f"VT -2 5  2 0 1 0 {roof_uv}",
        "VT -2 4 -2 0 -1 0 0.75 0.25",
        "VT  2 4 -2 0 -1 0 0.75 0.25",
        "VT  2 4  2 0 -1 0 0.75 0.25",
        "VT -2 4  2 0 -1 0 0.75 0.25",
        "VT -2 0 -2 0 0 -1 0.75 0.25",
        "VT  2 0 -2 0 0 -1 0.75 0.25",
        "VT  2 5 -2 0 0 -1 0.75 0.25",
        "VT -2 5 -2 0 0 -1 0.75 0.25",
        "IDX10 0 1 2 2 3 0 4 6 5 6",
        "IDX10 4 7 8 9 10 10 11 8",
        "TRIS 0 18",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_split_texture(path, roof_rgb, other_rgb=(25, 55, 190)):
    texture = np.empty((8, 8, 3), dtype=np.uint8)
    texture[:] = np.asarray(roof_rgb, dtype=np.uint8)
    texture[:, 4:] = np.asarray(other_rgb, dtype=np.uint8)
    Image.fromarray(texture, mode="RGB").save(path)


def _asset(path, bounds, descriptor):
    return {
        "kind": "object",
        "path": path,
        "bounds_m": bounds,
        "source": "visual-test",
        "footprint_class": BLD.BLD_CLASS_MEDIUM,
        "roof_color_complete": True,
        "roof_color_consistent": True,
        "roof_color_variants": (descriptor,),
    }


def _selector_context(assets, detection_size):
    table = BLD._build_yolo_object_candidate_index({
        BLD.BLD_CLASS_MEDIUM: assets,
    })
    polygon = BLD._points_from_yolo_heading(
        (40.0, 40.0), detection_size, detection_size, 0.0
    )
    detection = {
        "length_m": detection_size,
        "width_m": detection_size,
        "area_m2": detection_size * detection_size,
        "placement_class": BLD.BLD_CLASS_MEDIUM,
    }
    return table, detection, np.rint(polygon).astype(np.int32)


def _new_panel(title, subtitle):
    image = Image.new("RGB", PANEL_SIZE, PANEL_BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), title, fill=TEXT, font=_font(25, bold=True))
    draw.text((24, 54), subtitle, fill=MUTED, font=_font(16))
    return image, draw


def _descriptor_text(descriptor):
    lab = descriptor["lab"]
    rgb = descriptor["rgb"]
    return (
        f'{descriptor["family"]}\n'
        f'RGB {tuple(rgb)}\n'
        f'Lab ({lab[0]:.1f}, {lab[1]:.1f}, {lab[2]:.1f})'
    )


def _texture_panel(texture_path, obj_path, descriptor):
    panel, draw = _new_panel(
        "OBJ8 roof texture sampling",
        "Green marker = retained top-envelope UV sample; blue half is wall/underside",
    )
    texture = Image.open(texture_path).convert("RGB")
    preview = texture.resize((320, 320), Image.Resampling.NEAREST)
    panel.paste(preview, (24, 88))

    mesh = ROOF._parse_obj8_mesh(obj_path)
    samples = ROOF._roof_surface_samples(*mesh)
    for _cell, _y, u_coord, v_coord, _weight in samples:
        u_coord -= np.floor(u_coord)
        v_coord -= np.floor(v_coord)
        x = 24 + int(round(u_coord * 319))
        y = 88 + int(round((1.0 - v_coord) * 319))
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=GREEN, width=3)

    draw.rectangle((365, 105, 425, 165), fill=descriptor["rgb"], outline=TEXT)
    draw.text(
        (365, 180),
        _descriptor_text(descriptor),
        fill=TEXT,
        font=_font(15),
        spacing=5,
    )
    draw.text(
        (365, 260),
        f'{descriptor["sample_count"]} roof samples\n'
        "Wrapped UV: (1.25, -0.75)\n"
        "Resolved UV: (0.25, 0.25)\n\n"
        "The lower horizontal surface\n"
        "and vertical wall are excluded.",
        fill=MUTED,
        font=_font(16),
        spacing=6,
    )
    return panel


def _rooftop_panel(source_image, polygon, descriptor):
    panel, draw = _new_panel(
        "Detected rooftop image sampling",
        "The 80% inset avoids the blue edge; the median remains the warm roof colour",
    )
    preview = Image.fromarray(source_image, mode="RGB").resize(
        (320, 320), Image.Resampling.NEAREST
    )
    panel.paste(preview, (24, 88))

    scale = 320.0 / source_image.shape[1]
    outer = [(24 + float(x) * scale, 88 + float(y) * scale) for x, y in polygon]
    center = np.asarray(polygon, dtype=np.float32).mean(axis=0)
    inset = center + (np.asarray(polygon, dtype=np.float32) - center) * 0.8
    inner = [(24 + float(x) * scale, 88 + float(y) * scale) for x, y in inset]
    draw.line(outer + [outer[0]], fill=AMBER, width=4)
    draw.line(inner + [inner[0]], fill=GREEN, width=4)

    draw.rectangle((365, 105, 425, 165), fill=descriptor["rgb"], outline=TEXT)
    draw.text(
        (365, 180),
        _descriptor_text(descriptor),
        fill=TEXT,
        font=_font(15),
        spacing=5,
    )
    draw.text(
        (365, 265),
        f'{descriptor["pixel_count"]} inset pixels sampled\n\n'
        "Amber: YOLO polygon\n"
        "Green: sampling inset",
        fill=MUTED,
        font=_font(16),
        spacing=7,
    )
    return panel


def _map_points(points, origin=(-115, -55), scale=8.0):
    return [
        (origin[0] + float(x) * scale, origin[1] + float(y) * scale)
        for x, y in np.asarray(points).reshape(-1, 2)
    ]


def _selection_panel(
    title,
    subtitle,
    yolo_polygon,
    candidates,
    selected,
    source_descriptor,
    annotations,
):
    panel, draw = _new_panel(title, subtitle)
    map_box = (45, 105, 365, 425)
    draw.rectangle(map_box, fill=(20, 24, 30), outline=(80, 90, 105), width=2)
    yolo_points = _map_points(yolo_polygon)
    draw.polygon(yolo_points, fill=(70, 62, 35), outline=AMBER, width=4)

    for candidate, color, width in candidates:
        points = _map_points(candidate)
        draw.line(points + [points[0]], fill=color, width=width)
    selected_points = _map_points(selected["footprint_poly"])
    draw.line(selected_points + [selected_points[0]], fill=GREEN, width=5)

    draw.rectangle((390, 105, 440, 155), fill=source_descriptor["rgb"], outline=TEXT)
    draw.text((455, 107), "source roof", fill=MUTED, font=_font(14))
    draw.text((455, 130), source_descriptor["family"], fill=TEXT, font=_font(17, True))
    draw.text((390, 180), annotations, fill=TEXT, font=_font(16), spacing=7)
    draw.text((390, 365), "Amber: detection", fill=AMBER, font=_font(14))
    draw.text((390, 387), "Green: selected fit", fill=GREEN, font=_font(14))
    return panel


def _make_selection_case(blue_descriptor, warm_descriptor, source_descriptor):
    blue = _asset("blue.obj", (-7.0, 7.0, -7.0, 7.0), blue_descriptor)
    warm = _asset("warm.obj", (-6.0, 6.0, -6.0, 6.0), warm_descriptor)
    assets = [blue, warm]
    table, detection, yolo_polygon = _selector_context(assets, 20.0)
    all_assets = {asset["path"]: asset for asset in assets}
    by_family = ROOF.color_assets_by_family({BLD.BLD_CLASS_MEDIUM: assets})

    legacy, legacy_status = BLD._select_yolo_object_candidate(
        table,
        detection,
        yolo_polygon,
        40,
        40,
        0.0,
        1.0,
        min_coverage=0.30,
        enabled_assets_by_path=all_assets,
    )
    selected, status = BLD._select_yolo_object_candidate(
        table,
        detection,
        yolo_polygon,
        40,
        40,
        0.0,
        1.0,
        min_coverage=0.30,
        enabled_assets_by_path=all_assets,
        roof_color=source_descriptor,
        color_assets_by_family=by_family,
    )
    assert legacy_status == status == "selected"
    assert legacy["asset"]["path"] == "blue.obj"
    assert selected["asset"]["path"] == "warm.obj"
    assert selected["roof_color_match"] is True

    panel = _selection_panel(
        "Colour preference after fit gates",
        "Both objects fit; the warm object wins even though geometry-only chose blue",
        yolo_polygon,
        [
            (legacy["footprint_poly"], BLUE, 3),
            (selected["footprint_poly"], WARM, 3),
        ],
        selected,
        source_descriptor,
        "Legacy choice: blue.obj\n"
        "Colour choice: warm.obj\n"
        "Result: colour match\n\n"
        "Both footprints pass the\n"
        "existing containment gates.",
    )
    return panel, {
        "legacy_asset": legacy["asset"]["path"],
        "selected_asset": selected["asset"]["path"],
        "status": status,
        "roof_color_match": selected["roof_color_match"],
    }


def _make_fallback_case(blue_descriptor, warm_descriptor, source_descriptor):
    blue = _asset("blue.obj", (-5.0, 5.0, -5.0, 5.0), blue_descriptor)
    warm = _asset("warm-too-large.obj", (-9.0, 9.0, -9.0, 9.0), warm_descriptor)
    assets = [blue, warm]
    table, detection, yolo_polygon = _selector_context(assets, 12.0)
    all_assets = {asset["path"]: asset for asset in assets}
    by_family = ROOF.color_assets_by_family({BLD.BLD_CLASS_MEDIUM: assets})
    selected, status = BLD._select_yolo_object_candidate(
        table,
        detection,
        yolo_polygon,
        40,
        40,
        0.0,
        1.0,
        min_coverage=0.30,
        enabled_assets_by_path=all_assets,
        roof_color=source_descriptor,
        color_assets_by_family=by_family,
    )
    assert status == "selected"
    assert selected["asset"]["path"] == "blue.obj"
    assert selected["roof_color_match"] is False

    rejected_poly = BLD._footprint_poly(40, 40, warm["bounds_m"], 0.0, 1.0)
    panel = _selection_panel(
        "Fit-first fallback behaviour",
        "The warm object exceeds the detection, so the unchanged blue geometry fit wins",
        yolo_polygon,
        [
            (rejected_poly, WARM, 3),
            (selected["footprint_poly"], BLUE, 3),
        ],
        selected,
        source_descriptor,
        "Warm candidate: rejected\n"
        "Legacy fit: blue.obj\n"
        "Result: fit fallback\n\n"
        "Colour never bypasses the\n"
        "existing containment gate.",
    )
    return panel, {
        "rejected_color_asset": warm["path"],
        "selected_asset": selected["asset"]["path"],
        "status": status,
        "roof_color_match": selected["roof_color_match"],
    }


def generate(output_dir):
    output_dir = Path(output_dir).resolve()
    fixture_dir = output_dir / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    warm_obj = fixture_dir / "warm.obj"
    warm_texture = fixture_dir / "warm.png"
    blue_obj = fixture_dir / "blue.obj"
    blue_texture = fixture_dir / "blue.png"
    _write_obj(warm_obj, texture_name=warm_texture.name, wrapped=True)
    _write_split_texture(warm_texture, (190, 55, 35))
    _write_obj(blue_obj, texture_name=blue_texture.name)
    _write_split_texture(blue_texture, (30, 65, 195))

    cache_dir = output_dir / "cache"
    warm_descriptor = ROOF.analyze_obj_roof_color(warm_obj, cache_dir=cache_dir)
    blue_descriptor = ROOF.analyze_obj_roof_color(blue_obj, cache_dir=cache_dir)
    assert warm_descriptor and warm_descriptor["family"] == "warm"
    assert blue_descriptor and blue_descriptor["family"] == "blue"

    source_image = np.zeros((20, 20, 3), dtype=np.uint8)
    source_image[2:18, 2:18] = (20, 40, 210)
    source_image[5:15, 5:15] = (210, 45, 25)
    source_polygon = np.asarray(
        [[2, 2], [17, 2], [17, 17], [2, 17]], dtype=np.float32
    )
    source_descriptor = ROOF.sample_rooftop_color(source_image, source_polygon)
    assert source_descriptor and source_descriptor["family"] == "warm"

    texture_panel = _texture_panel(warm_texture, warm_obj, warm_descriptor)
    rooftop_panel = _rooftop_panel(
        source_image, source_polygon, source_descriptor
    )
    match_panel, match_result = _make_selection_case(
        blue_descriptor, warm_descriptor, source_descriptor
    )
    fallback_panel, fallback_result = _make_fallback_case(
        blue_descriptor, warm_descriptor, source_descriptor
    )

    panels = {
        "obj_texture_sampling": texture_panel,
        "rooftop_sampling": rooftop_panel,
        "color_match_selection": match_panel,
        "fit_fallback_selection": fallback_panel,
    }
    outputs = {}
    for name, panel in panels.items():
        path = output_dir / f"{name}.png"
        panel.save(path)
        outputs[name] = str(path)

    sheet = Image.new(
        "RGB",
        (PANEL_SIZE[0] * 2 + 24, PANEL_SIZE[1] * 2 + 24),
        BACKGROUND,
    )
    sheet.paste(texture_panel, (8, 8))
    sheet.paste(rooftop_panel, (PANEL_SIZE[0] + 16, 8))
    sheet.paste(match_panel, (8, PANEL_SIZE[1] + 16))
    sheet.paste(fallback_panel, (PANEL_SIZE[0] + 16, PANEL_SIZE[1] + 16))
    sheet_path = output_dir / "roof_color_visual_tests.png"
    sheet.save(sheet_path)
    outputs["contact_sheet"] = str(sheet_path)

    manifest = {
        "analyzer_version": ROOF.ROOF_COLOR_ANALYZER_VERSION,
        "source_roof": source_descriptor,
        "warm_obj": warm_descriptor,
        "blue_obj": blue_descriptor,
        "color_match_case": match_result,
        "fit_fallback_case": fallback_result,
        "outputs": outputs,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return sheet_path, manifest_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "test_output" / "sfr_roof_color_visuals"),
        help="Directory for generated previews and fixtures",
    )
    args = parser.parse_args(argv)
    sheet_path, manifest_path = generate(args.output)
    print(f"Visual test sheet: {sheet_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
