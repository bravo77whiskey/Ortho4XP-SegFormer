import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_SFR_Asset_Inventory as ASSETINV
import O4_SFR_Building_Overlay as BLD
import O4_SFR_Roof_Color as ROOF
from scripts import generate_sfr_roof_color_visuals as ROOF_VISUALS


def _descriptor(family, lab):
    return {
        "family": family,
        "lab": tuple(float(value) for value in lab),
        "rgb": (128, 128, 128),
    }


def _asset(path, bounds, family=None, lab=None):
    asset = {
        "kind": "object",
        "path": path,
        "bounds_m": bounds,
        "source": "test",
        "footprint_class": BLD.BLD_CLASS_MEDIUM,
    }
    if family is not None:
        asset.update({
            "roof_color_complete": True,
            "roof_color_consistent": True,
            "roof_color_variants": (
                _descriptor(family, lab),
            ),
        })
    return asset


def _write_obj(path, texture_name="roof.png", wrapped=False):
    roof_uv = "1.25 -0.75" if wrapped else "0.25 0.25"
    lines = [
        "I",
        "800",
        "OBJ",
        "",
        f"TEXTURE {texture_name}",
        "POINT_COUNTS 12 0 0 18",
        # Top roof: red texture quarter.
        f"VT -2 5 -2 0 1 0 {roof_uv}",
        f"VT  2 5 -2 0 1 0 {roof_uv}",
        f"VT  2 5  2 0 1 0 {roof_uv}",
        f"VT -2 5  2 0 1 0 {roof_uv}",
        # Coincident lower surface: maps blue, but must lose the top envelope.
        "VT -2 4 -2 0 -1 0 0.75 0.25",
        "VT  2 4 -2 0 -1 0 0.75 0.25",
        "VT  2 4  2 0 -1 0 0.75 0.25",
        "VT -2 4  2 0 -1 0 0.75 0.25",
        # Vertical wall: also maps blue and must fail the roof-normal gate.
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


def _write_split_texture(path, roof_rgb=(190, 55, 35), other_rgb=(25, 55, 190)):
    texture = np.empty((8, 8, 3), dtype=np.uint8)
    texture[:] = np.asarray(roof_rgb, dtype=np.uint8)
    texture[:, 4:] = np.asarray(other_rgb, dtype=np.uint8)
    Image.fromarray(texture, mode="RGB").save(path)


def test_lab_family_boundaries():
    assert ROOF.descriptor_from_lab((34.9, 0.0, 0.0))["family"] == "neutral_dark"
    assert ROOF.descriptor_from_lab((35.0, 0.0, 0.0))["family"] == "neutral_mid"
    assert ROOF.descriptor_from_lab((72.0, 0.0, 0.0))["family"] == "neutral_mid"
    assert ROOF.descriptor_from_lab((72.1, 0.0, 0.0))["family"] == "neutral_light"
    assert ROOF.descriptor_from_lab((55.0, 30.0, 15.0))["family"] == "warm"
    assert ROOF.descriptor_from_lab((55.0, -30.0, 20.0))["family"] == "green"
    assert ROOF.descriptor_from_lab((55.0, -10.0, -30.0))["family"] == "blue"
    assert ROOF.descriptor_from_lab((55.0, 25.0, -20.0))["family"] == "violet"


def test_rooftop_sampler_uses_rgb_and_inset_pixels():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[2:18, 2:18] = (20, 40, 210)
    image[5:15, 5:15] = (210, 45, 25)
    polygon = np.array([[2, 2], [17, 2], [17, 17], [2, 17]], dtype=np.float32)

    descriptor = ROOF.sample_rooftop_color(image, polygon)

    assert descriptor["family"] == "warm"
    assert descriptor["rgb"][0] > descriptor["rgb"][2]
    assert descriptor["pixel_count"] >= 9


def test_rooftop_sampler_falls_back_to_full_tiny_polygon():
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    image[2:4, 2:4] = (30, 170, 45)
    polygon = np.array([[2, 2], [3, 2], [3, 3], [2, 3]], dtype=np.float32)

    descriptor = ROOF.sample_rooftop_color(image, polygon)

    assert descriptor is not None
    assert descriptor["family"] == "green"
    assert descriptor["pixel_count"] < 9


def test_obj_analyzer_keeps_top_roof_and_supports_wrapped_uvs():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        obj = root / "house.obj"
        _write_obj(obj, wrapped=True)
        _write_split_texture(root / "roof.png")

        descriptor = ROOF.analyze_obj_roof_color(str(obj), cache_dir=str(root / "cache"))

        assert descriptor is not None
        assert descriptor["family"] == "warm"
        assert descriptor["rgb"][0] > descriptor["rgb"][2]
        assert descriptor["sample_count"] > 0


def test_obj_analyzer_uses_dds_sibling_when_declared_png_is_missing():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        obj = root / "house.obj"
        _write_obj(obj, texture_name="roof.png")
        texture = np.full((8, 8, 3), (35, 65, 190), dtype=np.uint8)
        Image.fromarray(texture, mode="RGB").save(root / "roof.dds", format="DDS")

        descriptor = ROOF.analyze_obj_roof_color(str(obj))

        assert descriptor is not None
        assert descriptor["family"] == "blue"
        assert descriptor["texture_path"].lower().endswith("roof.dds")


def test_obj_analyzer_returns_none_for_missing_texture():
    with tempfile.TemporaryDirectory() as tmp:
        obj = Path(tmp) / "house.obj"
        _write_obj(obj, texture_name="missing.png")

        assert ROOF.analyze_obj_roof_color(str(obj)) is None


def test_obj_color_cache_invalidates_when_texture_changes():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        obj = root / "house.obj"
        texture = root / "roof.png"
        cache = root / "cache"
        _write_obj(obj)
        _write_split_texture(texture, roof_rgb=(190, 45, 30))
        first = ROOF.analyze_obj_roof_color(str(obj), cache_dir=str(cache))

        _write_split_texture(texture, roof_rgb=(30, 65, 195))
        stat = texture.stat()
        os.utime(texture, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
        second = ROOF.analyze_obj_roof_color(str(obj), cache_dir=str(cache))

        assert first["family"] == "warm"
        assert second["family"] == "blue"


def test_alias_requires_every_physical_variant_to_match():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        warm_a = root / "warm_a.obj"
        warm_b = root / "warm_b.obj"
        blue = root / "blue.obj"
        for obj in (warm_a, warm_b, blue):
            _write_obj(obj, texture_name=obj.with_suffix(".png").name)
        _write_split_texture(warm_a.with_suffix(".png"), roof_rgb=(190, 55, 35))
        _write_split_texture(warm_b.with_suffix(".png"), roof_rgb=(165, 70, 45))
        _write_split_texture(blue.with_suffix(".png"), roof_rgb=(30, 65, 195))

        def export(path):
            return ASSETINV.LibraryExport(
                package_name="test",
                package_dir=str(root),
                library_txt=str(root / "library.txt"),
                command="EXPORT",
                virtual_path="test/house.obj",
                physical_path=Path(path).name,
                resolved_path=str(path),
            )

        pools = {BLD.BLD_CLASS_MEDIUM: [
            _asset("test/house.obj", (-2.0, 2.0, -2.0, 2.0))
        ]}
        counts = ROOF.enrich_asset_roof_colors(
            pools, [export(warm_a), export(warm_b)], cache_dir=str(root / "cache")
        )
        assert counts["safe"] == 1
        assert pools[BLD.BLD_CLASS_MEDIUM][0]["roof_color_consistent"]

        backup = ASSETINV.LibraryExport(
            package_name="test",
            package_dir=str(root),
            library_txt=str(root / "library.txt"),
            command="EXPORT_BACKUP",
            virtual_path="test/house.obj",
            physical_path="missing-backup.obj",
            resolved_path=None,
        )
        pools = {BLD.BLD_CLASS_MEDIUM: [
            _asset("test/house.obj", (-2.0, 2.0, -2.0, 2.0))
        ]}
        counts = ROOF.enrich_asset_roof_colors(
            pools, [export(warm_a), backup], cache_dir=str(root / "cache")
        )
        assert counts["safe"] == 1

        pools = {BLD.BLD_CLASS_MEDIUM: [
            _asset("test/house.obj", (-2.0, 2.0, -2.0, 2.0))
        ]}
        counts = ROOF.enrich_asset_roof_colors(
            pools, [export(warm_a), export(blue)], cache_dir=str(root / "cache")
        )
        assert counts["inconsistent"] == 1
        assert not pools[BLD.BLD_CLASS_MEDIUM][0]["roof_color_consistent"]

        missing = export(root / "missing.obj")
        pools = {BLD.BLD_CLASS_MEDIUM: [
            _asset("test/house.obj", (-2.0, 2.0, -2.0, 2.0))
        ]}
        counts = ROOF.enrich_asset_roof_colors(pools, [export(warm_a), missing])
        assert counts["unavailable"] == 1
        assert not pools[BLD.BLD_CLASS_MEDIUM][0]["roof_color_complete"]


def _selector_context(assets, detection_size=20.0):
    table = BLD._build_yolo_object_candidate_index({
        BLD.BLD_CLASS_MEDIUM: assets,
    })
    yolo_poly = BLD._points_from_yolo_heading(
        (40.0, 40.0), detection_size, detection_size, 0.0
    )
    detection = {
        "length_m": detection_size,
        "width_m": detection_size,
        "area_m2": detection_size * detection_size,
        "placement_class": BLD.BLD_CLASS_MEDIUM,
    }
    return table, detection, np.rint(yolo_poly).astype(np.int32)


def test_selector_prefers_later_color_match_that_still_fits():
    blue = _asset("blue.obj", (-7.0, 7.0, -7.0, 7.0), "blue", (45, -5, -30))
    warm = _asset("warm.obj", (-6.0, 6.0, -6.0, 6.0), "warm", (45, 30, 20))
    table, detection, yolo_poly = _selector_context([blue, warm])
    all_assets = {asset["path"]: asset for asset in (blue, warm)}
    by_family = ROOF.color_assets_by_family({
        BLD.BLD_CLASS_MEDIUM: [blue, warm]
    })

    legacy, _ = BLD._select_yolo_object_candidate(
        table, detection, yolo_poly, 40, 40, 0.0, 1.0,
        min_coverage=0.30,
        enabled_assets_by_path=all_assets,
    )
    selected, status = BLD._select_yolo_object_candidate(
        table, detection, yolo_poly, 40, 40, 0.0, 1.0,
        min_coverage=0.30,
        enabled_assets_by_path=all_assets,
        roof_color=_descriptor("warm", (47, 28, 18)),
        color_assets_by_family=by_family,
    )

    assert legacy["asset"]["path"] == "blue.obj"
    assert status == "selected"
    assert selected["asset"]["path"] == "warm.obj"
    assert selected["roof_color_match"] is True


def test_selector_rejects_nonfitting_color_match_then_uses_legacy_fit():
    blue = _asset("blue.obj", (-5.0, 5.0, -5.0, 5.0), "blue", (45, -5, -30))
    warm_too_large = _asset(
        "warm-large.obj", (-9.0, 9.0, -9.0, 9.0), "warm", (45, 30, 20)
    )
    table, detection, yolo_poly = _selector_context(
        [blue, warm_too_large], detection_size=12.0
    )
    all_assets = {asset["path"]: asset for asset in (blue, warm_too_large)}
    by_family = ROOF.color_assets_by_family({
        BLD.BLD_CLASS_MEDIUM: [blue, warm_too_large]
    })

    selected, status = BLD._select_yolo_object_candidate(
        table, detection, yolo_poly, 40, 40, 0.0, 1.0,
        min_coverage=0.30,
        enabled_assets_by_path=all_assets,
        roof_color=_descriptor("warm", (47, 28, 18)),
        color_assets_by_family=by_family,
    )

    assert status == "selected"
    assert selected["asset"]["path"] == "blue.obj"
    assert selected["roof_color_match"] is False


def test_selector_rejects_color_match_on_occupancy_then_uses_clear_legacy_fit():
    blue = _asset(
        "blue.obj", (1.0, 9.0, -4.0, 4.0), "blue", (45, -5, -30)
    )
    warm = _asset(
        "warm.obj", (-9.0, -1.0, -4.0, 4.0), "warm", (45, 30, 20)
    )
    assets = [blue, warm]
    table, detection, yolo_poly = _selector_context(assets, detection_size=24.0)
    all_assets = {asset["path"]: asset for asset in assets}
    by_family = ROOF.color_assets_by_family({
        BLD.BLD_CLASS_MEDIUM: assets
    })
    static = np.zeros((80, 80), dtype=np.uint8)
    spacing = np.zeros_like(static)

    def footprint_mask(asset, heading):
        mask = np.zeros_like(static)
        polygon = BLD._footprint_poly(
            40, 40, asset["bounds_m"], heading, 1.0
        )
        cv2.fillPoly(mask, [np.int32(polygon)], 1)
        return mask

    clear_blue = footprint_mask(blue, -90.0)
    for warm_heading in (-90.0, 0.0):
        warm_mask = footprint_mask(warm, warm_heading)
        ys, xs = np.nonzero((warm_mask != 0) & (clear_blue == 0))
        assert xs.size
        spacing[ys[0], xs[0]] = 1
    assert not np.any((spacing != 0) & (clear_blue != 0))

    selected, status = BLD._select_yolo_object_candidate(
        table, detection, yolo_poly, 40, 40, 0.0, 1.0,
        min_coverage=0.10,
        enabled_assets_by_path=all_assets,
        roof_color=_descriptor("warm", (47, 28, 18)),
        color_assets_by_family=by_family,
        static_occ_mask=static,
        building_spacing_mask=spacing,
        scratch_mask=np.zeros_like(static),
        static_occ_integral=cv2.integral(static, sdepth=cv2.CV_32S),
        spacing_occ_integral=cv2.integral(spacing, sdepth=cv2.CV_32S),
    )

    assert status == "selected"
    assert selected["asset"]["path"] == "blue.obj"
    assert selected["roof_color_match"] is False


def test_selector_without_compatible_alias_preserves_legacy_choice_and_status():
    blue = _asset("blue.obj", (-5.0, 5.0, -5.0, 5.0), "blue", (45, -5, -30))
    table, detection, yolo_poly = _selector_context([blue], detection_size=12.0)
    all_assets = {"blue.obj": blue}
    by_family = ROOF.color_assets_by_family({
        BLD.BLD_CLASS_MEDIUM: [blue]
    })

    legacy, legacy_status = BLD._select_yolo_object_candidate(
        table, detection, yolo_poly, 40, 40, 0.0, 1.0,
        min_coverage=0.30,
        enabled_assets_by_path=all_assets,
    )
    selected, status = BLD._select_yolo_object_candidate(
        table, detection, yolo_poly, 40, 40, 0.0, 1.0,
        min_coverage=0.30,
        enabled_assets_by_path=all_assets,
        roof_color=_descriptor("warm", (47, 28, 18)),
        color_assets_by_family=by_family,
    )

    assert status == legacy_status
    assert selected["asset"]["path"] == legacy["asset"]["path"]
    assert selected["heading"] == legacy["heading"]
    assert np.array_equal(selected["footprint_poly"], legacy["footprint_poly"])
    assert selected["roof_color_match"] is False


def test_targeted_library_export_filter_keeps_only_requested_alias():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "one.obj").write_text("", encoding="utf-8")
        (root / "two.obj").write_text("", encoding="utf-8")
        library = root / "library.txt"
        library.write_text(
            "A\n800\nLIBRARY\n"
            "EXPORT test/one.obj one.obj\n"
            "EXPORT test/two.obj two.obj\n",
            encoding="utf-8",
        )

        exports = ASSETINV.parse_library_exports(
            str(library),
            target_virtual_paths=["/test/two.obj"],
        )

        assert [export.virtual_path for export in exports] == ["test/two.obj"]


def test_visual_diagnostic_exercises_match_and_fit_fallback():
    with tempfile.TemporaryDirectory() as tmp:
        sheet_path, manifest_path = ROOF_VISUALS.generate(tmp)

        with Image.open(sheet_path) as sheet:
            assert sheet.size == (
                ROOF_VISUALS.PANEL_SIZE[0] * 2 + 24,
                ROOF_VISUALS.PANEL_SIZE[1] * 2 + 24,
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["source_roof"]["family"] == "warm"
        assert manifest["warm_obj"]["family"] == "warm"
        assert manifest["blue_obj"]["family"] == "blue"
        assert manifest["color_match_case"] == {
            "legacy_asset": "blue.obj",
            "roof_color_match": True,
            "selected_asset": "warm.obj",
            "status": "selected",
        }
        assert manifest["fit_fallback_case"] == {
            "rejected_color_asset": "warm-too-large.obj",
            "roof_color_match": False,
            "selected_asset": "blue.obj",
            "status": "selected",
        }


def test_production_visual_overlay_uses_asset_roof_color_and_status_legend():
    asset = {
        "roof_color_variants": (
            {"rgb": (180, 50, 30)},
            {"rgb": (200, 70, 50)},
        ),
    }
    assert BLD._asset_roof_viz_rgb(asset) == (190, 60, 40)

    image = Image.new("RGBA", (400, 180), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    polygon = np.array([[20, 20], [40, 20], [40, 40], [20, 40]])
    records = [
        (polygon, "match", (190, 60, 40)),
        (polygon, "fallback", (30, 65, 195)),
        (polygon, "unavailable", (235, 232, 220)),
    ]
    BLD._draw_roof_color_viz_legend(draw, records, 400)

    rendered = np.asarray(image)
    assert np.count_nonzero(rendered[:, :, 3]) > 100
