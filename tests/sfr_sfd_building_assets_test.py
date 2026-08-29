import sys
import tempfile
import unittest
import bz2
import math
import os
from unittest import mock
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_SFR_Building_Overlay as BLD
import O4_SFR_Asset_Inventory as ASSETINV
import O4_SFR_DSF_Utils as DSF


def _max_corner_deviation_deg(points):
    """Largest departure from a right angle across a quad's four corners."""
    pts = np.asarray(points, dtype=np.float64).reshape(4, 2)
    worst = 0.0
    for idx in range(4):
        a = pts[(idx - 1) % 4] - pts[idx]
        b = pts[(idx + 1) % 4] - pts[idx]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 90.0
        cos_ang = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        worst = max(worst, abs(math.degrees(math.acos(cos_ang)) - 90.0))
    return worst


def _is_rectangle(points, tol_deg=0.5):
    return _max_corner_deviation_deg(points) <= tol_deg


def _paths_for_classes(pools, classes):
    return {
        asset["path"]
        for cls in classes
        for asset in pools.get(cls, ())
    }


def _write_tile_dsf(package: Path, lat=22, lon=120):
    lat_group = int(lat // 10) * 10
    lon_group = int(lon // 10) * 10
    lat_block = f"{'+' if lat_group >= 0 else '-'}{abs(lat_group):02d}"
    lon_block = f"{'+' if lon_group >= 0 else '-'}{abs(lon_group):03d}"
    lat_tile = f"{'+' if lat >= 0 else '-'}{abs(lat):02d}"
    lon_tile = f"{'+' if lon >= 0 else '-'}{abs(lon):03d}"
    dsf = package / "Earth nav data" / f"{lat_block}{lon_block}" / f"{lat_tile}{lon_tile}.dsf"
    dsf.parent.mkdir(parents=True, exist_ok=True)
    dsf.write_text("", encoding="utf-8")
    return dsf


def _write_library_package(custom: Path, name: str, lines):
    package = custom / name
    package.mkdir(parents=True, exist_ok=True)
    (package / "library.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return package


def _activate_packages(custom: Path, *packages: Path):
    (custom / "scenery_packs.ini").write_text(
        "".join(
            f"SCENERY_PACK Custom Scenery/{package.name}/\n"
            for package in packages
        ),
        encoding="utf-8",
    )


def _write_obj8(path: Path, vertices):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["I", "800", "OBJ", ""]
    lines.extend(f"VT {x} {y} {z} 0 0 0 0 0" for x, y, z in vertices)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _poly_long_axis_heading(points):
    pts = np.asarray(points, dtype=np.float32)
    edges = np.roll(pts, -1, axis=0) - pts
    long_vec = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
    return (
        math.degrees(math.atan2(float(long_vec[0]), -float(long_vec[1]))) + 360.0
    ) % 180.0


class SfdBuildingAssetTests(unittest.TestCase):
    def test_yolo_analysis_size_downscales_high_zl_to_target_scale(self):
        self.assertEqual(
            BLD._yolo_analysis_size(4096, 4096, source_zl=19, target_zl=16),
            (512, 512),
        )
        self.assertEqual(
            BLD._yolo_analysis_size(4096, 4096, source_zl=16, target_zl=16),
            (4096, 4096),
        )

    def test_yolo_default_stride_matches_model_input_size(self):
        self.assertEqual(BLD.DEFAULT_YOLO_OBB_STRIDE, BLD.DEFAULT_YOLO_OBB_IMGSZ)

    def test_build_yolo_zl16_analysis_image_writes_downscaled_cache(self):
        source = np.zeros((8, 8, 3), dtype=np.uint8)
        source[:, :, 0] = 255

        with tempfile.TemporaryDirectory() as tmp:
            analysis_path = BLD.build_yolo_zl16_analysis_image(
                tmp,
                til_y_top=100,
                til_x_left=200,
                provider="BI",
                source_zl=18,
                target_zl=16,
                cache_dir=tmp,
                source_image=source,
                source_path=None,
                bounds=(1.0, 0.0, 2.0, 3.0),
            )

            with Image.open(analysis_path) as image:
                self.assertEqual(image.size, (2, 2))

            self.assertTrue(Path(analysis_path).with_suffix(".json").exists())

    def test_scale_yolo_detections_to_image_maps_points_and_center_only(self):
        detections = [{
            "points": [[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [0.0, 3.0]],
            "center": [1.0, 1.5],
            "area_m2": 42.0,
            "confidence": 0.75,
        }]

        scaled = BLD._scale_yolo_detections_to_image(
            detections,
            scale_x=4.0,
            scale_y=4.0,
            img_w=16,
            img_h=16,
        )

        self.assertEqual(scaled[0]["points"][2], [8.0, 12.0])
        self.assertEqual(scaled[0]["center"], [4.0, 6.0])
        self.assertEqual(scaled[0]["area_m2"], 42.0)

    def test_covered_fractions_full_parent_coverage(self):
        # ZL16 parent (28320, 54528) with all 4 ZL17 children present: the
        # children's rects union to the whole parent footprint.
        entries = [
            (28320, 54528, 16, "28320_54528_Arc16.dds"),
            (56640, 109056, 17, "56640_109056_BI17.dds"),
            (56640, 109072, 17, "56640_109072_BI17.dds"),
            (56656, 109056, 17, "56656_109056_BI17.dds"),
            (56656, 109072, 17, "56656_109072_BI17.dds"),
        ]
        covered = BLD.compute_covered_fractions(entries)
        rects = covered["28320_54528_Arc16.dds"]
        self.assertEqual(len(rects), 4)
        self.assertEqual(
            set(rects),
            {
                (0.0, 0.0, 0.5, 0.5),
                (0.5, 0.0, 1.0, 0.5),
                (0.0, 0.5, 0.5, 1.0),
                (0.5, 0.5, 1.0, 1.0),
            },
        )
        # children themselves are never marked covered
        self.assertNotIn("56640_109056_BI17.dds", covered)

    def test_covered_fractions_partial_and_multi_level(self):
        # One ZL18 grandchild inside the ZL16 parent's NW ZL17 quadrant.
        entries = [
            (28320, 54528, 16, "28320_54528_Arc16.dds"),
            (113280, 218112, 18, "113280_218112_BI18.dds"),
        ]
        covered = BLD.compute_covered_fractions(entries)
        rects = covered["28320_54528_Arc16.dds"]
        self.assertEqual(rects, ((0.0, 0.0, 0.25, 0.25),))

    def test_covered_rects_px_and_detection_filter(self):
        rects_px = BLD.covered_rects_to_px(((0.5, 0.5, 1.0, 1.0),), 4096, 4096)
        self.assertEqual(rects_px, ((2048, 2048, 4096, 4096),))
        detections = [
            {"points": [], "center": [2100.0, 2100.0]},
            {"points": [], "center": [100.0, 100.0]},
        ]
        kept, dropped = BLD.filter_detections_by_coverage(detections, rects_px)
        self.assertEqual(dropped, 1)
        self.assertEqual(kept[0]["center"], [100.0, 100.0])

    def test_yolo_crops_skip_fully_covered_windows(self):
        image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        rects_px = ((512, 512, 1024, 1024),)
        offsets = [
            (x, y) for x, y, _ in BLD._iter_yolo_crops(
                image, 512, skip_rects_px=rects_px
            )
        ]
        self.assertEqual(offsets, [(0, 0), (512, 0), (0, 512)])
        # no skip rects: all 4 crops
        self.assertEqual(
            len(list(BLD._iter_yolo_crops(image, 512))), 4
        )

    def test_geo_boxes_and_stock_coverage_filter(self):
        rects_px = ((2048, 2048, 4096, 4096),)
        geo = BLD.geo_boxes_from_covered_px(
            rects_px, 4096, 4096,
            lat_n=23.5, lat_s=23.0, lon_w=119.0, lon_e=119.5,
        )
        self.assertEqual(len(geo), 1)
        west, south, east, north = geo[0]
        self.assertAlmostEqual(west, 119.25)
        self.assertAlmostEqual(east, 119.5)
        self.assertAlmostEqual(north, 23.25)
        self.assertAlmostEqual(south, 23.0)

        stock = type("StockRes", (), {})()
        stock.placed_objects = [
            (119.3, 23.1, 0.0, "a.obj"),   # inside covered box -> dropped
            (119.1, 23.4, 0.0, "b.obj"),   # outside -> kept
        ]
        stock.placed_facades = [
            ([(119.3, 23.05), (119.31, 23.05), (119.31, 23.06)], "f.fac", 8.0),
        ]
        stock.placed_draped = []
        stock.occupied_px_polys = [
            np.array([[2100, 2100], [2200, 2100], [2200, 2200], [2100, 2200]]),
            np.array([[10, 10], [20, 10], [20, 20], [10, 20]]),
        ]
        dropped = BLD.filter_stock_results_by_coverage(stock, rects_px, geo)
        self.assertEqual(dropped, 2)
        self.assertEqual([p[3] for p in stock.placed_objects], ["b.obj"])
        self.assertEqual(stock.placed_facades, [])
        self.assertEqual(len(stock.occupied_px_polys), 1)

    def test_mask_covered_regions_blanks_rects(self):
        mask = np.ones((8, 8), dtype=np.uint8)
        BLD.mask_covered_regions(mask, ((4, 4, 8, 8),))
        self.assertEqual(int(mask.sum()), 64 - 16)

    def test_yolo_cache_key_tracks_coverage_and_native_mode(self):
        base = dict(
            fname="f.dds", img_w=4096, img_h=4096, checkpoint=None,
            imgsz=512, stride=512, conf=0.18, iou=0.5, max_det=3000,
        )
        native_key = BLD._yolo_obb_cache_key(**base)
        covered_key = BLD._yolo_obb_cache_key(
            **base, covered_rects=((2048, 2048, 4096, 4096),)
        )
        analysis_key = BLD._yolo_obb_cache_key(
            **base, analysis_target_zl=16, analysis_signature={"x": 1}
        )
        self.assertNotEqual(native_key, covered_key)
        self.assertNotEqual(native_key, analysis_key)
        self.assertIsNone(native_key["analysis_target_zl"])
        self.assertIsNone(native_key["covered_rects"])

    def test_stock_yolo_batch_default_tolerates_legacy_module(self):
        legacy_stock = type("LegacyStockYolo", (), {})()

        with mock.patch.object(BLD, "STOCKYOLO", legacy_stock):
            self.assertEqual(BLD._stock_yolo_batch_default(), 1)

    def test_stock_yolo_batch_metadata_tolerates_legacy_results(self):
        legacy_result = type("LegacyStockYoloResults", (), {})()

        effective, fell_back = BLD._stock_yolo_batch_metadata(
            legacy_result,
            requested_batch_size=4,
        )

        self.assertEqual(effective, 4)
        self.assertFalse(fell_back)

    def test_stock_yolo_batch_metadata_preserves_fallback_results(self):
        result = type("StockYoloResultsLike", (), {
            "effective_batch_size": 1,
            "batch_fell_back": True,
        })()

        effective, fell_back = BLD._stock_yolo_batch_metadata(
            result,
            requested_batch_size=8,
        )

        self.assertEqual(effective, 1)
        self.assertTrue(fell_back)

    def test_stock_yolo_pass_compat_retries_without_new_filter_kwargs(self):
        calls = []

        def legacy_run_stock_yolo_pass(image, *, model, img_w, img_h):
            calls.append((image, model, img_w, img_h))
            return "legacy-result"

        legacy_stock = type("LegacyStockYolo", (), {
            "run_stock_yolo_pass": staticmethod(legacy_run_stock_yolo_pass),
        })()

        with mock.patch.object(BLD, "STOCKYOLO", legacy_stock):
            result = BLD._run_stock_yolo_pass_compat(
                "image",
                model="model",
                img_w=512,
                img_h=512,
                asset_map={},
                static_classes=(),
            )

        self.assertEqual(result, "legacy-result")
        self.assertEqual(calls, [("image", "model", 512, 512)])

    def test_yolo_obb_height_class_decode_returns_height_bin(self):
        model_class = (
            (BLD.BLD_CLASS_APARTMENT_BLOCK - 1) * BLD.YOLO_HEIGHT_BIN_COUNT
            + BLD.YOLO_HEIGHT_BINS_M.index(60.0)
        )

        placement, height_m = BLD._decode_yolo_obb_detection_class(
            model_class,
            model_class_count=len(BLD.BLD_PLACEMENT_CLASSES) * BLD.YOLO_HEIGHT_BIN_COUNT,
        )

        self.assertEqual(placement, BLD.BLD_CLASS_APARTMENT_BLOCK)
        self.assertEqual(height_m, 60.0)

    def test_legacy_yolo_class_decode_returns_placement_only(self):
        placement, height_m = BLD._decode_yolo_obb_detection_class(
            BLD.BLD_CLASS_MEDIUM - 1,
            model_class_count=len(BLD.BLD_PLACEMENT_CLASSES),
        )

        self.assertEqual(placement, BLD.BLD_CLASS_MEDIUM)
        self.assertIsNone(height_m)

    def test_height_priors_use_finite_asset_range_per_class(self):
        pools = {cls: [] for cls in BLD.BLD_PLACEMENT_CLASSES}
        pools[BLD.BLD_CLASS_MEDIUM] = [
            {"height_m": 5.0},
            {"height_m": 12.0},
            {"height_m": None},
            {"height_m": BLD.FACADE_HEIGHT_PRIOR_CAP_M + 1.0},
        ]

        priors = BLD._height_priors_by_class(pools)

        self.assertEqual(
            priors[BLD.BLD_CLASS_MEDIUM],
            (5.0, 12.0, BLD.DEFAULT_FACADE_HEIGHT_M[BLD.BLD_CLASS_MEDIUM]),
        )

    def test_randomized_facade_height_is_seeded_and_bounded(self):
        priors = {BLD.BLD_CLASS_MEDIUM: (5.0, 12.0, 7.0)}
        rng_a = np.random.default_rng(12345)
        rng_b = np.random.default_rng(12345)

        heights_a = [
            BLD._randomized_facade_height_m(
                rng_a, priors, BLD.BLD_CLASS_MEDIUM
            )
            for _ in range(8)
        ]
        heights_b = [
            BLD._randomized_facade_height_m(
                rng_b, priors, BLD.BLD_CLASS_MEDIUM
            )
            for _ in range(8)
        ]

        self.assertEqual(heights_a, heights_b)
        self.assertTrue(all(5.0 <= height <= 12.0 for height in heights_a))

    def test_randomized_facade_height_falls_back_for_empty_or_single_range(self):
        empty_pools = {cls: [] for cls in BLD.BLD_PLACEMENT_CLASSES}
        empty_priors = BLD._height_priors_by_class(empty_pools)

        self.assertEqual(
            BLD._randomized_facade_height_m(
                np.random.default_rng(1),
                empty_priors,
                BLD.BLD_CLASS_SMALL_RESIDENTIAL,
            ),
            BLD.DEFAULT_FACADE_HEIGHT_M[BLD.BLD_CLASS_SMALL_RESIDENTIAL],
        )
        self.assertEqual(
            BLD._randomized_facade_height_m(
                np.random.default_rng(1),
                {BLD.BLD_CLASS_MEDIUM: (9.0, 9.0, 9.0)},
                BLD.BLD_CLASS_MEDIUM,
            ),
            9.0,
        )

    def test_regional_height_fallback_replaces_only_class_defaults(self):
        detections = [
            {
                "height_m": 7.0,
                "height_source": "class_default",
                "placement_class": BLD.BLD_CLASS_MEDIUM,
            },
            {
                "height_m": 18.0,
                "height_source": "height_bins",
                "placement_class": BLD.BLD_CLASS_MEDIUM,
            },
        ]

        BLD._apply_regional_height_fallbacks(detections, "africa")

        prior = BLD.HEIGHTPRIORS.height_prior("africa", BLD.BLD_CLASS_MEDIUM)
        self.assertAlmostEqual(detections[0]["height_m"], prior.median_m)
        self.assertEqual(detections[0]["height_source"], "regional_class_default")
        self.assertEqual(detections[0]["height_prior_region"], "africa")
        self.assertEqual(detections[1]["height_m"], 18.0)
        self.assertEqual(detections[1]["height_source"], "height_bins")

    def test_unknown_region_uses_generic_height_prior(self):
        detection = {
            "height_m": 7.0,
            "height_source": "class_default",
            "placement_class": BLD.BLD_CLASS_SMALL_RESIDENTIAL,
        }

        BLD._apply_regional_height_fallbacks([detection], "not-a-region")

        prior = BLD.HEIGHTPRIORS.height_prior(
            "generic", BLD.BLD_CLASS_SMALL_RESIDENTIAL
        )
        self.assertAlmostEqual(detection["height_m"], prior.median_m)
        self.assertEqual(detection["height_prior_region"], "generic")

    def test_dominant_detection_landcover_ignores_buildings_and_background(self):
        veg_map = np.full((9, 9), BLD._SF_BUILDING, dtype=np.int16)
        veg_map[1:8, 1:8] = BLD._SF_BARELAND
        veg_map[4, 4] = 0

        dominant = BLD._dominant_detection_landcover(
            veg_map, 4, 4, m_per_px=20.0
        )

        self.assertEqual(dominant, BLD._SF_BARELAND)

    def test_heightnet_uses_regional_limits_and_contextual_industrial_cap(self):
        detections = [
            {
                "height_m": 8.0,
                "height_source": "class_default",
                "placement_class": BLD.BLD_CLASS_LARGE,
                "center": (2.0, 2.0),
            },
            {
                "height_m": 10.0,
                "height_source": "class_default",
                "placement_class": BLD.BLD_CLASS_EXTRA_LARGE,
                "center": (5.0, 5.0),
            },
            {
                "height_m": 12.0,
                "height_source": "class_default",
                "placement_class": BLD.BLD_CLASS_APARTMENT_BLOCK,
                "area_m2": 1_200.0,
                "max_side_m": 45.0,
                "center": (5.0, 2.0),
            },
            {
                "height_m": 13.0,
                "height_source": "class_default",
                "placement_class": BLD.BLD_CLASS_APARTMENT_BLOCK,
                "area_m2": 8_000.0,
                "max_side_m": 120.0,
                "center": (2.0, 5.0),
            },
            {
                "height_m": 14.0,
                "height_source": "class_default",
                "placement_class": BLD.BLD_CLASS_LARGE,
                "center": (4.0, 4.0),
            },
        ]
        veg_map = np.full((8, 8), BLD._SF_DEVELOPED, dtype=np.int16)
        veg_map[:4, :4] = BLD._SF_BARELAND

        with mock.patch.object(
            BLD.HEIGHTMODEL,
            "predict_detection_heights",
            return_value=np.asarray(
                [80.0, 80.0, 80.0, 80.0, np.nan], dtype=np.float64
            ),
        ):
            BLD._apply_heightnet_to_detections(
                object(),
                np.zeros((8, 8, 3), dtype=np.uint8),
                detections,
                20.0,
                region="europe",
                veg_map=veg_map,
            )

        self.assertEqual(detections[0]["height_raw_m"], 80.0)
        self.assertEqual(detections[0]["height_m"], BLD.LARGE_FOOTPRINT_HEIGHT_CAP_M)
        self.assertEqual(detections[0]["height_source"], "heightnet")
        self.assertEqual(
            detections[1]["height_m"],
            BLD.HEIGHTPRIORS.height_prior(
                "europe", BLD.BLD_CLASS_EXTRA_LARGE
            ).hard_ceiling_m,
        )
        self.assertEqual(
            detections[2]["height_m"],
            BLD.HEIGHTPRIORS.height_prior(
                "europe", BLD.BLD_CLASS_APARTMENT_BLOCK
            ).hard_ceiling_m,
        )
        self.assertEqual(
            detections[3]["height_m"],
            BLD.HEIGHTPRIORS.height_prior(
                "europe", BLD.BLD_CLASS_APARTMENT_BLOCK
            ).hard_ceiling_m,
        )
        self.assertNotIn("height_raw_m", detections[4])
        self.assertEqual(detections[4]["height_m"], 14.0)
        self.assertEqual(detections[4]["height_source"], "class_default")

    def test_heightnet_softly_reduces_only_p95_excess(self):
        class_id = BLD.BLD_CLASS_TINY_RESIDENTIAL
        prior = BLD.HEIGHTPRIORS.height_prior("europe", class_id)
        raw_above_p95 = prior.p95_m + 4.0
        detections = [
            {"placement_class": class_id, "height_m": 3.0},
            {"placement_class": class_id, "height_m": 3.0},
        ]

        with mock.patch.object(
            BLD.HEIGHTMODEL,
            "predict_detection_heights",
            return_value=np.asarray(
                [prior.p95_m - 1.0, raw_above_p95], dtype=np.float64
            ),
        ):
            BLD._apply_heightnet_to_detections(
                object(),
                np.zeros((8, 8, 3), dtype=np.uint8),
                detections,
                1.0,
                region="europe",
            )

        self.assertAlmostEqual(detections[0]["height_m"], prior.p95_m - 1.0)
        self.assertAlmostEqual(
            detections[1]["height_m"],
            prior.p95_m + 4.0 * BLD.HEIGHTPRIORS.P95_EXCESS_RETAIN,
        )
        self.assertEqual(detections[1]["height_adjustment"], "soft_p95")

    def test_pipeline_yolo_defaults_match_config_defaults(self):
        import O4_Cfg_Vars as CFG
        import O4_SFR_Pipeline as PIPE

        self.assertEqual(
            PIPE.sfr_bld_yolo_conf,
            CFG.cfg_tile_vars["sfr_bld_yolo_conf"]["default"],
        )
        self.assertEqual(
            PIPE.sfr_bld_yolo_max_det,
            CFG.cfg_tile_vars["sfr_bld_yolo_max_det"]["default"],
        )
        self.assertEqual(
            PIPE.sfr_bld_yolo_min_coverage,
            CFG.cfg_tile_vars["sfr_bld_yolo_min_coverage"]["default"],
        )

    def test_alias_oversized_footprint_fraction(self):
        class Ex:
            def __init__(self, rp):
                self.resolved_path = rp
        declared = BLD._bounds_from_dimensions(16.0, 12.0)  # max-side 16
        bounds_map = {
            "big1": BLD._bounds_from_dimensions(60.0, 10.0),   # 60 >> 16
            "big2": BLD._bounds_from_dimensions(58.0, 10.0),
            "small": BLD._bounds_from_dimensions(16.5, 12.2),
        }
        orig = BLD._read_obj8_bounds
        BLD._read_obj8_bounds = lambda rp, cd=None: bounds_map.get(rp)
        try:
            # All variants resolve to a much bigger footprint -> frac 1.0 (drop).
            frac, n = BLD._alias_oversized_footprint_fraction(
                declared, [Ex("big1"), Ex("big2")], None)
            self.assertEqual((frac, n), (1.0, 2))
            # One oversized among three -> minority (kept).
            frac, n = BLD._alias_oversized_footprint_fraction(
                declared, [Ex("big1"), Ex("small"), Ex("small")], None)
            self.assertEqual(n, 3)
            self.assertLess(frac, 0.5)
            # Declared-large (industrial) vs same-scale mesh -> not oversized.
            big_declared = BLD._bounds_from_dimensions(60.0, 120.0)
            frac, n = BLD._alias_oversized_footprint_fraction(
                big_declared, [Ex("big1")], None)
            self.assertEqual(frac, 0.0)
        finally:
            BLD._read_obj8_bounds = orig

    def test_drop_oversized_aliased_assets_no_exports_keeps_all(self):
        pools = {BLD.BLD_CLASS_MEDIUM: [
            {'kind': 'object', 'path': 'simheaven/x/y.obj',
             'bounds_m': BLD._bounds_from_dimensions(16.0, 12.0)},
        ]}
        out, dropped, paths = BLD._drop_oversized_aliased_assets(pools, [], None)
        self.assertEqual(dropped, 0)
        self.assertEqual(len(out[BLD.BLD_CLASS_MEDIUM]), 1)

    def test_footprint_containment_rejects_real_overhang_keeps_quant_slop(self):
        # Detection polygon: 100x100 px square.
        outer = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)

        def inside(inner):
            return BLD._footprint_inside_detection(
                np.asarray(inner, dtype=np.float32), outer
            )

        # Fully inside -> accepted.
        self.assertTrue(inside([[10, 10], [90, 10], [90, 90], [10, 90]]))

        # Sub-pixel overhang (1px sliver along one edge) within the quantisation
        # margin -> accepted.
        self.assertTrue(inside([[-1, 10], [90, 10], [90, 90], [-1, 90]]))

        # Gross overhang (~30% of the footprint outside the polygon) -> rejected,
        # regardless of the old fractional tolerance.
        self.assertFalse(inside([[-30, 10], [70, 10], [70, 90], [-30, 90]]))

        # The allowance is an absolute perimeter band, not a fraction of area:
        # a large object overhanging by the same *fraction* is still rejected.
        big_outer = np.array(
            [[0, 0], [1000, 0], [1000, 1000], [0, 1000]], dtype=np.float32
        )
        big_inner = np.array(
            [[-300, 100], [700, 100], [700, 900], [-300, 900]], dtype=np.float32
        )
        self.assertFalse(BLD._footprint_inside_detection(big_inner, big_outer))

    def test_osm_tile_peer_path_handles_all_road_sources(self):
        base = r"C:\O4XP\OSM_data\+30+110\+36+117\+36+117_big_roads.osm.bz2"

        self.assertEqual(
            BLD._osm_tile_peer_path(base, "_all_roads.osm.bz2"),
            r"C:\O4XP\OSM_data\+30+110\+36+117\+36+117_all_roads.osm.bz2",
        )
        self.assertEqual(
            BLD._osm_tile_peer_path(
                r"C:\O4XP\OSM_data\+30+110\+36+117\+36+117_all_roads.osm.bz2",
                "_small_roads.osm.bz2",
            ),
            r"C:\O4XP\OSM_data\+30+110\+36+117\+36+117_small_roads.osm.bz2",
        )

    def test_transient_cache_peer_path_uses_temp_dir_for_missing_peer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = os.path.join(tmpdir, "+36+117_big_roads.osm.bz2")
            transient = os.path.join(tmpdir, "run-cache")
            os.makedirs(transient)
            Path(base).write_bytes(b"x")

            peer = BLD._transient_cache_peer_path(
                base,
                "_excl_bld_rail_res.osm.bz2",
                transient,
            )

            self.assertEqual(
                peer,
                os.path.join(transient, "+36+117_excl_bld_rail_res.osm.bz2"),
            )

    def test_transient_cache_peer_path_keeps_existing_external_peer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = os.path.join(tmpdir, "+36+117_big_roads.osm.bz2")
            existing_peer = os.path.join(tmpdir, "+36+117_all_roads.osm.bz2")
            transient = os.path.join(tmpdir, "run-cache")
            os.makedirs(transient)
            Path(base).write_bytes(b"x")
            Path(existing_peer).write_bytes(b"x")

            peer = BLD._transient_cache_peer_path(
                base,
                "_all_roads.osm.bz2",
                transient,
            )

            self.assertEqual(peer, existing_peer)

    def test_load_osm_roads_parses_generic_highway_extract(self):
        xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<osm>
  <node id="1" lat="36.0" lon="117.0" />
  <node id="2" lat="36.1" lon="117.1" />
  <way id="10">
    <nd ref="1" />
    <nd ref="2" />
    <tag k="highway" v="residential" />
  </way>
</osm>"""
        with tempfile.TemporaryDirectory() as tmpdir:
            osm_path = Path(tmpdir) / "+36+117_all_roads.osm.bz2"
            with bz2.open(osm_path, "wb") as handle:
                handle.write(xml)

            roads = BLD._load_osm_roads(str(osm_path))

        self.assertEqual(len(roads), 1)
        self.assertEqual(roads[0]["type"], "residential")
        self.assertEqual(roads[0]["pts"], [(36.0, 117.0), (36.1, 117.1)])

    def test_medium_is_an_explicit_placement_class(self):
        self.assertIn(BLD.BLD_CLASS_MEDIUM, BLD.BLD_PLACEMENT_CLASSES)
        self.assertEqual(BLD.BLD_CLASS_LABELS[BLD.BLD_CLASS_MEDIUM], "medium footprint")
        self.assertEqual(BLD.BLD_CLASS_STANDARD_RESIDENTIAL, BLD.BLD_CLASS_MEDIUM)

    def test_building_spacing_is_edge_gap_plus_class_span(self):
        class_spans = {
            BLD.BLD_CLASS_TINY_RESIDENTIAL: 6.0,
            BLD.BLD_CLASS_SMALL_RESIDENTIAL: 10.0,
            BLD.BLD_CLASS_COMPACT_RESIDENTIAL: 8.0,
            BLD.BLD_CLASS_MEDIUM: 20.0,
            BLD.BLD_CLASS_SMALL_APARTMENT: 30.0,
            BLD.BLD_CLASS_APARTMENT_BLOCK: 40.0,
            BLD.BLD_CLASS_LARGE: 50.0,
            BLD.BLD_CLASS_EXTRA_LARGE: 80.0,
        }

        self.assertEqual(
            {
                cls: BLD._spacing_for_zone_class_m(20.0, cls, class_spans)
                for cls in BLD.BLD_PLACEMENT_CLASSES
            },
            {
                BLD.BLD_CLASS_TINY_RESIDENTIAL: 26.0,
                BLD.BLD_CLASS_SMALL_RESIDENTIAL: 30.0,
                BLD.BLD_CLASS_COMPACT_RESIDENTIAL: 28.0,
                BLD.BLD_CLASS_MEDIUM: 40.0,
                BLD.BLD_CLASS_SMALL_APARTMENT: 50.0,
                BLD.BLD_CLASS_APARTMENT_BLOCK: 60.0,
                BLD.BLD_CLASS_LARGE: 70.0,
                BLD.BLD_CLASS_EXTRA_LARGE: 100.0,
            },
        )

    def test_orientation_order_aligns_long_footprint_side_first(self):
        self.assertEqual(
            BLD._orientation_angles_for_bounds((-5.0, 5.0, -20.0, 20.0), 90.0),
            (90.0, 0.0),
        )
        self.assertEqual(
            BLD._orientation_angles_for_bounds((-20.0, 20.0, -5.0, 5.0), 90.0),
            (0.0, 90.0),
        )

    def test_yolo_obb_detection_conversion_extracts_heading_and_class(self):
        # Trained YOLO emits classes 0..7 which map to BLD_PLACEMENT_CLASSES
        # 1..8 via the `+1` shift in `_yolo_obb_detection_from_points`. cls=0
        # therefore lands in BLD_CLASS_TINY_RESIDENTIAL regardless of the
        # detection geometry.
        detection = BLD._yolo_obb_detection_from_points(
            np.array([
                [10.0, 10.0],
                [30.0, 10.0],
                [30.0, 20.0],
                [10.0, 20.0],
            ]),
            confidence=0.8,
            cls=0,
            img_w=64,
            img_h=64,
            m_per_px=1.0,
        )

        self.assertIsNotNone(detection)
        self.assertAlmostEqual(detection["center"][0], 20.0)
        self.assertAlmostEqual(detection["center"][1], 15.0)
        self.assertAlmostEqual(detection["heading"], 90.0)
        self.assertEqual(detection["placement_class"], BLD.BLD_CLASS_TINY_RESIDENTIAL)
        self.assertEqual(detection["model_class"], 0)

    def test_yolo_obb_inference_streams_crop_results(self):
        class FakeObb:
            def __init__(self):
                self.xyxyxyxy = BLD.torch.tensor(
                    [[[10.0, 10.0], [30.0, 10.0], [30.0, 20.0], [10.0, 20.0]]]
                )
                self.conf = BLD.torch.tensor([0.8])
                self.cls = BLD.torch.tensor([0.0])

        class FakeResult:
            def __init__(self):
                self.obb = FakeObb()

        class FakeYolo:
            def __init__(self):
                self.calls = []
                self.results_consumed = 0

            def predict(self, **kwargs):
                self.calls.append(kwargs)
                self.assert_stream = kwargs.get("stream")

                def _results():
                    yield FakeResult()
                    self.results_consumed += 1

                return _results()

        model = FakeYolo()
        image = np.zeros((512, 1024, 3), dtype=np.uint8)

        detections = BLD._run_yolo_obb_inference(
            model,
            image,
            imgsz=512,
            stride=512,
            conf=0.18,
            iou=0.5,
            max_det=1000,
            device="cpu",
            m_per_px=1.0,
        )

        self.assertEqual(len(model.calls), 2)
        self.assertTrue(all(call["stream"] for call in model.calls))
        self.assertEqual(model.results_consumed, 2)
        self.assertEqual(len(detections), 2)
        self.assertAlmostEqual(detections[0]["center"][0], 20.0)
        self.assertAlmostEqual(detections[1]["center"][0], 532.0)

    def test_yolo_obb_batched_inference_matches_legacy_offsets(self):
        class FakeObb:
            def __init__(self):
                self.xyxyxyxy = BLD.torch.tensor(
                    [[[10.0, 10.0], [30.0, 10.0], [30.0, 20.0], [10.0, 20.0]]]
                )
                self.conf = BLD.torch.tensor([0.8])
                self.cls = BLD.torch.tensor([0.0])

        class FakeResult:
            def __init__(self):
                self.obb = FakeObb()

        class FakeYolo:
            def __init__(self):
                self.calls = []

            def predict(self, **kwargs):
                self.calls.append(kwargs)
                source = kwargs["source"]
                n_results = len(source) if isinstance(source, list) else 1

                def _results():
                    for _ in range(n_results):
                        yield FakeResult()

                return _results()

        image = np.zeros((512, 1536, 3), dtype=np.uint8)
        common = dict(
            imgsz=512,
            stride=512,
            conf=0.18,
            iou=0.5,
            max_det=1000,
            device="cpu",
            m_per_px=1.0,
        )

        legacy_model = FakeYolo()
        batched_model = FakeYolo()
        legacy = BLD._run_yolo_obb_inference(
            legacy_model, image, batch_size=1, **common
        )
        batched = BLD._run_yolo_obb_inference(
            batched_model, image, batch_size=2, **common
        )

        self.assertEqual(legacy, batched)
        self.assertEqual(len(legacy_model.calls), 3)
        self.assertEqual(len(batched_model.calls), 2)
        self.assertTrue(all(call["stream"] for call in batched_model.calls))
        self.assertEqual(batched_model.calls[0]["batch"], 2)

    def test_yolo_obb_oom_fallback_reports_metadata_and_discards_partial_results(self):
        class FakeObb:
            def __init__(self):
                self.xyxyxyxy = BLD.torch.tensor(
                    [[[10.0, 10.0], [30.0, 10.0], [30.0, 20.0], [10.0, 20.0]]]
                )
                self.conf = BLD.torch.tensor([0.8])
                self.cls = BLD.torch.tensor([0.0])

        class FakeResult:
            def __init__(self):
                self.obb = FakeObb()

        class OomOnBatchYolo:
            def __init__(self):
                self.calls = []

            def predict(self, **kwargs):
                self.calls.append(kwargs)
                source = kwargs["source"]
                if isinstance(source, list):
                    def _failing():
                        yield FakeResult()
                        raise RuntimeError("CUDA out of memory.")

                    return _failing()
                return iter((FakeResult(),))

        image = np.zeros((512, 1536, 3), dtype=np.uint8)
        common = dict(
            imgsz=512,
            stride=512,
            conf=0.18,
            iou=0.5,
            max_det=1000,
            device="cpu",
            m_per_px=1.0,
        )

        baseline_model = OomOnBatchYolo()
        baseline = BLD._run_yolo_obb_inference(
            baseline_model, image, batch_size=1, **common
        )
        fallback_model = OomOnBatchYolo()
        fallback = BLD._run_yolo_obb_inference(
            fallback_model,
            image,
            batch_size=8,
            return_metadata=True,
            **common,
        )

        self.assertEqual(baseline, fallback["detections"])
        self.assertEqual(fallback["requested_batch"], 8)
        self.assertEqual(fallback["effective_batch"], 1)
        self.assertTrue(fallback["fell_back"])
        self.assertEqual(len(fallback_model.calls), 1 + len(baseline_model.calls))

    def test_yolo_obb_fallback_cache_key_uses_effective_batch(self):
        detections = [{"area_m2": 100.0, "confidence": 0.8}]
        requested_key = BLD._yolo_obb_cache_key(
            "1_2_BI18.dds", 512, 512, None, 512, 512, 0.18, 0.5, 1000,
            batch_size=8,
        )
        effective_key = BLD._yolo_obb_cache_key(
            "1_2_BI18.dds", 512, 512, None, 512, 512, 0.18, 0.5, 1000,
            batch_size=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "1_2_BI18_yolo_obb.pkl")
            BLD._save_yolo_obb_cache(cache_path, effective_key, detections)

            self.assertNotEqual(requested_key, effective_key)
            self.assertIsNone(BLD._load_yolo_obb_cache(cache_path, requested_key))
            self.assertEqual(
                BLD._load_yolo_obb_cache(cache_path, effective_key),
                detections,
            )

    def test_yolo_overlap_suppression_removes_lower_confidence_duplicates(self):
        detections = [
            {
                "confidence": 0.90,
                "area_m2": 400.0,
                "points": [[10, 10], [30, 10], [30, 30], [10, 30]],
            },
            {
                "confidence": 0.60,
                "area_m2": 400.0,
                "points": [[12, 12], [32, 12], [32, 32], [12, 32]],
            },
            {
                "confidence": 0.50,
                "area_m2": 400.0,
                "points": [[60, 60], [80, 60], [80, 80], [60, 80]],
            },
        ]

        kept, dropped = BLD._suppress_overlapping_yolo_detections(
            detections,
            coverage_threshold=0.35,
            min_overlap_m2=25.0,
            m_per_px=1.0,
        )

        self.assertEqual(dropped, 1)
        self.assertEqual([det["confidence"] for det in kept], [0.90, 0.50])

    @staticmethod
    def _small_pair_with(big_detection):
        return [
            {
                "id": "small-a",
                "confidence": 0.70,
                "area_m2": 400.0,
                "max_side_m": 20.0,
                "points": [[10, 10], [30, 10], [30, 30], [10, 30]],
            },
            {
                "id": "small-b",
                "confidence": 0.65,
                "area_m2": 400.0,
                "max_side_m": 20.0,
                "points": [[55, 55], [75, 55], [75, 75], [55, 75]],
            },
            big_detection,
        ]

    def test_warehouse_scale_detection_survives_max_footprint_gate(self):
        # The max-footprint gate now sits at the world's largest building, so
        # warehouse-scale OBBs reach overlap suppression instead of being
        # dropped up front; the pair rule decides who wins from there.
        detections = self._small_pair_with({
            "id": "warehouse",
            "confidence": 0.95,
            "area_m2": 9_800.0,
            "max_side_m": 140.0,
            "points": [[0, 0], [140, 0], [140, 70], [0, 70]],
        })

        filtered, oversize_dropped = BLD._filter_oversized_direct_yolo_detections(
            detections
        )
        kept, overlap_dropped = BLD._suppress_overlapping_yolo_detections(
            filtered,
            coverage_threshold=0.0,
            min_overlap_m2=0.0,
            m_per_px=1.0,
            pair_rule=True,
        )

        self.assertEqual(oversize_dropped, 0)
        # The two smalls explain far too little of its ground to be the real
        # buildings, so it survives and evicts them as sub-structure boxes.
        self.assertEqual(overlap_dropped, 2)
        self.assertEqual([det["id"] for det in kept], ["warehouse"])

    def test_above_world_scale_detection_cannot_suppress_smaller_overlaps(self):
        detections = self._small_pair_with({
            "id": "field-sized",
            "confidence": 0.95,
            "area_m2": 900_000.0,
            "max_side_m": 1_500.0,
            "points": [[0, 0], [1500, 0], [1500, 600], [0, 600]],
        })

        filtered, oversize_dropped = BLD._filter_oversized_direct_yolo_detections(
            detections
        )
        kept, overlap_dropped = BLD._suppress_overlapping_yolo_detections(
            filtered,
            coverage_threshold=0.0,
            min_overlap_m2=0.0,
            m_per_px=1.0,
            pair_rule=True,
        )

        self.assertEqual(oversize_dropped, 1)
        self.assertEqual(overlap_dropped, 0)
        self.assertEqual([det["id"] for det in kept], ["small-a", "small-b"])

    def test_yolo_obb_detection_clips_bounds(self):
        detection = BLD._yolo_obb_detection_from_points(
            np.array([
                [-5.0, -5.0],
                [20.0, -5.0],
                [20.0, 10.0],
                [-5.0, 10.0],
            ]),
            confidence=0.9,
            cls=0,
            img_w=16,
            img_h=16,
            m_per_px=1.0,
        )

        self.assertIsNotNone(detection)
        points = np.asarray(detection["points"])
        self.assertGreaterEqual(points.min(), 0.0)
        self.assertLessEqual(points[:, 0].max(), 15.0)
        self.assertLessEqual(points[:, 1].max(), 15.0)

    def test_yolo_obb_detection_stays_rectangular_when_clipped(self):
        # A rotated box hanging off the texture border used to be clamped
        # corner-by-corner, which sheared it into an irregular quad and — via
        # the direct-YOLO facade fallback, which extrudes these corners
        # verbatim — produced facades with non-right angles in the sim.
        rot = math.radians(30.0)
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        half_long, half_short = 30.0, 12.0
        center = np.array([12.0, 40.0])
        corners = np.array([
            center + s * half_long * np.array([cos_r, sin_r])
                   + t * half_short * np.array([-sin_r, cos_r])
            for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ])

        detection = BLD._yolo_obb_detection_from_points(
            corners,
            confidence=0.9,
            cls=0,
            img_w=128,
            img_h=128,
            m_per_px=1.0,
        )

        self.assertIsNotNone(detection)
        points = np.asarray(detection["points"], dtype=np.float64)
        self.assertGreaterEqual(points.min(), 0.0)
        self.assertLessEqual(points.max(), 127.0)
        self.assertTrue(_is_rectangle(points))
        # Orientation is preserved: the box is trimmed along its own axes.
        self.assertAlmostEqual(
            float(detection["heading"]), (30.0 + 90.0) % 180.0, places=3
        )

    def test_rect_clip_obb_leaves_in_bounds_box_untouched(self):
        corners = np.array([
            [10.0, 10.0], [40.0, 20.0], [35.0, 35.0], [5.0, 25.0],
        ], dtype=np.float32)
        clipped = BLD._rect_clip_obb_to_image(corners, 128, 128)
        np.testing.assert_allclose(clipped, corners)

    def test_rect_clip_obb_survives_axis_aligned_and_corner_overhang(self):
        for corners in (
            # axis-aligned, overhanging left and top
            np.array([[-5.0, -5.0], [20.0, -5.0], [20.0, 10.0], [-5.0, 10.0]]),
            # 45 deg, poking out of the bottom-right corner
            np.array([[50.0, 40.0], [70.0, 60.0], [60.0, 70.0], [40.0, 50.0]]),
        ):
            with self.subTest(corners=corners.tolist()):
                clipped = BLD._rect_clip_obb_to_image(corners, 64, 64)
                self.assertIsNotNone(clipped)
                clipped = np.asarray(clipped, dtype=np.float64)
                self.assertGreaterEqual(clipped.min(), -1e-4)
                self.assertLessEqual(clipped.max(), 63.0 + 1e-4)
                self.assertTrue(_is_rectangle(clipped))

    def test_rect_clip_obb_drops_box_that_cannot_be_trimmed(self):
        # Entirely outside the image: no same-orientation sub-rectangle fits.
        corners = np.array([
            [200.0, 200.0], [260.0, 220.0], [255.0, 240.0], [195.0, 220.0],
        ])
        self.assertIsNone(BLD._rect_clip_obb_to_image(corners, 64, 64))

    def test_yolo_guidance_prefers_nearby_detection_heading_and_class(self):
        guidance = BLD._build_yolo_guidance(
            [
                {
                    "center": [8.0, 8.0],
                    "heading": 45.0,
                    "confidence": 0.7,
                    "area_m2": 220.0,
                    "placement_class": BLD.BLD_CLASS_COMPACT_RESIDENTIAL,
                    "points": [
                        [4.0, 6.0],
                        [12.0, 6.0],
                        [12.0, 10.0],
                        [4.0, 10.0],
                    ],
                }
            ],
            img_h=32,
            img_w=32,
            m_per_px=1.0,
        )
        cand_cls = np.array([BLD.BLD_CLASS_MEDIUM], dtype=np.uint8)

        refined, changed = BLD._refine_candidate_classes_from_yolo(
            np.array([10], dtype=np.int32),
            np.array([8], dtype=np.int32),
            cand_cls,
            guidance,
        )

        self.assertEqual(changed, 1)
        self.assertEqual(int(refined[0]), BLD.BLD_CLASS_COMPACT_RESIDENTIAL)
        self.assertAlmostEqual(BLD._yolo_heading_for_candidate(guidance, 10, 8), 45.0)

    def test_yolo_template_translates_nearest_obb_shape(self):
        guidance = BLD._build_yolo_guidance(
            [
                {
                    "center": [8.0, 8.0],
                    "heading": 45.0,
                    "confidence": 0.7,
                    "area_m2": 220.0,
                    "placement_class": BLD.BLD_CLASS_COMPACT_RESIDENTIAL,
                    "points": [
                        [4.0, 6.0],
                        [12.0, 6.0],
                        [12.0, 10.0],
                        [4.0, 10.0],
                    ],
                }
            ],
            img_h=32,
            img_w=32,
            m_per_px=1.0,
        )

        template = BLD._yolo_template_for_candidate(guidance, 18, 18)

        self.assertIsNotNone(template)
        points, cls, heading = template
        np.testing.assert_array_equal(
            points,
            np.array([[14, 16], [22, 16], [22, 20], [14, 20]], dtype=np.int32),
        )
        self.assertEqual(cls, BLD.BLD_CLASS_COMPACT_RESIDENTIAL)
        self.assertAlmostEqual(heading, 45.0)

    def test_yolo_consensus_template_uses_fuzzy_majority_heading_and_shape(self):
        def tpl(center, heading, long_len, short_len, confidence=0.8):
            return {
                "center": np.asarray(center, dtype=np.float32),
                "points": BLD._points_from_yolo_heading(center, long_len, short_len, heading),
                "class": BLD.BLD_CLASS_COMPACT_RESIDENTIAL,
                "heading": heading,
                "confidence": confidence,
            }

        consensus = BLD._consensus_yolo_template(
            [
                tpl((10, 10), 88.0, 20.0, 10.0),
                tpl((30, 10), 91.0, 21.0, 9.5),
                tpl((50, 10), 94.0, 19.0, 10.5),
                tpl((70, 10), 25.0, 48.0, 12.0),
                tpl((90, 10), 27.0, 50.0, 12.0),
            ],
            heading_tol_deg=8.0,
            shape_rel_tol=0.15,
            shape_abs_tol_px=3.0,
        )

        self.assertIsNotNone(consensus)
        self.assertLessEqual(BLD._angle_delta_180(consensus["heading"], 91.0), 3.0)
        metrics = BLD._yolo_template_metrics(consensus)
        self.assertIsNotNone(metrics)
        self.assertLessEqual(abs(metrics["long_len"] - 20.0), 1.0)
        self.assertLessEqual(abs(metrics["short_len"] - 10.0), 1.0)
        self.assertEqual(consensus["consensus_heading_votes"], 3)
        self.assertEqual(consensus["consensus_shape_votes"], 3)

    def test_yolo_consensus_heading_wraps_at_180_degrees(self):
        templates = []
        for center, heading in [((10, 10), 178.0), ((30, 10), 1.0), ((50, 10), 3.0)]:
            templates.append({
                "center": np.asarray(center, dtype=np.float32),
                "points": BLD._points_from_yolo_heading(center, 18.0, 8.0, heading),
                "class": BLD.BLD_CLASS_SMALL_RESIDENTIAL,
                "heading": heading,
                "confidence": 0.8,
            })

        consensus = BLD._consensus_yolo_template(templates, heading_tol_deg=6.0)

        self.assertIsNotNone(consensus)
        self.assertLessEqual(
            min(
                BLD._angle_delta_180(consensus["heading"], 0.0),
                BLD._angle_delta_180(consensus["heading"], 180.0),
            ),
            2.0,
        )

    def test_neighbor_yolo_templates_only_supply_zones_without_obb(self):
        labels = np.zeros((20, 30), dtype=np.int32)
        labels[3:17, 2:12] = 1
        labels[3:17, 15:27] = 2
        stats = np.zeros((3, 5), dtype=np.int32)
        stats[1, BLD.cv2.CC_STAT_LEFT] = 2
        stats[1, BLD.cv2.CC_STAT_TOP] = 3
        stats[1, BLD.cv2.CC_STAT_WIDTH] = 10
        stats[1, BLD.cv2.CC_STAT_HEIGHT] = 14
        stats[1, BLD.cv2.CC_STAT_AREA] = 140
        stats[2, BLD.cv2.CC_STAT_LEFT] = 15
        stats[2, BLD.cv2.CC_STAT_TOP] = 3
        stats[2, BLD.cv2.CC_STAT_WIDTH] = 12
        stats[2, BLD.cv2.CC_STAT_HEIGHT] = 14
        stats[2, BLD.cv2.CC_STAT_AREA] = 168
        yolo_template = {
            "center": np.asarray([7.0, 10.0], dtype=np.float32),
            "points": BLD._points_from_yolo_heading((7.0, 10.0), 8.0, 4.0, 90.0),
            "class": BLD.BLD_CLASS_COMPACT_RESIDENTIAL,
            "heading": 90.0,
            "confidence": 0.9,
        }

        neighbors = BLD._neighbor_yolo_templates_by_zone(
            labels,
            stats,
            np.array([1, 2], dtype=np.int32),
            {1: [yolo_template]},
            radius_px=4,
        )

        self.assertNotIn(1, neighbors)
        self.assertIn(2, neighbors)
        self.assertIs(neighbors[2][0], yolo_template)

    def test_neighbor_yolo_templates_choose_closest_obb_zone(self):
        labels = np.zeros((20, 50), dtype=np.int32)
        labels[3:17, 2:10] = 1
        labels[3:17, 20:28] = 2
        labels[3:17, 34:42] = 3
        stats = np.zeros((4, 5), dtype=np.int32)
        for label, left in ((1, 2), (2, 20), (3, 34)):
            stats[label, BLD.cv2.CC_STAT_LEFT] = left
            stats[label, BLD.cv2.CC_STAT_TOP] = 3
            stats[label, BLD.cv2.CC_STAT_WIDTH] = 8
            stats[label, BLD.cv2.CC_STAT_HEIGHT] = 14
            stats[label, BLD.cv2.CC_STAT_AREA] = 112
        west_template = {
            "center": np.asarray([6.0, 10.0], dtype=np.float32),
            "points": BLD._points_from_yolo_heading((6.0, 10.0), 8.0, 4.0, 0.0),
            "class": BLD.BLD_CLASS_SMALL_RESIDENTIAL,
            "heading": 0.0,
            "confidence": 0.8,
        }
        east_template = {
            "center": np.asarray([38.0, 10.0], dtype=np.float32),
            "points": BLD._points_from_yolo_heading((38.0, 10.0), 8.0, 4.0, 90.0),
            "class": BLD.BLD_CLASS_EXTRA_LARGE,
            "heading": 90.0,
            "confidence": 0.8,
        }

        neighbors = BLD._neighbor_yolo_templates_by_zone(
            labels,
            stats,
            np.array([1, 2, 3], dtype=np.int32),
            {1: [west_template], 3: [east_template]},
            radius_px=100,
        )

        self.assertNotIn(1, neighbors)
        self.assertNotIn(3, neighbors)
        self.assertEqual(len(neighbors[2]), 1)
        self.assertIs(neighbors[2][0], east_template)

    def test_nearest_obb_zone_heading_ignores_class(self):
        stats = np.zeros((4, 5), dtype=np.int32)
        centroids = np.zeros((4, 2), dtype=np.float32)
        # Source OBB zone, intentionally a different placement class.
        stats[1, BLD.cv2.CC_STAT_LEFT] = 0
        stats[1, BLD.cv2.CC_STAT_TOP] = 0
        stats[1, BLD.cv2.CC_STAT_WIDTH] = 10
        stats[1, BLD.cv2.CC_STAT_HEIGHT] = 10
        centroids[1] = (5.0, 5.0)
        # Non-OBB target zone next to source.
        stats[2, BLD.cv2.CC_STAT_LEFT] = 15
        stats[2, BLD.cv2.CC_STAT_TOP] = 0
        stats[2, BLD.cv2.CC_STAT_WIDTH] = 8
        stats[2, BLD.cv2.CC_STAT_HEIGHT] = 8
        centroids[2] = (19.0, 4.0)
        # Farther non-OBB target zone also receives the same nearest source.
        stats[3, BLD.cv2.CC_STAT_LEFT] = 80
        stats[3, BLD.cv2.CC_STAT_TOP] = 0
        stats[3, BLD.cv2.CC_STAT_WIDTH] = 8
        stats[3, BLD.cv2.CC_STAT_HEIGHT] = 8
        centroids[3] = (84.0, 4.0)
        template = {
            "center": np.asarray([5.0, 5.0], dtype=np.float32),
            "points": BLD._points_from_yolo_heading((5.0, 5.0), 12.0, 4.0, 37.0),
            "class": BLD.BLD_CLASS_EXTRA_LARGE,
            "heading": 37.0,
            "confidence": 0.9,
        }

        headings, counts, source_labels, distances = BLD._nearest_obb_zone_headings(
            stats,
            centroids,
            np.array([1, 2, 3], dtype=np.int32),
            {1: [template]},
        )

        self.assertTrue(np.isnan(headings[1]))
        self.assertAlmostEqual(float(headings[2]), 37.0)
        self.assertEqual(int(counts[2]), 1)
        self.assertEqual(int(source_labels[2]), 1)
        self.assertAlmostEqual(float(distances[2]), 6.0)
        self.assertAlmostEqual(float(headings[3]), 37.0)

    def test_retarget_yolo_template_heading_preserves_shape(self):
        template = {
            "center": np.asarray([20.0, 20.0], dtype=np.float32),
            "points": BLD._points_from_yolo_heading((20.0, 20.0), 12.0, 6.0, 10.0),
            "class": BLD.BLD_CLASS_MEDIUM,
            "heading": 10.0,
            "confidence": 0.7,
        }

        retargeted = BLD._retarget_yolo_template_heading(template, 85.0)

        self.assertIsNotNone(retargeted)
        self.assertAlmostEqual(retargeted["heading"], 85.0)
        old_metrics = BLD._yolo_template_metrics(template)
        new_metrics = BLD._yolo_template_metrics(retargeted)
        self.assertAlmostEqual(old_metrics["long_len"], new_metrics["long_len"], places=4)
        self.assertAlmostEqual(old_metrics["short_len"], new_metrics["short_len"], places=4)
        self.assertEqual(retargeted["class"], BLD.BLD_CLASS_MEDIUM)

    def test_largest_fitting_yolo_template_shrinks_only_too_large_dimension(self):
        template = {
            "center": np.asarray([15.0, 30.0], dtype=np.float32),
            "points": BLD._points_from_yolo_heading((15.0, 30.0), 30.0, 12.0, 90.0),
            "class": BLD.BLD_CLASS_MEDIUM,
            "heading": 90.0,
            "confidence": 0.8,
        }
        static_occ_mask = np.zeros((60, 24), dtype=np.uint8)
        spacing_mask = np.zeros_like(static_occ_mask)
        scratch = np.zeros_like(static_occ_mask)

        poly, scales = BLD._largest_fitting_yolo_template_poly(
            template,
            15,
            30,
            img_w=24,
            img_h=60,
            static_occ_mask=static_occ_mask,
            building_spacing_mask=spacing_mask,
            scratch_mask=scratch,
        )

        self.assertIsNotNone(poly)
        long_scale, short_scale = scales
        self.assertLess(long_scale, 1.0)
        self.assertAlmostEqual(short_scale, 1.0)
        metrics = BLD._yolo_template_metrics({"points": poly})
        self.assertLess(metrics["long_len"], 30.0)
        self.assertGreaterEqual(metrics["short_len"], 11.0)

    def test_direct_yolo_poly_fits_clear_mask(self):
        static_occ_mask = np.zeros((32, 32), dtype=np.uint8)
        spacing_mask = np.zeros_like(static_occ_mask)
        scratch = np.zeros_like(static_occ_mask)
        yolo_poly = np.array(
            [[10, 10], [20, 10], [20, 20], [10, 20]],
            dtype=np.int32,
        )

        self.assertTrue(
            BLD._direct_yolo_poly_fits(
                static_occ_mask,
                spacing_mask,
                yolo_poly,
                scratch_mask=scratch,
                static_occ_integral=BLD.cv2.integral(static_occ_mask, sdepth=BLD.cv2.CV_32S),
            )
        )

    def test_prepared_direct_yolo_bbox_matches_standard_fit(self):
        detection = {
            "center": [30.2, 30.4],
            "points": [[20.4, 18.6], [42.2, 20.1], [40.8, 43.7], [19.3, 41.9]],
        }
        prepared = BLD._prepare_direct_yolo_detection(detection, 64, 64)
        poly = prepared["poly"]
        expected_bbox = BLD._fit_bbox_for_poly(poly, 64, 64)

        self.assertTrue(prepared["valid"])
        self.assertEqual(prepared["bbox"], expected_bbox)
        self.assertEqual((prepared["jx"], prepared["jy"]), (30, 30))
        self.assertGreater(prepared["area_px"], 1.0)

        occ_mask = np.zeros((64, 64), dtype=np.uint8)
        scratch = np.zeros_like(occ_mask)
        self.assertEqual(
            BLD._poly_fits(occ_mask, poly, scratch),
            BLD._poly_fits(occ_mask, poly, scratch, bbox=prepared["bbox"]),
        )
        occ_mask[30, 30] = 1
        self.assertEqual(
            BLD._poly_fits(occ_mask, poly, scratch),
            BLD._poly_fits(occ_mask, poly, scratch, bbox=prepared["bbox"]),
        )

    def test_direct_yolo_fit_with_bbox_matches_unbounded_path(self):
        static_occ_mask = np.zeros((64, 64), dtype=np.uint8)
        spacing_mask = np.zeros_like(static_occ_mask)
        static_occ_mask[30:34, :] = 1
        spacing_mask[12, 12] = 1
        scratch = np.zeros_like(static_occ_mask)
        yolo_poly = np.array(
            [[20, 20], [44, 20], [44, 44], [20, 44]],
            dtype=np.int32,
        )
        bbox = BLD._fit_bbox_for_poly(yolo_poly, 64, 64)

        standard = BLD._direct_yolo_poly_fits(
            static_occ_mask,
            spacing_mask,
            yolo_poly,
            scratch_mask=scratch,
            static_occ_integral=BLD.cv2.integral(static_occ_mask, sdepth=BLD.cv2.CV_32S),
        )
        bbox_path = BLD._direct_yolo_poly_fits(
            static_occ_mask,
            spacing_mask,
            yolo_poly,
            scratch_mask=scratch,
            static_occ_integral=BLD.cv2.integral(static_occ_mask, sdepth=BLD.cv2.CV_32S),
            bbox=bbox,
        )

        self.assertEqual(standard, bbox_path)

    def test_placed_yolo_overlap_gate_matches_exact_poly_fit(self):
        placed_mask = np.zeros((64, 64), dtype=np.uint8)
        placed_poly = np.array(
            [[10, 10], [20, 10], [20, 20], [10, 20]],
            dtype=np.int32,
        )
        BLD.cv2.fillPoly(placed_mask, [placed_poly], 1)
        integral = BLD.cv2.integral(placed_mask, sdepth=BLD.cv2.CV_32S)
        recent_mask = np.zeros_like(placed_mask)
        scratch = np.zeros_like(placed_mask)
        cases = [
            np.array([[24, 10], [34, 10], [34, 20], [24, 20]], dtype=np.int32),
            np.array([[18, 10], [28, 10], [28, 20], [18, 20]], dtype=np.int32),
            np.array([[20, 10], [30, 10], [30, 20], [20, 20]], dtype=np.int32),
        ]

        for poly in cases:
            bbox = BLD._fit_bbox_for_poly(poly, 64, 64)
            self.assertEqual(
                BLD._poly_fits(placed_mask, poly, scratch, bbox=bbox),
                BLD._placed_yolo_poly_fits(
                    placed_mask,
                    poly,
                    scratch,
                    placed_yolo_integral=integral,
                    recent_yolo_mask=recent_mask,
                    bbox=bbox,
                ),
            )

    def test_placed_yolo_overlap_gate_checks_recent_placements(self):
        placed_mask = np.zeros((64, 64), dtype=np.uint8)
        stale_integral = BLD.cv2.integral(placed_mask, sdepth=BLD.cv2.CV_32S)
        recent_mask = np.zeros_like(placed_mask)
        placed_poly = np.array(
            [[10, 10], [20, 10], [20, 20], [10, 20]],
            dtype=np.int32,
        )
        BLD.cv2.fillPoly(placed_mask, [placed_poly], 1)
        BLD.cv2.fillPoly(recent_mask, [placed_poly], 1)
        overlap_poly = np.array(
            [[18, 10], [28, 10], [28, 20], [18, 20]],
            dtype=np.int32,
        )

        self.assertFalse(
            BLD._placed_yolo_poly_fits(
                placed_mask,
                overlap_poly,
                np.zeros_like(placed_mask),
                placed_yolo_integral=stale_integral,
                recent_yolo_mask=recent_mask,
                bbox=BLD._fit_bbox_for_poly(overlap_poly, 64, 64),
            )
        )

    def test_yolo_object_selection_keeps_footprint_occupancy_gate(self):
        asset = {
            "kind": "object",
            "path": "mapped.obj",
            "bounds_m": (-5.0, 5.0, -5.0, 5.0),
            "mark_bounds_m": (-5.0, 5.0, -5.0, 5.0),
            "source": "test",
            "footprint_class": BLD.BLD_CLASS_MEDIUM,
        }
        table = BLD._build_yolo_object_candidate_index({BLD.BLD_CLASS_MEDIUM: [asset]})
        yolo_poly = BLD._points_from_yolo_heading((30.0, 30.0), 12.0, 10.0, 0.0)
        detection = {
            "length_m": 12.0,
            "width_m": 10.0,
            "area_m2": 120.0,
            "placement_class": BLD.BLD_CLASS_MEDIUM,
        }
        static_occ_mask = np.zeros((64, 64), dtype=np.uint8)
        spacing_mask = np.zeros_like(static_occ_mask)
        spacing_mask[30, 30] = 1

        selected, status = BLD._select_yolo_object_candidate(
            table,
            detection,
            np.rint(yolo_poly).astype(np.int32),
            30,
            30,
            0.0,
            1.0,
            static_occ_mask=static_occ_mask,
            building_spacing_mask=spacing_mask,
            scratch_mask=np.zeros_like(static_occ_mask),
            static_occ_integral=BLD.cv2.integral(static_occ_mask, sdepth=BLD.cv2.CV_32S),
        )

        self.assertIsNone(selected)
        self.assertEqual(status, "occupancy_reject")

    def test_direct_yolo_footprint_rejects_simheaven_object_overlap(self):
        simheaven_objects = {
            "lat": np.array([0.5], dtype=np.float32),
            "lon": np.array([0.5], dtype=np.float32),
            "heading": np.array([0.0], dtype=np.float32),
            "w_m": np.array([80.0], dtype=np.float32),
            "h_m": np.array([80.0], dtype=np.float32),
        }
        static_occ_mask = BLD._rasterize_simheaven_objects(
            simheaven_objects,
            lat_n=1.0,
            lat_s=0.0,
            lon_w=0.0,
            lon_e=1.0,
            img_h=100,
            img_w=100,
            m_per_px=2.0,
            margin_m=0.0,
        )
        spacing_mask = np.zeros_like(static_occ_mask)
        scratch = np.zeros_like(static_occ_mask)
        yolo_poly = np.array(
            [[40, 40], [60, 40], [60, 60], [40, 60]],
            dtype=np.int32,
        )

        self.assertFalse(
            BLD._direct_yolo_poly_fits(
                static_occ_mask,
                spacing_mask,
                yolo_poly,
                scratch_mask=scratch,
                static_occ_integral=BLD.cv2.integral(static_occ_mask, sdepth=BLD.cv2.CV_32S),
            )
        )

    def test_direct_yolo_footprint_rejects_skinny_road_crossing(self):
        static_occ_mask = np.zeros((64, 64), dtype=np.uint8)
        static_occ_mask[30:34, :] = 1
        spacing_mask = np.zeros_like(static_occ_mask)
        scratch = np.zeros_like(static_occ_mask)
        yolo_poly = np.array(
            [[20, 20], [44, 20], [44, 44], [20, 44]],
            dtype=np.int32,
        )

        self.assertFalse(
            BLD._direct_yolo_poly_fits(
                static_occ_mask,
                spacing_mask,
                yolo_poly,
                scratch_mask=scratch,
                static_occ_integral=BLD.cv2.integral(static_occ_mask, sdepth=BLD.cv2.CV_32S),
            )
        )

    def test_yolo_detection_records_oriented_meter_dimensions(self):
        points = BLD._points_from_yolo_heading((50.0, 50.0), 20.0, 8.0, 35.0)

        detection = BLD._yolo_obb_detection_from_points(
            points,
            confidence=0.9,
            cls=BLD.BLD_CLASS_MEDIUM - 1,
            img_w=100,
            img_h=100,
            m_per_px=1.0,
            model_class_count=len(BLD.BLD_PLACEMENT_CLASSES),
        )

        self.assertIsNotNone(detection)
        self.assertAlmostEqual(detection["length_m"], 20.0, delta=0.2)
        self.assertAlmostEqual(detection["width_m"], 8.0, delta=0.2)
        self.assertEqual(BLD._yolo_object_dimension_key(detection), (20, 8))

    def test_yolo_detection_heading_uses_polygon_when_xywhr_disagrees(self):
        points = BLD._points_from_yolo_heading((50.0, 50.0), 20.0, 8.0, 35.0)

        detection = BLD._yolo_obb_detection_from_points(
            points,
            confidence=0.9,
            cls=BLD.BLD_CLASS_MEDIUM - 1,
            img_w=100,
            img_h=100,
            m_per_px=1.0,
            xywhr=[50.0, 50.0, 20.0, 8.0, 0.0],
            model_class_count=len(BLD.BLD_PLACEMENT_CLASSES),
        )

        self.assertIsNotNone(detection)
        self.assertAlmostEqual(detection["heading"], 35.0, delta=0.2)
        self.assertAlmostEqual(detection["length_m"], 20.0, delta=0.2)
        self.assertAlmostEqual(detection["width_m"], 8.0, delta=0.2)

    def test_yolo_detection_diagonal_heading_uses_compass_axis(self):
        points = np.array(
            [
                [42.93, 32.32],
                [57.07, 46.46],
                [51.41, 52.12],
                [37.27, 37.98],
            ],
            dtype=np.float32,
        )

        detection = BLD._yolo_obb_detection_from_points(
            points,
            confidence=0.9,
            cls=BLD.BLD_CLASS_MEDIUM - 1,
            img_w=100,
            img_h=100,
            m_per_px=1.0,
            model_class_count=len(BLD.BLD_PLACEMENT_CLASSES),
        )

        self.assertIsNotNone(detection)
        self.assertAlmostEqual(detection["heading"], 135.0, delta=0.2)

    def test_points_from_yolo_heading_matches_compass_diagonal(self):
        points = BLD._points_from_yolo_heading((50.0, 50.0), 20.0, 8.0, 135.0)
        edges = np.roll(points, -1, axis=0) - points
        long_vec = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
        img_angle = math.degrees(math.atan2(float(long_vec[1]), float(long_vec[0])))

        self.assertAlmostEqual(img_angle, 45.0, delta=0.2)

    def test_yolo_object_fit_table_includes_both_asset_orientations(self):
        asset = {
            "kind": "object",
            "path": "rect.obj",
            "bounds_m": (-4.5, 4.5, -6.0, 6.0),
            "source": "test",
        }
        pools = {BLD.BLD_CLASS_MEDIUM: [asset]}

        table = BLD._build_yolo_object_fit_table(pools)

        self.assertIn((12, 9), table)
        self.assertIn((9, 12), table)
        self.assertEqual(table[(12, 9)][0]["asset"]["path"], "rect.obj")
        self.assertEqual(table[(9, 12)][0]["asset"]["path"], "rect.obj")

    def test_yolo_object_fit_table_rejects_less_than_eighty_percent_coverage(self):
        asset = {
            "kind": "object",
            "path": "small.obj",
            "bounds_m": (-5.0, 5.0, -5.0, 5.0),
            "source": "test",
        }
        pools = {BLD.BLD_CLASS_MEDIUM: [asset]}

        table = BLD._build_yolo_object_fit_table(pools)

        self.assertIn((10, 10), table)
        self.assertNotIn((13, 10), table)

    def test_yolo_object_fit_table_uses_centered_dimensions_for_offcenter_bounds(self):
        asset = {
            "kind": "object",
            "path": "offcenter.obj",
            "bounds_m": (-4.0, 6.0, -5.0, 5.0),
            "source": "test",
        }
        pools = {BLD.BLD_CLASS_MEDIUM: [asset]}

        table = BLD._build_yolo_object_fit_table(pools)

        self.assertNotIn((10, 10), table)
        self.assertIn((10, 12), table)
        self.assertIn((12, 10), table)

    def test_yolo_object_selection_places_when_mapping_fits_polygon(self):
        asset = {
            "kind": "object",
            "path": "mapped.obj",
            "bounds_m": (-5.0, 5.0, -5.0, 5.0),
            "mark_bounds_m": (-5.0, 5.0, -5.0, 5.0),
            "source": "test",
        }
        table = BLD._build_yolo_object_fit_table({BLD.BLD_CLASS_MEDIUM: [asset]})
        yolo_poly = BLD._points_from_yolo_heading((30.0, 30.0), 12.0, 10.0, 0.0)
        detection = {
            "length_m": 12.0,
            "width_m": 10.0,
            "area_m2": 120.0,
        }

        selected, status = BLD._select_yolo_object_candidate(
            table,
            detection,
            np.rint(yolo_poly).astype(np.int32),
            30,
            30,
            0.0,
            1.0,
        )

        self.assertEqual(status, "selected")
        self.assertEqual(selected["asset"]["path"], "mapped.obj")
        self.assertIsNotNone(selected["footprint_poly"])

    def test_yolo_object_selection_aligns_local_x_long_axis_to_detection(self):
        asset = {
            "kind": "object",
            "path": "x_long.obj",
            "bounds_m": (-6.0, 6.0, -2.0, 2.0),
            "source": "test",
        }
        table = BLD._build_yolo_object_candidate_index({BLD.BLD_CLASS_MEDIUM: [asset]})
        yolo_heading = 35.0
        yolo_poly = BLD._points_from_yolo_heading((40.0, 40.0), 12.0, 4.0, yolo_heading)
        detection = {
            "length_m": 12.0,
            "width_m": 4.0,
            "area_m2": 48.0,
            "placement_class": BLD.BLD_CLASS_MEDIUM,
        }

        selected, status = BLD._select_yolo_object_candidate(
            table,
            detection,
            np.rint(yolo_poly).astype(np.int32),
            40,
            40,
            yolo_heading,
            1.0,
        )

        self.assertEqual(status, "selected")
        self.assertAlmostEqual(selected["heading"], 305.0)
        self.assertLessEqual(
            BLD._angle_delta_180(
                _poly_long_axis_heading(selected["footprint_poly"]),
                yolo_heading,
            ),
            0.5,
        )

    def test_yolo_object_selection_falls_back_when_no_mapping_exists(self):
        asset = {
            "kind": "object",
            "path": "mapped.obj",
            "bounds_m": (-5.0, 5.0, -5.0, 5.0),
            "source": "test",
        }
        table = BLD._build_yolo_object_fit_table({BLD.BLD_CLASS_MEDIUM: [asset]})
        yolo_poly = BLD._points_from_yolo_heading((30.0, 30.0), 20.0, 20.0, 0.0)
        detection = {
            "length_m": 20.0,
            "width_m": 20.0,
            "area_m2": 400.0,
        }

        selected, status = BLD._select_yolo_object_candidate(
            table,
            detection,
            np.rint(yolo_poly).astype(np.int32),
            30,
            30,
            0.0,
            1.0,
        )

        self.assertIsNone(selected)
        self.assertEqual(status, "miss")

    def test_yolo_object_selection_reports_residential_context_skip(self):
        asset = {
            "kind": "object",
            "path": "simheaven/houses/house_09x12x2.obj",
            "bounds_m": (-4.5, 4.5, -6.0, 6.0),
            "source": "test",
        }
        table = BLD._build_yolo_object_fit_table({BLD.BLD_CLASS_MEDIUM: [asset]})
        yolo_poly = BLD._points_from_yolo_heading((30.0, 30.0), 12.0, 9.0, 0.0)
        detection = {
            "length_m": 12.0,
            "width_m": 9.0,
            "area_m2": 108.0,
        }

        selected, status = BLD._select_yolo_object_candidate(
            table,
            detection,
            np.rint(yolo_poly).astype(np.int32),
            30,
            30,
            0.0,
            1.0,
            residential_context=False,
        )

        self.assertIsNone(selected)
        self.assertEqual(status, "context_skipped")

    def test_scored_yolo_selection_uses_smaller_residential_object_before_facade(self):
        asset = {
            "kind": "object",
            "path": "simheaven/houses/house_10x10x2.obj",
            "bounds_m": (-5.0, 5.0, -5.0, 5.0),
            "source": "test",
            "footprint_class": BLD.BLD_CLASS_SMALL_RESIDENTIAL,
        }
        index = BLD._build_yolo_object_candidate_index({
            BLD.BLD_CLASS_SMALL_RESIDENTIAL: [asset],
        })
        yolo_poly = BLD._points_from_yolo_heading((30.0, 30.0), 16.0, 12.0, 0.0)
        detection = {
            "length_m": 16.0,
            "width_m": 12.0,
            "area_m2": 192.0,
            "placement_class": BLD.BLD_CLASS_SMALL_RESIDENTIAL,
        }

        selected, status = BLD._select_yolo_object_candidate(
            index,
            detection,
            np.rint(yolo_poly).astype(np.int32),
            30,
            30,
            0.0,
            1.0,
        )

        self.assertEqual(status, "selected")
        self.assertEqual(selected["asset"]["path"], asset["path"])

    def test_scored_yolo_selection_handles_non_integer_near_size_match(self):
        asset = {
            "kind": "object",
            "path": "simheaven/houses/house_9p9x7p9.obj",
            "bounds_m": (-4.95, 4.95, -3.95, 3.95),
            "source": "test",
            "footprint_class": BLD.BLD_CLASS_TINY_RESIDENTIAL,
        }
        old_table = BLD._build_yolo_object_fit_table({
            BLD.BLD_CLASS_TINY_RESIDENTIAL: [asset],
        })
        index = BLD._build_yolo_object_candidate_index({
            BLD.BLD_CLASS_TINY_RESIDENTIAL: [asset],
        })
        yolo_poly = BLD._points_from_yolo_heading((30.0, 30.0), 9.95, 7.95, 0.0)
        detection = {
            "length_m": 9.95,
            "width_m": 7.95,
            "area_m2": 9.95 * 7.95,
            "placement_class": BLD.BLD_CLASS_TINY_RESIDENTIAL,
        }

        self.assertNotIn(BLD._yolo_object_dimension_key(detection), old_table)
        selected, status = BLD._select_yolo_object_candidate(
            index,
            detection,
            np.rint(yolo_poly).astype(np.int32),
            30,
            30,
            0.0,
            1.0,
        )

        self.assertEqual(status, "selected")
        self.assertEqual(selected["asset"]["path"], asset["path"])

    def test_scored_yolo_selection_keeps_large_classes_at_strict_coverage(self):
        asset = {
            "kind": "object",
            "path": "commercial_10x10.obj",
            "bounds_m": (-5.0, 5.0, -5.0, 5.0),
            "source": "test",
            "footprint_class": BLD.BLD_CLASS_MEDIUM,
        }
        index = BLD._build_yolo_object_candidate_index({
            BLD.BLD_CLASS_MEDIUM: [asset],
        })
        yolo_poly = BLD._points_from_yolo_heading((30.0, 30.0), 13.0, 10.0, 0.0)
        detection = {
            "length_m": 13.0,
            "width_m": 10.0,
            "area_m2": 130.0,
            "placement_class": BLD.BLD_CLASS_MEDIUM,
        }

        selected, status = BLD._select_yolo_object_candidate(
            index,
            detection,
            np.rint(yolo_poly).astype(np.int32),
            30,
            30,
            0.0,
            1.0,
        )

        self.assertIsNone(selected)
        self.assertEqual(status, "coverage_reject")

    def test_optional_library_house_asset_requires_residential_context(self):
        asset = {
            "kind": "object",
            "path": "opensceneryx/objects/buildings/houses/suburban_1.obj",
        }

        self.assertTrue(BLD._asset_requires_residential_context(asset))

    def test_building_zone_cell_mask_tracks_only_occupied_grid_cells(self):
        bld_zone = np.zeros((8, 8), dtype=np.uint8)
        bld_zone[1, 1] = 1
        bld_zone[6, 7] = 1

        cell_mask = BLD._building_zone_cell_mask(bld_zone, 8, 8, 4)

        self.assertEqual(int(cell_mask.sum()), 2)
        self.assertTrue(cell_mask[0, 0])
        self.assertTrue(cell_mask[3, 3])
        self.assertFalse(cell_mask[0, 3])

    def test_image_heading_grid_can_be_limited_to_building_cells(self):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        cell_mask = np.zeros((4, 4), dtype=bool)
        cell_mask[2, 1] = True
        calls = []

        def fake_edge_hist(_patch, bin_deg=5.0):
            calls.append(_patch.shape)
            hist = np.zeros(int(180 / bin_deg), dtype=np.float32)
            hist[0] = 1.0
            return hist

        with mock.patch.object(BLD, "_cell_edge_hist", side_effect=fake_edge_hist):
            hgrid = BLD._image_heading_grid(img, 8, 8, 4, cell_mask=cell_mask)

        self.assertEqual(len(calls), 1)
        self.assertFalse(np.isnan(hgrid[2, 1]))
        self.assertTrue(np.isnan(hgrid[0, 0]))

    def test_oversized_polygon_rasterization_fills_covered_tile(self):
        poly = [
            (-1.0, -1.0),
            (-1.0, 2.0),
            (2.0, 2.0),
            (2.0, -1.0),
            (-1.0, -1.0),
        ]

        mask = BLD._rasterize_polygons(
            [poly],
            lat_n=1.0,
            lat_s=0.0,
            lon_w=0.0,
            lon_e=1.0,
            img_h=32,
            img_w=32,
        )

        self.assertTrue(np.all(mask == 1))

    def test_oversized_polygon_rasterization_clips_partial_water(self):
        poly = [
            (-1.0, -1.0),
            (-1.0, 0.5),
            (2.0, 0.5),
            (2.0, -1.0),
            (-1.0, -1.0),
        ]

        mask = BLD._rasterize_polygons(
            [poly],
            lat_n=1.0,
            lat_s=0.0,
            lon_w=0.0,
            lon_e=1.0,
            img_h=32,
            img_w=32,
        )

        self.assertTrue(np.all(mask[:, :16] == 1))
        self.assertTrue(np.all(mask[:, 18:] == 0))

    def test_mesh_water_reader_does_not_treat_land_type_as_water(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mesh_path = Path(tmpdir) / "Data+00+000.mesh"
            mesh_path.write_text(
                "\n".join(
                    [
                        "MeshVersionFormatted 1.3",
                        "Dimension 3",
                        "Vertices",
                        "unused",
                        "4",
                        "0.0 0.0 0.0",
                        "1.0 0.0 0.0",
                        "1.0 1.0 0.0",
                        "0.0 1.0 0.0",
                        "Normals",
                        "unused",
                        "unused",
                        "0.0 0.0",
                        "0.0 0.0",
                        "0.0 0.0",
                        "0.0 0.0",
                        "Triangles",
                        "unused",
                        "2",
                        "1 2 3 0",
                        "1 3 4 2",
                    ]
                ),
                encoding="utf-8",
            )

            tris = BLD._read_mesh_water_triangles(str(mesh_path))

        self.assertEqual(tris.shape, (1, 3, 2))
        self.assertTrue(np.allclose(tris[0], [(0.0, 0.0), (1.0, 1.0), (1.0, 0.0)]))

    def test_component_side_heading_counts_touched_sides_not_road_length(self):
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[30:70, 30:70] = 1
        stats = np.zeros((2, 5), dtype=np.int32)
        stats[1, BLD.cv2.CC_STAT_LEFT] = 30
        stats[1, BLD.cv2.CC_STAT_TOP] = 30
        stats[1, BLD.cv2.CC_STAT_WIDTH] = 40
        stats[1, BLD.cv2.CC_STAT_HEIGHT] = 40
        stats[1, BLD.cv2.CC_STAT_AREA] = 1600

        def ll(px, py):
            return (1.0 - py / 100.0, px / 100.0)

        roads = [
            {"pts": [ll(31 + i * 4, 28), ll(33 + i * 4, 28)]}
            for i in range(10)
        ] + [
            {"pts": [ll(28, 20), ll(28, 80)]},
            {"pts": [ll(72, 20), ll(72, 80)]},
        ]

        headings, side_counts = BLD._component_side_touch_headings(
            labels,
            stats,
            np.array([1], dtype=np.int32),
            roads,
            lat_n=1.0,
            lat_s=0.0,
            lon_w=0.0,
            lon_e=1.0,
            img_h=100,
            img_w=100,
            band_px=4,
        )

        self.assertEqual(int(side_counts[1]), 3)
        self.assertLessEqual(min(abs(float(headings[1])), abs(float(headings[1]) - 360.0)), 5.0)

    def test_dead_end_road_contact_can_supply_heading(self):
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[30:70, 30:70] = 1
        stats = np.zeros((2, 5), dtype=np.int32)
        stats[1, BLD.cv2.CC_STAT_LEFT] = 30
        stats[1, BLD.cv2.CC_STAT_TOP] = 30
        stats[1, BLD.cv2.CC_STAT_WIDTH] = 40
        stats[1, BLD.cv2.CC_STAT_HEIGHT] = 40
        stats[1, BLD.cv2.CC_STAT_AREA] = 1600

        def ll(px, py):
            return (1.0 - py / 100.0, px / 100.0)

        headings, contact_counts = BLD._component_side_touch_headings(
            labels,
            stats,
            np.array([1], dtype=np.int32),
            [{"pts": [ll(50, 10), ll(50, 32)]}],
            lat_n=1.0,
            lat_s=0.0,
            lon_w=0.0,
            lon_e=1.0,
            img_h=100,
            img_w=100,
            band_px=4,
        )

        self.assertGreaterEqual(int(contact_counts[1]), 1)
        self.assertLessEqual(min(abs(float(headings[1])), abs(float(headings[1]) - 360.0)), 5.0)

    def test_touching_local_road_beats_nearby_large_road_for_zone_heading(self):
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[30:70, 30:70] = 1
        stats = np.zeros((2, 5), dtype=np.int32)
        stats[1, BLD.cv2.CC_STAT_LEFT] = 30
        stats[1, BLD.cv2.CC_STAT_TOP] = 30
        stats[1, BLD.cv2.CC_STAT_WIDTH] = 40
        stats[1, BLD.cv2.CC_STAT_HEIGHT] = 40
        stats[1, BLD.cv2.CC_STAT_AREA] = 1600

        def ll(px, py):
            return (1.0 - py / 100.0, px / 100.0)

        roads = [
            {"pts": [ll(28, 20), ll(28, 80)]},
            {"pts": [ll(0, 22), ll(99, 22)]},
        ]

        headings, contact_counts = BLD._component_side_touch_headings(
            labels,
            stats,
            np.array([1], dtype=np.int32),
            roads,
            lat_n=1.0,
            lat_s=0.0,
            lon_w=0.0,
            lon_e=1.0,
            img_h=100,
            img_w=100,
            band_px=12,
        )

        self.assertEqual(int(contact_counts[1]), 1)
        self.assertLessEqual(min(abs(float(headings[1])), abs(float(headings[1]) - 360.0)), 5.0)

    def test_road_mask_contact_ignores_non_touching_road_in_heading_band(self):
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[30:70, 30:70] = 1
        stats = np.zeros((2, 5), dtype=np.int32)
        stats[1, BLD.cv2.CC_STAT_LEFT] = 30
        stats[1, BLD.cv2.CC_STAT_TOP] = 30
        stats[1, BLD.cv2.CC_STAT_WIDTH] = 40
        stats[1, BLD.cv2.CC_STAT_HEIGHT] = 40
        stats[1, BLD.cv2.CC_STAT_AREA] = 1600
        road_mask = np.zeros((100, 100), dtype=np.uint8)
        road_mask[20:24, :] = 1
        road_mask[:, 24:31] = 1

        def ll(px, py):
            return (1.0 - py / 100.0, px / 100.0)

        roads = [
            {"pts": [ll(24, 20), ll(24, 80)]},
            {"pts": [ll(0, 22), ll(99, 22)]},
        ]

        headings, contact_counts = BLD._component_side_touch_headings(
            labels,
            stats,
            np.array([1], dtype=np.int32),
            roads,
            lat_n=1.0,
            lat_s=0.0,
            lon_w=0.0,
            lon_e=1.0,
            img_h=100,
            img_w=100,
            band_px=12,
            contact_px=5,
            road_mask=road_mask,
        )

        self.assertEqual(int(contact_counts[1]), 1)
        self.assertLessEqual(min(abs(float(headings[1])), abs(float(headings[1]) - 360.0)), 5.0)

    def test_straight_contact_patches_are_preferred_over_noisy_fragments(self):
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[30:70, 30:70] = 1
        stats = np.zeros((2, 5), dtype=np.int32)
        stats[1, BLD.cv2.CC_STAT_LEFT] = 30
        stats[1, BLD.cv2.CC_STAT_TOP] = 30
        stats[1, BLD.cv2.CC_STAT_WIDTH] = 40
        stats[1, BLD.cv2.CC_STAT_HEIGHT] = 40
        stats[1, BLD.cv2.CC_STAT_AREA] = 1600

        def ll(px, py):
            return (1.0 - py / 100.0, px / 100.0)

        roads = [{"pts": [ll(28, 20), ll(28, 80)]}]
        roads.extend(
            {"pts": [ll(35 + i * 4, 28), ll(36 + i * 4, 31)]}
            for i in range(7)
        )

        headings, contact_counts = BLD._component_side_touch_headings(
            labels,
            stats,
            np.array([1], dtype=np.int32),
            roads,
            lat_n=1.0,
            lat_s=0.0,
            lon_w=0.0,
            lon_e=1.0,
            img_h=100,
            img_w=100,
            band_px=4,
        )

        self.assertGreaterEqual(int(contact_counts[1]), 1)
        self.assertLessEqual(min(abs(float(headings[1])), abs(float(headings[1]) - 360.0)), 5.0)

    def test_simheaven_buildings_supply_zone_heading_after_split(self):
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[20:80, 20:80] = 1
        objects = {
            "lat": np.array([0.70, 0.60, 0.50], dtype=np.float32),
            "lon": np.array([0.30, 0.40, 0.50], dtype=np.float32),
            "heading": np.array([30.0, 32.0, 120.0], dtype=np.float32),
            "w_m": np.array([10.0, 10.0, 20.0], dtype=np.float32),
            "h_m": np.array([20.0, 20.0, 20.0], dtype=np.float32),
        }

        headings, counts = BLD._simheaven_building_zone_headings(
            objects,
            labels,
            np.array([1], dtype=np.int32),
            lat_n=1.0,
            lat_s=0.0,
            lon_w=0.0,
            lon_e=1.0,
            img_h=100,
            img_w=100,
        )

        self.assertEqual(int(counts[1]), 2)
        self.assertLessEqual(abs(float(headings[1]) - 32.5), 5.0)

    def test_asset_regions_use_non_rectangular_boundaries(self):
        self.assertEqual(BLD._asset_region(45.0, -75.0), "north_america_ne")
        self.assertEqual(BLD._asset_region(35.0, -120.0), "north_america_west")
        self.assertEqual(BLD._asset_region(42.0, 12.0), "mediterranean")
        self.assertEqual(BLD._asset_region(10.0, 25.0), "africa")
        self.assertEqual(BLD._asset_region(35.0, -40.0), "generic")

    def test_simheaven_package_region_parser_handles_xworld_and_legacy_names(self):
        cases = {
            "simHeaven_X-World_Europe-6-scenery": "europe",
            "simHeaven_X-World_America-1-vfr": "america",
            "simHeaven_X-World_Americas-6-scenery": "america",
            "simHeaven_X-World_Asia-3-details": "asia",
            "simHeaven_X-World_Africa-6-scenery": "africa",
            "simHeaven_X-World_Australia-Oceania-6-scenery": "australia_oceania",
            "simHeaven_X-World_Antarctica-6-scenery": "antarctica",
            "simHeaven_X-Europe-1-vfr": "europe",
            "simHeaven_X-Asia-6-scenery": "asia",
        }

        for folder_name, expected in cases.items():
            with self.subTest(folder_name=folder_name):
                self.assertEqual(
                    DSF.simheaven_package_region_from_name(folder_name),
                    expected,
                )
        self.assertIsNone(
            DSF.simheaven_package_region_from_name("simHeaven_Vegetation_Library")
        )

    def test_simheaven_package_region_search_respects_active_scenery_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            custom = Path(tmpdir) / "Custom Scenery"
            europe = custom / "simHeaven_X-World_Europe-6-scenery"
            asia = custom / "simHeaven_X-World_Asia-6-scenery"
            _write_tile_dsf(europe)
            _write_tile_dsf(asia)
            (custom / "scenery_packs.ini").write_text(
                "\n".join(
                    [
                        "SCENERY_PACK Custom Scenery/simHeaven_X-World_Europe-6-scenery/",
                        "SCENERY_PACK Custom Scenery/simHeaven_X-World_Asia-6-scenery/",
                    ]
                ),
                encoding="utf-8",
            )

            region, folder_name = DSF.find_simheaven_package_region_for_tile(
                custom, 22, 120
            )

        self.assertEqual(region, "europe")
        self.assertEqual(folder_name, "simHeaven_X-World_Europe-6-scenery")

    def test_simheaven_package_region_search_ignores_disabled_packages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            custom = Path(tmpdir) / "Custom Scenery"
            asia = custom / "simHeaven_X-World_Asia-6-scenery"
            _write_tile_dsf(asia)
            (custom / "scenery_packs.ini").write_text(
                "SCENERY_PACK_DISABLED Custom Scenery/simHeaven_X-World_Asia-6-scenery/\n",
                encoding="utf-8",
            )

            region, folder_name = DSF.find_simheaven_package_region_for_tile(
                custom, 22, 120
            )

        self.assertIsNone(region)
        self.assertIsNone(folder_name)

    def test_simheaven_package_region_search_falls_back_without_scenery_packs_ini(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            custom = Path(tmpdir) / "Custom Scenery"
            asia = custom / "simHeaven_X-World_Asia-6-scenery"
            _write_tile_dsf(asia)

            region, folder_name = DSF.find_simheaven_package_region_for_tile(
                custom, 22, 120
            )

        self.assertEqual(region, "asia")
        self.assertEqual(folder_name, "simHeaven_X-World_Asia-6-scenery")

    def test_simheaven_package_region_wins_for_default_asset_pools(self):
        natural_na = BLD._default_object_catalog_paths(35.0, -120.0)
        europe_package = BLD._default_object_catalog_paths(
            35.0,
            -120.0,
            asset_region=BLD._asset_region(35.0, -120.0, "europe"),
        )

        self.assertIn("/lib/global8/us/feat_Building_50_40_600r40.obj", natural_na)
        self.assertIn("/lib/global8/us/hill_sq_30_30r.obj", europe_package)
        self.assertNotIn(
            "/lib/global8/us/feat_Building_50_40_600r40.obj",
            europe_package,
        )

    def test_simheaven_america_package_refines_generic_lonlat_to_north_america(self):
        self.assertEqual(BLD._asset_region(35.0, -40.0), "generic")
        self.assertEqual(BLD._asset_region(35.0, -40.0, "america"), "north_america")
        paths = BLD._default_object_catalog_paths(
            35.0,
            -40.0,
            asset_region=BLD._asset_region(35.0, -40.0, "america"),
        )

        self.assertIn("/lib/global8/us/feat_Building_50_40_600r40.obj", paths)
        self.assertNotIn("/lib/global8/us/hill_sq_30_30r.obj", paths)

    def test_simheaven_asia_package_refines_generic_lonlat_to_asia(self):
        self.assertEqual(BLD._asset_region(0.0, 80.0), "generic")
        self.assertEqual(BLD._asset_region(0.0, 80.0, "asia"), "asia")
        paths = _paths_for_classes(
            BLD._build_sfd_asset_pools(
                0.0,
                80.0,
                asset_region=BLD._asset_region(0.0, 80.0, "asia"),
            ),
            BLD.BLD_PLACEMENT_CLASSES,
        )

        self.assertIn("SFD_Global/Asia/Suburban_1.obj", paths)
        self.assertNotIn("SFD_Global/Med/Residential/Suburban_1.obj", paths)

    def test_residential_pools_have_expanded_regional_suburban_variety(self):
        cases = (
            (65.0, 20.0, "SFD_Global/Scandinavia/Residential/Suburban_", 8),
            (10.0, 25.0, "SFD_Global/Africa/Residential/Suburban_", 8),
            (42.0, 12.0, "SFD_Global/Med/Residential/Suburban_", 8),
            (45.0, -75.0, "SFD_Global/New_England/Residential/Suburban_", 8),
            (35.0, -120.0, "SFD_Global/US_West_Coast/Suburban_", 8),
            (-20.0, -60.0, "SFD_Global/South_America/Suburban_", 10),
            (35.5, 139.5, "SFD_Global/Asia/Suburban_", 10),
        )

        for lat, lon, prefix, expected_count in cases:
            with self.subTest(prefix=prefix):
                pools = BLD._build_sfd_asset_pools(lat, lon)
                residential_paths = _paths_for_classes(
                    pools,
                    (
                        BLD.BLD_CLASS_TINY_RESIDENTIAL,
                        BLD.BLD_CLASS_SMALL_RESIDENTIAL,
                        BLD.BLD_CLASS_COMPACT_RESIDENTIAL,
                        BLD.BLD_CLASS_MEDIUM,
                        BLD.BLD_CLASS_SMALL_APARTMENT,
                    ),
                )
                regional_suburban = [
                    path for path in residential_paths
                    if path.startswith(prefix)
                ]

                self.assertGreaterEqual(len(regional_suburban), expected_count)

    def test_regional_apartment_like_assets_expand_where_sfd_exports_them(self):
        med_pools = BLD._build_sfd_asset_pools(42.0, 12.0)
        med_apartments = _paths_for_classes(
            med_pools,
            (
                BLD.BLD_CLASS_MEDIUM,
                BLD.BLD_CLASS_SMALL_APARTMENT,
                BLD.BLD_CLASS_APARTMENT_BLOCK,
            ),
        )
        south_america_pools = BLD._build_sfd_asset_pools(-20.0, -60.0)
        south_america_medium = _paths_for_classes(
            south_america_pools,
            (
                BLD.BLD_CLASS_COMPACT_RESIDENTIAL,
                BLD.BLD_CLASS_MEDIUM,
                BLD.BLD_CLASS_SMALL_APARTMENT,
                BLD.BLD_CLASS_APARTMENT_BLOCK,
            ),
        )

        self.assertGreaterEqual(
            len([
                path for path in med_apartments
                if path.startswith("SFD_Global/Med/Residential/Apartment_North_")
            ]),
            8,
        )
        self.assertGreaterEqual(
            len([
                path for path in south_america_medium
                if path.startswith("SFD_Global/South_America/Med_")
            ]),
            8,
        )

    def test_urban_residential_assets_are_included_by_footprint(self):
        pools = BLD._build_sfd_asset_pools(42.0, 12.0)
        residential_and_apartment_paths = _paths_for_classes(
            pools,
            (
                BLD.BLD_CLASS_COMPACT_RESIDENTIAL,
                BLD.BLD_CLASS_MEDIUM,
                BLD.BLD_CLASS_SMALL_APARTMENT,
                BLD.BLD_CLASS_APARTMENT_BLOCK,
            ),
        )
        urban_paths = [
            path for path in residential_and_apartment_paths
            if path.startswith("SFD_Global/Med/Residential/Urban_Mid_")
        ]

        self.assertGreaterEqual(len(urban_paths), 16)
        self.assertIn("SFD_Global/Med/Residential/Urban_Mid_30m.obj", urban_paths)
        self.assertEqual(
            BLD._class_for_footprint(
                BLD._bounds_for_object_path(
                    "SFD_Global/Med/Residential/Urban_Mid_30m.obj"
                )
            ),
            BLD.BLD_CLASS_SMALL_APARTMENT,
        )

    def test_compact_residential_excludes_midrise_row_blocks(self):
        pools = BLD._build_sfd_asset_pools(42.0, 12.0)
        compact_paths = _paths_for_classes(
            pools,
            (BLD.BLD_CLASS_COMPACT_RESIDENTIAL,),
        )
        medium_paths = _paths_for_classes(
            pools,
            (BLD.BLD_CLASS_MEDIUM,),
        )

        self.assertNotIn("SFD_Global/Med/Residential/Urban_Mid_7m.obj", compact_paths)
        self.assertIn("SFD_Global/Med/Residential/Urban_Mid_7m.obj", medium_paths)

    def test_compact_simheaven_residential_stays_one_or_two_floor(self):
        pools = BLD._build_simheaven_asset_pools([], 45.0, 7.0)
        compact_paths = _paths_for_classes(
            pools,
            (BLD.BLD_CLASS_COMPACT_RESIDENTIAL,),
        )
        small_paths = _paths_for_classes(
            pools,
            (BLD.BLD_CLASS_SMALL_RESIDENTIAL,),
        )
        medium_paths = _paths_for_classes(
            pools,
            (BLD.BLD_CLASS_MEDIUM,),
        )

        self.assertIn("simheaven/houses/house_09x12x2.obj", small_paths)
        self.assertIn("simheaven/houses/house_12x15x2.obj", compact_paths)
        self.assertNotIn("simheaven/residential/residential_10x10x3.obj", compact_paths)
        self.assertIn("simheaven/residential/residential_10x10x3.obj", medium_paths)

    def test_fit_selection_does_not_retry_after_selected_asset_fails(self):
        pool = [
            {
                "kind": "object",
                "path": "too-large.obj",
                "bounds_m": (-8.0, 8.0, -8.0, 8.0),
                "fit_bounds_m": (-8.0, 8.0, -8.0, 8.0),
                "mark_bounds_m": (-8.0, 8.0, -8.0, 8.0),
            },
            {
                "kind": "object",
                "path": "small-enough.obj",
                "bounds_m": (-2.0, 2.0, -2.0, 2.0),
                "fit_bounds_m": (-2.0, 2.0, -2.0, 2.0),
                "mark_bounds_m": (-2.0, 2.0, -2.0, 2.0),
            },
        ]
        occ_mask = np.zeros((40, 40), dtype=np.uint8)
        occ_mask[20, 27] = 1
        scratch = np.zeros_like(occ_mask)
        counts = {}

        building_spacing_mask = np.zeros_like(occ_mask)

        asset, final_h, final_poly, spacing_poly, skipped = BLD._find_fitting_asset(
            pool,
            np.random.default_rng(1),
            20,
            20,
            0.0,
            1.0,
            occ_mask,
            building_spacing_mask,
            scratch,
            counts,
        )

        self.assertIsNone(asset)
        self.assertIsNone(final_h)
        self.assertEqual(skipped, 0)
        self.assertEqual(counts["fit_checks"], 2)
        self.assertIsNone(final_poly)
        self.assertIsNone(spacing_poly)

    def test_largest_fit_mode_selects_largest_asset_that_fits(self):
        pool = [
            {
                "kind": "object",
                "path": "small.obj",
                "bounds_m": (-2.0, 2.0, -2.0, 2.0),
                "fit_bounds_m": (-2.0, 2.0, -2.0, 2.0),
                "mark_bounds_m": (-2.0, 2.0, -2.0, 2.0),
                "footprint_area_m2": 16.0,
                "footprint_max_side_m": 4.0,
            },
            {
                "kind": "object",
                "path": "large.obj",
                "bounds_m": (-6.0, 6.0, -4.0, 4.0),
                "fit_bounds_m": (-6.0, 6.0, -4.0, 4.0),
                "mark_bounds_m": (-6.0, 6.0, -4.0, 4.0),
                "footprint_area_m2": 96.0,
                "footprint_max_side_m": 12.0,
            },
        ]
        occ_mask = np.zeros((40, 40), dtype=np.uint8)
        spacing_mask = np.zeros_like(occ_mask)
        counts = {}

        asset, final_h, final_poly, spacing_poly, skipped = BLD._find_fitting_asset(
            pool,
            np.random.default_rng(1),
            20,
            20,
            0.0,
            1.0,
            occ_mask,
            spacing_mask,
            np.zeros_like(occ_mask),
            counts,
            prefer_largest_fit=True,
        )

        self.assertEqual(asset["path"], "large.obj")
        self.assertIsNotNone(final_h)
        self.assertIsNotNone(final_poly)
        self.assertIsNotNone(spacing_poly)
        self.assertEqual(skipped, 0)
        self.assertEqual(counts["largest_fit_asset_selected"], 1)

    def test_static_integral_fit_path_matches_standard_fit_path(self):
        pool = [
            {
                "kind": "object",
                "path": "small.obj",
                "bounds_m": (-2.0, 2.0, -2.0, 2.0),
                "fit_bounds_m": (-6.0, 6.0, -6.0, 6.0),
                "mark_bounds_m": (-10.0, 10.0, -10.0, 10.0),
            },
        ]
        static_occ_mask = np.zeros((50, 50), dtype=np.uint8)
        static_occ_mask[20, 25] = 1
        building_spacing_mask = np.zeros_like(static_occ_mask)

        standard_counts = {}
        standard = BLD._find_fitting_asset(
            pool,
            np.random.default_rng(1),
            20,
            20,
            0.0,
            1.0,
            static_occ_mask,
            building_spacing_mask,
            np.zeros_like(static_occ_mask),
            standard_counts,
        )

        integral_counts = {}
        integral = BLD._find_fitting_asset(
            pool,
            np.random.default_rng(1),
            20,
            20,
            0.0,
            1.0,
            static_occ_mask,
            building_spacing_mask,
            np.zeros_like(static_occ_mask),
            integral_counts,
            static_occ_integral=BLD.cv2.integral(static_occ_mask, sdepth=BLD.cv2.CV_32S),
        )

        self.assertEqual(standard[0]["path"], integral[0]["path"])
        self.assertEqual(standard[1], integral[1])
        np.testing.assert_array_equal(standard[2], integral[2])
        np.testing.assert_array_equal(standard[3], integral[3])
        self.assertEqual(standard[4], integral[4])
        self.assertEqual(standard_counts, integral_counts)

    def test_fit_selection_checks_only_selected_asset_orientations(self):
        pool = [
            {
                "kind": "object",
                "path": "too-large-a.obj",
                "bounds_m": (-8.0, 8.0, -8.0, 8.0),
                "fit_bounds_m": (-8.0, 8.0, -8.0, 8.0),
                "mark_bounds_m": (-8.0, 8.0, -8.0, 8.0),
            },
            {
                "kind": "object",
                "path": "too-large-b.obj",
                "bounds_m": (-8.0, 8.0, -8.0, 8.0),
                "fit_bounds_m": (-8.0, 8.0, -8.0, 8.0),
                "mark_bounds_m": (-8.0, 8.0, -8.0, 8.0),
            },
            {
                "kind": "object",
                "path": "small-enough.obj",
                "bounds_m": (-2.0, 2.0, -2.0, 2.0),
                "fit_bounds_m": (-2.0, 2.0, -2.0, 2.0),
                "mark_bounds_m": (-2.0, 2.0, -2.0, 2.0),
            },
        ]
        occ_mask = np.zeros((40, 40), dtype=np.uint8)
        occ_mask[20, 27] = 1
        building_spacing_mask = np.zeros_like(occ_mask)
        counts = {}

        with mock.patch.object(
            BLD, "_footprint_poly_with_bbox_basis", wraps=BLD._footprint_poly_with_bbox_basis
        ) as footprint_poly:
            asset, final_h, final_poly, spacing_poly, skipped = BLD._find_fitting_asset(
                pool,
                np.random.default_rng(11),
                20,
                20,
                0.0,
                1.0,
                occ_mask,
                building_spacing_mask,
                np.zeros_like(occ_mask),
                counts,
            )

        self.assertIsNone(asset)
        self.assertIsNone(final_h)
        self.assertEqual(skipped, 0)
        self.assertEqual(counts["fit_checks"], 2)
        self.assertIsNone(final_poly)
        self.assertIsNone(spacing_poly)
        self.assertEqual(footprint_poly.call_count, 4)

    def test_static_blockers_use_raw_footprint_not_spacing_pad(self):
        pool = [
            {
                "kind": "object",
                "path": "small.obj",
                "bounds_m": (-2.0, 2.0, -2.0, 2.0),
                "fit_bounds_m": (-6.0, 6.0, -6.0, 6.0),
                "mark_bounds_m": (-10.0, 10.0, -10.0, 10.0),
            },
        ]
        static_occ_mask = np.zeros((50, 50), dtype=np.uint8)
        static_occ_mask[20, 25] = 1
        building_spacing_mask = np.zeros_like(static_occ_mask)
        scratch = np.zeros_like(static_occ_mask)
        counts = {}

        asset, final_h, final_poly, spacing_poly, skipped = BLD._find_fitting_asset(
            pool,
            np.random.default_rng(1),
            20,
            20,
            0.0,
            1.0,
            static_occ_mask,
            building_spacing_mask,
            scratch,
            counts,
        )

        self.assertEqual(asset["path"], "small.obj")
        self.assertEqual(final_h, 0.0)
        self.assertEqual(skipped, 0)
        self.assertIsNotNone(final_poly)
        self.assertIsNotNone(spacing_poly)

    def test_generated_building_spacing_still_uses_padded_footprint(self):
        pool = [
            {
                "kind": "object",
                "path": "small.obj",
                "bounds_m": (-2.0, 2.0, -2.0, 2.0),
                "fit_bounds_m": (-6.0, 6.0, -6.0, 6.0),
                "mark_bounds_m": (-10.0, 10.0, -10.0, 10.0),
            },
        ]
        static_occ_mask = np.zeros((50, 50), dtype=np.uint8)
        building_spacing_mask = np.zeros_like(static_occ_mask)
        building_spacing_mask[20, 25] = 1
        scratch = np.zeros_like(static_occ_mask)
        counts = {}

        asset, final_h, final_poly, spacing_poly, skipped = BLD._find_fitting_asset(
            pool,
            np.random.default_rng(1),
            20,
            20,
            0.0,
            1.0,
            static_occ_mask,
            building_spacing_mask,
            scratch,
            counts,
        )

        self.assertIsNone(asset)
        self.assertIsNone(final_h)
        self.assertIsNone(final_poly)
        self.assertIsNone(spacing_poly)
        self.assertEqual(skipped, 0)

    def test_fit_selection_tries_rotated_asset_before_next_asset(self):
        pool = [
            {
                "kind": "object",
                "path": "rotates-to-fit.obj",
                "bounds_m": (-2.0, 2.0, -8.0, 8.0),
                "fit_bounds_m": (-2.0, 2.0, -8.0, 8.0),
                "mark_bounds_m": (-2.0, 2.0, -8.0, 8.0),
            },
            {
                "kind": "object",
                "path": "fallback.obj",
                "bounds_m": (-1.0, 1.0, -1.0, 1.0),
                "fit_bounds_m": (-1.0, 1.0, -1.0, 1.0),
                "mark_bounds_m": (-1.0, 1.0, -1.0, 1.0),
            },
        ]
        static_occ_mask = np.zeros((50, 50), dtype=np.uint8)
        static_occ_mask[24, 20] = 1
        building_spacing_mask = np.zeros_like(static_occ_mask)
        scratch = np.zeros_like(static_occ_mask)
        counts = {}

        asset, final_h, final_poly, spacing_poly, skipped = BLD._find_fitting_asset(
            pool,
            np.random.default_rng(1),
            20,
            20,
            0.0,
            1.0,
            static_occ_mask,
            building_spacing_mask,
            scratch,
            counts,
        )

        self.assertEqual(asset["path"], "rotates-to-fit.obj")
        self.assertIn(final_h, (90.0, 270.0))
        self.assertIsNotNone(final_poly)
        self.assertIsNotNone(spacing_poly)
        self.assertEqual(skipped, 0)

    def test_nonresidential_context_skips_house_like_assets_not_facades(self):
        pool = [
            {
                "kind": "object",
                "path": "simheaven/houses/house_09x12x2.obj",
                "bounds_m": (-4.5, 4.5, -6.0, 6.0),
                "fit_bounds_m": (-4.5, 4.5, -6.0, 6.0),
                "mark_bounds_m": (-4.5, 4.5, -6.0, 6.0),
            },
            {
                "kind": "facade",
                "path": "lib/buildings/facades/commercial/low_commercial_01.fac",
                "bounds_m": (-5.0, 5.0, -5.0, 5.0),
                "fit_bounds_m": (-5.0, 5.0, -5.0, 5.0),
                "mark_bounds_m": (-5.0, 5.0, -5.0, 5.0),
            },
        ]
        static_occ_mask = np.zeros((40, 40), dtype=np.uint8)
        building_spacing_mask = np.zeros_like(static_occ_mask)
        scratch = np.zeros_like(static_occ_mask)
        counts = {}

        asset, final_h, final_poly, spacing_poly, skipped = BLD._find_fitting_asset(
            pool,
            np.random.default_rng(1),
            20,
            20,
            0.0,
            1.0,
            static_occ_mask,
            building_spacing_mask,
            scratch,
            counts,
            residential_context=False,
        )

        self.assertEqual(asset["kind"], "facade")
        self.assertEqual(counts["residential_asset_skipped"], 1)
        self.assertEqual(skipped, 0)

    def test_nonresidential_retry_context_preserves_selection_and_skip_counts(self):
        pool = [
            {
                "kind": "object",
                "path": "simheaven/houses/house_09x12x2.obj",
                "bounds_m": (-4.5, 4.5, -6.0, 6.0),
                "fit_bounds_m": (-4.5, 4.5, -6.0, 6.0),
                "mark_bounds_m": (-4.5, 4.5, -6.0, 6.0),
            },
            {
                "kind": "facade",
                "path": "too-large.fac",
                "bounds_m": (-8.0, 8.0, -8.0, 8.0),
                "fit_bounds_m": (-8.0, 8.0, -8.0, 8.0),
                "mark_bounds_m": (-8.0, 8.0, -8.0, 8.0),
            },
            {
                "kind": "object",
                "path": "SFD_Global/Asia/Suburban_1.obj",
                "bounds_m": (-5.0, 5.0, -5.0, 5.0),
                "fit_bounds_m": (-5.0, 5.0, -5.0, 5.0),
                "mark_bounds_m": (-5.0, 5.0, -5.0, 5.0),
            },
            {
                "kind": "facade",
                "path": "small.fac",
                "bounds_m": (-2.0, 2.0, -2.0, 2.0),
                "fit_bounds_m": (-2.0, 2.0, -2.0, 2.0),
                "mark_bounds_m": (-2.0, 2.0, -2.0, 2.0),
            },
        ]
        static_occ_mask = np.zeros((50, 50), dtype=np.uint8)
        static_occ_mask[20, 27] = 1
        building_spacing_mask = np.zeros_like(static_occ_mask)

        standard_counts = {}
        standard = BLD._find_fitting_asset(
            pool,
            np.random.default_rng(0),
            20,
            20,
            0.0,
            1.0,
            static_occ_mask,
            building_spacing_mask,
            np.zeros_like(static_occ_mask),
            standard_counts,
            residential_context=False,
        )

        context_counts = {}
        context = BLD._find_fitting_asset(
            pool,
            np.random.default_rng(0),
            20,
            20,
            0.0,
            1.0,
            static_occ_mask,
            building_spacing_mask,
            np.zeros_like(static_occ_mask),
            context_counts,
            residential_context=False,
            retry_context=BLD._asset_retry_context(pool),
        )

        self.assertEqual(standard[0]["path"], context[0]["path"])
        self.assertEqual(standard[1], context[1])
        np.testing.assert_array_equal(standard[2], context[2])
        np.testing.assert_array_equal(standard[3], context[3])
        self.assertEqual(standard[4], context[4])
        self.assertEqual(standard_counts, context_counts)

    def test_asset_retry_order_prefers_smaller_footprints(self):
        pools = {
            cls: []
            for cls in BLD.BLD_PLACEMENT_CLASSES
        }
        pools[BLD.BLD_CLASS_COMPACT_RESIDENTIAL] = [
            {
                "path": "larger.obj",
                "bounds_m": (-5.0, 5.0, -5.0, 5.0),
                "footprint_area_m2": 100.0,
                "footprint_max_side_m": 10.0,
            },
            {
                "path": "smaller.obj",
                "bounds_m": (-2.0, 2.0, -2.0, 2.0),
                "footprint_area_m2": 16.0,
                "footprint_max_side_m": 4.0,
            },
        ]

        BLD._sort_asset_pools_for_retry(pools)

        self.assertEqual(
            [asset["path"] for asset in pools[BLD.BLD_CLASS_COMPACT_RESIDENTIAL]],
            ["smaller.obj", "larger.obj"],
        )

    def test_keep_smallest_asset_per_class_keeps_one_minimal_asset(self):
        pools = {
            cls: []
            for cls in BLD.BLD_PLACEMENT_CLASSES
        }
        pools[BLD.BLD_CLASS_MEDIUM] = [
            {
                "path": "larger.obj",
                "bounds_m": (-5.0, 5.0, -5.0, 5.0),
                "footprint_area_m2": 100.0,
                "footprint_max_side_m": 10.0,
            },
            {
                "path": "smaller.obj",
                "bounds_m": (-2.0, 2.0, -2.0, 2.0),
                "footprint_area_m2": 16.0,
                "footprint_max_side_m": 4.0,
            },
        ]

        BLD._keep_smallest_asset_per_class(pools)

        self.assertEqual(
            [asset["path"] for asset in pools[BLD.BLD_CLASS_MEDIUM]],
            ["smaller.obj"],
        )

    def test_class_min_footprint_span_uses_smallest_raw_asset_span(self):
        pools = {
            cls: []
            for cls in BLD.BLD_PLACEMENT_CLASSES
        }
        pools[BLD.BLD_CLASS_COMPACT_RESIDENTIAL] = [
            {
                "path": "wider.obj",
                "bounds_m": (-5.0, 5.0, -3.0, 3.0),
            },
            {
                "path": "narrower.obj",
                "bounds_m": (-2.0, 2.0, -4.0, 4.0),
            },
        ]

        min_span = BLD._class_min_footprint_span_m(pools)

        self.assertEqual(min_span[BLD.BLD_CLASS_COMPACT_RESIDENTIAL], 8.0)

    def test_mark_pad_for_edge_spacing_accounts_for_fit_pad(self):
        self.assertEqual(BLD._mark_pad_for_edge_spacing_m(20.0), 16.0)

    def test_asset_retry_sequence_wraps_from_random_offset(self):
        pool = [
            {"path": "a.obj"},
            {"path": "b.obj"},
            {"path": "c.obj"},
            {"path": "d.obj"},
        ]

        sequence = list(BLD._asset_retry_sequence(pool, np.random.default_rng(1)))

        self.assertEqual(
            [asset["path"] for asset in sequence],
            ["b.obj", "c.obj", "d.obj", "a.obj"],
        )

    def test_class_min_fit_inradius_uses_smallest_available_asset(self):
        pools = {
            cls: []
            for cls in BLD.BLD_PLACEMENT_CLASSES
        }
        pools[BLD.BLD_CLASS_COMPACT_RESIDENTIAL] = [
            {
                "path": "larger.obj",
                "fit_bounds_m": (-7.0, 7.0, -5.0, 5.0),
            },
            {
                "path": "smaller.obj",
                "fit_bounds_m": (-3.0, 3.0, -2.0, 2.0),
            },
        ]

        min_radius = BLD._class_min_fit_inradius_m(pools)

        self.assertEqual(min_radius[BLD.BLD_CLASS_COMPACT_RESIDENTIAL], 2.0)

    def test_dynamic_center_blocker_marks_future_impossible_centers(self):
        masks = {
            cls: np.zeros((50, 50), dtype=np.uint8)
            for cls in BLD.BLD_PLACEMENT_CLASSES
        }
        min_radius = {
            cls: 0.0
            for cls in BLD.BLD_PLACEMENT_CLASSES
        }
        min_radius[BLD.BLD_CLASS_COMPACT_RESIDENTIAL] = 3.0

        BLD._mark_dynamic_center_blockers(
            masks,
            25,
            25,
            0.0,
            (-2.0, 2.0, -2.0, 2.0),
            1.0,
            min_radius,
        )

        self.assertEqual(masks[BLD.BLD_CLASS_COMPACT_RESIDENTIAL][25, 25], 1)
        self.assertEqual(masks[BLD.BLD_CLASS_COMPACT_RESIDENTIAL][25, 29], 1)
        self.assertEqual(masks[BLD.BLD_CLASS_COMPACT_RESIDENTIAL][25, 32], 0)
        self.assertEqual(masks[BLD.BLD_CLASS_MEDIUM][25, 25], 0)

    def test_leftover_gap_candidates_are_component_capped(self):
        leftover = np.zeros((30, 30), dtype=np.uint8)
        leftover[4:14, 4:14] = 1
        leftover[18:28, 18:28] = 1
        zone_class = np.zeros_like(leftover, dtype=np.uint8)
        zone_class[leftover != 0] = BLD.BLD_CLASS_MEDIUM

        gap_x, gap_y, gap_cls, _, n_components, n_dropped = BLD._leftover_gap_candidates(
            leftover,
            zone_class,
            max_candidates=3,
            rng=np.random.default_rng(1),
            max_per_component=2,
        )

        self.assertEqual(n_components, 2)
        self.assertEqual(gap_x.size, 3)
        self.assertEqual(gap_y.size, 3)
        self.assertGreaterEqual(n_dropped, 1)
        self.assertTrue(np.all(gap_cls == BLD.BLD_CLASS_MEDIUM))

    def test_apartment_pool_height_gate(self):
        pools = BLD._build_sfd_asset_pools(35.5, 139.5)
        apartment_paths = _paths_for_classes(
            pools,
            (
                BLD.BLD_CLASS_SMALL_APARTMENT,
                BLD.BLD_CLASS_APARTMENT_BLOCK,
            ),
        )

        # 30 m apartments sit inside the 40 m pool gate since the HeightNet
        # ceiling removal (height-fit selection selects them only for
        # accordingly tall predictions).
        self.assertIn("SFD_Global/Buildings/Apartment_30m_1.obj", apartment_paths)
        self.assertIn("SFD_Global/Buildings/Apartment_30m_2.obj", apartment_paths)
        self.assertFalse(
            BLD._is_very_tall_building_asset(
                "SFD_Global/Buildings/Apartment_30m_2.obj",
                BLD._object_estimated_height_m("SFD_Global/Buildings/Apartment_30m_2.obj"),
            )
        )
        # Above the gate (or skyscraper-named) assets stay out of the pools.
        self.assertTrue(
            BLD._is_very_tall_building_asset(
                "SFD_Global/Buildings/Apartment_45m_1.obj",
                BLD.MAX_GENERATED_BUILDING_HEIGHT_M + 5.0,
            )
        )
        self.assertTrue(
            BLD._is_very_tall_building_asset("some/lib/office_tower_1.obj")
        )

    def test_tiny_fillers_are_excluded_from_building_pools(self):
        asia_paths = _paths_for_classes(
            BLD._build_sfd_asset_pools(35.5, 139.5),
            (BLD.BLD_CLASS_COMPACT_RESIDENTIAL,),
        )
        simheaven_paths = _paths_for_classes(
            BLD._build_simheaven_asset_pools(tile_lat=35.5, tile_lon=139.5),
            BLD.BLD_PLACEMENT_CLASSES,
        )

        self.assertNotIn("SFD_Global/Asia/Carport_1.obj", asia_paths)
        self.assertNotIn("SFD_Global/Asia/Carport_2.obj", asia_paths)
        self.assertNotIn("SFD_Global/Asia/Shed_1.obj", asia_paths)
        self.assertNotIn("simheaven/houses/house_05x05x1.obj", simheaven_paths)
        self.assertNotIn("simheaven/sheds/shed_02x03x1.obj", simheaven_paths)

    def test_small_accessory_building_classes_are_included_when_reasonable(self):
        north_america_paths = _paths_for_classes(
            BLD._build_sfd_asset_pools(45.0, -75.0),
            BLD.BLD_PLACEMENT_CLASSES,
        )
        australia_paths = _paths_for_classes(
            BLD._build_sfd_asset_pools(-33.0, 151.0),
            BLD.BLD_PLACEMENT_CLASSES,
        )

        self.assertIn("SFD_Global/New_England/Residential/Garage.obj", north_america_paths)
        self.assertIn("SFD_Global/Australia/Shed.obj", australia_paths)
        self.assertNotIn("SFD_Global/Australia/Carport.obj", australia_paths)

    def test_default_assets_include_facade_and_rectangular_object_variety(self):
        pools = BLD._build_default_asset_pools(45.0, -75.0)
        medium_paths = _paths_for_classes(pools, (BLD.BLD_CLASS_MEDIUM,))
        apartment_paths = _paths_for_classes(pools, (BLD.BLD_CLASS_APARTMENT_BLOCK,))
        large_paths = _paths_for_classes(pools, (BLD.BLD_CLASS_LARGE,))

        self.assertIn("lib/buildings/facades/generic/mid_modern_05.fac", medium_paths)
        self.assertIn("lib/buildings/facades/commercial/low_commercial_08.fac", medium_paths)
        self.assertIn("lib/buildings/facades/generic/high_glass_03.fac", apartment_paths)
        self.assertIn("lib/buildings/facades/generic/high_metallic_01.fac", apartment_paths)
        self.assertIn("lib/buildings/facades/generic/high_modern_07.fac", apartment_paths)
        self.assertIn("lib/buildings/facades/generic/high_universal_02.fac", apartment_paths)
        self.assertIn(
            "lib/buildings/facades/industrial/warehouse_07_90x40.fac",
            large_paths,
        )
        self.assertNotIn("/lib/global8/us/feat_Building_50_40_600r40.obj", large_paths)
        self.assertGreater(BLD._default_object_height_m("/lib/global8/us/feat_Building_50_40_600r40.obj"), 24.0)
        self.assertEqual(
            BLD._class_for_footprint(BLD._bounds_from_dimensions(50.0, 40.0)),
            BLD.BLD_CLASS_LARGE,
        )

    def test_context_facade_pools_are_non_empty_facades(self):
        for key, variants in BLD.CONTEXT_FACADE_VARIANTS.items():
            self.assertIn(key[0], BLD.BLD_PLACEMENT_CLASSES)
            self.assertIn(key[1], {
                BLD._SF_BARELAND,
                BLD._SF_RANGELAND,
                BLD._SF_DEVELOPED,
                BLD._SF_ROAD,
                BLD._SF_TREE,
                BLD._SF_WATER,
                BLD._SF_AGRICULTURE,
            })
            self.assertTrue(variants, f"{key} must have at least one variant")
            self.assertTrue(
                all(path.endswith(".fac") for path in variants),
                f"{key} contains a non-facade path",
            )

    def test_facade_picker_filters_simheaven_when_unavailable(self):
        veg_map = np.full((64, 64), BLD._SF_DEVELOPED, dtype=np.uint8)

        paths = {
            BLD._facade_for_detection(
                BLD.BLD_CLASS_APARTMENT_BLOCK,
                veg_map,
                jx,
                32,
                1.0,
                lat=45.0,
                lon=7.0,
                include_simheaven_assets=False,
            )
            for jx in range(8, 56)
        }

        self.assertTrue(paths)
        self.assertFalse(any(path.startswith("simheaven/") for path in paths))

    def test_facade_picker_can_use_expanded_simheaven_groups(self):
        veg_map = np.full((64, 64), BLD._SF_DEVELOPED, dtype=np.uint8)
        expanded_groups = {
            "simheaven/facades/bld-high-res.fac",
            "simheaven/facades/bld-high-com.fac",
            "simheaven/facades/retail.fac",
            "simheaven/facades/hotel.fac",
            "simheaven/facades/school.fac",
            "simheaven/facades/college.fac",
            "simheaven/facades/university.fac",
        }

        paths = {
            BLD._facade_for_detection(
                BLD.BLD_CLASS_APARTMENT_BLOCK,
                veg_map,
                jx,
                jy,
                1.0,
                lat=45.0,
                lon=7.0,
                include_simheaven_assets=True,
            )
            for jy in range(8, 56)
            for jx in range(8, 56)
        }

        self.assertTrue(paths & expanded_groups)

    def test_facade_picker_is_deterministic_for_same_detection(self):
        veg_map = np.full((64, 64), BLD._SF_ROAD, dtype=np.uint8)
        kwargs = dict(
            facade_cls=BLD.BLD_CLASS_MEDIUM,
            veg_map=veg_map,
            jx=23,
            jy=41,
            m_per_px=1.0,
            lat=45.123456,
            lon=7.654321,
            include_simheaven_assets=True,
        )

        self.assertEqual(
            BLD._facade_for_detection(**kwargs),
            BLD._facade_for_detection(**kwargs),
        )

    def test_default_object_catalog_keeps_regional_context(self):
        north_america = _paths_for_classes(
            BLD._build_default_asset_pools(45.0, -75.0),
            BLD.BLD_PLACEMENT_CLASSES,
        )
        europe = _paths_for_classes(
            BLD._build_default_asset_pools(45.0, 7.0),
            BLD.BLD_PLACEMENT_CLASSES,
        )
        asia = _paths_for_classes(
            BLD._build_default_asset_pools(35.5, 139.5),
            BLD.BLD_PLACEMENT_CLASSES,
        )

        self.assertNotIn("/lib/global8/us/feat_Building_50_40_600r40.obj", north_america)
        self.assertNotIn("/lib/global8/us/feat_Building_50_40_600r40.obj", europe)
        self.assertIn("/lib/global8/us/hill_sq_30_30r.obj", europe)
        self.assertNotIn("/lib/global8/us/hill_sq_30_30r.obj", asia)

    def test_simheaven_catalog_expands_all_footprint_classes(self):
        pools = BLD._build_simheaven_asset_pools([], 45.0, 7.0)
        small_paths = _paths_for_classes(pools, (BLD.BLD_CLASS_SMALL_RESIDENTIAL,))
        compact_paths = _paths_for_classes(pools, (BLD.BLD_CLASS_COMPACT_RESIDENTIAL,))
        medium_paths = _paths_for_classes(pools, (BLD.BLD_CLASS_MEDIUM,))
        apartment_paths = _paths_for_classes(
            pools,
            (
                BLD.BLD_CLASS_SMALL_APARTMENT,
                BLD.BLD_CLASS_APARTMENT_BLOCK,
            ),
        )
        large_paths = _paths_for_classes(pools, (BLD.BLD_CLASS_LARGE,))

        self.assertIn("simheaven/houses/house_09x12x1.obj", small_paths)
        self.assertIn("simheaven/houses/house_12x15x2.obj", compact_paths)
        self.assertIn("simheaven/residential/residential_15x20x4.obj", medium_paths)
        self.assertIn("simheaven/residential/residential_20x30x3.obj", apartment_paths)
        self.assertIn("simheaven/industrial/industrial_30x60.obj", large_paths)

    def test_simheaven_catalog_keeps_regional_context(self):
        asia_paths = _paths_for_classes(
            BLD._build_simheaven_asset_pools([], 35.5, 139.5),
            BLD.BLD_PLACEMENT_CLASSES,
        )
        europe_paths = _paths_for_classes(
            BLD._build_simheaven_asset_pools([], 45.0, 7.0),
            BLD.BLD_PLACEMENT_CLASSES,
        )

        self.assertIn("simheaven/houses/house_09x12x1.obj", asia_paths)
        self.assertIn("simheaven/commercial/commercial_18x42.obj", asia_paths)
        self.assertIn("simheaven/industrial/industrial_30x60.obj", asia_paths)
        self.assertIn("simheaven/industrial/industrial_30x60.obj", europe_paths)

    def test_simheaven_special_landmarks_are_not_reused_as_assets(self):
        special_objects = [
            {
                "path": "simheaven/landmarks/church_20x30.obj",
                "w_m": 20.0,
                "h_m": 30.0,
            },
            {
                "path": "simheaven/landmarks/cathedral_40x60.obj",
                "w_m": 40.0,
                "h_m": 60.0,
            },
            {
                "path": "simheaven/commercial/school_30x40.obj",
                "w_m": 30.0,
                "h_m": 40.0,
            },
            {
                "path": "simheaven/commercial/petrol_18x18.obj",
                "w_m": 18.0,
                "h_m": 18.0,
            },
            {
                "path": "simheaven/commercial/supermarket_30x24.obj",
                "w_m": 30.0,
                "h_m": 24.0,
            },
            {
                "path": "simheaven/commercial/hospital_30x40.obj",
                "w_m": 30.0,
                "h_m": 40.0,
            },
            {
                "path": "simheaven/commercial/townhall_20x20.obj",
                "w_m": 20.0,
                "h_m": 20.0,
            },
            {
                "path": "simheaven/commercial/bank_15x20.obj",
                "w_m": 15.0,
                "h_m": 20.0,
            },
            {
                "path": "simheaven/residential/residential_15x20x4.obj",
                "w_m": 15.0,
                "h_m": 20.0,
            },
        ]
        paths = _paths_for_classes(
            BLD._build_simheaven_asset_pools(special_objects, 45.0, 7.0),
            BLD.BLD_PLACEMENT_CLASSES,
        )

        self.assertTrue(
            BLD._is_simheaven_building_object("simheaven/landmarks/church_20x30.obj")
        )
        self.assertNotIn("simheaven/landmarks/church_20x30.obj", paths)
        self.assertNotIn("simheaven/landmarks/cathedral_40x60.obj", paths)
        self.assertNotIn("simheaven/commercial/school_30x40.obj", paths)
        self.assertNotIn("simheaven/commercial/petrol_18x18.obj", paths)
        self.assertNotIn("simheaven/commercial/supermarket_30x24.obj", paths)
        self.assertNotIn("simheaven/commercial/hospital_30x40.obj", paths)
        self.assertNotIn("simheaven/commercial/townhall_20x20.obj", paths)
        self.assertNotIn("simheaven/commercial/bank_15x20.obj", paths)
        self.assertIn("simheaven/residential/residential_15x20x4.obj", paths)

        catalog_paths = _paths_for_classes(
            BLD._build_simheaven_asset_pools([], 45.0, 7.0),
            BLD.BLD_PLACEMENT_CLASSES,
        )
        self.assertNotIn("simheaven/commercial/petrol_18x18.obj", catalog_paths)
        self.assertNotIn("simheaven/commercial/supermarket_30x24.obj", catalog_paths)

    def test_long_slab_and_industrial_assets_stay_out_of_residential_pools(self):
        pools = BLD._build_sfd_asset_pools(35.5, 139.5)
        residential_and_apartment_paths = _paths_for_classes(
            pools,
            (
                BLD.BLD_CLASS_COMPACT_RESIDENTIAL,
                BLD.BLD_CLASS_MEDIUM,
                BLD.BLD_CLASS_SMALL_APARTMENT,
                BLD.BLD_CLASS_APARTMENT_BLOCK,
            ),
        )

        self.assertNotIn("SFD_Global/Asia/Apartment_2.obj", residential_and_apartment_paths)
        self.assertTrue(
            all("/Industry_" not in path for path in residential_and_apartment_paths)
        )
        self.assertEqual(
            BLD._class_for_footprint(
                BLD._bounds_for_object_path("SFD_Global/Asia/Apartment_2.obj")
            ),
            BLD.BLD_CLASS_LARGE,
        )

    def test_sfd_catalog_is_a_strict_allowlist(self):
        pools = BLD._build_sfd_asset_pools(3.0, 102.0)
        all_paths = _paths_for_classes(pools, BLD.BLD_PLACEMENT_CLASSES)

        self.assertNotIn("SFD_Global/Asia/Suburban_South_Test.obj", all_paths)
        self.assertNotIn("SFD_Global/Asia/blank.obj", all_paths)
        self.assertNotIn("SFD_Global/Asia/Store.obj", all_paths)
        self.assertNotIn("SFD_Global/Asia/Sign_Drugstore.obj", all_paths)
        self.assertNotIn("SFD_Global/Australia/Pool.obj", all_paths)
        self.assertNotIn("SFD_Global/Australia/Trampoline.obj", all_paths)
        self.assertIsNone(
            BLD._bounds_for_object_path("SFD_Global/Asia/Suburban_South_Test.obj")
        )

    def test_simheaven_export_discovery_filters_and_deduplicates_assets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            custom = Path(tmpdir) / "Custom Scenery"
            package = _write_library_package(
                custom,
                "simHeaven_X-World_Africa-6-scenery",
                [
                    "EXPORT simheaven/houses/house_04x06x1.obj objects/a.obj",
                    "EXPORT_BACKUP simheaven/houses/house_04x06x1.obj objects/b.obj",
                    "EXPORT simheaven/houses/house_05x05x1.obj objects/missing.obj",
                    "EXPORT simheaven/houses/house_09x12x3.obj objects/c.obj",
                    "EXPORT simheaven/residential/residential_20x20x8.obj objects/tall.obj",
                    "EXPORT simheaven/residential/residential_20x20x14.obj objects/tall14.obj",
                    "EXPORT simheaven/sheds/shed_02x03x1.obj objects/shed.obj",
                    "EXPORT simheaven/commercial/school_30x40.obj objects/school.obj",
                ],
            )
            _activate_packages(custom, package)

            pools = BLD._build_simheaven_asset_pools(
                [],
                0.0,
                0.0,
                "africa",
                custom_scenery_dir=custom,
            )
            paths = _paths_for_classes(pools, BLD.BLD_PLACEMENT_CLASSES)

        self.assertIn("simheaven/houses/house_04x06x1.obj", paths)
        self.assertIn("simheaven/houses/house_09x12x3.obj", paths)
        self.assertNotIn("simheaven/houses/house_05x05x1.obj", paths)
        # 8 floors (25.6 m) sits inside the 40 m pool gate since the
        # HeightNet ceiling removal; 14 floors (44.8 m) stays out.
        self.assertIn("simheaven/residential/residential_20x20x8.obj", paths)
        self.assertNotIn("simheaven/residential/residential_20x20x14.obj", paths)
        self.assertNotIn("simheaven/sheds/shed_02x03x1.obj", paths)
        self.assertNotIn("simheaven/commercial/school_30x40.obj", paths)
        self.assertEqual(
            sum(
                1
                for pool in pools.values()
                for asset in pool
                if asset["path"] == "simheaven/houses/house_04x06x1.obj"
            ),
            1,
        )

    def test_simheaven_pool_excludes_non_exported_dsf_local_objects(self):
        export = type("Export", (), {
            "virtual_path": "simheaven/houses/house_04x06x1.obj",
            "resolved_path": None,
        })()
        excluded_export = type("Export", (), {
            "virtual_path": "simheaven/houses/house_05x05x1.obj",
            "resolved_path": None,
        })()
        simheaven_objects = [
            {
                "path": "simheaven/houses/house_05x05x1.obj",
                "w_m": 5.0,
                "h_m": 5.0,
            },
            {
                "path": "simheaven/houses/house_05x06x1.obj",
                "w_m": 5.0,
                "h_m": 6.0,
            },
            {
                "path": "simheaven/houses/house_04x06x1.obj",
                "w_m": 4.0,
                "h_m": 6.0,
            },
        ]

        pools = BLD._build_simheaven_asset_pools(
            simheaven_objects,
            0.0,
            0.0,
            "africa",
            library_exports=[export, excluded_export],
        )
        paths = _paths_for_classes(pools, BLD.BLD_PLACEMENT_CLASSES)

        self.assertIn("simheaven/houses/house_04x06x1.obj", paths)
        self.assertNotIn("simheaven/houses/house_05x05x1.obj", paths)
        self.assertNotIn("simheaven/houses/house_05x06x1.obj", paths)

    def test_unexported_simheaven_object_placements_are_filtered(self):
        placements = [
            (120.1, 22.1, 0.0, "simheaven/houses/house_04x06x1.obj"),
            (120.2, 22.2, 0.0, "simheaven/houses/house_05x06x1.obj"),
            (120.3, 22.3, 0.0, "o4sfr/asia/residential/house_4.3x3.5x1.obj"),
        ]

        filtered, dropped = BLD._filter_unexported_simheaven_object_placements(
            placements,
            {"simheaven/houses/house_04x06x1.obj"},
        )

        self.assertEqual(dropped, 1)
        self.assertEqual(
            [placement[3] for placement in filtered],
            [
                "simheaven/houses/house_04x06x1.obj",
                "o4sfr/asia/residential/house_4.3x3.5x1.obj",
            ],
        )

    def test_unexported_simheaven_object_placements_fail_loudly(self):
        placements = [
            (120.1, 22.1, 0.0, "simheaven/houses/house_05x06x1.obj"),
            (120.2, 22.2, 0.0, "o4sfr/asia/residential/house_4.3x3.5x1.obj"),
        ]

        with self.assertRaisesRegex(RuntimeError, "house_05x06x1.obj"):
            BLD._assert_no_unexported_simheaven_object_placements(
                placements,
                {"simheaven/houses/house_04x06x1.obj"},
            )

    def test_exported_simheaven_landmark_object_is_valid_output_ref(self):
        export = type("Export", (), {
            "virtual_path": "simheaven/landmarks/gantry-crane.obj",
            "resolved_path": None,
        })()
        placements = [
            (120.1, 22.1, 0.0, "simheaven/landmarks/gantry-crane.obj"),
        ]

        exported_paths = BLD._exported_simheaven_object_paths([export])

        self.assertIn("simheaven/landmarks/gantry-crane.obj", exported_paths)
        BLD._assert_no_unexported_simheaven_object_placements(
            placements,
            exported_paths,
        )

    def test_stock_yolo_asset_filter_disables_unexported_gantry_crane(self):
        asset_map, dropped_paths, dropped_classes = BLD._filter_stock_yolo_asset_map(
            {
                7: (
                    "object",
                    ("simheaven/landmarks/gantry-crane.obj",),
                    None,
                ),
                14: (
                    "object",
                    (
                        "lib/garden/pools/pool_Small_7x10.obj",
                        "SFD_Global/Australia/Pool.obj",
                    ),
                    None,
                ),
            },
            exported_paths=set(),
        )

        self.assertNotIn(7, asset_map)
        self.assertIn(7, dropped_classes)
        self.assertIn("simheaven/landmarks/gantry-crane.obj", dropped_paths)
        self.assertEqual(
            asset_map[14][1],
            ("lib/garden/pools/pool_Small_7x10.obj",),
        )

    def test_sfd_export_measurement_adds_region_asset_with_offcenter_bounds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            custom = Path(tmpdir) / "Custom Scenery"
            package = _write_library_package(
                custom,
                "SFD Global Autogen",
                [
                    "EXPORT SFD_Global/Asia/Industry_30x70.obj Asia/Industry_30x70.obj",
                    "EXPORT SFD_Global/Asia/Gas_Station.obj Asia/Gas_Station.obj",
                ],
            )
            _write_obj8(
                package / "Asia" / "Industry_30x70.obj",
                [(-12.0, 0.0, -35.0), (18.0, 0.0, -35.0), (18.0, 0.0, 35.0), (-12.0, 0.0, 35.0)],
            )
            _write_obj8(
                package / "Asia" / "Gas_Station.obj",
                [(-8.0, 0.0, -8.0), (8.0, 0.0, 8.0)],
            )
            _activate_packages(custom, package)

            pools = BLD._build_sfd_asset_pools(
                35.0,
                139.0,
                "asia",
                custom_scenery_dir=custom,
            )
            assets = [
                asset
                for pool in pools.values()
                for asset in pool
                if asset["path"] == "SFD_Global/Asia/Industry_30x70.obj"
            ]
            paths = _paths_for_classes(pools, BLD.BLD_PLACEMENT_CLASSES)

        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0]["bounds_m"], (-12.0, 18.0, -35.0, 35.0))
        self.assertIn("SFD_Global/Asia/Industry_30x70.obj", paths)
        self.assertNotIn("SFD_Global/Asia/Gas_Station.obj", paths)

    def test_sfd_export_discovery_keeps_audited_exclusions_out(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            custom = Path(tmpdir) / "Custom Scenery"
            package = _write_library_package(
                custom,
                "SFD Global Autogen",
                [
                    "EXPORT SFD_Global/Asia/Apartment_2.obj Asia/Apartment_2.obj",
                ],
            )
            _write_obj8(
                package / "Asia" / "Apartment_2.obj",
                [
                    (-30.0, 0.0, -5.185),
                    (30.0, 0.0, -5.185),
                    (30.0, 0.0, 5.185),
                    (-30.0, 0.0, 5.185),
                ],
            )
            _activate_packages(custom, package)

            pools = BLD._build_sfd_asset_pools(
                35.0,
                139.0,
                "asia",
                custom_scenery_dir=custom,
            )
            paths = _paths_for_classes(pools, BLD.BLD_PLACEMENT_CLASSES)

        self.assertNotIn("SFD_Global/Asia/Apartment_2.obj", paths)

    def test_optional_library_classifier_maps_known_regional_assets(self):
        self.assertEqual(
            BLD._optional_library_asset_regions(
                "objects/houses/EU/GB/RES_05x11_2_UK_1.obj",
                "world-models",
            ),
            ("europe",),
        )
        self.assertEqual(
            BLD._optional_library_asset_regions(
                "objects/commercial/EU/med/commercial_20.0x10.0_18_lugano.obj",
                "world-models",
            ),
            ("mediterranean",),
        )
        self.assertEqual(
            BLD._optional_library_asset_regions(
                "objects/houses/US/AZ/RES_10x12_US.obj",
                "world-models",
            ),
            ("north_america",),
        )
        self.assertEqual(
            BLD._optional_library_asset_regions(
                "objects/houses/NZ/RES_10.00x10.00_NZ.obj",
                "world-models",
            ),
            ("australia_oceania",),
        )
        self.assertEqual(
            BLD._optional_library_asset_regions(
                "CDB-Library/buildings/samoa/house_samoa1.obj",
                "cdb-library",
            ),
            ("australia_oceania",),
        )
        self.assertEqual(
            BLD._optional_library_asset_regions(
                "CDB-Library/buildings/houses/papua_house1.obj",
                "cdb-library",
            ),
            ("australia_oceania",),
        )
        self.assertEqual(
            BLD._optional_library_asset_regions(
                "CDB-Library/buildings/houses/hihifo_house1.obj",
                "cdb-library",
            ),
            ("australia_oceania",),
        )

    def test_optional_library_rejects_unclassified_and_special_landmarks(self):
        self.assertEqual(
            BLD._optional_library_rejection_reason(
                "opensceneryx/objects/buildings/industrial/wind_turbines/1.obj",
                "opensceneryx",
                "europe",
            ),
            "special-landmark",
        )
        self.assertEqual(
            BLD._optional_library_rejection_reason(
                "R2_Library/industrial/elektrarny/velektrarna_100m.obj",
                "r2-library",
                "europe",
            ),
            "special-landmark",
        )
        # An opensceneryx path with no entry in the visual-triage overrides
        # file and no per-library hardcoded rule stays unclassified.
        # ("commercial/hotels/" is not yet triaged; "offices/brick/" used to
        # be unclassified but is now tagged generic via the override file.)
        self.assertEqual(
            BLD._optional_library_rejection_reason(
                "opensceneryx/objects/buildings/commercial/hotels/1.obj",
                "opensceneryx",
                "europe",
            ),
            "unclassified",
        )
        # The override file promotes opensceneryx wooden houses to europe.
        self.assertIsNone(
            BLD._optional_library_rejection_reason(
                "opensceneryx/objects/buildings/residential/houses/wooden/3.obj",
                "opensceneryx",
                "europe",
            ),
        )
        self.assertEqual(
            BLD._optional_library_rejection_reason(
                "objects/houses/US/AZ/RES_10x12_US.obj",
                "world-models",
                "europe",
            ),
            "region-mismatch",
        )
        self.assertIsNone(
            BLD._optional_library_rejection_reason(
                "objects/houses/US/AZ/RES_10x12_US.obj",
                "world-models",
                "north_america_west",
            )
        )

    def test_optional_library_policy_controls_regional_measured_assets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            custom = Path(tmpdir) / "Custom Scenery"
            osx_package = _write_library_package(
                custom,
                "OpenSceneryX",
                [
                    "EXPORT opensceneryx/objects/buildings/industrial/warehouse_1.obj objects/warehouse.obj",
                    "EXPORT opensceneryx/objects/buildings/industrial/wind_turbines/1.obj objects/wind.obj",
                    "EXPORT opensceneryx/objects/buildings/industrial/chimney.obj objects/chimney.obj",
                    "EXPORT opensceneryx/objects/buildings/industrial/silo_1.obj objects/silo.obj",
                    "EXPORT opensceneryx/objects/buildings/industrial/storage_tank.obj objects/tank.obj",
                    "EXPORT opensceneryx/objects/buildings/marine/lighthouses/1.obj objects/lighthouse.obj",
                ],
            )
            world_package = _write_library_package(
                custom,
                "world-models",
                [
                    "EXPORT objects/houses/EU/GB/RES_05x11_2_UK_1.obj objects/eu_house.obj",
                    "EXPORT objects/houses/US/AZ/RES_10x12_US.obj objects/us_house.obj",
                    "EXPORT objects/houses/NZ/RES_10.00x10.00_NZ.obj objects/nz_house.obj",
                ],
            )
            cdb_package = _write_library_package(
                custom,
                "CDB-Library",
                [
                    "EXPORT CDB-Library/buildings/samoa/house_samoa1.obj objects/samoa.obj",
                    "EXPORT CDB-Library/buildings/houses/building_house01.obj objects/generic_house.obj",
                ],
            )
            r2_package = _write_library_package(
                custom,
                "R2_Library",
                [
                    "EXPORT R2_Library/industrial/elektrarny/velektrarna_100m.obj objects/wind_power.obj",
                    "EXPORT R2_Library/industrial/kominy/komin_B30.obj objects/komin.obj",
                    "EXPORT R2_Library/industrial/elektrarny/chladici_vez_A95.obj objects/chladici_vez.obj",
                    "EXPORT R2_Library/industrial/elektrarny/reaktor.obj objects/reaktor.obj",
                    "EXPORT R2_Library/industrial/nadrze/nadrz88m.obj objects/nadrz.obj",
                ],
            )
            _write_obj8(
                osx_package / "objects" / "warehouse.obj",
                [(-10.0, 0.0, -15.0), (10.0, 0.0, -15.0), (10.0, 0.0, 15.0), (-10.0, 0.0, 15.0)],
            )
            _write_obj8(
                osx_package / "objects" / "wind.obj",
                [(-10.0, 0.0, -15.0), (10.0, 0.0, -15.0), (10.0, 0.0, 15.0), (-10.0, 0.0, 15.0)],
            )
            _write_obj8(
                osx_package / "objects" / "chimney.obj",
                [(-3.0, 0.0, -3.0), (3.0, 0.0, -3.0), (3.0, 0.0, 3.0), (-3.0, 0.0, 3.0)],
            )
            _write_obj8(
                osx_package / "objects" / "silo.obj",
                [(-4.0, 0.0, -4.0), (4.0, 0.0, -4.0), (4.0, 0.0, 4.0), (-4.0, 0.0, 4.0)],
            )
            _write_obj8(
                osx_package / "objects" / "tank.obj",
                [(-6.0, 0.0, -6.0), (6.0, 0.0, -6.0), (6.0, 0.0, 6.0), (-6.0, 0.0, 6.0)],
            )
            _write_obj8(
                osx_package / "objects" / "lighthouse.obj",
                [(-2.0, 0.0, -2.0), (2.0, 0.0, 2.0)],
            )
            for package_dir, name in (
                (world_package, "eu_house"),
                (world_package, "us_house"),
                (world_package, "nz_house"),
                (cdb_package, "samoa"),
                (cdb_package, "generic_house"),
                (r2_package, "wind_power"),
            ):
                _write_obj8(
                    package_dir / "objects" / f"{name}.obj",
                    [(-5.0, 0.0, -6.0), (5.0, 0.0, -6.0), (5.0, 0.0, 6.0), (-5.0, 0.0, 6.0)],
                )
            for name in ("komin", "chladici_vez", "reaktor", "nadrz"):
                _write_obj8(
                    r2_package / "objects" / f"{name}.obj",
                    [(-5.0, 0.0, -5.0), (5.0, 0.0, -5.0), (5.0, 0.0, 5.0), (-5.0, 0.0, 5.0)],
                )
            _activate_packages(custom, osx_package, world_package, cdb_package, r2_package)

            with mock.patch.dict(os.environ, {"O4_SFR_BLD_EXTRA_LIBRARIES": "auto"}):
                europe_pools = BLD._build_optional_library_asset_pools(
                    custom_scenery_dir=custom,
                    asset_region="europe",
                )
                north_america_pools = BLD._build_optional_library_asset_pools(
                    custom_scenery_dir=custom,
                    asset_region="north_america",
                )
                oceania_pools = BLD._build_optional_library_asset_pools(
                    custom_scenery_dir=custom,
                    asset_region="australia_oceania",
                )
            with mock.patch.dict(os.environ, {"O4_SFR_BLD_EXTRA_LIBRARIES": "off"}):
                off_pools = BLD._build_optional_library_asset_pools(
                    custom_scenery_dir=custom,
                    asset_region="europe",
                )

        europe_paths = _paths_for_classes(europe_pools, BLD.BLD_PLACEMENT_CLASSES)
        north_america_paths = _paths_for_classes(north_america_pools, BLD.BLD_PLACEMENT_CLASSES)
        oceania_paths = _paths_for_classes(oceania_pools, BLD.BLD_PLACEMENT_CLASSES)
        off_paths = _paths_for_classes(off_pools, BLD.BLD_PLACEMENT_CLASSES)
        self.assertIn(
            "objects/houses/EU/GB/RES_05x11_2_UK_1.obj",
            europe_paths,
        )
        self.assertNotIn(
            "objects/houses/US/AZ/RES_10x12_US.obj",
            europe_paths,
        )
        self.assertIn(
            "objects/houses/US/AZ/RES_10x12_US.obj",
            north_america_paths,
        )
        self.assertIn(
            "objects/houses/NZ/RES_10.00x10.00_NZ.obj",
            oceania_paths,
        )
        self.assertIn(
            "CDB-Library/buildings/samoa/house_samoa1.obj",
            oceania_paths,
        )
        # building_house01 was previously unclassified; after Track A visual
        # triage it is tagged "generic" -> usable on every tile (with lower
        # priority than a region-locked match).  Its presence on the oceania
        # pool is now expected.
        self.assertIn(
            "CDB-Library/buildings/houses/building_house01.obj",
            oceania_paths,
        )
        self.assertNotIn(
            "opensceneryx/objects/buildings/industrial/warehouse_1.obj",
            europe_paths,
        )
        self.assertNotIn(
            "opensceneryx/objects/buildings/industrial/wind_turbines/1.obj",
            europe_paths,
        )
        self.assertNotIn(
            "opensceneryx/objects/buildings/marine/lighthouses/1.obj",
            europe_paths,
        )
        self.assertNotIn(
            "opensceneryx/objects/buildings/industrial/chimney.obj",
            europe_paths,
        )
        self.assertNotIn(
            "opensceneryx/objects/buildings/industrial/silo_1.obj",
            europe_paths,
        )
        self.assertNotIn(
            "opensceneryx/objects/buildings/industrial/storage_tank.obj",
            europe_paths,
        )
        self.assertNotIn(
            "R2_Library/industrial/elektrarny/velektrarna_100m.obj",
            europe_paths,
        )
        self.assertNotIn(
            "R2_Library/industrial/kominy/komin_B30.obj",
            europe_paths,
        )
        self.assertNotIn(
            "R2_Library/industrial/elektrarny/chladici_vez_A95.obj",
            europe_paths,
        )
        self.assertNotIn(
            "R2_Library/industrial/elektrarny/reaktor.obj",
            europe_paths,
        )
        self.assertNotIn(
            "R2_Library/industrial/nadrze/nadrz88m.obj",
            europe_paths,
        )
        self.assertEqual(off_paths, set())

    def test_optional_library_diagnostics_reports_scan_and_pool_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            custom = Path(tmpdir) / "Custom Scenery"
            package = _write_library_package(
                custom,
                "world-models",
                [
                    "EXPORT objects/houses/EU/GB/RES_05x11_2_UK_1.obj objects/eu_house.obj",
                    "EXPORT objects/houses/US/AZ/RES_10x12_US.obj objects/us_house.obj",
                ],
            )
            _write_obj8(
                package / "objects" / "eu_house.obj",
                [(-5.0, 0.0, -6.0), (5.0, 0.0, -6.0), (5.0, 0.0, 6.0), (-5.0, 0.0, 6.0)],
            )
            _write_obj8(
                package / "objects" / "us_house.obj",
                [(-5.0, 0.0, -6.0), (5.0, 0.0, -6.0), (5.0, 0.0, 6.0), (-5.0, 0.0, 6.0)],
            )
            _activate_packages(custom, package)

            enabled = ("world-models",)
            exports = BLD._scan_runtime_library_exports(
                custom,
                extra_library_ids=enabled,
            )
            pools = BLD._build_optional_library_asset_pools(
                custom_scenery_dir=custom,
                library_exports=exports,
                enabled_library_ids=enabled,
                asset_region="europe",
            )
            diagnostics = BLD._describe_optional_library_diagnostics(
                exports,
                enabled,
                "europe",
                pools,
            )

        self.assertIn("enabled=world-models", diagnostics)
        self.assertIn("exports=2", diagnostics)
        self.assertIn("candidates=2", diagnostics)
        self.assertIn("accepted=1", diagnostics)
        self.assertIn("world-models:1", diagnostics)
        self.assertIn("region-mismatch:1", diagnostics)

    def test_asset_inventory_dry_run_reports_exports_without_mutating(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            custom = Path(tmpdir) / "Custom Scenery"
            package = _write_library_package(
                custom,
                "simHeaven_X-World_Africa-6-scenery",
                ["EXPORT simheaven/houses/house_04x06x1.obj objects/a.obj"],
            )
            _activate_packages(custom, package)

            before = sorted(path.relative_to(custom) for path in custom.rglob("*"))
            exports = ASSETINV.scan_library_exports(
                custom_scenery_dir=custom,
                package_name_patterns=("simheaven",),
                suffixes=(".obj",),
            )
            after = sorted(path.relative_to(custom) for path in custom.rglob("*"))

        self.assertEqual(before, after)
        self.assertEqual(len(exports), 1)
        self.assertEqual(exports[0].virtual_path, "simheaven/houses/house_04x06x1.obj")


class RoadExclusionTests(unittest.TestCase):
    """Metre-accurate per-class road exclusion of placed building footprints."""

    # A 100x100 px tile spanning a tiny lat/lon box; 1 m/px keeps the
    # metre->pixel maths readable.
    BOUNDS = dict(lat_n=0.001, lat_s=0.0, lon_w=0.0, lon_e=0.001,
                  img_h=100, img_w=100)

    def _way(self, road_type, y_frac=0.5):
        lat = self.BOUNDS["lat_n"] * (1.0 - y_frac)
        return {
            "pts": [(lat, 0.0), (lat, self.BOUNDS["lon_e"])],
            "type": road_type,
        }

    def _mask_for(self, roads, rails=(), m_per_px=1.0):
        return BLD._rasterize_route_exclusion(
            roads, list(rails), m_per_px=m_per_px, **self.BOUNDS
        )

    def _band_height(self, mask):
        rows = np.flatnonzero(mask.any(axis=1))
        return 0 if rows.size == 0 else int(rows[-1] - rows[0] + 1)

    def test_widths_follow_highway_class(self):
        residential = self._mask_for([self._way("residential")])
        motorway = self._mask_for([self._way("motorway")])
        self.assertLessEqual(
            self._band_height(residential),
            round(BLD.ROAD_EXCLUSION_WIDTH_M["residential"]) + 1,
        )
        self.assertGreater(
            self._band_height(motorway), self._band_height(residential)
        )

    def test_no_coarse_pixel_floor_on_narrow_roads(self):
        # At 2.4 m/px (~ZL16) a service alley must stay a 1 px sliver, not the
        # old >=6 px lattice band.
        mask = self._mask_for([self._way("service")], m_per_px=2.4)
        self.assertEqual(self._band_height(mask), 1)

    def test_unlisted_types_never_block(self):
        for road_type in ("footway", "path", "cycleway", "network", None):
            self.assertIsNone(
                self._mask_for([self._way(road_type)]), msg=str(road_type)
            )

    def test_rails_always_block(self):
        mask = self._mask_for([], rails=[self._way("network")])
        self.assertIsNotNone(mask)
        self.assertGreater(int(mask.sum()), 0)

    def test_poly_mask_overlap_frac(self):
        occ = np.zeros((64, 64), dtype=np.uint8)
        occ[:, 30:32] = 1  # 2 px vertical road strip
        poly = np.array([[20, 20], [40, 20], [40, 40], [20, 40]], dtype=np.int32)
        frac = BLD._poly_mask_overlap_frac(occ, poly)
        # 2 of the 21 columns covered by the footprint sit on the strip.
        self.assertAlmostEqual(frac, 2 / 21, delta=0.02)
        self.assertEqual(BLD._poly_mask_overlap_frac(np.zeros_like(occ), poly), 0.0)

    def test_edge_clip_passes_straddle_fails_facade_tolerance(self):
        occ = np.zeros((64, 64), dtype=np.uint8)
        occ[:, 30:32] = 1
        edge_clip = np.array([[10, 10], [31, 10], [31, 30], [10, 30]], dtype=np.int32)
        straddling = np.array([[26, 10], [36, 10], [36, 30], [26, 30]], dtype=np.int32)
        self.assertLessEqual(
            BLD._poly_mask_overlap_frac(occ, edge_clip),
            BLD.DIRECT_FACADE_ROAD_OVERLAP_FRAC,
        )
        self.assertGreater(
            BLD._poly_mask_overlap_frac(occ, straddling),
            BLD.DIRECT_FACADE_ROAD_OVERLAP_FRAC,
        )


if __name__ == "__main__":
    unittest.main()
