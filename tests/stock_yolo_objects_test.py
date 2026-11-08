import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import torch
except ImportError:  # pragma: no cover - exercised only in minimal envs
    torch = None

import O4_Stock_Yolo_Objects as STOCK


class StockYoloAssetMapTests(unittest.TestCase):
    def test_default_batch_constant_is_positive(self):
        self.assertGreaterEqual(STOCK.DEFAULT_STOCK_YOLO_BATCH, 1)

    def test_static_dota_classes_excludes_moving_and_already_rendered(self):
        # Aircraft (X-Plane parks its own), harbour (asset mismatch), DOTA's
        # vehicle classes (vehicles come from DIOR) and already-rendered
        # bridge / roundabout must never be placed. Ships (1) are placed, but
        # only when moored (see the ship rule tests).
        forbidden = {0, 7, 8, 9, 10, 11, 12}
        self.assertFalse(
            forbidden.intersection(STOCK.STATIC_DOTA_CLASSES),
            f"STATIC_DOTA_CLASSES includes forbidden classes: "
            f"{forbidden.intersection(STOCK.STATIC_DOTA_CLASSES)}",
        )

    def test_asset_map_covers_every_static_class(self):
        for cls in STOCK.STATIC_DOTA_CLASSES:
            self.assertIn(
                cls, STOCK.STOCK_YOLO_ASSET_MAP,
                f"DOTA class {cls} ({STOCK.DOTA_CLASS_NAMES.get(cls)}) "
                f"in STATIC_DOTA_CLASSES but missing from STOCK_YOLO_ASSET_MAP",
            )

    def test_asset_map_uses_only_object_or_facade(self):
        # Draped .pol polygons are not allowed: the project disabled ground
        # polygons globally and the disable should not regress here.
        for cls, (placement_type, paths, _h) in STOCK.STOCK_YOLO_ASSET_MAP.items():
            self.assertIn(
                placement_type, ("object", "facade"),
                f"DOTA class {cls}: unsupported placement_type "
                f"{placement_type!r} (only 'object' / 'facade' allowed)",
            )
            self.assertIsInstance(paths, tuple,
                                  f"DOTA class {cls}: asset_paths must be a tuple")
            self.assertTrue(paths, f"DOTA class {cls}: empty asset_paths")
            for p in paths:
                self.assertNotIn(
                    ".pol", p,
                    f"DOTA class {cls}: draped polygon asset {p!r} disallowed",
                )

    def test_asset_paths_resolve_to_known_library_namespaces(self):
        # Every asset path must start with a known library prefix so the sim
        # has a chance of resolving it. Catches typos like "simhaven/..." or
        # forgotten "lib/" prefixes.
        allowed_prefixes = (
            "simheaven/",         # simHeaven X-World
            "lib/",               # X-Plane 12 default
            "SFD_Global/",        # SFD Global
            # Curated extra libraries (see CURATED_EXTRA_BUILDING_LIBRARIES
            # in O4_Building_Overlay): the building overlay keeps these
            # only when the installed library set actually exports them.
            "handyobjects/",      # The Handy Objects Library
            "opensceneryx/",      # OpenSceneryX
            "MisterX_Library/",   # MisterX Library
            "objects/",           # world-models
            "CDB-Library/",       # CDB-Library
        )
        for cls, (_kind, paths, _h) in STOCK.STOCK_YOLO_ASSET_MAP.items():
            for p in paths:
                self.assertTrue(
                    p.startswith(allowed_prefixes),
                    f"DOTA class {cls}: asset {p!r} has unknown library prefix",
                )

    def test_size_limits_defined_for_every_static_class(self):
        for cls in STOCK.STATIC_DOTA_CLASSES:
            self.assertIn(cls, STOCK._MIN_LONG_SIDE_M,
                          f"_MIN_LONG_SIDE_M missing class {cls}")
            self.assertIn(cls, STOCK._MAX_LONG_SIDE_M,
                          f"_MAX_LONG_SIDE_M missing class {cls}")
            self.assertLess(
                STOCK._MIN_LONG_SIDE_M[cls],
                STOCK._MAX_LONG_SIDE_M[cls],
                f"Class {cls}: min long-side >= max long-side",
            )

    def test_heading_flag_defined_for_every_static_class(self):
        for cls in STOCK.STATIC_DOTA_CLASSES:
            self.assertIn(cls, STOCK._USE_OBB_HEADING_PER_CLASS,
                          f"_USE_OBB_HEADING_PER_CLASS missing class {cls}")
            self.assertIsInstance(STOCK._USE_OBB_HEADING_PER_CLASS[cls], bool)

    def test_no_sports_class_is_ever_placed(self):
        # Sports fields were removed from the stock-YOLO pass entirely: the
        # available assets are fixed-size models that never matched the pitch
        # already painted in the ortho.
        sports_classes = {3, 4, 5, 6, 13}
        self.assertFalse(
            sports_classes.intersection(STOCK.STATIC_DOTA_CLASSES),
            "STATIC_DOTA_CLASSES still detects sports classes: "
            f"{sorted(sports_classes.intersection(STOCK.STATIC_DOTA_CLASSES))}",
        )
        self.assertFalse(
            sports_classes.intersection(STOCK.STOCK_YOLO_ASSET_MAP),
            "STOCK_YOLO_ASSET_MAP still maps sports classes: "
            f"{sorted(sports_classes.intersection(STOCK.STOCK_YOLO_ASSET_MAP))}",
        )

    def test_no_sports_assets_remain_in_the_map(self):
        sports_tokens = (
            "grandstand", "stadium", "sports", "cancha", "deportes",
            "baseball", "soccer", "tennis", "basketball", "football",
        )
        for cls, (_kind, paths, _h) in STOCK.STOCK_YOLO_ASSET_MAP.items():
            for path in paths:
                lowered = path.lower()
                for token in sports_tokens:
                    self.assertNotIn(
                        token, lowered,
                        f"DOTA class {cls}: {path!r} is a sports asset",
                    )

    def test_facade_entries_carry_positive_default_height(self):
        for cls, (placement_type, _paths, height_m) in STOCK.STOCK_YOLO_ASSET_MAP.items():
            if placement_type == "facade":
                self.assertIsNotNone(
                    height_m, f"Class {cls}: facade entry needs default_height_m"
                )
                self.assertGreater(height_m, 0.0,
                                   f"Class {cls}: facade height must be > 0")


class StockYoloVariantPickerTests(unittest.TestCase):
    def test_single_entry_pool_returns_that_entry(self):
        self.assertEqual(
            STOCK._pick_variant(("only.fac",), 22.5, 120.5, 100, 200),
            "only.fac",
        )

    def test_empty_pool_raises(self):
        with self.assertRaises(ValueError):
            STOCK._pick_variant((), 0.0, 0.0, 0, 0)

    def test_picker_is_deterministic_per_location(self):
        pool = ("a.fac", "b.fac", "c.fac", "d.fac")
        first = STOCK._pick_variant(pool, 22.123, 120.456, 1500, 2400)
        for _ in range(10):
            self.assertEqual(
                first,
                STOCK._pick_variant(pool, 22.123, 120.456, 1500, 2400),
                "picker should be stable across repeated calls with same key",
            )

    def test_picker_spreads_across_pool_for_distinct_locations(self):
        pool = ("a.fac", "b.fac", "c.fac", "d.fac")
        picks = set()
        # Sweep a grid of locations large enough to reasonably hit every entry.
        for jx in range(0, 4000, 200):
            for jy in range(0, 4000, 200):
                picks.add(STOCK._pick_variant(pool, 22.0, 120.0, jx, jy))
                if len(picks) == len(pool):
                    break
            if len(picks) == len(pool):
                break
        self.assertEqual(picks, set(pool), "picker should cover the whole pool")

    def test_picker_keys_on_pixel_coordinates(self):
        # Different pixel coordinates within the same tile must be able to
        # land on different assets — otherwise asset variety collapses to
        # one-per-tile.
        pool = ("a.fac", "b.fac", "c.fac", "d.fac")
        seen = {
            STOCK._pick_variant(pool, 22.0, 120.0, jx, jy)
            for jx in (50, 1500, 3000)
            for jy in (50, 1500, 3000)
        }
        self.assertGreater(len(seen), 1)


class EnsureStockYoloCheckpointTests(unittest.TestCase):
    def test_returns_existing_path_without_calling_downloader(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "yolo26x-obb.pt"
            existing.write_bytes(b"stub")
            with mock.patch(
                "ultralytics.utils.downloads.attempt_download_asset"
            ) as dl:
                result = STOCK.ensure_stock_yolo_checkpoint(str(existing))
            self.assertEqual(Path(result), existing)
            dl.assert_not_called()

    def test_invokes_ultralytics_download_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "subdir" / "yolo26x-obb.pt"

            def fake_download(path, *args, **kwargs):
                # Ultralytics writes to the requested path on success.
                p = Path(str(path))
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"downloaded")
                return str(p)

            with mock.patch(
                "ultralytics.utils.downloads.attempt_download_asset",
                side_effect=fake_download,
            ) as dl:
                result = STOCK.ensure_stock_yolo_checkpoint(str(target))

            dl.assert_called_once()
            self.assertEqual(Path(result), target)
            self.assertTrue(target.exists())

    def test_raises_when_download_silently_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "yolo26x-obb.pt"
            # Simulate Ultralytics returning a path but not actually
            # producing the file (network/HTTP edge case).
            with mock.patch(
                "ultralytics.utils.downloads.attempt_download_asset",
                return_value=str(target),
            ):
                with self.assertRaises(FileNotFoundError):
                    STOCK.ensure_stock_yolo_checkpoint(str(target))

    def test_packaged_build_keeps_checkpoint_with_user_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            internal = Path(tmp) / "_internal"
            (internal / "overlay_scripts").mkdir(parents=True)
            (internal / "Ortho4XP_Data").mkdir()
            self.assertEqual(
                Path(STOCK._default_stock_yolo_checkpoint(str(internal / "overlay_scripts"))),
                internal / "Ortho4XP_Data" / "yolo26x-obb.pt",
            )

    def test_source_checkout_keeps_checkpoint_in_repository_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "Ortho4XP_Data").mkdir()
            repo = Path(tmp) / "repo"
            repo.mkdir()
            self.assertEqual(
                Path(STOCK._default_stock_yolo_checkpoint(str(repo))),
                repo / "yolo26x-obb.pt",
            )

    def test_moves_checkpoint_left_in_overlay_scripts(self):
        with tempfile.TemporaryDirectory() as tmp:
            scripts = Path(tmp) / "overlay_scripts"
            scripts.mkdir()
            (scripts / "yolo26x-obb.pt").write_bytes(b"old")
            target = Path(tmp) / "Ortho4XP_Data" / "yolo26x-obb.pt"
            with mock.patch.object(STOCK, "_MODULE_ROOT", str(scripts)), mock.patch(
                "ultralytics.utils.downloads.attempt_download_asset"
            ) as dl:
                result = STOCK.ensure_stock_yolo_checkpoint(str(target))
            dl.assert_not_called()
            self.assertEqual(Path(result), target)
            self.assertEqual(target.read_bytes(), b"old")
            self.assertFalse((scripts / "yolo26x-obb.pt").exists())


@unittest.skipIf(torch is None, "torch is not installed")
class StockYoloFacadeGeometryTests(unittest.TestCase):
    def _run_fake_detection(self, cls, m_per_px=1.0, **kwargs):
        class FakeObb:
            def __init__(self, cls):
                self.xyxyxyxy = torch.tensor(
                    [[[100.0, 100.0], [140.0, 100.0], [140.0, 120.0], [100.0, 120.0]]]
                )
                self.conf = torch.tensor([0.9])
                self.cls = torch.tensor([float(cls)])

        class FakeResult:
            def __init__(self, cls):
                self.obb = FakeObb(cls)

        class FakeYolo:
            def __init__(self, cls):
                self.cls = cls

            def predict(self, **_kwargs):
                return iter((FakeResult(self.cls),))

        return STOCK.run_stock_yolo_pass(
            np.zeros((256, 256, 3), dtype=np.uint8),
            model=FakeYolo(cls),
            img_w=256,
            img_h=256,
            lat=0,
            lon=0,
            lat_n=1.0,
            lat_s=0.0,
            lon_w=0.0,
            lon_e=1.0,
            m_per_px=m_per_px,
            stride=256,
            imgsz=256,
            device="cpu",
            **kwargs,
        )

    def test_storage_tank_facade_uses_closed_circular_ring(self):
        result = self._run_fake_detection(2)

        self.assertEqual(len(result.placed_facades), 1)
        ring, path, _height_m = result.placed_facades[0]
        self.assertEqual(path, "simheaven/facades/tank.fac")
        self.assertEqual(len(ring), STOCK.STORAGE_TANK_CIRCLE_SEGMENTS + 1)
        self.assertEqual(ring[0], ring[-1])
        self.assertGreater(STOCK._signed_lonlat_ring_area(ring), 0.0)

        center_lon = 120.0 / 256.0
        center_lat = 1.0 - 110.0 / 256.0
        distances = [
            ((lon - center_lon) ** 2 + (lat - center_lat) ** 2) ** 0.5
            for lon, lat in ring[:-1]
        ]
        self.assertAlmostEqual(max(distances), min(distances), places=6)

    def test_storage_tank_occupancy_uses_same_circular_polygon(self):
        result = self._run_fake_detection(2)

        self.assertEqual(len(result.occupied_px_polys), 1)
        occupied = result.occupied_px_polys[0]
        self.assertEqual(occupied.shape[0], STOCK.STORAGE_TANK_CIRCLE_SEGMENTS + 1)
        self.assertTrue(np.array_equal(occupied[0], occupied[-1]))

    def test_sports_detections_place_nothing(self):
        # Sports classes are no longer detected or mapped, so a detection at
        # any scale must leave the results (and the occupancy masks that feed
        # the building pass) untouched.
        for cls, m_per_px in ((3, 2.35), (4, 0.6), (5, 0.6), (6, 3.5), (13, 3.63)):
            with self.subTest(dota_class=cls):
                result = self._run_fake_detection(cls, m_per_px=m_per_px)
                self.assertEqual(len(result.placed_objects), 0)
                self.assertEqual(len(result.placed_facades), 0)
                self.assertEqual(len(result.occupied_px_polys), 0)
                self.assertEqual(result.counts_by_class, {})

    def test_rect_clip_keeps_border_crossing_obb_rectangular(self):
        # Per-corner clamping sheared boxes that hang off the texture border
        # into irregular quads, and the facade ring is written straight from
        # these corners.
        rot = math.radians(35.0)
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        long_axis = np.array([cos_r, sin_r])
        short_axis = np.array([-sin_r, cos_r])
        center = np.array([14.0, 60.0])
        corners = np.array([
            center + s * 26.0 * long_axis + t * 11.0 * short_axis
            for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ])

        clipped = STOCK._rect_clip_obb_to_image(corners, 256, 256)

        self.assertIsNotNone(clipped)
        clipped = np.asarray(clipped, dtype=np.float64)
        self.assertGreaterEqual(clipped.min(), -1e-4)
        self.assertLessEqual(clipped.max(), 255.0 + 1e-4)
        for idx in range(4):
            a = clipped[(idx - 1) % 4] - clipped[idx]
            b = clipped[(idx + 1) % 4] - clipped[idx]
            cos_ang = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            self.assertAlmostEqual(cos_ang, 0.0, places=5)

    def test_storage_tank_radius_shrinks_instead_of_flattening(self):
        # A tank overlapping the texture border must stay round: clamping the
        # circle's vertices would flatten the crossing arc into a straight edge.
        radius = STOCK._circle_radius_within_image(20.0, 128.0, 50.0, 256, 256)
        self.assertAlmostEqual(radius, 20.0)

        ring = STOCK._pixel_circle_polygon(20.0, 128.0, radius)
        self.assertGreaterEqual(ring.min(), -1e-4)
        self.assertLessEqual(ring.max(), 255.0 + 1e-4)
        distances = np.linalg.norm(
            np.asarray(ring[:-1], dtype=np.float64) - np.array([20.0, 128.0]),
            axis=1,
        )
        self.assertAlmostEqual(float(distances.max()), float(distances.min()), places=4)

    def test_storage_tank_radius_untouched_when_circle_fits(self):
        self.assertAlmostEqual(
            STOCK._circle_radius_within_image(128.0, 128.0, 40.0, 256, 256), 40.0
        )

    def test_rect_clip_leaves_in_bounds_obb_untouched(self):
        corners = np.array([
            [100.0, 100.0], [140.0, 100.0], [140.0, 120.0], [100.0, 120.0],
        ], dtype=np.float32)
        np.testing.assert_allclose(
            STOCK._rect_clip_obb_to_image(corners, 256, 256), corners
        )

    def test_custom_asset_map_can_disable_detected_class(self):
        result = self._run_fake_detection(
            7,
            asset_map={},
            static_classes=(),
        )

        self.assertEqual(result.placed_objects, [])
        self.assertEqual(result.placed_facades, [])
        self.assertEqual(result.counts_by_class, {})


@unittest.skipIf(torch is None, "torch is not installed")
class StockYoloBatchingTests(unittest.TestCase):
    def test_batched_pass_matches_legacy_offsets_and_placements(self):
        class FakeObb:
            def __init__(self):
                self.xyxyxyxy = torch.tensor(
                    [[[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]]]
                )
                self.conf = torch.tensor([0.9])
                self.cls = torch.tensor([2.0])

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

        image = np.zeros((1024, 2048, 3), dtype=np.uint8)
        common = dict(
            img_w=2048,
            img_h=1024,
            lat=22,
            lon=120,
            lat_n=23.0,
            lat_s=22.0,
            lon_w=120.0,
            lon_e=121.0,
            m_per_px=1.0,
            stride=1024,
            imgsz=1024,
            device="cpu",
        )

        legacy_model = FakeYolo()
        batched_model = FakeYolo()
        legacy = STOCK.run_stock_yolo_pass(
            image, model=legacy_model, batch_size=1, **common
        )
        batched = STOCK.run_stock_yolo_pass(
            image, model=batched_model, batch_size=2, **common
        )

        self.assertEqual(legacy.placed_objects, batched.placed_objects)
        self.assertEqual(legacy.placed_facades, batched.placed_facades)
        self.assertEqual(legacy.counts_by_class, batched.counts_by_class)
        self.assertEqual(
            [poly.tolist() for poly in legacy.occupied_px_polys],
            [poly.tolist() for poly in batched.occupied_px_polys],
        )
        self.assertEqual(len(legacy_model.calls), 2)
        self.assertEqual(len(batched_model.calls), 1)
        self.assertEqual(batched_model.calls[0]["batch"], 2)


@unittest.skipIf(torch is None, "torch is not installed")
class StockYoloOomFallbackTests(unittest.TestCase):
    class _FakeObb:
        def __init__(self):
            self.xyxyxyxy = torch.tensor(
                [[[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]]]
            )
            self.conf = torch.tensor([0.9])
            self.cls = torch.tensor([2.0])

    class _FakeResult:
        def __init__(self):
            self.obb = StockYoloOomFallbackTests._FakeObb()

    class _OomOnBatchYolo:
        """Yields one detection per crop at batch=1; OOMs on batched calls."""

        def __init__(self):
            self.calls = []

        def predict(self, **kwargs):
            self.calls.append(kwargs)
            source = kwargs["source"]
            if isinstance(source, list):
                # Yield one result, then blow up mid-stream so partial
                # results exist when the OOM hits.
                def _failing():
                    yield StockYoloOomFallbackTests._FakeResult()
                    raise RuntimeError("CUDA out of memory.")

                return _failing()
            return iter((StockYoloOomFallbackTests._FakeResult(),))

    def _common_kwargs(self):
        return dict(
            img_w=2048,
            img_h=1024,
            lat=22,
            lon=120,
            lat_n=23.0,
            lat_s=22.0,
            lon_w=120.0,
            lon_e=121.0,
            m_per_px=1.0,
            stride=1024,
            imgsz=1024,
            device="cpu",
        )

    def test_oom_at_batch_falls_back_to_exact_batch_1_results(self):
        image = np.zeros((1024, 2048, 3), dtype=np.uint8)

        baseline_model = self._OomOnBatchYolo()
        baseline = STOCK.run_stock_yolo_pass(
            image, model=baseline_model, batch_size=1, **self._common_kwargs()
        )

        fallback_model = self._OomOnBatchYolo()
        fallback = STOCK.run_stock_yolo_pass(
            image, model=fallback_model, batch_size=8, **self._common_kwargs()
        )

        # No partial results from the aborted batched attempt may leak.
        self.assertEqual(baseline.placed_objects, fallback.placed_objects)
        self.assertEqual(baseline.placed_facades, fallback.placed_facades)
        self.assertEqual(baseline.counts_by_class, fallback.counts_by_class)
        self.assertEqual(
            [poly.tolist() for poly in baseline.occupied_px_polys],
            [poly.tolist() for poly in fallback.occupied_px_polys],
        )
        self.assertEqual(fallback.requested_batch_size, 8)
        self.assertEqual(fallback.effective_batch_size, 1)
        self.assertTrue(fallback.batch_fell_back)
        # One failed batched call, then one batch-1 call per crop.
        self.assertEqual(len(fallback_model.calls), 1 + len(baseline_model.calls))

    def test_non_oom_runtime_error_propagates(self):
        class AlwaysFailsYolo:
            def predict(self, **_kwargs):
                raise RuntimeError("device-side assert triggered")

        with self.assertRaises(RuntimeError):
            STOCK.run_stock_yolo_pass(
                np.zeros((1024, 2048, 3), dtype=np.uint8),
                model=AlwaysFailsYolo(),
                batch_size=8,
                **self._common_kwargs(),
            )

    def test_oom_at_batch_1_propagates(self):
        class OomAlwaysYolo:
            def predict(self, **_kwargs):
                raise RuntimeError("CUDA out of memory.")

        with self.assertRaises(RuntimeError):
            STOCK.run_stock_yolo_pass(
                np.zeros((1024, 2048, 3), dtype=np.uint8),
                model=OomAlwaysYolo(),
                batch_size=1,
                **self._common_kwargs(),
            )


def _fake_model(quads, cls, conf=0.9):
    """A YOLO stand-in returning one result with the given crop-pixel quads."""

    class FakeObb:
        def __init__(self):
            self.xyxyxyxy = torch.tensor(np.asarray(quads, dtype=np.float32))
            self.conf = torch.tensor([float(conf)] * len(quads))
            self.cls = torch.tensor([float(cls)] * len(quads))

    class FakeResult:
        def __init__(self):
            self.obb = FakeObb()

    class FakeYolo:
        def predict(self, **_kwargs):
            return iter((FakeResult(),))

    return FakeYolo()


def _rect(cx, cy, long_px, short_px):
    hl, hs = long_px / 2.0, short_px / 2.0
    return [[cx - hl, cy - hs], [cx + hl, cy - hs], [cx + hl, cy + hs], [cx - hl, cy + hs]]


class StockYoloBlacklistTests(unittest.TestCase):
    def test_dior_contributes_vehicles_only(self):
        self.assertEqual(STOCK.STATIC_DIOR_CLASSES, (STOCK.VEHICLE_CLASS,))
        self.assertEqual(STOCK.STOCK_CLASS_NAMES[STOCK.VEHICLE_CLASS], "vehicle")

    def test_no_blacklisted_dior_twin_is_mapped(self):
        # airplane, bridge, chimney, harbour, overpass, sports, windmill ...
        for dior_cls in (0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 16, 17, 19):
            cls = STOCK.DIOR_CLASS_OFFSET + dior_cls
            self.assertNotIn(cls, STOCK.STOCK_YOLO_ASSET_MAP)
            self.assertNotIn(cls, STOCK.STATIC_DIOR_CLASSES)

    def test_ladder_paths_are_all_in_the_asset_map(self):
        for cls, ladder in STOCK._SIZE_LADDERS_M.items():
            mapped = set(STOCK.STOCK_YOLO_ASSET_MAP[cls][1])
            for rung in ladder:
                for path in rung[1]:
                    self.assertIn(path, mapped)
                    self.assertTrue(path.endswith((".obj", ".agp")), path)

    def test_never_uses_dynamic_multi_part_ship_exports(self):
        for path in STOCK.STOCK_YOLO_ASSET_MAP[STOCK.SHIP_DOTA_CLASS][1]:
            if path.startswith("lib/ships/") and path.endswith(".obj"):
                self.assertTrue(path.startswith("lib/ships/powered/"), path)


class StockYoloSizeLadderTests(unittest.TestCase):
    def _pick(self, cls, long_m, kind=None):
        paths = STOCK.STOCK_YOLO_ASSET_MAP[cls][1]
        return STOCK._pick_sized_variant(cls, long_m, paths, 22.5, 120.3, 10, 10, kind=kind)

    def test_car_and_bus_lengths_pick_matching_models(self):
        self.assertEqual(self._pick(STOCK.VEHICLE_CLASS, 4.6), "lib/cars/car_static.obj")
        self.assertEqual(self._pick(STOCK.VEHICLE_CLASS, 12.5), "lib/cars/bus_static.obj")

    def test_no_rung_within_tolerance_places_nothing(self):
        self.assertIsNone(self._pick(STOCK.VEHICLE_CLASS, 30.0))
        # A 95 m ferry sits between the 65 m yacht and the 156 m carriers.
        self.assertIsNone(self._pick(STOCK.SHIP_DOTA_CLASS, 95.0))

    def test_ship_kind_prefers_matching_rungs(self):
        container = self._pick(STOCK.SHIP_DOTA_CLASS, 158.0, kind="container")
        cargo = self._pick(STOCK.SHIP_DOTA_CLASS, 158.0, kind="cargo")
        self.assertIn("Container", container)
        self.assertIn("BulkCarrier_155", cargo)

    def test_only_installed_paths_are_picked(self):
        only = ("lib/ships/powered/over_10mt/Cruiser_1200_any.obj",)
        got = STOCK._pick_sized_variant(STOCK.SHIP_DOTA_CLASS, 12.0, only, 0, 0, 0, 0)
        self.assertEqual(got, only[0])
        self.assertIsNone(STOCK._pick_sized_variant(STOCK.SHIP_DOTA_CLASS, 200.0, only, 0, 0, 0, 0))


class StockYoloHeadingTests(unittest.TestCase):
    def test_image_vectors_map_to_compass_headings(self):
        self.assertAlmostEqual(STOCK._image_vector_heading(0, -1), 0.0)    # up = north
        self.assertAlmostEqual(STOCK._image_vector_heading(1, 0), 90.0)    # right = east
        self.assertAlmostEqual(STOCK._image_vector_heading(0, 1), 180.0)
        self.assertAlmostEqual(STOCK._image_vector_heading(-1, 0), 270.0)
        # Diagonals are not mirrored (the old 90 - angle formula gave 135 here).
        self.assertAlmostEqual(STOCK._image_vector_heading(1, -1), 45.0)

    def test_quad_heading_follows_long_axis(self):
        quad = np.array([[0, 0], [30, 30], [25, 35], [-5, 5]], dtype=np.float32)
        _cx, _cy, long_m, _short_m, heading = STOCK._quad_geometry(quad, 1.0)
        self.assertGreater(long_m, 40.0)
        self.assertAlmostEqual(heading % 180.0, 135.0, places=3)

    def _offset_m(self, lon, lat, lon0, lat0):
        east = (lon - lon0) * 111320.0 * math.cos(math.radians(lat0))
        north = (lat - lat0) * 111320.0
        return east, north

    def test_centred_minus_z_models_are_placed_as_detected(self):
        lon, lat, h = STOCK.model_origin_for_centre(120.3, 22.6, 37.0, "lib/cars/car_static.obj")
        self.assertEqual((lon, lat, h), (120.3, 22.6, 37.0))
        _lon, _lat, h = STOCK.model_origin_for_centre(
            120.3, 22.6, 37.0, "opensceneryx/objects/vehicles/boats_ships/fishing/2.obj")
        self.assertAlmostEqual(h, 217.0)

    def test_off_centre_x_axis_ships_put_the_hull_on_the_detection(self):
        lat0 = 22.6
        # Freighter, bow north: bow is +X, origin at the stern 31.7 m south.
        lon, lat, h = STOCK.model_origin_for_centre(120.3, lat0, 0.0, "CDB-Library/ship/ship_pesca.obj")
        self.assertAlmostEqual(h, 270.0)
        east, north = self._offset_m(lon, lat, 120.3, lat0)
        self.assertAlmostEqual(north, -31.71, places=1)
        self.assertAlmostEqual(east, 0.03, places=1)
        # Tug, bow east: bow is -X, so the origin (near the bow) lies east of the centre.
        lon, lat, h = STOCK.model_origin_for_centre(
            120.3, lat0, 90.0, "CDB-Library/ship/ship_trawler1_senza_scia.obj")
        self.assertAlmostEqual(h, 180.0)
        east, north = self._offset_m(lon, lat, 120.3, lat0)
        self.assertAlmostEqual(east, 22.37, places=1)
        self.assertAlmostEqual(north, 5.93, places=1)

    def test_moored_mid_size_hulls_get_working_boats_not_the_yacht(self):
        ladder_paths = {p for rung in STOCK._SIZE_LADDERS_M[STOCK.SHIP_DOTA_CLASS] for p in rung[1]}
        self.assertFalse(any("Luxury_Yacht" in p for p in ladder_paths))
        for path in ("CDB-Library/ship/ship_pesca.obj", "CDB-Library/ship/ship_trawler1_senza_scia.obj"):
            self.assertIn(path, ladder_paths)
            self.assertIn(path, STOCK._MODEL_FRAME)


class StockYoloSeamTests(unittest.TestCase):
    def test_inner_seam_cuts_but_image_edge_does_not(self):
        box = [[0, 50], [100, 50], [100, 70], [0, 70]]
        # Touching the crop's left edge: a seam only when a crop lies left.
        self.assertTrue(STOCK.touches_inner_crop_seam(box, 1024, 0, 1024, 1024, 4096, 4096))
        self.assertFalse(STOCK.touches_inner_crop_seam(box, 0, 0, 1024, 1024, 4096, 4096))
        inside = [[100, 50], [200, 50], [200, 70], [100, 70]]
        self.assertFalse(STOCK.touches_inner_crop_seam(inside, 1024, 1024, 1024, 1024, 4096, 4096))


@unittest.skipIf(torch is None, "torch not installed")
class StockYoloShipAndVehicleRuleTests(unittest.TestCase):
    SIZE = 256

    def _run(self, model, *, m_per_px=0.3, class_offset=0, mesh_water=None, class_map=None):
        return STOCK.run_stock_yolo_pass(
            np.full((self.SIZE, self.SIZE, 3), 90, dtype=np.uint8),
            model=model,
            img_w=self.SIZE, img_h=self.SIZE,
            lat=0, lon=0, lat_n=1.0, lat_s=0.0, lon_w=0.0, lon_e=1.0,
            m_per_px=m_per_px, stride=self.SIZE, imgsz=self.SIZE, device="cpu",
            class_offset=class_offset,
            mesh_water_mask=mesh_water, class_map=class_map,
            water_class=6, road_class=4,
        )

    def _harbour(self):
        water = np.zeros((self.SIZE, self.SIZE), dtype=bool)
        water[:, 60:] = True              # quay at x < 60
        class_map = np.where(water, 6, 1).astype(np.int8)
        return water, class_map

    def test_moored_ship_is_placed_with_a_sized_boat(self):
        water, class_map = self._harbour()
        # 12 m boat (40 px at 0.3 m/px) alongside the quay.
        model = _fake_model([_rect(85, 120, 40, 12)], STOCK.SHIP_DOTA_CLASS)
        res = self._run(model, mesh_water=water, class_map=class_map)
        self.assertEqual(len(res.placed_objects), 1, res.rejected)
        self.assertIn(res.placed_objects[0][3], STOCK.STOCK_YOLO_ASSET_MAP[1][1])

    def test_ship_far_from_land_is_under_way(self):
        water = np.ones((self.SIZE, self.SIZE), dtype=bool)
        class_map = np.full((self.SIZE, self.SIZE), 6, dtype=np.int8)
        model = _fake_model([_rect(128, 128, 40, 12)], STOCK.SHIP_DOTA_CLASS)
        res = self._run(model, mesh_water=water, class_map=class_map)
        self.assertEqual(res.placed_objects, [])
        self.assertEqual(res.rejected.get("ship_under_way"), 1)

    def test_ship_on_land_and_without_water_data_are_dropped(self):
        water, class_map = self._harbour()
        model = _fake_model([_rect(25, 120, 30, 10)], STOCK.SHIP_DOTA_CLASS)
        res = self._run(model, mesh_water=water, class_map=class_map)
        self.assertEqual(res.rejected.get("ship_on_land"), 1)
        res = self._run(_fake_model([_rect(85, 120, 40, 12)], STOCK.SHIP_DOTA_CLASS))
        self.assertEqual(res.rejected.get("ship_no_water_data"), 1)

    def test_parked_car_is_placed_from_the_dior_offset_class(self):
        class_map = np.ones((self.SIZE, self.SIZE), dtype=np.int8)
        model = _fake_model([_rect(100, 100, 15, 6)], 18)   # DIOR vehicle, 4.5 m
        res = self._run(model, class_offset=STOCK.DIOR_CLASS_OFFSET, class_map=class_map)
        self.assertEqual([p[3] for p in res.placed_objects], ["lib/cars/car_static.obj"])
        self.assertEqual(res.counts_by_class, {STOCK.VEHICLE_CLASS: 1})

    def test_vehicle_on_road_or_water_is_dropped(self):
        class_map = np.ones((self.SIZE, self.SIZE), dtype=np.int8)
        class_map[90:110, :] = 4                            # SegFormer road
        class_map[200:, :] = 6                              # water
        model = _fake_model([_rect(100, 100, 15, 6), _rect(100, 220, 15, 6)], 18)
        res = self._run(model, class_offset=STOCK.DIOR_CLASS_OFFSET, class_map=class_map)
        self.assertEqual(res.placed_objects, [])
        self.assertEqual(res.rejected, {"vehicle_on_road": 1, "vehicle_on_water": 1})

    def test_vehicles_skipped_on_coarse_textures(self):
        model = _fake_model([_rect(100, 100, 5, 2)], 18)
        res = self._run(model, m_per_px=1.2, class_offset=STOCK.DIOR_CLASS_OFFSET,
                        class_map=np.ones((self.SIZE, self.SIZE), dtype=np.int8))
        self.assertEqual(res.placed_objects, [])

    def test_dota_vehicle_ids_are_not_placed(self):
        # DOTA 'small vehicle' (10) is not a static class even with masks.
        model = _fake_model([_rect(100, 100, 15, 6)], 10)
        res = self._run(model, class_map=np.ones((self.SIZE, self.SIZE), dtype=np.int8))
        self.assertEqual(res.placed_objects, [])


class StockYoloMergeTests(unittest.TestCase):
    def test_merge_rebases_owner_indices(self):
        a = STOCK.StockYoloResults()
        a.placed_objects = [(0, 0, 0, "a.obj")]
        a.occupied_px_polys = [np.zeros((4, 2), np.int32)]
        a.occupied_owner = [("object", 0)]
        a.counts_by_class = {14: 1}
        b = STOCK.StockYoloResults()
        b.placed_objects = [(1, 1, 0, "b.obj"), (2, 2, 0, "c.obj")]
        b.occupied_px_polys = [np.zeros((4, 2), np.int32)] * 2
        b.occupied_owner = [("object", 0), ("object", 1)]
        b.counts_by_class = {STOCK.VEHICLE_CLASS: 2}
        b.rejected = {"vehicle_on_road": 3}
        merged = STOCK.merge_stock_results(a, b)
        self.assertEqual(merged.occupied_owner, [("object", 0), ("object", 1), ("object", 2)])
        self.assertEqual([p[3] for p in merged.placed_objects], ["a.obj", "b.obj", "c.obj"])
        self.assertEqual(merged.counts_by_class, {14: 1, STOCK.VEHICLE_CLASS: 2})
        self.assertEqual(merged.rejected, {"vehicle_on_road": 3})


class _Evil:
    """Pickles to a call of open(marker, 'w'); a real unpickle creates it."""

    marker = None

    def __reduce__(self):
        return (open, (self.marker, "w"))


@unittest.skipIf(torch is None, "torch not installed")
class SafeCheckpointRebuildTests(unittest.TestCase):
    def test_rebuild_loads_weights_without_running_pickled_code(self):
        from ultralytics.nn.tasks import OBBModel

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            net = OBBModel(cfg="yolov8n-obb.yaml", nc=3, verbose=False)
            marker = tmp / "pwned.txt"
            evil = _Evil()
            evil.marker = str(marker)
            src = tmp / "foreign.pt"
            torch.save({"model": net, "ema": None, "payload": evil}, src)
            self.assertFalse(marker.exists())
            dst = tmp / "rebuilt.pt"
            names = STOCK.rebuild_checkpoint_safely(str(src), str(dst))
            self.assertFalse(marker.exists(), "foreign pickle code ran")
            self.assertEqual(len(names), 3)
            rebuilt = torch.load(dst, map_location="cpu", weights_only=False)["model"]
            for (k, v), (k2, v2) in zip(net.state_dict().items(), rebuilt.state_dict().items()):
                self.assertEqual(k, k2)
                self.assertTrue(torch.allclose(v.float(), v2.float(), atol=1e-2), k)


if __name__ == "__main__":
    unittest.main()
