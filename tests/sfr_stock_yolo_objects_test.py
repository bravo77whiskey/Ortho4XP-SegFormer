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

import O4_SFR_Stock_Yolo_Objects as STOCK


class StockYoloAssetMapTests(unittest.TestCase):
    def test_default_batch_constant_is_positive(self):
        self.assertGreaterEqual(STOCK.DEFAULT_STOCK_YOLO_BATCH, 1)

    def test_static_dota_classes_excludes_moving_and_already_rendered(self):
        # Moving (plane, ship, vehicles, helicopter) and already-rendered
        # (bridge, roundabout) must never appear in the placement map.
        forbidden = {0, 1, 8, 9, 10, 11, 12}
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
            "simheaven/",       # simHeaven X-World
            "lib/",             # X-Plane 12 default
            "SFD_Global/",      # SFD Global
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
        # land on different assets — otherwise stadium variety collapses to
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


@unittest.skipIf(torch is None, "torch is not installed")
class StockYoloFacadeGeometryTests(unittest.TestCase):
    def _run_fake_detection(self, cls, **kwargs):
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
            m_per_px=1.0,
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

    def test_non_tank_facade_keeps_obb_ring(self):
        result = self._run_fake_detection(3)

        self.assertEqual(len(result.placed_facades), 1)
        ring, path, _height_m = result.placed_facades[0]
        self.assertIn(path, STOCK.STOCK_YOLO_ASSET_MAP[3][1])
        self.assertEqual(len(ring), 5)
        self.assertEqual(ring[0], ring[-1])
        self.assertGreater(STOCK._signed_lonlat_ring_area(ring), 0.0)
        self.assertEqual(result.occupied_px_polys[0].shape[0], 4)

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


if __name__ == "__main__":
    unittest.main()
