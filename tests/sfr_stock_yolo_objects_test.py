import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_SFR_Stock_Yolo_Objects as STOCK


class StockYoloAssetMapTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
