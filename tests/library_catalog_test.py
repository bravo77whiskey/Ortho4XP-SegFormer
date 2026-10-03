import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_Asset_Inventory as ASSETINV
import O4_Building_Overlay as BLD
import O4_Library_Catalog as LIBCAT


def _write_obj(path, half_width=2.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "I\n800\nOBJ\n"
        f"VT {-half_width} 0 {-half_width} 0 1 0 0 0\n"
        f"VT {half_width} 5 {half_width} 0 1 0 1 1\n",
        encoding="utf-8",
    )


def _write_library(package, text):
    package.mkdir(parents=True, exist_ok=True)
    (package / "library.txt").write_text(text, encoding="utf-8")


def _bump_mtime(path):
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))


class LibraryCatalogTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.custom = tmp / "Custom Scenery"
        self.cache = tmp / "cache"
        self.simheaven = self.custom / "simHeaven_X-World_Europe-6-scenery"
        self.sfd = self.custom / "SFD Global Autogen"
        self.airport = self.custom / "Airport Pack"
        _write_obj(self.simheaven / "objects" / "house.obj")
        _write_obj(self.sfd / "objects" / "shed.obj", half_width=3.0)
        _write_obj(self.airport / "objects" / "hangar.obj")
        _write_library(
            self.simheaven,
            "A\n800\nLIBRARY\n"
            "EXPORT simheaven/houses/house_10x10x2.obj objects/house.obj\n"
            "EXPORT simheaven/houses/wall.fac objects/wall.fac\n"
            "EXPORT simheaven/houses/broken.obj objects/missing.obj\n",
        )
        _write_library(
            self.sfd,
            "A\n800\nLIBRARY\n"
            "REGION_DEFINE scandinavia\nREGION_RECT 0 50 30 72\n"
            "REGION scandinavia\n"
            "EXPORT sfd_global/scandinavia/suburban/shed.obj objects/shed.obj\n"
            "REGION all\n"
            "EXPORT_RATIO 0.5 simheaven/houses/house_10x10x2.obj objects/shed.obj\n",
        )
        _write_library(
            self.airport,
            "A\n800\nLIBRARY\n"
            "EXPORT lib/custom/hangar.obj objects/hangar.obj\n"
            "EXPORT_EXTEND lib/custom/extended.obj objects/hangar.obj\n",
        )
        self.ini = self.custom / "scenery_packs.ini"
        self._write_ini(self.simheaven, self.sfd, self.airport)
        BLD._LIBRARY_CATALOG = None

    def tearDown(self):
        BLD._LIBRARY_CATALOG = None
        self._tmp.cleanup()

    def _write_ini(self, *active):
        lines = ["I", "1000 Version", "SCENERY"]
        lines += [f"SCENERY_PACK Custom Scenery/{pack.name}/" for pack in active]
        self.ini.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _open(self):
        return LIBCAT.open_catalog(str(self.custom), str(self.cache))

    def test_runtime_exports_match_disk_scan(self):
        catalog = self._open()
        for patterns in (("simheaven",), ("sfd global autogen", "simheaven"), ()):
            for suffixes in ((".obj",), (".obj", ".fac")):
                self.assertEqual(
                    catalog.runtime_exports(patterns, suffixes),
                    ASSETINV.scan_library_exports(
                        custom_scenery_dir=str(self.custom),
                        package_name_patterns=patterns,
                        suffixes=suffixes,
                    ),
                    (patterns, suffixes),
                )
        self.assertIsNone(catalog.runtime_exports(("simheaven",), (".for",)))

    def test_overlay_helpers_match_disk_fallback(self):
        disk = (
            BLD._find_library_export(self.custom, "sfd_global/"),
            BLD._find_library_export(self.custom, "simheaven/"),
            BLD._find_library_export(self.custom, "opensceneryx/"),
            BLD._active_custom_library_index(self.custom),
            BLD._scan_runtime_library_exports(
                self.custom, include_sfd=True, include_simheaven=True
            ),
        )
        BLD._LIBRARY_CATALOG = self._open()
        cached = (
            BLD._find_library_export(self.custom, "sfd_global/"),
            BLD._find_library_export(self.custom, "simheaven/"),
            BLD._find_library_export(self.custom, "opensceneryx/"),
            BLD._active_custom_library_index(self.custom),
            BLD._scan_runtime_library_exports(
                self.custom, include_sfd=True, include_simheaven=True
            ),
        )
        self.assertEqual(cached, disk)
        self.assertEqual(
            set(cached[3]),
            {
                "simheaven/houses/house_10x10x2.obj",
                "sfd_global/scandinavia/suburban/shed.obj",
                "lib/custom/hangar.obj",
            },
        )

    def test_other_custom_scenery_folder_falls_back_to_disk(self):
        BLD._LIBRARY_CATALOG = self._open()
        with tempfile.TemporaryDirectory() as other:
            self.assertEqual(BLD._scan_runtime_library_exports(
                other, include_simheaven=True
            ), [])
            self.assertIsNone(BLD._library_catalog_for(other))

    def test_reopen_reuses_catalog_and_bounds(self):
        catalog = self._open()
        obj = str(self.simheaven / "objects" / "house.obj")
        self.assertEqual(catalog.bounds_for(obj), (-2.0, 2.0, -2.0, 2.0))
        self.assertTrue(catalog.save_if_dirty())

        reopened = self._open()
        self.assertEqual(reopened.sync()["parsed"], 0)
        self.assertIn(obj, reopened.obj_bounds)
        self.assertFalse(reopened.dirty)

    def test_changed_library_txt_reparses_only_that_package(self):
        catalog = self._open()
        house = str(self.simheaven / "objects" / "house.obj")
        shed = str(self.sfd / "objects" / "shed.obj")
        catalog.bounds_for(house)
        catalog.bounds_for(shed)
        catalog.save()

        library_txt = self.simheaven / "library.txt"
        library_txt.write_text(
            library_txt.read_text(encoding="utf-8")
            + "EXPORT simheaven/houses/house_12x12x2.obj objects/house.obj\n",
            encoding="utf-8",
        )
        _bump_mtime(library_txt)

        reopened = self._open()
        self.assertTrue(reopened.dirty)
        self.assertNotIn(house, reopened.obj_bounds)
        self.assertIn(shed, reopened.obj_bounds)
        virtual = {
            export.virtual_path
            for export in reopened.runtime_exports(("simheaven",), (".obj",))
        }
        self.assertIn("simheaven/houses/house_12x12x2.obj", virtual)

    def test_deactivated_and_removed_packages_drop_out(self):
        catalog = self._open()
        self.assertTrue(catalog.has_prefix("sfd_global/"))

        self._write_ini(self.simheaven, self.airport)
        catalog = self._open()
        self.assertEqual(catalog.runtime_exports(("sfd global",), (".obj",)), [])
        self.assertNotIn(
            "sfd_global/scandinavia/suburban/shed.obj",
            catalog.active_object_index(),
        )

        catalog.save()
        (self.sfd / "library.txt").unlink()
        catalog = self._open()
        self.assertEqual(len(catalog.packages), 2)
        self.assertFalse(catalog.has_prefix("sfd_global/"))

    def test_rescan_rebuilds_and_forgets_measured_bounds(self):
        catalog = self._open()
        obj = str(self.airport / "objects" / "hangar.obj")
        catalog.bounds_for(obj)
        catalog.save()
        # An OBJ replaced in place leaves library.txt untouched: only a
        # rescan picks it up.
        _write_obj(self.airport / "objects" / "hangar.obj", half_width=9.0)
        self.assertEqual(self._open().bounds_for(obj), (-2.0, 2.0, -2.0, 2.0))

        LIBCAT.rescan(str(self.custom), str(self.cache))
        self.assertEqual(self._open().bounds_for(obj), (-9.0, 9.0, -9.0, 9.0))

    def test_catalog_for_other_folder_is_ignored(self):
        self._open().save()
        payload_path = LIBCAT.catalog_file(str(self.cache))
        with tempfile.TemporaryDirectory() as other_root:
            other = Path(other_root) / "Custom Scenery"
            other.mkdir()
            catalog = LIBCAT.open_catalog(str(other), str(self.cache))
            self.assertEqual(catalog.packages, {})
        self.assertTrue(os.path.isfile(payload_path))

    def test_no_cache_dir_means_no_catalog(self):
        self.assertIsNone(LIBCAT.open_catalog(str(self.custom), None))
        self.assertIsNone(LIBCAT.open_catalog(None, str(self.cache)))

    def test_default_scenery_exports_match_disk_scan(self):
        default_lib = self.custom.parent / "Resources" / "default scenery" / "1000 autogen"
        _write_obj(default_lib / "objects" / "garage.obj")
        _write_library(
            default_lib,
            "A\n800\nLIBRARY\n"
            "EXPORT lib/g10/autogen/garage.obj objects/garage.obj\n"
            "EXPORT lib/g10/autogen/other.obj objects/garage.obj\n",
        )
        pools = {1: [
            {"kind": "object", "path": "lib/g10/autogen/garage.obj"},
            {"kind": "object", "path": "simheaven/houses/house_10x10x2.obj"},
        ]}
        disk = BLD._scan_default_object_exports(self.custom, pools)
        BLD._LIBRARY_CATALOG = self._open()
        cached = BLD._scan_default_object_exports(self.custom, pools)
        self.assertEqual(cached, disk)
        self.assertEqual(
            [export.virtual_path for export in cached],
            ["lib/g10/autogen/garage.obj"],
        )
        # Default libraries never count as Custom Scenery packages.
        self.assertNotIn(
            "lib/g10/autogen/garage.obj", BLD._active_custom_library_index(self.custom)
        )

    def test_cached_results_persist_and_drop_with_their_package(self):
        catalog = self._open()
        house = str(self.simheaven / "objects" / "house.obj")
        shed = str(self.sfd / "objects" / "shed.obj")
        calls = []

        def build(path):
            calls.append(path)
            return f"result:{os.path.basename(path)}"

        for path in (house, shed, house):
            catalog.cached("probe:v1", path, lambda path=path: build(path))
        self.assertEqual(calls, [house, shed])
        catalog.save()

        library_txt = self.sfd / "library.txt"
        library_txt.write_text(library_txt.read_text(encoding="utf-8") + "\n",
                               encoding="utf-8")
        _bump_mtime(library_txt)
        reopened = self._open()
        self.assertEqual(reopened.derived["probe:v1"], {house: "result:house.obj"})

        LIBCAT.rescan(str(self.custom), str(self.cache))
        self.assertEqual(self._open().derived, {})

    def test_roof_colours_come_from_catalog_whatever_the_tile_cache(self):
        from unittest import mock

        pools = {1: [{"kind": "object", "path": "simheaven/houses/house_10x10x2.obj"}]}
        exports = BLD._scan_runtime_library_exports(self.custom, include_simheaven=True)
        BLD._LIBRARY_CATALOG = self._open()
        descriptor = {"family": "red", "lab": (50.0, 40.0, 30.0)}
        with mock.patch.object(
            BLD.ROOFCOLOR, "analyze_obj_roof_color", return_value=descriptor
        ) as analyze:
            BLD._prepare_asset_roof_colors(True, self.custom, pools, exports, None)
            BLD._LIBRARY_CATALOG.save()
            BLD._LIBRARY_CATALOG = self._open()
            counts = BLD._prepare_asset_roof_colors(
                True, self.custom, pools, exports, None
            )
        # Analysed once, then served from the reopened catalog.
        self.assertEqual(analyze.call_count, 1)
        self.assertEqual(counts["safe"], 1)
        self.assertEqual(pools[1][0]["roof_color_variants"], (descriptor,))

    def test_library_obj_bounds_uses_catalog_when_open(self):
        obj = str(self.simheaven / "objects" / "house.obj")
        self.assertEqual(BLD._library_obj_bounds(obj), (-2.0, 2.0, -2.0, 2.0))
        BLD._LIBRARY_CATALOG = self._open()
        BLD._LIBRARY_CATALOG.obj_bounds[obj] = (0.0, 1.0, 0.0, 1.0)
        self.assertEqual(BLD._library_obj_bounds(obj), (0.0, 1.0, 0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
