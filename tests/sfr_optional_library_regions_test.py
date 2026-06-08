"""Region routing tests for the curated optional building libraries.

Covers the Track-A library additions (FFLibrary, BS2001, RuScenery, ZDP,
AR_Library, OB_Library, MisterX, Vectors to Final) and the shipped
``o4sfr`` library that uses an inline ``/<region>/`` path token.
"""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_SFR_Building_Overlay as BLD


class OptionalLibraryRegionsTest(unittest.TestCase):
    def _accepted(self, obj_path, lib_id, tile_region):
        return BLD._optional_library_rejection_reason(obj_path, lib_id, tile_region) is None

    # --- Track A: per-library fixed regions ---------------------------------

    def test_ff_library_is_europe_only(self):
        path = "ff_library/houses/residential/townhouse_03.obj"
        self.assertEqual(
            BLD._optional_library_asset_regions(path, "ff-library"),
            ("europe",),
        )
        self.assertTrue(self._accepted(path, "ff-library", "europe"))
        self.assertTrue(self._accepted(path, "ff-library", "mediterranean"))
        self.assertFalse(self._accepted(path, "ff-library", "north_america"))
        self.assertFalse(self._accepted(path, "ff-library", "asia"))

    def test_ruscenery_is_europe_only(self):
        path = "ruscenery/buildings/residential/khrushchyovka.obj"
        self.assertEqual(
            BLD._optional_library_asset_regions(path, "ruscenery"),
            ("europe",),
        )
        self.assertTrue(self._accepted(path, "ruscenery", "europe"))
        self.assertFalse(self._accepted(path, "ruscenery", "africa"))

    def test_bs2001_is_europe_only(self):
        path = "bs2001/buildings/houses/house_eu_01.obj"
        self.assertEqual(
            BLD._optional_library_asset_regions(path, "bs2001"),
            ("europe",),
        )
        self.assertTrue(self._accepted(path, "bs2001", "scandinavia"))
        self.assertFalse(self._accepted(path, "bs2001", "south_america"))

    def test_ar_library_is_south_america_only(self):
        path = "ar_library/buildings/residential/casa_01.obj"
        self.assertEqual(
            BLD._optional_library_asset_regions(path, "ar-library"),
            ("south_america",),
        )
        self.assertTrue(self._accepted(path, "ar-library", "south_america"))
        self.assertFalse(self._accepted(path, "ar-library", "europe"))
        self.assertFalse(self._accepted(path, "ar-library", "north_america"))

    def test_vectors_to_final_is_europe_only(self):
        path = "vectors_to_final/buildings/residential/barracks.obj"
        self.assertEqual(
            BLD._optional_library_asset_regions(path, "vectors-to-final"),
            ("europe",),
        )
        self.assertTrue(self._accepted(path, "vectors-to-final", "europe"))
        self.assertFalse(self._accepted(path, "vectors-to-final", "asia"))

    def test_zdp_library_serves_americas_and_europe(self):
        path = "zdp_library/buildings/houses/house_01.obj"
        self.assertEqual(
            BLD._optional_library_asset_regions(path, "zdp-library"),
            ("north_america", "europe"),
        )
        self.assertTrue(self._accepted(path, "zdp-library", "north_america"))
        self.assertTrue(self._accepted(path, "zdp-library", "europe"))
        self.assertFalse(self._accepted(path, "zdp-library", "asia"))
        self.assertFalse(self._accepted(path, "zdp-library", "africa"))

    # --- Generic libraries act as a wildcard --------------------------------

    def test_ob_library_is_generic_wildcard(self):
        path = "ob_library/buildings/houses/house_a.obj"
        self.assertEqual(
            BLD._optional_library_asset_regions(path, "ob-library"),
            ("generic",),
        )
        for region in (
            "europe",
            "north_america",
            "south_america",
            "asia",
            "se_asia",
            "africa",
            "australia_oceania",
            "mediterranean",
            "scandinavia",
        ):
            self.assertTrue(
                self._accepted(path, "ob-library", region),
                msg=f"OB_Library should serve {region}",
            )

    def test_misterx_is_generic_wildcard(self):
        path = "misterx_library/buildings/residential/house.obj"
        self.assertEqual(
            BLD._optional_library_asset_regions(path, "misterx"),
            ("generic",),
        )
        self.assertTrue(self._accepted(path, "misterx", "asia"))
        self.assertTrue(self._accepted(path, "misterx", "africa"))

    # --- o4sfr library uses an inline /<region>/ path token -----------------

    def test_o4sfr_europe_path_token(self):
        path = "o4sfr/europe/residential/townhouse_03.obj"
        self.assertEqual(
            BLD._optional_library_asset_regions(path, "o4sfr"),
            ("europe",),
        )
        self.assertTrue(self._accepted(path, "o4sfr", "europe"))
        self.assertFalse(self._accepted(path, "o4sfr", "north_america"))

    def test_o4sfr_north_america_path_token(self):
        path = "o4sfr/north_america/residential/ranch_01.obj"
        self.assertEqual(
            BLD._optional_library_asset_regions(path, "o4sfr"),
            ("north_america",),
        )
        self.assertTrue(self._accepted(path, "o4sfr", "north_america"))
        self.assertTrue(self._accepted(path, "o4sfr", "north_america_west"))
        self.assertFalse(self._accepted(path, "o4sfr", "europe"))

    def test_o4sfr_asia_path_token(self):
        path = "o4sfr/asia/residential/house_jp_01.obj"
        regions = BLD._optional_library_asset_regions(path, "o4sfr")
        self.assertEqual(regions, ("asia",))
        self.assertTrue(self._accepted(path, "o4sfr", "asia"))
        self.assertTrue(self._accepted(path, "o4sfr", "se_asia"))
        self.assertFalse(self._accepted(path, "o4sfr", "europe"))

    def test_o4sfr_southeast_asia_path_token(self):
        path = "o4sfr/se_asia/residential/stilt_house.obj"
        self.assertEqual(
            BLD._optional_library_asset_regions(path, "o4sfr"),
            ("se_asia",),
        )
        self.assertTrue(self._accepted(path, "o4sfr", "asia"))
        self.assertTrue(self._accepted(path, "o4sfr", "se_asia"))
        self.assertFalse(self._accepted(path, "o4sfr", "europe"))

    def test_o4sfr_africa_and_oceania_path_tokens(self):
        for token, tile_region, off_region in (
            ("africa", "africa", "europe"),
            ("australia_oceania", "australia_oceania", "asia"),
            ("mediterranean", "mediterranean", "scandinavia"),
        ):
            path = f"o4sfr/{token}/residential/sample.obj"
            with self.subTest(token=token):
                regions = BLD._optional_library_asset_regions(path, "o4sfr")
                self.assertIn(token, regions)
                self.assertTrue(self._accepted(path, "o4sfr", tile_region))
                self.assertFalse(self._accepted(path, "o4sfr", off_region))

    def test_o4sfr_without_region_token_is_generic(self):
        path = "o4sfr/uncategorized/residential/house.obj"
        self.assertEqual(
            BLD._optional_library_asset_regions(path, "o4sfr"),
            ("generic",),
        )
        for region in ("europe", "north_america", "asia", "africa"):
            self.assertTrue(self._accepted(path, "o4sfr", region))

    # --- Registry sanity ----------------------------------------------------

    def test_curated_libraries_entries_have_required_keys(self):
        for lib_id, entry in BLD.CURATED_EXTRA_BUILDING_LIBRARIES.items():
            with self.subTest(lib_id=lib_id):
                self.assertIn("label", entry)
                self.assertIn("package_patterns", entry)
                self.assertIn("virtual_prefixes", entry)
                regions = entry.get("regions")
                if regions is not None:
                    self.assertIsInstance(regions, tuple)
                    for region in regions:
                        self.assertTrue(
                            region == "generic"
                            or region in BLD.OPTIONAL_ASSET_REGION_ALIASES,
                            msg=f"{lib_id} declared unknown region {region!r}",
                        )

    def test_region_priority_regional_beats_generic(self):
        # europe-tagged asset for a europe tile → priority 0
        self.assertEqual(
            BLD._region_priority_for_asset_regions(("europe",), "europe"),
            0,
        )
        # generic asset matches but at lower priority (1)
        self.assertEqual(
            BLD._region_priority_for_asset_regions(("generic",), "europe"),
            1,
        )
        # europe-tagged asset on a north_america tile → falls to 1
        self.assertEqual(
            BLD._region_priority_for_asset_regions(("europe",), "north_america"),
            1,
        )
        # empty regions tuple defaults to 1 (no signal)
        self.assertEqual(
            BLD._region_priority_for_asset_regions((), "europe"),
            1,
        )

    def test_asset_retry_sort_key_orders_regional_first(self):
        regional = {
            "kind": "object",
            "path": "lib/regional.obj",
            "source": "Track A",
            "footprint_area_m2": 200.0,
            "footprint_max_side_m": 20.0,
            "region_priority": 0,
        }
        generic = {
            "kind": "object",
            "path": "lib/generic.obj",
            "source": "Track A",
            "footprint_area_m2": 80.0,   # smaller footprint
            "footprint_max_side_m": 10.0,
            "region_priority": 1,
        }
        # Despite generic having a smaller footprint, regional must come first.
        sorted_assets = sorted(
            [generic, regional], key=BLD._asset_retry_sort_key
        )
        self.assertEqual(sorted_assets[0]["path"], "lib/regional.obj")
        self.assertEqual(sorted_assets[1]["path"], "lib/generic.obj")

    def test_opensceneryx_override_routes_wooden_houses_to_europe(self):
        # Sanity check that the compiled override file is loaded and applied
        # for the canonical opensceneryx Haussmannian-style wooden houses.
        regions = BLD._optional_library_asset_regions(
            "opensceneryx/objects/buildings/residential/houses/wooden/7.obj",
            "opensceneryx",
        )
        self.assertIn("europe", regions)

    def test_track_a_libraries_registered(self):
        expected = {
            "ff-library", "ruscenery", "bs2001", "ar-library",
            "ob-library", "zdp-library", "misterx", "vectors-to-final",
            "o4sfr",
        }
        registered = set(BLD.CURATED_EXTRA_BUILDING_LIBRARIES)
        missing = expected - registered
        self.assertFalse(missing, msg=f"missing registrations: {sorted(missing)}")


if __name__ == "__main__":
    unittest.main()
