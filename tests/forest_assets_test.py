import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock
import contextlib

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_Forest_Assets as FOREST_ASSETS
import O4_SFR_Climate_Regions as CLIMATE_REGIONS
import O4_SFR_Vegetation_Overlay as SFR_VEG
import O4_SFR_Inference as SEGFORMER


class _IndexRng:
    def __init__(self, index):
        self.index = index

    def integers(self, low, high=None):
        if high is None:
            high = low
            low = 0
        span = max(1, high - low)
        return low + (self.index % span)


class ForestAssetPolicyTests(unittest.TestCase):
    def test_load_dds_or_none_skips_problem_texture(self):
        with mock.patch.object(
            SEGFORMER,
            "_load_dds",
            side_effect=PermissionError(13, "Permission denied"),
        ):
            out = StringIO()
            with contextlib.redirect_stdout(out):
                arr = SEGFORMER.load_dds_or_none(
                    str(ROOT / "missing.dds"),
                    log_prefix="[SFR Veg]",
                    display_name="264000_315952_BI19.dds",
                )
        self.assertIsNone(arr)
        self.assertIn("264000_315952_BI19.dds", out.getvalue())
        self.assertIn("Permission denied", out.getvalue())
        self.assertIn("skipping", out.getvalue())

    def test_allowlist_contains_default_and_gfv2_assets(self):
        approved = set(FOREST_ASSETS.APPROVED_GENERATED_FOREST_PATHS)
        self.assertEqual(FOREST_ASSETS.HEIGHT_CUTOFF_METERS, 24.0)
        self.assertIn("lib/vegetation/forests/broadleaves/warm_dry.for", approved)
        self.assertIn("lib/vegetation/forests/broadleaves/cold_low.for", approved)
        self.assertNotIn("lib/vegetation/forests/conifers/warm_dry.for", approved)
        self.assertIn(
            "forests/tropical/woodland/tropical_woodland_100_y2.for",
            approved,
        )
        self.assertIn(
            "forests/northmiddle/cropland/northmiddle_cropland_25_y2.for",
            approved,
        )
        self.assertIn(
            "forests/northnorth/mixed/northnorth_mixed_25_y1.for",
            approved,
        )

    def test_allowlist_excludes_known_tall_default_and_gfv2_assets(self):
        approved = set(FOREST_ASSETS.APPROVED_GENERATED_FOREST_PATHS)
        for path in FOREST_ASSETS.KNOWN_TALL_DEFAULT_FOREST_PATHS:
            self.assertNotIn(path, approved)
        for path in FOREST_ASSETS.KNOWN_TALL_GFV2_FOREST_PATHS:
            self.assertNotIn(path, approved)

    def test_default_assets_stay_in_tree_mix(self):
        candidates = SFR_VEG._short_tree_candidates("subtropical", 75)
        self.assertTrue(any(path.startswith("lib/vegetation/") for path in candidates))
        self.assertTrue(any(path.startswith("forests/") for path in candidates))
        northnorth = SFR_VEG._short_tree_candidates("northnorth", 75)
        self.assertIn(
            "forests/northnorth/mixed/northnorth_mixed_75_y1.for",
            northnorth,
        )
        self.assertTrue(any(path.startswith("lib/vegetation/") for path in northnorth))

    def test_koppen_grid_drives_vegetation_climate_region(self):
        self.assertEqual(CLIMATE_REGIONS.koppen_code(1.35, 103.8), "Af")
        self.assertEqual(FOREST_ASSETS.climate_region(1.35, 103.8), "tropical")
        self.assertEqual(CLIMATE_REGIONS.koppen_code(42.0, 12.0), "Csa")
        self.assertEqual(FOREST_ASSETS.climate_region(42.0, 12.0), "northsouth")
        self.assertEqual(CLIMATE_REGIONS.koppen_code(43.7, -79.4), "Dfb")
        self.assertEqual(FOREST_ASSETS.climate_region(43.7, -79.4), "northmiddle")

    def test_latitude_only_climate_region_keeps_legacy_fallback(self):
        self.assertEqual(FOREST_ASSETS.climate_region(0.0), "tropical")
        self.assertEqual(FOREST_ASSETS.climate_region(34.0), "northsouth")

    def test_tree_selection_biases_toward_lighter_gfv2_assets(self):
        region = "northsouth"
        dlevel = 75
        candidates = FOREST_ASSETS.short_tree_candidates(region, dlevel)
        weights = FOREST_ASSETS.tree_candidate_weights(candidates)
        self.assertEqual(
            candidates,
            (
                "forests/northsouth/woodland/northsouth_woodland_75_y1.for",
                "lib/vegetation/forests/broadleaves/warm_dry.for",
            ),
        )
        self.assertEqual(weights, (48, 15))
        for rng_index in range(48):
            choice = FOREST_ASSETS.choose_tree_path(
                region,
                dlevel,
                _IndexRng(rng_index),
                context="managed",
            )
            self.assertEqual(
                choice,
                "forests/northsouth/woodland/northsouth_woodland_75_y1.for",
            )
        self.assertEqual(
            FOREST_ASSETS.choose_tree_path(
                region,
                dlevel,
                _IndexRng(48),
                context="managed",
            ),
            "lib/vegetation/forests/broadleaves/warm_dry.for",
        )
        self.assertEqual(
            FOREST_ASSETS.mesh_defs_for_path(
                "lib/vegetation/forests/broadleaves/warm_dry.for"
            ),
            45,
        )
        for rng_index in range(20):
            choice = FOREST_ASSETS.choose_tree_path(
                region,
                dlevel,
                _IndexRng(rng_index),
                context="bulk",
            )
            self.assertEqual(
                choice,
                "forests/northsouth/woodland/northsouth_woodland_75_y1.for",
            )

    def test_sfr_for_entry_emits_only_allowlisted_paths(self):
        cases = (
            (SEGFORMER.CLASS_TREE, 0.80, "area", None),
            (SEGFORMER.CLASS_TREE, 0.35, "treeline", "tree_row_linear"),
            (SEGFORMER.CLASS_TREE, 0.55, "area", "natural_woodland_open"),
            (SEGFORMER.CLASS_RANGELAND, 0.60, "area", None),
            (SEGFORMER.CLASS_AGRICULTURE, 0.25, "treeline", None),
        )
        regions = ("subtropical", "northmiddle", "northnorth")
        for rng_index in range(6):
            rng = _IndexRng(rng_index)
            for region in regions:
                for veg_cls, frac, shape, veg_type in cases:
                    path, _density = SFR_VEG._for_entry(
                        veg_cls,
                        frac,
                        shape,
                        region,
                        rng,
                        density_override=None,
                        veg_type=veg_type,
                    )
                    self.assertTrue(
                        FOREST_ASSETS.is_approved_generated_forest_path(path),
                        msg=path,
                    )

    def test_dense_area_tree_polygons_use_gfv2_only(self):
        for rng_index in range(20):
            path, _density = SFR_VEG._for_entry(
                SEGFORMER.CLASS_TREE,
                0.80,
                "area",
                "northsouth",
                _IndexRng(rng_index),
                density_override=None,
                veg_type=None,
            )
            self.assertTrue(path.startswith("forests/"), msg=path)

    def test_gfv2_path_parser_rejects_palm_coconut_and_shrub_type_sources(self):
        self.assertEqual(
            FOREST_ASSETS.parse_gfv2_path(
                "forests/tropical/woodland/tropical_woodland_75_y1.for"
            )["family"],
            "woodland",
        )
        for path in (
            "forests/tropical/palm/tropical_palm_75_y1.for",
            "forests/tropical/coconut/tropical_coconut_75_y1.for",
            "forests/subtropical/shrub/subtropical_shrub_50_y1.for",
            "forests/northmiddle/scrub/northmiddle_scrub_50_y1.for",
        ):
            self.assertFalse(FOREST_ASSETS.is_acceptable_gfv2_type_source(path))

    def test_sfr_for_entry_uses_nearest_gfv2_type_hint_when_available(self):
        for rng_index in range(8):
            path, _density = SFR_VEG._for_entry(
                SEGFORMER.CLASS_TREE,
                0.72,
                "area",
                "northnorth",
                _IndexRng(rng_index),
                density_override=None,
                veg_type="natural_forest_closed",
                gfv2_type_path="forests/tropical/woodland/tropical_woodland_75_y2.for",
            )
            self.assertTrue(path.startswith("forests/tropical/woodland/"), msg=path)

    def test_nearest_gfv2_type_hint_skips_rejected_sources(self):
        records = [
            {
                "pts": [(0.0, 0.0), (0.0001, 0.0), (0.0001, 0.0001)],
                "path": "forests/tropical/palm/tropical_palm_75_y1.for",
                "_centroid": (0.00002, 0.00002),
            },
            {
                "pts": [(0.0, 0.0), (0.0002, 0.0), (0.0002, 0.0002)],
                "path": "forests/tropical/woodland/tropical_woodland_75_y1.for",
                "_centroid": (0.00008, 0.00008),
            },
        ]

        self.assertEqual(
            SFR_VEG._nearest_acceptable_gfv2_path((0.0, 0.0), records, max_distance_m=50.0),
            "forests/tropical/woodland/tropical_woodland_75_y1.for",
        )

    def test_optimized_nearest_gfv2_lookup_skips_rejected_sources(self):
        records = [
            {
                "pts": [(0.0, 0.0), (0.0001, 0.0), (0.0001, 0.0001)],
                "path": "forests/tropical/shrub/tropical_shrub_75_y1.for",
                "_centroid": (0.00002, 0.00002),
            },
            {
                "pts": [(0.0, 0.0), (0.0002, 0.0), (0.0002, 0.0002)],
                "path": "forests/tropical/woodland/tropical_woodland_75_y1.for",
                "_centroid": (0.00008, 0.00008),
            },
        ]

        lookup = SFR_VEG._build_gfv2_type_lookup(records, origin_lat=0.0)

        self.assertEqual(
            SFR_VEG._nearest_acceptable_gfv2_path((0.0, 0.0), lookup, max_distance_m=50.0),
            "forests/tropical/woodland/tropical_woodland_75_y1.for",
        )

    def test_climate_mode_does_not_call_gfv2_proximity_lookup(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[2:18, 2:18] = 255

        with mock.patch.object(
            SFR_VEG,
            "_nearest_acceptable_gfv2_path",
            side_effect=AssertionError("proximity lookup should be disabled"),
        ):
            polygons = SFR_VEG._process_dds_mask(
                mask,
                SEGFORMER.CLASS_TREE,
                img_w=20,
                img_h=20,
                lat_n=1.0,
                lat_s=0.99,
                lon_w=2.0,
                lon_e=2.01,
                tile_lat=0.0,
                tile_lon=2.0,
                m_per_px=2.0,
                min_area_px=1.0,
                simplify_px=1.0,
                region="northsouth",
                rng=_IndexRng(0),
                density_override=None,
                context_masks={},
                type_counts={},
                gfv2_type_records=None,
            )

        self.assertTrue(polygons)
        self.assertTrue(polygons[0][0].startswith("forests/northsouth/"))

    def test_cli_accepts_gfv2_asset_proximity_flag(self):
        argv = [
            "generate_veg_overlay.py",
            "textures",
            "1",
            "2",
            "--gfv2-asset-proximity",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = SFR_VEG.parse_args()

        self.assertTrue(args.gfv2_asset_proximity)

    def test_managed_tree_context_can_still_use_defaults(self):
        default_seen = False
        for rng_index in range(80):
            path, _density = SFR_VEG._for_entry(
                SEGFORMER.CLASS_TREE,
                0.55,
                "area",
                "northsouth",
                _IndexRng(rng_index),
                density_override=None,
                veg_type="park_or_managed_green",
            )
            self.assertTrue(
                FOREST_ASSETS.is_approved_generated_forest_path(path),
                msg=path,
            )
            if path.startswith("lib/vegetation/"):
                default_seen = True
        self.assertTrue(default_seen)

    def test_spec_does_not_bundle_library_txt(self):
        spec_text = (ROOT / "Ortho4XP.spec").read_text(encoding="utf-8")
        self.assertNotIn("library.txt", spec_text)

    def test_repo_has_no_generated_library_txt(self):
        ignored_roots = {".git", ".venv", "build", "dist", "dist_build", "__pycache__"}
        found = []
        for path in ROOT.rglob("library.txt"):
            if any(part in ignored_roots for part in path.parts):
                continue
            found.append(path)
        self.assertEqual(found, [])

    def test_vegetation_aux_cache_key_tracks_mesh_water_signature(self):
        base_args = (
            "34336_26384_BI16.dds",
            1.0,
            0.9,
            2.0,
            2.1,
            64,
            64,
            2.4,
            ("roads",),
            ("res-roads",),
            ("tree-rows",),
            {"water_polys": (0, 0, 0.0)},
            (),
            (0, 0, 0.0),
            (0, 0.0),
            12.0,
            None,
            10.0,
        )

        key_a = SFR_VEG._dds_mask_cache_key(*base_args, ("mesh-a", 10, 20))
        key_b = SFR_VEG._dds_mask_cache_key(*base_args, ("mesh-b", 10, 20))

        self.assertNotEqual(key_a, key_b)
        self.assertEqual(key_a["version"], 2)

    def test_vegetation_polygon_cache_key_tracks_asset_selection_mode(self):
        base_args = (
            "34336_26384_BI16.dds",
            1.0,
            0.9,
            2.0,
            2.1,
            64,
            64,
            ("veg-cache",),
            {"mask": "cache-key"},
            5,
            2,
            10.0,
            1.5,
            None,
            10.0,
            "northsouth",
            "Csa",
        )

        climate_key = SFR_VEG._dds_polygon_cache_key(
            *base_args,
            asset_selection_mode="climate",
            gfv2_type_source_sig=None,
        )
        proximity_key = SFR_VEG._dds_polygon_cache_key(
            *base_args,
            asset_selection_mode="gfv2_proximity",
            gfv2_type_source_sig=(2, 7, 0.123, 456),
        )

        self.assertNotEqual(climate_key, proximity_key)
        self.assertEqual(climate_key["version"], 3)

    def test_vegetation_optional_mask_or_handles_missing_masks(self):
        lhs = None
        rhs = np.zeros((3, 3), dtype=np.uint8)
        rhs[1, 1] = 1
        self.assertIs(SFR_VEG._or_optional_masks(lhs, rhs), rhs)

        lhs = np.zeros((3, 3), dtype=np.uint8)
        lhs[0, 0] = 1
        combined = SFR_VEG._or_optional_masks(lhs, rhs)
        self.assertEqual(int(combined.sum()), 2)


if __name__ == "__main__":
    unittest.main()
