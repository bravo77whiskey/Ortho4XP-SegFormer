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

    def test_allowlist_contains_only_simheaven_assets(self):
        approved = set(FOREST_ASSETS.APPROVED_GENERATED_FOREST_PATHS)
        self.assertEqual(FOREST_ASSETS.HEIGHT_CUTOFF_METERS, 24.0)
        self.assertEqual(
            approved,
            {
                "simheaven/forests/broad.for",
                "simheaven/forests/coni.for",
                "simheaven/forests/mixed.for",
            },
        )

    def test_allowlist_excludes_known_tall_default_assets(self):
        approved = set(FOREST_ASSETS.APPROVED_GENERATED_FOREST_PATHS)
        for path in FOREST_ASSETS.KNOWN_TALL_DEFAULT_FOREST_PATHS:
            self.assertNotIn(path, approved)

    def test_tree_candidates_are_simheaven_only(self):
        candidates = SFR_VEG._short_tree_candidates("subtropical", 75)
        self.assertTrue(candidates)
        self.assertTrue(all(path.startswith("simheaven/") for path in candidates))
        northnorth = SFR_VEG._short_tree_candidates("northnorth", 75)
        self.assertIn(
            "simheaven/forests/mixed.for",
            northnorth,
        )
        self.assertTrue(all(path.startswith("simheaven/") for path in northnorth))

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

    def test_tree_selection_uses_simheaven_assets(self):
        region = "northsouth"
        dlevel = 75
        candidates = FOREST_ASSETS.short_tree_candidates(region, dlevel)
        weights = FOREST_ASSETS.tree_candidate_weights(candidates)
        self.assertEqual(
            candidates,
            (
                "simheaven/forests/broad.for",
                "simheaven/forests/mixed.for",
            ),
        )
        self.assertEqual(weights, (15, 11))
        for rng_index in range(20):
            choice = FOREST_ASSETS.choose_tree_path(
                region,
                dlevel,
                _IndexRng(rng_index),
                context="managed",
            )
            self.assertIn(choice, candidates)

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

    def test_dense_area_tree_polygons_use_simheaven_assets_only(self):
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
            self.assertTrue(path.startswith("simheaven/"), msg=path)

    def test_gfv2_path_parser_remains_available_for_legacy_alias_installer(self):
        self.assertEqual(
            FOREST_ASSETS.parse_gfv2_path(
                "forests/tropical/woodland/tropical_woodland_75_y1.for"
            )["family"],
            "woodland",
        )

    def test_simheaven_tree_path_normalization_rejects_non_tree_families(self):
        self.assertEqual(
            FOREST_ASSETS.normalize_simheaven_tree_path(
                r"SIMHEAVEN\FORESTS\MIXED.FOR"
            ),
            "simheaven/forests/mixed.for",
        )
        for path in (
            "simheaven/forests/orchard.for",
            "simheaven/forests/vineyard.for",
            "simheaven/forests/shrubs.for",
            "simheaven/forests/wetland.for",
        ):
            self.assertIsNone(
                FOREST_ASSETS.normalize_simheaven_tree_path(path)
            )

    def test_sfr_for_entry_uses_nearby_simheaven_tree_family(self):
        for rng_index in range(8):
            path, _density = SFR_VEG._for_entry(
                SEGFORMER.CLASS_TREE,
                0.72,
                "area",
                "northnorth",
                _IndexRng(rng_index),
                density_override=None,
                veg_type="natural_forest_closed",
                simheaven_type_path="simheaven/forests/mixed.for",
            )
            self.assertEqual(path, "simheaven/forests/mixed.for")

    def test_dominant_simheaven_type_picks_most_common_tree_path(self):
        records = [
            {"path": "simheaven/forests/broad.for"},
            {"path": "simheaven/forests/mixed.for"},
            {"path": "simheaven/forests/mixed.for"},
            {"path": "simheaven/forests/orchard.for"},
        ]

        self.assertEqual(
            SFR_VEG._dominant_simheaven_tree_path(records),
            "simheaven/forests/mixed.for",
        )

    def test_dominant_simheaven_type_skips_non_tree_sources(self):
        records = [
            {"path": "simheaven/forests/orchard.for"},
            {"path": "simheaven/forests/vineyard.for"},
            {"path": "simheaven/forests/shrubs.for"},
            {"path": "simheaven/forests/wetland.for"},
            {"path": "simheaven/forests/coni.for"},
        ]

        self.assertEqual(
            SFR_VEG._dominant_simheaven_tree_path(records),
            "simheaven/forests/coni.for",
        )

    def test_dominant_simheaven_type_tie_breaks_by_path(self):
        records = [
            {"path": "simheaven/forests/mixed.for"},
            {"path": "simheaven/forests/broad.for"},
        ]

        self.assertEqual(
            SFR_VEG._dominant_simheaven_tree_path(records),
            "simheaven/forests/broad.for",
        )

    def test_dominant_simheaven_type_returns_none_without_tree_sources(self):
        records = [
            {"path": "simheaven/forests/orchard.for"},
            {"path": "simheaven/forests/shrubs.for"},
            {"path": None},
        ]

        self.assertIsNone(SFR_VEG._dominant_simheaven_tree_path(records))

    def test_climate_mode_does_not_call_simheaven_type_hint_mapping(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[2:18, 2:18] = 255

        with mock.patch.object(
            FOREST_ASSETS,
            "simheaven_type_hint_candidates",
            side_effect=AssertionError("simHeaven type mapping should be disabled"),
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
                simheaven_type_path=None,
            )

        self.assertTrue(polygons)
        self.assertTrue(polygons[0][0].startswith("simheaven/"))

    def test_process_dds_mask_uses_simheaven_type_hint(self):
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[2:16, 2:16] = 255
        mask[22:36, 22:36] = 255

        polygons = SFR_VEG._process_dds_mask(
            mask,
            SEGFORMER.CLASS_TREE,
            img_w=40,
            img_h=40,
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
            simheaven_type_path="simheaven/forests/coni.for",
        )

        self.assertEqual(len(polygons), 2)
        for path, _density, _ring in polygons:
            self.assertEqual(path, "simheaven/forests/coni.for")

    def test_process_dds_mask_asks_resolver_for_each_tree_polygon(self):
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[2:16, 2:16] = 255
        mask[22:36, 22:36] = 255
        resolver = mock.Mock()
        resolver.resolve.return_value = "simheaven/forests/mixed.for"

        polygons = SFR_VEG._process_dds_mask(
            mask,
            SEGFORMER.CLASS_TREE,
            img_w=40,
            img_h=40,
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
            simheaven_type_path="simheaven/forests/broad.for",
            simheaven_type_resolver=resolver,
        )

        self.assertEqual(len(polygons), 2)
        self.assertEqual(resolver.resolve.call_count, 2)
        self.assertTrue(
            all(path == "simheaven/forests/mixed.for" for path, _, _ in polygons)
        )

    def test_cli_uses_simheaven_asset_proximity_by_default(self):
        argv = [
            "generate_veg_overlay.py",
            "textures",
            "1",
            "2",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = SFR_VEG.parse_args()

        self.assertFalse(args.no_simheaven_asset_proximity)

    def test_managed_tree_context_uses_simheaven(self):
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
            self.assertTrue(path.startswith("simheaven/"), msg=path)

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
        self.assertEqual(key_a["version"], 3)

    def test_vegetation_aux_cache_key_tracks_exclusion_zones(self):
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
            ("mesh-a", 10, 20),
        )

        key_a = SFR_VEG._dds_mask_cache_key(*base_args, excl_zone_sig=(0, 0, 0.0, 0))
        key_b = SFR_VEG._dds_mask_cache_key(*base_args, excl_zone_sig=(1, 4, 3.5, 7))

        self.assertNotEqual(key_a, key_b)

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
            asset_selection_mode="simheaven_climate_v1",
            forest_type_source_path=None,
        )
        dominant_key = SFR_VEG._dds_polygon_cache_key(
            *base_args,
            asset_selection_mode="simheaven_tile_dominant",
            forest_type_source_path="simheaven/forests/broad.for",
        )
        other_dominant_key = SFR_VEG._dds_polygon_cache_key(
            *base_args,
            asset_selection_mode="simheaven_tile_dominant",
            forest_type_source_path="simheaven/forests/mixed.for",
        )

        closest_key = SFR_VEG._dds_polygon_cache_key(
            *base_args,
            asset_selection_mode="simheaven_closest_v1",
            forest_type_source_path="simheaven/forests/broad.for",
            forest_type_sig=(12, 34, 5.6, 78),
        )
        covered_key = SFR_VEG._dds_polygon_cache_key(
            *base_args,
            asset_selection_mode="simheaven_climate_v1",
            forest_type_source_path=None,
            covered_fracs=((0.5, 0.5, 1.0, 1.0),),
        )

        self.assertNotEqual(climate_key, dominant_key)
        self.assertNotEqual(dominant_key, other_dominant_key)
        self.assertNotEqual(dominant_key, closest_key)
        self.assertNotEqual(climate_key, covered_key)
        self.assertEqual(climate_key["version"], 7)

    def test_dominant_simheaven_path_ignores_non_tree_votes(self):
        records = [
            {"path": "simheaven/forests/orchard.for"},
        ] * 50 + [
            {"path": "simheaven/forests/mixed.for"},
        ] * 3 + [
            {"path": "simheaven/forests/broad.for"},
        ] * 2
        self.assertEqual(
            SFR_VEG._dominant_simheaven_tree_path(records),
            "simheaven/forests/mixed.for",
        )

    def test_dominant_simheaven_path_none_when_only_non_tree_sources(self):
        records = [
            {"path": "simheaven/forests/orchard.for"},
        ] * 10
        self.assertIsNone(SFR_VEG._dominant_simheaven_tree_path(records))

    def test_simheaven_type_resolver_prefers_nearest_tree_polygon(self):
        records = [
            {
                "path": "simheaven/forests/broad.for",
                "_bounds": (23.09, 23.11, 119.09, 119.11),
            },
            {
                "path": "simheaven/forests/coni.for",
                "_bounds": (23.49, 23.51, 119.49, 119.51),
            },
            {
                "path": "simheaven/forests/orchard.for",
                "_bounds": (23.0, 23.3, 119.0, 119.3),
            },
        ]
        resolver = SFR_VEG._SimHeavenTypeResolver(
            records,
            dominant_path="simheaven/forests/mixed.for",
            tile_lat=23,
        )
        self.assertEqual(resolver.n, 2)
        self.assertEqual(
            resolver.resolve(119.10, 23.10),
            "simheaven/forests/broad.for",
        )
        self.assertEqual(
            resolver.resolve(119.50, 23.50),
            "simheaven/forests/coni.for",
        )

    def test_simheaven_type_resolver_falls_back_to_dominant_beyond_radius(self):
        records = [
            {
                "path": "simheaven/forests/broad.for",
                "_bounds": (23.0, 23.001, 119.0, 119.001),
            },
        ]
        resolver = SFR_VEG._SimHeavenTypeResolver(
            records,
            dominant_path="simheaven/forests/mixed.for",
            tile_lat=23,
            max_radius_m=1000.0,
        )
        # ~50 km away -> dominant fallback
        self.assertEqual(
            resolver.resolve(119.5, 23.5),
            "simheaven/forests/mixed.for",
        )

    def test_simheaven_type_resolver_empty_records_returns_dominant(self):
        resolver = SFR_VEG._SimHeavenTypeResolver(
            [],
            "simheaven/forests/mixed.for",
            tile_lat=23,
        )
        self.assertEqual(
            resolver.resolve(119.1, 23.1),
            "simheaven/forests/mixed.for",
        )

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
