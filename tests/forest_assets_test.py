import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock
import contextlib


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_Forest_Assets as FOREST_ASSETS
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


if __name__ == "__main__":
    unittest.main()
