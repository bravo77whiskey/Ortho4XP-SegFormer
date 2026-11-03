"""Option G routing and native RGB regression checks, with no model loading."""
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import O4_Veg_Lowzoom as LOW
import O4_Vegetation_Overlay as VEG


class LowzoomMaskTests(unittest.TestCase):
    def test_zl16_and_up_use_the_merge(self):
        self.assertEqual([z for z in range(12, 22) if LOW.applies(z)], list(range(16, 22)))

    def test_fill_matches_reviewed_f_plus_segformer_forest(self):
        sys.path.insert(0, str(ROOT / "experiments" / "veg_rgb_lowzoom_20261020"))
        from research import detect
        from refine_direction import refine
        rng = np.random.default_rng(81)
        rgb = np.clip(rng.normal(0, 27, (256, 256, 1)) + [47, 85, 38], 0, 255).astype(np.uint8)
        classes = np.tile(np.arange(8, dtype=np.uint8), (256, 32))
        for mpp in (1.2, 2.4):
            f = refine(rgb, mpp, detect(rgb, mpp)["texture"])
            self.assertGreater(int(f.sum()), 1000)
            expected = f | (classes == 5)
            result = LOW.tree_mask(rgb, classes, mpp, min_area_m2=60)
            np.testing.assert_array_equal(result, expected)
            self.assertEqual(result.dtype, np.uint8)

    def test_rgb_plants_on_any_segformer_class(self):
        rng = np.random.default_rng(81)
        rgb = np.clip(rng.normal(0, 27, (256, 256, 1)) + [47, 85, 38], 0, 255).astype(np.uint8)
        classes = np.full((256, 256), 7, np.uint8)
        self.assertGreater(int(LOW.tree_mask(rgb, classes, 2.4).sum()), 1000)

    def test_area_limit_is_one_square_metre(self):
        mask = np.zeros((16, 16), bool)
        mask[1:3, 1:3] = True   # 4 px = 1 m2 at 0.5 m/px
        mask[8, 8] = True       # 1 px = 0.25 m2
        np.testing.assert_array_equal(LOW._remove_small(mask, 0.5, LOW.MIN_AREA_M2), mask & ~(
            np.arange(16)[:, None] == 8))
        self.assertTrue(LOW._remove_small(mask, 2.2, LOW.MIN_AREA_M2)[8, 8])

    def test_smooth_green_and_parallel_rows_are_not_canopy(self):
        for row_pattern in (False, True):
            rgb = np.full((256, 256, 3), [45, 90, 35], np.uint8)
            if row_pattern:
                rgb[::4] = [25, 50, 20]
            classes = np.zeros((256, 256), np.uint8)
            self.assertFalse(LOW.tree_mask(rgb, classes, 2.4).any())

    def test_invalid_native_grid_or_scale_fails(self):
        rgb = np.zeros((32, 32, 3), np.uint8)
        with self.assertRaisesRegex(ValueError, "same native"):
            LOW.tree_mask(rgb, np.zeros((16, 16)), 2)
        for mpp in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                LOW.tree_mask(rgb, np.zeros((32, 32)), mpp)


class LowzoomPipelineTests(unittest.TestCase):
    def run_texture(self, zl, *, res_m=None, missing_rgb=False, extra_gap=False, airport=False):
        """Capture the mask handed to production polygonization."""
        seg = VEG.SEGFORMER
        name = f"34336_26384_BI{zl}.dds"
        shape = (128, 128)
        classes = np.full(shape, seg.CLASS_TREE, np.int8)
        classes[40:55, 40:55] = seg.CLASS_BUILDING
        classes[60:80, 60:80] = seg.CLASS_RANGELAND
        rgb = np.full((*shape, 3), [40, 80, 30], np.uint8)
        g = np.zeros(shape, np.uint8)
        g[12:110, 12:110] = 1
        g[60:80, 60:80] = 0  # G rejects a hole within a model forest.
        if extra_gap:
            g[85:95, 85:95] = 0
        captured, cache_keys = [], []
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            np.save(Path(tmp) / name.replace(".dds", "_veg.npy"), classes)
            # ZL18/19 normally infer at ZL17.
            np.save(Path(tmp) / name.replace(".dds", "_z17_veg.npy"), classes)
            def patch(obj, key, **kwargs):
                return stack.enter_context(mock.patch.object(obj, key, **kwargs))
            patch(seg, "collect_source_texture_files", return_value=([name], "dds", None))
            patch(VEG.TEXSEL, "select_textures", return_value=([name], {}, {}))
            patch(VEG, "dds_bounds", return_value=(47.003, 47.0, 8.0, 8.004))
            patch(seg, "load_dds_or_none", return_value=None if missing_rgb else rgb)
            # A cached native map must not require support or enlarged inference.
            patch(seg, "tree_support_enabled", return_value=zl in (16, 17))
            patch(VEG, "_texture_density_enabled", return_value=False)
            patch(seg, "load_vegetation_model", side_effect=AssertionError("unexpected inference"))
            patch(VEG, "_upscaled_trees", side_effect=AssertionError("unexpected upscale"))
            patch(VEG, "_canopy_enabled", return_value=airport)
            if airport:
                maps = np.zeros((1 + len(VEG.CANOPY.LEVELS_M), *shape), np.float32)
                maps[0, 60:80, 60:80] = 5  # Canopy sees trees in the RGB hole.
                np.save(Path(tmp) / name.replace(".dds", "_vegcanopy.npy"), maps)
                inside = np.zeros(shape, bool)
                inside[12:30, 12:30] = True
                patch(VEG, "_canopy_airports_only", return_value=True)
                patch(VEG, "_airport_inside_mask", return_value=inside)
                patch(VEG, "_airports_signature", return_value=("airport",))
            patch(VEG, "_load_mesh_water_index", return_value=None)
            patch(VEG, "_load_osm_roads", return_value=[])
            patch(VEG, "_load_dds_mask_cache", return_value={
                "context_masks": {"forest": np.ones(shape, np.uint8)}} if res_m is None else None)
            patch(VEG, "_osm_forest_rescue_enabled", return_value=True)
            patch(VEG, "_osm_rescue_rangeland_enabled", return_value=True)
            patch(VEG, "_canopy_look", return_value=np.ones(shape, bool))
            patch(VEG, "_spacing_rules_enabled", return_value=False)
            calc = patch(LOW, "tree_mask", return_value=g)
            patch(VEG, "_load_dds_polygon_cache", side_effect=lambda path, key: cache_keys.append(key))
            patch(VEG, "_process_tree_mask",
                  side_effect=lambda mask, **kwargs: captured.append(mask.copy()) or [])
            stack.enter_context(redirect_stdout(StringIO()))
            VEG.run(tmp, 47, 8, str(Path(tmp) / "overlay.dsf"), tmp,
                    10, 3, 50, 3, False, res_m=res_m,
                    use_simheaven=False, use_simheaven_asset_proximity=False,
                    avoid_simheaven_buildings=False, avoid_simheaven_forests=False,
                    use_leaf_type_maps=False, use_xp_forests=False, bld_excl_m=0)
        return calc, captured, cache_keys, g

    def test_merge_keeps_mask_gaps_and_applies_building_exclusions(self):
        for zl in (16, 17):
            calc, masks, keys, g = self.run_texture(zl)
            calc.assert_called_once()
            result = masks[0]
            self.assertFalse(result[40:55, 40:55].any())
            self.assertFalse(result[60:80, 60:80].any())
            self.assertTrue(result[20:30, 20:30].all())
            self.assertFalse((result & ~g).any())
            self.assertEqual(keys[0]["lowzoom_sig"][0], LOW.SIGNATURE)

    def test_resampling_happens_after_native_g(self):
        calc, masks, _, g = self.run_texture(16, res_m=8)
        self.assertEqual(calc.call_args.args[0].shape[:2], (128, 128))
        self.assertEqual(calc.call_args.args[1].shape, (128, 128))
        self.assertEqual(masks[0].shape, (64, 64))
        small = cv2.resize(g, (64, 64), interpolation=cv2.INTER_NEAREST)
        self.assertFalse((masks[0] & ~small).any())

    def test_canopy_adds_trees_outside_airports_and_decides_inside(self):
        for zl in (18, 19):
            _, masks, _, _ = self.run_texture(zl, airport=True)
            self.assertFalse(masks[0][16:26, 16:26].any())
            self.assertTrue(masks[0][32:38, 32:38].all())
            self.assertTrue(masks[0][65:75, 65:75].all())

    def test_missing_rgb_does_not_fall_back_to_solid_model_forest(self):
        calc, masks, keys, _ = self.run_texture(16, missing_rgb=True)
        calc.assert_not_called()
        self.assertEqual(masks, [])
        self.assertEqual(keys, [])

    def test_rgb_mask_changes_invalidate_polygon_cache(self):
        _, _, first, _ = self.run_texture(16)
        _, _, changed, _ = self.run_texture(16, extra_gap=True)
        self.assertNotEqual(first[0]["lowzoom_sig"], changed[0]["lowzoom_sig"])


if __name__ == "__main__":
    unittest.main()
