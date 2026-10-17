import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_Cfg_Vars as CFGVARS
import O4_Forest_Spacing as FSP
import O4_Overlay_Pipeline as PIPELINE
import O4_Vegetation_Overlay as VEG


NEW_VARS = (
    "veg_canopy",
    "veg_canopy_min_conf",
    "veg_canopy_seg_min_h_m",
    "veg_forest_holes",
    "veg_point_tree_max_cells",
    "veg_hole_certain_cells",
    "veg_hole_unsure_cells",
    "veg_tree_sizes",
    "veg_forest_heights",
)


class TreeSettingsTests(unittest.TestCase):
    def test_settings_are_tile_and_global_config_vars(self):
        for name in NEW_VARS:
            self.assertIn(name, CFGVARS.cfg_vars)
            self.assertIn(name, CFGVARS.list_veg_vars)
            self.assertIn(CFGVARS.global_prefix + name, CFGVARS.list_global_veg_vars)
            self.assertEqual(getattr(PIPELINE, name), CFGVARS.cfg_vars[name]["default"])

    def test_old_hole_size_still_loads_but_is_not_shown(self):
        self.assertIn("veg_forest_hole_min_m2", CFGVARS.cfg_vars)
        self.assertIn("veg_forest_hole_min_m2", CFGVARS.list_veg_legacy_vars)
        self.assertNotIn("veg_forest_hole_min_m2", CFGVARS.list_veg_vars)

    def test_defaults_match_the_overlay_constants(self):
        self.assertEqual(CFGVARS.cfg_vars["veg_canopy_min_conf"]["default"], VEG.CANOPY_MIN_CONFIDENCE)
        self.assertEqual(CFGVARS.cfg_vars["veg_canopy_seg_min_h_m"]["default"], VEG.CANOPY_SEG_MIN_H_M)
        self.assertEqual(CFGVARS.cfg_vars["veg_point_tree_max_cells"]["default"], VEG.POINT_TREE_MAX_CELLS)
        self.assertEqual(CFGVARS.cfg_vars["veg_hole_certain_cells"]["default"], VEG.HOLE_CERTAIN_CELLS)
        self.assertEqual(CFGVARS.cfg_vars["veg_hole_unsure_cells"]["default"], VEG.HOLE_UNSURE_CELLS)

    def _env_for(self, **values):
        saved = {k: getattr(PIPELINE, k) for k in values}
        try:
            for k, v in values.items():
                setattr(PIPELINE, k, v)
            return PIPELINE._veg_tree_env()
        finally:
            for k, v in saved.items():
                setattr(PIPELINE, k, v)

    def test_config_values_reach_the_overlay_switches(self):
        env = self._env_for(veg_canopy=True, veg_canopy_min_conf=0.15,
                            veg_canopy_seg_min_h_m=0.5, veg_forest_holes=True,
                            veg_point_tree_max_cells=3.0, veg_hole_certain_cells=0.5,
                            veg_hole_unsure_cells=4.0, veg_tree_sizes=True,
                            veg_forest_heights=True)
        with mock.patch.dict(os.environ, env):
            self.assertTrue(VEG._canopy_enabled())
            self.assertEqual(VEG._canopy_rule(), (2.0, 0.15))
            self.assertEqual(VEG._canopy_seg_min_h(), 0.5)
            self.assertTrue(VEG._forest_holes_enabled())
            self.assertEqual(VEG._cell_rules(), (3.0, 0.5, 4.0))
            self.assertTrue(VEG._tree_sizes_enabled())
            self.assertTrue(VEG._forest_heights_enabled())

    def test_zero_and_false_turn_things_off(self):
        env = self._env_for(veg_canopy=False, veg_canopy_min_conf=0.0,
                            veg_canopy_seg_min_h_m=0.0, veg_forest_holes=False,
                            veg_tree_sizes=False, veg_forest_heights=False)
        with mock.patch.dict(os.environ, env):
            self.assertFalse(VEG._tree_sizes_enabled())
            self.assertFalse(VEG._forest_heights_enabled())
            self.assertFalse(VEG._canopy_enabled())
            self.assertEqual(VEG._canopy_rule(), (2.0, None))   # mean height >= 2 m
            self.assertIsNone(VEG._canopy_seg_min_h())
            self.assertFalse(VEG._forest_holes_enabled())


class _Spacing:
    """Every forest on a 10 m grid: one cell = 100 m2."""

    def cell_m2(self, path):
        return 100.0


class SpacingRuleTests(unittest.TestCase):
    def _run(self, mask, gap_certain, env):
        kwargs = dict(img_w=mask.shape[1], img_h=mask.shape[0], lat_n=10.002, lat_s=10.0,
                      lon_w=104.0, lon_e=104.002, tile_lat=10, tile_lon=104, m_per_px=1.0,
                      min_area_px=1, simplify_px=1.0, region="tropical",
                      rng=np.random.default_rng(0), density_override=1.0, context_masks={},
                      forest_spacing=_Spacing(), gap_certain=gap_certain)
        with mock.patch.dict(os.environ, dict({"O4_VEG_FOREST_HOLES": "1",
                                               "O4_VEG_POINT_TREES": "1"}, **env)):
            return VEG._process_dds_mask(mask, VEG.SEGFORMER.CLASS_TREE, **kwargs)

    def test_point_trees_below_two_cells_areas_above(self):
        mask = np.zeros((200, 200), np.uint8)
        mask[10:24, 10:24] = 1          # 196 m2 < 2 cells -> single trees
        mask[100:116, 100:116] = 1      # 256 m2 > 2 cells -> forest area
        polys = self._run(mask, None, {"O4_VEG_SPACING_RULES": "1"})
        modes = sorted(bool(den & VEG.POINT_MODE) for _p, den, _r in polys)
        self.assertEqual(modes, [False, True])
        fixed = self._run(mask, None, {"O4_VEG_SPACING_RULES": "0"})
        self.assertTrue(all(den & VEG.POINT_MODE for _p, den, _r in fixed))   # both < 600 m2

    def test_holes_need_one_cell_if_known_two_if_not(self):
        mask = np.zeros((300, 300), np.uint8)
        mask[20:280, 20:280] = 1
        mask[50:62, 50:62] = 0          # 144 m2 gap, known cause -> hole
        mask[150:162, 150:162] = 0      # 144 m2 gap, unknown cause -> stays forest
        mask[200:216, 50:66] = 0        # 256 m2 gap, unknown cause -> hole
        certain = np.zeros_like(mask, bool)
        certain[50:62, 50:62] = True
        polys = self._run(mask, certain, {"O4_VEG_SPACING_RULES": "1"})
        area = [ring for _p, den, ring in polys if not den & VEG.POINT_MODE]
        self.assertEqual(len(area), 1)
        self.assertEqual(len(area[0].holes), 2)


class ForestSpacingParseTests(unittest.TestCase):
    def test_reads_season_exports_and_paths_with_spaces(self):
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            cs = os.path.join(root, "Custom Scenery")
            pack = os.path.join(cs, "pack")
            forests = os.path.join(root, "Resources", "default scenery", "1200 forests", "sum")
            os.makedirs(pack)
            os.makedirs(forests)
            with open(os.path.join(forests, "a.for"), "w") as f:
                f.write("A\n800\nFOREST\nSPACING\t9 9\nRANDOM\t4 4\n")
            with open(os.path.join(pack, "library.txt"), "w") as f:
                f.write("A\n800\nLIBRARY\n"
                        "EXPORT_SEASON sum\tlib/x/a.for\t\t1200 forests/sum/a.for\n")
            sp = FSP.ForestSpacing(cs)
            self.assertEqual(sp.spacing("lib/x/a.for"), (9.0, 9.0))
            self.assertEqual(sp.cell_m2("lib/x/a.for"), 81.0)
            self.assertEqual(sp.spacing("lib/x/missing.for"),
                             (FSP.DEFAULT_SPACING_M, FSP.DEFAULT_SPACING_M))


if __name__ == "__main__":
    unittest.main()
