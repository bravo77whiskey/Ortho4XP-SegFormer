import os
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_Cfg_Vars as CFGVARS
import O4_Overlay_Pipeline as PIPELINE
import O4_Vegetation_Overlay as VEG


NEW_VARS = (
    "veg_canopy",
    "veg_canopy_min_conf",
    "veg_canopy_seg_min_h_m",
    "veg_forest_holes",
    "veg_forest_hole_min_m2",
)


class TreeSettingsTests(unittest.TestCase):
    def test_settings_are_tile_and_global_config_vars(self):
        for name in NEW_VARS:
            self.assertIn(name, CFGVARS.cfg_vars)
            self.assertIn(name, CFGVARS.list_veg_vars)
            self.assertIn(CFGVARS.global_prefix + name, CFGVARS.list_global_veg_vars)
            self.assertEqual(getattr(PIPELINE, name), CFGVARS.cfg_vars[name]["default"])

    def test_defaults_match_the_overlay_constants(self):
        self.assertEqual(CFGVARS.cfg_vars["veg_canopy_min_conf"]["default"], VEG.CANOPY_MIN_CONFIDENCE)
        self.assertEqual(CFGVARS.cfg_vars["veg_canopy_seg_min_h_m"]["default"], VEG.CANOPY_SEG_MIN_H_M)
        self.assertEqual(CFGVARS.cfg_vars["veg_forest_hole_min_m2"]["default"], VEG.FOREST_HOLE_MIN_AREA_M2)

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
                            veg_forest_hole_min_m2=500.0)
        with mock.patch.dict(os.environ, env):
            self.assertTrue(VEG._canopy_enabled())
            self.assertEqual(VEG._canopy_rule(), (2.0, 0.15))
            self.assertEqual(VEG._canopy_seg_min_h(), 0.5)
            self.assertTrue(VEG._forest_holes_enabled())
            self.assertEqual(VEG._forest_hole_min_m2(), 500.0)

    def test_zero_and_false_turn_things_off(self):
        env = self._env_for(veg_canopy=False, veg_canopy_min_conf=0.0,
                            veg_canopy_seg_min_h_m=0.0, veg_forest_holes=False,
                            veg_forest_hole_min_m2=200.0)
        with mock.patch.dict(os.environ, env):
            self.assertFalse(VEG._canopy_enabled())
            self.assertEqual(VEG._canopy_rule(), (2.0, None))   # mean height >= 2 m
            self.assertIsNone(VEG._canopy_seg_min_h())
            self.assertFalse(VEG._forest_holes_enabled())


if __name__ == "__main__":
    unittest.main()
