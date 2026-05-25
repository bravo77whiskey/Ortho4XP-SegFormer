import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_Config_Utils as CFG


class ConfigAliasTests(unittest.TestCase):
    def test_legacy_disable_cache_keys_map_to_new_names(self):
        self.assertEqual(
            CFG.normalize_config_entry("sfr_bld_del", "True"),
            ("sfr_bld_disable_cache", "True"),
        )
        self.assertEqual(
            CFG.normalize_config_entry("sfr_veg_del", "False"),
            ("sfr_veg_disable_cache", "False"),
        )

    def test_global_legacy_disable_cache_keys_map_to_new_names(self):
        self.assertEqual(
            CFG.normalize_config_entry("global_sfr_bld_del", "True"),
            ("global_sfr_bld_disable_cache", "True"),
        )
        self.assertEqual(
            CFG.normalize_config_entry("global_sfr_veg_del", "False"),
            ("global_sfr_veg_disable_cache", "False"),
        )

    def test_legacy_kernel_radius_keys_still_convert_to_metres(self):
        self.assertEqual(
            CFG.normalize_config_entry("sfr_bld_close_k", "15"),
            ("sfr_bld_close_m", "30.0"),
        )
        self.assertEqual(
            CFG.normalize_config_entry("global_sfr_bld_open_k", "5"),
            ("global_sfr_bld_open_m", "10.0"),
        )

    def test_tile_defaults_inherit_current_global_cache_settings(self):
        old_bld = CFG.global_sfr_bld_disable_cache
        old_veg = CFG.global_sfr_veg_disable_cache
        try:
            CFG.global_sfr_bld_disable_cache = True
            CFG.global_sfr_veg_disable_cache = True
            tile = CFG.Tile(12, 34, "")
            self.assertTrue(tile.sfr_bld_disable_cache)
            self.assertTrue(tile.sfr_veg_disable_cache)
        finally:
            CFG.global_sfr_bld_disable_cache = old_bld
            CFG.global_sfr_veg_disable_cache = old_veg

    def test_gfv2_asset_proximity_defaults_to_climate_selection(self):
        self.assertFalse(
            CFG.cfg_tile_vars["sfr_veg_use_gfv2_asset_proximity"]["default"]
        )
        self.assertFalse(CFG.global_sfr_veg_use_gfv2_asset_proximity)
        tile = CFG.Tile(12, 34, "")
        self.assertFalse(tile.sfr_veg_use_gfv2_asset_proximity)

    def test_yolo_checkpoint_uses_file_picker_metadata(self):
        tile_var = CFG.cfg_tile_vars["sfr_bld_yolo_checkpoint"]
        global_var = CFG.cfg_global_tile_vars["global_sfr_bld_yolo_checkpoint"]

        self.assertTrue(tile_var["file_picker"])
        self.assertEqual(tile_var["file_picker_title"], "Choose YOLO model checkpoint")
        self.assertIn(("All files", "*.*"), tile_var["filetypes"])
        self.assertTrue(global_var["file_picker"])
        self.assertEqual(global_var["filetypes"], tile_var["filetypes"])


if __name__ == "__main__":
    unittest.main()
