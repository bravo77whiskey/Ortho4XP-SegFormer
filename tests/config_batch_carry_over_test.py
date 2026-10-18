import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_Config_Utils as CFG
import O4_File_Names as FNAMES


class BatchCarryOverTests(unittest.TestCase):
    """A batch reads every tile into one Tile object (O4_Tile_Utils.build_tile_list)."""

    def test_a_setting_missing_from_the_next_tile_does_not_carry_over(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = tmp + "/"
            global_cfg = Path(tmp) / "Ortho4XP.cfg"
            global_cfg.write_text("grass_enabled=True\n")
            for (lat, lon), text in (((10, 103), "grass_full_tile=True\nveg_tree_sizes=False\n"),
                                     ((30, 114), "grass_enabled=True\n")):
                d = Path(FNAMES.build_dir(lat, lon, base))
                d.mkdir(parents=True)
                (d / f"Ortho4XP_{FNAMES.short_latlon(lat, lon)}.cfg").write_text(text)
            with mock.patch.object(CFG, "global_cfg_file", str(global_cfg)):
                tile = CFG.Tile(10, 103, base)
                tile.read_from_config()
                self.assertTrue(tile.grass_full_tile)
                self.assertFalse(tile.veg_tree_sizes)
                # The batch moves the same object on to the next tile.
                tile.lat, tile.lon = 30, 114
                tile.build_dir = FNAMES.build_dir(30, 114, base)
                tile.read_from_config()
                self.assertFalse(tile.grass_full_tile)
                self.assertEqual(tile.veg_tree_sizes, CFG.cfg_vars["veg_tree_sizes"]["default"])


if __name__ == "__main__":
    unittest.main()
