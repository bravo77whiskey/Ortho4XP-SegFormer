import ast
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_Config_Utils as CFG


class ConfigZoneListTests(unittest.TestCase):
    def test_write_to_config_uses_tile_zone_list_when_global_zone_list_is_empty(self):
        original_zone_list = CFG.zone_list
        tile_zone = [[36.25, 117.25, 36.25, 117.5, 36.5, 117.5, 36.5, 117.25, 36.25, 117.25], 18, "BI"]

        try:
            CFG.zone_list = []
            with tempfile.TemporaryDirectory() as tmpdir:
                tile = CFG.Tile(36, 117, "")
                tile.build_dir = tmpdir
                tile.zone_list = [tile_zone]

                self.assertEqual(tile.write_to_config(), 1)

                cfg_path = Path(tmpdir) / "Ortho4XP_+36+117.cfg"
                values = dict(
                    line.strip().split("=", 1)
                    for line in cfg_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )

                self.assertEqual(ast.literal_eval(values["zone_list"]), [tile_zone])
        finally:
            CFG.zone_list = original_zone_list

    def test_write_to_config_preserves_existing_zones_when_tile_has_no_zones_loaded(self):
        original_zone_list = CFG.zone_list
        tile_zone = [[36.25, 117.25, 36.25, 117.5, 36.5, 117.5, 36.5, 117.25, 36.25, 117.25], 18, "BI"]

        try:
            CFG.zone_list = []
            with tempfile.TemporaryDirectory() as tmpdir:
                cfg_path = Path(tmpdir) / "Ortho4XP_+36+117.cfg"
                cfg_path.write_text(f"zone_list={[tile_zone]}\n", encoding="utf-8")

                tile = CFG.Tile(36, 117, "")
                tile.build_dir = tmpdir
                tile.zone_list = []

                self.assertEqual(tile.write_to_config(), 1)

                values = dict(
                    line.strip().split("=", 1)
                    for line in cfg_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )

                self.assertEqual(ast.literal_eval(values["zone_list"]), [tile_zone])
        finally:
            CFG.zone_list = original_zone_list


if __name__ == "__main__":
    unittest.main()
