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

    def test_read_from_config_allows_equals_inside_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "name=visual.pt"
            checkpoint.write_bytes(b"")
            cfg_path = Path(tmpdir) / "Ortho4XP_+36+117.cfg"
            cfg_path.write_text(
                "\n".join(
                    [
                        "sfr_bld_o4sfr_library_assets=facades",
                        f"sfr_bld_yolo_checkpoint={checkpoint}",
                        "zone_list=[[[36.25, 117.25, 36.25, 117.5], 18, 'Arc=Custom']]",
                    ]
                ),
                encoding="utf-8",
            )

            tile = CFG.Tile(36, 117, "")
            self.assertEqual(tile.read_from_config(config_file=str(cfg_path)), 1)

            self.assertEqual(tile.sfr_bld_asset_mode, "facades")
            self.assertEqual(tile.sfr_bld_yolo_checkpoint, str(checkpoint))
            self.assertEqual(tile.zone_list[0][2], "Arc=Custom")

    def test_missing_tile_model_paths_fall_back_to_global(self):
        original_global_cfg = CFG.global_cfg_file
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                yolo = Path(tmpdir) / "global_yolo.pt"
                height = Path(tmpdir) / "global_height.pt"
                tile_yolo = Path(tmpdir) / "tile_yolo.pt"
                for path in (yolo, height, tile_yolo):
                    path.write_bytes(b"")
                global_cfg = Path(tmpdir) / "Ortho4XP.cfg"
                global_cfg.write_text(
                    f"sfr_bld_yolo_checkpoint={yolo}\n"
                    f"sfr_bld_height_checkpoint={height}\n"
                    "sfr_bld_yolo_conf=0.18\n",
                    encoding="utf-8",
                )
                CFG.global_cfg_file = str(global_cfg)

                tile = CFG.Tile(36, 117, "")
                tile.build_dir = tmpdir
                cfg_path = Path(tmpdir) / "Ortho4XP_+36+117.cfg"
                cfg_path.write_text(
                    "sfr_bld_yolo_checkpoint=Z:/gone/yolo.pt\n"
                    "sfr_bld_height_checkpoint=\n"
                    "sfr_bld_yolo_conf=0.3\n",
                    encoding="utf-8",
                )
                self.assertEqual(tile.read_from_config(), 1)
                self.assertEqual(tile.sfr_bld_yolo_checkpoint, str(yolo))
                self.assertEqual(tile.sfr_bld_height_checkpoint, str(height))
                self.assertEqual(tile.sfr_bld_yolo_conf, 0.3)

                # An existing tile model path still wins over the global one.
                cfg_path.write_text(
                    f"sfr_bld_yolo_checkpoint={tile_yolo}\n", encoding="utf-8"
                )
                self.assertEqual(tile.read_from_config(), 1)
                self.assertEqual(tile.sfr_bld_yolo_checkpoint, str(tile_yolo))
        finally:
            CFG.global_cfg_file = original_global_cfg


if __name__ == "__main__":
    unittest.main()
