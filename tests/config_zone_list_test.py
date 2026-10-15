import ast
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


class ConfigZoneListTests(unittest.TestCase):
    def test_gui_load_and_save_fold_legacy_zones_into_the_editable_list(self):
        import O4_File_Names as FNAMES
        import O4_GUI_Utils as GUI

        def value(initial):
            var = mock.Mock()
            var.get.return_value = str(initial)
            var.set.side_effect = lambda new: setattr(var.get, "return_value", str(new))
            return var

        enclosing = [[35.9, 116.9, 37.1, 116.9, 37.1, 118.1, 35.9, 118.1, 35.9, 116.9], 19, "BI"]
        with tempfile.TemporaryDirectory() as tmpdir:
            base = tmpdir + "/"
            build_dir = Path(FNAMES.build_dir(36, 117, base))
            build_dir.mkdir(parents=True)
            cfg = build_dir / "Ortho4XP_+36+117.cfg"
            cfg.write_text(f"airport_zone_list={[enclosing]}\nzone_list=[]\n")
            parent = mock.Mock()
            parent.custom_build_dir_entry = value(base)
            parent.get_lat_lon.return_value = (36, 117)
            parent.lat = value(36)
            parent.lon = value(117)
            parent.default_website = value("BI")
            parent.default_zl = value(16)
            window = mock.Mock()
            window.parent = parent
            window.v_ = {key: value(str(info["default"])) for key, info in CFG.cfg_vars.items()}
            window.reference_year_ok.return_value = True
            snapshot = {key: getattr(CFG, key) for key in CFG.list_tile_vars}
            with mock.patch.multiple(CFG, **snapshot), mock.patch.object(CFG, "zone_list", []):
                GUI.Ortho4XP_GUI.load_tile_cfg(parent, 36, 117)
                self.assertEqual(CFG.zone_list, [enclosing])
                CFG.Ortho4XP_Config.load_tile_cfg(window)
                self.assertEqual(ast.literal_eval(window.v_["zone_list"].get()), [enclosing])
                backup = Path(str(cfg) + ".bak")
                backup.write_text(cfg.read_text())
                window.v_["zone_list"].set("[]")
                CFG.Ortho4XP_Config.load_backup_tile_cfg(window)
                self.assertEqual(ast.literal_eval(window.v_["zone_list"].get()), [enclosing])
                CFG.Ortho4XP_Config.write_tile_cfg(window)
                values = dict(line.split("=", 1) for line in cfg.read_text().splitlines())
                self.assertEqual(ast.literal_eval(values["zone_list"]), [enclosing])
                self.assertNotIn("airport_zone_list", values)

    def test_legacy_airport_zones_migrate_to_zone_list_in_either_key_order(self):
        drawn = [[36.2, 117.2, 36.4, 117.2, 36.4, 117.4, 36.2, 117.2], 18, "GO2"]
        airport = [[36.1, 117.1, 36.5, 117.1, 36.5, 117.5, 36.1, 117.1], 19, "BI"]
        entries = [f"zone_list={[drawn]}\n", f"airport_zone_list={[airport]}\n"]
        for lines in (entries, entries[::-1]):
            with self.subTest(lines=lines), tempfile.TemporaryDirectory() as tmpdir:
                cfg = Path(tmpdir) / "Ortho4XP_+36+117.cfg"
                cfg.write_text("".join(lines), encoding="utf-8")
                tile = CFG.Tile(36, 117, "")
                tile.build_dir = tmpdir
                self.assertEqual(tile.read_from_config(config_file=str(cfg)), 1)
                self.assertEqual(tile.zone_list, [airport, drawn])
                self.assertFalse(hasattr(tile, "airport_zone_list"))
                self.assertEqual(tile.write_to_config(), 1)
                self.assertNotIn("airport_zone_list=", cfg.read_text())
                self.assertEqual(tile.read_from_config(config_file=str(cfg)), 1)
                self.assertEqual(tile.zone_list, [airport, drawn])

    def test_save_keeps_zones_enclosing_a_tile_in_all_hemispheres(self):
        for lat, lon in ((23, 120), (-23, 120), (23, -120), (-23, -120)):
            with self.subTest(lat=lat, lon=lon), tempfile.TemporaryDirectory() as tmpdir:
                coords = [lat - .1, lon - .1, lat + 1.1, lon - .1,
                          lat + 1.1, lon + 1.1, lat - .1, lon + 1.1, lat - .1, lon - .1]
                enclosing = [coords, 17, "BI"]
                elsewhere = [[value + 10 for value in coords], 18, "BI"]
                tile = CFG.Tile(lat, lon, "")
                tile.build_dir = tmpdir
                tile.zone_list = [enclosing, elsewhere]
                self.assertEqual(tile.write_to_config(), 1)
                cfg = Path(tmpdir) / f"Ortho4XP_{lat:+03d}{lon:+04d}.cfg"
                values = dict(line.split("=", 1) for line in cfg.read_text().splitlines())
                self.assertEqual(ast.literal_eval(values["zone_list"]), [enclosing])

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
                        "bld_o4sfr_library_assets=facades",
                        f"bld_yolo_checkpoint={checkpoint}",
                        "zone_list=[[[36.25, 117.25, 36.25, 117.5], 18, 'Arc=Custom']]",
                    ]
                ),
                encoding="utf-8",
            )

            tile = CFG.Tile(36, 117, "")
            self.assertEqual(tile.read_from_config(config_file=str(cfg_path)), 1)

            self.assertEqual(tile.bld_asset_mode, "facades")
            self.assertEqual(tile.bld_yolo_checkpoint, str(checkpoint))
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
                    f"bld_yolo_checkpoint={yolo}\n"
                    f"bld_height_checkpoint={height}\n"
                    "bld_yolo_conf=0.18\n",
                    encoding="utf-8",
                )
                CFG.global_cfg_file = str(global_cfg)

                tile = CFG.Tile(36, 117, "")
                tile.build_dir = tmpdir
                cfg_path = Path(tmpdir) / "Ortho4XP_+36+117.cfg"
                cfg_path.write_text(
                    "bld_yolo_checkpoint=Z:/gone/yolo.pt\n"
                    "bld_height_checkpoint=\n"
                    "bld_yolo_conf=0.3\n",
                    encoding="utf-8",
                )
                self.assertEqual(tile.read_from_config(), 1)
                self.assertEqual(tile.bld_yolo_checkpoint, str(yolo))
                self.assertEqual(tile.bld_height_checkpoint, str(height))
                self.assertEqual(tile.bld_yolo_conf, 0.3)

                # An existing tile model path still wins over the global one.
                cfg_path.write_text(
                    f"bld_yolo_checkpoint={tile_yolo}\n", encoding="utf-8"
                )
                self.assertEqual(tile.read_from_config(), 1)
                self.assertEqual(tile.bld_yolo_checkpoint, str(tile_yolo))
        finally:
            CFG.global_cfg_file = original_global_cfg


if __name__ == "__main__":
    unittest.main()
