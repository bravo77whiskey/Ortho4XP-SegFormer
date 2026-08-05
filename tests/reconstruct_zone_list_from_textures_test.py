import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "reconstruct_zone_list_from_textures.py"
SPEC = importlib.util.spec_from_file_location("reconstruct_zone_list_from_textures", SCRIPT)
RECON = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RECON
SPEC.loader.exec_module(RECON)


class ReconstructZoneListFromTexturesTests(unittest.TestCase):
    def test_discover_tile_dirs_accepts_gui_string_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tile_dir = Path(tmpdir) / "zOrtho4XP_+36+117"
            tile_dir.mkdir()

            discovered = RECON.discover_tile_dirs([tmpdir])

        self.assertEqual(discovered, [tile_dir])

    def test_reconstruct_leaves_existing_zone_list_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tile_dir = Path(tmpdir) / "zOrtho4XP_+36+117"
            tile_dir.mkdir()
            existing_zone = [
                [36.1, 117.1, 36.1, 117.2, 36.2, 117.2, 36.2, 117.1, 36.1, 117.1],
                18,
                "GO2",
            ]
            (tile_dir / "Ortho4XP_+36+117.cfg").write_text(
                f"default_website=BI\ndefault_zl=16\nzone_list={[existing_zone]!r}\n",
                encoding="utf-8",
            )

            result = RECON.reconstruct_zone_list(tile_dir)

        self.assertEqual(result["source"], "current")
        self.assertEqual(result["zone_list"], [existing_zone])

    def test_reconstruct_omits_default_provider_and_zoomlevel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tile_dir = Path(tmpdir) / "zOrtho4XP_+36+117"
            textures_dir = tile_dir / "textures"
            textures_dir.mkdir(parents=True)
            (tile_dir / "Ortho4XP_+36+117.cfg").write_text(
                "default_website=BI\ndefault_zl=16\nzone_list=[]\n",
                encoding="utf-8",
            )
            (textures_dir / "25680_54080_BI16.dds").write_bytes(b"default")

            result = RECON.reconstruct_zone_list(tile_dir)

        self.assertEqual(result["source"], "textures")
        self.assertEqual(result["zone_count"], 0)

    def test_reconstruct_picks_newest_provider_for_duplicate_footprint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tile_dir = Path(tmpdir) / "zOrtho4XP_+36+117"
            textures_dir = tile_dir / "textures"
            textures_dir.mkdir(parents=True)
            (tile_dir / "Ortho4XP_+36+117.cfg").write_text(
                "default_website=BI\n"
                "default_zl=16\n"
                "zone_list=[]\n",
                encoding="utf-8",
            )

            default_texture = textures_dir / "25680_54080_BI16.dds"
            older_duplicate = textures_dir / "102400_216512_BI18.dds"
            newer_duplicate = textures_dir / "102400_216512_GO218.dds"
            default_texture.write_bytes(b"default")
            older_duplicate.write_bytes(b"old")
            newer_duplicate.write_bytes(b"new")
            os.utime(default_texture, (100.0, 100.0))
            os.utime(older_duplicate, (200.0, 200.0))
            os.utime(newer_duplicate, (300.0, 300.0))

            result = RECON.reconstruct_zone_list(tile_dir)

        self.assertEqual(result["duplicate_footprints"], 1)
        self.assertEqual(result["zone_count"], 1)
        self.assertEqual(result["zone_list"][0][1:], [18, "GO2"])
        self.assertEqual(result["selected_textures"][0]["file"], "102400_216512_GO218.dds")

    def test_write_zone_list_replaces_existing_value_and_keeps_backup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "Ortho4XP_+36+117.cfg"
            cfg_path.write_text(
                "default_website=BI\n"
                "zone_list=[]\n",
                encoding="utf-8",
            )
            zone_list = [[[36.0, 117.0, 36.0, 117.1, 36.1, 117.1, 36.1, 117.0, 36.0, 117.0], 18, "GO2"]]

            backup_path = RECON.write_zone_list(cfg_path, zone_list)

            self.assertTrue(backup_path.exists())
            self.assertIn("zone_list=[]", backup_path.read_text(encoding="utf-8"))
            self.assertIn(f"zone_list={zone_list!r}", cfg_path.read_text(encoding="utf-8"))

    def test_write_zone_list_preserves_existing_bak_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "Ortho4XP_+36+117.cfg"
            bak_path = cfg_path.with_suffix(cfg_path.suffix + ".bak")
            cfg_path.write_text(
                "default_website=BI\n"
                "zone_list=[]\n",
                encoding="utf-8",
            )
            original_bak = (
                "default_website=Arc\n"
                "zone_list=[[[36.0, 117.0, 36.0, 117.1, 36.1, 117.1, 36.1, 117.0, 36.0, 117.0], 18, 'Arc']]\n"
            )
            bak_path.write_text(original_bak, encoding="utf-8")
            zone_list = [[[36.2, 117.2, 36.2, 117.3, 36.3, 117.3, 36.3, 117.2, 36.2, 117.2], 19, "GO2"]]

            backup_path = RECON.write_zone_list(cfg_path, zone_list)

            self.assertNotEqual(backup_path, bak_path)
            self.assertTrue(backup_path.exists())
            self.assertEqual(bak_path.read_text(encoding="utf-8"), original_bak)
            self.assertIn("zone_list=[]", backup_path.read_text(encoding="utf-8"))
            self.assertIn(f"zone_list={zone_list!r}", cfg_path.read_text(encoding="utf-8"))

    def test_zone_intersection_does_not_include_shared_boundary_only(self):
        east_tile_zone = [
            [36.1, 118.0, 36.1, 118.2, 36.2, 118.2, 36.2, 118.0, 36.1, 118.0],
            18,
            "GO2",
        ]

        self.assertFalse(RECON.zone_intersects_tile(east_tile_zone, 36, 117))
        self.assertTrue(RECON.zone_intersects_tile(east_tile_zone, 36, 118))


if __name__ == "__main__":
    unittest.main()
