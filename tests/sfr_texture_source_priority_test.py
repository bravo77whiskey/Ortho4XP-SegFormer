import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_SFR_Inference as SEGFORMER
import O4_SFR_Pipeline as SFR_PIPELINE


DDS_NAME = "112256_218048_BI18.dds"


def _make_tile_dirs(root):
    tex_dir = root / "Tiles" / "zOrtho4XP_+12+034" / "textures"
    ortho_dir = root / "Orthophotos" / "+10+030" / "+12+034" / "BI_18"
    tex_dir.mkdir(parents=True, exist_ok=True)
    ortho_dir.mkdir(parents=True, exist_ok=True)
    return tex_dir, ortho_dir


class SfrTextureSourcePriorityTests(unittest.TestCase):
    def test_tile_dds_wins_when_cached_orthophoto_also_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            tex_dir, ortho_dir = _make_tile_dirs(Path(tmp))
            (tex_dir / DDS_NAME).write_bytes(b"dds")
            (ortho_dir / DDS_NAME.replace(".dds", ".jpg")).write_bytes(b"jpg")

            files, source_mode, source_dir = SEGFORMER.collect_source_texture_files(
                str(tex_dir), 12, 34
            )

        self.assertEqual(files, [DDS_NAME])
        self.assertEqual(source_mode, "dds")
        self.assertIsNone(source_dir)

    def test_cached_orthophoto_is_used_when_tile_dds_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tex_dir, ortho_dir = _make_tile_dirs(Path(tmp))
            (ortho_dir / DDS_NAME.replace(".dds", ".jpg")).write_bytes(b"jpg")

            files, source_mode, source_dir = SEGFORMER.collect_source_texture_files(
                str(tex_dir), 12, 34
            )

        self.assertEqual(files, [DDS_NAME])
        self.assertEqual(source_mode, "orthophoto")
        self.assertTrue(source_dir.endswith(str(Path("Orthophotos") / "+10+030" / "+12+034")))

    def test_missing_tile_dds_and_orthophotos_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            tex_dir = Path(tmp) / "Tiles" / "zOrtho4XP_+12+034" / "textures"
            tex_dir.mkdir(parents=True)

            with self.assertRaises(FileNotFoundError):
                SEGFORMER.collect_source_texture_files(str(tex_dir), 12, 34)

    def test_pipeline_preflight_checks_valid_tile_dds_before_orthophotos(self):
        with tempfile.TemporaryDirectory() as tmp:
            tex_dir, _ortho_dir = _make_tile_dirs(Path(tmp))
            (tex_dir / "112256_218048_ZL18.dds").write_bytes(b"mask")
            self.assertFalse(
                SFR_PIPELINE._check_tile_imagery(
                    str(tex_dir), 12, 34, "SegFormer test overlay"
                )
            )

            (tex_dir / DDS_NAME).write_bytes(b"dds")
            self.assertTrue(
                SFR_PIPELINE._check_tile_imagery(
                    str(tex_dir), 12, 34, "SegFormer test overlay"
                )
            )


if __name__ == "__main__":
    unittest.main()
