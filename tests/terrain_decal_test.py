from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DECALS = ROOT / "scripts" / "asset_pipeline" / "decals"
for p in (ROOT / "src", DECALS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import O4_DSF_Utils as DSF  # noqa: E402
import set_terrain_decal as SET  # noqa: E402
from quilt import quilt  # noqa: E402
from shader_sim import repetition_score  # noqa: E402


class QuiltTests(unittest.TestCase):
    def test_output_tiles_and_does_not_repeat_the_source(self):
        rng = np.random.default_rng(0)
        # Tileable source: smooth periodic noise via a wrapped box blur.
        src = rng.random((64, 64, 2)).astype(np.float32)
        for _ in range(3):
            src = (src + np.roll(src, 1, 0) + np.roll(src, 1, 1)) / 3
        out = quilt(src, 192, patch=24, overlap=8, candidates=32, seed=1)
        self.assertEqual(out.shape, (192, 192, 2))
        inner = np.abs(np.diff(out, axis=1)).mean()
        wrap = np.abs(out[:, 0] - out[:, -1]).mean()
        self.assertLess(wrap, 1.5 * inner)
        wrap_v = np.abs(out[0] - out[-1]).mean()
        self.assertLess(wrap_v, 1.5 * inner)
        # The source period (64 px) must not survive in the output.
        score, _ = repetition_score(out[..., 0], 1.0, min_lag_m=8)
        tiled_score, _ = repetition_score(np.tile(src[..., 0], (3, 3)), 1.0, min_lag_m=8)
        self.assertGreater(tiled_score, 0.9)
        self.assertLess(score, 0.5)


class TerrainDecalTableTests(unittest.TestCase):
    def test_script_table_matches_builder(self):
        self.assertEqual(SET.TERRAIN_DECALS, DSF.TERRAIN_DECALS)


class SetTerrainDecalTests(unittest.TestCase):
    def test_rewrites_only_decal_lines(self):
        with tempfile.TemporaryDirectory() as tile:
            terrain = Path(tile) / "terrain"
            terrain.mkdir()
            with_decal = "A\n800\nTERRAIN\n\nBASE_TEX_NOWRAP ../textures/a.dds\nDECAL_LIB lib/g10/decals/maquify_2_green_key.dcl\nNO_ALPHA\n"
            without = "A\n800\nTERRAIN\n\nBASE_TEX_NOWRAP ../textures/b.dds\nNO_ALPHA\n"
            (terrain / "a.ter").write_text(with_decal)
            (terrain / "b.ter").write_text(without)
            changed, total = SET.retarget(tile, SET.TERRAIN_DECALS["o4xp"])
            self.assertEqual((changed, total), (1, 1))
            self.assertEqual(
                (terrain / "a.ter").read_text(),
                with_decal.replace("lib/g10/decals/maquify_2_green_key.dcl", "o4xp/decals/ortho_detail.dcl"),
            )
            self.assertEqual((terrain / "b.ter").read_text(), without)
            self.assertEqual(SET.retarget(tile, SET.TERRAIN_DECALS["o4xp"]), (0, 1))


if __name__ == "__main__":
    unittest.main()
