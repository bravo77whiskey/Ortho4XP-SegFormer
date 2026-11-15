"""Rare edge thicket heights merge into a lower one the tile keeps (VRAM, 2026-11-15)."""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_Forest_Heights as FH
import O4_Vegetation_Overlay as VEG

F = "o4xp/forests/{}/broad_hot_rain.for"


class MergeRuleTests(unittest.TestCase):
    def test_thickets_move_one_level_down_only(self):
        weights = {F.format("m7.5-su"): 50.0, F.format("m9-su"): 1.0, F.format("m6-su"): 1.0}
        moves = FH.merge_rare_copies(weights)
        # m9 joins m7.5; m6 cannot rise to it (thickets under runway caps stay under).
        self.assertEqual(moves, {F.format("m9-su"): F.format("m7.5-su")})

    def test_levels_more_than_one_step_apart_stay(self):
        weights = {F.format("m4.5"): 50.0, F.format("m7.5"): 1.0}
        self.assertEqual(FH.merge_rare_copies(weights), {})

    def test_common_levels_stay(self):
        weights = {F.format("m7.5"): 100.0, F.format("m9"): 6.0}   # 5.7 % of the group
        self.assertEqual(FH.merge_rare_copies(weights), {})

    def test_groups_are_one_forest_and_season_kind(self):
        weights = {F.format("m7.5-ev"): 100.0, F.format("m9-su"): 1.0,
                   "o4xp/forests/m7.5-su/coni_hot_rain.for": 100.0}
        self.assertEqual(FH.merge_rare_copies(weights), {})   # nothing to merge into

    def test_all_rare_keeps_the_heaviest(self):
        weights = {F.format("m7.5"): 1.0, F.format("m9"): 1.5}
        self.assertEqual(FH.merge_rare_copies(weights, share=0.9), {})   # only down: m9 stays

    def test_other_copies_and_x_plane_forests_are_left_alone(self):
        weights = {"lib/g8/broad_hot_rain.for": 100.0, F.format("h6"): 1.0, F.format("h9"): 100.0,
                   F.format("su"): 1.0}
        self.assertEqual(FH.merge_rare_copies(weights), {})


class OverlayTests(unittest.TestCase):
    def test_tile_pass_weighs_area_and_points(self):
        sq = [(120.0, 22.0), (120.01, 22.0), (120.01, 22.01), (120.0, 22.01), (120.0, 22.0)]
        tiny = [(120.0, 22.0), (120.0001, 22.0), (120.0001, 22.0001), (120.0, 22.0)]
        polys = [(F.format("m7.5-su"), VEG.LINE_MODE + 255, sq), (F.format("m9-su"), 255, tiny),
                 ("lib/g8/broad_hot_rain.for", 255, tiny),
                 ("o4xp/forests/m7.5/trees/x/a.for", VEG.POINT_MODE + 255, [(120.0, 22.0)] * 50),
                 ("o4xp/forests/m9/trees/x/a.for", VEG.POINT_MODE + 255, [(120.0, 22.0)])]
        out, moves = VEG._merge_rare_copies(polys, 22.5)
        self.assertEqual(set(moves), {F.format("m9-su"), "o4xp/forests/m9/trees/x/a.for"})
        self.assertEqual([p for p, _d, _r in out], [F.format("m7.5-su"), F.format("m7.5-su"),
                                                    "lib/g8/broad_hot_rain.for",
                                                    "o4xp/forests/m7.5/trees/x/a.for",
                                                    "o4xp/forests/m7.5/trees/x/a.for"])
        self.assertIs(out[1][2], tiny)   # rings and densities kept

    def test_holes_and_line_density_count(self):
        sq = VEG.HoledRing([(0.0, 0.0), (0.01, 0.0), (0.01, 0.01), (0.0, 0.01)],
                           [[(0.0, 0.0), (0.0, 0.005), (0.01, 0.005), (0.01, 0.0)]])
        full = VEG._polygon_weight(255, list(sq), 0.0)
        self.assertAlmostEqual(VEG._polygon_weight(255, sq, 0.0), full / 2, delta=full * 1e-6)
        self.assertAlmostEqual(VEG._polygon_weight(VEG.LINE_MODE + 51, list(sq), 0.0), full / 5,
                               delta=full * 1e-6)

    def test_setting_switch(self):
        with mock.patch.dict(os.environ, {"O4_VEG_MERGE_RARE_COPIES": "0"}):
            self.assertFalse(VEG._merge_rare_copies_enabled())
        env = {k: v for k, v in os.environ.items() if k != "O4_VEG_MERGE_RARE_COPIES"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(VEG._merge_rare_copies_enabled())


if __name__ == "__main__":
    unittest.main()
