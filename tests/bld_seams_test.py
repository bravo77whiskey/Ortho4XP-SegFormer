"""Tests for biggest piece wins (O4_Bld_Seams)."""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_Bld_Seams as SEAMS


def _box(x0, x1, y0, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _piece(name, til_y, til_x, quad, area, zl=16, stride=512):
    lines = SEAMS.seam_lines_touched(quad, 4096, 4096, stride)
    return SEAMS.SeamPiece(name, til_y, til_x, zl, quad, area, lines)


# A ZL16 texture and its east neighbour.
TY, TX = 28448, 54448


class SeamLinesTest(unittest.TestCase):
    def test_window_and_texture_edges(self):
        lines = SEAMS.seam_lines_touched(_box(480, 513, 10, 40), 4096, 4096, 512)
        self.assertEqual(lines, [(0, 512.0, -1, "window")])
        lines = SEAMS.seam_lines_touched(_box(4000, 4095, 10, 40), 4096, 4096, 512)
        self.assertIn((0, 4096.0, -1, "texture"), lines)

    def test_interior_box_touches_nothing(self):
        self.assertEqual(SEAMS.seam_lines_touched(_box(600, 700, 600, 700), 4096, 4096, 512), [])

    def test_no_window_lines_without_a_stride(self):
        self.assertEqual(SEAMS.seam_lines_touched(_box(480, 513, 10, 40), 4096, 4096, None), [])


class GroupPiecesTest(unittest.TestCase):
    def test_biggest_piece_wins_across_window_and_texture_seams(self):
        pieces = [
            _piece("tex_left", TY, TX, _box(4000, 4095, 100, 140), 300),
            _piece("tex_right", TY, TX + 16, _box(0, 30, 105, 135), 90),
            _piece("win_left", TY, TX, _box(480, 513, 300, 330), 50),
            _piece("win_right", TY, TX, _box(511, 560, 302, 328), 80),
            _piece("alone", TY, TX, _box(1000, 1024, 1500, 1520), 20),
        ]
        drop, counts = SEAMS.group_pieces(pieces)
        self.assertEqual(sorted(p.placement for p in drop), ["tex_right", "win_left"])
        self.assertEqual(counts, {"texture": 1, "window": 1})

    def test_four_pieces_at_a_window_corner_keep_one(self):
        pieces = [
            _piece("nw", TY, TX, _box(490, 512, 490, 512), 30),
            _piece("ne", TY, TX, _box(512, 540, 490, 512), 40),
            _piece("sw", TY, TX, _box(490, 512, 512, 530), 20),
            _piece("se", TY, TX, _box(512, 540, 512, 530), 25),
        ]
        drop, _counts = SEAMS.group_pieces(pieces)
        self.assertEqual(sorted(p.placement for p in drop), ["nw", "se", "sw"])

    def test_pieces_across_zoom_levels(self):
        # A ZL16 piece meeting a ZL17 piece on the shared texture edge.
        left = _piece("zl16", TY, TX, _box(4000, 4095, 100, 140), 300)
        right = _piece("zl17", TY * 2, (TX + 16) * 2, _box(0, 40, 200, 280), 60, zl=17)
        drop, counts = SEAMS.group_pieces([left, right])
        self.assertEqual([p.placement for p in drop], ["zl17"])
        self.assertEqual(counts, {"texture": 1})

    def test_pieces_on_the_same_side_stay(self):
        # Two neighbours ending on the same window edge from the same side.
        pieces = [
            _piece("a", TY, TX, _box(480, 512, 300, 330), 50),
            _piece("b", TY, TX, _box(470, 512, 331, 360), 60),
        ]
        self.assertEqual(SEAMS.group_pieces(pieces)[0], [])

    def test_far_apart_pieces_stay(self):
        pieces = [
            _piece("a", TY, TX, _box(480, 512, 300, 330), 50),
            _piece("b", TY, TX, _box(512, 560, 900, 930), 60),
        ]
        self.assertEqual(SEAMS.group_pieces(pieces)[0], [])


class SwitchTest(unittest.TestCase):
    def test_on_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("O4_BLD_SEAM_OWNER", None)
            self.assertTrue(SEAMS.owner_enabled())
        with mock.patch.dict(os.environ, {"O4_BLD_SEAM_OWNER": "0"}):
            self.assertFalse(SEAMS.owner_enabled())


if __name__ == "__main__":
    unittest.main()
