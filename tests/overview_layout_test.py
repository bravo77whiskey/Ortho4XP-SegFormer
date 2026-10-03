"""Layout of the building overlay's overview images (make_viz)."""

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_Building_Overlay as BLD  # noqa: E402


def _layout(files, tile_viz=512):
    run = BLD._BuildingOverlayRun.__new__(BLD._BuildingOverlayRun)
    run.files = list(files)
    run.TILE_VIZ = tile_viz
    zls = [int(BLD._DDS_NAME_RE.match(name).group(4)) for name in files]
    run._layout_overview(min(zls))
    return run


def _overlaps(a, b):
    ax, ay, asize = a
    bx, by, bsize = b
    return ax < bx + bsize and bx < ax + asize and ay < by + bsize and by < ay + asize


class OverviewLayoutTests(unittest.TestCase):
    def test_contiguous_single_zoom_grid_is_one_panel_per_texture(self):
        files = [f"{y}_{x}_BI16.dds" for y in (23440, 23456) for x in (34128, 34144, 34160)]
        run = _layout(files)
        self.assertEqual((run.n_cols, run.n_rows), (3, 2))
        self.assertEqual(run._overview_panel("23440_34128_BI16.dds"), (0, 0, 512))
        self.assertEqual(run._overview_panel("23456_34160_BI16.dds"), (1024, 512, 512))

    def test_textures_with_gaps_keep_their_own_panels(self):
        # The old grid used the gap between the first two textures as the
        # column width, putting the last two in the same column.
        files = ["23312_34032_BI16.dds", "23440_34128_BI16.dds", "23440_34144_BI16.dds"]
        run = _layout(files)
        panels = [run._overview_panel(name) for name in files]
        self.assertEqual((run.n_cols, run.n_rows), (8, 9))
        for i in range(len(panels)):
            for j in range(i + 1, len(panels)):
                self.assertFalse(_overlaps(panels[i], panels[j]), (files[i], files[j]))

    def test_mixed_zoom_levels_share_a_small_canvas(self):
        # A ZL17 texture inside a ZL16 tile used to blow the canvas up to
        # tens of gigabytes because tile numbers of both levels were mixed.
        files = ["23440_34128_BI16.dds", "46880_68256_BI17.dds"]
        run = _layout(files)
        self.assertLessEqual(run.n_cols * run.n_rows, 4)
        x16, y16, size16 = run._overview_panel(files[0])
        x17, y17, size17 = run._overview_panel(files[1])
        self.assertEqual(size17 * 2, size16)
        # the ZL17 texture lies inside the ZL16 one it refines
        self.assertTrue(x16 <= x17 and x17 + size17 <= x16 + size16)
        self.assertTrue(y16 <= y17 and y17 + size17 <= y16 + size16)

    def test_paste_panel_clips_to_the_canvas(self):
        import numpy as np
        canvas = np.zeros((10, 10), dtype=np.uint8)
        BLD._paste_panel(canvas, np.ones((6, 6), dtype=np.uint8), 7, 8)
        self.assertEqual(int(canvas.sum()), 3 * 2)


if __name__ == "__main__":
    unittest.main()
