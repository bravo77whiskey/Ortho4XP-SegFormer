"""Colour correction / cloud removal as batch steps: which textures they redo."""

import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_Color_Build as CCB  # noqa: E402


class _Tile:
    lat, lon = 45, 5
    color_reference_year = 2025

    def __init__(self, build_dir):
        self.build_dir = build_dir


class ColorStepsTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.tile = _Tile(self.dir.name)
        os.makedirs(os.path.join(self.dir.name, "terrain"))
        os.makedirs(os.path.join(self.dir.name, "textures"))
        providers = {"BI": {"code": "BI"}, "GO2": {"code": "GO2"}, "EOX": {"code": "EOX"}}
        for p in (
            mock.patch.object(CCB.IMG, "providers_dict", providers),
            mock.patch.object(CCB.IMG, "combined_providers_dict", {}),
            mock.patch.object(CCB.IMG, "local_combined_providers_dict", {}),
            mock.patch.object(CCB, "_jpeg_path", lambda tile, *a: os.path.join(tile.build_dir, "none.jpg")),
        ):
            p.start()
            self.addCleanup(p.stop)

    def _texture(self, name, mark=None, ters=("",)):
        head = bytearray(128)
        head[:4] = b"DDS "
        if mark is not None:
            head[32:36] = CCB.MARK
            struct.pack_into("<II", head, 36, *mark)
        path = os.path.join(self.dir.name, "textures", name)
        with open(path, "wb") as f:
            f.write(bytes(head))
        for suffix in ters:
            with open(os.path.join(self.dir.name, "terrain", name[:-4] + suffix + ".ter"), "w") as f:
                f.write("A\n800\nTERRAIN\n\nLOAD_CENTER 1 2 3 4096\n"
                        f"BASE_TEX_NOWRAP ../textures/{name}\n")
        return path

    def test_textures_come_from_the_terrain_files(self):
        self._texture("100_200_GO216.dds", ters=("", "_sea", "_water_overlay"))
        self._texture("104_208_BI17.dds")
        found = dict(CCB.tile_textures(self.tile))
        self.assertEqual(sorted(found), [(200, 100, 16, "GO2"), (208, 104, 17, "BI")])

    def test_no_terrain_folder_means_no_step_3(self):
        self.assertIsNone(CCB.tile_textures(_Tile(os.path.join(self.dir.name, "nope"))))

    def test_what_each_step_redoes(self):
        year = 2025 << 16
        self._texture("1_1_BI16.dds")                                       # never corrected
        self._texture("2_2_BI16.dds", (CCB.MARK_VERSION, year))             # corrected, no fill
        self._texture("3_3_BI16.dds", (CCB.MARK_VERSION, year | 1))         # corrected and filled
        self._texture("4_4_BI16.dds", (CCB.MARK_VERSION, (2023 << 16) | 1))  # another year
        self._texture("5_5_BI12.dds")                                       # below the reference ZL
        color = {a: r for a, r, _ in CCB.plan_tile(self.tile, True, False)}
        clouds = {a: r for a, r, _ in CCB.plan_tile(self.tile, False, True)}
        both = {a: r for a, r, _ in CCB.plan_tile(self.tile, True, True)}
        self.assertEqual(sorted(color), [(1, 1, 16, "BI"), (4, 4, 16, "BI")])
        self.assertEqual(sorted(clouds), [(1, 1, 16, "BI"), (2, 2, 16, "BI"), (4, 4, 16, "BI")])
        self.assertEqual(sorted(both), sorted(clouds))
        self.assertEqual(clouds[(1, 1, 16, "BI")], "clouds not removed yet")
        CCB.forget(self.tile)

    def test_cloud_removal_alone_keeps_colours(self):
        year = 2025 << 16
        kept = (CCB.MARK_VERSION, year | CCB.FLAG_FILL | CCB.FLAG_KEPT)
        self._texture("1_1_BI16.dds", kept)                                  # filled, colours kept
        self._texture("2_2_BI16.dds", (CCB.MARK_VERSION - 1, (2023 << 16) | 3))  # old version and year
        self.assertEqual(CCB.plan_tile(self.tile, False, True), [])
        color = {a: r for a, r, _ in CCB.plan_tile(self.tile, True, False)}
        self.assertEqual(color, {(1, 1, 16, "BI"): "not colour-corrected yet",
                                 (2, 2, 16, "BI"): "not colour-corrected yet"})
        # Their DDS still has the provider's colours: corrected in place, no download.
        self.assertTrue(all(CCB.uses_existing_dds(self.tile, a) for a in color))
        CCB.forget(self.tile)

    def test_a_step_never_undoes_another(self):
        year = 2025 << 16
        corrected = self._texture("1_1_BI16.dds", (CCB.MARK_VERSION, year))
        kept = self._texture("2_2_BI16.dds", (CCB.MARK_VERSION, year | CCB.FLAG_FILL | CCB.FLAG_KEPT))
        fresh = self._texture("3_3_BI16.dds")
        self.assertEqual(CCB.texture_mode(corrected, False, True), (True, True))
        self.assertEqual(CCB.texture_mode(kept, True, False), (True, True))
        self.assertEqual(CCB.texture_mode(fresh, False, True), (False, True))
        self.assertEqual(CCB.texture_mode(fresh, True, False), (True, False))

    def test_only_an_uncorrected_dds_is_corrected_in_place(self):
        self._texture("1_1_BI16.dds")
        self._texture("4_4_BI16.dds", (CCB.MARK_VERSION, (2023 << 16)))
        plan = {a: download for a, _, download in CCB.plan_tile(self.tile, True, False)}
        self.assertEqual(plan, {(1, 1, 16, "BI"): False, (4, 4, 16, "BI"): True})
        self.assertTrue(CCB.uses_existing_dds(self.tile, (1, 1, 16, "BI")))
        CCB.forget(self.tile)
        self.assertFalse(CCB.uses_existing_dds(self.tile, (1, 1, 16, "BI")))

    def test_step_3_alone_never_touches_corrected_textures(self):
        path = self._texture("3_3_BI16.dds", (CCB.MARK_VERSION, 2025 << 16))
        self.assertIsNone(CCB.rebuild_reason(self.tile, (3, 3, 16, "BI"), path))


if __name__ == "__main__":
    unittest.main()
