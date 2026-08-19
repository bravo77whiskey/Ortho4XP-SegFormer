"""Custom Scenery link creation, de-duplication and removal."""

import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_UI_Utils as UI
import O4_Scenery_Links as SLINK


def _make_tile_dir(root, lat, lon, name=None, with_dsf=True):
    """Create a build directory looking like a finished Ortho4XP tile."""
    import O4_File_Names as FNAMES

    path = Path(root) / (name or FNAMES.tile_dir(lat, lon))
    if with_dsf:
        dsf = path / "Earth nav data" / (FNAMES.long_latlon(lat, lon) + ".dsf")
        dsf.parent.mkdir(parents=True, exist_ok=True)
        dsf.write_bytes(b"XPLNEDSF")
    else:
        path.mkdir(parents=True, exist_ok=True)
    return str(path)


class SceneryLinksTest(unittest.TestCase):
    def setUp(self):
        self._verbosity = UI.verbosity
        self._log = UI.log
        UI.verbosity = 0
        UI.log = False
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.scenery = self.root / "X-Plane 12" / "Custom Scenery"
        self.scenery.mkdir(parents=True)
        self.tiles = self.root / "Tiles"
        self.tiles.mkdir()

    def tearDown(self):
        UI.verbosity = self._verbosity
        UI.log = self._log
        self._tmp.cleanup()

    # -- creation ---------------------------------------------------------
    def test_link_is_created_and_resolves_to_the_tile(self):
        target = _make_tile_dir(self.tiles, 50, 8)
        result = SLINK.ensure_tile_link(str(self.scenery), target, 50, 8)
        self.assertEqual(result.status, SLINK.CREATED)
        link = self.scenery / "zOrtho4XP_+50+008"
        self.assertTrue(SLINK.is_link(str(link)))
        self.assertTrue(SLINK.same_path(os.path.realpath(link), target))
        self.assertTrue(
            (link / "Earth nav data" / "+50+000" / "+50+008.dsf").is_file()
        )

    def test_second_call_is_a_no_op(self):
        target = _make_tile_dir(self.tiles, 50, 8)
        SLINK.ensure_tile_link(str(self.scenery), target, 50, 8)
        result = SLINK.ensure_tile_link(str(self.scenery), target, 50, 8)
        self.assertEqual(result.status, SLINK.ALREADY_LINKED)
        self.assertEqual(len(list(SLINK.iter_links(str(self.scenery)))), 1)

    def test_xplane_root_is_accepted_in_place_of_custom_scenery(self):
        target = _make_tile_dir(self.tiles, 50, 8)
        result = SLINK.ensure_tile_link(
            str(self.root / "X-Plane 12"), target, 50, 8
        )
        self.assertEqual(result.status, SLINK.CREATED)
        self.assertTrue(SLINK.is_link(str(self.scenery / "zOrtho4XP_+50+008")))

    # -- de-duplication ---------------------------------------------------
    def test_renamed_link_is_recognised_by_its_target(self):
        target = _make_tile_dir(self.tiles, 50, 8)
        SLINK.ensure_tile_link(str(self.scenery), target, 50, 8)
        os.rename(
            self.scenery / "zOrtho4XP_+50+008", self.scenery / "aaa_my_own_name"
        )
        result = SLINK.ensure_tile_link(str(self.scenery), target, 50, 8)
        self.assertEqual(result.status, SLINK.ALREADY_LINKED)
        self.assertEqual(os.path.basename(result.link), "aaa_my_own_name")
        self.assertFalse(os.path.lexists(self.scenery / "zOrtho4XP_+50+008"))

    def test_same_tile_from_another_build_dir_is_not_linked_twice(self):
        other = self.root / "OldTiles"
        other.mkdir()
        first = _make_tile_dir(other, 50, 8)
        SLINK.ensure_tile_link(str(self.scenery), first, 50, 8)
        second = _make_tile_dir(self.tiles, 50, 8)
        result = SLINK.ensure_tile_link(str(self.scenery), second, 50, 8)
        self.assertEqual(result.status, SLINK.DUPLICATE_TILE)
        self.assertEqual(len(list(SLINK.iter_links(str(self.scenery)))), 1)

    def test_neighbouring_tiles_do_not_look_like_duplicates(self):
        SLINK.ensure_tile_link(
            str(self.scenery), _make_tile_dir(self.tiles, 50, 8), 50, 8
        )
        result = SLINK.ensure_tile_link(
            str(self.scenery), _make_tile_dir(self.tiles, 50, 9), 50, 9
        )
        self.assertEqual(result.status, SLINK.CREATED)
        self.assertEqual(len(list(SLINK.iter_links(str(self.scenery)))), 2)

    def test_grouped_build_dir_is_linked_under_its_own_name(self):
        group = self.root / "MyRegion"
        _make_tile_dir(group, 50, 8, name=".")
        result = SLINK.ensure_tile_link(
            str(self.scenery), str(group), 50, 8, grouped=True
        )
        self.assertEqual(result.status, SLINK.CREATED)
        self.assertTrue(SLINK.is_link(str(self.scenery / "zOrtho4XP_MyRegion")))

    # -- refusals ---------------------------------------------------------
    def test_foreign_directory_keeping_the_name_is_left_alone(self):
        squatter = self.scenery / "zOrtho4XP_+50+008"
        (squatter / "Earth nav data").mkdir(parents=True)
        target = _make_tile_dir(self.tiles, 50, 8)
        result = SLINK.ensure_tile_link(str(self.scenery), target, 50, 8)
        self.assertEqual(result.status, SLINK.NAME_TAKEN)
        self.assertFalse(SLINK.is_link(str(squatter)))
        self.assertTrue((squatter / "Earth nav data").is_dir())

    def test_broken_link_on_the_canonical_name_is_replaced(self):
        gone = _make_tile_dir(self.tiles, 50, 8, name="gone")
        SLINK.make_link(str(self.scenery / "zOrtho4XP_+50+008"), gone)
        shutil.rmtree(gone)
        self.assertTrue(SLINK.is_broken_link(str(self.scenery / "zOrtho4XP_+50+008")))
        target = _make_tile_dir(self.tiles, 50, 8)
        result = SLINK.ensure_tile_link(str(self.scenery), target, 50, 8)
        self.assertEqual(result.status, SLINK.CREATED)
        self.assertTrue(
            SLINK.same_path(
                os.path.realpath(self.scenery / "zOrtho4XP_+50+008"), target
            )
        )

    def test_dangling_link_for_the_same_tile_is_swept_up(self):
        gone = _make_tile_dir(self.tiles, 50, 8, name="gone")
        SLINK.make_link(str(self.scenery / "my_+50+008_pack"), gone)
        shutil.rmtree(gone)
        target = _make_tile_dir(self.tiles, 50, 8)
        result = SLINK.ensure_tile_link(str(self.scenery), target, 50, 8)
        self.assertEqual(result.status, SLINK.CREATED)
        self.assertEqual(
            [name for (name, _p, _t) in SLINK.iter_links(str(self.scenery))],
            ["zOrtho4XP_+50+008"],
        )

    def test_tile_built_inside_custom_scenery_needs_no_link(self):
        target = _make_tile_dir(self.scenery, 50, 8)
        result = SLINK.ensure_tile_link(str(self.scenery), target, 50, 8)
        self.assertEqual(result.status, SLINK.IN_PLACE)
        self.assertEqual(list(SLINK.iter_links(str(self.scenery))), [])

    def test_unset_scenery_dir_is_reported_not_raised(self):
        target = _make_tile_dir(self.tiles, 50, 8)
        self.assertEqual(
            SLINK.ensure_tile_link("", target, 50, 8).status, SLINK.NO_SCENERY_DIR
        )

    def test_missing_target_is_reported_not_raised(self):
        result = SLINK.ensure_tile_link(
            str(self.scenery), str(self.tiles / "nope"), 50, 8
        )
        self.assertEqual(result.status, SLINK.NO_TARGET)

    # -- removal ----------------------------------------------------------
    def test_removal_finds_the_link_under_any_name_and_spares_the_tile(self):
        target = _make_tile_dir(self.tiles, 50, 8)
        SLINK.ensure_tile_link(str(self.scenery), target, 50, 8)
        os.rename(
            self.scenery / "zOrtho4XP_+50+008", self.scenery / "aaa_my_own_name"
        )
        removed = SLINK.remove_tile_link(str(self.scenery), target, 50, 8)
        self.assertEqual([os.path.basename(p) for p in removed], ["aaa_my_own_name"])
        self.assertEqual(list(SLINK.iter_links(str(self.scenery))), [])
        self.assertTrue(os.path.isdir(target))

    def test_removal_clears_a_link_left_dangling_by_a_deleted_tile(self):
        target = _make_tile_dir(self.tiles, 50, 8)
        SLINK.ensure_tile_link(str(self.scenery), target, 50, 8)
        shutil.rmtree(target)
        removed = SLINK.remove_tile_link(str(self.scenery), target, 50, 8)
        self.assertEqual(len(removed), 1)
        self.assertFalse(os.path.lexists(self.scenery / "zOrtho4XP_+50+008"))

    def test_is_tile_linked_tracks_creation_and_removal(self):
        target = _make_tile_dir(self.tiles, 50, 8)
        self.assertFalse(SLINK.is_tile_linked(str(self.scenery), target))
        SLINK.ensure_tile_link(str(self.scenery), target, 50, 8)
        self.assertTrue(SLINK.is_tile_linked(str(self.scenery), target))
        SLINK.remove_tile_link(str(self.scenery), target, 50, 8)
        self.assertFalse(SLINK.is_tile_linked(str(self.scenery), target))


class AutoLinkTest(unittest.TestCase):
    """The build-time hook, with a stand-in for the tkinter-heavy config module."""

    def setUp(self):
        self._verbosity = UI.verbosity
        self._log = UI.log
        UI.verbosity = 0
        UI.log = False
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.scenery = self.root / "Custom Scenery"
        self.scenery.mkdir()
        self.tiles = self.root / "Tiles"
        self.tiles.mkdir()
        self._saved_cfg = sys.modules.get("O4_Config_Utils")
        self.cfg = types.ModuleType("O4_Config_Utils")
        self.cfg.custom_scenery_dir = str(self.scenery)
        self.cfg.auto_link_custom_scenery = True
        sys.modules["O4_Config_Utils"] = self.cfg

    def tearDown(self):
        UI.verbosity = self._verbosity
        UI.log = self._log
        if self._saved_cfg is None:
            sys.modules.pop("O4_Config_Utils", None)
        else:
            sys.modules["O4_Config_Utils"] = self._saved_cfg
        self._tmp.cleanup()

    def _tile(self, lat=50, lon=8, with_dsf=True):
        tile = types.SimpleNamespace(
            lat=lat,
            lon=lon,
            grouped=False,
            build_dir=_make_tile_dir(self.tiles, lat, lon, with_dsf=with_dsf),
        )
        return tile

    def test_built_tile_is_linked(self):
        result = SLINK.auto_link_tile(self._tile())
        self.assertEqual(result.status, SLINK.CREATED)
        self.assertTrue(SLINK.is_link(str(self.scenery / "zOrtho4XP_+50+008")))

    def test_toggle_off_links_nothing(self):
        self.cfg.auto_link_custom_scenery = False
        self.assertIsNone(SLINK.auto_link_tile(self._tile()))
        self.assertEqual(list(SLINK.iter_links(str(self.scenery))), [])

    def test_unset_scenery_dir_links_nothing(self):
        self.cfg.custom_scenery_dir = ""
        self.assertIsNone(SLINK.auto_link_tile(self._tile()))

    def test_tile_without_dsf_is_not_linked(self):
        self.assertIsNone(SLINK.auto_link_tile(self._tile(with_dsf=False)))
        self.assertEqual(list(SLINK.iter_links(str(self.scenery))), [])

    def test_hook_never_raises_when_the_scenery_dir_is_bogus(self):
        self.cfg.custom_scenery_dir = str(self.root / "not" / "there")
        self.assertEqual(
            SLINK.auto_link_tile(self._tile()).status, SLINK.NO_SCENERY_DIR
        )


if __name__ == "__main__":
    unittest.main()
