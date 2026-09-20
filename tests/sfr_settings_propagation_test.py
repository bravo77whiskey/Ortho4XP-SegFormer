"""Guard: SFR building settings must propagate on BOTH build paths.

The building overlay runs in a venv subprocess driven by module-level vars in
O4_SFR_Pipeline. Two call sites copy the tile's settings into that module before
launching the build:

  * O4_GUI_Utils.build_sfr_bld        (GUI single-tile build button)
  * O4_Tile_Utils  batch build loop   (Build-all / batch)

If a new ``sfr_bld_yolo_*`` setting is wired into one but not the other, the
missing path silently uses the pipeline module default. That is exactly how
a retired overlap toggle once got ignored on the batch path, producing tiles
with no inter-detection overlap avoidance.

This test fails if the batch path does not assign every ``SFR.sfr_bld_*`` that
the GUI path assigns.
"""

import re
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_Cfg_Vars as CFGVARS

_ASSIGN_RE = re.compile(r"SFR\.(sfr_bld_\w+)\s*=\s*tile\.")
_HIDDEN_LEGACY_BLD_VARS = {"sfr_bld_close_k", "sfr_bld_open_k"}


def _assigned_sfr_bld_vars(file_name: str) -> set[str]:
    text = (SRC / file_name).read_text(encoding="utf-8")
    return set(_ASSIGN_RE.findall(text))


class SfrBldSettingsPropagationTests(unittest.TestCase):
    def test_batch_path_copies_every_gui_sfr_bld_setting(self):
        gui_vars = _assigned_sfr_bld_vars("O4_GUI_Utils.py")
        batch_vars = _assigned_sfr_bld_vars("O4_Tile_Utils.py")
        # Sanity: the GUI path must actually wire the YOLO settings.
        self.assertIn("sfr_bld_yolo_enabled", gui_vars)
        missing = gui_vars - batch_vars
        self.assertEqual(
            missing,
            set(),
            f"O4_Tile_Utils batch build does not propagate these SFR bld "
            f"settings that O4_GUI_Utils does: {sorted(missing)}",
        )

    def test_every_sfr_bld_config_var_is_listed_for_gui(self):
        cfg_vars = {
            name for name in CFGVARS.cfg_tile_vars
            if name.startswith("sfr_bld_")
        } - _HIDDEN_LEGACY_BLD_VARS
        listed = set(CFGVARS.list_sfr_bld_vars)
        self.assertEqual(
            cfg_vars - listed,
            set(),
            "These sfr_bld_* settings exist but are not displayed in the "
            f"SegFormer Bld GUI list: {sorted(cfg_vars - listed)}",
        )


_VEG_ASSIGN_RE = re.compile(r"SFR\.(sfr_veg_\w+)\s*=\s*tile\.")
# Retired Global Forests settings: still loadable so old configs stay valid,
# deliberately absent from the SegFormer Veg list and from both build paths.
_HIDDEN_LEGACY_VEG_VARS = set(CFGVARS.list_sfr_veg_legacy_vars)


def _assigned_sfr_veg_vars(file_name: str) -> set[str]:
    text = (SRC / file_name).read_text(encoding="utf-8")
    return set(_VEG_ASSIGN_RE.findall(text))


class SfrVegSettingsPropagationTests(unittest.TestCase):
    """The same guard for vegetation, including the tile-replacement pair.

    ``sfr_veg_exclude_lower_forests`` suppresses every lower-priority forest
    across a whole degree square. A path that silently fell back to the
    pipeline default would either leave a tile double-planted or strip one that
    was never meant to be replaced, so both build paths must carry it.
    """

    def test_batch_path_copies_every_gui_sfr_veg_setting(self):
        gui_vars = _assigned_sfr_veg_vars("O4_GUI_Utils.py")
        batch_vars = _assigned_sfr_veg_vars("O4_Tile_Utils.py")
        self.assertIn("sfr_veg_exclude_lower_forests", gui_vars)
        self.assertIn("sfr_veg_remove_embedded_trees", gui_vars)
        missing = gui_vars - batch_vars
        self.assertEqual(
            missing,
            set(),
            "O4_Tile_Utils batch build does not propagate these SFR veg "
            f"settings that O4_GUI_Utils does: {sorted(missing)}",
        )

    def test_both_paths_cover_every_listed_veg_setting(self):
        listed = {
            name for name in CFGVARS.list_sfr_veg_vars
            if name.startswith("sfr_veg_")
        }
        for file_name in ("O4_GUI_Utils.py", "O4_Tile_Utils.py"):
            missing = listed - _assigned_sfr_veg_vars(file_name)
            self.assertEqual(
                missing,
                set(),
                f"{file_name} does not copy these SegFormer Veg settings into "
                f"the pipeline: {sorted(missing)}",
            )

    def test_every_sfr_veg_config_var_is_listed_for_gui(self):
        cfg_vars = {
            name for name in CFGVARS.cfg_tile_vars
            if name.startswith("sfr_veg_")
        } - _HIDDEN_LEGACY_VEG_VARS
        listed = set(CFGVARS.list_sfr_veg_vars)
        self.assertEqual(
            cfg_vars - listed,
            set(),
            "These sfr_veg_* settings exist but are not displayed in the "
            f"SegFormer Veg GUI list: {sorted(cfg_vars - listed)}",
        )

    def test_the_replacement_settings_default_to_off(self):
        for name in ("sfr_veg_exclude_lower_forests",
                     "sfr_veg_remove_embedded_trees"):
            self.assertIs(CFGVARS.cfg_vars[name]["default"], False, name)
            self.assertIs(CFGVARS.cfg_vars[name]["type"], bool, name)

    def test_the_gui_hint_says_the_exclusion_covers_the_whole_tile(self):
        hint = CFGVARS.cfg_vars["sfr_veg_exclude_lower_forests"]["hint"]
        self.assertIn("WHOLE one-degree tile", hint)
        self.assertIn("no trees", hint)

    def test_the_gui_hint_says_a_checkbox_does_not_undo_installed_edits(self):
        hint = CFGVARS.cfg_vars["sfr_veg_remove_embedded_trees"]["hint"]
        self.assertIn("does NOT undo installed edits", hint)


class SfrVegPipelineWiringTests(unittest.TestCase):
    """The pipeline is what actually reaches the generation subprocess."""

    def _pipeline(self) -> str:
        return (SRC / "O4_SFR_Pipeline.py").read_text(encoding="utf-8")

    def test_the_pipeline_defines_both_settings(self):
        source = self._pipeline()
        self.assertIn("sfr_veg_exclude_lower_forests = False", source)
        self.assertIn("sfr_veg_remove_embedded_trees = False", source)

    def test_the_subprocess_receives_the_exclusion_setting(self):
        self.assertIn(
            "exclude_lower_forests = {sfr_veg_exclude_lower_forests!r}",
            self._pipeline(),
        )

    def test_embedded_tree_removal_requires_forest_replacement(self):
        self.assertIn(
            "if remove_embedded_trees and not exclude_lower_forests:",
            self._pipeline(),
        )

    def test_the_cli_exposes_the_exclusion_flag(self):
        source = (SRC / "O4_SFR_Vegetation_Overlay.py").read_text(
            encoding="utf-8")
        self.assertIn("'--exclude-lower-forests'", source)
        self.assertIn("exclude_lower_forests = args.exclude_lower_forests",
                      source)


if __name__ == "__main__":
    unittest.main()
