"""Guard: overlay settings must reach the pipeline on BOTH build paths.

The overlays run in a .venv subprocess driven by module-level settings in
O4_Overlay_Pipeline. Two call sites copy a tile's settings onto that module
before launching a step:

  * O4_GUI_Utils.build_bld_overlay / build_veg_overlay  (single-tile buttons)
  * O4_Tile_Utils batch build loop                      (Build-all / batch)

Both go through O4_Overlay_Pipeline.apply_tile_settings() with the shared
O4_Cfg_Vars.list_*_pipeline_vars lists. Before that, each path kept its own
hand-written copy, and a retired overlap toggle once got ignored on the batch
path, producing tiles with no inter-detection overlap avoidance.
"""

import re
import sys
import types
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_Cfg_Vars as CFGVARS
import O4_Overlay_Pipeline as PIPELINE

# Retired Global Forests settings: still loadable so old configs stay valid,
# deliberately absent from the vegetation settings list in the GUI.
_HIDDEN_LEGACY_VEG_VARS = set(CFGVARS.list_veg_legacy_vars)


def _calls(file_name: str) -> list[str]:
    text = (SRC / file_name).read_text(encoding="utf-8")
    return re.findall(r"PIPELINE\.apply_tile_settings\(tile, CFGVARS\.(\w+)\)", text)


class PipelineSettingsTests(unittest.TestCase):
    def test_both_build_paths_apply_both_setting_lists(self):
        for file_name in ("O4_GUI_Utils.py", "O4_Tile_Utils.py"):
            self.assertEqual(
                sorted(_calls(file_name)),
                ["list_bld_pipeline_vars", "list_veg_pipeline_vars"],
                file_name,
            )

    def test_every_pipeline_setting_is_a_tile_setting(self):
        for name in CFGVARS.list_bld_pipeline_vars + CFGVARS.list_veg_pipeline_vars:
            self.assertIn(name, CFGVARS.cfg_tile_vars, name)

    def test_apply_tile_settings_copies_every_listed_value(self):
        names = CFGVARS.list_bld_pipeline_vars + CFGVARS.list_veg_pipeline_vars
        tile = types.SimpleNamespace(**{name: f"value-{name}" for name in names})
        saved = {name: getattr(PIPELINE, name) for name in names}
        try:
            PIPELINE.apply_tile_settings(tile, CFGVARS.list_bld_pipeline_vars)
            PIPELINE.apply_tile_settings(tile, CFGVARS.list_veg_pipeline_vars)
            for name in names:
                self.assertEqual(getattr(PIPELINE, name), f"value-{name}", name)
        finally:
            for name, value in saved.items():
                setattr(PIPELINE, name, value)

    def test_an_unknown_setting_is_rejected(self):
        tile = types.SimpleNamespace(bld_not_a_setting=1)
        with self.assertRaises(AttributeError):
            PIPELINE.apply_tile_settings(tile, ["bld_not_a_setting"])

    def test_the_building_step_receives_the_yolo_settings(self):
        self.assertIn("bld_yolo_enabled", CFGVARS.list_bld_pipeline_vars)
        self.assertIn("segformer_patch_size", CFGVARS.list_bld_pipeline_vars)

    def test_the_vegetation_step_receives_the_replacement_settings(self):
        self.assertIn("veg_exclude_lower_forests", CFGVARS.list_veg_pipeline_vars)
        self.assertIn("veg_remove_embedded_trees", CFGVARS.list_veg_pipeline_vars)


class GuiSettingsListTests(unittest.TestCase):
    def test_every_bld_config_var_is_listed_for_gui(self):
        cfg_vars = {
            name for name in CFGVARS.cfg_tile_vars
            if name.startswith("bld_")
        }
        listed = set(CFGVARS.list_bld_vars)
        self.assertEqual(
            cfg_vars - listed,
            set(),
            "These bld_* settings exist but are not displayed in the "
            f"building overlay GUI list: {sorted(cfg_vars - listed)}",
        )

    def test_every_veg_config_var_is_listed_for_gui(self):
        cfg_vars = {
            name for name in CFGVARS.cfg_tile_vars
            if name.startswith("veg_")
        } - _HIDDEN_LEGACY_VEG_VARS
        listed = set(CFGVARS.list_veg_vars)
        self.assertEqual(
            cfg_vars - listed,
            set(),
            "These veg_* settings exist but are not displayed in the "
            f"vegetation overlay GUI list: {sorted(cfg_vars - listed)}",
        )


class VegSettingsTests(unittest.TestCase):
    def test_the_replacement_settings_default_to_off(self):
        for name in ("veg_exclude_lower_forests",
                     "veg_remove_embedded_trees"):
            self.assertIs(CFGVARS.cfg_vars[name]["default"], False, name)
            self.assertIs(CFGVARS.cfg_vars[name]["type"], bool, name)

    def test_the_gui_hint_says_the_exclusion_covers_the_whole_tile(self):
        hint = CFGVARS.cfg_vars["veg_exclude_lower_forests"]["hint"]
        self.assertIn("WHOLE one-degree tile", hint)
        self.assertIn("no trees", hint)

    def test_the_gui_hint_says_a_checkbox_does_not_undo_installed_edits(self):
        hint = CFGVARS.cfg_vars["veg_remove_embedded_trees"]["hint"]
        self.assertIn("does NOT undo installed edits", hint)


class VegPipelineWiringTests(unittest.TestCase):
    """The pipeline is what actually reaches the generation subprocess."""

    def _pipeline(self) -> str:
        return (SRC / "O4_Overlay_Pipeline.py").read_text(encoding="utf-8")

    def test_the_pipeline_defines_both_settings(self):
        source = self._pipeline()
        self.assertIn("veg_exclude_lower_forests = False", source)
        self.assertIn("veg_remove_embedded_trees = False", source)

    def test_the_subprocess_receives_the_exclusion_setting(self):
        self.assertIn(
            "exclude_lower_forests = {veg_exclude_lower_forests!r}",
            self._pipeline(),
        )

    def test_embedded_tree_removal_requires_forest_replacement(self):
        self.assertIn(
            "if remove_embedded_trees and not exclude_lower_forests:",
            self._pipeline(),
        )

    def test_the_cli_exposes_the_exclusion_flag(self):
        source = (SRC / "O4_Vegetation_Overlay.py").read_text(
            encoding="utf-8")
        self.assertIn("'--exclude-lower-forests'", source)
        self.assertIn("exclude_lower_forests = args.exclude_lower_forests",
                      source)


if __name__ == "__main__":
    unittest.main()
