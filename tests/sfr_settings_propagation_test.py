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
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

_ASSIGN_RE = re.compile(r"SFR\.(sfr_bld_\w+)\s*=\s*tile\.")


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


if __name__ == "__main__":
    unittest.main()
