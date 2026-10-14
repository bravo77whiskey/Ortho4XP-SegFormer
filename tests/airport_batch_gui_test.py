"""Airport selection's optional batch launch, without building scenery."""

import sys
import tkinter as tk
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_GUI_Utils as GUI


class AirportBatchGuiTest(unittest.TestCase):
    def setUp(self):
        self.earth = mock.Mock()
        self.earth.step_ckbtn = GUI.Ortho4XP_Earth_Preview.step_ckbtn
        self.earth.v_ = {label: mock.Mock() for label in self.earth.step_ckbtn.values()}
        self.earth.selected_steps.return_value = {
            step: False for step in self.earth.step_ckbtn
        }
        self.parent = SimpleNamespace(
            custom_build_dir_entry=mock.Mock(),
            config=mock.Mock(),
            update_idletasks=mock.Mock(),
            get_lat_lon=mock.Mock(return_value=(0, 0)),
            load_tile_cfg=mock.Mock(),
            earth_window=self.earth,
        )
        self.parent.custom_build_dir_entry.get.return_value = "test-tiles/"
        self.result = {
            "icao": "RCKH", "home": (22, 120),
            "tiles": [(22, 120), (23, 120)], "created": [], "listed": False,
        }
        for patch in (
            mock.patch.object(GUI, "Ortho4XP_Airport_Tiles_Dialog"),
            mock.patch.object(GUI.TILE, "prepare_airport_batch", return_value=self.result),
            mock.patch.object(GUI.UI, "vprint"),
            mock.patch.object(GUI.messagebox, "showwarning"),
            mock.patch.object(GUI.messagebox, "showerror"),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def choose(self, choice):
        GUI.Ortho4XP_Airport_Tiles_Dialog.return_value.result = choice
        GUI.Ortho4XP_GUI.select_airport_tiles(self.parent)

    def test_cancel_does_not_prepare_or_build(self):
        self.choose(None)
        GUI.TILE.prepare_airport_batch.assert_not_called()
        self.earth.batch_build.assert_not_called()

    def test_unchecked_selects_tiles_without_building(self):
        self.choose(("RCKH", False))
        GUI.TILE.prepare_airport_batch.assert_called_once_with("RCKH", "test-tiles/")
        self.earth.show_tiles.assert_called_once_with(self.result["tiles"], (22, 120))
        self.earth.batch_build.assert_not_called()

    def test_checked_selects_tiles_before_launch_and_defaults_to_all_in_one(self):
        self.choose(("RCKH", True))
        calls = [call[0] for call in self.earth.mock_calls]
        self.assertLess(calls.index("show_tiles"), calls.index("batch_build"))
        self.earth.batch_build.assert_called_once_with()
        for step, label in self.earth.step_ckbtn.items():
            if step in ("osm", "mesh", "mask", "dsf"):
                self.earth.v_[label].set.assert_called_once_with(1)
            else:
                self.earth.v_[label].set.assert_not_called()

    def test_checked_preserves_existing_batch_steps(self):
        self.earth.selected_steps.return_value["bld_overlay"] = True
        self.choose(("RCKH", True))
        self.earth.batch_build.assert_called_once_with()
        for variable in self.earth.v_.values():
            variable.set.assert_not_called()

    def test_invalid_airport_never_launches(self):
        GUI.TILE.prepare_airport_batch.side_effect = ValueError("Unknown airport")
        self.choose(("ZZZZ", True))
        GUI.messagebox.showwarning.assert_called_once()
        self.earth.show_tiles.assert_not_called()
        self.earth.batch_build.assert_not_called()
        self.parent.config.assert_called_with(cursor="")

    def test_new_batch_window_is_created_before_launch(self):
        self.earth.lift.side_effect = tk.TclError("destroyed window")
        with mock.patch.object(GUI, "Ortho4XP_Earth_Preview", return_value=self.earth) as preview:
            self.choose(("RCKH", True))
        preview.assert_called_once_with(self.parent, 22, 120)
        self.earth.show_tiles.assert_called_once()
        self.earth.batch_build.assert_called_once_with()


class AirportDialogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.root = tk.Tk()
        except tk.TclError as exc:
            raise unittest.SkipTest(f"no Tk display: {exc}")
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def test_checkbox_is_beside_entry_and_off_by_default(self):
        captured = {}

        def accept():
            dialog = next(
                child for child in self.root.winfo_children()
                if isinstance(child, GUI.Ortho4XP_Airport_Tiles_Dialog)
            )
            entry = dialog.icao_entry
            checkbox = next(
                child for child in entry.master.winfo_children()
                if isinstance(child, tk.Checkbutton)
            )
            captured["default"] = dialog.start_batch.get()
            captured["beside"] = (
                checkbox.winfo_x() >= entry.winfo_x() + entry.winfo_width()
                and abs(checkbox.winfo_y() - entry.winfo_y()) < 10
            )
            entry.insert(0, " rckh ")
            dialog.start_batch.set(True)
            dialog.ok()

        self.root.after(100, accept)
        dialog = GUI.Ortho4XP_Airport_Tiles_Dialog(self.root)
        self.assertFalse(captured["default"])
        self.assertTrue(captured["beside"])
        self.assertEqual(dialog.result, ("rckh", True))

    def test_blank_code_keeps_dialog_open(self):
        dialog = SimpleNamespace(icao_entry=mock.Mock())
        dialog.icao_entry.get.return_value = "   "
        with mock.patch.object(GUI.messagebox, "showwarning") as warning:
            self.assertFalse(GUI.Ortho4XP_Airport_Tiles_Dialog.validate(dialog))
        warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
