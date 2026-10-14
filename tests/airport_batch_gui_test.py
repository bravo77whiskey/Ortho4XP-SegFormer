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
            "airports": [
                {"icao": "RCKH", "home": (22, 120), "tiles": [(22, 120), (23, 120)]},
            ],
            "tiles": [(22, 120), (23, 120)], "center": (22, 120),
            "created": [], "listed": [],
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
        self.choose(("RCKH", None))
        GUI.TILE.prepare_airport_batch.assert_called_once_with("RCKH", "test-tiles/")
        self.earth.show_tiles.assert_called_once_with(self.result["tiles"], (22, 120))
        self.earth.batch_build.assert_not_called()

    def steps(self, *wanted):
        return {step: step in wanted for step in self.earth.step_ckbtn}

    def test_checked_builds_the_chosen_steps_after_selecting_tiles(self):
        self.choose(("RCKH", self.steps("dsf", "bld_overlay")))
        calls = [call[0] for call in self.earth.mock_calls]
        self.assertLess(calls.index("show_tiles"), calls.index("batch_build"))
        self.earth.batch_build.assert_called_once_with()
        # The batch window shows exactly the chosen steps, unticking others.
        for step, label in self.earth.step_ckbtn.items():
            self.earth.v_[label].set.assert_called_once_with(
                1 if step in ("dsf", "bld_overlay") else 0
            )


    def test_several_airports_are_selected_together(self):
        self.result.update({
            "airports": [
                {"icao": "RCKH", "home": (22, 120), "tiles": [(22, 120), (23, 120)]},
                {"icao": "RCBS", "home": (24, 118), "tiles": [(24, 118)]},
            ],
            "tiles": [(22, 120), (23, 120), (24, 118)],
            "center": (23, 119),
            "listed": [("RCKH", (22, 120)), ("RCBS", (24, 118))],
        })
        self.parent.get_lat_lon.return_value = (24, 118)
        self.choose(("RCKH, RCBS", None))
        GUI.TILE.prepare_airport_batch.assert_called_once_with("RCKH, RCBS", "test-tiles/")
        self.earth.show_tiles.assert_called_once_with(
            [(22, 120), (23, 120), (24, 118)], (23, 119)
        )
        # The active tile just had an airport added: its settings are reloaded.
        self.parent.load_tile_cfg.assert_called_once_with(24, 118)
        self.earth.batch_build.assert_not_called()

    def test_invalid_airport_never_launches(self):
        GUI.TILE.prepare_airport_batch.side_effect = ValueError("Unknown airport")
        self.choose(("ZZZZ", self.steps("osm")))
        GUI.messagebox.showwarning.assert_called_once()
        self.earth.show_tiles.assert_not_called()
        self.earth.batch_build.assert_not_called()
        self.parent.config.assert_called_with(cursor="")

    def test_new_batch_window_is_created_before_launch(self):
        self.earth.lift.side_effect = tk.TclError("destroyed window")
        with mock.patch.object(GUI, "Ortho4XP_Earth_Preview", return_value=self.earth) as preview:
            self.choose(("RCKH", self.steps("osm")))
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

    def run_dialog(self, act):
        """Open the dialog, let act(dialog) inspect and fill it, return result."""
        captured = {}

        def step():
            dialog = next(
                child for child in self.root.winfo_children()
                if isinstance(child, GUI.Ortho4XP_Airport_Tiles_Dialog)
            )
            dialog.update_idletasks()
            act(dialog, captured)

        self.root.after(100, step)
        dialog = GUI.Ortho4XP_Airport_Tiles_Dialog(self.root)
        return dialog.result, captured

    def test_checkbox_is_beside_entry_and_off_by_default(self):
        def act(dialog, captured):
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
            dialog.ok()

        result, captured = self.run_dialog(act)
        self.assertFalse(captured["default"])
        self.assertTrue(captured["beside"])
        self.assertEqual(result, ("rckh", None))

    def test_all_steps_ticked_by_default_and_unlock_with_start(self):
        def act(dialog, captured):
            captured["initial"] = dialog.chosen_steps()
            captured["locked"] = {
                str(button.cget("state")) for button in dialog.step_buttons
            }
            dialog.start_batch.set(True)
            dialog.update_step_state()
            captured["unlocked"] = {
                str(button.cget("state")) for button in dialog.step_buttons
            }
            dialog.step_vars["mask"].set(False)
            dialog.step_vars["clouds"].set(False)
            dialog.icao_entry.insert(0, "RCKH, RCBS")
            dialog.ok()

        result, captured = self.run_dialog(act)
        steps = GUI.Ortho4XP_Earth_Preview.step_ckbtn
        self.assertEqual(captured["initial"], {step: True for step in steps})
        self.assertEqual(captured["locked"], {"disabled"})
        self.assertEqual(captured["unlocked"], {"normal"})
        self.assertEqual(
            result,
            ("RCKH, RCBS",
             {step: step not in ("mask", "clouds") for step in steps}),
        )

    def _validate(self, icaos, start, steps):
        dialog = SimpleNamespace(
            icao_entry=mock.Mock(),
            start_batch=mock.Mock(),
            chosen_steps=lambda: steps,
        )
        dialog.icao_entry.get.return_value = icaos
        dialog.start_batch.get.return_value = start
        with mock.patch.object(GUI.messagebox, "showwarning") as warning:
            ok = GUI.Ortho4XP_Airport_Tiles_Dialog.validate(dialog)
        return ok, warning.call_count

    def test_blank_code_keeps_dialog_open(self):
        self.assertEqual(self._validate("   ", False, {}), (False, 1))

    def test_starting_without_any_step_keeps_dialog_open(self):
        self.assertEqual(self._validate("RCKH", True, {"osm": False}), (False, 1))
        self.assertEqual(self._validate("RCKH", False, {"osm": False}), (True, 0))
        self.assertEqual(self._validate("RCKH", True, {"osm": True}), (True, 0))


if __name__ == "__main__":
    unittest.main()
