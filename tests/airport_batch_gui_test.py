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

STEPS = list(GUI.Ortho4XP_Earth_Preview.step_ckbtn)


def steps(*wanted):
    return {step: step in wanted for step in STEPS}


class AirportBatchGuiTest(unittest.TestCase):
    def setUp(self):
        self.earth = mock.Mock()
        self.parent = SimpleNamespace(
            custom_build_dir_entry=mock.Mock(),
            config=mock.Mock(),
            update_idletasks=mock.Mock(),
            get_lat_lon=mock.Mock(return_value=(0, 0)),
            load_tile_cfg=mock.Mock(),
            earth_window=self.earth,
            map_list=["BI", "GO2"],
            default_website=mock.Mock(),
        )
        self.parent.default_website.get.return_value = "BI"
        self.parent.custom_build_dir_entry.get.return_value = "test-tiles/"
        self.result = {
            "airports": [
                {"icao": "RCKH", "home": (22, 120), "tiles": [(22, 120), (23, 120)]},
            ],
            "tiles": [(22, 120), (23, 120)], "center": (22, 120),
            "created": [], "written": [], "listed": [], "year_conflicts": [],
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
        self.choose(("RCKH", {"RCKH": 2024}, {"RCKH": "GO2"}, None))
        GUI.Ortho4XP_Airport_Tiles_Dialog.assert_called_once_with(
            self.parent, mock.ANY, ["BI", "GO2"], "BI"
        )
        GUI.TILE.prepare_airport_batch.assert_called_once_with(
            "RCKH", "test-tiles/", years={"RCKH": 2024}, providers={"RCKH": "GO2"}
        )
        self.earth.show_tiles.assert_called_once_with(self.result["tiles"], (22, 120))
        self.earth.batch_build.assert_not_called()

    def test_checked_builds_each_tile_its_airports_steps(self):
        self.result.update({
            "airports": [
                {"icao": "RCKH", "home": (22, 120), "tiles": [(22, 120), (23, 120)]},
                {"icao": "RCTP", "home": (25, 121), "tiles": [(25, 121), (23, 120)]},
                {"icao": "RCBS", "home": (24, 118), "tiles": [(24, 118)]},
            ],
            "tiles": [(22, 120), (23, 120), (25, 121), (24, 118)],
        })
        self.choose((
            "RCKH, RCTP, RCBS",
            {"RCKH": 2025, "RCTP": 2025, "RCBS": 2025},
            {"RCKH": "BI", "RCTP": "BI", "RCBS": "BI"},
            {"RCKH": steps("dsf"), "RCTP": steps("bld_overlay"), "RCBS": steps()},
        ))
        calls = [call[0] for call in self.earth.mock_calls]
        self.assertLess(calls.index("show_tiles"), calls.index("batch_build"))
        # A tile two airports reach builds both their steps; an airport with
        # no step still has its tiles in the batch, so its zones are planned.
        self.earth.batch_build.assert_called_once_with(tile_steps={
            (22, 120): {"dsf"},
            (23, 120): {"dsf", "bld_overlay"},
            (25, 121): {"bld_overlay"},
            (24, 118): set(),
        })

    def test_several_airports_are_selected_together(self):
        self.result.update({
            "airports": [
                {"icao": "RCKH", "home": (22, 120), "tiles": [(22, 120), (23, 120)]},
                {"icao": "RCBS", "home": (24, 118), "tiles": [(24, 118)]},
            ],
            "tiles": [(22, 120), (23, 120), (24, 118)],
            "center": (23, 119),
            "written": [(22, 120), (24, 118)],
            "listed": [("RCKH", (22, 120)), ("RCBS", (24, 118))],
            "year_conflicts": [((23, 120), ("RCKH", 2025), ("RCTP", 2023))],
            "provider_conflicts": [((23, 120), ("RCKH", "BI"), ("RCTP", "GO2"))],
        })
        self.parent.get_lat_lon.return_value = (24, 118)
        self.choose(("RCKH, RCBS", {"RCKH": 2025, "RCBS": 2025}, {"RCKH": "BI", "RCBS": "BI"}, None))
        self.earth.show_tiles.assert_called_once_with(
            [(22, 120), (23, 120), (24, 118)], (23, 119)
        )
        # The active tile's config was just written: its settings are reloaded.
        self.parent.load_tile_cfg.assert_called_once_with(24, 118)
        self.earth.batch_build.assert_not_called()
        logged = " ".join(str(c) for c in GUI.UI.vprint.call_args_list)
        self.assertIn("keeps RCKH's reference year 2025, not 2023", logged)
        self.assertIn("keeps RCKH's imagery BI, not GO2", logged)

    def test_invalid_airport_never_launches(self):
        GUI.TILE.prepare_airport_batch.side_effect = ValueError("Unknown airport")
        self.choose(("ZZZZ", {"ZZZZ": 2025}, {"ZZZZ": "BI"}, {"ZZZZ": steps("osm")}))
        GUI.messagebox.showwarning.assert_called_once()
        self.earth.show_tiles.assert_not_called()
        self.earth.batch_build.assert_not_called()
        self.parent.config.assert_called_with(cursor="")

    def test_new_batch_window_is_created_before_launch(self):
        self.earth.lift.side_effect = tk.TclError("destroyed window")
        with mock.patch.object(GUI, "Ortho4XP_Earth_Preview", return_value=self.earth) as preview:
            self.choose(("RCKH", {"RCKH": 2025}, {"RCKH": "BI"}, {"RCKH": steps("osm")}))
        preview.assert_called_once_with(self.parent, 22, 120)
        self.earth.show_tiles.assert_called_once()
        self.earth.batch_build.assert_called_once_with(
            tile_steps={(22, 120): {"osm"}, (23, 120): {"osm"}}
        )


class BatchWindowTileStepsTest(unittest.TestCase):
    """batch_build(tile_steps=...) builds exactly those tiles with their steps."""

    def setUp(self):
        self.window = SimpleNamespace(
            parent=SimpleNamespace(config_window=None),
            step_ckbtn=GUI.Ortho4XP_Earth_Preview.step_ckbtn,
            v_={label: mock.Mock() for label in
                list(GUI.Ortho4XP_Earth_Preview.step_ckbtn.values()) + ["Override tile configs"]},
            dico_tiles_todo={(1, 1): None},
            custom_build_dir="",
            confirm_no_live_overlap=mock.Mock(return_value=True),
            confirm_resume=mock.Mock(return_value=""),
            launch_batch=mock.Mock(),
        )
        self.window.v_["Override tile configs"].get.return_value = 0
        self.window.selected_steps = lambda: {
            step: self.window.v_[label].set.call_args == mock.call(1)
            for step, label in self.window.step_ckbtn.items()
        }

    def test_steps_union_is_shown_and_tiles_come_from_the_plan(self):
        tile_steps = {(22, 120): {"dsf"}, (23, 120): {"dsf", "veg_overlay"}}
        GUI.Ortho4XP_Earth_Preview.batch_build(self.window, tile_steps=tile_steps)
        for step, label in self.window.step_ckbtn.items():
            self.window.v_[label].set.assert_called_once_with(
                1 if step in ("dsf", "veg_overlay") else 0
            )
        args = self.window.launch_batch.call_args[0]
        self.assertEqual(args[0], [(22, 120), (23, 120)])  # not the map selection
        self.assertEqual(args[4], tile_steps)
        self.assertEqual(self.window.confirm_resume.call_args[0][3], tile_steps)


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

    def run_dialog(self, act, default_year=2025):
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
        dialog = GUI.Ortho4XP_Airport_Tiles_Dialog(
            self.root, default_year, ["BI", "GO2", "EOX"], "BI"
        )
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
        self.assertEqual(result, ("RCKH", {"RCKH": 2025}, {"RCKH": "BI"}, None))

    def test_each_airport_gets_a_row_with_its_own_year_and_steps(self):
        def act(dialog, captured):
            captured["locked"] = {
                str(button.cget("state")) for button in dialog.step_buttons
            }
            dialog.icao_entry.insert(0, "RCKH, RCBS")
            captured["rows"] = sorted(dialog.rows)
            captured["initial"] = dialog.chosen_steps()
            dialog.start_batch.set(True)
            dialog.update_step_state()
            captured["unlocked"] = {
                str(button.cget("state")) for button in dialog.step_buttons
            }
            dialog.row("RCBS")["year"].set("2023")
            dialog.row("RCBS")["steps"]["clouds"].set(False)
            # The All imagery reaches every airport; a row can then differ.
            dialog.all_provider.set("GO2")
            dialog.set_all_providers()
            dialog.row("RCBS")["provider"].set("EOX")
            # The All column toggle reaches every airport.
            dialog.all_steps["mask"].set(False)
            dialog.set_all_steps("mask")
            dialog.ok()

        result, captured = self.run_dialog(act)
        self.assertEqual(captured["locked"], {"disabled"})
        self.assertEqual(captured["unlocked"], {"normal"})
        self.assertEqual(captured["rows"], ["RCBS", "RCKH"])
        self.assertEqual(
            captured["initial"],
            {"RCKH": steps(*STEPS), "RCBS": steps(*STEPS)},
        )
        icaos, years, providers, chosen = result
        self.assertEqual(icaos, "RCKH, RCBS")
        self.assertEqual(years, {"RCKH": 2025, "RCBS": 2023})
        self.assertEqual(providers, {"RCKH": "GO2", "RCBS": "EOX"})
        everything_but = lambda *off: steps(*(s for s in STEPS if s not in off))  # noqa: E731
        self.assertEqual(chosen, {
            "RCKH": everything_but("mask"),
            "RCBS": everything_but("mask", "clouds"),
        })

    def test_a_row_keeps_its_settings_while_the_code_is_retyped(self):
        def act(dialog, captured):
            dialog.icao_var.set("RCKH, RCBS")
            dialog.row("RCBS")["year"].set("2022")
            dialog.icao_var.set("RCKH")
            dialog.icao_var.set("RCKH, RCBS")
            captured["year"] = dialog.row("RCBS")["year"].get()
            dialog.all_year.set("2024")
            dialog.set_all_years()
            captured["all"] = dialog.chosen_years()
            dialog.cancel()

        result, captured = self.run_dialog(act)
        self.assertIsNone(result)
        self.assertEqual(captured["year"], "2022")
        self.assertEqual(captured["all"], {"RCKH": 2024, "RCBS": 2024})

    def _validate(self, codes, start, chosen_steps, years=None, providers=None):
        dialog = SimpleNamespace(
            codes=lambda: codes,
            start_batch=mock.Mock(),
            chosen_years=lambda: years if years is not None else {c: 2025 for c in codes},
            chosen_providers=lambda: (
                providers if providers is not None else {c: "BI" for c in codes}
            ),
            chosen_steps=lambda: chosen_steps,
            years_on_server=lambda _years: True,
        )
        dialog.start_batch.get.return_value = start
        with mock.patch.object(GUI.messagebox, "showwarning") as warning:
            ok = GUI.Ortho4XP_Airport_Tiles_Dialog.validate(dialog)
        return ok, warning.call_count

    def test_blank_code_keeps_dialog_open(self):
        self.assertEqual(self._validate([], False, {}), (False, 1))

    def test_starting_without_any_step_keeps_dialog_open(self):
        none = {"RCKH": steps(), "RCBS": steps()}
        self.assertEqual(self._validate(["RCKH", "RCBS"], True, none), (False, 1))
        self.assertEqual(self._validate(["RCKH", "RCBS"], False, none), (True, 0))
        some = {"RCKH": steps(), "RCBS": steps("dsf")}
        self.assertEqual(self._validate(["RCKH", "RCBS"], True, some), (True, 0))

    def test_an_airport_without_imagery_keeps_dialog_open(self):
        some = {"RCKH": steps("dsf")}
        self.assertEqual(
            self._validate(["RCKH"], True, some, providers={"RCKH": ""}), (False, 1)
        )

    def test_a_year_that_is_not_a_number_keeps_dialog_open(self):
        dialog = SimpleNamespace(
            codes=lambda: ["RCKH"],
            chosen_years=mock.Mock(side_effect=ValueError),
        )
        with mock.patch.object(GUI.messagebox, "showwarning") as warning:
            self.assertFalse(GUI.Ortho4XP_Airport_Tiles_Dialog.validate(dialog))
        warning.assert_called_once()

    def test_years_other_than_the_default_are_checked_on_the_server(self):
        import O4_Color_Build as CCB

        dialog = SimpleNamespace(default_year="2025")
        with mock.patch.object(CCB, "check_year", return_value=(False, "no 2015")) as check, \
                mock.patch.object(GUI.messagebox, "showwarning") as warning:
            ok = GUI.Ortho4XP_Airport_Tiles_Dialog.years_on_server(
                dialog, {"RCKH": 2025, "RCBS": 2015}
            )
        self.assertFalse(ok)
        check.assert_called_once_with(2015)
        warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
