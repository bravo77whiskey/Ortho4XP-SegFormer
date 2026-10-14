"""The progress tree and filtered log of the main window (O4_Progress_Panel)."""

import sys
import tkinter as tk
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_Progress as PROG  # noqa: E402
import O4_Progress_Panel as PANEL  # noqa: E402


class PanelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.root = tk.Tk()
        except tk.TclError as exc:  # pragma: no cover - no display
            raise unittest.SkipTest(f"no Tk display: {exc}")
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def setUp(self):
        PROG.is_stopped = lambda: False
        self.retries = []
        self.panel = PANEL.ProgressPanel(self.root, on_retry=lambda t, l: self.retries.append((t, l)))
        PROG.begin_run([(22, 112), (22, 113)], ["dsf", "bld_overlay"],
                       launch={"custom_build_dir": "X:/tiles", "override_cfg": False})

    def tearDown(self):
        self.panel.paned.destroy()

    def _draw(self):
        self.panel.refresh()
        self.root.update_idletasks()

    def test_rows_follow_the_run(self):
        self._draw()
        tree = self.panel.tree
        self.assertEqual(tree.get_children(""), ("run",))
        self.assertEqual(tree.get_children("run"), ("t:+22+112", "t:+22+113"))
        self.assertEqual(tree.get_children("t:+22+112"), ("s:+22+112:dsf", "s:+22+112:bld_overlay"))
        with PROG.step(22, 112, "dsf"):
            PROG.items(part="Download", done=30, total=120)
            self._draw()
            self.assertEqual(tree.item("s:+22+112:dsf", "text"), "▶ DSF")
            self.assertTrue(tree.item("t:+22+112", "open"))
            self.assertTrue(tree.item("s:+22+112:dsf", "open"))
            part = "p:+22+112:dsf:Download"
            self.assertEqual(tree.item(part, "text"), "Download")
            self.assertTrue(tree.item(part, "values")[0].endswith("30/120"))
            PROG.issue("a.dds", "download", "failed after 3 attempts")
        self._draw()
        # Done: its parts go, its issue stays, and running rows fold away.
        self.assertFalse(tree.exists("p:+22+112:dsf:Download"))
        self.assertEqual(tree.item("i:+22+112:dsf:0", "text"), "⚠ a.dds")
        self.assertEqual(tree.item("s:+22+112:dsf", "text"), "⚠ DSF")
        self.assertFalse(tree.item("s:+22+112:dsf", "open"))
        self.assertIn("done with issues", tree.item("s:+22+112:dsf", "values")[0])

    def test_a_row_opened_by_hand_stays_open(self):
        self._draw()
        tree = self.panel.tree
        tree.item("t:+22+113", open=True)
        tree.focus("t:+22+113")
        self.panel._on_toggle()
        PROG.step_skipped(22, 113, "dsf")
        self._draw()
        self.assertTrue(tree.item("t:+22+113", "open"))

    def test_a_new_run_replaces_the_rows(self):
        self._draw()
        PROG.begin_run([(1, 2)], ["osm"])
        self._draw()
        self.assertEqual(self.panel.tree.get_children("run"), ("t:+01+002",))

    def test_selecting_a_row_filters_the_log(self):
        self.panel.add_lines([
            ("+22+112", "dsf", "downloading\n"),
            ("+22+113", "dsf", "other tile\n"),
            ("+22+112", "bld_overlay", "WARNING: skipped a.dds\n"),
            (None, None, "Batch build launched\n"),
        ])
        text = lambda: self.panel.console.get("1.0", "end")  # noqa: E731
        self.assertIn("+22+112 DSF   │ downloading", text())
        self.assertIn("Batch build launched", text())
        self._draw()
        self.panel.tree.selection_set("s:+22+112:bld_overlay")
        self.panel._on_select()
        self.assertEqual(text().strip(), "+22+112 BLD   │ WARNING: skipped a.dds")
        self.assertIn("warn", self.panel.console.tag_names("1.20"))
        self.panel.tree.selection_set("t:+22+112")
        self.panel._on_select()
        self.assertNotIn("other tile", text())
        self.assertIn("downloading", text())
        self.panel.show_all()
        self.assertIn("other tile", text())
        # Lines arriving while filtered only show when they match.
        self.panel._set_filter(("+22+113", None))
        self.panel.add_lines([("+22+112", "dsf", "hidden\n"), ("+22+113", "dsf", "shown\n")])
        self.assertNotIn("hidden", text())
        self.assertIn("shown", text())

    def _menu_labels(self, iid):
        self._draw()
        event = mock.Mock(x_root=0, y_root=0)
        with mock.patch.object(self.panel.tree, "identify_row", return_value=iid), \
                mock.patch.object(self.panel.menu, "tk_popup"):
            self.panel._on_right_click(event)
        return [self.panel.menu.entrycget(i, "label") for i in range(self.panel.menu.index("end") + 1)]

    def test_retry_targets_failed_steps_and_issues(self):
        with PROG.step(22, 112, "dsf") as scope:
            PROG.issue("a.dds", "download", "white")
            scope.result(1)
        with PROG.step(22, 112, "bld_overlay") as scope:
            scope.result(1)
        with PROG.step(22, 113, "dsf") as scope:
            scope.result(0)
        PROG.end_run()
        self._draw()
        self.assertEqual(self.panel._retry_targets("run"), {(22, 112): ["dsf"], (22, 113): ["dsf"]})
        self.assertEqual(self.panel._retry_targets("t:+22+113"), {(22, 113): ["dsf"]})
        self.assertEqual(self.panel._retry_targets("i:+22+112:dsf:0"), {(22, 112): ["dsf"]})
        self.assertEqual(self.panel._retry_targets("s:+22+112:bld_overlay"), {})
        self.assertEqual(self._menu_labels("s:+22+112:bld_overlay"), ["Nothing to retry here"])
        labels = self._menu_labels("run")
        self.assertEqual(labels, ["Retry every failed step and issue (2 steps)"])
        self.panel.menu.invoke(0)
        self.assertEqual(
            self.retries,
            [({(22, 112): ["dsf"], (22, 113): ["dsf"]}, {"custom_build_dir": "X:/tiles", "override_cfg": False})],
        )

    def test_no_retry_while_the_run_is_going(self):
        with PROG.step(22, 113, "dsf") as scope:
            scope.result(0)
        labels = self._menu_labels("t:+22+113")
        self.assertTrue(labels[0].endswith("wait for the build to end"))
        self.assertEqual(self.panel.menu.entrycget(0, "state"), "disabled")


class FormatTest(unittest.TestCase):
    def test_bars_and_prefixes(self):
        self.assertEqual(PANEL.text_bar(0.5), "█" * 8 + "░" * 8)
        self.assertEqual(PANEL._measure(3, 4, None), PANEL.text_bar(0.75) + " 3/4")
        self.assertEqual(PANEL._measure(None, None, 40), PANEL.text_bar(0.4) + " 40%")
        self.assertEqual(PANEL._measure(None, None, None), "")
        self.assertEqual(PANEL.log_prefix(None, None), "")
        self.assertEqual(PANEL.log_prefix("+22+112", None), "+22+112       │ ")
        self.assertEqual(PANEL.log_prefix("+22+112", "veg_overlay"), "+22+112 VEG   │ ")


if __name__ == "__main__":
    unittest.main()
