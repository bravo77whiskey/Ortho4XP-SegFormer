"""Pause/resume of tile builds: the pause gate and the batch journal."""

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_UI_Utils as UI
import O4_Build_State as BSTATE


class PauseGateTest(unittest.TestCase):
    def setUp(self):
        UI.red_flag = False
        UI._clear_pause()
        self.addCleanup(self._reset)

    def _reset(self):
        UI.red_flag = False
        UI._clear_pause()

    def test_check_pause_returns_at_once_when_running(self):
        started = time.time()
        UI.check_pause()
        self.assertLess(time.time() - started, 0.2)

    def test_worker_parks_until_resumed(self):
        passed = threading.Event()

        def worker():
            UI.check_pause()
            passed.set()

        self.assertTrue(UI.request_pause())
        self.assertTrue(UI.is_paused())
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        self.assertFalse(passed.wait(0.4), "worker ran while the build was paused")

        self.assertTrue(UI.request_resume())
        self.assertTrue(passed.wait(2), "worker stayed parked after Resume")
        thread.join(2)
        self.assertFalse(UI.is_paused())

    def test_stop_releases_a_paused_worker(self):
        # Stop must never be swallowed by a pause, or the GUI would hang.
        passed = threading.Event()

        def worker():
            UI.check_pause()
            passed.set()

        UI.request_pause()
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        self.assertFalse(passed.wait(0.4))
        UI.red_flag = True
        self.assertTrue(passed.wait(2), "Stop did not release the paused worker")
        thread.join(2)

    def test_kill_all_subprocesses_lifts_the_pause(self):
        UI.request_pause()
        UI.kill_all_subprocesses()
        self.assertFalse(UI.is_paused())

    def test_stop_requested_reports_the_red_flag(self):
        self.assertFalse(UI.stop_requested())
        UI.red_flag = True
        self.assertTrue(UI.stop_requested())

    def test_toggle_flips_both_ways(self):
        self.assertTrue(UI.toggle_pause())
        self.assertTrue(UI.is_paused())
        self.assertTrue(UI.toggle_pause())
        self.assertFalse(UI.is_paused())


class BuildJournalTest(unittest.TestCase):
    steps = {"osm": True, "mesh": True, "dsf": True}
    tiles = [(45, 5), (45, 6)]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name) / ".batch_build_state.json"
        self._orig_state_file = BSTATE.state_file
        BSTATE.state_file = lambda: str(path)
        self.path = path
        self.addCleanup(self._restore)

    def _restore(self):
        BSTATE.state_file = self._orig_state_file
        BSTATE.close()

    def _begin(self, resume=False):
        return BSTATE.begin(self.tiles, self.steps, "", False, resume=resume)

    def test_only_selected_steps_are_journalled(self):
        state = self._begin()
        self.assertEqual(state["steps"], ["osm", "mesh", "dsf"])
        self.assertTrue(self.path.is_file())

    def test_progress_survives_a_reload(self):
        self._begin()
        BSTATE.mark_done(45, 5, "osm")
        BSTATE.mark_done(45, 5, "mesh")
        BSTATE.close()

        state = BSTATE.resumable()
        self.assertIsNotNone(state)
        self.assertEqual(BSTATE.pending_tiles(state), [(45, 5), (45, 6)])

        self._begin(resume=True)
        self.assertTrue(BSTATE.is_done(45, 5, "osm"))
        self.assertTrue(BSTATE.is_done(45, 5, "mesh"))
        self.assertFalse(BSTATE.is_done(45, 5, "dsf"))
        self.assertFalse(BSTATE.is_done(45, 6, "osm"))

    def test_a_finished_tile_drops_out_of_the_pending_list(self):
        self._begin()
        for step in self.steps:
            BSTATE.mark_done(45, 5, step)
        state = BSTATE.load()
        self.assertEqual(BSTATE.pending_tiles(state), [(45, 6)])

    def test_completed_batch_is_not_offered_for_resume(self):
        self._begin()
        for lat, lon in self.tiles:
            for step in self.steps:
                BSTATE.mark_done(lat, lon, step)
        BSTATE.complete()
        self.assertIsNone(BSTATE.resumable())

    def test_a_stopped_batch_is_offered_for_resume(self):
        self._begin()
        BSTATE.mark_done(45, 5, "osm")
        BSTATE.set_status("stopped")
        BSTATE.close()
        self.assertIsNotNone(BSTATE.resumable())

    def test_resume_against_a_different_batch_starts_afresh(self):
        # Skipping steps recorded for another tile set would ship half-built
        # tiles, so a mismatched journal is thrown away rather than trusted.
        self._begin()
        BSTATE.mark_done(45, 5, "osm")
        BSTATE.close()

        BSTATE.begin([(45, 5), (46, 6)], self.steps, "", False, resume=True)
        self.assertFalse(BSTATE.is_done(45, 5, "osm"))

    def test_matches_is_picky_about_the_whole_batch(self):
        state = self._begin()
        self.assertTrue(BSTATE.matches(state, self.tiles, self.steps, "", False))
        self.assertFalse(
            BSTATE.matches(state, [(45, 5)], self.steps, "", False)
        )
        self.assertFalse(
            BSTATE.matches(state, self.tiles, {"osm": True}, "", False)
        )
        self.assertFalse(
            BSTATE.matches(state, self.tiles, self.steps, "H:\\Tiles", False)
        )
        self.assertFalse(
            BSTATE.matches(state, self.tiles, self.steps, "", True)
        )

    def test_discard_removes_the_journal(self):
        self._begin()
        BSTATE.discard()
        self.assertFalse(self.path.exists())
        self.assertIsNone(BSTATE.load())

    def test_a_corrupt_journal_is_ignored(self):
        self.path.write_text("{not json", encoding="utf-8")
        self.assertIsNone(BSTATE.load())
        self.assertIsNone(BSTATE.resumable())

    def test_a_journal_from_another_version_is_ignored(self):
        self._begin()
        state = json.loads(self.path.read_text(encoding="utf-8"))
        state["version"] = BSTATE.VERSION + 1
        self.path.write_text(json.dumps(state), encoding="utf-8")
        self.assertIsNone(BSTATE.load())


class BuildTileListWiringTest(unittest.TestCase):
    """The batch loop must expose resume and honour the journal's skips."""

    def test_build_tile_list_accepts_resume(self):
        import inspect

        import O4_Tile_Utils as TILE

        signature = inspect.signature(TILE.build_tile_list)
        self.assertIn("resume", signature.parameters)
        self.assertIs(signature.parameters["resume"].default, False)

    def test_every_journal_step_is_wired_into_the_batch_loop(self):
        source = (SRC / "O4_Tile_Utils.py").read_text(encoding="utf-8")
        for step in BSTATE.STEPS:
            self.assertIn(
                'BSTATE.is_done(lat, lon, "%s")' % step,
                source,
                "step %r is never checked against the journal" % step,
            )
            self.assertIn(
                'BSTATE.mark_done(lat, lon, "%s")' % step,
                source,
                "step %r is never recorded in the journal" % step,
            )

    def test_a_step_is_only_journalled_when_it_succeeded(self):
        # Recording a failed step would make the resume skip a tile that was
        # never actually built.
        lines = (SRC / "O4_Tile_Utils.py").read_text(encoding="utf-8").split("\n")
        for index, line in enumerate(lines):
            if "BSTATE.mark_done(lat, lon," not in line:
                continue
            self.assertEqual(
                lines[index - 1].strip(),
                "if done:",
                "unguarded mark_done on line %d: %s" % (index + 1, line.strip()),
            )

    def test_build_masks_reports_success_like_the_other_steps(self):
        source = (SRC / "O4_Mask_Utils.py").read_text(encoding="utf-8")
        body = source.split("def build_masks(")[1].split("\ndef ")[0]
        self.assertIn("\n    return 1\n", body)


if __name__ == "__main__":
    unittest.main()
