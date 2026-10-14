"""The progress model behind the main window's build tree (O4_Progress)."""

import os
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_Progress as PROG  # noqa: E402


def _step(snap, tile, key):
    for t in snap["run"]["tiles"]:
        if t["name"] == tile:
            for s in t["steps"]:
                if s["key"] == key:
                    return s
    raise KeyError((tile, key))


def _tile(snap, name):
    return next(t for t in snap["run"]["tiles"] if t["name"] == name)


class StepStatesTest(unittest.TestCase):
    def setUp(self):
        PROG.is_stopped = lambda: False
        PROG.begin_run([(22, 112), (22, 113)], ["osm", "dsf"])

    def test_steps_start_queued_and_end_done(self):
        self.assertEqual(_step(PROG.snapshot(), "+22+112", "osm")["state"], PROG.QUEUED)
        with PROG.step(22, 112, "osm") as scope:
            self.assertEqual(_step(PROG.snapshot(), "+22+112", "osm")["state"], PROG.RUNNING)
            scope.result(1)
        self.assertEqual(_step(PROG.snapshot(), "+22+112", "osm")["state"], PROG.DONE)

    def test_a_falsy_result_or_an_exception_fails_the_step(self):
        with PROG.step(22, 112, "osm") as scope:
            scope.result(0)
        with self.assertRaises(ValueError):
            with PROG.step(22, 112, "dsf"):
                raise ValueError("boom")
        snap = PROG.snapshot()
        self.assertEqual(_step(snap, "+22+112", "osm")["state"], PROG.FAILED)
        self.assertEqual(_step(snap, "+22+112", "dsf")["state"], PROG.FAILED)
        self.assertEqual(_tile(snap, "+22+112")["state"], PROG.FAILED)

    def test_issues_make_a_step_done_with_issues(self):
        with PROG.step(22, 112, "dsf") as scope:
            PROG.issue("a.dds", "download", "failed after 3 attempts")
            scope.result(1)
        snap = PROG.snapshot()
        step = _step(snap, "+22+112", "dsf")
        self.assertEqual(step["state"], PROG.ISSUES)
        self.assertEqual(step["issues"], [("a.dds", "download", "failed after 3 attempts")])
        self.assertEqual(PROG.issues_by_tile(), {(22, 112): {"dsf": step["issues"]}})
        self.assertIn("   DSF: a.dds: download: failed after 3 attempts", PROG.summary_lines())

    def test_a_stop_marks_the_step_stopped(self):
        PROG.is_stopped = lambda: True
        with PROG.step(22, 112, "osm") as scope:
            scope.result(0)
        self.assertEqual(_step(PROG.snapshot(), "+22+112", "osm")["state"], PROG.STOPPED)

    def test_a_step_outside_any_run_opens_and_closes_its_own(self):
        PROG.end_run()
        with PROG.step(10, 20, "mesh") as scope:
            run = PROG.snapshot()["run"]
            self.assertEqual(run["state"], PROG.RUNNING)
            self.assertEqual([t["name"] for t in run["tiles"]], ["+10+020"])
            scope.result(1)
        self.assertEqual(PROG.snapshot()["run"]["state"], PROG.DONE)

    def test_skips(self):
        PROG.tile_skipped(22, 113)
        PROG.step_skipped(22, 112, "osm")
        snap = PROG.snapshot()
        self.assertEqual(_tile(snap, "+22+113")["state"], PROG.SKIPPED)
        self.assertEqual(_step(snap, "+22+112", "osm")["state"], PROG.SKIPPED)


class ElapsedTest(unittest.TestCase):
    """Wall clock from start to now (or end): overlapping steps are never summed."""

    def test_parallel_steps_do_not_add_up(self):
        PROG.is_stopped = lambda: False
        PROG.begin_run([(22, 112)], ["bld_overlay", "veg_overlay"])
        clock = [1000.0]
        with mock.patch.object(PROG.time, "time", lambda: clock[0]):
            with PROG.step(22, 112, "bld_overlay"):
                with PROG.step(22, 112, "veg_overlay"):
                    clock[0] += 20
                clock[0] += 5
            clock[0] += 100  # long after both ended
            snap = PROG.snapshot()
        self.assertEqual(_step(snap, "+22+112", "veg_overlay")["elapsed"], 20)
        self.assertEqual(_step(snap, "+22+112", "bld_overlay")["elapsed"], 25)
        self.assertEqual(_tile(snap, "+22+112")["elapsed"], 25)

    def test_running_times_tick_and_end_run_stops_the_clock(self):
        clock = [500.0]
        with mock.patch.object(PROG.time, "time", lambda: clock[0]):
            PROG.begin_run([(1, 2)], ["osm"])
            clock[0] += 7
            self.assertEqual(PROG.snapshot()["run"]["elapsed"], 7)
            PROG.end_run()
            clock[0] += 50
            self.assertEqual(PROG.snapshot()["run"]["elapsed"], 7)

    def test_format(self):
        self.assertEqual(PROG.format_elapsed(None), "")
        self.assertEqual(PROG.format_elapsed(9.6), "9s")
        self.assertEqual(PROG.format_elapsed(125), "2m05s")
        self.assertEqual(PROG.format_elapsed(3 * 3600 + 7 * 60 + 3), "3h07m")


class ThreadContextTest(unittest.TestCase):
    def setUp(self):
        PROG.is_stopped = lambda: False
        PROG.begin_run([(22, 112)], ["dsf"])

    def test_bound_threads_report_to_their_parents_step(self):
        with PROG.step(22, 112, "dsf"):
            def worker():
                PROG.bar(2, 40)
                PROG.items(part="Download", done=3, total=10)
            t = threading.Thread(target=PROG.bind(worker))
            t.start()
            t.join()
            parts = {p["label"]: p for p in _step(PROG.snapshot(), "+22+112", "dsf")["parts"]}
        self.assertEqual(parts["Download"]["pct"], 40)
        self.assertEqual((parts["Download"]["done"], parts["Download"]["total"]), (3, 10))

    def test_bar_one_is_the_steps_own_progress_outside_step_3(self):
        PROG.begin_run([(22, 112)], ["mask"])
        with PROG.step(22, 112, "mask"):
            PROG.bar(1, 55)
            step = _step(PROG.snapshot(), "+22+112", "mask")
        self.assertEqual(step["pct"], 55)
        self.assertEqual(step["parts"], [])

    def test_an_unbound_thread_reports_nowhere_in_the_tree(self):
        with PROG.step(22, 112, "dsf"):
            t = threading.Thread(target=lambda: PROG.bar(1, 70))
            t.start()
            t.join()
            snap = PROG.snapshot()
        self.assertEqual(_step(snap, "+22+112", "dsf")["parts"], [])
        self.assertEqual(snap["loose"], {1: 70})

    def test_context_is_restored_after_the_step(self):
        self.assertEqual(PROG.current(), (None, None))
        with PROG.step(22, 112, "dsf"):
            self.assertEqual(PROG.current(), ("+22+112", "dsf"))
        self.assertEqual(PROG.current(), (None, None))


class WorkerEventsTest(unittest.TestCase):
    def setUp(self):
        PROG.is_stopped = lambda: False
        PROG.begin_run([(22, 112)], ["bld_overlay"])

    def test_event_lines_are_applied_and_swallowed(self):
        with PROG.step(22, 112, "bld_overlay") as scope:
            self.assertFalse(PROG.handle_line("  [ 1/64] a.dds (starting)\n"))
            self.assertTrue(PROG.handle_line(PROG.EVENT_PREFIX + '{"ev":"items","done":0,"total":4}\n'))
            for _ in range(5):  # one too many never goes past the total
                PROG.handle_line(PROG.EVENT_PREFIX + '{"ev":"items","add":1}\n')
            PROG.handle_line(PROG.EVENT_PREFIX + '{"ev":"issue","item":"b.dds","stage":"building overlay","msg":"OOM"}\n')
            self.assertTrue(PROG.handle_line(PROG.EVENT_PREFIX + "not json\n"))
            scope.result(1)
        step = _step(PROG.snapshot(), "+22+112", "bld_overlay")
        self.assertEqual((step["done"], step["total"]), (4, 4))
        self.assertEqual(step["state"], PROG.ISSUES)
        self.assertEqual(step["issues"], [("b.dds", "building overlay", "OOM")])

    def test_emit_is_quiet_unless_a_parent_listens(self):
        out = []
        with mock.patch.object(sys, "stdout", mock.Mock(write=out.append, flush=lambda: None)):
            with mock.patch.dict(os.environ, {PROG.EVENT_ENV: ""}):
                PROG.emit(ev="items", add=1)
            self.assertEqual(out, [])
            with mock.patch.dict(os.environ, {PROG.EVENT_ENV: "1"}):
                PROG.emit(ev="items", add=1)
        self.assertEqual(out, [PROG.EVENT_PREFIX + '{"ev":"items","add":1}\n'])


class VersionTest(unittest.TestCase):
    def test_every_change_bumps_the_version(self):
        PROG.begin_run([(1, 1)], ["osm"])
        v = PROG.version()
        with PROG.step(1, 1, "osm"):
            PROG.bar(1, 10)
        self.assertGreaterEqual(PROG.version(), v + 3)


if __name__ == "__main__":
    unittest.main()
