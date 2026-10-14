"""Pause/resume of tile builds: the pause gate and the per-run batch journal."""

import json
import os
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


class JournalTestCase(unittest.TestCase):
    """Redirects the journal directory and keeps module state out of the way."""

    steps = {"osm": True, "mesh": True, "dsf": True}
    tiles = [(45, 5), (45, 6)]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name) / "build_state"
        self._orig_state_dir = BSTATE.state_dir
        self._orig_user_path = BSTATE.FNAMES.user_path
        BSTATE.state_dir = lambda: str(self.dir)
        # Keeps the legacy-file migration looking at the sandbox, not the repo.
        BSTATE.FNAMES.user_path = lambda rel: str(Path(self.tmp.name) / rel)
        self.addCleanup(self._restore)

    def _restore(self):
        BSTATE.close()
        BSTATE.state_dir = self._orig_state_dir
        BSTATE.FNAMES.user_path = self._orig_user_path

    def _begin(self, tiles=None, resume_path=None):
        return BSTATE.begin(
            tiles or self.tiles, self.steps, "", False, resume_path=resume_path
        )

    def _journals(self):
        return sorted(p.name for p in self.dir.glob("batch-*.json"))


class BuildJournalTest(JournalTestCase):
    def test_only_selected_steps_are_journalled(self):
        state = self._begin()
        self.assertEqual(state["steps"], ["osm", "mesh", "dsf"])
        self.assertEqual(len(self._journals()), 1)

    def test_progress_survives_a_reload(self):
        self._begin()
        BSTATE.mark_done(45, 5, "osm")
        BSTATE.mark_done(45, 5, "mesh")
        BSTATE.close()

        found = BSTATE.resumable()
        self.assertIsNotNone(found)
        path, state = found
        self.assertEqual(BSTATE.pending_tiles(state), [(45, 5), (45, 6)])

        self._begin(resume_path=path)
        self.assertTrue(BSTATE.is_done(45, 5, "osm"))
        self.assertTrue(BSTATE.is_done(45, 5, "mesh"))
        self.assertFalse(BSTATE.is_done(45, 5, "dsf"))
        self.assertFalse(BSTATE.is_done(45, 6, "osm"))

    def test_a_finished_tile_drops_out_of_the_pending_list(self):
        self._begin()
        for step in self.steps:
            BSTATE.mark_done(45, 5, step)
        state = BSTATE.scan()[0][1]
        self.assertEqual(BSTATE.pending_tiles(state), [(45, 6)])

    def test_a_completed_batch_leaves_no_journal_behind(self):
        self._begin()
        for lat, lon in self.tiles:
            for step in self.steps:
                BSTATE.mark_done(lat, lon, step)
        BSTATE.complete()
        self.assertEqual(self._journals(), [])
        self.assertIsNone(BSTATE.resumable())

    def test_a_stopped_batch_is_offered_for_resume(self):
        self._begin()
        BSTATE.mark_done(45, 5, "osm")
        BSTATE.set_status("stopped")
        BSTATE.close()
        self.assertIsNotNone(BSTATE.resumable())

    def test_a_journal_with_pre_rename_step_keys_still_resumes(self):
        steps = {"dsf": True, "bld_overlay": True, "veg_overlay": True}
        BSTATE.begin(self.tiles, steps, "", False)
        path = BSTATE.journal_path()
        BSTATE.close()
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        state["steps"] = ["dsf", "sfr_bld", "sfr_veg"]
        state["progress"] = {"+45+005": ["dsf", "sfr_bld"]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f)

        found = BSTATE.resumable()
        self.assertIsNotNone(found)
        self.assertEqual(found[1]["steps"], ["dsf", "bld_overlay", "veg_overlay"])
        BSTATE.begin(self.tiles, steps, "", False, resume_path=found[0])
        self.assertTrue(BSTATE.is_done(45, 5, "bld_overlay"))
        self.assertFalse(BSTATE.is_done(45, 5, "veg_overlay"))

    def test_resume_against_a_different_batch_starts_afresh(self):
        # Skipping steps recorded for another tile set would ship half-built
        # tiles, so a mismatched journal is not trusted.
        self._begin()
        BSTATE.mark_done(45, 5, "osm")
        path = BSTATE.journal_path()
        BSTATE.close()

        self._begin(tiles=[(45, 5), (46, 6)], resume_path=path)
        self.assertFalse(BSTATE.is_done(45, 5, "osm"))

    def _stopped(self, tiles, steps=None, folder=""):
        BSTATE.begin(tiles, steps or self.steps, folder, False)
        BSTATE.set_status("stopped")
        BSTATE.close()

    def test_a_new_batch_supersedes_stopped_ones_on_its_tiles(self):
        # Reported 2026-09-18: stop a 1-tile and a 2-tile batch, finish one of
        # the tiles in a third batch - both stopped journals kept being
        # announced as unfinished.
        self._stopped([(13, 79)])
        self._stopped([(13, 77), (13, 79)])
        self._begin(tiles=[(13, 77), (13, 79)])
        self.assertEqual(len(self._journals()), 1)
        BSTATE.complete()
        self.assertEqual(self._journals(), [])
        self.assertIsNone(BSTATE.resumable())

    def test_a_stopped_batch_keeps_the_tiles_the_new_one_skips(self):
        self._stopped([(13, 77), (13, 79)])
        self._begin(tiles=[(13, 77)])
        BSTATE.complete()
        entries = BSTATE.resumable_all()
        self.assertEqual(len(entries), 1)
        self.assertEqual(BSTATE.pending_tiles(entries[0][1]), [(13, 79)])

    def test_a_stopped_batch_keeps_the_steps_the_new_one_skips(self):
        self._stopped([(45, 5)])
        BSTATE.begin([(45, 5)], {"osm": True}, "", False)
        BSTATE.complete()
        path, state = BSTATE.resumable()
        self.assertEqual(state["progress"], {"+45+005": ["osm"]})
        self.assertEqual(BSTATE.pending_tiles(state), [(45, 5)])

    def test_another_base_folder_is_a_different_build(self):
        self._stopped([(45, 5)], folder="H:\\Tiles")
        self._begin(tiles=[(45, 5)])
        BSTATE.complete()
        self.assertEqual(len(BSTATE.resumable_all()), 1)

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
        self.assertEqual(self._journals(), [])
        self.assertIsNone(BSTATE.resumable())

    def test_a_corrupt_journal_is_ignored(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "batch-1-deadbeef.json").write_text(
            "{not json", encoding="utf-8"
        )
        self.assertEqual(BSTATE.scan(), [])
        self.assertIsNone(BSTATE.resumable())

    def test_a_journal_from_another_version_is_ignored(self):
        self._begin()
        path = Path(BSTATE.journal_path())
        BSTATE.close()
        state = json.loads(path.read_text(encoding="utf-8"))
        state["version"] = BSTATE.VERSION + 1
        path.write_text(json.dumps(state), encoding="utf-8")
        self.assertEqual(BSTATE.scan(), [])

    def test_a_single_file_journal_is_carried_over(self):
        legacy = Path(self.tmp.name) / ".batch_build_state.json"
        legacy.write_text(
            json.dumps(
                {
                    "version": 1,
                    "started": "2026-09-03 10:00:00",
                    "status": "stopped",
                    "custom_build_dir": "",
                    "override_cfg": False,
                    "steps": ["osm", "mesh", "dsf"],
                    "tiles": [[45, 5], [45, 6]],
                    "progress": {"+45+005": ["osm"]},
                }
            ),
            encoding="utf-8",
        )
        found = BSTATE.resumable()
        self.assertIsNotNone(found)
        _path, state = found
        self.assertEqual(state["progress"]["+45+005"], ["osm"])
        self.assertFalse(legacy.exists())


class AirportBatchJournalTest(JournalTestCase):
    """An airport batch builds each tile its own steps; its journal says so."""

    tile_steps = {(45, 5): {"dsf"}, (45, 6): {"osm", "dsf"}}

    def test_the_journal_keeps_each_tiles_steps(self):
        state = BSTATE.begin(self.tiles, self.steps, "", False, tile_steps=self.tile_steps)
        self.assertEqual(state["tile_steps"], {"+45+005": ["dsf"], "+45+006": ["osm", "dsf"]})
        self.assertEqual(BSTATE.tile_steps_of(state), self.tile_steps)
        BSTATE.close()
        _path, reread = BSTATE.resumable()
        self.assertEqual(BSTATE.tile_steps_of(reread), self.tile_steps)

    def test_an_ordinary_batch_has_no_tile_steps(self):
        self.assertIsNone(BSTATE.tile_steps_of(self._begin()))

    def test_matches_compares_each_tiles_steps(self):
        state = BSTATE.begin(self.tiles, self.steps, "", False, tile_steps=self.tile_steps)
        self.assertTrue(
            BSTATE.matches(state, self.tiles, self.steps, "", False, self.tile_steps)
        )
        self.assertFalse(BSTATE.matches(state, self.tiles, self.steps, "", False))
        self.assertFalse(BSTATE.matches(
            state, self.tiles, self.steps, "", False,
            {(45, 5): {"dsf"}, (45, 6): {"dsf"}},
        ))

    def test_it_only_strikes_its_own_steps_off_stopped_batches(self):
        BSTATE.begin(self.tiles, {"osm": True, "dsf": True}, "", False)
        BSTATE.set_status("stopped")
        BSTATE.close()
        BSTATE.begin(
            self.tiles, {"osm": True, "dsf": True}, "", False,
            tile_steps={(45, 5): {"dsf"}, (45, 6): {"osm"}},
        )
        BSTATE.complete()
        _path, state = BSTATE.resumable()
        self.assertEqual(state["progress"], {"+45+005": ["dsf"], "+45+006": ["osm"]})
        self.assertEqual(BSTATE.pending_tiles(state), [(45, 5), (45, 6)])


class ParallelInstanceTest(JournalTestCase):
    """Several copies of Ortho4XP run at once; journals must not collide."""

    _serial = 0

    def _foreign_journal(self, tiles, owner):
        """Write a journal as if another process had started that batch."""
        state = BSTATE._blank(tiles, self.steps, "", False)
        state["owner"] = owner
        ParallelInstanceTest._serial += 1
        path = str(
            self.dir
            / ("batch-%s-%08d.json" % (os.getpid(), ParallelInstanceTest._serial))
        )
        BSTATE._write(path, state)
        return path

    @staticmethod
    def _dead_owner():
        # A pid that exists but was created at the dawn of time cannot be the
        # owner; the mismatch is what rules out a recycled pid.
        return {"pid": os.getpid(), "created": 1.0}

    @staticmethod
    def _live_owner():
        return BSTATE._owner_block()

    def test_two_runs_keep_separate_journals(self):
        first = self._begin()
        self.assertEqual(first["tiles"], [[45, 5], [45, 6]])
        first_path = BSTATE.journal_path()
        BSTATE.mark_done(45, 5, "osm")
        BSTATE.close()

        self._begin(tiles=[(46, 7)])
        BSTATE.mark_done(46, 7, "osm")
        second_path = BSTATE.journal_path()

        self.assertNotEqual(first_path, second_path)
        self.assertEqual(len(self._journals()), 2)
        # The first run's progress is untouched by the second.
        state = BSTATE._read(first_path)
        self.assertEqual(state["progress"]["+45+005"], ["osm"])

    def test_a_batch_owned_by_a_live_instance_is_not_offered(self):
        path = self._foreign_journal([(50, 1)], self._live_owner())
        self.assertEqual(BSTATE.resumable_all(), [])
        self.assertEqual([p for p, _s in BSTATE.live_batches()], [path])

    def test_a_batch_whose_instance_died_is_offered(self):
        self._foreign_journal([(50, 1)], self._dead_owner())
        self.assertEqual(len(BSTATE.resumable_all()), 1)
        self.assertEqual(BSTATE.live_batches(), [])

    def test_our_own_running_batch_is_not_offered_to_us(self):
        self._begin()
        self.assertEqual(BSTATE.resumable_all(), [])

    def test_busy_tiles_come_from_live_instances_only(self):
        self._foreign_journal([(50, 1), (50, 2)], self._live_owner())
        self._foreign_journal([(60, 3)], self._dead_owner())
        self.assertEqual(BSTATE.busy_tiles(), {(50, 1), (50, 2)})

    def test_claiming_a_journal_hands_over_ownership(self):
        path = self._foreign_journal([(50, 1)], self._dead_owner())
        claimed = BSTATE.claim(path)
        self.assertIsNotNone(claimed)
        self.assertFalse(os.path.exists(path))
        state = BSTATE._read(claimed)
        self.assertEqual(state["owner"]["pid"], os.getpid())
        self.assertEqual(state["progress"], {})

    def test_only_one_instance_can_claim_a_journal(self):
        # The rename is the claim, so the loser gets None rather than a
        # duplicate build of the same tiles.
        path = self._foreign_journal([(50, 1)], self._dead_owner())
        self.assertIsNotNone(BSTATE.claim(path))
        self.assertIsNone(BSTATE.claim(path))

    def test_several_abandoned_batches_are_all_listed(self):
        self._foreign_journal([(50, 1)], self._dead_owner())
        self._foreign_journal([(60, 2)], {"pid": os.getpid(), "created": 2.0})
        self.assertEqual(len(BSTATE.resumable_all()), 2)

    def test_clearing_abandons_dead_batches_but_spares_live_ones(self):
        # Another instance is still writing the live journal; deleting it
        # would lose that run's progress record mid-build.
        live = self._foreign_journal([(50, 1)], self._live_owner())
        self._foreign_journal([(60, 2)], self._dead_owner())
        self._foreign_journal([(70, 3)], {"pid": os.getpid(), "created": 2.0})

        self.assertEqual(BSTATE.discard_abandoned(), 2)
        self.assertEqual(BSTATE.resumable_all(), [])
        self.assertEqual([p for p, _s in BSTATE.live_batches()], [live])
        self.assertTrue(os.path.exists(live))

    def test_clearing_spares_our_own_running_batch(self):
        self._begin()
        ours = BSTATE.journal_path()
        self.assertEqual(BSTATE.discard_abandoned(), 0)
        self.assertTrue(os.path.exists(ours))

    def test_clearing_also_sweeps_journals_with_nothing_left(self):
        path = self._foreign_journal([(50, 1)], self._dead_owner())
        state = BSTATE._read(path)
        state["progress"] = {"+50+001": list(self.steps)}
        BSTATE._write(path, state)
        self.assertEqual(BSTATE.resumable_all(), [])
        self.assertEqual(BSTATE.discard_abandoned(), 1)
        self.assertFalse(os.path.exists(path))

    def test_clearing_with_nothing_on_record_is_a_no_op(self):
        self.assertEqual(BSTATE.discard_abandoned(), 0)

    def test_an_unowned_journal_is_judged_by_its_heartbeat(self):
        # Journals carried over from the single-file layout have no owner, so
        # a stale mtime is the only sign their instance is gone.
        path = self._foreign_journal([(50, 1)], None)
        self.assertEqual(BSTATE.resumable_all(), [], "a fresh touch means live")
        old = time.time() - BSTATE.STALE_AFTER_S - 60
        os.utime(path, (old, old))
        self.assertEqual(len(BSTATE.resumable_all()), 1)


class StopEndsTheBatchTest(JournalTestCase):
    """Stop finishes a batch; closing the window or a crash does not."""

    def setUp(self):
        super().setUp()
        import O4_Tile_Utils as TILE

        self.TILE = TILE
        self.addCleanup(self._reset_flags)

    def _reset_flags(self):
        UI.red_flag = False
        UI.user_stopped = False

    def test_stop_button_drops_the_journal(self):
        self._begin()
        BSTATE.mark_done(45, 5, "osm")
        UI.user_stopped = True
        UI.red_flag = True
        self.TILE._end_journal_on_stop()
        BSTATE.release()
        self.assertEqual(self._journals(), [])
        self.assertIsNone(BSTATE.resumable())

    def test_closing_the_window_keeps_the_batch_resumable(self):
        # exit_prg raises red_flag without user_stopped.
        self._begin()
        BSTATE.mark_done(45, 5, "osm")
        UI.red_flag = True
        self.TILE._end_journal_on_stop()
        BSTATE.release()
        self.assertIsNotNone(BSTATE.resumable())

    def test_only_the_stop_button_marks_a_user_stop(self):
        source = (SRC / "O4_GUI_Utils.py").read_text(encoding="utf-8")
        stop = source.split("def set_red_flag(self):")[1].split("\n    def ")[0]
        self.assertIn("UI.user_stopped = True", stop)
        close = source.split("def exit_prg(self)")[1].split("\n    def ")[0]
        self.assertNotIn("user_stopped", close)

    def test_every_stop_exit_in_the_batch_loop_settles_the_journal(self):
        source = (SRC / "O4_Tile_Utils.py").read_text(encoding="utf-8")
        body = (
            source.split("def _build_tile_list(")[1].split("\ndef ")[0]
            + source.split("def _build_tiles(")[1].split("\ndef ")[0]
        )
        self.assertNotIn('BSTATE.set_status("stopped")', body)
        self.assertEqual(
            body.count("if UI.red_flag:"), body.count("_end_journal_on_stop()")
        )
        self.assertIn("UI.user_stopped = False", body)


class AirportBatchResumeTest(JournalTestCase):
    """An airport batch cut short resumes with each tile's own steps."""

    tiles = [(45, 5), (45, 6), (45, 7)]
    tile_steps = {(45, 5): {"dsf", "veg_overlay"}, (45, 6): {"dsf"}, (45, 7): {"osm"}}

    def setUp(self):
        super().setUp()
        from unittest import mock

        import O4_Tile_Utils as TILE

        self.TILE = TILE
        self.ran = []
        self.interrupt_at = None

        def step(key):
            def run(tile, *_args):
                self.ran.append((tile.lat, tile.lon, key))
                if key == "dsf" and (tile.lat, tile.lon) == self.interrupt_at:
                    UI.red_flag = True  # the window closed during this DSF
                    return 0
                return 1
            return run

        def overlays(lat, lon, build_dir, do_bld=True, do_veg=True):
            errors = {}
            if do_bld:
                self.ran.append((lat, lon, "bld_overlay"))
                errors["bld"] = None
            if do_veg:
                self.ran.append((lat, lon, "veg_overlay"))
                errors["veg"] = None
            return errors

        class FakeTile:
            lat = lon = None
            custom_build_dir = ""
            build_dir = ""
            dem = None

            def read_from_config(self, *a, **kw):
                return 1

            def make_dirs(self):
                pass

        self.tile = FakeTile()
        for patch in (
            mock.patch.object(TILE.VMAP, "build_poly_file", step("osm")),
            mock.patch.object(TILE.MESH, "build_mesh", step("mesh")),
            mock.patch.object(TILE.MASK, "build_masks", step("mask")),
            mock.patch.object(TILE, "_build_tile_retrying_white", step("dsf")),
            mock.patch.object(TILE, "_batch_airport_zone_plan", lambda *a: {}),
            mock.patch.object(TILE.PIPELINE, "process_overlays_tile", overlays),
            mock.patch.object(TILE.PIPELINE, "apply_tile_settings", lambda *a, **kw: None),
            mock.patch.object(TILE.SLINK, "auto_link_tile", lambda *a, **kw: None),
            mock.patch.object(TILE, "_clear_todo_marker", lambda *a, **kw: None),
            mock.patch.object(TILE.FNAMES, "build_dir", lambda *a, **kw: ""),
            mock.patch.object(UI, "lvprint", lambda *a, **kw: None),
            mock.patch.object(UI, "vprint", lambda *a, **kw: None),
            mock.patch.object(UI, "exit_message_and_bottom_line", lambda *a, **kw: None),
            mock.patch.dict(os.environ, {"O4_BATCH_PIPELINE": "0"}),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        UI.red_flag = False
        UI.user_stopped = False
        UI.is_working = 0
        self.addCleanup(setattr, UI, "red_flag", False)

    def _build(self, steps, tile_steps, resume_path=None):
        return self.TILE.build_tile_list(
            self.tile, self.tiles,
            steps.get("osm"), steps.get("mesh"), steps.get("mask"), steps.get("dsf"),
            steps.get("ovl"), steps.get("bld_overlay"), steps.get("veg_overlay"),
            False, resume_path, steps.get("color"), steps.get("clouds"),
            tile_steps=tile_steps,
        )

    def test_resume_runs_only_what_each_tile_still_owes(self):
        steps = {"osm": True, "dsf": True, "veg_overlay": True}
        self.interrupt_at = (45, 6)
        self.assertEqual(self._build(steps, self.tile_steps), 0)
        self.assertEqual(self.ran, [
            (45, 5, "dsf"), (45, 5, "veg_overlay"), (45, 6, "dsf"),
        ])

        # Resume Build: the journal brings back each tile's steps.
        path, state = BSTATE.resumable()
        self.assertEqual(BSTATE.pending_tiles(state), [(45, 6), (45, 7)])
        tile_steps = BSTATE.tile_steps_of(state)
        self.assertEqual(tile_steps, self.tile_steps)
        claimed = BSTATE.claim(path)
        self.ran.clear()
        UI.red_flag = False
        self.interrupt_at = None
        resumed = {step: step in state["steps"] for step in BSTATE.STEPS}
        self.assertEqual(self._build(resumed, tile_steps, claimed), 1)
        # The interrupted DSF is redone; nothing already built or never asked
        # for runs - no OSM on +45+006, no DSF or VEG on +45+007.
        self.assertEqual(self.ran, [(45, 6, "dsf"), (45, 7, "osm")])
        self.assertIsNone(BSTATE.resumable())

        snap = self.TILE.PROG.snapshot()["run"]
        shown = {t["name"]: {s["key"]: s["state"] for s in t["steps"]} for t in snap["tiles"]}
        self.assertEqual(shown["+45+005"], {"dsf": "skipped", "veg_overlay": "skipped"})
        self.assertEqual(shown["+45+006"], {"dsf": "done"})
        self.assertEqual(shown["+45+007"], {"osm": "done"})

    def test_rerunning_the_same_airport_batch_offers_its_journal(self):
        steps = {"osm": True, "dsf": True, "veg_overlay": True}
        self.interrupt_at = (45, 6)
        self._build(steps, self.tile_steps)
        _path, state = BSTATE.resumable()
        # The batch window's "resume this exact batch?" check.
        self.assertTrue(BSTATE.matches(state, self.tiles, steps, "", False, self.tile_steps))
        other = {**self.tile_steps, (45, 7): {"osm", "dsf"}}
        self.assertFalse(BSTATE.matches(state, self.tiles, steps, "", False, other))


class BuildTileListWiringTest(unittest.TestCase):
    """The batch loop must take a journal and honour the skips it records."""

    def test_build_tile_list_accepts_a_journal_to_resume(self):
        import inspect

        import O4_Tile_Utils as TILE

        signature = inspect.signature(TILE.build_tile_list)
        self.assertIn("resume_path", signature.parameters)
        self.assertIsNone(signature.parameters["resume_path"].default)

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
