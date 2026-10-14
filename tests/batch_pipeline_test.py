"""Batch builds run one tile's overlays while the next tile's steps are built."""

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

import O4_UI_Utils as UI  # noqa: E402
import O4_Tile_Utils as TILE  # noqa: E402


class _Tile:
    def __init__(self):
        self.lat = self.lon = None
        self.custom_build_dir = ""
        self.build_dir = ""
        self.dem = None

    def read_from_config(self, *a, **kw):
        return 1

    def make_dirs(self):
        pass


class BatchPipelineTest(unittest.TestCase):
    TILES = [(10, 20), (10, 21), (10, 22)]

    def _run(self, pipelined, overlay_s=0.3, dsf_s=0.05, overlay_errors=None,
             do_color=False, do_clouds=False, retry=None, tile_steps=None):
        events = []
        self.overlay_calls = []
        lock = threading.Lock()
        done = set()
        inline = []
        fills = []

        def log(kind, lat, lon):
            with lock:
                events.append((time.perf_counter(), kind, (lat, lon)))

        def build_tile(tile):
            log("dsf_start", tile.lat, tile.lon)
            inline.append((tile.color_correction, tile.color_fill_clouds))
            time.sleep(dsf_s)
            log("dsf_end", tile.lat, tile.lon)
            return 1

        def build_color(tile, fill):
            log("color", tile.lat, tile.lon)
            fills.append(fill)
            return 1

        def overlays(lat, lon, build_dir, do_bld=True, do_veg=True):
            log("ovl_start", lat, lon)
            self.overlay_calls.append(((lat, lon), do_bld, do_veg))
            time.sleep(overlay_s)
            log("ovl_end", lat, lon)
            return dict(overlay_errors or {"bld": None, "veg": None})

        patches = [
            mock.patch.object(TILE, "build_tile", build_tile),
            mock.patch.object(TILE, "build_color", build_color),
            mock.patch.object(TILE.PIPELINE, "process_overlays_tile", overlays),
            mock.patch.object(TILE.PIPELINE, "apply_tile_settings", lambda *a, **kw: None),
            mock.patch.object(TILE.SLINK, "auto_link_tile", lambda *a, **kw: None),
            mock.patch.object(TILE, "_clear_todo_marker", lambda *a, **kw: None),
            mock.patch.object(TILE.BSTATE, "begin", lambda *a, **kw: None),
            mock.patch.object(TILE.BSTATE, "release", lambda *a, **kw: None),
            mock.patch.object(TILE.BSTATE, "complete", lambda *a, **kw: None),
            mock.patch.object(TILE.BSTATE, "is_done", lambda lat, lon, step: (lat, lon, step) in done),
            mock.patch.object(TILE.BSTATE, "mark_done", lambda lat, lon, step: done.add((lat, lon, step))),
            mock.patch.object(TILE.FNAMES, "build_dir", lambda *a, **kw: ""),
            mock.patch.object(UI, "lvprint", lambda *a, **kw: self.logged.append(a)),
            mock.patch.object(UI, "vprint", lambda *a, **kw: None),
            mock.patch.object(UI, "exit_message_and_bottom_line", lambda *a, **kw: None),
            mock.patch.dict(os.environ, {"O4_BATCH_PIPELINE": "1" if pipelined else "0"}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        UI.red_flag = False
        UI.is_working = 0
        self.logged = []
        result = TILE.build_tile_list(
            _Tile(), self.TILES, False, False, False, True, False,
            do_bld_overlay=True, do_veg_overlay=True,
            do_color=do_color, do_clouds=do_clouds, retry=retry,
            tile_steps=tile_steps,
        )
        self.inline = inline
        self.fills = fills
        return result, events, done

    @staticmethod
    def _at(events, kind, tile):
        return next(t for t, k, tl in events if k == kind and tl == tile)

    def test_next_tile_is_built_while_overlays_run(self):
        result, events, done = self._run(pipelined=True)
        self.assertEqual(result, 1)
        a, b = self.TILES[0], self.TILES[1]
        self.assertLess(self._at(events, "dsf_start", b), self._at(events, "ovl_end", a))

    def test_overlays_of_two_tiles_never_overlap(self):
        _, events, _ = self._run(pipelined=True)
        for a, b in zip(self.TILES, self.TILES[1:]):
            self.assertLessEqual(self._at(events, "ovl_end", a), self._at(events, "ovl_start", b))

    def test_a_tiles_overlays_start_after_its_own_dsf(self):
        _, events, _ = self._run(pipelined=True)
        for tile in self.TILES:
            self.assertLessEqual(self._at(events, "dsf_end", tile), self._at(events, "ovl_start", tile))

    def test_every_overlay_step_is_journalled(self):
        _, _, done = self._run(pipelined=True)
        for lat, lon in self.TILES:
            self.assertIn((lat, lon, "bld_overlay"), done)
            self.assertIn((lat, lon, "veg_overlay"), done)

    def test_pipelining_can_be_switched_off(self):
        _, events, _ = self._run(pipelined=False)
        a, b = self.TILES[0], self.TILES[1]
        self.assertGreaterEqual(self._at(events, "dsf_start", b), self._at(events, "ovl_end", a))

    def test_a_retry_runs_only_the_steps_it_retries(self):
        result, events, done = self._run(
            pipelined=True, overlay_s=0.01, retry={(10, 21): {"veg_overlay"}, (10, 22): {"dsf"}},
        )
        self.assertEqual(result, 1)
        self.assertEqual([tile for _t, kind, tile in events if kind == "dsf_start"], [(10, 22)])
        self.assertEqual(self.overlay_calls, [((10, 21), False, True)])
        snap = TILE.PROG.snapshot()["run"]
        self.assertEqual(snap["title"], "Retry")
        states = {(t["name"], s["key"]): s["state"] for t in snap["tiles"] for s in t["steps"]}
        self.assertEqual(states[("+10+022", "dsf")], TILE.PROG.DONE)
        self.assertEqual(states[("+10+020", "dsf")], TILE.PROG.SKIPPED)
        self.assertEqual(states[("+10+021", "bld_overlay")], TILE.PROG.SKIPPED)

    def test_an_airport_batch_builds_each_tile_its_own_steps(self):
        result, events, done = self._run(
            pipelined=True, overlay_s=0.01,
            tile_steps={(10, 20): {"dsf"}, (10, 21): {"veg_overlay"}, (10, 22): set()},
        )
        self.assertEqual(result, 1)
        self.assertEqual([tile for _t, kind, tile in events if kind == "dsf_start"], [(10, 20)])
        self.assertEqual(self.overlay_calls, [((10, 21), False, True)])
        snap = TILE.PROG.snapshot()["run"]
        self.assertEqual(snap["title"], "Batch")
        # The progress tree shows only the steps each tile builds.
        shown = {t["name"]: [s["key"] for s in t["steps"]] for t in snap["tiles"]}
        self.assertEqual(shown, {"+10+020": ["dsf"], "+10+021": ["veg_overlay"], "+10+022": []})
        states = {(t["name"], s["key"]): s["state"] for t in snap["tiles"] for s in t["steps"]}
        self.assertEqual(states[("+10+020", "dsf")], TILE.PROG.DONE)
        # The log names the steps each tile builds.
        lines = [" ".join(map(str, a[1:])) for a in self.logged]
        self.assertIn("[Batch] Steps for this tile: DSF.", lines)
        self.assertIn("[Batch] Steps for this tile: VEG.", lines)

    def test_a_failed_overlay_is_not_journalled(self):
        _, _, done = self._run(
            pipelined=True, overlay_errors={"bld": RuntimeError("boom"), "veg": None},
        )
        for lat, lon in self.TILES:
            self.assertNotIn((lat, lon, "bld_overlay"), done)
            self.assertIn((lat, lon, "veg_overlay"), done)
    def test_colour_steps_run_after_dsf_and_before_the_overlays(self):
        _, events, done = self._run(pipelined=True, do_color=True, do_clouds=True)
        for tile in self.TILES:
            steps = [t for t, k, tl in events if k == "color" and tl == tile]
            self.assertEqual(len(steps), 2)
            self.assertLessEqual(self._at(events, "dsf_end", tile), min(steps))
            self.assertLessEqual(max(steps), self._at(events, "ovl_start", tile))
            self.assertIn((*tile, "color"), done)
            self.assertIn((*tile, "clouds"), done)
        # With both selected the colour step fills clouds too; the cloud step
        # then only checks.
        self.assertEqual(self.fills, [True, True] * len(self.TILES))
        # Step 3 corrects inline when the colour steps follow it.
        self.assertEqual(self.inline, [(True, True)] * len(self.TILES))

    def test_without_colour_steps_step_3_does_not_correct(self):
        _, events, _ = self._run(pipelined=True)
        self.assertEqual(self.inline, [(False, False)] * len(self.TILES))
        self.assertFalse([e for e in events if e[1] == "color"])

    def test_colour_step_alone_does_not_fill(self):
        _, _, done = self._run(pipelined=True, do_color=True)
        self.assertEqual(self.fills, [False] * len(self.TILES))
        self.assertEqual(self.inline, [(True, False)] * len(self.TILES))
        self.assertNotIn((10, 20, "clouds"), done)


if __name__ == "__main__":
    unittest.main()
