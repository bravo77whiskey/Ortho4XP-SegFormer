"""A texture that fails is left without buildings; the building overlay carries on."""

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_Building_Overlay as B  # noqa: E402
import O4_Progress as PROG  # noqa: E402


def _run(n_files, fail):
    """A building run over ``n_files`` textures whose stages are stubs.

    ``fail``: indices (1-based) whose placement raises after placing one object.
    """
    run = B._BuildingOverlayRun.__new__(B._BuildingOverlayRun)
    run.files = [f"{1000 + i}_2000_BI16.dds" for i in range(n_files)]
    run._par = None
    run.timings = {}
    run.verbose_log = False
    run.detail_timing = False
    run.slow_timing_s = 0
    run.yolo_cuda_cleanup_every = 0
    run.disable_cache = False
    run.keep_seg_for_veg = False
    run.cache_dir = ""
    run.placed_stock_objects = []
    run.placed_facades = []
    run.placed_draped = []
    run._seam_pieces = []
    run._review_pending = []
    run._review_placements = {}
    run._dedup_fallback = None
    run.direct_yolo_facade_placements = 0

    def prepare(fi, fname, m):
        return {"file_timings": {}, "file_counts": {}, "local_timings": {}}

    def place(fi, fname, m, prep, file_timings, file_counts):
        run.placed_stock_objects.append(("obj", fi))
        if fi in fail:
            raise RuntimeError(f"CUDA out of memory on {fi}")

    run._prepare_dds_inference = prepare
    run._place_texture = place
    return run


class TextureSkipTest(unittest.TestCase):
    def setUp(self):
        self.issues = []
        patcher = mock.patch.object(
            B.PROG, "report_issue", lambda item, stage, msg: self.issues.append((item, stage, msg))
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_failed_texture_is_skipped_and_its_partial_placements_dropped(self):
        run = _run(4, fail={2})
        run._process_textures()
        self.assertEqual(run.placed_stock_objects, [("obj", 1), ("obj", 3), ("obj", 4)])
        self.assertEqual(len(self.issues), 1)
        item, stage, message = self.issues[0]
        self.assertEqual(item, run.files[1])
        self.assertEqual(stage, "building overlay")
        self.assertIn("CUDA out of memory on 2", message)

    def test_a_run_of_failures_still_fails_the_tile(self):
        run = _run(8, fail=set(range(2, 9)))
        with self.assertRaises(RuntimeError) as caught:
            run._process_textures()
        self.assertIn(f"{B._MAX_TEXTURE_FAILURES_IN_A_ROW} textures in a row failed", str(caught.exception))
        self.assertEqual(len(self.issues), B._MAX_TEXTURE_FAILURES_IN_A_ROW - 1)

    def test_scattered_failures_never_add_up_to_a_run(self):
        run = _run(12, fail={2, 3, 4, 5, 7, 8, 9, 10})
        run._process_textures()
        self.assertEqual([fi for _o, fi in run.placed_stock_objects], [1, 6, 11, 12])
        self.assertEqual(len(self.issues), 8)

    def test_progress_counts_every_texture_once(self):
        events = []
        with mock.patch.object(B.PROG, "emit", lambda **ev: events.append(ev)):
            _run(3, fail={2})._process_textures()
        self.assertEqual(events[0], {"ev": "items", "done": 0, "total": 3})
        self.assertEqual(events[1:], [{"ev": "items", "add": 1}] * 3)

    def test_a_parallel_helper_stores_an_empty_result_for_a_failed_texture(self):
        run = _run(3, fail={2})
        stored = {}

        class Par:
            helper = True

            def claim(self, fi):
                return True

            def store(self, _run, fi, result):
                stored[fi] = result

        run._par = Par()
        run._process_textures()
        self.assertEqual(sorted(stored), [1, 2, 3])
        self.assertEqual(stored[2]["objects"], [])
        self.assertEqual(stored[1]["objects"], [("obj", 1)])


if __name__ == "__main__":
    unittest.main()
