"""GPU steps keep the VRAM reserve free: they wait, are capped and retry on OOM.

Runs against a fake torch, so no GPU is needed; the ledger goes to a temp dir.
"""

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_GPU_Budget as GPUB  # noqa: E402

MB = GPUB.MB
TOTAL = 16 * 1024 * MB


class FakeCuda:
    def __init__(self, free):
        self.free = free
        self.reserved = 0
        self.allocated = 0
        self.peak = 0
        self.peak_reserved = 0
        self.fraction = 1.0
        self.emptied = 0

    def is_available(self):
        return True

    def current_device(self):
        return 0

    def get_device_properties(self, _dev):
        return SimpleNamespace(total_memory=TOTAL)

    def mem_get_info(self):
        return self.free, TOTAL

    def memory_reserved(self):
        return self.reserved

    def memory_allocated(self):
        return self.allocated

    def max_memory_allocated(self):
        return self.peak

    def max_memory_reserved(self):
        return self.peak_reserved

    def reset_peak_memory_stats(self):
        self.peak = self.allocated
        self.peak_reserved = self.reserved

    def empty_cache(self):
        self.emptied += 1
        self.free += self.reserved - self.allocated
        self.reserved = self.allocated

    def set_per_process_memory_fraction(self, fraction):
        self.fraction = fraction

    def alloc(self, nbytes):
        """Allocate like the caching allocator: fail past the cap."""
        if self.allocated + nbytes > self.fraction * TOTAL:
            raise RuntimeError("CUDA out of memory. Tried to allocate")
        self.allocated += nbytes
        self.peak = max(self.peak, self.allocated)
        grow = max(0, self.allocated - self.reserved)
        self.reserved += grow
        self.free -= grow
        self.peak_reserved = max(self.peak_reserved, self.reserved)

    def release(self, nbytes):
        self.allocated -= nbytes


class BudgetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="gpu_budget_test_")
        self.cuda = FakeCuda(free=6 * 1024 * MB)
        fake_torch = SimpleNamespace(cuda=self.cuda)
        patches = [
            mock.patch.object(GPUB.tempfile, "gettempdir", return_value=self.tmp),
            mock.patch.dict(os.environ, {"O4_VRAM_RESERVE_MB": "1024"}),
            mock.patch.dict(GPUB._state, {"torch": fake_torch, "checked": True, "off": False,
                                          "watcher": True, "paths": None, "nvml": None}),
            mock.patch.object(GPUB, "POLL_S", 0.02),
            mock.patch.object(GPUB, "ANNOUNCE_S", 0.0),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        GPUB._active.clear()
        GPUB._local_taint.clear()

    def _ledger(self):
        with GPUB._ledger() as led:
            return json.loads(json.dumps(led))

    def _book_other(self, nbytes):
        """A booking held by another live process (this test's parent)."""
        with GPUB._ledger() as led:
            led["steps"]["other"] = {"pid": os.getppid(), "label": "other",
                                     "bytes": int(nbytes), "t": time.time()}

    def _drop_other(self):
        with GPUB._ledger() as led:
            led["steps"].pop("other", None)

    def test_step_waits_until_the_reserve_is_safe(self):
        # 6 GB free, 1 GB reserve, 4.5 GB booked elsewhere: a 1 GB step must wait
        self._book_other(4608 * MB)
        threading.Timer(0.3, self._drop_other).start()
        t0 = time.monotonic()
        with GPUB.step("probe", need_mb=1024):
            waited = time.monotonic() - t0
            self.assertEqual(len(self._ledger()["steps"]), 1)
        self.assertGreaterEqual(waited, 0.25)
        self.assertEqual(self._ledger()["steps"], {})

    def test_older_waiter_goes_first(self):
        # 5 GB above the reserve, 4.5 GB of it wanted by a step already waiting
        # elsewhere: a new 1 GB step must not jump the queue
        with GPUB._ledger() as led:
            led["waiting"][f"{os.getppid()}:1"] = {"t": time.time() - 5, "seen": time.time(),
                                                   "want": 4608 * MB}

        def leave():
            with GPUB._ledger() as led:
                led["waiting"].clear()

        threading.Timer(0.3, leave).start()
        t0 = time.monotonic()
        with GPUB.step("probe", need_mb=1024):
            self.assertGreaterEqual(time.monotonic() - t0, 0.25)

    def test_runs_within_what_is_left_when_nothing_can_free_up(self):
        # 2 GB above the reserve, nothing else running or caching: a 3 GB step
        # stops waiting after the give-back time and is capped at what is left
        self.cuda.free = 3 * 1024 * MB
        with mock.patch.object(GPUB, "GIVE_BACK_S", 0.2):
            t0 = time.monotonic()
            with GPUB.step("probe", need_mb=3072):
                self.assertLess(time.monotonic() - t0, 2.0)
                self.assertLessEqual(self.cuda.fraction * TOTAL, 2048 * MB + 1)

    def test_step_is_capped_at_its_booking(self):
        with GPUB.step("probe", need_mb=512):
            cap = self.cuda.fraction * TOTAL
            self.assertLess(cap, 1024 * MB)
            with self.assertRaises(RuntimeError):
                self.cuda.alloc(1024 * MB)
        self.assertEqual(self.cuda.fraction, 1.0)   # uncapped between steps

    def test_oom_is_retried_with_a_bigger_booking_and_learned(self):
        tries = []

        def body():
            tries.append(self.cuda.fraction * TOTAL)
            self.cuda.alloc(900 * MB)
            self.cuda.release(900 * MB)

        with mock.patch.dict(GPUB.DEFAULT_NEED_MB, {"probe": 512}):
            GPUB.call("probe", body)
        self.assertEqual(len(tries), 2)
        self.assertGreater(tries[1], tries[0])
        self.assertGreaterEqual(self._ledger()["need"]["probe"], 768 * MB)

    def test_measured_need_replaces_the_default(self):
        def body():
            self.cuda.alloc(300 * MB)
            self.cuda.release(300 * MB)

        GPUB.call("probe", body)
        self.assertEqual(self._ledger()["need"]["probe"], 300 * MB)

    def test_dead_process_bookings_are_dropped(self):
        with GPUB._ledger() as led:
            led["steps"]["ghost"] = {"pid": 2 ** 30, "label": "x", "bytes": 10 ** 12,
                                     "t": time.time()}
        with GPUB.step("probe", need_mb=256):
            pass
        self.assertNotIn("ghost", self._ledger()["steps"])

    def test_nested_steps_book_once(self):
        with GPUB.step("outer", need_mb=256):
            with GPUB.step("inner", need_mb=256):
                self.assertEqual(len(self._ledger()["steps"]), 1)

    def test_reserve_zero_turns_it_off(self):
        GPUB._state["checked"] = False
        GPUB._state["torch"] = None
        with mock.patch.dict(os.environ, {"O4_VRAM_RESERVE_MB": "0"}):
            self.assertEqual(GPUB.call("probe", lambda: 42), 42)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "o4xp_gpu")))


if __name__ == "__main__":
    unittest.main()
