"""TensorRT engines: the setting turns them off, one process builds at a time,
and anything missing leaves the caller on its torch model. No GPU needed."""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_TensorRT as TRT  # noqa: E402


class SwitchTests(unittest.TestCase):
    def test_on_unless_turned_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(TRT.ENV_ON, None)
            self.assertTrue(TRT.enabled())
        for off in ("0", "false", "off"):
            with mock.patch.dict(os.environ, {TRT.ENV_ON: off}):
                self.assertFalse(TRT.enabled())

    def test_off_never_touches_tensorrt(self):
        export = mock.Mock()
        with mock.patch.dict(os.environ, {TRT.ENV_ON: "0"}), \
                mock.patch.object(TRT, "_trt") as trt:
            self.assertIsNone(TRT.engine("m", "src", export))
        trt.assert_not_called()
        export.assert_not_called()

    def test_missing_tensorrt_keeps_torch(self):
        with mock.patch.dict(os.environ, {TRT.ENV_ON: "1"}), \
                mock.patch.object(TRT, "_trt", return_value=None):
            self.assertIsNone(TRT.engine("m", "src", mock.Mock()))

    def test_engine_dir_from_environment(self):
        with mock.patch.dict(os.environ, {TRT.ENV_DIR: "E:/x/_tensorrt"}):
            self.assertEqual(TRT.engine_dir(), "E:/x/_tensorrt")
        with mock.patch.dict(os.environ, {TRT.ENV_DIR: ""}):
            self.assertTrue(TRT.engine_dir().endswith("_tensorrt"))


class LockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lock = os.path.join(self.tmp.name, "m.engine.lock")

    def tearDown(self):
        self.tmp.cleanup()

    def test_second_builder_waits_its_turn(self):
        self.assertTrue(TRT._take_lock(self.lock))
        self.assertFalse(TRT._take_lock(self.lock))

    def test_stale_lock_is_taken_over(self):
        with open(self.lock, "w") as f:
            f.write("1")
        old = time.time() - TRT.LOCK_STALE_S - 60
        os.utime(self.lock, (old, old))
        self.assertTrue(TRT._take_lock(self.lock))


if __name__ == "__main__":
    unittest.main()
