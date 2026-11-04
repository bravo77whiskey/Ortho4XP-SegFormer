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


class BookingLabelTests(unittest.TestCase):
    """Engines book VRAM under their own label, so the VRAM guard measures
    them apart from the torch models they replace."""

    def _label(self, module, fn):
        with mock.patch.object(module.GPUB, "call", return_value="ok") as call:
            fn()
        return call.call_args[0][0]

    def test_segformer_engine_has_its_own_label(self):
        import O4_Landcover_Inference as INF
        cuda = mock.Mock(type="cuda")
        img = mock.Mock()
        trt_model = INF._TensorRTSegformer(mock.Mock(), mock.Mock())
        self.assertEqual(
            self._label(INF, lambda: INF.run_inference(trt_model, cuda, img)), "segformer trt")
        self.assertEqual(
            self._label(INF, lambda: INF.run_inference(mock.Mock(), cuda, img)), "segformer")

    def test_heightnet_engine_has_its_own_label(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        import numpy as np
        import O4_Height_Model as HEIGHT

        def predict(model):
            HEIGHT.predict_detection_heights(model, np.zeros((8, 8, 3), np.uint8),
                                             [mock.Mock()], 0.5)

        model = mock.Mock(_o4_device="cuda", _o4_tensorrt=mock.Mock(),
                          _o4_window_m=HEIGHT.WINDOW_M, _o4_crop_px=HEIGHT.CROP_PX,
                          _o4_scalar_layout=HEIGHT.LEGACY_SCALAR_LAYOUT)
        with mock.patch.object(HEIGHT, "_fp16_enabled", return_value=True):
            self.assertEqual(self._label(HEIGHT, lambda: predict(model)), "heightnet trt")
        with mock.patch.object(HEIGHT, "_fp16_enabled", return_value=False):
            self.assertEqual(self._label(HEIGHT, lambda: predict(model)), "heightnet")
        model._o4_tensorrt = None
        self.assertEqual(self._label(HEIGHT, lambda: predict(model)), "heightnet")


if __name__ == "__main__":
    unittest.main()
