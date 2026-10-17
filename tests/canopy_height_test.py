import io
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import torch
except ImportError:  # pragma: no cover - exercised only in minimal envs
    torch = None

import O4_Canopy_Height as CANOPY
import O4_Vegetation_Overlay as VEG


class CanopySwitchTests(unittest.TestCase):
    def test_canopy_on_by_default_and_in_feature_sig(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("O4_VEG_CANOPY", None)
            self.assertTrue(VEG._canopy_enabled())
            on = VEG._veg_feature_sig()
            os.environ["O4_VEG_CANOPY"] = "0"
            self.assertFalse(VEG._canopy_enabled())
            self.assertNotEqual(on, VEG._veg_feature_sig())
            os.environ.pop("O4_VEG_CANOPY", None)

    def test_narrow_rows_keep_tree_lines_and_drop_blobs(self):
        mpp = 0.2
        mask = np.zeros((1000, 1000), np.uint8)
        mask[100:900, 100:120] = 1      # 160 m x 4 m roadside row
        mask[300:700, 400:800] = 1      # 80 m x 80 m blob on grass
        rows = VEG._narrow_rows(mask, mpp)
        self.assertTrue(rows[500, 110])
        self.assertFalse(rows[500, 600])
        self.assertLess(rows[300:700, 400:800].mean(), 0.05)

    def test_only_textures_above_zl17_are_shrunk(self):
        self.assertEqual(VEG._veg_infer_factor(16), 1)
        self.assertEqual(VEG._veg_infer_factor(17), 1)
        self.assertEqual(VEG._veg_infer_factor(18), 2)
        self.assertEqual(VEG._veg_infer_factor(19), 4)


class CanopyHelperTests(unittest.TestCase):
    def test_tree_mask_upsamples_and_thresholds(self):
        heights = np.zeros((4, 4), np.float16)
        heights[:2, :2] = 10.0
        mask = CANOPY.tree_mask(heights, (16, 16), 2.0)
        self.assertEqual(mask.shape, (16, 16))
        self.assertTrue(mask[0, 0])
        self.assertFalse(mask[15, 15])

    def test_model_dir_is_shared_beside_the_tile_caches(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("O4_CANOPY_MODEL_DIR", None)
            tile = os.path.join("root", "cache", "+50+010", "+52+013")
            self.assertEqual(
                CANOPY.default_model_dir(tile),
                os.path.join(os.path.abspath(os.path.join("root", "cache")), "_canopy_height"),
            )
            os.environ["O4_CANOPY_MODEL_DIR"] = "X:/models"
            self.assertEqual(CANOPY.default_model_dir(tile), "X:/models")

    def test_norm_unpickler_stubs_lightning_and_refuses_others(self):
        class Evil:
            def __reduce__(self):
                return (os.system, ("echo pwned",))

        data = pickle.dumps(Evil())
        with self.assertRaises(pickle.UnpicklingError):
            CANOPY._NormUnpickler(io.BytesIO(data)).load()
        stub_payload = b"cpytorch_lightning.utilities\nAttributeDict\n."
        self.assertIs(CANOPY._NormUnpickler(io.BytesIO(stub_payload)).load(), CANOPY._Stub)


@unittest.skipIf(torch is None, "torch is not installed")
class CanopyInferenceTests(unittest.TestCase):
    def test_heights_run_at_zl17_scale_with_aerial_normalisation(self):
        seen = {}

        class FakeNet(torch.nn.Module):
            def forward(self, x):
                seen["shape"] = tuple(x.shape)
                # 1.5 -> 15 m after the network's x10 scaling.
                return torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), 1.5)

        class FakeNorm(torch.nn.Module):
            def forward(self, x):
                q = torch.tensor([0.1, 0.1, 0.1, 0.6, 0.6, 0.6])
                return q.expand(x.shape[0], 6)

        model = CANOPY.CanopyModel(torch, torch.device("cpu"), FakeNet(), FakeNorm())
        img = np.random.default_rng(0).integers(0, 255, (1024, 1024, 3), dtype=np.uint8)
        with mock.patch.object(CANOPY.GPUB, "call", lambda label, fn, *a, **k: fn(*a)):
            heights = model.heights(img, 19)
        side = 1024 // 2 ** (19 - CANOPY.INFER_ZL)
        self.assertEqual(heights.shape, (side, side))
        self.assertEqual(seen["shape"][1:], (3, CANOPY.PATCH, CANOPY.PATCH))
        self.assertTrue(np.allclose(heights.astype(np.float32), 15.0, atol=0.1))

    def test_load_returns_none_without_cuda(self):
        logs = []
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with tempfile.TemporaryDirectory() as d:
                self.assertIsNone(CANOPY.CanopyModel.load(d, log=logs.append))
        self.assertTrue(any("no CUDA" in line for line in logs))


if __name__ == "__main__":
    unittest.main()
