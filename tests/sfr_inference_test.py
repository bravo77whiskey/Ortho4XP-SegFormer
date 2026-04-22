import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import torch
except ImportError:  # pragma: no cover - exercised only in minimal envs
    torch = None

import O4_SFR_Inference as INF


@unittest.skipIf(torch is None, "torch is not installed")
class SfrInferenceTests(unittest.TestCase):
    def test_run_inference_returns_expected_class_map(self):
        class Config:
            num_labels = 3

        class Output:
            def __init__(self, logits):
                self.logits = logits

        class FakeModel(torch.nn.Module):
            config = Config()

            def forward(self, pixel_values):
                batch_size, _channels, height, width = pixel_values.shape
                logits = torch.zeros(
                    (batch_size, 3, max(1, height // 4), max(1, width // 4)),
                    dtype=pixel_values.dtype,
                )
                logits[:, 1] = 3.0
                return Output(logits)

        old_patch_size = INF.segformer_patch_size
        old_overlap = INF.segformer_overlap
        old_threshold = INF.segformer_confidence_threshold
        try:
            INF.segformer_patch_size = 32
            INF.segformer_overlap = 8
            INF.segformer_confidence_threshold = 0.1
            image = np.zeros((40, 48, 3), dtype=np.uint8)

            result = INF.run_inference(FakeModel(), torch.device("cpu"), image)

            self.assertEqual(result.shape, (40, 48))
            self.assertEqual(result.dtype, np.int8)
            self.assertTrue((result[2:-2, 2:-2] == 1).all())
        finally:
            INF.segformer_patch_size = old_patch_size
            INF.segformer_overlap = old_overlap
            INF.segformer_confidence_threshold = old_threshold


if __name__ == "__main__":
    unittest.main()
