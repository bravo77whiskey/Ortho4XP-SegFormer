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

    def test_run_inference_batch_size_preserves_class_map(self):
        class Config:
            num_labels = 3

        class Output:
            def __init__(self, logits):
                self.logits = logits

        class FakeModel(torch.nn.Module):
            config = Config()

            def forward(self, pixel_values):
                # Depend only on per-pixel channel values so batch grouping cannot
                # change the resulting argmax map.
                batch_size, _channels, height, width = pixel_values.shape
                logits = torch.zeros(
                    (batch_size, 3, max(1, height // 4), max(1, width // 4)),
                    dtype=pixel_values.dtype,
                )
                mean_val = pixel_values[:, 0].mean(dim=(1, 2)).view(batch_size, 1, 1)
                logits[:, 1] = mean_val
                logits[:, 2] = 1.0 - logits[:, 1]
                return Output(logits)

        old_patch_size = INF.segformer_patch_size
        old_overlap = INF.segformer_overlap
        old_threshold = INF.segformer_confidence_threshold
        old_batch_size = INF.segformer_batch_size
        try:
            INF.segformer_patch_size = 32
            INF.segformer_overlap = 8
            INF.segformer_confidence_threshold = 0.1
            image = np.zeros((72, 80, 3), dtype=np.uint8)
            image[:, 40:, 0] = 255

            INF.segformer_batch_size = 1
            legacy = INF.run_inference(FakeModel(), torch.device("cpu"), image)
            INF.segformer_batch_size = 3
            batched = INF.run_inference(FakeModel(), torch.device("cpu"), image)

            self.assertTrue(np.array_equal(legacy, batched))
        finally:
            INF.segformer_patch_size = old_patch_size
            INF.segformer_overlap = old_overlap
            INF.segformer_confidence_threshold = old_threshold
            INF.segformer_batch_size = old_batch_size


if __name__ == "__main__":
    unittest.main()
