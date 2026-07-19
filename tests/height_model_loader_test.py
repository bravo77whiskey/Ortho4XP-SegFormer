import math
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

import O4_SFR_Height_Model as HEIGHT


@unittest.skipIf(torch is None, "torch is not installed")
class HeightModelLoaderTests(unittest.TestCase):
    def _write_checkpoint(self, path: Path, *, head_dropout: bool) -> None:
        model = HEIGHT._build_heightnet(head_dropout=head_dropout)
        torch.save(
            {"model": model.state_dict(), "window_m": HEIGHT.WINDOW_M},
            path,
        )

    def test_load_height_model_accepts_legacy_head_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "legacy-heightnet.pt"
            self._write_checkpoint(checkpoint, head_dropout=False)

            model = HEIGHT.load_height_model(str(checkpoint), device="cpu")

            self.assertEqual(model._sfr_heightnet_head_layout, "legacy")
            self.assertEqual(model._sfr_heightnet_arch, "v1-legacy")
            self.assertEqual(model._sfr_crop_px, HEIGHT.CROP_PX)
            self.assertEqual(model._sfr_scalar_layout, "legacy-log10")
            self.assertIsInstance(model.head[2], torch.nn.Linear)

    def test_load_height_model_accepts_dropout_head_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "dropout-heightnet.pt"
            self._write_checkpoint(checkpoint, head_dropout=True)

            model = HEIGHT.load_height_model(str(checkpoint), device="cpu")

            self.assertEqual(model._sfr_heightnet_head_layout, "dropout")
            self.assertEqual(model._sfr_heightnet_arch, "v1-legacy")
            self.assertEqual(model._sfr_crop_px, HEIGHT.CROP_PX)
            self.assertEqual(model._sfr_scalar_layout, "legacy-log10")
            self.assertIsInstance(model.head[2], torch.nn.Dropout)
            self.assertIsInstance(model.head[3], torch.nn.Linear)

    def test_load_height_model_autodetects_v2s_metadata(self):
        class TinyV2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))

            def forward(self, image, scalars):
                return image.new_zeros((len(image),))

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "heightnet-v2s.pt"
            torch.save(
                {
                    "model": TinyV2().state_dict(),
                    "window_m": HEIGHT.WINDOW_M,
                    "arch": "v2s",
                    "crop_px": 128,
                },
                checkpoint,
            )
            with mock.patch.object(
                HEIGHT, "_build_heightnet_v2", return_value=TinyV2()
            ) as build:
                model = HEIGHT.load_height_model(str(checkpoint), device="cpu")

        build.assert_called_once_with(backbone="small")
        self.assertEqual(model._sfr_heightnet_arch, "v2s")
        self.assertEqual(model._sfr_crop_px, 128)
        self.assertEqual(model._sfr_scalar_layout, "v2-normalized-ln")
        self.assertEqual(model._sfr_heightnet_head_layout, "ordinal")

    def test_load_height_model_maps_v2_to_tiny_backbone(self):
        class TinyV2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))

            def forward(self, image, scalars):
                return image.new_zeros((len(image),))

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "heightnet-v2.pt"
            torch.save(
                {
                    "model": TinyV2().state_dict(),
                    "window_m": HEIGHT.WINDOW_M,
                    "arch": "v2",
                    "crop_px": 112,
                },
                checkpoint,
            )
            with mock.patch.object(
                HEIGHT, "_build_heightnet_v2", return_value=TinyV2()
            ) as build:
                model = HEIGHT.load_height_model(str(checkpoint), device="cpu")

        build.assert_called_once_with(backbone="tiny")
        self.assertEqual(model._sfr_heightnet_arch, "v2")
        self.assertEqual(model._sfr_crop_px, 112)

    def test_load_height_model_rejects_unknown_architecture(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "heightnet-unknown.pt"
            torch.save(
                {
                    "model": {},
                    "window_m": HEIGHT.WINDOW_M,
                    "arch": "future-net",
                    "crop_px": 128,
                },
                checkpoint,
            )

            with self.assertRaisesRegex(ValueError, "unsupported arch"):
                HEIGHT.load_height_model(str(checkpoint), device="cpu")

    def test_load_height_model_rejects_invalid_crop_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "heightnet-bad-crop.pt"
            torch.save(
                {
                    "model": {},
                    "window_m": HEIGHT.WINDOW_M,
                    "arch": "v2s",
                    "crop_px": 0,
                },
                checkpoint,
            )

            with self.assertRaisesRegex(ValueError, "crop_px"):
                HEIGHT.load_height_model(str(checkpoint), device="cpu")

    def test_scalar_layouts_match_declared_inference_contracts(self):
        legacy = HEIGHT._encode_scalars(
            area_m2=99.0,
            long_side_m=9.0,
            m_per_px=0.5,
            layout="legacy-log10",
        )
        modern = HEIGHT._encode_scalars(
            area_m2=99.0,
            long_side_m=9.0,
            m_per_px=0.5,
            layout="v2-normalized-ln",
        )

        np.testing.assert_allclose(legacy, (2.0, 1.0, 0.5), rtol=0, atol=1e-7)
        np.testing.assert_allclose(
            modern,
            (math.log1p(99.0) / 8.0, math.log1p(9.0) / 5.0,
             math.log(0.5) / 2.0),
            rtol=0,
            atol=1e-7,
        )

    def test_prediction_uses_model_crop_and_scalar_metadata(self):
        class RecordingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.shapes = []
                self.scalars = []
                self._sfr_device = "cpu"
                self._sfr_crop_px = 128
                self._sfr_scalar_layout = "v2-normalized-ln"

            def forward(self, image, scalars):
                self.shapes.append(tuple(image.shape))
                self.scalars.append(scalars.detach().cpu().numpy())
                return image.new_full((len(image),), math.log1p(12.5))

        model = RecordingModel()
        image = np.full((80, 80, 3), 127, dtype=np.uint8)
        detections = [{
            "center": (2.0, 3.0),
            "area_m2": 99.0,
            "length_m": 9.0,
        }]

        heights = HEIGHT.predict_detection_heights(
            model, image, detections, m_per_px=0.5
        )

        np.testing.assert_allclose(heights, [12.5], rtol=0, atol=1e-5)
        self.assertEqual(model.shapes, [(1, 3, 128, 128)])
        np.testing.assert_allclose(
            model.scalars[0][0],
            (math.log1p(99.0) / 8.0, math.log1p(9.0) / 5.0,
             math.log(0.5) / 2.0),
            rtol=0,
            atol=1e-7,
        )


if __name__ == "__main__":
    unittest.main()
