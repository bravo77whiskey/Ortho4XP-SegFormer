import sys
import tempfile
import unittest
from pathlib import Path


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
            self.assertIsInstance(model.head[2], torch.nn.Linear)

    def test_load_height_model_accepts_dropout_head_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "dropout-heightnet.pt"
            self._write_checkpoint(checkpoint, head_dropout=True)

            model = HEIGHT.load_height_model(str(checkpoint), device="cpu")

            self.assertEqual(model._sfr_heightnet_head_layout, "dropout")
            self.assertIsInstance(model.head[2], torch.nn.Dropout)
            self.assertIsInstance(model.head[3], torch.nn.Linear)


if __name__ == "__main__":
    unittest.main()
