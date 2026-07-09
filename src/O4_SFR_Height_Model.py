"""Second-stage building-height regressor (HeightNet) for SFR placement.

Runs after YOLO OBB building detection: each detection's fixed 64 m ground
window is cropped from the source texture, resized to 96x96 and pushed with
three footprint scalars through a small CNN that outputs log1p(height_m).
Weights come from the user's external open-buildings-training repo
(``I:\\building-models\\heightnet.pt``; updated 2026-07-09, val MAE 2.40 m
over 2.89 M buildings).

The architecture below was reconstructed from the checkpoint state dict and
validated against SimHeaven ground-truth heights (MAE 2.58 m, matching the
reported val MAE) — see tmp/heightnet_tests/heightnet_arch_probe*.py for the
probes that pinned the wiring:
  - trunk: 8x [Conv3x3(bias=False) -> BN -> ReLU], channels
    32,32,64,64,128,128,256,256 with stride 2 on the channel-up convs;
    adaptive average pooling to 256 features.
  - head: concat [log10(area_m2+1), log10(long_side_m+1), m_per_px] ->
    Linear(259,128) -> ReLU -> optional Dropout -> Linear(128,1).
  - input pixels are RGB / 255 (no ImageNet normalization).

Known limitation: buildings much larger than the 64 m window (warehouses,
big-box) can over-predict badly (roof-only crops leave the footprint scalars
unchecked).  Callers apply a floor only; over-predictions are harmless to
height-fit selection, which just picks the tallest asset in the class pool.
"""

from __future__ import annotations

import math
import os

import numpy as np

WINDOW_M = 64.0
CROP_PX = 96
DEFAULT_HEIGHT_CHECKPOINT = r"H:\model_training\models\heightnet.pt"
HEIGHT_MODEL_MIN_M = 2.5
# Batch size for the tiny CNN; 512 crops is ~18 MB of input on device.
DEFAULT_HEIGHT_BATCH = 512
HEAD_DROPOUT_P = 0.20

_BLOCK_CHANNELS = ((3, 32), (32, 32), (32, 64), (64, 64),
                   (64, 128), (128, 128), (128, 256), (256, 256))
# Downsample where the channel count doubles (blocks 2, 4, 6).
_STRIDE_BLOCKS = frozenset({2, 4, 6})


def _build_heightnet(head_dropout=False):
    import torch.nn as nn

    class HeightNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = []
            for i, (cin, cout) in enumerate(_BLOCK_CHANNELS):
                layers += [
                    nn.Conv2d(cin, cout, 3, 2 if i in _STRIDE_BLOCKS else 1, 1,
                              bias=False),
                    nn.BatchNorm2d(cout),
                    nn.ReLU(inplace=True),
                ]
            self.body = nn.Sequential(*layers)
            head_layers = [nn.Linear(256 + 3, 128), nn.ReLU(inplace=True)]
            if head_dropout:
                # Dropout has no parameters and is disabled in eval mode, but
                # it shifts the final Linear layer to head.3 in new checkpoints.
                head_layers.append(nn.Dropout(p=HEAD_DROPOUT_P))
            head_layers.append(nn.Linear(128, 1))
            self.head = nn.Sequential(*head_layers)

        def forward(self, image, scalars):
            import torch.nn.functional as F
            x = self.body(image)
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
            import torch
            return self.head(torch.cat([x, scalars], dim=1)).squeeze(1)

    return HeightNet()


def _checkpoint_head_uses_dropout(state_dict):
    has_legacy_head = "head.2.weight" in state_dict or "head.2.bias" in state_dict
    has_dropout_head = "head.3.weight" in state_dict or "head.3.bias" in state_dict
    if has_legacy_head and has_dropout_head:
        raise ValueError(
            "HeightNet checkpoint has both legacy head.2 and dropout head.3 keys"
        )
    if has_dropout_head:
        return True
    if has_legacy_head:
        return False
    raise ValueError(
        "HeightNet checkpoint does not contain recognizable head.2 or head.3 keys"
    )


def default_checkpoint_path():
    return os.environ.get(
        "O4_SFR_BLD_HEIGHT_CHECKPOINT", DEFAULT_HEIGHT_CHECKPOINT
    )


def load_height_model(checkpoint_path=None, device=None):
    """Load HeightNet onto ``device``; raises on a bad/missing checkpoint."""
    import torch

    path = checkpoint_path or default_checkpoint_path()
    # weights_only: the checkpoint is a plain tensor/state dict from an
    # external repo — never unpickle arbitrary objects from it.
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    window_m = float(ckpt.get("window_m", WINDOW_M))
    if abs(window_m - WINDOW_M) > 1e-6:
        raise ValueError(
            f"HeightNet checkpoint window_m={window_m} != expected {WINDOW_M}"
        )
    state_dict = ckpt["model"]
    head_dropout = _checkpoint_head_uses_dropout(state_dict)
    model = _build_heightnet(head_dropout=head_dropout)
    model.load_state_dict(state_dict, strict=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    model._sfr_device = device
    model._sfr_heightnet_head_layout = "dropout" if head_dropout else "legacy"
    return model


def _detection_scalars(detection, m_per_px):
    """Return (cx, cy, area_m2, long_side_m) or None when incomplete."""
    try:
        center = detection.get("center")
        cx, cy = float(center[0]), float(center[1])
        area_m2 = float(detection.get("area_m2") or 0.0)
        long_m = float(detection.get("length_m") or 0.0)
    except (TypeError, ValueError, IndexError):
        return None
    if not (math.isfinite(cx) and math.isfinite(cy)):
        return None
    if area_m2 <= 0.0 or long_m <= 0.0:
        return None
    return cx, cy, area_m2, long_m


def predict_detection_heights(model, image, detections, m_per_px,
                              batch_size=DEFAULT_HEIGHT_BATCH):
    """Predict height_m for each detection; NaN where prediction is impossible.

    ``image`` is the RGB uint8 texture array the detections are in pixel
    coordinates of (the same array YOLO inference cropped from), ``m_per_px``
    its ground resolution. Returns a float64 array aligned with
    ``detections``. Predictions are raw model output (expm1) — callers
    apply the HEIGHT_MODEL_MIN_M floor.
    """
    import cv2
    import torch

    n = len(detections)
    out = np.full((n,), np.nan, dtype=np.float64)
    if n == 0 or model is None or image is None:
        return out
    img_h, img_w = image.shape[:2]
    m_per_px = float(m_per_px)
    if not math.isfinite(m_per_px) or m_per_px <= 0.0:
        return out
    half_px = (WINDOW_M / 2.0) / m_per_px

    valid_idx = []
    crops = []
    scalars = []
    for i, det in enumerate(detections):
        parsed = _detection_scalars(det, m_per_px)
        if parsed is None:
            continue
        cx, cy, area_m2, long_m = parsed
        x0 = int(round(cx - half_px))
        x1 = int(round(cx + half_px))
        y0 = int(round(cy - half_px))
        y1 = int(round(cy + half_px))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(img_w, x1), min(img_h, y1)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        window = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
        window[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = image[sy0:sy1, sx0:sx1]
        crops.append(cv2.resize(window, (CROP_PX, CROP_PX),
                                interpolation=cv2.INTER_LINEAR))
        scalars.append((
            math.log10(area_m2 + 1.0),
            math.log10(long_m + 1.0),
            m_per_px,
        ))
        valid_idx.append(i)

    if not valid_idx:
        return out
    device = getattr(model, "_sfr_device", "cpu")
    crops_np = np.stack(crops)
    scalars_np = np.asarray(scalars, dtype=np.float32)
    preds = np.empty((len(valid_idx),), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(valid_idx), int(batch_size)):
            stop = start + int(batch_size)
            imgs = torch.from_numpy(
                crops_np[start:stop].astype(np.float32).transpose(0, 3, 1, 2)
                / 255.0
            ).to(device)
            scal = torch.from_numpy(scalars_np[start:stop]).to(device)
            pred = model(imgs, scal).float().cpu().numpy()
            # Clip the log-space output before expm1: wrong-regime inputs can
            # otherwise overflow to inf (12 -> ~163 km, far past any clamp).
            preds[start:stop] = np.expm1(np.clip(pred, -2.0, 8.0))
    out[np.asarray(valid_idx, dtype=np.int64)] = preds
    return out
