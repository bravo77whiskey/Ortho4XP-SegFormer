"""Second-stage building-height regressors for SFR placement.

Runs after YOLO OBB building detection: each detection's fixed 64 m ground
window is cropped from the source texture and combined with three footprint
scalars. Checkpoint metadata selects the matching inference contract:

* metadata-free legacy checkpoints preserve Ortho4XP's deployed reconstructed
  96 px CNN and log10 scalar encoding;
* ``arch=v2`` / ``arch=v2s`` checkpoints use a 64-bin ordinal head on an
  ImageNet-normalized ConvNeXt-Tiny / ConvNeXt-Small trunk, the checkpoint's
  ``crop_px``, and normalized natural-log scalars.

Every model returns expected ``log1p(height_m)`` so the public prediction API
and downstream placement rules stay unchanged.

The legacy architecture below was reconstructed from the checkpoint state
dict and is retained for backward compatibility:
  - trunk: 8x [Conv3x3(bias=False) -> BN -> ReLU], channels
    32,32,64,64,128,128,256,256 with stride 2 on the channel-up convs;
    adaptive average pooling to 256 features.
  - head: concat [log10(area_m2+1), log10(long_side_m+1), m_per_px] ->
    Linear(259,128) -> ReLU -> optional Dropout -> Linear(128,1).
  - input pixels are RGB / 255 (no ImageNet normalization).

Known limitation: buildings much larger than the 64 m window (warehouses,
big-box) can over-predict badly. Callers apply a floor, and large-footprint
placement classes cap outlier heights before object/facade selection.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping

import numpy as np

WINDOW_M = 64.0
CROP_PX = 96
DEFAULT_HEIGHT_CHECKPOINT = r"I:\building-models\heightnet.pt"
HEIGHT_MODEL_MIN_M = 2.5
# Batch size for the tiny CNN; 512 crops is ~18 MB of input on device.
DEFAULT_HEIGHT_BATCH = 512
HEAD_DROPOUT_P = 0.20
V2_HEIGHT_BINS = 64
V2_HEIGHT_MAX_M = 400.0

LEGACY_SCALAR_LAYOUT = "legacy-log10"
V2_SCALAR_LAYOUT = "v2-normalized-ln"

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


def _build_heightnet_v2(backbone="tiny"):
    """Build the ConvNeXt ordinal model without downloading pretrained weights."""
    import torch
    import torch.nn as nn

    if backbone == "small":
        from torchvision.models import convnext_small
        body = convnext_small(weights=None).features
    elif backbone == "tiny":
        from torchvision.models import convnext_tiny
        body = convnext_tiny(weights=None).features
    else:
        raise ValueError(f"unsupported HeightNet v2 backbone: {backbone!r}")

    class HeightNetV2(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = body
            self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.head = nn.Sequential(
                nn.LayerNorm(768 + 3),
                nn.Linear(768 + 3, 256),
                nn.SiLU(),
                nn.Dropout(HEAD_DROPOUT_P),
                nn.Linear(256, V2_HEIGHT_BINS),
            )
            self.register_buffer(
                "bin_centers",
                torch.linspace(0.0, math.log1p(V2_HEIGHT_MAX_M), V2_HEIGHT_BINS),
            )
            self.register_buffer(
                "in_mean",
                torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            )
            self.register_buffer(
                "in_std",
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            )

        def logits(self, image, scalars):
            normalized = (image - self.in_mean) / self.in_std
            features = self.pool(self.body(normalized))
            return self.head(torch.cat([features, scalars], dim=1))

        def expectation(self, logits):
            probabilities = logits.float().softmax(dim=1)
            return (probabilities * self.bin_centers).sum(dim=1)

        def forward(self, image, scalars):
            return self.expectation(self.logits(image, scalars))

    return HeightNetV2()


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


def _fp16_enabled():
    """CUDA fp16 autocast for the forward pass (O4_SFR_BLD_HEIGHT_FP16=0 to
    disable). Measured drift vs fp32 is <5 mm on v2s — far below the model's
    MAE — but parity A/B runs need the exact fp32 numbers."""
    return os.environ.get("O4_SFR_BLD_HEIGHT_FP16", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )


def _checkpoint_crop_px(ckpt, arch):
    if arch == "v1-legacy":
        raw_crop_px = ckpt.get("crop_px", CROP_PX)
    else:
        if "crop_px" not in ckpt:
            raise ValueError(
                f"HeightNet checkpoint arch={arch!r} is missing required crop_px"
            )
        raw_crop_px = ckpt["crop_px"]
    if isinstance(raw_crop_px, bool):
        raise ValueError(f"HeightNet checkpoint crop_px is invalid: {raw_crop_px!r}")
    try:
        crop_px = int(raw_crop_px)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"HeightNet checkpoint crop_px is invalid: {raw_crop_px!r}"
        ) from exc
    if crop_px <= 0 or crop_px != raw_crop_px:
        raise ValueError(f"HeightNet checkpoint crop_px is invalid: {raw_crop_px!r}")
    if arch == "v1-legacy" and crop_px != CROP_PX:
        raise ValueError(
            f"Legacy HeightNet checkpoint crop_px={crop_px} != expected {CROP_PX}"
        )
    return crop_px


def _checkpoint_architecture(ckpt):
    raw_arch = ckpt.get("arch")
    if raw_arch is None or str(raw_arch).strip() == "":
        return "v1-legacy"
    arch = str(raw_arch).strip().lower()
    if arch not in {"v2", "v2s"}:
        raise ValueError(f"HeightNet checkpoint has unsupported arch={raw_arch!r}")
    return arch


def load_height_model(checkpoint_path=None, device=None):
    """Load an autodetected HeightNet onto ``device``.

    Legacy checkpoints intentionally have no ``arch`` metadata. New ordinal
    checkpoints must declare ``arch`` and ``crop_px`` so incompatible formats
    fail explicitly instead of producing plausible-looking wrong heights.
    """
    import torch

    path = checkpoint_path or default_checkpoint_path()
    # weights_only: the checkpoint is a plain tensor/state dict from an
    # external repo — never unpickle arbitrary objects from it.
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, Mapping):
        raise ValueError("HeightNet checkpoint must contain a mapping")
    try:
        window_m = float(ckpt.get("window_m", WINDOW_M))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"HeightNet checkpoint window_m is invalid: {ckpt.get('window_m')!r}"
        ) from exc
    if not math.isfinite(window_m) or abs(window_m - WINDOW_M) > 1e-6:
        raise ValueError(
            f"HeightNet checkpoint window_m={window_m} != expected {WINDOW_M}"
        )
    arch = _checkpoint_architecture(ckpt)
    crop_px = _checkpoint_crop_px(ckpt, arch)
    try:
        state_dict = ckpt["model"]
    except KeyError as exc:
        raise ValueError("HeightNet checkpoint is missing model state dict") from exc
    if not isinstance(state_dict, Mapping):
        raise ValueError("HeightNet checkpoint model must be a state dict mapping")

    if arch == "v1-legacy":
        head_dropout = _checkpoint_head_uses_dropout(state_dict)
        model = _build_heightnet(head_dropout=head_dropout)
        head_layout = "dropout" if head_dropout else "legacy"
        scalar_layout = LEGACY_SCALAR_LAYOUT
    else:
        backbone = "small" if arch == "v2s" else "tiny"
        model = _build_heightnet_v2(backbone=backbone)
        head_layout = "ordinal"
        scalar_layout = V2_SCALAR_LAYOUT
    model.load_state_dict(state_dict, strict=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    model._sfr_device = device
    model._sfr_heightnet_arch = arch
    model._sfr_heightnet_head_layout = head_layout
    model._sfr_crop_px = crop_px
    model._sfr_scalar_layout = scalar_layout
    model._sfr_window_m = window_m
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


def _encode_scalars(area_m2, long_side_m, m_per_px, layout):
    """Encode the three auxiliary features for one checkpoint family."""
    if layout == LEGACY_SCALAR_LAYOUT:
        return (
            math.log10(area_m2 + 1.0),
            math.log10(long_side_m + 1.0),
            m_per_px,
        )
    if layout == V2_SCALAR_LAYOUT:
        return (
            math.log1p(area_m2) / 8.0,
            math.log1p(long_side_m) / 5.0,
            math.log(m_per_px) / 2.0,
        )
    raise ValueError(f"unsupported HeightNet scalar layout: {layout!r}")


def predict_detection_heights(model, image, detections, m_per_px,
                              batch_size=DEFAULT_HEIGHT_BATCH):
    """Predict height_m for each detection; NaN where prediction is impossible.

    ``image`` is the RGB uint8 texture array the detections are in pixel
    coordinates of (the same array YOLO inference cropped from), ``m_per_px``
    its ground resolution. Returns a float64 array aligned with
    ``detections``. Predictions are raw model output (expm1); callers apply
    the HEIGHT_MODEL_MIN_M floor and any placement-class-specific caps.
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
    window_m = float(getattr(model, "_sfr_window_m", WINDOW_M))
    crop_px = int(getattr(model, "_sfr_crop_px", CROP_PX))
    scalar_layout = getattr(model, "_sfr_scalar_layout", LEGACY_SCALAR_LAYOUT)
    half_px = (window_m / 2.0) / m_per_px

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
        crops.append(cv2.resize(window, (crop_px, crop_px),
                                interpolation=cv2.INTER_LINEAR))
        scalars.append(_encode_scalars(
            area_m2, long_m, m_per_px, scalar_layout
        ))
        valid_idx.append(i)

    if not valid_idx:
        return out
    device = getattr(model, "_sfr_device", "cpu")
    crops_np = np.stack(crops)
    scalars_np = np.asarray(scalars, dtype=np.float32)
    preds = np.empty((len(valid_idx),), dtype=np.float64)
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("HeightNet batch_size must be positive")
    use_fp16 = _fp16_enabled() and str(device).startswith("cuda")
    with torch.inference_mode():
        for start in range(0, len(valid_idx), batch_size):
            stop = start + batch_size
            imgs = torch.from_numpy(
                crops_np[start:stop].astype(np.float32).transpose(0, 3, 1, 2)
                / 255.0
            ).to(device)
            scal = torch.from_numpy(scalars_np[start:stop]).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                pred = model(imgs, scal)
            pred = pred.float().cpu().numpy()
            # Clip the log-space output before expm1: wrong-regime inputs can
            # otherwise overflow to inf (12 -> ~163 km, far past any clamp).
            preds[start:stop] = np.expm1(np.clip(pred, -2.0, 8.0))
    out[np.asarray(valid_idx, dtype=np.int64)] = preds
    return out
