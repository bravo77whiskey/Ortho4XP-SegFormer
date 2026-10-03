"""Create a height-bin YOLO OBB checkpoint initialized from an 8-class checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch

from building_training_lib import YOLO_HEIGHT_BINS_M, YOLO_HEIGHT_LABEL_BINS, yolo_obb_class_names


CLASS_HEAD_KEYS = (
    "model.23.cv3.0.2.weight",
    "model.23.cv3.0.2.bias",
    "model.23.cv3.1.2.weight",
    "model.23.cv3.1.2.bias",
    "model.23.cv3.2.2.weight",
    "model.23.cv3.2.2.bias",
    "model.23.one2one_cv3.0.2.weight",
    "model.23.one2one_cv3.0.2.bias",
    "model.23.one2one_cv3.1.2.weight",
    "model.23.one2one_cv3.1.2.bias",
    "model.23.one2one_cv3.2.2.weight",
    "model.23.one2one_cv3.2.2.bias",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, required=True, help="8-class YOLO OBB checkpoint")
    parser.add_argument("--template", type=Path, required=True, help="160-class height-bin checkpoint template")
    parser.add_argument("--output", type=Path, required=True, help="Output checkpoint path")
    parser.add_argument("--note", type=Path, default=None, help="Optional JSON note path")
    parser.add_argument(
        "--bias-offset",
        type=float,
        default=0.0,
        help="Optional additive bias offset for expanded class logits",
    )
    return parser.parse_args()


def checkpoint_model(checkpoint: dict[str, Any]) -> torch.nn.Module:
    model = checkpoint.get("ema") or checkpoint.get("model")
    if model is None:
        raise KeyError("Checkpoint has neither 'ema' nor 'model'")
    return model.float()


def expand_class_head_tensor(source: torch.Tensor, target: torch.Tensor, bias_offset: float) -> torch.Tensor:
    if source.shape[0] != 8 or target.shape[0] != 8 * len(YOLO_HEIGHT_BINS_M):
        raise ValueError(f"Unexpected class-head shapes: source={tuple(source.shape)} target={tuple(target.shape)}")
    expanded = target.detach().clone()
    for placement_index in range(8):
        start = placement_index * len(YOLO_HEIGHT_BINS_M)
        end = start + len(YOLO_HEIGHT_BINS_M)
        expanded[start:end] = source[placement_index].detach().clone()
        if source.ndim == 1 and bias_offset:
            expanded[start:end] += float(bias_offset)
    return expanded


def main() -> int:
    args = parse_args()
    legacy_ckpt = torch.load(args.legacy, map_location="cpu", weights_only=False)
    template_ckpt = torch.load(args.template, map_location="cpu", weights_only=False)

    legacy_model = checkpoint_model(legacy_ckpt)
    target_model = checkpoint_model(template_ckpt)
    legacy_state = legacy_model.state_dict()
    target_state = target_model.state_dict()

    copied_same_shape: list[str] = []
    skipped_shape: dict[str, dict[str, tuple[int, ...]]] = {}
    for key, target_tensor in list(target_state.items()):
        source_tensor = legacy_state.get(key)
        if source_tensor is None:
            continue
        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            target_state[key] = source_tensor.detach().clone()
            copied_same_shape.append(key)
        else:
            skipped_shape[key] = {
                "legacy": tuple(source_tensor.shape),
                "target": tuple(target_tensor.shape),
            }

    expanded_heads: list[str] = []
    for key in CLASS_HEAD_KEYS:
        if key not in legacy_state or key not in target_state:
            raise KeyError(f"Missing required class-head key: {key}")
        target_state[key] = expand_class_head_tensor(legacy_state[key], target_state[key], args.bias_offset)
        expanded_heads.append(key)

    target_model.load_state_dict(target_state)
    target_model.names = yolo_obb_class_names(YOLO_HEIGHT_LABEL_BINS)
    target_model.model[-1].nc = len(target_model.names)

    output_ckpt = copy.copy(template_ckpt)
    output_ckpt["model"] = target_model
    output_ckpt["ema"] = None
    output_ckpt["optimizer"] = None
    output_ckpt["scaler"] = None
    output_ckpt["epoch"] = -1
    output_ckpt["best_fitness"] = None
    output_ckpt["train_args"] = dict(output_ckpt.get("train_args") or {})
    output_ckpt["train_args"]["height_surgery_source"] = str(args.legacy)
    output_ckpt["train_args"]["height_surgery_template"] = str(args.template)
    output_ckpt["train_args"]["height_surgery_bias_offset"] = float(args.bias_offset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_ckpt, args.output)

    note = {
        "legacy": str(args.legacy),
        "template": str(args.template),
        "output": str(args.output),
        "bias_offset": float(args.bias_offset),
        "height_bins": list(YOLO_HEIGHT_BINS_M),
        "copied_same_shape_count": len(copied_same_shape),
        "expanded_heads": expanded_heads,
        "skipped_shape": skipped_shape,
    }
    if args.note:
        args.note.parent.mkdir(parents=True, exist_ok=True)
        args.note.write_text(json.dumps(note, indent=2), encoding="utf-8")
    print(json.dumps(note, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
