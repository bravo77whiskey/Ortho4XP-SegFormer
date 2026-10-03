"""Run a debug placement-model inference pass on one image."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from building_training_lib import NUM_PLACEMENT_CLASSES, PlacementModel, _image_to_float_tensor


def main() -> int:
    parser = argparse.ArgumentParser(description="Render placement model debug overlays.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chip-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--center-threshold", type=float, default=0.55)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--max-centers", type=int, default=2000)
    args = parser.parse_args()

    import torch

    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = output_dir / "checkpoint_snapshot.pt"
    shutil.copy2(checkpoint_path, snapshot_path)

    checkpoint = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    model = PlacementModel(
        num_classes=NUM_PLACEMENT_CLASSES,
        encoder_name=(checkpoint.get("config") or {}).get("encoder_name"),
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with Image.open(image_path) as image_handle:
        image = image_handle.convert("RGB")

    width, height = image.size
    center = np.zeros((height, width), dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.float32)

    with torch.inference_mode():
        for top in range(0, height, args.chip_size):
            for left in range(0, width, args.chip_size):
                right = min(width, left + args.chip_size)
                bottom = min(height, top + args.chip_size)
                chip = image.crop((left, top, right, bottom))
                if chip.size != (args.chip_size, args.chip_size):
                    padded = Image.new("RGB", (args.chip_size, args.chip_size))
                    padded.paste(chip, (0, 0))
                    chip = padded
                tensor = _image_to_float_tensor(chip).unsqueeze(0).to(device)
                outputs = model(tensor)
                center_prob = torch.sigmoid(outputs["center"])[0, 0].cpu().numpy()
                mask_prob = torch.sigmoid(outputs["mask"])[0, 0].cpu().numpy()
                tile_h = bottom - top
                tile_w = right - left
                center[top:bottom, left:right] += center_prob[:tile_h, :tile_w]
                mask[top:bottom, left:right] += mask_prob[:tile_h, :tile_w]
                coverage[top:bottom, left:right] += 1.0

    coverage = np.maximum(coverage, 1.0)
    center /= coverage
    mask /= coverage

    center_heat = _heatmap(center)
    mask_heat = _heatmap(mask)
    center_heat.save(output_dir / "center_heat.png")
    mask_heat.save(output_dir / "mask_heat.png")

    mask_overlay = _overlay_mask(image, mask, args.mask_threshold)
    center_overlay = _overlay_centers(
        image,
        center,
        threshold=args.center_threshold,
        max_centers=args.max_centers,
    )
    combined = _overlay_mask(image, mask, args.mask_threshold)
    combined = _overlay_centers(
        combined,
        center,
        threshold=args.center_threshold,
        max_centers=args.max_centers,
    )
    mask_overlay.save(output_dir / "mask_overlay.png")
    center_overlay.save(output_dir / "center_overlay.png")
    combined.save(output_dir / "combined_overlay.png")

    metadata = {
        "image": str(image_path),
        "checkpoint": str(snapshot_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_total_steps": checkpoint.get("total_steps"),
        "image_size": [width, height],
        "center": _array_stats(center, args.center_threshold),
        "mask": _array_stats(mask, args.mask_threshold),
        "outputs": {
            "center_heat": str(output_dir / "center_heat.png"),
            "mask_heat": str(output_dir / "mask_heat.png"),
            "mask_overlay": str(output_dir / "mask_overlay.png"),
            "center_overlay": str(output_dir / "center_overlay.png"),
            "combined_overlay": str(output_dir / "combined_overlay.png"),
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def _array_stats(values: np.ndarray, threshold: float) -> dict[str, float]:
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "threshold": float(threshold),
        "fraction_over_threshold": float((values >= threshold).mean()),
    }


def _heatmap(values: np.ndarray) -> Image.Image:
    values = np.clip(values, 0.0, 1.0)
    red = (values * 255).astype(np.uint8)
    green = (np.sqrt(values) * 180).astype(np.uint8)
    blue = ((1.0 - values) * 80).astype(np.uint8)
    return Image.fromarray(np.stack([red, green, blue], axis=2), mode="RGB")


def _overlay_mask(image: Image.Image, mask: np.ndarray, threshold: float) -> Image.Image:
    base = image.convert("RGBA")
    rgba = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    strength = np.clip((mask - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)
    rgba[:, :, 0] = 255
    rgba[:, :, 1] = 80
    rgba[:, :, 3] = (strength * 150).astype(np.uint8)
    overlay = Image.fromarray(rgba, mode="RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


def _overlay_centers(
    image: Image.Image,
    center: np.ndarray,
    threshold: float,
    max_centers: int,
) -> Image.Image:
    output = image.convert("RGB")
    draw = ImageDraw.Draw(output)
    points = _local_maxima(center, threshold=threshold, max_points=max_centers)
    for x, y, score in points:
        radius = 2 if score < 0.75 else 3
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(255, 0, 0), width=1)
    return output


def _local_maxima(
    values: np.ndarray,
    threshold: float,
    max_points: int,
) -> list[tuple[int, int, float]]:
    padded = np.pad(values, 1, mode="edge")
    maxima = values >= threshold
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            maxima &= values >= padded[1 + dy : 1 + dy + values.shape[0], 1 + dx : 1 + dx + values.shape[1]]
    ys, xs = np.nonzero(maxima)
    if len(xs) == 0:
        return []
    scores = values[ys, xs]
    order = np.argsort(scores)[::-1][:max_points]
    return [(int(xs[idx]), int(ys[idx]), float(scores[idx])) for idx in order]


if __name__ == "__main__":
    raise SystemExit(main())
