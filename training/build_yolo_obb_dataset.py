"""Build a resumable YOLO OBB dataset from Ortho4XP textures and SimHeaven labels."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageFile
from shapely.geometry import Polygon, box

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))

from building_training_lib import (  # noqa: E402
    CLASS_BACKGROUND,
    DEFAULT_TRAINING_ROOT,
    YOLO_HEIGHT_LABEL_MODES,
    YOLO_HEIGHT_LABEL_OFF,
    _choose_crop_box,
    _crop_bounds_from_pixels,
    _features_from_sqlite_for_bounds,
    _iter_polygons,
    append_jsonl,
    encode_yolo_obb_class,
    iter_jsonl,
    lonlat_to_pixel,
    yolo_obb_class_names,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass(frozen=True)
class YoloBuildConfig:
    manifest: Path
    output_root: Path
    chip_size: int = 1024
    max_train: int = 24000
    max_val: int = 4000
    seed: int = 1337
    jpeg_quality: int = 92
    retries: int = 8
    progress_interval: int = 100
    height_labels: str = YOLO_HEIGHT_LABEL_OFF


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _split_dirs(root: Path, split: str) -> tuple[Path, Path]:
    return root / "images" / split, root / "labels" / split


def _completion_manifest(root: Path) -> Path:
    return root / "build_manifest.jsonl"


def _progress_path(root: Path) -> Path:
    return root / "build_progress.json"


def _load_completed(root: Path) -> set[str]:
    return {str(row["source_chip_id"]) for row in iter_jsonl(_completion_manifest(root)) if row.get("ok")}


def _row_bounds(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["west"]),
        float(row["south"]),
        float(row["east"]),
        float(row["north"]),
    )


def _load_rows(path: Path, split: str, limit: int, seed: int) -> list[dict[str, Any]]:
    rows = [
        row
        for row in iter_jsonl(path)
        if row.get("split") == split
        and row.get("region") == "standard"
        and row.get("image")
        and row.get("labels")
    ]
    rng = random.Random(seed + (17 if split == "val" else 0))
    rng.shuffle(rows)
    if limit and limit > 0:
        rows = rows[:limit]
    return rows


def _polygon_to_pixel_parts(
    geometry: Any,
    crop_bounds: tuple[float, float, float, float],
    chip_size: int,
) -> Iterable[Polygon]:
    query = box(*crop_bounds)
    clipped = geometry.intersection(query)
    if clipped.is_empty:
        return
    for polygon in _iter_polygons(clipped):
        exterior = [
            lonlat_to_pixel(lon, lat, chip_size, chip_size, crop_bounds)
            for lon, lat in polygon.exterior.coords
        ]
        if len(exterior) < 4:
            continue
        pixel_poly = Polygon(exterior)
        if pixel_poly.is_valid and pixel_poly.area >= 4.0:
            yield pixel_poly


def _ordered_rect_corners(rect: Polygon) -> list[tuple[float, float]]:
    coords = list(rect.exterior.coords)[:-1]
    if len(coords) != 4:
        return []
    cx = sum(x for x, _ in coords) / 4.0
    cy = sum(y for _, y in coords) / 4.0
    ordered = sorted(coords, key=lambda pt: math.atan2(pt[1] - cy, pt[0] - cx))
    start = min(range(4), key=lambda idx: (ordered[idx][1], ordered[idx][0]))
    return ordered[start:] + ordered[:start]


def _obb_labels_for_crop(
    features: list[Any],
    crop_bounds: tuple[float, float, float, float],
    chip_size: int,
    height_labels: str = YOLO_HEIGHT_LABEL_OFF,
) -> list[str]:
    lines: list[str] = []
    for feature in features:
        if int(feature.placement_class) == CLASS_BACKGROUND:
            continue
        class_id = encode_yolo_obb_class(
            int(feature.placement_class),
            getattr(feature, "height", None),
            height_labels=height_labels,
        )
        if class_id is None:
            continue
        for pixel_poly in _polygon_to_pixel_parts(feature.geometry, crop_bounds, chip_size):
            rect = pixel_poly.minimum_rotated_rectangle
            corners = _ordered_rect_corners(rect)
            if len(corners) != 4:
                continue
            xs = [x for x, _ in corners]
            ys = [y for _, y in corners]
            if max(xs) - min(xs) < 2.0 or max(ys) - min(ys) < 2.0:
                continue
            values: list[str] = [str(class_id)]
            for x, y in corners:
                x = min(chip_size - 1.0, max(0.0, x)) / chip_size
                y = min(chip_size - 1.0, max(0.0, y)) / chip_size
                values.extend([f"{x:.6f}", f"{y:.6f}"])
            lines.append(" ".join(values))
    return lines


def _choose_labelled_crop(
    row: dict[str, Any],
    rng: random.Random,
    chip_size: int,
    retries: int,
    height_labels: str = YOLO_HEIGHT_LABEL_OFF,
) -> tuple[tuple[int, int, int, int], tuple[float, float, float, float], list[str]]:
    image_width = int(row.get("width") or chip_size)
    image_height = int(row.get("height") or chip_size)
    full_bounds = _row_bounds(row)
    full_features = _features_from_sqlite_for_bounds(row["labels"], full_bounds)
    feature_records = [{"b": list(feature.geometry.bounds)} for feature in full_features]

    original_state = random.getstate()
    try:
        for _attempt in range(max(1, retries)):
            random.seed(rng.randrange(1 << 30))
            crop_box = _choose_crop_box(
                feature_records,
                full_bounds,
                image_width,
                image_height,
                chip_size,
                random_crop=True,
            )
            crop_bounds = _crop_bounds_from_pixels(full_bounds, image_width, image_height, crop_box)
            crop_features = _features_from_sqlite_for_bounds(row["labels"], crop_bounds)
            labels = _obb_labels_for_crop(
                crop_features,
                crop_bounds,
                chip_size,
                height_labels=height_labels,
            )
            if labels:
                return crop_box, crop_bounds, labels
        crop_box = _choose_crop_box(
            feature_records,
            full_bounds,
            image_width,
            image_height,
            chip_size,
            random_crop=False,
        )
        crop_bounds = _crop_bounds_from_pixels(full_bounds, image_width, image_height, crop_box)
        crop_features = _features_from_sqlite_for_bounds(row["labels"], crop_bounds)
        return crop_box, crop_bounds, _obb_labels_for_crop(
            crop_features,
            crop_bounds,
            chip_size,
            height_labels=height_labels,
        )
    finally:
        random.setstate(original_state)


def _write_chip(
    row: dict[str, Any],
    split: str,
    index: int,
    config: YoloBuildConfig,
    rng: random.Random,
) -> dict[str, Any]:
    chip_id = f"{_safe_name(str(row['chip_id']))}_{index:06d}"
    image_dir, label_dir = _split_dirs(config.output_root, split)
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{chip_id}.jpg"
    label_path = label_dir / f"{chip_id}.txt"

    crop_box, crop_bounds, labels = _choose_labelled_crop(
        row,
        rng,
        config.chip_size,
        config.retries,
        height_labels=config.height_labels,
    )
    if not labels:
        raise ValueError("crop has no OBB labels")

    if not image_path.exists():
        with Image.open(row["image"]) as image:
            image = image.convert("RGB")
            if image.size[0] < config.chip_size or image.size[1] < config.chip_size:
                image = image.resize((config.chip_size, config.chip_size), Image.Resampling.LANCZOS)
            else:
                image = image.crop(crop_box)
            image.save(image_path, quality=config.jpeg_quality, optimize=True)

    label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
    return {
        "ok": True,
        "source_chip_id": row["chip_id"],
        "yolo_chip_id": chip_id,
        "split": split,
        "image": str(image_path),
        "label": str(label_path),
        "labels": len(labels),
        "tile": row.get("tile"),
        "region": row.get("region"),
        "crop_bounds": [round(value, 9) for value in crop_bounds],
        "time": time.time(),
    }


def _write_dataset_yaml(root: Path, height_labels: str) -> Path:
    names = yolo_obb_class_names(height_labels)
    lines = [
        f"path: {str(root).replace(chr(92), '/')}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for idx, name in sorted(names.items()):
        lines.append(f"  {idx}: {name}")
    yaml_path = root / "dataset.yaml"
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def _build_split(config: YoloBuildConfig, split: str, rows: list[dict[str, Any]], completed: set[str]) -> int:
    rng = random.Random(config.seed + (101 if split == "val" else 0))
    written = 0
    skipped = 0
    for index, row in enumerate(rows):
        if str(row["chip_id"]) in completed:
            continue
        try:
            result = _write_chip(row, split, index, config, rng)
        except Exception as exc:  # noqa: BLE001 - keep long builds moving and resumable.
            skipped += 1
            append_jsonl(
                config.output_root / "build_skipped.jsonl",
                {
                    "ok": False,
                    "source_chip_id": row.get("chip_id"),
                    "split": split,
                    "reason": type(exc).__name__,
                    "message": str(exc),
                    "time": time.time(),
                },
            )
            continue
        append_jsonl(_completion_manifest(config.output_root), result)
        completed.add(str(row["chip_id"]))
        written += 1
        if written % config.progress_interval == 0:
            progress = {
                "split": split,
                "written": written,
                "skipped": skipped,
                "completed_total": len(completed),
                "target_rows": len(rows),
                "time": time.time(),
            }
            _progress_path(config.output_root).write_text(json.dumps(progress, indent=2), encoding="utf-8")
            print(
                f"[yolo-obb-build] split={split} written={written} skipped={skipped} completed={len(completed)}",
                flush=True,
            )
    return written


def build_yolo_obb_dataset(config: YoloBuildConfig) -> Path:
    config.output_root.mkdir(parents=True, exist_ok=True)
    completed = _load_completed(config.output_root)
    train_rows = _load_rows(config.manifest, "train", config.max_train, config.seed)
    val_rows = _load_rows(config.manifest, "val", config.max_val, config.seed)
    print(
        f"[yolo-obb-build] root={config.output_root} train_rows={len(train_rows)} val_rows={len(val_rows)} completed={len(completed)}",
        flush=True,
    )
    train_written = _build_split(config, "train", train_rows, completed)
    val_written = _build_split(config, "val", val_rows, completed)
    yaml_path = _write_dataset_yaml(config.output_root, config.height_labels)
    summary = {
        "dataset_yaml": str(yaml_path),
        "train_written_this_run": train_written,
        "val_written_this_run": val_written,
        "completed_total": len(completed),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "time": time.time(),
    }
    (config.output_root / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[yolo-obb-build] dataset_yaml={yaml_path}", flush=True)
    return yaml_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_TRAINING_ROOT / "manifests" / "dataset.jsonl")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_TRAINING_ROOT / "yolo_obb" / "placement_v1")
    parser.add_argument("--chip-size", type=int, default=1024)
    parser.add_argument("--max-train", type=int, default=24000)
    parser.add_argument("--max-val", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--height-labels", choices=YOLO_HEIGHT_LABEL_MODES, default=YOLO_HEIGHT_LABEL_OFF)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = YoloBuildConfig(
        manifest=args.manifest,
        output_root=args.output_root,
        chip_size=args.chip_size,
        max_train=args.max_train,
        max_val=args.max_val,
        seed=args.seed,
        jpeg_quality=args.jpeg_quality,
        retries=args.retries,
        progress_interval=args.progress_interval,
        height_labels=args.height_labels,
    )
    build_yolo_obb_dataset(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
