"""Train YOLO OBB directly from the existing DDS + SQLite manifest.

This avoids materializing a second YOLO image/label dataset. Each batch opens the
source texture, chooses a labelled crop, and converts the SimHeaven SQLite
geometry to YOLO OBB tensors in memory.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO
from ultralytics.models.yolo.obb import OBBTrainer
from ultralytics.utils import LOGGER
from ultralytics.utils.ops import xyxyxyxy2xywhr

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))

from building_training_lib import (  # noqa: E402
    DEFAULT_TRAINING_ROOT,
    YOLO_HEIGHT_LABEL_MODES,
    YOLO_HEIGHT_LABEL_OFF,
    iter_jsonl,
    yolo_obb_class_names,
)
from build_yolo_obb_dataset import _choose_labelled_crop  # noqa: E402
from train_yolo_obb import _add_step_callbacks  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _class_names(height_labels: str) -> dict[int, str]:
    return yolo_obb_class_names(height_labels)


def _load_manifest_rows(manifest: Path, split: str, max_rows: int, seed: int) -> list[dict[str, Any]]:
    rows = [
        row
        for row in iter_jsonl(manifest)
        if row.get("split") == split
        and row.get("region") == "standard"
        and row.get("image")
        and row.get("labels")
    ]
    rng = random.Random(seed + (41 if split == "val" else 0))
    rng.shuffle(rows)
    if max_rows and max_rows > 0:
        rows = rows[:max_rows]
    return rows


def _parse_obb_lines(lines: list[str], imgsz: int) -> tuple[torch.Tensor, torch.Tensor]:
    classes: list[int] = []
    corners: list[list[float]] = []
    for line in lines:
        parts = line.split()
        if len(parts) != 9:
            continue
        classes.append(int(float(parts[0])))
        corners.append([float(value) * imgsz for value in parts[1:]])
    if not corners:
        return torch.zeros((0, 1), dtype=torch.float32), torch.zeros((0, 5), dtype=torch.float32)
    corner_tensor = torch.tensor(corners, dtype=torch.float32)
    boxes = xyxyxyxy2xywhr(corner_tensor)
    boxes[:, [0, 2]] /= imgsz
    boxes[:, [1, 3]] /= imgsz
    cls = torch.tensor(classes, dtype=torch.float32).view(-1, 1)
    return cls, boxes


class OrthoSQLiteOBBDataset(Dataset):
    """On-demand OBB samples from existing texture rows and SQLite labels."""

    def __init__(
        self,
        manifest: Path,
        split: str,
        imgsz: int,
        max_rows: int,
        seed: int,
        retries: int,
        max_instances: int,
        height_labels: str,
    ) -> None:
        self.manifest = manifest
        self.split = split
        self.imgsz = int(imgsz)
        self.seed = int(seed)
        self.retries = int(retries)
        self.max_instances = int(max_instances)
        self.height_labels = str(height_labels)
        self.rows = _load_manifest_rows(manifest, split, max_rows, seed)
        self.im_files = [str(row["image"]) for row in self.rows]
        if not self.rows:
            raise RuntimeError(f"No standard-region rows found for split={split!r} in {manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        worker = torch.utils.data.get_worker_info()
        worker_seed = 0 if worker is None else worker.seed
        for offset in range(8):
            row_index = (index + offset) % len(self.rows)
            row = self.rows[row_index]
            rng = random.Random(self.seed + worker_seed + row_index * 1009 + random.randrange(1 << 20))
            try:
                crop_box, _crop_bounds, lines = _choose_labelled_crop(
                    row,
                    rng,
                    self.imgsz,
                    self.retries,
                    height_labels=self.height_labels,
                )
                if self.max_instances > 0 and len(lines) > self.max_instances:
                    lines = rng.sample(lines, self.max_instances)
                cls, boxes = _parse_obb_lines(lines, self.imgsz)
                if len(cls) == 0:
                    continue
                with Image.open(row["image"]) as image:
                    image = image.convert("RGB")
                    if image.size[0] < self.imgsz or image.size[1] < self.imgsz:
                        image = image.resize((self.imgsz, self.imgsz), Image.Resampling.LANCZOS)
                    else:
                        image = image.crop(crop_box)
                array = np.asarray(image, dtype=np.uint8)
                img = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
                return {
                    "img": img,
                    "cls": cls,
                    "bboxes": boxes,
                    "batch_idx": torch.zeros((len(cls),), dtype=torch.float32),
                    "im_file": f"{row.get('chip_id', row_index)}",
                    "ori_shape": (self.imgsz, self.imgsz),
                    "resized_shape": (self.imgsz, self.imgsz),
                    "ratio_pad": ((1.0, 1.0), (0.0, 0.0)),
                }
            except Exception as exc:  # noqa: BLE001
                if offset == 7:
                    raise RuntimeError(f"Could not build OBB sample for {row.get('chip_id')}") from exc
        raise IndexError(index)

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        new_batch: dict[str, Any] = {}
        keys = batch[0].keys()
        for key in keys:
            values = [sample[key] for sample in batch]
            if key == "img":
                new_batch[key] = torch.stack(values, 0)
            elif key in {"bboxes", "cls"}:
                new_batch[key] = torch.cat(values, 0)
            elif key == "batch_idx":
                adjusted = [value + idx for idx, value in enumerate(values)]
                new_batch[key] = torch.cat(adjusted, 0)
            else:
                new_batch[key] = values
        return new_batch


class ResettableDataLoader(DataLoader):
    """DataLoader adapter for Ultralytics trainer versions that call reset()."""

    def reset(self) -> None:
        return None


class StreamingOBBTrainer(OBBTrainer):
    """OBBTrainer that sources data from OrthoSQLiteOBBDataset."""

    stream_manifest: Path = DEFAULT_TRAINING_ROOT / "manifests" / "dataset.jsonl"
    stream_max_train: int = 0
    stream_max_val: int = 0
    stream_seed: int = 1337
    stream_retries: int = 8
    stream_max_instances: int = 1000
    stream_height_labels: str = YOLO_HEIGHT_LABEL_OFF

    def get_dataset(self) -> dict[str, Any]:
        names = _class_names(self.stream_height_labels)
        return {
            "train": str(self.stream_manifest),
            "val": str(self.stream_manifest),
            "nc": len(names),
            "names": names,
            "channels": 3,
            "yaml_file": str(self.stream_manifest),
        }

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None) -> OrthoSQLiteOBBDataset:
        split = "train" if mode == "train" else "val"
        max_rows = self.stream_max_train if split == "train" else self.stream_max_val
        dataset = OrthoSQLiteOBBDataset(
            manifest=self.stream_manifest,
            split=split,
            imgsz=int(self.args.imgsz),
            max_rows=max_rows,
            seed=self.stream_seed,
            retries=self.stream_retries,
            max_instances=self.stream_max_instances,
            height_labels=self.stream_height_labels,
        )
        LOGGER.info(
            "%sStreaming Ortho4XP SQLite OBB split=%s rows=%s max_instances=%s height_labels=%s source=%s",
            "train: " if split == "train" else "val: ",
            split,
            len(dataset),
            self.stream_max_instances,
            self.stream_height_labels,
            self.stream_manifest,
        )
        return dataset

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train") -> DataLoader:
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        return ResettableDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=mode == "train",
            num_workers=self.args.workers if mode == "train" else max(0, self.args.workers),
            pin_memory=True,
            persistent_workers=self.args.workers > 0,
            collate_fn=OrthoSQLiteOBBDataset.collate_fn,
        )

    def final_eval(self) -> None:
        """Skip Ultralytics' YAML-based final revalidation for the manifest-backed stream.

        Normal end-of-epoch validation still runs. The stock final_eval path reloads
        args.data as a YOLO YAML, which is deliberately not what this trainer uses.
        """

        LOGGER.info("Skipping YAML final_eval for streaming SQLite dataset; epoch validation metrics were already run.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_TRAINING_ROOT / "manifests" / "dataset.jsonl")
    parser.add_argument("--model", type=Path, default=Path("yolo26x-obb.pt"))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_TRAINING_ROOT / "runs" / "yolo_obb_v1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-train", type=int, default=24000)
    parser.add_argument("--max-val", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--max-instances", type=int, default=1000)
    parser.add_argument("--height-labels", choices=YOLO_HEIGHT_LABEL_MODES, default=YOLO_HEIGHT_LABEL_OFF)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--best-step-min-delta", type=float, default=1e-4)
    parser.add_argument("--cuda-empty-cache-interval", type=int, default=0)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cuda_memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, 0)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    StreamingOBBTrainer.stream_manifest = args.manifest
    StreamingOBBTrainer.stream_max_train = args.max_train
    StreamingOBBTrainer.stream_max_val = args.max_val
    StreamingOBBTrainer.stream_seed = args.seed
    StreamingOBBTrainer.stream_retries = args.retries
    StreamingOBBTrainer.stream_max_instances = args.max_instances
    StreamingOBBTrainer.stream_height_labels = args.height_labels

    last_checkpoint = args.run_dir / "weights" / "last.pt"
    model_path = last_checkpoint if args.resume and last_checkpoint.exists() else args.model
    model = YOLO(str(model_path))
    _add_step_callbacks(
        model,
        args.run_dir,
        checkpoint_interval=args.checkpoint_interval,
        progress_interval=args.progress_interval,
        min_delta=args.best_step_min_delta,
        cuda_empty_cache_interval=args.cuda_empty_cache_interval,
    )
    model.train(
        trainer=StreamingOBBTrainer,
        data=str(args.manifest),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(args.run_dir.parent),
        name=args.run_dir.name,
        exist_ok=True,
        pretrained=True,
        task="obb",
        save=True,
        save_period=1,
        patience=100,
        plots=False,
        resume=bool(args.resume and last_checkpoint.exists()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
