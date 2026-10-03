"""Train the YOLO OBB alternate placement model with step checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))

from building_training_lib import DEFAULT_TRAINING_ROOT, append_jsonl  # noqa: E402


def _float_loss(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "float"):
            value = value.float()
        if hasattr(value, "mean"):
            value = value.mean()
        if hasattr(value, "cpu"):
            value = value.cpu()
        loss = float(value)
    except Exception:  # noqa: BLE001
        return None
    if not math.isfinite(loss):
        return None
    return loss


def _current_loss(trainer: Any) -> float | None:
    return _float_loss(getattr(trainer, "loss", None)) or _float_loss(getattr(trainer, "tloss", None))


def _release_cuda_cache() -> None:
    """Return unused CUDA blocks to the driver after temporary checkpoint spikes."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def _save_ultralytics_checkpoint(trainer: Any, target: Path, fitness: float) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    old_metrics = getattr(trainer, "metrics", None)
    old_fitness = getattr(trainer, "fitness", None)
    try:
        trainer.metrics = old_metrics or {}
        trainer.fitness = fitness
        ok = bool(trainer.save_model())
        last_path = Path(getattr(trainer, "last"))
        if ok and last_path.exists():
            shutil.copy2(last_path, target)
            return True
    except Exception as exc:  # noqa: BLE001
        append_jsonl(
            target.parent.parent / "step_checkpoint_errors.jsonl",
            {"time": time.time(), "target": str(target), "error": type(exc).__name__, "message": str(exc)},
        )
    finally:
        trainer.metrics = old_metrics
        trainer.fitness = old_fitness
        _release_cuda_cache()
    return False


def _last_jsonl_step(path: Path) -> int:
    if not path.exists():
        return 0
    step = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = max(step, int(row.get("step") or 0))
    return step


def _add_step_callbacks(
    model: Any,
    run_dir: Path,
    checkpoint_interval: int,
    progress_interval: int,
    min_delta: float,
    cuda_empty_cache_interval: int = 0,
) -> None:
    progress_path = run_dir / "yolo_progress.jsonl"
    state = {
        "step": _last_jsonl_step(progress_path),
        "best_step_loss": math.inf,
        "best_step": 0,
        "last_checkpoint_step": 0,
    }
    weights_dir = run_dir / "weights"
    best_step_path = weights_dir / "best_step.pt"
    latest_step_path = weights_dir / "latest_step.pt"

    existing_best = run_dir / "best_step_state.json"
    if existing_best.exists():
        try:
            saved = json.loads(existing_best.read_text(encoding="utf-8"))
            state["best_step_loss"] = float(saved.get("best_step_loss", math.inf))
            state["best_step"] = int(saved.get("best_step", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    def on_batch_end(trainer: Any) -> None:
        state["step"] += 1
        loss = _current_loss(trainer)
        epoch = int(getattr(trainer, "epoch", -1))
        row = {
            "time": time.time(),
            "step": state["step"],
            "epoch": epoch,
            "loss": loss,
            "best_step": state["best_step"],
            "best_step_loss": None if math.isinf(state["best_step_loss"]) else state["best_step_loss"],
        }
        should_log = state["step"] == 1 or state["step"] % progress_interval == 0
        should_latest = state["step"] == 1 or state["step"] % checkpoint_interval == 0
        if loss is not None and loss + min_delta < state["best_step_loss"]:
            if _save_ultralytics_checkpoint(trainer, best_step_path, -loss):
                state["best_step_loss"] = loss
                state["best_step"] = state["step"]
                row["saved_best_step"] = str(best_step_path)
                existing_best.write_text(
                    json.dumps(
                        {
                            "best_step": state["best_step"],
                            "best_step_loss": state["best_step_loss"],
                            "time": time.time(),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                should_log = True
        if should_latest:
            if loss is not None:
                _save_ultralytics_checkpoint(trainer, latest_step_path, -loss)
                row["saved_latest_step"] = str(latest_step_path)
                state["last_checkpoint_step"] = state["step"]
            should_log = True
        if cuda_empty_cache_interval > 0 and state["step"] % cuda_empty_cache_interval == 0:
            _release_cuda_cache()
        if should_log:
            append_jsonl(progress_path, row)

    def on_epoch_end(trainer: Any) -> None:
        append_jsonl(
            run_dir / "yolo_epoch_progress.jsonl",
            {
                "time": time.time(),
                "step": state["step"],
                "epoch": int(getattr(trainer, "epoch", -1)),
                "metrics": getattr(trainer, "metrics", None),
                "best_fitness": getattr(trainer, "best_fitness", None),
            },
        )

    model.add_callback("on_train_batch_end", on_batch_end)
    model.add_callback("on_fit_epoch_end", on_epoch_end)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_TRAINING_ROOT / "yolo_obb" / "placement_v1" / "dataset.yaml")
    parser.add_argument("--model", type=Path, default=Path("yolo26x-obb.pt"))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_TRAINING_ROOT / "runs" / "yolo_obb_v1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--best-step-min-delta", type=float, default=1e-4)
    parser.add_argument("--cuda-empty-cache-interval", type=int, default=0)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cuda_memory_fraction > 0:
        import torch

        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, 0)
    from ultralytics import YOLO

    args.run_dir.mkdir(parents=True, exist_ok=True)
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
    append_jsonl(
        args.run_dir / "launches.jsonl",
        {
            "time": time.time(),
            "model": str(model_path),
            "data": str(args.data),
            "run_dir": str(args.run_dir),
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "resume": bool(args.resume and last_checkpoint.exists()),
        },
    )
    model.train(
        data=str(args.data),
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
        cache=args.cache,
        patience=args.patience,
        plots=False,
        resume=bool(args.resume and last_checkpoint.exists()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
