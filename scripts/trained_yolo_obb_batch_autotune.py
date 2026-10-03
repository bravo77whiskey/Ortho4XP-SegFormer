"""Benchmark trained YOLO-OBB building-detection batch sizes against parity.

Sibling of stock_yolo_batch_autotune.py for the trained building OBB model
(_run_yolo_obb_inference). This is a diagnostic harness. It does not change
pipeline defaults; it reports whether a faster batch size is exact-parity safe
or only visually comparable to batch=1.

Run with pipeline-matched parameters (the defaults below mirror
DEFAULT_YOLO_OBB_* in O4_SFR_Building_Overlay.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import stock_yolo_batch_autotune as SAB

import O4_SFR_Building_Overlay as BLD


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _serialize_detections(detections: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {"detections": _jsonable(detections)}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["sha256"] = hashlib.sha256(blob).hexdigest()
    return payload


def _inventory(payload: dict[str, Any]) -> dict[str, Any]:
    counts_by_class: dict[str, int] = {}
    for det in payload["detections"]:
        cls = str(det.get("placement_class"))
        counts_by_class[cls] = counts_by_class.get(cls, 0) + 1
    return {
        "detection_count": len(payload["detections"]),
        "counts_by_class": dict(sorted(counts_by_class.items())),
    }


def _centers(payload: dict[str, Any]) -> np.ndarray:
    centers = [det["center"] for det in payload["detections"] if "center" in det]
    if not centers:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(centers, dtype=np.float64).reshape(-1, 2)


def _compare_payloads(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    baseline_inventory = _inventory(baseline)
    candidate_inventory = _inventory(candidate)
    exact_equal = baseline["sha256"] == candidate["sha256"]
    inventory_equal = baseline_inventory == candidate_inventory
    counts_equal = (
        baseline_inventory["counts_by_class"] == candidate_inventory["counts_by_class"]
    )
    visual_count_equal = (
        baseline_inventory["detection_count"] == candidate_inventory["detection_count"]
    )

    center_delta_px = {}
    nearest_center_delta_px = {}
    b_centers = _centers(baseline)
    c_centers = _centers(candidate)
    if len(b_centers) == len(c_centers):
        deltas = np.linalg.norm(b_centers - c_centers, axis=1)
        center_delta_px = {
            "mean": float(deltas.mean()) if len(deltas) else 0.0,
            "max": float(deltas.max()) if len(deltas) else 0.0,
        }
        nearest_center_delta_px = SAB._nearest_center_delta(b_centers, c_centers)

    return {
        "exact_equal": bool(exact_equal),
        "inventory_equal": bool(inventory_equal),
        "counts_equal": bool(counts_equal),
        "placement_counts_equal": bool(visual_count_equal),
        "visual_count_equal": bool(counts_equal and visual_count_equal),
        "center_delta_px": center_delta_px,
        "nearest_center_delta_px": nearest_center_delta_px,
        "baseline_hash": baseline["sha256"],
        "candidate_hash": candidate["sha256"],
    }


def _run_once(
    model: Any,
    image: np.ndarray,
    image_path: Path,
    args: argparse.Namespace,
    batch: int,
) -> dict[str, Any]:
    til_y_top, til_x_left, zl = SAB._parse_dds_filename(image_path)
    lat_n, lat_s, lon_w, lon_e = SAB._dds_bounds(til_y_top, til_x_left, zl)
    img_h, img_w = image.shape[:2]
    m_per_px = SAB._dds_m_per_px(lat_n, lat_s, lon_w, lon_e, img_h, img_w)

    t0 = time.perf_counter()
    detections = BLD._run_yolo_obb_inference(
        model,
        image,
        imgsz=args.imgsz,
        stride=args.stride,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        m_per_px=m_per_px,
        batch_size=batch,
    )
    elapsed_s = time.perf_counter() - t0
    return {
        "elapsed_s": elapsed_s,
        "payload": _serialize_detections(detections),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="DDS/JPG/PNG files to benchmark.")
    parser.add_argument(
        "--batches", type=SAB._parse_batches, default=SAB._parse_batches("1,2,4,8,16")
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Unmeasured pre-sweep runs per image to remove first-predict CUDA warmup bias.",
    )
    parser.add_argument("--warmup-batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--checkpoint", default=BLD.DEFAULT_YOLO_OBB_CHECKPOINT)
    parser.add_argument(
        "--fuse",
        action="store_true",
        help="Load the model fused (conv+BN). Not bit-exact with unfused weights; "
        "only for evaluating a fuse-by-default change.",
    )
    parser.add_argument("--conf", type=float, default=BLD.DEFAULT_YOLO_OBB_CONF)
    parser.add_argument("--iou", type=float, default=BLD.DEFAULT_YOLO_OBB_IOU)
    parser.add_argument("--stride", type=int, default=BLD.DEFAULT_YOLO_OBB_STRIDE)
    parser.add_argument("--imgsz", type=int, default=BLD.DEFAULT_YOLO_OBB_IMGSZ)
    parser.add_argument("--max-det", type=int, default=BLD.DEFAULT_YOLO_OBB_MAX_DET)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be >= 0")
    if args.warmup_batch < 1:
        raise ValueError("--warmup-batch must be >= 1")
    if 1 not in args.batches:
        args.batches = [1, *args.batches]

    images = [Path(p).resolve() for p in args.images]

    out_path = (
        Path(args.out).resolve()
        if args.out
        else ROOT / "tmp" / "trained_yolo_obb_batch_autotune" / (
            time.strftime("%Y%m%d_%H%M%S") + "_summary.json"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[obb-autotune] python={sys.executable}", flush=True)
    print(f"[obb-autotune] checkpoint={args.checkpoint} fuse={args.fuse}", flush=True)
    print(f"[obb-autotune] batches={args.batches} repeats={args.repeats}", flush=True)
    model = BLD._load_yolo_obb_model(args.checkpoint, fuse=args.fuse)

    report: dict[str, Any] = {
        "python": sys.executable,
        "checkpoint": str(args.checkpoint),
        "fuse": bool(args.fuse),
        "batches": args.batches,
        "repeats": args.repeats,
        "warmup_runs": args.warmup_runs,
        "warmup_batch": args.warmup_batch,
        "params": {
            "imgsz": args.imgsz,
            "stride": args.stride,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
        },
        "images": [],
    }

    for image_path in images:
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        image = SAB._load_image(image_path)
        image_report: dict[str, Any] = {
            "path": str(image_path),
            "shape": list(image.shape),
            "batches": {},
        }
        baseline_payload = None
        print(f"[obb-autotune] image={image_path}", flush=True)
        for warmup_idx in range(args.warmup_runs):
            SAB._empty_cuda_cache()
            run = _run_once(model, image, image_path, args, args.warmup_batch)
            print(
                "[obb-autotune] "
                f"warmup={warmup_idx + 1}/{args.warmup_runs} "
                f"batch={args.warmup_batch} time={run['elapsed_s']:.3f}s",
                flush=True,
            )
        for batch in args.batches:
            runs = []
            for repeat in range(args.repeats):
                SAB._empty_cuda_cache()
                run = _run_once(model, image, image_path, args, batch)
                runs.append(run)
                inventory = _inventory(run["payload"])
                print(
                    "[obb-autotune] "
                    f"batch={batch} repeat={repeat + 1}/{args.repeats} "
                    f"time={run['elapsed_s']:.3f}s "
                    f"detections={inventory['detection_count']} "
                    f"hash={run['payload']['sha256'][:12]}",
                    flush=True,
                )

            if batch == 1:
                baseline_payload = runs[0]["payload"]
            assert baseline_payload is not None

            repeat_hashes = [run["payload"]["sha256"] for run in runs]
            compare = _compare_payloads(baseline_payload, runs[0]["payload"])
            image_report["batches"][batch] = {
                "times_s": [float(run["elapsed_s"]) for run in runs],
                "median_elapsed_s": float(median(run["elapsed_s"] for run in runs)),
                "repeat_exact_equal": len(set(repeat_hashes)) == 1,
                "inventory": _inventory(runs[0]["payload"]),
                "compare_to_batch_1": compare,
            }
        report["images"].append(image_report)

    exact_candidate = SAB._summarize_candidate(
        report["images"], args.batches, "exact_equal"
    )
    inventory_candidate = SAB._summarize_candidate(
        report["images"], args.batches, "inventory_equal"
    )
    visual_candidate = SAB._summarize_candidate(
        report["images"], args.batches, "visual_count_equal"
    )
    report["recommended_exact_candidate"] = exact_candidate
    report["recommended_visual_inventory_candidate"] = inventory_candidate
    report["recommended_visual_count_candidate"] = visual_candidate

    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[obb-autotune] wrote {out_path}", flush=True)
    print(f"[obb-autotune] exact candidate: {exact_candidate}", flush=True)
    print(f"[obb-autotune] visual-inventory candidate: {inventory_candidate}", flush=True)
    print(f"[obb-autotune] visual-count candidate: {visual_candidate}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
