"""Benchmark stock YOLO-OBB batch sizes against exact and visual parity.

This is a diagnostic harness. It does not change pipeline defaults; it reports
whether a faster batch size is exact-parity safe or only visually comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
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

import O4_SFR_Stock_Yolo_Objects as STOCK


DEFAULT_IMAGES = (
    Path(r"G:\XP12 Addons2\Custom Scenery\zOrtho4XP_+22+120\textures\114176_218672_Arc18.dds"),
    Path(r"G:\XP12 Addons2\Custom Scenery\zOrtho4XP_+25+121\textures\112160_219360_Arc18.dds"),
)

DDS_STD_RE = re.compile(
    r"^(?P<til_y>\d+)_(?P<til_x>\d+)_(?P<provider>[A-Za-z][A-Za-z0-9_]*)(?P<zl>\d{2})\.dds$",
    re.IGNORECASE,
)
TILE_RE = re.compile(r"zOrtho4XP_(?P<lat>[+-]\d{2})(?P<lon>[+-]\d{3})", re.IGNORECASE)


def _parse_batches(value: str) -> list[int]:
    batches = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        batch = int(part)
        if batch < 1:
            raise argparse.ArgumentTypeError("batch sizes must be >= 1")
        batches.append(batch)
    if not batches:
        raise argparse.ArgumentTypeError("at least one batch size is required")
    return sorted(set(batches))


def _load_image(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _parse_dds_filename(path: Path) -> tuple[int, int, int]:
    match = DDS_STD_RE.match(path.name)
    if not match:
        raise ValueError(f"Cannot parse standard Ortho4XP DDS name: {path.name}")
    return int(match.group("til_y")), int(match.group("til_x")), int(match.group("zl"))


def _infer_tile_lat_lon(path: Path, lat: int | None, lon: int | None) -> tuple[int, int]:
    if lat is not None and lon is not None:
        return int(lat), int(lon)
    for parent in (path.parent, *path.parents):
        match = TILE_RE.search(parent.name)
        if match:
            return int(match.group("lat")), int(match.group("lon"))
    raise ValueError(
        f"Cannot infer tile lat/lon from {path}; pass --lat and --lon explicitly"
    )


def _gtile_to_wgs84(til_x: int, til_y: int, zl: int) -> tuple[float, float]:
    rat_x = til_x / (2 ** (zl - 1)) - 1
    rat_y = 1 - til_y / (2 ** (zl - 1))
    lon = rat_x * 180
    lat = 360 / math.pi * math.atan(math.exp(math.pi * rat_y)) - 90
    return lat, lon


def _dds_bounds(til_y_top: int, til_x_left: int, zl: int) -> tuple[float, float, float, float]:
    lat_n, lon_w = _gtile_to_wgs84(til_x_left, til_y_top, zl)
    lat_s, lon_e = _gtile_to_wgs84(til_x_left + 16, til_y_top + 16, zl)
    return lat_n, lat_s, lon_w, lon_e


def _dds_m_per_px(
    lat_n: float, lat_s: float, lon_w: float, lon_e: float, img_h: int, img_w: int
) -> float:
    mid_lat_rad = math.radians((lat_n + lat_s) / 2)
    lon_span_m = (lon_e - lon_w) * 111320.0 * math.cos(mid_lat_rad)
    lat_span_m = (lat_n - lat_s) * 110540.0
    return ((lon_span_m / float(img_w)) + (lat_span_m / float(img_h))) / 2.0


def _float(value: Any) -> float:
    return float(value)


def _ring_payload(ring: list[tuple[float, float]]) -> list[list[float]]:
    return [[_float(lon), _float(lat)] for lon, lat in ring]


def _serialize_results(result: STOCK.StockYoloResults) -> dict[str, Any]:
    payload = {
        "objects": [
            [_float(lon), _float(lat), _float(heading), str(asset)]
            for lon, lat, heading, asset in result.placed_objects
        ],
        "facades": [
            [_ring_payload(ring), str(asset), _float(height_m)]
            for ring, asset, height_m in result.placed_facades
        ],
        "draped": [
            [_ring_payload(ring), str(asset)]
            for ring, asset in result.placed_draped
        ],
        "occupied_px_polys": [
            np.asarray(poly).astype(int).tolist()
            for poly in result.occupied_px_polys
        ],
        "counts_by_class": {
            str(cls): int(count)
            for cls, count in sorted(result.counts_by_class.items())
        },
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["sha256"] = hashlib.sha256(blob).hexdigest()
    return payload


def _inventory(payload: dict[str, Any]) -> dict[str, Any]:
    object_assets: dict[str, int] = {}
    facade_assets: dict[str, int] = {}
    for _lon, _lat, _heading, asset in payload["objects"]:
        object_assets[asset] = object_assets.get(asset, 0) + 1
    for _ring, asset, _height_m in payload["facades"]:
        facade_assets[asset] = facade_assets.get(asset, 0) + 1
    return {
        "object_count": len(payload["objects"]),
        "facade_count": len(payload["facades"]),
        "draped_count": len(payload["draped"]),
        "occupied_poly_count": len(payload["occupied_px_polys"]),
        "counts_by_class": payload["counts_by_class"],
        "object_assets": dict(sorted(object_assets.items())),
        "facade_assets": dict(sorted(facade_assets.items())),
    }


def _poly_centers(payload: dict[str, Any]) -> np.ndarray:
    centers = []
    for poly in payload["occupied_px_polys"]:
        arr = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        centers.append(arr.mean(axis=0))
    if not centers:
        return np.zeros((0, 2), dtype=np.float64)
    return np.vstack(centers)


def _nearest_center_delta(baseline: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if len(baseline) != len(candidate):
        return {}
    if len(baseline) == 0:
        return {"mean": 0.0, "max": 0.0}

    remaining = list(range(len(candidate)))
    deltas = []
    for base_center in baseline:
        cand = candidate[remaining]
        distances = np.linalg.norm(cand - base_center, axis=1)
        local_idx = int(np.argmin(distances))
        deltas.append(float(distances[local_idx]))
        del remaining[local_idx]
    arr = np.asarray(deltas, dtype=np.float64)
    return {"mean": float(arr.mean()), "max": float(arr.max())}


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
    placement_counts_equal = all(
        baseline_inventory[key] == candidate_inventory[key]
        for key in ("object_count", "facade_count", "draped_count", "occupied_poly_count")
    )
    asset_multiset_equal = (
        baseline_inventory["object_assets"] == candidate_inventory["object_assets"]
        and baseline_inventory["facade_assets"] == candidate_inventory["facade_assets"]
    )

    center_delta_px = {}
    nearest_center_delta_px = {}
    b_centers = _poly_centers(baseline)
    c_centers = _poly_centers(candidate)
    if len(b_centers) == len(c_centers):
        deltas = np.linalg.norm(b_centers - c_centers, axis=1)
        center_delta_px = {
            "mean": float(deltas.mean()) if len(deltas) else 0.0,
            "max": float(deltas.max()) if len(deltas) else 0.0,
        }
        nearest_center_delta_px = _nearest_center_delta(b_centers, c_centers)

    return {
        "exact_equal": bool(exact_equal),
        "inventory_equal": bool(inventory_equal),
        "counts_equal": bool(counts_equal),
        "placement_counts_equal": bool(placement_counts_equal),
        "asset_multiset_equal": bool(asset_multiset_equal),
        "visual_count_equal": bool(counts_equal and placement_counts_equal),
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
    til_y_top, til_x_left, zl = _parse_dds_filename(image_path)
    lat, lon = _infer_tile_lat_lon(image_path, args.lat, args.lon)
    lat_n, lat_s, lon_w, lon_e = _dds_bounds(til_y_top, til_x_left, zl)
    img_h, img_w = image.shape[:2]
    m_per_px = _dds_m_per_px(lat_n, lat_s, lon_w, lon_e, img_h, img_w)

    t0 = time.perf_counter()
    result = STOCK.run_stock_yolo_pass(
        image,
        img_w=img_w,
        img_h=img_h,
        lat=lat,
        lon=lon,
        lat_n=lat_n,
        lat_s=lat_s,
        lon_w=lon_w,
        lon_e=lon_e,
        m_per_px=m_per_px,
        model=model,
        conf=args.conf,
        iou=args.iou,
        stride=args.stride,
        imgsz=args.imgsz,
        max_det=args.max_det,
        device=args.device,
        batch_size=batch,
    )
    elapsed_s = time.perf_counter() - t0
    return {
        "elapsed_s": elapsed_s,
        "model_elapsed_s": float(result.inference_time_s),
        "payload": _serialize_results(result),
    }


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


def _summarize_candidate(
    image_results: list[dict[str, Any]], batches: list[int], parity_key: str
) -> dict[str, Any] | None:
    baseline_total = 0.0
    totals = {batch: 0.0 for batch in batches}
    ok = {batch: True for batch in batches}
    for image_result in image_results:
        baseline_time = image_result["batches"][1]["median_elapsed_s"]
        baseline_total += baseline_time
        for batch in batches:
            batch_result = image_result["batches"][batch]
            totals[batch] += batch_result["median_elapsed_s"]
            if batch != 1 and not batch_result["compare_to_batch_1"][parity_key]:
                ok[batch] = False

    candidates = []
    for batch in batches:
        if batch == 1 or not ok[batch]:
            continue
        speedup_pct = (
            (baseline_total - totals[batch]) / baseline_total * 100.0
            if baseline_total > 0 else 0.0
        )
        candidates.append((speedup_pct, -totals[batch], batch))
    if not candidates:
        return None
    speedup_pct, neg_time, batch = max(candidates)
    return {
        "batch": batch,
        "speedup_pct": float(speedup_pct),
        "total_median_elapsed_s": float(-neg_time),
        "parity": parity_key,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", help="DDS/JPG/PNG files to benchmark.")
    parser.add_argument("--batches", type=_parse_batches, default=_parse_batches("1,2,4,8,16"))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Unmeasured pre-sweep runs per image to remove first-predict CUDA warmup bias.",
    )
    parser.add_argument("--warmup-batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--checkpoint", default=STOCK.DEFAULT_STOCK_YOLO_CHECKPOINT)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--max-det", type=int, default=500)
    parser.add_argument("--lat", type=int, default=None)
    parser.add_argument("--lon", type=int, default=None)
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
    if not images:
        images = [path for path in DEFAULT_IMAGES if path.exists()]
    if not images:
        print(
            "[stock-autotune] no images supplied and default positive DDS files were not found",
            file=sys.stderr,
        )
        return 2

    out_path = (
        Path(args.out).resolve()
        if args.out
        else ROOT / "tmp" / "stock_yolo_batch_autotune" / (
            time.strftime("%Y%m%d_%H%M%S") + "_summary.json"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[stock-autotune] python={sys.executable}", flush=True)
    print(f"[stock-autotune] checkpoint={args.checkpoint}", flush=True)
    print(f"[stock-autotune] batches={args.batches} repeats={args.repeats}", flush=True)
    model = STOCK.load_stock_yolo_model(args.checkpoint)

    report: dict[str, Any] = {
        "python": sys.executable,
        "checkpoint": str(args.checkpoint),
        "batches": args.batches,
        "repeats": args.repeats,
        "warmup_runs": args.warmup_runs,
        "warmup_batch": args.warmup_batch,
        "images": [],
    }

    for image_path in images:
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        image = _load_image(image_path)
        image_report: dict[str, Any] = {
            "path": str(image_path),
            "shape": list(image.shape),
            "batches": {},
        }
        baseline_payload = None
        print(f"[stock-autotune] image={image_path}", flush=True)
        for warmup_idx in range(args.warmup_runs):
            _empty_cuda_cache()
            run = _run_once(model, image, image_path, args, args.warmup_batch)
            print(
                "[stock-autotune] "
                f"warmup={warmup_idx + 1}/{args.warmup_runs} "
                f"batch={args.warmup_batch} time={run['elapsed_s']:.3f}s",
                flush=True,
            )
        for batch in args.batches:
            runs = []
            for repeat in range(args.repeats):
                _empty_cuda_cache()
                run = _run_once(model, image, image_path, args, batch)
                runs.append(run)
                inventory = _inventory(run["payload"])
                print(
                    "[stock-autotune] "
                    f"batch={batch} repeat={repeat + 1}/{args.repeats} "
                    f"time={run['elapsed_s']:.3f}s "
                    f"objects={inventory['object_count']} facades={inventory['facade_count']} "
                    f"hash={run['payload']['sha256'][:12]}",
                    flush=True,
                )

            if batch == 1:
                baseline_payload = runs[0]["payload"]
            assert baseline_payload is not None

            repeat_hashes = [run["payload"]["sha256"] for run in runs]
            compare = _compare_payloads(baseline_payload, runs[0]["payload"])
            batch_report = {
                "times_s": [float(run["elapsed_s"]) for run in runs],
                "model_times_s": [float(run["model_elapsed_s"]) for run in runs],
                "median_elapsed_s": float(median(run["elapsed_s"] for run in runs)),
                "median_model_elapsed_s": float(median(run["model_elapsed_s"] for run in runs)),
                "repeat_exact_equal": len(set(repeat_hashes)) == 1,
                "inventory": _inventory(runs[0]["payload"]),
                "compare_to_batch_1": compare,
            }
            image_report["batches"][batch] = batch_report
        report["images"].append(image_report)

    exact_candidate = _summarize_candidate(report["images"], args.batches, "exact_equal")
    inventory_candidate = _summarize_candidate(
        report["images"], args.batches, "inventory_equal"
    )
    visual_candidate = _summarize_candidate(
        report["images"], args.batches, "visual_count_equal"
    )
    report["recommended_exact_candidate"] = exact_candidate
    report["recommended_visual_inventory_candidate"] = inventory_candidate
    report["recommended_visual_count_candidate"] = visual_candidate

    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[stock-autotune] wrote {out_path}", flush=True)
    print(f"[stock-autotune] exact candidate: {exact_candidate}", flush=True)
    print(f"[stock-autotune] visual-inventory candidate: {inventory_candidate}", flush=True)
    print(f"[stock-autotune] visual-count candidate: {visual_candidate}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
