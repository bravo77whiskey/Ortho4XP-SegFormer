"""Compare two HeightNet checkpoints on identical YOLO OBB detections.

The command is intentionally analysis-only: it writes CSV/JSON/PNG artifacts
to an output directory and never edits Ortho4XP.cfg or X-Plane scenery.
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import glob
import heapq
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_SFR_Building_Overlay as BLD
import O4_SFR_Debug_Images as DEBUG
import O4_SFR_Height_Model as HEIGHT


HEIGHT_BANDS = (
    ("0-4m", 0.0, 4.0),
    ("4-8m", 4.0, 8.0),
    ("8-15m", 8.0, 15.0),
    ("15-30m", 15.0, 30.0),
    ("30m+", 30.0, math.inf),
)

CSV_FIELDS = (
    "source",
    "detection_index",
    "center_x",
    "center_y",
    "confidence",
    "placement_class",
    "area_m2",
    "long_side_m",
    "old_raw_m",
    "old_final_m",
    "new_raw_m",
    "new_final_m",
    "raw_delta_m",
    "final_delta_m",
    "old_floored",
    "old_capped",
    "new_floored",
    "new_capped",
)


def finalize_heights(detections, raw_heights):
    """Return post-floor/cap heights plus masks describing both adjustments."""
    raw = np.asarray(raw_heights, dtype=np.float64)
    final = np.full(raw.shape, np.nan, dtype=np.float64)
    floored = np.zeros(raw.shape, dtype=bool)
    capped = np.zeros(raw.shape, dtype=bool)
    floor_m = float(HEIGHT.HEIGHT_MODEL_MIN_M)
    for index, (detection, height) in enumerate(zip(detections, raw)):
        if not math.isfinite(float(height)):
            continue
        floor_applied = max(float(height), floor_m)
        capped_height = float(BLD._capped_detection_height_m(
            detection, floor_applied
        ))
        final[index] = capped_height
        floored[index] = float(height) < floor_m
        capped[index] = capped_height < floor_applied - 1e-9
    return final, floored, capped


def summarize_predictions(values, *, inference_seconds, floor_count, cap_count):
    """Return JSON-safe distribution and throughput statistics."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    count = int(len(finite))
    bands = {
        label: int(np.count_nonzero((finite >= low) & (finite < high)))
        for label, low, high in HEIGHT_BANDS
    }
    if count:
        percentiles = np.percentile(finite, [10, 25, 50, 75, 90]).tolist()
        mean_m = float(np.mean(finite))
        min_m = float(np.min(finite))
        max_m = float(np.max(finite))
    else:
        percentiles = [math.nan] * 5
        mean_m = min_m = max_m = math.nan
    elapsed = float(inference_seconds)
    return {
        "count": count,
        "mean_m": mean_m,
        "min_m": min_m,
        "p10_m": float(percentiles[0]),
        "p25_m": float(percentiles[1]),
        "median_m": float(percentiles[2]),
        "p75_m": float(percentiles[3]),
        "p90_m": float(percentiles[4]),
        "max_m": max_m,
        "bands": bands,
        "floor_count": int(floor_count),
        "cap_count": int(cap_count),
        "inference_seconds": elapsed,
        "throughput_buildings_per_s": (
            float(count / elapsed) if elapsed > 0.0 else math.nan
        ),
    }


def summarize_differences(old_values, new_values):
    """Summarize aligned ``new - old`` prediction differences."""
    old = np.asarray(old_values, dtype=np.float64)
    new = np.asarray(new_values, dtype=np.float64)
    valid = np.isfinite(old) & np.isfinite(new)
    delta = new[valid] - old[valid]
    if not len(delta):
        return {
            "count": 0,
            "mean_delta_m": math.nan,
            "median_delta_m": math.nan,
            "median_abs_delta_m": math.nan,
            "p90_abs_delta_m": math.nan,
            "new_taller_count": 0,
            "old_taller_count": 0,
            "within_1m_count": 0,
        }
    absolute = np.abs(delta)
    return {
        "count": int(len(delta)),
        "mean_delta_m": float(np.mean(delta)),
        "median_delta_m": float(np.median(delta)),
        "median_abs_delta_m": float(np.median(absolute)),
        "p90_abs_delta_m": float(np.percentile(absolute, 90)),
        "new_taller_count": int(np.count_nonzero(delta > 0.0)),
        "old_taller_count": int(np.count_nonzero(delta < 0.0)),
        "within_1m_count": int(np.count_nonzero(absolute <= 1.0)),
    }


def _resolve_inputs(values, limit=0, skip=0):
    paths = []
    for value in values:
        expanded = [Path(item) for item in glob.glob(value)]
        if not expanded:
            expanded = [Path(value)]
        for candidate in expanded:
            if candidate.is_dir():
                paths.extend(sorted(candidate.glob("*.dds")))
            elif candidate.is_file() and candidate.suffix.lower() == ".dds":
                paths.append(candidate)
            else:
                raise FileNotFoundError(f"DDS input not found: {candidate}")
    unique = list(dict.fromkeys(path.resolve() for path in paths))
    if skip > 0:
        unique = unique[skip:]
    if limit > 0:
        unique = unique[:limit]
    if not unique:
        raise ValueError("No DDS inputs were selected")
    return unique


def _assert_safe_output(path):
    if any(part.casefold() == "custom scenery" for part in path.resolve().parts):
        raise ValueError("Comparison output may not be written into Custom Scenery")


def _tile_name(path):
    return next(
        (part for part in reversed(path.parts) if part.lower().startswith("zortho4xp_")),
        path.parent.name,
    )


def _source_key(path):
    tile = _tile_name(path)
    return f"{tile}_{path.stem}".replace(" ", "_")


def _image_m_per_px(path, image):
    metadata = DEBUG.parse_texture_metadata(path)
    lat_n, lat_s, lon_w, lon_e = BLD.dds_bounds(
        metadata.til_y_top, metadata.til_x_left, metadata.zoomlevel
    )
    return DEBUG._dds_m_per_px(
        lat_n, lat_s, lon_w, lon_e, image.shape[0], image.shape[1]
    )


def _cuda_sync(device):
    import torch

    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _predict_timed(model, image, detections, m_per_px, batch_size, device):
    import torch

    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        baseline = int(torch.cuda.memory_allocated())
    else:
        baseline = 0
    _cuda_sync(device)
    started = time.perf_counter()
    raw = HEIGHT.predict_detection_heights(
        model, image, detections, m_per_px, batch_size=batch_size
    )
    _cuda_sync(device)
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated()) if use_cuda else 0
    return raw, elapsed, max(0, peak - baseline), peak


def _height_color(height):
    normalized = min(max(float(height), 0.0), 60.0) / 60.0
    red, green, blue = colorsys.hsv_to_rgb((1.0 - normalized) * 0.66, 0.9, 1.0)
    return int(red * 255), int(green * 255), int(blue * 255)


def _delta_color(delta):
    strength = min(1.0, abs(float(delta)) / 15.0)
    if delta >= 0.0:
        return 255, int(220 * (1.0 - strength)), 60
    return 60, int(220 * (1.0 - strength)), 255


def _annotated_panel(image, detections, values, title, *, deltas=False,
                     max_side=1200):
    base = Image.fromarray(image, "RGB")
    scale = min(1.0, float(max_side) / max(base.size))
    if scale < 1.0:
        base = base.resize(
            (max(1, round(base.width * scale)), max(1, round(base.height * scale))),
            Image.Resampling.LANCZOS,
        )
    header_h = 34
    panel = Image.new("RGB", (base.width, base.height + header_h), (25, 25, 30))
    panel.paste(base, (0, header_h))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    draw.text((10, 10), title, fill=(245, 245, 245), font=font)
    finite_indices = [
        index for index, value in enumerate(values) if math.isfinite(float(value))
    ]
    label_indices = set(sorted(
        finite_indices,
        key=lambda index: abs(float(values[index])),
        reverse=True,
    )[:30])
    for index in finite_indices:
        points = np.asarray(detections[index].get("points"), dtype=np.float64)
        if points.shape != (4, 2):
            continue
        polygon = [
            (float(x) * scale, float(y) * scale + header_h) for x, y in points
        ]
        value = float(values[index])
        color = _delta_color(value) if deltas else _height_color(value)
        draw.line(polygon + [polygon[0]], fill=color, width=2)
        if index in label_indices and (not deltas or abs(value) >= 2.0):
            cx = float(detections[index]["center"][0]) * scale
            cy = float(detections[index]["center"][1]) * scale + header_h
            label = f"{value:+.1f}" if deltas else f"{value:.1f}"
            draw.text((cx + 2, cy + 2), label, fill=color, font=font,
                      stroke_width=1, stroke_fill=(0, 0, 0))
    return panel


def _save_preview(path, image, detections, old_final, new_final, max_side):
    delta = np.asarray(new_final) - np.asarray(old_final)
    panels = [
        _annotated_panel(image, detections, old_final, "Old final height (m)",
                         max_side=max_side),
        _annotated_panel(image, detections, new_final, "New final height (m)",
                         max_side=max_side),
        _annotated_panel(image, detections, delta, "New - old (m)", deltas=True,
                         max_side=max_side),
    ]
    canvas = Image.new(
        "RGB",
        (sum(panel.width for panel in panels), max(panel.height for panel in panels)),
        (20, 20, 24),
    )
    offset = 0
    for panel in panels:
        canvas.paste(panel, (offset, 0))
        offset += panel.width
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _number(value):
    value = float(value)
    return round(value, 6) if math.isfinite(value) else ""


def _concatenate(chunks):
    if chunks:
        return np.concatenate(chunks)
    return np.asarray([], dtype=np.float64)


def _write_detection_rows(writer, source, detections, old, new):
    old_raw, old_final, old_floored, old_capped = old
    new_raw, new_final, new_floored, new_capped = new
    for index, detection in enumerate(detections):
        writer.writerow({
            "source": source,
            "detection_index": index,
            "center_x": _number(detection["center"][0]),
            "center_y": _number(detection["center"][1]),
            "confidence": _number(detection.get("confidence", math.nan)),
            "placement_class": int(detection.get("placement_class", -1)),
            "area_m2": _number(detection.get("area_m2", math.nan)),
            "long_side_m": _number(detection.get("length_m", math.nan)),
            "old_raw_m": _number(old_raw[index]),
            "old_final_m": _number(old_final[index]),
            "new_raw_m": _number(new_raw[index]),
            "new_final_m": _number(new_final[index]),
            "raw_delta_m": _number(new_raw[index] - old_raw[index]),
            "final_delta_m": _number(new_final[index] - old_final[index]),
            "old_floored": bool(old_floored[index]),
            "old_capped": bool(old_capped[index]),
            "new_floored": bool(new_floored[index]),
            "new_capped": bool(new_capped[index]),
        })


def _model_metadata(path, model, load_seconds):
    return {
        "checkpoint": str(Path(path).resolve()),
        "arch": model._sfr_heightnet_arch,
        "crop_px": int(model._sfr_crop_px),
        "scalar_layout": model._sfr_scalar_layout,
        "head_layout": model._sfr_heightnet_head_layout,
        "load_seconds": float(load_seconds),
    }


def _write_report(path, summary):
    old = summary["models"]["old"]["final"]
    new = summary["models"]["new"]["final"]
    difference = summary["differences"]["final"]
    lines = [
        "# HeightNet A/B comparison",
        "",
        f"DDS inputs: {summary['inputs']}  ",
        f"Shared YOLO detections: {difference['count']}",
        "",
        "| Model | Architecture | Crop | Median final | P90 final | Throughput | Floors | Caps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, stats in (("Old", old), ("New", new)):
        metadata = summary["models"][label.lower()]["metadata"]
        lines.append(
            f"| {label} | {metadata['arch']} | {metadata['crop_px']} px | "
            f"{stats['median_m']:.2f} m | {stats['p90_m']:.2f} m | "
            f"{stats['throughput_buildings_per_s']:.1f} bld/s | "
            f"{stats['floor_count']} | {stats['cap_count']} |"
        )
    lines.extend([
        "",
        "## Difference (new - old)",
        "",
        f"- Mean: {difference['mean_delta_m']:.2f} m",
        f"- Median: {difference['median_delta_m']:.2f} m",
        f"- Median absolute: {difference['median_abs_delta_m']:.2f} m",
        f"- 90th percentile absolute: {difference['p90_abs_delta_m']:.2f} m",
        f"- New taller: {difference['new_taller_count']:,}",
        f"- Old taller: {difference['old_taller_count']:,}",
        "",
        "The configured/default checkpoint was not changed.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="DDS file(s), glob(s), or texture directories")
    parser.add_argument("--old-checkpoint", default=HEIGHT.DEFAULT_HEIGHT_CHECKPOINT)
    parser.add_argument("--new-checkpoint", default=r"I:\building-models\heightnet.pt")
    parser.add_argument("--yolo-checkpoint", default=BLD.DEFAULT_YOLO_OBB_CHECKPOINT)
    parser.add_argument("--output-dir", default="tmp/height_model_ab/comparison")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height-batch", type=int, default=HEIGHT.DEFAULT_HEIGHT_BATCH)
    parser.add_argument("--yolo-batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=BLD.DEFAULT_YOLO_OBB_IMGSZ)
    parser.add_argument("--stride", type=int, default=BLD.DEFAULT_YOLO_OBB_STRIDE)
    parser.add_argument("--conf", type=float, default=BLD.DEFAULT_YOLO_OBB_CONF)
    parser.add_argument("--iou", type=float, default=BLD.DEFAULT_YOLO_OBB_IOU)
    parser.add_argument("--max-det", type=int, default=BLD.DEFAULT_YOLO_OBB_MAX_DET)
    parser.add_argument("--preview-count", type=int, default=16)
    parser.add_argument("--preview-max-side", type=int, default=1200)
    parser.add_argument("--limit", type=int, default=0, help="Limit DDS count for a quick smoke run")
    parser.add_argument("--skip", type=int, default=0, help="Skip sorted DDS inputs when resuming a run")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    inputs = _resolve_inputs(args.inputs, args.limit, args.skip)
    output_dir = Path(args.output_dir).resolve()
    _assert_safe_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    device = args.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {device!r} was requested, but this Python environment "
            "does not have CUDA-enabled PyTorch. Choose --device cpu explicitly "
            "or run the comparison with the deployed CUDA environment."
        )
    load_started = time.perf_counter()
    old_model = HEIGHT.load_height_model(args.old_checkpoint, device=device)
    old_load_seconds = time.perf_counter() - load_started
    load_started = time.perf_counter()
    new_model = HEIGHT.load_height_model(args.new_checkpoint, device=device)
    new_load_seconds = time.perf_counter() - load_started
    yolo_model = BLD._load_yolo_obb_model(args.yolo_checkpoint)

    old_raw_chunks = []
    old_final_chunks = []
    new_raw_chunks = []
    new_final_chunks = []
    old_floor_count = old_cap_count = 0
    new_floor_count = new_cap_count = 0
    old_seconds = new_seconds = 0.0
    old_peak_delta = old_peak_total = 0
    new_peak_delta = new_peak_total = 0
    image_summaries = []
    tile_chunks = defaultdict(lambda: {
        "old_raw": [], "old_final": [], "new_raw": [], "new_final": [],
    })
    preview_heap = []
    preview_count = max(0, int(args.preview_count))

    csv_path = output_dir / "detections.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for source_index, source_path in enumerate(inputs, 1):
            print(f"[{source_index}/{len(inputs)}] {source_path}", flush=True)
            image = DEBUG._load_rgb_image(source_path)
            m_per_px = _image_m_per_px(source_path, image)
            detect_started = time.perf_counter()
            detections = BLD._run_yolo_obb_inference(
                yolo_model,
                image,
                imgsz=args.imgsz,
                stride=args.stride,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                device=device,
                m_per_px=m_per_px,
                batch_size=args.yolo_batch,
            )
            _cuda_sync(device)
            detect_seconds = time.perf_counter() - detect_started

            old_raw, elapsed, peak_delta, peak_total = _predict_timed(
                old_model, image, detections, m_per_px, args.height_batch, device
            )
            old_seconds += elapsed
            old_peak_delta = max(old_peak_delta, peak_delta)
            old_peak_total = max(old_peak_total, peak_total)
            old_final, old_floored, old_capped = finalize_heights(
                detections, old_raw
            )

            new_raw, elapsed, peak_delta, peak_total = _predict_timed(
                new_model, image, detections, m_per_px, args.height_batch, device
            )
            new_seconds += elapsed
            new_peak_delta = max(new_peak_delta, peak_delta)
            new_peak_total = max(new_peak_total, peak_total)
            new_final, new_floored, new_capped = finalize_heights(
                detections, new_raw
            )

            old_floor_count += int(np.count_nonzero(old_floored))
            old_cap_count += int(np.count_nonzero(old_capped))
            new_floor_count += int(np.count_nonzero(new_floored))
            new_cap_count += int(np.count_nonzero(new_capped))
            old_raw_chunks.append(old_raw)
            old_final_chunks.append(old_final)
            new_raw_chunks.append(new_raw)
            new_final_chunks.append(new_final)
            tile = _tile_name(source_path)
            tile_chunks[tile]["old_raw"].append(old_raw)
            tile_chunks[tile]["old_final"].append(old_final)
            tile_chunks[tile]["new_raw"].append(new_raw)
            tile_chunks[tile]["new_final"].append(new_final)

            source_key = _source_key(source_path)
            _write_detection_rows(
                writer,
                source_key,
                detections,
                (old_raw, old_final, old_floored, old_capped),
                (new_raw, new_final, new_floored, new_capped),
            )
            difference = summarize_differences(old_final, new_final)
            image_summaries.append({
                "source": str(source_path),
                "key": source_key,
                "tile": tile,
                "m_per_px": float(m_per_px),
                "detections": int(len(detections)),
                "yolo_seconds": float(detect_seconds),
                "final_difference": difference,
            })
            score = float(difference["p90_abs_delta_m"])
            if preview_count and math.isfinite(score):
                item = (
                    score,
                    source_index,
                    str(source_path),
                    detections,
                    old_final,
                    new_final,
                )
                if len(preview_heap) < preview_count:
                    heapq.heappush(preview_heap, item)
                elif score > preview_heap[0][0]:
                    heapq.heapreplace(preview_heap, item)

    old_raw_all = _concatenate(old_raw_chunks)
    old_final_all = _concatenate(old_final_chunks)
    new_raw_all = _concatenate(new_raw_chunks)
    new_final_all = _concatenate(new_final_chunks)
    tile_summaries = {}
    for tile, chunks in sorted(tile_chunks.items()):
        tile_summaries[tile] = {
            "raw_difference": summarize_differences(
                _concatenate(chunks["old_raw"]),
                _concatenate(chunks["new_raw"]),
            ),
            "final_difference": summarize_differences(
                _concatenate(chunks["old_final"]),
                _concatenate(chunks["new_final"]),
            ),
        }

    old_metadata = _model_metadata(args.old_checkpoint, old_model, old_load_seconds)
    new_metadata = _model_metadata(args.new_checkpoint, new_model, new_load_seconds)
    old_metadata.update({
        "peak_inference_delta_mib": old_peak_delta / 2**20,
        "peak_total_allocated_mib": old_peak_total / 2**20,
    })
    new_metadata.update({
        "peak_inference_delta_mib": new_peak_delta / 2**20,
        "peak_total_allocated_mib": new_peak_total / 2**20,
    })
    summary = {
        "inputs": len(inputs),
        "device": str(device),
        "yolo_checkpoint": str(Path(args.yolo_checkpoint).resolve()),
        "models": {
            "old": {
                "metadata": old_metadata,
                "raw": summarize_predictions(
                    old_raw_all,
                    inference_seconds=old_seconds,
                    floor_count=old_floor_count,
                    cap_count=old_cap_count,
                ),
                "final": summarize_predictions(
                    old_final_all,
                    inference_seconds=old_seconds,
                    floor_count=old_floor_count,
                    cap_count=old_cap_count,
                ),
            },
            "new": {
                "metadata": new_metadata,
                "raw": summarize_predictions(
                    new_raw_all,
                    inference_seconds=new_seconds,
                    floor_count=new_floor_count,
                    cap_count=new_cap_count,
                ),
                "final": summarize_predictions(
                    new_final_all,
                    inference_seconds=new_seconds,
                    floor_count=new_floor_count,
                    cap_count=new_cap_count,
                ),
            },
        },
        "differences": {
            "raw": summarize_differences(old_raw_all, new_raw_all),
            "final": summarize_differences(old_final_all, new_final_all),
        },
        "tiles": tile_summaries,
        "images": image_summaries,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(output_dir / "report.md", summary)

    preview_dir = output_dir / "previews"
    for _score, _index, source, detections, old_final, new_final in sorted(
        preview_heap, reverse=True
    ):
        source_path = Path(source)
        image = DEBUG._load_rgb_image(source_path)
        _save_preview(
            preview_dir / f"{_source_key(source_path)}.png",
            image,
            detections,
            old_final,
            new_final,
            args.preview_max_side,
        )

    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Report: {output_dir / 'report.md'}")
    print("Configured/default checkpoint unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
