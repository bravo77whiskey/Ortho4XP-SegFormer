"""Merge completed HeightNet A/B CSV shards into one validated report.

This is primarily useful after resuming a long comparison with ``--skip``.
Input shards must cover disjoint source textures and use the CSV schema emitted
by ``compare_sfr_height_models.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import compare_sfr_height_models as compare


VALUE_COLUMNS = ("old_raw_m", "old_final_m", "new_raw_m", "new_final_m")
FLAG_COLUMNS = ("old_floored", "old_capped", "new_floored", "new_capped")
TILE_RE = re.compile(r"^(zOrtho4XP_[+-]\d{2}[+-]\d{3})_", re.IGNORECASE)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_shards", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--timing-summary",
        help="Completed shard summary whose CUDA timing and model metadata are reusable.",
    )
    parser.add_argument(
        "--input-count",
        type=int,
        help="Total DDS inputs, including sources that emitted no detection rows.",
    )
    return parser.parse_args(argv)


def _tile_from_source(source):
    match = TILE_RE.match(str(source))
    if not match:
        raise ValueError(f"Unrecognized comparison source key: {source!r}")
    return match.group(1)


def _merge_csv_files(paths, destination):
    expected_header = None
    with destination.open("wb") as output:
        for index, path in enumerate(paths):
            with path.open("rb") as source:
                header = source.readline()
                if expected_header is None:
                    expected_header = header
                    output.write(header)
                elif header != expected_header:
                    raise ValueError(f"CSV header mismatch in {path}")
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)


def _summarize_shards(paths):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Merging large comparison shards requires pandas in this Python environment"
        ) from exc

    use_columns = ("source", *VALUE_COLUMNS, *FLAG_COLUMNS)
    arrays = defaultdict(lambda: defaultdict(list))
    flag_counts = defaultdict(Counter)
    shard_sources = []
    source_peak_delta = {}

    for path in paths:
        sources = set()
        for chunk in pd.read_csv(path, usecols=use_columns, chunksize=250_000):
            chunk["tile"] = chunk["source"].map(_tile_from_source)
            sources.update(chunk["source"].unique().tolist())
            absolute_delta = (chunk["new_final_m"] - chunk["old_final_m"]).abs()
            for source, value in absolute_delta.groupby(chunk["source"]).max().items():
                source_peak_delta[source] = max(
                    float(value), source_peak_delta.get(source, -math.inf)
                )
            for tile, group in chunk.groupby("tile", sort=False):
                for column in VALUE_COLUMNS:
                    arrays[tile][column].append(
                        group[column].to_numpy(dtype=np.float64, copy=True)
                    )
                for column in FLAG_COLUMNS:
                    flag_counts[tile][column] += int(group[column].sum())
        for previous in shard_sources:
            overlap = sources.intersection(previous)
            if overlap:
                example = sorted(overlap)[0]
                raise ValueError(
                    f"Comparison shards overlap on {len(overlap)} sources; example: {example}"
                )
        shard_sources.append(sources)

    tile_values = {
        tile: {
            column: compare._concatenate(chunks)
            for column, chunks in columns.items()
        }
        for tile, columns in arrays.items()
    }
    all_values = {
        column: compare._concatenate([
            values[column] for values in tile_values.values()
        ])
        for column in VALUE_COLUMNS
    }
    total_flags = Counter()
    for counts in flag_counts.values():
        total_flags.update(counts)
    detected_source_count = sum(len(sources) for sources in shard_sources)
    return (tile_values, flag_counts, all_values, total_flags,
            detected_source_count, source_peak_delta)


def _prediction_summary(values, seconds, flags, prefix):
    return compare.summarize_predictions(
        values,
        inference_seconds=seconds,
        floor_count=flags[f"{prefix}_floored"],
        cap_count=flags[f"{prefix}_capped"],
    )


def _append_regional_report(path, summary):
    lines = [
        "",
        "## Regional final-height distributions",
        "",
        "| Tile | Buildings | Old median | New median | Median delta | P90 abs delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for tile, stats in sorted(summary["tiles"].items()):
        old = stats["old_final"]
        new = stats["new_final"]
        delta = stats["final_difference"]
        lines.append(
            f"| {tile} | {delta['count']:,} | {old['median_m']:.2f} m | "
            f"{new['median_m']:.2f} m | {delta['median_delta_m']:.2f} m | "
            f"{delta['p90_abs_delta_m']:.2f} m |"
        )
    if summary.get("timing_scope"):
        lines.extend([
            "",
            f"Timing/VRAM scope: {summary['timing_scope']}.",
        ])
    path.write_text(path.read_text(encoding="utf-8") + "\n".join(lines) + "\n",
                    encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    paths = [Path(value).resolve() for value in args.csv_shards]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir = Path(args.output_dir).resolve()
    compare._assert_safe_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timing = None
    if args.timing_summary:
        timing = json.loads(Path(args.timing_summary).read_text(encoding="utf-8"))
    old_seconds = float(timing["models"]["old"]["final"]["inference_seconds"]) if timing else 0.0
    new_seconds = float(timing["models"]["new"]["final"]["inference_seconds"]) if timing else 0.0

    (tile_values, tile_flags, all_values, total_flags, detected_source_count,
     source_peak_delta) = _summarize_shards(paths)
    source_count = int(args.input_count or detected_source_count)
    if source_count < detected_source_count:
        raise ValueError(
            f"--input-count {source_count} is smaller than the "
            f"{detected_source_count} sources represented in the CSV shards"
        )
    merged_csv = output_dir / "detections.csv"
    _merge_csv_files(paths, merged_csv)

    tiles = {}
    for tile, values in sorted(tile_values.items()):
        flags = tile_flags[tile]
        tiles[tile] = {
            "old_raw": _prediction_summary(values["old_raw_m"], 0.0, flags, "old"),
            "old_final": _prediction_summary(values["old_final_m"], 0.0, flags, "old"),
            "new_raw": _prediction_summary(values["new_raw_m"], 0.0, flags, "new"),
            "new_final": _prediction_summary(values["new_final_m"], 0.0, flags, "new"),
            "raw_difference": compare.summarize_differences(
                values["old_raw_m"], values["new_raw_m"]
            ),
            "final_difference": compare.summarize_differences(
                values["old_final_m"], values["new_final_m"]
            ),
        }

    metadata_old = timing["models"]["old"]["metadata"] if timing else {}
    metadata_new = timing["models"]["new"]["metadata"] if timing else {}
    summary = {
        "inputs": source_count,
        "inputs_with_detections": detected_source_count,
        "device": timing.get("device") if timing else None,
        "yolo_checkpoint": timing.get("yolo_checkpoint") if timing else None,
        "source_csv_shards": [str(path) for path in paths],
        "timing_scope": (
            f"CUDA-completed resumed shard only ({timing['inputs']} of {source_count} inputs)"
            if timing else None
        ),
        "models": {
            "old": {
                "metadata": metadata_old,
                "raw": _prediction_summary(all_values["old_raw_m"], old_seconds,
                                           total_flags, "old"),
                "final": _prediction_summary(all_values["old_final_m"], old_seconds,
                                             total_flags, "old"),
            },
            "new": {
                "metadata": metadata_new,
                "raw": _prediction_summary(all_values["new_raw_m"], new_seconds,
                                           total_flags, "new"),
                "final": _prediction_summary(all_values["new_final_m"], new_seconds,
                                             total_flags, "new"),
            },
        },
        "differences": {
            "raw": compare.summarize_differences(
                all_values["old_raw_m"], all_values["new_raw_m"]
            ),
            "final": compare.summarize_differences(
                all_values["old_final_m"], all_values["new_final_m"]
            ),
        },
        "tiles": tiles,
        "preview_source_candidates": [
            source for source, _value in sorted(
                source_peak_delta.items(), key=lambda item: item[1], reverse=True
            )[:16]
        ],
        "preview_source_candidates_by_tile": {
            tile: [
                source for source, _value in sorted(
                    (
                        (source, value)
                        for source, value in source_peak_delta.items()
                        if _tile_from_source(source) == tile
                    ),
                    key=lambda item: item[1],
                    reverse=True,
                )[:4]
            ]
            for tile in sorted(tile_values)
        },
    }
    if timing:
        for model_name in ("old", "new"):
            for layout in ("raw", "final"):
                measured = timing["models"][model_name][layout]
                merged = summary["models"][model_name][layout]
                merged["inference_seconds"] = float(measured["inference_seconds"])
                merged["throughput_buildings_per_s"] = float(
                    measured["throughput_buildings_per_s"]
                )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path = output_dir / "report.md"
    compare._write_report(report_path, summary)
    _append_regional_report(report_path, summary)
    print(f"Merged CSV: {merged_csv}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
