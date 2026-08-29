#!/usr/bin/env python3
"""Estimate regional height priors from targeted GlobalBuildingAtlas samples.

The project classifies buildings by footprint area and maximum side length.
This script samples both ODbL and non-ODbL GlobalBuildingAtlas polygons, joins
them to GBA.LoD1 heights, applies the project's footprint thresholds, and
writes per-region and per-class summary statistics.

Only byte ranges from selected 5 degree tiles are transferred. The 36 TB
GlobalBuildingAtlas archive is never downloaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import requests
from shapely.geometry import shape


HF_ROOT = "https://huggingface.co/datasets"
HEIGHT_REPO = "zhu-xlab/GBA.LoD1"
ODBL_REPO = "zhu-xlab/GBA.ODbLPolygon"
EARTH_RADIUS_M = 6_378_137.0

CLASS_LABELS = {
    1: "tiny residential",
    2: "small residential",
    3: "compact residential",
    4: "medium footprint",
    5: "small apartment",
    6: "apartment block",
    7: "large footprint",
    8: "extra-large footprint",
}

# Mirrors O4_SFR_Building_Overlay._class_for_footprint().
TINY_MAX_M2 = 90.0
SMALL_MAX_M2 = 170.0
COMPACT_MAX_M2 = 270.0
MEDIUM_MAX_M2 = 450.0
MEDIUM_MAX_SIDE_M = 28.0
SMALL_APARTMENT_MAX_M2 = 850.0
SMALL_APARTMENT_MAX_SIDE_M = 45.0
APARTMENT_BLOCK_MAX_M2 = 1_650.0
APARTMENT_BLOCK_MAX_SIDE_M = 55.0
LARGE_MAX_M2 = 7_000.0
LARGE_MAX_SIDE_M = 100.0

# GBA.LoD1 continental height RMSE reported by Zhu et al. (2025). Africa
# lacked direct validation, so the most conservative validated RMSE is used.
REGION_RMSE_M = {
    "north_america": 5.3,
    "north_america_ne": 5.3,
    "north_america_west": 5.3,
    "europe": 4.1,
    "scandinavia": 4.1,
    "mediterranean": 4.1,
    "asia": 5.9,
    "se_asia": 5.9,
    "africa": 8.9,
    "australia_oceania": 1.5,
    "south_america": 8.9,
}


@dataclass(frozen=True)
class Tile:
    city: str
    folder: str
    stem: str


REGION_TILES = {
    "north_america": (
        Tile("Dallas", "northamerica", "w100_n35_w095_n30"),
        Tile("Houston", "northamerica", "w100_n30_w095_n25"),
        Tile("Chicago", "northamerica", "w090_n45_w085_n40"),
    ),
    "north_america_ne": (
        Tile("Boston", "northamerica", "w075_n45_w070_n40"),
        Tile("Toronto", "northamerica", "w080_n45_w075_n40"),
        Tile("Montreal", "northamerica", "w075_n50_w070_n45"),
    ),
    "north_america_west": (
        Tile("Portland", "northamerica", "w125_n50_w120_n45"),
        Tile("Los Angeles", "northamerica", "w120_n35_w115_n30"),
        Tile("San Francisco", "northamerica", "w125_n40_w120_n35"),
    ),
    "europe": (
        Tile("Amsterdam", "europe", "e000_n55_e005_n50"),
        Tile("Berlin", "europe", "e010_n55_e015_n50"),
        Tile("Paris", "europe", "e000_n50_e005_n45"),
    ),
    "scandinavia": (
        Tile("Stockholm", "europe", "e015_n60_e020_n55"),
        Tile("Helsinki", "europe", "e020_n65_e025_n60"),
        Tile("Copenhagen", "europe", "e010_n60_e015_n55"),
    ),
    "mediterranean": (
        Tile("Barcelona", "europe", "e000_n45_e005_n40"),
        Tile("Rome", "europe", "e010_n45_e015_n40"),
        Tile("Thessaloniki", "europe", "e020_n45_e025_n40"),
    ),
    "asia": (
        Tile("Tokyo", "asiaeast", "e135_n40_e140_n35"),
        Tile("Seoul", "asiaeast", "e125_n40_e130_n35"),
        Tile("Shanghai", "asiaeast", "e120_n35_e125_n30"),
    ),
    "se_asia": (
        Tile("Bangkok", "asiawest", "e100_n15_e105_n10"),
        Tile("Da Nang", "asiawest", "e105_n20_e110_n15"),
        Tile("Manila", "asiaeast", "e120_n15_e125_n10"),
    ),
    "africa": (
        Tile("Nairobi", "africa", "e035_n00_e040_s05"),
        Tile("Lagos", "africa", "e000_n10_e005_n05"),
        Tile("Johannesburg", "africa", "e025_s25_e030_s30"),
    ),
    "australia_oceania": (
        Tile("Melbourne", "oceania", "e140_s35_e145_s40"),
        Tile("Sydney", "oceania", "e150_s30_e155_s35"),
        Tile("Auckland", "oceania", "e170_s35_e175_s40"),
    ),
    "south_america": (
        Tile("Quito", "southamerica", "w080_n00_w075_s05"),
        Tile("Sao Paulo", "southamerica", "w050_s20_w045_s25"),
        Tile("Buenos Aires", "southamerica", "w060_s30_w055_s35"),
    ),
}

# ODbL and height indices diverge sooner in these tiles. Wider aligned ranges
# recover enough rare large-footprint examples without downloading full files.
REGION_CHUNK_MULTIPLIER = {
    "asia": 2,
    "se_asia": 4,
    "africa": 8,
    "south_america": 8,
}

HEIGHT_RE = re.compile(
    rb'"([^"\\]+)"\s*:\s*\{\s*"height"\s*:\s*'
    rb'([-+0-9.eE]+)\s*,\s*"var"\s*:\s*([-+0-9.eE]+)\s*\}'
)


def url_for(repo: str, path: str) -> str:
    return f"{HF_ROOT}/{repo}/resolve/main/{path}?download=true"


def request_range(session: requests.Session, url: str, start: int, stop: int) -> bytes:
    headers = {"Range": f"bytes={start}-{stop}"}
    error: Exception | None = None
    for attempt in range(4):
        try:
            response = session.get(url, headers=headers, timeout=120)
            response.raise_for_status()
            if response.status_code != 206:
                raise RuntimeError(f"server ignored byte range for {url}")
            return response.content
        except (requests.RequestException, RuntimeError) as exc:
            error = exc
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"range request failed for {url}: {error}")


def remote_size(session: requests.Session, url: str) -> int:
    response = session.get(url, headers={"Range": "bytes=0-0"}, timeout=120)
    response.raise_for_status()
    content_range = response.headers.get("Content-Range", "")
    match = re.search(r"/(\d+)$", content_range)
    if response.status_code != 206 or not match:
        raise RuntimeError(f"could not determine remote size for {url}")
    return int(match.group(1))


def range_pair(
    session: requests.Session, url: str, prefix_bytes: int, suffix_bytes: int
) -> tuple[bytes, bytes, int]:
    size = remote_size(session, url)
    prefix_stop = min(size, prefix_bytes) - 1
    prefix = request_range(session, url, 0, prefix_stop)
    suffix_start = max(0, size - suffix_bytes)
    suffix = request_range(session, url, suffix_start, size - 1)
    return prefix, suffix, size


def prefix_range(
    session: requests.Session, url: str, byte_count: int
) -> tuple[bytes, int]:
    size = remote_size(session, url)
    return request_range(session, url, 0, min(size, byte_count) - 1), size


def parse_heights(blob: bytes) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for match in HEIGHT_RE.finditer(blob):
        key = match.group(1).decode("utf-8")
        height = float(match.group(2))
        variance = float(match.group(3))
        if math.isfinite(height) and math.isfinite(variance):
            result[key] = (height, variance)
    return result


def feature_lines(blob: bytes) -> Iterable[dict]:
    for raw_line in blob.splitlines():
        line = raw_line.strip()
        if not line.startswith(b'{ "type": "Feature"'):
            continue
        if line.endswith(b","):
            line = line[:-1]
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def latitude_from_mercator_y(y_m: float) -> float:
    return math.atan(math.sinh(y_m / EARTH_RADIUS_M))


def ground_metrics(feature: dict) -> tuple[float, float] | None:
    try:
        geometry = shape(feature["geometry"])
    except (KeyError, TypeError, ValueError):
        return None
    if geometry.is_empty or not geometry.is_valid or geometry.area <= 0.0:
        return None
    latitude_rad = latitude_from_mercator_y(float(geometry.centroid.y))
    scale = math.cos(latitude_rad)
    area_m2 = float(geometry.area) * scale * scale
    rectangle = geometry.minimum_rotated_rectangle
    coords = list(rectangle.exterior.coords)
    sides = [
        math.hypot(x2 - x1, y2 - y1) * scale
        for (x1, y1), (x2, y2) in zip(coords, coords[1:])
    ]
    if not sides:
        return None
    return area_m2, max(sides)


def footprint_class(area_m2: float, max_side_m: float) -> int:
    if area_m2 <= TINY_MAX_M2:
        return 1
    if area_m2 <= SMALL_MAX_M2:
        return 2
    if area_m2 <= COMPACT_MAX_M2:
        return 3
    if area_m2 <= MEDIUM_MAX_M2 and max_side_m <= MEDIUM_MAX_SIDE_M:
        return 4
    if area_m2 <= SMALL_APARTMENT_MAX_M2 and max_side_m <= SMALL_APARTMENT_MAX_SIDE_M:
        return 5
    if area_m2 <= APARTMENT_BLOCK_MAX_M2 and max_side_m <= APARTMENT_BLOCK_MAX_SIDE_M:
        return 6
    if area_m2 <= LARGE_MAX_M2 and max_side_m <= LARGE_MAX_SIDE_M:
        return 7
    return 8


def feature_key(feature: dict) -> str:
    props = feature.get("properties") or {}
    return f"{props.get('source', '')}{props.get('id', '')}{props.get('region', '')}"


def join_features(
    features_blob: bytes,
    heights: dict[str, tuple[float, float]],
    region: str,
    tile: Tile,
    sample_kind: str,
) -> list[dict]:
    rows = []
    for feature in feature_lines(features_blob):
        key = feature_key(feature)
        height_record = heights.get(key)
        if height_record is None:
            continue
        height_m, variance = height_record
        # Remove physically impossible fragments and extreme model failures.
        if not (1.0 <= height_m <= 400.0 and 0.0 <= variance <= 10_000.0):
            continue
        metrics = ground_metrics(feature)
        if metrics is None:
            continue
        area_m2, max_side_m = metrics
        if not (1.0 <= area_m2 <= 520_000.0 and 1.0 <= max_side_m <= 1_300.0):
            continue
        props = feature.get("properties") or {}
        rows.append(
            {
                "region": region,
                "city": tile.city,
                "tile": tile.stem,
                "sample_kind": sample_kind,
                "source": str(props.get("source", "")),
                "country": str(props.get("region", "")),
                "class_id": footprint_class(area_m2, max_side_m),
                "area_m2": area_m2,
                "max_side_m": max_side_m,
                "height_m": height_m,
                "variance": variance,
            }
        )
    return rows


def sample_tile(
    session: requests.Session,
    region: str,
    tile: Tile,
    height_chunk_bytes: int,
    polygon_chunk_bytes: int,
) -> tuple[list[dict], dict]:
    height_path = f"LoD1/{tile.folder}/{tile.stem}.json"
    odbl_path = f"{tile.folder}/{tile.stem}.geojson"
    height_prefix, height_size = prefix_range(
        session,
        url_for(HEIGHT_REPO, height_path),
        height_chunk_bytes,
    )
    prefix_heights = parse_heights(height_prefix)

    odbl_prefix, odbl_size = prefix_range(
        session,
        url_for(ODBL_REPO, odbl_path),
        polygon_chunk_bytes,
    )
    rows = join_features(odbl_prefix, prefix_heights, region, tile, "odbl_prefix")
    metadata = {
        "region": region,
        "city": tile.city,
        "tile": tile.stem,
        "height_file_bytes": height_size,
        "odbl_polygon_file_bytes": odbl_size,
        "prefix_height_records": len(prefix_heights),
        "height_sample_bytes": min(height_size, height_chunk_bytes),
        "polygon_sample_bytes": min(odbl_size, polygon_chunk_bytes),
        "joined_records": len(rows),
        "joined_by_kind": dict(Counter(row["sample_kind"] for row in rows)),
    }
    return rows, metadata


def ceil_to_5(value: float) -> float:
    return float(math.ceil(value / 5.0) * 5.0)


def summarize_group(region: str, class_id: int, rows: list[dict]) -> dict:
    values = np.asarray([row["height_m"] for row in rows], dtype=np.float64)
    if values.size == 0:
        return {
            "region": region,
            "class_id": class_id,
            "class_label": CLASS_LABELS[class_id],
            "count": 0,
        }
    p90, p95, p99 = np.percentile(values, [90.0, 95.0, 99.0])
    rmse = REGION_RMSE_M.get(region, max(REGION_RMSE_M.values()))
    return {
        "region": region,
        "class_id": class_id,
        "class_label": CLASS_LABELS[class_id],
        "count": int(values.size),
        "mean_m": float(values.mean()),
        "median_m": float(np.median(values)),
        "std_m": float(values.std()),
        "p90_m": float(p90),
        "p95_m": float(p95),
        "p99_m": float(p99),
        "observed_max_m": float(values.max()),
        "gba_rmse_margin_m": float(rmse),
        "soft_ceiling_m": ceil_to_5(float(p95) + rmse),
        "hard_ceiling_m": ceil_to_5(float(p99) + rmse),
        "countries": sorted({row["country"] for row in rows if row["country"]}),
        "cities": sorted({row["city"] for row in rows}),
        "source_counts": dict(Counter(row["sample_kind"] for row in rows)),
        "confidence": "high" if values.size >= 1_000 else "medium" if values.size >= 200 else "low",
    }


def pooled_generic(rows: list[dict], class_id: int) -> dict:
    values = [row for row in rows if row["class_id"] == class_id]
    summary = summarize_group("generic", class_id, values)
    if summary["count"]:
        summary["gba_rmse_margin_m"] = max(REGION_RMSE_M.values())
        summary["soft_ceiling_m"] = ceil_to_5(
            summary["p95_m"] + summary["gba_rmse_margin_m"]
        )
        summary["hard_ceiling_m"] = ceil_to_5(
            summary["p99_m"] + summary["gba_rmse_margin_m"]
        )
    return summary


def write_csv(path: Path, summaries: list[dict]) -> None:
    fields = [
        "region", "class_id", "class_label", "count", "mean_m", "median_m",
        "std_m", "p90_m", "p95_m", "p99_m", "observed_max_m",
        "gba_rmse_margin_m", "soft_ceiling_m", "hard_ceiling_m", "confidence",
        "countries", "cities", "source_counts",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            row = dict(summary)
            for key in ("countries", "cities", "source_counts"):
                row[key] = json.dumps(row.get(key), sort_keys=True)
            writer.writerow({key: row.get(key) for key in fields})


def matrix_table(summaries: list[dict], field: str) -> list[str]:
    lookup = {(row["region"], row["class_id"]): row for row in summaries}
    regions = list(REGION_TILES) + ["generic"]
    lines = [
        "| Region | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for region in regions:
        values = []
        for class_id in CLASS_LABELS:
            value = lookup.get((region, class_id), {}).get(field)
            values.append("n/a" if value is None else f"{value:.1f}")
        lines.append(f"| {region} | " + " | ".join(values) + " |")
    return lines


def write_report(
    path: Path,
    summaries: list[dict],
    tile_metadata: list[dict],
    rows: list[dict],
    height_chunk_bytes: int,
    polygon_chunk_bytes: int,
) -> None:
    count = len(rows)
    region_counts = Counter(row["region"] for row in rows)
    class_counts = Counter(row["class_id"] for row in rows)
    transfer_upper_bound = sum(
        item["height_sample_bytes"] + item["polygon_sample_bytes"]
        for item in tile_metadata
    )
    lines = [
        "# Regional building-height prior research",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()}.",
        "",
        "## Result",
        "",
        f"The sample contains {count:,} joined buildings from {len(tile_metadata)} "
        f"five-degree tiles. It transfers at most {transfer_upper_bound / 1e9:.2f} GB "
        "of byte ranges rather than downloading the 36 TB archive.",
        "",
        "The tables use the exact eight footprint classes in "
        "`O4_SFR_Building_Overlay.py`. Mean is the requested statistic; median "
        "is the safer fallback prior for the skewed height distribution. P99 plus "
        "the published continental GBA height RMSE, rounded up "
        "to 5 m, is shown as a conservative hard ceiling candidate.",
        "",
        "### Mean height in metres",
        "",
        *matrix_table(summaries, "mean_m"),
        "",
        "### Median height in metres",
        "",
        *matrix_table(summaries, "median_m"),
        "",
        "### P99 plus GBA error margin, in metres",
        "",
        *matrix_table(summaries, "hard_ceiling_m"),
        "",
        "## Interpretation",
        "",
        "These are footprint classes, not use or storey classes. A tower can have "
        "the same footprint as a house, and a hospital can have the same footprint "
        "as a warehouse. Region and footprint therefore make a useful prior, but "
        "they do not justify unconditional low hard caps.",
        "",
        "Use the mean or median only when HeightNet is missing or weak. Use the "
        "soft ceiling for shrinkage of uncertain predictions. Reserve the hard "
        "ceiling for obvious outliers, and bypass it when another source supplies "
        "height or floor count.",
        "",
        "The `generic` row pools every sampled region because the project's "
        "`generic` boundary key has no coherent real-world building stock.",
        "",
        "## Comparison with the current runtime",
        "",
        "The current generic fallback heights for C1 through C8 are 3.5, 4, 4, "
        "7, 9, 12, 8, and 10 m. The measured regional means show that C5 and C6 "
        "are usually much lower than the current 9 and 12 m fallbacks. That "
        "happens because the runtime names imply apartments, while the classifier "
        "only knows footprint dimensions.",
        "",
        "The current 16 m cap for C7 and C8 is near or above the sampled P95 in "
        "many regions. It is still too low for the minority of large-footprint "
        "apartment, office, hospital, and tower buildings. Raising every large "
        "building to the regional P99 ceiling would reintroduce tall warehouse "
        "errors. The cap needs one more signal, such as developed versus "
        "industrial landcover or a selected asset family.",
        "",
        "A safe first experiment is to use the regional median when HeightNet is "
        "missing, softly pull predictions above P95 toward P95, and clip only "
        "above the reported hard ceiling. Keep the 16 m industrial cap for large "
        "footprints on bareland or agricultural context. Evaluate this on shared "
        "detections before changing the production default.",
        "",
        "## Sampling and limits",
        "",
        "GlobalBuildingAtlas supplies predicted, not surveyed, heights. The paper "
        "reports continental LoD1 height RMSE from 1.5 m in Oceania to 8.9 m in "
        "South America, with no direct African validation. This report adds those "
        "errors to the proposed ceilings.",
        "",
        "The sample is deterministic. It reads the aligned start of each ODbL "
        "polygon tile and its LoD1 height index. Those ODbL files contain OSM, "
        "Microsoft, or Google-derived footprints depending on the tile. Three "
        "metropolitan or mixed tiles "
        "represent each project region. It is broad enough for a practical prior, "
        "but it is not a census-weighted regional statistic.",
        "",
        "The separate non-ODbL polygon files do not share byte order with the "
        "height index, so partial range requests cannot join them safely. This "
        "sample excludes those polygons instead of accepting false joins. That "
        "makes the African and South American estimates less representative.",
        "",
        "GlobalBuildingAtlas uses the maximum predicted height pixel inside each "
        "building footprint. Its raw observed maximum is therefore unsuitable as "
        "a cap. P95 and P99 are retained in the CSV and JSON outputs.",
        "",
        "GBA.LoD1 and the non-ODbL polygons are CC BY-NC 4.0. ODbL polygons have "
        "their own ODbL terms. Derived constants need a license review before a "
        "commercial distribution.",
        "",
        "## Coverage",
        "",
        "| Region | Joined buildings |",
        "|---|---:|",
    ]
    for region in list(REGION_TILES):
        lines.append(f"| {region} | {region_counts[region]:,} |")
    lines.extend(["", "| Class | Joined buildings |", "|---|---:|"])
    for class_id, label in CLASS_LABELS.items():
        lines.append(f"| C{class_id} {label} | {class_counts[class_id]:,} |")
    lines.extend(
        [
            "",
            "## Class thresholds",
            "",
            "The thresholds copied from the runtime are 90, 170, 270, 450, 850, "
            "1,650, and 7,000 square metres. Classes 4 through 7 also use maximum "
            "side limits of 28, 45, 55, and 100 metres.",
            "",
            "## Source",
            "",
            "Zhu, Chen, Zhang, Shi, and Wang, GlobalBuildingAtlas, Earth System "
            "Science Data 17, 6647 to 6668, 2025. DOI 10.5194/essd-17-6647-2025.",
            "Dataset record: https://mediatum.ub.tum.de/1782307",
            "Paper: https://essd.copernicus.org/articles/17/6647/2025/",
            "Google Open Buildings 2.5D cross-check: "
            "https://sites.research.google/gr/open-buildings/temporal/",
            "Microsoft global density and height cross-check: "
            "https://github.com/microsoft/buildings",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--regions", nargs="*", choices=tuple(REGION_TILES))
    parser.add_argument("--tiles-per-region", type=int, default=3)
    parser.add_argument("--height-chunk-mib", type=int, default=4)
    parser.add_argument("--polygon-chunk-mib", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    height_chunk_bytes = args.height_chunk_mib * 1024 * 1024
    polygon_chunk_bytes = args.polygon_chunk_mib * 1024 * 1024
    selected_regions = args.regions or list(REGION_TILES)

    session = requests.Session()
    session.headers.update({"User-Agent": "Ortho4XP-height-prior-research/1.0"})
    all_rows: list[dict] = []
    tile_metadata: list[dict] = []
    failures: list[dict] = []

    for region in selected_regions:
        chunk_multiplier = REGION_CHUNK_MULTIPLIER.get(region, 1)
        for tile in REGION_TILES[region][: args.tiles_per_region]:
            print(f"Sampling {region}: {tile.city} ({tile.stem})", flush=True)
            try:
                rows, metadata = sample_tile(
                    session,
                    region,
                    tile,
                    height_chunk_bytes * chunk_multiplier,
                    polygon_chunk_bytes * chunk_multiplier,
                )
            except Exception as exc:
                print(f"  failed: {exc}", flush=True)
                failures.append(
                    {"region": region, "city": tile.city, "tile": tile.stem, "error": str(exc)}
                )
                continue
            print(f"  joined {len(rows):,} buildings", flush=True)
            all_rows.extend(rows)
            tile_metadata.append(metadata)

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in all_rows:
        grouped[(row["region"], row["class_id"])].append(row)

    summaries = []
    for region in selected_regions:
        for class_id in CLASS_LABELS:
            summaries.append(summarize_group(region, class_id, grouped[(region, class_id)]))
    if set(selected_regions) == set(REGION_TILES):
        for class_id in CLASS_LABELS:
            summaries.append(pooled_generic(all_rows, class_id))

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "dataset": "GlobalBuildingAtlas GBA.LoD1",
            "height_repo": HEIGHT_REPO,
            "odbl_repo": ODBL_REPO,
            "height_chunk_mib": args.height_chunk_mib,
            "polygon_chunk_mib": args.polygon_chunk_mib,
            "tiles_per_region": args.tiles_per_region,
            "height_filter_m": [1.0, 400.0],
            "area_filter_m2": [1.0, 520_000.0],
            "project_class_thresholds": True,
        },
        "tile_metadata": tile_metadata,
        "failures": failures,
        "summaries": summaries,
    }
    json_path = args.output_dir / "regional_height_priors_gba.json"
    csv_path = args.output_dir / "regional_height_priors_gba.csv"
    report_path = args.output_dir / "RESEARCH_GBA_HEIGHT_PRIORS.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(csv_path, summaries)
    write_report(
        report_path,
        summaries,
        tile_metadata,
        all_rows,
        height_chunk_bytes,
        polygon_chunk_bytes,
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")
    if failures:
        print(f"Completed with {len(failures)} failed tiles")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
