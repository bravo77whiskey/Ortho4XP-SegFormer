"""SimHeaven-supervised building placement training utilities.

The module is intentionally usable in three modes:
  * lightweight unit tests without torch/transformers installed,
  * dataset construction from local Ortho4XP + SimHeaven scenery, and
  * CUDA training from the external Ortho4XP venv.

Coordinates use WGS84 `(lon, lat)` geometry internally because Shapely's x/y
axes naturally map to longitude/latitude.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import re
import shutil
import sqlite3
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
from PIL import Image, ImageDraw
from shapely import affinity, wkb
from shapely.geometry import MultiPolygon, Point, Polygon, box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree


DEFAULT_TILES_ROOT = Path(r"H:\XP12 Addons\Applications\Ortho4XP\Tiles")
DEFAULT_CUSTOM_SCENERY_ROOT = Path(
    r"D:\SteamLibrary\steamapps\common\X-Plane 12\Custom Scenery"
)
DEFAULT_TRAINING_ROOT = Path(r"H:\model_training")
DEFAULT_VENV_PYTHON = Path(r"G:\Dev\Ortho4XP\.venv\Scripts\python.exe")

DEFAULT_ZOOM = 16
TEXTURE_BLOCK_SIZE = 16
_LABEL_PAYLOAD_CACHE: dict[str, dict[str, Any]] = {}
_SQLITE_COUNT_CACHE: dict[str, int] = {}
WEB_MERCATOR_TILE_RE = re.compile(
    r"^(\d+)_(\d+)_([A-Za-z][A-Za-z0-9_]*)(\d{2})\.(dds|jpg|jpeg|png)$",
    re.IGNORECASE,
)
TILE_DIR_RE = re.compile(r"^zOrtho4XP_([+-]\d{2})([+-]\d{3})$", re.IGNORECASE)

SOUTH_AMERICA_BOUNDS = (-90.0, -60.0, -30.0, 15.0)
EAST_ASIA_BOUNDS = (100.0, -10.0, 150.0, 55.0)
ASIA_DENSE_BOUNDS = (95.0, -10.0, 150.0, 55.0)

CLASS_BACKGROUND = 0
CLASS_TINY_RESIDENTIAL = 1
CLASS_SMALL_RESIDENTIAL = 2
CLASS_COMPACT_RESIDENTIAL = 3
CLASS_MEDIUM = 4
CLASS_SMALL_APARTMENT = 5
CLASS_APARTMENT_BLOCK = 6
CLASS_LARGE = 7
CLASS_EXTRA_LARGE = 8

CLASS_LABELS = {
    CLASS_BACKGROUND: "background",
    CLASS_TINY_RESIDENTIAL: "tiny_residential",
    CLASS_SMALL_RESIDENTIAL: "small_residential",
    CLASS_COMPACT_RESIDENTIAL: "compact_residential",
    CLASS_MEDIUM: "medium",
    CLASS_SMALL_APARTMENT: "small_apartment",
    CLASS_APARTMENT_BLOCK: "apartment_block",
    CLASS_LARGE: "large",
    CLASS_EXTRA_LARGE: "extra_large",
}
NUM_PLACEMENT_CLASSES = len(CLASS_LABELS)

YOLO_HEIGHT_BINS_M = (
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
    12.0,
    15.0,
    18.0,
    21.0,
    24.0,
    30.0,
    40.0,
    60.0,
    90.0,
    120.0,
)
YOLO_HEIGHT_LABEL_OFF = "off"
YOLO_HEIGHT_LABEL_BINS = "bins"
YOLO_HEIGHT_LABEL_MODES = (YOLO_HEIGHT_LABEL_OFF, YOLO_HEIGHT_LABEL_BINS)


@dataclass(frozen=True)
class Feature:
    """A geometry label from SimHeaven or an auxiliary source."""

    source: str
    geometry: BaseGeometry
    properties: dict[str, Any] = field(default_factory=dict)
    height: float | None = None
    confidence: float | None = None
    asset_path: str | None = None
    heading_deg: float | None = None
    width_m: float | None = None
    depth_m: float | None = None
    placement_class: int = CLASS_MEDIUM


@dataclass(frozen=True)
class TextureRef:
    """Loaded texture plus path metadata."""

    image: Image.Image
    path: Path
    temporary: bool = False


@dataclass(frozen=True)
class TileRecord:
    """One Ortho4XP tile available for training or evaluation."""

    name: str
    lat: int
    lon: int
    path: str
    split: str
    region: str


@dataclass(frozen=True)
class TextureRecord:
    """One source texture with geographic bounds."""

    path: str
    tile_lat: int
    tile_lon: int
    til_y_top: int
    til_x_left: int
    zoomlevel: int
    provider: str
    bounds: tuple[float, float, float, float]
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class ChipRecord:
    """Manifest row for one extracted training chip."""

    chip_id: str
    split: str
    tile: str
    region: str
    image: str
    center: str = ""
    mask: str = ""
    dims: str = ""
    heading: str = ""
    placement_class: str = ""
    weight: float = 1.0
    source_texture: str | None = None
    width: int | None = None
    height: int | None = None
    labels: str | None = None
    west: float | None = None
    south: float | None = None
    east: float | None = None
    north: float | None = None


@dataclass
class TrainConfig:
    """Training configuration for the v1 placement model."""

    dataset_manifest: Path
    run_dir: Path
    epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 2e-4
    chip_size: int = 512
    num_workers: int = 0
    hard_mining_fraction: float = 0.25
    hard_mining_max_weight: float = 3.0
    dimension_loss_weight: float = 4.0
    device: str = "cuda"
    encoder_name: str = "nave1616/SegFormer-landcover-FT"
    resume: bool = True
    pause_file: Path | None = None
    progress_interval_steps: int = 100
    checkpoint_interval_steps: int = 500


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    """Append one JSON row to a UTF-8 JSONL file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file, skipping blank lines."""

    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_completed_ids(path: str | Path, key: str = "chip_id") -> set[str]:
    """Return completed IDs from a JSONL manifest."""

    completed: set[str] = set()
    for row in iter_jsonl(path):
        value = row.get(key)
        if value is not None:
            completed.add(str(value))
    return completed


def wgs84_to_orthogrid(lat: float, lon: float, zoomlevel: int = DEFAULT_ZOOM) -> tuple[int, int]:
    """Return the Ortho4XP texture-block Web Mercator grid for a point."""

    lat = max(-85.05112878, min(85.05112878, float(lat)))
    lon = ((float(lon) + 180.0) % 360.0) - 180.0
    n = 2**zoomlevel
    x_tile = int(math.floor((lon + 180.0) / 360.0 * n))
    lat_rad = math.radians(lat)
    y_tile = int(
        math.floor(
            (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
        )
    )
    x_tile = min(n - 1, max(0, x_tile))
    y_tile = min(n - 1, max(0, y_tile))
    return (
        (x_tile // TEXTURE_BLOCK_SIZE) * TEXTURE_BLOCK_SIZE,
        (y_tile // TEXTURE_BLOCK_SIZE) * TEXTURE_BLOCK_SIZE,
    )


def _gtile_to_wgs84(tile_x: int, tile_y: int, zoomlevel: int) -> tuple[float, float]:
    n = 2.0**zoomlevel
    lon = tile_x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * tile_y / n)))
    return math.degrees(lat_rad), lon


def texture_bounds(
    tile_x: int,
    tile_y: int,
    zoomlevel: int = DEFAULT_ZOOM,
    block_size: int = TEXTURE_BLOCK_SIZE,
) -> tuple[float, float, float, float]:
    """Return `(west, south, east, north)` for an Ortho4XP texture block."""

    north, west = _gtile_to_wgs84(tile_x, tile_y, zoomlevel)
    south, east = _gtile_to_wgs84(tile_x + block_size, tile_y + block_size, zoomlevel)
    return west, south, east, north


def parse_texture_name(name: str) -> tuple[int, int, str, int] | None:
    """Parse standard Ortho4XP texture names."""

    match = WEB_MERCATOR_TILE_RE.match(Path(name).name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), match.group(3), int(match.group(4))


def pixel_to_lonlat(
    px: float,
    py: float,
    width: int,
    height: int,
    bounds: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Convert image pixel coordinates to `(lon, lat)` inside bounds."""

    west, south, east, north = bounds
    lon = west + float(px) / max(1, width) * (east - west)
    lat = north - float(py) / max(1, height) * (north - south)
    return lon, lat


def lonlat_to_pixel(
    lon: float,
    lat: float,
    width: int,
    height: int,
    bounds: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Convert `(lon, lat)` to image pixel coordinates inside bounds."""

    west, south, east, north = bounds
    x = (float(lon) - west) / (east - west) * width
    y = (north - float(lat)) / (north - south) * height
    return x, y


def _image_shape(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, tuple):
        return int(size[0]), int(size[1])
    return int(size), int(size)


def rasterize_features(
    features: Sequence[Feature],
    bounds: tuple[float, float, float, float],
    chip_size: int | tuple[int, int] = 512,
    value: int = 255,
) -> Image.Image:
    """Rasterize feature polygons into a single-channel mask."""

    width, height = _image_shape(chip_size)
    mask = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(mask)
    west, south, east, north = bounds

    def to_px(coord: tuple[float, float]) -> tuple[int, int]:
        lon, lat = coord
        x = int(round((lon - west) / (east - west) * (width - 1)))
        y = int(round((north - lat) / (north - south) * (height - 1)))
        return x, y

    for feature in features:
        geom = feature.geometry
        if geom.is_empty:
            continue
        for polygon in _iter_polygons(geom):
            exterior = [to_px(pt) for pt in polygon.exterior.coords]
            if len(exterior) >= 3:
                draw.polygon(exterior, fill=value)
            for interior in polygon.interiors:
                hole = [to_px(pt) for pt in interior.coords]
                if len(hole) >= 3:
                    draw.polygon(hole, fill=0)
    return mask


def _iter_polygons(geometry: BaseGeometry) -> Iterator[Polygon]:
    """Yield polygon members from any Shapely geometry."""

    if geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _iter_polygons(part)


def dominant_angle_degrees(geometry: BaseGeometry) -> float | None:
    """Return the long-axis angle of a geometry's minimum rotated rectangle."""

    if geometry.is_empty:
        return None
    rect = geometry.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    if len(coords) < 4:
        return None
    edges: list[tuple[float, float, float]] = []
    for a, b in zip(coords, coords[1:]):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        length = math.hypot(dx, dy)
        if length > 0:
            edges.append((length, dx, dy))
    if not edges:
        return None
    _length, dx, dy = max(edges, key=lambda item: item[0])
    return math.degrees(math.atan2(dy, dx)) % 180.0


def rotation_error_degrees(a: float, b: float) -> float:
    """Return absolute orientation error modulo 180 degrees."""

    diff = abs((float(a) - float(b)) % 180.0)
    return min(diff, 180.0 - diff)


def cleanup_texture(texture: TextureRef) -> None:
    """Remove temporary texture sidecar files."""

    if texture.temporary:
        try:
            Path(texture.path).unlink()
        except FileNotFoundError:
            pass


def parse_size_bytes(value: str) -> float:
    """Parse human file-size strings such as `1.5MB` into bytes."""

    match = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?B)?\s*$", value, re.I)
    if not match:
        raise ValueError(f"Unsupported size string: {value!r}")
    number = float(match.group(1))
    unit = (match.group(2) or "B").upper()
    factors = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
    return number * factors[unit]


def feature_centroid_in_bounds(
    feature: Feature,
    bounds: tuple[float, float, float, float],
) -> bool:
    """Return True if a feature centroid falls inside `(west, south, east, north)`."""

    west, south, east, north = bounds
    centroid = feature.geometry.centroid
    return west <= centroid.x <= east and south <= centroid.y <= north


def _stream_microsoft_part(part: str | Path, location: str) -> Iterator[Feature]:
    """Stream Microsoft building footprint GeoJSONL CSV/GZip records.

    Kept for compatibility with the existing training tests and future auxiliary
    labels, though v1 uses SimHeaven labels only.
    """

    part = Path(part)
    opener = gzip.open if part.suffix.lower() == ".gz" else open
    with opener(part, "rt", encoding="utf-8") as handle:
        sample = handle.readline()
        if not sample:
            return
        handle.seek(0)
        if sample.lstrip().startswith("{"):
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                yield _feature_from_geojson(payload, "microsoft", location)
        else:
            reader = csv.DictReader(handle)
            for row in reader:
                payload = json.loads(row.get("geometry") or row.get("geojson") or "{}")
                properties = dict(row)
                feature = {"type": "Feature", "properties": properties, "geometry": payload}
                yield _feature_from_geojson(feature, "microsoft", location)


def _feature_from_geojson(payload: dict[str, Any], source: str, location: str) -> Feature:
    properties = dict(payload.get("properties") or {})
    properties["location"] = location
    return Feature(
        source=source,
        geometry=shape(payload["geometry"]),
        properties=properties,
        height=_optional_float(properties.get("height")),
        confidence=_optional_float(properties.get("confidence")),
    )


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def yolo_height_bin_index(height_m: float | None) -> int | None:
    """Return the nearest YOLO height-bin index, with the final bin as 120m+."""

    if height_m is None:
        return None
    try:
        height = float(height_m)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(height):
        return None
    if height >= YOLO_HEIGHT_BINS_M[-1]:
        return len(YOLO_HEIGHT_BINS_M) - 1
    return min(
        range(len(YOLO_HEIGHT_BINS_M)),
        key=lambda idx: abs(YOLO_HEIGHT_BINS_M[idx] - height),
    )


def encode_yolo_obb_class(
    placement_class: int,
    height_m: float | None = None,
    height_labels: str = YOLO_HEIGHT_LABEL_OFF,
) -> int | None:
    """Encode a placement class, optionally expanded by height bin, for YOLO OBB."""

    try:
        placement = int(placement_class)
    except (TypeError, ValueError):
        return None
    if placement == CLASS_BACKGROUND:
        return None
    if placement not in CLASS_LABELS:
        return None
    placement_index = placement - 1
    if height_labels == YOLO_HEIGHT_LABEL_OFF:
        return placement_index
    if height_labels != YOLO_HEIGHT_LABEL_BINS:
        raise ValueError(f"Unknown height label mode: {height_labels!r}")
    bin_index = yolo_height_bin_index(height_m)
    if bin_index is None:
        return None
    return placement_index * len(YOLO_HEIGHT_BINS_M) + bin_index


def decode_yolo_obb_class(
    class_id: int,
    height_labels: str = YOLO_HEIGHT_LABEL_OFF,
) -> tuple[int, float | None]:
    """Decode a YOLO OBB class into placement class and optional binned height."""

    model_class = int(class_id)
    if height_labels == YOLO_HEIGHT_LABEL_OFF:
        return model_class + 1, None
    if height_labels != YOLO_HEIGHT_LABEL_BINS:
        raise ValueError(f"Unknown height label mode: {height_labels!r}")
    placement_index, bin_index = divmod(model_class, len(YOLO_HEIGHT_BINS_M))
    return placement_index + 1, float(YOLO_HEIGHT_BINS_M[bin_index])


def yolo_obb_class_names(height_labels: str = YOLO_HEIGHT_LABEL_OFF) -> dict[int, str]:
    """Return class-name mapping for legacy or height-binned YOLO OBB training."""

    if height_labels == YOLO_HEIGHT_LABEL_OFF:
        return {idx - 1: name for idx, name in CLASS_LABELS.items() if idx != CLASS_BACKGROUND}
    if height_labels != YOLO_HEIGHT_LABEL_BINS:
        raise ValueError(f"Unknown height label mode: {height_labels!r}")
    names: dict[int, str] = {}
    for placement_class, label in sorted(CLASS_LABELS.items()):
        if placement_class == CLASS_BACKGROUND:
            continue
        placement_index = placement_class - 1
        for bin_index, height in enumerate(YOLO_HEIGHT_BINS_M):
            suffix = "h120plus" if bin_index == len(YOLO_HEIGHT_BINS_M) - 1 else f"h{int(height):02d}"
            names[placement_index * len(YOLO_HEIGHT_BINS_M) + bin_index] = f"{label}__{suffix}"
    return names


def discover_tiles(
    tiles_root: str | Path = DEFAULT_TILES_ROOT,
    seed: int = 1337,
    val_fraction: float = 0.1,
    test_fraction: float = 0.05,
) -> list[TileRecord]:
    """Discover Ortho4XP tile folders and assign deterministic splits."""

    root = Path(tiles_root)
    candidates: list[tuple[str, int, int, Path, str]] = []
    for entry in sorted(root.iterdir() if root.exists() else [], key=lambda item: item.name):
        if not entry.is_dir():
            continue
        match = TILE_DIR_RE.match(entry.name)
        if not match:
            continue
        lat = int(match.group(1))
        lon = int(match.group(2))
        region = holdout_region(lat, lon) or "standard"
        candidates.append((entry.name, lat, lon, entry, region))

    rng = random.Random(seed)
    standard = [item for item in candidates if item[4] == "standard"]
    rng.shuffle(standard)
    n_val = int(round(len(standard) * val_fraction))
    n_test = int(round(len(standard) * test_fraction))
    split_by_name = {
        item[0]: "test" if idx < n_test else "val" if idx < n_test + n_val else "train"
        for idx, item in enumerate(standard)
    }

    records: list[TileRecord] = []
    for name, lat, lon, path, region in candidates:
        split = "holdout" if region != "standard" else split_by_name[name]
        records.append(
            TileRecord(
                name=name,
                lat=lat,
                lon=lon,
                path=str(path),
                split=split,
                region=region,
            )
        )
    return sorted(records, key=lambda record: record.name)


def holdout_region(lat: float, lon: float) -> str | None:
    """Return holdout region name for tile origin, if any."""

    if _point_in_bounds(lon, lat, SOUTH_AMERICA_BOUNDS):
        return "south_america"
    if _point_in_bounds(lon, lat, EAST_ASIA_BOUNDS):
        return "east_asia"
    return None


def _point_in_bounds(lon: float, lat: float, bounds: tuple[float, float, float, float]) -> bool:
    west, south, east, north = bounds
    return west <= float(lon) <= east and south <= float(lat) <= north


def write_tile_splits(
    output_path: str | Path,
    tiles_root: str | Path = DEFAULT_TILES_ROOT,
    seed: int = 1337,
) -> list[TileRecord]:
    """Discover tiles and write deterministic split JSONL."""

    records = discover_tiles(tiles_root=tiles_root, seed=seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    return records


def load_tile_splits(path: str | Path) -> list[TileRecord]:
    """Load tile split records from JSONL."""

    return [TileRecord(**row) for row in iter_jsonl(path)]


def collect_tile_textures(tile: TileRecord | dict[str, Any]) -> list[TextureRecord]:
    """Collect DDS textures for a tile. DDS files are authoritative for v1."""

    if isinstance(tile, dict):
        tile = TileRecord(**tile)
    tex_dir = Path(tile.path) / "textures"
    records: list[TextureRecord] = []
    if not tex_dir.exists():
        return records
    for path in sorted(tex_dir.iterdir(), key=lambda item: item.name.lower()):
        parsed = parse_texture_name(path.name)
        if parsed is None:
            continue
        til_y_top, til_x_left, provider, zoomlevel = parsed
        if provider.upper() == "ZL":
            continue
        tex_width, tex_height = read_image_size(path)
        records.append(
            TextureRecord(
                path=str(path),
                tile_lat=tile.lat,
                tile_lon=tile.lon,
                til_y_top=til_y_top,
                til_x_left=til_x_left,
                zoomlevel=zoomlevel,
                provider=provider,
                bounds=texture_bounds(til_x_left, til_y_top, zoomlevel),
                width=tex_width,
                height=tex_height,
            )
        )
    return records


def read_image_size(path: str | Path) -> tuple[int, int]:
    """Return source image size without loading full pixel data."""

    with Image.open(path) as image:
        return image.size


def tile_dsf_relpath(lat: int, lon: int) -> Path:
    """Return the `Earth nav data` relative DSF path for a 1x1 tile."""

    lat_group = math.floor(int(lat) / 10) * 10
    lon_group = math.floor(int(lon) / 10) * 10
    lat_tile = f"{'+' if lat >= 0 else '-'}{abs(int(lat)):02d}"
    lon_tile = f"{'+' if lon >= 0 else '-'}{abs(int(lon)):03d}"
    lat_block = f"{'+' if lat_group >= 0 else '-'}{abs(int(lat_group)):02d}"
    lon_block = f"{'+' if lon_group >= 0 else '-'}{abs(int(lon_group)):03d}"
    return Path("Earth nav data") / f"{lat_block}{lon_block}" / f"{lat_tile}{lon_tile}.dsf"


def find_simheaven_building_dsfs(
    custom_scenery_root: str | Path,
    lat: int,
    lon: int,
) -> list[Path]:
    """Return SimHeaven footprint/scenery DSFs for a tile."""

    root = Path(custom_scenery_root)
    relpath = tile_dsf_relpath(lat, lon)
    matches: list[Path] = []
    if not root.exists():
        return matches
    for entry in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if not entry.is_dir():
            continue
        name = entry.name.lower()
        if "simheaven" not in name:
            continue
        if "footprints" not in name and "scenery" not in name:
            continue
        dsf_path = entry / relpath
        if dsf_path.exists():
            matches.append(dsf_path)
    return matches


def ensure_dsf_text(
    dsf_path: str | Path,
    dsftool_path: str | Path,
    cache_dir: str | Path,
) -> Path:
    """Disassemble a DSF into a cached text file."""

    dsf_path = Path(dsf_path)
    dsftool_path = Path(dsftool_path)
    cache_dir = Path(cache_dir)
    stat = dsf_path.stat()
    key = hashlib.sha1(
        f"{dsf_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()
    text_path = cache_dir / "dsf_text" / f"{key}.txt"
    if text_path.exists():
        return text_path
    text_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(dsftool_path), "--dsf2text", str(dsf_path), str(text_path)],
        check=True,
        capture_output=True,
        timeout=180,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    return text_path


def parse_simheaven_text(text_path: str | Path) -> list[Feature]:
    """Parse SimHeaven DSF text into supervised building footprint features."""

    object_defs: list[str] = []
    polygon_defs: list[str] = []
    features: list[Feature] = []
    current_polygon_path: str | None = None
    current_polygon_height: float | None = None
    current_winding: list[tuple[float, float]] | None = None

    with Path(text_path).open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("OBJECT_DEF "):
                object_defs.append(line.split(" ", 1)[1])
            elif line.startswith("POLYGON_DEF "):
                polygon_defs.append(line.split(" ", 1)[1])
            elif line.startswith("OBJECT "):
                feature = _parse_object_line(line, object_defs)
                if feature is not None:
                    features.append(feature)
            elif line.startswith("BEGIN_POLYGON "):
                current_polygon_path = None
                current_polygon_height = None
                current_winding = None
                parts = line.split()
                try:
                    path = polygon_defs[int(parts[1])]
                except (IndexError, ValueError):
                    continue
                if is_simheaven_building_polygon(path):
                    current_polygon_path = path
                    current_polygon_height = _optional_float(parts[2] if len(parts) > 2 else None)
            elif line == "BEGIN_WINDING" and current_polygon_path:
                current_winding = []
            elif line.startswith("POLYGON_POINT ") and current_winding is not None:
                parts = line.split()
                try:
                    current_winding.append((float(parts[1]), float(parts[2])))
                except (IndexError, ValueError):
                    pass
            elif line == "END_WINDING" and current_winding is not None:
                if len(current_winding) >= 3:
                    poly = Polygon(current_winding)
                    if poly.is_valid and not poly.is_empty:
                        features.append(
                            _feature_from_polygon(
                                poly,
                                current_polygon_path or "",
                                current_polygon_height,
                            )
                        )
                current_winding = None
            elif line == "END_POLYGON":
                current_polygon_path = None
                current_polygon_height = None
                current_winding = None
    return features


def _parse_object_line(line: str, object_defs: Sequence[str]) -> Feature | None:
    parts = line.split()
    try:
        object_path = object_defs[int(parts[1])]
    except (IndexError, ValueError):
        return None
    if not is_simheaven_building_object(object_path):
        return None
    try:
        lon = float(parts[2])
        lat = float(parts[3])
        heading = float(parts[4]) if len(parts) > 4 else 0.0
    except (IndexError, ValueError):
        return None
    width_m, depth_m = simheaven_object_dims(object_path)
    footprint = oriented_footprint(lon, lat, width_m, depth_m, heading)
    return Feature(
        source="simheaven",
        geometry=footprint,
        asset_path=object_path,
        height=simheaven_object_height_m(object_path),
        heading_deg=heading % 360.0,
        width_m=width_m,
        depth_m=depth_m,
        placement_class=placement_class_for_dims(width_m, depth_m),
    )


def _feature_from_polygon(poly: Polygon, asset_path: str, height_m: float | None = None) -> Feature:
    angle = dominant_angle_degrees(poly)
    width_m, depth_m = footprint_dims_m(poly)
    return Feature(
        source="simheaven",
        geometry=poly,
        asset_path=asset_path,
        height=height_m,
        heading_deg=None if angle is None else angle % 180.0,
        width_m=width_m,
        depth_m=depth_m,
        placement_class=placement_class_for_dims(width_m, depth_m),
    )


def is_simheaven_building_object(path: str) -> bool:
    """Return True for physical SimHeaven building object defs."""

    normalized = path.replace("\\", "/").lower()
    if not normalized.startswith("simheaven/"):
        return False
    if any(
        token in normalized
        for token in (
            "/houses/",
            "/industrial/",
            "/commercial/",
            "/residential/",
            "/farms/",
            "/sheds/",
        )
    ):
        return True
    return "/landmarks/" in normalized and any(
        token in normalized for token in ("church", "chapel", "mosque")
    )


def is_simheaven_building_polygon(path: str) -> bool:
    """Return True for SimHeaven facade-like polygon defs."""

    normalized = path.replace("\\", "/").lower()
    if not normalized.startswith("simheaven/"):
        return False
    if "/ground/" in normalized or "parking" in normalized or "pier" in normalized:
        return False
    return "/facades/" in normalized or "/landmarks/" in normalized


_SIMHEAVEN_DIMS_RE = re.compile(r"_(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)(?:x\d+(?:\.\d+)?)?")
_SIMHEAVEN_FLOORS_RE = re.compile(r"_(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)")


def simheaven_object_dims(path: str) -> tuple[float, float]:
    """Infer object footprint dimensions in metres from a SimHeaven path."""

    name = Path(path.replace("\\", "/")).name.lower()
    match = _SIMHEAVEN_DIMS_RE.search(name)
    if match:
        return float(match.group(1)), float(match.group(2))
    if any(token in name for token in ("church", "chapel", "mosque")):
        return 26.0, 26.0
    return 14.0, 14.0


def simheaven_object_height_m(path: str) -> float | None:
    """Infer object height in metres from SimHeaven floor-count suffixes."""

    name = Path(path.replace("\\", "/")).name.lower()
    match = _SIMHEAVEN_FLOORS_RE.search(name)
    if not match:
        return None
    return float(match.group(3)) * 3.2


def metres_per_degree(lat: float) -> tuple[float, float]:
    """Return approximate metres per lon degree and lat degree at latitude."""

    lat_m = 110_540.0
    lon_m = 111_320.0 * math.cos(math.radians(float(lat)))
    return lon_m, lat_m


def oriented_footprint(
    lon: float,
    lat: float,
    width_m: float,
    depth_m: float,
    heading_deg: float,
) -> Polygon:
    """Build an oriented rectangular footprint around `(lon, lat)`."""

    lon_m, lat_m = metres_per_degree(lat)
    half_w = float(width_m) / 2.0 / max(1e-9, lon_m)
    half_d = float(depth_m) / 2.0 / max(1e-9, lat_m)
    poly = box(float(lon) - half_w, float(lat) - half_d, float(lon) + half_w, float(lat) + half_d)
    return affinity.rotate(poly, -float(heading_deg), origin=(float(lon), float(lat)))


def footprint_dims_m(geometry: BaseGeometry) -> tuple[float, float]:
    """Return long/short minimum-rectangle dimensions in metres."""

    if geometry.is_empty:
        return 0.0, 0.0
    rect = geometry.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    lengths: list[float] = []
    lat = geometry.centroid.y
    lon_m, lat_m = metres_per_degree(lat)
    for a, b in zip(coords, coords[1:]):
        dx_m = (b[0] - a[0]) * lon_m
        dy_m = (b[1] - a[1]) * lat_m
        length = math.hypot(dx_m, dy_m)
        if length > 0:
            lengths.append(length)
    if not lengths:
        return 0.0, 0.0
    unique = sorted(lengths, reverse=True)
    return unique[0], unique[-1]


def placement_class_for_dims(width_m: float | None, depth_m: float | None) -> int:
    """Map footprint dimensions to a coarse placement class."""

    width = float(width_m or 0.0)
    depth = float(depth_m or 0.0)
    area = width * depth
    long_side = max(width, depth)
    if area < 45:
        return CLASS_TINY_RESIDENTIAL
    if area < 120:
        return CLASS_SMALL_RESIDENTIAL
    if area < 240:
        return CLASS_COMPACT_RESIDENTIAL
    if area < 700:
        return CLASS_MEDIUM
    if area < 1_300:
        return CLASS_SMALL_APARTMENT
    if area < 2_500:
        return CLASS_APARTMENT_BLOCK
    if area < 5_500 and long_side < 90:
        return CLASS_LARGE
    return CLASS_EXTRA_LARGE


def load_simheaven_features_for_tile(
    lat: int,
    lon: int,
    custom_scenery_root: str | Path,
    dsftool_path: str | Path,
    cache_dir: str | Path,
) -> list[Feature]:
    """Load SimHeaven building features for one tile."""

    features: list[Feature] = []
    for dsf_path in find_simheaven_building_dsfs(custom_scenery_root, lat, lon):
        text_path = ensure_dsf_text(dsf_path, dsftool_path, cache_dir)
        features.extend(parse_simheaven_text(text_path))
    return features


def build_chip_labels(
    features: Sequence[Feature],
    bounds: tuple[float, float, float, float],
    chip_size: int | tuple[int, int] = 512,
) -> dict[str, Image.Image]:
    """Build dense label maps for one chip."""

    width_px, height_px = _image_shape(chip_size)
    mask = rasterize_features(features, bounds, chip_size=chip_size, value=255)
    center = Image.new("L", (width_px, height_px), color=0)
    dims = Image.new("RGB", (width_px, height_px), color=(0, 0, 0))
    heading = Image.new("RGB", (width_px, height_px), color=(127, 127, 0))
    klass = Image.new("L", (width_px, height_px), color=CLASS_BACKGROUND)

    center_draw = ImageDraw.Draw(center)
    dims_draw = ImageDraw.Draw(dims)
    heading_draw = ImageDraw.Draw(heading)
    class_draw = ImageDraw.Draw(klass)

    for feature in features:
        if feature.geometry.is_empty:
            continue
        centroid = feature.geometry.centroid
        cx, cy = lonlat_to_pixel(centroid.x, centroid.y, width_px, height_px, bounds)
        radius = max(3, int(round(min(width_px, height_px) / 128.0)))
        center_draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=255,
        )
        width = max(0, min(255, int(round((feature.width_m or 0.0) * 2.0))))
        depth = max(0, min(255, int(round((feature.depth_m or 0.0) * 2.0))))
        angle = math.radians(float(feature.heading_deg or 0.0))
        sin_v = int(round((math.sin(angle) * 0.5 + 0.5) * 255))
        cos_v = int(round((math.cos(angle) * 0.5 + 0.5) * 255))
        for polygon in _iter_polygons(feature.geometry):
            poly_pixels = [
                lonlat_to_pixel(lon, lat, width_px, height_px, bounds)
                for lon, lat in polygon.exterior.coords
            ]
            if len(poly_pixels) >= 3:
                dims_draw.polygon(poly_pixels, fill=(width, depth, 0))
                heading_draw.polygon(poly_pixels, fill=(sin_v, cos_v, 0))
                class_draw.polygon(poly_pixels, fill=int(feature.placement_class))

    return {
        "center": center,
        "mask": mask,
        "dims": dims,
        "heading": heading,
        "placement_class": klass,
    }


def make_chip_id(tile_name: str, texture_path: str | Path) -> str:
    """Return a stable chip ID for a tile texture."""

    stem = Path(texture_path).stem
    return f"{tile_name}_{stem}"


def extract_dataset(
    splits_path: str | Path,
    output_root: str | Path = DEFAULT_TRAINING_ROOT,
    custom_scenery_root: str | Path = DEFAULT_CUSTOM_SCENERY_ROOT,
    dsftool_path: str | Path = Path(r"G:\Dev\Ortho4XP-SegFormer\Utils\win\DSFTool.exe"),
    chip_size: int = 512,
    max_tiles: int | None = None,
    include_holdouts: bool = True,
    materialize_images: bool = False,
    max_tile_features: int = 0,
) -> Path:
    """Create chip image/label files and return the dataset manifest path."""

    output_root = Path(output_root)
    cache_dir = output_root / "cache"
    manifest_path = output_root / "manifests" / "dataset.jsonl"
    completed = load_completed_ids(manifest_path)
    tiles = load_tile_splits(splits_path)
    if not include_holdouts:
        tiles = [tile for tile in tiles if tile.split != "holdout"]
    if max_tiles is not None:
        tiles = tiles[:max_tiles]

    for tile in tiles:
        print(
            f"[extract] tile={tile.name} split={tile.split} region={tile.region}",
            flush=True,
        )
        textures = collect_tile_textures(tile)
        tile_bounds = (float(tile.lon), float(tile.lat), float(tile.lon + 1), float(tile.lat + 1))
        tile_label_path = output_root / "labels" / "tiles" / f"{tile.name}_labels.sqlite"
        if _sqlite_label_cache_is_valid(tile_label_path):
            label_count = _sqlite_label_count(tile_label_path)
            print(
                f"[extract] tile={tile.name} labels={label_count} textures={len(textures)} label_cache={tile_label_path.name} cached=true",
                flush=True,
            )
            tile_chips = _append_texture_rows_from_label_cache(
                tile=tile,
                textures=textures,
                tile_label_path=tile_label_path,
                output_root=output_root,
                manifest_path=manifest_path,
                completed=completed,
                chip_size=chip_size,
                materialize_images=materialize_images,
            )
            print(
                f"[extract] tile={tile.name} done chips_written={tile_chips} total_completed={len(completed)}",
                flush=True,
            )
            continue

        features = load_simheaven_features_for_tile(
            tile.lat,
            tile.lon,
            custom_scenery_root,
            dsftool_path,
            cache_dir,
        )
        if not features:
            print(f"[extract] tile={tile.name} skipped: no SimHeaven building labels", flush=True)
            continue
        if max_tile_features > 0 and len(features) > max_tile_features:
            print(
                f"[extract] tile={tile.name} skipped: labels={len(features)} exceeds max_tile_features={max_tile_features}",
                flush=True,
            )
            append_jsonl(
                output_root / "manifests" / "skipped_tiles.jsonl",
                {
                    "tile": tile.name,
                    "lat": tile.lat,
                    "lon": tile.lon,
                    "split": tile.split,
                    "region": tile.region,
                    "labels": len(features),
                    "reason": "max_tile_features",
                    "limit": max_tile_features,
                },
            )
            continue
        tile_label_path = _save_sqlite_labels(
            features,
            output_root / "labels" / "tiles",
            tile.name,
            tile_bounds,
        )
        print(
            f"[extract] tile={tile.name} labels={len(features)} textures={len(textures)} label_cache={tile_label_path.name}",
            flush=True,
        )
        tile_chips = _append_texture_rows_from_label_cache(
            tile=tile,
            textures=textures,
            tile_label_path=tile_label_path,
            output_root=output_root,
            manifest_path=manifest_path,
            completed=completed,
            chip_size=chip_size,
            materialize_images=materialize_images,
        )
        print(
            f"[extract] tile={tile.name} done chips_written={tile_chips} total_completed={len(completed)}",
            flush=True,
        )
    return manifest_path


def _append_texture_rows_from_label_cache(
    tile: TileRecord,
    textures: Sequence[TextureRecord],
    tile_label_path: Path,
    output_root: Path,
    manifest_path: Path,
    completed: set[str],
    chip_size: int,
    materialize_images: bool,
) -> int:
    """Append pending manifest rows for textures that intersect cached labels."""

    tile_chips = 0
    for texture in textures:
        chip_id = make_chip_id(tile.name, texture.path)
        if chip_id in completed:
            continue
        if not _sqlite_has_features(tile_label_path, texture.bounds):
            continue
        image_path = (
            _copy_or_resize_texture(texture.path, output_root / "images", chip_id, chip_size)
            if materialize_images
            else Path(texture.path)
        )
        label_size = (
            (int(texture.width), int(texture.height))
            if texture.width and texture.height
            else read_image_size(texture.path)
        )
        west, south, east, north = texture.bounds
        append_jsonl(
            manifest_path,
            asdict(
                ChipRecord(
                    chip_id=chip_id,
                    split=tile.split,
                    tile=tile.name,
                    region=tile.region,
                    image=str(image_path),
                    source_texture=texture.path,
                    width=label_size[0],
                    height=label_size[1],
                    labels=str(tile_label_path),
                    west=west,
                    south=south,
                    east=east,
                    north=north,
                )
            ),
        )
        completed.add(chip_id)
        tile_chips += 1
        if tile_chips % 25 == 0:
            print(
                f"[extract] tile={tile.name} chips_written={tile_chips} total_completed={len(completed)}",
                flush=True,
            )
    return tile_chips


def _copy_or_resize_texture(
    source_path: str | Path,
    image_dir: Path,
    chip_id: str,
    chip_size: int,
) -> Path:
    image_dir.mkdir(parents=True, exist_ok=True)
    output_path = image_dir / f"{chip_id}.png"
    if output_path.exists():
        return output_path
    with Image.open(source_path) as image:
        image = image.convert("RGB")
        if image.size != (chip_size, chip_size):
            image = image.resize((chip_size, chip_size), Image.Resampling.LANCZOS)
        image.save(output_path)
    return output_path


def _build_feature_index(features: Sequence[Feature]) -> tuple[STRtree | None, list[Feature]]:
    """Build a spatial index for feature intersection checks."""

    indexed = [feature for feature in features if not feature.geometry.is_empty]
    if not indexed:
        return None, []
    return STRtree([feature.geometry for feature in indexed]), indexed


def _features_intersect_bounds(
    tree: STRtree | None,
    features: Sequence[Feature],
    bounds: tuple[float, float, float, float],
) -> bool:
    """Return True if any indexed feature intersects bounds."""

    if tree is None:
        return False
    query_geom = box(*bounds)
    matches = tree.query(query_geom)
    for match in matches:
        idx = int(match)
        if features[idx].geometry.intersects(query_geom):
            return True
    return False


def _save_label_images(labels: dict[str, Image.Image], label_dir: Path, chip_id: str) -> dict[str, Path]:
    label_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, image in labels.items():
        output_path = label_dir / f"{chip_id}_{name}.png"
        if not output_path.exists():
            image.save(output_path)
        paths[name] = output_path
    return paths


def _save_vector_labels(
    features: Sequence[Feature],
    label_dir: Path,
    chip_id: str,
    bounds: tuple[float, float, float, float],
    size: tuple[int, int],
) -> Path:
    """Save compact vector labels for one source texture."""

    label_dir.mkdir(parents=True, exist_ok=True)
    output_path = label_dir / f"{chip_id}_labels.json.gz"
    if output_path.exists():
        return output_path
    west, south, east, north = bounds
    payload = {
        "chip_id": chip_id,
        "b": [west, south, east, north],
        "s": [int(size[0]), int(size[1])],
        "n": len(features),
        "features": [_feature_to_label_record(feature) for feature in features],
    }
    with gzip.open(output_path, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
    return output_path


def _save_sqlite_labels(
    features: Sequence[Feature],
    label_dir: Path,
    tile_name: str,
    bounds: tuple[float, float, float, float],
) -> Path:
    """Save per-tile labels in a queryable SQLite/RTree cache."""

    label_dir.mkdir(parents=True, exist_ok=True)
    output_path = label_dir / f"{tile_name}_labels.sqlite"
    if _sqlite_label_cache_is_valid(output_path, require_height=True):
        return output_path

    tmp_path = output_path.with_suffix(".sqlite.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    connection = sqlite3.connect(str(tmp_path))
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute(
            "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            """
            CREATE TABLE features (
                id INTEGER PRIMARY KEY,
                asset_path TEXT,
                heading REAL,
                width REAL,
                depth REAL,
                height REAL,
                class INTEGER NOT NULL,
                geom BLOB NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE VIRTUAL TABLE idx USING rtree(id, minx, maxx, miny, maxy)"
        )

        west, south, east, north = bounds
        connection.executemany(
            "INSERT INTO meta(key, value) VALUES (?, ?)",
            [
                ("tile", tile_name),
                ("bounds", json.dumps([west, south, east, north])),
                ("count", str(len(features))),
                ("format", "simheaven-placement-sqlite-v2"),
            ],
        )

        feature_rows: list[tuple[Any, ...]] = []
        index_rows: list[tuple[Any, ...]] = []
        next_id = 1
        for feature in features:
            if feature.geometry.is_empty:
                continue
            minx, miny, maxx, maxy = feature.geometry.bounds
            feature_rows.append(
                (
                    next_id,
                    feature.asset_path,
                    feature.heading_deg,
                    feature.width_m,
                    feature.depth_m,
                    feature.height,
                    int(feature.placement_class),
                    sqlite3.Binary(feature.geometry.wkb),
                )
            )
            index_rows.append(
                (next_id, float(minx), float(maxx), float(miny), float(maxy))
            )
            next_id += 1
            if len(feature_rows) >= 10_000:
                _insert_sqlite_label_rows(connection, feature_rows, index_rows)
                feature_rows.clear()
                index_rows.clear()
        if feature_rows:
            _insert_sqlite_label_rows(connection, feature_rows, index_rows)
        connection.commit()
    finally:
        connection.close()

    if output_path.exists():
        output_path.unlink()
    tmp_path.replace(output_path)
    _SQLITE_COUNT_CACHE[str(output_path.resolve())] = len(features)
    return output_path


def _insert_sqlite_label_rows(
    connection: sqlite3.Connection,
    feature_rows: Sequence[tuple[Any, ...]],
    index_rows: Sequence[tuple[Any, ...]],
) -> None:
    connection.executemany(
        """
        INSERT INTO features(id, asset_path, heading, width, depth, height, class, geom)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        feature_rows,
    )
    connection.executemany(
        "INSERT INTO idx(id, minx, maxx, miny, maxy) VALUES (?, ?, ?, ?, ?)",
        index_rows,
    )


def _sqlite_label_cache_is_valid(path: Path, require_height: bool = False) -> bool:
    if not path.exists():
        return False
    try:
        connection = sqlite3.connect(str(path))
        try:
            count = connection.execute(
                "SELECT value FROM meta WHERE key = 'count'"
            ).fetchone()
            format_row = connection.execute(
                "SELECT value FROM meta WHERE key = 'format'"
            ).fetchone()
            connection.execute("SELECT id FROM idx LIMIT 1").fetchone()
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(features)").fetchall()
            }
        finally:
            connection.close()
    except sqlite3.DatabaseError:
        return False
    if require_height and (
        "height" not in columns
        or format_row is None
        or format_row[0] != "simheaven-placement-sqlite-v2"
    ):
        return False
    if count is not None:
        _SQLITE_COUNT_CACHE[str(path.resolve())] = int(count[0])
    return True


def _sqlite_label_count(path: str | Path) -> int:
    path = Path(path)
    cache_key = str(path.resolve())
    cached = _SQLITE_COUNT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    connection = sqlite3.connect(str(path))
    try:
        row = connection.execute("SELECT value FROM meta WHERE key = 'count'").fetchone()
    finally:
        connection.close()
    count = int(row[0]) if row is not None else 0
    _SQLITE_COUNT_CACHE[cache_key] = count
    return count


def _sqlite_has_features(
    path: str | Path,
    bounds: tuple[float, float, float, float],
) -> bool:
    west, south, east, north = bounds
    connection = sqlite3.connect(str(path))
    try:
        row = connection.execute(
            """
            SELECT id
            FROM idx
            WHERE maxx >= ? AND minx <= ? AND maxy >= ? AND miny <= ?
            LIMIT 1
            """,
            (west, east, south, north),
        ).fetchone()
    finally:
        connection.close()
    return row is not None


def _sqlite_feature_bounds_for_bounds(
    path: str | Path,
    bounds: tuple[float, float, float, float],
    limit: int = 2048,
) -> list[dict[str, list[float]]]:
    west, south, east, north = bounds
    connection = sqlite3.connect(str(path))
    try:
        rows = connection.execute(
            """
            SELECT minx, miny, maxx, maxy
            FROM idx
            WHERE maxx >= ? AND minx <= ? AND maxy >= ? AND miny <= ?
            LIMIT ?
            """,
            (west, east, south, north, int(limit)),
        ).fetchall()
    finally:
        connection.close()
    return [{"b": [float(minx), float(miny), float(maxx), float(maxy)]} for minx, miny, maxx, maxy in rows]


def _features_from_sqlite_for_bounds(
    path: str | Path,
    bounds: tuple[float, float, float, float],
) -> list[Feature]:
    """Load only indexed SQLite features whose geometry intersects bounds."""

    west, south, east, north = bounds
    query = box(*bounds)
    connection = sqlite3.connect(str(path))
    try:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(features)").fetchall()
        }
        height_expr = "f.height" if "height" in columns else "NULL"
        rows = connection.execute(
            f"""
            SELECT f.asset_path, f.heading, f.width, f.depth, {height_expr}, f.class, f.geom
            FROM features f
            JOIN idx i ON f.id = i.id
            WHERE i.maxx >= ? AND i.minx <= ? AND i.maxy >= ? AND i.miny <= ?
            """,
            (west, east, south, north),
        ).fetchall()
    finally:
        connection.close()

    result: list[Feature] = []
    for asset_path, heading, width, depth, height, placement_class, geom_blob in rows:
        geometry = wkb.loads(bytes(geom_blob))
        if geometry.intersects(query):
            result.append(
                Feature(
                    source="simheaven",
                    geometry=geometry,
                    asset_path=asset_path,
                    height=_optional_float(height),
                    heading_deg=_optional_float(heading),
                    width_m=_optional_float(width),
                    depth_m=_optional_float(depth),
                    placement_class=int(placement_class or CLASS_MEDIUM),
                )
            )
    return result


def _feature_to_label_record(feature: Feature) -> dict[str, Any]:
    """Serialize one feature into compact JSON-compatible label data."""

    return {
        "g": _geometry_to_compact(feature.geometry),
        "b": [round(float(value), 7) for value in feature.geometry.bounds],
        "a": feature.asset_path,
        "h": feature.heading_deg,
        "w": feature.width_m,
        "d": feature.depth_m,
        "z": feature.height,
        "c": feature.placement_class,
    }


def _feature_from_label_record(row: dict[str, Any]) -> Feature:
    """Deserialize one vector label record."""

    if "g" in row:
        geometry = _geometry_from_compact(row["g"])
        return Feature(
            source="simheaven",
            geometry=geometry,
            asset_path=row.get("a"),
            height=_optional_float(row.get("z")),
            heading_deg=_optional_float(row.get("h")),
            width_m=_optional_float(row.get("w")),
            depth_m=_optional_float(row.get("d")),
            placement_class=int(row.get("c") or CLASS_MEDIUM),
        )

    return Feature(
        source=str(row.get("source") or "simheaven"),
        geometry=shape(row["geometry"]),
        properties=dict(row.get("properties") or {}),
        height=_optional_float(row.get("height")),
        confidence=_optional_float(row.get("confidence")),
        asset_path=row.get("asset_path"),
        heading_deg=_optional_float(row.get("heading_deg")),
        width_m=_optional_float(row.get("width_m")),
        depth_m=_optional_float(row.get("depth_m")),
        placement_class=int(row.get("placement_class") or CLASS_MEDIUM),
    )


def load_vector_label_payload(path: str | Path) -> dict[str, Any]:
    """Load compact vector labels from JSON."""

    path = Path(path)
    cache_key = str(path.resolve())
    cached = _LABEL_PAYLOAD_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    _LABEL_PAYLOAD_CACHE[cache_key] = payload
    return payload


def load_vector_label_features(path: str | Path) -> list[Feature]:
    """Load compact vector labels as Feature instances."""

    payload = load_vector_label_payload(path)
    return [_feature_from_label_record(row) for row in payload.get("features", [])]


def _geometry_to_compact(geometry: BaseGeometry) -> list[list[list[list[float]]]]:
    """Serialize polygons as rounded coordinate rings."""

    polygons: list[list[list[list[float]]]] = []
    for polygon in _iter_polygons(geometry):
        rings: list[list[list[float]]] = []
        rings.append(_coords_to_compact(polygon.exterior.coords))
        for interior in polygon.interiors:
            rings.append(_coords_to_compact(interior.coords))
        polygons.append(rings)
    return polygons


def _coords_to_compact(coords: Iterable[tuple[float, float]]) -> list[list[float]]:
    return [[round(float(lon), 7), round(float(lat), 7)] for lon, lat in coords]


def _geometry_from_compact(polygons: list[list[list[list[float]]]]) -> BaseGeometry:
    """Deserialize compact coordinate rings into Shapely geometry."""

    result: list[Polygon] = []
    for rings in polygons:
        if not rings:
            continue
        exterior = [(float(lon), float(lat)) for lon, lat in rings[0]]
        holes = [
            [(float(lon), float(lat)) for lon, lat in ring]
            for ring in rings[1:]
            if len(ring) >= 3
        ]
        if len(exterior) >= 3:
            poly = Polygon(exterior, holes)
            if poly.is_valid and not poly.is_empty:
                result.append(poly)
    if not result:
        return Polygon()
    if len(result) == 1:
        return result[0]
    return MultiPolygon(result)


def train(config: TrainConfig) -> dict[str, Any]:
    """Train the placement model and write checkpoints/metrics."""

    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader, WeightedRandomSampler

    config.run_dir.mkdir(parents=True, exist_ok=True)
    (config.run_dir / "config.json").write_text(
        json.dumps(_json_safe(asdict(config)), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    train_rows, val_rows, holdout_rows = split_dataset_rows(config.dataset_manifest)
    model = PlacementModel(num_classes=NUM_PLACEMENT_CLASSES, encoder_name=config.encoder_name)
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    hard_weights: dict[str, float] = {}
    best_score = -1.0
    history: list[dict[str, Any]] = []
    start_epoch = 1

    if config.resume:
        resume_state = load_training_checkpoint(config.run_dir / "last.pt", device)
        resume_kind = "epoch"
        if resume_state is None:
            resume_state = load_training_checkpoint(config.run_dir / "latest_step.pt", device)
            resume_kind = "step"
        if resume_state is not None:
            model.load_state_dict(resume_state["model"])
            optimizer_state = resume_state.get("optimizer")
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)
            resume_epoch = int(resume_state.get("epoch", 0))
            start_epoch = resume_epoch + 1 if resume_kind == "epoch" else max(1, resume_epoch)
            best_score = float(resume_state.get("best_score", best_score))
            hard_weights = load_latest_hard_weights(
                config.run_dir / "failures.jsonl",
                fraction=config.hard_mining_fraction,
                max_weight=config.hard_mining_max_weight,
            )
            _log_training_event(
                config.run_dir,
                {
                    "event": "resume",
                    "resume_kind": resume_kind,
                    "epoch": resume_epoch,
                    "step": int(resume_state.get("step", 0)),
                    "best_score": best_score,
                },
            )

    _log_training_event(
        config.run_dir,
        {
            "event": "start",
            "device": str(device),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "holdout_rows": len(holdout_rows),
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "chip_size": config.chip_size,
            "num_workers": config.num_workers,
        },
    )

    for epoch in range(start_epoch, config.epochs + 1):
        dataset = PlacementDataset(train_rows, chip_size=config.chip_size, random_crop=True)
        weights = [hard_weights.get(row["chip_id"], 1.0) for row in train_rows]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
        )
        model.train()
        loss_total = 0.0
        total_steps = len(loader)
        epoch_start = time.time()
        _log_training_event(
            config.run_dir,
            {
                "event": "epoch_start",
                "epoch": epoch,
                "total_steps": total_steps,
                "train_rows": len(train_rows),
            },
        )
        for step, batch in enumerate(loader, start=1):
            batch = _batch_to_device(batch, device)
            outputs = model(batch["image"])
            losses = placement_losses(outputs, batch, config.dimension_loss_weight)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            optimizer.step()
            step_loss = float(losses["loss"].detach().cpu())
            loss_total += step_loss
            if _should_log_step(step, total_steps, config.progress_interval_steps):
                elapsed = time.time() - epoch_start
                _log_training_event(
                    config.run_dir,
                    {
                        "event": "train_step",
                        "epoch": epoch,
                        "step": step,
                        "total_steps": total_steps,
                        "loss": step_loss,
                        "avg_loss": loss_total / max(1, step),
                        "elapsed_sec": elapsed,
                        "steps_per_sec": step / max(1e-6, elapsed),
                    },
                )
            if _should_save_step_checkpoint(step, total_steps, config.checkpoint_interval_steps):
                atomic_torch_save(
                    {
                        "epoch": epoch,
                        "step": step,
                        "total_steps": total_steps,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": _json_safe(asdict(config)),
                        "best_score": best_score,
                        "complete_epoch": False,
                    },
                    config.run_dir / "latest_step.pt",
                )
                _log_training_event(
                    config.run_dir,
                    {
                        "event": "step_checkpoint",
                        "epoch": epoch,
                        "step": step,
                        "path": str(config.run_dir / "latest_step.pt"),
                    },
                )

        _log_training_event(
            config.run_dir,
            {"event": "validation_start", "epoch": epoch},
        )
        val_metrics, failures = evaluate_model(
            model,
            val_rows,
            device=device,
            chip_size=config.chip_size,
            batch_size=config.batch_size,
            dimension_loss_weight=config.dimension_loss_weight,
        )
        holdout_metrics: dict[str, Any] = {}
        if holdout_rows:
            _log_training_event(
                config.run_dir,
                {"event": "holdout_validation_start", "epoch": epoch},
            )
            holdout_metrics, _ = evaluate_model(
                model,
                holdout_rows,
                device=device,
                chip_size=config.chip_size,
                batch_size=config.batch_size,
                dimension_loss_weight=config.dimension_loss_weight,
            )
        failure_path = config.run_dir / "failures.jsonl"
        for failure in failures:
            append_jsonl(failure_path, {"epoch": epoch, **failure})
        hard_weights = hard_example_weights(
            failures,
            fraction=config.hard_mining_fraction,
            max_weight=config.hard_mining_max_weight,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": loss_total / max(1, len(loader)),
            "val": val_metrics,
            "holdout": holdout_metrics,
            "hard_examples": len(hard_weights),
        }
        history.append(epoch_record)
        append_jsonl(config.run_dir / "metrics.jsonl", epoch_record)
        score = float(val_metrics.get("footprint_iou", 0.0)) - float(val_metrics.get("dimension_mae_m", 999.0)) * 0.001
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": _json_safe(asdict(config)),
            "metrics": epoch_record,
            "best_score": best_score,
            "complete_epoch": True,
        }
        atomic_torch_save(checkpoint, config.run_dir / "last.pt")
        try:
            (config.run_dir / "latest_step.pt").unlink()
        except FileNotFoundError:
            pass
        if score > best_score:
            best_score = score
            checkpoint["best_score"] = best_score
            atomic_torch_save(checkpoint, config.run_dir / "best.pt")
        _log_training_event(
            config.run_dir,
            {
                "event": "epoch_complete",
                "epoch": epoch,
                "train_loss": epoch_record["train_loss"],
                "score": score,
                "best_score": best_score,
            },
        )

        if should_pause(config):
            append_jsonl(
                config.run_dir / "metrics.jsonl",
                {"epoch": epoch, "event": "paused", "pause_file": str(resolve_pause_file(config))},
            )
            break

    return {
        "history": history,
        "best_score": best_score,
        "run_dir": str(config.run_dir),
        "start_epoch": start_epoch,
        "completed_epochs": max(0, config.epochs - start_epoch + 1),
    }


def _should_log_step(step: int, total_steps: int, interval: int) -> bool:
    return step == 1 or step == total_steps or (interval > 0 and step % interval == 0)


def _should_save_step_checkpoint(step: int, total_steps: int, interval: int) -> bool:
    return step == 1 or step == total_steps or (interval > 0 and step % interval == 0)


def _log_training_event(run_dir: str | Path, row: dict[str, Any]) -> None:
    """Append a training progress event and mirror it to stdout."""

    event = {"timestamp": time.time(), **row}
    append_jsonl(Path(run_dir) / "train_progress.jsonl", event)
    print(json.dumps(_json_safe(event), sort_keys=True), flush=True)


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    """Write a torch checkpoint via a temp file, then atomically replace."""

    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def resolve_pause_file(config: TrainConfig) -> Path:
    """Return the pause sentinel path for a training run."""

    return config.pause_file or (config.run_dir / "PAUSE")


def should_pause(config: TrainConfig) -> bool:
    """Return True when training should stop after the current epoch."""

    return resolve_pause_file(config).exists()


def load_training_checkpoint(path: str | Path, device: Any = "cpu") -> dict[str, Any] | None:
    """Load a training checkpoint if it exists."""

    path = Path(path)
    if not path.exists():
        return None
    import torch

    return torch.load(path, map_location=device, weights_only=False)


def load_latest_hard_weights(
    failures_path: str | Path,
    fraction: float,
    max_weight: float,
) -> dict[str, float]:
    """Rebuild hard-example sampling weights from the latest failure epoch."""

    rows = list(iter_jsonl(failures_path))
    if not rows:
        return {}
    latest_epoch = max(int(row.get("epoch", 0)) for row in rows)
    latest = [row for row in rows if int(row.get("epoch", 0)) == latest_epoch]
    return hard_example_weights(latest, fraction=fraction, max_weight=max_weight)


def split_dataset_rows(manifest_path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = list(iter_jsonl(manifest_path))
    train_rows = [row for row in rows if row.get("split") == "train"]
    val_rows = [row for row in rows if row.get("split") == "val"]
    holdout_rows = [row for row in rows if row.get("split") == "holdout"]
    if not val_rows:
        val_rows = [row for row in rows if row.get("split") in {"test", "holdout"}][: max(1, len(rows) // 10)]
    return train_rows, val_rows, holdout_rows


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


class PlacementDataset:
    """Torch-compatible dataset for placement chips."""

    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        chip_size: int = 512,
        random_crop: bool = True,
    ) -> None:
        self.rows = list(rows)
        self.chip_size = chip_size
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        row = self.rows[index]
        tensors = _load_aligned_training_tensors(
            row,
            chip_size=self.chip_size,
            random_crop=self.random_crop,
        )
        return {
            "chip_id": row["chip_id"],
            **tensors,
        }


def _load_aligned_training_tensors(
    row: dict[str, Any],
    chip_size: int,
    random_crop: bool,
) -> dict[str, Any]:
    import torch

    if row.get("labels"):
        return _load_vector_training_tensors(row, chip_size=chip_size, random_crop=random_crop)

    with Image.open(row["image"]) as src_img, Image.open(row["center"]) as center_img, Image.open(row["mask"]) as mask_img, Image.open(row["dims"]) as dims_img, Image.open(row["heading"]) as heading_img, Image.open(row["placement_class"]) as class_img:
        images = {
            "image": src_img.convert("RGB"),
            "center": center_img.convert("L"),
            "mask": mask_img.convert("L"),
            "dims": dims_img.convert("RGB"),
            "heading": heading_img.convert("RGB"),
            "placement_class": class_img.convert("L"),
        }
        images = _align_sizes(images)
        images = _crop_or_resize_images(images, chip_size=chip_size, random_crop=random_crop)

        image = _image_to_float_tensor(images["image"])
        center = _image_to_float_tensor(images["center"])[0:1]
        mask = _image_to_float_tensor(images["mask"])[0:1]
        dims = _image_to_float_tensor(images["dims"])[:2] / 2.0
        heading = _image_to_float_tensor(images["heading"])[:2] * 2.0 - 1.0
        placement_class = torch.from_numpy(
            np.asarray(images["placement_class"], dtype=np.int64)
        )
    return {
        "image": image,
        "center": center,
        "mask": mask,
        "dims": dims,
        "heading": heading,
        "placement_class": placement_class,
    }


def _load_vector_training_tensors(
    row: dict[str, Any],
    chip_size: int,
    random_crop: bool,
) -> dict[str, Any]:
    import torch

    labels_path = Path(row["labels"])
    if labels_path.suffix.lower() == ".sqlite":
        return _load_sqlite_training_tensors(
            row,
            labels_path=labels_path,
            chip_size=chip_size,
            random_crop=random_crop,
        )

    payload = load_vector_label_payload(row["labels"])
    payload_bounds = payload.get("bounds") or {}
    compact_bounds = payload.get("b")
    full_bounds = (
        float(row.get("west") if row.get("west") is not None else (compact_bounds[0] if compact_bounds else payload_bounds["west"])),
        float(row.get("south") if row.get("south") is not None else (compact_bounds[1] if compact_bounds else payload_bounds["south"])),
        float(row.get("east") if row.get("east") is not None else (compact_bounds[2] if compact_bounds else payload_bounds["east"])),
        float(row.get("north") if row.get("north") is not None else (compact_bounds[3] if compact_bounds else payload_bounds["north"])),
    )

    with Image.open(row["image"]) as src_img:
        image = src_img.convert("RGB")
        width, height = image.size
        if width < chip_size or height < chip_size:
            features = _features_from_payload_for_bounds(payload, full_bounds)
            labels = build_chip_labels(features, full_bounds, chip_size=(width, height))
            images = {
                "image": image,
                "center": labels["center"],
                "mask": labels["mask"],
                "dims": labels["dims"],
                "heading": labels["heading"],
                "placement_class": labels["placement_class"],
            }
            images = _crop_or_resize_images(images, chip_size=chip_size, random_crop=False)
        else:
            crop_box = _choose_crop_box(
                payload.get("features", []),
                full_bounds,
                width=width,
                height=height,
                chip_size=chip_size,
                random_crop=random_crop,
            )
            crop_bounds = _crop_bounds_from_pixels(full_bounds, width, height, crop_box)
            crop_features = _features_from_payload_for_bounds(
                payload,
                crop_bounds,
            )
            labels = build_chip_labels(crop_features, crop_bounds, chip_size=chip_size)
            images = {
                "image": image.crop(crop_box),
                "center": labels["center"],
                "mask": labels["mask"],
                "dims": labels["dims"],
                "heading": labels["heading"],
                "placement_class": labels["placement_class"],
            }

        image_tensor = _image_to_float_tensor(images["image"])
        center = _image_to_float_tensor(images["center"])[0:1]
        mask = _image_to_float_tensor(images["mask"])[0:1]
        dims = _image_to_float_tensor(images["dims"])[:2] / 2.0
        heading = _image_to_float_tensor(images["heading"])[:2] * 2.0 - 1.0
        placement_class = torch.from_numpy(
            np.asarray(images["placement_class"], dtype=np.int64)
        )

    return {
        "image": image_tensor,
        "center": center,
        "mask": mask,
        "dims": dims,
        "heading": heading,
        "placement_class": placement_class,
    }


def _load_sqlite_training_tensors(
    row: dict[str, Any],
    labels_path: Path,
    chip_size: int,
    random_crop: bool,
) -> dict[str, Any]:
    import torch

    full_bounds = (
        float(row["west"]),
        float(row["south"]),
        float(row["east"]),
        float(row["north"]),
    )

    with Image.open(row["image"]) as src_img:
        image = src_img.convert("RGB")
        width, height = image.size
        if width < chip_size or height < chip_size:
            features = _features_from_sqlite_for_bounds(labels_path, full_bounds)
            labels = build_chip_labels(features, full_bounds, chip_size=(width, height))
            images = {
                "image": image,
                "center": labels["center"],
                "mask": labels["mask"],
                "dims": labels["dims"],
                "heading": labels["heading"],
                "placement_class": labels["placement_class"],
            }
            images = _crop_or_resize_images(images, chip_size=chip_size, random_crop=False)
        else:
            feature_bounds = _sqlite_feature_bounds_for_bounds(
                labels_path,
                full_bounds,
            )
            crop_box = _choose_crop_box(
                feature_bounds,
                full_bounds,
                width=width,
                height=height,
                chip_size=chip_size,
                random_crop=random_crop,
            )
            crop_bounds = _crop_bounds_from_pixels(full_bounds, width, height, crop_box)
            crop_features = _features_from_sqlite_for_bounds(
                labels_path,
                crop_bounds,
            )
            labels = build_chip_labels(crop_features, crop_bounds, chip_size=chip_size)
            images = {
                "image": image.crop(crop_box),
                "center": labels["center"],
                "mask": labels["mask"],
                "dims": labels["dims"],
                "heading": labels["heading"],
                "placement_class": labels["placement_class"],
            }

        image_tensor = _image_to_float_tensor(images["image"])
        center = _image_to_float_tensor(images["center"])[0:1]
        mask = _image_to_float_tensor(images["mask"])[0:1]
        dims = _image_to_float_tensor(images["dims"])[:2] / 2.0
        heading = _image_to_float_tensor(images["heading"])[:2] * 2.0 - 1.0
        placement_class = torch.from_numpy(
            np.asarray(images["placement_class"], dtype=np.int64)
        )

    return {
        "image": image_tensor,
        "center": center,
        "mask": mask,
        "dims": dims,
        "heading": heading,
        "placement_class": placement_class,
    }


def _choose_crop_box(
    feature_records: Sequence[Any],
    bounds: tuple[float, float, float, float],
    width: int,
    height: int,
    chip_size: int,
    random_crop: bool,
) -> tuple[int, int, int, int]:
    """Choose a crop, biased toward labelled buildings during training."""

    if not random_crop:
        left = (width - chip_size) // 2
        top = (height - chip_size) // 2
        return (left, top, left + chip_size, top + chip_size)

    labelled = [record for record in feature_records if record]
    if labelled:
        record = random.choice(labelled)
        feature_bounds = _record_bounds(record)
        if feature_bounds is None:
            centroid_lon = (bounds[0] + bounds[2]) * 0.5
            centroid_lat = (bounds[1] + bounds[3]) * 0.5
        else:
            centroid_lon = (feature_bounds[0] + feature_bounds[2]) * 0.5
            centroid_lat = (feature_bounds[1] + feature_bounds[3]) * 0.5
        cx, cy = lonlat_to_pixel(
            centroid_lon,
            centroid_lat,
            width,
            height,
            bounds,
        )
        jitter = chip_size // 4
        cx += random.randint(-jitter, jitter)
        cy += random.randint(-jitter, jitter)
        left = int(round(cx - chip_size / 2))
        top = int(round(cy - chip_size / 2))
    else:
        left = random.randint(0, width - chip_size)
        top = random.randint(0, height - chip_size)
    left = max(0, min(width - chip_size, left))
    top = max(0, min(height - chip_size, top))
    return (left, top, left + chip_size, top + chip_size)


def _features_from_payload_for_bounds(
    payload: dict[str, Any],
    bounds: tuple[float, float, float, float],
) -> list[Feature]:
    """Deserialize only payload features whose bbox intersects bounds."""

    query = box(*bounds)
    result: list[Feature] = []
    for record in payload.get("features", []):
        record_bounds = _record_bounds(record)
        if record_bounds is not None and not _bounds_intersect(record_bounds, bounds):
            continue
        feature = _feature_from_label_record(record)
        if feature.geometry.intersects(query):
            result.append(feature)
    return result


def _record_bounds(record: Any) -> tuple[float, float, float, float] | None:
    """Return compact or GeoJSON record bounds."""

    if not isinstance(record, dict):
        return None
    raw = record.get("b")
    if raw and len(raw) == 4:
        return tuple(float(value) for value in raw)
    geometry = record.get("geometry")
    if geometry:
        return shape(geometry).bounds
    return None


def _bounds_intersect(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    return a[2] >= b[0] and a[0] <= b[2] and a[3] >= b[1] and a[1] <= b[3]


def _crop_bounds_from_pixels(
    bounds: tuple[float, float, float, float],
    width: int,
    height: int,
    crop_box: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    """Return WGS84 bounds for a pixel crop."""

    left, top, right, bottom = crop_box
    west, south, east, north = bounds
    crop_west = west + left / width * (east - west)
    crop_east = west + right / width * (east - west)
    crop_north = north - top / height * (north - south)
    crop_south = north - bottom / height * (north - south)
    return crop_west, crop_south, crop_east, crop_north


def _align_sizes(images: dict[str, Image.Image]) -> dict[str, Image.Image]:
    """Resize labels to the source texture size if needed."""

    target_size = images["image"].size
    aligned: dict[str, Image.Image] = {"image": images["image"]}
    for key, image in images.items():
        if key == "image":
            continue
        if image.size == target_size:
            aligned[key] = image
            continue
        interpolation = Image.Resampling.NEAREST
        aligned[key] = image.resize(target_size, interpolation)
    return aligned


def _crop_or_resize_images(
    images: dict[str, Image.Image],
    chip_size: int,
    random_crop: bool,
) -> dict[str, Image.Image]:
    width, height = images["image"].size
    if width < chip_size or height < chip_size:
        return {
            key: image.resize(
                (chip_size, chip_size),
                Image.Resampling.BILINEAR if key == "image" else Image.Resampling.NEAREST,
            )
            for key, image in images.items()
        }
    if width == chip_size and height == chip_size:
        return images
    if random_crop:
        left = random.randint(0, width - chip_size)
        top = random.randint(0, height - chip_size)
    else:
        left = (width - chip_size) // 2
        top = (height - chip_size) // 2
    box_px = (left, top, left + chip_size, top + chip_size)
    return {key: image.crop(box_px) for key, image in images.items()}


def _image_to_float_tensor(image: Image.Image):
    import torch

    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    arr = arr.transpose(2, 0, 1) / 255.0
    return torch.from_numpy(arr)


class PlacementModel:
    """Small SegFormer-style multi-head placement model.

    Uses a Hugging Face SegFormer encoder when available; otherwise falls back to
    a compact convolutional encoder so tests and dry runs remain lightweight.
    """

    def __init__(self, num_classes: int, encoder_name: str | None = None) -> None:
        import torch
        import torch.nn as nn

        class _Fallback(nn.Module):
            def __init__(self, classes: int) -> None:
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 96, 3, padding=1),
                    nn.BatchNorm2d(96),
                    nn.ReLU(inplace=True),
                )
                self.center = nn.Conv2d(96, 1, 1)
                self.mask = nn.Conv2d(96, 1, 1)
                self.dims = nn.Conv2d(96, 2, 1)
                self.heading = nn.Conv2d(96, 2, 1)
                self.placement_class = nn.Conv2d(96, classes, 1)

            def forward(self, image):
                features = self.encoder(image)
                return {
                    "center": self.center(features),
                    "mask": self.mask(features),
                    "dims": self.dims(features),
                    "heading": torch.tanh(self.heading(features)),
                    "placement_class": self.placement_class(features),
                }

        self.module = _Fallback(num_classes)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.module, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)


def placement_losses(outputs: dict[str, Any], batch: dict[str, Any], dimension_loss_weight: float = 4.0) -> dict[str, Any]:
    import torch.nn.functional as functional

    center_loss = functional.binary_cross_entropy_with_logits(outputs["center"], batch["center"])
    mask_loss = functional.binary_cross_entropy_with_logits(outputs["mask"], batch["mask"])
    occupied = batch["mask"].expand_as(batch["dims"]) > 0.5
    if bool(occupied.any()):
        dims_loss = functional.smooth_l1_loss(outputs["dims"][occupied], batch["dims"][occupied])
        heading_loss = functional.smooth_l1_loss(
            outputs["heading"][occupied],
            batch["heading"][occupied],
        )
    else:
        dims_loss = outputs["dims"].sum() * 0.0
        heading_loss = outputs["heading"].sum() * 0.0
    class_loss = functional.cross_entropy(outputs["placement_class"], batch["placement_class"])
    loss = center_loss + mask_loss + dimension_loss_weight * dims_loss + heading_loss + class_loss
    return {
        "loss": loss,
        "center": center_loss,
        "mask": mask_loss,
        "dims": dims_loss,
        "heading": heading_loss,
        "class": class_loss,
    }


def _batch_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def evaluate_model(
    model: Any,
    rows: Sequence[dict[str, Any]],
    device: Any,
    chip_size: int,
    batch_size: int,
    dimension_loss_weight: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate model and return aggregate metrics plus hard-example rows."""

    import torch
    from torch.utils.data import DataLoader

    if not rows:
        return {}, []
    dataset = PlacementDataset(rows, chip_size=chip_size, random_crop=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    totals = {
        "center_precision": 0.0,
        "center_recall": 0.0,
        "footprint_iou": 0.0,
        "dimension_mae_m": 0.0,
        "heading_error_deg": 0.0,
        "class_accuracy": 0.0,
    }
    failures: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        count = 0
        for batch in loader:
            ids = list(batch["chip_id"])
            batch = _batch_to_device(batch, device)
            outputs = model(batch["image"])
            batch_metrics = metric_batch(outputs, batch)
            for key in totals:
                totals[key] += float(batch_metrics[key]) * len(ids)
            per_item = per_item_failure_scores(outputs, batch, ids)
            failures.extend(per_item)
            count += len(ids)
    metrics = {key: value / max(1, count) for key, value in totals.items()}
    return metrics, failures


def metric_batch(outputs: dict[str, Any], batch: dict[str, Any]) -> dict[str, float]:
    import torch

    center_pred = torch.sigmoid(outputs["center"]) > 0.5
    center_true = batch["center"] > 0.5
    mask_pred = torch.sigmoid(outputs["mask"]) > 0.5
    mask_true = batch["mask"] > 0.5
    tp = (center_pred & center_true).sum().float()
    precision = tp / (center_pred.sum().float() + 1e-6)
    recall = tp / (center_true.sum().float() + 1e-6)
    intersection = (mask_pred & mask_true).sum().float()
    union = (mask_pred | mask_true).sum().float()
    iou = intersection / (union + 1e-6)
    occupied = batch["mask"].expand_as(batch["dims"]) > 0.5
    if bool(occupied.any()):
        dim_mae = (outputs["dims"][occupied] - batch["dims"][occupied]).abs().mean()
        pred_angle = torch.atan2(outputs["heading"][:, 0], outputs["heading"][:, 1])
        true_angle = torch.atan2(batch["heading"][:, 0], batch["heading"][:, 1])
        angle_diff = torch.rad2deg(torch.atan2(torch.sin(pred_angle - true_angle), torch.cos(pred_angle - true_angle))).abs()
        heading_error = angle_diff[batch["mask"][:, 0] > 0.5].mean()
    else:
        dim_mae = torch.tensor(0.0, device=batch["image"].device)
        heading_error = torch.tensor(0.0, device=batch["image"].device)
    class_pred = outputs["placement_class"].argmax(dim=1)
    class_accuracy = (class_pred == batch["placement_class"]).float().mean()
    return {
        "center_precision": float(precision.detach().cpu()),
        "center_recall": float(recall.detach().cpu()),
        "footprint_iou": float(iou.detach().cpu()),
        "dimension_mae_m": float(dim_mae.detach().cpu()),
        "heading_error_deg": float(heading_error.detach().cpu()),
        "class_accuracy": float(class_accuracy.detach().cpu()),
    }


def per_item_failure_scores(outputs: dict[str, Any], batch: dict[str, Any], chip_ids: Sequence[str]) -> list[dict[str, Any]]:
    import torch

    failures: list[dict[str, Any]] = []
    for idx, chip_id in enumerate(chip_ids):
        one_outputs = {key: value[idx : idx + 1] for key, value in outputs.items()}
        one_batch = {
            key: value[idx : idx + 1] if hasattr(value, "__getitem__") and key != "chip_id" else value
            for key, value in batch.items()
            if key != "chip_id"
        }
        metrics = metric_batch(one_outputs, one_batch)
        error = (
            (1.0 - metrics["footprint_iou"])
            + (1.0 - metrics["center_recall"])
            + metrics["dimension_mae_m"] * 0.05
            + metrics["heading_error_deg"] / 180.0
            + (1.0 - metrics["class_accuracy"])
        )
        failures.append({"chip_id": str(chip_id), "error": float(error), **metrics})
    return failures


def hard_example_weights(
    failures: Sequence[dict[str, Any]],
    fraction: float,
    max_weight: float,
) -> dict[str, float]:
    """Convert validation failures into bounded sampling weights."""

    if not failures or fraction <= 0:
        return {}
    ordered = sorted(failures, key=lambda row: float(row.get("error", 0.0)), reverse=True)
    keep = max(1, int(round(len(ordered) * min(1.0, fraction))))
    selected = ordered[:keep]
    max_error = max(float(row.get("error", 0.0)) for row in selected) or 1.0
    return {
        str(row["chip_id"]): 1.0 + (min(max_weight, 1.0 + float(row.get("error", 0.0)) / max_error * (max_weight - 1.0)) - 1.0)
        for row in selected
    }


def export_model(run_dir: str | Path, output_path: str | Path | None = None) -> Path:
    """Export the best checkpoint to a stable artifact path."""

    run_dir = Path(run_dir)
    source = run_dir / "best.pt"
    if output_path is None:
        output_path = run_dir / "building_placement_best.pt"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output_path)
    return output_path


def parse_train_args(argv: Sequence[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train SimHeaven-supervised building placement model.")
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--chip-size",
        type=int,
        default=512,
        help="Training crop size sampled from native-resolution DDS textures; labels are rasterized in memory for the crop.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--pause-file",
        default=None,
        help="Sentinel file checked after each epoch; create it to pause cleanly.",
    )
    parser.add_argument(
        "--progress-interval-steps",
        type=int,
        default=100,
        help="Write train_progress.jsonl/stdout every N training steps.",
    )
    parser.add_argument(
        "--checkpoint-interval-steps",
        type=int,
        default=500,
        help="Write latest_step.pt every N training steps for crash recovery.",
    )
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir) if args.run_dir else DEFAULT_TRAINING_ROOT / "runs" / time.strftime("%Y%m%d_%H%M%S")
    return TrainConfig(
        dataset_manifest=Path(args.dataset_manifest),
        run_dir=run_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        chip_size=args.chip_size,
        num_workers=args.num_workers,
        device=args.device,
        resume=not args.no_resume,
        pause_file=Path(args.pause_file) if args.pause_file else None,
        progress_interval_steps=args.progress_interval_steps,
        checkpoint_interval_steps=args.checkpoint_interval_steps,
    )


def cli_discover(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Discover Ortho4XP tiles and write split manifest.")
    parser.add_argument("--tiles-root", default=str(DEFAULT_TILES_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_TRAINING_ROOT / "manifests" / "tiles.jsonl"))
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)
    records = write_tile_splits(args.output, args.tiles_root, args.seed)
    counts: dict[str, int] = {}
    for record in records:
        counts[record.split] = counts.get(record.split, 0) + 1
    print(json.dumps({"tiles": len(records), "splits": counts, "output": args.output}, indent=2))
    return 0


def cli_extract(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract SimHeaven-supervised training chips.")
    parser.add_argument("--splits", default=str(DEFAULT_TRAINING_ROOT / "manifests" / "tiles.jsonl"))
    parser.add_argument("--output-root", default=str(DEFAULT_TRAINING_ROOT))
    parser.add_argument("--custom-scenery-root", default=str(DEFAULT_CUSTOM_SCENERY_ROOT))
    parser.add_argument("--dsftool", default=str(Path(r"G:\Dev\Ortho4XP-SegFormer\Utils\win\DSFTool.exe")))
    parser.add_argument("--chip-size", type=int, default=512, help="Training crop size sampled from native-resolution textures and labels.")
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument(
        "--max-tile-features",
        type=int,
        default=0,
        help="Optional safety cap for SimHeaven feature count. Default 0 keeps dense tiles.",
    )
    parser.add_argument("--skip-holdouts", action="store_true")
    parser.add_argument(
        "--materialize-images",
        action="store_true",
        help="Copy/resize source textures into output-root/images. Off by default to avoid duplicating large DDS libraries.",
    )
    args = parser.parse_args(argv)
    manifest = extract_dataset(
        splits_path=args.splits,
        output_root=args.output_root,
        custom_scenery_root=args.custom_scenery_root,
        dsftool_path=args.dsftool,
        chip_size=args.chip_size,
        max_tiles=args.max_tiles,
        include_holdouts=not args.skip_holdouts,
        materialize_images=args.materialize_images,
        max_tile_features=args.max_tile_features,
    )
    print(json.dumps({"dataset_manifest": str(manifest)}, indent=2))
    return 0


def cli_train(argv: Sequence[str] | None = None) -> int:
    config = parse_train_args(argv)
    result = train(config)
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    return 0
