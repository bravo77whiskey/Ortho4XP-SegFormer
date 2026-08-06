"""Recover Ortho4XP custom zones from backups and texture artifacts."""

from __future__ import annotations

import ast
import math
import re
import shutil
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path


TILE_RE = re.compile(r"^zOrtho4XP_(?P<lat>[+-]\d{2})(?P<lon>[+-]\d{3})$")
DDS_RE = re.compile(
    r"^(?P<y>\d+)_(?P<x>\d+)_(?P<provider>.+?)(?P<zl>\d{2})\.dds$",
    re.IGNORECASE,
)
MASK_RE = re.compile(
    r"^(?P<y>\d+)_(?P<x>\d+)_ZL(?P<zl>\d{2})\.png$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TextureRecord:
    path: Path
    x_left: int
    y_top: int
    provider: str
    zoomlevel: int
    mtime: float

    @property
    def footprint_key(self) -> tuple[int, int, int]:
        return (self.x_left, self.y_top, self.zoomlevel)


@dataclass(frozen=True)
class TileConfig:
    default_website: str
    default_zl: int
    values: dict[str, str]
    zone_list: list = field(default_factory=list)


def short_latlon(lat: int, lon: int) -> str:
    return f"{lat:+03d}{lon:+04d}"


def tile_coordinates(tile_dir: Path) -> tuple[int, int]:
    match = TILE_RE.match(tile_dir.name)
    if not match:
        raise ValueError(f"Cannot infer tile coordinates from {tile_dir.name}")
    return int(match.group("lat")), int(match.group("lon"))


def gtile_to_wgs84(til_x: int, til_y: int, zoomlevel: int) -> tuple[float, float]:
    rat_x = til_x / (2 ** (zoomlevel - 1)) - 1
    rat_y = 1 - til_y / (2 ** (zoomlevel - 1))
    lon = rat_x * 180
    lat = 360 / math.pi * math.atan(math.exp(math.pi * rat_y)) - 90
    return lat, lon


def parse_cfg(cfg_path: Path) -> TileConfig:
    values: dict[str, str] = {}
    if cfg_path.exists():
        for line in cfg_path.read_text(encoding="utf-8").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    try:
        zone_list = ast.literal_eval(values.get("zone_list", "[]"))
    except (SyntaxError, ValueError):
        zone_list = []
    return TileConfig(
        default_website=values.get("default_website", ""),
        default_zl=int(values.get("default_zl", "16")),
        values=values,
        zone_list=zone_list,
    )


def discover_tile_dirs(paths: list[Path]) -> list[Path]:
    tile_dirs: list[Path] = []
    for path in paths:
        path = Path(path).resolve()
        if TILE_RE.match(path.name):
            tile_dirs.append(path)
            continue
        if not path.is_dir():
            continue
        tile_dirs.extend(
            child for child in sorted(path.iterdir()) if TILE_RE.match(child.name)
        )
    return sorted(set(tile_dirs))


def parse_texture(path: Path) -> TextureRecord | None:
    match = DDS_RE.match(path.name)
    if not match:
        return None
    return TextureRecord(
        path=path,
        x_left=int(match.group("x")),
        y_top=int(match.group("y")),
        provider=match.group("provider"),
        zoomlevel=int(match.group("zl")),
        mtime=path.stat().st_mtime,
    )


def parse_mask(path: Path) -> TextureRecord | None:
    """Parse a texture mask; its provider is inferred during reconstruction."""
    match = MASK_RE.match(path.name)
    if not match:
        return None
    return TextureRecord(
        path=path,
        x_left=int(match.group("x")),
        y_top=int(match.group("y")),
        provider="",
        zoomlevel=int(match.group("zl")),
        mtime=path.stat().st_mtime,
    )


def winning_textures(textures_dir: Path) -> tuple[list[TextureRecord], int]:
    by_footprint: dict[tuple[int, int, int], TextureRecord] = {}
    duplicate_count = 0
    if not textures_dir.is_dir():
        return [], 0
    paths = list(textures_dir.glob("*.dds")) + list(textures_dir.glob("*.png"))
    for path in paths:
        record = (
            parse_texture(path)
            if path.suffix.lower() == ".dds"
            else parse_mask(path)
        )
        if record is None:
            continue
        current = by_footprint.get(record.footprint_key)
        if current is not None:
            duplicate_count += 1
        if current is None:
            by_footprint[record.footprint_key] = record
            continue
        if (record.mtime, record.path.name) > (current.mtime, current.path.name):
            if not record.provider and current.provider:
                record = replace(record, provider=current.provider)
            by_footprint[record.footprint_key] = record
        elif not current.provider and record.provider:
            by_footprint[record.footprint_key] = replace(
                current,
                provider=record.provider,
            )
    return list(by_footprint.values()), duplicate_count


def texture_to_zone(
    record: TextureRecord,
    tile_lat: int,
    tile_lon: int,
) -> list[object] | None:
    lat_max, lon_min = gtile_to_wgs84(record.x_left, record.y_top, record.zoomlevel)
    lat_min, lon_max = gtile_to_wgs84(
        record.x_left + 16,
        record.y_top + 16,
        record.zoomlevel,
    )
    lat_min = max(lat_min, tile_lat)
    lat_max = min(lat_max, tile_lat + 1)
    lon_min = max(lon_min, tile_lon)
    lon_max = min(lon_max, tile_lon + 1)
    if lat_min >= lat_max or lon_min >= lon_max:
        return None
    return [
        [
            lat_min,
            lon_min,
            lat_min,
            lon_max,
            lat_max,
            lon_max,
            lat_max,
            lon_min,
            lat_min,
            lon_min,
        ],
        record.zoomlevel,
        record.provider,
    ]


def zone_intersects_tile(zone: list, lat: int, lon: int) -> bool:
    try:
        coordinates = zone[0]
        zone_lats = coordinates[0::2]
        zone_lons = coordinates[1::2]
        return (
            max(zone_lats) > lat
            and min(zone_lats) < lat + 1
            and max(zone_lons) > lon
            and min(zone_lons) < lon + 1
        )
    except (IndexError, TypeError, ValueError):
        return False


def reconstruct_zone_list(
    tile_dir: Path,
    include_default_textures: bool = False,
    lat: int | None = None,
    lon: int | None = None,
) -> dict[str, object]:
    tile_dir = Path(tile_dir)
    if lat is None or lon is None:
        lat, lon = tile_coordinates(tile_dir)
    cfg_path = tile_dir / f"Ortho4XP_{short_latlon(lat, lon)}.cfg"
    base_result = {
        "tile": short_latlon(lat, lon),
        "lat": lat,
        "lon": lon,
        "cfg_path": str(cfg_path),
        "duplicate_footprints": 0,
        "default_matching_textures": 0,
        "zone_count": 0,
        "zone_list": [],
        "selected_textures": [],
    }
    if not cfg_path.exists():
        return {**base_result, "source": "skip", "reason": "no config"}

    cfg = parse_cfg(cfg_path)
    textures_dir = tile_dir / "textures"
    if textures_dir.is_dir():
        texture_count = len(list(textures_dir.glob("*.dds")))
        mask_count = len(
            [path for path in textures_dir.glob("*.png") if MASK_RE.match(path.name)]
        )
    else:
        texture_count = 0
        mask_count = 0
    common = {
        **base_result,
        "default_website": cfg.default_website,
        "default_zl": cfg.default_zl,
        "textures": texture_count,
        "texture_masks": mask_count,
    }
    if cfg.zone_list:
        return {
            **common,
            "source": "current",
            "zone_count": len(cfg.zone_list),
            "zone_list": cfg.zone_list,
        }

    backup_path = cfg_path.with_suffix(cfg_path.suffix + ".bak")
    if backup_path.exists():
        backup_cfg = parse_cfg(backup_path)
        if backup_cfg.zone_list:
            return {
                **common,
                "source": "backup",
                "zone_count": len(backup_cfg.zone_list),
                "zone_list": backup_cfg.zone_list,
            }

    winners, duplicate_count = winning_textures(textures_dir)
    zones: list[list[object]] = []
    selected: list[dict[str, object]] = []
    default_matching_textures = 0
    ordered = sorted(
        winners,
        key=lambda item: (
            -item.zoomlevel,
            -item.mtime,
            item.y_top,
            item.x_left,
            item.provider,
        ),
    )
    for record in ordered:
        provider = record.provider or cfg.default_website
        zone = texture_to_zone(record, lat, lon)
        if zone is None:
            continue
        matches_defaults = (
            provider == cfg.default_website
            and record.zoomlevel == cfg.default_zl
        )
        if matches_defaults:
            default_matching_textures += 1
        if not include_default_textures and matches_defaults:
            continue
        zone[2] = provider
        zones.append(zone)
        selected.append(
            {
                "file": record.path.name,
                "provider": provider,
                "format": record.path.suffix.lower().lstrip("."),
                "zoomlevel": record.zoomlevel,
                "x_left": record.x_left,
                "y_top": record.y_top,
                "mtime": datetime.fromtimestamp(record.mtime).isoformat(
                    timespec="seconds"
                ),
            }
        )
    return {
        **common,
        "source": "textures",
        "duplicate_footprints": duplicate_count,
        "default_matching_textures": default_matching_textures,
        "zone_count": len(zones),
        "zone_list": zones,
        "selected_textures": selected,
    }


def write_zone_list(cfg_path: Path, zone_list: list[list[object]]) -> Path:
    cfg_path = Path(cfg_path)
    backup_path = cfg_path.with_suffix(cfg_path.suffix + ".bak")
    if cfg_path.exists():
        if backup_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = cfg_path.with_suffix(cfg_path.suffix + f".bak.{timestamp}")
            suffix = 1
            while backup_path.exists():
                backup_path = cfg_path.with_suffix(
                    cfg_path.suffix + f".bak.{timestamp}.{suffix}"
                )
                suffix += 1
        shutil.copy2(cfg_path, backup_path)
        lines = cfg_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    rendered = "zone_list=" + repr(zone_list)
    new_lines: list[str] = []
    wrote = False
    for line in lines:
        if line.startswith("zone_list="):
            new_lines.append(rendered)
            wrote = True
        else:
            new_lines.append(line)
    if not wrote:
        new_lines.append(rendered)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return backup_path
