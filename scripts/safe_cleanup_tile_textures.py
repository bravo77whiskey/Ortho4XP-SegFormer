from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw


TILE_RE = re.compile(r"^zOrtho4XP_(?P<lat>[+-]\d{2})(?P<lon>[+-]\d{3})$")
DDS_RE = re.compile(
    r"^(?P<y>\d+)_(?P<x>\d+)_(?P<provider>.+?)(?P<zl>\d{2})\.dds$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TileConfig:
    default_website: str
    default_zl: int
    mesh_zl: int
    zone_list: list[list[object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete Ortho4XP DDS textures that are not implied by each tile's "
            "own config file."
        )
    )
    parser.add_argument("--tiles-root", required=True, type=Path)
    parser.add_argument("--lat-min", required=True, type=int)
    parser.add_argument("--lat-max", required=True, type=int)
    parser.add_argument("--lon-min", required=True, type=int)
    parser.add_argument("--lon-max", required=True, type=int)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSON manifest output path. Defaults under analysis_tmp/.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files listed by the config-only comparison.",
    )
    return parser.parse_args()


def gtile_to_wgs84(til_x: int, til_y: int, zoomlevel: int) -> tuple[float, float]:
    rat_x = (til_x / (2 ** (zoomlevel - 1)) - 1)
    rat_y = 1 - til_y / (2 ** (zoomlevel - 1))
    lon = rat_x * 180
    lat = 360 / math.pi * math.atan(math.exp(math.pi * rat_y)) - 90
    return lat, lon


def wgs84_to_orthogrid(lat: float, lon: float, zoomlevel: int) -> tuple[int, int]:
    ratio_x = lon / 180
    ratio_y = math.log(math.tan((90 + lat) * math.pi / 360)) / math.pi
    mult = 2 ** (zoomlevel - 5)
    til_x = int((ratio_x + 1) * mult) * 16
    til_y = int((1 - ratio_y) * mult) * 16
    return til_x, til_y


def dds_name(x_left: int, y_top: int, zoomlevel: int, provider: str) -> str:
    return f"{y_top}_{x_left}_{provider}{zoomlevel}.dds"


def parse_cfg(cfg_path: Path) -> TileConfig:
    values: dict[str, str] = {}
    for line in cfg_path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    zone_list_raw = values.get("zone_list", "[]")
    try:
        zone_list = ast.literal_eval(zone_list_raw)
    except (SyntaxError, ValueError):
        zone_list = []
    return TileConfig(
        default_website=values.get("default_website", ""),
        default_zl=int(values.get("default_zl", "16")),
        mesh_zl=int(values.get("mesh_zl", "19")),
        zone_list=zone_list,
    )


def expected_dds_for_tile(lat: int, lon: int, cfg: TileConfig) -> set[str]:
    mask_image = Image.new("L", (4096, 4096), color=0)
    draw = ImageDraw.Draw(mask_image)
    zone_lookup: dict[int, tuple[int, str]] = {}

    base_zone = (
        (lat, lon, lat, lon + 1, lat + 1, lon + 1, lat + 1, lon, lat, lon),
        cfg.default_zl,
        cfg.default_website,
    )

    next_zone_id = 1
    for region in [base_zone, *reversed(cfg.zone_list)]:
        zone_lookup[next_zone_id] = (int(region[1]), str(region[2]))
        polygon = [
            (round((x - lon) * 4095), round((lat + 1 - y) * 4095))
            for x, y in zip(region[0][1::2], region[0][::2])
        ]
        draw.polygon(polygon, fill=next_zone_id)
        next_zone_id += 1

    expected: set[str] = set()
    til_x_min, til_y_min = wgs84_to_orthogrid(lat + 1, lon, cfg.mesh_zl)
    til_x_max, til_y_max = wgs84_to_orthogrid(lat, lon + 1, cfg.mesh_zl)

    for til_x in range(til_x_min, til_x_max + 1, 16):
        for til_y in range(til_y_min, til_y_max + 1, 16):
            lat_point, lon_point = gtile_to_wgs84(til_x + 8, til_y + 8, cfg.mesh_zl)
            lon_point = max(min(lon_point, lon + 1), lon)
            lat_point = max(min(lat_point, lat + 1), lat)
            px = round((lon_point - lon) * 4095)
            py = round((lat + 1 - lat_point) * 4095)
            mask_idx = mask_image.getpixel((px, py))
            zoomlevel, provider = zone_lookup[mask_idx]
            til_x_text = 16 * ((int(til_x / 2 ** (cfg.mesh_zl - zoomlevel))) // 16)
            til_y_text = 16 * ((int(til_y / 2 ** (cfg.mesh_zl - zoomlevel))) // 16)
            expected.add(dds_name(til_x_text, til_y_text, zoomlevel, provider))
    return expected


def default_manifest_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("analysis_tmp") / f"tile_texture_cleanup_{timestamp}.json"


def scan_tiles(args: argparse.Namespace) -> dict[str, object]:
    tiles_root = args.tiles_root
    results: list[dict[str, object]] = []
    total_deleted = 0

    for tile_dir in sorted(tiles_root.iterdir()):
        if not tile_dir.is_dir():
            continue
        match = TILE_RE.match(tile_dir.name)
        if not match:
            continue
        lat = int(match.group("lat"))
        lon = int(match.group("lon"))
        if lat < args.lat_min or lat > args.lat_max or lon < args.lon_min or lon > args.lon_max:
            continue

        cfg_path = tile_dir / f"Ortho4XP_{match.group('lat')}{match.group('lon')}.cfg"
        textures_dir = tile_dir / "textures"
        if not cfg_path.exists() or not textures_dir.exists():
            continue

        cfg = parse_cfg(cfg_path)
        expected = expected_dds_for_tile(lat, lon, cfg)
        actual = {
            path.name: path
            for path in textures_dir.glob("*.dds")
            if DDS_RE.match(path.name)
        }
        excess_names = sorted(set(actual) - expected)
        if not excess_names:
            continue

        provider_counts = Counter(
            DDS_RE.match(name).group("provider") for name in excess_names if DDS_RE.match(name)
        )
        total_deleted += len(excess_names)
        results.append(
            {
                "tile": tile_dir.name,
                "lat": lat,
                "lon": lon,
                "config": {
                    "default_website": cfg.default_website,
                    "default_zl": cfg.default_zl,
                    "mesh_zl": cfg.mesh_zl,
                    "zone_providers": sorted({str(zone[2]) for zone in cfg.zone_list}),
                },
                "counts": {
                    "actual_dds": len(actual),
                    "expected_dds": len(expected),
                    "excess_dds": len(excess_names),
                },
                "excess_provider_counts": dict(sorted(provider_counts.items())),
                "delete_files": [
                    str(actual[name].resolve()) for name in excess_names
                ],
            }
        )

    return {
        "tiles_root": str(tiles_root.resolve()),
        "bounds": {
            "lat_min": args.lat_min,
            "lat_max": args.lat_max,
            "lon_min": args.lon_min,
            "lon_max": args.lon_max,
        },
        "mode": "apply" if args.apply else "dry-run",
        "tile_count": len(results),
        "delete_count": total_deleted,
        "tiles": results,
    }


def delete_files(manifest: dict[str, object]) -> int:
    deleted = 0
    for tile in manifest["tiles"]:
        for file_name in tile["delete_files"]:
            path = Path(file_name)
            if not path.exists():
                continue
            path.unlink()
            deleted += 1
    return deleted


def print_summary(manifest: dict[str, object], deleted: int | None = None) -> None:
    print(
        f"{manifest['mode']}: {manifest['tile_count']} tile(s), "
        f"{manifest['delete_count']} config-invalid DDS file(s)"
    )
    for tile in manifest["tiles"]:
        counts = tile["counts"]
        providers = ", ".join(
            f"{name}:{count}" for name, count in tile["excess_provider_counts"].items()
        )
        print(
            f"- {tile['tile']}: {counts['excess_dds']} excess "
            f"({providers or 'unknown providers'})"
        )
    if deleted is not None:
        print(f"deleted: {deleted}")


def main() -> int:
    args = parse_args()
    args.tiles_root = args.tiles_root.resolve()
    manifest_path = (args.manifest or default_manifest_path()).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = scan_tiles(args)
    deleted = delete_files(manifest) if args.apply else None
    manifest["deleted_count"] = deleted
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print_summary(manifest, deleted=deleted)
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
