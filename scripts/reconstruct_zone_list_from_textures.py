from __future__ import annotations

import argparse
import ast
import json
import math
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


TILE_RE = re.compile(r"^zOrtho4XP_(?P<lat>[+-]\d{2})(?P<lon>[+-]\d{3})$")
DDS_RE = re.compile(
    r"^(?P<y>\d+)_(?P<x>\d+)_(?P<provider>.+?)(?P<zl>\d{2})\.dds$",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct Ortho4XP tile zone_list entries from existing DDS "
            "textures. Duplicate files with the same x/y/ZL footprint are "
            "resolved by newest modification time."
        )
    )
    parser.add_argument(
        "tile_dirs",
        nargs="*",
        type=Path,
        help=(
            "One or more zOrtho4XP tile directories, or a Tiles root containing tile "
            "directories. Defaults to the current directory when omitted."
        ),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the reconstructed zone_list into each tile cfg. Default is dry-run.",
    )
    parser.add_argument(
        "--include-default-textures",
        action="store_true",
        help=(
            "Include winning textures that match the tile default_website/default_zl. "
            "By default those base textures are omitted from zone_list."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSON manifest path. Defaults under analysis_tmp/.",
    )
    return parser.parse_args()


def short_latlon(lat: int, lon: int) -> str:
    return f"{lat:+03d}{lon:+04d}"


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
        path = path.resolve()
        if TILE_RE.match(path.name):
            tile_dirs.append(path)
            continue
        if not path.is_dir():
            continue
        tile_dirs.extend(child for child in sorted(path.iterdir()) if TILE_RE.match(child.name))
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


def winning_textures(textures_dir: Path) -> tuple[list[TextureRecord], int]:
    by_footprint: dict[tuple[int, int, int], TextureRecord] = {}
    duplicate_count = 0
    for path in textures_dir.glob("*.dds"):
        record = parse_texture(path)
        if record is None:
            continue
        current = by_footprint.get(record.footprint_key)
        if current is not None:
            duplicate_count += 1
        if current is None or record.mtime > current.mtime:
            by_footprint[record.footprint_key] = record
    return list(by_footprint.values()), duplicate_count


def texture_to_zone(record: TextureRecord, tile_lat: int, tile_lon: int) -> list[object] | None:
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


def reconstruct_zone_list(
    tile_dir: Path,
    include_default_textures: bool = False,
) -> dict[str, object]:
    match = TILE_RE.match(tile_dir.name)
    if not match:
        raise ValueError(f"Not an Ortho4XP tile directory: {tile_dir}")
    lat = int(match.group("lat"))
    lon = int(match.group("lon"))
    cfg_path = tile_dir / f"Ortho4XP_{short_latlon(lat, lon)}.cfg"
    if not cfg_path.exists():
        return {"tile": tile_dir.name, "source": "skip", "zone_count": 0, "duplicate_footprints": 0, "cfg_path": str(cfg_path)}
    cfg = parse_cfg(cfg_path)
    textures_dir = tile_dir / "textures"
    texture_count = len(list(textures_dir.glob("*.dds"))) if textures_dir.exists() else 0

    bak_path = cfg_path.with_suffix(cfg_path.suffix + ".bak")
    if bak_path.exists():
        bak_cfg = parse_cfg(bak_path)
        if bak_cfg.zone_list:
            return {
                "tile": tile_dir.name,
                "lat": lat,
                "lon": lon,
                "cfg_path": str(cfg_path),
                "default_website": cfg.default_website,
                "default_zl": cfg.default_zl,
                "textures": texture_count,
                "duplicate_footprints": 0,
                "zone_count": len(bak_cfg.zone_list),
                "source": "bak",
                "zone_list": bak_cfg.zone_list,
                "selected_textures": [],
            }

    winners, duplicate_count = winning_textures(textures_dir)

    zones: list[list[object]] = []
    selected: list[dict[str, object]] = []
    for record in sorted(winners, key=lambda item: (-item.zoomlevel, -item.mtime, item.y_top, item.x_left, item.provider)):
        if (
            not include_default_textures
            and record.provider == cfg.default_website
            and record.zoomlevel == cfg.default_zl
        ):
            continue
        zone = texture_to_zone(record, lat, lon)
        if zone is None:
            continue
        zones.append(zone)
        selected.append(
            {
                "file": record.path.name,
                "provider": record.provider,
                "zoomlevel": record.zoomlevel,
                "x_left": record.x_left,
                "y_top": record.y_top,
                "mtime": datetime.fromtimestamp(record.mtime).isoformat(timespec="seconds"),
            }
        )

    return {
        "tile": tile_dir.name,
        "lat": lat,
        "lon": lon,
        "cfg_path": str(cfg_path),
        "default_website": cfg.default_website,
        "default_zl": cfg.default_zl,
        "textures": texture_count,
        "duplicate_footprints": duplicate_count,
        "zone_count": len(zones),
        "source": "textures",
        "zone_list": zones,
        "selected_textures": selected,
    }


def write_zone_list(cfg_path: Path, zone_list: list[list[object]]) -> Path:
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
    wrote = False
    new_lines: list[str] = []
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


def default_manifest_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("analysis_tmp") / f"reconstructed_zone_list_{timestamp}.json"


def main() -> int:
    args = parse_args()
    tile_dirs = discover_tile_dirs(args.tile_dirs or [Path(".")])
    manifest = {
        "mode": "write" if args.write else "dry-run",
        "include_default_textures": args.include_default_textures,
        "tile_count": len(tile_dirs),
        "tiles": [],
    }
    errors: list[str] = []
    for tile_dir in tile_dirs:
        try:
            result = reconstruct_zone_list(
                tile_dir,
                include_default_textures=args.include_default_textures,
            )
            if args.write and result["source"] != "skip":
                backup_path = write_zone_list(
                    Path(result["cfg_path"]),
                    result["zone_list"],
                )
                result["backup_path"] = str(backup_path)
        except Exception as exc:
            msg = f"{tile_dir.name}: {exc}"
            errors.append(msg)
            result = {"tile": tile_dir.name, "error": str(exc), "zone_count": 0, "source": "error", "duplicate_footprints": 0}
        manifest["tiles"].append(result)

    manifest_path = (args.manifest or default_manifest_path()).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"{manifest['mode']}: {manifest['tile_count']} tile(s)")
    for tile in manifest["tiles"]:
        if "error" in tile:
            print(f"- {tile['tile']}: ERROR — {tile['error']}")
        elif tile["source"] == "skip":
            print(f"- {tile['tile']}: skipped (no cfg)")
        else:
            print(
                f"- {tile['tile']}: {tile['zone_count']} zone(s) [{tile['source']}], "
                f"{tile['duplicate_footprints']} duplicate footprint(s)"
            )
    if errors:
        print(f"{len(errors)} tile(s) failed")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
