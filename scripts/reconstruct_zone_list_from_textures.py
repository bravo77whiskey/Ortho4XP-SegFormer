from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from O4_Zone_Utils import (  # noqa: E402
    DDS_RE,
    MASK_RE,
    TILE_RE,
    TextureRecord,
    TileConfig,
    discover_tile_dirs,
    gtile_to_wgs84,
    parse_cfg,
    parse_mask,
    parse_texture,
    reconstruct_zone_list,
    short_latlon,
    texture_to_zone,
    tile_coordinates,
    winning_textures,
    write_zone_list,
    zone_intersects_tile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover Ortho4XP zone_list entries from a config backup, DDS "
            "textures, or PNG masks. Higher zoom levels are ordered first; "
            "duplicate x/y/ZL footprints use the newest file."
        )
    )
    parser.add_argument(
        "tile_dirs",
        nargs="*",
        type=Path,
        help=(
            "One or more zOrtho4XP tile directories, or a Tiles root. "
            "Defaults to the current directory."
        ),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write recovered zones into empty tile configs. Default is dry-run.",
    )
    parser.add_argument(
        "--include-default-textures",
        action="store_true",
        help="Also turn base provider/ZL textures into zone entries.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSON manifest path. Defaults under analysis_tmp/.",
    )
    return parser.parse_args()


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
    errors = 0
    for tile_dir in tile_dirs:
        try:
            result = reconstruct_zone_list(
                tile_dir,
                include_default_textures=args.include_default_textures,
            )
            if args.write and result["source"] in ("backup", "textures"):
                if result["zone_count"]:
                    result["backup_path"] = str(
                        write_zone_list(Path(result["cfg_path"]), result["zone_list"])
                    )
        except Exception as exc:
            errors += 1
            result = {
                "tile": tile_dir.name,
                "source": "error",
                "zone_count": 0,
                "duplicate_footprints": 0,
                "error": str(exc),
            }
        manifest["tiles"].append(result)

    manifest_path = (args.manifest or default_manifest_path()).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"{manifest['mode']}: {manifest['tile_count']} tile(s)")
    for tile in manifest["tiles"]:
        if tile["source"] == "error":
            print(f"- {tile['tile']}: ERROR - {tile['error']}")
        elif tile["source"] == "skip":
            print(f"- {tile['tile']}: skipped (no cfg)")
        else:
            print(
                f"- {tile['tile']}: {tile['zone_count']} zone(s) "
                f"[{tile['source']}], "
                f"{tile['duplicate_footprints']} duplicate footprint(s)"
            )
    print(f"manifest: {manifest_path}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
