#!/usr/bin/env python
"""
generate_overlay.py — standalone CLI for AI-based Ortho4XP overlay generation.

Examples
--------
# Tile at lat=45, lon=7 using the default Tiles/ directory
    python generate_overlay.py --lat 45 --lon 7

# Multiple tiles
    python generate_overlay.py --lat 45 --lon 7 --lat 46 --lon 8

# Specify a custom build directory for the tile
    python generate_overlay.py --lat 45 --lon 7 --build-dir D:/Scenery/zOrtho4XP_+45+007

# Vegetation only (skip building detection)
    python generate_overlay.py --lat 45 --lon 7 --no-buildings

# Force CPU (useful for debugging)
    python generate_overlay.py --lat 45 --lon 7 --cpu
"""

import argparse
import sys
import os

# Ensure src/ is on the path whether we run from repo root or elsewhere
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "src"))


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate AI vegetation/building DSF overlays for Ortho4XP tiles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--lat", type=int, action="append", required=True,
                   metavar="LAT",
                   help="Tile latitude (floor, integer). Repeat for multiple tiles.")
    p.add_argument("--lon", type=int, action="append", required=True,
                   metavar="LON",
                   help="Tile longitude (floor, integer). Repeat for multiple tiles.")
    p.add_argument("--build-dir", default=None,
                   help="Override tile build directory (for a single tile only).")
    p.add_argument("--no-vegetation", action="store_true",
                   help="Skip vegetation detection.")
    p.add_argument("--no-buildings", action="store_true",
                   help="Skip building detection.")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU inference (slow; default: use CUDA if available).")
    return p.parse_args()


def main():
    args = parse_args()

    if len(args.lat) != len(args.lon):
        print("ERROR: --lat and --lon must be provided the same number of times.",
              file=sys.stderr)
        sys.exit(1)

    tiles = list(zip(args.lat, args.lon))

    import torch
    device = torch.device("cpu") if args.cpu else None  # None → auto-detect in module

    from O4_AI_Overlay import process_tile

    ok_count = 0
    for lat, lon in tiles:
        print(f"\n{'='*60}")
        print(f"  Tile  lat={lat:+d}  lon={lon:+d}")
        print(f"{'='*60}")
        build_dir = args.build_dir if len(tiles) == 1 else None
        result = process_tile(
            lat, lon,
            build_dir=build_dir,
            device=device,
            do_vegetation=not args.no_vegetation,
            do_buildings=not args.no_buildings,
        )
        if result:
            ok_count += 1

    print(f"\n[AI Overlay] Finished: {ok_count}/{len(tiles)} tiles succeeded.")
    sys.exit(0 if ok_count == len(tiles) else 1)


if __name__ == "__main__":
    main()
