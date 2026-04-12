"""CLI wrapper for the combined SegFormer overlay module."""

import argparse
import os
import sys


_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_ROOT_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from O4_SegFormer_Overlay import process_sfr_tile


def parse_args():
    """Parse CLI arguments for combined SegFormer overlay generation."""
    parser = argparse.ArgumentParser(
        description="Generate SegFormer vegetation/building DSF overlays for Ortho4XP tiles."
    )
    parser.add_argument("--lat", type=int, action="append", required=True, metavar="LAT")
    parser.add_argument("--lon", type=int, action="append", required=True, metavar="LON")
    parser.add_argument("--build-dir", default=None)
    parser.add_argument("--no-vegetation", action="store_true")
    parser.add_argument("--no-buildings", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    """Run the combined SegFormer overlay CLI."""
    args = parse_args()
    if len(args.lat) != len(args.lon):
        print("ERROR: --lat and --lon must be provided the same number of times.", file=sys.stderr)
        raise SystemExit(1)

    import torch

    device = torch.device("cpu") if args.cpu else None
    success_count = 0
    tiles = list(zip(args.lat, args.lon))
    for lat, lon in tiles:
        build_dir = args.build_dir if len(tiles) == 1 else None
        result = process_sfr_tile(
            lat,
            lon,
            build_dir=build_dir,
            device=device,
            do_vegetation=not args.no_vegetation,
            do_buildings=not args.no_buildings,
        )
        if result:
            success_count += 1

    print(f"\n[SegFormer] Finished: {success_count}/{len(tiles)} tiles succeeded.")
    raise SystemExit(0 if success_count == len(tiles) else 1)


if __name__ == "__main__":
    main()
