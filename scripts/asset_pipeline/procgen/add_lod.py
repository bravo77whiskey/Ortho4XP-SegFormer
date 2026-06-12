"""Manage ATTR_LOD draw-distance limits in generated OBJ8s (in place).

The library DEFAULT matches simHeaven / SFD Global building objects: NO
explicit ATTR_LOD, letting X-Plane compute the draw distance from each
object's size and the user's object-detail setting (scales with graphics
settings instead of a hard-coded ceiling, and stays instancing-friendly).

``--strip`` removes any explicit ATTR_LOD (restoring the default).
``--inject`` adds a single size-scaled band as an experiment knob:

  far = clamp(800 + 40*sqrt(footprint_area) + 30*height, 1300, 6000) m
    -> houses ~1.3 km, apartment blocks ~3 km, big-boxes up to 6 km.

Both modes are idempotent.

Run:  python add_lod.py --output <pkg> --strip
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

LOD_MIN_M = 1300.0
LOD_MAX_M = 6000.0
FLOOR_HEIGHT_M = 3.2


def lod_far_m(length_m: float, width_m: float, floors: int) -> int:
    area = float(length_m) * float(width_m)
    height = int(floors) * FLOOR_HEIGHT_M
    far = 800.0 + 40.0 * math.sqrt(area) + 30.0 * height
    return int(round(min(LOD_MAX_M, max(LOD_MIN_M, far)) / 50.0) * 50)


def inject_lod(obj_path: str, far_m: int) -> str:
    """Insert ATTR_LOD before the first draw command; returns a status."""
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as fh:
        lines = fh.read().splitlines()
    out = []
    inserted = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("ATTR_LOD"):
            return "already"
        if not inserted and stripped.startswith("TRIS"):
            out.append(f"ATTR_LOD 0 {far_m}")
            inserted = True
        out.append(line)
    if not inserted:
        return "no-tris"
    with open(obj_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out) + "\n")
    return "patched"


def strip_lod(obj_path: str) -> str:
    """Remove every ATTR_LOD line; returns a status."""
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as fh:
        lines = fh.read().splitlines()
    kept = [line for line in lines if not line.strip().startswith("ATTR_LOD")]
    if len(kept) == len(lines):
        return "already"
    with open(obj_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(kept) + "\n")
    return "patched"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default=os.path.join(HERE, "procgen_manifest.json")
    )
    parser.add_argument("--output", required=True,
                        help="Library package folder to patch in place.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--strip", action="store_true",
                      help="Remove explicit ATTR_LOD (simHeaven/SFD default).")
    mode.add_argument("--inject", action="store_true",
                      help="Add the size-scaled hard LOD band.")
    args = parser.parse_args(argv)

    with open(args.manifest, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    stats = {"patched": 0, "already": 0, "missing": 0, "no-tris": 0}
    seen = set()
    for asset in manifest["assets"]:
        if not asset.get("enabled"):
            continue
        far = lod_far_m(asset["length_m"], asset["width_m"], asset["floors"])
        for variant in asset["variants"]:
            physical = variant["physical_path"]
            if physical in seen:
                continue
            seen.add(physical)
            obj_path = os.path.join(args.output, physical)
            if not os.path.isfile(obj_path):
                stats["missing"] += 1
                continue
            if args.strip:
                stats[strip_lod(obj_path)] += 1
            else:
                stats[inject_lod(obj_path, far)] += 1

    action = "strip" if args.strip else "inject"
    print(
        f"LOD {action}: patched={stats['patched']}  already={stats['already']}  "
        f"missing={stats['missing']}  no-tris={stats['no-tris']}"
    )
    return 1 if stats["no-tris"] else 0


if __name__ == "__main__":
    sys.exit(main())
