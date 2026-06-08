"""Surface Track A families that are still unclassified by the SFR overlay.

Reads ``track_a_triage/inventory.csv`` and runs each family's representative
path through ``_optional_library_asset_regions``.  Families that come back
empty (i.e. neither the visual-triage override file nor the hardcoded
per-library path rules nor the library's static ``regions`` registration
covers them) are printed grouped by ``lib_id`` and by a coarse path-depth
prefix so the user can see what residual triage work remains.

This is a planning aid -- it does not modify any files.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_SFR_Building_Overlay as BLD  # noqa: E402


def _coarse_group(family_prefix: str, depth: int = 3) -> str:
    """Coarsen a family prefix to its first `depth` path segments.

    Helps cluster the (potentially long) residual list into recognisable
    sub-trees that the user can triage in batches.
    """
    parts = family_prefix.replace("\\", "/").lower().strip("/").split("/")
    if len(parts) <= depth:
        return "/".join(parts)
    return "/".join(parts[:depth])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--lib", default=None,
                        help="Optional substring filter on lib_id.")
    parser.add_argument("--depth", type=int, default=3,
                        help="Path-segment depth for the coarse grouping "
                             "(default: 3).")
    parser.add_argument("--show-paths", action="store_true",
                        help="Print every residual family path under each "
                             "group; off by default to keep output compact.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with open(args.inventory, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.lib:
        rows = [r for r in rows if args.lib in r["lib_id"].lower()]

    by_lib: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        lib_id = r["lib_id"]
        rep = r["representative_virtual_path"]
        regions = BLD._optional_library_asset_regions(rep, lib_id)
        if regions:
            continue
        by_lib[lib_id].append(r)

    if not by_lib:
        print("No residual families -- every Track A family is classified.")
        return 0

    grand_total = 0
    for lib_id in sorted(by_lib):
        rows_for_lib = by_lib[lib_id]
        grand_total += len(rows_for_lib)
        print()
        print(f"=== {lib_id} ===   {len(rows_for_lib)} residual families")
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in rows_for_lib:
            groups[_coarse_group(r["family_prefix"], args.depth)].append(r)
        for group, members in sorted(
            groups.items(), key=lambda kv: (-len(kv[1]), kv[0])
        ):
            family_count = len(members)
            member_total = sum(int(m["member_count"]) for m in members)
            print(f"  {group:60} {family_count:4} families  "
                  f"({member_total} objects)")
            if args.show_paths:
                for m in sorted(members, key=lambda r: r["family_prefix"]):
                    print(f"      {m['family_prefix']}  "
                          f"x{m['member_count']}")
    print()
    print(f"TOTAL RESIDUAL: {grand_total} families")
    return 0


if __name__ == "__main__":
    sys.exit(main())
