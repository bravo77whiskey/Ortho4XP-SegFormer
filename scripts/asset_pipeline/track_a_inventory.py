"""Inventory residential-adjacent building objects in installed Track A
X-Plane libraries (OpenSceneryX, R2_Library, world-models, etc.).

For every EXPORT line in each library.txt we apply the same filtering the SFR
overlay uses (OPTIONAL_LIBRARY_INCLUDE_TOKENS, GENERIC_BUILDING_EXCLUDE_TOKENS,
OPTIONAL_LIBRARY_SPECIAL_LANDMARK_TOKENS) so the inventory matches what the
overlay would actually consider as a placement candidate.

Objects are then grouped into "families" by stripping the trailing numeric
``/<n>.obj`` suffix. Most Track A libraries store visually-similar variants
together (e.g. ``houses/wooden/1.obj`` through ``houses/wooden/21.obj``),
so one render per family is enough to triage all members in a single pass.

Output: ``scripts/asset_pipeline/track_a_triage/inventory.csv`` with columns
  lib_id, family_prefix, representative_virtual_path, representative_physical_path,
  member_count, sample_virtual_paths
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Reuse the SFR overlay's filter constants so the inventory tracks reality.
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_SFR_Building_Overlay as BLD  # noqa: E402


_TRAILING_NUMBER_RE = re.compile(r"/(\d+)\.obj$", re.IGNORECASE)


def family_prefix(virtual_path: str) -> str:
    """Strip a trailing numeric ``/<n>.obj`` suffix when present.

    ``houses/wooden/1.obj``         -> ``houses/wooden/``
    ``houses/wooden/big_house.obj`` -> ``houses/wooden/big_house.obj`` (unchanged)
    """
    p = virtual_path.replace("\\", "/")
    match = _TRAILING_NUMBER_RE.search(p)
    if match:
        return p[: match.start() + 1]
    return p


def parse_library_txt(library_txt: Path) -> list[tuple[str, str]]:
    """Yield ``(virtual_path, physical_path)`` from EXPORT lines."""
    exports: list[tuple[str, str]] = []
    with open(library_txt, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            tag = parts[0].upper()
            if tag in ("EXPORT", "EXPORT_EXTEND", "EXPORT_RATIO"):
                # EXPORT virtual physical, optionally with leading ratio token.
                if tag == "EXPORT_RATIO" and len(parts) >= 4:
                    virtual = parts[2]
                    physical = " ".join(parts[3:])
                elif len(parts) >= 3:
                    virtual = parts[1]
                    physical = " ".join(parts[2:])
                else:
                    continue
                if virtual.lower().endswith(".obj"):
                    exports.append((virtual, physical))
    return exports


def is_building_candidate(virtual_path: str) -> bool:
    """Apply the same filter as the SFR overlay."""
    if not BLD._is_optional_library_building_candidate(virtual_path):
        return False
    p = virtual_path.replace("\\", "/").lower()
    if BLD._path_has_any(p, BLD.OPTIONAL_LIBRARY_SPECIAL_LANDMARK_TOKENS):
        return False
    return True


def find_library_txt(custom_scenery_dir: Path, lib_id: str) -> Path | None:
    """Locate the library.txt of an installed curated library."""
    entry = BLD.CURATED_EXTRA_BUILDING_LIBRARIES.get(lib_id)
    if not entry:
        return None
    patterns = [p.lower() for p in entry.get("package_patterns") or ()]
    if not patterns:
        return None
    for child in sorted(custom_scenery_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name.lower()
        if not any(p in name for p in patterns):
            continue
        lib_txt = child / "library.txt"
        if lib_txt.is_file():
            return lib_txt
    return None


def inventory_library(custom_scenery_dir: Path, lib_id: str) -> list[dict]:
    """Return per-family rows for one library."""
    library_txt = find_library_txt(custom_scenery_dir, lib_id)
    if library_txt is None:
        return []
    package_dir = library_txt.parent
    exports = parse_library_txt(library_txt)

    families: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for virtual, physical in exports:
        if not is_building_candidate(virtual):
            continue
        families[family_prefix(virtual)].append((virtual, physical))

    rows: list[dict] = []
    for prefix, members in sorted(families.items()):
        members.sort()
        rep_virtual, rep_physical = members[0]
        abs_physical = (package_dir / rep_physical).resolve()
        rows.append({
            "lib_id": lib_id,
            "family_prefix": prefix,
            "representative_virtual_path": rep_virtual,
            "representative_physical_path": str(abs_physical),
            "member_count": len(members),
            "sample_virtual_paths": "|".join(v for v, _ in members[:5]),
        })
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--custom-scenery-dir", required=True,
                        help="X-Plane Custom Scenery directory.")
    parser.add_argument("--output", required=True,
                        help="Output CSV path.")
    parser.add_argument("--libraries", nargs="*", default=None,
                        help="Limit to specific lib_id(s); default is all "
                             "registered Track A libraries.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    custom_scenery_dir = Path(args.custom_scenery_dir)
    if not custom_scenery_dir.is_dir():
        raise SystemExit(f"Custom Scenery dir not found: {custom_scenery_dir}")

    lib_ids = args.libraries or list(BLD.CURATED_EXTRA_BUILDING_LIBRARIES.keys())
    all_rows: list[dict] = []
    for lib_id in lib_ids:
        rows = inventory_library(custom_scenery_dir, lib_id)
        print(f"{lib_id:18} {len(rows):4} families")
        all_rows.extend(rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["lib_id", "family_prefix",
              "representative_virtual_path", "representative_physical_path",
              "member_count", "sample_virtual_paths"]
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print()
    print(f"inventory: {output_path} ({len(all_rows)} families)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
