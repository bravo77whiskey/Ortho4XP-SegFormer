"""Retarget an already-built library's TEXTURE directives to atlas pages.

Texture pages (config atlas.pages) share the strip layout, so switching a
built OBJ between pages is a pure header rewrite -- no Blender, no UV
changes.  This walks procgen_manifest.json, computes each variant's page
from its seed (same rule as generate_procedural_buildings.py) and rewrites
the OBJ8 TEXTURE line in place.  Run it after atlas.py whenever pages are
added or the page mapping changes; running it twice is a no-op.

Run:  python retarget_texture_pages.py --output <Custom Scenery>/O4SFR_ProcGen_Library
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from atlas import texture_name, texture_normal_name  # noqa: E402
from generate_procedural_buildings import (  # noqa: E402
    _expected_texture_lines, _flatten_rows, _modern_archetypes,
    _normalize_texture_directive,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default=os.path.join(HERE, "procgen_manifest.json")
    )
    parser.add_argument("--output", required=True,
                        help="Library package folder (Custom Scenery).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing.")
    args = parser.parse_args(argv)

    with open(args.manifest, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    pages = max(1, int((manifest.get("atlas") or {}).get("pages", 1)))

    modern_set = _modern_archetypes(manifest)
    missing_textures = []
    for flavor in sorted({a["flavor"] for a in manifest["assets"]}):
        for page in range(pages):
            for modern in (False, True):
                for name in (texture_name(flavor, page, modern),
                             texture_normal_name(flavor, page, modern)):
                    path = os.path.join(args.output, "textures", name)
                    if not os.path.isfile(path):
                        missing_textures.append(path)
    if missing_textures:
        for path in missing_textures:
            print(f"  ! missing atlas page: {path}")
        raise SystemExit("run atlas.py --output <pkg> first")

    rows = _flatten_rows(manifest)
    changed = skipped = absent = 0
    for row in rows:
        obj_path = os.path.join(args.output, row["physical_path"])
        if not os.path.isfile(obj_path):
            absent += 1
            continue
        expected = _expected_texture_lines(
            row["physical_path"], row["flavor"], row["seed"], pages,
            row["archetype"] in modern_set)
        with open(obj_path, "r", encoding="utf-8", errors="ignore") as fh:
            current = fh.read()
        if "\n" + "\n".join(expected) + "\n" in current:
            skipped += 1
            continue
        if not args.dry_run:
            _normalize_texture_directive(obj_path, expected)
        changed += 1
    verb = "would retarget" if args.dry_run else "retargeted"
    print(f"{verb} {changed} OBJs across {pages} pages "
          f"(already-correct={skipped}, missing-obj={absent})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
