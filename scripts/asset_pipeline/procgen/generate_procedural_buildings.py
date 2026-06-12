"""Host-side driver: batch-generate the procgen library through Blender.

Reads procgen_manifest.json (grid.py) and atlas_layout.json (atlas.py),
flattens enabled assets into per-OBJ rows, chunks them, and runs one headless
Blender process per chunk via blender_worker.py.  A post-pass normalizes the
TEXTURE directive of every produced OBJ8 to the shared atlas.

Run:
    python atlas.py --output <pkg>            # textures must exist first
    python generate_procedural_buildings.py --output <pkg> [--jobs 2]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
WORKER = os.path.join(HERE, "blender_worker.py")

# Banded LOD: the post-pass merges far-LOD meshes into every OBJ
# (apply_lod_shells.patch_shell) so dense tiles stay renderable — full mesh,
# silhouette shell, flat-top box, roof quad, then culled. Pass
# --lod-bands 0 0 0 0 to skip (plain single-band objects).
from apply_lod_shells import DEFAULT_BANDS, patch_shell  # noqa: E402

_CHUNK_DONE_RE = re.compile(r"^\[chunk-done\] ok=(\d+) fail=(\d+)", re.MULTILINE)


def _flatten_rows(manifest: dict, *, only: str = None,
                  archetypes: tuple = ()) -> list:
    rows = []
    seen_paths = set()
    for asset in manifest["assets"]:
        if not asset.get("enabled"):
            continue
        if only and only not in asset["id"]:
            continue
        for variant in asset["variants"]:
            if archetypes and variant["archetype"] not in archetypes:
                continue
            # Band overlaps can map one physical OBJ into several virtual
            # paths; build it once.
            if variant["physical_path"] in seen_paths:
                continue
            seen_paths.add(variant["physical_path"])
            stem = os.path.splitext(
                os.path.basename(variant["physical_path"])
            )[0]
            rows.append({
                "stem": stem,
                "physical_path": variant["physical_path"],
                "archetype": variant["archetype"],
                "length_m": asset["length_m"],
                "width_m": asset["width_m"],
                "floors": asset["floors"],
                "seed": variant["seed"],
                "flavor": asset["flavor"],
            })
    return rows


def _run_chunk(blender: str, chunk_path: str, output_root: str,
               layout_path: str) -> tuple[int, int, str]:
    cmd = [
        blender, "--background", "--factory-startup",
        "--python", WORKER, "--",
        chunk_path, output_root, layout_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                check=False)
    except FileNotFoundError as exc:
        raise SystemExit(f"Blender not found at {blender!r}: {exc}") from exc
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    match = _CHUNK_DONE_RE.search(output)
    if not match:
        return 0, -1, output  # fail==-1 flags "worker crashed before summary"
    return int(match.group(1)), int(match.group(2)), output


def _expected_texture_line(physical_path: str, flavor: str) -> str:
    depth = physical_path.replace("\\", "/").count("/")
    up = "../" * depth
    return f"TEXTURE {up}textures/o4sfr_procgen_atlas_{flavor}.png"


def _normalize_texture_directive(obj_path: str, texture_line: str) -> None:
    """Force the OBJ8 header's TEXTURE line to the shared atlas."""
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as fh:
        lines = fh.read().splitlines()
    # Pass 1: drop every plain TEXTURE line the exporter may have written.
    stripped_lines = [
        line for line in lines
        if not (line.strip().startswith("TEXTURE")
                and not line.strip().startswith("TEXTURE_"))
    ]
    # Pass 2: insert exactly one directive after the OBJ header marker.
    out = []
    inserted = False
    for line in stripped_lines:
        out.append(line)
        if not inserted and line.strip() in ("OBJ", "OBJ8"):
            out.append("")
            out.append(texture_line)
            inserted = True
    if not inserted:
        out = ["A", "800", "OBJ", "", texture_line, ""] + out
    with open(obj_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default=os.path.join(HERE, "procgen_manifest.json")
    )
    parser.add_argument(
        "--layout", default=os.path.join(HERE, "atlas_layout.json")
    )
    parser.add_argument("--output", required=True,
                        help="Library package folder (Custom Scenery).")
    parser.add_argument("--blender",
                        default=os.environ.get("O4SFR_BLENDER", "blender"),
                        help="Path to the Blender executable.")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Parallel Blender processes on disjoint chunks.")
    parser.add_argument("--chunk-size", type=int, default=150)
    parser.add_argument("--limit", type=int, default=0,
                        help="Generate at most N OBJ files (0 = all).")
    parser.add_argument("--only", default=None,
                        help="Only assets whose id contains this substring.")
    parser.add_argument("--archetypes", default=None,
                        help="Comma-separated archetype filter.")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate OBJs that already exist.")
    parser.add_argument("--lod-bands", type=int, nargs=4,
                        default=list(DEFAULT_BANDS),
                        metavar=("FULL", "SHELL", "BOX", "QUAD"),
                        help="LOD band boundaries in metres "
                             "(first value 0 = no LOD bands).")
    args = parser.parse_args(argv)

    with open(args.manifest, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not os.path.isfile(args.layout):
        raise SystemExit(
            f"atlas layout missing at {args.layout}; run atlas.py first"
        )
    flavors = set()
    rows = _flatten_rows(
        manifest,
        only=args.only,
        archetypes=tuple(
            part.strip() for part in (args.archetypes or "").split(",")
            if part.strip()
        ),
    )
    pending = []
    for row in rows:
        target = os.path.join(args.output, row["physical_path"])
        if os.path.isfile(target) and not args.force:
            continue
        pending.append(row)
        flavors.add(row["flavor"])
    if args.limit > 0:
        pending = pending[: args.limit]
        flavors = {row["flavor"] for row in pending}

    for flavor in sorted(flavors):
        texture = os.path.join(
            args.output, "textures", f"o4sfr_procgen_atlas_{flavor}.png"
        )
        if not os.path.isfile(texture):
            raise SystemExit(
                f"atlas texture missing: {texture}; run atlas.py --output "
                f"{args.output} first"
            )

    total = len(rows)
    print(f"{total} obj files in scope, {len(pending)} to generate")
    if not pending:
        return 0

    chunk_dir = tempfile.mkdtemp(prefix="o4sfr_procgen_")
    failures: list[str] = []
    try:
        chunks = [
            pending[i: i + args.chunk_size]
            for i in range(0, len(pending), args.chunk_size)
        ]
        chunk_paths = []
        for index, chunk in enumerate(chunks):
            path = os.path.join(chunk_dir, f"chunk_{index:04d}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"assets": chunk}, fh)
            chunk_paths.append(path)

        def _job(item):
            index, path = item
            ok, fail, output = _run_chunk(
                args.blender, path, args.output, args.layout
            )
            return index, ok, fail, output

        results = []
        if args.jobs > 1:
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                results = list(pool.map(_job, enumerate(chunk_paths)))
        else:
            results = [_job(item) for item in enumerate(chunk_paths)]

        ok_total = 0
        for index, ok, fail, output in sorted(results):
            ok_total += ok
            if fail != 0:
                failures.append(f"chunk {index}: ok={ok} fail={fail}")
                tail = "\n".join(output.splitlines()[-15:])
                print(f"--- chunk {index} problems ---\n{tail}")

        # Post-pass: deterministic TEXTURE directive + far-LOD bands.
        layout = None
        if args.lod_bands[0] > 0:
            with open(args.layout, "r", encoding="utf-8") as fh:
                layout = json.load(fh)
        normalized = 0
        for row in pending:
            obj_path = os.path.join(args.output, row["physical_path"])
            if not os.path.isfile(obj_path):
                continue
            _normalize_texture_directive(
                obj_path,
                _expected_texture_line(row["physical_path"], row["flavor"]),
            )
            if layout is not None:
                patch_shell(
                    obj_path, row["archetype"], row["length_m"],
                    row["width_m"], row["floors"], row["seed"],
                    row["flavor"], layout, tuple(args.lod_bands),
                )
            normalized += 1
        print(f"generated ok={ok_total}, texture+shell-normalized={normalized}, "
              f"failed-chunks={len(failures)}")
    finally:
        shutil.rmtree(chunk_dir, ignore_errors=True)

    if failures:
        for line in failures:
            print(f"  ! {line}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
