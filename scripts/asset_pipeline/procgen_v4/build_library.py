"""Build the full v4 library as a drop-in for the old ProcGen package.

    python build_library.py --out <pkg> [--workers 12] [--limit N]

Reads the OLD library's manifest (procgen/procgen_manifest.json): same
virtual paths, footprints, floors, regions, variant seeds and physical
paths -- so the overlay's matcher, the pool's bounds checks and any
existing tile DSFs keep resolving -- and writes every OBJ with the v4
builders, then library.txt (one EXPORT per physical variant, as before)
and build_info.json.  Pure Python: no Blender needed for geometry.

Every o4xp/ virtual path is also exported under its pre-rename o4sfr/ name,
so tiles built before the library was renamed keep their buildings.

Validate with the old verifier, unchanged:
    python ../procgen/verify_procgen_library.py --output <pkg>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
OLD_MANIFEST = os.path.normpath(os.path.join(HERE, "..", "procgen",
                                             "procgen_manifest.json"))

import buildings  # noqa: E402
from meshkit import write_obj8  # noqa: E402

HEADER = "A\n800\nLIBRARY\n\n"
PREFIX = "o4xp/"
LEGACY_PREFIX = "o4sfr/"


def _build_chunk(args):
    out, rows = args
    stats = {"objs": 0, "tris": [0, 0, 0, 0]}
    for row in rows:
        page, bands = buildings.build(row["archetype"], row["L"], row["W"],
                                      row["F"], row["seed"], row["region"])
        dists, _ = buildings.band_distances(row["L"], row["W"], row["F"],
                                                 bands)
        path = os.path.join(out, row["physical"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_obj8(path, [(bands[i], dists[i], dists[i + 1])
                          for i in range(4)],
                   f"../../textures/o4v4_{page}.png",
                   f"../../textures/o4v4_{page}_nml.png")
        stats["objs"] += 1
        for i in range(4):
            stats["tris"][i] += bands[i].tri_count()
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=OLD_MANIFEST)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 4))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    out = os.path.abspath(args.out)
    with open(args.manifest, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    rows = []
    lines = []
    paths = 0
    for asset in manifest["assets"]:
        if not asset.get("enabled"):
            continue
        paths += 1
        for v in asset["variants"]:
            rows.append({"archetype": v["archetype"], "L": asset["length_m"],
                         "W": asset["width_m"], "F": asset["floors"],
                         "seed": v["seed"], "region": asset["region"],
                         "physical": v["physical_path"]})
            lines.append(f"EXPORT {asset['virtual_path']} "
                         f"{v['physical_path']}")
            if asset["virtual_path"].startswith(PREFIX):
                legacy = LEGACY_PREFIX + asset["virtual_path"][len(PREFIX):]
                lines.append(f"EXPORT {legacy} {v['physical_path']}")
    if args.limit:
        rows = rows[:args.limit]
    t0 = time.time()
    chunk = 400
    jobs = [(out, rows[i:i + chunk]) for i in range(0, len(rows), chunk)]
    total = {"objs": 0, "tris": [0, 0, 0, 0]}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, st in enumerate(pool.map(_build_chunk, jobs)):
            total["objs"] += st["objs"]
            for i in range(4):
                total["tris"][i] += st["tris"][i]
            if k % 25 == 0:
                print(f"  {total['objs']}/{len(rows)} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    if not args.limit:
        with open(os.path.join(out, "library.txt"), "w", encoding="utf-8",
                  newline="\n") as fh:
            fh.write(HEADER + "\n".join(lines) + "\n")
    info = {
        "built_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"),
        "generator": "procgen_v4",
        "virtual_paths": paths,
        "obj_files": total["objs"],
        "mean_tris_per_band": [t / max(1, total["objs"])
                               for t in total["tris"]],
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(out, "build_info.json"), "w",
              encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
