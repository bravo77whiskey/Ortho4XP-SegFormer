"""Enumerate the procgen dimension grid into a deterministic manifest.

Reads config.yaml and writes procgen_manifest.json next to it.  The manifest
is the single source of truth for the library build: every asset row carries
its exact dimensions, archetype, region, variant seeds and output paths.

Run:  python grid.py [--config config.yaml] [--out procgen_manifest.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import yaml  # noqa: E402

from archetypes import ARCHETYPES  # noqa: E402
from archetypes.common import fmt_dim  # noqa: E402

MANIFEST_SCHEMA_VERSION = 1
FLOOR_HEIGHT_M = 3.2  # _SIMHEAVEN_FLOORS_RE contract: height = floors * 3.2
MAX_FLOORS = 12       # 12 * 3.2 = 38.4 < MAX_GENERATED_BUILDING_HEIGHT_M (40)

# Mirror of the overlay's filename filters that could silently reject our
# assets.  The authoritative audit (importing the real tuples from
# O4_SFR_Building_Overlay) lives in verify_procgen_library.py; this short
# denylist just fails fast on obviously bad archetype names.
_NAME_DENYLIST = (
    "tower", "store", "tank", "silo", "hangar", "chimney", "carport",
    "school", "hotel", "terminal", "parking", "mast", "antenna",
)


def _ladder(start_m: float, ratio: float, stop_m: float) -> list[float]:
    values = []
    value = float(start_m)
    while value <= float(stop_m) + 1e-9:
        values.append(round(value, 1))
        value *= float(ratio)
    return values


def _asset_seed(lib_version: int, asset_id: str, archetype: str,
                variant: int) -> int:
    digest = hashlib.sha1(
        f"{lib_version}:{asset_id}:{archetype}:v{variant}".encode("utf-8")
    ).hexdigest()
    return int(digest[:8], 16)


# Virtual-path stem per bucket.  One virtual path per (key, floors, region,
# bucket) carries EVERY matching archetype as a physical EXPORT variant:
# the matcher ranks candidates deterministically (alphabetical at equal
# coverage), so per-archetype virtual paths would make one archetype win a
# whole dimension key; folding archetypes into X-Plane's per-placement
# variant randomization is what makes neighborhoods look mixed.
BUCKET_STEMS = {
    "residential": "house",
    "apartments": "apt",
    "commercial": "com",
    "industrial": "ind",
}


def _footprint_keys(ladder: list[float], *, min_area: float, max_area: float,
                    max_aspect: float, max_side: float,
                    min_side: float = 0.0) -> list[tuple[float, float]]:
    keys = []
    for length in ladder:
        for width in ladder:
            if width > length:
                continue
            area = length * width
            if not (min_area - 1e-6 <= area <= max_area + 1e-6):
                continue
            if length / width > max_aspect + 1e-9:
                continue
            if length > max_side:
                continue
            if max(length, width) < min_side:
                continue
            keys.append((length, width))
    return keys


def build_manifest(config: dict) -> dict:
    lib_version = int(config["lib_version"])
    limits = config["limits"]
    ladders = {
        name: {
            "values": _ladder(spec["start_m"], spec["ratio"], spec["stop_m"]),
            "max_aspect": float(spec["max_aspect"]),
        }
        for name, spec in config["ladders"].items()
    }
    archetype_buckets = config.get("archetype_buckets") or {}
    region_flavors = config["region_flavors"]

    assets = []
    seen_ids = set()
    for band in config["bands"]:
        ladder = ladders[band["ladder"]]
        max_aspect = float(band.get("max_aspect", ladder["max_aspect"]))
        keys = _footprint_keys(
            ladder["values"],
            min_area=float(band["min_area_m2"]),
            max_area=min(float(band["max_area_m2"]), float(limits["max_area_m2"])),
            max_aspect=max_aspect,
            max_side=min(
                float(band.get("max_side_m", limits["max_side_m"])),
                float(limits["max_side_m"]),
            ),
            min_side=float(band.get("min_side_m", 0.0)),
        )
        # Group the band's archetypes by their effective bucket so each
        # virtual path stays semantically pure (residential-context flags
        # and include tokens are per virtual path).
        by_bucket = {}
        for archetype in band["archetypes"]:
            if any(token in archetype for token in _NAME_DENYLIST):
                raise SystemExit(
                    f"archetype name {archetype!r} hits a building-overlay "
                    f"exclude token; rename it"
                )
            bucket = archetype_buckets.get(archetype, band["bucket"])
            by_bucket.setdefault(bucket, []).append(archetype)

        archetype_regions = config.get("archetype_regions") or {}
        for bucket, bucket_archetypes in sorted(by_bucket.items()):
            stem_prefix = BUCKET_STEMS[bucket]
            for floors in band["floors"]:
                if int(floors) > MAX_FLOORS:
                    raise SystemExit(
                        f"band {band['name']}: floors {floors} exceeds the "
                        f"{MAX_FLOORS}-floor "
                        f"({MAX_FLOORS * FLOOR_HEIGHT_M:.1f} m) ingestion cap"
                    )
                for region in band["regions"]:
                    flavor = region_flavors[region]
                    # Per-region archetype routing: some massing types only
                    # belong to specific regions (shophouses in Asia, ...).
                    region_archetypes = [
                        archetype for archetype in sorted(bucket_archetypes)
                        if region in archetype_regions.get(
                            archetype, (region,)
                        )
                    ]
                    if not region_archetypes:
                        continue
                    for length, width in keys:
                        dims = f"{fmt_dim(length)}x{fmt_dim(width)}x{floors}"
                        stem = f"{stem_prefix}_{dims}"
                        asset_id = f"{region}/{bucket}/{stem}"
                        if asset_id in seen_ids:
                            continue
                        seen_ids.add(asset_id)
                        variants = [
                            {
                                "archetype": archetype,
                                "variant": k,
                                "seed": _asset_seed(
                                    lib_version, asset_id, archetype, k
                                ),
                                "physical_path": (
                                    f"{region}/{bucket}/"
                                    f"{archetype}_{dims}_v{k}.obj"
                                ),
                            }
                            for archetype in region_archetypes
                            if archetype in ARCHETYPES
                            for k in range(1, int(band["variants"]) + 1)
                        ]
                        assets.append({
                            "id": asset_id,
                            "virtual_path": f"o4sfr/{asset_id}.obj",
                            "archetypes": region_archetypes,
                            "length_m": length,
                            "width_m": width,
                            "floors": int(floors),
                            "height_m": round(int(floors) * FLOOR_HEIGHT_M, 2),
                            "region": region,
                            "bucket": bucket,
                            "band": band["name"],
                            "flavor": flavor,
                            "enabled": bool(variants),
                            "variants": variants,
                        })

    assets.sort(key=lambda row: row["id"])
    config_blob = json.dumps(config, sort_keys=True).encode("utf-8")
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "lib_version": lib_version,
        "global_seed": int(config["global_seed"]),
        "config_sha1": hashlib.sha1(config_blob).hexdigest(),
        "output_package_name": config["output_package_name"],
        "atlas": dict(config["atlas"]),
        "counts": {
            "virtual_paths": len(assets),
            "enabled_virtual_paths": sum(1 for a in assets if a["enabled"]),
            "obj_files": sum(len(a["variants"]) for a in assets),
            "enabled_obj_files": sum(
                len(a["variants"]) for a in assets if a["enabled"]
            ),
        },
        "assets": assets,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    parser.add_argument(
        "--out", default=os.path.join(HERE, "procgen_manifest.json")
    )
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    manifest = build_manifest(config)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
        fh.write("\n")

    counts = manifest["counts"]
    per_arch = {}
    missing = set()
    for asset in manifest["assets"]:
        built = {v["archetype"] for v in asset["variants"]}
        for archetype in asset["archetypes"]:
            per_arch[archetype] = per_arch.get(archetype, 0) + (
                1 if archetype in built else 0
            )
            if archetype not in built:
                missing.add(archetype)
    print(
        f"manifest: {counts['virtual_paths']} virtual paths, "
        f"{counts['obj_files']} obj files "
        f"({counts['enabled_virtual_paths']} / {counts['enabled_obj_files']} "
        f"enabled with implemented archetypes)"
    )
    for name in sorted(per_arch):
        state = "  [not implemented yet]" if name in missing else ""
        print(f"  {name:10s} {per_arch[name]:5d} variant paths{state}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
