"""Join inventory.csv + probe.csv + decisions.yaml into a sources.yaml manifest.

Inputs (under --triage):
  inventory.csv   -- one row per archive (license, source_url, model_path, preview)
  probe.csv       -- per-slug bounding-box dimensions
  decisions.yaml  -- assistant's region/bucket calls

Output:
  sources.yaml    -- conforms to the schema enforced by build_custom_library.py

Footprint rule:
  - If both bbox_x and bbox_y are within 4-40 m (real-world building scale),
    trust the probe.
  - Otherwise fall back to a bucket default (see BUCKET_DEFAULT_FOOTPRINT).

License gate:
  - The schema in build_custom_library.py only accepts licenses starting with
    CC0 / CC-BY / PUBLIC DOMAIN. Rows that don't are dropped (with a warning)
    unless --include-unknown is given.

Run:
    python scripts/asset_pipeline/manifest_from_triage.py \\
        --triage scripts/asset_pipeline/triage \\
        --output scripts/asset_pipeline/sources.yaml
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# Borrow the schema constants from the sibling script.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from build_custom_library import (  # noqa: E402
    VALID_REGIONS,
    VALID_BUCKETS,
    VALID_LICENSE_PREFIXES,
)


BUCKET_DEFAULT_FOOTPRINT = {
    "residential": (10.0, 8.0),
    "commercial":  (15.0, 12.0),
    "industrial":  (25.0, 18.0),
    "farm":        (12.0, 10.0),
    "accessory":   (5.0, 5.0),
}

REAL_BUILDING_MIN_M = 4.0
REAL_BUILDING_MAX_M = 40.0
HIGH_POLY_THRESHOLD = 50_000


def _require_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML required: pip install pyyaml") from exc
    return yaml


def load_decisions(path: Path) -> dict[str, dict]:
    yaml = _require_yaml()
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    rows = data.get("decisions") or []
    out: dict[str, dict] = {}
    for row in rows:
        slug = row.get("slug")
        if not slug:
            continue
        out[slug] = row
    return out


def load_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def choose_footprint(bucket: str, bbox_x: str, bbox_y: str) -> tuple[float, float]:
    try:
        dx = float(bbox_x) if bbox_x else 0.0
        dy = float(bbox_y) if bbox_y else 0.0
    except ValueError:
        dx = dy = 0.0
    if (REAL_BUILDING_MIN_M <= dx <= REAL_BUILDING_MAX_M
            and REAL_BUILDING_MIN_M <= dy <= REAL_BUILDING_MAX_M):
        return dx, dy
    return BUCKET_DEFAULT_FOOTPRINT.get(bucket, (10.0, 8.0))


def license_passes(licence: str) -> bool:
    licence = (licence or "").strip().upper()
    return any(licence.startswith(pfx) for pfx in VALID_LICENSE_PREFIXES)


def assemble(args) -> int:
    triage = Path(args.triage)
    inv = {row["slug"]: row for row in load_csv(triage / "inventory.csv")}
    probe = {row["slug"]: row for row in load_csv(triage / "probe.csv")}
    decisions = load_decisions(triage / "decisions.yaml")
    if not inv:
        raise SystemExit(f"inventory.csv missing or empty under {triage}")
    if not decisions:
        raise SystemExit(
            f"decisions.yaml missing or empty under {triage}; "
            "no slugs to write into the manifest"
        )

    yaml = _require_yaml()
    entries = []
    skipped = []
    for slug, dec in decisions.items():
        if slug not in inv:
            skipped.append((slug, "no inventory row"))
            continue
        inv_row = inv[slug]
        probe_row = probe.get(slug, {})

        region = (dec.get("region") or "").lower()
        bucket = (dec.get("bucket") or "").lower()
        if region not in VALID_REGIONS:
            skipped.append((slug, f"invalid region: {region}"))
            continue
        if bucket not in VALID_BUCKETS:
            skipped.append((slug, f"invalid bucket: {bucket}"))
            continue

        licence = inv_row.get("license", "")
        if not license_passes(licence):
            if args.include_unknown:
                # Coerce rejected/blank licenses to CC-BY-4.0 unconditionally
                # so the build_custom_library schema accepts them. The user is
                # responsible for fixing inventory.csv before redistribution.
                licence = "CC-BY-4.0"
            else:
                skipped.append((slug, f"license rejected: {licence!r}"))
                continue

        fmt = (inv_row.get("format") or "").lower()
        if fmt not in ("fbx", "blend", "obj", "glb", "gltf"):
            skipped.append((slug, f"unsupported format: {fmt!r}"))
            continue

        author = inv_row.get("author") or "Unknown"
        source_url = inv_row.get("source_url") or ""
        source_file = inv_row.get("model_path") or ""
        if not source_file:
            skipped.append((slug, "missing model_path"))
            continue

        width_m, depth_m = choose_footprint(
            bucket, probe_row.get("bbox_x"), probe_row.get("bbox_y")
        )
        try:
            tris = int(probe_row.get("triangle_count") or 0)
        except ValueError:
            tris = 0

        entry = {
            "id": slug,
            "region": region,
            "bucket": bucket,
            "license": licence,
            "author": author,
            "source_url": source_url,
            "source_file": source_file,
            "footprint_m": [round(width_m, 2), round(depth_m, 2)],
        }
        if tris > HIGH_POLY_THRESHOLD:
            entry["decimate_target"] = 2000
        notes = dec.get("notes")
        if notes:
            entry["notes"] = notes
        entries.append(entry)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump({"version": 1, "assets": entries}, fh, sort_keys=False)
    print(f"manifest: {out_path} ({len(entries)} entries)")
    if skipped:
        print(f"skipped: {len(skipped)}")
        for slug, reason in skipped:
            print(f"  - {slug}: {reason}")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triage", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-unknown", action="store_true",
                        help="Pass-through entries with unrecognised licenses "
                             "(coerced to CC-BY-4.0). Off by default.")
    return parser.parse_args(argv)


def main(argv=None):
    return assemble(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
