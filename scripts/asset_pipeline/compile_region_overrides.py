"""Compile Track A visual-triage decisions into a JSON override file.

Reads ``track_a_triage/decisions.yaml`` and emits
``src/O4_SFR_Region_Overrides.json`` -- a flat list of
``[lib_id, family_prefix, [regions...]]`` triples loaded at import time by
``_optional_library_asset_regions()``.

Region validity is enforced against the canonical taxonomy
(``OPTIONAL_ASSET_REGION_ALIASES`` plus the ``generic`` wildcard).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_SFR_Building_Overlay as BLD  # noqa: E402


VALID_REGIONS = set(BLD.OPTIONAL_ASSET_REGION_ALIASES) | {"generic"}


def _require_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML required: pip install pyyaml") from exc
    return yaml


def compile_overrides(decisions_path: Path, output_path: Path) -> int:
    yaml = _require_yaml()
    if not decisions_path.is_file():
        raise SystemExit(f"decisions.yaml not found: {decisions_path}")
    with open(decisions_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    rows = data.get("decisions") or []
    if not rows:
        raise SystemExit("No decisions found in YAML.")

    # We index overrides as a list of (lib_id, family_prefix_lower, regions).
    # The overlay matches the longest-prefix-first, then the lib_id, then
    # falls back to the static registration.
    out: list[dict] = []
    seen_keys = set()
    for entry in rows:
        lib_id = (entry.get("lib_id") or "").strip().lower()
        family = (entry.get("family_prefix") or "").strip()
        regions = entry.get("regions") or []
        if not lib_id or not family or not regions:
            raise SystemExit(
                f"decisions.yaml entry missing fields: {entry!r}"
            )
        family_lower = family.replace("\\", "/").lower()
        regions_norm = []
        for r in regions:
            r_l = str(r).strip().lower()
            if r_l not in VALID_REGIONS:
                raise SystemExit(
                    f"invalid region {r_l!r} for {lib_id}:{family}"
                )
            regions_norm.append(r_l)
        key = (lib_id, family_lower)
        if key in seen_keys:
            raise SystemExit(f"duplicate override for {lib_id}:{family}")
        seen_keys.add(key)
        out.append({
            "lib_id": lib_id,
            "family_prefix": family_lower,
            "regions": tuple(regions_norm),
        })

    # Order by descending prefix length so the overlay's matcher can short
    # circuit on the most specific path token first.
    out.sort(key=lambda e: (-len(e["family_prefix"]), e["lib_id"]))

    # Convert tuples to lists for JSON compatibility.
    json_payload = {
        "version": 1,
        "overrides": [
            {**e, "regions": list(e["regions"])}
            for e in out
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(json_payload, fh, indent=2, sort_keys=False)
    print(f"overrides: {output_path} ({len(out)} entries)")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decisions", required=True,
        help="Path to track_a_triage/decisions.yaml.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSON file the SFR overlay will load.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return compile_overrides(Path(args.decisions), Path(args.output))


if __name__ == "__main__":
    sys.exit(main())
