"""Build the shipped O4SFR_Library Custom Scenery folder from sources.yaml.

Inputs: a manifest file with one entry per asset (see sources.yaml).
Outputs (under --output):
  library.txt       -- one EXPORT line per asset, virtual paths o4sfr/<region>/...
  LICENSE           -- short summary of how individual asset licenses apply
  ATTRIBUTIONS.md   -- per-asset author/source/license credit (required for CC-BY)

This script does NOT touch the original 3D-model files or call Blender. The
xplane2blender conversion step is done by convert_to_xplane_obj.py. We assume
the converted .obj files already live at <output>/<region>/<bucket>/<id>.obj
(use --skip-missing to emit a manifest even if some are not yet built).

Run via:
    python scripts/asset_pipeline/build_custom_library.py \
        --manifest scripts/asset_pipeline/sources.yaml \
        --output "C:/X-Plane 12/Custom Scenery/O4SFR_Library"
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Iterable, NamedTuple


# Region keys accepted in the manifest. Kept in sync with
# _O4SFR_PATH_REGION_TOKENS in src/O4_SFR_Building_Overlay.py.
VALID_REGIONS = frozenset({
    "europe",
    "scandinavia",
    "mediterranean",
    "north_america",
    "south_america",
    "asia",
    "se_asia",
    "africa",
    "australia_oceania",
    "generic",
})

VALID_BUCKETS = frozenset({
    "residential",
    "commercial",
    "industrial",
    "farm",
    "accessory",
})

VALID_LICENSE_PREFIXES = (
    "CC0",
    "CC-BY",
    "PUBLIC DOMAIN",
)


class Asset(NamedTuple):
    id: str
    region: str
    bucket: str
    license: str
    author: str
    source_url: str
    source_file: str
    footprint_m: tuple
    height_m: float
    # Optional Blender conversion knobs. Both have sane defaults; override per
    # asset for chunky photogrammetry models or stylised low-poly inputs.
    decimate_target: int = 2000
    texture_max_px: int = 1024

    @property
    def virtual_path(self) -> str:
        return f"o4sfr/{self.region}/{self.bucket}/{self.id}.obj"

    @property
    def physical_path(self) -> str:
        return f"{self.region}/{self.bucket}/{self.id}.obj"


def _require_yaml():
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "PyYAML is required: pip install pyyaml"
        ) from exc
    return yaml


def load_manifest(path: str) -> list[Asset]:
    yaml = _require_yaml()
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if data.get("version") != 1:
        raise SystemExit(f"Unsupported manifest version: {data.get('version')!r}")
    raw_assets = data.get("assets") or []
    seen_ids = set()
    assets: list[Asset] = []
    for entry in raw_assets:
        asset = _parse_entry(entry)
        if asset.id in seen_ids:
            raise SystemExit(f"Duplicate asset id: {asset.id!r}")
        seen_ids.add(asset.id)
        assets.append(asset)
    return assets


def _parse_entry(entry) -> Asset:
    required = ("id", "region", "bucket", "license", "author", "source_url",
                "source_file", "footprint_m")
    missing = [key for key in required if key not in entry]
    if missing:
        raise SystemExit(f"Manifest entry {entry!r} is missing keys: {missing}")
    region = str(entry["region"]).lower()
    if region not in VALID_REGIONS:
        raise SystemExit(f"Invalid region {region!r} for asset {entry['id']!r}")
    bucket = str(entry["bucket"]).lower()
    if bucket not in VALID_BUCKETS:
        raise SystemExit(f"Invalid bucket {bucket!r} for asset {entry['id']!r}")
    licence = str(entry["license"]).strip().upper()
    if not any(licence.startswith(pfx) for pfx in VALID_LICENSE_PREFIXES):
        raise SystemExit(
            f"Unsupported license {licence!r} for asset {entry['id']!r}; "
            f"manifest accepts only {VALID_LICENSE_PREFIXES}"
        )
    footprint = tuple(entry["footprint_m"])
    if len(footprint) != 2 or any(not isinstance(v, (int, float)) for v in footprint):
        raise SystemExit(
            f"footprint_m for {entry['id']!r} must be [width_m, depth_m]"
        )
    height = float(entry.get("height_m") or 0.0)
    decimate_target = int(entry.get("decimate_target", 2000))
    if decimate_target < 100:
        raise SystemExit(
            f"decimate_target for {entry['id']!r} must be >= 100 triangles"
        )
    texture_max_px = int(entry.get("texture_max_px", 1024))
    if texture_max_px < 64:
        raise SystemExit(
            f"texture_max_px for {entry['id']!r} must be >= 64"
        )
    return Asset(
        id=str(entry["id"]),
        region=region,
        bucket=bucket,
        license=licence,
        author=str(entry["author"]),
        source_url=str(entry["source_url"]),
        source_file=str(entry["source_file"]),
        footprint_m=tuple(float(v) for v in footprint),
        height_m=height,
        decimate_target=decimate_target,
        texture_max_px=texture_max_px,
    )


def render_library_txt(assets: Iterable[Asset]) -> str:
    lines = [
        "I",
        "800",
        "LIBRARY",
        "",
        "# O4SFR_Library -- shipped by Ortho4XP-SegFormer.",
        "# Per-asset region is encoded as a path token: o4sfr/<region>/<bucket>/<id>.obj",
        "",
    ]
    for asset in sorted(assets, key=lambda a: (a.region, a.bucket, a.id)):
        lines.append(f"EXPORT {asset.virtual_path} {asset.physical_path}")
    lines.append("")
    return "\n".join(lines)


def render_license(assets: Iterable[Asset]) -> str:
    return (
        "O4SFR_Library bundles open-source 3D models from third-party authors.\n"
        "Each asset retains its original license; see ATTRIBUTIONS.md for the\n"
        "per-asset credit, source URL, and license terms.\n\n"
        "CC0 assets may be redistributed without attribution. CC-BY assets\n"
        "require attribution as recorded in ATTRIBUTIONS.md. No assets in\n"
        "this library use restrictive licenses (NC, ND, SA).\n"
    )


def render_attributions(assets: Iterable[Asset]) -> str:
    groups: dict[str, list[Asset]] = defaultdict(list)
    for asset in assets:
        groups[asset.license].append(asset)
    out = ["# O4SFR_Library attributions", ""]
    for licence in sorted(groups):
        out.append(f"## {licence}")
        out.append("")
        for asset in sorted(groups[licence], key=lambda a: (a.region, a.id)):
            out.append(
                f"- `{asset.virtual_path}` -- {asset.author} -- "
                f"<{asset.source_url}>"
            )
        out.append("")
    return "\n".join(out)


def _verify_physical_files(assets: Iterable[Asset], output_dir: str) -> list[Asset]:
    missing = []
    for asset in assets:
        absolute = os.path.join(output_dir, asset.physical_path)
        if not os.path.isfile(absolute):
            missing.append(asset)
    return missing


def build(args):
    assets = load_manifest(args.manifest)
    if not assets:
        raise SystemExit("Manifest contains no assets.")
    os.makedirs(args.output, exist_ok=True)
    missing = _verify_physical_files(assets, args.output)
    if missing and not args.skip_missing:
        rel_paths = "\n  ".join(asset.physical_path for asset in missing)
        raise SystemExit(
            f"{len(missing)} asset .obj files are not built yet:\n  {rel_paths}\n"
            "Run convert_to_xplane_obj.py first, or pass --skip-missing to\n"
            "drop unbuilt entries from library.txt."
        )

    # When --skip-missing is set, exclude unbuilt entries from library.txt
    # entirely. Otherwise X-Plane logs "resource cannot be found" errors per
    # missing virtual path at scenery load time.
    if missing:
        missing_ids = {a.id for a in missing}
        emitted_assets = [a for a in assets if a.id not in missing_ids]
    else:
        emitted_assets = assets

    written = []
    for filename, render in (
        ("library.txt", render_library_txt(emitted_assets)),
        ("LICENSE", render_license(emitted_assets)),
        ("ATTRIBUTIONS.md", render_attributions(emitted_assets)),
    ):
        path = os.path.join(args.output, filename)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(render)
        written.append(path)
    print(f"Built O4SFR_Library at {args.output}")
    for path in written:
        print(f"  wrote {os.path.basename(path)}")
    if missing:
        print(f"  assets emitted: {len(emitted_assets)} "
              f"(dropped {len(missing)} missing .obj entries)")
    else:
        print(f"  assets emitted: {len(emitted_assets)}")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True,
                        help="Path to sources.yaml.")
    parser.add_argument("--output", required=True,
                        help="Custom Scenery folder to build into.")
    parser.add_argument("--skip-missing", action="store_true",
                        help="Emit library.txt even if some .obj files are absent.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return build(args)


if __name__ == "__main__":
    sys.exit(main())
