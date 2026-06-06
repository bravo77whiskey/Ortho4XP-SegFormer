"""Audit installed X-Plane building assets usable by SFR placement."""

import argparse
from collections import Counter, defaultdict
import os
import sys


_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import O4_SFR_Asset_Inventory as ASSETINV
import O4_SFR_Building_Overlay as BLD


CLASS_LABELS = {
    BLD.BLD_CLASS_TINY_RESIDENTIAL: "tiny",
    BLD.BLD_CLASS_SMALL_RESIDENTIAL: "small",
    BLD.BLD_CLASS_COMPACT_RESIDENTIAL: "compact",
    BLD.BLD_CLASS_MEDIUM: "medium",
    BLD.BLD_CLASS_SMALL_APARTMENT: "small_apt",
    BLD.BLD_CLASS_APARTMENT_BLOCK: "apt_block",
    BLD.BLD_CLASS_LARGE: "large",
    BLD.BLD_CLASS_EXTRA_LARGE: "extra_large",
}


def _asset_counts(pools):
    counts = Counter()
    by_source = Counter()
    for cls, assets in pools.items():
        counts[CLASS_LABELS.get(cls, str(cls))] += len(assets)
        for asset in assets:
            by_source[asset.get("source", "unknown")] += 1
    return counts, by_source


def _print_counter(title, counter):
    print(title)
    if not counter:
        print("  none")
        return
    for key, value in sorted(counter.items()):
        print(f"  {key}: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit installed building object assets for SFR placement."
    )
    parser.add_argument(
        "--custom-scenery-dir",
        default=None,
        help="X-Plane Custom Scenery directory or X-Plane root.",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=45.0,
        help="Representative tile latitude for region-specific pools.",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=7.0,
        help="Representative tile longitude for region-specific pools.",
    )
    parser.add_argument(
        "--asset-region",
        default=None,
        help="Override effective asset region.",
    )
    parser.add_argument(
        "--extra-libraries",
        default=None,
        help="Optional library policy: auto, off, or comma-separated curated ids.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Maximum unmapped recommended samples to print.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.extra_libraries is not None:
        os.environ["O4_SFR_BLD_EXTRA_LIBRARIES"] = args.extra_libraries

    custom_scenery_dir = ASSETINV.resolve_custom_scenery_dir(args.custom_scenery_dir)
    if not custom_scenery_dir:
        custom_scenery_dir = getattr(getattr(BLD.SEGFORMER, "CFG", None), "custom_scenery_dir", None)
    region = args.asset_region or BLD._asset_region(args.lat, args.lon)
    enabled_extra = BLD._enabled_extra_library_ids()

    exports = BLD._scan_runtime_library_exports(
        custom_scenery_dir,
        include_sfd=True,
        include_simheaven=True,
        extra_library_ids=enabled_extra,
    )
    all_exports = ASSETINV.scan_library_exports(
        custom_scenery_dir=custom_scenery_dir,
        include_default=True,
        suffixes=(".obj", ".fac"),
    )

    default_pools = BLD._build_default_asset_pools(args.lat, args.lon, region)
    sfd_pools = BLD._build_sfd_asset_pools(
        args.lat,
        args.lon,
        region,
        library_exports=exports,
    )
    simheaven_pools = BLD._build_simheaven_asset_pools(
        [],
        args.lat,
        args.lon,
        region,
        library_exports=exports,
    )
    extra_pools = BLD._build_optional_library_asset_pools(
        custom_scenery_dir=custom_scenery_dir,
        library_exports=exports,
        enabled_library_ids=enabled_extra,
    )
    merged = BLD._merge_asset_pools(default_pools, sfd_pools, simheaven_pools, extra_pools)

    print(f"Custom Scenery: {custom_scenery_dir or 'unavailable'}")
    print(f"Asset region: {region}")
    print(f"Library exports scanned: runtime={len(exports)} all={len(all_exports)}")
    print(f"Extra library policy: {BLD._extra_library_policy()} enabled={', '.join(enabled_extra) or 'none'}")
    counts, by_source = _asset_counts(merged)
    _print_counter("Mapped placement assets by class:", counts)
    _print_counter("Mapped placement assets by source:", by_source)

    mapped_paths = {
        asset["path"].lower()
        for pool in merged.values()
        for asset in pool
        if asset.get("path")
    }
    grouped = ASSETINV.unique_virtual_exports(all_exports, suffix=".obj")
    recommended = defaultdict(list)
    skipped = Counter()
    for _, group in grouped.items():
        path = group[0].virtual_path
        key = path.lower()
        if key in mapped_paths:
            continue
        if key.startswith("simheaven/"):
            if path in BLD._scanned_simheaven_catalog_paths(library_exports=group):
                recommended["simHeaven dimension-coded"].append(path)
            else:
                skipped["simHeaven filtered"] += 1
        elif key.startswith("sfd_global/"):
            if (
                BLD._is_sfd_building_asset_candidate(path)
                and BLD._sfd_export_matches_region(path, region)
            ):
                recommended["SFD measured candidate"].append(path)
            else:
                skipped["SFD filtered"] += 1
        elif any(BLD._is_optional_library_export_enabled(group[0], (lib_id,)) for lib_id in enabled_extra):
            if BLD._is_optional_library_building_candidate(path):
                recommended["optional measured candidate"].append(path)
            else:
                skipped["optional filtered"] += 1

    print("Recommended unmapped candidates:")
    if not recommended:
        print("  none")
    for label, paths in sorted(recommended.items()):
        print(f"  {label}: {len(paths)}")
        for path in sorted(paths)[:max(0, args.samples)]:
            print(f"    {path}")
    _print_counter("Skipped candidate groups:", skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
