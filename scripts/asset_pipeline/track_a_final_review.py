"""End-to-end review of Track A building object coverage.

Cross-tabulates every accepted candidate by:

    library    x    region tag(s)    x    BLD_CLASS_*    x    footprint dims

so we can see -- at a glance -- where the asset pool is thick or thin per
region, what footprint sizes are available, and where gaps remain.

Uses the same scan + filter + measurement helpers as the placement pipeline
itself (``_scan_runtime_library_exports``, ``_measured_bounds_for_exports``,
``_optional_library_asset_regions``, ``_class_for_object_asset``) so the
numbers reflect what the SFR overlay will actually consider at runtime.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_SFR_Asset_Inventory as ASSETINV  # noqa: E402
import O4_SFR_Building_Overlay as BLD  # noqa: E402
import O4_SFR_DSF_Utils as DSF  # noqa: E402


def _dim_bucket(width_m: float, depth_m: float) -> str:
    """Coarsen a footprint into a human-readable size bucket."""
    longer = max(width_m, depth_m)
    if longer <= 6:
        return "tiny (≤6m)"
    if longer <= 10:
        return "small (6–10m)"
    if longer <= 16:
        return "mid (10–16m)"
    if longer <= 25:
        return "large (16–25m)"
    if longer <= 40:
        return "x-large (25–40m)"
    return "landmark (>40m)"


def _format_regions(regions):
    if not regions:
        return "<unclassified>"
    return ",".join(sorted(regions))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--custom-scenery-dir", required=True)
    parser.add_argument("--cache-dir", default=None,
                        help="Optional bounds cache directory.")
    parser.add_argument("--policy", default=None,
                        help="Override O4_SFR_BLD_EXTRA_LIBRARIES; default is "
                             "whatever the environment / 'auto' produces.")
    parser.add_argument("--show-empty-cells", action="store_true",
                        help="Include zero-count cells in the per-library "
                             "tables.  Off by default.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    custom = ASSETINV.resolve_custom_scenery_dir(args.custom_scenery_dir)
    if not custom:
        raise SystemExit(f"--custom-scenery-dir not usable: {args.custom_scenery_dir}")
    if args.policy is not None:
        os.environ["O4_SFR_BLD_EXTRA_LIBRARIES"] = args.policy
    enabled = BLD._enabled_extra_library_ids()

    exports = BLD._scan_runtime_library_exports(
        custom,
        extra_library_ids=enabled,
    )
    grouped = BLD._library_exports_by_virtual_path(exports)

    # rows: list of (lib_id, regions, zone_class, bounds_m, obj_path)
    rows = []
    skipped = Counter()
    for _key, exports_for_path in grouped.items():
        export = exports_for_path[0]
        obj_path = export.virtual_path
        lib_id = BLD._optional_library_id_for_export(export, enabled)
        if not lib_id:
            continue
        if not BLD._is_optional_library_building_candidate(obj_path):
            skipped["not-residential-candidate"] += 1
            continue
        reason = BLD._optional_library_rejection_reason(obj_path, lib_id, "europe")
        # "europe" is just a sentinel here -- region acceptance is handled per
        # row when we build the report.  We only use the rejection helper to
        # drop special landmarks and unclassified entries the placement layer
        # would never look at.
        if reason in ("special-landmark", "unclassified"):
            skipped[f"rejected-{reason}"] += 1
            continue
        regions = BLD._optional_library_asset_regions(obj_path, lib_id)
        bounds_m = BLD._measured_bounds_for_exports(
            exports_for_path, args.cache_dir
        )
        if not BLD._footprint_within_limits(
            bounds_m, max_area_m2=7_000.0, max_side_m=110.0
        ):
            skipped["footprint-out-of-bounds"] += 1
            continue
        zone_class = BLD._class_for_object_asset(obj_path, bounds_m)
        rows.append((lib_id, regions, zone_class, bounds_m, obj_path))

    if not rows:
        print("No accepted candidates found.  Check that Track A libraries are "
              "installed in the Custom Scenery directory.")
        return 0

    # ---- Totals ------------------------------------------------------------
    print("=" * 76)
    print("Track A coverage summary")
    print("=" * 76)
    print(f"Custom Scenery: {custom}")
    print(f"Policy:         {BLD._extra_library_policy()}")
    print(f"Enabled libs:   {', '.join(enabled)}")
    print(f"Accepted candidates: {len(rows)}")
    if skipped:
        print("Skipped:")
        for reason, n in sorted(skipped.items(), key=lambda kv: -kv[1]):
            print(f"  {reason:30} {n}")

    # ---- By region (collapsed) --------------------------------------------
    region_counter: Counter = Counter()
    for _lib, regions, _cls, _b, _p in rows:
        if not regions:
            region_counter["<unclassified>"] += 1
        else:
            for r in regions:
                region_counter[r] += 1
    print()
    print("--- Accepted candidates per region (multi-region entries count once per region) ---")
    for region, n in sorted(region_counter.items(), key=lambda kv: -kv[1]):
        print(f"  {region:20} {n}")

    # ---- By class ---------------------------------------------------------
    class_counter: Counter = Counter()
    for _lib, _r, cls, _b, _p in rows:
        class_counter[BLD.BLD_CLASS_LABELS.get(cls, str(cls))] += 1
    print()
    print("--- Accepted candidates per BLD_CLASS_* ---")
    for cls_label in [
        BLD.BLD_CLASS_LABELS[c] for c in BLD.BLD_PLACEMENT_CLASSES
    ]:
        n = class_counter.get(cls_label, 0)
        print(f"  {cls_label:24} {n}")

    # ---- By dimension bucket ----------------------------------------------
    dim_counter: Counter = Counter()
    for _lib, _r, _cls, bounds_m, _p in rows:
        w, d = _footprint_wd(bounds_m)
        dim_counter[_dim_bucket(w, d)] += 1
    print()
    print("--- Accepted candidates per footprint dimension bucket ---")
    for bucket in (
        "tiny (≤6m)", "small (6–10m)", "mid (10–16m)",
        "large (16–25m)", "x-large (25–40m)", "landmark (>40m)",
    ):
        n = dim_counter.get(bucket, 0)
        print(f"  {bucket:24} {n}")

    # ---- Region x class cross-tab -----------------------------------------
    print()
    print("--- Region x class (accepted candidate counts) ---")
    region_class: dict[tuple, Counter] = defaultdict(Counter)
    region_names = []
    for _lib, regions, cls, _b, _p in rows:
        cls_label = BLD.BLD_CLASS_LABELS.get(cls, str(cls))
        keys = regions if regions else ("<unclassified>",)
        for r in keys:
            region_class[r][cls_label] += 1
    region_order = [
        "europe", "scandinavia", "mediterranean",
        "north_america", "north_america_ne", "north_america_west",
        "south_america", "asia", "se_asia", "africa", "australia_oceania",
        "generic", "<unclassified>",
    ]
    seen_regions = [r for r in region_order if r in region_class]
    for r in sorted(region_class):
        if r not in seen_regions:
            seen_regions.append(r)
    class_order = [BLD.BLD_CLASS_LABELS[c] for c in BLD.BLD_PLACEMENT_CLASSES]
    header = f"  {'region':18}" + "".join(f"{c[:9]:>10}" for c in class_order)
    print(header)
    for r in seen_regions:
        row_counts = [region_class[r].get(c, 0) for c in class_order]
        if not args.show_empty_cells and not any(row_counts):
            continue
        cells = "".join(f"{n:>10}" for n in row_counts)
        print(f"  {r:18}{cells}")

    # ---- SimHeaven X-World pack contribution -----------------------------
    print()
    print("--- SimHeaven X-World packs (region-specific via package folder) ---")
    print("    SimHeaven exports the same virtual paths (simheaven/facades/*,")
    print("    simheaven/landmarks/*) for every X-World pack; the visual region")
    print("    comes from which pack contains the tile's DSF, not the asset path.")
    sh_exports = BLD._scan_runtime_library_exports(
        custom, include_simheaven=True
    )
    sh_by_pack: dict[str, list] = defaultdict(list)
    for export in sh_exports:
        pkg = (getattr(export, "package_name", "") or "")
        path = (getattr(export, "virtual_path", "") or "").replace("\\", "/")
        if not path.lower().startswith("simheaven/"):
            continue
        if not path.lower().endswith(".obj"):
            continue
        sh_by_pack[pkg].append(export)
    if not sh_by_pack:
        print("    (no SimHeaven X-World packs installed)")
    else:
        for pkg, pkg_exports in sorted(sh_by_pack.items()):
            region = DSF.simheaven_package_region_from_name(pkg) or "<unknown>"
            # Count repeatable candidate assets the SFR overlay would accept.
            count = sum(
                1 for e in pkg_exports
                if BLD._is_repeatable_simheaven_asset(e.virtual_path)
            )
            print(f"    [{region:18}] {pkg}  ({count} repeatable candidates)")

    # Map SimHeaven pack regions -> natural-earth regions for the cross-tab.
    sh_region_expansion = {
        "europe": ("europe", "scandinavia", "mediterranean"),
        "america": ("north_america", "south_america"),
        "asia": ("asia", "se_asia"),
        "africa": ("africa",),
        "australia_oceania": ("australia_oceania",),
        "antarctica": ("generic",),
    }
    sh_region_class_counts: dict[str, Counter] = defaultdict(Counter)
    for pkg, pkg_exports in sh_by_pack.items():
        pack_region = DSF.simheaven_package_region_from_name(pkg)
        if not pack_region:
            continue
        for nat_region in sh_region_expansion.get(pack_region, (pack_region,)):
            for e in pkg_exports:
                if not BLD._is_repeatable_simheaven_asset(e.virtual_path):
                    continue
                bounds = BLD._measured_bounds_for_exports([e], args.cache_dir)
                if not BLD._footprint_within_limits(
                    bounds, max_area_m2=7_000.0, max_side_m=110.0
                ):
                    continue
                cls = BLD._class_for_object_asset(e.virtual_path, bounds)
                cls_label = BLD.BLD_CLASS_LABELS.get(cls, str(cls))
                sh_region_class_counts[nat_region][cls_label] += 1
    if sh_region_class_counts:
        print()
        print("--- SimHeaven region x class (expanded via X-World pack -> natural regions) ---")
        sh_seen = [r for r in region_order if r in sh_region_class_counts]
        for r in sorted(sh_region_class_counts):
            if r not in sh_seen:
                sh_seen.append(r)
        print(header)
        for r in sh_seen:
            row_counts = [
                sh_region_class_counts[r].get(c, 0) for c in class_order
            ]
            if not args.show_empty_cells and not any(row_counts):
                continue
            cells = "".join(f"{n:>10}" for n in row_counts)
            print(f"  {r:18}{cells}")

    # ---- Default X-Plane 12 + SFD Global per region ----------------------
    # Both pools are region-conditional: the same builder returns different
    # catalogs for different region keys.  Tally each natural-earth region.
    default_region_class: dict[str, Counter] = defaultdict(Counter)
    sfd_region_class: dict[str, Counter] = defaultdict(Counter)
    default_kind_class: dict[str, Counter] = defaultdict(Counter)
    sfd_kind_class: dict[str, Counter] = defaultdict(Counter)
    natural_regions = (
        "europe", "scandinavia", "mediterranean",
        "north_america", "north_america_ne", "north_america_west",
        "south_america", "asia", "se_asia", "africa", "australia_oceania",
        "generic",
    )
    runtime_exports = BLD._scan_runtime_library_exports(
        custom, include_sfd=True
    )
    for nat_region in natural_regions:
        # Default pools (built-in X-Plane 12).
        default_pool = BLD._build_default_asset_pools(
            tile_lat=45.0, tile_lon=7.0, asset_region=nat_region
        )
        for cls in BLD.BLD_PLACEMENT_CLASSES:
            cls_label = BLD.BLD_CLASS_LABELS.get(cls, str(cls))
            entries = default_pool.get(cls, ())
            default_region_class[nat_region][cls_label] += len(entries)
            for asset in entries:
                kind = asset.get("kind", "object")
                default_kind_class[kind][cls_label] += 0  # ensure key exists
                default_kind_class[kind][cls_label] += 1
        # SFD Global pools.
        try:
            sfd_pool = BLD._build_sfd_asset_pools(
                tile_lat=45.0, tile_lon=7.0,
                asset_region=nat_region,
                library_exports=runtime_exports,
                cache_dir=args.cache_dir,
            )
        except Exception:
            sfd_pool = {cls: [] for cls in BLD.BLD_PLACEMENT_CLASSES}
        for cls in BLD.BLD_PLACEMENT_CLASSES:
            cls_label = BLD.BLD_CLASS_LABELS.get(cls, str(cls))
            entries = sfd_pool.get(cls, ())
            sfd_region_class[nat_region][cls_label] += len(entries)
            for asset in entries:
                kind = asset.get("kind", "object")
                sfd_kind_class[kind][cls_label] += 1

    print()
    print("--- Default X-Plane 12 pools per region (facades + default objects) ---")
    default_seen = [r for r in region_order if r in default_region_class
                    and any(default_region_class[r].values())]
    print(header)
    for r in default_seen:
        row_counts = [default_region_class[r].get(c, 0) for c in class_order]
        cells = "".join(f"{n:>10}" for n in row_counts)
        print(f"  {r:18}{cells}")
    if default_kind_class:
        print("    by kind:")
        for kind, counts in sorted(default_kind_class.items()):
            total = sum(counts.values()) // max(1, len(default_seen))
            print(f"      {kind:10} ~{total} entries per region")

    print()
    print("--- SFD Global pools per region (measured + catalogued buildings) ---")
    sfd_seen = [r for r in region_order if r in sfd_region_class
                and any(sfd_region_class[r].values())]
    if not sfd_seen:
        print("    (no SFD Global / SFD-flavoured assets found in scan)")
    else:
        print(header)
        for r in sfd_seen:
            row_counts = [sfd_region_class[r].get(c, 0) for c in class_order]
            cells = "".join(f"{n:>10}" for n in row_counts)
            print(f"  {r:18}{cells}")

    # ---- Combined Track A + SimHeaven + Default + SFD cross-tab ----------
    print()
    print("--- COMBINED region x class (ALL sources: Track A + SimHeaven + Default + SFD) ---")
    combined: dict[str, Counter] = defaultdict(Counter)
    for r, counts in region_class.items():
        for cls, n in counts.items():
            combined[r][cls] += n
    for r, counts in sh_region_class_counts.items():
        for cls, n in counts.items():
            combined[r][cls] += n
    for r, counts in default_region_class.items():
        for cls, n in counts.items():
            combined[r][cls] += n
    for r, counts in sfd_region_class.items():
        for cls, n in counts.items():
            combined[r][cls] += n
    combined_seen = [r for r in region_order if r in combined]
    for r in sorted(combined):
        if r not in combined_seen:
            combined_seen.append(r)
    print(header)
    for r in combined_seen:
        row_counts = [combined[r].get(c, 0) for c in class_order]
        if not args.show_empty_cells and not any(row_counts):
            continue
        cells = "".join(f"{n:>10}" for n in row_counts)
        print(f"  {r:18}{cells}")
    print()
    print("--- Source contribution per region (totals across all classes) ---")
    print(f"  {'region':18}{'TrackA':>10}{'SimHvn':>10}{'Default':>10}{'SFD':>10}{'TOTAL':>10}")
    for r in combined_seen:
        ta = sum(region_class.get(r, {}).values())
        sh = sum(sh_region_class_counts.get(r, {}).values())
        de = sum(default_region_class.get(r, {}).values())
        sf = sum(sfd_region_class.get(r, {}).values())
        total = ta + sh + de + sf
        if not args.show_empty_cells and total == 0:
            continue
        print(f"  {r:18}{ta:>10}{sh:>10}{de:>10}{sf:>10}{total:>10}")

    # ---- Per-library breakdown --------------------------------------------
    print()
    print("--- Per-library breakdown (region -> class counts) ---")
    by_lib: dict[str, list] = defaultdict(list)
    for lib_id, regions, cls, bounds_m, obj_path in rows:
        by_lib[lib_id].append((regions, cls, bounds_m))
    for lib_id in sorted(by_lib):
        lib_rows = by_lib[lib_id]
        lib_label = BLD.CURATED_EXTRA_BUILDING_LIBRARIES.get(lib_id, {}).get(
            "label", lib_id
        )
        print()
        print(f"  [{lib_id}] {lib_label}  ({len(lib_rows)} accepted candidates)")
        lib_rc: dict[str, Counter] = defaultdict(Counter)
        for regions, cls, _b in lib_rows:
            cls_label = BLD.BLD_CLASS_LABELS.get(cls, str(cls))
            keys = regions if regions else ("<unclassified>",)
            for r in keys:
                lib_rc[r][cls_label] += 1
        regions_here = [r for r in seen_regions if r in lib_rc]
        for r in regions_here:
            row_counts = [lib_rc[r].get(c, 0) for c in class_order]
            if not args.show_empty_cells and not any(row_counts):
                continue
            cells = "".join(f"{n:>10}" for n in row_counts)
            print(f"    {r:18}{cells}")

    return 0


def _footprint_wd(bounds_m):
    if not bounds_m:
        return 0.0, 0.0
    xmin, xmax, ymin, ymax = bounds_m
    return max(0.0, xmax - xmin), max(0.0, ymax - ymin)


if __name__ == "__main__":
    sys.exit(main())
