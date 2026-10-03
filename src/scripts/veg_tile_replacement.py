"""Command line for tile-wide vegetation replacement.

The generated vegetation overlay carries the forest exclusion itself; this tool
handles everything around it — inspecting what a tile's other scenery draws,
removing the embedded trees the exclusion cannot reach, migrating tiles that
were generated before the feature existed, and putting all of it back.

    inventory   list what each enabled package draws in a tile
    preview     show the exact changes and the storage they need
    apply       install them, with verified backups and a journal
    migrate     add or remove the exclusion on an already generated overlay
    restore     undo a tile's replacement from its verified backups
    status      report installed tiles and any interrupted operation
    adopt       reuse verified originals from the old simHeaven cleanup run

Every destructive subcommand refuses to run while X-Plane or a generation run
is active, and stops rather than overwriting a file it does not recognise.
"""

import argparse
import json
import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import O4_Veg_Replacement as CLEANUP  # noqa: E402


def _default_managed_root():
    return os.path.join(
        os.path.dirname(_SRC), CLEANUP.MANAGED_DIR_NAME
    )


def _default_overlay_dsf(data_dir, lat, lon):
    """Where the generated overlay for a tile lives under the data directory."""
    return os.path.join(
        data_dir or os.path.dirname(_SRC),
        CLEANUP.OVERLAY_PACK_NAME,
        CLEANUP.tile_dsf_relpath(lat, lon),
    )


def _tile_args(args):
    if args.tile:
        return CLEANUP._tile_latlon(args.tile)
    if args.lat is None or args.lon is None:
        raise SystemExit("give --tile +47+008, or --lat and --lon")
    return int(args.lat), int(args.lon)


def _build(args):
    lat, lon = _tile_args(args)
    store = CLEANUP.ManagedStore(args.managed_root or _default_managed_root())
    inventory = CLEANUP.inventory_tile(
        args.custom_scenery_dir, lat, lon, rules=store.rules()
    )
    plan = CLEANUP.plan_tile(inventory, store)
    conflicts = CLEANUP.validate_scenery_order(args.custom_scenery_dir, lat, lon)
    return lat, lon, store, inventory, plan, conflicts


def cmd_inventory(args):
    _lat, _lon, _store, inventory, _plan, conflicts = _build(args)
    print(json.dumps({"inventory": inventory, "scenery_order": conflicts},
                     indent=2, sort_keys=True, default=str))
    return 0


def cmd_preview(args):
    _lat, _lon, store, inventory, plan, conflicts = _build(args)
    print(CLEANUP.format_report(plan, inventory, conflicts, store=store))
    return 0


def cmd_apply(args):
    lat, lon, store, inventory, plan, conflicts = _build(args)
    blocking = [item for item in conflicts if item["severity"] == "error"]
    if blocking and not args.force_order:
        for item in blocking:
            print("scenery order: %s" % item["message"], file=sys.stderr)
        print("fix scenery_packs.ini, or pass --force-order to proceed anyway",
              file=sys.stderr)
        return 2
    overlay = args.overlay_dsf or _default_overlay_dsf(args.xplane_data_dir, lat, lon)
    if args.exclusion:
        problem = CLEANUP.check_overlay_suitable(overlay, lat, lon)
        if problem:
            print("overlay not ready: %s" % problem, file=sys.stderr)
            return 2
        CLEANUP.set_overlay_exclusion(overlay, lat, lon, True, store)
    manifest = CLEANUP.apply_plan(
        plan, store, force_space=args.force_space, ignore_busy=args.ignore_busy
    )
    print(CLEANUP.format_report(plan, inventory, conflicts, applied=manifest,
                                store=store))
    return 0


def cmd_migrate(args):
    lat, lon = _tile_args(args)
    store = CLEANUP.ManagedStore(args.managed_root or _default_managed_root())
    overlay = args.overlay_dsf or _default_overlay_dsf(args.xplane_data_dir, lat, lon)
    problem = CLEANUP.check_overlay_suitable(overlay, lat, lon)
    if args.enable and problem:
        print("overlay not suitable: %s" % problem, file=sys.stderr)
        return 2
    result = CLEANUP.set_overlay_exclusion(
        overlay, lat, lon, args.enable, store, preview=args.preview
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.preview or not args.enable or args.exclusion_only:
        return 0
    _, _, store, inventory, plan, conflicts = _build(args)
    manifest = CLEANUP.apply_plan(
        plan, store, force_space=args.force_space, ignore_busy=args.ignore_busy
    )
    print(CLEANUP.format_report(plan, inventory, conflicts, applied=manifest,
                                store=store))
    return 0


def cmd_restore(args):
    lat, lon = _tile_args(args)
    store = CLEANUP.ManagedStore(args.managed_root or _default_managed_root())
    overlay = args.overlay_dsf or _default_overlay_dsf(args.xplane_data_dir, lat, lon)
    result = CLEANUP.restore_tile(
        store, CLEANUP.tile_name(lat, lon),
        overlay_dsf=None if args.keep_exclusion else overlay,
        ignore_busy=args.ignore_busy,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 1 if result.get("conflicts") else 0


def cmd_status(args):
    store = CLEANUP.ManagedStore(args.managed_root or _default_managed_root())
    pending = store.pending()
    payload = {
        "managed_root": store.root,
        "tiles": store.tiles(),
        "pending": pending,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if pending:
        print(
            "\nAn operation on tile %s was interrupted after %d change(s). "
            "Re-run apply to finish it, or restore to put it back."
            % (pending.get("tile"), len(pending.get("completed", []))),
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_adopt(args):
    store = CLEANUP.ManagedStore(args.managed_root or _default_managed_root())
    result = CLEANUP.adopt_backups_from_strip_output(store, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _add_tile_options(parser):
    parser.add_argument("--tile", help="tile name, e.g. +47+008")
    parser.add_argument("--lat", type=int)
    parser.add_argument("--lon", type=int)
    parser.add_argument("--managed-root", default=None,
                        help="where backups, the journal and tile manifests live")
    parser.add_argument("--overlay-dsf", default=None,
                        help="the generated vegetation DSF for this tile")
    parser.add_argument("--xplane-data-dir", default=None,
                        help="root holding %s" % CLEANUP.OVERLAY_PACK_NAME)


def _add_scenery_options(parser):
    parser.add_argument("--custom-scenery-dir", required=True,
                        help="X-Plane root or its Custom Scenery folder")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inventory = sub.add_parser("inventory", help="list what a tile draws")
    _add_tile_options(inventory)
    _add_scenery_options(inventory)
    inventory.set_defaults(func=cmd_inventory)

    preview = sub.add_parser("preview", help="show planned changes, change nothing")
    _add_tile_options(preview)
    _add_scenery_options(preview)
    preview.set_defaults(func=cmd_preview)

    apply_cmd = sub.add_parser("apply", help="install the embedded-tree cleanup")
    _add_tile_options(apply_cmd)
    _add_scenery_options(apply_cmd)
    apply_cmd.add_argument("--exclusion", action="store_true",
                           help="also add the tile exclusion to the overlay")
    apply_cmd.add_argument("--force-space", action="store_true")
    apply_cmd.add_argument("--force-order", action="store_true")
    apply_cmd.add_argument("--ignore-busy", action="store_true",
                           help="skip the X-Plane / generation running check")
    apply_cmd.set_defaults(func=cmd_apply)

    migrate = sub.add_parser(
        "migrate", help="add or remove the exclusion on an existing overlay")
    _add_tile_options(migrate)
    _add_scenery_options(migrate)
    migrate.add_argument("--enable", dest="enable", action="store_true",
                         default=True)
    migrate.add_argument("--disable", dest="enable", action="store_false")
    migrate.add_argument("--preview", action="store_true")
    migrate.add_argument("--exclusion-only", action="store_true",
                         help="do not run the embedded-tree cleanup")
    migrate.add_argument("--force-space", action="store_true")
    migrate.add_argument("--ignore-busy", action="store_true")
    migrate.set_defaults(func=cmd_migrate)

    restore = sub.add_parser("restore", help="undo a tile's replacement")
    _add_tile_options(restore)
    restore.add_argument("--keep-exclusion", action="store_true",
                         help="restore source scenery but leave the overlay alone")
    restore.add_argument("--ignore-busy", action="store_true")
    restore.set_defaults(func=cmd_restore)

    status = sub.add_parser("status", help="installed tiles and interrupted work")
    status.add_argument("--managed-root", default=None)
    status.set_defaults(func=cmd_status)

    adopt = sub.add_parser(
        "adopt", help="reuse verified originals from the old cleanup run")
    adopt.add_argument("--managed-root", default=None)
    adopt.add_argument("--output", required=True,
                       help="the --output folder of strip_simheaven_vegetation")
    adopt.set_defaults(func=cmd_adopt)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CLEANUP.ConflictError as exc:
        print("stopped: %s" % exc, file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
