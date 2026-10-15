"""List library exports whose mesh is far from the size its simHeaven path names.

simHeaven virtual paths carry their footprint (house_09x08x1.obj = 9 m along
x by 8 m along z, 1 floor). Other libraries may export the same path to their
own meshes; SFD Global Autogen's X-World integration draws house_06x04x2 as a
12.6 x 13.6 m house in Africa. The overlay and simHeaven's DSFs space
buildings by the named size, so oversized aliases interpenetrate.

    python scripts/library_audit/audit_simheaven_aliases.py [--pack "SFD Global Autogen"] [--csv out.csv]
"""
import argparse
import csv
import os
import re
import sys

CUSTOM_SCENERY = r"D:\SteamLibrary\steamapps\common\X-Plane 12\Custom Scenery"
NAME_DIMS = re.compile(r'_(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)(?:x(\d+(?:\.\d+)?))?\.obj$', re.I)
EXPORT_KINDS = ('EXPORT', 'EXPORT_EXCLUDE', 'EXPORT_EXTEND', 'EXPORT_BACKUP')


def name_dims(virtual_path):
    """(width_x, depth_z, floors) from a simHeaven path, or None."""
    m = NAME_DIMS.search(virtual_path)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), float(m.group(3)) if m.group(3) else None


def obj_bounds(path, cache={}):
    """(xmin, xmax, zmin, zmax) of an OBJ8's vertices, or None."""
    if path in cache:
        return cache[path]
    xs = []
    zs = []
    try:
        with open(path, errors='ignore') as handle:
            for line in handle:
                if line.startswith('VT'):
                    parts = line.split()
                    xs.append(float(parts[1]))
                    zs.append(float(parts[3]))
    except OSError:
        pass
    cache[path] = (min(xs), max(xs), min(zs), max(zs)) if xs else None
    return cache[path]


def library_exports(pack_dir, prefix='simheaven/'):
    """Yield (line_no, region, kind, virtual, real) for exports under ``prefix``."""
    region = None
    with open(os.path.join(pack_dir, 'library.txt'), errors='ignore') as handle:
        for line_no, raw in enumerate(handle, 1):
            parts = raw.split()
            if not parts:
                continue
            if parts[0] == 'REGION' and len(parts) > 1:
                region = parts[1]
            elif parts[0] in EXPORT_KINDS and len(parts) >= 3 and parts[1].lower().startswith(prefix):
                yield line_no, region, parts[0], parts[1], ' '.join(parts[2:])


def audit(pack_dir):
    rows = []
    for line_no, region, kind, virtual, real in library_exports(pack_dir):
        dims = name_dims(virtual)
        bounds = obj_bounds(os.path.join(pack_dir, real))
        row = {
            'line': line_no, 'region': region or '', 'kind': kind,
            'virtual': virtual, 'real': real,
            'name_w': '', 'name_d': '', 'mesh_w': '', 'mesh_d': '',
            'area_ratio': '', 'off_centre_m': '',
        }
        if bounds:
            xmin, xmax, zmin, zmax = bounds
            row['mesh_w'] = round(xmax - xmin, 1)
            row['mesh_d'] = round(zmax - zmin, 1)
            row['off_centre_m'] = round(max(abs(xmin + xmax), abs(zmin + zmax)) / 2, 1)
        if dims:
            row['name_w'], row['name_d'] = dims[0], dims[1]
            if bounds:
                row['area_ratio'] = round(row['mesh_w'] * row['mesh_d'] / (dims[0] * dims[1]), 2)
        rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--scenery', default=CUSTOM_SCENERY)
    ap.add_argument('--pack', default='SFD Global Autogen')
    ap.add_argument('--csv')
    ap.add_argument('--min-ratio', type=float, default=1.5,
                    help='report exports whose mesh area is at least this many times the named area')
    args = ap.parse_args(argv)
    rows = audit(os.path.join(args.scenery, args.pack))
    if args.csv:
        with open(args.csv, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    sized = [r for r in rows if r['area_ratio'] != '']
    over = [r for r in sized if r['area_ratio'] >= args.min_ratio]
    under = [r for r in sized if r['area_ratio'] <= 1 / args.min_ratio]
    print(f"{args.pack}: {len(rows)} simHeaven exports, {len(sized)} measurable, "
          f"{len(over)} >= {args.min_ratio}x the named area, {len(under)} <= 1/{args.min_ratio}x, "
          f"{sum(1 for r in rows if r['mesh_w'] == '')} meshes missing")
    by_region = {}
    for r in over:
        by_region.setdefault((r['region'], r['kind']), []).append(r)
    for (region, kind), group in sorted(by_region.items()):
        worst = max(group, key=lambda r: r['area_ratio'])
        print(f"  {region:20s} {kind:14s} {len(group):4d} oversized; worst {worst['virtual']} "
              f"-> {worst['real']} {worst['mesh_w']}x{worst['mesh_d']} m ({worst['area_ratio']}x)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
