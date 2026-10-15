"""Build a library pack that redraws oversized simHeaven aliases at their named size.

SFD Global Autogen's X-World integration exports simHeaven house paths to SFD
meshes regardless of size (house_06x04x2 drawn 12.6 x 13.6 m in Africa), so
buildings spaced by the named size interpenetrate. This builds a pack that sits
above the target pack in scenery_packs.ini and, per region, EXPORT_EXCLUDEs each
oversized path to a mesh that fits: simHeaven's own mesh for that region where it
has one, else the largest building mesh of the source packs that fits inside the
named footprint. The pack links (or copies) the meshes and textures it uses, as
X-Plane libraries cannot point outside their own folder. Picks are steered by
library_size_overrides.json.

Dry run (default) writes a report only:
    python scripts/library_audit/build_library_overrides.py
Install the pack and list it above the target pack (scenery_packs.ini backed up):
    python scripts/library_audit/build_library_overrides.py --install
"""
import argparse
import csv
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_simheaven_aliases import (  # noqa: E402
    CUSTOM_SCENERY, EXPORT_KINDS, library_exports, name_dims, obj_bounds,
)

DEFINITION = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'library_size_overrides.json')
DEFAULT_OUT = r"E:\Ortho4XP_Review\library_overrides"
MARKER = 'o4xp_library_overrides.json'
BUILDING_PREFIXES = ('simheaven/houses/', 'simheaven/residential/')
SIMHEAVEN_PACK = re.compile(r'^simHeaven_X-World_(.+?)-\d', re.I)
# SFD keeps each region's meshes in one folder
SFD_STYLE_FOLDER = {
    'africa': 'Africa', 'asia': 'Asia', 'asia_south': 'Asia', 'asia_LHD': 'Asia',
    'australia': 'Australia', 'south_america': 'South_America', 'scandinavia': 'Scandinavia',
    'mediterranean': 'Mediterranean', 'mediterranean_north': 'Mediterranean',
    'new_england': 'New_England', 'west_coast': 'US_West_Coast',
}
TEXTURE_KEYS = ('TEXTURE', 'TEXTURE_LIT', 'TEXTURE_NORMAL', 'TEXTURE_DRAPED', 'TEXTURE_DRAPED_NORMAL')
IMAGE_EXTS = ('.dds', '.png')


@dataclass(frozen=True)
class Mesh:
    pack: str
    rel: str
    width: float   # of the origin-centred box holding the mesh, x
    depth: float   # same, z
    style: str     # SFD region folder or simHeaven continent

    @property
    def area(self):
        return self.width * self.depth

    @property
    def ref(self):
        return f"{self.pack}/{self.rel}"


@dataclass
class Group:
    region: str
    virtual: str
    name_w: float
    name_d: float
    target: list = field(default_factory=list)       # (line, kind, real, mesh)
    action: str = ''
    picks: list = field(default_factory=list)        # Mesh
    note: str = ''

    @property
    def worst_ratio(self):
        ratios = [m.area / (self.name_w * self.name_d) for _, _, _, m in self.target if m]
        return max(ratios) if ratios else 0.0


def measure(scenery, pack, rel, style):
    bounds = obj_bounds(os.path.join(scenery, pack, rel))
    if bounds is None:
        return None
    xmin, xmax, zmin, zmax = bounds
    return Mesh(pack, rel.replace('\\', '/'), 2 * max(-xmin, xmax), 2 * max(-zmin, zmax), style)


def simheaven_packs(scenery):
    for folder in sorted(os.listdir(scenery)):
        m = SIMHEAVEN_PACK.match(folder)
        if m and os.path.isfile(os.path.join(scenery, folder, 'library.txt')):
            yield folder, m.group(1)


def fits(mesh, group, cfg):
    slack = float(cfg['fit_slack'])
    return (
        mesh.width <= group.name_w * slack and mesh.depth <= group.name_d * slack and
        mesh.area >= float(cfg['min_fill']) * group.name_w * group.name_d
    )


def plan(scenery, cfg):
    """Groups of (region, path) the target pack draws oversized, with their picks."""
    target = cfg['target_pack']
    max_ratio = float(cfg['max_area_ratio'])
    groups = {}
    for line_no, region, kind, virtual, real in library_exports(os.path.join(scenery, target)):
        if not virtual.lower().startswith(BUILDING_PREFIXES):
            continue
        dims = name_dims(virtual)
        if not dims or not region or region == 'all' or kind == 'EXPORT_BACKUP':
            continue
        key = (region, virtual)
        if key not in groups:
            groups[key] = Group(region, virtual, dims[0], dims[1])
        groups[key].target.append(
            (line_no, kind, real, measure(scenery, target, real, real.split('/')[0]))
        )
    groups = {k: g for k, g in groups.items() if g.worst_ratio >= max_ratio}

    # simHeaven's own exports, and every building mesh the source packs use
    sh_exports = {}
    pool = {}
    for folder, continent in simheaven_packs(scenery):
        for _line, _region, kind, virtual, real in library_exports(os.path.join(scenery, folder)):
            if not virtual.lower().startswith(BUILDING_PREFIXES):
                continue
            mesh = measure(scenery, folder, real, continent)
            if mesh is None:
                continue
            pool[mesh.ref] = mesh
            if kind != 'EXPORT_BACKUP':
                sh_exports.setdefault((continent, virtual), {})[mesh.ref] = mesh
    for group in groups.values():
        for _line, _kind, _real, mesh in group.target:
            if mesh is not None:
                pool[mesh.ref] = mesh
    for _line, _region, _kind, virtual, real in library_exports(os.path.join(scenery, target)):
        if virtual.lower().startswith(BUILDING_PREFIXES) and f"{target}/{real}" not in pool:
            mesh = measure(scenery, target, real, real.split('/')[0])
            if mesh is not None:
                pool[mesh.ref] = mesh

    manual = cfg.get('manual') or []
    for group in sorted(groups.values(), key=lambda g: (g.region, g.virtual)):
        rule = next((
            r for r in manual
            if r.get('virtual') == group.virtual and r.get('region', group.region) == group.region
        ), None)
        if rule is not None:
            refs = rule['mesh'] if isinstance(rule['mesh'], list) else [rule['mesh']]
            if refs == ['keep']:
                group.action = 'keep (manual)'
                continue
            group.action = 'manual'
            for ref in refs:
                pack, _, rel = ref.partition('/')
                mesh = pool.get(ref) or measure(scenery, pack, rel, '')
                if mesh is None:
                    group.note = f"manual mesh not found: {ref}"
                else:
                    group.picks.append(mesh)
            continue
        own = [
            m for continent in cfg['simheaven_packs_by_region'].get(group.region, ())
            for m in sh_exports.get((continent, group.virtual), {}).values()
            if m.area < max_ratio * group.name_w * group.name_d
        ]
        if own:
            group.action = 'simheaven'
            group.picks = sorted({m.ref: m for m in own}.values(), key=lambda m: m.ref)
            continue
        styles = (
            SFD_STYLE_FOLDER.get(group.region),
            *cfg['simheaven_packs_by_region'].get(group.region, ()),
        )
        fitting = [m for m in pool.values() if fits(m, group, cfg)]
        if not fitting:
            def overshoot(m):
                return max(m.width / group.name_w, m.depth / group.name_d)
            near = [m for m in pool.values() if overshoot(m) <= float(cfg['closest_max_overshoot'])]
            if not near:
                group.action = 'unresolved'
                group.note = 'no mesh fits inside the named footprint; target exports kept'
                continue
            group.action = 'closest'
            group.picks = [min(near, key=lambda m: (round(overshoot(m), 2), m.style not in styles, m.ref))]
            group.note = f"nothing fits inside; sides up to {overshoot(group.picks[0]):.2f}x the named size"
            continue
        best = max(m.area for m in fitting)
        group.action = 'fit'
        group.picks = [max(
            fitting,
            key=lambda m: (
                m.area >= float(cfg['same_style_within']) * best and m.style in styles,
                m.area, m.ref,
            ),
        )]
    return list(groups.values())


def texture_files(scenery, mesh):
    """Pack-relative paths of the textures an OBJ uses, with their .dds/.png twins."""
    obj_path = os.path.join(scenery, mesh.pack, mesh.rel)
    pack_root = os.path.join(scenery, mesh.pack)
    found = []
    with open(obj_path, errors='ignore') as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2 or parts[0] not in TEXTURE_KEYS:
                continue
            tex = os.path.normpath(os.path.join(os.path.dirname(obj_path), parts[1]))
            stem = os.path.splitext(tex)[0]
            for candidate in [tex] + [stem + ext for ext in IMAGE_EXTS]:
                rel = os.path.relpath(candidate, pack_root)
                if rel.startswith('..') or not os.path.isfile(candidate):
                    continue
                rel = rel.replace('\\', '/')
                # X-Plane and Windows ignore case: GR_texture.dds is gr_texture.dds
                if rel.lower() not in {f.lower() for f in found}:
                    found.append(rel)
    return found


def pack_tag(pack):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', pack)


def region_definitions(scenery, target):
    """REGION_DEFINE blocks of the target library, as {region: [lines]}."""
    defs = {}
    current = None
    with open(os.path.join(scenery, target, 'library.txt'), errors='ignore') as handle:
        for raw in handle:
            parts = raw.split()
            if not parts:
                continue
            if parts[0] == 'REGION_DEFINE':
                current = parts[1]
                defs[current] = [raw.strip()]
            elif current and parts[0].startswith('REGION_') and parts[0] != 'REGION':
                defs[current].append(raw.strip())
            else:
                current = None
    return defs


def library_text(groups, region_defs, cfg):
    lines = [
        'A', '800', 'LIBRARY', '',
        f"# Generated by scripts/library_audit/build_library_overrides.py from",
        f"# library_size_overrides.json. Redraws simHeaven paths that {cfg['target_pack']}",
        '# exports oversized at their named size. Rebuild instead of editing.', '',
    ]
    regions = sorted({g.region for g in groups if g.picks})
    for region in regions:
        lines.extend(region_defs[region])
    for region in regions:
        lines += ['', f"REGION {region}"]
        for group in sorted((g for g in groups if g.picks and g.region == region), key=lambda g: g.virtual):
            for mesh in group.picks:
                lines.append(f"EXPORT_EXCLUDE {group.virtual} objects/{pack_tag(mesh.pack)}/{mesh.rel}")
    return '\n'.join(lines) + '\n'


def files_to_place(scenery, groups):
    """{pack-relative destination: absolute source} for every mesh and texture used."""
    files = {}
    seen = set()
    for mesh in {m.ref: m for g in groups for m in g.picks}.values():
        base = f"objects/{pack_tag(mesh.pack)}/"
        for rel in [mesh.rel] + texture_files(scenery, mesh):
            if (base + rel).lower() not in seen:  # case-insensitive, as on disk
                seen.add((base + rel).lower())
                files[base + rel] = os.path.join(scenery, mesh.pack, rel)
    return files


def write_report(groups, files, cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'dryrun_overrides.csv')
    with open(csv_path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow([
            'region', 'virtual', 'named_m', 'target_meshes', 'worst_area_ratio',
            'action', 'replacement', 'replacement_m', 'fill', 'note',
        ])
        for g in sorted(groups, key=lambda g: (g.region, g.virtual)):
            target = '; '.join(
                f"{real} {m.width:.1f}x{m.depth:.1f}" if m else f"{real} (missing)"
                for _l, _k, real, m in g.target
            )
            writer.writerow([
                g.region, g.virtual, f"{g.name_w:g}x{g.name_d:g}", target, round(g.worst_ratio, 2),
                g.action, '; '.join(m.ref for m in g.picks),
                '; '.join(f"{m.width:.1f}x{m.depth:.1f}" for m in g.picks),
                '; '.join(f"{m.area / (g.name_w * g.name_d):.2f}" for m in g.picks), g.note,
            ])
    size = sum(os.path.getsize(p) for p in files.values() if os.path.isfile(p))
    counts = {}
    for g in groups:
        counts.setdefault(g.region, {}).setdefault(g.action, 0)
        counts[g.region][g.action] += 1

    def short(ref):
        return ref.replace('simHeaven_X-World_', 'sH ').replace('SFD Global Autogen', 'SFD')

    md = [
        f"# {cfg['pack_name']} dry run", '',
        f"{len(groups)} (region, path) pairs where {cfg['target_pack']} draws a simHeaven path "
        f"at least {cfg['max_area_ratio']}x its named area. Fill = replacement area / named area.", '',
        '| region | ' + ' | '.join(sorted({g.action for g in groups})) + ' |',
        '|---|' + '---|' * len({g.action for g in groups}),
    ]
    for region, by_action in sorted(counts.items()):
        md.append(f"| {region} | " + ' | '.join(
            str(by_action.get(a, 0)) for a in sorted({g.action for g in groups})) + ' |')
    md += ['', f"Pack: {len(files)} files, {size / 1e6:.1f} MB (hard-linked on the same drive).", '']
    for region in sorted(counts):
        md += [f"## {region}", '', '| path | named | drawn now (worst) | action | replacement | size | fill |',
               '|---|---|---|---|---|---|---|']
        for g in sorted((g for g in groups if g.region == region), key=lambda g: g.virtual):
            worst = max((m for *_, m in g.target if m), key=lambda m: m.area, default=None)
            md.append(
                f"| {g.virtual.split('/', 1)[1]} | {g.name_w:g}x{g.name_d:g} | "
                f"{(f'{worst.width:.1f}x{worst.depth:.1f} ({g.worst_ratio:.1f}x)') if worst else '-'} | "
                f"{g.action} | {'<br>'.join(short(m.ref) for m in g.picks) or g.note} | "
                f"{'<br>'.join(f'{m.width:.1f}x{m.depth:.1f}' for m in g.picks)} | "
                f"{'<br>'.join(f'{m.area / (g.name_w * g.name_d):.2f}' for m in g.picks)} |"
            )
        md.append('')
    with open(os.path.join(out_dir, 'dryrun_overrides.md'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(md))
    print(f"{len(groups)} oversized (region, path) pairs in {cfg['target_pack']} "
          f"(worst mesh >= {cfg['max_area_ratio']}x the named area)")
    for region, by_action in sorted(counts.items()):
        print(f"  {region:20s} " + '  '.join(f"{a}={n}" for a, n in sorted(by_action.items())))
    print(f"pack would hold {len(files)} files, {size / 1e6:.1f} MB (hard links on the same drive)")
    print(f"report: {csv_path}")
    return csv_path


def link_or_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        try:
            shutil.copy2(src, dst)
        except OSError as exc:
            raise SystemExit(f"cannot copy {src}: {exc}")


def list_above_target(scenery, pack_name, target):
    """Put the pack directly above the target pack in scenery_packs.ini (backed up once)."""
    ini = os.path.join(scenery, 'scenery_packs.ini')
    with open(ini, encoding='utf-8', errors='ignore') as handle:
        lines = handle.read().splitlines()
    ours = f"SCENERY_PACK Custom Scenery/{pack_name}/"
    target_line = f"SCENERY_PACK Custom Scenery/{target}/"
    if target_line not in lines:
        raise SystemExit(f"{target} is not an enabled pack in scenery_packs.ini")
    if ours in lines and lines.index(ours) < lines.index(target_line):
        return False
    backup = ini + '.bak_before_o4xp_library_overrides'
    if not os.path.exists(backup):
        shutil.copy2(ini, backup)
    lines = [line for line in lines if line != ours]
    lines.insert(lines.index(target_line), ours)
    with open(ini, 'w', encoding='utf-8', newline='\n') as handle:
        handle.write('\n'.join(lines) + '\n')
    return True


def install(scenery, groups, files, cfg):
    pack_dir = os.path.join(scenery, cfg['pack_name'])
    if os.path.exists(pack_dir):
        if not os.path.isfile(os.path.join(pack_dir, MARKER)):
            raise SystemExit(f"{pack_dir} exists and was not built by this script; not touching it")
        shutil.rmtree(pack_dir)
    # the marker first, so a failed build can be rebuilt over
    os.makedirs(pack_dir)
    with open(os.path.join(pack_dir, MARKER), 'w') as handle:
        json.dump({'target_pack': cfg['target_pack'], 'overrides': []}, handle)
    region_defs = region_definitions(scenery, cfg['target_pack'])
    for region in {g.region for g in groups if g.picks}:
        for line in region_defs[region]:
            parts = line.split()
            if parts[0] == 'REGION_BITMAP':
                link_or_copy(
                    os.path.join(scenery, cfg['target_pack'], parts[1]),
                    os.path.join(pack_dir, parts[1]),
                )
    for rel, src in files.items():
        link_or_copy(src, os.path.join(pack_dir, rel))
    with open(os.path.join(pack_dir, 'library.txt'), 'w', newline='\n') as handle:
        handle.write(library_text(groups, region_defs, cfg))
    with open(os.path.join(pack_dir, MARKER), 'w') as handle:
        json.dump({
            'target_pack': cfg['target_pack'],
            'overrides': [
                {'region': g.region, 'virtual': g.virtual, 'action': g.action,
                 'meshes': [m.ref for m in g.picks]}
                for g in groups if g.picks
            ],
        }, handle, indent=1)
    moved = list_above_target(scenery, cfg['pack_name'], cfg['target_pack'])
    print(f"installed {pack_dir}" + (" and listed it above " + cfg['target_pack'] if moved else ''))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--scenery', default=CUSTOM_SCENERY)
    ap.add_argument('--definition', default=DEFINITION)
    ap.add_argument('--out', default=DEFAULT_OUT, help='where the dry-run report goes')
    ap.add_argument('--install', action='store_true')
    args = ap.parse_args(argv)
    with open(args.definition) as handle:
        cfg = json.load(handle)
    groups = plan(args.scenery, cfg)
    files = files_to_place(args.scenery, groups)
    write_report(groups, files, cfg, args.out)
    with open(os.path.join(args.out, 'library.txt.preview'), 'w', newline='\n') as handle:
        handle.write(library_text(groups, region_definitions(args.scenery, cfg['target_pack']), cfg))
    if args.install:
        install(args.scenery, groups, files, cfg)
    return 0


if __name__ == '__main__':
    sys.exit(main())
