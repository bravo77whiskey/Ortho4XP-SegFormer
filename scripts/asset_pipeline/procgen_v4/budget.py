"""Performance contract: compare every v4 asset with the old library's.

    python budget.py [--sample 3000] [--all]

The old library's build (procgen/) is importable, so each v4 asset is
compared against the OLD asset at the same manifest row -- same archetype,
footprint, floors, region and seed -- not against an average.

Frame cost of an instanced object is modelled as triangles x the number of
instances drawn at that LOD, and instances in an LOD ring grow with the
ring's AREA (uniform building density).  The per-asset load is therefore

    load = sum_b  tris_b * (far_b^2 - near_b^2)

Rules checked per asset:
  * cull distance (last band end) identical to the old asset's;
  * load <= LOAD_RATIO x old load;
  * band 4 no heavier than the old one plus a small allowance;
  * band 3 at most two wall quads per wall per unit face (see
    buildings.far_group) plus the roof: BAND3_MAX triangles;
  * one texture page per OBJ (checked by construction in buildings.py).
Library-wide: the mean load ratio stays <= MEAN_LOAD_RATIO.

2026-09-27 smooth-LOD round: bands 2 and 3 now keep what band 1 shows at
the switch distance (chimneys, dormers, porches, shop canopies, eave line,
unit colours, the window texture), which raised the modelled load to about
1.35x the old library.  The model counts triangles x instances; nothing
changes the instance count, texture pages or draw batching.
Texture memory per tile is reported separately (pages a tile can load).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.normpath(os.path.join(HERE, "..", "procgen"))
for p in (HERE, OLD):
    if p not in sys.path:
        sys.path.insert(0, p)

import buildings  # noqa: E402

LOAD_RATIO = 4.5          # per asset; 7-unit terraces carry per-unit colours
MEAN_LOAD_RATIO = 1.40    # library mean
FAR_ALLOWANCE = 4
BAND3_MAX = 96


def _old_modules():
    from archetypes import build_archetype  # old package
    from archetypes.combo_styles import group_for_archetype
    import apply_lod_shells as als
    from lod_shell import roof_quad_mesh, shell_mesh
    return build_archetype, group_for_archetype, als, shell_mesh, \
        roof_quad_mesh


def _tris(faces):
    return sum(len(f) - 2 for f in faces)


def old_reference(row, variant, layouts, mods):
    build_archetype, group_for, als, shell_mesh, roof_quad_mesh = mods
    L, W, F = row["length_m"], row["width_m"], row["floors"]
    arch = variant["archetype"]
    lay = layouts["combos"][f"{row['flavor']}/{group_for(arch)}"]
    spec = build_archetype(arch, L, W, F, variant["seed"], lay, row["flavor"])
    meta = spec.meta
    scale = als._band_scale(L, W, F * 3.2)
    d = [0] + [int(round(b * scale)) for b in als.DEFAULT_BANDS]
    wall = als._strip_band(lay, meta["wall_strip"])
    plain = als._plain_strip_band(lay, meta["wall_strip"])
    roofb = als._strip_band(lay, meta["roof_strip"])
    kind = meta["kind"]
    b2 = shell_mesh(L, W, F, kind, meta["ridge_z"], meta["eave_z"], wall,
                    roofb, wall_mode="floors")
    top = meta["eave_z"] if kind == "pitched" else meta["ridge_z"]
    b3 = shell_mesh(L, W, F, "flat", top, meta["eave_z"], plain, roofb,
                    wall_mode="stretch")
    b4 = roof_quad_mesh(L, W, top, roofb)
    tris = [spec.tri_count(), _tris(b2[1]), _tris(b3[1]), _tris(b4[1])]
    return tris, d


def load(tris, d):
    return sum(t * (d[i + 1] ** 2 - d[i] ** 2) for i, t in enumerate(tris))


def check(rows, mods, layouts):
    out = []
    for row, variant in rows:
        L, W, F = row["length_m"], row["width_m"], row["floors"]
        otris, od = old_reference(row, variant, layouts, mods)
        _page, bands = buildings.build(variant["archetype"], L, W, F,
                                       variant["seed"], row["region"])
        ntris = [b.tri_count() for b in bands]
        nd, _ = buildings.band_distances(L, W, F, bands)
        ol, nl = load(otris, od), load(ntris, list(nd))
        problems = []
        if nd[-1] != od[-1]:
            problems.append(f"cull {nd[-1]} != old {od[-1]}")
        if nl > LOAD_RATIO * ol:
            problems.append(f"load x{nl / ol:.2f}")
        if ntris[3] > otris[3] + FAR_ALLOWANCE:
            problems.append(f"band4 {ntris[3]} > old {otris[3]}")
        if ntris[2] > BAND3_MAX:
            problems.append(f"band3 {ntris[2]} > {BAND3_MAX}")
        out.append({"arch": variant["archetype"], "region": row["region"],
                    "old": otris, "new": ntris, "ratio": nl / ol,
                    "problems": problems, "id": row["id"]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=3000)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    with open(os.path.join(OLD, "procgen_manifest.json")) as fh:
        manifest = json.load(fh)
    with open(os.path.join(OLD, "atlas_layout.json")) as fh:
        layouts = json.load(fh)
    rows = [(a, v) for a in manifest["assets"] if a.get("enabled")
            for v in a["variants"]]
    if not args.all:
        rows = random.Random(7).sample(rows, min(args.sample, len(rows)))
    res = check(rows, _old_modules(), layouts)
    by = {}
    for r in res:
        by.setdefault(r["arch"], []).append(r)
    print(f"{'archetype':10s} {'n':>5s}  old tris (b1/b2/b3/b4)   "
          f"new tris (b1/b2/b3/b4)   load ratio mean/max  fails")
    for arch in sorted(by):
        rs = by[arch]

        def mean_band(key, i):
            return statistics.mean(r[key][i] for r in rs)
        old = "/".join(f"{mean_band('old', i):.0f}" for i in range(4))
        new = "/".join(f"{mean_band('new', i):.0f}" for i in range(4))
        ratios = [r["ratio"] for r in rs]
        fails = sum(1 for r in rs if r["problems"])
        print(f"{arch:10s} {len(rs):5d}  {old:24s} {new:24s} "
              f"{statistics.mean(ratios):5.2f} / {max(ratios):5.2f}   "
              f"{fails}")
    total_old = sum(load(r["old"], [0, 1, 2, 3, 4]) for r in res)
    ratios = [r["ratio"] for r in res]
    fails = [r for r in res if r["problems"]]
    print(f"\nall: {len(res)} assets, load ratio mean {statistics.mean(ratios):.3f}"
          f" median {statistics.median(ratios):.3f} max {max(ratios):.3f};"
          f" {len(fails)} contract failures")
    for r in fails[:15]:
        print("  !", r["id"], r["arch"], r["problems"], r["old"], r["new"])
    mean_ok = statistics.mean(ratios) <= MEAN_LOAD_RATIO
    if not mean_ok:
        print(f"  ! library mean load ratio > {MEAN_LOAD_RATIO}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh)
    return 1 if (fails or not mean_ok) else 0


if __name__ == "__main__":
    sys.exit(main())
