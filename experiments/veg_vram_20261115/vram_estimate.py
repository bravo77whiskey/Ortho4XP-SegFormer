"""Estimated X-Plane VRAM of the vegetation a review run plants, per location.

  python vram_estimate.py RUN [RUN ...] [--pack DIR] [--json OUT]

X-Plane loads every distinct .for file a scene references, each with its own
3-d tree meshes (the meshes are inline in the file, and nothing in the forest
format shares them between files), and keeps instance data for the plants
within each forest's LOD. This script counts, per review location (its ZL16
and ZL17-19 samples together, a stand-in for one scene):

  * forest files: distinct physical .for files (a copy pack file, or
    X-Plane's own), their mesh vertices and indices, and an estimate of
    their VRAM at VERTEX_BYTES / INDEX_BYTES;
  * plants: area x density/255 / SPACING^2 x layers (GROUP layers of the
    .for), point-mode polygons one plant per point, by kind (forest,
    thicket, shrub, single tree, grass).

The byte sizes are assumptions (11 floats per VERTEX line as written, 32-bit
indices); X-Plane's real layout is not documented. Use the numbers to
compare runs, not as absolute VRAM.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

import O4_Forest_Heights as FH  # noqa: E402

DATA = os.environ.get("O4_REVIEW_DATA", r"G:\Ortho4XP_Review")
XP = r"D:\SteamLibrary\steamapps\common\X-Plane 12\Resources\default scenery\1200 forests"
PACK = os.path.join(DATA, "_tmp", FH.PACK_NAME)
VERTEX_BYTES = 44
INDEX_BYTES = 4
LOCATIONS = ("khh", "szx", "wnz", "shouguang")
TEX_PX = 4096


def library(lib_dir, season="sum"):
    """{virtual path: physical file} of a library for one season (plain EXPORT as fallback)."""
    out = {}
    path = os.path.join(lib_dir, "library.txt")
    if not os.path.isfile(path):
        return out
    for line in open(path, encoding="utf-8", errors="ignore"):
        p = line.split()
        if len(p) == 4 and p[0] == "EXPORT_SEASON" and p[1] == season:
            out[p[2]] = os.path.normpath(os.path.join(lib_dir, p[3]))
        elif len(p) == 3 and p[0] == "EXPORT":
            out.setdefault(p[1], os.path.normpath(os.path.join(lib_dir, p[2])))
    return out


_FILE = {}


def for_stats(path):
    """(vertices, indices, spacing x, spacing z, layers) of a .for file, or None."""
    if path in _FILE:
        return _FILE[path]
    res = None
    if path and os.path.isfile(path):
        verts = idx = 0
        sx = sz = None
        layers = set()
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                c = line[:1]
                if c in "VI ":   # VERTEX / IDX lines: most of the file
                    continue
                p = line.split()
                if not p:
                    continue
                if p[0] == "MESH" and len(p) >= 6:
                    verts += int(float(p[4]))
                    idx += int(float(p[5]))
                elif p[0] == "SPACING":
                    sx, sz = float(p[1]), float(p[2])
                elif p[0] in ("TREE", "TREE2"):
                    layers.add(p[-2] if p[0] == "TREE" else p[-2])
                elif p[0] == "\tTREE" or p[0] == "\tTREE2":
                    pass
        if sx:
            res = (verts, idx, sx, sz, max(1, len(layers)))
    _FILE[path] = res
    return res


def area_px(ring):
    a = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
        a += x0 * y1 - x1 * y0
    return abs(a) / 2


def perimeter_px(ring):
    return sum(math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]))


def load(run, key):
    path = os.path.join(DATA, "runs", run, "BI", key + ".js")
    text = open(path, encoding="utf-8").read()
    return json.loads(text[text.index("] = ") + 4:].rstrip().rstrip(";"))


_GRASS = {}


def grass_file(run, path):
    """The grass .for a review run wrote for ``path`` (grass/<tile>/cNN.for), or None."""
    key = (run, path)
    if key not in _GRASS:
        found = None
        builds = os.path.join(DATA, "runs", run, "BI", "_builds")
        for b in sorted(os.listdir(builds)) if os.path.isdir(builds) else ():
            f = os.path.join(builds, b, "out", "yOrtho4XP_Grass_Overlays", *path.split("/"))
            if os.path.isfile(f):
                found = f
                break
        _GRASS[key] = found
    return _GRASS[key]


def kind_of(path, den):
    if path.startswith("grass/"):
        return "grass"
    parsed = FH.parse_virtual(path)
    kind = parsed[1] if parsed else ""
    src = parsed[0] if parsed else path
    if kind == FH.MANTLE_KIND:
        return "thicket"
    if den >= 512:
        return "single"
    if kind == "s" or "/shrubs_" in src or "/shrb_" in src or src.startswith("lib/g8/shrb"):
        return "shrub"
    return "forest"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--pack", default=PACK)
    ap.add_argument("--json")
    ap.add_argument("--share", type=float, default=FH.MERGE_SHARE,
                    help="merge what-if: copies under this share of their group move")
    args = ap.parse_args()
    libs = library(XP)
    libs.update(library(args.pack))
    meta = json.load(open(os.path.join(DATA, "sets", "hotspots", "BI", "samples.json")))
    mpp_of = {}
    for s in meta:
        lat_s, lon_w, lat_n, lon_e = s["bbox"]
        mpp_of[s["key"]] = (lon_e - lon_w) * math.cos(math.radians((lat_s + lat_n) / 2)) * 111320 / TEX_PX
    report = {}
    for run in args.runs:
        keys = sorted(k[:-3] for k in os.listdir(os.path.join(DATA, "runs", run, "BI")) if k.endswith(".js"))
        per_loc = defaultdict(lambda: {"files": {}, "plants": defaultdict(float),
                                       "use": defaultdict(float), "unknown": set(), "polys": 0,
                                       "phys": {}})
        for key in keys:
            loc = next((l for l in LOCATIONS if key.startswith(l)), key)
            try:
                d = load(run, key)
            except (OSError, ValueError):
                continue
            mpp = mpp_of.get(key, 1.0)
            L = per_loc[loc]
            for v in d["veg"]:
                path = v["a"]
                den = int(float(v.get("d", 255)))
                kind = kind_of(path, den)
                L["polys"] += 1
                if kind == "grass":
                    phys = grass_file(run, path)
                    st = for_stats(phys)
                    if st is not None:
                        L["files"].setdefault(phys, (st[0], st[1], kind))
                else:
                    phys = libs.get(path)
                    st = for_stats(phys)
                    if st is None:
                        L["unknown"].add(path)
                        continue
                    if phys not in L["files"] or L["files"][phys][2] == "single":
                        L["files"][phys] = (st[0], st[1], kind)
                    L["phys"][path] = phys
                if v.get("pt"):
                    n = float(v["pt"])
                elif st is None:
                    continue
                else:
                    rings = v["r"]
                    if den % 512 >= 256:   # line mode: one row along the outline
                        n = perimeter_px(rings[0]) * mpp / st[2] * st[4]
                    else:
                        a = (area_px(rings[0]) - sum(area_px(h) for h in rings[1:])) * mpp * mpp
                        n = a * (den % 256) / 255.0 / (st[2] * st[3]) * st[4]
                L["plants"][kind] += n
                L["use"][path] += n
        out = {}
        for loc, L in sorted(per_loc.items()):
            verts = sum(f[0] for f in L["files"].values())
            idx = sum(f[1] for f in L["files"].values())
            mesh_mb = (verts * VERTEX_BYTES + idx * INDEX_BYTES) / 2 ** 20
            by_kind = defaultdict(int)
            for _v, _i, k in L["files"].values():
                by_kind[k] += 1
            # What merge_rare_copies would leave (runs made with it off).
            moves = FH.merge_rare_copies(dict(L["use"]), share=args.share)
            targets = {moves.get(path, path) for path in L["phys"]}   # targets are planted paths
            files2 = {L["phys"][t] for t in targets} | {ph for ph, f in L["files"].items() if f[2] == "grass"}
            v2 = sum(L["files"][ph][0] for ph in files2)
            i2 = sum(L["files"][ph][1] for ph in files2)
            merged = {"files": len(files2),
                      "mesh_mb": round((v2 * VERTEX_BYTES + i2 * INDEX_BYTES) / 2 ** 20, 1),
                      "moves": len(moves)}
            out[loc] = {
                "files": len(L["files"]), "files_by_kind": dict(by_kind),
                "mesh_vertices": verts, "mesh_mb": round(mesh_mb, 1),
                "plants": {k: int(n) for k, n in sorted(L["plants"].items())},
                "polygons": L["polys"],
                "unknown": sorted(L["unknown"])[:5],
                "if_merged": merged,
            }
            p = out[loc]
            print(f"{run:26s} {loc:9s} files {p['files']:4d} {dict(by_kind)}  mesh ~{p['mesh_mb']:7.1f} MB")
            print(f"{'':36s} plants " + "  ".join(f"{k} {n:,}" for k, n in p["plants"].items()))
            if moves:
                print(f"{'':36s} if merged: {merged['moves']} copies move -> files ~{merged['files']} "
                      f"mesh ~{merged['mesh_mb']} MB")
            if L["unknown"]:
                print(f"{'':36s} unresolved: {', '.join(sorted(L['unknown'])[:4])}", file=sys.stderr)
        report[run] = out
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
