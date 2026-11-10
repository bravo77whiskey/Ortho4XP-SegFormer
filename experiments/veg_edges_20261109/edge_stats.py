"""Polygons, vertices, area and estimated plants per review run sample, forests vs edge thickets.

  python edge_stats.py RUN [RUN ...] [--samples key ...]

Estimated plants = area x density/255 / SPACING^2 x layers of the .for (the
forest copy pack of the review runs, else X-Plane's own forests).
"""

import argparse
import json
import os
import re
import sys

DATA = os.environ.get("O4_REVIEW_DATA", r"G:\Ortho4XP_Review")
XP = r"D:\SteamLibrary\steamapps\common\X-Plane 12\Resources\default scenery\1200 forests"
PACK = os.path.join(DATA, "_tmp", "O4XP_Forest_Heights")
MANTLE = re.compile(r"^o4xp/forests/m[0-9]")


def library(lib_dir):
    out = {}
    path = os.path.join(lib_dir, "library.txt")
    if os.path.isfile(path):
        for line in open(path, encoding="utf-8", errors="ignore"):
            p = line.split()
            if len(p) == 4 and p[0] == "EXPORT_SEASON" and p[1] == "sum":
                out[p[2]] = os.path.join(lib_dir, p[3])
            elif len(p) == 3 and p[0] == "EXPORT":
                out.setdefault(p[1], os.path.join(lib_dir, p[2]))
    return out


LIBS = library(XP)
LIBS.update(library(PACK))
_GRID = {}


def grid(path):
    """(spacing x, spacing z, layers) of a forest path, or None."""
    if path not in _GRID:
        f = LIBS.get(path)
        res = None
        if f and os.path.isfile(f):
            sx = sz = None
            layers = set()
            for line in open(f, encoding="utf-8", errors="ignore"):
                p = line.split()
                if p[:1] == ["SPACING"]:
                    sx, sz = float(p[1]), float(p[2])
                elif p[:1] == ["GROUP"]:
                    layers.add(p[1])
            if sx:
                res = (sx, sz, max(1, len(layers)))
        _GRID[path] = res
    return _GRID[path]


def area_px(ring):
    a = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
        a += x0 * y1 - x1 * y0
    return abs(a) / 2


def load(run, key):
    path = os.path.join(DATA, "runs", run, "BI", key + ".js")
    text = open(path, encoding="utf-8").read()
    return json.loads(text[text.index("] = ") + 4:].rstrip().rstrip(";"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--samples", nargs="*")
    args = ap.parse_args()
    meta = json.load(open(os.path.join(DATA, "sets", "hotspots", "BI", "samples.json")))
    mpp_of = {}
    for s in meta:
        lat_s, lon_w, lat_n, lon_e = s["bbox"]
        import math
        mpp_of[s["key"]] = (lon_e - lon_w) * math.cos(math.radians((lat_s + lat_n) / 2)) * 111320 / 4096
    for run in args.runs:
        keys = args.samples or sorted(k[:-3] for k in os.listdir(os.path.join(DATA, "runs", run, "BI"))
                                      if k.endswith(".js"))
        for key in keys:
            try:
                d = load(run, key)
            except (OSError, ValueError):
                continue
            mpp = mpp_of.get(key, 1.0)
            stats = {"forest": [0, 0, 0.0, 0.0], "edge": [0, 0, 0.0, 0.0], "single": [0, 0, 0.0, 0.0]}
            unknown = set()
            for v in d["veg"]:
                kind = "single" if v.get("pt") else "edge" if MANTLE.match(v["a"]) else "forest"
                st = stats[kind]
                st[0] += 1
                st[1] += sum(len(r) for r in v["r"])
                if kind == "single":
                    st[3] += v["pt"]
                    continue
                a = area_px(v["r"][0]) - sum(area_px(h) for h in v["r"][1:])
                a_m2 = a * mpp * mpp
                st[2] += a_m2
                g = grid(v["a"])
                if g is None:
                    unknown.add(v["a"])
                    continue
                dens = (v["d"] % 256) / 255.0
                st[3] += a_m2 * dens / (g[0] * g[1]) * g[2]
            line = "  ".join(f"{k}: {st[0]} polys {st[1]} pts {st[2] / 1e4:.1f} ha ~{st[3]:.0f} plants"
                             for k, st in stats.items())
            print(f"{run:22s} {key:9s} {line}")
            if unknown:
                print("   unknown spacing:", ", ".join(sorted(unknown)[:5]), file=sys.stderr)


if __name__ == "__main__":
    main()
