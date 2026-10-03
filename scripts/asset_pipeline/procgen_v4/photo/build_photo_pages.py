"""Bake ProcGen v4 pages with CC0 photoscanned materials.

    python photo/build_photo_pages.py --out <dir> [--pages <page> ...]
        [--base <pkg with pages>] [--cache <bake cache>] [--wear 0]
        [--windows] [--size 512]

Same strips, geometry, UVs and bay tables as the v4 bake; only the wall,
plinth, lintel and roof materials sample real photos (photo_kits.py)
instead of procedural palettes, and photo roofs are one flat sheet (the
photo carries its own tiles).  No forced wear by default on the regional
pages: the photos are real, already-weathered surfaces.  The industrial
pages keep their clean / mild / worn grades.  ``--windows`` pastes the
real window and door cut-outs (window_cutouts.py, window_compose.py).
``--base`` copies another package's pages first.  ``--size`` also writes
box-downsampled copies to ``<out>/textures_<size>`` (the installed
library uses 512 px pages).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.dirname(HERE)
sys.path.insert(0, V4)
sys.path.insert(0, HERE)

import bake_pages  # noqa: E402
import regions  # noqa: E402
from photo_kits import DEFAULT_TEX_DIR, REPO, apply_mapping, mapping_for  # noqa: E402
import window_compose  # noqa: E402


def _kits():
    """{page or "ind": (kit, region, layout kind, setter)}; the setter puts
    a patched kit back where bake_pages reads it."""
    out = {}
    for region, kit in regions.REGIONS.items():
        for i, res in enumerate(kit["res"]):
            def put(k, kit=kit, i=i):
                kit["res"][i] = k
            out[f"{region}_res_{'ab'[i]}"] = (res, region, "res", put)

        def put_apt(k, kit=kit):
            kit["apt"] = k
        out[f"{region}_apt"] = (kit["apt"], region, "apt", put_apt)

    def put_ind(k):
        regions.IND_PAGE = k
        bake_pages.IND_PAGE = k
    out["ind"] = (regions.IND_PAGE, None, "ind", put_ind)
    return out


def _current(page, region, kind):
    if kind == "res":
        return regions.REGIONS[region]["res"]["ab".index(page[-1])]
    return regions.REGIONS[region]["apt"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pages", nargs="*", default=None,
                    help="page names (default: all, incl. ind_*)")
    ap.add_argument("--base", default=None)
    ap.add_argument("--cache", default=os.path.join(
        REPO, "test_output", "procgen_v4", "O4XP_ProcGen_v4", "_bake_cache"))
    ap.add_argument("--wear", type=float, default=0.0)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--windows", action="store_true")
    ap.add_argument("--size", type=int, default=None)
    args = ap.parse_args(argv)
    out = os.path.abspath(args.out)
    tex = os.path.join(out, "textures")
    os.makedirs(tex, exist_ok=True)
    if args.base:
        src = os.path.join(args.base, "textures")
        for name in os.listdir(src):
            shutil.copy2(os.path.join(src, name), os.path.join(tex, name))
    all_pages = [p[0] for p in bake_pages.page_specs(list(regions.REGIONS))]
    pages = args.pages or all_pages
    kits = _kits()
    lib = os.path.join(DEFAULT_TEX_DIR, "windows", "lib")
    pools, types = window_compose.load_library(lib)
    assignments = {}
    for key, (kit, region, kind, put) in kits.items():
        if not any(p == key or (key == "ind" and p.startswith("ind_"))
                   for p in pages):
            continue
        patched = apply_mapping(kit, mapping_for(key, kit))
        if args.windows:
            a = window_compose.assign(key, patched, region, types)
            window_compose.apply_sizes(patched, a, pools, types)
            assignments[key] = a
        put(patched)
    regional = [p for p in pages if not p.startswith("ind_")]
    industrial = [p for p in pages if p.startswith("ind_")]
    common = ["--out", out, "--cache", args.cache, "--jobs", str(args.jobs),
              "--samples", str(args.samples), "--regions",
              *regions.REGIONS]
    if regional:
        bake_pages.main(common + ["--pages", *regional, "--wear-scale",
                                  str(args.wear)])
    if industrial:
        bake_pages.main(common + ["--pages", *industrial])
    if args.windows:
        report = {}
        for page in regional:
            _kit, region, kind, _put = kits[page]
            a = assignments.get(page, {})
            sets = window_compose.sets_for(a, pools)
            if not sets:
                continue
            rows = [(n, r, j) for n, r, j in bake_pages.strip_jobs(
                page, kind, _current(page, region, kind), region)
                if r["kind"] == "facade"]
            n = window_compose.compose_page(
                os.path.join(tex, f"o4v4_{page}.png"),
                os.path.join(tex, f"o4v4_{page}_nml.png"), rows, sets)
            report[page] = {s: {"window": w, "door": d}
                            for s, (w, d) in a.items()}
            print(f"{page}: {n} photo openings", flush=True)
        with open(os.path.join(tex, "window_assignment.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
        shutil.copy2(os.path.join(lib, "SOURCES.json"),
                     os.path.join(tex, "SOURCES_cc0_windows.json"))
    shutil.copy2(os.path.join(DEFAULT_TEX_DIR, "LICENSES.json"),
                 os.path.join(tex, "LICENSES_cc0_photo.json"))
    if args.size:
        small = os.path.join(out, f"textures_{args.size}")
        os.makedirs(small, exist_ok=True)
        for name in os.listdir(tex):
            src = os.path.join(tex, name)
            if name.endswith(".png") and any(
                    name.startswith(f"o4v4_{p}") for p in pages):
                Image.open(src).resize((args.size, args.size),
                                       Image.BOX).save(
                    os.path.join(small, name), compress_level=6)
            elif name.endswith(".json"):
                shutil.copy2(src, os.path.join(small, name))
        print(f"wrote {small}")


if __name__ == "__main__":
    main()
