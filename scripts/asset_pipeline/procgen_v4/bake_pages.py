"""Bake and assemble every atlas page of the v4 library.

    python bake_pages.py --out <pkg> [--cache <dir>] [--regions europe ...]
                         [--blender <exe>] [--samples 64] [--jobs 2]

Per page: one Blender job per strip (cached by a hash of the strip spec, so
re-running after a kit edit only re-bakes what changed), then weather.py
applies the region's wear and the strips are stacked into
``textures/o4v4_<page>.png`` (albedo, sRGB) and ``..._nml.png`` (normal
RG, base reflectance B, gloss A).  The industrial page is baked once and
post-processed three times (clean / mild / worn): wear is a post step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from kits import BASE_MATS, materials_for, roof_materials  # noqa: E402
from layout import PAGE_PX, TRIM_BANDS, page_layout  # noqa: E402
from regions import (FLAT_ROOF_KINDS, IND_PAGE, IND_WEAR, REGIONS,  # noqa: E402
                     ROOF_OBJECTS)
import weather  # noqa: E402

DEFAULT_BLENDER = r"F:\SteamLibrary\steamapps\common\Blender\blender.exe"
# Bump when strip recipes change in ways the job spec cannot see (e.g. the
# bay-motif tables of v2), so cached bakes are not reused against new UVs.
BAKE_VERSION = 2


def page_specs(regions):
    """[(page_name, layout_kind, kit_dict, region_for_wear, wear_scale)]."""
    pages = []
    for region in regions:
        kit = REGIONS[region]
        for i, res in enumerate(kit["res"]):
            pages.append((f"{region}_res_{'ab'[i]}", "res", res, region,
                          1.0))
        pages.append((f"{region}_apt", "apt", kit["apt"], region, 1.0))
    for grade, (wear_region, scale) in IND_WEAR.items():
        pages.append((f"ind_{grade}", "ind", IND_PAGE, wear_region, scale))
    return pages


def _scheme_key(strip):
    return strip.split("_")[0]


def strip_jobs(page, kind, kit, region=None):
    lay = page_layout(kind)
    objects = ROOF_OBJECTS.get("ind" if kind == "ind" else region, {})
    jobs = []
    for name, row in lay.items():
        W, H = row["world_w"], row["world_h"]
        px = [PAGE_PX, row["px"]]
        seed = int(hashlib.sha1(f"{page}/{name}".encode()).hexdigest()[:6],
                   16)
        if row["kind"] == "roof":
            spec = dict(kit[name])
            slot = "flat" if spec["kind"] in FLAT_ROOF_KINDS else "pitched"
            if objects.get(slot):
                spec["objects"] = objects[slot]
            job = {"kind": "roof", "roof": spec, "mats": roof_materials(spec)}
        elif row["kind"] == "trim":
            mats = dict(BASE_MATS)
            for band, m in kit["trim"].items():
                mats[f"trim_{band}"] = m
            job = {"kind": "band", "band": list(TRIM_BANDS), "mats": mats}
        elif row["kind"] == "plain":
            sch = kit[_scheme_key(name) if name != "porch" else "porch"]
            job = {"kind": "plain", "scheme": sch,
                   "mats": materials_for(sch), "vents": name != "porch"}
        else:
            sch = kit[_scheme_key(name)]
            level = "ground" if name.endswith("_ground") else "upper"
            job = {"kind": "facade", "level": level, "scheme": sch,
                   "mats": materials_for(sch)}
        job.update({"W": W, "H": H, "px": px, "seed": seed,
                    "bake_version": BAKE_VERSION})
        key = hashlib.sha1(json.dumps(job, sort_keys=True,
                                      default=str).encode()).hexdigest()[:16]
        job["name"] = f"{page}/{name}"
        job["key"] = key
        jobs.append((name, row, job))
    return jobs


def run_blender(jobs, cache, blender, samples, n_proc):
    todo = [j for j in jobs if not os.path.isfile(j["out"])]
    if not todo:
        return
    chunks = [todo[i::n_proc] for i in range(n_proc)]
    procs = []
    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
        path = os.path.join(cache, f"_jobs_{os.getpid()}_{i}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"samples": samples, "skip_existing": True,
                       "jobs": chunk}, fh)
        log = open(path + ".log", "w", encoding="utf-8")
        procs.append((subprocess.Popen(
            [blender, "--background", "--factory-startup",
             "--python-exit-code", "1", "--python",
             os.path.join(HERE, "bake_blender.py"), "--", path],
            stdout=log, stderr=subprocess.STDOUT), path, log))
    failed = False
    for proc, path, log in procs:
        rc = proc.wait()
        log.close()
        if rc:
            failed = True
            print(f"blender failed ({rc}); see {path}.log")
    if failed:
        raise SystemExit(1)


def wall_kind_of(job):
    """Wall material family, so wear picks plausible layers (bricks
    effloresce, render peels and gets repainted)."""
    sch = job.get("scheme")
    if not sch:
        return "render"
    pat = ((sch.get("mats") or {}).get("wall") or {}).get("pattern") or {}
    return {"brick": "brick", "siding": "siding", "tiles": "tiles",
            "panels": "panels", "boards": "metal"}.get(pat.get("type"),
                                                       "render")


def assemble(page, kind, jobs, region, wear_scale, out_dir):
    albedo = np.zeros((PAGE_PX, PAGE_PX, 3), np.uint8)
    normal = np.zeros((PAGE_PX, PAGE_PX, 4), np.uint8)
    for name, row, job in jobs:
        data = np.load(job["out"])
        ppm = row["px"] / row["world_h"]
        level = "ground" if name.endswith("_ground") else "upper"
        rkind = "roof" if row["kind"] == "roof" else "facade"
        a, n = weather.process(data, region=region, kind=rkind, level=level,
                               px_per_m=ppm, seed=int(job["key"][:6], 16),
                               wear_scale=wear_scale,
                               wall_kind=wall_kind_of(job),
                               roof_flat=(job.get("roof") or {}).get(
                                   "kind") in FLAT_ROOF_KINDS)
        r0 = row["row0"]
        albedo[r0:r0 + row["px"]] = a
        normal[r0:r0 + row["px"]] = n
    tex = os.path.join(out_dir, "textures")
    os.makedirs(tex, exist_ok=True)
    # truecolor only: X-Plane renders palette PNGs with phantom holes
    Image.fromarray(albedo, "RGB").save(
        os.path.join(tex, f"o4v4_{page}.png"), compress_level=6)
    Image.fromarray(normal, "RGBA").save(
        os.path.join(tex, f"o4v4_{page}_nml.png"), compress_level=6)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--regions", nargs="*", default=list(REGIONS))
    ap.add_argument("--pages", nargs="*", default=None,
                    help="only these page names")
    ap.add_argument("--blender", default=DEFAULT_BLENDER)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--wear-scale", type=float, default=None,
                    help="override every page's wear (0 = clean)")
    args = ap.parse_args(argv)
    cache = os.path.abspath(args.cache or os.path.join(args.out, "_bake_cache"))
    os.makedirs(cache, exist_ok=True)
    pages = page_specs(args.regions)
    if args.pages:
        pages = [p for p in pages if p[0] in set(args.pages)]
    if args.wear_scale is not None:
        pages = [p[:4] + (args.wear_scale,) for p in pages]
    # bake: industrial pages share one bake
    all_jobs = {}
    for page, kind, kit, region, scale in pages:
        bake_page = "ind" if kind == "ind" else page
        jobs = strip_jobs(bake_page, kind, kit,
                          None if kind == "ind" else region)
        for _name, _row, job in jobs:
            job["out"] = os.path.join(cache, f"{job['key']}.npz")
            all_jobs[job["key"]] = job
        all_jobs.setdefault("_pages", []).append(
            (page, kind, jobs, region, scale))
    page_rows = all_jobs.pop("_pages")
    t0 = time.time()
    run_blender(list(all_jobs.values()), cache, args.blender, args.samples,
                args.jobs)
    print(f"bake: {len(all_jobs)} strips ready in {time.time() - t0:.0f}s")
    from concurrent.futures import ProcessPoolExecutor
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(assemble, page, kind, jobs, region, scale,
                            args.out): page
                for page, kind, jobs, region, scale in page_rows}
        for fut, page in futs.items():
            fut.result()
            print(f"page {page} done ({time.time() - t1:.0f}s)", flush=True)
    manifest = {kind: page_layout(kind) for kind in ("res", "apt", "ind")}
    with open(os.path.join(args.out, "atlas_layout_v4.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)


if __name__ == "__main__":
    main()
