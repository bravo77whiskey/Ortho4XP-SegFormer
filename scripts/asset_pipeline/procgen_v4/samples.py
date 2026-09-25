"""Write sample OBJ8s and Blender review scenes.

    python samples.py --pkg <pkg> --out <render_dir> [--regions ...]
                      [--render] [--blender exe]

Per region: a street scene (houses both sides of a road, apartments and
shops at the end, a warehouse behind), a close pass over two houses, and
cross-region comparison sheets.  Every building is a real exported OBJ8
from buildings.py, rendered by render_preview.py.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import buildings  # noqa: E402
from meshkit import write_obj8  # noqa: E402

DEFAULT_BLENDER = r"F:\SteamLibrary\steamapps\common\Blender\blender.exe"

GROUND = {
    "europe": "#5d6b44", "north_america": "#61703f", "mediterranean":
    "#8a8058", "asia": "#5b6a45", "africa": "#8a6a48", "south_america":
    "#667044", "australia_oceania": "#7d7a52", "generic": "#626c48",
}
BUCKET = {"gable": "residential", "hip": "residential", "lshape":
          "residential", "rowhouse": "residential", "flatres": "residential",
          "aptslab": "apartments", "aptblock": "apartments",
          "shophouse": "commercial", "flatcom": "commercial",
          "warehouse": "industrial", "bigbox": "industrial"}


def write_building(pkg, region, archetype, L, W, F, seed, force=None):
    page, bands = buildings.build(archetype, L, W, F, seed, region, force)
    dists, _old = buildings.band_distances(L, W, F, bands)
    rel = os.path.join(region, BUCKET[archetype],
                       f"{archetype}_{L:g}x{W:g}x{F}_s{seed}.obj")
    path = os.path.join(pkg, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lod = [(bands[i], dists[i], dists[i + 1]) for i in range(4)]
    write_obj8(path, lod, f"../../textures/o4v4_{page}.png",
               f"../../textures/o4v4_{page}_nml.png")
    return path, [b.tri_count() for b in bands]


def street(pkg, region, seed=1):
    """Houses both sides of a road; apartments/shops at the far end."""
    rng = random.Random(seed)
    objs = []
    stats = []
    x = -60.0
    res = ["gable", "hip", "gable", "lshape", "hip", "gable"]
    if region in ("mediterranean", "africa", "south_america", "asia"):
        res += ["flatres", "flatres"]
    while x < 40:
        for side in (-1, 1):
            arch = rng.choice(res)
            L = round(rng.uniform(10, 15), 1)
            W = round(rng.uniform(8.5, 11.5), 1)
            F = rng.choice([1, 2, 2]) if arch != "flatres" else \
                rng.choice([1, 2, 2, 3])
            if arch == "lshape":
                L, W = L + 3, W + 3
            p, t = write_building(pkg, region, arch, L, W, F,
                                  rng.randrange(1 << 30))
            y = side * (6.5 + W / 2 + rng.uniform(4, 7))
            objs.append({"obj": p, "loc": [x + L / 2, y],
                         "rot": 0 if side > 0 else 180, "band": 0})
            stats.append((arch, t))
        x += 15 + rng.uniform(2, 5)
    # an urban end: apartments + shops on the far side of a cross street
    urban = [("aptslab", 32, 13, rng.choice([4, 5, 6])),
             ("flatcom", 24, 16, 3),
             ("aptblock", 24, 20, rng.choice([5, 6, 7]))]
    if region == "asia":
        urban[1] = ("shophouse", 24, 13, 3)
    if region == "europe":
        urban.append(("rowhouse", 30, 10, 3))
    ux = 72
    for i, (arch, L, W, F) in enumerate(urban):
        p, t = write_building(pkg, region, arch, L, W, F,
                              rng.randrange(1 << 30))
        side = 1 if i % 2 == 0 else -1
        y = side * (6.5 + W / 2 + 3)
        objs.append({"obj": p, "loc": [ux + L / 2, y],
                     "rot": 0 if side > 0 else 180, "band": 0})
        stats.append((arch, t))
        if i % 2 == 1:
            ux += max(L, 30) + 10
    p, t = write_building(pkg, region, "warehouse", 60, 30, 3,
                          rng.randrange(1 << 30))
    objs.append({"obj": p, "loc": [-20, 75], "rot": 0, "band": 0})
    stats.append(("warehouse", t))
    roads = [[-90, 0, 190, 0, 7.0], [62, -120, 62, 120, 7.0]]
    return objs, roads, stats


HERO = {
    "europe": [("gable", 12.5, 9.5, 2, {"dormer_p": 1, "chimney_p": 1}),
               ("rowhouse", 26, 9.5, 3, {}),
               ("aptblock", 26, 14, 5, {"apt_pitched_p": 1})],
    "north_america": [("gable", 13, 10.5, 2, {"porch_p": 1,
                                               "veranda_p": 0}),
                      ("hip", 14, 11, 1, {"porch_p": 0}),
                      ("flatcom", 26, 16, 2, {"canopy_p": 1})],
    "mediterranean": [("hip", 12, 10, 2, {"veranda_p": 0, "porch_p": 0}),
                      ("flatres", 11, 9, 2, {}),
                      ("aptslab", 28, 13, 6, {"apt_pitched_p": 0})],
    "asia": [("gable", 11, 8.5, 2, {}), ("shophouse", 20, 12, 3, {}),
             ("aptslab", 30, 13, 8, {})],
    "africa": [("hip", 13, 10.5, 1, {"veranda_p": 1, "porch_p": 0}),
               ("flatres", 11, 9, 2, {}),
               ("flatcom", 22, 14, 2, {"canopy_p": 1})],
    "south_america": [("flatres", 11, 9, 2, {}),
                      ("gable", 12, 9, 1, {"veranda_p": 0}),
                      ("aptslab", 26, 13, 9, {"apt_pitched_p": 0})],
    "australia_oceania": [("hip", 15, 12, 1, {"veranda_p": 1,
                                               "porch_p": 0}),
                          ("hip", 14, 11, 1, {"veranda_p": 0}),
                          ("aptblock", 26, 14, 3, {"apt_pitched_p": 1})],
    "generic": [("gable", 12, 9, 2, {}), ("hip", 13, 10, 2, {}),
                ("aptslab", 28, 13, 5, {})],
}


def hero_scene(pkg, region, out_dir, seed=3, band=0, suffix="hero",
               res=(1280, 720), cam=None):
    objs = []
    x = 0.0
    rng = random.Random(seed)
    for arch, L, W, F, force in HERO[region]:
        p, _t = write_building(pkg, region, arch, L, W, F,
                               rng.randrange(1 << 30), force)
        objs.append({"obj": p, "loc": [x + L / 2, W / 2 + 4], "rot": 0,
                     "band": band})
        x += L + 6
    cx = x / 2
    cam = cam or {"loc": [cx - 32, -48, 34], "look_at": [cx, 12, 5],
                  "lens": 32}
    return {"objects": objs, "roads": [[-40, -4, x + 40, -4, 7.0]],
            "ground": GROUND.get(region, "#626c48"),
            "sun": {"elev": 38, "azim": 145, "strength": 4.2},
            "samples": 128, "res": list(res), "camera": cam,
            "out": os.path.join(out_dir, f"{region}_{suffix}.png")}


def lod_scenes(pkg, out_dir, region="europe"):
    """One cluster at bands 1-4 from the same camera (close-up)."""
    specs = []
    for band in range(4):
        specs.append(hero_scene(pkg, region, out_dir, band=band,
                                suffix=f"lod{band + 1}", res=(960, 540)))
    return specs


def compose(paths, labels, out, cols, cell=(960, 540)):
    from PIL import Image, ImageDraw, ImageFont
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell[0], rows * (cell[1] + 44)),
                      (24, 26, 28))
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 26)
    except OSError:
        font = ImageFont.load_default()
    d = ImageDraw.Draw(sheet)
    for i, (p, lab) in enumerate(zip(paths, labels)):
        if not os.path.isfile(p):
            continue
        im = Image.open(p).convert("RGB").resize(cell, Image.LANCZOS)
        x = (i % cols) * cell[0]
        y = (i // cols) * (cell[1] + 44)
        sheet.paste(im, (x, y))
        d.text((x + 14, y + cell[1] + 6), lab, fill=(230, 232, 235),
               font=font)
    sheet.save(out)


def scenes_for(pkg, region, out_dir):
    objs, roads, stats = street(pkg, region)
    sun = {"elev": 36, "azim": 150, "strength": 4.2}
    ground = GROUND.get(region, "#626c48")
    base = {"objects": objs, "roads": roads, "ground": ground, "sun": sun,
            "samples": 128}
    return [
        dict(base, out=os.path.join(out_dir, f"{region}_street.png"),
             res=[1920, 1080],
             camera={"loc": [-150, -150, 115], "look_at": [20, 5, 0],
                     "lens": 45}),
        dict(base, out=os.path.join(out_dir, f"{region}_close.png"),
             res=[1920, 1080],
             camera={"loc": [-68, -34, 12], "look_at": [-40, 12, 5],
                     "lens": 35}),
        dict(base, out=os.path.join(out_dir, f"{region}_urban.png"),
             res=[1920, 1080],
             camera={"loc": [40, -95, 60], "look_at": [95, 10, 8],
                     "lens": 40}),
    ], stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--regions", nargs="*", default=["europe"])
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--only", nargs="*", default=None,
                    help="scene suffixes: street close urban")
    ap.add_argument("--blender", default=DEFAULT_BLENDER)
    ap.add_argument("--sheets", action="store_true",
                    help="regional lineup + LOD sheets instead of streets")
    args = ap.parse_args(argv)
    args.pkg = os.path.abspath(args.pkg)
    args.out = os.path.abspath(args.out)
    os.makedirs(args.out, exist_ok=True)
    all_scenes = []
    if args.sheets:
        for region in args.regions:
            all_scenes.append(hero_scene(args.pkg, region, args.out))
        all_scenes += lod_scenes(args.pkg, args.out)
        path = os.path.join(args.out, "sheets.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(all_scenes, fh)
        if args.render:
            subprocess.run([args.blender, "--background",
                            "--factory-startup", "--python-exit-code", "1",
                            "--python", os.path.join(HERE,
                                                     "render_preview.py"),
                            "--", path], check=True)
        names = {"europe": "Europe", "north_america": "North America",
                 "mediterranean": "Mediterranean", "asia": "Asia",
                 "africa": "Africa", "south_america": "South America",
                 "australia_oceania": "Australia / Oceania",
                 "generic": "Generic"}
        compose([os.path.join(args.out, f"{r}_hero.png")
                 for r in args.regions],
                [names.get(r, r) for r in args.regions],
                os.path.join(args.out, "sheet_regions.png"), cols=2)
        compose([os.path.join(args.out, f"europe_lod{b}.png")
                 for b in range(1, 5)],
                ["LOD 1  full (0 - 0.6 x old band-1 range)",
                 "LOD 2  shell (per-floor facades)",
                 "LOD 3  box + roof silhouette", "LOD 4  roof quad"],
                os.path.join(args.out, "sheet_lod.png"), cols=2)
        return
    for region in args.regions:
        scenes, stats = scenes_for(args.pkg, region, args.out)
        if args.only:
            scenes = [s for s in scenes if any(
                s["out"].endswith(f"_{o}.png") for o in args.only)]
        all_scenes += scenes
        print(region, stats)
    path = os.path.join(args.out, "scenes.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(all_scenes, fh)
    if args.render:
        subprocess.run([args.blender, "--background", "--factory-startup",
                        "--python-exit-code", "1", "--python",
                        os.path.join(HERE, "render_preview.py"), "--", path],
                       check=True)


if __name__ == "__main__":
    main()
