"""Cut clean (unweathered) v4 strips into chunks for codex img2img.

    python image-gen/prep_ai.py --pkg <v4 pkg with _bake_cache> --pages europe_res_a

Per page, writes ``image-gen/work/<page>/in/<group>_cN.png`` plus ``jobs.json``
(the exact crop of every chunk, for compose_ai.py) and one prompt per group
(``prompt_<group>.md``) for ``codex exec``.

Groups:
  * facade ``sN``: the scheme's upper strip stacked on its ground strip
    (512 px = 6.4 m), cut along u into 4 chunks of 1024 px core plus 128 px
    of overlap each side (wrapping), so neighbours can be stitched with a
    min-error cut.
  * pitched roofs and 512 px flat roofs: same chunking, one strip each.
  * short strips (plain walls, porch, 256 px flat roofs): their 4 chunks
    are stacked vertically with black separator bars into one image, so a
    single generation covers the whole strip in one consistent material.

The trim strip keeps its Cycles bake.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.dirname(HERE)
if V4 not in sys.path:
    sys.path.insert(0, V4)

from bake_pages import page_specs, strip_jobs, wall_kind_of  # noqa: E402
from layout import PAGE_PX  # noqa: E402
from regions import FLAT_ROOF_KINDS, REGIONS  # noqa: E402
import weather  # noqa: E402

CORE = 1024
OVL = 128
N_CHUNKS = PAGE_PX // CORE
SEP = 16            # black separator between stacked short strips
WORK = os.path.join(HERE, "work")

ROOF_WORDS = {
    "clay_interlock": "interlocking red clay roof tiles",
    "concrete_tile": "grey-brown concrete roof tiles",
    "slate": "natural slate roof",
    "gravel": "flat roof covered in grey gravel ballast",
    "membrane": "flat roof with a bitumen / PVC membrane",
}
WALL_WORDS = {
    "render": "painted render (stucco) wall",
    "brick": "brick wall",
    "siding": "siding wall",
    "tiles": "tiled wall",
    "panels": "panel wall",
    "metal": "metal-clad wall",
}


def clean_strip(job, row, region):
    """Albedo of one strip with wear switched off (glass, occlusion kept)."""
    data = np.load(job["out"])
    ppm = row["px"] / row["world_h"]
    rkind = "roof" if row["kind"] == "roof" else "facade"
    a, _n = weather.process(
        data, region=region, kind=rkind,
        level="ground" if job["name"].endswith("_ground") else "upper",
        px_per_m=ppm, seed=int(job["key"][:6], 16), wear_scale=0.0,
        wall_kind=wall_kind_of(job),
        roof_flat=(job.get("roof") or {}).get("kind") in FLAT_ROOF_KINDS)
    return a


def cls_strip(job):
    """Material class id per pixel (strips.CLS_*), uint8."""
    return np.rint(np.load(job["out"])["cls"].astype(np.float32) * 16.0
                   ).astype(np.uint8)


def wrap_cols(img, x0, x1):
    idx = np.arange(x0, x1) % img.shape[1]
    return img[:, idx]


def chunks_of(strip):
    """[(x0, crop)] with x0 the u of the crop's first column (may be <0)."""
    out = []
    for i in range(N_CHUNKS):
        x0 = i * CORE - OVL
        out.append((x0, wrap_cols(strip, x0, x0 + CORE + 2 * OVL)))
    return out


def prep_page(page, kind, kit, region, cache, cls_only=False):
    jobs = strip_jobs(page, kind, kit, region)
    by = {}
    for name, row, job in jobs:
        job["out"] = os.path.join(cache, f"{job['key']}.npz")
        by[name] = (row, job)
    wdir = os.path.join(WORK, page)
    os.makedirs(os.path.join(wdir, "in"), exist_ok=True)
    os.makedirs(os.path.join(wdir, "out"), exist_ok=True)
    groups = []

    def save(name, img):
        path = os.path.join(wdir, "in", name + ".png")
        if not cls_only:
            Image.fromarray(img).save(path)

    def save_cls(name, cls):
        Image.fromarray(cls).save(os.path.join(wdir, "in", name + "_cls.png"))

    # facades: upper over ground
    schemes = sorted({n.split("_")[0] for n in by if n.endswith("_upper")})
    for s in schemes:
        up_row, up_job = by[f"{s}_upper"]
        gr_row, gr_job = by[f"{s}_ground"]
        up = clean_strip(up_job, up_row, region)
        gr = clean_strip(gr_job, gr_row, region)
        stack = np.concatenate([up, gr], 0)
        cstack = np.concatenate([cls_strip(up_job), cls_strip(gr_job)], 0)
        wk = wall_kind_of(up_job)
        g = {"group": s, "type": "facade", "wall_kind": wk,
             "rows": [[f"{s}_upper", 0, up_row["px"]],
                      [f"{s}_ground", up_row["px"], gr_row["px"]]],
             "chunks": []}
        for i, (x0, crop) in enumerate(chunks_of(stack)):
            nm = f"{s}_c{i}"
            save(nm, crop)
            save_cls(nm, wrap_cols(cstack, x0, x0 + CORE + 2 * OVL))
            g["chunks"].append({"name": nm, "x0": x0, "y0": 0})
        groups.append(g)

    # roofs
    for name, (row, job) in by.items():
        if row["kind"] != "roof":
            continue
        strip = clean_strip(job, row, region)
        rk = job["roof"]["kind"]
        if row["px"] >= 512:
            g = {"group": name, "type": "roof", "roof_kind": rk,
                 "px_per_m": row["px"] / row["world_h"],
                 "rows": [[name, 0, row["px"]]], "chunks": []}
            cls = cls_strip(job)
            for i, (x0, crop) in enumerate(chunks_of(strip)):
                nm = f"{name}_c{i}"
                save(nm, crop)
                save_cls(nm, wrap_cols(cls, x0, x0 + CORE + 2 * OVL))
                g["chunks"].append({"name": nm, "x0": x0, "y0": 0})
            groups.append(g)
        else:
            groups.append(_stacked(name, "roof", [(name, row, strip)], save,
                                   roof_kind=rk,
                                   px_per_m=row["px"] / row["world_h"]))

    # plain walls + porch: one stacked image per strip
    for name, (row, job) in by.items():
        if row["kind"] != "plain":
            continue
        strip = clean_strip(job, row, region)
        groups.append(_stacked(name, "plain", [(name, row, strip)], save,
                               wall_kind=wall_kind_of(job)))

    if cls_only:
        return groups
    with open(os.path.join(wdir, "jobs.json"), "w", encoding="utf-8") as fh:
        json.dump({"page": page, "kind": kind, "region": region,
                   "core": CORE, "ovl": OVL, "sep": SEP, "groups": groups},
                  fh, indent=1)
    for g in groups:
        write_prompt(wdir, page, region, g)
    return groups


def _stacked(name, typ, strips, save, **extra):
    (_n, row, strip), = strips
    h = row["px"]
    parts = []
    chunks = []
    y = 0
    for i, (x0, crop) in enumerate(chunks_of(strip)):
        if i:
            parts.append(np.zeros((SEP, crop.shape[1], 3), np.uint8))
            y += SEP
        parts.append(crop)
        chunks.append({"x0": x0, "y0": y})
        y += h
    img = np.concatenate(parts, 0)
    save(f"{name}_stack", img)
    g = {"group": name, "type": typ, "stacked": True,
         "rows": [[name, 0, h]],
         "chunks": [dict(c, name=f"{name}_stack") for c in chunks]}
    g.update(extra)
    return g


REGION_WORDS = {
    "europe": "central/northern European",
    "north_america": "North American suburban",
    "mediterranean": "Mediterranean",
    "asia": "East Asian (Chinese / Taiwanese)",
    "africa": "sub-Saharan African",
    "south_america": "South American",
    "australia_oceania": "Australian",
    "generic": "generic modern",
}

COMMON = """\
# Image re-rendering ONLY - no coding

Your only job: re-render each input image below as a PHOTOGRAPH with your
image-generation tool, and save each result under ./out/ with the SAME
filename as its input. Do NOT write, edit or run any Python or other code
and do not touch any other file. Shell use is allowed solely to copy each
generated image from your image cache into ./out/<name>.png.
Skip any output file that already exists.

The inputs are Blender renders of texture strips for a flight simulator's
{region} buildings. They are geometrically exact but look synthetic.
Your re-render must look like a real photo of the same surface while
keeping the EXACT layout, because the image is mapped onto 3-D models by
pixel coordinates:
- same image size and aspect ({w}x{h}; generate at {gw}x{gh} if your tool
  allows, otherwise the closest size with the SAME aspect ratio),
- every window, door, frame, sill, shutter, vent, tile course, panel and
  rooftop object stays at the same position and size, same count,
- same colour family as the input.
Only the surface realism changes: real material grain, real glass with
reflections of an overcast sky and curtains/rooms behind it, real frames
with depth, and REALISTIC, MODERATE weathering that suits the material
(soft rain streaks under sills and ledges, dust and splash-back near the
ground, slight fading, some grime in corners; nothing heavy or dramatic).

The input's faint rectangular tone blocks and its perfectly regular
per-brick / per-tile colour speckle are rendering artefacts: do NOT
reproduce them as patches; paint the natural, continuous variation a real
surface has instead.

Small shifts, slight size changes or a slightly different aspect ratio are
corrected automatically afterwards, so NEVER give up on a file: if an
attempt is not pixel-perfect, save your best attempt anyway.

Style for ALL images: photorealistic, strictly orthographic (camera
perpendicular to the surface), flat overcast daylight, no directional
shadows, no perspective, no vignetting, no text, no watermarks, no people,
no plants in front of the surface, material fills the frame edge to edge.
"""


def write_prompt(wdir, page, region, g):
    first = g["chunks"][0]["name"]
    names = sorted({c["name"] for c in g["chunks"]})
    im = Image.open(os.path.join(wdir, "in", names[0] + ".png"))
    w, h = im.size
    gw, gh = w * 2, h * 2
    while gw * gh > 8_000_000:
        gw, gh = gw * 3 // 4, gh * 3 // 4
    gw, gh = gw // 16 * 16, gh // 16 * 16
    txt = COMMON.format(region=REGION_WORDS.get(region, region), w=w, h=h,
                        gw=gw, gh=gh)
    if g["type"] == "facade":
        wk = WALL_WORDS.get(g["wall_kind"], "wall")
        txt += f"""
## What the inputs show

{len(names)} adjacent pieces (left to right: {', '.join(names)}) of ONE
continuous two-storey facade band, a {wk}. The TOP half is an upper
floor, the BOTTOM half is the ground floor (its bottom edge is ground
level). Each piece is 16 m wide by 6.4 m tall. Neighbouring pieces overlap
by 1.6 m at each side, and the last piece wraps around into the first, so
the overlapping parts must look the same in both.

The upper floor is repeated on top of itself on tall buildings, so the
wall must continue straight through the top edge and the middle line: no
cornice, cap, shadow band or colour change at the top edge or at the
floor line.

## Consistency (important)

All pieces are ONE photographed wall. Generate {first} first. Then, for
every other piece, give your tool BOTH that piece's input and your {first}
result as references, and keep the wall material, colour, grain scale,
window frame colour, glass look and weathering level identical to {first}.
"""
    elif g["type"] == "roof" and not g.get("stacked"):
        rw = ROOF_WORDS.get(g["roof_kind"], "roof")
        m_w = (CORE + 2 * OVL) / g["px_per_m"]
        m_h = h / g["px_per_m"]
        txt += f"""
## What the inputs show

{len(names)} adjacent pieces (left to right: {', '.join(names)}) of ONE
continuous roof strip seen from DIRECTLY ABOVE: {rw}. Each piece is
{m_w:.0f} m wide by {m_h:.0f} m tall; the roof slope runs top to bottom (the
eave is the bottom edge). Neighbouring pieces overlap by
{OVL / g['px_per_m']:.0f} m at each side and the last wraps into the first.
Keep the exact number and spacing of tile courses and of tiles per course
- count them in the input. It is a ROOF seen from above, never a wall and
never brick. Keep any baked rooftop items (solar panels, skylights, vents,
chimney bases, hatches) exactly where they are, photographic.
Weathering suitable for roofs: dirt collecting toward the eave, some
lichen and moss, faded or replaced pieces - moderate.

## Consistency (important)

All pieces are ONE roof. Generate {first} first. Then, for every other
piece, give your tool BOTH that piece's input and your {first} result as
references, and keep material, colour, tile scale and weathering
identical.
"""
    else:
        what = (ROOF_WORDS.get(g.get("roof_kind"), "roof") + ", seen from "
                "DIRECTLY ABOVE" if g["type"] == "roof" else
                WALL_WORDS.get(g.get("wall_kind"), "wall") + " without "
                "windows (a blank side wall, with small vents), seen "
                "straight on")
        txt += f"""
## What the input shows

ONE image, {names[0]}.png, made of {len(g['chunks'])} horizontal bands
separated by {len(g['chunks']) - 1} thin pure-black bars. Every band is
a piece of the same surface: {what}. Keep ALL {len(g['chunks']) - 1}
black bars, pure black, full width, at the same heights; they are cut
away afterwards, so the output must again have exactly
{len(g['chunks'])} bands. The bands must all
look like the same material, colour and weathering level. The left and
right edges of neighbouring bands continue into each other.
"""
    txt += f"""
## Files

Inputs (attached, and on disk in ./in/): {', '.join(n + '.png' for n in names)}
Outputs: ./out/<same name>.png

When every output exists, list ./out/ and stop.
"""
    with open(os.path.join(wdir, f"prompt_{g['group']}.md"), "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write(txt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--pages", nargs="+", required=True)
    ap.add_argument("--cls-only", action="store_true",
                    help="only (re)write the class maps next to the inputs")
    args = ap.parse_args(argv)
    cache = os.path.join(args.pkg, "_bake_cache")
    specs = {p[0]: p for p in page_specs(list(REGIONS))}
    for page in args.pages:
        _p, kind, kit, region, _scale = specs[page]
        groups = prep_page(page, kind, kit, region, cache, args.cls_only)
        print(page, len(groups), "groups",
              sum(len({c['name'] for c in g['chunks']}) for g in groups),
              "images")


if __name__ == "__main__":
    main()
