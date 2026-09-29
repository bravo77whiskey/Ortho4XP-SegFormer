"""Level 2: free facade designs on the v4 bay grid -- sketches and prompts.

    python image-gen/design_prep.py --pkg <v4 pkg with _bake_cache>
                             --pages asia_res_a asia_res_b asia_apt ind

Per facade scheme the model designs two three-storey elevations (ground +
two upper floors, K bays wide) from a written brief and a zone sketch, plus
one blank side wall.  Only the grid is fixed:

* floors are 3.2 m, bays are the scheme's exact bay period;
* every bay holds at most one opening, centred in the bay, sized like the
  zone drawn for its kind (window, small window, door, loggia, shop; a
  garage or loading dock spans two bays), and blank bays hold none;
* the bay kinds are a K-bay window of the bake's own tables, so doors keep
  the positions buildings.py expects (``Facade.bay_start_with``).

Everything else -- materials, colours, window design, surrounds, shutters,
ornament, wear -- is the model's.  design_compose.py conforms the results
back onto the grid and rebuilds the strips.

Roofs (for pages without level-1 roofs, i.e. everything but europe): one
re-render per roof strip of a 2.5:1 crop of its clean bake, tiled across the
strip by design_compose.py.

The industrial page is prepared once as ``ind`` (the bake page all three
wear grades share) and written to ind_clean / ind_mild / ind_worn.

Writes ``image-gen/design_work/<page>/``: ``in/<key>_sketch_{a,b}.png``,
``prompt_<key>.md`` and ``jobs.json`` (run with run_codex.py --work).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.dirname(HERE)
if V4 not in sys.path:
    sys.path.insert(0, V4)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from bake_pages import page_specs, strip_jobs  # noqa: E402
from buildings import strip_seed  # noqa: E402
from layout import bay_period, page_layout  # noqa: E402
from regions import IND_PAGE, REGIONS  # noqa: E402
import strips as strip_recipes  # noqa: E402
from design_briefs import BRIEFS, REGION_LINE, ROOF_WORDS  # noqa: E402

WORK = os.path.join(HERE, "design_work")
K = 6               # bays per elevation (default)
K_BY_KIND = {"ind": 4}          # industrial bays are 5.7 m
FLOOR_H = 3.2
PX_M = 80           # strip resolution (px per metre)
SKETCH_W = 1600     # sketch width in px
SPAN2 = ("garage", "dock")      # kinds that span two bays
CONT = "+"                      # suffix of a span's second bay

# opening zones: (x0, x1) as a fraction of the bay (a two-bay kind reaches
# up to 2.0), (z0, z1) in metres above the floor
ZONES = {
    "win": ((0.12, 0.88), (0.55, 2.85)),
    "small": ((0.30, 0.70), (1.40, 2.60)),
    "door": ((0.20, 0.80), (0.00, 2.75)),
    "loggia": ((0.06, 0.94), (0.00, 2.95)),
    "shop": ((0.05, 0.95), (0.00, 3.05)),
}
LINTEL = 0.2       # lintel / roller-shutter box above a window head (m)
MIN_WALL_ABOVE = 0.7


def scheme_zones(sch, p):
    """Opening boxes for one scheme, sized from its kit (regions.py) so the
    regional window proportions survive: {level: {kind: ((fx0, fx1),
    (z0, z1))}}.  The model is asked to match them, not just stay inside."""
    win = sch["window"]
    w, h, sill = win["w"], win["h"], win.get("sill_z", 0.9)
    # user 2026-09-28: AI windows read too big -> never wider than half a
    # bay (industrial strip windows excepted), and keep >= 0.7 m of wall
    # between lintel and the floor above
    if sch.get("ground") != "industrial":
        w = min(w, 0.5 * p / 1.08)
    h = min(h, FLOOR_H - MIN_WALL_ABOVE - LINTEL - sill)
    pier = sch.get("loggia_pier", 0.35)

    def xz(width, z0, z1, span=1):
        f = min(0.9 * span, width / p)
        c = span / 2.0
        return ((c - f / 2, c + f / 2), (round(z0, 3), round(z1, 3)))

    out = {}
    for level in ("upper", "ground"):
        sl = sill + (sch.get("ground_raise", 0.0) if level == "ground"
                     else 0.0)
        sh = h * 0.55
        dz = sch.get("door_z", 0.15)
        out[level] = {
            "win": xz(w * 1.08, sl - 0.05, sl + h + LINTEL),
            "small": xz(w * 0.6, sl + h - sh - 0.05, sl + h + LINTEL / 2),
            "door": xz(sch.get("door_w", 1.0) * 1.15, 0.0,
                       dz + sch.get("door_h", 2.25) + 0.15),
            "loggia": ((pier / 2 / p, 1 - pier / 2 / p),
                       (0.12, FLOOR_H - 0.42)),
            "shop": ZONES["shop"],
            # the bake's own two-bay openings (strips.facade_strip)
            "garage": xz(min(2 * p - 0.6, 2.6 * 1.5 + 0.3), 0.0, 2.35,
                         span=2),
            "dock": xz(min(2 * p - 0.8, sch.get("dock_w", 3.6)), 0.0,
                       FLOOR_H - 0.35, span=2),
        }
    return out


ZONE_RGB = {"win": (40, 100, 230), "small": (30, 160, 60),
            "door": (220, 40, 40), "loggia": (240, 140, 20),
            "shop": (200, 40, 200), "garage": (130, 80, 30),
            "dock": (20, 170, 170)}
LEGEND = {
    "win": "BLUE box = a normal window, including its lintel or "
           "roller-shutter box",
    "small": "GREEN box = a small window (bathroom / stair window)",
    "door": "RED box = the front entrance door of a house or stair hall "
            "(door reaches the ground)",
    "loggia": "ORANGE box = a recessed loggia / balcony (painted depth, "
              "balustrade, a window or door behind it)",
    "shop": "MAGENTA box = one small shop unit: shop window + shop door, "
            "separated from the next unit by a narrow wall pier on the bay "
            "line",
    "garage": "BROWN box (spans two bays) = a garage door (reaches the "
              "ground)",
    "dock": "CYAN box (spans two bays) = a loading-dock opening with a "
            "sectional or roller door (reaches the ground)",
    "blank": "GREY hatching = plain wall, NO opening in this bay",
}


def expand(table):
    """[(kind, span)] -> one entry per bay; a span's later bays are
    ``kind + CONT``."""
    out = []
    for kind, span in table:
        out.append(kind)
        out += [kind + CONT] * (span - 1)
    return out


def scheme_info(page, kind, kit, key):
    lay = page_layout(kind)
    sch = kit[key]
    motif = strip_recipes.motif_len(sch)
    p, n = bay_period(lay[f"{key}_upper"]["world_w"], sch["bay"], motif)
    bake_page = "ind" if kind == "ind" else page
    tables = {}
    for level in ("ground", "upper"):
        if f"{key}_{level}" not in lay:
            continue
        rng = random.Random(strip_seed(bake_page, f"{key}_{level}"))
        tables[level] = expand(strip_recipes._bay_table(rng, n, level, sch))
    return {"p": p, "n": n, "motif": motif, "tables": tables}


def pick_windows(table, k, count=2):
    """``count`` circular k-bay windows of ``table`` that between them
    cover as many of its kinds as possible (doors first).  A window never
    starts or ends inside a two-bay opening."""
    n = len(table)
    wins = []
    for s in range(n):
        seq = [table[(s + i) % n] for i in range(k)]
        if seq[0].endswith(CONT) or table[(s + k) % n].endswith(CONT):
            continue
        wins.append((s, seq))

    def score(seq, have):
        kinds = {x for x in seq if not x.endswith(CONT)}
        return (len(kinds - have) * 10 + ("door" in kinds) * 5
                + len(kinds) + seq.count("door") * 0.5)

    chosen, have = [], set()
    for _ in range(count):
        pool = [w for w in wins if w[0] not in [c[0] for c in chosen]] \
            or wins
        best = max(pool, key=lambda w: (score(w[1], have), -w[0]))
        chosen.append(best)
        have |= set(best[1])
    # every kind of the table must appear in some sketch, or the rebuild
    # has no source bay for it: swap the last pick for one that adds the
    # missing kind (only happens when the scoring above missed it)
    need = {x for x in table if not x.endswith(CONT)}
    missing = need - {x for _s, seq in chosen for x in seq}
    for kind in sorted(missing):
        keep = [x for _s, seq in chosen[:-1] for x in seq]
        pool = [w for w in wins if kind in w[1]]
        if pool:
            chosen[-1] = max(pool, key=lambda w: (
                len({x for x in w[1] if not x.endswith(CONT)}
                    - set(keep)), -w[0]))
    return chosen


def draw_sketch(path, rows, levels, p, zones):
    """rows: top to bottom, each a list of kinds; levels: their strip level."""
    k = len(rows[0])
    w_m, h_m = k * p, len(rows) * FLOOR_H
    s = SKETCH_W / w_m
    W, H = SKETCH_W, int(round(h_m * s))
    im = Image.new("RGB", (W, H), (232, 232, 232))
    d = ImageDraw.Draw(im)
    for r, kinds in enumerate(rows):
        y_floor = (r + 1) * FLOOR_H * s          # bottom of this floor
        for b, kind in enumerate(kinds):
            x0 = b * p * s
            if kind.endswith(CONT):
                continue
            if kind == "blank":
                cw, ch = int(p * s), int(FLOOR_H * s)
                cell = Image.new("RGB", (cw, ch), (232, 232, 232))
                cd = ImageDraw.Draw(cell)
                for t in range(-ch, cw, 14):
                    cd.line([(t, ch), (t + ch, 0)], fill=(196, 196, 196),
                            width=2)
                im.paste(cell, (int(x0), int(y_floor - ch)))
                continue
            (fx0, fx1), (z0, z1) = zones[levels[r]][kind]
            box = [x0 + fx0 * p * s, y_floor - z1 * s,
                   x0 + fx1 * p * s, y_floor - z0 * s]
            d.rectangle(box, outline=ZONE_RGB[kind], width=4)
        for b in range(k + 1):
            x = b * p * s
            for y in range(int(r * FLOOR_H * s), int(y_floor), 18):
                d.line([(x, y), (x, min(y_floor, y + 9))],
                       fill=(90, 90, 90), width=2)
        d.line([(0, y_floor - 1), (W, y_floor - 1)], fill=(60, 60, 60),
               width=3)
    im.save(path)
    return W, H


def gen_size(w, h):
    """Generation size with the sketch's aspect (multiples of 16)."""
    scale = min(3840 / w, (8_000_000 / (w * h)) ** 0.5, 2560 / max(w, h))
    return int(w * scale) // 16 * 16, int(h * scale) // 16 * 16


PROMPT = """\
# Facade design + image generation ONLY - no coding

Your only job: design and generate {n_img} photographic images with your
image-generation tool and save them under ./out/ with the exact filenames
below. Do NOT write, edit or run any Python or other code and do not touch
any other file. Shell use is allowed solely to copy each generated image
from your image cache into ./out/<name>.png. Skip any output that already
exists. Small misalignments are corrected automatically afterwards, so
NEVER give up on a file: always save your best attempt.

These are facade textures for a flight simulator's buildings on {region}.
You are the architect: you design the facade. Brief:

> {brief}

Make it look like a real photograph of a real, lived-in building of that
type: real materials and colour variation, glass reflecting an overcast
sky with curtains / blinds / rooms behind it, frames and reveals with
depth, sills, and CLEARLY VISIBLE, honest weathering of a building that
has not been repainted for 15-25 years: {wear} -- worn and real, but not
derelict. Vary the details from bay to bay the way a real building does
(curtains, blinds half down, an open window, a plant pot), but keep ONE
coherent design.

## The grid (the only hard constraints)

The sketch (attached, ./in/{sketch_a}.png) is a DIAGRAM of the facade: it
is {w_m:.1f} m wide and {h_m:.1f} m tall, three storeys of exactly 3.2 m
(thick horizontal lines = floor lines; the bottom edge is the ground),
{K} bays of {p:.2f} m (dashed vertical lines). Legend:
{legend}

Rules:
1. The image frame IS the facade: its edges are the sketch's edges; no
   sky, no roof, no ground, no street, no neighbouring buildings, no
   margin. The facade continues past the left and right edges.
2. Each bay has at most one opening (window, door, loggia, shop unit;
   garages and loading docks take two bays), centred in its box. Each
   coloured box IS the size of that opening: match its width and height
   within about 10%. Do not enlarge windows: a blue box is about
   {win_w:.2f} m wide and {win_h:.2f} m tall (sill to the top of the
   lintel / shutter box), and above it there are {wall_above:.2f} m of
   plain wall up to the next floor line. Everything outside the boxes is
   plain wall. You choose the design inside the box (frames, glazing
   bars, surrounds, shutters, grilles, sills).
   Bays with grey hatching have no opening at all.
3. Openings must never cross a floor line. Keep the same window design on
   both upper floors so the windows line up vertically.
4. Do not draw the sketch's lines, boxes or colours: the result is a
   photo, not a diagram. No floor-line cornices at the floor lines unless
   the style really has string courses there.
5. Strictly orthographic, straight-on, flat overcast daylight, no
   perspective, no shadows cast by other objects, no people, no trees or
   cars in front, no text, characters or logos anywhere (sign boards stay
   blank).

## Files

1. out/{key}_elev_a.png -- from the sketch ./in/{sketch_a}.png. Generate
   at {gw}x{gh} if your tool allows (same aspect as the sketch).
2. out/{key}_elev_b.png -- ANOTHER stretch of the SAME building, from the
   sketch ./in/{sketch_b}.png. Give your tool that sketch AND your
   {key}_elev_a result as references: identical materials, colours,
   window/door design and weathering level; only which bay holds what
   (per the sketch) and the small lived-in details differ.
3. out/{key}_plain.png -- the SAME building's blank side wall (a gable
   end without windows), 12 m wide by 4 m tall ({pw}x{ph}), straight on,
   same wall material, colour and weathering as {key}_elev_a (give that
   image as reference), with only one or two small vents. NO downpipes,
   pipes, cables, lamps, signs or any other vertical element. It must tile
   left-to-right. Nothing but wall in the frame.

When all three exist, list ./out/ and stop.
"""

WEAR = {
    "default": "rain streaks and dark run-off under sills and ledges, "
               "grime in corners, green algae and dirt splash-back near the "
               "ground, faded and patched paint or render, a repaired crack",
    "asia": "black mould and run-off streaks from every ledge, sill and "
            "air-conditioner, rust bleeding from window grilles and "
            "brackets, grime-darkened tile joints, patchy repainting, "
            "water stains at the base",
    "africa": "red-brown dust and dirt splash-back, sun-faded and chalky "
              "paint, cracked and patched plaster, rust streaks under "
              "metal grilles and roof edges, dark water marks under sills",
    "ind": "dirt and rust streaks from fixings and roof edges, dented and "
           "replaced cladding panels, faded paint, tyre and forklift "
           "scuffs near the ground, grime around the dock doors",
}

ROOF_PROMPT = """\
# Image re-rendering ONLY - no coding

Your only job: re-render each input image below as a PHOTOGRAPH with your
image-generation tool and save each result under ./out/ with the SAME
filename as its input. Do NOT write, edit or run any Python or other code
and do not touch any other file. Shell use is allowed solely to copy each
generated image from your image cache into ./out/<name>.png. Skip any
output that already exists. Small misalignments are corrected
automatically afterwards, so NEVER give up on a file: always save your
best attempt.

The inputs are Blender renders of roof strips for a flight simulator's
buildings on {region}, seen from DIRECTLY ABOVE (the roof slope runs top
to bottom; the eave is the bottom edge). Each image is re-used side by
side, so it must TILE LEFT-TO-RIGHT (its left and right edges continue
into each other). Keep the input's roofing type, colour family, and the
number and scale of tile courses / sheet ribs / panels; keep any rooftop
items (solar panels, water tanks, skylights, vents, hatches) roughly where
they are, photographic. It is a ROOF seen from above, never a wall.
Weathering: {wear}.

Style: photorealistic, strictly orthographic top-down, flat overcast
light, no shadows cast by other objects, no perspective, no vignetting, no
text, material fills the frame edge to edge.

## Files

{files}

When every output exists, list ./out/ and stop.
"""

WEAR.update({
    "mediterranean": "sun-faded and chalky paint, flaking lime render, "
                     "rust stains under iron grilles and balconies, dark "
                     "water marks under sills, dust at the base",
    "south_america": "faded and peeling paint, rust streaks under window "
                     "bars, dark humidity stains, grime splash-back at "
                     "the base, patched cement",
    "north_america": "weathered and chalky siding, mildew streaks, dirt "
                     "splash near the ground, faded trim paint",
    "australia_oceania": "sun-bleached paint, red-brown dust, efflorescence "
                         "on brick, dark streaks under sills",
})

ROOF_WEAR = {
    "default": "moderate: dirt collecting toward the eave, lichen and moss "
               "on tiles, faded and replaced pieces, rust on metal",
    "asia": "humid climate: black algae streaks, moss in the laps, rust "
            "runs on metal sheets, dirt toward the eave",
    "africa": "hot and dusty: red dust, sun-bleached colour, rust on "
              "corrugated iron, patched sheets",
}


def prep_facades(page, kind, kit, region, wdir):
    groups = []
    k_bays = K_BY_KIND.get(kind, K)
    wear = WEAR.get("ind" if kind == "ind" else region, WEAR["default"])
    for key in BRIEFS[page]:
        info = scheme_info(page, kind, kit, key)
        p = info["p"]
        zones = scheme_zones(kit[key], p)
        wz = zones["upper"]["win"]
        win_w = (wz[0][1] - wz[0][0]) * p
        win_h = wz[1][1] - wz[1][0]
        wall_above = FLOOR_H - wz[1][1]
        m = info["motif"]
        up = info["tables"]["upper"]
        u_wins = [(ph, [up[(ph + i) % len(up)] for i in range(k_bays)])
                  for ph in (0, 1 % m)]
        if "ground" in info["tables"]:
            g_wins = pick_windows(info["tables"]["ground"], k_bays)
            levels = ["upper", "upper", "ground"]
        else:                                   # upper-only scheme
            g_wins = [(None, None), (None, None)]
            levels = ["upper", "upper", "upper"]
        sketches = {}
        for tag, (gs, gseq), (us, useq) in zip("ab", g_wins, u_wins):
            rows = [useq, useq, gseq if gseq else useq]
            name = f"{key}_sketch_{tag}"
            w, h = draw_sketch(os.path.join(wdir, "in", name + ".png"),
                               rows, levels, p, zones)
            sketches[tag] = {"name": name, "rows": rows, "levels": levels,
                             "ground_start": gs, "upper_start": us,
                             "size": [w, h]}
        kinds_used = {x for s in sketches.values() for r in s["rows"]
                      for x in r}
        legend = "\n".join(f"- {LEGEND[x]}" for x in LEGEND
                           if x in kinds_used)
        gw, gh = gen_size(*sketches["a"]["size"])
        txt = PROMPT.format(
            n_img=3, region=REGION_LINE.get(region, region),
            brief=BRIEFS[page][key], wear=wear,
            sketch_a=sketches["a"]["name"], sketch_b=sketches["b"]["name"],
            w_m=k_bays * p, h_m=3 * FLOOR_H, K=k_bays, p=p, legend=legend,
            key=key, gw=gw, gh=gh, win_w=win_w, win_h=win_h,
            wall_above=wall_above, pw=1536, ph=512)
        with open(os.path.join(wdir, f"prompt_{key}.md"), "w",
                  encoding="utf-8", newline="\n") as fh:
            fh.write(txt)
        groups.append({"group": key, "p": p, "n": info["n"], "K": k_bays,
                       "zones": zones, "motif": m, "tables": info["tables"],
                       "sketches": sketches,
                       "inputs": [sketches["a"]["name"],
                                  sketches["b"]["name"]],
                       "outputs": [f"{key}_elev_a", f"{key}_elev_b",
                                   f"{key}_plain"]})
    return groups


def prep_roofs(page, kind, kit, region, wdir, cache):
    """One 2.5:1 crop of each clean roof bake -> a tileable photo."""
    from prep_ai import clean_strip
    bake_page = "ind" if kind == "ind" else page
    jobs = strip_jobs(bake_page, kind, kit,
                      None if kind == "ind" else region)
    names, files = [], []
    for name, row, job in jobs:
        if row["kind"] != "roof":
            continue
        job["out"] = os.path.join(cache, f"{job['key']}.npz")
        strip = clean_strip(job, row, "generic" if kind == "ind"
                            else region)
        w = int(round(2.5 * row["px"]))
        crop = strip[:, :w]
        nm = f"roof_{name}" if not name.startswith("roof") else name
        Image.fromarray(crop).save(os.path.join(wdir, "in", nm + ".png"))
        rk = job["roof"]["kind"]
        m_w, m_h = w / (row["px"] / row["world_h"]), row["world_h"]
        files.append(f"- ./in/{nm}.png -> out/{nm}.png: "
                     f"{ROOF_WORDS.get(rk, rk)}, {m_w:.0f} m wide x "
                     f"{m_h:.0f} m (generate at "
                     f"{min(2560, 5 * row['px'])}x{min(1024, 2 * row['px'])})")
        names.append((nm, name))
    if not names:
        return []
    txt = ROOF_PROMPT.format(
        region=REGION_LINE.get(region, region),
        wear=ROOF_WEAR.get(region, ROOF_WEAR["default"]),
        files="\n".join(files))
    with open(os.path.join(wdir, "prompt_roofs.md"), "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write(txt)
    return [{"group": "roofs", "type": "roofs",
             "strips": {nm: strip for nm, strip in names},
             "inputs": [nm for nm, _ in names],
             "outputs": [nm for nm, _ in names]}]


def prep_page(page, kind, kit, region, cache=None, roofs=True):
    wdir = os.path.join(WORK, page)
    os.makedirs(os.path.join(wdir, "in"), exist_ok=True)
    os.makedirs(os.path.join(wdir, "out"), exist_ok=True)
    groups = prep_facades(page, kind, kit, region, wdir)
    if roofs and cache:
        groups += prep_roofs(page, kind, kit, region, wdir, cache)
    with open(os.path.join(wdir, "jobs.json"), "w", encoding="utf-8") as fh:
        json.dump({"page": page, "kind": kind, "region": region,
                   "K": K_BY_KIND.get(kind, K), "px_m": PX_M,
                   "groups": groups}, fh, indent=1)
    return groups


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", nargs="+", required=True,
                    help="page names (e.g. asia_res_a), or 'ind'")
    ap.add_argument("--pkg", default=None,
                    help="v4 package with _bake_cache (needed for roofs)")
    ap.add_argument("--no-roofs", action="store_true")
    args = ap.parse_args(argv)
    specs = {p[0]: p for p in page_specs(list(REGIONS))}
    specs["ind"] = ("ind", "ind", IND_PAGE, "ind", 1.0)
    cache = os.path.join(args.pkg, "_bake_cache") if args.pkg else None
    for page in args.pages:
        _p, kind, kit, region, _s = specs[page]
        for g in prep_page(page, kind, kit, region, cache,
                           not args.no_roofs):
            if g.get("type") == "roofs":
                print(page, "roofs", g["inputs"])
                continue
            print(page, g["group"], f"p={g['p']:.3f} K={g['K']}",
                  {t: " ".join(s["rows"][0]) + " / " + " ".join(s["rows"][2])
                   for t, s in g["sketches"].items()})


if __name__ == "__main__":
    main()
