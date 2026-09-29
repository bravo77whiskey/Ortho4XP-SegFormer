"""Level 2: conform free facade designs onto the v4 grid, rebuild pages.

    python image-gen/design_compose.py --base <pkg with level-1 pages> --out <pkg>
                                --pages europe_res_a [--size 2048] [--diag]

Per scheme (``design_work/<page>/jobs.json`` from design_prep.py):

1. **Openings.**  Each elevation is resized to its nominal size (K bays x
   3 floors at 80 px/m).  Pixels far from the wall colour (blurred Lab
   distance plus edge energy, Otsu threshold) form an opening mask; in every
   non-blank cell the component that best fills that cell's zone is its
   opening.
2. **Conform.**  Upper floors: the spacing of the two window rows sets the
   vertical scale.  The ground floor is what lies below.  Each floor is
   then warped in x, piecewise-linearly, so every measured opening centre
   lands on its bay centre.
3. **Rebuild.**  The strip's bay table (the bake's own, so doors stay
   where buildings.py expects them) is covered greedily by runs of source
   bays with matching kinds, taken from the conformed floors of both
   elevations.  Runs are joined with a minimum-error cut through the wall
   at the bay line.
4. **Maps.**  Normals are derived from the albedo.  Gloss is high on
   glass (dark pixels inside openings) and low elsewhere.  The plain side
   wall is made seamless in u and tiled.

Porch and trim come from ``--base``; roofs too, unless the page has
roof jobs (one tileable photo per roof strip, design_prep.prep_roofs).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.dirname(HERE)
if V4 not in sys.path:
    sys.path.insert(0, V4)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from layout import PAGE_PX, page_layout  # noqa: E402
from compose_ai import derived_normal, min_cut  # noqa: E402
from design_prep import FLOOR_H, LINTEL, WORK  # noqa: E402

FH = 256            # floor height in px (80 px/m)
OV = 20             # half-width of a run join (px)
GLOSS_WALL, GLOSS_GLASS, GLOSS_DOOR = 26, 190, 60
MAX_RESIDUAL = 5.0      # px; worse elevations are not used as bay sources
COMPAT = {"blank": ("small", "win"), "small": ("win", "blank"),
          "win": ("small",), "loggia": ("win",), "door": ("door",),
          "shop": ("shop",), "garage": ("garage", "door"),
          "garage+": ("blank", "win"), "dock": ("dock", "door"),
          "dock+": ("blank", "win")}
SPAN2 = ("garage", "dock")
CONT = "+"


def no_opening(kind):
    return kind == "blank" or kind.endswith(CONT)


def default_levels(rows):
    return ["upper"] * (len(rows) - 1) + ["ground"]


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


# ------------------------------------------------------------ openings

def opening_mask(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab = cv2.GaussianBlur(lab, (0, 0), 3.0)
    ref = np.median(lab.reshape(-1, 3), 0)
    d = np.sqrt(((lab - ref) ** 2 * np.array([1.0, 2.0, 2.0])).sum(-1))
    lum = lab[..., 0]
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1)
    g = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 4.0)
    score = d + 0.6 * g
    s8 = np.clip(score / max(1e-3, np.percentile(score, 99.5)) * 255, 0,
                 255).astype(np.uint8)
    _t, m = cv2.threshold(s8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return m > 0


def zone_box(kind, b, r, P, levels, zones):
    """Zone of (row r from the top, bay b) in nominal image px."""
    (fx0, fx1), (z0, z1) = zones[levels[r]][kind]
    y_floor = (r + 1) * FH
    return (b * P + fx0 * P, y_floor - z1 * FH / FLOOR_H,
            b * P + fx1 * P, y_floor - z0 * FH / FLOOR_H)


def find_openings(mask, rows, P, zones, levels=None):
    """{(r, b): (x0, y0, x1, y1)} for every non-blank cell found.

    The mask is clipped to each cell's (slightly grown) zone before
    labelling, so a splash band or plinth that touches several openings
    cannot merge them into one wall-wide blob."""
    H, W = mask.shape
    levels = levels or default_levels(rows)
    found = {}
    for r, kinds in enumerate(rows):
        for b, kind in enumerate(kinds):
            if no_opening(kind):
                continue
            zx0, zy0, zx1, zy1 = zone_box(kind, b, r, P, levels, zones)
            cx0 = int(max(0, zx0 - 0.08 * P))
            cx1 = int(min(W, zx1 + 0.08 * P))
            cy0 = int(max(0, zy0 - 0.15 * FH / FLOOR_H))
            cy1 = int(min(H, zy1 + 0.15 * FH / FLOOR_H))
            sub = mask[cy0:cy1, cx0:cx1].astype(np.uint8)
            if not sub.any():
                continue
            n, _lab, stats, _ = cv2.connectedComponentsWithStats(sub, 8)
            cw, ch = sub.shape[1], sub.shape[0]
            zone_a = max(1.0, (zx1 - zx0) * (zy1 - zy0))
            best, best_s = None, 0.0
            for i in range(1, n):
                x, y, w, h, a = stats[i]
                spans = x <= 1 and x + w >= cw - 1
                if (spans and kind not in ("shop", "loggia") + SPAN2
                        and h < 0.5 * ch):
                    continue        # a horizontal band through the cell
                s = (w * h) / zone_a + 0.3 * a / zone_a
                if s > best_s:
                    best, best_s = (x + cx0, y + cy0, x + w + cx0,
                                    y + h + cy0), s
            if best is not None and best_s > 0.08:
                found[(r, b)] = best
    return found


# ------------------------------------------------------------ conform

def conform(img, rows, P, K, zones, levels=None):
    """Return (floor images top->bottom [FH, W, 3], floor masks, report)."""
    W = int(round(K * P))
    H = len(rows) * FH
    levels = levels or default_levels(rows)
    im = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    mask = opening_mask(im)
    found = find_openings(mask, rows, P, zones, levels)

    def zc(r, b):
        z = zone_box(rows[r][b], b, r, P, levels, zones)
        return (z[0] + z[2]) / 2

    if found:
        offs = {k: (bx[0] + bx[2]) / 2 - zc(*k) for k, bx in found.items()}
        med = float(np.median(list(offs.values())))
        found = {k: bx for k, bx in found.items()
                 if abs(offs[k] - med) <= 0.3 * P}
    rep = {"found": len(found),
           "expected": sum(not no_opening(k) for r in rows for k in r)}
    # vertical: spacing of the two upper window rows
    cy = {}
    for r in range(len(rows) - 1):
        ys = [(bx[1] + bx[3]) / 2 for (rr, b), bx in found.items()
              if rr == r and rows[r][b] == "win"]
        if ys:
            cy[r] = float(np.median(ys))
    scale = 1.0
    if 0 in cy and 1 in cy:
        scale = float(np.clip((cy[1] - cy[0]) / FH, 0.85, 1.18))
    rep["y_scale"] = round(scale, 3)
    # upper floors: slide each band so its window row sits at the scheme's
    # intended height (kit sill / head), whatever height the model drew it
    # at -- the roof fascia hides the top ~0.3 m of the top floor, so
    # windows drawn high read as touching the eaves
    # The head is what matters (the model tends to draw windows taller than
    # asked, and a sill a little low is far less visible than a head
    # hidden under the eaves), so the window heads are aligned.
    (_x, (z0, z1)) = zones["upper"]["win"]
    want = FH - (z1 - LINTEL) * FH / FLOOR_H
    heads = {}
    for r in range(len(rows) - 1):
        ys = [bx[1] for (rr, b), bx in found.items()
              if rr == r and rows[r][b] == "win"]
        if ys:
            heads[r] = float(np.median(ys))
    bands = []
    for r in range(len(rows) - 1):
        top = r * FH * scale
        if r in heads:
            top = heads[r] - want * scale
        bands.append((top, top + FH * scale))
    rep["upper_shift_px"] = [round(b[0] - r * FH * scale, 1)
                             for r, b in enumerate(bands)]
    bands.append((bands[-1][1], float(H)))
    # horizontal: per floor, measured centres -> bay centres
    # shop units are asymmetric (window + door) and fill their bay: their
    # centres say nothing about the grid, so shop rows follow the others
    found = {k: bx for k, bx in found.items()
             if rows[k[0]][k[1]] != "shop"}
    # measured position of each bay's centre = its ideal centre plus the
    # offset of its opening from that opening's zone centre (a two-bay
    # garage / dock moves both of its bays)
    T = np.array([(b + 0.5) * P for b in range(K)])
    offs = {}
    for (r, b), bx in found.items():
        o = (bx[0] + bx[2]) / 2 - zc(r, b)
        offs[(r, b)] = o
        if rows[r][b] in SPAN2 and b + 1 < K:
            offs[(r, b + 1)] = o
    col_med = {}
    for b in range(K):
        xs = [o for (r, bb), o in offs.items() if bb == b]
        if xs:
            col_med[b] = T[b] + float(np.median(xs))
    floors, masks, errs = [], [], []
    for r, (y0, y1) in enumerate(bands):
        meas = {b: T[b] + o for (rr, b), o in offs.items() if rr == r}
        for b, x in col_med.items():
            meas.setdefault(b, x)
        if len(meas) >= 2:
            bs = sorted(meas)
            fit = np.polyfit([T[b] for b in bs], [meas[b] for b in bs], 1)
            S = np.array([meas.get(b, np.polyval(fit, T[b]))
                          for b in range(K)])
            slope = fit[0]
        else:
            S, slope = T.copy(), 1.0
        errs += [abs(S[b] - T[b]) for b in meas]
        Te = np.r_[0.0, T, float(W)]
        Se = np.r_[S[0] - T[0] * slope, S, S[-1] + (W - T[-1]) * slope]
        xs = np.interp(np.arange(W, dtype=np.float32), Te, Se)
        ys = y0 + (np.arange(FH, dtype=np.float32) + 0.5) * (y1 - y0) / FH \
            - 0.5
        if r == 0 and len(bands) > 2:
            # above the image top: the same wall one floor down (under the
            # top row's windows) instead of a mirror of the lintels
            ys = np.where(ys < 0, ys + (bands[1][0] - bands[0][0]), ys)
        mapx = np.repeat(xs[None, :], FH, 0).astype(np.float32)
        mapy = np.repeat(ys[:, None], W, 1).astype(np.float32)
        floors.append(cv2.remap(im, mapx, mapy, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT))
        masks.append(cv2.remap(mask.astype(np.uint8) * 255, mapx, mapy,
                               cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_REFLECT) > 0)
    rep["centre_err_px"] = round(float(np.median(errs)), 1) if errs else None
    rep["centre_err_max_px"] = round(float(np.max(errs)), 1) if errs else None
    # residual: openings of the conformed floors against the bay centres
    stack = np.concatenate(floors, 0)
    after = find_openings(opening_mask(stack), rows, P, zones, levels)
    res = [abs((bx[0] + bx[2]) / 2 - zc(r, b))
           for (r, b), bx in after.items() if rows[r][b] != "shop"]
    rep["residual_px"] = round(float(np.median(res)), 1) if res else None
    rep["residual_max_px"] = round(float(np.max(res)), 1) if res else None
    return floors, masks, rep, (im, mask, found)


def gloss_row(floor, mask, kinds, P):
    """Per-pixel gloss (uint8) for one conformed floor.

    Glass = the darker pixels inside the bay's opening box, trimmed at the
    frame and above the sill: the opening mask also holds sills, lintels
    and their shadows, and a glossy brick sill sparkles white in the sim."""
    lum = cv2.cvtColor(floor, cv2.COLOR_RGB2GRAY).astype(np.float32)
    out = np.full(lum.shape, GLOSS_WALL, np.uint8)
    for b, kind in enumerate(kinds):
        if no_opening(kind):
            continue
        span = 2 if kind in SPAN2 else 1
        x0, x1 = int(b * P), min(mask.shape[1], int((b + span) * P))
        m = mask[:, x0:x1].astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if n < 2:
            continue
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        bx, by, bw, bh = stats[i, :4]
        ix0, ix1 = bx + int(0.08 * bw), bx + bw - int(0.08 * bw)
        iy0, iy1 = by + int(0.06 * bh), by + bh - int(0.18 * bh)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        cell = lum[iy0:iy1, x0 + ix0:x0 + ix1]
        glass = cell < np.percentile(cell, 55)
        val = GLOSS_DOOR if kind in ("door",) + SPAN2 else GLOSS_GLASS
        out[iy0:iy1, x0 + ix0:x0 + ix1][glass] = val
    return out


def match_to(src, ref, src_mask, ref_mask):
    """Per-channel mean/std of src's wall pixels -> ref's."""
    s = src.astype(np.float32)
    a = s[~src_mask]
    b = ref.astype(np.float32)[~ref_mask]
    if len(a) < 100 or len(b) < 100:
        return src
    out = (s - a.mean(0)) * (b.std(0) / np.maximum(a.std(0), 1e-3)) \
        + b.mean(0)
    return np.clip(out, 0, 255).astype(np.uint8)


# ------------------------------------------------------------ rebuild

def plan_runs(table, sources, K, rng):
    """Cover ``table`` with runs (i, source, j, L) of matching kinds."""
    n = len(table)
    runs, i, prev = [], 0, None
    uses = {}
    while i < n:
        cands = []
        for si, src in enumerate(sources):
            for j in range(K):
                L = 0
                while (j + L < K and i + L < n
                       and src["kinds"][j + L] == table[i + L]):
                    L += 1
                # never end a run between the two bays of a garage / dock
                while L and i + L < n and table[i + L].endswith(CONT):
                    L -= 1
                if L and table[i].endswith(CONT):
                    L = 0
                if L:
                    # long runs join less often; reused stretches repeat
                    w = L ** 1.5 / (1.0 + 2.0 * uses.get((si, j), 0))
                    if (si, j) == prev:
                        w *= 0.1
                    cands.append((w, si, j, L))
        if not cands:
            for alt in COMPAT.get(table[i], ()):
                for si, src in enumerate(sources):
                    for j in range(K):
                        if src["kinds"][j] == alt:
                            cands.append((1.0, si, j, 1))
                if cands:
                    break
        if not cands:
            raise RuntimeError(f"no source bay for kind {table[i]!r}")
        tot = sum(c[0] for c in cands)
        pick = rng.random() * tot
        for w, si, j, L in cands:
            pick -= w
            if pick <= 0:
                break
        runs.append((i, si, j, L))
        prev = (si, j)
        for k in range(L):
            uses[(si, j + k)] = uses.get((si, j + k), 0) + 1
        i += L
    return runs


def render_runs(runs, sources, n, P, width=PAGE_PX, key="img"):
    """Each run as a full-width layer (NaN where not covered)."""
    layers = []
    for i, si, j, L in runs:
        src = sources[si][key]
        c0 = int(np.floor(i * P - OV))
        c1 = int(np.ceil((i + L) * P + OV))
        cols = np.arange(c0, c1)
        srcx = (cols - i * P + j * P).astype(np.float32)
        h = src.shape[0]
        mapx = np.repeat(srcx[None, :], h, 0)
        mapy = np.repeat(np.arange(h, dtype=np.float32)[:, None], len(cols),
                         1)
        interp = cv2.INTER_NEAREST if src.dtype == bool else cv2.INTER_LINEAR
        s = src.astype(np.uint8) if src.dtype == bool else src
        vals = cv2.remap(s, mapx, mapy, interp,
                         borderMode=cv2.BORDER_REFLECT)
        layers.append((cols % width, vals))
    return layers


def level_layers(runs, layers, P):
    """Remove low-frequency tone steps at run joins: both sides of a join
    meet at their mean (per row band, per channel); the correction ramps
    linearly to zero across each run."""
    R = len(runs)
    if R < 2:
        return layers

    def overlap(layer, want):
        cols, vals = layer
        pos = {c: k for k, c in enumerate(cols)}
        idx = [pos.get(c) for c in want]
        if any(k is None for k in idx):
            return None
        return vals[:, idx].astype(np.float32).mean(1)        # [h, 3]

    deltas = []
    for r in range(R):
        b = int(round(runs[r][0] * P)) % PAGE_PX
        want = np.arange(b - OV, b + OV) % PAGE_PX
        a, c = overlap(layers[r - 1], want), overlap(layers[r], want)
        if a is None or c is None:
            deltas.append(np.zeros((layers[r][1].shape[0], 3), np.float32))
            continue
        d = c - a
        deltas.append(cv2.GaussianBlur(d[:, None, :], (0, 0), 0.1,
                                       sigmaY=24.0)[:, 0, :])
    out = []
    for r, (cols, vals) in enumerate(layers):
        w = vals.shape[1]
        t = (np.arange(w, dtype=np.float32) / max(1, w - 1))[None, :, None]
        corr = -(1 - t) * deltas[r][:, None, :] / 2.0 \
            + t * deltas[(r + 1) % R][:, None, :] / 2.0
        out.append((cols, np.clip(vals.astype(np.float32) + corr, 0, 255)))
    return out


def stitch_runs(runs, sources, n, P, key="img"):
    layers = render_runs(runs, sources, n, P, key=key)
    ch = sources[0][key].shape[2] if sources[0][key].ndim == 3 else 0
    h = sources[0][key].shape[0]
    shape = (h, PAGE_PX, ch) if ch else (h, PAGE_PX)
    out = np.zeros(shape, np.float32)
    if key == "img":
        layers = level_layers(runs, layers, P)
    # cores first
    for (i, _si, _j, L), (cols, vals) in zip(runs, layers):
        core = (cols >= 0)
        out[:, cols[core]] = vals[:, core]
    if key != "img":
        return out
    # joins: run r-1 (left) meets run r (right) at column i_r * P
    R = len(runs)
    for r in range(R):
        i = runs[r][0]
        b = int(round(i * P)) % PAGE_PX
        want = (np.arange(b - OV, b + OV)) % PAGE_PX

        def take(layer):
            cols, vals = layer
            pos = {c: k for k, c in enumerate(cols)}
            idx = [pos.get(c) for c in want]
            if any(k is None for k in idx):
                return None
            return vals[:, idx].astype(np.float32)
        a = take(layers[r - 1])
        c = take(layers[r])
        if a is None or c is None:
            continue
        cut = min_cut(a, c)[:, None]
        xs = np.arange(2 * OV)[None, :]
        t = np.clip((xs - cut + 3) / 6.0, 0, 1)[..., None]
        out[:, want] = a * (1 - t) + c * t
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def plain_strip(img, px_h, world_h, src_w_m=12.0):
    """Blank side wall -> seamless full-width strip."""
    ppm = px_h / world_h
    w = max(32, int(round(src_w_m * ppm)))
    tile = cv2.resize(img, (w, px_h), interpolation=cv2.INTER_AREA)
    o = max(8, w // 8)
    tile = tile.astype(np.float32)
    # seamless in u: min-error cut between the tile's two ends
    a, b = tile[:, w - o:], tile[:, :o]
    cut = min_cut(a, b)[:, None]
    t = (np.arange(o)[None, :] >= cut).astype(np.float32)[..., None]
    seam = a * (1 - t) + b * t
    core = np.concatenate([seam, tile[:, o:w - o]], 1)
    reps = max(1, int(round(PAGE_PX / core.shape[1])))
    row = np.concatenate([core] * reps, 1)
    row = cv2.resize(row, (PAGE_PX, px_h), interpolation=cv2.INTER_AREA)
    return np.clip(row + 0.5, 0, 255).astype(np.uint8)


# ------------------------------------------------------------ page

def compose_scheme(wdir, g, K, rng, report, diag=False):
    P = PAGE_PX / g["n"]
    K = g.get("K", K)
    outs = {}
    conformed = {}
    for tag in "ab":
        path = os.path.join(wdir, "out", f"{g['group']}_elev_{tag}.png")
        if not os.path.isfile(path):
            continue
        sk = g["sketches"][tag]
        rows = sk["rows"]
        levels = sk.get("levels") or default_levels(rows)
        floors, masks, rep, dbg = conform(load_rgb(path), rows, P, K,
                                          g["zones"], levels)
        report[f"{g['group']}_{tag}"] = rep
        conformed[tag] = (floors, masks, rows, levels)
        if diag:
            save_diag(wdir, f"{g['group']}_{tag}", dbg, rows, levels, P,
                      floors, g["zones"])
    if "a" not in conformed:
        return None
    # a badly conformed elevation (the model drew its openings off the
    # grid) would cut openings in half when its bays are reused: drop it
    # if the other elevation still covers every bay kind of the tables
    bad = [t for t in conformed
           if (report[f"{g['group']}_{t}"].get("residual_px") or 0)
           > MAX_RESIDUAL]
    for t in bad:
        rest = [x for x in conformed if x != t]
        if not rest:
            continue
        have = {k for x in rest for row in conformed[x][2] for k in row}
        need = {k for tab in g["tables"].values() for k in tab}
        if need <= have:
            del conformed[t]
            report[f"{g['group']}_{t}"]["dropped"] = True
    if "a" not in conformed:
        # keep the names below simple: the survivor becomes "a"
        conformed = {"a": next(iter(conformed.values()))}
    if "b" in conformed:
        fa = np.concatenate(conformed["a"][0], 0)
        ma = np.concatenate(conformed["a"][1], 0)
        fb, mb, rb, lb = conformed["b"]
        full_b = match_to(np.concatenate(fb, 0), fa,
                          np.concatenate(mb, 0), ma)
        conformed["b"] = (np.split(full_b, len(fb), 0), mb, rb, lb)
    sources = {"upper": [], "ground": []}
    for tag, (floors, masks, rows, levels) in conformed.items():
        for r, (fl, m) in enumerate(zip(floors, masks)):
            sources[levels[r]].append({
                "img": fl, "kinds": rows[r],
                "gloss": gloss_row(fl, m, rows[r], P), "tag": f"{tag}{r}"})
    for level in [lv for lv in ("upper", "ground") if lv in g["tables"]]:
        table = g["tables"][level]
        if not sources[level]:
            continue
        runs = plan_runs(table, sources[level], K, rng)
        img = stitch_runs(runs, sources[level], g["n"], P)
        gl = stitch_runs(runs, sources[level], g["n"], P, key="gloss")
        outs[f"{g['group']}_{level}"] = (img, gl.astype(np.uint8))
        report[f"{g['group']}_{level}_runs"] = [
            f"{sources[level][si]['tag']}:{j}+{L}" for _i, si, j, L in runs]
    return outs


def save_diag(wdir, name, dbg, rows, levels, P, floors, zones):
    im, mask, found = dbg
    d = os.path.join(wdir, "diag")
    os.makedirs(d, exist_ok=True)
    vis = im.copy()
    vis[mask] = (vis[mask] * 0.5 + np.array([255, 0, 255]) * 0.5).astype(
        np.uint8)
    for r, kinds in enumerate(rows):
        for b, kind in enumerate(kinds):
            if not no_opening(kind):
                x0, y0, x1, y1 = [int(v) for v in zone_box(kind, b, r, P,
                                                           levels, zones)]
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 90, 255), 2)
    for (r, b), (x0, y0, x1, y1) in found.items():
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 220, 0), 3)
    conf = np.concatenate(floors, 0).copy()
    for b in range(len(rows[0]) + 1):
        cv2.line(conf, (int(b * P), 0), (int(b * P), conf.shape[0]),
                 (255, 255, 0), 1)
    Image.fromarray(np.concatenate([vis, conf], 0)).save(
        os.path.join(d, name + ".png"))


def roof_strip(img, px_h, crop_w):
    """A tileable roof photo -> the full-width strip (seamless in u)."""
    tile = cv2.resize(img, (crop_w, px_h), interpolation=cv2.INTER_AREA)
    tile = tile.astype(np.float32)
    o = max(8, crop_w // 8)
    a, b = tile[:, crop_w - o:], tile[:, :o]
    cut = min_cut(a, b)[:, None]
    t = (np.arange(o)[None, :] >= cut).astype(np.float32)[..., None]
    core = np.concatenate([a * (1 - t) + b * t, tile[:, o:crop_w - o]], 1)
    reps = max(1, int(round(PAGE_PX / core.shape[1])))
    row = np.concatenate([core] * reps, 1)
    row = cv2.resize(row, (PAGE_PX, px_h), interpolation=cv2.INTER_AREA)
    return np.clip(row + 0.5, 0, 255).astype(np.uint8)


def paste(albedo, normal, lay, name, img, gloss, strength=1.4):
    r0, px = lay[name]["row0"], lay[name]["px"]
    albedo[r0:r0 + px] = img
    normal[r0:r0 + px, :, :2] = derived_normal(img, strength=strength
                                               ).astype(np.uint8)
    normal[r0:r0 + px, :, 2] = 10
    normal[r0:r0 + px, :, 3] = gloss


def build_rows(wdir, jobs, lay, rng, report, diag):
    """{strip name: (albedo rows, gloss rows or scalar, normal strength)}."""
    rows = {}
    for g in jobs["groups"]:
        if g.get("type") == "roofs":
            for nm, strip in g["strips"].items():
                op = os.path.join(wdir, "out", nm + ".png")
                if not os.path.isfile(op):
                    continue
                px = lay[strip]["px"]
                rows[strip] = (roof_strip(load_rgb(op), px,
                                          int(round(2.5 * px))),
                               GLOSS_WALL, 2.5)
            continue
        outs = compose_scheme(wdir, g, jobs["K"], rng, report, diag)
        if outs is None:
            print(f"  {jobs['page']}/{g['group']}: no elevations yet")
            continue
        for name, (img, gl) in outs.items():
            rows[name] = (img, gl, 1.4)
        pp = os.path.join(wdir, "out", f"{g['group']}_plain.png")
        pname = f"{g['group']}_plain"
        if os.path.isfile(pp) and pname in lay:
            row = lay[pname]
            rows[pname] = (plain_strip(load_rgb(pp), row["px"],
                                       row["world_h"]), GLOSS_WALL, 1.0)
    return rows


IND_PAGES = ("ind_clean", "ind_mild", "ind_worn")


def compose_page(base, out, page, size, rep_all, diag, work=WORK):
    """``page`` is a page name, or ``ind`` for the three industrial wear
    grades (one design; each grade keeps its own trim)."""
    wdir = os.path.join(work, page)
    with open(os.path.join(wdir, "jobs.json"), encoding="utf-8") as fh:
        jobs = json.load(fh)
    jobs.setdefault("page", page)
    lay = page_layout(jobs["kind"])
    report = {}
    rows = build_rows(wdir, jobs, lay, random.Random(page), report, diag)
    targets = IND_PAGES if page == "ind" else (page,)
    tex = os.path.join(out, "textures")
    os.makedirs(tex, exist_ok=True)
    for target in targets:
        albedo = np.array(Image.open(os.path.join(
            base, "textures", f"o4v4_{target}.png")).convert("RGB"))
        normal = np.array(Image.open(os.path.join(
            base, "textures", f"o4v4_{target}_nml.png")).convert("RGBA"))
        if albedo.shape[0] != PAGE_PX:
            raise SystemExit("--base must hold 4096 pages")
        for name, (img, gl, strength) in rows.items():
            paste(albedo, normal, lay, name, img, gl, strength)
        alb = Image.fromarray(albedo, "RGB")
        nml = Image.fromarray(normal, "RGBA")
        if size != PAGE_PX:
            alb = alb.resize((size, size), Image.BOX)
            nml = nml.resize((size, size), Image.BOX)
        alb.save(os.path.join(tex, f"o4v4_{target}.png"), compress_level=6)
        nml.save(os.path.join(tex, f"o4v4_{target}_nml.png"),
                 compress_level=6)
        print(f"  {target}: replaced {len(rows)} strips")
    rep_all[page] = {"replaced": sorted(rows), "schemes": report}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True,
                    help="package whose pages supply roofs/trim/porch")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pages", nargs="+", required=True,
                    help="page names, or 'ind' for the industrial pages")
    ap.add_argument("--size", type=int, default=PAGE_PX)
    ap.add_argument("--work", default=WORK)
    ap.add_argument("--diag", action="store_true",
                    help="write detection / conform overlays to diag/")
    args = ap.parse_args(argv)
    rep = {}
    for page in args.pages:
        compose_page(args.base, args.out, page, args.size, rep, args.diag,
                     args.work)
    path = os.path.join(args.out, "design_report.json")
    old = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            old = json.load(fh)
    old.update(rep)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(old, fh, indent=1)
    for page, r in rep.items():
        for k, v in r["schemes"].items():
            if not k.endswith("_runs"):
                print(page, k, v)


if __name__ == "__main__":
    main()
