"""Split forest canopy into stands and give each stand its own height.

Compares stand shapes and sizes on the hotspots ZL18/19 textures:

  current      today's rule: crown tops smoothed by a 20 m Gaussian, cut into
               height zones, zone patches under 2000 m2 merged, one height
               per polygon (median)
  grid         square cells
  voronoi      cells around jittered seeds
  warped       voronoi cells sampled through a fractal noise warp (wavy edges)
  watershed    seeds flooded over the height map (cv2.watershed): edges follow
               height changes, wander with added noise elsewhere

Heights are measured on canopy run A and scored against run B (the same
texture through shifted model windows, canopy_maps.py): stands so small that
they fit the model's noise score worse against B than against A.

  .venv/Scripts/python.exe experiments/veg_stands_20261105/stands.py [--sheets]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import O4_Forest_Stands as STANDS  # noqa: E402

DATA = os.environ.get("O4_REVIEW_DATA", "G:/Ortho4XP_Review")
SET = os.path.join(DATA, "sets", "hotspots", "BI")
SCRATCH = os.path.join(DATA, "_scratch", "veg_stands")
BASELINE = os.path.join(DATA, "runs", "hotspots-baseline", "BI")

CROWN_R_M = 3.0
MIN_TREE_H = 2.0          # CANOPY_MIN_TREE_H_M
MEASURED_SHARE = 0.1      # FOREST_MEASURED_MIN_SHARE
BANDS = (6.0, 9.0, 12.0, 15.0)
MIN_GAIN = 3.0
SHRUB_H = 5.0             # MIN_TREE_ASSET_M: under it shrubs
GRID_M = 1.1              # working grid


def load_sample(key):
    samples = {s["key"]: s for s in json.load(open(os.path.join(SET, "samples.json")))}
    s = samples[key]
    lat = 0.5 * (s["bbox"][0] + s["bbox"][2])
    deg_w = s["bbox"][3] - s["bbox"][1]
    tex_m = deg_w * 111320.0 * np.cos(np.radians(lat))
    return s, tex_m


def baseline_polys(key):
    t = open(os.path.join(BASELINE, key + ".js")).read()
    return json.loads(t[t.index("= {") + 2:].rstrip().rstrip(";"))


def tree_mask_and_stock(key, size_px):
    """Area-forest pixels of the baseline run, on a size_px grid; the stock height used there."""
    d = baseline_polys(key)
    scale = size_px / 4096.0
    m = np.zeros((size_px, size_px), np.uint8)
    stock = {}
    for v in d["veg"]:
        r = v["r"]
        if not r or not isinstance(r[0][0], list):
            continue   # single trees
        outer = np.round(np.asarray(r[0]) * scale).astype(np.int32)
        cv2.fillPoly(m, [outer], 1)
        for hole in r[1:]:
            cv2.fillPoly(m, [np.round(np.asarray(hole) * scale).astype(np.int32)], 0)
        if "lib/g8/" in v["a"] and "h" in v:
            stock[round(v["h"], 1)] = stock.get(round(v["h"], 1), 0) + 1
    stock_h = max(stock) if stock else 17.2   # the full-height forest is the tallest
    return m.astype(bool), stock_h


def crown_tops(h, px_m):
    r = max(1, int(round(CROWN_R_M / px_m)))
    return cv2.dilate(h, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))


def band_of(h, stock):
    """Planted canopy height for a measured height (nan: unknown -> stock)."""
    out = np.full(h.shape, stock, np.float32)
    known = ~np.isnan(h)
    choices = np.asarray([b for b in BANDS if b <= stock - MIN_GAIN] + [stock], np.float32)
    idx = np.abs(h[known, None] - choices[None, :]).argmin(axis=1)
    out[known] = choices[idx]
    out[known & (h < SHRUB_H)] = 3.0
    return out


# ── stand labelings ───────────────────────────────────────────────────────────

def seeds(shape, spacing_px, rng):
    """Jittered-grid seeds (one per spacing cell, +-0.45 cell)."""
    h, w = shape
    ys, xs = np.mgrid[0:h:spacing_px, 0:w:spacing_px]
    ys = ys + (rng.random(ys.shape) - 0.5) * 0.9 * spacing_px + spacing_px / 2
    xs = xs + (rng.random(xs.shape) - 0.5) * 0.9 * spacing_px + spacing_px / 2
    ys = np.clip(ys, 0, h - 1).astype(int).ravel()
    xs = np.clip(xs, 0, w - 1).astype(int).ravel()
    return ys, xs


def voronoi(shape, spacing_px, rng):
    ys, xs = seeds(shape, spacing_px, rng)
    img = np.ones(shape, np.uint8)
    img[ys, xs] = 0
    _d, lab = cv2.distanceTransformWithLabels(img, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    return lab.astype(np.int32)


def fractal_noise(shape, scale_px, rng, octaves=3):
    """Smooth noise in [-1, 1]-ish, features ~scale_px, finer octaves at half amplitude."""
    h, w = shape
    out = np.zeros(shape, np.float32)
    amp, s, total = 1.0, float(scale_px), 0.0
    for _ in range(octaves):
        gh, gw = max(2, int(h / s) + 2), max(2, int(w / s) + 2)
        g = rng.standard_normal((gh, gw)).astype(np.float32)
        out += amp * cv2.resize(g, (int(gw * s), int(gh * s)), interpolation=cv2.INTER_CUBIC)[:h, :w]
        total += amp
        amp *= 0.5
        s /= 2.0
    return out / total


def warped(shape, spacing_px, rng, amount=0.45):
    lab = voronoi(shape, spacing_px, rng)
    h, w = shape
    dx = fractal_noise(shape, spacing_px * 0.6, rng) * amount * spacing_px
    dy = fractal_noise(shape, spacing_px * 0.6, rng) * amount * spacing_px
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(lab.astype(np.float32), gx + dx, gy + dy, cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_REFLECT).astype(np.int32)


def watershed(shape, spacing_px, rng, height, mask, edge_m=4.0, noise=0.35):
    """Seeds flooded over an image of the smoothed height plus noise: stand
    edges follow height steps of ~edge_m and wander through flat canopy."""
    ys, xs = seeds(shape, spacing_px, rng)
    markers = np.zeros(shape, np.int32)
    markers[ys, xs] = np.arange(1, len(ys) + 1)
    hs = cv2.GaussianBlur(np.nan_to_num(height), (0, 0), 2.0)
    n = fractal_noise(shape, spacing_px * 0.5, rng)
    v = hs / edge_m * 40.0 + n * noise * 40.0
    v = np.where(mask, v, v + 160.0)   # crossing open ground costs a lot
    v = np.clip(v - v.min(), 0, 255).astype(np.uint8)
    lab = cv2.watershed(cv2.merge([v, v, v]), markers)
    bad = lab <= 0
    if bad.any():
        lab2 = cv2.dilate(np.where(bad, 0, lab).astype(np.float32), np.ones((3, 3), np.uint8))
        lab = np.where(bad, lab2.astype(np.int32), lab)
    return lab


def grid(shape, spacing_px, rng):
    h, w = shape
    oy, ox = rng.integers(0, spacing_px, 2)
    yy, xx = np.mgrid[0:h, 0:w]
    return ((yy + oy) // spacing_px) * (w // spacing_px + 2) + (xx + ox) // spacing_px + 1


# ── heights per stand ─────────────────────────────────────────────────────────

def stand_heights(labels, tops, mask, px_m, min_patch_m2=0.0):
    """Per pixel: the median measured crown top of its stand (nan: unmeasured)."""
    lab = np.where(mask, labels, 0)
    if min_patch_m2:
        lab = split_disconnected(lab)
    meas = mask & (tops >= MIN_TREE_H)
    n = int(lab.max()) + 1
    cnt = np.bincount(lab[mask].ravel(), minlength=n)
    mcnt = np.bincount(lab[meas].ravel(), minlength=n)
    med = np.full(n, np.nan, np.float32)
    order = np.argsort(lab[meas].ravel(), kind="stable")
    vals = tops[meas].ravel()[order]
    labs = lab[meas].ravel()[order]
    if len(labs):
        starts = np.searchsorted(labs, np.arange(n))
        ends = np.searchsorted(labs, np.arange(n), side="right")
        for i in np.nonzero((mcnt >= np.maximum(1, MEASURED_SHARE * cnt)) & (cnt > 0))[0]:
            med[i] = np.median(vals[starts[i]:ends[i]])
    out = med[lab]
    out[~mask] = np.nan
    return out, lab


def split_disconnected(lab):
    """Pieces of a stand cut apart by open ground become their own stands."""
    out = np.zeros_like(lab)
    n, cc = cv2.connectedComponents((lab > 0).astype(np.uint8), connectivity=8)
    key = cc.astype(np.int64) * (int(lab.max()) + 1) + lab
    _u, inv = np.unique(key[lab > 0], return_inverse=True)
    out[lab > 0] = inv + 1
    return out


def merge_small(band, mask, px_m, min_m2):
    """Same-band patches under min_m2 take the band of the nearest bigger patch (_merge_small_zones)."""
    min_px = min_m2 / (px_m * px_m)
    vals = np.unique(band[mask])
    code = np.zeros(band.shape, np.uint8)
    for i, v in enumerate(vals):
        code[mask & (band == v)] = i + 1
    small = np.zeros(band.shape, bool)
    for z in range(1, len(vals) + 1):
        _n, lab, st, _ = cv2.connectedComponentsWithStats((code == z).astype(np.uint8), connectivity=8)
        tiny = np.nonzero(st[1:, cv2.CC_STAT_AREA] < min_px)[0] + 1
        if len(tiny):
            small |= np.isin(lab, tiny)
    keep = mask & ~small
    if small.any() and keep.any():
        _d, near = cv2.distanceTransformWithLabels((~keep).astype(np.uint8), cv2.DIST_L2, 5,
                                                   labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(keep)
        lut = np.zeros(int(near.max()) + 1, np.float32)
        lut[near[ys, xs]] = band[ys, xs]
        band = band.copy()
        band[small] = lut[near[small]]
    return band


def smoothed_map(tops, mask, px_m):
    """Today's 20 m smoothed height map, NaN where little canopy around is measured."""
    meas = (mask & (tops >= MIN_TREE_H)).astype(np.float32)
    sig = 20.0 / px_m
    num = cv2.GaussianBlur(tops * meas, (0, 0), sig)
    den = cv2.GaussianBlur(meas, (0, 0), sig)
    can = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sig)
    hmap = num / np.maximum(den, 1e-3)
    hmap[den / np.maximum(can, 1e-3) < MEASURED_SHARE] = np.nan
    return hmap


def current_rule(tops, mask, px_m, stock):
    """Today's zones -> polygons -> median height per polygon."""
    hmap = smoothed_map(tops, mask, px_m)
    edges = (SHRUB_H, 7.5, 10.5, 13.5, 16.5, 20.0)
    zone = np.digitize(np.nan_to_num(hmap), edges) + 1
    zone[np.isnan(hmap)] = 99
    zone = np.where(mask, zone, 0).astype(np.float32)
    zone = merge_small(zone, mask, px_m, 2000.0)
    # polygons = connected components of a zone; one median each
    out = np.full(mask.shape, np.nan, np.float32)
    for z in np.unique(zone[mask]):
        n, cc = cv2.connectedComponents((mask & (zone == z)).astype(np.uint8), connectivity=8)
        flat = cc.ravel()
        hv = hmap.ravel()
        ok = ~np.isnan(hv) & (flat > 0)
        order = np.argsort(flat[ok], kind="stable")
        labs, vals = flat[ok][order], hv[ok][order]
        starts = np.searchsorted(labs, np.arange(n))
        ends = np.searchsorted(labs, np.arange(n), side="right")
        med = np.full(n, np.nan, np.float32)
        for i in range(1, n):
            if ends[i] > starts[i]:
                med[i] = np.median(vals[starts[i]:ends[i]])
        sel = cc > 0
        out[sel] = med[cc[sel]]
    return out


# ── scoring ──────────────────────────────────────────────────────────────────

def score(planted, measured, truth_tops, mask, px_m):
    """Error of planted canopy height against the crown tops of run B, where B measures trees."""
    ev = mask & (truth_tops >= MIN_TREE_H)
    truth = cv2.GaussianBlur(truth_tops, (0, 0), 2.0 / px_m)
    e_band = float(np.abs(planted[ev] - truth[ev]).mean())
    known = ev & ~np.isnan(measured)
    e_cont = float(np.abs(measured[known] - truth[known]).mean()) if known.any() else np.nan
    # patches: connected same-height areas (what X-Plane plants as one polygon)
    sizes = []
    edges = 0
    for v in np.unique(planted[mask]):
        part = (mask & (planted == v)).astype(np.uint8)
        n, _cc, st, _ = cv2.connectedComponentsWithStats(part, connectivity=8)
        sizes.extend(st[1:, cv2.CC_STAT_AREA].tolist())
        edges += int((cv2.morphologyEx(part, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0).sum())
    sizes = np.asarray(sizes, np.float64) * px_m * px_m
    canopy = mask.sum() * px_m * px_m
    big = sizes[sizes >= 10000].sum() / canopy
    vals, cnts = np.unique(planted[mask], return_counts=True)
    p = cnts / cnts.sum()
    return dict(e_band=e_band, e_cont=e_cont, patches=len(sizes),
                patch_ha=float(np.average(sizes, weights=sizes) / 1e4),
                big_share=float(big), entropy=float(-(p * np.log2(p)).sum()),
                edge_km=edges * px_m / 2000.0)


def run_methods(key, sizes_m, rng_seed=7, min_patch=(600.0,), methods=("grid", "voronoi", "warped", "watershed")):
    s, tex_m = load_sample(key)
    z = np.load(os.path.join(SCRATCH, "maps", key + ".npz"))
    a, b = z["a"].astype(np.float32), z["b"].astype(np.float32)
    size_px = int(round(tex_m / GRID_M))
    px_m = tex_m / size_px
    a = cv2.resize(a, (size_px, size_px), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (size_px, size_px), interpolation=cv2.INTER_AREA)
    mask, stock = tree_mask_and_stock(key, size_px)
    ta, tb = crown_tops(a, px_m), crown_tops(b, px_m)
    rows = []
    cur = current_rule(ta, mask, px_m, stock)
    fallback = smoothed_map(ta, mask, px_m)
    planted = band_of(cur, stock)
    rows.append(("current", 0, 2000, score(planted, cur, tb, mask, px_m), planted, None))
    for size_m in sizes_m:
        sp = max(2, int(round(size_m / px_m)))
        for name in methods:
            rng = np.random.default_rng(rng_seed)
            if name == "grid":
                lab = grid(mask.shape, sp, rng)
            elif name == "voronoi":
                lab = voronoi(mask.shape, sp, rng)
            else:   # the shipping code (O4_Forest_Stands)
                lab = STANDS.stand_labels(ta, mask, mask & (ta >= MIN_TREE_H), px_m, size_m, name,
                                          seed=rng_seed).astype(np.int32)
            meas, slab = stand_heights(lab, ta, mask, px_m, min_patch_m2=1)
            gap = np.isnan(meas) & mask
            meas[gap] = fallback[gap]   # as the overlay: undecided stands keep the smoothed map
            for mp in min_patch:
                planted = band_of(meas, stock)
                if mp:
                    planted = merge_small(planted, mask, px_m, mp)
                rows.append((name, size_m, mp, score(planted, meas, tb, mask, px_m), planted, slab))
    return dict(key=key, px_m=px_m, stock=stock, mask=mask, ta=ta, tb=tb,
                canopy_ha=mask.sum() * px_m * px_m / 1e4,
                measured_share=float((mask & (ta >= MIN_TREE_H)).sum() / max(1, mask.sum())),
                rows=rows)


def print_rows(res):
    print(f"\n{res['key']}: canopy {res['canopy_ha']:.0f} ha, measured {100 * res['measured_share']:.0f} %, "
          f"stock {res['stock']} m")
    print(f"  {'method':10s} {'size':>5s} {'minp':>5s} {'err band':>8s} {'err cont':>8s} "
          f"{'patches':>7s} {'patch ha':>8s} {'>=1ha':>6s} {'entropy':>7s} {'edge km':>7s}")
    for name, size, mp, sc, _p, _l in res["rows"]:
        print(f"  {name:10s} {size:5.0f} {mp:5.0f} {sc['e_band']:8.2f} {sc['e_cont']:8.2f} "
              f"{sc['patches']:7d} {sc['patch_ha']:8.2f} {100 * sc['big_share']:5.0f}% "
              f"{sc['entropy']:7.2f} {sc['edge_km']:7.1f}")


# ── pictures ─────────────────────────────────────────────────────────────────

HEIGHT_COLOURS = {   # planted canopy height -> BGR
    3.0: (60, 220, 255),     # shrubs: yellow
    6.0: (80, 255, 160),     # light green
    9.0: (40, 200, 60),
    12.0: (30, 140, 20),
    15.0: (90, 90, 0),       # teal
}
STOCK_COLOUR = (170, 40, 120)   # full-height X-Plane forest: purple


def _colour(planted, mask):
    out = np.zeros(planted.shape + (3,), np.uint8)
    for v in np.unique(planted[mask]):
        out[mask & (planted == v)] = HEIGHT_COLOURS.get(float(v), STOCK_COLOUR)
    return out


def best_window(mask, ta, win_px):
    meas = (mask & (ta >= MIN_TREE_H)).astype(np.float32)
    s = cv2.boxFilter(meas, -1, (win_px, win_px), normalize=False)
    h = win_px // 2
    s[:h], s[-h:], s[:, :h], s[:, -h:] = 0, 0, 0, 0
    y, x = np.unravel_index(int(s.argmax()), s.shape)
    return y - h, x - h


def panel(tex, planted, slab, mask, y0, x0, win, out_px, title):
    """Texture crop with planted heights tinted on, patch edges black, stand edges white."""
    crop = lambda a: a[y0:y0 + win, x0:x0 + win]
    t = cv2.resize(tex, (out_px, out_px), interpolation=cv2.INTER_AREA)
    if planted is None:
        img = t
    else:
        m = cv2.resize(crop(mask).astype(np.uint8), (out_px, out_px), interpolation=cv2.INTER_NEAREST) > 0
        p = cv2.resize(crop(planted), (out_px, out_px), interpolation=cv2.INTER_NEAREST)
        col = _colour(p, m)
        img = t.copy()
        img[m] = (0.45 * t[m] + 0.55 * col[m]).astype(np.uint8)
        if slab is not None:
            sl = cv2.resize(crop(slab).astype(np.float32), (out_px, out_px), interpolation=cv2.INTER_NEAREST)
            e = (cv2.morphologyEx(sl, cv2.MORPH_GRADIENT, np.ones((2, 2), np.uint8)) > 0) & m
            img[e] = (0.5 * img[e] + 0.5 * 255).astype(np.uint8)
        pe = (cv2.morphologyEx(np.where(m, p, -1).astype(np.float32), cv2.MORPH_GRADIENT,
                               np.ones((3, 3), np.uint8)) > 0) & m
        img[pe] = (20, 20, 20)
    img = img.copy()
    cv2.rectangle(img, (0, 0), (out_px, 34), (0, 0, 0), -1)
    cv2.putText(img, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def legend(width):
    img = np.zeros((40, width, 3), np.uint8)
    x = 10
    items = [("shrubs 3 m", HEIGHT_COLOURS[3.0])] + [
        (f"{int(h)} m", c) for h, c in HEIGHT_COLOURS.items() if h > 3] + [("full height", STOCK_COLOUR)]
    for label, c in items:
        cv2.rectangle(img, (x, 10), (x + 24, 30), c, -1)
        cv2.putText(img, label, (x + 30, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        x += 60 + 12 * len(label)
    cv2.putText(img, "black = height changes   white = stand edges", (x + 20, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def sheet(res, picks, out_path, win_m=600.0, out_px=900):
    s, _tex_m = load_sample(res["key"])
    tex = cv2.imread(os.path.join(SET, s["file"]))
    size_px = res["mask"].shape[0]
    tex = cv2.resize(tex, (size_px, size_px), interpolation=cv2.INTER_AREA)
    win = int(round(win_m / res["px_m"]))
    y0, x0 = best_window(res["mask"], res["ta"], win)
    rows = {(n, sz, mp): (p, l) for n, sz, mp, _sc, p, l in res["rows"]}
    panels = [panel(tex, None, None, res["mask"], y0, x0, win, out_px, f"{res['key']} texture ({win_m:.0f} m)")]
    for pick in picks:
        p, l = rows[pick]
        title = "today" if pick[0] == "current" else f"{pick[0]} {pick[1]:.0f} m"
        panels.append(panel(tex, p, l, res["mask"], y0, x0, win, out_px, title))
    top = np.hstack(panels)
    cv2.imwrite(out_path, np.vstack([top, legend(top.shape[1])]), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="*")
    ap.add_argument("--sizes", nargs="*", type=float, default=[15, 25, 40, 60, 90, 140])
    ap.add_argument("--minp", nargs="*", type=float, default=[0.0, 600.0, 2000.0])
    ap.add_argument("--methods", nargs="*", default=["warped", "watershed"])
    ap.add_argument("--json", default=os.path.join(SCRATCH, "scores.json"))
    args = ap.parse_args()
    keys = args.keys or sorted(f[:-4] for f in os.listdir(os.path.join(SCRATCH, "maps")) if f.endswith(".npz"))
    allrows = {}
    for key in keys:
        res = run_methods(key, args.sizes, min_patch=args.minp, methods=args.methods)
        print_rows(res)
        allrows[key] = dict(canopy_ha=res["canopy_ha"], measured=res["measured_share"],
                            rows=[(n, s, mp, sc) for n, s, mp, sc, _p, _l in res["rows"]])
        sys.stdout.flush()
    json.dump(allrows, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
