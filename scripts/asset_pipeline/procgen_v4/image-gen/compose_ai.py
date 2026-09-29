"""Assemble codex img2img chunks back into v4 atlas pages.

    python image-gen/compose_ai.py --pkg <v4 pkg> --out <new pkg> --pages europe_res_a
                            [--size 4096|2048]

For every group in ``image-gen/work/<page>/jobs.json``:

1. resize each output to its input's size and register it onto the input
   (ECC affine on heavily blurred luminance, so brick or tile texture does
   not drive the fit);
2. snap features one by one: every window (glass in the bake's class map)
   or rooftop item is template-matched in the AI image, and a smooth
   displacement field moves it back onto its bake position, because the
   OBJ UVs expect windows exactly there.  The residual is reported;
3. even out brightness/colour between the chunks of one group (the group's
   mean, per channel), so stitched strips do not step at chunk joins;
4. stitch neighbouring chunks with a minimum-error vertical cut through
   their 256 px overlap (wrapping, so the strip still tiles in u);
5. paste the strip rows into a copy of the original page.  Strips without
   AI output (trim) keep the Cycles bake.

Facade normal maps are copied from the original page (after snapping the
relief still matches).  Roof normals are derived from the AI albedo,
because the model's tile courses do not land on the bake's.  Pitched roofs
skip the global fit: periodic tiles make it ambiguous.  ``--size 2048``
halves both pages (UVs are normalised, the pow2 strip layout still holds).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.dirname(HERE)
if V4 not in sys.path:
    sys.path.insert(0, V4)

from layout import PAGE_PX, page_layout  # noqa: E402
from regions import FLAT_ROOF_KINDS  # noqa: E402
from strips import CLS_GLASS, CLS_OBJECT  # noqa: E402

WORK = os.path.join(HERE, "work")


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def _lum(img, sigma=1.5):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return cv2.GaussianBlur(g, (0, 0), sigma)


def _grad(img, sigma=2.0):
    g = _lum(img, sigma)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return np.sqrt(gx * gx + gy * gy)


def register(ai, ref, sigma=4.0):
    """Warp ``ai`` (already ref-sized) onto ``ref``: (img, score, warp)."""
    a, r = _lum(ai, sigma), _lum(ref, sigma)
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    try:
        # coarse-to-fine: a half-size pass first gets big shifts right
        s = 0.5
        a2 = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        r2 = cv2.resize(r, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        _, w2 = cv2.findTransformECC(r2, a2, warp.copy(), cv2.MOTION_AFFINE,
                                     crit, None, 5)
        w2[:, 2] /= s
        score, warp = cv2.findTransformECC(r, a, w2, cv2.MOTION_AFFINE, crit,
                                           None, 5)
    except cv2.error:
        return ai, float("nan"), np.eye(2, 3, dtype=np.float32)
    h, w = ref.shape[:2]
    out = cv2.warpAffine(ai, warp, (w, h),
                         flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_REFLECT)
    return out, score, warp


def features(cls, feat_cls, min_px=150):
    """Bounding boxes (x0, y0, x1, y1) of windows / rooftop items."""
    m = np.isin(cls, feat_cls).astype(np.uint8)
    # merge the panes of one window into one feature
    m = cv2.dilate(m, np.ones((9, 9), np.uint8))
    n, _lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a >= min_px:
            out.append((x, y, x + w, y + h))
    return out


def match_shifts(ai, ref, boxes, margin=10, search=28, min_score=0.35):
    """Per feature: where it sits in ``ai`` relative to ``ref``."""
    ga, gr = _grad(ai), _grad(ref)
    H, W = gr.shape
    res = []
    for x0, y0, x1, y1 in boxes:
        tx0, ty0 = max(0, x0 - margin), max(0, y0 - margin)
        tx1, ty1 = min(W, x1 + margin), min(H, y1 + margin)
        tpl = gr[ty0:ty1, tx0:tx1]
        sx0, sy0 = max(0, tx0 - search), max(0, ty0 - search)
        sx1, sy1 = min(W, tx1 + search), min(H, ty1 + search)
        win = ga[sy0:sy1, sx0:sx1]
        if (win.shape[0] <= tpl.shape[0] or win.shape[1] <= tpl.shape[1]
                or tpl.std() < 1e-4):
            continue
        r = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(r)
        if score < min_score:
            continue
        res.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0,
                    float(loc[0] + sx0 - tx0), float(loc[1] + sy0 - ty0),
                    float(score)))
    return res


def snap(ai, ref, cls, feat_cls, sigma=110.0, search=28):
    """Warp ``ai`` so each matched feature lands on its ``ref`` position:
    a Gaussian-weighted field of the feature shifts, fading to no shift away
    from features.  Returns (img, n_matched, median err before, after)."""
    boxes = features(cls, feat_cls)
    m = match_shifts(ai, ref, boxes, search=search)
    if not m:
        return ai, 0, 0.0, 0.0
    H, W = ref.shape[:2]
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    num_x = np.zeros((H, W), np.float32)
    num_y = np.zeros((H, W), np.float32)
    den = np.full((H, W), 0.02, np.float32)      # zero-shift prior
    for cx, cy, dx, dy, sc in m:
        w = np.exp(-((xs - cx) ** 2 + ((ys - cy) * 1.5) ** 2)
                   / (2 * sigma * sigma)) * sc
        num_x += w * dx
        num_y += w * dy
        den += w
    out = cv2.remap(ai, xs + num_x / den, ys + num_y / den, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT)
    before = float(np.median([np.hypot(d[2], d[3]) for d in m]))
    m2 = match_shifts(out, ref, boxes, search=search)
    after = float(np.median([np.hypot(d[2], d[3]) for d in m2])) if m2 else 0.0
    return out, len(m), before, after


def derived_normal(albedo, strength=2.5, sigma_hp=6.0):
    """Tangent normal RG (0..255) from albedo luminance used as height."""
    g = cv2.cvtColor(albedo, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    h = g - cv2.GaussianBlur(g, (0, 0), sigma_hp)
    h = cv2.GaussianBlur(h, (0, 0), 0.8)
    gx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3) * strength
    gy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3) * strength
    nz = 1.0 / np.sqrt(gx * gx + gy * gy + 1.0)
    # image rows run down, texture t runs up: flip the y component
    nx, ny = -gx * nz, gy * nz
    return ((np.stack([nx, ny], -1) * 0.5 + 0.5).clip(0, 1) * 255.0 + 0.5)


def min_cut(a, b):
    """Column index per row where ``a`` (left) hands over to ``b``."""
    d = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).sum(-1)
    d = cv2.GaussianBlur(d, (0, 0), 2.0)
    h, w = d.shape
    acc = d.copy()
    back = np.zeros((h, w), np.int8)
    for y in range(1, h):
        prev = acc[y - 1]
        cand = np.stack([np.r_[np.inf, prev[:-1]], prev,
                         np.r_[prev[1:], np.inf]])
        k = cand.argmin(0)
        acc[y] += cand[k, np.arange(w)]
        back[y] = k - 1
    x = int(acc[-1].argmin())
    cut = np.zeros(h, np.int32)
    for y in range(h - 1, -1, -1):
        cut[y] = x
        x = int(np.clip(x + back[y, x], 0, w - 1))
    return cut


def stitch(chunks, core, ovl, feather=4):
    """chunks: [H, core+2*ovl, 3] crops, chunk i starting at u = i*core-ovl.
    Returns the wrapped strip [H, n*core, 3]."""
    n = len(chunks)
    h = chunks[0].shape[0]
    W = n * core
    out = np.zeros((h, W, 3), np.float32)
    for i, c in enumerate(chunks):
        out[:, i * core:(i + 1) * core] = c[:, ovl:ovl + core]
    xs = np.arange(2 * ovl)[None, :]
    for i in range(n):
        left, right = chunks[i], chunks[(i + 1) % n]
        a = left[:, core:core + 2 * ovl].astype(np.float32)
        b = right[:, 0:2 * ovl].astype(np.float32)
        cut = min_cut(a, b)[:, None]
        t = np.clip((xs - cut + feather) / (2.0 * feather), 0, 1)[..., None]
        u0 = (i + 1) * core - ovl
        out[:, np.arange(u0, u0 + 2 * ovl) % W] = a * (1 - t) + b * t
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def split_bands(ai, n, band_h, width):
    """Cut a stacked output at its black separator bars (the model moves
    and resizes them) and return ``n`` bands of exactly band_h x width, or
    None when the bars cannot be found."""
    lum = cv2.cvtColor(ai, cv2.COLOR_RGB2GRAY).astype(np.float32).mean(1)
    dark = lum < 28
    runs, y0 = [], None
    for y, d in enumerate(np.r_[dark, True]):
        if not d and y0 is None:
            y0 = y
        elif d and y0 is not None:
            runs.append((y0, y))
            y0 = None
    runs = [r for r in runs if r[1] - r[0] > 0.4 * ai.shape[0] / n]
    if len(runs) != n:
        return None
    out = []
    for a, b in runs:
        # a couple of rows of bar anti-aliasing hug each band edge
        a2, b2 = a + 2 if a else a, b - 2 if b < ai.shape[0] else b
        out.append(cv2.resize(ai[a2:b2], (width, band_h),
                              interpolation=cv2.INTER_AREA))
    return out


def level_joins(chunks, core, ovl):
    """Remove low-frequency tone steps at chunk joins: at each join the two
    sides meet at their mean (per row, per channel) and the correction
    ramps linearly to zero across each chunk."""
    n = len(chunks)
    f = [c.astype(np.float32) for c in chunks]
    h, w = f[0].shape[:2]
    deltas = []
    for i in range(n):
        a = f[i][:, core:core + 2 * ovl].mean(1)
        b = f[(i + 1) % n][:, 0:2 * ovl].mean(1)
        d = cv2.GaussianBlur((b - a)[:, None, :], (0, 0), 0.1,
                             sigmaY=h / 8.0)[:, 0, :]
        deltas.append(d)
    ramp = (np.arange(w, dtype=np.float32) / (w - 1))[None, :, None]
    out = []
    for i in range(n):
        corr = ramp * deltas[i][:, None, :] / 2.0 \
            - (1 - ramp) * deltas[i - 1][:, None, :] / 2.0
        out.append(np.clip(f[i] + corr, 0, 255).astype(np.uint8))
    return out


def match_group(chunks):
    """Equalise per-channel mean/std across a group's chunks."""
    arr = [c.astype(np.float32) for c in chunks]
    means = np.array([c.reshape(-1, 3).mean(0) for c in arr])
    stds = np.array([c.reshape(-1, 3).std(0) for c in arr])
    gm, gs = means.mean(0), stds.mean(0)
    return [np.clip((c - m) * (gs / np.maximum(s, 1e-3)) + gm, 0, 255)
            .astype(np.uint8) for c, m, s in zip(arr, means, stds)]


def compose_group(wdir, g, core, ovl, report):
    names = []
    for c in g["chunks"]:
        if c["name"] not in names:
            names.append(c["name"])
    roof = g["type"] == "roof"
    pitched = (roof and not g.get("stacked")
               and g.get("roof_kind") not in FLAT_ROOF_KINDS)
    hh = sum(r[2] for r in g["rows"])
    if g.get("stacked"):
        nm = names[0]
        op = os.path.join(wdir, "out", nm + ".png")
        if not os.path.isfile(op):
            return None
        inp = load_rgb(os.path.join(wdir, "in", nm + ".png"))
        ai = load_rgb(op)
        bands = split_bands(ai, len(g["chunks"]), hh, inp.shape[1])
        found = bands is not None
        if not found:
            ai_r = cv2.resize(ai, inp.shape[1::-1],
                              interpolation=cv2.INTER_AREA)
            bands = [ai_r[c["y0"]:c["y0"] + hh] for c in g["chunks"]]
        report.append({"chunk": nm, "ai_size": list(ai.shape[1::-1]),
                       "bands_found": found})
        crops = level_joins(match_group(bands), core, ovl)
        strip = stitch(crops, core, ovl)
        return {name: strip[y0:y0 + px] for name, y0, px in g["rows"]}
    regs = {}
    for nm in names:
        inp = load_rgb(os.path.join(wdir, "in", nm + ".png"))
        op = os.path.join(wdir, "out", nm + ".png")
        if not os.path.isfile(op):
            return None
        ai = load_rgb(op)
        h, w = inp.shape[:2]
        ai_r = cv2.resize(ai, (w, h), interpolation=cv2.INTER_AREA)
        if pitched:
            reg, score, warp = ai_r, float("nan"), np.eye(2, 3)
        else:
            reg, score, warp = register(ai_r, inp)
            if not np.isfinite(score) or score < 0.5:
                reg, warp = ai_r, np.eye(2, 3)
        n_feat, e0, e1 = 0, 0.0, 0.0
        cls_p = os.path.join(wdir, "in", nm + "_cls.png")
        if os.path.isfile(cls_p):
            cls = np.asarray(Image.open(cls_p))
            reg, n_feat, e0, e1 = snap(
                reg, inp, cls, (CLS_OBJECT,) if roof else (CLS_GLASS,),
                search=64 if roof else 28)
        report.append({"chunk": nm, "ai_size": list(ai.shape[1::-1]),
                       "ecc": round(float(score), 3),
                       "shift_px": [round(float(warp[0, 2]), 1),
                                    round(float(warp[1, 2]), 1)],
                       "scale": [round(float(warp[0, 0]), 3),
                                 round(float(warp[1, 1]), 3)],
                       "features": n_feat,
                       "feat_err_px": [round(e0, 1), round(e1, 1)]})
        regs[nm] = reg
    crops = match_group([regs[c["name"]][c["y0"]:c["y0"] + hh]
                         for c in g["chunks"]])
    strip = stitch(level_joins(crops, core, ovl), core, ovl)
    return {name: strip[y0:y0 + px] for name, y0, px in g["rows"]}


def compose_page(pkg, out, page, size, report_all):
    wdir = os.path.join(WORK, page)
    with open(os.path.join(wdir, "jobs.json"), encoding="utf-8") as fh:
        jobs = json.load(fh)
    lay = page_layout(jobs["kind"])
    albedo = np.array(Image.open(os.path.join(
        pkg, "textures", f"o4v4_{page}.png")).convert("RGB"))
    normal = np.array(Image.open(os.path.join(
        pkg, "textures", f"o4v4_{page}_nml.png")).convert("RGBA"))
    report = []
    done = []
    for g in jobs["groups"]:
        rows = compose_group(wdir, g, jobs["core"], jobs["ovl"], report)
        if rows is None:
            print(f"  {page}/{g['group']}: missing outputs, kept bake")
            continue
        for name, img in rows.items():
            r0, px = lay[name]["row0"], lay[name]["px"]
            albedo[r0:r0 + px] = img
            if g["type"] == "roof":
                normal[r0:r0 + px, :, :2] = derived_normal(img).astype(
                    np.uint8)
            done.append(name)
    tex = os.path.join(out, "textures")
    os.makedirs(tex, exist_ok=True)
    alb = Image.fromarray(albedo, "RGB")
    nml = Image.fromarray(normal, "RGBA")
    if size != PAGE_PX:
        alb = alb.resize((size, size), Image.BOX)
        nml = nml.resize((size, size), Image.BOX)
    alb.save(os.path.join(tex, f"o4v4_{page}.png"), compress_level=6)
    nml.save(os.path.join(tex, f"o4v4_{page}_nml.png"), compress_level=6)
    report_all[page] = {"replaced": done, "chunks": report}
    print(f"  {page}: replaced {len(done)} strips")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pkg", required=True, help="original v4 package")
    ap.add_argument("--out", required=True, help="package to write pages to")
    ap.add_argument("--pages", nargs="+", required=True)
    ap.add_argument("--size", type=int, default=PAGE_PX)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    rep = {}
    for page in args.pages:
        compose_page(args.pkg, args.out, page, args.size, rep)
    with open(os.path.join(args.out, "ai_compose_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    for page, r in rep.items():
        for c in r["chunks"]:
            print(page, c)


if __name__ == "__main__":
    main()
