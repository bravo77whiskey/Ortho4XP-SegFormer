"""Reference-anchored colour normalisation, prototype.

Idea: a cloud-free, globally balanced satellite mosaic (Sentinel-2 cloudless)
is radiometrically consistent everywhere, across climates, tiles and
providers.  It is far too coarse to use as imagery, but at ~40 m it tells us
what colour every place *should* have relative to every other place.  So:

  1. Reduce the ortho and the reference to a common coarse grid (~35-40 m,
     the ZL12 pixel), where resolution/MTF and small misregistration stop
     mattering.
  2. In Lab, fit a local low-DOF map  ortho -> reference  at every point:
         L'  = g * L + o          (contrast / exposure / veil)
         ab' = s * ab + t         (saturation / colour cast, no hue rotation)
     with Gaussian windows, robust reweighting (clouds, changed land, glint),
     and shrinkage toward the region-wide fit where a window has no signal.
  3. Compose with one *constant* "look" map that takes reference space to a
     pleasing ortho appearance.  Constant => cannot create seams.

The local map depends only on data near a point and on the reference, never
on texture or tile membership, so neighbouring textures (and other ZLs, and
other providers) converge to the same answer.
"""
import json
import os

import cv2
import numpy as np
from PIL import Image

DATA = os.environ.get("O4_REVIEW_DATA", r"E:\Ortho4XP_Review")
SET = os.path.join(DATA, "color", "set")
TEX = 4096
COARSE = 16          # ortho px per coarse px at ZL16 (ZL16 -> ZL12 grid)


class P:
    mode = "dt"            # "tv" (small windows + Huber-TV) or "dt" (edge-aware windows)
    tv_win_m = 150.0       # Gaussian sigma of the small fit windows
    tv_lambda = 30.0       # TV weight, Delta-E * px: discs of radius r lose 2*lam/r contrast
    tv_eps = 2.0           # Huber threshold (Delta-E per px): gentler gradients stay smooth
    tv_iters = 300
    tv_split = True        # gains from win_L_m windows, offsets from tv_win_m
    tv_attribute = True    # TV cheap where the ortho has the edge, dear where only the reference does
    tv_attr_sigma = 4.0    # px, scale at which edges are attributed to ortho or reference
    tv_lam_ref = 6.0       # lambda multiplier for reference-only edges (and edge-free areas)
    tv_conf_full = 0.6     # robust window weight that counts as full confidence
    tv_L_typ = 50.0        # typical L*, converts the gain into Delta-E
    tv_C_typ = 15.0        # typical chroma, converts the saturation gain
    edge_aware = True      # windows stop at provider patch boundaries
    ref_haze = True        # exclude haze/thin cloud found in the reference itself
    ref_haze_rise = 8.0    # reference L* this far above the ortho prediction ...
    ref_haze_range = 8.0   # L* step in the (smoothed) ortho that stops the prediction windows
    ref_haze_chroma = 0.75 # ... and chroma below this fraction of the prediction
    red_cap = 4.0          # a* units the local fit may go redder than the regional one (0 = off)
    dt_attribute = False   # ... weighted by how much each edge is the ortho's (mode "dt")
    dt_attr_floor = 0.1    # weight left on edges that are not the ortho's
    straight_only = False  # ... but only at long straight ones
    line_min_m = 1000.0    # shortest straight boundary run
    line_width = 3         # px, rasterised boundary thickness
    line_smooth_px = 1.5   # Gaussian on the difference image before edge finding
    line_gap_px = 6        # Hough: largest gap bridged along a line
    line_probe_px = 3      # how far either side of a line the step is measured
    line_min_step = 0.8    # median step across the line, in range units
    line_consistency = 0.8 # fraction of the line showing the step's sign
    range_L = 5.0          # L* step in (ortho - reference) that counts as a boundary
    range_ab = 5.0
    win_L_m = 500.0        # Gaussian sigma of the windows for g, o
    win_ab_m = 1000.0      # ... for s, t
    shrink_L = 6.0         # L* std below which the local gain is unreliable
    shrink_ab = 4.0        # chroma std below which the local sat is unreliable
    irls = 3
    resid_L = 10.0         # Cauchy scale of the residual, L* units
    resid_ab = 8.0
    water = True
    water_L = 35.0         # reference L* below which a smooth cell may be water
    water_rdet = 2.5       # reference L* std (~35 m) below which it is smooth
    water_odet = 4.0       # ortho L* std inside the cell: water is smooth even at 2 m, canopy is not
    fill_m = 1500.0        # Gaussian sigma of the fallback fit for excluded cells
    fill_support = 0.3     # usable window fraction below which the fallback wins
    g_range = (0.55, 1.8)  # relative to the region-wide gain
    s_range = (0.5, 1.6)


# Parameter overrides for experiments: REFNORM_SET="tv_lambda=12,tv_eps=1.5"
for _kv in filter(None, os.environ.get("REFNORM_SET", "").split(",")):
    _k, _v = _kv.split("=")
    _t = type(getattr(P, _k))
    setattr(P, _k, _v.lower() in ("1", "true", "yes") if _t is bool else _t(_v))


def srgb_to_lab(rgb_u8):
    return cv2.cvtColor(rgb_u8.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def lab_to_srgb(lab):
    rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    return np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


def missing_mask(rgb):
    """Pixels of 256-px provider tiles Ortho4XP could not download (filled white).

    Tested per tile so that saturated cloud tops, which are also pure white,
    are not mistaken for missing imagery.
    """
    h, w = rgb.shape[:2]
    white = rgb.min(-1) >= 254
    blk = white.reshape(h // 256, 256, w // 256, 256).mean(axis=(1, 3)) > 0.995
    return np.repeat(np.repeat(blk, 256, 0), 256, 1)


def gblur(x, sigma_px):
    return cv2.GaussianBlur(x, (0, 0), sigma_px, borderType=cv2.BORDER_REFLECT)


################################################################################
class Block(object):
    """n x n ZL16 textures of one provider plus the reference around them."""

    def __init__(self, key, prov):
        d = os.path.join(SET, key)
        self.meta = meta = json.load(open(os.path.join(d, "meta.json")))
        self.key, self.prov = key, prov
        if "n" in meta:
            self.n, x0, y0 = meta["n"], meta["x0"], meta["y0"]
            self.paths = {(i, j): os.path.join(d, prov, f"{y0 + 16 * j}_{x0 + 16 * i}.jpg")
                          for j in range(self.n) for i in range(self.n)}
        else:   # single-texture location
            self.n = 1
            self.paths = {(0, 0): os.path.join(d, f"{prov}16.jpg")}
        self.ref_path = os.path.join(d, "ref.png")
        self.px_m = 156543.034 * np.cos(np.radians(meta["lat"])) / 2 ** 16 * COARSE
        # Coarse-grid margin around the block (reference has one ZL14 tile).
        self.margin = meta["ref_origin_px"][0] * meta["ref_scale"] // COARSE

    def texture(self, ij):
        return np.asarray(Image.open(self.paths[ij]).convert("RGB"))

    def coarse(self):
        """Lab ortho, Lab reference and validity on the coarse grid (with margin).

        Also sets self.odet / self.rdet: fine detail (local L* std) of the ortho
        inside each coarse cell and of the reference around it, which is what
        separates smooth surfaces (water, cloud) from textured ones.
        """
        c = TEX // COARSE
        ref = np.asarray(Image.open(self.ref_path).convert("RGB"))
        k = COARSE // self.meta["ref_scale"]
        H = ref.shape[0] // k
        R = srgb_to_lab(cv2.resize(ref, (H, H), interpolation=cv2.INTER_AREA))
        rl = srgb_to_lab(ref)[..., 0]
        rm = cv2.blur(rl, (2 * k + 1, 2 * k + 1))
        rv = np.maximum(cv2.blur(rl * rl, (2 * k + 1, 2 * k + 1)) - rm * rm, 0)
        self.rdet = np.sqrt(cv2.resize(rv, (H, H), interpolation=cv2.INTER_AREA))
        O = np.zeros((H, H, 3), np.float32)
        valid = np.zeros((H, H), bool)
        self.odet = np.zeros((H, H), np.float32)
        m = self.margin
        for (i, j), _ in self.paths.items():
            t = self.texture((i, j))
            miss = cv2.resize(missing_mask(t).astype(np.float32), (c, c),
                              interpolation=cv2.INTER_AREA) > 0.01
            sl = (slice(m + j * c, m + (j + 1) * c), slice(m + i * c, m + (i + 1) * c))
            O[sl] = srgb_to_lab(cv2.resize(t, (c, c), interpolation=cv2.INTER_AREA))
            valid[sl] = ~miss
            # L* proxy from luma, std per cell (cheap; Lab of 16 Mpx is slow).
            lum = (t.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)) * (100 / 255.0)
            self.odet[sl] = lum.reshape(c, COARSE, c, COARSE).std(axis=(1, 3))
        return O, R, valid


################################################################################
class DomainTransform(object):
    """Edge-aware smoothing (Gastal & Oliveira 2011, recursive filter).

    Linear in the signal for a fixed guide, so weighted means computed as
    DT(w*x) / DT(w) are proper edge-aware normalised convolutions.
    """

    def __init__(self, guide, range_sigmas, iterations=3, boundary=None):
        # Domain distance between horizontal / vertical neighbours, in units of
        # spatial pixels: 1 + sum_k |dI_k| / range_k (sigma_s factored out below).
        # With `boundary`, only steps on those pixels count; elsewhere the
        # filter is a plain (isotropic) smoother.
        gx = np.abs(np.diff(guide, axis=1)) / np.asarray(range_sigmas, np.float32)
        gy = np.abs(np.diff(guide, axis=0)) / np.asarray(range_sigmas, np.float32)
        self.dx = gx.sum(-1)        # (H, W-1), multiplied by sigma_s at use
        self.dy = gy.sum(-1)        # (H-1, W)
        if boundary is not None and boundary.dtype == bool:
            self.dx *= boundary[:, :-1] | boundary[:, 1:]
            self.dy *= boundary[:-1] | boundary[1:]
        elif boundary is not None:      # soft weights: steps count this much
            self.dx *= np.maximum(boundary[:, :-1], boundary[:, 1:])
            self.dy *= np.maximum(boundary[:-1], boundary[1:])
        self.iterations = iterations

    def __call__(self, x, sigma_s):
        x = x.astype(np.float32).copy()
        squeeze = x.ndim == 2
        if squeeze:
            x = x[..., None]
        n = self.iterations
        dx = (1.0 + sigma_s * self.dx)[..., None]
        dy = (1.0 + sigma_s * self.dy)[..., None]
        for i in range(n):
            sh = sigma_s * np.sqrt(3.0) * 2 ** (n - i - 1) / np.sqrt(4 ** n - 1)
            a = np.exp(-np.sqrt(2.0) / sh)
            wx, wy = (a ** dx).astype(np.float32), (a ** dy).astype(np.float32)
            for c in range(1, x.shape[1]):              # left -> right
                x[:, c] += wx[:, c - 1] * (x[:, c - 1] - x[:, c])
            for c in range(x.shape[1] - 2, -1, -1):     # right -> left
                x[:, c] += wx[:, c] * (x[:, c + 1] - x[:, c])
            for r in range(1, x.shape[0]):              # top -> bottom
                x[r] += wy[r - 1] * (x[r - 1] - x[r])
            for r in range(x.shape[0] - 2, -1, -1):     # bottom -> top
                x[r] += wy[r] * (x[r + 1] - x[r])
        return x[..., 0] if squeeze else x


def boundary_guide(O, R, valid):
    """Robustly smoothed ortho - reference difference.

    Terrain edges appear in both images and cancel; what is left are steps in
    the provider's radiometry (patch / acquisition boundaries).  Two 5x5
    medians remove isolated content mismatches (new buildings, boats, roofs).
    """
    d = np.where(valid[..., None], O - R, 0.0).astype(np.float32)
    for _ in range(2):
        d = np.stack([cv2.medianBlur(np.ascontiguousarray(d[..., k]), 5) for k in range(3)], -1)
    return d


def straight_boundaries(guide, valid, px_m, p=P, exclude=None, debug=None):
    """Long straight steps in the ortho - reference difference.

    Provider mosaic boundaries are straight (acquisition strips, rectangles,
    polygons, tile staircases).  Cloud edges, reference artefacts and real
    land-cover change are blobby, and letting the windows stop at those makes
    the fit copy them, so only straight runs of at least line_min_m count, and
    each candidate must be a *step*: one side consistently differs from the
    other along its whole length (a road is a line, not a step).
    """
    rng = np.asarray((p.range_L, p.range_ab, p.range_ab), np.float32)
    g2 = gblur(guide / rng, p.line_smooth_px)
    ok = cv2.erode(valid.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    if exclude is not None:
        ok &= ~cv2.dilate(exclude.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    edges = np.zeros(guide.shape[:2], np.uint8)
    for k in range(3):
        u8 = np.clip(g2[..., k] * 24.0 + 128.0, 0, 255).astype(np.uint8)
        edges |= cv2.Canny(u8, 50, 90)
    edges[~ok] = 0
    Lmin = p.line_min_m / px_m
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=int(Lmin * 0.5),
                            minLineLength=Lmin, maxLineGap=p.line_gap_px)
    out = np.zeros(guide.shape[:2], np.uint8)
    kept = []
    H, W = out.shape
    if lines is not None:
        off = p.line_probe_px
        for x1, y1, x2, y2 in lines[:, 0].astype(np.float32):
            L = np.hypot(x2 - x1, y2 - y1)
            n = np.array([-(y2 - y1), x2 - x1], np.float32) / max(L, 1e-3)
            ts = np.linspace(0, 1, int(L) + 1)
            px_, py_ = x1 + ts * (x2 - x1), y1 + ts * (y2 - y1)
            ax, ay = np.clip(px_ + off * n[0], 0, W - 1), np.clip(py_ + off * n[1], 0, H - 1)
            bx, by = np.clip(px_ - off * n[0], 0, W - 1), np.clip(py_ - off * n[1], 0, H - 1)
            A = g2[ay.round().astype(int), ax.round().astype(int)]
            B = g2[by.round().astype(int), bx.round().astype(int)]
            inside = ok[ay.round().astype(int), ax.round().astype(int)] & ok[by.round().astype(int), bx.round().astype(int)]
            if inside.mean() < 0.8:
                continue
            d = (A - B)[inside]                        # (N, 3) in range units
            k = int(np.argmax(np.abs(np.median(d, 0))))
            step = np.median(d[:, k])
            consistent = np.mean(np.sign(d[:, k]) == np.sign(step))
            if abs(step) >= p.line_min_step and consistent >= p.line_consistency:
                kept.append((x1, y1, x2, y2))
                cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), 1, p.line_width)
    if debug is not None:
        debug["edges"], debug["lines"], debug["kept"] = edges, lines, kept
    return out.astype(bool)


def wmoments(x, w, sigma, smooth=None):
    """Weighted local mean and variance of x (H,W[,C]) with weights w (H,W)."""
    f = smooth or gblur
    sw = f(w, sigma) + 1e-6
    if x.ndim == 3:
        w, sw = w[..., None], sw[..., None]
    m = f(x * w, sigma) / sw
    v = f(x * x * w, sigma) / sw - m * m
    return m, np.maximum(v, 0.0)


def water_mask(R, rdet, odet, valid, p=P):
    """Smooth dark water in the reference that is also smooth in the ortho.

    Only used to *exclude* cells from the fit (they then take the correction
    of the surrounding land), so a false positive costs almost nothing: dense
    forest fails the ortho-detail test, bright smooth ground fails the dark
    test.  Production would use the tile's OSM / mesh water instead.
    """
    m = (R[..., 0] < p.water_L) & (rdet < p.water_rdet) & (odet < p.water_odet) & valid
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.dilate(m, np.ones((3, 3), np.uint8)).astype(bool)


def fit_fields(O, R, valid, px_m, p=P, exclude=None):
    """O, R: Lab on the same coarse grid.  Returns g, o, s, t fields and weights.

    `exclude` marks cells that must not inform the fit (water, cloud).  They,
    and anything else whose edge-aware window holds too little usable data,
    take the correction of a wide land-only Gaussian fit instead.
    """
    if p.mode == "tv":
        return fit_fields_tv(O, R, valid, px_m, p, exclude)
    base = valid & ~exclude if exclude is not None else valid
    base = base.astype(np.float32)
    sL, sab = p.win_L_m / px_m, p.win_ab_m / px_m
    w = base.copy()
    smooth = None
    if p.edge_aware:
        guide = boundary_guide(O, R, valid)
        bnd = None
        if p.dt_attribute:
            # Windows stop at the ortho's own edges (seams) but run through
            # edges only the reference has, so reference artefacts become
            # outliers inside a larger window instead of a window of their own.
            bnd = (p.dt_attr_floor + (1 - p.dt_attr_floor) * ortho_edge_share(O, R, p)).astype(np.float32)
        elif p.straight_only:
            bnd = straight_boundaries(guide, valid, px_m, p, exclude)
        smooth = DomainTransform(guide, (p.range_L, p.range_ab, p.range_ab), boundary=bnd)

    def fit(w, smooth, sL, sab):
        ww = w.sum() + 1e-6
        def gm(x): return (x * w).sum() / ww
        def gv(x): return gm(x * x) - gm(x) ** 2
        # Region-wide fit: the prior each window is shrunk toward.
        gL = np.sqrt(gv(R[..., 0]) / max(gv(O[..., 0]), 1e-6))
        gs = np.sqrt((gv(R[..., 1]) + gv(R[..., 2])) / max(gv(O[..., 1]) + gv(O[..., 2]), 1e-6))

        # Gain from the ratio of spreads (reduced-major-axis): unlike a least
        # squares slope it is not biased low by misregistration or noise.
        mO, vO = wmoments(O[..., 0], w, sL, smooth)
        mR, vR = wmoments(R[..., 0], w, sL, smooth)
        lam = vO / (vO + p.shrink_L ** 2)
        g = np.sqrt(vR / np.maximum(vO, 1e-6))
        g = np.clip(lam * g + (1 - lam) * gL, gL * p.g_range[0], gL * p.g_range[1])
        o = mR - g * mO

        mOa, vOa = wmoments(O[..., 1:], w, sab, smooth)
        mRa, vRa = wmoments(R[..., 1:], w, sab, smooth)
        vOc, vRc = vOa.sum(-1), vRa.sum(-1)
        lam = vOc / (vOc + p.shrink_ab ** 2)
        s = np.sqrt(vRc / np.maximum(vOc, 1e-6))
        s = np.clip(lam * s + (1 - lam) * gs, gs * p.s_range[0], gs * p.s_range[1])
        t = mRa - s[..., None] * mOa
        return g, o, s, t

    if p.ref_haze:
        # Haze or thin cloud in the *reference* (common in the tropics even in
        # a cloudless composite): the reference is much brighter and greyer
        # than a plain wide fit predicts from the ortho.  Edge-aware windows
        # would lock onto it and paint it into clear ortho, so keep it out.
        # The prediction must not cross provider seams (a wide plain fit
        # mixes a washed-out patch with its neighbour and then "finds" haze
        # all over the neighbour), but must cross reference haze, so its
        # windows stop only at edges the *ortho* has.
        og = O.copy()
        for _ in range(2):
            og = np.stack([cv2.medianBlur(np.ascontiguousarray(og[..., k]), 5) for k in range(3)], -1)
        dt_o = DomainTransform(og, (p.ref_haze_range, p.ref_haze_range * 1.5, p.ref_haze_range * 1.5))
        mid = p.win_ab_m / px_m
        rg, ro, rs, rt = fit(base, dt_o, mid, mid)
        predL = rg * O[..., 0] + ro
        predC = np.linalg.norm(rs[..., None] * O[..., 1:] + rt, axis=-1)
        Rc = np.linalg.norm(R[..., 1:], axis=-1)
        haze = (R[..., 0] - predL > p.ref_haze_rise) & (Rc < p.ref_haze_chroma * predC + 1.0)
        haze = cv2.dilate(haze.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        base = base * ~haze
        w = base.copy()
    for it in range(p.irls + 1):
        g, o, s, t = fit(w, smooth, sL, sab)
        if it == p.irls:
            break
        eL = (g * O[..., 0] + o - R[..., 0]) / p.resid_L
        eab = np.linalg.norm(s[..., None] * O[..., 1:] + t - R[..., 1:], axis=-1) / p.resid_ab
        w = (base / (1.0 + eL ** 2 + eab ** 2)).astype(np.float32)

    if exclude is not None and exclude.any():
        # Usable fraction of each edge-aware window; where it is small the
        # window sits inside an excluded region (or is cut off by its edge).
        f = smooth or gblur
        support = f(base, sL) / (f(np.ones_like(base), sL) + 1e-6)
        fg, fo, fs, ft = fit(w, None, p.fill_m / px_m, p.fill_m / px_m)
        a = np.clip(support / p.fill_support, 0.0, 1.0)
        g, o, s = a * g + (1 - a) * fg, a * o + (1 - a) * fo, a * s + (1 - a) * fs
        t = a[..., None] * t + (1 - a[..., None]) * ft

    if p.red_cap > 0:
        # Edge-aware windows can lock onto something only the reference has
        # (bare patches, artefacts) and paint it into the ortho: green forest
        # turning orange.  Limit how much redder than the regional correction
        # the local one may make an area.  One-sided on purpose: pulling red /
        # autumn foliage toward green (summer) and every other correction
        # stay free.
        rg, ro, rs, rt = fit(w, None, p.fill_m / px_m, p.fill_m / px_m)
        mOa, _ = wmoments(O[..., 1], base, sab)
        local = s * mOa + t[..., 0]
        regional = rs * mOa + rt[..., 0]
        t = t.copy()
        t[..., 0] -= np.maximum(local - regional - p.red_cap, 0.0)
    return g, o, s, t, w


################################################################################
def huber_tv(f, w, lam, eps, iters=300, init=None):
    """Weighted vectorial Huber-ROF:  min_u  sum w/2 |u - f|^2 + lam * H_eps(grad u).

    f: (H, W, C) data, w: (H, W) confidence (0 = no data, filled by the prior).
    Chambolle-Pock primal-dual.  Channels share one gradient norm so all the
    correction parameters jump at the same places.  The Huber norm keeps
    gentle ramps (haze gradients) smooth instead of terracing them, while
    steps stay sharp and regions too small for their perimeter are removed.
    """
    u = (f if init is None else init).copy()
    ub = u.copy()
    px = np.zeros_like(f)
    py = np.zeros_like(f)
    tau = sigma = 1.0 / np.sqrt(8.0)
    w = w[..., None].astype(np.float32)
    lam = np.broadcast_to(np.asarray(lam, np.float32), f.shape[:2])[..., None]
    for _ in range(iters):
        gx = np.zeros_like(ub); gy = np.zeros_like(ub)
        gx[:, :-1] = ub[:, 1:] - ub[:, :-1]
        gy[:-1] = ub[1:] - ub[:-1]
        px = (px + sigma * gx) / (1.0 + sigma * eps / lam)
        py = (py + sigma * gy) / (1.0 + sigma * eps / lam)
        nrm = np.sqrt((px * px + py * py).sum(-1, keepdims=True)) / lam
        nrm = np.maximum(nrm, 1.0)
        px /= nrm; py /= nrm
        div = np.zeros_like(px)
        div[:, :-1] += px[:, :-1]; div[:, 1:] -= px[:, :-1]
        div[:-1] += py[:-1]; div[1:] -= py[:-1]
        un = (u + tau * div + tau * w * f) / (1.0 + tau * w)
        ub = 2 * un - u
        u = un
    return u


def ortho_edge_share(O, R, p=P):
    """0..1: how much an edge here belongs to the ortho rather than the reference.

    A provider seam is an ortho edge the reference does not have (-> 1).  An
    edge only the reference has (artefact, land change since the ortho was
    taken) gives 0, and so do edges both share (terrain) and edge-free areas.
    Measured at ~140 m, where terrain texture averages out but steps survive.
    """
    def grad(x):
        x = gblur(x, p.tv_attr_sigma)
        gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx * gx + gy * gy).sum(-1)
    wch = np.array([1.0, 1.5, 1.5], np.float32)
    gO, gR = grad(O * wch), grad(R * wch)
    return np.clip((gO / (gO + gR + 0.5) - 0.5) / 0.3, 0, 1)


def fit_fields_tv(O, R, valid, px_m, p=P, exclude=None):
    """Small-window local fit, then Huber-TV denoising of the parameter field."""
    base = (valid & ~exclude) if exclude is not None else valid.copy()
    base = base.astype(np.float32)
    big, small = p.win_ab_m / px_m, p.tv_win_m / px_m

    def fit(w, sL, sab, s_off=None):
        """Gains from windows sL/sab; offsets from s_off windows (default same)."""
        ww = w.sum() + 1e-6
        def gm(x): return (x * w).sum() / ww
        def gv(x): return gm(x * x) - gm(x) ** 2
        gL = np.sqrt(gv(R[..., 0]) / max(gv(O[..., 0]), 1e-6))
        gs = np.sqrt((gv(R[..., 1]) + gv(R[..., 2])) / max(gv(O[..., 1]) + gv(O[..., 2]), 1e-6))
        mO, vO = wmoments(O[..., 0], w, sL)
        mR, vR = wmoments(R[..., 0], w, sL)
        lam = vO / (vO + p.shrink_L ** 2)
        g = np.sqrt(vR / np.maximum(vO, 1e-6))
        g = np.clip(lam * g + (1 - lam) * gL, gL * p.g_range[0], gL * p.g_range[1])
        if s_off is not None:
            mO, _ = wmoments(O[..., 0], w, s_off)
            mR, _ = wmoments(R[..., 0], w, s_off)
        o = mR - g * mO
        mOa, vOa = wmoments(O[..., 1:], w, sab)
        mRa, vRa = wmoments(R[..., 1:], w, sab)
        vOc, vRc = vOa.sum(-1), vRa.sum(-1)
        lam = vOc / (vOc + p.shrink_ab ** 2)
        s = np.sqrt(vRc / np.maximum(vOc, 1e-6))
        s = np.clip(lam * s + (1 - lam) * gs, gs * p.s_range[0], gs * p.s_range[1])
        if s_off is not None:
            mOa, _ = wmoments(O[..., 1:], w, s_off)
            mRa, _ = wmoments(R[..., 1:], w, s_off)
        t = mRa - s[..., None] * mOa
        return g, o, s, t

    cheap = ortho_edge_share(O, R, p)
    lam_map = p.tv_lambda
    if p.tv_attribute:
        lam_map = p.tv_lambda * (p.tv_lam_ref + (1.0 - p.tv_lam_ref) * cheap)
    medium = p.win_L_m / px_m

    # Robust weights.  Residuals are taken against the TV-cleaned field, not a
    # wide fit: a whole provider patch would be an "outlier" to a wide fit and
    # get dropped, whereas TV keeps large patches (they fit, keep their weight)
    # and removes small blobs (they don't, and lose it).  The first pass is
    # unweighted: seeding from a wide fit would drop the patches for good.
    sc = np.array([p.tv_L_typ, 1.0, p.tv_C_typ, 1.0, 1.0], np.float32)

    def residual_weights(g, o, s, t):
        eL = (g * O[..., 0] + o - R[..., 0]) / p.resid_L
        eab = np.linalg.norm(s[..., None] * O[..., 1:] + t - R[..., 1:], axis=-1) / p.resid_ab
        return (base / (1.0 + eL ** 2 + eab ** 2)).astype(np.float32)

    w = base.copy()
    u = None
    for it in range(p.irls):
        g, o, s, t = fit(w, medium, medium, small) if p.tv_split else fit(w, small, small)
        # Confidence: how much robust weight the small window actually held.
        conf = gblur(w, small) / (gblur(np.ones_like(w), small) + 1e-6)
        conf = np.clip(conf / p.tv_conf_full, 0, 1) ** 2
        f = np.stack([g, o, s, t[..., 0], t[..., 1]], -1) * sc
        u = huber_tv(f, conf, lam_map, p.tv_eps,
                     p.tv_iters if u is None else p.tv_iters // 3, init=u)
        uu = u / sc
        if it < p.irls - 1:
            w = residual_weights(uu[..., 0], uu[..., 1], uu[..., 2], uu[..., 3:5])
    u = u / sc
    return u[..., 0], u[..., 1], u[..., 2], u[..., 3:5], w


def compose_look(fields, look):
    """look = (G, O, S, T): constant affine applied after the local map."""
    g, o, s, t = fields
    G, O_, S, T = look
    return G * g, G * o + O_, S * s, S * t + np.asarray(T, np.float32)


def apply(lab, g, o, s, t):
    out = np.empty_like(lab)
    out[..., 0] = g * lab[..., 0] + o
    out[..., 1:] = s[..., None] * lab[..., 1:] + t
    return out


def texture_field(field, block, ij):
    """Bilinear upsample of a coarse field to one full-res texture.

    The crop carries one coarse cell of padding and is sliced so pixel centres
    line up exactly with the coarse grid, so two neighbouring textures read the
    same interpolated values along their shared edge.
    """
    c = TEX // COARSE
    m = block.margin
    i, j = ij
    y, x = m + j * c - 1, m + i * c - 1
    crop = field[y:y + c + 2, x:x + c + 2].astype(np.float32)
    up = cv2.resize(crop, ((c + 2) * COARSE, (c + 2) * COARSE), interpolation=cv2.INTER_LINEAR)
    return up[COARSE:COARSE + TEX, COARSE:COARSE + TEX]


def correct_texture(rgb, fields, block, ij):
    g, o, s, t = (texture_field(f, block, ij) for f in fields)
    out = lab_to_srgb(apply(srgb_to_lab(rgb), g, o, s, t))
    out[missing_mask(rgb)] = 255
    return out


################################################################################
# Clouds
#
# A cloud is where the ortho is much brighter than the reference says the
# ground is AND has lost the ground's colour.  Both are measured against the
# clear-sky ortho the local fit predicts from the reference, so bright roofs,
# sand and snow (bright in the reference too) do not trigger it.  The result is
# an opacity alpha in [0, 1] per ~10 m pixel, under the usual haze model
#     observed = ground * (1 - alpha) + cloud_colour * alpha.

MED = 4              # ortho px per medium px at ZL16 (= reference px at ZL14)


class CP:
    cloud_L = 92.0         # L* of an opaque cloud top
    min_rise = 12.0        # L* above the clear-sky prediction before it counts
    min_chroma_ref = 6.0   # below this the ground is grey and chroma says nothing
    alpha_floor = 0.12     # alphas below this are noise
    open_px = 2            # morphological opening (medium px) to drop specks
    feather_px = 1.5       # Gaussian feather of the final alpha (medium px)
    ref_cloud_L = 72.0     # reference pixels this bright and grey are cloud themselves
    ref_cloud_C = 9.0
    coarse_alpha = 0.25    # coarse cells with mean alpha above this leave the fit
    cand_rise = 28.0       # fit-free candidate: L* above the reference ...
    cand_chroma = 12.0     # ... and chroma below this


def reference_cloud(Rm):
    C = np.hypot(Rm[..., 1], Rm[..., 2])
    m = (Rm[..., 0] > CP.ref_cloud_L) & (C < CP.ref_cloud_C)
    return cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)


def medium_mosaic(block):
    """Lab ortho mosaic at reference resolution, aligned with ref.png, plus validity."""
    ref = np.asarray(Image.open(block.ref_path).convert("RGB"))
    Rm = srgb_to_lab(ref)
    H = ref.shape[0]
    Om = np.zeros((H, H, 3), np.float32)
    vm = np.zeros((H, H), bool)
    c = TEX // MED
    m = block.margin * COARSE // MED
    for (i, j) in block.paths:
        t = block.texture((i, j))
        sl = (slice(m + j * c, m + (j + 1) * c), slice(m + i * c, m + (i + 1) * c))
        Om[sl] = srgb_to_lab(cv2.resize(t, (c, c), interpolation=cv2.INTER_AREA))
        vm[sl] = cv2.resize((~missing_mask(t)).astype(np.uint8), (c, c),
                            interpolation=cv2.INTER_NEAREST) > 0
    return Om, Rm, vm


def cloud_candidates(Om, Rm, vm):
    """Fit-free first guess: far brighter than the reference and nearly grey.

    Needed because a local fit run with clouds included simply learns to map
    cloud-white to the ground colour, after which nothing looks like cloud.
    """
    C_obs = np.linalg.norm(Om[..., 1:], axis=-1)
    m = (Om[..., 0] - Rm[..., 0] > CP.cand_rise) & (C_obs < CP.cand_chroma) & vm
    return m | (reference_cloud(Rm) & vm)


def cloud_alpha(Om, Rm, vm, fields):
    """Cloud opacity at reference resolution from the *local* (pre-look) fields."""
    k = COARSE // MED
    H = Om.shape[0]
    up = lambda f: cv2.resize(f.astype(np.float32), (H, H), interpolation=cv2.INTER_LINEAR)
    g, o, s, t = up(fields[0]), up(fields[1]), up(fields[2]), up(fields[3])
    # Clear-sky ortho predicted from the reference (inverse of the local map).
    L_exp = (Rm[..., 0] - o) / np.maximum(g, 1e-3)
    ab_exp = (Rm[..., 1:] - t) / np.maximum(s, 1e-3)[..., None]
    C_exp = np.linalg.norm(ab_exp, axis=-1)
    C_obs = np.linalg.norm(Om[..., 1:], axis=-1)
    rise = Om[..., 0] - L_exp
    a_L = np.clip((rise - CP.min_rise) / np.maximum(CP.cloud_L - L_exp - CP.min_rise, 5.0), 0, 1)
    a_C = np.where(C_exp > CP.min_chroma_ref, np.clip(1 - C_obs / np.maximum(C_exp, 1e-3), 0, 1), 1.0)
    a = np.minimum(a_L, a_C)
    # Where the reference is itself cloudy the prediction is useless: fall back
    # to an absolute test (very bright, nearly grey).
    rc = reference_cloud(Rm)
    a_abs = np.clip((Om[..., 0] - 70.0) / 18.0, 0, 1) * (C_obs < 8.0)
    a = np.where(rc, a_abs, a)
    a = np.where(a < CP.alpha_floor, 0.0, a) * vm
    ker = np.ones((2 * CP.open_px + 1,) * 2, np.uint8)
    keep = cv2.morphologyEx((a > 0).astype(np.uint8), cv2.MORPH_OPEN, ker).astype(bool)
    a = gblur((a * keep).astype(np.float32), CP.feather_px)
    return np.clip(a, 0, 1), rc


def coarse_from_medium(x, block):
    k = COARSE // MED
    H = x.shape[0] // k
    return cv2.resize(x.astype(np.float32), (H, H), interpolation=cv2.INTER_AREA)


def texture_medium_field(field_m, block, ij):
    """Upsample a reference-resolution field to one full-res texture (aligned)."""
    c = TEX // MED
    m = block.margin * COARSE // MED
    i, j = ij
    y, x = m + j * c - 1, m + i * c - 1
    crop = field_m[y:y + c + 2, x:x + c + 2].astype(np.float32)
    up = cv2.resize(crop, ((c + 2) * MED, (c + 2) * MED), interpolation=cv2.INTER_LINEAR)
    return up[MED:MED + TEX, MED:MED + TEX]


def dehaze(rgb, alpha, cloud_rgb=(236, 238, 240), max_alpha=0.7):
    """Invert observed = J (1 - a) + A a in linear light, for thin cloud only."""
    lin = (rgb.astype(np.float32) / 255.0) ** 2.2
    A = (np.asarray(cloud_rgb, np.float32) / 255.0) ** 2.2
    a = np.minimum(alpha, max_alpha)[..., None]
    J = (lin - A * a) / (1 - a)
    return np.clip(np.clip(J, 0, 1) ** (1 / 2.2) * 255 + 0.5, 0, 255).astype(np.uint8)
