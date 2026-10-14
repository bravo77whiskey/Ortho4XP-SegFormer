"""Reference-anchored colour correction of orthophoto textures (Step 3).

Providers ship imagery with washed-out city centres (Bing), oversaturation and
hard-edged acquisition patches (Google), seams between acquisition strips and
clouds.  A cloud-free satellite mosaic with the same colour balance
everywhere (EOX Sentinel-2 cloudless, ~10 m) is far too coarse to use as
imagery but tells what colour every place should have relative to every
other, across climates, tiles and providers.  So, per (tile, provider, ZL):

  1. Every texture is reduced to a ~38 m grid (the ZL12 pixel) and compared
     with the reference in Lab.
  2. A local map ortho -> reference is fitted everywhere:
         L'  = g L + o       (contrast, exposure, veil)
         ab' = s ab + t      (saturation, colour cast; no hue rotation)
     with edge-aware windows that stop at the provider's own patch edges, so
     a patch is corrected as a unit.  Water, cloud, and haze inside the
     reference itself are kept out of the fit and take their surroundings'
     correction.
  3. One constant "look" maps reference colours to a provider-like rendering.
  4. Clouds are found against the reference (a cloud is much brighter and
     greyer than the reference says the ground is, so bright sand and roofs
     do not count) and filled from a second provider, corrected the same way
     so the two blend without a visible join.  Tiles Ortho4XP could not
     download (white squares) are filled the same way.

The correction is smooth and only rescales pixels, so a texture keeps all its
detail at any zoom level.  Everything heavy runs on the GPU (PyTorch, which
the overlays already need); without CUDA it runs on the CPU, slowly.

This module runs in the overlay .venv python (torch is not bundled into the
exe) and imports nothing from Ortho4XP.  O4_Color_Build.py drives it from the
main process with two JSON jobs per tile build: "fit" (fits every group,
reports cloudy textures) and, once the main process has downloaded the fill
provider's textures for those, "apply" (corrects, fills and converts every
texture to DDS with nvcompress).
"""
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image

import O4_Progress as PROG

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - torch ships with the overlays
    torch = None

REF_PROVIDER = "EOX"     # Sentinel-2 cloudless (s2maps.eu, EOX IT Services)
REF_ZL = 14
COARSE_ZL = 12           # fit grid
MARGIN_TILES = 1         # ZL14 reference tiles of context around a group
FILL_PROVIDERS = {"BI": "GO2", "GO2": "BI"}
# reference space -> provider-like rendering, pooled over Bing and Google
# around the world (constant, so it cannot create seams)
LOOK = (0.86555, 15.6366, 1.34055, (2.83361, -9.64584))


class P:
    """Fit parameters (the version reviewed as "v13")."""
    mode = "dt"
    edge_aware = True
    range_L = 5.0           # L* step in (ortho - reference) that stops a window
    range_ab = 5.0
    win_L_m = 500.0         # Gaussian sigma of the windows for g, o
    win_ab_m = 1000.0       # ... for s, t
    shrink_L = 6.0
    shrink_ab = 4.0
    irls = 3
    resid_L = 10.0
    resid_ab = 8.0
    g_range = (0.55, 1.8)
    s_range = (0.5, 1.6)
    fill_m = 1500.0         # fallback fit for excluded cells
    fill_support = 0.3
    red_cap = 4.0           # max a* the local fit may go redder than the regional one
    ref_haze = True
    ref_haze_rise = 8.0
    ref_haze_range = 8.0
    ref_haze_chroma = 0.75
    water_L = 35.0
    water_rdet = 2.5
    water_odet = 4.0
    # Huber-TV (fill source and near clouds)
    tv_win_m = 150.0
    tv_lambda = 30.0
    tv_eps = 2.0
    tv_iters = 300
    tv_split = True
    tv_attribute = True
    tv_attr_sigma = 4.0
    tv_lam_ref = 6.0
    tv_conf_full = 0.6
    tv_L_typ = 50.0
    tv_C_typ = 15.0


class TV(P):
    mode = "tv"
    tv_lambda = 12.0


class Wide(P):
    """Plain wide fit, only used to predict clear sky for cloud detection."""
    edge_aware = False
    red_cap = 0.0
    win_L_m = 1000.0
    win_ab_m = 1500.0


class CP:
    cloud_L = 92.0
    min_rise = 12.0
    min_chroma_ref = 6.0
    alpha_floor = 0.12
    open_px = 2
    feather_px = 1.5
    ref_cloud_L = 72.0
    ref_cloud_C = 9.0
    ref_cloud_std = 4.0     # L* std over ~50 m: clouds are smooth
    ref_cloud_open = 11     # px (~110 m): narrower bright objects are not cloud
    coarse_alpha = 0.25
    cand_rise = 28.0
    cand_chroma = 12.0
    min_cloud_m2 = 20000.0  # smaller bright blobs are roofs, not cloud
    roof_ref_L = 65.0       # reference L* from which a bright spot may be a roof
    compact_max_m2 = 100000.0   # blobs this large always count as cloud
    anchor_min_m2 = 50000.0     # ... and ragged ones from this size
    anchor_solidity = 0.75
    anchor_reach_m = 1000.0     # smaller blobs count only this close to an anchor
    core_alpha = 0.6            # near-opaque cloud ...
    core_min_m2 = 10000.0       # ... of which an anchor needs at least this much
    near_m = 400.0
    fill_grow = 4
    fill_full = 0.2


_lock = threading.Lock()
_DEV = None


def available():
    return torch is not None


def _dev():
    global _DEV
    if _DEV is None:
        _DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEV


def _np(t):
    return t.detach().cpu().numpy()


################################################################################
# Colour
_M = np.array([[0.412453, 0.357580, 0.180423],
               [0.212671, 0.715160, 0.072169],
               [0.019334, 0.119193, 0.950227]], np.float32)
_WHITE = np.array([0.950456, 1.0, 1.088754], np.float32)


def srgb_to_lab(x):
    """(3, H, W) sRGB in [0, 1] -> Lab (OpenCV conventions, exact gamma)."""
    lin = torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    M = torch.from_numpy(_M / _WHITE[:, None]).to(x.device)
    xyz = torch.einsum("ij,jhw->ihw", M, lin)
    f = torch.where(xyz > 0.008856, xyz.clamp_min(0) ** (1 / 3), 7.787 * xyz + 16 / 116)
    L = torch.where(xyz[1] > 0.008856, 116 * f[1] - 16, 903.3 * xyz[1])
    return torch.stack([L, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def lab_to_srgb(lab):
    L, a, b = lab[0], lab[1], lab[2]
    fy = (L + 16) / 116
    Y = torch.where(L > 8.0, fy ** 3, L / 903.3)
    fy = torch.where(L > 8.0, fy, 7.787 * Y + 16 / 116)
    fx, fz = fy + a / 500, fy - b / 200
    inv = lambda t: torch.where(t > 0.206893, t ** 3, (t - 16 / 116) / 7.787)
    xyz = torch.stack([inv(fx), Y, inv(fz)]) * torch.from_numpy(_WHITE).to(lab.device)[:, None, None]
    lin = torch.einsum("ij,jhw->ihw", torch.from_numpy(np.linalg.inv(_M).astype(np.float32)).to(lab.device),
                       xyz).clamp(0, 1)
    return torch.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055).clamp(0, 1)


def _cv_lab(mean_u8):
    """Lab of coarse uint8 means via OpenCV (the fit was tuned on these values)."""
    a = _np(mean_u8.permute(1, 2, 0)).astype(np.float32) / 255.0
    return torch.from_numpy(cv2.cvtColor(a, cv2.COLOR_RGB2LAB).transpose(2, 0, 1).copy()).to(_dev())


################################################################################
# Filters on (C, H, W) tensors
def _sym_pad(x, p, dim):
    """OpenCV BORDER_REFLECT (fedcba|abcdef|fedcba), valid for any pad width."""
    if p == 0:
        return x
    n = x.shape[dim]
    i = torch.arange(-p, n + p, device=x.device) % (2 * n)
    i = torch.where(i >= n, 2 * n - 1 - i, i)
    return x.index_select(dim, i)


_kernels = {}


def gblur(x, sigma):
    """cv2.GaussianBlur(x, (0, 0), sigma, borderType=BORDER_REFLECT) per channel."""
    k = int(round(sigma * 8 + 1)) | 1
    p = k // 2
    key = (k, float(sigma), str(x.device))
    if key not in _kernels:
        _kernels[key] = torch.from_numpy(cv2.getGaussianKernel(k, sigma).astype(np.float32)[:, 0]).to(x.device)
    w = _kernels[key]
    C = x.shape[0]
    y = F.conv2d(_sym_pad(x, p, 2)[None], w.view(1, 1, 1, k).expand(C, 1, 1, k), groups=C)
    y = F.conv2d(_sym_pad(y[0], p, 1)[None], w.view(1, 1, k, 1).expand(C, 1, k, 1), groups=C)
    return y[0]


def _scan_last(B, A):
    """y[n] = A[n] y[n-1] + B[n] along the last dim (A[..., 0] = 0), log2(N) doubling steps."""
    k, N = 1, B.shape[-1]
    while k < N:
        Bn = B.clone()
        Bn[..., k:] += A[..., k:] * B[..., :-k]
        An = A.clone()
        An[..., k:] *= A[..., :-k]
        B, A = Bn, An
        k *= 2
    return B


class DomainTransform(object):
    """Edge-aware smoothing (Gastal & Oliveira 2011 recursive filter, 3 iterations).

    The recursion is evaluated as a parallel prefix scan.  Linear in the signal
    for a fixed guide, so DT(w x) / DT(w) is an edge-aware weighted mean.
    """

    def __init__(self, guide, range_sigmas, iterations=3):
        g = torch.as_tensor(guide, dtype=torch.float32, device=_dev())
        r = torch.as_tensor(range_sigmas, dtype=torch.float32, device=_dev())
        self.dx = ((g[:, 1:] - g[:, :-1]).abs() / r).sum(-1)
        self.dy = ((g[1:] - g[:-1]).abs() / r).sum(-1)
        self.iterations = iterations

    def __call__(self, x, sigma_s):
        n = self.iterations
        H, W = x.shape[1:]
        zc = torch.zeros(H, 1, device=x.device)
        zr = torch.zeros(W, 1, device=x.device)
        for i in range(n):
            sh = sigma_s * np.sqrt(3.0) * 2 ** (n - i - 1) / np.sqrt(4 ** n - 1)
            a = float(np.exp(-np.sqrt(2.0) / sh))
            wx = a ** (1.0 + sigma_s * self.dx)
            wy = (a ** (1.0 + sigma_s * self.dy)).T
            A = torch.cat([zc, wx], 1)
            x = _scan_last((1 - A) * x, A)
            A = torch.cat([zc, wx.flip(1)], 1)
            x = _scan_last((1 - A) * x.flip(2), A).flip(2)
            xt = x.transpose(1, 2).contiguous()
            A = torch.cat([zr, wy], 1)
            xt = _scan_last((1 - A) * xt, A)
            A = torch.cat([zr, wy.flip(1)], 1)
            xt = _scan_last((1 - A) * xt.flip(2), A).flip(2)
            x = xt.transpose(1, 2).contiguous()
        return x


def moments(f, sigma, w, xs):
    """Weighted local means and variances of every signal in xs, one filter call."""
    S = torch.stack([w] + [x * w for x in xs] + [x * x * w for x in xs])
    Fs = f(S, sigma)
    sw = Fs[0] + 1e-6
    n = len(xs)
    m = [Fs[1 + i] / sw for i in range(n)]
    v = [(Fs[1 + n + i] / sw - m[i] * m[i]).clamp_min(0) for i in range(n)]
    return m, v


def _median5x2(a):
    a = np.ascontiguousarray(a, np.float32)
    for _ in range(2):
        a = np.stack([cv2.medianBlur(np.ascontiguousarray(a[..., k]), 5) for k in range(3)], -1)
    return a


def _dilate(m, k):
    return F.max_pool2d(m.float()[None, None], k, stride=1, padding=k // 2)[0, 0]


def _erode(m, k):
    return -F.max_pool2d(-m.float()[None, None], k, stride=1, padding=k // 2)[0, 0]


################################################################################
# Fits.  O, R: (3, H, W) Lab tensors on the coarse grid; valid / exclude: bool numpy.
def _region_fit(O, R, w, p, f, sL, sab, s_off=None):
    OL, Oa, Ob = O
    RL, Ra, Rb = R
    ww = w.sum(dtype=torch.float64) + 1e-6
    gm = lambda x: (x * w).sum(dtype=torch.float64) / ww
    gv = lambda x: gm(x * x) - gm(x) ** 2
    gL = torch.sqrt(gv(RL) / torch.clamp(gv(OL), min=1e-6)).float()
    gs = torch.sqrt((gv(Ra) + gv(Rb)) / torch.clamp(gv(Oa) + gv(Ob), min=1e-6)).float()
    f = f or gblur
    if s_off is not None:                      # TV: spreads from sL, means from s_off
        sig = [OL, RL, Oa, Ob, Ra, Rb]
        _, (vO, vR, vOa, vOb, vRa, vRb) = moments(f, sL, w, sig)
        (mO, mR, mOa, mOb, mRa, mRb), _ = moments(f, s_off, w, sig)
    elif sL == sab:
        (mO, mR, mOa, mOb, mRa, mRb), (vO, vR, vOa, vOb, vRa, vRb) = \
            moments(f, sL, w, [OL, RL, Oa, Ob, Ra, Rb])
    else:
        (mO, mR), (vO, vR) = moments(f, sL, w, [OL, RL])
        (mOa, mOb, mRa, mRb), (vOa, vOb, vRa, vRb) = moments(f, sab, w, [Oa, Ob, Ra, Rb])
    lam = vO / (vO + p.shrink_L ** 2)
    g = torch.sqrt(vR / vO.clamp_min(1e-6))
    g = torch.clamp(lam * g + (1 - lam) * gL, min=float(gL * p.g_range[0]), max=float(gL * p.g_range[1]))
    o = mR - g * mO
    vOc, vRc = vOa + vOb, vRa + vRb
    lam = vOc / (vOc + p.shrink_ab ** 2)
    s = torch.sqrt(vRc / vOc.clamp_min(1e-6))
    s = torch.clamp(lam * s + (1 - lam) * gs, min=float(gs * p.s_range[0]), max=float(gs * p.s_range[1]))
    return g, o, s, torch.stack([mRa - s * mOa, mRb - s * mOb])


def fit_dt(O, R, valid, px_m, p=P, exclude=None):
    v = valid if exclude is None else valid & ~exclude
    base = torch.as_tensor(v, dtype=torch.float32, device=_dev())
    sL, sab = p.win_L_m / px_m, p.win_ab_m / px_m
    OL, Oa, Ob = O
    RL, Ra, Rb = R
    O_np, R_np = _np(O.permute(1, 2, 0)), _np(R.permute(1, 2, 0))
    smooth = None
    if p.edge_aware:
        guide = _median5x2(np.where(valid[..., None], O_np - R_np, 0.0))
        smooth = DomainTransform(guide, (p.range_L, p.range_ab, p.range_ab))
    if p.ref_haze:
        # Haze in the reference itself: much brighter and greyer than an
        # ortho-edge-aware prediction says.  Keep it out of the fit.
        dt_o = DomainTransform(_median5x2(O_np), (p.ref_haze_range, p.ref_haze_range * 1.5,
                                                  p.ref_haze_range * 1.5))
        mid = p.win_ab_m / px_m
        rg, ro, rs, rt = _region_fit(O, R, base, p, dt_o, mid, mid)
        predL = rg * OL + ro
        predC = torch.sqrt((rs * Oa + rt[0]) ** 2 + (rs * Ob + rt[1]) ** 2)
        Rc = torch.sqrt(Ra ** 2 + Rb ** 2)
        haze = (RL - predL > p.ref_haze_rise) & (Rc < p.ref_haze_chroma * predC + 1.0)
        haze = cv2.dilate(_np(haze).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        base = base * torch.as_tensor(~haze, dtype=torch.float32, device=_dev())
    w = base.clone()
    for it in range(p.irls + 1):
        g, o, s, t = _region_fit(O, R, w, p, smooth, sL, sab)
        if it == p.irls:
            break
        eL = (g * OL + o - RL) / p.resid_L
        eab = torch.sqrt((s * Oa + t[0] - Ra) ** 2 + (s * Ob + t[1] - Rb) ** 2) / p.resid_ab
        w = base / (1.0 + eL ** 2 + eab ** 2)
    wide = None
    if exclude is not None and exclude.any():
        f = smooth or gblur
        sup = f(torch.stack([base, torch.ones_like(base)]), sL)
        support = sup[0] / (sup[1] + 1e-6)
        wide = _region_fit(O, R, w, p, None, p.fill_m / px_m, p.fill_m / px_m)
        fg, fo, fs, ft = wide
        a = torch.clamp(support / p.fill_support, 0.0, 1.0)
        g, o, s = a * g + (1 - a) * fg, a * o + (1 - a) * fo, a * s + (1 - a) * fs
        t = a * t + (1 - a) * ft
    if p.red_cap > 0:
        rg, ro, rs, rt = wide if wide is not None else \
            _region_fit(O, R, w, p, None, p.fill_m / px_m, p.fill_m / px_m)
        (mOa,), _ = moments(gblur, sab, base, [Oa])
        local, regional = s * mOa + t[0], rs * mOa + rt[0]
        t = t.clone()
        t[0] -= torch.clamp(local - regional - p.red_cap, min=0.0)
    return g, o, s, t


def _huber_tv(f, w, lam, eps, iters, init=None):
    u = (f if init is None else init).clone()
    ub = u.clone()
    px = torch.zeros_like(f)
    py = torch.zeros_like(f)
    tau = sigma = 1.0 / np.sqrt(8.0)
    w = w[None]
    lam = lam[None] if torch.is_tensor(lam) else torch.tensor(lam, device=f.device).view(1, 1, 1)
    damp = 1.0 + sigma * eps / lam
    for _ in range(iters):
        gx = torch.zeros_like(ub)
        gy = torch.zeros_like(ub)
        gx[:, :, :-1] = ub[:, :, 1:] - ub[:, :, :-1]
        gy[:, :-1] = ub[:, 1:] - ub[:, :-1]
        px = (px + sigma * gx) / damp
        py = (py + sigma * gy) / damp
        nrm = (torch.sqrt((px * px + py * py).sum(0, keepdim=True)) / lam).clamp_min(1.0)
        px, py = px / nrm, py / nrm
        div = torch.zeros_like(px)
        div[:, :, :-1] += px[:, :, :-1]
        div[:, :, 1:] -= px[:, :, :-1]
        div[:, :-1] += py[:, :-1]
        div[:, 1:] -= py[:, :-1]
        un = (u + tau * div + tau * w * f) / (1.0 + tau * w)
        ub = 2 * un - u
        u = un
    return u


def _ortho_edge_share(O, R, p):
    """0..1: how much an edge belongs to the ortho (a seam) rather than the reference."""
    wch = torch.tensor([1.0, 1.5, 1.5], device=O.device)[:, None, None]
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=O.device)

    def grad(x):
        x = gblur(x, p.tv_attr_sigma)
        xp = F.pad(x[None], (1, 1, 1, 1), mode="reflect")
        gx = F.conv2d(xp, kx.expand(3, 1, 3, 3), groups=3)[0]
        gy = F.conv2d(xp, kx.T.contiguous().expand(3, 1, 3, 3), groups=3)[0]
        return torch.sqrt(gx * gx + gy * gy).sum(0)
    gO, gR = grad(O * wch), grad(R * wch)
    return torch.clamp((gO / (gO + gR + 0.5) - 0.5) / 0.3, 0, 1)


def fit_tv(O, R, valid, px_m, p=TV, exclude=None):
    v = valid if exclude is None else valid & ~exclude
    base = torch.as_tensor(v, dtype=torch.float32, device=_dev())
    small, medium = p.tv_win_m / px_m, p.win_L_m / px_m
    OL, Oa, Ob = O
    RL, Ra, Rb = R
    lam = p.tv_lambda
    if p.tv_attribute:
        lam = p.tv_lambda * (p.tv_lam_ref + (1.0 - p.tv_lam_ref) * _ortho_edge_share(O, R, p))
    sc = torch.tensor([p.tv_L_typ, 1.0, p.tv_C_typ, 1.0, 1.0], device=_dev())[:, None, None]
    ones = gblur(torch.ones_like(base)[None], small)[0] + 1e-6
    w = base.clone()
    u = None
    for it in range(p.irls):
        g, o, s, t = _region_fit(O, R, w, p, None, medium, medium, small) if p.tv_split else \
            _region_fit(O, R, w, p, None, small, small)
        conf = torch.clamp(gblur(w[None], small)[0] / ones / p.tv_conf_full, 0, 1) ** 2
        f = torch.cat([g[None], o[None], s[None], t]) * sc
        u = _huber_tv(f, conf, lam, p.tv_eps, p.tv_iters if u is None else p.tv_iters // 3, init=u)
        uu = u / sc
        if it < p.irls - 1:
            eL = (uu[0] * OL + uu[1] - RL) / p.resid_L
            eab = torch.sqrt((uu[2] * Oa + uu[3] - Ra) ** 2 + (uu[2] * Ob + uu[4] - Rb) ** 2) / p.resid_ab
            w = base / (1.0 + eL ** 2 + eab ** 2)
    u = u / sc
    return u[0], u[1], u[2], u[3:5]


def fit_any(O, R, valid, px_m, p, exclude=None):
    return (fit_tv if p.mode == "tv" else fit_dt)(O, R, valid, px_m, p, exclude)


################################################################################
# Texture-level helpers
def missing_mask(img):
    """(H, W) bool: whole 256-px provider tiles left white (not downloaded)."""
    white = (img.min(0).values >= 254).float()
    blk = F.avg_pool2d(white[None, None], 256)[0, 0] > 0.995
    return blk.repeat_interleave(256, 0).repeat_interleave(256, 1)


def _upsample_crop(field, y, x, n, k):
    """Bilinear upsample (x k) of field[:, y:y+n, x:x+n] with one cell of context.

    Pixel centres line up with the coarse grid exactly, so two neighbouring
    textures read identical values along their shared edge.
    """
    crop = field[:, y - 1:y + n + 1, x - 1:x + n + 1]
    up = F.interpolate(crop[None], scale_factor=k, mode="bilinear", align_corners=False)[0]
    return up[:, k:k + n * k, k:k + n * k]


def save_tga(arr, path):
    """Uncompressed 24/32-bit TGA, top-left origin (nvcompress honours it)."""
    h, w, c = arr.shape
    hdr = bytearray(18)
    hdr[2] = 2
    hdr[12:14] = w.to_bytes(2, "little")
    hdr[14:16] = h.to_bytes(2, "little")
    hdr[16] = 8 * c
    hdr[17] = 0x20 | (8 if c == 4 else 0)
    px = arr[..., [2, 1, 0, 3] if c == 4 else [2, 1, 0]]
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(np.ascontiguousarray(px).tobytes())


def _prefetch(loader, provider, keys, depth=4):
    """Decode textures on worker threads, at most `depth` ahead (16 Mpx each)."""
    from concurrent.futures import ThreadPoolExecutor
    keys = list(keys)
    with ThreadPoolExecutor(depth) as ex:
        futs = [ex.submit(loader, provider, *key) for key in keys[:depth]]
        for n, key in enumerate(keys):
            rgb = futs[n].result()
            futs[n] = None
            if n + depth < len(keys):
                futs.append(ex.submit(loader, provider, *keys[n + depth]))
            yield key, rgb


def reference_window(zl, x0, y0, nx, ny):
    """Reference window in ZL14 pixels for textures x0.. (nx x ny) at zl (>= 14)."""
    f14 = 2 ** (zl - REF_ZL)
    m = MARGIN_TILES * 256
    return (x0 * 256 // f14 - m, y0 * 256 // f14 - m,
            nx * 4096 // f14 + 2 * m, ny * 4096 // f14 + 2 * m)


def reference_textures(zl, x0, y0, nx, ny):
    """ZL14 orthogrid textures (multiples of 16 tiles) covering the reference window."""
    rx0, ry0, rw, rh = reference_window(zl, x0, y0, nx, ny)
    return [(tx, ty) for ty in range((ry0 // 4096) * 16, (ry0 + rh - 1) // 4096 * 16 + 1, 16)
            for tx in range((rx0 // 4096) * 16, (rx0 + rw - 1) // 4096 * 16 + 1, 16)]


################################################################################
class Group(object):
    """All textures of one (tile, provider, zoomlevel) and their fitted correction."""

    def __init__(self, lat, zl, provider, textures, loader):
        self.zl, self.provider = zl, provider
        self.textures = sorted(textures)
        self.loader = loader
        self.k = 2 ** (zl - COARSE_ZL)              # texture px per coarse cell
        self.km = 2 ** (zl - REF_ZL)                # texture px per medium (reference) px
        self.c = 4096 // self.k                     # coarse cells per texture
        self.cm = 4096 // self.km                   # medium px per texture
        xs = [t[0] for t in self.textures]
        ys = [t[1] for t in self.textures]
        self.x0, self.y0 = min(xs), min(ys)
        self.nx = (max(xs) - self.x0) // 16 + 1
        self.ny = (max(ys) - self.y0) // 16 + 1
        self.mt = MARGIN_TILES * 256                # medium margin (one ZL14 tile)
        self.m = self.mt // (2 ** (REF_ZL - COARSE_ZL))
        self.px_m = 156543.034 * np.cos(np.radians(lat)) / 2 ** COARSE_ZL
        self.fields = None
        self.alpha = {}                             # (tx, ty) -> medium cloud alpha (float16 numpy)
        self.medium = {}                            # (tx, ty) -> medium uint8 RGB (numpy)

    def ij(self, tx, ty):
        return (tx - self.x0) // 16, (ty - self.y0) // 16

    # --- reference ------------------------------------------------------
    def reference_window(self):
        """(x, y, w, h) of the reference mosaic in ZL14 pixels: group + one ZL14 tile.

        Worked in pixels, not tiles: a ZL19 texture is half a ZL14 tile, so a
        ZL19 group can start or end in the middle of one.
        """
        return reference_window(self.zl, self.x0, self.y0, self.nx, self.ny)

    def load_reference(self, fetch):
        """Reference mosaic at ZL14 covering the group plus margin, uint8 RGB."""
        rx0, ry0, rw, rh = self.reference_window()
        ref = np.zeros((rh, rw, 3), np.uint8)
        for tx, ty in reference_textures(self.zl, self.x0, self.y0, self.nx, self.ny):
            img = fetch(tx, ty)
            if img is None:
                raise RuntimeError(f"reference texture {ty}_{tx} unavailable")
            px, py = tx * 256, ty * 256                    # texture origin in ZL14 px
            sx0, sy0 = max(px, rx0), max(py, ry0)
            sx1, sy1 = min(px + 4096, rx0 + rw), min(py + 4096, ry0 + rh)
            ref[sy0 - ry0:sy1 - ry0, sx0 - rx0:sx1 - rx0] = img[sy0 - py:sy1 - py, sx0 - px:sx1 - px]
        self.ref = ref

    def reference_stats(self):
        """Coarse Lab of the reference and its local detail, in row strips."""
        k = 2 ** (REF_ZL - COARSE_ZL)
        H, W = self.ref.shape[:2]
        R = np.zeros((3, H // k, W // k), np.float32)
        rdet = np.zeros((H // k, W // k), np.float32)
        step = 2048
        for y in range(0, H, step):
            y0, y1 = max(0, y - k), min(H, y + step + k)
            r = torch.from_numpy(self.ref[y0:y1].copy()).to(_dev()).permute(2, 0, 1).float()
            core = r[:, y - y0:y - y0 + min(step, H - y)]
            R[:, y // k:(y + core.shape[1]) // k] = _np(_cv_lab(torch.round(F.avg_pool2d(core[None], k)[0])))
            rl = srgb_to_lab(r / 255.0)[0][None, None]
            pad = F.pad(rl, (k, k, 0, 0), mode="reflect")
            top = k if y0 == 0 else 0
            bot = k if y1 == H else 0
            if top or bot:
                pad = F.pad(pad, (0, 0, top, bot), mode="reflect")
            rm = F.avg_pool2d(pad, 2 * k + 1, stride=1)
            rv = (F.avg_pool2d(pad * pad, 2 * k + 1, stride=1) - rm * rm).clamp_min(0)
            off = (y - y0) - (k - top)
            rv = rv[:, :, off:off + core.shape[1]]
            rdet[y // k:(y + core.shape[1]) // k] = _np(torch.sqrt(F.avg_pool2d(rv, k)[0, 0]))
        return torch.from_numpy(R).to(_dev()), torch.from_numpy(rdet).to(_dev())

    # --- ortho ----------------------------------------------------------
    def ortho_stats(self):
        Hc, Wc = self.ny * self.c + 2 * self.m, self.nx * self.c + 2 * self.m
        O = torch.zeros(3, Hc, Wc, device=_dev())
        odet = torch.zeros(Hc, Wc, device=_dev())
        valid = torch.zeros(Hc, Wc, dtype=torch.bool, device=_dev())
        c, m, k = self.c, self.m, self.k
        for (tx, ty), rgb in _prefetch(self.loader, self.provider, self.textures):
            if rgb is None:
                continue
            img = torch.from_numpy(rgb).to(_dev()).permute(2, 0, 1)
            x = img.float()
            i, j = self.ij(tx, ty)
            sl = (slice(m + j * c, m + (j + 1) * c), slice(m + i * c, m + (i + 1) * c))
            O[:, sl[0], sl[1]] = _cv_lab(torch.round(F.avg_pool2d(x[None], k)[0]))
            lum = (x[0] * 0.299 + x[1] * 0.587 + x[2] * 0.114) * (100 / 255.0)
            odet[sl] = lum.view(c, k, c, k).permute(0, 2, 1, 3).reshape(c, c, -1).std(-1, unbiased=False)
            miss = missing_mask(img)
            valid[sl] = ~(F.avg_pool2d(miss.float()[None, None], k)[0, 0] > 0.01)
            self.medium[(tx, ty)] = (_np(torch.round(F.avg_pool2d(x[None], self.km)[0])).astype(np.uint8),
                                     _np(F.avg_pool2d((~miss).float()[None, None], self.km)[0, 0] > 0.5))
        return O, odet, valid

    # --- medium-resolution chunks for cloud detection -------------------
    def _paste_neighbours(self, maps, tx, ty, pad, n_ch=None, dtype=np.float32):
        """Canvas of one texture's map plus `pad` px taken from its neighbours."""
        cm = self.cm
        n = cm + 2 * pad
        out = np.zeros(((n_ch,) if n_ch else ()) + (n, n), dtype)
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                src = maps.get((tx + 16 * di, ty + 16 * dj))
                if src is None:
                    continue
                y0, x0 = pad + dj * cm, pad + di * cm
                ys0, xs0 = max(0, -y0), max(0, -x0)
                ys1, xs1 = min(cm, n - y0), min(cm, n - x0)
                if ys1 > ys0 and xs1 > xs0:
                    out[..., y0 + ys0:y0 + ys1, x0 + xs0:x0 + xs1] = src[..., ys0:ys1, xs0:xs1]
        return out

    def medium_chunk(self, tx, ty, pad):
        """Om (Lab), Rm (Lab), vm for one texture plus `pad` medium px of context."""
        cm = self.cm
        n = cm + 2 * pad
        rgb = self._paste_neighbours({k: v[0] for k, v in self.medium.items()}, tx, ty, pad, 3, np.uint8)
        vm = self._paste_neighbours({k: v[1] for k, v in self.medium.items()}, tx, ty, pad, None, bool)
        i, j = self.ij(tx, ty)
        ry, rx = self.mt + j * cm - pad, self.mt + i * cm - pad
        ref = self.ref[ry:ry + n, rx:rx + n]
        Om = srgb_to_lab(torch.from_numpy(rgb).to(_dev()).float() / 255.0)
        Rm = srgb_to_lab(torch.from_numpy(ref.copy()).to(_dev()).permute(2, 0, 1).float() / 255.0)
        return Om, Rm, torch.from_numpy(vm).to(_dev())

    def _coarse_of(self, values):
        """Assemble per-texture medium maps (pooled to coarse) into a coarse grid."""
        Hc, Wc = self.ny * self.c + 2 * self.m, self.nx * self.c + 2 * self.m
        out = np.zeros((Hc, Wc), np.float32)
        r = 2 ** (REF_ZL - COARSE_ZL)
        for (tx, ty), v in values.items():
            i, j = self.ij(tx, ty)
            out[self.m + j * self.c:self.m + (j + 1) * self.c, self.m + i * self.c:self.m + (i + 1) * self.c] = \
                _np(F.avg_pool2d(torch.as_tensor(v, device=_dev()).float()[None, None], r)[0, 0])
        return out

    def cloud_maps(self, fields, pad=8):
        """Cloud alpha per texture (medium res) given clear-sky fields; also coarse alpha / ref-cloud."""
        r = 2 ** (REF_ZL - COARSE_ZL)
        stack = torch.cat([fields[0][None], fields[1][None], fields[2][None], fields[3]])
        alphas, rcs = {}, {}
        for (tx, ty) in self.medium:
            Om, Rm, vm = self.medium_chunk(tx, ty, pad)
            i, j = self.ij(tx, ty)
            cy, cx = self.m + j * self.c - pad // r, self.m + i * self.c - pad // r
            f = _upsample_crop(stack, cy, cx, self.c + 2 * pad // r, r)
            a, rc = _cloud_alpha(Om, Rm, vm, f)
            alphas[(tx, ty)] = _np(a[pad:-pad, pad:-pad])
            rcs[(tx, ty)] = _np(rc[pad:-pad, pad:-pad])
        return self._drop_small_clouds(alphas), rcs

    def _drop_small_clouds(self, alphas):
        """Remove cloud blobs below min_cloud_m2, decided on the whole group.

        Deciding per texture would cut a cloud that straddles a texture edge
        differently on the two sides; on the group's coarse grid it is one blob.
        """
        r = 2 ** (REF_ZL - COARSE_ZL)
        Hc, Wc = self.ny * self.c + 2 * self.m, self.nx * self.c + 2 * self.m
        det = np.zeros((Hc, Wc), np.uint8)
        cnt = np.zeros((Hc, Wc), np.float32)
        core_cnt = np.zeros((Hc, Wc), np.float32)
        for (tx, ty), a in alphas.items():
            i, j = self.ij(tx, ty)
            sl = (slice(self.m + j * self.c, self.m + (j + 1) * self.c),
                  slice(self.m + i * self.c, self.m + (i + 1) * self.c))
            core = (a > 0.1).reshape(self.c, r, self.c, r)
            det[sl] = core.any(axis=(1, 3))
            cnt[sl] = core.sum(axis=(1, 3))
            core_cnt[sl] = (a > CP.core_alpha).reshape(self.c, r, self.c, r).sum(axis=(1, 3))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(det, connectivity=8)
        # Area = pixels that are clearly cloud, in m^2 (feathered edges and
        # whole coarse cells would inflate a roof into a hectare-sized blob).
        med_m2 = (self.px_m / r) ** 2
        area = np.bincount(lab.ravel(), weights=cnt.ravel(), minlength=n) * med_m2
        core_area = np.bincount(lab.ravel(), weights=core_cnt.ravel(), minlength=n) * med_m2
        # Large flat roofs (warehouses, distribution centres) pass a size test
        # but are compact and convex; clouds of that size are ragged.  Small
        # or compact blobs are only kept near a real ("anchor") cloud: small
        # puffs come in cloud fields, roofs in cities far from them.
        solid = np.zeros(n)
        for k in np.nonzero((area >= CP.min_cloud_m2) & (area < CP.compact_max_m2))[0]:
            ys, xs = np.nonzero(lab == k)
            hull = cv2.convexHull(np.stack([xs, ys], 1).astype(np.float32))
            solid[k] = len(xs) / max(cv2.contourArea(hull), 1.0)
        # A real cloud has an opaque core; airport concrete, big car parks
        # and container yards are only moderately brighter than predicted.
        anchor = ((area >= CP.compact_max_m2) | ((area >= CP.anchor_min_m2) & (solid < CP.anchor_solidity)))             & (core_area >= CP.core_min_m2)
        anchor[0] = False
        rad = max(1, int(round(CP.anchor_reach_m / self.px_m)))
        near = cv2.dilate(anchor[lab].astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1)))
        near_k = np.bincount(lab.ravel(), weights=near.ravel(), minlength=n) > 0
        small = ~anchor & ~near_k
        small[0] = False
        keep = cv2.dilate((det.astype(bool) & ~small[lab]).astype(np.uint8), np.ones((3, 3), np.uint8))
        out = {}
        for (tx, ty), a in alphas.items():
            i, j = self.ij(tx, ty)
            k = keep[self.m + j * self.c:self.m + (j + 1) * self.c, self.m + i * self.c:self.m + (i + 1) * self.c]
            out[(tx, ty)] = a * np.repeat(np.repeat(k, r, 0), r, 1)
        return out

    def candidates(self, pad=8):
        cands = {}
        for (tx, ty) in self.medium:
            Om, Rm, vm = self.medium_chunk(tx, ty, pad)
            cands[(tx, ty)] = _np(_cloud_candidates(Om, Rm, vm)[pad:-pad, pad:-pad])
        return self._coarse_of(cands) > 0.15

    # --- the fit ----------------------------------------------------------
    def fit(self, fetch_reference, clouds=True, mode=P):
        self.load_reference(fetch_reference)
        R, rdet = self.reference_stats()
        O, odet, valid_t = self.ortho_stats()
        valid = _np(valid_t)
        water = ((R[0] < P.water_L) & (rdet < P.water_rdet) & (odet < P.water_odet) & valid_t)
        water = cv2.morphologyEx(_np(water).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        water = cv2.dilate(water, np.ones((3, 3), np.uint8)).astype(bool)
        self.O, self.R, self.valid = O, R, valid
        excl = water
        if clouds:
            cand = self.candidates()
            cloud_c = cand
            for _ in range(2):
                wf = fit_any(O, R, valid, self.px_m, Wide, water | cloud_c)
                alphas, rcs = self.cloud_maps(wf)
                cloud_c = cand | (self._coarse_of(alphas) > CP.coarse_alpha) | (self._coarse_of(rcs) > 0.3)
            self.alpha = {key: a.astype(np.float16) for key, a in alphas.items()}
            # The fit-free candidates only seed the detection (they include
            # roofs and airport concrete); exclude what survived as cloud.
            excl = water | (self._coarse_of(alphas) > CP.coarse_alpha) | (self._coarse_of(rcs) > 0.3)
        fields = fit_any(O, R, valid, self.px_m, mode, excl)
        self.cloudy = clouds and any(a.max() > 0.02 for a in self.alpha.values())
        if self.cloudy and mode is P:
            # Near clouds the edge-aware fit copies whatever the reference has
            # under the haze; the TV fit does not.  Patch fixing matters least
            # there (it is filled anyway), so blend to TV near cloud.
            tv = fit_tv(O, R, valid, self.px_m, TV, excl)
            cl = torch.from_numpy(self._coarse_of(self.alpha)).to(_dev())
            near = torch.clamp(gblur((cl > 0.05).float()[None], CP.near_m / self.px_m)[0] * 4.0, 0, 1)
            fields = tuple(near * a + (1 - near) * b for a, b in zip(tv, fields))
        G, O_, S, T = LOOK
        g, o, s, t = fields
        T = torch.tensor(T, dtype=torch.float32, device=_dev())[:, None, None]
        self.fields = torch.cat([(G * g)[None], (G * o + O_)[None], (S * s)[None], S * t + T])
        self.has_missing = {key for key, (a, v) in self.medium.items() if not v.all()}
        self.ref = None          # free the big arrays
        self.O = self.R = None
        self.medium = {}

    def cloudy_textures(self):
        """Textures with cloud to fill (after the fill region is grown)."""
        return [key for key, a in self.alpha.items() if a.max() > 0.02]

    # --- applying ---------------------------------------------------------
    def correct(self, rgb, tx, ty):
        """(H, W, 3) uint8 numpy -> corrected uint8 tensor (3, H, W) on the device."""
        img = torch.from_numpy(rgb).to(_dev()).permute(2, 0, 1)
        i, j = self.ij(tx, ty)
        up = _upsample_crop(self.fields, self.m + j * self.c, self.m + i * self.c, self.c, self.k)
        lab = srgb_to_lab(img.float() / 255.0)
        out = torch.stack([up[0] * lab[0] + up[1], up[2] * lab[1] + up[3], up[2] * lab[2] + up[4]])
        res = (lab_to_srgb(out) * 255.0 + 0.5).clamp(0, 255).to(torch.uint8)
        miss = missing_mask(img)
        res[:, miss] = 255
        return res, miss

    def grown_alpha(self, tx, ty, pad=12):
        """Fill weight source: alpha dilated and feathered, with neighbour context."""
        a = self._paste_neighbours(self.alpha, tx, ty, pad)
        t = torch.from_numpy(a).to(_dev())
        k = 2 * CP.fill_grow + 1
        return torch.clamp(gblur(_dilate(t, k)[None], 1.5)[0] / CP.fill_full, 0, 1)


def _reference_cloud(Rm):
    """Cloud in the reference itself: bright, grey, smooth and large.

    Bright grey alone also matches concrete, roofs and highways, which are
    detailed and small; a cloud is neither.  The opening removes anything
    narrower than ~110 m.
    """
    L = Rm[0][None, None]
    mean = F.avg_pool2d(F.pad(L, (2, 2, 2, 2), mode="reflect"), 5, stride=1)
    sq = F.avg_pool2d(F.pad(L * L, (2, 2, 2, 2), mode="reflect"), 5, stride=1)
    std = torch.sqrt((sq - mean * mean).clamp_min(0))[0, 0]
    C = torch.sqrt(Rm[1] ** 2 + Rm[2] ** 2)
    m = (Rm[0] > CP.ref_cloud_L) & (C < CP.ref_cloud_C) & (std < CP.ref_cloud_std)
    m = _dilate(_erode(m, CP.ref_cloud_open) > 0, CP.ref_cloud_open) > 0
    return _dilate(m, 5) > 0


def _ref_bright(Rm):
    """Brightest reference L* within ~25 m: absorbs the reference's blur and a
    few metres of misregistration, so a white roof (bright in the reference
    too) is not mistaken for cloud.  Only used where the reference is that
    bright: under a cloud over vegetation, a road or a patch of bare soil
    nearby must not punch a hole in the cloud."""
    mx = F.max_pool2d(Rm[0][None, None], 5, stride=1, padding=2)[0, 0]
    return torch.where(mx > CP.roof_ref_L, mx, Rm[0])


def _cloud_candidates(Om, Rm, vm):
    C_obs = torch.sqrt(Om[1] ** 2 + Om[2] ** 2)
    m = (Om[0] - _ref_bright(Rm) > CP.cand_rise) & (C_obs < CP.cand_chroma) & vm
    return m | (_reference_cloud(Rm) & vm)


def _cloud_alpha(Om, Rm, vm, f):
    g, o, s, t = f[0], f[1], f[2], f[3:5]
    L_exp = (_ref_bright(Rm) - o) / g.clamp_min(1e-3)
    ab_exp = (Rm[1:] - t) / s.clamp_min(1e-3)
    C_exp = torch.sqrt((ab_exp ** 2).sum(0))
    C_obs = torch.sqrt(Om[1] ** 2 + Om[2] ** 2)
    rise = Om[0] - L_exp
    a_L = torch.clamp((rise - CP.min_rise) / torch.clamp(CP.cloud_L - L_exp - CP.min_rise, min=5.0), 0, 1)
    a_C = torch.where(C_exp > CP.min_chroma_ref, torch.clamp(1 - C_obs / C_exp.clamp_min(1e-3), 0, 1),
                      torch.ones_like(C_exp))
    a = torch.minimum(a_L, a_C)
    rc = _reference_cloud(Rm)
    a = torch.where(rc, torch.clamp((Om[0] - 70.0) / 18.0, 0, 1) * (C_obs < 8.0), a)
    a = torch.where(a < CP.alpha_floor, torch.zeros_like(a), a) * vm
    kk = 2 * CP.open_px + 1
    keep = _dilate(_erode(a > 0, kk) > 0, kk)
    a = gblur((a * keep)[None], CP.feather_px)[0]
    return a.clamp(0, 1), rc


################################################################################
# Jobs (run in the .venv worker)
def _log(msg):
    print("[Colour] " + msg, flush=True)


def _state_path(state_dir, gid, what):
    return os.path.join(state_dir, f"{gid}.{what}")


def _group_from_job(gj, which="textures"):
    src = gj[which] if which == "textures" else gj["fill"]["textures"]
    paths = {(t[0], t[1]): t[2] for t in src}

    def loader(provider, tx, ty):
        p = paths.get((tx, ty))
        if not p or not os.path.isfile(p):
            return None
        return np.array(Image.open(p).convert("RGB"))
    refs = {(t[0], t[1]): t[2] for t in gj["reference"]}

    def fetch_reference(tx, ty):
        p = refs.get((tx, ty))
        return np.asarray(Image.open(p).convert("RGB")) if p and os.path.isfile(p) else None
    prov = gj["provider"] if which == "textures" else gj["fill"]["provider"]
    return Group(gj["lat"], gj["zl"], prov, paths.keys(), loader), fetch_reference


def _save_group(g, state_dir, gid):
    np.save(_state_path(state_dir, gid, "fields.npy"), _np(g.fields))
    np.savez_compressed(_state_path(state_dir, gid, "alpha.npz"),
                        **{f"{tx}_{ty}": a for (tx, ty), a in g.alpha.items() if a.max() > 0})


def _load_group(g, state_dir, gid):
    g.fields = torch.from_numpy(np.load(_state_path(state_dir, gid, "fields.npy"))).to(_dev())
    with np.load(_state_path(state_dir, gid, "alpha.npz")) as z:
        g.alpha = {tuple(int(v) for v in k.split("_")): z[k] for k in z.files}


def job_fit(job):
    """Fit every group; report the textures that need a fill source."""
    result = {}
    for gj in job["groups"]:
        gid = gj["id"]
        t0 = time.time()
        try:
            g, fetch_reference = _group_from_job(gj)
            with torch.no_grad():
                g.fit(fetch_reference, clouds=bool(gj.get("clouds", True)))
            _save_group(g, job["state_dir"], gid)
        except Exception as e:      # one group failing must not cost the others
            import traceback
            traceback.print_exc()
            _log(f"{gj['provider']} ZL{gj['zl']}: fit failed ({e}); converted without correction")
            result[gid] = {"failed": str(e)}
            continue
        tex = set(g.textures)
        cloudy = (set(g.cloudy_textures()) | g.has_missing) & tex
        need = {(tx + 16 * di, ty + 16 * dj) for (tx, ty) in cloudy
                for di in (-1, 0, 1) for dj in (-1, 0, 1)} & tex
        result[gid] = {"cloudy": sorted(cloudy), "need": sorted(need)}
        msg = f"{gj['provider']} ZL{gj['zl']}: fitted {len(tex)} textures in {time.time() - t0:.0f}s"
        if cloudy:
            msg += f", {len(cloudy)} with cloud or missing tiles"
        _log(msg)
    return result


def _stamp(dds_path, mark):
    """Write the colour-correction stamp into DDS_HEADER.dwReserved1[0..2]."""
    import struct
    tag, version, flags = mark
    with open(dds_path, "r+b") as f:
        f.seek(32)
        f.write(struct.pack("<4sII", tag.encode("ascii"), int(version), int(flags)))


def _convert(rgb, out_dds, mask_path, nvcompress, tmp_dir, mark=None):
    """uint8 RGB (+ optional alpha mask file) -> DDS via a temporary TGA."""
    dxt5 = False
    if mask_path and os.path.isfile(mask_path):
        mask = Image.open(mask_path).convert("L").resize((rgb.shape[1], rgb.shape[0]), Image.BICUBIC)
        rgb = np.dstack([rgb, np.asarray(mask)])
        dxt5 = True
    base = os.path.basename(out_dds)
    tga = os.path.join(tmp_dir, base[:-4] + ".tga")
    tmp_dds = out_dds + ".colour.tmp"
    save_tga(rgb, tga)
    ok = False
    try:
        cmd = [nvcompress, "-bc3" if dxt5 else "-bc1", "-fast", tga, tmp_dds]
        flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform.startswith("win") else {}
        for _ in range(3):
            rc = subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, **flags)
            if rc == 0 and os.path.isfile(tmp_dds):
                if mark:
                    _stamp(tmp_dds, mark)
                os.replace(tmp_dds, out_dds)
                ok = True
                break
    finally:
        for f in (tga, tmp_dds):
            try:
                os.remove(f)
            except OSError:
                pass
    if ok and dxt5:
        try:
            os.remove(mask_path)
        except OSError:
            pass
    return ok


def _material_map(rgb, out_dds, zl, job):
    """XP12 material map from the corrected pixels (O4_Terrain_Materials).

    Best effort: Step 3 makes any map missing here from the DDS afterwards.
    """
    try:
        import O4_Terrain_Materials as TMAT
        TMAT.write_map(rgb, os.path.join(os.path.dirname(out_dds),
                                         TMAT.map_file_name(os.path.basename(out_dds))),
                       zl, float(job["materials"]["relief"]), job["nvcompress"])
    except Exception as e:
        _log(f"material map for {os.path.basename(out_dds)} failed ({e})")


def job_apply(job):
    """Correct (and fill) every texture and convert it to DDS."""
    done = []
    workers = int(job.get("workers", 4))
    for gj in job["groups"]:
        try:
            done += _apply_group(job, gj, workers)
        except Exception as e:
            import traceback
            traceback.print_exc()
            _log(f"{gj['provider']} ZL{gj['zl']}: correction failed ({e}); converted without it")
    return {"done": done}


def _apply_group(job, gj, workers):
    """Correct, fill and convert one group's textures; returns the converted ones."""
    done = []
    gid = gj["id"]
    g, _ = _group_from_job(gj)
    _load_group(g, job["state_dir"], gid)
    ga = None
    fill_tex = set()
    alt_paths = {}
    if gj.get("fill") and gj["fill"]["textures"]:
        t0 = time.time()
        ga, fetch_reference = _group_from_job(gj, "fill")
        with torch.no_grad():
            ga.fit(fetch_reference, clouds=True, mode=TV)
        fill_tex = set(ga.textures)
        alt_paths = {(t[0], t[1]): t[2] for t in gj["fill"]["textures"]}
        _log(f"{gj['provider']} ZL{gj['zl']}: fitted {len(fill_tex)} {ga.provider} textures "
             f"for the cloud fill in {time.time() - t0:.0f}s")
    items = gj["targets"]

    def one(item):
        tx, ty, src, out_dds, mask = item[:5]
        rgb = np.array(Image.open(src).convert("RGB"))
        alt = None
        if (tx, ty) in fill_tex and os.path.isfile(alt_paths.get((tx, ty), "")):
            alt = np.array(Image.open(alt_paths[(tx, ty)]).convert("RGB"))
        with _lock, torch.no_grad():
            res, miss = g.correct(rgb, tx, ty)
            if alt is not None:
                cor_a, miss_a = ga.correct(alt, tx, ty)
                pad = 12
                wm = (g.grown_alpha(tx, ty, pad) * (1 - ga.grown_alpha(tx, ty, pad)))[None]
                wf = _upsample_crop(wm, pad, pad, g.cm, g.km)[0]
                wf = torch.maximum(wf, miss.float()).clamp(0, 1) * (~miss_a).float()
                res = (res.float() * (1 - wf) + cor_a.float() * wf + 0.5).to(torch.uint8)
            out = _np(res.permute(1, 2, 0))
        ok = _convert(out, out_dds, mask, job["nvcompress"], job["tmp_dir"], job.get("mark"))
        if ok and job.get("materials"):
            _material_map(out, out_dds, gj["zl"], job)
        return [tx, ty, gj["zl"], gj["provider"]] if ok else None

    t0 = time.time()
    n = 0
    part = f"{gj['provider']} ZL{gj['zl']}"
    PROG.emit(ev="items", part=part, done=0, total=len(items))
    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(one, items):
            n += 1
            PROG.emit(ev="items", part=part, done=n)
            if r is not None:
                done.append(r)
            if n % 10 == 0 or n == len(items):
                _log(f"{gj['provider']} ZL{gj['zl']}: {n}/{len(items)} textures converted")
    _log(f"{gj['provider']} ZL{gj['zl']}: corrected {len(items)} textures in {time.time() - t0:.0f}s")
    return done


def main(job_path):
    with open(job_path) as f:
        job = json.load(f)
    if not available():
        raise RuntimeError("torch is not available in this python")
    where = "GPU (" + torch.cuda.get_device_name(0) + ")" if _dev().type == "cuda" else "CPU"
    _log("running on " + where)
    result = (job_fit if job["phase"] == "fit" else job_apply)(job)
    with open(job["result"], "w") as f:
        json.dump(result, f)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
