"""GPU (PyTorch/CUDA) port of the clear-sky colour correction (refnorm, mode "dt" = v13).

Same method and parameters as refnorm.py (P is shared), restructured for speed:

  * every 16 Mpx operation (decode, statistics, colour conversion, applying the
    correction) runs on the GPU;
  * the edge-aware recursive filter, the only sequential part, is evaluated as
    a parallel prefix scan: y[n] = a[n] y[n-1] + b[n] is associative, so
    log2(N) doubling steps replace N sequential ones with identical results;
  * all moments a fit needs are stacked and smoothed in one filter call
    instead of one call per moment.

The small 2-D morphology (median, opening, dilation on the ~900 px coarse grid)
stays in OpenCV on the CPU; those arrays are tiny.

Colour conversion uses the exact sRGB/Lab formulas; OpenCV's float path uses a
gamma lookup table that differs by up to ~0.3 Lab units.
"""
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import refnorm as RN

DEV = torch.device("cuda")
P = RN.P
TEX, COARSE = RN.TEX, RN.COARSE

_M = torch.tensor([[0.412453, 0.357580, 0.180423],
                   [0.212671, 0.715160, 0.072169],
                   [0.019334, 0.119193, 0.950227]], dtype=torch.float32)
_WHITE = torch.tensor([0.950456, 1.0, 1.088754], dtype=torch.float32)


################################################################################
# Colour
def srgb_to_lab(x):
    """x: (3, H, W) float sRGB in [0, 1] -> (3, H, W) Lab (OpenCV conventions)."""
    lin = torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    M = (_M / _WHITE[:, None]).to(x.device)
    xyz = torch.einsum("ij,jhw->ihw", M, lin)
    f = torch.where(xyz > 0.008856, xyz.clamp_min(0) ** (1 / 3), 7.787 * xyz + 16 / 116)
    L = torch.where(xyz[1] > 0.008856, 116 * f[1] - 16, 903.3 * xyz[1])
    return torch.stack([L, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def lab_to_srgb(lab):
    """(3, H, W) Lab -> (3, H, W) float sRGB, clipped to [0, 1]."""
    L, a, b = lab[0], lab[1], lab[2]
    fy = (L + 16) / 116
    Y = torch.where(L > 8.0, fy ** 3, L / 903.3)
    fy = torch.where(L > 8.0, fy, 7.787 * Y + 16 / 116)
    fx, fz = fy + a / 500, fy - b / 200
    inv = lambda t: torch.where(t > 0.206893, t ** 3, (t - 16 / 116) / 7.787)
    xyz = torch.stack([inv(fx), Y, inv(fz)]) * _WHITE.to(lab.device)[:, None, None]
    lin = torch.einsum("ij,jhw->ihw", torch.linalg.inv(_M).to(lab.device), xyz).clamp(0, 1)
    return torch.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055).clamp(0, 1)


################################################################################
# Filters on (C, H, W) tensors
def _sym_pad(x, p, dim):
    """OpenCV BORDER_REFLECT (edge pixel repeated): fedcba|abcdef|fedcba."""
    if p == 0:
        return x
    n = x.shape[dim]
    lo = x.narrow(dim, 0, p).flip(dim)
    hi = x.narrow(dim, n - p, p).flip(dim)
    return torch.cat([lo, x, hi], dim)


_kernels = {}


def gblur(x, sigma):
    """cv2.GaussianBlur(x, (0, 0), sigma, borderType=BORDER_REFLECT), per channel."""
    k = int(round(sigma * 8 + 1)) | 1
    key = (k, float(sigma))
    if key not in _kernels:
        _kernels[key] = torch.from_numpy(cv2.getGaussianKernel(k, sigma).astype(np.float32)[:, 0]).to(DEV)
    w = _kernels[key]
    C = x.shape[0]
    p = k // 2
    y = _sym_pad(x, p, 2)[None]
    y = F.conv2d(y, w.view(1, 1, 1, k).expand(C, 1, 1, k), groups=C)
    y = _sym_pad(y[0], p, 1)[None]
    y = F.conv2d(y, w.view(1, 1, k, 1).expand(C, 1, k, 1), groups=C)
    return y[0]


def _scan_last(B, A):
    """Solve y[n] = A[n] y[n-1] + B[n] along the last dim (A[0] must be 0).

    B: (C, H, N), A: (H, N).  Hillis-Steele doubling, log2(N) steps.
    """
    k = 1
    N = B.shape[-1]
    while k < N:
        Bn = B.clone()
        Bn[..., k:] += A[..., k:] * B[..., :-k]
        An = A.clone()
        An[..., k:] *= A[..., :-k]
        B, A = Bn, An
        k *= 2
    return B


class DomainTransform(object):
    """GPU twin of refnorm.DomainTransform (recursive filter, 3 iterations)."""

    def __init__(self, guide, range_sigmas, iterations=3):
        g = torch.as_tensor(guide, dtype=torch.float32, device=DEV)      # (H, W, 3)
        r = torch.as_tensor(range_sigmas, dtype=torch.float32, device=DEV)
        self.dx = ((g[:, 1:] - g[:, :-1]).abs() / r).sum(-1)              # (H, W-1)
        self.dy = ((g[1:] - g[:-1]).abs() / r).sum(-1)                    # (H-1, W)
        self.iterations = iterations

    def __call__(self, x, sigma_s):
        x = x.float()
        n = self.iterations
        H, W = x.shape[1:]
        zc = torch.zeros(H, 1, device=DEV)
        zr = torch.zeros(W, 1, device=DEV)
        for i in range(n):
            sh = sigma_s * np.sqrt(3.0) * 2 ** (n - i - 1) / np.sqrt(4 ** n - 1)
            a = float(np.exp(-np.sqrt(2.0) / sh))
            wx = a ** (1.0 + sigma_s * self.dx)                            # (H, W-1)
            wy = (a ** (1.0 + sigma_s * self.dy)).T                        # (W, H-1)
            A = torch.cat([zc, wx], 1)                                     # left -> right
            x = _scan_last((1 - A) * x, A)
            A = torch.cat([zc, wx.flip(1)], 1)                             # right -> left
            x = _scan_last((1 - A) * x.flip(2), A).flip(2)
            xt = x.transpose(1, 2).contiguous()                            # (C, W, H)
            A = torch.cat([zr, wy], 1)                                     # top -> bottom
            xt = _scan_last((1 - A) * xt, A)
            A = torch.cat([zr, wy.flip(1)], 1)                             # bottom -> top
            xt = _scan_last((1 - A) * xt.flip(2), A).flip(2)
            x = xt.transpose(1, 2).contiguous()
        return x


def moments(f, sigma, w, xs):
    """Weighted local means and variances of every (H, W) signal in xs, one filter call."""
    S = torch.stack([w] + [x * w for x in xs] + [x * x * w for x in xs])
    Fs = f(S, sigma)
    sw = Fs[0] + 1e-6
    n = len(xs)
    m = [Fs[1 + i] / sw for i in range(n)]
    v = [(Fs[1 + n + i] / sw - m[i] * m[i]).clamp_min(0) for i in range(n)]
    return m, v


################################################################################
# Fit
def _np(t):
    return t.detach().cpu().numpy()


def _median5x2(a):
    """Two 5x5 medians per channel (CPU, coarse grid)."""
    a = np.ascontiguousarray(a, np.float32)
    for _ in range(2):
        a = np.stack([cv2.medianBlur(np.ascontiguousarray(a[..., k]), 5) for k in range(3)], -1)
    return a


def fit_fields(O, R, valid, px_m, p=P, exclude=None):
    """GPU twin of refnorm.fit_fields (mode "dt").  O, R: (3, H, W) Lab tensors."""
    v = valid if exclude is None else valid & ~exclude
    base = torch.as_tensor(v, dtype=torch.float32, device=DEV)
    sL, sab = p.win_L_m / px_m, p.win_ab_m / px_m
    OL, Oa, Ob = O
    RL, Ra, Rb = R

    O_np, R_np = _np(O.permute(1, 2, 0)), _np(R.permute(1, 2, 0))
    guide = _median5x2(np.where(valid[..., None], O_np - R_np, 0.0))
    smooth = DomainTransform(guide, (p.range_L, p.range_ab, p.range_ab)) if p.edge_aware else None

    def fit(w, f, sL, sab):
        ww = w.sum(dtype=torch.float64) + 1e-6
        gm = lambda x: (x * w).sum(dtype=torch.float64) / ww
        gv = lambda x: gm(x * x) - gm(x) ** 2
        gL = torch.sqrt(gv(RL) / torch.clamp(gv(OL), min=1e-6)).float()
        gs = torch.sqrt((gv(Ra) + gv(Rb)) / torch.clamp(gv(Oa) + gv(Ob), min=1e-6)).float()
        f = f or gblur
        if sL == sab:
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
        t = torch.stack([mRa - s * mOa, mRb - s * mOb])
        return g, o, s, t

    if p.ref_haze:
        og = _median5x2(O_np)
        dt_o = DomainTransform(og, (p.ref_haze_range, p.ref_haze_range * 1.5, p.ref_haze_range * 1.5))
        mid = p.win_ab_m / px_m
        rg, ro, rs, rt = fit(base, dt_o, mid, mid)
        predL = rg * OL + ro
        predC = torch.sqrt((rs * Oa + rt[0]) ** 2 + (rs * Ob + rt[1]) ** 2)
        Rc = torch.sqrt(Ra ** 2 + Rb ** 2)
        haze = (RL - predL > p.ref_haze_rise) & (Rc < p.ref_haze_chroma * predC + 1.0)
        haze = cv2.dilate(_np(haze).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        base = base * torch.as_tensor(~haze, dtype=torch.float32, device=DEV)
    w = base.clone()
    for it in range(p.irls + 1):
        g, o, s, t = fit(w, smooth, sL, sab)
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
        wide = fit(w, None, p.fill_m / px_m, p.fill_m / px_m)
        fg, fo, fs, ft = wide
        a = torch.clamp(support / p.fill_support, 0.0, 1.0)
        g, o, s = a * g + (1 - a) * fg, a * o + (1 - a) * fo, a * s + (1 - a) * fs
        t = a * t + (1 - a) * ft

    if p.red_cap > 0:
        rg, ro, rs, rt = wide if wide is not None else fit(w, None, p.fill_m / px_m, p.fill_m / px_m)
        (mOa,), _ = moments(gblur, sab, base, [Oa])
        local = s * mOa + t[0]
        regional = rs * mOa + rt[0]
        t = t.clone()
        t[0] -= torch.clamp(local - regional - p.red_cap, min=0.0)
    return g, o, s, t, w


################################################################################
# Block I/O and statistics
DECODER = os.environ.get("REFNORM_DECODER", "nvjpeg")   # "nvjpeg" (GPU) or "pil" (CPU)


def decode(path):
    """JPEG -> (3, H, W) uint8 RGB on the GPU.

    nvjpeg is fastest; PIL reproduces the CPU prototype bit for bit.  The two
    decoders differ by about one level per pixel, which the edge-aware fit
    can amplify in small enclosed windows, so one build must stick to one.
    """
    if DECODER == "pil":
        a = np.asarray(Image.open(path).convert("RGB"))
        return torch.from_numpy(a.copy()).to(DEV).permute(2, 0, 1).contiguous()
    import torchvision.io as tio
    return tio.decode_jpeg(tio.read_file(path), device=DEV)


def missing_mask(img):
    """(H, W) bool: whole 256-px provider tiles that are white (not downloaded)."""
    white = (img.min(0).values >= 254).float()
    blk = F.avg_pool2d(white[None, None], 256)[0, 0] > 0.995
    return blk.repeat_interleave(256, 0).repeat_interleave(256, 1)


def coarse_lab(mean_u8):
    """Lab of coarse uint8 means via OpenCV, so the fit sees exactly the CPU inputs.

    The fit is sensitive to sub-unit input changes (thresholds, medians), and
    OpenCV's float Lab differs from the exact formula by up to ~0.3; the
    coarse grid is tiny, so the CPU round trip costs milliseconds.
    """
    a = _np(mean_u8.permute(1, 2, 0)).astype(np.float32) / 255.0
    return torch.from_numpy(cv2.cvtColor(a, cv2.COLOR_RGB2LAB).transpose(2, 0, 1).copy()).to(DEV)


def texture_stats(img):
    """Coarse Lab means, luma detail and validity of one texture."""
    x = img.float()
    c = x.shape[1] // COARSE
    lab = coarse_lab(torch.round(F.avg_pool2d(x[None], COARSE)[0]))    # cv2 INTER_AREA on uint8 rounds
    lum = (x[0] * 0.299 + x[1] * 0.587 + x[2] * 0.114) * (100 / 255.0)
    det = lum.view(c, COARSE, c, COARSE).permute(0, 2, 1, 3).reshape(c, c, -1).std(-1, unbiased=False)
    miss = F.avg_pool2d(missing_mask(img).float()[None, None], COARSE)[0, 0] > 0.01
    return lab, det, ~miss


class Block(object):
    """GPU twin of refnorm.Block: n x n textures of one provider + reference."""

    def __init__(self, key, prov):
        self.cpu = RN.Block(key, prov)
        b = self.cpu
        self.meta, self.n, self.paths = b.meta, b.n, b.paths
        self.px_m, self.margin = b.px_m, b.margin

    def reference(self):
        ref = np.asarray(Image.open(self.cpu.ref_path).convert("RGB"))
        r = torch.from_numpy(ref.copy()).to(DEV).permute(2, 0, 1).float()
        k = COARSE // self.meta["ref_scale"]
        R = coarse_lab(torch.round(F.avg_pool2d(r[None], k)[0]))
        rl = srgb_to_lab(r / 255.0)[0][None, None]
        pad = F.pad(rl, (k, k, k, k), mode="reflect")                     # cv2.blur default border
        rm = F.avg_pool2d(pad, 2 * k + 1, stride=1)
        rv = (F.avg_pool2d(pad * pad, 2 * k + 1, stride=1) - rm * rm).clamp_min(0)
        rdet = torch.sqrt(F.avg_pool2d(rv, k)[0, 0])
        return R, rdet

    def coarse(self, images=None):
        """Lab ortho, Lab reference, validity, ortho detail, reference detail."""
        R, rdet = self.reference()
        H = R.shape[1]
        c = TEX // COARSE
        m = self.margin
        O = torch.zeros(3, H, H, device=DEV)
        odet = torch.zeros(H, H, device=DEV)
        valid = torch.zeros(H, H, dtype=torch.bool, device=DEV)
        for (i, j), path in self.paths.items():
            img = images[(i, j)] if images is not None else decode(path)
            lab, det, ok = texture_stats(img)
            sl = (slice(m + j * c, m + (j + 1) * c), slice(m + i * c, m + (i + 1) * c))
            O[:, sl[0], sl[1]] = lab
            odet[sl] = det
            valid[sl] = ok
        return O, R, valid, odet, rdet


def water_mask(R, rdet, odet, valid, p=P):
    m = ((R[0] < p.water_L) & (rdet < p.water_rdet) & (odet < p.water_odet) & valid)
    m = cv2.morphologyEx(_np(m).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.dilate(m, np.ones((3, 3), np.uint8)).astype(bool)


################################################################################
# Applying the correction
def compose_look(fields, look):
    g, o, s, t = fields
    G, O_, S, T = look
    T = torch.tensor(T, dtype=torch.float32, device=DEV)[:, None, None]
    return G * g, G * o + O_, S * s, S * t + T


def correct_texture(img, fields, block, ij):
    """img: (3, H, W) uint8 GPU tensor -> corrected (3, H, W) uint8 GPU tensor."""
    g, o, s, t = fields
    c = TEX // COARSE
    m = block.margin
    i, j = ij
    y, x = m + j * c - 1, m + i * c - 1
    stack = torch.cat([g[None], o[None], s[None], t])[:, y:y + c + 2, x:x + c + 2]
    up = F.interpolate(stack[None], scale_factor=COARSE, mode="bilinear", align_corners=False)[0]
    up = up[:, COARSE:COARSE + TEX, COARSE:COARSE + TEX]
    lab = srgb_to_lab(img.float() / 255.0)
    out = torch.stack([up[0] * lab[0] + up[1], up[2] * lab[1] + up[3], up[2] * lab[2] + up[4]])
    rgb = (lab_to_srgb(out) * 255.0 + 0.5).clamp(0, 255).to(torch.uint8)
    rgb[:, missing_mask(img)] = 255
    return rgb


def run_block(key, prov, look, keep_images=True):
    """Whole clear-sky pipeline for one block; returns {ij: (3, H, W) uint8 GPU tensor}."""
    b = Block(key, prov)
    images = {ij: decode(pth) for ij, pth in b.paths.items()} if keep_images else None
    O, R, valid, odet, rdet = b.coarse(images)
    valid_np = _np(valid)
    excl = water_mask(R, rdet, odet, valid, P) if P.water else None
    g, o, s, t, w = fit_fields(O, R, valid_np, b.px_m, P, excl)
    fields = compose_look((g, o, s, t), look)
    out = {}
    for ij, pth in b.paths.items():
        img = images[ij] if images is not None else decode(pth)
        out[ij] = correct_texture(img, fields, b, ij)
    return out, (g, o, s, t, w), b
