"""GPU port of the cloud path (v14fill): Huber-TV fit, cloud detection, fill.

Twin of refnorm.fit_fields_tv / refnorm.huber_tv / the cloud helpers in
refnorm.py and the pipeline in run_clouds.py (ALT_TV=1 NEAR_CLOUD_TV=1).
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import refnorm as RN
import refnorm_gpu as RG
from refnorm_gpu import DEV, P, TEX, COARSE, gblur, moments, _np, decode, missing_mask, \
    srgb_to_lab, compose_look, correct_texture, water_mask

MED = RN.MED
CP = RN.CP


################################################################################
# Huber-TV fit (fill source, and near clouds)
def huber_tv(f, w, lam, eps, iters=300, init=None):
    """f: (C, H, W), w: (H, W), lam: scalar or (H, W) tensor."""
    u = (f if init is None else init).clone()
    ub = u.clone()
    px = torch.zeros_like(f)
    py = torch.zeros_like(f)
    tau = sigma = 1.0 / np.sqrt(8.0)
    w = w[None]
    lam = torch.as_tensor(lam, dtype=torch.float32, device=DEV)
    lam = lam.expand(f.shape[1:])[None] if lam.ndim else lam.view(1, 1, 1)
    damp = 1.0 + sigma * eps / lam
    for _ in range(iters):
        gx = torch.zeros_like(ub)
        gy = torch.zeros_like(ub)
        gx[:, :, :-1] = ub[:, :, 1:] - ub[:, :, :-1]
        gy[:, :-1] = ub[:, 1:] - ub[:, :-1]
        px = (px + sigma * gx) / damp
        py = (py + sigma * gy) / damp
        nrm = (torch.sqrt((px * px + py * py).sum(0, keepdim=True)) / lam).clamp_min(1.0)
        px = px / nrm
        py = py / nrm
        div = torch.zeros_like(px)
        div[:, :, :-1] += px[:, :, :-1]
        div[:, :, 1:] -= px[:, :, :-1]
        div[:, :-1] += py[:, :-1]
        div[:, 1:] -= py[:, :-1]
        un = (u + tau * div + tau * w * f) / (1.0 + tau * w)
        ub = 2 * un - u
        u = un
    return u


def fit_fields_tv(O, R, valid, px_m, p, exclude=None):
    v = valid if exclude is None else valid & ~exclude
    base = torch.as_tensor(v, dtype=torch.float32, device=DEV)
    small, medium = p.tv_win_m / px_m, p.win_L_m / px_m
    OL, Oa, Ob = O
    RL, Ra, Rb = R

    def fit(w, sg, s_off):
        ww = w.sum(dtype=torch.float64) + 1e-6
        gm = lambda x: (x * w).sum(dtype=torch.float64) / ww
        gv = lambda x: gm(x * x) - gm(x) ** 2
        gL = torch.sqrt(gv(RL) / torch.clamp(gv(OL), min=1e-6)).float()
        gs = torch.sqrt((gv(Ra) + gv(Rb)) / torch.clamp(gv(Oa) + gv(Ob), min=1e-6)).float()
        sig = [OL, RL, Oa, Ob, Ra, Rb]
        _, (vO, vR, vOa, vOb, vRa, vRb) = moments(gblur, sg, w, sig)
        (mO, mR, mOa, mOb, mRa, mRb), _ = moments(gblur, s_off, w, sig)
        lam = vO / (vO + p.shrink_L ** 2)
        g = torch.sqrt(vR / vO.clamp_min(1e-6))
        g = torch.clamp(lam * g + (1 - lam) * gL, min=float(gL * p.g_range[0]), max=float(gL * p.g_range[1]))
        o = mR - g * mO
        vOc, vRc = vOa + vOb, vRa + vRb
        lam = vOc / (vOc + p.shrink_ab ** 2)
        s = torch.sqrt(vRc / vOc.clamp_min(1e-6))
        s = torch.clamp(lam * s + (1 - lam) * gs, min=float(gs * p.s_range[0]), max=float(gs * p.s_range[1]))
        return g, o, s, torch.stack([mRa - s * mOa, mRb - s * mOb])

    lam_map = p.tv_lambda
    if p.tv_attribute:
        O_np, R_np = _np(O.permute(1, 2, 0)), _np(R.permute(1, 2, 0))
        cheap = torch.from_numpy(RN.ortho_edge_share(O_np, R_np, p)).to(DEV)
        lam_map = p.tv_lambda * (p.tv_lam_ref + (1.0 - p.tv_lam_ref) * cheap)
    sc = torch.tensor([p.tv_L_typ, 1.0, p.tv_C_typ, 1.0, 1.0], device=DEV)[:, None, None]

    def residual_weights(g, o, s, t):
        eL = (g * OL + o - RL) / p.resid_L
        eab = torch.sqrt((s * Oa + t[0] - Ra) ** 2 + (s * Ob + t[1] - Rb) ** 2) / p.resid_ab
        return base / (1.0 + eL ** 2 + eab ** 2)

    w = base.clone()
    u = None
    ones = gblur(torch.ones_like(base)[None], small)[0] + 1e-6
    for it in range(p.irls):
        g, o, s, t = fit(w, medium, small) if p.tv_split else fit(w, small, small)
        conf = torch.clamp(gblur(w[None], small)[0] / ones / p.tv_conf_full, 0, 1) ** 2
        f = torch.cat([g[None], o[None], s[None], t]) * sc
        u = huber_tv(f, conf, lam_map, p.tv_eps, p.tv_iters if u is None else p.tv_iters // 3, init=u)
        uu = u / sc
        if it < p.irls - 1:
            w = residual_weights(uu[0], uu[1], uu[2], uu[3:5])
    u = u / sc
    return u[0], u[1], u[2], u[3:5], w


def fit_any(O, R, valid, px_m, p, exclude=None):
    return (fit_fields_tv if p.mode == "tv" else RG.fit_fields)(O, R, valid, px_m, p, exclude)


################################################################################
# Clouds at reference resolution
def dilate(m, k):
    return F.max_pool2d(m.float()[None, None], k, stride=1, padding=k // 2)[0, 0]


def erode(m, k):
    return -F.max_pool2d(-m.float()[None, None], k, stride=1, padding=k // 2)[0, 0]


def medium_mosaic(block, images):
    ref = np.asarray(Image.open(block.cpu.ref_path).convert("RGB"))
    Rm = srgb_to_lab(torch.from_numpy(ref.copy()).to(DEV).permute(2, 0, 1).float() / 255.0)
    H = ref.shape[0]
    Om = torch.zeros(3, H, H, device=DEV)
    vm = torch.zeros(H, H, dtype=torch.bool, device=DEV)
    c = TEX // MED
    m = block.margin * COARSE // MED
    for (i, j), img in images.items():
        sl = (slice(m + j * c, m + (j + 1) * c), slice(m + i * c, m + (i + 1) * c))
        Om[:, sl[0], sl[1]] = srgb_to_lab(torch.round(F.avg_pool2d(img.float()[None], MED)[0]) / 255.0)
        vm[sl] = F.avg_pool2d((~missing_mask(img)).float()[None, None], MED)[0, 0] > 0.5
    return Om, Rm, vm


def reference_cloud(Rm):
    C = torch.sqrt(Rm[1] ** 2 + Rm[2] ** 2)
    return dilate((Rm[0] > CP.ref_cloud_L) & (C < CP.ref_cloud_C), 5) > 0


def cloud_candidates(Om, Rm, vm):
    C_obs = torch.sqrt(Om[1] ** 2 + Om[2] ** 2)
    m = (Om[0] - Rm[0] > CP.cand_rise) & (C_obs < CP.cand_chroma) & vm
    return m | (reference_cloud(Rm) & vm)


def coarse_from_medium(x):
    return F.avg_pool2d(x.float()[None, None], COARSE // MED)[0, 0]


def cloud_alpha(Om, Rm, vm, fields):
    H = Om.shape[1]

    def up(f):
        x = f[None, None] if f.ndim == 2 else f[None]
        return F.interpolate(x, size=(H, H), mode="bilinear", align_corners=False)[0]
    g, o, s = up(fields[0])[0], up(fields[1])[0], up(fields[2])[0]
    t = up(fields[3])
    L_exp = (Rm[0] - o) / g.clamp_min(1e-3)
    ab_exp = (Rm[1:] - t) / s.clamp_min(1e-3)
    C_exp = torch.sqrt((ab_exp ** 2).sum(0))
    C_obs = torch.sqrt(Om[1] ** 2 + Om[2] ** 2)
    rise = Om[0] - L_exp
    a_L = torch.clamp((rise - CP.min_rise) / torch.clamp(CP.cloud_L - L_exp - CP.min_rise, min=5.0), 0, 1)
    a_C = torch.where(C_exp > CP.min_chroma_ref, torch.clamp(1 - C_obs / C_exp.clamp_min(1e-3), 0, 1),
                      torch.ones_like(C_exp))
    a = torch.minimum(a_L, a_C)
    rc = reference_cloud(Rm)
    a_abs = torch.clamp((Om[0] - 70.0) / 18.0, 0, 1) * (C_obs < 8.0)
    a = torch.where(rc, a_abs, a)
    a = torch.where(a < CP.alpha_floor, torch.zeros_like(a), a) * vm
    k = 2 * CP.open_px + 1
    keep = dilate(erode(a > 0, k) > 0, k)
    a = gblur((a * keep)[None], CP.feather_px)[0]
    return a.clamp(0, 1), rc


class Wide(P):
    edge_aware = False
    red_cap = 0.0
    win_L_m = 1000.0
    win_ab_m = 1500.0


class AltTV(P):
    mode = "tv"
    tv_lambda = 12.0


def fit_with_clouds(block, images, p=P):
    O, R, valid, odet, rdet = block.coarse(images)
    valid_np = _np(valid)
    water = water_mask(R, rdet, odet, valid)
    Om, Rm, vm = medium_mosaic(block, images)
    cand = _np(coarse_from_medium(cloud_candidates(Om, Rm, vm)) > 0.15)
    cloud_c = cand
    for _ in range(2):
        wf = fit_any(O, R, valid_np, block.px_m, Wide, water | cloud_c)
        alpha, rc = cloud_alpha(Om, Rm, vm, wf[:4])
        cloud_c = cand | _np(coarse_from_medium(alpha) > CP.coarse_alpha) | \
            _np(coarse_from_medium(rc) > 0.3)
    excl = water | cloud_c
    fields = fit_any(O, R, valid_np, block.px_m, p, excl)[:4]
    return fields, alpha, excl, (O, R, valid_np)


def texture_medium_field(field_m, block, ij):
    c = TEX // MED
    m = block.margin * COARSE // MED
    i, j = ij
    y, x = m + j * c - 1, m + i * c - 1
    crop = field_m[y:y + c + 2, x:x + c + 2]
    up = F.interpolate(crop[None, None], scale_factor=MED, mode="bilinear", align_corners=False)[0, 0]
    return up[MED:MED + TEX, MED:MED + TEX]


NEAR_M = 400.0
FILL_GROW = 4
FILL_FULL = 0.2


def run_cloud_block(key, look, near_cloud_tv=True, alt_tv=True):
    """v14fill: primary corrected (TV near cloud), cloud filled from the alternate."""
    meta = json.load(open(os.path.join(RN.SET, key, "meta.json")))
    prim, alt = meta["providers"][0], meta["providers"][1]
    res = {}
    for pv in (prim, alt):
        b = RG.Block(key, pv)
        images = {ij: decode(pth) for ij, pth in b.paths.items()}
        fields, alpha, excl, (O, R, valid_np) = fit_with_clouds(
            b, images, AltTV if (alt_tv and pv == alt) else P)
        if near_cloud_tv and pv == prim:
            tv = fit_any(O, R, valid_np, b.px_m, AltTV, excl)[:4]
            cl = coarse_from_medium(alpha)
            near = torch.clamp(gblur((cl > 0.05).float()[None], NEAR_M / b.px_m)[0] * 4.0, 0, 1)
            fields = tuple(near * a + (1 - near) * c for a, c in zip(tv, fields))
        res[pv] = (b, images, compose_look(fields, look), alpha)
    bP, imP, fP, aP = res[prim]
    bA, imA, fA, aA = res[alt]
    k = 2 * FILL_GROW + 1
    aPf = torch.clamp(gblur(dilate(aP, k)[None], 1.5)[0] / FILL_FULL, 0, 1)
    aAf = torch.clamp(gblur(dilate(aA, k)[None], 1.5)[0] / FILL_FULL, 0, 1)
    wfill_m = aPf * (1 - aAf)
    out_p, out_a = {}, {}
    for ij in bP.paths:
        corP = correct_texture(imP[ij], fP, bP, ij)
        corA = correct_texture(imA[ij], fA, bA, ij)
        wf = torch.maximum(texture_medium_field(wfill_m, bP, ij), missing_mask(imP[ij]).float())
        wf = wf.clamp(0, 1) * (~missing_mask(imA[ij])).float()
        out_p[ij] = (corP.float() * (1 - wf) + corA.float() * wf + 0.5).to(torch.uint8)
        out_a[ij] = corA
    return (prim, out_p, aP, bP), (alt, out_a, aA, bA)
