"""Fit a Sentinel-2 cloudless -> Esri World Imagery colour calibration.

Pairs (the user's picks, 2026-11-05): Kaohsiung + Shenzhen, EOX 2025 vs the
Esri Wayback release captured Dec 2024 - Mar 2025 (G:/Ortho4XP_Review/color/eox_refs).
Both are compared on the colour correction's ~38 m grid (4x4 ZL14 px), in
OpenCV Lab like the fit.  Model, fitted by matching distributions (not
pixel regression, which would shrink contrast and saturation):
  L'  = lut(L)                       monotone quantile map
  ab' = A (ab - mu_e) + mu_r         2x2 Monge-Kantorovich linear map
  C'' = chroma(hue, C')              per-hue chroma quantile map (hue kept)
Water (dark, flat) and clouds/no-data are left out.

Usage: python eox_calib_fit.py [year]     -> prints the table to paste
"""
import json, os, sys
import numpy as np, cv2

ROOT = os.path.join(os.environ.get("O4_REVIEW_DATA", r"G:\Ortho4XP_Review"), "color", "eox_refs")
YEAR = int(sys.argv[1]) if __name__ == "__main__" and sys.argv[1:] else 2025
# The user's picks: the 3x3 ZL14 windows round the hotspots plus the six Bing
# hotspot textures of the same cities (tex_<ty>_<tx>, ZL16 Esri averaged to ZL14).
KHH = ["khh", "tex_28528_54656", "tex_28528_54672", "tex_28544_54672"]
SZX = ["szx", "tex_28512_53472", "tex_28512_53488", "tex_28528_53488"]
PAIRS = {2025: [(s, "wb48925_z14.png") for s in KHH + SZX]}
HOLDOUT = os.environ.get("HOLDOUT")       # khh / szx: fit on the other city only


def coarse_lab(path):
    im = cv2.imread(path)[:, :, ::-1].astype(np.float32)
    h, w, _ = im.shape
    m = im.reshape(h // 4, 4, w // 4, 4, 3).mean((1, 3))
    lab = cv2.cvtColor(np.round(m).astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    L = cv2.cvtColor(im / 255, cv2.COLOR_RGB2LAB)[..., 0]
    det = L.reshape(h // 4, 4, w // 4, 4).std((1, 3))
    return lab, det, m


def sqrtm(S):
    w, V = np.linalg.eigh(S)
    return V @ np.diag(np.sqrt(np.clip(w, 1e-9, None))) @ V.T


def fit(pairs):
    E, R = [], []
    for spot, ref in pairs:
        le, de, me = coarse_lab(os.path.join(ROOT, spot, f"eox{YEAR}.png"))
        lr, dr, mr = coarse_lab(os.path.join(ROOT, spot, ref))
        water = (le[..., 0] < 40) & (de < 2.0)
        cloud = ((le[..., 0] > 80) & (np.hypot(le[..., 1], le[..., 2]) < 8)) | \
                ((lr[..., 0] > 85) & (np.hypot(lr[..., 1], lr[..., 2]) < 6))
        nodata = (np.abs(mr - 204).max(-1) < 3) | (mr.max(-1) < 3)
        ok = ~(water | cloud | nodata)
        print(f"{spot}: {ok.mean():.0%} of cells used (water {water.mean():.0%})")
        E.append(le[ok]); R.append(lr[ok])
    E, R = np.concatenate(E), np.concatenate(R)
    q = np.linspace(0.5, 99.5, 100)
    qe, qr = np.percentile(E[:, 0], q), np.percentile(R[:, 0], q)
    grid = np.arange(0, 101, 5.0)
    # Interpolate the quantile pairs onto a fixed L grid; linear beyond the data.
    lo_s = (qr[5] - qr[0]) / max(qe[5] - qe[0], 1e-3)
    hi_s = (qr[-1] - qr[-6]) / max(qe[-1] - qe[-6], 1e-3)
    lut = np.interp(grid, qe, qr)
    lut = np.where(grid < qe[0], qr[0] + (grid - qe[0]) * lo_s, lut)
    lut = np.where(grid > qe[-1], qr[-1] + (grid - qe[-1]) * hi_s, lut)
    lut = np.maximum.accumulate(np.clip(lut, 0, 100))
    ae, ar = E[:, 1:], R[:, 1:]
    mue, mur = ae.mean(0), ar.mean(0)
    Se, Sr = np.cov(ae.T), np.cov(ar.T)
    Se_h = sqrtm(Se); Se_ih = np.linalg.inv(Se_h)
    A = Se_ih @ sqrtm(Se_h @ Sr @ Se_h) @ Se_ih
    ab1 = (ae - mue) @ A.T + mur
    ch = chroma_table(ab1, ar)
    return {"year": YEAR, "hue_deg": HUES.tolist(), "C_grid": C_GRID.tolist(), "C_lut": ch, "L_grid": grid.tolist(), "L_lut": [round(float(v), 2) for v in lut],
            "A": [[round(float(v), 4) for v in r] for r in A], "mu_e": [round(float(v), 3) for v in mue],
            "mu_r": [round(float(v), 3) for v in mur]}, E, R


HUES = np.arange(0, 360, 30.0)
C_GRID = np.array([0, 4, 8, 12, 16, 20, 25, 30, 40, 60, 100.0])
HUE_HALF = 45.0     # hue window half-width of a sector
FULL_WEIGHT = 2000.0  # sector weight (cells) for full trust; fewer blend to identity


def _wq(x, w, q):
    o = np.argsort(x); x, w = x[o], w[o]
    cw = np.cumsum(w); cw /= cw[-1]
    return np.interp(q, cw, x)


def chroma_table(ab_e, ab_r):
    """C_lut[k][j]: chroma that EOX chroma C_GRID[j] at hue HUES[k] maps to."""
    q = np.linspace(0.02, 0.98, 49)
    out = []
    for h in HUES:
        row = []
        ws = []
        for ab in (ab_e, ab_r):
            hh = np.degrees(np.arctan2(ab[:, 1], ab[:, 0]))
            d = np.abs((hh - h + 180) % 360 - 180)
            ws.append(np.clip(1 - d / HUE_HALF, 0, None) * np.clip(np.hypot(ab[:, 0], ab[:, 1]) / 4, 0, 1))
        print(f"   hue {h:3.0f}: weight {ws[0].sum():7.0f} eox {ws[1].sum():7.0f} esri")
        conf = float(np.clip(min(ws[0].sum(), ws[1].sum()) / FULL_WEIGHT, 0, 1))
        if conf == 0:
            out.append(C_GRID.tolist()); continue
        ce = _wq(np.hypot(ab_e[:, 0], ab_e[:, 1]), ws[0], q)
        cr = _wq(np.hypot(ab_r[:, 0], ab_r[:, 1]), ws[1], q)
        g = np.interp(C_GRID, ce, cr / np.maximum(ce, 1e-3))     # gain, held flat beyond the data
        g = 1 + conf * (np.clip(g, 0.5, 1.25) - 1)      # thin hues stay at the linear map
        row = np.maximum.accumulate(C_GRID * g)
        out.append([round(float(v), 2) for v in row])
    return out


def apply_chroma(c, ab):
    C = np.hypot(ab[..., 0], ab[..., 1]); h = np.degrees(np.arctan2(ab[..., 1], ab[..., 0])) % 360
    lut = np.array(c["C_lut"]); hues = np.array(c["hue_deg"]); step = hues[1] - hues[0]
    k0 = np.floor(h / step).astype(int) % len(hues); k1 = (k0 + 1) % len(hues); t = (h / step) - np.floor(h / step)
    c0 = np.array([np.interp(C.ravel(), c["C_grid"], lut[k]) for k in range(len(hues))])  # (K, N)
    idx = np.arange(C.size)
    Cn = (1 - t.ravel()) * c0[k0.ravel(), idx] + t.ravel() * c0[k1.ravel(), idx]
    s = (Cn / np.maximum(C.ravel(), 1e-6)).reshape(C.shape)
    return ab * s[..., None]


def apply(c, lab, chroma=True):
    L = np.interp(lab[..., 0], c["L_grid"], c["L_lut"])
    ab = (lab[..., 1:] - c["mu_e"]) @ np.array(c["A"]).T + c["mu_r"]
    if chroma:
        ab = apply_chroma(c, ab)
    return np.concatenate([L[..., None], ab], -1)


def report(tag, X, R):
    def st(Y):
        C = np.hypot(Y[:, 1], Y[:, 2]); h = np.degrees(np.arctan2(Y[:, 2], Y[:, 1])) % 360
        warm = (h < 75) & (C > 8)
        return [Y[:, 0].mean(), Y[:, 0].std(), Y[:, 1].mean(), Y[:, 2].mean(), C.mean(),
                np.percentile(C, 90), 100 * warm.mean(), C[warm].mean()]
    print(f"{tag:<10}" + "".join(f"{v:7.1f}" for v in st(X)))


if __name__ == "__main__":
    pairs = PAIRS[YEAR]
    if HOLDOUT:
        held = KHH if HOLDOUT == "khh" else SZX
        pairs = [p for p in pairs if p[0] not in held]
    c, E, R = fit(pairs)
    print(f"{'':<10}" + "".join(f"{n:>7}" for n in ["L", "Lsd", "a*", "b*", "C", "C90", "warm%", "warmC"]))
    report("Esri", R, R); report("EOX", E, R); report("EOX lin", apply(c, E[None], False)[0], R)
    report("EOX cal", apply(c, E[None])[0], R)
    for spot, ref in ([] if HOLDOUT else pairs[:1] + pairs[4:5]):
        le = coarse_lab(os.path.join(ROOT, spot, f"eox{YEAR}.png"))[0].reshape(-1, 3)
        lr = coarse_lab(os.path.join(ROOT, spot, ref))[0].reshape(-1, 3)
        print(" ", spot); report("  Esri", lr, lr); report("  EOX", le, lr); report("  EOX cal", apply(c, le[None])[0], lr)
    print(c["C_lut"])
    if HOLDOUT:
        for name in (KHH if HOLDOUT == "khh" else SZX):
            le = coarse_lab(os.path.join(ROOT, name, f"eox{YEAR}.png"))[0].reshape(-1, 3)
            lr = coarse_lab(os.path.join(ROOT, name, "wb48925_z14.png"))[0].reshape(-1, 3)
            print(" held out", name); report("  Esri", lr, lr); report("  EOX", le, lr); report("  EOX cal", apply(c, le[None])[0], lr)
    else:
        json.dump(c, open(os.path.join(ROOT, f"calib_{YEAR}.json"), "w"), indent=1)
