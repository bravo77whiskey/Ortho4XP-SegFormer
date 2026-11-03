"""Score estimators of tree cover INSIDE the production tree mask.

Truth: 2019 LiDAR (veg >= 2 m) at ZL18, Brandenburg, images shrunk to lower zoom
scales. "z16low" is 2.2 m/px, what ZL16 really is at 25 degrees latitude.
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image

import density_lib as D
import O4_Landcover_Inference as SEGFORMER
import O4_Canopy_Height as CANOPY
import O4_Veg_Lowzoom as LOWZOOM

ROOT = r"E:\Ortho4XP_Review\canopy_height"
OUT = r"E:\Ortho4XP_Review\veg_density\in_mask"
os.makedirs(OUT, exist_ok=True)
CELL_M = 64.0
MPP18 = 0.37
_seg = {}
_canopy = {}


def seg_probs(rgb):
    """(classes, P(tree)) from the production SegFormer; same patch grid."""
    import torch
    if not _seg:
        _seg["m"] = SEGFORMER.load_vegetation_model()
    model, proc, device = _seg["m"]
    P, ovlp = SEGFORMER.segformer_patch_size, SEGFORMER.segformer_overlap
    stride = P - 2 * ovlp
    H, W = rgb.shape[:2]
    rows = sorted(set(list(range(0, H - P + 1, stride)) + [max(0, H - P)]))
    cols = sorted(set(list(range(0, W - P + 1, stride)) + [max(0, W - P)]))
    win = np.outer(np.hanning(P), np.hanning(P)).astype(np.float32) + 1e-3
    acc = np.zeros((model.config.num_labels, H, W), np.float32)
    wt = np.zeros((H, W), np.float32)
    x = ((rgb.astype(np.float32) / 255 - D.MEAN) / D.STD).transpose(2, 0, 1)
    coords = [(r, c) for r in rows for c in cols]
    dtype = next(model.parameters()).dtype
    for i in range(0, len(coords), 8):
        part = coords[i:i + 8]
        t = torch.from_numpy(np.stack([x[:, r:r + P, c:c + P] for r, c in part])).to(device, dtype)
        with torch.inference_mode():
            lo = model(pixel_values=t).logits
            lo = torch.nn.functional.interpolate(lo.float(), size=(P, P), mode="bilinear", align_corners=False)
            p = torch.softmax(lo, 1).cpu().numpy()
        for (r, c), q in zip(part, p):
            acc[:, r:r + P, c:c + P] += q * win
            wt[r:r + P, c:c + P] += win
    acc /= wt
    return acc.argmax(0).astype(np.uint8), acc[SEGFORMER.CLASS_TREE]


def canopy_maps(rgb):
    if "m" not in _canopy:
        _canopy["m"] = CANOPY.CanopyModel.load(r"E:\Ortho4XP_Review\veg_alt\chm_model\ckpt")
    m = _canopy["m"]
    if m is None:
        return None
    return m.maps(rgb, CANOPY.INFER_ZL).astype(np.float32)


def percentile_map(lum, mask, cell_px, q):
    """Local q-th percentile of lum over mask pixels, on a coarse grid, bilinear."""
    h, w = lum.shape
    ny, nx = max(1, h // cell_px), max(1, w // cell_px)
    glob = np.percentile(lum[mask], q) if mask.any() else 0.2
    out = np.full((ny, nx), np.nan, np.float32)
    for j in range(ny):
        for i in range(nx):
            s = (slice(j * h // ny, (j + 1) * h // ny), slice(i * w // nx, (i + 1) * w // nx))
            v = lum[s][mask[s]]
            if v.size >= 0.02 * lum[s].size:
                out[j, i] = np.percentile(v, q)
    out = np.where(np.isnan(out), glob, out)
    if ny == 1 and nx == 1:
        return np.full((h, w), float(out[0, 0]), np.float32)
    out = cv2.GaussianBlur(out, (0, 0), 1.0)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)


def gap_family(rgb, mask, mpp):
    """Canopy reference from the darker part of the masked pixels; brighter = ground."""
    f = rgb.astype(np.float32) / 255
    r, g, b = f.transpose(2, 0, 1)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    on = mask > 0
    out = {}
    for name, cell_m in (("tex", 0), ("600m", 600)):
        ref = percentile_map(lum, on, int(cell_m / mpp) if cell_m else 10 ** 6, 30)
        for k in (1.25, 1.5, 2.0):
            out[f"gap_{name}_k{k}"] = (lum < k * ref).astype(np.float32)
    return out


def in_mask_cells(a, mask, cell):
    num = D.cell_means(a * mask, cell)
    den = D.cell_means(mask.astype(np.float32), cell)
    return num / np.maximum(den, 1e-5), den


rows = []
want = sys.argv[1:] or ["schulz", "zeuthen"]
for site in want:
    rgb18 = np.array(Image.open(os.path.join(ROOT, f"img_{site}.png")).convert("RGB"))
    lidar = np.load(os.path.join(ROOT, "truth", f"lidar_{site}.npy")).astype(np.float32)
    for tag, f in (("z18", 1), ("z17", 2), ("z16", 4), ("z16low", 6)):
        size = 4096 // f
        rgb = cv2.resize(rgb18, (size, size), interpolation=cv2.INTER_AREA) if f > 1 else rgb18
        mpp = MPP18 * 4096 / size
        classes, ptree = seg_probs(rgb)
        mask = LOWZOOM.tree_mask(rgb, classes, mpp).astype(np.float32)
        truth_px = cv2.resize((lidar >= 2).astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
        cell = int(round(CELL_M / mpp))
        truth, share = in_mask_cells(truth_px, mask, cell)
        keep = share >= 0.25
        est = {"seg_ptree": ptree}
        for label, net, up in (("tcd_b2_x1", "tcd_b2", 1.0), ("tcd_b2_x2", "tcd_b2", 2.0), ("tcd_b0_x1", "tcd_b0", 1.0)):
            est[label] = D.tcd_prob(rgb, net, up)
        est.update(gap_family(rgb, mask, mpp))
        c = canopy_maps(rgb)
        if c is not None:
            c = np.stack([cv2.resize(ch, (size, size), interpolation=cv2.INTER_LINEAR) for ch in c])
            est["canopy_mean_h"] = np.clip(c[0] / 10, 0, 1)
            est["canopy_P2"] = c[1 + CANOPY.LEVELS_M.index(2)]
        print(f"--- {site} {tag} mpp={mpp:.2f} cells={int(keep.sum())} mean true cover in mask={truth[keep].mean():.2f} "
              f"p10-90={np.percentile(truth[keep], 10):.2f}-{np.percentile(truth[keep], 90):.2f} "
              f"flat MAE={np.abs(truth[keep] - truth[keep].mean()).mean():.3f}", flush=True)
        for label, a in est.items():
            e, _ = in_mask_cells(a, mask, cell)
            r, mae, n, mae_fit = D.score(e, truth, keep)
            rows.append((site, tag, label, r, mae, mae_fit))
            print(f"   {label:22s} r={r:5.2f} mae_fit={mae_fit:5.3f} mean={e[keep].mean():.2f}", flush=True)
np.save(os.path.join(OUT, "rows_" + "_".join(want) + ".npy"), np.array(rows, dtype=object), allow_pickle=True)
