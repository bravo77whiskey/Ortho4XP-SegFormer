"""LiDAR score of closure settings and hybrids with P(tree); caches the SegFormer output."""
import itertools
import os

import cv2
import numpy as np
from PIL import Image

import bimodal as B
import density_lib as D
import eval_in_mask_lib as L
import O4_Veg_Lowzoom as LOWZOOM

ROOT = r"E:\Ortho4XP_Review\canopy_height"
CACHE = r"E:\Ortho4XP_Review\veg_density\in_mask\cache"
os.makedirs(CACHE, exist_ok=True)
CELL_M = 64.0
SCALES = (("z18", 1), ("z17", 2), ("z16", 4), ("z16low", 6))
data = {}
for site in ("schulz", "zeuthen"):
    rgb18 = np.array(Image.open(os.path.join(ROOT, f"img_{site}.png")).convert("RGB"))
    lidar = np.load(os.path.join(ROOT, "truth", f"lidar_{site}.npy")).astype(np.float32)
    for tag, f in SCALES:
        size = 4096 // f
        rgb = cv2.resize(rgb18, (size, size), interpolation=cv2.INTER_AREA) if f > 1 else rgb18
        mpp = 0.37 * 4096 / size
        path = os.path.join(CACHE, f"{site}_{tag}.npz")
        if os.path.exists(path):
            z = np.load(path)
            ptree, mask = z["ptree"], z["mask"]
        else:
            classes, ptree = L.seg_probs(rgb)
            mask = LOWZOOM.tree_mask(rgb, classes, mpp).astype(np.float32)
            np.savez_compressed(path, ptree=ptree.astype(np.float16), mask=mask.astype(np.uint8))
        ptree = ptree.astype(np.float32)
        mask = mask.astype(np.float32)
        truth_px = cv2.resize((lidar >= 2).astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
        data[site, tag] = (rgb, ptree, mask, truth_px, mpp)


def report(label, fn):
    line = f"{label:44s}"
    rs = []
    for (site, tag), (rgb, ptree, mask, truth_px, mpp) in data.items():
        cell = int(round(CELL_M / mpp))
        truth, share = L.in_mask_cells(truth_px, mask, cell)
        keep = share >= 0.25
        e, _ = L.in_mask_cells(fn(rgb, ptree, mask, mpp), mask, cell)
        r, _m, _n, mae_fit = D.score(e, truth, keep)
        rs.append(r)
        line += f" {site[0]}{tag[1:]}={r:5.2f}/{mae_fit:.2f}/{e[keep].mean():.2f}"
    print(line + f"  | mean r {np.nanmean(rs):.2f}", flush=True)


print("site+scale = r / MAE after linear fit / mean estimate")
for site, tag in data:
    rgb, ptree, mask, truth_px, mpp = data[site, tag]
    t, share = L.in_mask_cells(truth_px, mask, int(round(CELL_M / mpp)))
    print(site, tag, "true cover in mask", round(float(t[share >= 0.25].mean()), 2))
report("P(tree)", lambda rgb, p, m, mpp: p)
for window, c0, c1, mid in itertools.product((96, 192), (0.12, 0.2), (0.3, 0.45), (0.5, 0.65)):
    report(f"closure w{window} c{c0}-{c1} mid{mid}",
           lambda rgb, p, m, mpp, w=window, a=c0, b=c1, c=mid: B.closure(rgb, m, mpp, w, a, b, c)[0])
for window, c0, c1, mid in ((192, 0.12, 0.3, 0.5), (192, 0.2, 0.45, 0.5), (96, 0.12, 0.3, 0.65)):
    close = lambda rgb, m, mpp, w=window, a=c0, b=c1, c=mid: B.closure(rgb, m, mpp, w, a, b, c)[0]
    report(f"max(P, closure {window},{c0},{c1},{mid})", lambda rgb, p, m, mpp, f=close: np.maximum(p, f(rgb, m, mpp)))
    report(f"mean(P, closure {window},{c0},{c1},{mid})", lambda rgb, p, m, mpp, f=close: 0.5 * (p + f(rgb, m, mpp)))
