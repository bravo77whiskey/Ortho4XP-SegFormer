"""The LiDAR sites and scorer of eval_unmix, importable without running its grid."""
import os

import cv2
import numpy as np
from PIL import Image

import density_lib as D
import eval_in_mask_lib as L

ROOT = r"E:\Ortho4XP_Review\canopy_height"
CACHE = r"E:\Ortho4XP_Review\veg_density\in_mask\cache"
CELL_M = 64.0
SCALES = (("z18", 1), ("z17", 2), ("z16", 4), ("z16low", 6))
data = {}
for site in ("schulz", "zeuthen"):
    rgb18 = np.array(Image.open(os.path.join(ROOT, f"img_{site}.png")).convert("RGB"))
    lidar = np.load(os.path.join(ROOT, "truth", f"lidar_{site}.npy")).astype(np.float32)
    for tag, f in SCALES:
        size = 4096 // f
        rgb = cv2.resize(rgb18, (size, size), interpolation=cv2.INTER_AREA) if f > 1 else rgb18
        z = np.load(os.path.join(CACHE, f"{site}_{tag}.npz"))
        truth_px = cv2.resize((lidar >= 2).astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
        data[site, tag] = (rgb, z["ptree"].astype(np.float32), z["mask"].astype(np.float32), truth_px, 0.37 * 4096 / size)


def report(label, fn):
    line = f"{label:26s}"
    rs = []
    for (site, tag), (rgb, ptree, mask, truth_px, mpp) in data.items():
        cell = int(round(CELL_M / mpp))
        truth, share = L.in_mask_cells(truth_px, mask, cell)
        keep = share >= 0.25
        e, _ = L.in_mask_cells(fn(rgb, ptree, mask, mpp), mask, cell)
        r, _m, _n, _f = D.score(e, truth, keep)
        rs.append(r)
        line += f" {site[0]}{tag[1:]}={r:4.2f}/{np.abs(e[keep] - truth[keep]).mean():.2f}/{e[keep].mean():.2f}"
    print(line + f" | mean r {np.nanmean(rs):.2f}", flush=True)
