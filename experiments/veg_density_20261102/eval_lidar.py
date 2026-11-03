"""Score tree-cover estimators against 2019 Brandenburg LiDAR, at ZL18/17/16 scale."""
import os
import numpy as np
import cv2
from PIL import Image

import density_lib as D

OUT = r"E:\Ortho4XP_Review\veg_density\lidar"
os.makedirs(OUT, exist_ok=True)
ROOT = r"E:\Ortho4XP_Review\canopy_height"
MPP18 = 0.37
CELL_M = 64.0
rows = []
for site in ("schulz", "zeuthen"):
    rgb18 = np.array(Image.open(os.path.join(ROOT, f"img_{site}.png")).convert("RGB"))
    lidar = np.load(os.path.join(ROOT, "truth", f"lidar_{site}.npy")).astype(np.float32)
    canopy = np.load(os.path.join(ROOT, f"canopymaps_{site}.npy"))
    for zl, f in ((18, 1), (17, 2), (16, 4)):
        size = 4096 // f
        rgb = cv2.resize(rgb18, (size, size), interpolation=cv2.INTER_AREA) if f > 1 else rgb18
        mpp = MPP18 * f
        cell = int(round(CELL_M / mpp))
        truth = D.cell_means((lidar >= 2).astype(np.float32), int(round(CELL_M / MPP18)))
        est = {}
        times = {}
        for label, name, up in (
            ("tcd_b0_x1", "tcd_b0", 1.0),
            ("tcd_b0_x2", "tcd_b0", 2.0),
            ("tcd_b2_x1", "tcd_b2", 1.0),
            ("tcd_b2_x2", "tcd_b2", 2.0),
        ):
            if size * up > 8192:
                continue
            est[label], times[label] = D.timed(D.tcd_prob, rgb, name, up)
        est["rgb_mask"], times["rgb_mask"] = D.timed(D.rgb_mask, rgb, mpp)
        est["green_dark"], times["green_dark"] = D.timed(D.green_dark, rgb, mpp)
        est["rough"], times["rough"] = D.timed(D.rough, rgb, mpp)
        if zl == 18:
            est["canopy_mean>=2"] = (canopy[0] >= 2).astype(np.float32)
        for label, a in est.items():
            e = D.cell_means(a, cell)
            n = min(e.shape[0], truth.shape[0]), min(e.shape[1], truth.shape[1])
            ee, tt = e[:n[0], :n[1]], truth[:n[0], :n[1]]
            # Only where something is planted today or truly grows
            keep = (tt >= 0.05) | (D.cell_means(est["rgb_mask"], cell)[:n[0], :n[1]] >= 0.05)
            r, mae, count, mae_fit = D.score(ee, tt, keep)
            rows.append((site, zl, label, r, mae, mae_fit, count, times.get(label, 0)))
            print(f"{site:8s} z{zl} {label:16s} r={r:5.2f} mae={mae:5.3f} mae_fit={mae_fit:5.3f} cells={count:5d} t={times.get(label, 0):5.1f}s", flush=True)
        np.savez_compressed(os.path.join(OUT, f"{site}_z{zl}.npz"), **{k: cv2.resize(v, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float16) for k, v in est.items()})
np.save(os.path.join(OUT, "rows.npy"), np.array(rows, dtype=object), allow_pickle=True)
