"""Other climates, no LiDAR: can a shrunk texture recover the cover seen at full zoom?

Reference: canopy model and TCD b2 on the real ZL18 texture (both track LiDAR with
r 0.84-0.91 in Brandenburg). Estimators see the same texture shrunk to the true
ZL16 and ZL17 pixel size, as a low-zoom texture would show it.
"""
import glob
import os

import cv2
import numpy as np
from PIL import Image

import density_lib as D
import eval_in_mask_lib as L
import O4_Canopy_Height as CANOPY
import O4_Veg_Lowzoom as LOWZOOM

SETS = r"E:\Ortho4XP_Review\sets"
OUT = r"E:\Ortho4XP_Review\veg_density\lowlat"
os.makedirs(OUT, exist_ok=True)
CELL_M = 64.0

files = []
for pat in (r"hotspots\BI\*BI18.jpg", r"gran-canaria\BI\*BI18.jpg", r"taipei\BI\*18.jpg"):
    found = sorted(glob.glob(os.path.join(SETS, pat)))
    files += found[:6]

all_rows = {}
for path in files:
    rel = os.path.basename(path)
    y = int(rel.split("_")[-2])
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + 8) / 2 ** 18))))
    mpp18 = 40075016.7 * np.cos(np.radians(lat)) / 2 ** 18 / 256
    rgb18 = np.array(Image.open(path).convert("RGB"))
    if rgb18.shape[0] != 4096:
        continue
    canopy = L.canopy_maps(rgb18)
    if canopy is None:
        raise SystemExit("no canopy model")
    ref_canopy = (canopy[0] >= 2).astype(np.float32)
    ref_canopy = cv2.resize(ref_canopy, (4096, 4096), interpolation=cv2.INTER_LINEAR)
    ref_tcd = D.tcd_prob(rgb18, "tcd_b2", 1.0)
    for tag, f in (("z17", 2), ("z16", 4)):
        size = 4096 // f
        rgb = cv2.resize(rgb18, (size, size), interpolation=cv2.INTER_AREA)
        mpp = mpp18 * f
        classes, ptree = L.seg_probs(rgb)
        mask = LOWZOOM.tree_mask(rgb, classes, mpp).astype(np.float32)
        cell = int(round(CELL_M / mpp))
        keep_share = D.cell_means(mask, cell) >= 0.25
        if keep_share.sum() < 20:
            continue
        refs = {}
        for name, ref in (("canopy", ref_canopy), ("tcd", ref_tcd)):
            small = cv2.resize(ref, (size, size), interpolation=cv2.INTER_AREA)
            refs[name], _ = L.in_mask_cells(small, mask, cell)
        est = {"seg_ptree": ptree}
        est.update(L.gap_family(rgb, mask, mpp))
        est["tcd_b2_x1"] = D.tcd_prob(rgb, "tcd_b2", 1.0)
        for label, a in est.items():
            e, _ = L.in_mask_cells(a, mask, cell)
            for name in refs:
                all_rows.setdefault((tag, label, name), []).append(
                    (e[keep_share].astype(np.float32), refs[name][keep_share].astype(np.float32), rel))
    print(rel, "done", flush=True)

print("\nper-estimator, per-reference (pooled over textures; per-texture r median)")
for key in sorted(all_rows):
    parts = all_rows[key]
    e = np.concatenate([p[0] for p in parts])
    t = np.concatenate([p[1] for p in parts])
    per = [np.corrcoef(p[0], p[1])[0, 1] for p in parts if p[0].size >= 20 and p[0].std() > 1e-6 and p[1].std() > 1e-6]
    print(f"{key[0]} {key[1]:20s} vs {key[2]:6s} pooled r={np.corrcoef(e, t)[0, 1]:5.2f} median tex r={np.median(per):5.2f} "
          f"texs={len(per)} cells={e.size} mean est={e.mean():.2f} mean ref={t.mean():.2f}", flush=True)
