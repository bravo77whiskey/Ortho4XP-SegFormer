"""Side-by-side crops: texture and each estimator's tree-cover map, for looking at."""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

import density_lib as D

Image.MAX_IMAGE_PIXELS = None
OUT = r"E:\Ortho4XP_Review\veg_density\sheets"
SETS = r"E:\Ortho4XP_Review\sets"

# (name, file, crop x, y): mixed density on purpose
SITES = [
    ("gc_pine_a", r"gran-canaria\BI\+27-016_27440_29920_BI16.jpg", 1000, 1000),
    ("gc_pine_b", r"gran-canaria\BI\+28-016_27440_29936_BI16.jpg", 1500, 1500),
    ("gc_z17", r"gran-canaria\BI\+27-016_54912_59888_BI17.jpg", 1000, 2000),
    ("szx_hill", r"hotspots\BI\28512_53488_BI16.jpg", 1500, 1500),
    ("khh", r"hotspots\BI\28544_54672_BI16.jpg", 2500, 500),
    ("khh_n", r"hotspots\BI\28528_54672_BI16.jpg", 1500, 1500),
]


def mpp_of(path):
    stem = os.path.basename(path).split("_")
    y = int(stem[-3])
    zl = int(stem[-1].split(".")[0][-2:])
    n = 2 ** zl
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + 8) / n))))
    return 156543.03 * np.cos(np.radians(lat)) / n * 256 / 256 * (n / 2 ** zl) * 16 * 256 / 4096 / (2 ** (zl - zl)) / (2 ** 0) * (1 / 1) if False else 40075016.7 * np.cos(np.radians(lat)) / n / 256


def heat(a):
    return cv2.cvtColor(cv2.applyColorMap((np.clip(a, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)


def panel(items, path, size=640):
    n = len(items)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * size, rows * (size + 18)), (18, 18, 18))
    d = ImageDraw.Draw(sheet)
    for i, (label, im) in enumerate(items):
        im = Image.fromarray(im).resize((size, size), Image.LANCZOS)
        x, y = (i % cols) * size, (i // cols) * (size + 18)
        sheet.paste(im, (x, y + 18))
        d.text((x + 4, y + 3), label, fill=(240, 240, 180))
    sheet.save(path, quality=90)


if __name__ == "__main__":
    want = set(sys.argv[1:])
    crop = 1024
    for name, rel, cx, cy in SITES:
        if want and name not in want:
            continue
        path = os.path.join(SETS, rel)
        rgb = np.array(Image.open(path).convert("RGB"))
        mpp = mpp_of(rel)
        est = {}
        times = {}
        for label, net, up in (("tcd_b0_x2", "tcd_b0", 2.0), ("tcd_b2_x1", "tcd_b2", 1.0), ("tcd_b2_x2", "tcd_b2", 2.0)):
            est[label], times[label] = D.timed(D.tcd_prob, rgb, net, up)
        est["rgb_mask"], times["rgb_mask"] = D.timed(D.rgb_mask, rgb, mpp)
        est["green_dark"], times["green_dark"] = D.timed(D.green_dark, rgb, mpp)
        x = cx, cx + crop
        y = cy, cy + crop
        items = [("texture mpp=%.2f" % mpp, rgb[y[0]:y[1], x[0]:x[1]])]
        for label, a in est.items():
            items.append((f"{label} {times[label]:.1f}s", heat(a[y[0]:y[1], x[0]:x[1]])))
        cell = max(3, int(round(64 / mpp)))
        for label in ("tcd_b2_x2", "rgb_mask", "green_dark"):
            smooth = cv2.blur(est[label], (cell, cell))
            items.append((label + " 64 m cover", heat(smooth[y[0]:y[1], x[0]:x[1]])))
        panel(items, os.path.join(OUT, f"sheet_{name}.jpg"))
        print(name, {k: round(v, 1) for k, v in times.items()}, flush=True)
