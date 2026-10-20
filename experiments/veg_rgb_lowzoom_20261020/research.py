"""Offline RGB-only vegetation experiments. No production imports or model input.

Run with the repository's existing .venv. Images are read from local review sets.
Thresholds are hypotheses, not calibrated probabilities or validated tree labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def window(metres, mpp):
    return max(3, int(round(metres / mpp)) | 1)


def mean(a, metres, mpp):
    k = window(metres, mpp)
    return cv2.blur(a.astype(np.float32), (k, k), borderType=cv2.BORDER_REFLECT)


def std(a, metres, mpp):
    mu = mean(a, metres, mpp)
    return np.sqrt(np.maximum(0, mean(a * a, metres, mpp) - mu * mu))


def remove_small(mask, mpp, min_m2=60):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] * mpp * mpp >= min_m2
    keep[0] = False
    return keep[labels]


def detect(rgb, mpp):
    """Return independent candidates at native resolution, given metres/pixel."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected HxWx3 uint8 RGB")
    if not math.isfinite(mpp) or mpp <= 0:
        raise ValueError("metres/pixel must be positive and finite")
    f = rgb.astype(np.float32) / 255
    r, g, b = f.transpose(2, 0, 1)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    exg = (2 * g - r - b) / np.maximum(r + g + b, 0.06)
    # Blue/cyan rejection helps some water but cannot identify water reliably.
    green = (exg > 0.065) & (g > r * 1.025) & (g > b * 1.08) & (lum > 0.055)
    green_share = mean(green, 12, mpp)
    colour = remove_small(green, mpp)
    dark = remove_small(green & (mean(lum, 8, mpp) < 0.38), mpp)

    # Suppress compression noise; retain texture at tree-crown scales.
    smooth = cv2.GaussianBlur(lum, (0, 0), max(0.45, 0.7 / mpp))
    local_mean = mean(smooth, 18, mpp)
    local_std = std(smooth, 18, mpp)
    contrast = local_std / np.maximum(local_mean, 0.08)
    coarse = cv2.GaussianBlur(smooth, (0, 0), max(0.7, 3.0 / mpp))
    fine_energy = np.sqrt(np.maximum(0, mean((smooth - coarse) ** 2, 18, mpp)))
    detail_ratio = fine_energy / np.maximum(local_std, 0.004)

    # Structure tensor: long coherent edges/parallel field rows are suspicious.
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    xx, yy, xy = mean(gx * gx, 22, mpp), mean(gy * gy, 22, mpp), mean(gx * gy, 22, mpp)
    coherence = np.sqrt((xx - yy) ** 2 + 4 * xy * xy) / (xx + yy + 1e-7)
    eligible = green & (green_share > 0.55) & (local_mean < 0.50)
    textured = (eligible & (local_std > 0.012) & (contrast > 0.065)
                & (detail_ratio > 0.28) & (coherence < 0.80))
    texture = remove_small(textured, mpp)
    seed = (textured & (local_std > 0.020) & (contrast > 0.10)
            & (coherence < 0.65))
    seed = remove_small(seed, mpp, 40)
    # Geodesic reconstruction for at most 8 m. Stops at unsupported pixels;
    # avoids filling a complete field connected to a single tree.
    support = (eligible & (local_std > 0.009) & (contrast > 0.045)
               & (detail_ratio > 0.20) & (coherence < 0.88))
    grown = seed.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(max(1, int(round(8 / mpp)))):
        grown = cv2.dilate(grown, kernel) & support.astype(np.uint8)
    regions = remove_small(grown, mpp)

    # A diagnostic only: dark linear objects include canals and field edges.
    size = window(14, mpp)
    blackhat = cv2.morphologyEx(
        smooth, cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    rows = remove_small(green & (blackhat > 0.025) & (coherence > 0.45), mpp, 30)
    return {
        "green": colour, "dark": dark, "texture": texture, "regions": regions,
        "rows_diagnostic": rows,
    }


def metres_per_pixel(sample, shape):
    south, west, north, east = sample["bbox"]
    radius = 6378137.0
    y = math.radians(north - south) * radius / shape[0]
    x = (math.radians(east - west) * radius
         * math.cos(math.radians((north + south) / 2)) / shape[1])
    return (x + y) / 2


def samples(root):
    selections = {
        "hotspots": None,
        "taipei": {
            "+24+121_28064_54944_Arc16", "+24+121_28080_54896_Arc16",
            "+24+121_28112_54832_Arc16", "+24+121_28256_54784_Arc16",
            "+25+121_56096_109664_BI17", "+25+121_56112_109680_BI17",
        },
        "gran-canaria": {
            "+27-016_54912_59904_BI17", "+27-016_54928_59920_BI17",
            "+27-016_54960_59872_BI17", "+27-016_27440_29920_BI16",
            "+27-016_27472_29904_BI16", "+28-016_27424_29920_BI16",
        },
    }
    for name, keys in selections.items():
        folder = root / "sets" / name / "BI"
        found = set()
        for s in json.loads((folder / "samples.json").read_text()):
            if s["zl"] not in (16, 17) or (keys is not None and s["key"] not in keys):
                continue
            found.add(s["key"])
            yield dict(s, set=name, path=str(folder / s["file"]), id=name + "_" + s["key"])
        if keys is not None and keys - found:
            raise ValueError(f"Missing keys: {keys - found}")


def save_preview(rgb, masks, dest, side=1000):
    h, w = rgb.shape[:2]
    size = (round(w * side / max(h, w)), round(h * side / max(h, w)))
    small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    Image.fromarray(small).save(dest / "source.jpg", quality=90)
    for name, mask in masks.items():
        # Soft coverage when previewing: avoids dropping sparse rows on shrink.
        coverage = cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_AREA)
        rgba = np.zeros((*coverage.shape, 4), np.uint8)
        rgba[..., :3] = (255, 64, 180)
        rgba[..., 3] = np.clip(coverage * 210, 0, 255).astype(np.uint8)
        Image.fromarray(rgba).save(dest / (name + ".png"), compress_level=1)


def gallery(out, records):
    import shutil
    # Keep the full original image available for true native-resolution zoom.
    for rec in records:
        target = out / rec["id"] / "source_native.jpg"
        source = Path(rec["path"])
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != rec["sha256"]:
            shutil.copyfile(source, target)
    data = json.dumps(records).replace("</", "<\\/")
    template = Path(__file__).with_name("gallery.html").read_text(encoding="utf-8")
    (out / "index.html").write_text(template.replace("__RECORDS_JSON__", data), encoding="utf-8")


def run(root, out):
    out.mkdir(parents=True, exist_ok=True)
    records = []
    cv2.setNumThreads(4)
    selected = list(samples(root))
    for sample in selected:
        path = Path(sample["path"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rgb = np.asarray(Image.open(path).convert("RGB"))
        mpp = metres_per_pixel(sample, rgb.shape)
        started = time.perf_counter()
        masks = detect(rgb, mpp)
        elapsed = time.perf_counter() - started
        dest = out / sample["id"]
        dest.mkdir(exist_ok=True)
        save_preview(rgb, masks, dest)
        for name, mask in masks.items():
            Image.fromarray(mask).save(dest / (name + "_native.png"), compress_level=1)
        record = dict(sample, mpp=mpp, shape=list(rgb.shape[:2]), sha256=digest,
                      seconds=elapsed, coverage_pct={k: float(v.mean() * 100) for k, v in masks.items()})
        records.append(record)
        print(sample["id"], f"{elapsed:.2f}s", record["coverage_pct"], flush=True)
    (out / "metrics.json").write_text(json.dumps({
        "version": 1, "opencv": cv2.__version__, "numpy": np.__version__,
        "threads": 4, "timing_scope": "detect only; excludes image decode and output encoding",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "samples": records,
    }, indent=2), encoding="utf-8")
    gallery(out, records)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(r"E:\Ortho4XP_Review"))
    ap.add_argument("--out", type=Path, default=Path(r"E:\Ortho4XP_Review\rgb_lowzoom_20261020"))
    args = ap.parse_args()
    run(args.root, args.out)
