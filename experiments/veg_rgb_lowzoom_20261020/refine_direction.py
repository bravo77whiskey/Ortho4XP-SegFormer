"""Exploratory crop-row suppression, informed by the visual patch diagnostics.

Runs after research.py. Reuses only its RGB-derived texture mask, never model
output. This is an in-sample ablation, not an independently validated improvement.
"""
import hashlib
import json
import time

import cv2
import numpy as np
from PIL import Image

from evaluate import OUT
from research import gallery, mean, remove_small, save_preview


def refine(rgb, mpp, texture):
    f = rgb.astype(np.float32) / 255
    lum = f @ np.array([0.299, 0.587, 0.114], np.float32)
    smooth = cv2.GaussianBlur(lum, (0, 0), max(0.45, 0.7 / mpp))
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    aligned = np.zeros(texture.shape, bool)
    for metres in (50, 80):
        xx, yy, xy = [mean(v, metres, mpp) for v in (gx * gx, gy * gy, gx * gy)]
        coherence = np.sqrt((xx - yy) ** 2 + 4 * xy * xy) / (xx + yy + 1e-7)
        aligned |= coherence > 0.55
    return remove_small(texture & ~aligned, mpp)


if __name__ == "__main__":
    cv2.setNumThreads(4)
    payload = json.loads((OUT / "metrics.json").read_text())
    for rec in payload["samples"]:
        dest = OUT / rec["id"]
        rgb = np.array(Image.open(rec["path"]).convert("RGB"))
        texture = np.array(Image.open(dest / "texture_native.png")).astype(bool)
        start = time.perf_counter()
        result = refine(rgb, rec["mpp"], texture)
        rec["direction_refine_seconds"] = time.perf_counter() - start
        Image.fromarray(result).save(dest / "field_suppressed_native.png", compress_level=1)
        save_preview(rgb, {"field_suppressed": result}, dest)
        rec["coverage_pct"]["field_suppressed"] = float(result.mean() * 100)
        print(rec["key"], round(rec["coverage_pct"]["texture"], 2),
              "->", round(rec["coverage_pct"]["field_suppressed"], 2), flush=True)
    payload["direction_refine_sha256"] = hashlib.sha256(
        __import__("pathlib").Path(__file__).read_bytes()).hexdigest()
    payload["direction_refine_note"] = "Thresholds explored on visual reference patches; not held-out accuracy."
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    gallery(OUT, payload["samples"])
