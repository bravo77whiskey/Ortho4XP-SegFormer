"""Small visual spot checks and resolution sensitivity, not a labelled benchmark."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import research


OUT = Path(r"E:\Ortho4XP_Review\rgb_lowzoom_20261020")
HERE = Path(__file__).resolve().parent
# Coordinates were selected by inspecting original RGB on a 1024x1024 grid.
# Regions are deliberately interior patches. They are visual judgements, not
# surveyed land-cover labels. No thresholds are fitted to these patches.
PATCHES = [
    ("khh_nw", "tree", "coastal wooded hill", [275, 165, 310, 200]),
    ("khh_nw", "tree", "coastal wooded hill south", [325, 270, 360, 305]),
    ("khh_nw", "non_tree", "open sea", [25, 525, 100, 600]),
    ("khh_nw", "non_tree", "harbour water", [420, 730, 455, 755]),
    ("khh_nw", "non_tree", "dense buildings", [805, 410, 845, 435]),
    ("szx_e", "tree", "dark woodland north", [160, 50, 210, 100]),
    ("szx_e", "tree", "dark woodland centre", [300, 155, 350, 205]),
    ("szx_e", "tree", "reservoir wooded island", [800, 415, 850, 465]),
    ("szx_e", "non_tree", "reservoir water", [700, 440, 730, 490]),
    ("szx_e", "non_tree", "reservoir water south", [792, 715, 835, 740]),
    ("szx_e", "non_tree", "industrial roofs", [450, 175, 478, 210]),
    ("wnz_z17", "non_tree", "dark crop field", [345, 160, 365, 220]),
    ("wnz_z17", "non_tree", "parallel crop beds", [425, 140, 470, 190]),
    ("wnz_z17", "non_tree", "muddy river", [935, 20, 990, 90]),
    ("wnz_z17", "non_tree", "canal", [227, 430, 233, 480]),
    ("wnz_z17", "non_tree", "green water", [700, 805, 740, 855]),
    ("wnz_z17", "tree", "small roadside tree clump", [610, 545, 628, 570]),
]


def spot_checks(records):
    rows = []
    sheet = Image.new("RGB", (1000, ((len(PATCHES) + 3) // 4) * 280), "#202124")
    draw = ImageDraw.Draw(sheet)
    for i, (key, label, desc, box) in enumerate(PATCHES):
        rec = next(r for r in records if r["set"] == "hotspots" and r["key"] == key)
        rgb = Image.open(rec["path"]).convert("RGB")
        w, h = rgb.size
        x0, y0, x1, y1 = box
        native = (round(x0 * w / 1024), round(y0 * h / 1024),
                  round(x1 * w / 1024), round(y1 * h / 1024))
        crop = rgb.crop(native)
        crop.thumbnail((240, 220))
        x, y = i % 4 * 250, i // 4 * 280
        sheet.paste(crop, (x, y))
        draw.text((x + 3, y + 223), f"{i+1}: {label}", fill="white")
        draw.text((x + 3, y + 241), desc, fill="white")
        draw.text((x + 3, y + 259), key, fill="white")
        coverage = {}
        for method in rec["coverage_pct"]:
            mask = np.asarray(Image.open(OUT / rec["id"] / (method + "_native.png")))
            a, b, c, d = native
            coverage[method] = float(mask[b:d, a:c].mean() * 100)
        rows.append(dict(key=key, label=label, description=desc,
                         box_on_1024=box, native_box=native, coverage_pct=coverage))
    sheet.save(HERE / "reference_patches.jpg", quality=95)
    summary = {}
    for label in ("tree", "non_tree"):
        subset = [r for r in rows if r["label"] == label]
        summary[label] = {
            k: float(np.mean([r["coverage_pct"][k] for r in subset]))
            for k in subset[0]["coverage_pct"]
        }
    return {"warning": "Equal-weight mean patch coverage, not precision/recall. "
                       "Visual reference patches are small, subjective, selected examples.",
            "patches": rows, "mean_patch_coverage_pct": summary}


def consistency(records):
    result = []
    for rec in records:
        if rec["zl"] != 17:
            continue
        rgb = np.asarray(Image.open(rec["path"]).convert("RGB"))
        h, w = rgb.shape[:2]
        small = cv2.resize(rgb, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        masks = research.detect(small, rec["mpp"] * 2)
        if "field_suppressed" in rec["coverage_pct"]:
            from refine_direction import refine
            masks["field_suppressed"] = refine(small, rec["mpp"] * 2, masks["texture"])
        metrics = {}
        for name, small_mask in masks.items():
            native = np.asarray(Image.open(OUT / rec["id"] / (name + "_native.png")))
            reference = cv2.resize(native.astype(np.float32), (w // 2, h // 2),
                                   interpolation=cv2.INTER_AREA) >= 0.5
            union = (reference | small_mask).sum()
            metrics[name] = {
                "iou": float((reference & small_mask).sum() / union) if union else None,
                "coverage_delta_percentage_points": float((small_mask.mean() - reference.mean()) * 100),
            }
        result.append(dict(id=rec["id"], metrics=metrics))
        print("scale check", rec["key"], flush=True)
    return {"warning": "Synthetic ZL16 from native ZL17: consistency only, not accuracy "
                       "or equivalence to separately acquired provider ZL16 imagery.",
            "samples": result}


def sanity():
    # Negative-control images must not produce textured tree candidates.
    results = {}
    for name, colour in {"black": [0, 0, 0], "white": [255, 255, 255],
                         "flat_green": [50, 110, 45], "flat_water": [35, 75, 70],
                         "grey": [90, 90, 90]}.items():
        img = np.full((128, 128, 3), colour, dtype=np.uint8)
        masks = research.detect(img, 2.2)
        assert not masks["texture"].any(), name
        assert not masks["regions"].any(), name
        assert all(m.shape == img.shape[:2] and m.dtype == bool for m in masks.values())
        results[name] = "passed"
    return results


if __name__ == "__main__":
    cv2.setNumThreads(4)
    records = json.loads((OUT / "metrics.json").read_text())["samples"]
    results = {"sanity": sanity(), "spot_checks": spot_checks(records),
               "resolution_consistency": consistency(records)}
    (OUT / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["spot_checks"]["mean_patch_coverage_pct"], indent=2))
