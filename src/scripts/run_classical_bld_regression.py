"""Visual regression harness for the classical (non-AI) building detector.

Runs O4_SFR_Classical_Buildings on the standard test textures and renders
confidence-band colour-coded overlays for visual judgement, plus optional
side-by-side composites against the YOLO OBB overlays.

Needs neither torch nor ultralytics (unless --segformer is passed).

Fast iteration on a single texture:
  python src/scripts/run_classical_bld_regression.py --label v1 --only 25680_54080

Full 8-texture regression with contact sheet:
  python src/scripts/run_classical_bld_regression.py --label v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_ROOT = Path(r"G:\Dev\Ortho4XP-SegFormer")
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import O4_SFR_Classical_Buildings as CLASSICAL  # noqa: E402

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "tmp" / "classical_bld_tests"
DEFAULT_YOLO_DIR = PROJECT_ROOT / "tmp" / "yolo_obb_tests" / "latest_step_20000_nonheight"

# Standard test texture set, copied from
# .agents/skills/ortho4xp-yolo-obb-training/scripts/run_texture_regression.py
# (importing it would pull in torch + ultralytics).
DEFAULT_SOURCES = [
    Path(r"G:\Dev\Ortho4XP-SegFormer\tmp\sfr_debug_images_full\28512_53472_BI16\28512_53472_BI16_source.png"),
    Path(r"H:\XP12 Addons\Applications\Ortho4XP\Tiles\zOrtho4XP_+36+117\textures\25680_54080_BI16.dds"),
    Path(r"H:\XP12 Addons\Applications\Ortho4XP\Tiles\zOrtho4XP_+36+117\textures\25680_54096_BI16.dds"),
    Path(r"G:\XP12 Addons2\Custom Scenery\zOrtho4XP_+22+120\textures\28528_54656_Arc16.dds"),
    Path(r"G:\XP12 Addons2\Custom Scenery\zOrtho4XP_+25+121\textures\28032_54832_Arc16.dds"),
    Path(r"G:\XP12 Addons2\Custom Scenery\zOrtho4XP_+25+121\textures\28048_54880_Arc16.dds"),
    Path(r"H:\XP12 Addons\Applications\Ortho4XP\Tiles\zOrtho4XP_+37+126\textures\203056_446832_Arc19.dds"),
    Path(r"H:\XP12 Addons\Applications\Ortho4XP\Tiles\zOrtho4XP_+37+126\textures\203024_446816_Arc19.dds"),
]

# Low -> high confidence band colours (orange, yellow, green, then extras).
BAND_COLORS = [
    (255, 150, 60),
    (255, 225, 70),
    (90, 235, 130),
    (110, 200, 255),
    (235, 120, 235),
]
SUB_BAND_COLOR = (150, 150, 150)  # detections below the first band edge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--sources", type=Path, nargs="+", default=None,
                        help="Override DEFAULT_SOURCES with explicit image paths")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Filter sources to stems containing any of these "
                             "substrings (fast single-texture iteration)")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--conf-bands", default="0.1,0.3,0.5",
                        help="Ascending band edges; boxes are coloured per band")
    parser.add_argument("--floor-conf", type=float, default=None,
                        help="Detector confidence floor; defaults to the first "
                             "band edge. Below-band detections draw grey.")
    parser.add_argument("--max-det", type=int, default=100000)
    parser.add_argument("--m-per-px", type=float, default=None,
                        help="Override for sources whose stem does not encode "
                             "tile coords + ZL")
    parser.add_argument("--veg-npy-dir", type=Path, default=None,
                        help="Directory with pipeline {stem}_veg.npy caches to "
                             "use as the landcover prior")
    parser.add_argument("--segformer", action="store_true",
                        help="Run SegFormer for missing landcover maps "
                             "(requires the torch venv)")
    parser.add_argument("--yolo-dir", type=Path, default=DEFAULT_YOLO_DIR,
                        help="YOLO regression output dir for side-by-side "
                             "composites")
    parser.add_argument("--no-sbs", action="store_true",
                        help="Skip side-by-side composites")
    parser.add_argument("--sbs-width", type=int, default=1600)
    parser.add_argument("--contact-thumb-size", type=int, default=1200)
    parser.add_argument("--save-debug-maps", action="store_true",
                        help="Save evidence/suppress/candidate maps per texture")
    parser.add_argument("--save-source", action="store_true",
                        help="Also save the decoded source PNG (large)")
    parser.add_argument("--param", action="append", default=[],
                        help="ClassicalParams override, e.g. --param evid_floor=0.15 "
                             "(repeatable; tuples as comma lists)")
    return parser.parse_args()


# copied from run_texture_regression.py
def stem_for(path: Path) -> str:
    if path.name.endswith("_source.png"):
        return path.name.removesuffix("_source.png")
    return path.stem


# copied from run_texture_regression.py
def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def text_size(draw: ImageDraw.ImageDraw, text: str) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def apply_param_overrides(params: CLASSICAL.ClassicalParams, overrides: list[str]):
    applied = {}
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--param expects key=value, got: {item}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not hasattr(params, key):
            raise SystemExit(f"Unknown ClassicalParams field: {key}")
        current = getattr(params, key)
        if isinstance(current, tuple):
            value: Any = tuple(float(x) for x in raw.split(","))
        elif isinstance(current, bool):
            value = raw.strip().lower() in ("1", "true", "yes")
        elif isinstance(current, int):
            value = int(raw)
        elif isinstance(current, dict):
            value = json.loads(raw)
        elif isinstance(current, str):
            value = raw.strip()
        else:
            value = float(raw)
        setattr(params, key, value)
        applied[key] = value
    return applied


def parse_bands(spec: str) -> list[float]:
    edges = sorted(float(x) for x in spec.split(",") if x.strip())
    if not edges:
        raise SystemExit("--conf-bands needs at least one edge")
    return edges


def band_index(confidence: float, edges: list[float]) -> int:
    """-1 below the first edge, else highest edge index <= confidence."""
    idx = -1
    for i, edge in enumerate(edges):
        if confidence >= edge:
            idx = i
    return idx


def band_color(idx: int) -> tuple[int, int, int]:
    if idx < 0:
        return SUB_BAND_COLOR
    return BAND_COLORS[min(idx, len(BAND_COLORS) - 1)]


def band_label(idx: int, edges: list[float]) -> str:
    if idx < 0:
        return f"<{edges[0]:g}"
    if idx == len(edges) - 1:
        return f">={edges[idx]:g}"
    return f"{edges[idx]:g}-{edges[idx + 1]:g}"


def draw_overlay(image: Image.Image, detections: list[dict],
                 edges: list[float]) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    # Low-confidence first so strong detections draw on top.
    for det in sorted(detections, key=lambda d: d["confidence"]):
        points = [tuple(p) for p in det["points"]]
        color = band_color(band_index(det["confidence"], edges))
        draw.polygon(points, outline=(*color, 235), fill=(*color, 38))
        draw.line(points + [points[0]], fill=(*color, 245), width=2)
    return overlay


def append_band_legend(image: Image.Image, edges: list[float],
                       counts: dict[str, int]) -> Image.Image:
    legend_h = 34
    out = Image.new("RGB", (image.width, image.height + legend_h), (22, 22, 26))
    out.paste(image, (0, 0))
    draw = ImageDraw.Draw(out)
    x = 12
    y = image.height + 8
    indices = ([-1] if counts.get(band_label(-1, edges)) else []) + list(range(len(edges)))
    for idx in indices:
        label = band_label(idx, edges)
        text = f"conf {label}: {counts.get(label, 0)}"
        color = band_color(idx)
        draw.rectangle((x, y, x + 16, y + 16), fill=color)
        draw.text((x + 22, y + 2), text, fill=(240, 240, 240))
        tw, _ = text_size(draw, text)
        x += 22 + tw + 28
    return out


def make_side_by_side(classical_overlay: Path, yolo_dir: Path, stem: str,
                      label: str, out_path: Path, width: int):
    yolo_candidates = sorted(yolo_dir.glob(f"{stem}_*_overlay.png")) if yolo_dir else []
    if not yolo_candidates:
        return None
    header_h = 34
    gap = 8
    with Image.open(classical_overlay) as left_img:
        left = left_img.convert("RGB")
        left.thumbnail((width, width), Image.Resampling.LANCZOS)
    with Image.open(yolo_candidates[0]) as right_img:
        right = right_img.convert("RGB")
        right.thumbnail((width, width), Image.Resampling.LANCZOS)
    canvas = Image.new(
        "RGB",
        (left.width + right.width + gap, max(left.height, right.height) + header_h),
        (18, 18, 20),
    )
    canvas.paste(left, (0, header_h))
    canvas.paste(right, (left.width + gap, header_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 9), f"CLASSICAL {label}", fill=(120, 235, 150))
    draw.text((left.width + gap + 10, 9), f"YOLO {yolo_candidates[0].parent.name}",
              fill=(120, 180, 255))
    canvas.save(out_path, quality=88)
    return out_path


# adapted from run_texture_regression.make_contact_sheet (no height legend)
def make_contact_sheet(out_dir: Path, label: str, sources: list[Path],
                       thumb_size: int) -> Path:
    thumb_w, thumb_h, label_h = int(thumb_size), int(thumb_size), 38
    cols = min(3, max(1, len(sources)))
    rows = int(np.ceil(len(sources) / cols)) if sources else 1
    sheet = Image.new("RGB", (thumb_w * cols, (thumb_h + label_h) * rows), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for idx, source in enumerate(sources):
        name = stem_for(source)
        path = out_dir / f"{name}_{label}_overlay.png"
        if not path.exists():
            continue
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            x = (idx % cols) * thumb_w + (thumb_w - image.width) // 2
            y = (idx // cols) * (thumb_h + label_h) + label_h
            sheet.paste(image, (x, y))
        draw.text(((idx % cols) * thumb_w + 10, (idx // cols) * (thumb_h + label_h) + 9),
                  name, fill=(240, 240, 240))
    sheet_path = out_dir / f"{label}_contact_sheet.jpg"
    sheet.save(sheet_path, quality=90)
    return sheet_path


def save_debug_maps(debug: dict, out_dir: Path, stem: str, label: str):
    debug_dir = out_dir / f"{stem}_{label}_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for key in ("evidence01", "mser_mask", "seam_road", "suppress", "shadow",
                "candidates", "prior_score"):
        arr = debug.get(key)
        if arr is None:
            continue
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        path = debug_dir / f"{key}.png"
        Image.fromarray(arr).save(path)
        saved.append(str(path))
    return saved


class VegResolver:
    """Loads {stem}_veg.npy caches, optionally running SegFormer on misses."""

    def __init__(self, veg_npy_dir: Path | None, use_segformer: bool,
                 cache_dir: Path):
        self.veg_npy_dir = veg_npy_dir
        self.use_segformer = use_segformer
        self.cache_dir = cache_dir
        self._segformer = None
        self._segformer_failed = False

    def _load_segformer(self):
        if self._segformer is not None or self._segformer_failed:
            return self._segformer
        try:
            import O4_SFR_Inference as SEGFORMER
            model, proc, device = SEGFORMER.load_vegetation_model(None)
            self._segformer = (SEGFORMER, model, proc, device)
        except Exception as exc:
            print(f"  segformer unavailable ({exc}); falling back to colour masks",
                  flush=True)
            self._segformer_failed = True
            self._segformer = None
        return self._segformer

    def resolve(self, stem: str, img_arr: np.ndarray):
        """Return (veg_map or None, source_tag)."""
        for directory in filter(None, (self.veg_npy_dir, self.cache_dir)):
            cache = directory / f"{stem}_veg.npy"
            if cache.exists():
                try:
                    return np.load(cache), f"npy:{cache}"
                except Exception as exc:
                    print(f"  bad veg cache {cache}: {exc}", flush=True)
        if self.use_segformer:
            bundle = self._load_segformer()
            if bundle is not None:
                segformer, model, proc, device = bundle
                veg_map = segformer.run_inference(model, device, img_arr, proc)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                cache = self.cache_dir / f"{stem}_veg.npy"
                np.save(cache, veg_map)
                return veg_map, "segformer"
        return None, "none"


def main() -> int:
    args = parse_args()
    edges = parse_bands(args.conf_bands)
    floor_conf = args.floor_conf if args.floor_conf is not None else edges[0]

    params = CLASSICAL.ClassicalParams()
    overrides = apply_param_overrides(params, args.param)

    out_dir = args.output_root / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    requested = args.sources if args.sources else DEFAULT_SOURCES
    if args.only:
        requested = [s for s in requested
                     if any(token in stem_for(s) for token in args.only)]
        if not requested:
            raise SystemExit(f"--only {args.only} matched no sources")
    sources = [s for s in requested if s.exists()]
    skipped_sources = [{"source": str(s), "reason": "missing"}
                       for s in requested if not s.exists()]
    if not sources:
        raise FileNotFoundError("No requested texture sources exist")

    veg_resolver = VegResolver(args.veg_npy_dir, args.segformer,
                               args.output_root / "veg_cache")

    settings = {
        "algo_version": CLASSICAL.ALGO_VERSION,
        "conf_bands": edges,
        "floor_conf": floor_conf,
        "max_det": args.max_det,
        "param_overrides": overrides,
        "params": asdict(params),
    }

    started = time.time()
    items: list[dict[str, Any]] = []
    for source in sources:
        item_started = time.time()
        stem = stem_for(source)
        image = load_rgb(source)
        width, height = image.size
        img_arr = np.asarray(image)

        m_per_px = CLASSICAL.m_per_px_for_texture_stem(stem, width, height)
        if m_per_px is None:
            m_per_px = args.m_per_px
        if m_per_px is None:
            print(f"{stem}: cannot derive m_per_px from stem; pass --m-per-px",
                  flush=True)
            skipped_sources.append({"source": str(source), "reason": "no_m_per_px"})
            continue

        veg_map, veg_source = veg_resolver.resolve(stem, img_arr)

        detect_started = time.time()
        detections, debug = CLASSICAL.run_classical_building_inference(
            img_arr,
            m_per_px,
            conf=floor_conf,
            veg_map=veg_map,
            max_det=args.max_det,
            params=params,
            return_debug=True,
        )
        detect_elapsed = time.time() - detect_started

        band_counts: dict[str, int] = {}
        for det in detections:
            key = band_label(band_index(det["confidence"], edges), edges)
            band_counts[key] = band_counts.get(key, 0) + 1

        overlay_png = out_dir / f"{stem}_{args.label}_overlay.png"
        overlay = draw_overlay(image, detections, edges)
        overlay = append_band_legend(overlay, edges, band_counts)
        overlay.save(overlay_png)

        if args.save_source:
            image.save(out_dir / f"{stem}_source.png")

        debug_paths = save_debug_maps(debug, out_dir, stem, args.label) \
            if args.save_debug_maps else []

        sbs_path = None
        if not args.no_sbs:
            sbs_path = make_side_by_side(
                overlay_png, args.yolo_dir, stem, args.label,
                out_dir / f"{stem}_{args.label}_sbs.jpg", args.sbs_width,
            )
            if sbs_path is None:
                print(f"  {stem}: no YOLO overlay found in {args.yolo_dir}, "
                      "skipping side-by-side", flush=True)

        det_json = out_dir / f"{stem}_{args.label}.json"
        det_json.write_text(
            json.dumps(
                {
                    "source": str(source),
                    "label": args.label,
                    "settings": settings,
                    "width": width,
                    "height": height,
                    "m_per_px": m_per_px,
                    "wgsd": debug.get("wgsd"),
                    "stage_timings": debug.get("timings"),
                    "gate_counters": debug.get("counters"),
                    "veg_source": veg_source,
                    "detection_count": len(detections),
                    "counts_by_conf_band": band_counts,
                    "detect_seconds": round(detect_elapsed, 3),
                    "detections": detections,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        item = {
            "source": str(source),
            "output_overlay": str(overlay_png),
            "side_by_side": str(sbs_path) if sbs_path else None,
            "debug_maps": debug_paths,
            "m_per_px": round(float(m_per_px), 4),
            "veg_source": veg_source,
            "detection_count": len(detections),
            "counts_by_conf_band": band_counts,
            "detect_seconds": round(detect_elapsed, 3),
            "elapsed_seconds": round(time.time() - item_started, 3),
        }
        items.append(item)
        bands_text = ", ".join(f"{k}: {v}" for k, v in sorted(band_counts.items()))
        timings = debug.get("timings") or {}
        timing_text = " ".join(f"{k}={v}s" for k, v in timings.items())
        print(f"{args.label} {stem}: {len(detections)} detections "
              f"({bands_text}) in {item['elapsed_seconds']}s "
              f"(detector {item['detect_seconds']}s | {timing_text})", flush=True)

    contact_sheet = None
    if len(items) > 1:
        contact_sheet = make_contact_sheet(
            out_dir, args.label,
            [Path(item["source"]) for item in items],
            args.contact_thumb_size,
        )

    summary = {
        "label": args.label,
        "started_at": started,
        "elapsed_seconds": round(time.time() - started, 3),
        "settings": settings,
        "contact_sheet": str(contact_sheet) if contact_sheet else None,
        "skipped_sources": skipped_sources,
        "total_detections": sum(item["detection_count"] for item in items),
        "items": items,
    }
    summary_path = out_dir / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "contact_sheet": str(contact_sheet) if contact_sheet else None,
                "elapsed_seconds": summary["elapsed_seconds"],
                "counts": {stem_for(Path(item["source"])): item["detection_count"]
                           for item in items},
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
