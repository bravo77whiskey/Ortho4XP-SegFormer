"""Prototype: per-building height from Depth Anything V2 + SegFormer landcover anchor.

Usage:
    python scripts/depth_height_prototype.py <image_path> [--device cuda|cpu]
                                              [--model small|large]
                                              [--out <debug_dir>]

Workflow per image:
  1. Load image (PNG/JPG/DDS — anything PIL or opencv can read).
  2. Run Depth Anything V2 to get a relative depth map (HxW float, larger = farther).
  3. (Optional) Run SegFormer landcover to identify flat-ground pixels
     (road / agriculture / bareland / rangeland) and use their median depth as a
     per-tile ground anchor.
  4. (Optional) Run the trained YOLO-OBB to get building OBBs.
  5. For each OBB compute:
       roof_depth   = mean depth inside the OBB
       ring_depth   = mean depth in a 10px ring outside the OBB
       height_proxy = max(ground_anchor, ring_depth) - roof_depth   (relative units)
  6. Calibrate: fit a single scalar `meters_per_unit` so that the median of the
     per-class height proxies matches the existing class-based heights from
     O4_SFR_Building_Overlay.DEFAULT_FACADE_HEIGHT_M (weak prior).
  7. Print results: per-class distribution, comparison vs class-based heights,
     and save side-by-side debug PNGs (RGB + depth + OBBs colored by height).

This is a diagnostic script — it does NOT modify the pipeline.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np


def _add_src_to_path():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.normpath(os.path.join(here, "..", "src"))
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


_add_src_to_path()


def load_image(path: str) -> np.ndarray:
    """Load an image to HxWx3 uint8 RGB. Supports DDS via PIL (drops alpha)."""
    from PIL import Image
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def load_depth_model(variant: str = "small", device: str = "cuda"):
    """Load Depth Anything V2 via transformers."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    import torch

    repo = {
        "small": "depth-anything/Depth-Anything-V2-Small-hf",
        "base":  "depth-anything/Depth-Anything-V2-Base-hf",
        "large": "depth-anything/Depth-Anything-V2-Large-hf",
    }[variant]
    print(f"[depth-proto] Loading {repo} on {device} …", flush=True)
    processor = AutoImageProcessor.from_pretrained(repo)
    model = AutoModelForDepthEstimation.from_pretrained(repo).to(device).eval()
    return model, processor


def run_depth_inference(model, processor, image_rgb: np.ndarray, device: str,
                        tile: int = 1024, overlap: int = 128) -> np.ndarray:
    """Run depth estimation; tile if image is large. Returns HxW float32 (relative depth)."""
    import torch
    from PIL import Image

    H, W = image_rgb.shape[:2]
    if max(H, W) <= tile:
        # Single forward pass
        inputs = processor(images=Image.fromarray(image_rgb), return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        depth = outputs.predicted_depth.squeeze().detach().cpu().float().numpy()
        # Resize back to original
        if depth.shape != (H, W):
            import cv2
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_CUBIC)
        return depth

    # Tiled inference
    print(f"[depth-proto] Tiled inference: {H}x{W} with tile={tile} overlap={overlap}", flush=True)
    out = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)
    step = tile - overlap
    y_starts = list(range(0, max(1, H - tile + 1), step))
    if y_starts[-1] + tile < H: y_starts.append(H - tile)
    x_starts = list(range(0, max(1, W - tile + 1), step))
    if x_starts[-1] + tile < W: x_starts.append(W - tile)
    feather = np.ones((tile, tile), dtype=np.float32)
    # Linear ramp on edges to feather overlaps
    ramp = np.linspace(0, 1, overlap, dtype=np.float32)
    feather[:overlap, :] *= ramp[:, None]
    feather[-overlap:, :] *= ramp[::-1, None]
    feather[:, :overlap] *= ramp[None, :]
    feather[:, -overlap:] *= ramp[None, ::-1]
    for yi, y0 in enumerate(y_starts):
        for xi, x0 in enumerate(x_starts):
            crop = image_rgb[y0:y0+tile, x0:x0+tile]
            inputs = processor(images=Image.fromarray(crop), return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = model(**inputs)
            d = outputs.predicted_depth.squeeze().detach().cpu().float().numpy()
            if d.shape != (tile, tile):
                import cv2
                d = cv2.resize(d, (tile, tile), interpolation=cv2.INTER_CUBIC)
            out[y0:y0+tile, x0:x0+tile]   += d * feather
            weight[y0:y0+tile, x0:x0+tile] += feather
        print(f"[depth-proto]   row {yi+1}/{len(y_starts)} done", flush=True)
    weight[weight < 1e-6] = 1.0
    return out / weight


def load_segformer_landcover(image_rgb: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Run SegFormer landcover model used by the project. Returns HxW int8 class map."""
    import O4_SFR_Inference as SEG  # noqa
    model, proc, _dev = SEG.load_vegetation_model(device)
    return SEG.run_inference(model, _dev, image_rgb, proc)


def load_yolo_obb(image_rgb: np.ndarray, checkpoint: str, device: str = "cuda",
                  conf: float = 0.18, iou: float = 0.5, stride: int = 512, imgsz: int = 1024):
    """Run the trained YOLO-OBB model the project already uses."""
    import O4_SFR_Building_Overlay as BLD
    model = BLD._load_yolo_obb_model(checkpoint)
    # Estimate m_per_px from the image — caller is responsible for context;
    # use a placeholder of 1.0 for relative computations (we only need pixel-space OBB).
    detections = BLD._run_yolo_obb_inference(
        model, image_rgb,
        imgsz=imgsz, stride=stride, conf=conf, iou=iou, max_det=2000,
        device=device, m_per_px=1.0,
    )
    return detections


def obb_mask(obb_pts_px: np.ndarray, H: int, W: int) -> np.ndarray:
    """Boolean mask for the polygon."""
    import cv2
    pts = np.asarray(obb_pts_px, dtype=np.int32).reshape(-1, 2)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def ring_mask(obb_pts_px: np.ndarray, H: int, W: int, expand_px: int = 8) -> np.ndarray:
    """Mask of an expand_px ring just outside the polygon (the building's
    immediate surroundings — best proxy for local ground level)."""
    import cv2
    pts = np.asarray(obb_pts_px, dtype=np.int32).reshape(-1, 2)
    inner = np.zeros((H, W), dtype=np.uint8)
    outer = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(inner, [pts], 1)
    # Dilate to expand outward
    k = max(1, expand_px)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
    outer = cv2.dilate(inner, kernel, iterations=1)
    return (outer.astype(bool)) & (~inner.astype(bool))


def compute_heights(depth: np.ndarray,
                    detections,
                    veg_map: np.ndarray | None = None,
                    ring_px: int = 8) -> list[dict]:
    """For each detection, compute roof_depth, ring_depth, and a relative height proxy.

    Convention: Depth Anything V2 outputs INVERSE depth (larger value = closer to camera).
    So roof should have LARGER depth value than ground. Height proxy = roof - ring.
    """
    H, W = depth.shape[:2]
    # Ground anchor from SegFormer: median depth over flat-ground classes
    ground_anchor = None
    if veg_map is not None:
        # SegFormer classes: 1=bareland, 2=rangeland, 3=developed, 4=road, 7=agriculture
        # Use the lowest-built classes as anchor: road + agriculture + rangeland + bareland
        flat_mask = (
            (veg_map == 1) | (veg_map == 2) | (veg_map == 4) | (veg_map == 7)
        )
        if flat_mask.any():
            ground_anchor = float(np.median(depth[flat_mask]))

    out = []
    for det in detections:
        pts = np.asarray(det.get('points', ()), dtype=np.float32).reshape(-1, 2)
        if pts.shape != (4, 2):
            continue
        inside = obb_mask(pts, H, W)
        if not inside.any():
            continue
        ring = ring_mask(pts, H, W, expand_px=ring_px)
        roof_depth = float(np.mean(depth[inside]))
        ring_depth = float(np.mean(depth[ring])) if ring.any() else float('nan')
        # Use ring depth if available, else fall back to ground anchor
        local_ground = ring_depth if not np.isnan(ring_depth) else (ground_anchor or roof_depth)
        # Depth Anything V2 emits INVERSE depth: higher = closer (roof should have higher value)
        height_proxy = roof_depth - local_ground
        out.append({
            'placement_class': int(det.get('placement_class', -1)),
            'model_class':    int(det.get('model_class', -1)),
            'area_m2':        float(det.get('area_m2', 0.0)),
            'max_side_m':     float(det.get('max_side_m', 0.0)),
            'confidence':     float(det.get('confidence', 0.0)),
            'center_xy':      tuple(map(int, det.get('center', (0, 0)))),
            'roof_depth':     roof_depth,
            'ring_depth':     ring_depth,
            'ground_anchor':  ground_anchor,
            'height_proxy':   height_proxy,
        })
    return out


def calibrate_to_meters(records: list[dict],
                        class_height_priors: dict[int, float]) -> tuple[float, float]:
    """Fit a single linear (scale, offset) so the per-class median of
    `height_proxy * scale + offset` matches `class_height_priors` (weak prior).

    Returns (scale, offset) such that height_m = scale * height_proxy + offset.
    Robust to per-class outliers via median aggregation.
    """
    by_cls = defaultdict(list)
    for r in records:
        by_cls[r['placement_class']].append(r['height_proxy'])
    # Build x (per-class median proxy) and y (per-class prior height)
    xs, ys = [], []
    for cls, proxies in by_cls.items():
        if cls not in class_height_priors or len(proxies) < 3:
            continue
        xs.append(float(np.median(proxies)))
        ys.append(float(class_height_priors[cls]))
    if len(xs) < 2:
        return 1.0, 0.0
    xs = np.asarray(xs); ys = np.asarray(ys)
    A = np.vstack([xs, np.ones_like(xs)]).T
    scale, offset = np.linalg.lstsq(A, ys, rcond=None)[0]
    return float(scale), float(offset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="Path to the aerial image (PNG/JPG/DDS).")
    ap.add_argument("--device", default=None, help="cuda or cpu (auto-detect by default).")
    ap.add_argument("--model", default="small", choices=["small", "base", "large"])
    ap.add_argument("--yolo", default=None,
                    help="Optional path to the trained YOLO-OBB checkpoint. "
                         "Defaults to the one in O4_SFR_Pipeline.py.")
    ap.add_argument("--no-segformer", action="store_true",
                    help="Skip SegFormer landcover anchor; use only ring depth.")
    ap.add_argument("--out", default=None, help="Optional output dir for debug PNGs.")
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[depth-proto] device={device}", flush=True)

    print(f"[depth-proto] Loading image {args.image} …", flush=True)
    image = load_image(args.image)
    print(f"[depth-proto] image shape: {image.shape}", flush=True)

    # 1) Depth
    t0 = time.perf_counter()
    dmodel, dproc = load_depth_model(args.model, device)
    depth = run_depth_inference(dmodel, dproc, image, device)
    print(f"[depth-proto] depth inference: {time.perf_counter()-t0:.1f}s, "
          f"shape={depth.shape}, range=[{depth.min():.3f}, {depth.max():.3f}]", flush=True)

    # 2) SegFormer (optional)
    veg_map = None
    if not args.no_segformer:
        t0 = time.perf_counter()
        try:
            veg_map = load_segformer_landcover(image, device)
            print(f"[depth-proto] segformer landcover: {time.perf_counter()-t0:.1f}s, "
                  f"classes present: {sorted(set(int(c) for c in np.unique(veg_map)))}",
                  flush=True)
        except Exception as exc:
            print(f"[depth-proto] segformer failed ({exc}); proceeding without anchor", flush=True)
            veg_map = None

    # 3) YOLO OBB
    yolo_ckpt = args.yolo or r"H:\model_training\runs\yolo_obb_v1\weights\visual_candidate_step_12000.pt"
    if not os.path.exists(yolo_ckpt):
        print(f"[depth-proto] WARNING: YOLO checkpoint not found at {yolo_ckpt}; "
              f"the prototype needs detections to compare against. Aborting.", flush=True)
        return 2
    t0 = time.perf_counter()
    detections = load_yolo_obb(image, yolo_ckpt, device)
    print(f"[depth-proto] yolo: {time.perf_counter()-t0:.1f}s, "
          f"{len(detections)} detections", flush=True)
    if not detections:
        print("[depth-proto] No detections — nothing to evaluate.", flush=True)
        return 2

    # 4) Per-detection height proxy
    records = compute_heights(depth, detections, veg_map=veg_map, ring_px=8)
    print(f"[depth-proto] computed proxies for {len(records)} detections", flush=True)

    # 5) Calibrate to meters via class priors (weak)
    import O4_SFR_Building_Overlay as BLD
    priors = dict(BLD.DEFAULT_FACADE_HEIGHT_M)
    scale, offset = calibrate_to_meters(records, priors)
    print(f"[depth-proto] linear fit (proxy → meters): scale={scale:.4f} offset={offset:.4f}",
          flush=True)

    # 6) Compute calibrated heights and report distribution per class
    for r in records:
        r['height_m_est'] = scale * r['height_proxy'] + offset

    print()
    print("=" * 78)
    print(f"{'class':<22} {'N':>5} {'prior(m)':>9} {'med(m)':>8} {'p25':>6} {'p75':>6} {'min':>6} {'max':>6}")
    print("-" * 78)
    by_cls = defaultdict(list)
    for r in records:
        by_cls[r['placement_class']].append(r['height_m_est'])
    for cls in sorted(by_cls):
        hs = np.asarray(by_cls[cls])
        label = BLD.BLD_CLASS_LABELS.get(cls, str(cls))[:22]
        prior = priors.get(cls, float('nan'))
        print(f"{label:<22} {len(hs):>5d} {prior:>9.2f} "
              f"{np.median(hs):>8.2f} {np.percentile(hs, 25):>6.2f} "
              f"{np.percentile(hs, 75):>6.2f} {hs.min():>6.2f} {hs.max():>6.2f}")
    print("=" * 78)
    print()

    # 7) Optional debug PNG
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        try:
            import cv2
            # Normalize depth for visualization
            d_norm = depth - depth.min()
            d_norm = d_norm / max(1e-6, d_norm.max())
            d_vis = (d_norm * 255).astype(np.uint8)
            d_color = cv2.applyColorMap(d_vis, cv2.COLORMAP_TURBO)
            # Overlay OBBs colored by estimated height
            overlay = image.copy()
            for r, det in zip(records, detections[:len(records)]):
                pts = np.asarray(det['points'], dtype=np.int32).reshape(-1, 2)
                # Color from height: blue (low) → red (high)
                h = max(0.0, min(30.0, r['height_m_est']))
                t = h / 30.0
                color = (int(255*(1-t)), 0, int(255*t))
                cv2.polylines(overlay, [pts], True, color, 2)
                cv2.putText(overlay, f"{r['height_m_est']:.0f}",
                            (int(pts.mean(axis=0)[0]), int(pts.mean(axis=0)[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            out_rgb = os.path.join(args.out, "overlay_heights.png")
            out_depth = os.path.join(args.out, "depth.png")
            cv2.imwrite(out_rgb, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            cv2.imwrite(out_depth, d_color)
            print(f"[depth-proto] wrote {out_rgb} and {out_depth}", flush=True)
        except Exception as exc:
            print(f"[depth-proto] debug image write failed: {exc}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
