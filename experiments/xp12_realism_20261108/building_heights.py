"""Building height rasters for the relief test (sim_ab.py building-heights).

Runs the live building detector (trained YOLO OBB) and HeightNet on every
texture of a tile copy, the same way the building overlay does (native
resolution, the live cfg's YOLO settings, regional height priors), and
rasterises each footprint at its height.  Saves <stem>.npy: float16 metres at
MAP px x MAP px (area-averaged, so footprint edges are already soft).

    <CUDA python> building_heights.py <textures dir> <out dir> <yolo .pt> <heightnet .pt>
                                      [conf iou stride imgsz batch region]

Run it with the CUDA venv from the repo's src/ (imports O4_Building_Overlay).
"""
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

MAP = 1024
TEX_RE = re.compile(r"^(\d+)_(\d+)_([A-Za-z]+)(\d{2})\.dds$")


def metres_per_px(til_y, zl, width):
    import math
    n = math.pi - 2 * math.pi * (til_y + 8) / 2 ** zl     # texture centre row (16 tiles)
    lat = math.degrees(math.atan(math.sinh(n)))
    return 40075016.686 * math.cos(math.radians(lat)) / 2 ** zl * 16 / width


def main():
    tex_dir, out_dir, yolo_pt, height_pt = sys.argv[1:5]
    conf, iou, stride, imgsz, batch = (float(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7]),
                                       int(sys.argv[8]), int(sys.argv[9])) if len(sys.argv) > 9 else \
        (0.05, 0.5, 512, 512, 8)
    region = sys.argv[10] if len(sys.argv) > 10 else "asia"
    import cv2
    import torch
    from PIL import Image
    import O4_Building_Overlay as B
    import O4_Height_Model as HM

    os.makedirs(out_dir, exist_ok=True)
    yolo = B._load_yolo_obb_model(yolo_pt)
    heights = HM.load_height_model(height_pt, device="cuda" if torch.cuda.is_available() else "cpu")
    names = sorted(f for f in os.listdir(tex_dir) if TEX_RE.match(f))
    t0 = time.time()
    for i, f in enumerate(names):
        dst = os.path.join(out_dir, f[:-4] + ".npy")
        if os.path.isfile(dst):
            continue
        m = TEX_RE.match(f)
        img = np.asarray(Image.open(os.path.join(tex_dir, f)).convert("RGB"))
        h, w = img.shape[:2]
        mpp = metres_per_px(int(m.group(1)), int(m.group(4)), w)
        dets = B._run_yolo_obb_inference(yolo, img, imgsz=imgsz, stride=stride, conf=conf, iou=iou,
                                         max_det=100000, m_per_px=mpp, batch_size=batch)
        if isinstance(dets, dict):
            dets = dets.get("detections", [])
        B._apply_heightnet_to_detections(heights, img, dets, mpp, region=region)
        raster = np.zeros((h, w), np.float32)
        placed = 0
        for d in sorted(dets, key=lambda d: float(d.get("height_m") or 0)):
            hm = d.get("height_m")
            pts = d.get("points")
            if hm is None or pts is None or not np.isfinite(hm):
                continue
            poly = np.round(np.asarray(pts, np.float64).reshape(-1, 2)).astype(np.int32)
            cv2.fillPoly(raster, [poly], float(hm))
            placed += 1
        small = cv2.resize(raster, (MAP, MAP), interpolation=cv2.INTER_AREA)
        np.save(dst, small.astype(np.float16))
        print(f"  [{i + 1}/{len(names)}] {f}: {placed} buildings, max {raster.max():.0f} m", flush=True)
    print(f"building heights: {len(names)} textures in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
