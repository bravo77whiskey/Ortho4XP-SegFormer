"""Density per hotspot texture from two veg-only review runs, and maps coloured by density."""
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

ROOT = "E:/Ortho4XP_Review/runs"
SETS = "E:/Ortho4XP_Review/sets"
OUT = "E:/Ortho4XP_Review/veg_density/sheets"
OFF, ON = "veg-density-off", "veg-density-solid"


def load(run, key):
    with open(f"{ROOT}/{run}/BI/{key}.js", encoding="utf-8") as f:
        text = f.read()
    return json.loads(text[text.index("] = {") + 4:].rstrip().rstrip(";"))


def paint(veg, size, flags):
    """uint8 raster of the DSF density (0 = no polygon) for polygons of one mode."""
    out = np.zeros((size, size), np.uint8)
    for p in veg:
        d = int(p["d"])
        if (d >> 8) != flags or not p["r"]:
            continue
        cv2.fillPoly(out, [np.int32(np.round(p["r"][0]))], max(1, d & 255))
        for hole in p["r"][1:]:
            cv2.fillPoly(out, [np.int32(np.round(hole))], 0)
    return out


def stats(veg, size=4096):
    area = paint(veg, size, 0)
    on = area > 0
    # m2 cancels: trees relative to the same area planted at 255.
    return dict(n=len(veg), line=sum(int(p["d"]) >> 8 == 1 for p in veg),
                point=sum(int(p["d"]) >> 8 == 2 for p in veg),
                frac=float(on.mean()), mean_d=float(area[on].mean()) if on.any() else 0.0,
                hist=np.bincount(area[on], minlength=256) / max(1, on.sum())), area


def density_view(jpg, area):
    img = cv2.imread(jpg)
    size = area.shape[0]
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    lut = np.zeros((256, 1, 3), np.uint8)
    lut[:, 0, 0] = 90
    lut[:, 0, 1] = np.linspace(0, 255, 256)
    lut[:, 0, 2] = np.linspace(255, 0, 256)
    colour = cv2.applyColorMap(area, lut)
    out = np.where((area > 0)[..., None], colour // 2 + img // 2, img)
    return out


def main(keys):
    print("key zl | polys off->on | treeline off->on | points off->on | area% off->on | mean density off->on | share at top density")
    for path in sorted(glob.glob(f"{ROOT}/{ON}/BI/*.js")):
        key = os.path.basename(path)[:-3]
        off, on = load(OFF, key), load(ON, key)
        a, area_a = stats(off["veg"])
        b, area_b = stats(on["veg"])
        top = int(np.nonzero(b["hist"])[0].max()) if b["hist"].any() else 0
        print(f"{key:15s} z{on['zl']} | {a['n']:5d}->{b['n']:5d} | {a['line']:5d}->{b['line']:5d} | "
              f"{a['point']:5d}->{b['point']:5d} | {100*a['frac']:4.1f}->{100*b['frac']:4.1f} | "
              f"{a['mean_d']:4.0f}->{b['mean_d']:4.0f} | {100*b['hist'][top]:3.0f}% at {top}")
        if key in keys:
            jpg = glob.glob(f"{SETS}/hotspots/**/{on['file']}", recursive=True)
            if not jpg:
                print("  no jpg for", key)
                continue
            view = density_view(jpg[0], area_b)
            src = cv2.resize(cv2.imread(jpg[0]), (4096, 4096), interpolation=cv2.INTER_AREA)
            sheet = np.hstack([src, view])
            cv2.imwrite(f"{OUT}/hotspot_{key}.jpg", cv2.resize(sheet, (2400, 1200), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main(set(sys.argv[1:]))
