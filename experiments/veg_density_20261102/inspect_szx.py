"""szx_e: density planted in the dark forest block west of the reservoir vs the forest around it."""
import glob
import json

import cv2
import numpy as np

from compare_runs import ON, SETS, load, paint

data = load(ON, "szx_e")
area = paint(data["veg"], 4096, 0)
jpg = glob.glob(f"{SETS}/hotspots/**/{data['file']}", recursive=True)[0]
img = cv2.resize(cv2.imread(jpg), (4096, 4096), interpolation=cv2.INTER_AREA)
lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Regions in sheet coordinates (2000 x 1000 display, right half = x - 1000), scaled to 4096.
REGIONS = {
    "dark block (mid density?)": (240, 380, 400, 480),
    "forest north of it (full)": (100, 100, 350, 220),
    "reservoir island (thin)": (780, 390, 840, 440),
    "forest south-east (full)": (550, 680, 680, 800),
}
scale = 4096 / 1000
for name, (x0, y0, x1, y1) in REGIONS.items():
    x0, y0, x1, y1 = [int(v * scale) for v in (x0, y0, x1, y1)]
    a = area[y0:y1, x0:x1]
    m = a > 0
    hist = {int(v): round(float((a[m] == v).mean()), 2) for v in np.unique(a[m])} if m.any() else {}
    print(f"{name:28s} mask {m.mean():.2f} lum {lum[y0:y1, x0:x1].mean():5.1f} densities {json.dumps(hist)}")
crop = np.hstack([img[1400:2200, 800:1900], cv2.applyColorMap(area[1400:2200, 800:1900], cv2.COLORMAP_VIRIDIS)])
cv2.imwrite("E:/Ortho4XP_Review/veg_density/sheets/szx_e_block.jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
