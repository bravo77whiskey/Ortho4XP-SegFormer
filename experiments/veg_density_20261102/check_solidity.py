"""Does the solid-mask weight fire at all? Solidity and the share of pixels it can change."""
import glob

import numpy as np
from PIL import Image

import eval_interior as I
import eval_unmix_data as E
import sheet2 as S
import sheet3 as H


def line(name, rgb, p, mask, mpp):
    on = mask > 0
    sol = I.solidity(mask, mpp, 200)
    diff = np.abs(I.cover(rgb, p, mask, mpp, radius_m=200, lo=0.7, hi=0.9) - I.cover(rgb, p, mask, mpp))
    print(f"{name:12s} mpp {mpp:5.2f} shape {mask.shape} mask range {mask.min():.1f}-{mask.max():.1f} on {on.mean():.2f} "
          f"solid p50/p90 {np.percentile(sol[on], 50):.2f}/{np.percentile(sol[on], 90):.2f} "
          f"P<0.5&solid>0.7 {((p < 0.5) & (sol > 0.7) & on).sum() / max(1, on.sum()):.3f} "
          f"mean change {diff[on].mean():.3f} max {diff[on].max():.2f}")


for (site, tag), (rgb, p, mask, _truth, mpp) in E.data.items():
    line(f"{site[0]}{tag}", rgb, p, mask, mpp)
name = "szx_e"
head = open(f"{I.RUNS}/{name}.js", encoding="utf-8").read(400)
path = glob.glob(f"{I.SETS}/**/{head.split(chr(34) + 'file' + chr(34) + ':' + chr(34))[1].split(chr(34))[0]}", recursive=True)[0]
rgb = np.array(Image.open(path).convert("RGB"))
mpp = S.mpp_of(22.7, 16)
p, mask = H.seg_cached(path, rgb, mpp)
line(name, rgb, p, mask, mpp)
y0, y1, x0, x1 = 1400, 2200, 800, 1900
print("szx_e block P percentiles", np.percentile(p[y0:y1, x0:x1][mask[y0:y1, x0:x1] > 0], [10, 50, 90]).round(2))
