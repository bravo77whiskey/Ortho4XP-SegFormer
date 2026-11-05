"""Planted forest heights of two review runs side by side on the texture, in the
400 m window where they differ most.

  .venv/Scripts/python.exe experiments/veg_stands_20261105/run_ab_images.py veg-stands-off veg-stands-25-m600 khh_z19 shouguang_z18
"""
import json, os, sys, numpy as np, cv2
R = "G:/Ortho4XP_Review/runs"; SET = "G:/Ortho4XP_Review/sets/hotspots/BI"
OUT = "G:/Ortho4XP_Review/_scratch/veg_stands/sheets"
samples = {s["key"]: s for s in json.load(open(os.path.join(SET, "samples.json")))}
COL = [(2.5, (0, 200, 255)), (5, (60, 220, 255)), (7.5, (80, 255, 160)), (10.5, (40, 200, 60)),
       (13.5, (30, 140, 20)), (16.5, (90, 90, 0)), (99, (170, 40, 120))]
def col(h):
    for lim, c in COL:
        if h < lim: return c
def raster(run, key, n):
    t = open(os.path.join(R, run, "BI", key + ".js")).read()
    d = json.loads(t[t.index("= {") + 2:].rstrip().rstrip(";"))
    hmap = np.zeros((n, n), np.float32); s = n / 4096
    for v in d["veg"]:
        r = v["r"]
        if not r or not isinstance(r[0][0], list): continue
        m = np.zeros((n, n), np.uint8)
        cv2.fillPoly(m, [np.round(np.asarray(r[0]) * s).astype(np.int32)], 1)
        for hole in r[1:]: cv2.fillPoly(m, [np.round(np.asarray(hole) * s).astype(np.int32)], 0)
        hmap[m > 0] = v.get("h", 99)
    return hmap
def tint(tex, hmap):
    img = tex.copy(); on = hmap > 0
    c = np.zeros_like(tex)
    for lim_lo, (lim, cc) in zip([0] + [l for l, _ in COL[:-1]], COL):
        c[on & (hmap >= lim_lo) & (hmap < lim)] = cc
    img[on] = (0.45 * tex[on] + 0.55 * c[on]).astype(np.uint8)
    e = (cv2.morphologyEx(hmap, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0)
    img[e] = (20, 20, 20)
    return img
RUN_A, RUN_B = sys.argv[1], sys.argv[2]
for key in sys.argv[3:]:
    n = 2048
    tex = cv2.resize(cv2.imread(os.path.join(SET, samples[key]["file"])), (n, n), interpolation=cv2.INTER_AREA)
    a, b = raster(RUN_A, key, n), raster(RUN_B, key, n)
    diff = (np.abs(a - b) > 1).astype(np.float32)
    win = 512
    s = cv2.boxFilter(diff, -1, (win, win), normalize=False)
    h = win // 2; s[:h] = s[-h:] = 0; s[:, :h] = s[:, -h:] = 0
    y, x = np.unravel_index(int(s.argmax()), s.shape); y0, x0 = y - h, x - h
    crop = lambda im: cv2.resize(im[y0:y0 + win, x0:x0 + win], (768, 768), interpolation=cv2.INTER_NEAREST)
    panels = [crop(tex), crop(tint(tex, a)), crop(tint(tex, b))]
    for p, t in zip(panels, ["texture", RUN_A, RUN_B]):
        cv2.rectangle(p, (0, 0), (768, 30), (0, 0, 0), -1)
        cv2.putText(p, f"{key} {t}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    leg = np.zeros((36, 768 * 3, 3), np.uint8); xx = 10
    for lab, (_l, c) in zip(["shrub<2.5", "shrub<5", "6", "9", "12", "15", "full"], COL):
        cv2.rectangle(leg, (xx, 8), (xx + 22, 28), c, -1)
        cv2.putText(leg, lab, (xx + 28, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA); xx += 140
    cv2.imwrite(os.path.join(OUT, f"{key}_{RUN_A}_vs_{RUN_B}.jpg"), np.vstack([np.hstack(panels), leg]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(key, "window", x0, y0, "changed px share %.1f%%" % (100 * diff.sum() / max(1, ((a > 0) | (b > 0)).sum())))
