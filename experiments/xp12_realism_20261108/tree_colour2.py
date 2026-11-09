"""Corrected tree-colour comparison: XP atlas leaves vs ortho canopy, both as
per-texture medians of a* and b* (CIELAB), canopy = SegFormer tree class
(eroded crowns) from the cached class map. Supersedes tree_colour.py, which
read the bit-packed tree-support file as if it were the tree mask."""
import glob, os, sys, numpy as np, cv2
from PIL import Image
sys.path.insert(0, r"G:\Dev\Ortho4XP-SegFormer\src")
from O4_Landcover_Inference import CLASS_TREE
XP = r"D:\SteamLibrary\steamapps\common\X-Plane 12\Resources\default scenery\1200 forests\textures"
SETS = r"G:\Ortho4XP_Review\sets\hotspots\BI"
def lab(rgb):
    l = cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    return np.stack([l[:, 0] * 100 / 255, l[:, 1] - 128, l[:, 2] - 128], 1)
def summary(x):
    a, b = np.median(x[:, 1]), np.median(x[:, 2])
    return np.median(x[:, 0]), a, b, np.hypot(a, b), np.degrees(np.arctan2(b, a))
def leaves(name):
    a = np.asarray(Image.open(os.path.join(XP, name)).convert("RGBA"))
    m = (a[..., 3] > 200) & (a[..., 1].astype(int) >= a[..., 0]) & (a[..., 1].astype(int) > a[..., 2])
    return lab(a[..., :3][m])
rows = [("XP " + n, summary(leaves(n))) for n in ("trees_3D1_ALB.dds", "trees_bb1_ALB.dds", "trees_3D1_ALB_sp.dds")]
seen = set()
for veg_npy in sorted(glob.glob(r"G:\Ortho4XP_Review\_cache\w0\*\*\*_veg.npy")):
    stem = os.path.basename(veg_npy)[:-len("_veg.npy")]
    jpg = os.path.join(SETS, stem + ".jpg")
    if stem in seen or not os.path.exists(jpg):
        continue
    seen.add(stem)
    im = np.asarray(Image.open(jpg).convert("RGB"))
    t = (np.load(veg_npy) == CLASS_TREE).astype(np.uint8)
    if t.shape != im.shape[:2]:
        t = cv2.resize(t, im.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    t = cv2.erode(t, np.ones((5, 5), np.uint8)) > 0
    if t.sum() < 5000:
        continue
    rows.append((stem, summary(lab(im[t][::7]))))
for n, (L, a, b, C, h) in rows:
    print(f"{n:24s} L {L:5.1f}  a {a:6.1f}  b {b:6.1f}  C {C:5.1f}  hue {h:6.1f}")
