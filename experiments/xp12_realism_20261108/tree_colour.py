"""SUPERSEDED by tree_colour2.py: this read the bit-packed tree-support file as the tree mask.
Measure colour mismatch: XP12 forest atlas leaves vs ortho canopy (SegFormer tree mask).
Compares CIELAB chroma (a*, b*) and lightness. Ortho is a lit photo, atlas is albedo,
so L* is only indicative; hue/chroma is the comparable part."""
import sys, glob, os, numpy as np, cv2
from PIL import Image
XP = r"D:\SteamLibrary\steamapps\common\X-Plane 12\Resources\default scenery\1200 forests\textures"
def lab(rgb):  # rgb uint8 Nx3 -> Lab float (OpenCV scales L 0..255)
    l = cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    return np.stack([l[:, 0] * 100 / 255, l[:, 1] - 128, l[:, 2] - 128], 1)
def atlas(name):
    a = np.asarray(Image.open(os.path.join(XP, name)).convert("RGBA"))
    m = (a[..., 3] > 200) & (a[..., 1].astype(int) >= a[..., 0]) & (a[..., 1].astype(int) > a[..., 2])
    return lab(a[..., :3][m])
def ortho(jpg, tree_npy):
    im = np.asarray(Image.open(jpg).convert("RGB"))
    t = np.load(tree_npy)
    if t.dtype != bool: t = t > (0.5 if t.max() <= 1 else 127)
    if t.shape != im.shape[:2]:
        t = cv2.resize(t.astype(np.uint8), im.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
    t = cv2.erode(t.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0   # crown interiors
    return lab(im[t])
def stats(x):
    return dict(L=float(np.median(x[:, 0])), a=float(np.median(x[:, 1])), b=float(np.median(x[:, 2])),
                C=float(np.median(np.hypot(x[:, 1], x[:, 2]))), hue=float(np.degrees(np.arctan2(np.median(x[:, 2]), np.median(x[:, 1])))))
rows = []
for n in ["trees_bb1_ALB.dds", "trees_3D1_ALB.dds"]:
    s = stats(atlas(n)); rows.append(("XP " + n, s))
for npy in sorted(glob.glob(r"G:\Ortho4XP_Review\_cache\w*\*\*\*_vegtree.npy")):
    stem = os.path.basename(npy).replace("_vegtree.npy", "")
    jpg = rf"G:\Ortho4XP_Review\sets\hotspots\BI\{stem}.jpg"
    if os.path.exists(jpg) and not any(r[0].endswith(stem) for r in rows):
        try: rows.append(("ortho " + stem, stats(ortho(jpg, npy))))
        except Exception as e: print(stem, e)
for name, s in rows:
    print(f"{name:34s} L {s['L']:5.1f}  a {s['a']:6.1f}  b {s['b']:6.1f}  C {s['C']:5.1f}  hue {s['hue']:6.1f}")
