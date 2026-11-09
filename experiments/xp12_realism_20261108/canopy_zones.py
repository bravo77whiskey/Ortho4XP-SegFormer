"""Distribution of locally smoothed ortho canopy colour (CIELAB a*, b*) over the hotspot textures,
weighted by canopy area: what range the forest tint buckets must cover."""
import glob, os, sys, numpy as np, cv2
from PIL import Image
SIGMA_M = 60.0
sys.path.insert(0, r'G:\Dev\Ortho4XP-SegFormer\src')
from O4_Landcover_Inference import CLASS_TREE
def lab_img(rgb):
    l = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    return l[..., 0] * 100 / 255, l[..., 1] - 128, l[..., 2] - 128
out = []
for npy in sorted(glob.glob(r"G:\Ortho4XP_Review\_cache\w*\*\*\*_vegtree.npy")):
    stem = os.path.basename(npy).replace("_vegtree.npy", "")
    jpg = rf"G:\Ortho4XP_Review\sets\hotspots\BI\{stem}.jpg"
    if not os.path.exists(jpg):
        continue
    im = np.asarray(Image.open(jpg).convert("RGB"))
    veg = np.load(npy.replace("_vegtree.npy", "_veg.npy"))
    t = (veg == CLASS_TREE)
    raw = np.load(npy); h, w = (int(v) for v in raw[:8].view(np.uint32))
    if (h, w) == veg.shape:
        t |= np.unpackbits(raw[8:], count=h * w).reshape(h, w).astype(bool)
    t = t.astype(np.uint8)
    if t.shape != im.shape[:2]:
        t = cv2.resize(t, im.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    t = cv2.erode(t, np.ones((5, 5), np.uint8)).astype(np.float32)
    zl = int(stem.split("_")[-1][-2:])
    mpp = 156543.03 * np.cos(np.radians(23)) / 2 ** zl * 256 / im.shape[1] * 16
    L, a, b = lab_img(im)
    sig = SIGMA_M / mpp
    w = cv2.GaussianBlur(t, (0, 0), sig)
    sa = cv2.GaussianBlur(a * t, (0, 0), sig) / np.maximum(w, 1e-6)
    sb = cv2.GaussianBlur(b * t, (0, 0), sig) / np.maximum(w, 1e-6)
    on = t > 0
    C = np.hypot(sa[on], sb[on]); h = np.degrees(np.arctan2(sb[on], sa[on]))
    print(f"{stem:24s} px {on.sum():9d}  C p10/50/90 {np.percentile(C,10):5.1f} {np.median(C):5.1f} {np.percentile(C,90):5.1f}"
          f"  hue p10/50/90 {np.percentile(h,10):6.1f} {np.median(h):6.1f} {np.percentile(h,90):6.1f}")
    step = max(1, on.sum() // 20000)
    out.append(np.stack([sa[on][::step], sb[on][::step]], 1))
allv = np.concatenate(out)
np.save(r"G:\Ortho4XP_Review\_scratch\realism\tint\canopy_ab.npy", allv)
C = np.hypot(allv[:, 0], allv[:, 1]); h = np.degrees(np.arctan2(allv[:, 1], allv[:, 0]))
print("ALL  C pct 5/25/50/75/95", np.percentile(C, [5, 25, 50, 75, 95]).round(1),
      " hue", np.percentile(h, [5, 25, 50, 75, 95]).round(1))
