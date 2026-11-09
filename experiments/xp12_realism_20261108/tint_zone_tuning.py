"""How many leaf colour buckets a texture gets, and how many tint-zone pieces, for a few
smoothing radii / minimum zone sizes (cached SegFormer tree class + BI hotspot jpgs)."""
import glob, os, sys, numpy as np, cv2
from PIL import Image
sys.path.insert(0, r"G:\Dev\Ortho4XP-SegFormer\src")
import O4_Forest_Tint as T
import O4_Vegetation_Overlay as VEG
from O4_Landcover_Inference import CLASS_TREE
STEMS = sys.argv[1:] or ["28528_53488_BI16", "28528_54672_BI16", "25536_54384_BI16", "27472_54752_BI16"]
for stem in STEMS:
    veg = glob.glob(rf"G:\Ortho4XP_Review\_cache\w*\*\*\{stem}_veg.npy")
    if not veg: print(stem, "no cache"); continue
    im = np.asarray(Image.open(rf"G:\Ortho4XP_Review\sets\hotspots\BI\{stem}.jpg").convert("RGB"))
    mask = (np.load(veg[0]) == CLASS_TREE).astype(np.uint8)
    mpp = 156543.03 * np.cos(np.radians(25)) / 2 ** 16 * 256 / mask.shape[1] * 16
    im = cv2.resize(im, mask.shape[::-1], interpolation=cv2.INTER_AREA)
    for sig, minm2 in ((60, 5000), (120, 20000), (200, 50000)):
        z = T.tint_zones(im, mask, mpp, sigma_m=sig)
        z = VEG._merge_small_zones(z, mpp, minm2)
        ids, area = np.unique(z[z > 0], return_counts=True)
        frac = area / area.sum()
        pieces = sum(cv2.connectedComponents((z == i).astype(np.uint8))[0] - 1 for i in ids)
        top = ", ".join(f"t{i-1}:{f:.0%}" for i, f in sorted(zip(ids, frac), key=lambda x: -x[1])[:5])
        print(f"{stem} sigma {sig:3d} m min {minm2/1e4:.1f} ha: {len(ids)} colours ({(frac>0.05).sum()} over 5%), {pieces} zone pieces | {top}")
