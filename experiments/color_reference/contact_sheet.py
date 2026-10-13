"""Side-by-side BI | GO2 | reference per location, 1024 px each."""
import json, os, sys
import numpy as np
from PIL import Image
DATA = os.environ.get("O4_REVIEW_DATA", r"E:\Ortho4XP_Review")
SET = os.path.join(DATA, "color", "set")

def load_ref(d, meta, size):
    ref = np.asarray(Image.open(os.path.join(d, "ref.png")).convert("RGB"))
    o = meta["ref_origin_px"][0]; n = 4096 // meta["ref_scale"]
    return Image.fromarray(ref[o:o + n, o:o + n]).resize((size, size), Image.LANCZOS)

def sheet(key, size=1024, cols=("BI16.jpg", "GO216.jpg")):
    d = os.path.join(SET, key); meta = json.load(open(os.path.join(d, "meta.json")))
    ims = [Image.open(os.path.join(d, c)).convert("RGB").resize((size, size), Image.LANCZOS) for c in cols]
    ims.append(load_ref(d, meta, size))
    out = Image.new("RGB", (size * len(ims) + 8 * (len(ims) - 1), size), (40, 40, 40))
    for i, im in enumerate(ims): out.paste(im, (i * (size + 8), 0))
    return out

if __name__ == "__main__":
    for key in sys.argv[1:] or sorted(os.listdir(SET)):
        sheet(key, 640).save(os.path.join(DATA, "color", "sheets", f"{key}_raw.jpg"), quality=88)
        print(key)
