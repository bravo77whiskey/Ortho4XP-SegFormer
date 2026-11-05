"""Colour statistics of each dated Esri / Bing reference vs the EOX year it's paired with (ZL14 grid)."""
import json, os
import numpy as np, cv2

ROOT = os.path.join(os.environ.get("O4_REVIEW_DATA", r"G:\Ortho4XP_Review"), "color", "eox_refs")


def lab(p):
    im = cv2.imread(p)[:, :, ::-1]
    return cv2.cvtColor(im.astype(np.float32) / 255, cv2.COLOR_RGB2LAB), im


def stats(L, valid):
    l, a, b = L[..., 0][valid], L[..., 1][valid], L[..., 2][valid]
    c = np.hypot(a, b); h = np.degrees(np.arctan2(b, a)) % 360
    warm = (h < 75) & (c > 8)
    return {"L": l.mean(), "Lsd": l.std(), "L5_95": np.percentile(l, 95) - np.percentile(l, 5),
            "a": a.mean(), "b": b.mean(), "C": c.mean(), "C90": np.percentile(c, 90),
            "warm%": 100 * warm.mean(), "warmC": c[warm].mean() if warm.any() else 0}


cands = json.load(open(os.path.join(ROOT, "candidates.json")))
out = {}
fmt = "{:<34}" + " {:>6}" * 9
print(fmt.format("source", "L", "Lsd", "L5-95", "a*", "b*", "C", "C90", "warm%", "warmC"))
for spot, c in cands.items():
    print("==", spot)
    d = os.path.join(ROOT, spot)
    rows = [(f"wb{n}_z14.png", lab_, y) for n, lab_, y in c["wayback"]] + [("bing_z14.png", "Bing (undated, <=2023)", 2022)]
    out[spot] = []
    for f, label, year in rows:
        R, rim = lab(os.path.join(d, f)); E, _ = lab(os.path.join(d, f"eox{year}.png"))
        valid = (rim.std(-1) > 0.5) | (rim.mean(-1) > 5)  # drop flat 'no data' fills
        valid &= ~((np.abs(rim.astype(int) - 204).max(-1) < 3))
        sr, se = stats(R, valid), stats(E, valid)
        print(fmt.format(label[:34], *[f"{sr[k]:.1f}" for k in sr]))
        print(fmt.format(f"   EOX {year}", *[f"{se[k]:.1f}" for k in se]))
        out[spot].append({"file": f, "label": label, "year": year, "ref": {k: float(v) for k, v in sr.items()},
                          "eox": {k: float(v) for k, v in se.items()}, "valid%": float(100 * valid.mean())})
json.dump(out, open(os.path.join(ROOT, "stats.json"), "w"), indent=1)
