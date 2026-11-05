"""A/B of the colour correction with the EOX reference raw vs calibrated to Esri.

Runs src/O4_Color_Correction.Group (fit + correct) on the visual-review
hotspot Bing ZL16 textures of Kaohsiung and Shenzhen, against EOX 2025:
  cur   - production (raw EOX, current look + dewarm)
  calA  - calibrated EOX, current look + dewarm
  calB  - calibrated EOX, no look (reference colours are already Esri's)
  prod  - the production code as it is (calibration inside O4_Color_Correction)
and compares each with the Esri release captured Dec 2024 - Mar 2025.
Writes <DATA>/color/eox_refs/ab/<variant>/<ty>_<tx>.jpg (+ esri/, bi/).

Usage (CUDA venv): python eox_calib_ab.py [variant ...]
"""
import io, json, os, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np, requests, cv2, torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)
import O4_Color_Correction as CC  # noqa: E402
import eox_calib_fit as CF  # noqa: E402

DATA = os.environ.get("O4_REVIEW_DATA", r"G:\Ortho4XP_Review")
ROOT = os.path.join(DATA, "color", "eox_refs")
AB = os.path.join(ROOT, "ab")
SET = os.path.join(DATA, "sets", "hotspots", "BI")
YEAR = 2025
GROUPS = {"khh": (22.63, [(54656, 28528), (54672, 28528), (54672, 28544)]),
          "szx": (22.71, [(53472, 28512), (53488, 28512), (53488, 28528)])}
# Not in the fit (validation where red-brown fields are common).  Esri there:
# Wenzhou Apr 2025 (51127); Shouguang has no clean 2025 release (10842 = 2020-2025 mix).
CHECK = {"wnz": (27.9, [(54768, 27456), (54752, 27472), (54768, 27472), (54752, 27488)]),
         "shouguang": (36.82, [(54384, 25536)])}
RELEASE = {"khh": 48925, "szx": 48925, "wnz": 51127, "shouguang": 10842}
if os.environ.get("CHECK"):
    GROUPS = CHECK
WB = ("https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/"
      "default028mm/MapServer/tile/{rel}/{z}/{y}/{x}")
EOX = ("https://tiles.maps.eox.at/wmts/?layer=s2cloudless-{year}_3857&style=default"
       "&tilematrixset=GoogleMapsCompatible&Service=WMTS&Request=GetTile&Version=1.0.0"
       "&Format=image%2Fjpeg&TileMatrix={z}&TileCol={x}&TileRow={y}")
S = requests.Session()
S.headers["User-Agent"] = "Ortho4XP colour-reference experiment"
S.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=32))


def get(url):
    import time
    for i in range(8):
        try:
            r = S.get(url, timeout=40)
            if r.ok and r.content:
                return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))
        except Exception:
            pass
        time.sleep(2 * (i + 1))
    raise RuntimeError(url)


def texture(url_fn, tx, ty, path):
    """16x16 tiles from (tx, ty) as one 4096 px texture, cached."""
    if os.path.isfile(path):
        return np.asarray(Image.open(path).convert("RGB"))
    coords = [(tx + i, ty + j) for j in range(16) for i in range(16)]
    with ThreadPoolExecutor(8) as ex:
        tiles = list(ex.map(lambda c: get(url_fn(*c)), coords))
    m = np.zeros((4096, 4096, 3), np.uint8)
    for (x, y), t in zip(coords, tiles):
        m[(y - ty) * 256:(y - ty + 1) * 256, (x - tx) * 256:(x - tx + 1) * 256] = t[:256, :256]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(m).save(path, quality=92) if path.endswith(".jpg") else Image.fromarray(m).save(path)
    return m


CAL = json.load(open(os.path.join(ROOT, f"calib_{YEAR}.json")))


def calibrate(rgb):
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    out = np.empty_like(lab)
    for y in range(0, lab.shape[0], 512):
        out[y:y + 512] = CF.apply(CAL, lab[y:y + 512])
    rgb2 = cv2.cvtColor(out.astype(np.float32), cv2.COLOR_LAB2RGB)
    return np.clip(rgb2 * 255 + 0.5, 0, 255).astype(np.uint8)


def run(variant):
    cal = variant.startswith("cal")
    real_dewarm = CC.dewarm
    for key, (lat, tex) in GROUPS.items():
        def loader(provider, tx, ty):
            return np.array(Image.open(os.path.join(SET, f"{ty}_{tx}_BI16.jpg")).convert("RGB"))

        def fetch_reference(tx, ty):
            raw = texture(lambda x, y: EOX.format(year=YEAR, z=14, x=x, y=y), tx, ty,
                          os.path.join(ROOT, "_ref", f"EOX{YEAR}", f"{ty}_{tx}.png"))
            if not cal:
                return raw
            p = os.path.join(ROOT, "_ref", f"EOX{YEAR}cal", f"{ty}_{tx}.png")
            if not os.path.isfile(p):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                Image.fromarray(calibrate(raw)).save(p)
            return np.asarray(Image.open(p).convert("RGB"))
        g = CC.Group(lat, 16, "BI", tex, loader, YEAR)
        if variant != "prod":
            g.calibration = None             # these variants bring their own reference
        if variant in ("cur", "calA"):       # the look before calibration (MARK_VERSION 2)
            G, O_, S_ = CC.LOOK
            grey = CC.REF_GREY[YEAR]
            g.look = (G, O_, S_, tuple(CC.PROVIDER_GREY[i] - S_ * grey[i] for i in range(2)), grey)
        with torch.no_grad():
            g.fit(fetch_reference, clouds=True)
            for tx, ty in tex:
                res, _ = g.correct(loader("BI", tx, ty), tx, ty)
                out = os.path.join(AB, variant, f"{ty}_{tx}.jpg")
                os.makedirs(os.path.dirname(out), exist_ok=True)
                Image.fromarray(res.permute(1, 2, 0).cpu().numpy()).save(out, quality=92)
        CC.dewarm = real_dewarm
        print(variant, key, "done", flush=True)


def stats_of(rgb):
    m = rgb.astype(np.float32).reshape(256, 16, 256, 16, 3).mean((1, 3))     # ~38 m cells
    lab = cv2.cvtColor(np.round(m).astype(np.float32) / 255, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    full = cv2.cvtColor(rgb[::4, ::4].astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[..., 0]
    C = np.hypot(lab[:, 1], lab[:, 2]); h = np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) % 360
    warm = (h < 75) & (C > 8)
    return [lab[:, 0].mean(), lab[:, 0].std(), full.std(), lab[:, 1].mean(), lab[:, 2].mean(), C.mean(),
            np.percentile(C, 90), 100 * warm.mean(), C[warm].mean() if warm.any() else 0]


def compare(variants):
    names = ["L", "Lsd38", "Lsd", "a*", "b*", "C", "C90", "warm%", "warmC"]
    print(f"{'':<14}" + "".join(f"{n:>7}" for n in names))
    for key, (lat, tex) in GROUPS.items():
        print("==", key)
        rows = {"esri": [], "bi": []} | {v: [] for v in variants}
        for tx, ty in tex:
            e = texture(lambda x, y: WB.format(rel=RELEASE[key], z=16, x=x, y=y), tx, ty, os.path.join(AB, "esri", f"{ty}_{tx}.jpg"))
            rows["esri"].append(stats_of(e))
            rows["bi"].append(stats_of(np.asarray(Image.open(os.path.join(SET, f"{ty}_{tx}_BI16.jpg")).convert("RGB"))))
            for v in variants:
                rows[v].append(stats_of(np.asarray(Image.open(os.path.join(AB, v, f"{ty}_{tx}.jpg")).convert("RGB"))))
        for k, r in rows.items():
            print(f"{k:<14}" + "".join(f"{x:7.1f}" for x in np.mean(r, 0)))


if __name__ == "__main__":
    vs = sys.argv[1:] or ["cur", "calA", "calB"]
    for v in vs:
        run(v)
    compare(vs)
