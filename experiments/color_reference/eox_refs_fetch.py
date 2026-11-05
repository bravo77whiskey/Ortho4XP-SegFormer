"""Fetch EOX S2-cloudless years + dated Esri Wayback releases + current Bing
over a 3x3 ZL14 window round each colour test spot.

Reads <DATA>/color/eox_refs/candidates.json:
  {spot: {"eox_years": [...], "wayback": [[release_num, label], ...]}}
Writes <DATA>/color/eox_refs/<spot>/{eox<year>.png, wb<num>.jpg, wb<num>_z14.png,
bing.jpg, bing_z14.png}.  High-res sources are fetched at ZL16 and area-averaged
down to ZL14 so they sit pixel-for-pixel on EOX.
"""
import io, json, os, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np, requests
from PIL import Image

DATA = os.environ.get("O4_REVIEW_DATA", r"G:\Ortho4XP_Review")
ROOT = os.path.join(DATA, "color", "eox_refs")
S = requests.Session()
S.headers["User-Agent"] = "Ortho4XP colour-reference experiment"
a = requests.adapters.HTTPAdapter(pool_maxsize=32)
S.mount("https://", a); S.mount("http://", a)
N = 3  # ZL14 tiles per side
EOX = ("https://tiles.maps.eox.at/wmts/?layer=s2cloudless-{year}_3857&style=default"
       "&tilematrixset=GoogleMapsCompatible&Service=WMTS&Request=GetTile&Version=1.0.0"
       "&Format=image%2Fjpeg&TileMatrix={z}&TileCol={x}&TileRow={y}")
WB = ("https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/"
      "default028mm/MapServer/tile/{num}/{z}/{y}/{x}")
BING = "http://r{s}.ortho.tiles.virtualearth.net/tiles/a{q}.jpeg?g=136"


def qk(x, y, z):
    return "".join(str(((x >> (i - 1)) & 1) + 2 * ((y >> (i - 1)) & 1)) for i in range(z, 0, -1))


def get(url):
    for _ in range(5):
        try:
            r = S.get(url, timeout=40)
            if r.ok and r.content:
                return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))
        except Exception:
            pass
    raise RuntimeError(url)


def mosaic(fn, x0, y0, n):
    coords = [(x0 + i, y0 + j) for j in range(n) for i in range(n)]
    with ThreadPoolExecutor(24) as ex:
        tiles = list(ex.map(lambda c: fn(*c), coords))
    m = np.zeros((n * 256, n * 256, 3), np.uint8)
    for (x, y), t in zip(coords, tiles):
        m[(y - y0) * 256:(y - y0 + 1) * 256, (x - x0) * 256:(x - x0 + 1) * 256] = t[:256, :256]
    return m


def down4(img):
    h, w, _ = img.shape
    f = img.astype(np.float32).reshape(h // 4, 4, w // 4, 4, 3).mean((1, 3))
    return np.clip(f + 0.5, 0, 255).astype(np.uint8)


def hires(fn, x0, y0, d, name):
    jp, zp = os.path.join(d, name + ".jpg"), os.path.join(d, name + "_z14.png")
    if os.path.isfile(zp):
        return
    m = mosaic(fn, x0 * 4, y0 * 4, N * 4)
    Image.fromarray(m).save(jp, quality=90)
    Image.fromarray(down4(m)).save(zp)
    print("  ", name, flush=True)


def main(only):
    survey = json.load(open(os.path.join(ROOT, "survey.json")))
    cands = json.load(open(os.path.join(ROOT, "candidates.json")))
    for spot, c in cands.items():
        if only and spot not in only:
            continue
        x0, y0 = survey[spot]["x0"], survey[spot]["y0"]
        d = os.path.join(ROOT, spot)
        os.makedirs(d, exist_ok=True)
        print(spot, flush=True)
        for year in c["eox_years"]:
            p = os.path.join(d, f"eox{year}.png")
            if not os.path.isfile(p):
                Image.fromarray(mosaic(lambda x, y: get(EOX.format(year=year, z=14, x=x, y=y)), x0, y0, N)).save(p)
                print("   eox", year, flush=True)
        for num, *_ in c["wayback"]:
            hires(lambda x, y: get(WB.format(num=num, z=16, x=x, y=y)), x0, y0, d, f"wb{num}")
        hires(lambda x, y: get(BING.format(s=(x + y) % 4, q=qk(x, y, 16))), x0, y0, d, "bing")


if __name__ == "__main__":
    main(sys.argv[1:])
