"""Survey Esri World Imagery Wayback capture dates around the colour test spots.

For every Wayback release, query its metadata layer 7 (the source drawn at ZL16)
over a 3x3 ZL14 window round each spot and record which captures cover it.
Output: <DATA>/color/eox_refs/survey.json
"""
import json, math, os, sys
from concurrent.futures import ThreadPoolExecutor
import requests

DATA = os.environ.get("O4_REVIEW_DATA", r"G:\Ortho4XP_Review")
OUT = os.path.join(DATA, "color", "eox_refs")
SPOTS = {  # visual-review hotspot centres (scripts/visual_review/fetch_set.py)
    "khh": ("Kaohsiung city grid", 22.63024, 120.29238),
    "szx": ("Shenzhen, Shajing", 22.71134, 113.78848),
    "wnz": ("Wenzhou, by the runway", 27.89657, 120.86191),
    "shouguang": ("Shouguang greenhouse fields", 36.82336, 118.80352),
}
S = requests.Session()
S.headers["User-Agent"] = "Ortho4XP colour-reference experiment"


def window(lat, lon, z=14, n=3):
    t = 2 ** z
    cx = int((lon + 180) / 360 * t)
    cy = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * t)
    x0, y0 = cx - n // 2, cy - n // 2
    def ll(x, y):
        lo = x / t * 360 - 180
        la = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / t))))
        return lo, la
    w, n_ = ll(x0, y0)
    e, s = ll(x0 + n, y0 + n)
    return x0, y0, (w, s, e, n_)


def query(url, env):
    q = {"geometry": ",".join(f"{v:.6f}" for v in env), "geometryType": "esriGeometryEnvelope",
         "inSR": 4326, "spatialRel": "esriSpatialRelIntersects", "returnGeometry": "false",
         "outFields": "SRC_DATE2,SRC_DESC,SRC_RES,NICE_DESC", "f": "json"}
    for _ in range(4):
        try:
            r = S.get(url + "/7/query", params=q, timeout=60)
            return [f["attributes"] for f in r.json().get("features", [])]
        except Exception:
            pass
    return None


def main():
    cfg = S.get("https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json").json()
    rel = sorted(((v["itemTitle"].split("Wayback ")[1].rstrip(")"), k, v["metadataLayerUrl"]) for k, v in cfg.items()))
    out = {}
    for key, (label, lat, lon) in SPOTS.items():
        x0, y0, env = window(lat, lon)
        with ThreadPoolExecutor(16) as ex:
            res = list(ex.map(lambda r: query(r[2], env), rel))
        rows = []
        for (date, num, _), feats in zip(rel, res):
            caps = sorted({(f["SRC_DESC"], f["NICE_DESC"],
                            __import__("datetime").datetime.utcfromtimestamp(f["SRC_DATE2"] / 1000).strftime("%Y-%m-%d") if f.get("SRC_DATE2") else None,
                            f["SRC_RES"]) for f in (feats or [])}, key=lambda c: str(c[2]))
            rows.append({"release": date, "num": num, "captures": caps})
        out[key] = {"label": label, "lat": lat, "lon": lon, "x0": x0, "y0": y0, "rows": rows}
        print(key, "done", flush=True)
        last = None
        for r in rows:
            sig = tuple(c[2] for c in r["captures"])
            if sig != last:
                print("  ", r["release"], r["num"], [(c[1], c[2], c[3]) for c in r["captures"]])
                last = sig
    with open(os.path.join(OUT, "survey.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
