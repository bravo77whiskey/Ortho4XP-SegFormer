"""Download the photo texture sources and record their licenses.

    python photo/fetch_sources.py [--out photo/sources]

Commons files come as the ``COMMONS_WIDTH`` px rendition; each file page's
license field must read public domain or CC0, or the fetch stops.  Poly
Haven assets (all CC0) come as their colour map.  ``LICENSES.json`` in the
output lists every source with its page, author and license; the texture
builder copies it next to the pages and the library build ships it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sources as S  # noqa: E402

DEFAULT_OUT = os.path.join(HERE, "sources")
UA = {"User-Agent": "O4XP-greenhouse-photo/1.0 "
      "(https://github.com/bravo77whiskey/Ortho4XP-SegFormer)"}
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
POLYHAVEN_API = "https://api.polyhaven.com"


def _get(url, tries=6):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            return urllib.request.urlopen(req, timeout=120).read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                time.sleep(min(90, int(e.headers.get("Retry-After") or 0) or 5 * 2 ** k))
                continue
            raise
    raise RuntimeError(f"gave up on {url}")


def commons_name(title):
    """Local file name of a Commons title (hash keeps non-Latin titles apart)."""
    return (re.sub(r"[^A-Za-z0-9]+", "_", title[5:])[:50].strip("_") + "_" +
            hashlib.sha1(title.encode("utf-8")).hexdigest()[:6] + ".jpg")


def fetch_commons(title, out):
    q = urllib.parse.urlencode(dict(action="query", titles=title, prop="imageinfo",
                                    format="json", iiprop="url|size|extmetadata",
                                    iiurlwidth=S.COMMONS_WIDTH))
    page = next(iter(json.loads(_get(f"{COMMONS_API}?{q}"))["query"]["pages"].values()))
    ii = page["imageinfo"][0]
    meta = ii["extmetadata"]
    lic = meta.get("LicenseShortName", {}).get("value", "")
    if "CC0" not in lic and "Public domain" not in lic:
        raise RuntimeError(f"{title}: license {lic!r} is not CC0 / public domain")
    url = ii.get("thumburl") or ii["url"]
    path = os.path.join(out, commons_name(title))
    if not os.path.isfile(path):
        with open(path, "wb") as fh:
            fh.write(_get(url))
        time.sleep(1)
    strip = lambda key: re.sub("<[^>]+>", "", meta.get(key, {}).get("value", "")).strip()  # noqa: E731
    return {"source": "Wikimedia Commons", "title": title, "file": os.path.basename(path),
            "page": ii["descriptionurl"], "license": lic, "author": strip("Artist"),
            "credit": strip("Credit")[:300], "original_size": [ii["width"], ii["height"]],
            "url": url}


def fetch_polyhaven(asset_id, res, out):
    info = json.loads(_get(f"{POLYHAVEN_API}/info/{asset_id}"))
    entry = json.loads(_get(f"{POLYHAVEN_API}/files/{asset_id}"))["Diffuse"][res]["jpg"]
    path = os.path.join(out, f"{asset_id}_diff_{res}.jpg")
    if not (os.path.isfile(path) and _md5(path) == entry["md5"]):
        with open(path, "wb") as fh:
            fh.write(_get(entry["url"]))
    if _md5(path) != entry["md5"]:
        raise RuntimeError(f"md5 mismatch for {path}")
    return {"source": "Poly Haven", "title": info["name"], "file": os.path.basename(path),
            "page": f"https://polyhaven.com/a/{asset_id}",
            "license": "CC0 1.0 (https://polyhaven.com/license)",
            "author": ", ".join(info["authors"]),
            "size_mm": [round(v) for v in info["dimensions"]],
            "url": entry["url"], "md5": entry["md5"]}


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    records = []
    for title in S.COMMONS:
        records.append(fetch_commons(title, args.out))
        print(f"{records[-1]['file']}: {records[-1]['license']}", flush=True)
    for asset_id, res in S.POLYHAVEN.items():
        records.append(fetch_polyhaven(asset_id, res, args.out))
        print(f"{records[-1]['file']}: CC0", flush=True)
    with open(os.path.join(args.out, "LICENSES.json"), "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
