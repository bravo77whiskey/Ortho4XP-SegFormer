"""Download the CC0 photo textures the photo pages use, with provenance.

    python photo/fetch_cc0.py [--out <dir>]

Every texture comes from Poly Haven (polyhaven.com/license: CC0, no
attribution required, redistribution allowed). Poly Haven textures are
photoscanned on location. For each asset this writes the colour map, the
displacement and AO maps, and a line in ``LICENSES.json`` with the asset
page, authors, real-world size and file hashes, so the textures can be
traced if the library is ever shared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from photo_kits import ASSETS, DEFAULT_TEX_DIR  # noqa: E402

API = "https://api.polyhaven.com"
UA = {"User-Agent": "O4SFR-procgen-photo/1.0"}
MAPS = {"diff": ("Diffuse", "jpg"), "disp": ("Displacement", "png"),
        "ao": ("AO", "jpg")}


def _get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def fetch(asset_id, res, out_dir):
    info = json.loads(_get(f"{API}/info/{asset_id}"))
    files = json.loads(_get(f"{API}/files/{asset_id}"))
    record = {
        "id": asset_id, "name": info["name"],
        "page": f"https://polyhaven.com/a/{asset_id}",
        "license": "CC0 1.0 (https://polyhaven.com/license)",
        "authors": info["authors"],
        "size_mm": [round(v) for v in info["dimensions"]],
        "res": res, "files": {},
    }
    for key, (api_name, ext) in MAPS.items():
        entry = files[api_name][res][ext]
        path = os.path.join(out_dir, f"{asset_id}_{key}_{res}.{ext}")
        if not (os.path.isfile(path) and _md5(path) == entry["md5"]):
            data = _get(entry["url"])
            with open(path, "wb") as fh:
                fh.write(data)
        if _md5(path) != entry["md5"]:
            raise RuntimeError(f"md5 mismatch for {path}")
        record["files"][key] = {"file": os.path.basename(path),
                                "url": entry["url"], "md5": entry["md5"]}
    return record


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_TEX_DIR)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    records = []
    for asset_id, res in sorted(ASSETS.items()):
        records.append(fetch(asset_id, res, args.out))
        print(f"{asset_id}: ok ({res})", flush=True)
    with open(os.path.join(args.out, "LICENSES.json"), "w",
              encoding="utf-8") as fh:
        json.dump(records, fh, indent=1)


if __name__ == "__main__":
    main()
