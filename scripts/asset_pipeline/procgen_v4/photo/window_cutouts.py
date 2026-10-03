"""Cut real windows and doors out of CC0 facade photos (Wikimedia Commons).

    python photo/window_cutouts.py [--review <dir>] [--only <id> ...]

Each source in ``LIBRARY`` is a near-frontal photo with one or more
openings marked by a rough box (pixels of the 1920 px rendition). Every
side of the box snaps to the strongest edge within ``SNAP`` px, measured
separately on the two halves of the side so a slight keystone is kept,
and the four corners are warped to a rectangle. The cut-out is the
opening only: the wall-side surround, the sill and the lintel stay the
bake's, so the photo window sits in the bake's own reveal shadow.

Cut-outs go to ``<tex dir>/windows/lib/<source>_<n>.png`` with
``windows/lib/SOURCES.json``, which records file page, author, license,
window type and the refined corners. ``--review`` writes the photos with
the refined quads drawn on them. window_compose.py decides which source
covers which scheme.
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

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from photo_kits import DEFAULT_TEX_DIR  # noqa: E402

UA = {"User-Agent": "O4XP-procgen-photo/1.0 "
      "(https://github.com/bravo77whiskey/Ortho4XP-SegFormer)"}
API = "https://commons.wikimedia.org/w/api.php"
WIDTH = 1920
SNAP = 22               # px search radius around each rough side
MAX_TILT = 4            # px a side may lean between its two halves
OUT_PX_PER_M = 200      # cut-out resolution (pages are 80 px/m)

_PARAND = ("File:برج های مسکونی در منطقه کوزو ۶ شهر پرند- عکاسی با "
           "موبایل- منظر شهری 05.jpg")
_DRESDEN = "File:Holbeinstraße 94 Haustür Dresden 2011.jpg"
_PITT_407 = "File:Front Door, South Side, 2019-10-07, 05.jpg"
_PITT_427 = "File:Front Door, South Side, Pittsburgh, 2019-10-12, 03.jpg"

# id -> (Commons file, kind, boxes, inset, window type).  The window type
# is the kit window a source may stand in for.  A box is either a rough
# (x0, y0, x1, y1) that refine() snaps, or {"quad", "size"} for an oblique
# shot whose leaf corners are marked by hand.  ``inset`` (top, right,
# bottom, left fractions, or one per box) trims wall and side faces that
# the snapped quad kept.  Sources shot at an angle that shows a door's
# jamb or a slanted floor are not used at all (loppersum, torgau_brown,
# pitt_tall, pitt_maroon were dropped for that).
LIBRARY = {
    # ---- casements (Europe, Mediterranean, South America, generic)
    "hockenheim": ("File:20110627Kaiserstr30-32 Hockenheim2.jpg", "win",
                   [(700, 938, 1250, 1563)], (0, 0.03, 0, 0), "casement"),
    "altlussheim": ("File:20110915Altlussheim05.jpg", "win",
                    [(245, 419, 478, 822), (718, 414, 952, 822),
                     (1443, 413, 1702, 829)],
                    [(0, 0, 0, 0.04), (0, 0, 0, 0.04), None], "casement"),
    "brick_square": ("File:Brick and square window (Unsplash).jpg", "win",
                     [(658, 137, 1243, 1079)], None, "casement"),
    "dresden_win": (_DRESDEN, "win",
                    [(282, 374, 455, 778), (1430, 202, 1763, 758)], None,
                    "casement"),
    # ---- double-hung (North America)
    "pitt_dh": (_PITT_427, "win", [(264, 653, 589, 1505)], None,
                "double_hung"),
    # ---- sliders / modern (Asia, Australia, apartments)
    "parand_up": (_PARAND, "win",
                  [(595, 215, 826, 495), (1104, 207, 1334, 484),
                   (1663, 204, 1897, 476)],
                  [(0.07, 0.03, 0.0, 0.12), (0.07, 0.03, 0.0, 0.07),
                   (0.07, 0.03, 0.0, 0.07)], "slider"),
    "parand_lo": (_PARAND, "win",
                  [(15, 876, 255, 1152), (586, 868, 816, 1152),
                   (1098, 856, 1332, 1148), (1663, 854, 1901, 1148)],
                  # the reveal's side face shows left of the frame
                  [(0.07, 0.03, 0.0, 0.19), (0.07, 0.03, 0.0, 0.155),
                   (0.07, 0.03, 0.0, 0.115), (0.07, 0.03, 0.0, 0.115)],
                  "slider"),
    # ---- doors, European
    "titting": ("File:Am Stock 15 (Titting) Haustür.jpg", "door",
                # the jamb's side face shows on the right
                [(720, 269, 1123, 1037)], (0, 0.11, 0, 0), "door_eu"),
    "beratzhausen": ("File:Königsmühle (Beratzhausen) Haustür.jpg", "door",
                     [(726, 173, 1213, 994)], None, "door_eu"),
    "dresden_door": (_DRESDEN, "door", [(745, 336, 1110, 1206)], None,
                     "door_eu"),
    "torgau_wood": ("File:Portal Elbstraße 11 Torgau (2).jpg", "door",
                    [(434, 799, 1332, 2312)], None, "door_eu"),
    "torgau_dark": ("File:Portal Verwaltungsbau Markt 1 Torgau.jpg", "door",
                    [(814, 640, 1242, 1767)], None, "door_eu"),
    # ---- doors, North American (Pittsburgh South Side)
    "pitt_red": (_PITT_407, "door", [(307, 986, 761, 2287)], None,
                 "door_na"),
    "pitt_double": ("File:Front Door, South Side, Pittsburgh, 2019-10-05, "
                    "07.jpg", "door", [(501, 674, 1148, 1738)], None,
                    "door_na"),
    "pitt_blue": ("File:Front door on 17th Street, South Side, 2022-10-08, "
                  "03.jpg", "door", [(615, 1536, 1291, 3114)], None,
                  "door_na"),
    "pitt_storm": (_PITT_427, "door", [(1165, 917, 1562, 1784)], None,
                   "door_na"),
    # ---- doors, Asia (shot from the left: right leaf edge 10% shorter)
    "taipei": ("File:臺北工業學校書庫入口.jpg", "door",
               [{"quad": [(527, 720), (1007, 732), (994, 1647),
                          (524, 1737)], "size": (1.3, 2.3)}], None,
               "door_asia"),
}


def _get(url, tries=6):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            return urllib.request.urlopen(req, timeout=90).read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                time.sleep(min(90, int(e.headers.get("Retry-After") or 0)
                               or 5 * 2 ** k))
                continue
            raise
    raise RuntimeError(f"gave up on {url}")


def fetch(title, out_dir):
    """The 1920 px rendition of a Commons file plus its license record."""
    # non-Latin titles collapse to "_" alone: add a hash of the title
    safe = (re.sub(r"[^A-Za-z0-9]+", "_", title[5:])[:60].strip("_") + "_" +
            hashlib.sha1(title.encode("utf-8")).hexdigest()[:8])
    path = os.path.join(out_dir, safe + ".jpg")
    meta_path = path + ".json"
    if os.path.isfile(path) and os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            return path, json.load(fh)
    q = urllib.parse.urlencode(dict(action="query", titles=title,
                                    prop="imageinfo", format="json",
                                    iiprop="url|size|extmetadata",
                                    iiurlwidth=WIDTH))
    page = next(iter(json.loads(_get(f"{API}?{q}"))["query"]["pages"]
                     .values()))
    ii = page["imageinfo"][0]
    m = ii["extmetadata"]
    lic = m.get("LicenseShortName", {}).get("value", "")
    if "CC0" not in lic and "Public domain" not in lic:
        raise RuntimeError(f"{title}: license {lic!r} is not CC0/PD")
    url = ii.get("thumburl") or ii["url"]
    with open(path, "wb") as fh:
        fh.write(_get(url))
    meta = {"title": title, "page": ii["descriptionurl"], "license": lic,
            "author": re.sub("<[^>]+>", "",
                             m.get("Artist", {}).get("value", "")).strip(),
            "original_size": [ii["width"], ii["height"]], "url": url}
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    time.sleep(1)
    return path, meta


def _edge_maps(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                         .astype(np.float32), (0, 0), 1.2)
    return (np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)),
            np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)))


def _snap(profile_fn, pos, lo, hi):
    best, best_v = pos, -1.0
    for p in range(int(pos - SNAP), int(pos + SNAP) + 1):
        if p < lo or p >= hi:
            continue
        v = profile_fn(p)
        if v > best_v:
            best, best_v = p, v
    return float(best)


def refine(img, box):
    """Corners (tl, tr, br, bl) with every side snapped per half."""
    gx, gy = _edge_maps(img)
    h, w = gx.shape
    x0, y0, x1, y1 = box
    bh, bw = y1 - y0, x1 - x0
    halves_y = [(y0 + 0.1 * bh, y0 + 0.5 * bh), (y0 + 0.5 * bh, y1 - 0.1 * bh)]
    halves_x = [(x0 + 0.1 * bw, x0 + 0.5 * bw), (x0 + 0.5 * bw, x1 - 0.1 * bw)]

    def col(a, b):
        return lambda x: float(gx[int(a):int(b), x].mean())

    def row(a, b):
        return lambda y: float(gy[y, int(a):int(b)].mean())

    full_y = (y0 + 0.1 * bh, y1 - 0.1 * bh)
    full_x = (x0 + 0.1 * bw, x1 - 0.1 * bw)

    def side(fn, pos, halves, full, lim):
        # two halves -> a slight keystone, but only when both halves
        # agree on it; a disagreement is clutter (a curtain edge, a
        # shadow), and then the whole side snaps as one
        a, b = (_snap(fn(*hh), pos, 0, lim) for hh in halves)
        if abs(a - b) > MAX_TILT:
            a = b = _snap(fn(*full), pos, 0, lim)
        return a, b

    left = side(col, x0, halves_y, full_y, w)
    right = side(col, x1, halves_y, full_y, w)
    top = side(row, y0, halves_x, full_x, h)
    bottom = side(row, y1, halves_x, full_x, h)
    # each half is measured at its middle: extrapolate to the corners
    def ends(v):
        a, b = v
        return a - (b - a) * 0.5, b + (b - a) * 0.5
    lt, lb = ends(left)
    rt, rb = ends(right)
    tl_, tr_ = ends(top)
    bl_, br_ = ends(bottom)
    return [(lt, tl_), (rt, tr_), (rb, br_), (lb, bl_)]


def warp(img, corners, size_px):
    w, h = size_px
    dst = np.float32([(0, 0), (w, 0), (w, h), (0, h)])
    m = cv2.getPerspectiveTransform(np.float32(corners), dst)
    return cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_AREA,
                               borderMode=cv2.BORDER_REFLECT)


def quad_size_m(corners, height_m):
    (a, b, c, d) = np.float32(corners)
    wpx = (np.linalg.norm(b - a) + np.linalg.norm(c - d)) / 2
    hpx = (np.linalg.norm(d - a) + np.linalg.norm(c - b)) / 2
    return height_m * wpx / hpx, height_m


# heights the cut-outs are given (only their aspect is used downstream)
HEIGHT_M = {"win": 1.6, "door": 2.3}


def build(tex_dir, review=None, only=None):
    src_dir = os.path.join(tex_dir, "commons")
    out_dir = os.path.join(tex_dir, "windows", "lib")
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "SOURCES.json")
    report = {}
    if only and os.path.isfile(report_path):
        with open(report_path, encoding="utf-8") as fh:
            report = json.load(fh)
    shown_by_path = {}
    for sid, (title, kind, boxes, inset, wtype) in LIBRARY.items():
        if only and sid not in only:
            continue
        insets = inset if isinstance(inset, list) else [inset] * len(boxes)
        path, meta = fetch(title, src_dir)
        img = cv2.imread(path)
        shown = shown_by_path.setdefault(path, img.copy())
        for n, box in enumerate(boxes):
            if isinstance(box, dict):
                # an oblique shot: the leaf's own corners, marked by hand
                # (snapping would catch the jamb's side face), and its real
                # size, since averaged side lengths under-read the
                # receding width
                corners = [tuple(c) for c in box["quad"]]
                w_m, h_m = box["size"]
                xs = [c[0] for c in corners]
                ys = [c[1] for c in corners]
                box = (int(min(xs)), int(min(ys)), int(max(xs)),
                       int(max(ys)))
            else:
                corners = refine(img, box)
                w_m, h_m = quad_size_m(corners, HEIGHT_M[kind])
            size = (int(round(w_m * OUT_PX_PER_M)),
                    int(round(h_m * OUT_PX_PER_M)))
            cut = warp(img, corners, size)
            t, r, b, l = insets[n] or (0, 0, 0, 0)
            hh, ww = cut.shape[:2]
            cut = cut[int(t * hh):hh - int(b * hh),
                      int(l * ww):ww - int(r * ww)]
            w_m *= 1 - l - r
            h_m *= 1 - t - b
            name = f"{sid}_{n}.png"
            cv2.imwrite(os.path.join(out_dir, name), cut)
            report[name] = dict(meta, source=sid, kind=kind, type=wtype,
                                rough_box=list(box),
                                corners=[[round(x, 1), round(y, 1)]
                                         for x, y in corners],
                                size_m=[round(w_m, 3), round(h_m, 3)])
            cv2.polylines(shown, [np.int32(corners)], True,
                          (0, 255, 255), 3)
            cv2.rectangle(shown, tuple(box[:2]), tuple(box[2:]),
                          (255, 0, 255), 1)
    if review:
        os.makedirs(review, exist_ok=True)
        for path, shown in shown_by_path.items():
            cv2.imwrite(os.path.join(review, os.path.basename(path)[:-4] +
                                     "_quads.jpg"), shown)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tex", default=DEFAULT_TEX_DIR)
    ap.add_argument("--review", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args(argv)
    rep = build(args.tex, args.review, args.only)
    for k, v in rep.items():
        print(k, v["size_m"], v["author"][:30], v["license"])


if __name__ == "__main__":
    main()
