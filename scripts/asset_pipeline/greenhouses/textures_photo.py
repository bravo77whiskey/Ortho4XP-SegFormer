"""Photo texture set: real photographs composited into the layout.

    python photo/fetch_sources.py
    python textures_photo.py --out <pages dir>

Every pixel comes from a public-domain or CC0 photograph listed in
``photo/sources.py`` (provenance in ``photo/sources/LICENSES.json``, copied
next to the pages).  This script unrolls, crops, tiles at real-world scale,
mirrors and pastes them; it never paints.  Normal maps are derived from
luminance as in ``textures_ai.py``.

* tunnel and solar film: one USDA raspberry tunnel, unrolled hoop by hoop
  (``photo/unroll.py``);
* tunnel gables: open tunnel ends from the same USDA set, so ``doors.json``
  has no door frames for them;
* masonry, render, thatch, quilts / straw mats, soil, concrete: Poly Haven
  scans tiled at their measured size, with real doors pasted where the
  layout puts one.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "photo"))
import layout as L  # noqa: E402
import textures_ai as A  # noqa: E402
import fetch_sources as F  # noqa: E402
import sources as S  # noqa: E402
import unroll as U  # noqa: E402

SRC = os.path.join(HERE, "photo", "sources")

# real sizes (m) the regions stand for, for tiling scans at their own scale
TUNNEL_ARCH_M = 10.0               # foot-to-foot arc of a 7 x 3.2 m tunnel
SOLAR_FILM_M = 10.0                # kerb-to-ridge arc of a 14 m deep house
QUILT_ROUND_M = 2.1                # circumference of a rolled quilt
BACK_SLOPE_M = 1.5                 # back roof, ridge to coping
WALL_H_M = 3.6                     # back wall height of a 14 m deep house
GABLE_M = (14.0, 4.65)             # solar gable: depth, height
ANTE_M = (11.0, 2.6)               # anteroom perimeter, height
ANTE_ROOF_M = (3.5, 4.0)
FILM_HOOPS = {"roof_a": 10, "solar": 10, "roof_b": 8}   # hoops per 12 m


def load(name):
    return np.asarray(Image.open(os.path.join(SRC, name)).convert("RGB"),
                      dtype=np.float32) / 255.0


def load_commons(title):
    return load(F.commons_name(title))


def load_scan(asset_id):
    return load(f"{asset_id}_diff_{S.POLYHAVEN[asset_id]}.jpg")


def scan_size_m(asset_id):
    with open(os.path.join(SRC, "LICENSES.json"), encoding="utf-8") as fh:
        rec = next(r for r in json.load(fh) if r.get("page", "").endswith("/" + asset_id))
    return rec["size_mm"][0] / 1000.0, rec["size_mm"][1] / 1000.0


def tile(scan, scan_m, w_px, h_px, w_m, h_m, offset=(0.0, 0.0), whole=(False, True)):
    """A seamless scan repeated over w_m x h_m metres at w_px x h_px.

    ``whole`` (horizontal, vertical) rounds that repeat count so the result
    tiles along it too (the along axis of a tiling region is vertical).
    """
    sh, sw = scan.shape[:2]
    rx = w_m / scan_m[0]
    ry = h_m / scan_m[1]
    if whole[0]:
        rx = max(1, round(rx))
    if whole[1]:
        ry = max(1, round(ry))
    u = (np.arange(w_px) + 0.5) / w_px * rx * sw + offset[0] * sw
    v = (np.arange(h_px) + 0.5) / h_px * ry * sh + offset[1] * sh
    # pre-blur to the output pixel footprint so fine detail does not alias
    sigma = (0.4 * ry * sh / h_px, 0.4 * rx * sw / w_px)
    src = np.stack([ndimage.gaussian_filter(scan[..., c], sigma, mode="wrap") for c in range(3)], -1)
    vv, uu = np.meshgrid(v, u, indexing="ij")
    return np.stack([ndimage.map_coordinates(src[..., c], [vv, uu], order=1, mode="grid-wrap")
                     for c in range(3)], -1)


def film_window(strip, rows, hoops, first, w, h, mirror):
    """``hoops`` hoop spacings from hoop ``first`` (counted from the near
    end), plus the overlap ``seamless_y`` cuts, as a tiling w x h strip."""
    period = rows[1] - rows[0]
    near = rows[-1] - first * period
    span = hoops * period * (1 + 0.12)
    top = near - span
    if top < 0:
        raise ValueError(f"film window from hoop {first}: only {near / period:.1f} hoops")
    win = strip[int(round(top)):int(round(near))]
    if mirror:
        win = win[:, ::-1]
    win = U.flatten(win, 60)
    return A.seamless_y(win, w, h)


def paste(img, patch, x0, y0):
    """Paste ``patch`` with its top-left at (x0, y0), rows from the top."""
    h, w = patch.shape[:2]
    img[y0:y0 + h, x0:x0 + w] = patch
    return img


def door_patch(name, w_px, h_px):
    title, (x0, y0, x1, y1) = S.DOORS[name]
    return A.resize(load_commons(title)[y0:y1, x0:x1], max(1, w_px), max(1, h_px))


def gable_end(title, gl, gr, apex, size):
    """Open tunnel end: the quad on the ground line and through the apex."""
    img = load_commons(title)
    slope = (gr[1] - gl[1]) / (gr[0] - gl[0])
    top = lambda x: apex[1] + (x - apex[0]) * slope  # noqa: E731
    quad = [(gl[0], top(gl[0])), (gr[0], top(gr[0])), gr, gl]
    return U.warp_quad(img, quad, *size)


def build(out):
    pages = {p: (Image.new("RGB", (L.PAGE_PX, L.PAGE_PX), (128, 128, 128)),
                 Image.new("RGBA", (L.PAGE_PX, L.PAGE_PX), (128, 128, 10, 60)))
             for p in ("tunnels", "solar")}
    T, SO = L.TUNNELS, L.SOLAR

    strip, rows, _ = U.unroll(load_commons(S.FILM_SOURCE), S.FILM_GAP_LEFT,
                              S.FILM_GAP_RIGHT, S.FILM_Y, S.FILM_COLUMNS)

    mud = load_scan("brown_mud_dry")
    mud_m = scan_size_m("brown_mud_dry")
    for name, seed in (("roof_a", 1), ("roof_b", 2)):
        w, h = T[name].size_px
        img = film_window(strip, rows, FILM_HOOPS[name], S.FILM_WINDOWS[name], w, h,
                          mirror=(name == "roof_b"))
        soil = tile(mud, mud_m, w, h, TUNNEL_ARCH_M, L.ALONG_PERIOD_M, (0.3 * seed, 0))
        img = A.soil_feet(img, soil, L.FOOT_SOIL, 0.1, seed)
        A.place(pages, "tunnels", T[name], img, 2.0, 0.6)

    doors = {}
    for gname in L.TUNNEL_GABLES:
        w, h = T[gname].size_px
        A.place(pages, "tunnels", T[gname], gable_end(*S.GABLE_ENDS[gname], (w, h)), 2.5, 0.5)
        doors[gname] = None                     # open ends: no door frame

    w, h = SO["film"].size_px
    img = film_window(strip, rows, FILM_HOOPS["solar"], S.FILM_WINDOWS["solar"], w, h, mirror=True)
    soil = tile(mud, mud_m, w, h, SOLAR_FILM_M, L.ALONG_PERIOD_M, (0.6, 0.2))
    # the film starts on top of the masonry kerb: only a little mud splash
    img = A.soil_feet(img, soil, 0.0, 0.05, 3, both=False)
    a = (np.arange(w) + 0.5) / w
    img = img * (1 - 0.35 * A.smooth(a, 0.84, 0.92))[None, :, None]   # quilt shade at the ridge
    A.place(pages, "solar", SO["film"], img, 2.0, 0.55)

    # rolled covers side by side along the roll: fabric quilts and straw
    # mats (straws wrap round the roll, i.e. run across)
    w, h = SO["quilt"].size_px
    thatch = np.ascontiguousarray(np.rot90(load_scan("thatch_roof_angled")))
    covers = [(load_scan(f), scan_size_m(f)) for f in ("knitted_fleece", "denim_fabric_05")]
    covers.append((thatch, scan_size_m("thatch_roof_angled")[::-1]))
    rng = np.random.default_rng(7)
    parts, filled = [], 0
    while filled < h:
        seg = int(rng.integers(h // 8, h // 5))        # 1.5-2.4 m wide covers
        scan, size = covers[int(rng.integers(len(covers)))]
        piece = tile(scan, size, w, seg, QUILT_ROUND_M, L.ALONG_PERIOD_M * seg / h,
                     tuple(rng.random(2)), whole=(False, False))
        parts.append(piece * float(rng.uniform(0.85, 1.1)))
        filled += seg
    img = np.concatenate(parts, axis=0)[:h]
    A.place(pages, "solar", SO["quilt"], np.clip(img, 0, 1), 3.0, 0.1)

    # straw back roof: strands run down the slope (across)
    w, h = SO["back"].size_px
    tm = scan_size_m("thatch_roof_angled")
    A.place(pages, "solar", SO["back"],
            tile(thatch, tm[::-1], w, h, BACK_SLOPE_M, L.ALONG_PERIOD_M), 3.0, 0.1)

    # back walls (across = height, bottom first; along = wall length) and
    # gables with a rusty steel door near the north end
    gw_m, gh_m = GABLE_M
    door_w, door_h, door_from_n = S.GABLE_DOOR_M
    for v, (wall_name, gable_name) in L.SOLAR_FINISH.items():
        scan_id = "worn_brick_wall" if v == "a" else "concrete_wall_008"
        scan, size = load_scan(scan_id), scan_size_m(scan_id)
        if v == "b":
            # its stains would repeat every 2.7 m along a 100 m wall
            scan = U.flatten(scan, scan.shape[0] / 5)
        w, h = SO[wall_name].size_px
        # tile as a wall (rows = height, top first; whole repeats along its
        # length), then turn so the height runs across from the bottom and
        # the length runs along
        face = tile(scan, size, h, w, L.ALONG_PERIOD_M, WALL_H_M, whole=(True, False))
        face = np.ascontiguousarray(np.transpose(face[::-1], (1, 0, 2)))
        A.place(pages, "solar", SO[wall_name], face, 1.5, 0.12)

        w, h = SO[gable_name].size_px
        g = tile(scan, size, w, h, gw_m, gh_m, (0.37, 0.0), whole=(False, False))
        dw, dh = round(door_w / gw_m * w), round(door_h / gh_m * h)
        x1 = round((gw_m - door_from_n) / gw_m * w)
        paste(g, door_patch("rusty", dw, dh), x1 - dw, h - dh)
        A.place(pages, "solar", SO[gable_name], g, 3.0, 0.12)

    # anteroom: the back wall's brick, blue steel door on the south face
    # (ANTE_DOOR)
    w, h = SO["ante_wall"].size_px
    img = tile(load_scan("worn_brick_wall"), scan_size_m("worn_brick_wall"), w, h, *ANTE_M,
               (0.41, 0.0), whole=(False, False))
    a0, a1, b0, b1 = L.ANTE_DOOR
    x0, x1 = round(a0 * w), round(a1 * w)
    dh = round((b1 - b0) * h)
    paste(img, door_patch("blue", x1 - x0, dh), x0, h - round(b0 * h) - dh)
    A.place(pages, "solar", SO["ante_wall"], img, 3.0, 0.1)

    w, h = SO["ante_roof"].size_px
    A.place(pages, "solar", SO["ante_roof"],
            tile(load_scan("dirty_concrete"), scan_size_m("dirty_concrete"), w, h, *ANTE_ROOF_M,
                 whole=(False, False)), 2.5, 0.15)

    os.makedirs(out, exist_ok=True)
    for p, (a_img, n_img) in pages.items():
        a_img.save(os.path.join(out, f"{p}.png"))
        n_img.save(os.path.join(out, f"{p}_nml.png"))
    with open(os.path.join(out, "doors.json"), "w", encoding="utf-8") as fh:
        json.dump(doors, fh, indent=1)
    shutil.copyfile(os.path.join(SRC, "LICENSES.json"), os.path.join(out, "LICENSES.json"))
    print(f"photo pages -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    build(os.path.abspath(ap.parse_args().out))


if __name__ == "__main__":
    main()
