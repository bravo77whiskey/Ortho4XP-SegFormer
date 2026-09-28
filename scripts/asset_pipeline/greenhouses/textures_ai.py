"""AI-image texture set: composite ``ai/sources/*.png`` into the layout.

    python textures_ai.py --out <pages dir>

The source pixels come from an image-generation model (codex, prompt in
``ai/PROMPT_IMAGES.md``); this script only crops, rotates, scales,
mirror-tiles and blends them, and derives normal maps from luminance.

Tiling along a greenhouse is closed with a minimum-error cut: the last
rows of a strip overlap its first rows and each column switches from one
to the other where they already agree (image quilting), so the join has
no mirror line and no cross-fade ghosting.  Gable doors differ per image, so their positions
go to ``doors.json`` beside the pages for the geometry's door frames.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import layout as L  # noqa: E402

SRC = os.path.join(HERE, "ai", "sources")

# arch bounding boxes (px) and door rectangles (a0, a1, b0, b1 inside the
# arch box, b from the ground) measured on the two generated end walls
ENDS = {
    "tunnel_end_a.png": ((5, 139, 1244, 1073), (0.38, 0.62, 0.0, 0.74)),
    "tunnel_end_b.png": ((7, 78, 1247, 1166), (0.21, 0.41, 0.0, 0.60)),
}


def load(name):
    return np.asarray(Image.open(os.path.join(SRC, name)).convert("RGB"),
                      dtype=np.float32) / 255.0


def resize(img, w, h):
    return np.asarray(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
                      .resize((int(w), int(h)), Image.LANCZOS), np.float32) / 255.0


def rot_cw(img):
    return np.ascontiguousarray(np.rot90(img, k=-1))


def mirror_tile_y(img, w, h, copies):
    """``copies`` (even) alternately flipped copies stacked to h px."""
    assert copies % 2 == 0
    each = h // copies
    tile = resize(img, w, each)
    parts = [tile if k % 2 == 0 else tile[::-1] for k in range(copies)]
    out = np.concatenate(parts, axis=0)
    if out.shape[0] < h:
        out = np.concatenate([out, out[: h - out.shape[0]][::-1]], axis=0)
    return out[:h]


def wrap_min_cut(img, overlap):
    """Make ``img`` tile vertically; returns it ``overlap`` rows shorter.

    Row r of the overlap comes from the strip's end above the cut and from
    its start below it; the cut row per column is a minimum-cost path whose
    step between neighbouring columns is at most one row.
    """
    h = img.shape[0]
    top, bot = img[:overlap], img[h - overlap:]
    cost = ((top - bot) ** 2).sum(axis=-1)            # (overlap, w)
    o, w = cost.shape
    acc = cost.copy()
    back = np.zeros((o, w), np.int64)
    for x in range(1, w):
        prev = acc[:, x - 1]
        cand = np.stack([np.roll(prev, 1), prev, np.roll(prev, -1)])
        cand[0, 0] = np.inf
        cand[2, -1] = np.inf
        k = np.argmin(cand, axis=0)
        back[:, x] = np.arange(o) + (k - 1)
        acc[:, x] += cand[k, np.arange(o)]
    cut = np.zeros(w, np.int64)
    cut[-1] = int(np.argmin(acc[:, -1]))
    for x in range(w - 1, 0, -1):
        cut[x - 1] = back[cut[x], x]
    out = img[: h - overlap].copy()
    rows = np.arange(o)[:, None]
    take_bot = rows < cut[None, :]
    out[:o][take_bot] = bot[take_bot]
    return out


def seamless_y(img, w, h, copies=1, overlap_frac=0.12, prep=None):
    """``copies`` repeats of ``img`` filling h rows, joins cut, wrap cut."""
    each = h // copies
    o = max(4, int(each * overlap_frac))
    tile = resize(img, w, each + o)
    if prep is not None:
        tile = prep(tile)
    tile = wrap_min_cut(tile, o)
    out = np.concatenate([tile] * copies, axis=0)
    if out.shape[0] < h:
        out = np.concatenate([out, tile[: h - out.shape[0]]], axis=0)
    return out[:h]


def mirror_tile_x(img, w, h, copies):
    return np.ascontiguousarray(
        np.transpose(mirror_tile_y(np.transpose(img, (1, 0, 2)), h, w, copies), (1, 0, 2)))


def smooth(x, e0, e1):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def soil_feet(img, soil, band, splash_to, seed, both=True):
    """Soil over the film feet (across ends) and mud splashed above them."""
    h, w, _ = img.shape
    rng = np.random.default_rng(seed)
    a = (np.arange(w) + 0.5) / w
    foot = np.minimum(a, 1 - a) if both else a
    s = resize(soil, w, h)
    noise = ndimage.gaussian_filter(rng.standard_normal((h, w)), 3, mode="wrap")
    noise = (noise - noise.min()) / (np.ptp(noise) + 1e-6)
    buried = (foot < band)[None, :] * np.ones((h, 1))
    splash = (1 - smooth(foot, band, splash_to))[None, :] * (0.4 + 0.6 * noise)
    out = img * (1 - 0.55 * splash[..., None]) + s * 0.8 * (0.55 * splash[..., None])
    return out * (1 - buried[..., None]) + s * buried[..., None]


def normal_map(img, strength, gloss, gloss_var=0.15):
    lum = img @ np.array([0.3, 0.59, 0.11], np.float32)
    hp = lum - ndimage.gaussian_filter(lum, 6, mode="wrap")
    hgt = ndimage.gaussian_filter(hp, 0.8, mode="wrap") * strength
    gy, gx = np.gradient(hgt)
    n = np.stack([-gx, gy, np.ones_like(gx)], axis=-1)     # image y down -> v up
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    g = np.clip(gloss + gloss_var * (lum - lum.mean()) / (lum.std() + 1e-6) * 0.5, 0, 1)
    return np.stack([(n[..., 0] * 0.5 + 0.5) * 255, (n[..., 1] * 0.5 + 0.5) * 255,
                     np.full_like(g, 10.0), g * 255], axis=-1).clip(0, 255).astype(np.uint8)


def place(pages, page, region, img, strength, gloss):
    """Put a natural-orientation (across x along) image into its region."""
    alb = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    nml = normal_map(img, strength, gloss)
    if region.rotated:
        alb = rot_cw(alb)
        nml = rot_cw(nml).copy()
        nx = nml[..., 0].astype(np.int16) - 128
        ny = nml[..., 1].astype(np.int16) - 128
        nml[..., 0] = np.clip(ny + 128, 0, 255)
        nml[..., 1] = np.clip(-nx + 128, 0, 255)
    # natural images are authored with row 0 = far end of "along"
    a_img, n_img = pages[page]
    a_img.paste(Image.fromarray(alb), (region.x0, region.y0))
    n_img.paste(Image.fromarray(nml, "RGBA"), (region.x0, region.y0))


def build(out):
    pages = {p: (Image.new("RGB", (L.PAGE_PX, L.PAGE_PX), (128, 128, 128)),
                 Image.new("RGBA", (L.PAGE_PX, L.PAGE_PX), (128, 128, 10, 60)))
             for p in ("tunnels", "solar")}
    soil = load("farm_soil.png")
    T, S = L.TUNNELS, L.SOLAR

    for name, src, seed in (("roof_a", "tunnel_film_a.png", 1), ("roof_b", "tunnel_film_b.png", 2)):
        w, h = T[name].size_px
        img = seamless_y(load(src), w, h,
                         prep=lambda t, s=seed: soil_feet(t, soil, L.FOOT_SOIL, 0.1, s))
        place(pages, "tunnels", T[name], img, 2.0, 0.6)

    doors = {}
    for gname, (src, flip) in zip(L.TUNNEL_GABLES, (("tunnel_end_a.png", False), ("tunnel_end_b.png", False),
                                                     ("tunnel_end_a.png", True), ("tunnel_end_b.png", True))):
        (x0, y0, x1, y1), (a0, a1, b0, b1) = ENDS[src]
        img = load(src)[y0:y1, x0:x1]
        if flip:
            img = img[:, ::-1]
            a0, a1 = 1 - a1, 1 - a0
        w, h = T[gname].size_px
        place(pages, "tunnels", T[gname], resize(img, w, h), 2.5, 0.5)
        doors[gname] = [a0, a1, b0, b1]

    w, h = S["film"].size_px
    # the film starts on top of the masonry kerb: only a little mud splash
    img = seamless_y(load("solar_film.png"), w, h,
                     prep=lambda t: soil_feet(t, soil, 0.0, 0.05, 3, both=False))
    a = (np.arange(w) + 0.5) / w
    img = img * (1 - 0.35 * smooth(a, 0.84, 0.92))[None, :, None]   # quilt shade at the ridge
    place(pages, "solar", S["film"], img, 2.0, 0.55)

    # quilts: each roll of the source becomes a lengthwise section
    q = load("insulation_quilts.png")
    bands = 6
    bh = q.shape[0] // bands
    rng = np.random.default_rng(7)
    w, h = S["quilt"].size_px
    parts, filled = [], 0
    while filled < h:
        k = int(rng.integers(bands))
        roll = rot_cw(q[k * bh:(k + 1) * bh])            # roll axis now vertical
        seg = int(rng.integers(h // 8, h // 4))
        y0 = int(rng.integers(0, roll.shape[0] - roll.shape[0] // 3))
        piece = resize(roll[y0:y0 + roll.shape[0] // 3], w, seg)
        parts.append(piece)
        filled += seg
    img = np.concatenate(parts, axis=0)[:h]
    place(pages, "solar", S["quilt"], img, 3.0, 0.1)

    w, h = S["back"].size_px
    b = rot_cw(load("straw_back_roof.png"))
    b = b[:, : int(b.shape[1] * 0.45)]
    place(pages, "solar", S["back"], seamless_y(b, w, h, copies=4), 3.0, 0.1)

    # built back walls and gables: brick (variant a), cement render (b);
    # the gable images have their door at the north (right) end
    for v, (wall_name, gable_name) in L.SOLAR_FINISH.items():
        finish = "brick" if v == "a" else "render"
        w, h = S[wall_name].size_px
        place(pages, "solar", S[wall_name],
              seamless_y(rot_cw(load(f"wall_{finish}.png")), w, h), 1.5, 0.12)
        w, h = S[gable_name].size_px
        place(pages, "solar", S[gable_name], resize(load(f"gable_{finish}.png"), w, h), 3.0, 0.12)

    br = load("brick_anteroom.png")
    w, h = S["ante_wall"].size_px
    per = 2 * 3.5 + 4.0
    south = resize(br, round(w * 3.5 / per), h)
    brick = br[:, int(br.shape[1] * 0.40): int(br.shape[1] * 0.62)]
    rest = w - south.shape[1]
    tiles = mirror_tile_x(brick, rest, h, 6)
    place(pages, "solar", S["ante_wall"], np.concatenate([south, tiles], axis=1), 3.0, 0.1)

    w, h = S["ante_roof"].size_px
    place(pages, "solar", S["ante_roof"], resize(load("concrete_flat_roof.png"), w, h), 2.5, 0.15)

    os.makedirs(out, exist_ok=True)
    for p, (a_img, n_img) in pages.items():
        a_img.save(os.path.join(out, f"{p}.png"))
        n_img.save(os.path.join(out, f"{p}_nml.png"))
    with open(os.path.join(out, "doors.json"), "w", encoding="utf-8") as fh:
        json.dump(doors, fh, indent=1)
    print(f"AI pages -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    build(os.path.abspath(ap.parse_args().out))


if __name__ == "__main__":
    main()
