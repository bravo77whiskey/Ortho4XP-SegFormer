"""Paste the real window / door cut-outs into a baked photo page.

The bake's openings come from the same ``strips.facade_strip`` call the
bake made (same scheme, level, W, H and seed), so every cut-out lands on
the exact rectangle the OBJ UVs show as a window or door. Only the opening
is replaced: the wall, sill, lintel and the bake's reveal shadow around it
stay. In the normal page the opening is flattened (the photo carries its
own relief), with gloss on the glass and matte frames.

Every scheme uses one source photo for its windows (a real facade repeats
one window design) and one for its door. The cut-outs of a source are
spread over the bays with a fixed per-strip shuffle.
"""

from __future__ import annotations

import json
import os
import random
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.dirname(HERE)
sys.path.insert(0, V4)

import strips  # noqa: E402

PASTE_KINDS = {"win": "win", "small": "win", "door": "door"}
WB_STRENGTH = 0.25         # gray-world white balance, fraction applied
EXPOSURE_P = 95            # this luminance percentile of a cut-out ...
EXPOSURE_TARGET = 0.80     # ... maps to this (sRGB), gain clamped below
GAIN_RANGE = (0.75, 1.7)


# kit window type -> library window types it may take; doors by region
WINDOW_TYPES = {"casement": ("casement",), "double_hung": ("double_hung",),
                "slider": ("slider",), "modern": ("slider",)}
DOOR_TYPES = {"north_america": ("door_na",),
              "australia_oceania": ("door_na", "door_eu"),
              "asia": ("door_asia",),
              "africa": ("door_eu", "door_na"),
              "south_america": ("door_eu", "door_na")}
DEFAULT_DOOR_TYPES = ("door_eu",)
OFFICE_W = 2.0
STRETCH = {"slider": 1.4}  # max horizontal stretch of a cut-out by type     # kit windows this wide are glazing bands, not windows
# the first two prototype pages keep their hand-picked sources
EXPLICIT = {
    "europe_res_a": {"s1": ("hockenheim", "titting"),
                     "s2": ("altlussheim", "beratzhausen"),
                     "s3": ("brick_square", "dresden_door")},
    "asia_res_a": {"s1": ("parand_up", "taipei"),
                   "s2": (None, None),
                   "s3": ("parand_lo", None)},
}


def load_library(lib_dir):
    """({source: [(name, image, size_m)]}, {source: type})."""
    with open(os.path.join(lib_dir, "SOURCES.json"), encoding="utf-8") as fh:
        report = json.load(fh)
    pools, types = {}, {}
    for name, rec in sorted(report.items()):
        im = np.asarray(Image.open(os.path.join(lib_dir, name))
                        .convert("RGB"), np.float32) / 255.0
        pools.setdefault(rec["source"], []).append(
            (name, _normalise(im), rec["size_m"]))
        types[rec["source"]] = rec["type"]
    return pools, types


def _normalise(im):
    """Even out a cut-out's colour cast and exposure between photos."""
    mean = im.reshape(-1, 3).mean(0)
    wb = mean.mean() / np.maximum(mean, 1e-3)
    im = im * (1.0 + WB_STRENGTH * (wb - 1.0))
    lum = im @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    gain = EXPOSURE_TARGET / max(np.percentile(lum, EXPOSURE_P), 1e-3)
    return np.clip(im * np.clip(gain, *GAIN_RANGE), 0.0, 1.0)


def _stable(key):
    import hashlib
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)


def _schemes(kit):
    return sorted(k for k, v in kit.items()
                  if isinstance(v, dict) and "window" in v and k != "porch")


def assign(page, kit, region, types):
    """{scheme: (window source or None, door source or None)}.

    One window source and one door source per scheme, so a facade repeats
    one design as a real one does; schemes of a page draw different
    sources where the pool allows.  Barred windows keep the bake (the bars
    are relief in front of the glass), and so do office glazing bands."""
    if page in EXPLICIT:
        return EXPLICIT[page]
    if page.startswith("ind"):
        # industrial: strip glazing and steel personnel doors stay baked
        return {}
    by_type = {}
    for sid, t in sorted(types.items()):
        by_type.setdefault(t, []).append(sid)
    out = {}
    used = set()
    for scheme in _schemes(kit):
        win = kit[scheme]["window"]
        wsrc = None
        if not win.get("bars") and win.get("w", 1.0) < OFFICE_W:
            pool = [sid for t in WINDOW_TYPES.get(win["type"], ())
                    for sid in by_type.get(t, ())]
            wsrc = _pick(pool, used, f"{page}/{scheme}/win")
        dpool = [sid for t in DOOR_TYPES.get(region, DEFAULT_DOOR_TYPES)
                 for sid in by_type.get(t, ())]
        dsrc = _pick(dpool, used, f"{page}/{scheme}/door")
        out[scheme] = (wsrc, dsrc)
    return out


def _pick(pool, used, key):
    if not pool:
        return None
    fresh = [p for p in pool if p not in used] or pool
    sid = fresh[_stable(key) % len(fresh)]
    used.add(sid)
    return sid


def apply_sizes(kit, assignment, pools, types):
    """Scheme windows and doors take the photos' proportions at the kit's
    own heights (sill and head heights stay).  Openings are texture
    content only, so OBJs and UVs do not change."""
    for scheme, (wsrc, dsrc) in assignment.items():
        sch = kit[scheme]
        bay = sch.get("bay", 3.0)
        if wsrc:
            aspect = np.mean([s[0] / s[1] for _n, _i, s in pools[wsrc]])
            h = sch["window"]["h"]
            # keep the kit's width where the photo can stretch that far
            # (a two-pane slider reads fine 40% wider, a casement 20%)
            stretch = STRETCH.get(types[wsrc], 1.2)
            w = float(np.clip(sch["window"]["w"], h * aspect,
                              h * aspect * stretch))
            sch["window"] = dict(sch["window"],
                                 w=round(min(w, 0.6 * bay), 3))
        if dsrc:
            aspect = np.mean([s[0] / s[1] for _n, _i, s in pools[dsrc]])
            h = sch.get("door_h", 2.25)
            sch["door_w"] = round(float(min(h * aspect, 0.55 * bay)), 3)
    return kit


def sets_for(assignment, pools):
    """{(scheme, "win"|"door"): pool} for compose_page."""
    sets = {}
    for scheme, (wsrc, dsrc) in assignment.items():
        if wsrc:
            sets[(scheme, "win")] = pools[wsrc]
        if dsrc:
            sets[(scheme, "door")] = pools[dsrc]
    return sets


def _paste(dst_a, dst_n, im, x0, x1, r0, r1):
    """Resize ``im`` onto the float pixel box, with a 1 px feather."""
    ix0, ix1 = int(np.floor(x0)), int(np.ceil(x1))
    iy0, iy1 = int(np.floor(r0)), int(np.ceil(r1))
    w, h = ix1 - ix0, iy1 - iy0
    if w < 2 or h < 2:
        return
    res = np.asarray(Image.fromarray((im * 255 + 0.5).astype(np.uint8))
                     .resize((w, h), Image.LANCZOS), np.float32) / 255.0
    # coverage of each pixel column/row by the true (fractional) box
    cx = np.clip(np.minimum(np.arange(ix0, ix1) + 1, x1) -
                 np.maximum(np.arange(ix0, ix1), x0), 0, 1)
    cy = np.clip(np.minimum(np.arange(iy0, iy1) + 1, r1) -
                 np.maximum(np.arange(iy0, iy1), r0), 0, 1)
    a = (cy[:, None] * cx[None, :])[..., None]
    region = dst_a[iy0:iy1, ix0:ix1].astype(np.float32) / 255.0
    dst_a[iy0:iy1, ix0:ix1] = np.clip(
        (region * (1 - a) + res * a) * 255 + 0.5, 0, 255).astype(np.uint8)
    lum = res @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    sat = res.max(-1) - res.min(-1)
    glass = ((lum < 0.35) & (sat < 0.18)).astype(np.float32)
    gloss = 0.2 + 0.45 * glass
    nrm = dst_n[iy0:iy1, ix0:ix1].astype(np.float32)
    flat = np.stack([np.full_like(lum, 128.0), np.full_like(lum, 128.0),
                     np.full_like(lum, 0.04 * 255), gloss * 255], -1)
    dst_n[iy0:iy1, ix0:ix1] = np.clip(nrm * (1 - a) + flat * a + 0.5, 0,
                                      255).astype(np.uint8)


def compose_page(page_png, nml_png, strip_rows, sets):
    """``strip_rows``: [(name, layout row, bake job)] of facade strips."""
    albedo = np.array(Image.open(page_png).convert("RGB"))
    normal = np.array(Image.open(nml_png).convert("RGBA"))
    placed = 0
    for name, row, job in strip_rows:
        scheme = name.split("_")[0]
        W, H = row["world_w"], row["world_h"]
        ppm = row["px"] / H
        _geo, info = strips.facade_strip(job["scheme"], job["level"], W, H,
                                         job["seed"])
        rng = random.Random(job["seed"])
        order = {}
        for kind, (x0, x1, z0, z1), k in info["openings"]:
            src = PASTE_KINDS.get(kind)
            pool = sets.get((scheme, src))
            if not pool:
                continue
            if src not in order:
                order[src] = rng.sample(range(len(pool)), len(pool))
            _n, im, _size = pool[order[src][k % len(pool)]]
            r0 = row["row0"] + (H - z1) * ppm
            r1 = row["row0"] + (H - z0) * ppm
            _paste(albedo, normal, im, x0 * ppm, x1 * ppm, r0, r1)
            placed += 1
    Image.fromarray(albedo, "RGB").save(page_png, compress_level=6)
    Image.fromarray(normal, "RGBA").save(nml_png, compress_level=6)
    return placed
