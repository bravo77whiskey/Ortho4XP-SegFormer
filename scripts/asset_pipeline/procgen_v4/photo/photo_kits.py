"""Which CC0 photo texture covers which v4 material, on every page.

A photo material keeps its procedural spec (cls, roughness, drip...) and
gains a ``photo`` block that bake_materials reads instead of the palette:

    {"diff", "disp", "ao": paths, "size": [m, m] real-world size,
     "gain": linear RGB, "ao_k": AO exponent, "bump": bump distance (m),
     "rot90": bool}

``gain`` pulls the photo's mean colour toward the scheme's palette by
``tint`` (0 = the photo as shot, 1 = the palette's mean), so the regional
colourways survive while every pixel's detail stays photographic.
"""

from __future__ import annotations

import copy
import json
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
DEFAULT_TEX_DIR = os.path.join(REPO, "test_output", "procgen_v4_photo",
                               "textures_cc0")

# Poly Haven asset id -> resolution to download.  2k covers >= 680 px/m on
# the small repeats (bake strips are 80 px/m), 4k for the 4-8 m roofs.
ASSETS = {
    "white_stucco": "2k",
    "beige_wall_001": "2k",
    "red_brick": "2k",
    "plaster_grey_04": "2k",
    "clay_roof_tiles_03": "2k",
    "grey_roof_01": "4k",
    "roof_09": "4k",
    "gravel_stones": "2k",
    # asia
    "exterior_wall_cladding_03": "2k",
    "rectangular_facade_tiles": "2k",
    "grey_roof_tiles": "2k",
    "corrugated_iron_03": "2k",
    "box_profile_metal_sheet": "2k",
    "concrete_floor_worn_001": "2k",
    # all regions
    "yellow_brick": "2k",
    "white_planks_clean": "2k",
    "concrete_slab_wall_02": "2k",
    "plastered_wall_04": "2k",
    "roof_slates_03": "2k",
    "ceramic_roof_01": "4k",
    "asbestos_sheet": "2k",
    "bitumen": "2k",
}

# material slot -> (asset, palette target or None, tint, ao_k, bump, extra)
EUROPE_RES_A = {
    ("s1", "wall"): ("white_stucco", "#ecebe4", 0.8, 0.5, 0.004,
                     {"flatten": 0.25}),
    ("s1", "plinth"): ("plaster_grey_04", "#66635f", 0.8, 0.5, 0.004,
                       {"flatten": 0.25}),
    ("s2", "wall"): ("beige_wall_001", "#ecdcaa", 0.8, 0.5, 0.004,
                     {"flatten": 0.25}),
    ("s3", "wall"): ("red_brick", "#8c3b2a", 0.4, 2.0, 0.012, {}),
    ("s3", "lintel"): ("red_brick", "#6d2d22", 0.6, 2.0, 0.012,
                       {"rot90": True}),
    ("porch", "wall"): ("white_stucco", "#e8e4da", 0.8, 0.5, 0.004,
                        {"flatten": 0.25}),
    ("roof_a", None): ("clay_roof_tiles_03", "#a8472c", 0.4, 1.0, 0.03, {}),
    ("roof_b", None): ("grey_roof_01", "#3a3c3f", 0.4, 1.0, 0.03, {}),
    ("roof_c", None): ("roof_09", "#5e3a2a", 0.4, 1.0, 0.03, {}),
    ("roof_flat", None): ("gravel_stones", "#6e6d69", 0.5, 0.8, 0.015, {}),
}


# asia res A: siding / small mosaic tiles / grey render, kawara + painted
# metal roofs, concrete slab.  ``desat`` recolours a painted surface (a red
# box-profile sheet becomes the scheme's green) keeping the photo's detail.
ASIA_RES_A = {
    ("s1", "wall"): ("exterior_wall_cladding_03", "#d8d0c0", 0.8, 0.6,
                     0.004, {"flatten": 0.25}),
    ("s2", "wall"): ("rectangular_facade_tiles", "#e8e6e0", 1.0, 0.8,
                     0.004, {"desat": 0.7}),
    ("s3", "wall"): ("white_stucco", "#b8b6ae", 1.0, 0.5, 0.004,
                     {"flatten": 0.25, "desat": 0.5}),
    ("porch", "wall"): ("white_stucco", "#dedad0", 0.8, 0.5, 0.004,
                        {"flatten": 0.25}),
    ("roof_a", None): ("grey_roof_tiles", "#6b7075", 0.6, 1.0, 0.03,
                       {"desat": 0.5}),
    ("roof_b", None): ("corrugated_iron_03", "#3d5f8a", 0.9, 0.8, 0.02,
                       {"desat": 0.6}),
    ("roof_c", None): ("box_profile_metal_sheet", "#4f7a55", 1.0, 0.8, 0.02,
                       {"desat": 1.0}),
    ("roof_flat", None): ("concrete_floor_worn_001", "#9c9a94", 0.5, 0.6,
                          0.008, {}),
}

def _srgb_to_lin(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _hex_lin(h):
    h = h.lstrip("#")
    return _srgb_to_lin([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def _licenses(tex_dir):
    with open(os.path.join(tex_dir, "LICENSES.json"), encoding="utf-8") as fh:
        return {r["id"]: r for r in json.load(fh)}


_MEANS = {}


def photo_mean(path):
    """Mean linear colour of a colour map (downsampled)."""
    if path not in _MEANS:
        im = Image.open(path).convert("RGB")
        im = im.resize((256, 256), Image.BOX)
        _MEANS[path] = _srgb_to_lin(np.asarray(im, np.float64) / 255.0) \
            .reshape(-1, 3).mean(0)
    return _MEANS[path]


def flattened(path, size_m, sigma_m):
    """The colour map with its tone below ``sigma_m`` metres removed.

    A render photo's large-scale shading (light falloff across the shot,
    patchy paint) repeats once per photo; on a facade that is once per
    bay and reads as banding.  Dividing by a wrap-around blur keeps the
    grain, sand and hairline cracks and drops the repeat.  Written next to
    the source as ``*_flat<sigma>.png`` (a derivative of the CC0 file)."""
    from scipy import ndimage
    out = path.replace("_diff_", f"_flat{int(sigma_m * 100)}_")
    out = os.path.splitext(out)[0] + ".png"
    if os.path.isfile(out):
        return out
    im = np.asarray(Image.open(path).convert("RGB"), np.float64) / 255.0
    lin = _srgb_to_lin(im)
    sig = sigma_m / size_m * im.shape[1]
    low = np.stack([ndimage.gaussian_filter(lin[..., c], sig, mode="wrap")
                    for c in range(3)], -1)
    flat = lin / np.maximum(low, 1e-4) * lin.reshape(-1, 3).mean(0)
    srgb = np.where(flat <= 0.0031308, flat * 12.92,
                    1.055 * np.clip(flat, 0, None) ** (1 / 2.4) - 0.055)
    Image.fromarray((np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8)
                    ).save(out)
    return out


def _size_m(rec, path):
    """Real-world [w, h] in metres.  The API lists some non-square maps
    (beige_wall_002 is 2048x512) as square, so height follows the image's
    own aspect from the listed width."""
    w, h = Image.open(path).size
    sx = rec["size_mm"][0] / 1000.0
    return [sx, sx * h / w]


def photo_block(asset, target, tint, ao_k, bump, extra, tex_dir):
    rec = _licenses(tex_dir)[asset]
    files = {k: os.path.join(tex_dir, v["file"])
             for k, v in rec["files"].items()}
    extra = dict(extra)
    if extra.get("flatten"):
        files["diff"] = flattened(files["diff"], rec["size_mm"][0] / 1000.0,
                                  extra.pop("flatten"))
    gain = np.ones(3)
    if target and tint:
        mean = photo_mean(files["diff"])
        d = extra.get("desat", 0.0)
        lum = float(mean @ np.array([0.2126, 0.7152, 0.0722]))
        mean = mean * (1 - d) + lum * d
        ratio = _hex_lin(target) / np.maximum(mean, 1e-4)
        gain = 1.0 + tint * (ratio - 1.0)
    block = {"diff": files["diff"], "disp": files["disp"], "ao": files["ao"],
             "size": _size_m(rec, files["diff"]),
             "gain": [round(float(g), 4) for g in gain], "ao_k": ao_k,
             "bump": bump, "asset": asset}
    block.update(extra)
    return block


def apply_mapping(kit, mapping, tex_dir=DEFAULT_TEX_DIR):
    """A copy of a res kit with the mapping's photo materials."""
    kit = copy.deepcopy(kit)
    for (slot, mat_key), args in mapping.items():
        block = photo_block(*args, tex_dir)
        if mat_key is None:                     # a roof covering
            kit[slot]["mat"]["photo"] = block
        else:
            kit[slot]["mats"][mat_key]["photo"] = block
    return kit


# ------------------------------------------------- every region, by family
# Pages without a hand-written table get photos by material family: the
# wall pattern (brick / siding / tiles / panels / boards / render /
# concrete) and the roof kind pick the photo, and the scheme's own palette
# tints it, so every colourway keeps its colour and gets photographic
# detail.  Shingle roofs stay procedural: neither CC0 library has a
# photographed asphalt shingle.

RENDERS = ("white_stucco", "beige_wall_001", "plaster_grey_04")
ROOF_PHOTOS = {
    # kind: (asset, tint, ao_k, bump, extra)
    "clay_interlock": ("clay_roof_tiles_03", 0.5, 1.0, 0.03, {}),
    "concrete_tile": ("grey_roof_01", 0.5, 1.0, 0.03, {}),
    "slate": ("roof_slates_03", 0.6, 1.0, 0.02, {"desat": 0.5}),
    "barrel": ("ceramic_roof_01", 0.6, 1.0, 0.03, {"desat": 0.4}),
    "kawara": ("grey_roof_tiles", 0.6, 1.0, 0.03, {"desat": 0.5}),
    "trapezoid": ("box_profile_metal_sheet", 1.0, 0.8, 0.02,
                  {"desat": 1.0}),
    "standing_seam": ("box_profile_metal_sheet", 1.0, 0.8, 0.02,
                      {"desat": 1.0}),
    "corrugated": ("corrugated_iron_03", 0.9, 0.8, 0.02, {"desat": 0.6}),
    "fibre_cement": ("asbestos_sheet", 0.8, 0.8, 0.015, {"desat": 0.6}),
    "membrane": ("bitumen", 0.8, 0.6, 0.006, {"desat": 0.5}),
    "gravel": ("gravel_stones", 0.5, 0.8, 0.015, {}),
    "concrete_slab": ("concrete_floor_worn_001", 0.5, 0.6, 0.008, {}),
}


def _mean_hex(colours):
    lin = np.mean([_hex_lin(c) for c in colours], axis=0)
    srgb = np.where(lin <= 0.0031308, lin * 12.92,
                    1.055 * lin ** (1 / 2.4) - 0.055)
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in srgb)


def _stable(key):
    import hashlib
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)


def wall_photo(spec, key):
    """(asset, tint, ao_k, bump, extra) for a wall-like material spec."""
    pat = (spec.get("pattern") or {}).get("type")
    if pat == "brick":
        warm = _hex_lin(spec["base"][0])
        yellow = warm[1] > 0.55 * warm[0]
        return ("yellow_brick" if yellow else "red_brick", 0.45, 2.0, 0.012,
                {})
    if pat == "siding":
        return ("white_planks_clean", 1.0, 0.8, 0.004,
                {"rot90": True, "desat": 1.0})
    if pat == "tiles":
        return ("rectangular_facade_tiles", 1.0, 0.8, 0.004, {"desat": 0.7})
    if pat == "panels":
        return ("concrete_slab_wall_02", 0.9, 0.8, 0.006, {"desat": 0.7})
    if pat == "boards":
        return ("box_profile_metal_sheet", 1.0, 0.8, 0.012, {"desat": 1.0})
    if pat is None and "bump_scale" not in spec:
        # concrete(): fair-faced concrete, not render
        return ("plastered_wall_04", 0.9, 0.6, 0.004,
                {"flatten": 0.25, "desat": 0.7})
    asset = RENDERS[_stable(key) % len(RENDERS)]
    return (asset, 1.0, 0.5, 0.004, {"flatten": 0.25, "desat": 0.6})


def auto_mapping(kit, page):
    """{(slot, mat key or None): photo args} for every scheme and roof of a
    kit dict (a res colourway, an apt kit or the industrial page)."""
    out = {}
    for slot, value in kit.items():
        if not isinstance(value, dict):
            continue
        if "mats" in value:
            for key in ("wall", "plinth", "lintel"):
                spec = value["mats"].get(key)
                if not spec or spec.get("cls", 1) not in (1, 2, 7):
                    continue
                if key != "wall" and (spec.get("pattern") or {}).get(
                        "type") not in (None, "brick"):
                    continue
                asset, tint, ao_k, bump, extra = wall_photo(
                    spec, f"{page}/{slot}/{key}")
                if key == "lintel" and asset.endswith("brick"):
                    extra = dict(extra, rot90=True)
                out[(slot, key)] = (asset, _mean_hex(spec["base"]), tint,
                                    ao_k, bump, extra)
        elif "kind" in value and "mat" in value:
            args = ROOF_PHOTOS.get(value["kind"])
            if args:
                asset, tint, ao_k, bump, extra = args
                out[(slot, None)] = (asset, _mean_hex(value["mat"]["base"]),
                                     tint, ao_k, bump, extra)
    return out


EXPLICIT = {"europe_res_a": EUROPE_RES_A, "asia_res_a": ASIA_RES_A}


def mapping_for(page, kit):
    return EXPLICIT.get(page) or auto_mapping(kit, page)

