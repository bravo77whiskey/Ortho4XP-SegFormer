"""Atlas page layouts shared by the bake, the geometry builders and LODs.

A page is 4096 x 4096.  Every strip spans the full page width, so it wraps
in u (a wall of any length maps to a u-range that simply runs past 1.0),
and is bounded in v.  Strips are stacked tallest first so each one starts
on a multiple of its own (power-of-two) height: box-filtered mip levels
then never mix two strips until the strip itself has shrunk to one texel,
which is what produced the old library's concentric "rainbow roof" rings
(its 24 strips had arbitrary heights and bled into each other from mip 3).

``world_h`` is the strip's height in metres; its width in metres is
``PAGE_PX * world_h / px``, i.e. texels are square.
"""

from __future__ import annotations

PAGE_PX = 4096
TRIM_BANDS = ("fascia", "soffit", "ridge", "chimney", "coping", "dark",
              "light", "metal")
TRIM_BAND_PX = 32
TRIM_BAND_H = 0.4           # metres per trim band (80 px/m)

# (name, px, kind, world_h)
_RES = [
    ("roof_a", 512, "roof", 8.0),
    ("roof_b", 512, "roof", 8.0),
    ("roof_c", 512, "roof", 8.0),
    ("roof_flat", 256, "roof", 16.0),
    ("s1_upper", 256, "facade", 3.2),
    ("s1_ground", 256, "facade", 3.2),
    ("s2_upper", 256, "facade", 3.2),
    ("s2_ground", 256, "facade", 3.2),
    ("s3_upper", 256, "facade", 3.2),
    ("s3_ground", 256, "facade", 3.2),
    ("trim", 256, "trim", 3.2),
    ("s1_plain", 128, "plain", 4.0),
    ("s2_plain", 128, "plain", 4.0),
    ("s3_plain", 128, "plain", 4.0),
    ("porch", 128, "plain", 4.0),
]

_APT = [
    ("roof_a", 512, "roof", 8.0),
    ("roof_flat", 512, "roof", 32.0),
    ("roof_flat_b", 256, "roof", 16.0),
    ("a1_upper", 256, "facade", 3.2),
    ("a1_ground", 256, "facade", 3.2),
    ("a2_upper", 256, "facade", 3.2),
    ("a2_ground", 256, "facade", 3.2),
    ("c1_upper", 256, "facade", 3.2),
    ("c1_ground", 256, "facade", 3.2),
    ("c2_upper", 256, "facade", 3.2),
    ("c2_ground", 256, "facade", 3.2),
    ("trim", 256, "trim", 3.2),
    ("a1_plain", 128, "plain", 4.0),
    ("a2_plain", 128, "plain", 4.0),
    ("c1_plain", 128, "plain", 4.0),
    ("c2_plain", 128, "plain", 4.0),
]

_IND = [
    ("roof_a", 512, "roof", 16.0),
    ("roof_b", 512, "roof", 16.0),
    ("roof_flat", 512, "roof", 32.0),
    ("i1_upper", 256, "facade", 3.2),
    ("i1_ground", 256, "facade", 3.2),
    ("i2_upper", 256, "facade", 3.2),
    ("i2_ground", 256, "facade", 3.2),
    ("i3_upper", 256, "facade", 3.2),
    ("i3_ground", 256, "facade", 3.2),
    ("i4_upper", 256, "facade", 3.2),
    ("trim", 256, "trim", 3.2),
    ("i1_plain", 128, "plain", 4.0),
    ("i2_plain", 128, "plain", 4.0),
    ("i3_plain", 128, "plain", 4.0),
    ("i4_plain", 128, "plain", 4.0),
]

PAGE_KINDS = {"res": _RES, "apt": _APT, "ind": _IND}


def page_layout(kind):
    """{strip: {v0, v1, px, row0, world_h, world_w, kind}} (v up, 0..1)."""
    rows = PAGE_KINDS[kind]
    out = {}
    y = 0
    for name, px, skind, world_h in rows:
        if y % px:
            raise ValueError(f"{kind}/{name}: row {y} not aligned to {px}")
        world_w = PAGE_PX * world_h / px
        out[name] = {
            "row0": y, "px": px, "kind": skind, "world_h": world_h,
            "world_w": world_w,
            # texture t runs bottom-up: the strip's top row is its v1
            "v0": 1.0 - (y + px) / PAGE_PX,
            "v1": 1.0 - y / PAGE_PX,
        }
        y += px
    if y != PAGE_PX:
        raise ValueError(f"{kind}: rows sum to {y}, not {PAGE_PX}")
    if "trim" in out:
        t = out["trim"]
        bands = {}
        for i, band in enumerate(TRIM_BANDS):
            r0 = t["row0"] + i * TRIM_BAND_PX
            bands[band] = {
                "v0": 1.0 - (r0 + TRIM_BAND_PX) / PAGE_PX,
                "v1": 1.0 - r0 / PAGE_PX,
                "world_h": TRIM_BAND_H,
                "world_w": PAGE_PX * TRIM_BAND_H / TRIM_BAND_PX,
            }
        out["trim"]["bands"] = bands
    return out


def bay_period(world_w, bay_m, motif=1):
    """The exact bay period the bake used for a facade strip."""
    from strips import bay_count
    n = bay_count(world_w, bay_m, motif)
    return world_w / n, n
