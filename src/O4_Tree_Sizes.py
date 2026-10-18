"""Single trees sized to the canopy height measured on the ortho.

X-Plane 12 ships its tree species as single-species forests in up to four
sizes (``lib/vegetation/trees/<group>/<species>_<size>.for``), each with a
fixed height range. A point-mode tree (a small canopy patch planted as single
trees) on a canopy-height texture gets the size of a species whose range
covers the height measured at its crown.

The species come from the canopy layer of X-Plane's own forest for the
tree family (broad / coni / mixed) and climate band at the tree, so every
climate gets trees X-Plane itself plants there. Below the smallest size of
those species a tree becomes a shrub: X-Plane's deciduous or pine shrubs, or
in hot climates the band's own shrub forest (``lib/g8/shrb_*``).
"""

from __future__ import annotations

import re

import numpy as np

# (virtual path, min height m, max height m), smallest first; the heights are
# the TREE min/max of the .for files in Resources/default scenery/1200 forests.
_T = "lib/vegetation/trees/"
SIZED_SPECIES = {
    "oak": ((_T + "deciduous/oak_medium.for", 8.0, 12.0),
            (_T + "deciduous/oak_big.for", 11.7, 18.0),
            (_T + "deciduous/oak_tall.for", 17.0, 26.0)),
    "red_oak": ((_T + "deciduous/red_oak_medium.for", 7.0, 12.0),
                (_T + "deciduous/red_oak_big.for", 11.0, 19.0),
                (_T + "deciduous/red_oak_tall.for", 17.0, 24.0)),
    "elm": ((_T + "deciduous/elm_medium.for", 9.0, 12.0),
            (_T + "deciduous/elm_big.for", 12.0, 18.0),
            (_T + "deciduous/elm_tall.for", 17.6, 25.0)),
    "maple": ((_T + "deciduous/maple_small.for", 5.0, 8.0),
              (_T + "deciduous/maple_medium.for", 8.0, 12.0),
              (_T + "deciduous/maple_big.for", 11.0, 18.0)),
    "birch": ((_T + "deciduous/birch_medium.for", 7.2, 11.0),
              (_T + "deciduous/birch_big.for", 9.4, 15.0),
              (_T + "deciduous/birch_tall.for", 15.0, 22.0)),
    "aspen": ((_T + "deciduous/aspen_medium.for", 8.0, 11.0),
              (_T + "deciduous/aspen_big.for", 14.7, 21.0),
              (_T + "deciduous/aspen_tall.for", 19.0, 32.0)),
    "black_locust": ((_T + "deciduous/black_locust_small.for", 5.0, 7.0),
                     (_T + "deciduous/black_locust_medium.for", 8.8, 15.0),
                     (_T + "deciduous/black_locust_big.for", 15.0, 21.0)),
    "pine": ((_T + "coniferous/pine_medium.for", 9.0, 12.0),
             (_T + "coniferous/pine_big.for", 12.0, 18.0),
             (_T + "coniferous/pine_tall.for", 20.0, 28.0)),
    "spruce": ((_T + "coniferous/spruce_small.for", 5.1, 8.0),
               (_T + "coniferous/spruce_medium.for", 10.0, 15.0),
               (_T + "coniferous/spruce_big.for", 18.0, 24.0),
               (_T + "coniferous/spruce_tall.for", 21.6, 32.0)),
    "fir": ((_T + "coniferous/fir_small.for", 5.1, 8.0),
            (_T + "coniferous/fir_medium.for", 10.2, 16.0),
            (_T + "coniferous/fir_big.for", 18.0, 24.0),
            (_T + "coniferous/fir_tall.for", 22.5, 32.0)),
    "kapok": ((_T + "evergreen/kapok_tree_medium.for", 8.0, 12.0),
              (_T + "evergreen/kapok_tree_thin.for", 14.4, 24.0),
              (_T + "evergreen/kapok_tree_big.for", 25.0, 36.0)),
    "brazil_nut": ((_T + "evergreen/brazil_nut_medium.for", 8.1, 12.0),
                   (_T + "evergreen/brazil_nut_big.for", 12.6, 20.0),
                   (_T + "evergreen/brazil_nut_thin.for", 15.3, 24.0),
                   (_T + "evergreen/brazil_nut_tall.for", 19.0, 32.0)),
    "shrub": ((_T + "deciduous/shrubs_small.for", 1.0, 3.0),
              (_T + "deciduous/shrubs_medium.for", 3.4, 6.0),
              (_T + "deciduous/shrubs_big.for", 6.5, 11.0)),
    "pine_shrub": ((_T + "coniferous/shrubs_pines.for", 1.8, 4.4),),
}

SIZED_TREE_PATHS = frozenset(p for sizes in SIZED_SPECIES.values() for p, _lo, _hi in sizes)

# Canopy-layer species of X-Plane's lib/g8 forests per climate temperature,
# (wet, dry) where the band's semi-dry/dry forest differs. X-Plane's palms
# and dead trees have no sized single-tree set and are left out.
_BROAD = {
    "vcld": ("birch", "aspen", "elm"),
    "cld": ("birch", "aspen", "elm", "oak", "black_locust"),
    "tmp": ("oak", "red_oak", "maple", "elm", "birch", "black_locust", "aspen"),
    "wrm": ("oak", "red_oak", "maple", "elm", "black_locust"),
    "hot": ("kapok", "brazil_nut"),
    "vhot": ("kapok", "brazil_nut"),
}
_BROAD_DRY = {"wrm": ("oak", "maple", "elm", "black_locust")}
_CONI = {t: ("pine", "spruce", "fir") for t in ("vcld", "cld", "tmp", "wrm")}
_CONI.update({"hot": ("pine",), "vhot": ("pine",)})
_CONI_DRY = {"wrm": ("pine",)}
_SHRUB_BY_FAMILY = {"broad": ("shrub",), "coni": ("pine_shrub",), "mixed": ("shrub", "pine_shrub")}
_TROPICAL = ("hot", "vhot")

# Tallest tree of X-Plane's hot-band shrub forests (lib/g8/shrb_{hot,vhot}_*).
TROPICAL_SHRUB_MAX_M = 6.0

# Under runway approaches a height limit below SHRUB_CAP_M plants shrubs
# instead of cut-down trees (shrub_path), and where vegetation is found no
# limit plants less than SHRUB_FLOOR_M of shrub, the inner strip before the
# threshold included.
SHRUB_CAP_M = 9.0
SHRUB_FLOOR_M = 1.0
# Smallest single-species tree X-Plane has (maple / black locust small, 5 m);
# forests whose canopy is measured lower are planted as shrubs (shrub_for_height).
MIN_TREE_ASSET_M = 5.0

# Bump when the tables or the rule change: part of the polygon cache key.
SIGNATURE = "tree-sizes-v3"

# Below this the canopy network did not see a tree at the crown (SegFormer
# rows, height-gated SegFormer trees): the tree keeps the forest it had.
MIN_MEASURED_H_M = 1.5
# A species qualifies when its best size is within this of the best species.
SPECIES_TOLERANCE_M = 1.0


_G8_RE = re.compile(r"^lib/g8/(broad|coni|mixed)_(vcld|cld|tmp|wrm|hot|vhot)_(dry|sdry|wet|rain)\.for$")
_G8_ANY_RE = re.compile(r"^lib/g8/([a-z_]+?)_(vcld|cld|tmp|wrm|hot|vhot)_(dry|sdry|wet|rain)\.for$")


def band_of_path(path):
    """(family, temp, moist) of an X-Plane climate forest path, else None."""
    match = _G8_RE.match(str(path or "").replace("\\", "/"))
    return match.groups() if match else None


def _shrub_sizes(path):
    """(tropical shrub forest or None, sized shrubs) for a tree polygon of forest ``path``.

    Hot bands take their own shrub forest (lib/g8/shrb_*); elsewhere conifer
    forests take X-Plane's pine shrubs and the rest its deciduous shrubs.
    """
    m = _G8_ANY_RE.match(str(path or "").replace("\\", "/"))
    family, temp, moist = m.groups() if m else ("broad", None, None)
    if temp in _TROPICAL:
        return f"lib/g8/shrb_{temp}_{moist}.for", ()
    return None, SIZED_SPECIES["pine_shrub" if family == "coni" else "shrub"]


def shrub_path(path, cap_m):
    """The shrubs that replace a tree polygon of ``path`` where nothing may be taller than ``cap_m``:
    the largest size that can still grow under the cap (the caller cuts it down
    to the cap where its tallest shrubs overtop it)."""
    tropical, sizes = _shrub_sizes(path)
    if tropical:
        return tropical
    fits = [size for size in sizes if size[1] < cap_m]
    return (fits[-1] if fits else sizes[0])[0]


def shrub_for_height(path, height_m):
    """The shrubs that replace a tree polygon of ``path`` whose canopy is ``height_m`` tall:
    the size whose height range is nearest."""
    tropical, sizes = _shrub_sizes(path)
    if tropical:
        return tropical
    return min(sizes, key=lambda size: (_size_miss(height_m, size[1], size[2]), size[1]))[0]


def fits_under(path, cap_m):
    """True for a sized tree or shrub whose tallest tree is at most ``cap_m``."""
    return any(p == path and hi <= cap_m + 1e-6
               for sizes in SIZED_SPECIES.values() for p, _lo, hi in sizes)


def species_pool(family, temp, moist):
    """Species names X-Plane plants for ``family`` in band ``temp``/``moist``."""
    dry = moist in ("dry", "sdry")
    if family == "mixed":
        return species_pool("broad", temp, moist) + species_pool("coni", temp, moist)
    table, dry_table = (_BROAD, _BROAD_DRY) if family == "broad" else (_CONI, _CONI_DRY)
    if dry and temp in dry_table:
        return dry_table[temp]
    return table.get(temp, ())


def _size_miss(height, lo, hi):
    return 0.0 if lo <= height <= hi else min(abs(height - lo), abs(height - hi))


def sized_tree_path(family, temp, moist, height, rng=None, max_h=None):
    """Sized single-tree path for a crown ``height`` m tall, or None.

    None when the family has no sized species in that band; the caller keeps
    its forest path. In hot bands, trees shorter than the smallest sized
    species take the band's own shrub forest. With ``max_h`` only sizes
    whose tallest tree is at most that tall qualify (None when none does).
    """
    if family not in ("broad", "coni", "mixed") or height < MIN_MEASURED_H_M:
        return None
    pool = species_pool(family, temp, moist)
    if not pool:
        return None
    if max_h is not None:
        height = min(height, max_h)
    if temp in _TROPICAL:
        smallest = min(SIZED_SPECIES[s][0][1] for s in pool)
        if height < smallest - SPECIES_TOLERANCE_M:
            if max_h is not None and max_h < TROPICAL_SHRUB_MAX_M:
                return None
            return f"lib/g8/shrb_{temp}_{moist}.for"
    else:
        pool = pool + _SHRUB_BY_FAMILY[family]
    best = {}
    for name in pool:
        sizes = [(_size_miss(height, lo, hi), path) for path, lo, hi in SIZED_SPECIES[name]
                 if max_h is None or hi <= max_h + 1e-6]
        if sizes:
            best[name] = min(sizes)
    if not best:
        return None
    floor = min(miss for miss, _p in best.values())
    choices = sorted(path for miss, path in best.values() if miss <= floor + SPECIES_TOLERANCE_M)
    if rng is None or len(choices) == 1:
        return choices[0]
    return choices[int(rng.integers(len(choices)))]


def crown_heights(height_map, points_px, radius_px):
    """Tallest canopy height within ``radius_px`` of each crown (x, y) pixel."""
    h, w = height_map.shape
    r = max(0, int(round(radius_px)))
    out = []
    for x, y in points_px:
        x0, x1 = max(0, int(x) - r), min(w, int(x) + r + 1)
        y0, y1 = max(0, int(y) - r), min(h, int(y) + r + 1)
        win = height_map[y0:y1, x0:x1]
        out.append(float(win.max()) if win.size else 0.0)
    return np.asarray(out, dtype=np.float32)
