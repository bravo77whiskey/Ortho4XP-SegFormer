"""Shorter copies of X-Plane 12's climate forests.

X-Plane's climate forests (``lib/g8/{broad,coni,mixed}_<temp>_<moist>.for``)
plant one canopy layer of fixed height, ~13-27 m in most bands. Two kinds of
copy of a ``lib/g8`` forest keep its species, understory, spacing, textures
and seasons and only make trees shorter:

  * height bands ``h<band>``: the canopy-layer trees scaled so their mean is
    one of BAND_HEIGHTS_M, and understory trees taller than that scaled down
    to it (X-Plane's understory reaches 13 m), for forests whose canopy
    height was measured on the ortho (hedges, young stands, gardens);
  * height caps ``c<level>``: every tree taller than one of CAP_LEVELS_M
    scaled down to it, for approach paths (O4_Approach_Heights);
  * shrubs ``s<level>`` (tiles built before shrubs followed the tree rules,
    2026-11-15): X-Plane's shrubs cut down to the cap level like a ``c``
    copy (``s0``: none), on their own grid. New builds plant uncut shrubs
    from X-Plane's own forest and cut ones from its ``c`` copy;
  * evergreen ``ev`` (and ``<kind><height>-ev`` for the copies above): the
    summer file alone, without its SEASONAL autumn texture, exported for
    every season. X-Plane runs its seasons by the calendar of the hemisphere,
    so on the equator its temperate-band forests (which it plants on tropical
    highlands such as Eldoret and Addis Ababa) turn brown in October; its own
    hot-band forests are already exported like this;
  * summer leaves ``su`` (and ``<kind><height>-su``): every season as
    X-Plane has it, but the summer file without its SEASONAL autumn
    texture. X-Plane's summer forests blend toward autumn as the season goes
    on (strongly autumnal by mid-July in the sim). These replaced the leaf
    colour copies ``t<k>`` (2026-11-09, retired 2026-11-16: imagery with off
    colours gave unnatural trees), whose paths no longer parse.

A tree's 3-d meshes are modelled at its minimum height (in all of X-Plane's
forests, mesh top / TREE min height is 1.00-1.02) and X-Plane scales them
with the tree, so the TREE heights and the vertices of the meshes it uses
are scaled together. Scaling only the TREE heights (forest-heights-v1)
shrank the billboards but left the 3-d trees full height: from a distance
the forest looked thin and short, then full-size trees popped in near the
camera. A mesh shared by trees scaled by different factors is duplicated.

Every mesh of a copy is renamed (``<name>_o4<kind><height>``): X-Plane's
season files share their meshes by name, and a copy's shorter mesh under the
stock name could be swapped with the full-size one of the stock forests in
the same scene (seen in the sim as dark shards from trunks). The planting
grid (SPACING / RANDOM) shrinks with the canopy trees, down to
MIN_SPACING_M, so a shorter forest closes up like the original instead of
standing as small trees far apart.

The copies are written on demand into the ``O4XP_Forest_Heights`` library
pack in Custom Scenery, made from the X-Plane install the overlay builds for:
``<season>_<kind><height>/<file>.for`` beside a ``textures`` link to
X-Plane's forest textures (their ``../textures/`` paths then resolve
unchanged), and ``library.txt`` exports them as
``o4xp/forests/<kind><height>/<g8 name>.for`` for every season the original
exports. X-Plane adds the new pack to scenery_packs.ini when it next starts.
Copies made by an earlier rule are remade in place, under the same paths.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from collections import namedtuple


PACK_NAME = "O4XP_Forest_Heights"
VIRTUAL_PREFIX = "o4xp/forests/"
# Canopy-layer mean heights of the band copies, metres.
BAND_HEIGHTS_M = (6.0, 9.0, 12.0, 15.0)
# A band copy is used only when it is at least this much shorter than the original.
MIN_GAIN_M = 3.0
# Tallest-tree heights of the cap copies, metres; under the lowest no tree
# fits. Finer at the bottom, where the shrubs (O4_Tree_Sizes.shrub_path) are.
CAP_LEVELS_M = (1.0, 2.0, 3.0, 4.5, 6.0, 7.5) + tuple(float(h) for h in range(9, 34, 3))
SIGNATURE = "forest-heights-v5"
# Rule revision of one kind of copy. A change remakes only that kind's copies
# in the pack (a SIGNATURE change remakes all of them, ~20 GB of copies on a
# well-used install). Kinds not listed are revision 1.
#   s 2 (2026-11-15): shrubs keep X-Plane's own grid, like the tree copies.
KIND_REVISIONS = {"s": 2}


def version_text():
    """What the pack's VERSION_FILE holds for copies made by the current rules."""
    return "\n".join([SIGNATURE] + [f"{k}={v}" for k, v in sorted(KIND_REVISIONS.items())])

# Densest planting grid of a copy, metres: X-Plane's own small shrubs.
MIN_SPACING_M = 4.0
# Edge thicket copies ``m<top>`` ("mantle", the shrub and young-tree belt a
# real forest edge has): planted only in a band along a forest's open edges
# (O4_Vegetation_Overlay), under the forest's own trees, so the bare trunks
# below X-Plane's canopy are hidden from the oblique views of flight. Each
# is the forest's own species on a MANTLE_SPACING_M grid: one layer of its
# canopy trees as young trees topping out at ``top`` m (MANTLE_YOUNG_SHARE)
# mixed with its shrubs, and a second layer of shrubs (its understory, the
# forest-floor clutter under MANTLE_MIN_SHRUB_M left out, cut to ``top``),
# visible to MANTLE_LOD_M instead of X-Plane's 2-5 km. ``top`` is
# MANTLE_SHARE of the forest's canopy height, one of MANTLE_LEVELS_M.
# Young trees alone read as thin poles (scaled-down tall trees keep their
# bare trunks); the bushy shrubs are what closes the edge.
MANTLE_LEVELS_M = (3.0, 4.5, 6.0, 7.5, 9.0)
MANTLE_SHARE = 0.45
MANTLE_YOUNG_SHARE = 0.4
MANTLE_SPACING_M = 4.0
MANTLE_RANDOM_M = 3.5
MANTLE_MIN_SHRUB_M = 2.0
# Understory that is not living cover (stumps, broken trunks, forest-floor
# stones, branches and ferns) stays out of the thicket.
MANTLE_SKIP_NOTES = ("ff_", "stump", "broken")
MANTLE_LOD_M = 6000.0
MANTLE_KIND = "m"
# Suffix of an evergreen copy's kind, and the kind of a plain evergreen copy.
EVERGREEN_SUFFIX = "-ev"
EVERGREEN_KIND = "ev"
SUMMER_KIND = "su"
_SEASONAL_RE = re.compile(r"^\s*SEASONAL\s")
# Forests a copy can be made of: X-Plane's climate forests and its single-species trees.
_SOURCES = ("lib/g8/", "lib/vegetation/trees/")
VERSION_FILE = "o4xp_version.txt"
_FAMILIES = ("broad_", "coni_", "mixed_")
_LOCK_STALE_S = 120.0
_VERTEX_RE = re.compile(r"^(\s*VERTEX\s+)(\S+)(\s+)(\S+)(\s+)(\S+)(.*)$", re.S)

# Of a forest's summer file: canopy-layer (1) mean height, tallest canopy
# tree, tallest other tree, tallest tree.
ForestStats = namedtuple("ForestStats", "canopy_mean canopy_max other_max max")


def _log_default(*args):
    print(*args, flush=True)


def xp_forests_dir(custom_scenery_dir):
    root = os.path.dirname(os.path.abspath(custom_scenery_dir))
    return os.path.join(root, "Resources", "default scenery", "1200 forests")


def read_exports(library_txt):
    """{virtual path: {season: relative .for path}} from EXPORT_SEASON and EXPORT lines.

    A plain EXPORT (X-Plane's evergreen hot-band forests) has season "".
    """
    out = {}
    with open(library_txt, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) == 4 and parts[0] == "EXPORT_SEASON":
                _kw, season, virtual, rel = parts
            elif len(parts) == 3 and parts[0] == "EXPORT":
                _kw, virtual, rel = parts
                season = ""
            else:
                continue
            out.setdefault(virtual, {})[season] = rel.replace("\\", "/")
    return out


def _tree_fields(parts):
    """(min h index, max h index, nominal h index or None, layer index) of a TREE/TREE2 line."""
    if parts[0] == "TREE" and len(parts) >= 11:
        return 7, 8, None, 10
    if parts[0] == "TREE2" and len(parts) >= 13:
        return 7, 8, 9, 12
    return None


def tree_rows(text):
    """(layer, frequency, min h, max h) of every TREE / TREE2 line."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        fields = parts and _tree_fields(parts)
        if not fields:
            continue
        lo_i, hi_i, _nom_i, lay_i = fields
        try:
            rows.append((int(float(parts[lay_i])), abs(float(parts[6])),
                         float(parts[lo_i]), float(parts[hi_i])))
        except ValueError:
            continue
    return rows


def forest_stats(text):
    """ForestStats of a .for, or None without trees."""
    rows = tree_rows(text)
    if not rows:
        return None
    canopy = [(f, lo, hi) for layer, f, lo, hi in rows if layer == 1]
    weight = sum(f for f, _lo, _hi in canopy)
    mean = (sum(f * 0.5 * (lo + hi) for f, lo, hi in canopy) / weight) if weight > 0 else None
    canopy_max = max((hi for _f, _lo, hi in canopy), default=0.0)
    other_max = max((hi for layer, _f, _lo, hi in rows if layer != 1), default=0.0)
    return ForestStats(mean, canopy_max, other_max, max(canopy_max, other_max))


def canopy_mean_height(text):
    """Frequency-weighted mean height of a .for's canopy (layer 1) trees, or None."""
    stats = forest_stats(text)
    return stats.canopy_mean if stats else None


def max_tree_height(text):
    """Tallest TREE / TREE2 max height in a .for, or None."""
    stats = forest_stats(text)
    return stats.max if stats else None


def _fmt(value):
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _fmt_fine(value):
    """Plain decimals, never exponent notation: X-Plane reads "-6.9e-09" as -6.9, and
    the near-zero vertices of trunks then shot out as long dark spikes."""
    text = f"{value:.7f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def _rebuild(line, parts):
    indent = line[:len(line) - len(line.lstrip())]
    ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
    return indent + "\t".join(parts) + ending


def _scaled_mesh(block, factor, name=None):
    """A MESH block (header, VERTEX and IDX lines) scaled by ``factor``, renamed to ``name``."""
    out = []
    for line in block:
        parts = line.split()
        if parts[0] == "MESH":
            if name:
                parts[1] = name
            if factor != 1.0 and len(parts) > 7:
                try:
                    parts[7] = _fmt_fine(float(parts[7]) * factor)   # max wind displacement, m
                except ValueError:
                    pass
            if name or factor != 1.0:
                line = _rebuild(line, parts)
        elif parts[0] == "VERTEX" and factor != 1.0:
            m = _VERTEX_RE.match(line)
            if m:
                try:
                    x, y, z = (float(m.group(i)) * factor for i in (2, 4, 6))
                    line = (m.group(1) + _fmt_fine(x) + m.group(3) + _fmt_fine(y) + m.group(5)
                            + _fmt_fine(z) + m.group(7))
                except ValueError:
                    pass
        out.append(line)
    return out


def scale_trees(text, factor_for, mesh_suffix="", spacing_m=None, spacing_min_m=None,
                line_factor=None):
    """``text`` with every tree scaled by ``factor_for(layer, min h, max h)``,
    or by ``line_factor(fields of the TREE / TREE2 line)`` when given.

    A tree's min/max (and TREE2 nominal) heights and the vertices of its 3-d
    meshes are scaled by the same factor. A mesh that trees with different
    factors share keeps one factor and gets a renamed copy per other one.
    Every mesh name gets ``mesh_suffix``. SPACING and RANDOM shrink by the
    canopy (layer 1) trees' frequency-weighted factor, to MIN_SPACING_M, or
    with ``spacing_m`` to that grid (never wider than the original); with
    ``spacing_min_m`` too, that grid shrinks by the same factor, to
    ``spacing_min_m``.
    """
    lines = text.splitlines(keepends=True)
    tree_factor = {}
    refs = []
    uses = {}
    current = None
    canopy = []
    spacing = None
    for i, line in enumerate(lines):
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "SPACING" and spacing is None and len(parts) > 1:
            try:
                spacing = float(parts[1])
            except ValueError:
                pass
        if parts[0] in ("TREE", "TREE2"):
            current = None
            fields = _tree_fields(parts)
            if fields:
                lo_i, hi_i, _nom_i, lay_i = fields
                try:
                    layer = int(float(parts[lay_i]))
                    current = float(line_factor(parts) if line_factor is not None
                                    else factor_for(layer, float(parts[lo_i]), float(parts[hi_i])))
                    if layer == 1:
                        canopy.append((abs(float(parts[6])), current))
                except ValueError:
                    current = 1.0
                tree_factor[i] = current
        elif parts[0] == "MESH_3D" and len(parts) >= 2:
            factor = 1.0 if current is None else current
            refs.append((i, parts[1], factor))
            uses.setdefault(parts[1], set()).add(factor)
    primary, variants = {}, {}
    for mesh, factors in uses.items():
        ordered = sorted(factors, key=lambda f: (f != 1.0, f))
        primary[mesh] = ordered[0]
        for j, factor in enumerate(ordered[1:], 1):
            variants[(mesh, factor)] = f"{mesh}_o4s{j}{mesh_suffix}"
    ref_name = {i: variants.get((mesh, factor), mesh + mesh_suffix) for i, mesh, factor in refs}
    weight = sum(f for f, _k in canopy)
    grid = (sum(f * k for f, k in canopy) / weight) if weight > 0 else 1.0
    floor = MIN_SPACING_M
    if spacing_m is not None:
        if spacing_min_m is not None:
            spacing_m = max(spacing_min_m, spacing_m * grid)
        grid = min(1.0, spacing_m / spacing) if spacing else 1.0
        floor = 0.0

    out = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        parts = line.split()
        if parts and parts[0] == "MESH" and len(parts) >= 2:
            j = i + 1
            while j < n and lines[j].split()[:1] in (["VERTEX"], ["IDX"]):
                j += 1
            block, mesh = lines[i:j], parts[1]
            out.extend(_scaled_mesh(block, primary.get(mesh, 1.0),
                                    (mesh + mesh_suffix) if mesh_suffix else None))
            for (name, factor), new_name in sorted(variants.items()):
                if name == mesh:
                    out.extend(_scaled_mesh(block, factor, new_name))
            i = j
            continue
        if parts and parts[0] in ("SPACING", "RANDOM") and grid < 1.0:
            values = []
            for v in parts[1:]:
                try:
                    v = float(v)
                    values.append(_fmt(max(min(v, floor), v * grid)
                                       if parts[0] == "SPACING" else v * grid))
                except ValueError:
                    values.append(v)
            line = _rebuild(line, [parts[0]] + values)
        elif i in tree_factor and tree_factor[i] != 1.0:
            lo_i, hi_i, nom_i, _lay_i = _tree_fields(parts)
            for k in (lo_i, hi_i) + ((nom_i,) if nom_i is not None else ()):
                parts[k] = _fmt(float(parts[k]) * tree_factor[i])
            line = _rebuild(line, parts)
        elif i in ref_name and ref_name[i] != parts[1]:
            parts[1] = ref_name[i]
            line = _rebuild(line, parts)
        out.append(line)
        i += 1
    return "".join(out)


def scale_canopy(text, factor, mesh_suffix=""):
    """``text`` with every canopy-layer tree (and its meshes) scaled by ``factor``."""
    return scale_trees(text, lambda layer, _lo, _hi: factor if layer == 1 else 1.0, mesh_suffix)


def band_trees(text, band_m, stock_mean, mesh_suffix=""):
    """Band copy: canopy trees scaled to a ``band_m`` mean, taller understory down to ``band_m``."""
    factor = band_m / stock_mean
    return scale_trees(text, lambda layer, _lo, hi: factor if layer == 1
                       else (min(1.0, band_m / hi) if hi > 0 else 1.0), mesh_suffix)


def cap_trees(text, cap_m, mesh_suffix=""):
    """``text`` with every tree taller than ``cap_m`` (and its meshes) scaled down to it."""
    return scale_trees(text, lambda _layer, _lo, hi: min(1.0, cap_m / hi) if hi > 0 else 1.0,
                       mesh_suffix)


def shrub_trees(text, cap_m=None, mesh_suffix=""):
    """Shrub copy (kind ``s``, kept for tiles that plant them): the tree copies' rule,
    so shrubs taller than ``cap_m`` are cut down to it and the grid closes up by
    their mean cut, never under MIN_SPACING_M; X-Plane's own grid otherwise.

    Until 2026-11-15 shrubs were planted on a 3 m grid where X-Plane spaces its
    shrubs 4-6 m (10 m in the hot bands): up to 11x as many plants.
    """
    if cap_m:
        return cap_trees(text, cap_m, mesh_suffix)
    return scale_trees(text, lambda _layer, _lo, _hi: 1.0, mesh_suffix)


def _mantle_layout(text):
    """``text`` arranged for an edge thicket copy (MANTLE_KIND).

    Understory trees under MANTLE_MIN_SHRUB_M or named in MANTLE_SKIP_NOTES are
    dropped; the other understory (the shrubs) is drawn out to MANTLE_LOD_M and
    also joins every canopy (layer 1) group, at frequencies that make the
    canopy trees MANTLE_YOUNG_SHARE of it. Groups and meshes left without a
    tree are dropped.
    """
    lines = text.splitlines(keepends=True)
    n = len(lines)
    blocks = {}   # first line -> (end, fields, layer, hi)
    i = 0
    while i < n:
        parts = lines[i].split()
        fields = parts and parts[0] in ("TREE", "TREE2") and _tree_fields(parts)
        if not fields:
            i += 1
            continue
        j = i + 1
        while j < n and lines[j].split()[:1] == ["MESH_3D"]:
            j += 1
        try:
            layer, hi = int(float(parts[fields[3]])), float(parts[fields[1]])
        except ValueError:
            layer, hi = None, 0.0
        blocks[i] = (j, fields, layer, hi)
        i = j

    def is_shrub(start):
        _end, _fields, layer, hi = blocks[start]
        notes = lines[start].split()[-1]
        return (layer is not None and layer != 1 and hi >= MANTLE_MIN_SHRUB_M
                and not notes.startswith(MANTLE_SKIP_NOTES))

    def shrub_block(start, layer=None, freq_scale=1.0):
        end, fields, _layer, _hi = blocks[start]
        parts = lines[start].split()
        if parts[0] == "TREE2":
            try:
                parts[10] = _fmt(max(float(parts[10]), MANTLE_LOD_M))
            except ValueError:
                pass
        if layer is not None:
            parts[fields[3]] = str(layer)
        if freq_scale != 1.0:
            parts[6] = _fmt(abs(float(parts[6])) * freq_scale)
        return [_rebuild(lines[start], parts)] + lines[start + 1:end]

    shrubs = [b for b in blocks if is_shrub(b)]
    shrub_freq = sum(abs(float(lines[b].split()[6])) for b in shrubs)

    out = []
    group_layer, young_freq = None, 0.0

    def close_group():
        if group_layer == 1 and young_freq > 0 and shrub_freq > 0:
            scale = (1.0 - MANTLE_YOUNG_SHARE) / MANTLE_YOUNG_SHARE * young_freq / shrub_freq
            for b in shrubs:
                out.extend(shrub_block(b, layer=1, freq_scale=scale))

    i = 0
    while i < n:
        parts = lines[i].split()
        head = parts[0] if parts else ""
        if i in blocks:
            end, _fields, layer, _hi = blocks[i]
            if layer == 1:
                out.extend(lines[i:end])
                young_freq += abs(float(parts[6]))
            elif is_shrub(i):
                out.extend(shrub_block(i))
            i = end
            continue
        if head and not head.startswith("#"):
            close_group()
            group_layer, young_freq = None, 0.0
            if head == "GROUP" and len(parts) > 1:
                group_layer = int(float(parts[1]))
        out.append(lines[i])
        i += 1
    close_group()

    def _head(ln):
        return (ln.split()[:1] or [""])[0]
    keep = [True] * len(out)
    for k, ln in enumerate(out):
        if _head(ln) == "GROUP":
            m = k + 1
            while m < len(out) and _head(out[m]) not in ("GROUP", "TREE", "TREE2"):
                m += 1
            keep[k] = m < len(out) and _head(out[m]) in ("TREE", "TREE2")
    used = {ln.split()[1] for ln in out if _head(ln) == "MESH_3D" and len(ln.split()) > 1}
    k = 0
    while k < len(out):
        parts = out[k].split()
        if parts[:1] == ["MESH"] and len(parts) > 1 and parts[1] not in used:
            m = k + 1
            while m < len(out) and _head(out[m]) in ("VERTEX", "IDX"):
                m += 1
            for q in range(k, m):
                keep[q] = False
            k = m
            continue
        k += 1
    return "".join(ln for ln, ok in zip(out, keep) if ok)


def mantle_trees(text, top_m, mesh_suffix=""):
    """Edge thicket copy (MANTLE_KIND): the forest's canopy trees as young trees topping out
    at ``top_m`` mixed with its shrubs, more shrubs as a second layer, everything cut to
    ``top_m``, on the MANTLE grid."""
    text = _mantle_layout(text)

    def line_factor(parts):
        _lo_i, hi_i, _nom_i, _lay_i = _tree_fields(parts)
        hi = float(parts[hi_i])
        return min(1.0, top_m / hi) if hi > 0 else 1.0

    text = scale_trees(text, None, mesh_suffix, line_factor=line_factor)
    out = []
    random_line = ["RANDOM", _fmt(MANTLE_RANDOM_M), _fmt(MANTLE_RANDOM_M)]
    has_random = any(ln.split()[:1] == ["RANDOM"] for ln in text.splitlines())
    for line in text.splitlines(keepends=True):
        parts = line.split()
        if parts[:1] == ["SPACING"]:
            line = _rebuild(line, ["SPACING", _fmt(MANTLE_SPACING_M), _fmt(MANTLE_SPACING_M)])
            if not has_random:
                out.append(line)
                line = _rebuild(line, random_line)
        elif parts[:1] == ["RANDOM"]:
            line = _rebuild(line, random_line)
        out.append(line)
    return "".join(out)


def mantle_top(canopy_h, cap_m=None):
    """Top of the edge thicket under a canopy ``canopy_h`` m tall (and ``cap_m``): the
    MANTLE_LEVELS_M nearest MANTLE_SHARE of it, none above the cap; 0.0 when none fits."""
    want = MANTLE_SHARE * canopy_h
    levels = [h for h in MANTLE_LEVELS_M if cap_m is None or h <= cap_m + 1e-6]
    if not levels:
        return 0.0
    return min(levels, key=lambda h: (abs(h - want), h))


def evergreen_text(text):
    """A summer .for without its SEASONAL (autumn blend) textures."""
    return "".join(ln for ln in text.splitlines(keepends=True) if not _SEASONAL_RE.match(ln))


def cap_level(cap_m):
    """The tallest of CAP_LEVELS_M at or under ``cap_m``; 0.0 under the lowest."""
    fit = [h for h in CAP_LEVELS_M if h <= cap_m + 1e-6]
    return fit[-1] if fit else 0.0


def band_virtual_path(g8_path, band, kind="h"):
    """``o4xp/forests/<kind><height>/<g8 name>``, or ``.../trees/<group>/<name>`` for a
    lib/vegetation/trees forest."""
    return f"{VIRTUAL_PREFIX}{kind}{_fmt(band)}/{_copy_name(g8_path)}"


def _copy_name(g8_path):
    return (g8_path[len("lib/vegetation/"):] if g8_path.startswith("lib/vegetation/trees/")
            else os.path.basename(g8_path))


def cap_virtual_path(g8_path, level):
    return band_virtual_path(g8_path, level, "c")


def _split_virtual(path):
    """(kind segment, rest) of a copy's path, else None."""
    path = str(path)
    if not path.startswith(VIRTUAL_PREFIX):
        return None
    parts = path[len(VIRTUAL_PREFIX):].split("/", 1)
    return parts if len(parts) == 2 else None


def _seg_parts(seg):
    """(base, evergreen, summer leaves) of a kind segment.

    ``<kind><height>`` (h/c/s/m copies, or nothing), then ``-ev`` for an
    evergreen copy or ``-su`` for a summer-leaves one; a copy with no height
    change is ``ev`` or ``su``.
    """
    tokens = seg.split("-")
    base, mods = tokens[0], tokens[1:]
    if base in (EVERGREEN_KIND, SUMMER_KIND):
        mods.insert(0, base)
        base = ""
    return base, EVERGREEN_KIND in mods, SUMMER_KIND in mods


def _seg_join(base, evergreen, summer):
    parts = [base] if base else []
    if evergreen:
        parts.append(EVERGREEN_KIND)   # the summer file all year: no autumn blend either
    elif summer:
        parts.append(SUMMER_KIND)
    return "-".join(parts)


def parse_virtual(path):
    """(source forest path, kind 'h', 'c', 's', 'm', 'e' or 'su', height) of a copy's path, else None.

    An evergreen or summer-leaves copy of an h/c/s/m copy parses as that copy
    (is_evergreen / is_summer tell them apart); a plain evergreen copy is kind
    'e', a plain summer-leaves one kind 'su', both height 0.
    """
    parts = _split_virtual(path)
    if not parts:
        return None
    seg, rest = parts
    source = ("lib/vegetation/" if rest.startswith("trees/") else "lib/g8/") + rest
    base, evergreen, summer = _seg_parts(seg)
    if not base:
        if evergreen:
            return source, "e", 0.0
        return (source, SUMMER_KIND, 0.0) if summer else None
    if len(base) < 2 or base[0] not in "hcs" + MANTLE_KIND:
        return None
    try:
        height = float(base[1:])
    except ValueError:
        return None
    return source, base[0], height


def is_evergreen(path):
    """True for an evergreen copy's path."""
    parts = _split_virtual(path)
    return bool(parts) and _seg_parts(parts[0])[1]


def is_summer(path):
    """True for a summer-leaves copy's path."""
    parts = _split_virtual(path)
    return bool(parts) and _seg_parts(parts[0])[2]


def is_capped(path):
    """True for copies that only exist to keep trees under a height cap
    (approach caps, cut shrubs): without the pack their trees are left out."""
    parsed = parse_virtual(path)
    return bool(parsed) and (parsed[1] == "c" or (parsed[1] == "s" and parsed[2] > 0))


def is_mantle(path):
    """True for an edge thicket copy's path (MANTLE_KIND)."""
    parsed = parse_virtual(path)
    return bool(parsed) and parsed[1] == MANTLE_KIND


def needs_pack(path):
    """True for copies whose trees are left out, not planted as their full-height original,
    when the pack cannot be written: capped copies and edge thickets (an original
    would plant a second full forest along the edge)."""
    return is_capped(path) or is_mantle(path)


def evergreen_virtual_path(path):
    """The evergreen copy of an X-Plane forest or of one of its copies (unchanged if already one)."""
    path = str(path)
    if is_evergreen(path):
        return path
    parts = _split_virtual(path)
    if parts:
        base = _seg_parts(parts[0])[0]
        return f"{VIRTUAL_PREFIX}{_seg_join(base, True, False)}/{parts[1]}"
    return f"{VIRTUAL_PREFIX}{EVERGREEN_KIND}/{_copy_name(path)}"


def summer_virtual_path(path):
    """The summer-leaves copy of an X-Plane forest or of one of its copies (an evergreen
    copy, which has no autumn blend either, unchanged)."""
    path = str(path)
    parts = _split_virtual(path)
    if parts:
        base, evergreen, _summer = _seg_parts(parts[0])
        return f"{VIRTUAL_PREFIX}{_seg_join(base, evergreen, True)}/{parts[1]}"
    return f"{VIRTUAL_PREFIX}{SUMMER_KIND}/{_copy_name(path)}"


# Rare copies (merge_rare_copies). X-Plane loads every forest file a scene
# references with its own copy of the tree meshes (~60k vertices for a
# climate forest), so each leaf colour x height x thicket level a tile plants
# costs a few MB of VRAM however few trees it holds (2026-11-15). An edge
# thicket level under MERGE_SHARE of its forest's thicket plantings moves to
# the nearest level the forest keeps at most one level lower (never higher:
# thickets under runway approaches stay under the cap).
MERGE_SHARE = 0.05
MERGE_MANTLE_STEP_M = 1.5


def _kept(weights, share):
    """Members of a group with at least ``share`` of its weight (else its heaviest)."""
    total = sum(weights.values())
    kept = {k for k, w in weights.items() if w >= share * total}
    return kept or {max(weights, key=lambda k: (weights[k], str(k)))}


def merge_rare_copies(weights, share=MERGE_SHARE, mantle_step_m=MERGE_MANTLE_STEP_M):
    """{path: replacement path} for the forest copies in ``weights`` ({path: how much
    it plants, e.g. area x density}) that are rare in their group.

    Thicket copies of one forest (evergreen and summer-leaves copies apart)
    form a group; a level under ``share`` of the group's weight moves to the
    nearest kept level at most ``mantle_step_m`` lower. Other paths, and
    levels with nowhere close to go, are left alone.
    """
    groups = {}
    for path, w in weights.items():
        parts = _split_virtual(path)
        if not parts:
            continue
        base, ev, summer = _seg_parts(parts[0])
        if len(base) < 2 or base[0] != MANTLE_KIND:
            continue
        try:
            level = float(base[1:])
        except ValueError:
            continue
        groups.setdefault((ev, summer, parts[1]), {}).setdefault(level, [0.0, path])[0] += w
    moves = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        kept = _kept({h: m[0] for h, m in members.items()}, share)
        for h, (_w, path) in members.items():
            lower = [k for k in kept if h - mantle_step_m - 1e-9 <= k < h]
            if h not in kept and lower:
                moves[path] = members[max(lower)][1]
    return moves


def source_path(path):
    """The X-Plane forest a copy was made from; other paths unchanged."""
    parsed = parse_virtual(path)
    return parsed[0] if parsed else str(path)


def _link_dir(link, target):
    """Directory link (junction on Windows), else a copy."""
    if os.path.isdir(link):
        return
    try:
        if sys.platform.startswith("win"):
            subprocess.run(["cmd", "/c", "mklink", "/J", link, target], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.symlink(target, link, target_is_directory=True)
    except (OSError, subprocess.CalledProcessError):
        shutil.copytree(target, link)


class _PackLock:
    def __init__(self, pack_dir):
        self.path = os.path.join(pack_dir, ".lock")

    def __enter__(self):
        deadline = time.time() + 2 * _LOCK_STALE_S
        while True:
            try:
                os.close(os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                return self
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(self.path) > _LOCK_STALE_S:
                        os.remove(self.path)
                        continue
                except OSError:
                    pass
                if time.time() > deadline:
                    raise TimeoutError(f"{self.path} held too long")
                time.sleep(0.2)

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except OSError:
            pass


class ForestHeights:
    """Picks and writes shorter copies of ``lib/g8`` forests."""

    def __init__(self, custom_scenery_dir, log=_log_default, xp_dir=None):
        self.log = log
        self.xp_dir = xp_dir or (xp_forests_dir(custom_scenery_dir) if custom_scenery_dir else "")
        # O4_FOREST_HEIGHTS_PACK puts the pack elsewhere (review runs keep it out of X-Plane).
        self.pack_dir = (os.environ.get("O4_FOREST_HEIGHTS_PACK")
                         or (os.path.join(custom_scenery_dir, PACK_NAME) if custom_scenery_dir else ""))
        self.exports = {}
        lib = os.path.join(self.xp_dir, "library.txt")
        if self.xp_dir and os.path.isfile(lib):
            try:
                self.exports = {v: s for v, s in read_exports(lib).items() if v.startswith(_SOURCES)}
            except OSError:
                self.exports = {}
        self._stats = {}
        self._spacing = {}

    @property
    def available(self):
        return bool(self.exports)

    def stats(self, g8_path):
        """ForestStats of the forest X-Plane exports at ``g8_path`` (summer), or None."""
        if g8_path not in self._stats:
            seasons = self.exports.get(g8_path) or {}
            rel = seasons.get("sum") or seasons.get("") or next(iter(seasons.values()), None)
            stats = None
            if rel:
                try:
                    with open(os.path.join(self.xp_dir, rel), "r", encoding="utf-8", errors="ignore") as f:
                        stats = forest_stats(f.read())
                except OSError:
                    stats = None
            self._stats[g8_path] = stats
        return self._stats[g8_path]

    def is_seasonal(self, path):
        """True when X-Plane exports the forest a path is (or was copied from) by season."""
        seasons = self.exports.get(source_path(path)) or {}
        return any(seasons)

    def evergreen_path(self, path):
        """The evergreen copy of ``path`` where X-Plane changes it with the seasons, else ``path``."""
        return evergreen_virtual_path(path) if self.is_seasonal(path) else path

    def summer_path(self, path):
        """The summer-leaves copy of ``path`` where X-Plane changes it with the seasons
        (one of its forests or a copy of one), else ``path``."""
        return summer_virtual_path(path) if self.is_seasonal(path) else path

    def spacing(self, path):
        """Tree spacing (SPACING x, metres) of X-Plane's forest at ``path`` or of a
        copy of one as it is written, or None when unknown."""
        parsed = parse_virtual(path)
        g8_path = source_path(path)
        key = (g8_path,) + (tuple(parsed[1:]) if parsed else ())
        if key not in self._spacing:
            result = None
            seasons = self.exports.get(g8_path) or {}
            rel = seasons.get("sum") or seasons.get("") or next(iter(seasons.values()), None)
            if rel:
                try:
                    with open(os.path.join(self.xp_dir, rel), "r", encoding="utf-8",
                              errors="ignore") as f:
                        text = f.read()
                except OSError:
                    text = None
                if text is not None and parsed:
                    kind, height = parsed[1], parsed[2]
                    stock = self.stock_height(g8_path) if kind == "h" else None
                    text = (None if kind == "h" and not stock
                            else self._copy_text(text, kind, height, stock, ""))
                if text is not None:
                    for line in text.splitlines():
                        parts = line.split()
                        if len(parts) >= 2 and parts[0] == "SPACING":
                            try:
                                result = float(parts[1])
                            except ValueError:
                                pass
                            break
            self._spacing[key] = result
        return self._spacing[key]

    @staticmethod
    def _copy_text(text, kind, height, stock, suffix):
        """The trees of a copy of kind ``kind`` made from the forest ``text``."""
        if kind == "h":
            return band_trees(text, height, stock, suffix)
        if kind == "s":
            return shrub_trees(text, height or None, suffix)
        if kind == "c":
            return cap_trees(text, height, suffix)
        if kind == MANTLE_KIND:
            return mantle_trees(text, height, suffix)
        return text

    def stock_height(self, g8_path):
        """Canopy mean height of the forest X-Plane exports at ``g8_path`` (summer), or None."""
        stats = self.stats(g8_path) if os.path.basename(g8_path).startswith(_FAMILIES) else None
        return stats.canopy_mean if stats else None

    def path_for(self, g8_path, measured_h, cap_m=None):
        """Forest path for a polygon whose canopy is ``measured_h`` m tall (None: not measured)
        where no tree may be taller than ``cap_m`` (None: no limit).

        The band copy nearest the measured height, else ``g8_path``; with a cap
        that path's tallest tree overtops, the cap copy at the cap's level.
        None when the cap leaves no tree: under the lowest level, or a forest
        whose heights are unknown (not one of X-Plane's own forests).
        """
        path = g8_path
        stats = self.stats(g8_path)
        top = stats.max if stats else None
        stock = self.stock_height(g8_path)
        if stock is not None and measured_h is not None:
            # The original competes as one more height; copies too close to it are not offered.
            choices = [b for b in BAND_HEIGHTS_M if b <= stock - MIN_GAIN_M] + [stock]
            best = min(choices, key=lambda b: abs(b - measured_h))
            if best != stock:
                path = band_virtual_path(g8_path, best)
                top = max(stats.canopy_max * best / stock, min(stats.other_max, best))
        if cap_m is None or (top is not None and top <= cap_m + 1e-6):
            return path
        level = cap_level(cap_m)
        if top is None or not level:
            return None
        return cap_virtual_path(g8_path, level)

    def shrub_path_for(self, shrub_path, cap_m=None):
        """The shrubs of ``shrub_path`` (an X-Plane shrub forest) under ``cap_m``, or None
        when X-Plane has no such forest or no shrub fits: the forest itself where its
        tallest shrub fits under the cap, else its cap copy, as for trees (path_for)."""
        stats = self.stats(shrub_path)
        if not stats:
            return None
        if cap_m is None or stats.max <= cap_m + 1e-6:
            return shrub_path
        level = cap_level(cap_m)
        return cap_virtual_path(shrub_path, level) if level else None

    def mantle_path_for(self, path, cap_m=None):
        """The edge thicket copy (MANTLE_KIND) for a forest polygon planted with ``path``
        (one of X-Plane's climate forests or a copy of one) where no tree may be taller
        than ``cap_m``; keeps its evergreen / leaf colour. None for other forests, shrubs,
        and where no thicket fits under the cap."""
        parsed = parse_virtual(path)
        source, kind, height = parsed if parsed else (str(path), "", 0.0)
        if kind in ("s", MANTLE_KIND):
            return None
        stock = self.stock_height(source)
        if stock is None:
            return None
        canopy = height if kind == "h" else stock
        if kind == "c":
            canopy = min(canopy, height)
            cap_m = height if cap_m is None else min(cap_m, height)
        top = mantle_top(canopy, cap_m)
        if not top:
            return None
        parts = _split_virtual(path)
        evergreen, summer = (_seg_parts(parts[0])[1:] if parts else (False, False))
        seg = _seg_join(f"{MANTLE_KIND}{_fmt(top)}", evergreen, summer)
        return f"{VIRTUAL_PREFIX}{seg}/{_copy_name(source)}"

    def ensure(self, virtual_paths):
        """Write every copy in ``virtual_paths`` that the pack lacks; False on failure."""
        wanted = sorted({p for p in virtual_paths if str(p).startswith(VIRTUAL_PREFIX)})
        if not wanted:
            return True
        if not self.available or not self.pack_dir:
            return False
        try:
            os.makedirs(self.pack_dir, exist_ok=True)
            with _PackLock(self.pack_dir):
                _link_dir(os.path.join(self.pack_dir, "textures"), os.path.join(self.xp_dir, "textures"))
                lines = self._read_pack_exports()
                remade = []
                stale = self._stale_lines(lines)
                if stale:
                    # Copies from an earlier rule: remade under the same paths,
                    # so tiles built with them pick up the new ones.
                    stale_paths = {ln.split()[-2] for ln in stale}
                    remade = sorted(stale_paths - set(wanted))
                    for ln in stale:
                        try:
                            os.remove(os.path.join(self.pack_dir, ln.split()[-1]))
                        except OSError:
                            pass
                    self.log(f"  [ForestHeights] remaking {len(stale_paths | set(wanted))} forest "
                             f"copies from an earlier {PACK_NAME}")
                    lines = [ln for ln in lines if ln.split()[-2] not in stale_paths]
                changed = bool(remade)
                for virtual in wanted + remade:
                    ok, added = self._write_copy(virtual, lines)
                    if not ok and virtual not in remade:
                        return False
                    changed |= added
                if changed or not os.path.isfile(os.path.join(self.pack_dir, "library.txt")):
                    self._write_library(lines)
                if self._pack_version() != version_text():
                    with open(os.path.join(self.pack_dir, VERSION_FILE), "w", encoding="utf-8") as f:
                        f.write(version_text() + "\n")
            return True
        except (OSError, TimeoutError) as exc:
            self.log(f"  [ForestHeights] could not write {PACK_NAME} ({exc})")
            return False

    def _write_copy(self, virtual, lines):
        """Write one copy's season files and add its exports to ``lines``: (ok, lines added)."""
        parsed = parse_virtual(virtual)
        if not parsed:
            self.log(f"  [ForestHeights] not a forest copy: {virtual}")
            return False, False
        g8_path, kind, height = parsed
        evergreen = is_evergreen(virtual)
        summer = is_summer(virtual)
        seasons = self.exports.get(g8_path)
        stock = self.stock_height(g8_path) if kind == "h" else None
        if not seasons or (kind == "h" and not stock):
            self.log(f"  [ForestHeights] no X-Plane forest for {virtual}")
            return False, False
        if evergreen:
            # The summer file for every season (plain EXPORT, season "").
            rel = seasons.get("sum") or seasons.get("") or next(iter(seasons.values()))
            seasons = {"": rel}
        added = False
        seg = _split_virtual(virtual)[0]
        for season, rel in sorted(seasons.items()):
            sub = f"{os.path.dirname(rel)}_{seg}"
            dst_rel = f"{sub}/{os.path.basename(rel)}"
            dst = os.path.join(self.pack_dir, sub, os.path.basename(rel))
            if not os.path.isfile(dst):
                with open(os.path.join(self.xp_dir, rel), "r", encoding="utf-8",
                          errors="ignore", newline="") as f:
                    text = f.read()
                suffix = f"_o4{kind}{_fmt(height)}".replace(".", "p")
                if evergreen or (summer and season == "sum"):
                    text = evergreen_text(text)
                text = self._copy_text(text, kind, height, stock, suffix)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                tmp = dst + ".tmp"
                with open(tmp, "w", encoding="utf-8", newline="") as f:
                    f.write(text)
                os.replace(tmp, dst)
            line = (f"EXPORT_SEASON {season}\t{virtual}\t{dst_rel}" if season
                    else f"EXPORT\t{virtual}\t{dst_rel}")
            if line not in lines:
                lines.append(line)
                added = True
        return True, added

    def _pack_version(self):
        try:
            with open(os.path.join(self.pack_dir, VERSION_FILE), "r", encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            return None

    def _stale_lines(self, lines):
        """The export lines of copies made by an earlier rule: all of them after a
        SIGNATURE change, else those of the kinds whose KIND_REVISIONS moved on."""
        if not lines:
            return []
        rows = (self._pack_version() or "").splitlines()
        if not rows or rows[0].strip() != SIGNATURE:
            return list(lines)
        revs = {}
        for row in rows[1:]:
            kind, _eq, rev = row.strip().partition("=")
            if rev.strip().isdigit():
                revs[kind.strip()] = int(rev)
        kinds = {k for k, v in KIND_REVISIONS.items() if revs.get(k, 1) != v}
        if not kinds:
            return []
        return [ln for ln in lines if (parse_virtual(ln.split()[-2]) or ("", ""))[1] in kinds]

    def _read_pack_exports(self):
        lib = os.path.join(self.pack_dir, "library.txt")
        if not os.path.isfile(lib):
            return []
        with open(lib, "r", encoding="utf-8") as f:
            return [ln.rstrip("\n") for ln in f if ln.startswith("EXPORT")]

    def _write_library(self, lines):
        lib = os.path.join(self.pack_dir, "library.txt")
        tmp = lib + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write("A\n1200\nLIBRARY\n\n")
            f.write("# Written by Ortho4XP: X-Plane 12 forests with shorter trees.\n")
            for ln in sorted(lines):
                f.write(ln + "\n")
        os.replace(tmp, lib)
