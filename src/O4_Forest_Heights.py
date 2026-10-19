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
  * shrubs ``s<level>``: X-Plane's shrubs planted on a SHRUB_SPACING_M grid
    (its shrublands use 10 m for 2-6 m shrubs, which reads as a few bushes
    on grass), cut down to the cap level where one applies (``s0``: none),
    the grid closing up with them down to SHRUB_MIN_SPACING_M;
  * evergreen ``ev`` (and ``<kind><height>-ev`` for the copies above): the
    summer file alone, without its SEASONAL autumn texture, exported for
    every season. X-Plane runs its seasons by the calendar of the hemisphere,
    so on the equator its temperate-band forests (which it plants on tropical
    highlands such as Eldoret and Addis Ababa) turn brown in October; its own
    hot-band forests are already exported like this.

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
# Densest planting grid of a copy, metres: X-Plane's own small shrubs.
MIN_SPACING_M = 4.0
# Shrub copies' grid, metres. X-Plane's shrub crowns are 3-4 m across, so at
# 4 m they stood apart (RCKH, 2026-10-19: "shrubs sparse"); at 3 m they close
# into cover. Shrubs cut down under an approach cap have smaller crowns, so
# the grid shrinks with them (by the shrubs' mean cut), to SHRUB_MIN_SPACING_M.
SHRUB_SPACING_M = 3.0
SHRUB_MIN_SPACING_M = 1.5
# Suffix of an evergreen copy's kind, and the kind of a plain evergreen copy.
EVERGREEN_SUFFIX = "-ev"
EVERGREEN_KIND = "ev"
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


def scale_trees(text, factor_for, mesh_suffix="", spacing_m=None, spacing_min_m=None):
    """``text`` with every tree scaled by ``factor_for(layer, min h, max h)``.

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
                    current = float(factor_for(layer, float(parts[lo_i]), float(parts[hi_i])))
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
    """Shrub copy: planted on a SHRUB_SPACING_M grid, shrubs taller than ``cap_m`` cut down to
    it and the grid closed up by their mean cut (to SHRUB_MIN_SPACING_M)."""
    factor_for = ((lambda _layer, _lo, hi: min(1.0, cap_m / hi) if hi > 0 else 1.0) if cap_m
                  else (lambda _layer, _lo, _hi: 1.0))
    return scale_trees(text, factor_for, mesh_suffix, spacing_m=SHRUB_SPACING_M,
                       spacing_min_m=SHRUB_MIN_SPACING_M)


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


def parse_virtual(path):
    """(source forest path, kind 'h', 'c', 's' or 'e', height) of a copy's path, else None.

    An evergreen copy of an h/c/s copy parses as that copy (is_evergreen tells them
    apart); a plain evergreen copy is kind 'e', height 0.
    """
    parts = _split_virtual(path)
    if not parts:
        return None
    seg, rest = parts
    source = ("lib/vegetation/" if rest.startswith("trees/") else "lib/g8/") + rest
    if seg == EVERGREEN_KIND:
        return source, "e", 0.0
    if seg.endswith(EVERGREEN_SUFFIX):
        seg = seg[:-len(EVERGREEN_SUFFIX)]
    if len(seg) < 2 or seg[0] not in "hcs":
        return None
    try:
        height = float(seg[1:])
    except ValueError:
        return None
    return source, seg[0], height


def is_evergreen(path):
    """True for an evergreen copy's path."""
    parts = _split_virtual(path)
    return bool(parts) and (parts[0] == EVERGREEN_KIND or parts[0].endswith(EVERGREEN_SUFFIX))


def evergreen_virtual_path(path):
    """The evergreen copy of an X-Plane forest or of one of its copies (unchanged if already one)."""
    path = str(path)
    if is_evergreen(path):
        return path
    parts = _split_virtual(path)
    if parts:
        return f"{VIRTUAL_PREFIX}{parts[0]}{EVERGREEN_SUFFIX}/{parts[1]}"
    return f"{VIRTUAL_PREFIX}{EVERGREEN_KIND}/{_copy_name(path)}"


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
        """The shrub copy of ``shrub_path`` (an X-Plane shrub forest) under ``cap_m``, or None
        when X-Plane has no such forest."""
        if not self.stats(shrub_path):
            return None
        stats = self.stats(shrub_path)
        level = cap_level(cap_m) if cap_m is not None and stats.max > cap_m + 1e-6 else 0.0
        return band_virtual_path(shrub_path, level, "s")

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
                if lines and self._pack_version() != SIGNATURE:
                    # Copies from an earlier rule: remade under the same paths,
                    # so tiles built with them pick up the new ones.
                    remade = sorted({ln.split()[-2] for ln in lines} - set(wanted))
                    for ln in lines:
                        try:
                            os.remove(os.path.join(self.pack_dir, ln.split()[-1]))
                        except OSError:
                            pass
                    self.log(f"  [ForestHeights] remaking {len(remade) + len(wanted)} forest "
                             f"copies from an earlier {PACK_NAME}")
                    lines = []
                changed = bool(remade)
                for virtual in wanted + remade:
                    ok, added = self._write_copy(virtual, lines)
                    if not ok and virtual not in remade:
                        return False
                    changed |= added
                if changed or not os.path.isfile(os.path.join(self.pack_dir, "library.txt")):
                    self._write_library(lines)
                if self._pack_version() != SIGNATURE:
                    with open(os.path.join(self.pack_dir, VERSION_FILE), "w", encoding="utf-8") as f:
                        f.write(SIGNATURE + "\n")
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
                if evergreen:
                    text = evergreen_text(text)
                if kind == "h":
                    text = band_trees(text, height, stock, suffix)
                elif kind == "s":
                    text = shrub_trees(text, height or None, suffix)
                elif kind == "c":
                    text = cap_trees(text, height, suffix)
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
