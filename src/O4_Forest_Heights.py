"""Height-banded copies of X-Plane 12's climate forests.

X-Plane's climate forests (``lib/g8/{broad,coni,mixed}_<temp>_<moist>.for``)
plant one canopy layer of fixed height, ~13-27 m in most bands. Canopy height
measured on the ortho is often much lower (hedges, young stands, gardens), so
forest polygons on canopy textures can take a copy of the same forest with
its canopy-layer trees scaled to one of BAND_HEIGHTS_M: same species,
understory, spacing, textures and seasons, only shorter.

The copies are written on demand into the ``O4XP_Forest_Heights`` library
pack in Custom Scenery, made from the X-Plane install the overlay builds for:
``<season>_h<band>/<file>.for`` beside a ``textures`` link to X-Plane's
forest textures (their ``../textures/`` paths then resolve unchanged), and
``library.txt`` exports them as ``o4xp/forests/h<band>/<g8 name>.for`` for
every season the original exports. X-Plane adds the new pack to
scenery_packs.ini when it next starts.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

PACK_NAME = "O4XP_Forest_Heights"
VIRTUAL_PREFIX = "o4xp/forests/"
# Canopy-layer mean heights of the copies, metres.
BAND_HEIGHTS_M = (6.0, 9.0, 12.0, 15.0)
# A copy is used only when it is at least this much shorter than the original.
MIN_GAIN_M = 3.0
SIGNATURE = "forest-heights-v1"
_FAMILIES = ("broad_", "coni_", "mixed_")
_LOCK_STALE_S = 120.0


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


def canopy_mean_height(text):
    """Frequency-weighted mean height of a .for's canopy (layer 1) trees, or None."""
    total = weight = 0.0
    for line in text.splitlines():
        parts = line.split()
        fields = parts and _tree_fields(parts)
        if not fields:
            continue
        lo_i, hi_i, _nom_i, lay_i = fields
        try:
            if int(float(parts[lay_i])) != 1:
                continue
            freq = abs(float(parts[6]))
            total += freq * 0.5 * (float(parts[lo_i]) + float(parts[hi_i]))
            weight += freq
        except ValueError:
            continue
    return total / weight if weight > 0 else None


def _fmt(value):
    return f"{value:.2f}".rstrip("0").rstrip(".")


def scale_canopy(text, factor):
    """``text`` with every canopy-layer tree's heights multiplied by ``factor``."""
    out = []
    for line in text.splitlines(keepends=True):
        parts = line.split()
        fields = parts and _tree_fields(parts)
        if fields:
            lo_i, hi_i, nom_i, lay_i = fields
            try:
                if int(float(parts[lay_i])) == 1:
                    for i in (lo_i, hi_i) + ((nom_i,) if nom_i is not None else ()):
                        parts[i] = _fmt(float(parts[i]) * factor)
                    indent = line[:len(line) - len(line.lstrip())]
                    ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
                    line = indent + "\t".join(parts) + ending
            except ValueError:
                pass
        out.append(line)
    return "".join(out)


def band_virtual_path(g8_path, band):
    return f"{VIRTUAL_PREFIX}h{int(band)}/{os.path.basename(g8_path)}"


def source_path(path):
    """The ``lib/g8`` forest a banded copy was made from; other paths unchanged."""
    path = str(path)
    if not path.startswith(VIRTUAL_PREFIX):
        return path
    parts = path[len(VIRTUAL_PREFIX):].split("/", 1)
    return "lib/g8/" + parts[1] if len(parts) == 2 else path


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
    """Picks and writes height-banded copies of ``lib/g8`` forests."""

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
                self.exports = {v: s for v, s in read_exports(lib).items()
                                if v.startswith("lib/g8/") and os.path.basename(v).startswith(_FAMILIES)}
            except OSError:
                self.exports = {}
        self._stock = {}

    @property
    def available(self):
        return bool(self.exports)

    def stock_height(self, g8_path):
        """Canopy mean height of the forest X-Plane exports at ``g8_path`` (summer), or None."""
        if g8_path not in self._stock:
            seasons = self.exports.get(g8_path) or {}
            rel = seasons.get("sum") or seasons.get("") or next(iter(seasons.values()), None)
            height = None
            if rel:
                try:
                    with open(os.path.join(self.xp_dir, rel), "r", encoding="utf-8", errors="ignore") as f:
                        height = canopy_mean_height(f.read())
                except OSError:
                    height = None
            self._stock[g8_path] = height
        return self._stock[g8_path]

    def path_for(self, g8_path, measured_h):
        """Banded copy for a polygon whose canopy is ``measured_h`` m tall, else ``g8_path``."""
        stock = self.stock_height(g8_path)
        if stock is None or measured_h is None:
            return g8_path
        # The original competes as one more height; copies too close to it are not offered.
        choices = [b for b in BAND_HEIGHTS_M if b <= stock - MIN_GAIN_M] + [stock]
        best = min(choices, key=lambda b: abs(b - measured_h))
        return g8_path if best == stock else band_virtual_path(g8_path, best)

    def _source_of(self, virtual):
        parts = virtual[len(VIRTUAL_PREFIX):].split("/", 1)
        if len(parts) != 2 or not parts[0].startswith("h"):
            return None, None
        try:
            band = float(parts[0][1:])
        except ValueError:
            return None, None
        return source_path(virtual), band

    def ensure(self, virtual_paths):
        """Write every banded copy in ``virtual_paths`` that the pack lacks; False on failure."""
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
                changed = False
                for virtual in wanted:
                    g8_path, band = self._source_of(virtual)
                    seasons = self.exports.get(g8_path)
                    stock = self.stock_height(g8_path) if seasons else None
                    if not seasons or not stock:
                        self.log(f"  [ForestHeights] no X-Plane forest for {virtual}")
                        return False
                    for season, rel in sorted(seasons.items()):
                        sub = f"{os.path.dirname(rel)}_h{int(band)}"
                        dst_rel = f"{sub}/{os.path.basename(rel)}"
                        dst = os.path.join(self.pack_dir, sub, os.path.basename(rel))
                        if not os.path.isfile(dst):
                            with open(os.path.join(self.xp_dir, rel), "r", encoding="utf-8",
                                      errors="ignore", newline="") as f:
                                text = f.read()
                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                            tmp = dst + ".tmp"
                            with open(tmp, "w", encoding="utf-8", newline="") as f:
                                f.write(scale_canopy(text, band / stock))
                            os.replace(tmp, dst)
                        line = (f"EXPORT_SEASON {season}\t{virtual}\t{dst_rel}" if season
                                else f"EXPORT\t{virtual}\t{dst_rel}")
                        if line not in lines:
                            lines.append(line)
                            changed = True
                if changed or not os.path.isfile(os.path.join(self.pack_dir, "library.txt")):
                    self._write_library(lines)
            return True
        except (OSError, TimeoutError) as exc:
            self.log(f"  [ForestHeights] could not write {PACK_NAME} ({exc})")
            return False

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
            f.write("# Written by Ortho4XP: X-Plane 12 climate forests with shorter canopy trees.\n")
            for ln in sorted(lines):
                f.write(ln + "\n")
        os.replace(tmp, lib)
