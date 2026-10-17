"""Tree spacing of the X-Plane forests the vegetation overlay plants.

X-Plane plants a forest polygon on a grid: one candidate tree every
``SPACING x z`` metres (from the .for file), each moved by up to ``RANDOM``,
kept where it falls inside the polygon. So the size at which a forest
polygon reliably gets a tree, and the size at which a hole in it reliably
removes one, is a number of grid cells of *that* forest, not a fixed area.

Forest virtual paths (``lib/g8/...``, ``simheaven/forests/...``) are looked up
in the library.txt files of the active Custom Scenery packs and of X-Plane's
default scenery, including ``EXPORT_SEASON`` lines (summer preferred), and the
first .for found gives the spacing. Unknown paths fall back to
DEFAULT_SPACING_M.
"""

import os

DEFAULT_SPACING_M = 12.0
_EXPORTS = ("EXPORT", "EXPORT_BACKUP", "EXPORT_EXTEND", "EXPORT_EXCLUDE")


def _read_spacing(for_path):
    try:
        with open(for_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 3 and parts[0] == "SPACING":
                    return float(parts[1]), float(parts[2])
    except (OSError, ValueError):
        pass
    return None


def _library_files(custom_scenery_dir):
    """(package_dir, library.txt) for active custom packs, then default scenery."""
    out = []
    if custom_scenery_dir and os.path.isdir(custom_scenery_dir):
        try:
            names = sorted(os.listdir(custom_scenery_dir), key=str.lower)
        except OSError:
            names = []
        for name in names:
            pkg = os.path.join(custom_scenery_dir, name)
            lib = os.path.join(pkg, "library.txt")
            if os.path.isfile(lib):
                out.append((pkg, lib))
        default_root = os.path.join(os.path.dirname(os.path.abspath(custom_scenery_dir)),
                                    "Resources", "default scenery")
        if os.path.isdir(default_root):
            for name in sorted(os.listdir(default_root), key=str.lower):
                pkg = os.path.join(default_root, name)
                lib = os.path.join(pkg, "library.txt")
                if os.path.isfile(lib):
                    out.append((pkg, lib))
    return out


class ForestSpacing:
    """``spacing(virtual_path) -> (x, z)`` metres, from the installed .for files."""

    def __init__(self, custom_scenery_dir):
        self.custom_scenery_dir = custom_scenery_dir
        self._exports = None     # virtual path (lower) -> [physical candidates]
        self._cache = {}
        self.default_root = None
        if custom_scenery_dir:
            self.default_root = os.path.join(
                os.path.dirname(os.path.abspath(custom_scenery_dir)), "Resources", "default scenery")

    def _index(self):
        if self._exports is not None:
            return self._exports
        exports = {}
        for pkg, lib in _library_files(self.custom_scenery_dir):
            try:
                with open(lib, "r", encoding="utf-8", errors="ignore") as handle:
                    lines = handle.readlines()
            except OSError:
                continue
            for raw in lines:
                head = raw.split(None, 1)
                if not head:
                    continue
                cmd = head[0].upper()
                # The physical path is the rest of the line: it may hold spaces
                # ("1200 forests/sum/broad_vcld.for").
                if cmd in _EXPORTS:
                    parts = raw.split(None, 2)
                    if len(parts) < 3:
                        continue
                    virt, phys, summer = parts[1], parts[2].strip(), True
                elif cmd in ("EXPORT_SEASON", "EXPORT_RATIO"):
                    parts = raw.split(None, 3)
                    if len(parts) < 4:
                        continue
                    summer = cmd == "EXPORT_RATIO" or "sum" in parts[1].lower().split(",")
                    virt, phys = parts[2], parts[3].strip()
                else:
                    continue
                if not virt.lower().endswith(".for"):
                    continue
                cands = [os.path.join(pkg, phys)]
                if self.default_root:
                    cands.append(os.path.join(self.default_root, phys))
                entry = exports.setdefault(virt.replace("\\", "/").lower(), [])
                # Summer exports first: the season the trees are planted for.
                (entry.insert(0, cands) if summer else entry.append(cands))
        self._exports = exports
        return exports

    def spacing(self, virtual_path):
        key = (virtual_path or "").replace("\\", "/").lower()
        if key in self._cache:
            return self._cache[key]
        result = None
        for cands in self._index().get(key, ()):
            for path in cands:
                if os.path.isfile(path):
                    result = _read_spacing(path)
                    if result:
                        break
            if result:
                break
        if not result or result[0] <= 1.0:
            result = (DEFAULT_SPACING_M, DEFAULT_SPACING_M)
        self._cache[key] = result
        return result

    def cell_m2(self, virtual_path):
        sx, sz = self.spacing(virtual_path)
        return sx * sz
