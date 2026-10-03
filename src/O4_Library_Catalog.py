"""Persistent catalog of installed X-Plane scenery libraries.

Every building-overlay tile used to parse every ``library.txt`` (checking each
export's file on disk), re-measure every library OBJ and re-analyse its roof
colour, which cost minutes a tile although installed libraries rarely change.
The catalog keeps the parsed Custom Scenery and default-scenery exports, the
``library.txt`` prefix hits, the measured OBJ8 footprints and other per-OBJ
results (:meth:`LibraryCatalog.cached`) in one file shared by all tiles. It is
independent of the per-tile cache setting. Per-OBJ results are computed the
first time a tile needs them.

Opening the catalog re-stats each package's ``library.txt`` (about a second)
and re-parses only packages whose file changed, appeared or vanished, dropping
the per-OBJ results stored under them; ``scenery_packs.ini`` activation is read
fresh every time. :func:`rescan` rebuilds it
from scratch for changes those stamps cannot see, such as OBJ files replaced
in place.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import pickle
import time

import O4_Asset_Inventory as ASSETINV
from O4_Scenery_Packs import active_scenery_pack_dirs, resolve_custom_scenery_dir


CATALOG_VERSION = 1
CATALOG_FILENAME = "library_catalog.pkl"
# ``library.txt`` substrings the overlay asks about. Hits are recorded per
# package so availability checks never re-read the files.
KNOWN_PREFIXES = ("sfd_global/", "simheaven/")
EXPORT_SUFFIXES = (".obj", ".fac")
# Commands the custom-scenery object index honours (EXPORT_EXTEND excluded).
INDEX_COMMANDS = frozenset({"EXPORT", "EXPORT_EXCLUDE", "EXPORT_BACKUP", "EXPORT_RATIO"})
_IO_WORKERS = 16


def measure_obj8_bounds(obj_path):
    """Return OBJ8 X/Z footprint bounds as ``(xmin, xmax, zmin, zmax)``, or None."""
    min_x = min_z = float("inf")
    max_x = max_z = float("-inf")
    n_vertices = 0
    try:
        handle = open(obj_path, "r", encoding="utf-8", errors="ignore")
    except OSError:
        return None
    with handle:
        for raw_line in handle:
            line = raw_line.lstrip()
            if not line.startswith("VT"):
                continue
            parts = line.split()
            if len(parts) < 4 or parts[0] != "VT":
                continue
            try:
                x = float(parts[1])
                z = float(parts[3])
            except ValueError:
                continue
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_z = min(min_z, z)
            max_z = max(max_z, z)
            n_vertices += 1
    if n_vertices <= 0 or max_x <= min_x or max_z <= min_z:
        return None
    return (float(min_x), float(max_x), float(min_z), float(max_z))


def _dir_key(path):
    return os.path.normcase(os.path.abspath(path))


def _export_has_suffix(export, suffixes):
    # Mirrors ASSETINV.parse_library_exports' suffix test.
    return (
        export.virtual_path.lower().endswith(suffixes)
        or export.physical_path.lower().endswith(suffixes)
    )


def _parse_package(name, package_dir, library_txt):
    """Return the stored record for one package's ``library.txt``."""
    try:
        with open(library_txt, "r", encoding="utf-8", errors="ignore") as handle:
            text = handle.read().lower()
    except OSError:
        text = ""
    return {
        "exports": ASSETINV.parse_library_exports(
            library_txt,
            package_name=name,
            package_dir=package_dir,
            suffixes=EXPORT_SUFFIXES,
        ),
        "prefixes": frozenset(prefix for prefix in KNOWN_PREFIXES if prefix in text),
    }


class LibraryCatalog:
    """Parsed library exports and per-OBJ results for one Custom Scenery folder."""

    def __init__(self, custom_scenery_dir, catalog_path):
        self.custom_scenery_dir = custom_scenery_dir
        self.catalog_path = catalog_path
        self.packages = {}        # dir key -> package record
        self.obj_bounds = {}      # resolved OBJ path -> bounds tuple or None
        self.derived = {}         # namespace -> {OBJ path: result}
        self.default_order = []   # dir keys of default-scenery libraries, walk order
        self.active = []          # [(name, dir)] scenery_packs.ini order
        self.listing = []         # [(name, dir)] Custom Scenery folders, sorted
        self.scanned_at = None
        self.dirty = False

    # -- persistence --------------------------------------------------------

    def _payload(self):
        return {
            "version": CATALOG_VERSION,
            "custom_scenery_dir": _dir_key(self.custom_scenery_dir),
            "scanned_at": self.scanned_at,
            "packages": self.packages,
            "obj_bounds": self.obj_bounds,
            "derived": self.derived,
        }

    def _restore(self, payload):
        if not isinstance(payload, dict):
            return False
        if payload.get("version") != CATALOG_VERSION:
            return False
        if payload.get("custom_scenery_dir") != _dir_key(self.custom_scenery_dir):
            return False
        self.packages = payload.get("packages") or {}
        self.obj_bounds = payload.get("obj_bounds") or {}
        self.derived = payload.get("derived") or {}
        self.scanned_at = payload.get("scanned_at")
        return True

    def save(self):
        """Write the catalog atomically; concurrent writers simply race."""
        if not self.catalog_path:
            return False
        tmp_path = f"{self.catalog_path}.{os.getpid()}.tmp"
        try:
            os.makedirs(os.path.dirname(self.catalog_path), exist_ok=True)
            with open(tmp_path, "wb") as handle:
                pickle.dump(self._payload(), handle, protocol=pickle.HIGHEST_PROTOCOL)
            for attempt in range(5):
                try:
                    os.replace(tmp_path, self.catalog_path)
                    break
                except PermissionError:
                    # Another build is reading the old file (Windows lock).
                    if attempt == 4:
                        raise
                    time.sleep(0.2)
        except OSError as exc:
            print(f"[Library] Could not save library catalog: {exc}", flush=True)
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            return False
        self.dirty = False
        return True

    def save_if_dirty(self):
        return self.save() if self.dirty else True

    # -- synchronisation ----------------------------------------------------

    def _current_layout(self):
        """Return ``{dir key: (name, dir, library_txt, stamp)}`` for installed packages."""
        cs_dir = self.custom_scenery_dir
        self.active = active_scenery_pack_dirs(cs_dir)
        try:
            entries = sorted(os.scandir(cs_dir), key=lambda item: item.name.lower())
        except OSError:
            entries = []
        self.listing = []
        for entry in entries:
            # One corrupt folder (WinError 1392) must not sink every build.
            try:
                is_dir = entry.is_dir()
            except OSError as exc:
                print(f"[Library] Skipping unreadable scenery folder {entry.path}: {exc}", flush=True)
                continue
            if is_dir:
                self.listing.append((entry.name, entry.path))
        layout = {}
        # Active names first: exports carry the scenery_packs.ini name.
        for name, package_dir in self.active + self.listing:
            key = _dir_key(package_dir)
            if key in layout:
                continue
            library_txt = os.path.join(package_dir, "library.txt")
            try:
                stat = os.stat(library_txt)
            except OSError:
                continue
            layout[key] = (
                name,
                package_dir,
                library_txt,
                (name, stat.st_size, stat.st_mtime_ns),
            )
        self.default_order = []
        xplane_root = self._xplane_root()
        if xplane_root:
            for name, package_dir, library_txt in ASSETINV.iter_library_txt_files(
                xplane_root=xplane_root, include_default=True
            ):
                key = _dir_key(package_dir)
                if key in layout:
                    continue
                try:
                    stat = os.stat(library_txt)
                except OSError:
                    continue
                self.default_order.append(key)
                layout[key] = (
                    name,
                    package_dir,
                    library_txt,
                    (name, stat.st_size, stat.st_mtime_ns),
                )
        return layout

    def _xplane_root(self):
        if os.path.basename(self.custom_scenery_dir).lower() != "custom scenery":
            return None
        return os.path.dirname(self.custom_scenery_dir)

    def _forget_results_under(self, package_dir):
        """Drop footprints and derived results for OBJs inside ``package_dir``."""
        prefix = _dir_key(package_dir).rstrip("\\/") + os.sep
        for results in (self.obj_bounds, *self.derived.values()):
            stale = [
                path for path in results
                if os.path.normcase(path).startswith(prefix)
            ]
            for path in stale:
                del results[path]

    def sync(self, *, full=False):
        """Bring the catalog in line with installed packages; return counts."""
        started = time.perf_counter()
        layout = self._current_layout()
        if full:
            self.packages = {}
            self.obj_bounds = {}
            self.derived = {}
        removed = [key for key in self.packages if key not in layout]
        for key in removed:
            self._forget_results_under(self.packages.pop(key)["dir"])
        todo = [
            (key, item) for key, item in layout.items()
            if self.packages.get(key, {}).get("stamp") != item[3]
        ]
        if todo:
            for key, (_name, package_dir, _lib, _stamp) in todo:
                if key in self.packages:
                    self._forget_results_under(package_dir)
            with ThreadPoolExecutor(max_workers=_IO_WORKERS) as pool:
                parsed = list(pool.map(
                    lambda job: _parse_package(job[1][0], job[1][1], job[1][2]),
                    todo,
                ))
            for (key, (name, package_dir, library_txt, stamp)), record in zip(todo, parsed):
                record.update({
                    "name": name,
                    "dir": package_dir,
                    "library_txt": library_txt,
                    "stamp": stamp,
                })
                self.packages[key] = record
        if full or todo or removed:
            self.scanned_at = time.time()
            self.dirty = True
        return {
            "packages": len(self.packages),
            "parsed": len(todo),
            "removed": len(removed),
            "seconds": time.perf_counter() - started,
        }

    # -- queries ------------------------------------------------------------

    def _record(self, package_dir):
        return self.packages.get(_dir_key(package_dir))

    def runtime_exports(self, package_name_patterns, suffixes):
        """Return what ``ASSETINV.scan_library_exports`` returns for these packages.

        Returns None when ``suffixes`` asks for more than the catalog stores.
        """
        suffixes = tuple(str(suffix).lower() for suffix in suffixes or ())
        if not suffixes or not set(suffixes) <= set(EXPORT_SUFFIXES):
            return None
        patterns = [
            pattern.strip().lower()
            for pattern in (package_name_patterns or ())
            if pattern and pattern.strip()
        ]
        exports = []
        # ASSETINV._package_dirs falls back to the folder listing without an ini.
        for name, package_dir in (self.active or self.listing):
            if patterns and not any(pattern in name.lower() for pattern in patterns):
                continue
            record = self._record(package_dir)
            if record is None:
                continue
            exports.extend(
                export for export in record["exports"]
                if _export_has_suffix(export, suffixes)
            )
        return exports

    def active_object_index(self):
        """Return ``{normalized virtual path: first resolved OBJ}`` over active packs."""
        index = {}
        for _name, package_dir in self.active:
            record = self._record(package_dir)
            if record is None:
                continue
            for export in record["exports"]:
                if export.command not in INDEX_COMMANDS:
                    continue
                if not export.resolved_path:
                    continue
                if not export.physical_path.lower().endswith(".obj"):
                    continue
                index.setdefault(
                    ASSETINV.normalize_library_path(export.virtual_path),
                    export.resolved_path,
                )
        return index

    def default_exports(self, target_virtual_paths, suffixes=(".obj",)):
        """Return default-scenery exports for these virtual paths.

        Matches ``ASSETINV.scan_library_exports(include_default=True, ...)``.
        """
        suffixes = tuple(str(suffix).lower() for suffix in suffixes or ())
        targets = {
            ASSETINV.normalize_library_path(path)
            for path in target_virtual_paths or ()
            if path
        }
        exports = []
        for key in self.default_order:
            record = self.packages.get(key)
            if record is None:
                continue
            exports.extend(
                export for export in record["exports"]
                if ASSETINV.normalize_library_path(export.virtual_path) in targets
                and (not suffixes or _export_has_suffix(export, suffixes))
            )
        return exports

    def has_prefix(self, prefix):
        """Return whether any Custom Scenery folder's library.txt mentions ``prefix``.

        Returns None for prefixes the catalog does not record.
        """
        prefix = (prefix or "").lower()
        if prefix not in KNOWN_PREFIXES:
            return None
        for _name, package_dir in self.listing:
            record = self._record(package_dir)
            if record is not None and prefix in record["prefixes"]:
                return True
        return False

    def bounds_for(self, obj_path):
        """Return cached OBJ8 footprint bounds, measuring on first use."""
        if not obj_path:
            return None
        try:
            return self.obj_bounds[obj_path]
        except KeyError:
            pass
        bounds = measure_obj8_bounds(obj_path)
        self.obj_bounds[obj_path] = bounds
        self.dirty = True
        return bounds

    def cached(self, namespace, obj_path, build):
        """Return ``build()`` for one OBJ, computed once and kept with the catalog.

        Results are dropped with the package the OBJ lives in and on rescan;
        put a version in ``namespace`` to retire results when ``build`` changes.
        """
        results = self.derived.setdefault(namespace, {})
        try:
            return results[obj_path]
        except KeyError:
            pass
        value = build()
        results[obj_path] = value
        self.dirty = True
        return value

    def summary(self):
        exports = sum(len(record["exports"]) for record in self.packages.values())
        derived = sum(len(results) for results in self.derived.values())
        return (
            f"{len(self.packages)} library packages, {exports} exports, "
            f"{len(self.obj_bounds)} OBJ footprints, {derived} other OBJ results"
        )


def catalog_file(cache_dir):
    return os.path.join(cache_dir, CATALOG_FILENAME) if cache_dir else None


def open_catalog(custom_scenery_dir, cache_dir):
    """Load and sync the catalog; None when caching or the folder is unavailable."""
    custom_scenery_dir = resolve_custom_scenery_dir(custom_scenery_dir)
    if not cache_dir or not custom_scenery_dir or not os.path.isdir(custom_scenery_dir):
        return None
    catalog = LibraryCatalog(custom_scenery_dir, catalog_file(cache_dir))
    loaded = False
    try:
        with open(catalog.catalog_path, "rb") as handle:
            loaded = catalog._restore(pickle.load(handle))
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"[Library] Ignoring unreadable library catalog: {exc}", flush=True)
    counts = catalog.sync()
    if not loaded:
        print(
            "[Library] No library catalog yet; scanned installed libraries "
            f"in {counts['seconds']:.1f}s ({catalog.summary()})",
            flush=True,
        )
    elif counts["parsed"] or counts["removed"]:
        print(
            f"[Library] Library catalog: re-read {counts['parsed']} changed and "
            f"dropped {counts['removed']} removed package(s) "
            f"in {counts['seconds']:.1f}s ({catalog.summary()})",
            flush=True,
        )
    else:
        print(
            f"[Library] Library catalog up to date ({catalog.summary()}; "
            f"checked in {counts['seconds']:.1f}s)",
            flush=True,
        )
    return catalog


def rescan(custom_scenery_dir, cache_dir):
    """Rebuild the catalog from scratch; per-OBJ results are recomputed on use."""
    custom_scenery_dir = resolve_custom_scenery_dir(custom_scenery_dir)
    if not custom_scenery_dir or not os.path.isdir(custom_scenery_dir):
        raise FileNotFoundError(f"Custom Scenery folder not found: {custom_scenery_dir!r}")
    if not cache_dir:
        raise ValueError("No library cache folder given")
    catalog = LibraryCatalog(custom_scenery_dir, catalog_file(cache_dir))
    print(f"[Library] Scanning libraries in {custom_scenery_dir} ...", flush=True)
    counts = catalog.sync(full=True)
    if not catalog.save():
        raise OSError(f"Could not write {catalog.catalog_path}")
    print(
        f"[Library] Library catalog rebuilt in {counts['seconds']:.1f}s "
        f"({catalog.summary()}); the next building build re-measures "
        "OBJ footprints and roof colours once.",
        flush=True,
    )
    return catalog
