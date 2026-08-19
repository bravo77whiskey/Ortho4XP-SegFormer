"""Creation and removal of Custom Scenery links pointing at Ortho4XP tiles.

X-Plane only sees a tile once its build directory shows up inside the sim's
``Custom Scenery`` folder.  Rather than copying gigabytes around, Ortho4XP puts
a link there: a symlink on Mac/Linux, a directory junction on Windows
(junctions need no administrator rights, unlike Windows symlinks).

Links are matched by *target*, not by name: a user is free to rename
``zOrtho4XP_+50+008`` to anything they like, and the tile must not be linked a
second time under the canonical name when that happens.
"""

import os
import stat
import subprocess

import O4_File_Names as FNAMES
import O4_UI_Utils as UI
from O4_SFR_DSF_Utils import resolve_custom_scenery_dir

LINK_PREFIX = "zOrtho4XP_"
OVERLAY_LINK_NAME = "yOrtho4XP_Overlays"

_WINDOWS = os.name == "nt"

# Statuses returned by ensure_link() / ensure_tile_link().
CREATED = "created"
ALREADY_LINKED = "already_linked"
NAME_TAKEN = "name_taken"
DUPLICATE_TILE = "duplicate_tile"
IN_PLACE = "in_place"
NO_SCENERY_DIR = "no_scenery_dir"
NO_TARGET = "no_target"
FAILED = "failed"

_OK_STATUSES = (CREATED, ALREADY_LINKED)


class LinkResult:
    """Outcome of a link request, with a message worth showing to the user."""

    def __init__(self, status, message, link=None, target=None):
        self.status = status
        self.message = message
        self.link = link
        self.target = target

    def __bool__(self):
        return self.status in _OK_STATUSES

    def __repr__(self):
        return f"LinkResult({self.status!r}, {self.message!r}, link={self.link!r})"


################################################################################
# Link primitives
################################################################################
def is_link(path):
    """True for a symlink or (on Windows) a directory junction."""
    try:
        if os.path.islink(path):
            return True
    except OSError:
        return False
    if not _WINDOWS:
        return False
    isjunction = getattr(os.path, "isjunction", None)  # Python 3.12+
    if isjunction is not None:
        try:
            return bool(isjunction(path))
        except OSError:
            return False
    try:
        attributes = os.lstat(path).st_file_attributes
    except (OSError, AttributeError):
        return False
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def link_target(path):
    """Absolute target of a link, or None when path is not a link."""
    if not is_link(path):
        return None
    return os.path.realpath(path)


def is_broken_link(path):
    return is_link(path) and not os.path.exists(os.path.realpath(path))


def same_path(path_a, path_b):
    if not path_a or not path_b:
        return False
    try:
        return os.path.samefile(path_a, path_b)
    except OSError:
        return os.path.normcase(os.path.normpath(path_a)) == os.path.normcase(
            os.path.normpath(path_b)
        )


def _is_within(child, parent):
    child = os.path.normcase(os.path.realpath(child))
    parent = os.path.normcase(os.path.realpath(parent))
    return child == parent or child.startswith(parent.rstrip("\\/") + os.sep)


def make_link(link, target):
    """Create link -> target. Returns True on success."""
    link = os.path.abspath(link)
    target = os.path.abspath(target)
    if not _WINDOWS:
        try:
            os.symlink(target, link)
            return True
        except OSError as e:
            UI.lvprint(1, "ERROR: could not create symlink", link, ":", e)
            return False
    try:
        import _winapi

        _winapi.CreateJunction(target, link)
        return True
    except (ImportError, AttributeError, OSError, ValueError) as e:
        UI.vprint(2, f"CreateJunction failed for {link} ({e}), falling back to mklink.")
    try:
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", link, target],
            check=True,
            capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return True
    except (OSError, subprocess.CalledProcessError) as e:
        detail = getattr(e, "stderr", b"") or getattr(e, "stdout", b"")
        if isinstance(detail, bytes):
            detail = detail.decode(errors="ignore").strip()
        UI.lvprint(1, "ERROR: could not create junction", link, ":", detail or e)
        return False


def remove_link(path):
    """Remove a link without touching what it points at. True on success."""
    if not is_link(path):
        return False
    # POSIX symlinks (even to directories) go through unlink; Windows junctions
    # and directory symlinks refuse unlink and need rmdir, which drops only the
    # reparse point and leaves the target alone.
    try:
        os.unlink(path)
        return True
    except OSError:
        pass
    try:
        os.rmdir(path)
        return True
    except OSError as e:
        UI.lvprint(1, "ERROR: could not remove link", path, ":", e)
        return False


def iter_links(custom_scenery_dir):
    """Yield (name, path, target) for every link inside the Custom Scenery dir."""
    custom_scenery_dir = resolve_custom_scenery_dir(custom_scenery_dir)
    if not custom_scenery_dir or not os.path.isdir(custom_scenery_dir):
        return
    try:
        entries = list(os.scandir(custom_scenery_dir))
    except OSError as e:
        UI.vprint(2, f"Could not list {custom_scenery_dir}: {e}")
        return
    for entry in entries:
        if not is_link(entry.path):
            continue
        yield (entry.name, entry.path, os.path.realpath(entry.path))


def linked_targets(custom_scenery_dir):
    """Normalised realpaths of everything Custom Scenery currently links to.

    Cheaper than repeated find_links_to() calls when many tiles are checked.
    """
    return {
        os.path.normcase(resolved)
        for (_name, _path, resolved) in iter_links(custom_scenery_dir)
    }


def in_linked_targets(targets, path):
    return os.path.normcase(os.path.realpath(path)) in targets


def find_links_to(custom_scenery_dir, target):
    """Every link in Custom Scenery resolving to target, whatever its name."""
    return [
        path
        for (_name, path, resolved) in iter_links(custom_scenery_dir)
        if same_path(resolved, target)
    ]


################################################################################
# Tile links
################################################################################
def tile_link_name(lat, lon):
    return FNAMES.tile_dir(lat, lon)


def group_link_name(build_dir):
    return LINK_PREFIX + os.path.basename(os.path.normpath(build_dir))


def link_name_for(build_dir, lat, lon, grouped=False):
    """Canonical Custom Scenery name for a tile (or for a grouped build dir)."""
    return group_link_name(build_dir) if grouped else tile_link_name(lat, lon)


def _duplicate_tile_links(custom_scenery_dir, lat, lon, target):
    """Live links that look like the same tile but point somewhere else.

    Same-tile links which dangle are cleared on the way: they name a build
    directory that no longer exists, so they are no reason to skip anything.
    """
    token = FNAMES.short_latlon(lat, lon)
    duplicates = []
    for (name, path, resolved) in list(iter_links(custom_scenery_dir)):
        if token not in name or same_path(resolved, target):
            continue
        if not os.path.exists(resolved):
            UI.vprint(1, "Removing broken link " + name + " from Custom Scenery.")
            remove_link(path)
            continue
        duplicates.append(path)
    return duplicates


def ensure_link(custom_scenery_dir, target, link_name, duplicate_tile=None):
    """Make sure target is reachable from Custom Scenery exactly once.

    ``duplicate_tile`` is an optional (lat, lon) pair; when given, a link whose
    name carries those tile coordinates but resolves elsewhere is treated as a
    duplicate of this tile and no second link is made.
    """
    custom_scenery_dir = resolve_custom_scenery_dir(custom_scenery_dir)
    if not custom_scenery_dir:
        return LinkResult(NO_SCENERY_DIR, "Custom Scenery directory not set.")
    if not os.path.isdir(custom_scenery_dir):
        return LinkResult(
            NO_SCENERY_DIR,
            "Custom Scenery directory not found: " + custom_scenery_dir,
        )
    if not target or not os.path.isdir(target):
        return LinkResult(NO_TARGET, "Nothing to link, no such directory: " + str(target))

    target = os.path.realpath(target)
    if _is_within(target, custom_scenery_dir):
        return LinkResult(
            IN_PLACE,
            os.path.basename(target)
            + " already sits in Custom Scenery, no link needed.",
            target=target,
        )

    existing = find_links_to(custom_scenery_dir, target)
    if existing:
        return LinkResult(
            ALREADY_LINKED,
            "Already linked as " + os.path.basename(existing[0]) + ".",
            link=existing[0],
            target=target,
        )

    if duplicate_tile is not None:
        duplicates = _duplicate_tile_links(custom_scenery_dir, *duplicate_tile, target)
        if duplicates:
            names = ", ".join(os.path.basename(p) for p in duplicates)
            return LinkResult(
                DUPLICATE_TILE,
                names
                + " already links this tile from another build directory, skipping "
                + link_name
                + ".",
                link=duplicates[0],
                target=target,
            )

    link = os.path.join(custom_scenery_dir, link_name)
    if is_broken_link(link):
        UI.vprint(1, "Replacing broken link " + link_name + " in Custom Scenery.")
        remove_link(link)
    if os.path.lexists(link):
        return LinkResult(
            NAME_TAKEN,
            link_name
            + " already exists in Custom Scenery and points elsewhere, "
            + "leaving it untouched.",
            link=link,
            target=target,
        )

    if not make_link(link, target):
        return LinkResult(
            FAILED, "Could not link " + link_name + " in Custom Scenery.", link, target
        )
    return LinkResult(
        CREATED, "Linked " + link_name + " in Custom Scenery.", link, target
    )


def ensure_tile_link(custom_scenery_dir, build_dir, lat, lon, grouped=False):
    """Link one built tile (or the whole grouped build dir) into Custom Scenery."""
    link_name = link_name_for(build_dir, lat, lon, grouped)
    return ensure_link(
        custom_scenery_dir,
        build_dir,
        link_name,
        duplicate_tile=None if grouped else (lat, lon),
    )


def remove_links_to(custom_scenery_dir, target, link_name=None):
    """Drop every Custom Scenery link pointing at target. Returns removed paths.

    ``link_name`` names the canonical link: it is cleared as well when it dangles,
    which happens once the target directory has been deleted.
    """
    custom_scenery_dir = resolve_custom_scenery_dir(custom_scenery_dir)
    if not custom_scenery_dir or not os.path.isdir(custom_scenery_dir):
        return []
    removed = []
    for path in find_links_to(custom_scenery_dir, os.path.realpath(target)):
        if remove_link(path):
            removed.append(path)
    if link_name:
        link = os.path.join(custom_scenery_dir, link_name)
        if link not in removed and is_broken_link(link) and remove_link(link):
            removed.append(link)
    return removed


def remove_tile_link(custom_scenery_dir, build_dir, lat, lon, grouped=False):
    """Drop every Custom Scenery link pointing at this tile. Returns removed paths."""
    return remove_links_to(
        custom_scenery_dir, build_dir, link_name_for(build_dir, lat, lon, grouped)
    )


def is_tile_linked(custom_scenery_dir, build_dir):
    """True when some Custom Scenery link resolves to this build directory."""
    if not build_dir or not os.path.isdir(build_dir):
        return False
    custom_scenery_dir = resolve_custom_scenery_dir(custom_scenery_dir)
    if not custom_scenery_dir or not os.path.isdir(custom_scenery_dir):
        return False
    if _is_within(build_dir, custom_scenery_dir):
        return True
    return bool(find_links_to(custom_scenery_dir, os.path.realpath(build_dir)))


def ensure_overlay_link(custom_scenery_dir, overlay_dir=None):
    """Link the shared yOrtho4XP_Overlays directory into Custom Scenery."""
    if overlay_dir is None:
        overlay_dir = FNAMES.Overlay_dir
    return ensure_link(custom_scenery_dir, overlay_dir, OVERLAY_LINK_NAME)


def remove_overlay_link(custom_scenery_dir, overlay_dir=None):
    """Drop the Custom Scenery link(s) to yOrtho4XP_Overlays. Returns removed paths."""
    if overlay_dir is None:
        overlay_dir = FNAMES.Overlay_dir
    return remove_links_to(custom_scenery_dir, overlay_dir, OVERLAY_LINK_NAME)


################################################################################
# Automatic linking after a build
################################################################################
def auto_link_enabled():
    # Imported lazily: O4_Config_Utils imports O4_Tile_Utils, which imports us.
    import O4_Config_Utils as CFG

    return bool(getattr(CFG, "auto_link_custom_scenery", False)) and bool(
        getattr(CFG, "custom_scenery_dir", "")
    )


def tile_has_dsf(build_dir, lat, lon):
    """True once the tile owns an activated DSF, i.e. X-Plane can load it."""
    return os.path.isfile(
        os.path.join(build_dir, "Earth nav data", FNAMES.long_latlon(lat, lon) + ".dsf")
    )


def auto_link_tile(tile):
    """Link a freshly built tile into Custom Scenery, if the user asked for it."""
    try:
        if not auto_link_enabled():
            return None
        if not tile_has_dsf(tile.build_dir, tile.lat, tile.lon):
            UI.vprint(2, "No DSF built yet, not linking into Custom Scenery.")
            return None
        import O4_Config_Utils as CFG

        result = ensure_tile_link(
            CFG.custom_scenery_dir,
            tile.build_dir,
            tile.lat,
            tile.lon,
            getattr(tile, "grouped", False),
        )
    except Exception as e:
        UI.lvprint(1, "ERROR: automatic Custom Scenery link failed:", e)
        return None
    if result.status == CREATED:
        UI.lvprint(1, result.message)
    elif result.status in (DUPLICATE_TILE, NAME_TAKEN, FAILED):
        UI.lvprint(1, "WARNING: " + result.message)
    else:
        UI.vprint(2, result.message)
    return result
