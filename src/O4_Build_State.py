"""Per-run journals for batch tile builds.

A batch over a few dozen tiles runs for many hours, so it has to survive a
Stop, a crash, or a reboot.  Every step that finishes is written to a small
JSON journal; a later run reads it back and skips the work that is already on
disk instead of starting from the top.  The journal records *steps*, not
tiles: a batch stopped after the mesh of the seventh tile resumes at that
tile's water masks, not at its vector data.

Several copies of Ortho4XP are commonly run side by side on different tiles,
so each run owns its own journal under ``build_state/`` rather than sharing
one file.  A journal names the process that owns it, and one whose owner is
still running is never offered for resume - otherwise a second instance would
cheerfully start rebuilding the tiles the first one is part-way through.

Ownership is established by the filename, which is handed over with a single
atomic rename, so two instances racing for the same abandoned journal cannot
both win.
"""

import json
import os
import threading
import time
import uuid

import O4_File_Names as FNAMES
import O4_UI_Utils as UI

VERSION = 2

# Ordered as build_tile_list runs them.
STEPS = (
    "osm",
    "mesh",
    "mask",
    "dsf",
    "ovl",
    "sfr_bld",
    "sfr_veg",
)

STEP_LABELS = {
    "osm": "Assemble vector data",
    "mesh": "Triangulate 3D mesh",
    "mask": "Draw water masks",
    "dsf": "Build imagery/DSF",
    "ovl": "Extract overlays",
    "sfr_bld": "SegFormer Bld",
    "sfr_veg": "SegFormer Veg",
}

# The owner touches its journal on this cadence. Only consulted when psutil is
# missing and the owning process cannot be interrogated directly.
HEARTBEAT_S = 30
STALE_AFTER_S = 150

_PREFIX = "batch-"
_SUFFIX = ".json"
# Single-file journal written by the first version of this feature.
_LEGACY_FILE = ".batch_build_state.json"

_lock = threading.RLock()
_state = None
_state_path = None
_heartbeat_stop = None


################################################################################
def state_dir():
    return FNAMES.user_path("build_state")


def journal_path():
    """The journal this process owns, or None when no batch is running."""
    return _state_path


################################################################################
# Ownership.
#
# psutil answers "is that process still there" exactly, and the recorded
# creation time rules out a recycled pid. Without it we fall back to the
# owner's heartbeat, which is also what identifies a journal carried over from
# the single-file layout: those have no owner at all.

def _psutil():
    try:
        import psutil
        return psutil
    except Exception:
        return None


def _owner_block():
    owner = {"pid": os.getpid(), "created": None}
    ps = _psutil()
    if ps is not None:
        try:
            owner["created"] = ps.Process(os.getpid()).create_time()
        except Exception:
            pass
    return owner


def owner_is_alive(state, path):
    """True when a live process is still working through the journal.

    A run that ended - normally, on Stop, or on an error - releases its
    journal, so only one still marked "running" can have an owner. That covers
    the case of a batch stopped and resumed inside a single session, where the
    pid is very much alive but no longer building anything.
    """
    if state.get("status") != "running":
        return False
    owner = state.get("owner") or {}
    pid = owner.get("pid")
    created = owner.get("created")
    ps = _psutil()
    if ps is not None and pid:
        try:
            process = ps.Process(int(pid))
        except Exception:
            return False
        if created is None:
            return True
        try:
            return abs(process.create_time() - float(created)) < 1.0
        except Exception:
            return False
    try:
        return (time.time() - os.path.getmtime(path)) < STALE_AFTER_S
    except OSError:
        return False


def _heartbeat(path, stop):
    while not stop.wait(HEARTBEAT_S):
        try:
            os.utime(path, None)
        except OSError:
            return


def _start_heartbeat_locked():
    global _heartbeat_stop
    _heartbeat_stop = threading.Event()
    threading.Thread(
        target=_heartbeat,
        args=(_state_path, _heartbeat_stop),
        name="build-journal-heartbeat",
        daemon=True,
    ).start()


def _stop_heartbeat_locked():
    global _heartbeat_stop
    if _heartbeat_stop is not None:
        _heartbeat_stop.set()
        _heartbeat_stop = None


################################################################################
def _key(lat, lon):
    return FNAMES.short_latlon(lat, lon)


def _new_journal_path():
    return os.path.join(
        state_dir(),
        "%s%d-%s%s" % (_PREFIX, os.getpid(), uuid.uuid4().hex[:8], _SUFFIX),
    )


def _blank(list_lat_lon, steps, custom_build_dir, override_cfg):
    return {
        "version": VERSION,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "running",
        "owner": _owner_block(),
        "custom_build_dir": custom_build_dir or "",
        "override_cfg": bool(override_cfg),
        "steps": [step for step in STEPS if steps.get(step)],
        "tiles": [[int(lat), int(lon)] for (lat, lon) in list_lat_lon],
        "progress": {},
    }


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        return None
    if not isinstance(state, dict) or state.get("version") != VERSION:
        return None
    if not state.get("tiles") or not state.get("steps"):
        return None
    return state


def _write(path, state):
    """Dump a journal atomically so a kill mid-write cannot corrupt it."""
    state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp_path = path + ".tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=1)
        os.replace(tmp_path, path)
    except Exception as exc:
        UI.vprint(2, "Could not write the build journal:", exc)


def _write_locked():
    if _state is not None and _state_path:
        _write(_state_path, _state)


def _remove(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except Exception as exc:
        UI.vprint(2, "Could not remove the build journal:", exc)


################################################################################
def _migrate_legacy():
    """Move a single-file journal from the first version into the directory.

    It keeps no owner, so its heartbeat decides whether it is still live: an
    instance from before the upgrade that is still writing it stays hidden.
    """
    legacy = FNAMES.user_path(_LEGACY_FILE)
    if not os.path.isfile(legacy):
        return
    try:
        with open(legacy, encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        _remove(legacy)
        return
    if isinstance(state, dict) and state.get("tiles") and state.get("steps"):
        state["version"] = VERSION
        state.setdefault("owner", None)
        target = os.path.join(
            state_dir(), "%slegacy-%s%s" % (_PREFIX, uuid.uuid4().hex[:8], _SUFFIX)
        )
        _write(target, state)
    _remove(legacy)


def scan():
    """Every readable journal on disk as (path, state), newest batch first."""
    _migrate_legacy()
    entries = []
    try:
        names = os.listdir(state_dir())
    except OSError:
        return entries
    for name in names:
        if not name.startswith(_PREFIX) or not name.endswith(_SUFFIX):
            continue
        path = os.path.join(state_dir(), name)
        state = _read(path)
        if state is not None:
            entries.append((path, state))
    entries.sort(key=lambda item: item[1].get("started", ""), reverse=True)
    return entries


def resumable_all():
    """Journals with work left whose owning process is gone, newest first."""
    entries = []
    for path, state in scan():
        if _state_path and os.path.abspath(path) == os.path.abspath(_state_path):
            continue
        if state.get("status") == "completed":
            continue
        if not pending_tiles(state):
            continue
        if owner_is_alive(state, path):
            continue
        entries.append((path, state))
    return entries


def resumable():
    """The most recent resumable journal as (path, state), or None."""
    entries = resumable_all()
    return entries[0] if entries else None


def live_batches():
    """Journals another running instance is working on right now."""
    entries = []
    for path, state in scan():
        if _state_path and os.path.abspath(path) == os.path.abspath(_state_path):
            continue
        if state.get("status") == "completed":
            continue
        if owner_is_alive(state, path):
            entries.append((path, state))
    return entries


def busy_tiles():
    """Tiles another running instance still has to build."""
    tiles = set()
    for path, state in live_batches():
        tiles.update(pending_tiles(state))
    return tiles


def claim(path):
    """Take an abandoned journal over, returning its new path.

    The rename is the claim: only one instance can move a given file, so a
    second one racing for the same journal comes away with None rather than a
    duplicate build.
    """
    target = _new_journal_path()
    try:
        os.makedirs(state_dir(), exist_ok=True)
        os.replace(path, target)
    except OSError:
        return None
    state = _read(target)
    if state is None:
        _remove(target)
        return None
    state["owner"] = _owner_block()
    _write(target, state)
    return target


################################################################################
def pending_tiles(state):
    """Tiles in ``state`` with at least one step still to run."""
    progress = state.get("progress", {})
    steps = state.get("steps", [])
    pending = []
    for lat, lon in state.get("tiles", []):
        done = set(progress.get(_key(lat, lon), []))
        if [step for step in steps if step not in done]:
            pending.append((int(lat), int(lon)))
    return pending


def describe(state):
    """One short paragraph about a journal, for the resume dialogs."""
    tiles = state.get("tiles", [])
    pending = pending_tiles(state)
    labels = [STEP_LABELS.get(step, step) for step in state.get("steps", [])]
    lines = [
        "Batch started %s, last progress %s."
        % (state.get("started", "?"), state.get("updated", "?")),
        "%d of %d tile(s) still have work left." % (len(pending), len(tiles)),
        "Steps: " + (", ".join(labels) if labels else "none"),
    ]
    if state.get("custom_build_dir"):
        lines.append("Base folder: " + state["custom_build_dir"])
    return "\n".join(lines)


def summarize(state):
    """One line for a list of journals."""
    pending = pending_tiles(state)
    first = (
        FNAMES.short_latlon(*pending[0]) if pending else "-"
    )
    return "%s   %d tile(s) left (from %s)   %d step(s)" % (
        state.get("started", "?"),
        len(pending),
        first,
        len(state.get("steps", [])),
    )


def matches(state, list_lat_lon, steps, custom_build_dir, override_cfg):
    """True when a journal was written for exactly this batch."""
    if state is None:
        return False
    wanted = [step for step in STEPS if steps.get(step)]
    return (
        state.get("steps") == wanted
        and [tuple(t) for t in state.get("tiles", [])]
        == [(int(lat), int(lon)) for (lat, lon) in list_lat_lon]
        and (state.get("custom_build_dir") or "") == (custom_build_dir or "")
        and bool(state.get("override_cfg")) == bool(override_cfg)
    )


################################################################################
def begin(list_lat_lon, steps, custom_build_dir="", override_cfg=False,
          resume_path=None):
    """Open this run's journal, keeping earlier progress when resuming."""
    global _state, _state_path
    with _lock:
        _stop_heartbeat_locked()
        state = None
        path = None
        if resume_path:
            state = _read(resume_path)
            if state is not None and not matches(
                state, list_lat_lon, steps, custom_build_dir, override_cfg
            ):
                # The journal describes a different batch - resuming against it
                # would skip steps that were never run for these tiles.
                UI.vprint(
                    1,
                    "Build journal does not match this batch; starting it "
                    "afresh.",
                )
                state = None
            if state is not None:
                path = resume_path
        if state is None:
            state = _blank(list_lat_lon, steps, custom_build_dir, override_cfg)
            path = _new_journal_path()
        state["owner"] = _owner_block()
        state["status"] = "running"
        _state = state
        _state_path = path
        _write_locked()
        _start_heartbeat_locked()
    return _state


def is_done(lat, lon, step):
    with _lock:
        if _state is None:
            return False
        return step in _state.get("progress", {}).get(_key(lat, lon), [])


def mark_done(lat, lon, step):
    with _lock:
        if _state is None:
            return
        done = _state.setdefault("progress", {}).setdefault(_key(lat, lon), [])
        if step not in done:
            done.append(step)
            _write_locked()


def set_status(status):
    with _lock:
        if _state is None:
            return
        _state["status"] = status
        _write_locked()


def complete():
    """The batch finished - its journal has nothing left to say."""
    global _state, _state_path
    with _lock:
        _stop_heartbeat_locked()
        if _state_path:
            _remove(_state_path)
        _state = None
        _state_path = None


def release():
    """Give up this run's journal without deleting it.

    Called however a batch ends, so an unfinished journal stops looking like
    live work the moment nothing is building it any more.
    """
    global _state, _state_path
    with _lock:
        _stop_heartbeat_locked()
        if _state is not None and _state_path:
            if _state.get("status") == "running":
                _state["status"] = "stopped"
            _state["owner"] = None
            _write_locked()
        _state = None
        _state_path = None


def close():
    """Forget the in-memory journal, leaving the file on disk."""
    release()


def discard(path=None):
    """Throw a journal away - the user chose to start over."""
    global _state, _state_path
    with _lock:
        target = path or _state_path
        if target and _state_path and os.path.abspath(target) == os.path.abspath(
            _state_path
        ):
            _stop_heartbeat_locked()
            _state = None
            _state_path = None
        if target:
            _remove(target)
