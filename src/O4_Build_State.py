"""Persistent journal for batch tile builds.

A batch over a few dozen tiles runs for many hours, so it has to survive a
Stop, a crash, or a reboot.  Every step that finishes is written to a small
JSON journal next to the other user data; a later run reads it back and skips
the work that is already on disk instead of starting from the top.

The journal records *steps*, not tiles: a batch stopped after the mesh of the
seventh tile resumes at that tile's water masks, not at its vector data.

Only one batch is journalled at a time - a second batch overwrites the first,
which is why the GUI asks before discarding a resumable one.
"""

import json
import os
import threading
import time

import O4_File_Names as FNAMES
import O4_UI_Utils as UI

VERSION = 1

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

_lock = threading.RLock()
_state = None


################################################################################
def state_file():
    return FNAMES.user_path(".batch_build_state.json")


################################################################################
def _key(lat, lon):
    return FNAMES.short_latlon(lat, lon)


def _blank(list_lat_lon, steps, custom_build_dir, override_cfg):
    return {
        "version": VERSION,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "running",
        "custom_build_dir": custom_build_dir or "",
        "override_cfg": bool(override_cfg),
        "steps": [step for step in STEPS if steps.get(step)],
        "tiles": [[int(lat), int(lon)] for (lat, lon) in list_lat_lon],
        "progress": {},
    }


def _write_locked():
    """Dump the journal atomically so a kill mid-write cannot corrupt it."""
    if _state is None:
        return
    _state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path = state_file()
    tmp_path = path + ".tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(_state, f, indent=1)
        os.replace(tmp_path, path)
    except Exception as exc:
        UI.vprint(2, "Could not write the build journal:", exc)


################################################################################
def load():
    """Return the journal on disk, or None when there is nothing usable."""
    try:
        with open(state_file(), encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        return None
    if not isinstance(state, dict) or state.get("version") != VERSION:
        return None
    if not state.get("tiles") or not state.get("steps"):
        return None
    return state


def resumable():
    """Return the journal only if it still has work left to do."""
    state = load()
    if state is None or state.get("status") == "completed":
        return None
    if not pending_tiles(state):
        return None
    return state


def pending_tiles(state):
    """Tiles in ``state`` with at least one step still to run."""
    progress = state.get("progress", {})
    steps = state.get("steps", [])
    pending = []
    for lat, lon in state.get("tiles", []):
        done = set(progress.get(_key(lat, lon), []))
        if [step for step in steps if step not in done]:
            pending.append((lat, lon))
    return pending


def describe(state):
    """One short paragraph about a journal, for the resume dialog."""
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
          resume=False):
    """Open the journal for a batch, keeping earlier progress when resuming."""
    global _state
    with _lock:
        state = load() if resume else None
        if state is not None and not matches(
            state, list_lat_lon, steps, custom_build_dir, override_cfg
        ):
            # The journal describes a different batch - resuming against it
            # would skip steps that were never run for these tiles.
            UI.vprint(
                1,
                "Build journal does not match this batch; starting it afresh.",
            )
            state = None
        if state is None:
            state = _blank(list_lat_lon, steps, custom_build_dir, override_cfg)
        else:
            state["status"] = "running"
        _state = state
        _write_locked()
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
    """Mark the batch finished; the journal stops being offered for resume."""
    set_status("completed")


def close():
    """Forget the in-memory journal, leaving the file alone."""
    global _state
    with _lock:
        _state = None


def discard():
    """Throw the journal away - the user chose to start over."""
    global _state
    with _lock:
        _state = None
        try:
            os.remove(state_file())
        except FileNotFoundError:
            pass
        except Exception as exc:
            UI.vprint(2, "Could not remove the build journal:", exc)
