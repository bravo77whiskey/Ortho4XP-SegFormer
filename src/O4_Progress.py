"""Live progress of tile builds: what is running, how far it got, what went wrong.

A batch overlaps its work - one tile's overlays run while the next tile's
mesh and DSF are built, and Step 3 downloads and converts while it writes the
DSF - so a single scrolling log and three anonymous bars cannot say where a
build stands. This module keeps that state as data instead: a run (a batch
or a single step started from the main window) holds its tiles, each tile its
steps, each step its own counters, sub-parts and issues. The GUI draws it;
nothing here knows about Tk.

Code reports into it from the thread doing the work. A thread carries the
tile and step it works for (see step(), scope() and bind()), so a progress
bar update or a log line lands on the right row without being told where.

An issue is one item of a step that went wrong while the step as a whole
still produced its output - a texture filled with white, a texture the
building overlay had to skip. A step that ends with issues is "done with
issues", not failed.

The overlay and colour-correction workers run as separate processes. They
report through their stdout with emit(), as lines that handle_line() picks
out of the stream before it reaches the console.
"""

import json
import os
import sys
import threading
import time
from contextlib import contextmanager

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
ISSUES = "issues"
FAILED = "failed"
SKIPPED = "skipped"
STOPPED = "stopped"

# Short names for the tree rows and the log prefixes.
STEP_NAMES = {
    "osm": "OSM",
    "mesh": "Mesh",
    "mask": "Mask",
    "dsf": "DSF",
    "color": "Colour",
    "clouds": "Clouds",
    "ovl": "Overlays",
    "bld_overlay": "BLD",
    "veg_overlay": "VEG",
}

# What UI.progress_bar(nbr, ...) measures inside each step. Bar 1 is the
# step's own progress except in Step 3, where it is the DSF being written
# alongside the downloads (bar 2) and the conversions (bar 3).
_BAR_PARTS = {
    "dsf": {1: "Mesh → DSF", 2: "Download", 3: "Convert"},
}
_DEFAULT_BAR_PARTS = {2: "Download", 3: "Convert"}

EVENT_PREFIX = "@@O4P "
# Set for the worker processes whose stdout handle_line() reads; without it
# emit() stays quiet, so a worker run by hand prints no event lines.
EVENT_ENV = "O4_PROGRESS_EVENTS"

_lock = threading.RLock()
_tls = threading.local()
_run = None
_version = 0
# Progress reported by threads working for no tile (the Custom ZL tools).
_loose = {}

# Set by O4_UI_Utils: whether the user stopped the build.
is_stopped = lambda: False  # noqa: E731


################################################################################
def tile_name(lat, lon):
    return f"{'+' if lat >= 0 else '-'}{abs(int(lat)):02d}{'+' if lon >= 0 else '-'}{abs(int(lon)):03d}"


class Issue:
    __slots__ = ("item", "stage", "message", "t")

    def __init__(self, item, stage, message):
        self.item = item
        self.stage = stage
        self.message = message
        self.t = time.time()


class Part:
    """A measurable piece of a step: Step 3's downloads, one colour group."""

    __slots__ = ("label", "done", "total", "pct")

    def __init__(self, label):
        self.label = label
        self.done = None
        self.total = None
        self.pct = None


class Step:
    def __init__(self, key):
        self.key = key
        self.state = QUEUED
        self.t0 = None
        self.t1 = None
        self.done = None
        self.total = None
        self.pct = None
        self.note = ""
        self.parts = {}
        self.issues = []
        # Started by ahead() inside an earlier step: step() carries on from it.
        self.ahead = False

    def elapsed(self, now):
        if self.t0 is None:
            return None
        return (self.t1 or now) - self.t0


class Tile:
    def __init__(self, lat, lon):
        self.lat = int(lat)
        self.lon = int(lon)
        self.name = tile_name(lat, lon)
        self.steps = {}
        self.skipped = False

    def step(self, key):
        node = self.steps.get(key)
        if node is None:
            node = self.steps[key] = Step(key)
        return node

    def state(self):
        states = [s.state for s in self.steps.values()]
        if self.skipped:
            return SKIPPED
        if not states or all(s == QUEUED for s in states):
            return QUEUED
        if RUNNING in states:
            return RUNNING
        if FAILED in states:
            return FAILED
        if STOPPED in states:
            return STOPPED
        if QUEUED in states:
            # Between two steps, or its overlays wait for the previous tile's.
            return RUNNING
        if ISSUES in states:
            return ISSUES
        return DONE

    def times(self, now):
        starts = [s.t0 for s in self.steps.values() if s.t0 is not None]
        if not starts:
            return None
        if any(s.state == RUNNING for s in self.steps.values()):
            end = now
        else:
            end = max((s.t1 for s in self.steps.values() if s.t1 is not None), default=now)
        return end - min(starts)


class Run:
    def __init__(self, title, tiles, steps, launch, auto, tile_steps=None):
        self.title = title
        self.t0 = time.time()
        self.t1 = None
        self.state = RUNNING
        self.steps = list(steps)
        self.tiles = {}
        for lat, lon in tiles:
            tile = Tile(lat, lon)
            # An airport batch builds each tile its own steps: show only those.
            wanted = (
                self.steps if tile_steps is None
                else [key for key in self.steps
                      if key in tile_steps.get((int(lat), int(lon)), ())]
            )
            for key in wanted:
                tile.step(key)
            self.tiles[tile.name] = tile
        # What the GUI needs to launch a retry of this run's issues.
        self.launch = dict(launch or {})
        # Opened by a step that ran outside any run: it ends with that step.
        self.auto = auto

    def tile(self, lat, lon):
        name = tile_name(lat, lon)
        node = self.tiles.get(name)
        if node is None:
            node = self.tiles[name] = Tile(lat, lon)
        return node


def _changed():
    global _version
    _version += 1


################################################################################
# Runs.

def begin_run(tiles, steps, title="Batch", launch=None, tile_steps=None):
    """Start showing a new run; the previous one is dropped from the panel.

    tile_steps: {(lat, lon): steps} when each tile builds only its own steps.
    """
    global _run
    with _lock:
        _run = Run(title, tiles, steps, launch, auto=False, tile_steps=tile_steps)
        _loose.clear()
        _changed()


def end_run():
    """The run is over and its clock stops."""
    with _lock:
        if _run is None or _run.t1 is not None:
            return
        _run.t1 = time.time()
        for tile in _run.tiles.values():
            for step in tile.steps.values():
                if step.state == RUNNING:
                    step.state = STOPPED
                    step.t1 = _run.t1
        _run.state = STOPPED if is_stopped() else DONE
        _changed()


def tile_skipped(lat, lon):
    """A resumed batch found every step of this tile already built."""
    with _lock:
        if _run is None:
            return
        tile = _run.tile(lat, lon)
        tile.skipped = True
        for step in tile.steps.values():
            step.state = SKIPPED
        _changed()


def step_skipped(lat, lon, key):
    """This step of the tile has nothing to do in this run."""
    with _lock:
        if _run is None:
            return
        node = _run.tile(lat, lon).steps.get(key)
        if node is not None and node.state == QUEUED:
            node.state = SKIPPED
            _changed()


class _StepScope:
    def __init__(self):
        self.ok = True

    def result(self, ok):
        """The step's own verdict: falsy means it did not produce its output."""
        self.ok = bool(ok)
        return ok


@contextmanager
def step(lat, lon, key):
    """Run one step of a tile: mark it running, and done/failed when it ends.

    The calling thread works for this tile and step until the block ends.
    Started outside any run (a step button with nothing else running), it
    opens a run of its own that ends with it.
    """
    global _run
    with _lock:
        if _run is None or _run.t1 is not None:
            _run = Run(
                f"{STEP_NAMES.get(key, key)} {tile_name(lat, lon)}",
                [(lat, lon)], [key], None, auto=True,
            )
            _loose.clear()
        node = _run.tile(lat, lon).step(key)
        if not (node.ahead and node.state == RUNNING):
            node.state = RUNNING
            node.t0 = time.time()
            node.t1 = None
            node.done = node.total = node.pct = None
            node.note = ""
            node.parts = {}
            node.issues = []
        node.ahead = False
        run = _run
        _changed()
    scope = _StepScope()
    previous = _context()
    _set_context(tile_name(lat, lon), key)
    failed = False
    try:
        yield scope
    except BaseException:
        failed = True
        raise
    finally:
        _set_context(*previous)
        with _lock:
            node.t1 = time.time()
            if is_stopped():
                node.state = STOPPED
            elif failed or not scope.ok:
                node.state = FAILED
            elif node.issues:
                node.state = ISSUES
            else:
                node.state = DONE
            _changed()
            if run.auto and run is _run:
                end_run()


@contextmanager
def ahead(lat, lon, keys):
    """Do the work of later steps ``keys`` inside the step running now.

    Step 3 colour-corrects inline when the batch's Colour and Clouds steps
    follow it. Those steps start here, so their clock runs while the work
    does, and the calling thread reports to the first of them for the block.
    They stay running afterwards: step() for them carries on from here and
    ends them. Only steps this run still has queued for the tile take part.
    """
    started = []
    with _lock:
        if _run is not None and _run.t1 is None:
            tile = _run.tiles.get(tile_name(lat, lon))
            for key in keys:
                node = None if tile is None else tile.steps.get(key)
                if node is None or node.state != QUEUED:
                    continue
                node.state = RUNNING
                node.t0 = time.time()
                node.t1 = None
                node.ahead = True
                started.append(key)
            if started:
                _changed()
    if not started:
        yield
        return
    with scope(tile_name(lat, lon), started[0]):
        yield


################################################################################
# Which tile and step the calling thread works for.

def _context():
    return getattr(_tls, "tile", None), getattr(_tls, "step", None)


def _set_context(tile, key):
    _tls.tile = tile
    _tls.step = key


def current():
    """(tile name, step key) the calling thread works for, or (None, None)."""
    return _context()


@contextmanager
def scope(tile, key):
    """Have the calling thread work for ``tile``/``key`` without starting a step."""
    previous = _context()
    _set_context(tile, key)
    try:
        yield
    finally:
        _set_context(*previous)


def bind(target):
    """``target`` wrapped to run for the calling thread's tile and step.

    Threads do not inherit it: hand this to threading.Thread instead of the
    bare target so a worker's bars and log lines land where its parent's do.
    """
    tile, key = _context()

    def run(*args, **kwargs):
        with scope(tile, key):
            return target(*args, **kwargs)

    return run


def _node():
    tile, key = _context()
    if _run is None or tile is None or key is None:
        return None
    node = _run.tiles.get(tile)
    return None if node is None else node.steps.get(key)


################################################################################
# Reports from the thread doing the work.

def bar(nbr, pct):
    """UI.progress_bar(nbr, pct), routed to the calling thread's step."""
    with _lock:
        node = _node()
        if node is None:
            _loose[nbr] = pct
            _changed()
            return
        label = _BAR_PARTS.get(node.key, _DEFAULT_BAR_PARTS).get(nbr)
        if label is None:
            node.pct = pct
        else:
            part = node.parts.get(label)
            if part is None:
                part = node.parts[label] = Part(label)
            part.pct = pct
        _changed()


def items(done=None, total=None, add=None, part=None, note=None):
    """Count-based progress of the calling thread's step, or of one of its parts."""
    with _lock:
        node = _node()
        if node is None:
            return
        target = node
        if part:
            target = node.parts.get(part)
            if target is None:
                target = node.parts[part] = Part(part)
        if total is not None:
            target.total = int(total)
        if done is not None:
            target.done = int(done)
        if add:
            target.done = (target.done or 0) + int(add)
            if target.total is not None:
                target.done = min(target.done, target.total)
        if note is not None and target is node:
            node.note = note
        _changed()


def issue(item, stage, message, lat=None, lon=None, key=None):
    """Record one item of a step that went wrong. Defaults to the calling thread's step."""
    with _lock:
        if _run is None:
            return
        if lat is not None and key is not None:
            node = _run.tile(lat, lon).step(key)
        else:
            node = _node()
        if node is None:
            return
        node.issues.append(Issue(str(item), str(stage), str(message)))
        _changed()


################################################################################
# Worker processes.

def emit(**event):
    """Report from a worker process; handle_line() in the parent applies it."""
    if os.environ.get(EVENT_ENV) != "1":
        return
    try:
        sys.stdout.write(EVENT_PREFIX + json.dumps(event, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


def report_issue(item, stage, message):
    """A worker's issue: a log line for people, an event for the panel."""
    print(f"  WARNING: {item}: {stage}: {message}", flush=True)
    emit(ev="issue", item=item, stage=stage, msg=str(message))


def handle_line(line):
    """Apply an event line from a worker; True when ``line`` was one."""
    if not line.startswith(EVENT_PREFIX):
        return False
    try:
        event = json.loads(line[len(EVENT_PREFIX):])
    except ValueError:
        return True
    kind = event.get("ev")
    if kind == "items":
        items(
            done=event.get("done"), total=event.get("total"), add=event.get("add"),
            part=event.get("part"), note=event.get("note"),
        )
    elif kind == "issue":
        issue(event.get("item", "?"), event.get("stage", "?"), event.get("msg", ""))
    return True


################################################################################
# Readers.

def version():
    return _version


def snapshot():
    """The run as plain data for the GUI, with elapsed times as of now."""
    with _lock:
        now = time.time()
        loose = {nbr: pct for nbr, pct in _loose.items() if 0 < pct < 100}
        if _run is None:
            return {"run": None, "loose": loose}
        tiles = []
        for tile in _run.tiles.values():
            steps = []
            for node in tile.steps.values():
                steps.append({
                    "key": node.key,
                    "name": STEP_NAMES.get(node.key, node.key),
                    "state": node.state,
                    "elapsed": node.elapsed(now),
                    "done": node.done,
                    "total": node.total,
                    "pct": node.pct,
                    "note": node.note,
                    "parts": [
                        {"label": p.label, "done": p.done, "total": p.total, "pct": p.pct}
                        for p in node.parts.values()
                    ],
                    "issues": [(i.item, i.stage, i.message) for i in node.issues],
                })
            tiles.append({
                "name": tile.name, "lat": tile.lat, "lon": tile.lon,
                "state": tile.state(), "elapsed": tile.times(now), "steps": steps,
            })
        return {
            "run": {
                "started": _run.t0,
                "title": _run.title,
                "state": _run.state,
                "elapsed": (_run.t1 or now) - _run.t0,
                "tiles": tiles,
                "launch": dict(_run.launch),
            },
            "loose": loose,
        }


def issues_by_tile():
    """{(lat, lon): {step key: [(item, stage, message), ...]}} for this run."""
    with _lock:
        out = {}
        if _run is None:
            return out
        for tile in _run.tiles.values():
            for node in tile.steps.values():
                if node.issues:
                    out.setdefault((tile.lat, tile.lon), {})[node.key] = [
                        (i.item, i.stage, i.message) for i in node.issues
                    ]
        return out


def summary_lines():
    """The run's issues and failures, grouped by tile, for the end of the log."""
    with _lock:
        if _run is None:
            return []
        lines = []
        for tile in _run.tiles.values():
            rows = []
            for node in tile.steps.values():
                name = STEP_NAMES.get(node.key, node.key)
                if node.state == FAILED:
                    rows.append(f"   {name}: failed")
                for i in node.issues:
                    rows.append(f"   {name}: {i.item}: {i.stage}: {i.message}")
            if rows:
                lines.append(tile.name)
                lines.extend(rows)
        return lines


def format_elapsed(seconds):
    if seconds is None:
        return ""
    seconds = int(seconds)
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"
