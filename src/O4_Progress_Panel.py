"""The main window's progress tree and the log under it.

The tree shows the run O4_Progress keeps: the run itself, its tiles, each
tile's steps, and under a step its parts (Step 3's downloads and
conversions) and its issues. Rows open while they run and fold away when
they are done, until the user opens or closes one by hand, which the panel
then leaves alone.

The log keeps every line with the tile and step that printed it. Selecting
a row shows only that row's lines; selecting the run row shows them all.
Right-click offers to retry what failed or has issues.
"""

import collections
import re
import time
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import END, N, S, E, W, VERTICAL

import O4_Progress as PROG
import O4_GUI_Theme as THEME

STATE_MARKS = {
    PROG.QUEUED: "·",
    PROG.RUNNING: "▶",
    PROG.DONE: "✓",
    PROG.ISSUES: "⚠",
    PROG.FAILED: "✗",
    PROG.SKIPPED: "–",
    PROG.STOPPED: "■",
}
RETRYABLE = (PROG.FAILED, PROG.ISSUES, PROG.STOPPED)

COLORS = {
    "light": {
        PROG.RUNNING: "#0b4fa8", PROG.ISSUES: "#a85d00", PROG.FAILED: "#b00020",
        PROG.STOPPED: "#a85d00", PROG.QUEUED: "#6b6b6b", PROG.SKIPPED: "#6b6b6b",
        PROG.DONE: "#1d6b2a", "dim": "#7a7a7a", "warn": "#a85d00", "error": "#b00020",
    },
    "dark": {
        PROG.RUNNING: "#8cc4ff", PROG.ISSUES: "#f0b45a", PROG.FAILED: "#ff7a7a",
        PROG.STOPPED: "#f0b45a", PROG.QUEUED: "#8a958a", PROG.SKIPPED: "#8a958a",
        PROG.DONE: "#9ad89a", "dim": "#7f8a7f", "warn": "#f0b45a", "error": "#ff7a7a",
    },
}

# Whole words only: "RuntimeError: ..." in an issue is a warning, not an error line.
_ERROR_RE = re.compile(r"\bERROR\b|^\s*Traceback", re.IGNORECASE)
_WARN_RE = re.compile(r"\bWARNING\b|\bfailed\b", re.IGNORECASE)

BAR_WIDTH = 16
LOG_KEEP = 200000        # lines kept for filtering
LOG_SHOWN_MAX = 60000    # lines in the Text widget before the oldest go


def text_bar(fraction):
    fraction = max(0.0, min(1.0, fraction))
    full = int(round(fraction * BAR_WIDTH))
    return "█" * full + "░" * (BAR_WIDTH - full)


def _measure(done, total, pct):
    """'███░░ 12/64' from counts, '███░░ 71%' from a percentage, '' from nothing."""
    if total:
        done = done or 0
        return f"{text_bar(done / total)} {done}/{total}"
    if pct is not None:
        return f"{text_bar(pct / 100.0)} {int(pct)}%"
    return ""


def log_prefix(tile, step):
    if tile is None:
        return ""
    name = PROG.STEP_NAMES.get(step, step) if step else ""
    return f"{tile} {name:<6}│ "


class ProgressPanel:
    """Tree on top, log below, in a vertical paned window placed by the caller."""

    def __init__(self, parent, on_retry):
        self.on_retry = on_retry
        self.colors = COLORS.get(THEME.ui_theme, COLORS["light"])
        palette = THEME.palette()

        self.paned = tk.PanedWindow(
            parent, orient=VERTICAL, sashwidth=6, bd=0,
            bg=palette["background"], sashrelief="flat",
        )

        # ── Tree ────────────────────────────────────────────────────────────
        tree_frame = tk.Frame(self.paned, **THEME.frame_options())
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        style = ttk.Style()
        style.configure(
            "Progress.Treeview",
            font="TkFixedFont",
            rowheight=20,
            background=palette["entry_background"],
            fieldbackground=palette["entry_background"],
            foreground=palette["foreground"],
        )
        style.map(
            "Progress.Treeview",
            background=[("selected", palette["header_background"])],
            foreground=[("selected", palette["header_foreground"])],
        )
        style.configure(
            "Progress.Treeview.Heading",
            background=palette["button_background"],
            foreground=palette["button_foreground"],
        )
        self.tree = ttk.Treeview(
            tree_frame, columns=("progress", "time"), style="Progress.Treeview",
            selectmode="browse", height=8,
        )
        self.tree.heading("#0", text="Build", anchor=W)
        self.tree.heading("progress", text="Progress", anchor=W)
        self.tree.heading("time", text="Elapsed", anchor=E)
        self.tree.column("#0", width=290, minwidth=150, stretch=False)
        self.tree.column("progress", width=520, minwidth=200, stretch=True)
        self.tree.column("time", width=80, minwidth=60, stretch=False, anchor=E)
        for state in STATE_MARKS:
            self.tree.tag_configure(state, foreground=self.colors[state])
        self.tree.tag_configure("note", foreground=self.colors["dim"])
        tree_scroll = ttk.Scrollbar(tree_frame, orient=VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.grid(row=0, column=0, sticky=N + S + E + W)
        tree_scroll.grid(row=0, column=1, sticky=N + S)
        self.paned.add(tree_frame, minsize=60, height=230, stretch="always")

        # ── Log ─────────────────────────────────────────────────────────────
        log_frame = tk.Frame(self.paned, **THEME.frame_options())
        log_frame.rowconfigure(1, weight=1)
        log_frame.columnconfigure(0, weight=1)
        bar = tk.Frame(log_frame, **THEME.frame_options())
        bar.grid(row=0, column=0, columnspan=2, sticky=E + W)
        self.filter_label = tk.Label(bar, anchor=W, text="Log: everything", **THEME.label_options())
        self.filter_label.pack(side="left", padx=2)
        self.show_all_button = ttk.Button(
            bar, text="Show all", takefocus=False, style="Flat.TButton",
            command=self.show_all,
        )
        self.console = tk.Text(log_frame, bd=0, **THEME.text_options())
        self.console.tag_configure("dim", foreground=self.colors["dim"])
        self.console.tag_configure("warn", foreground=self.colors["warn"])
        self.console.tag_configure("error", foreground=self.colors["error"])
        log_scroll = ttk.Scrollbar(log_frame, orient=VERTICAL, command=self.console.yview)
        self.console.configure(yscrollcommand=log_scroll.set)
        self.console.grid(row=1, column=0, sticky=N + S + E + W)
        log_scroll.grid(row=1, column=1, sticky=N + S)
        self.paned.add(log_frame, minsize=80, stretch="always")

        self.lines = collections.deque(maxlen=LOG_KEEP)
        self.filter = None          # None, (tile, None) or (tile, step)
        self.shown = 0

        self.rows = {}              # iid -> (text, values, tags) as last drawn
        self.user_set = set()       # iids opened or closed by hand
        self.run_id = None
        self.last_version = -1
        self.last_second = -1
        self.snap = None
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<<TreeviewOpen>>", self._on_toggle)
        self.tree.bind("<<TreeviewClose>>", self._on_toggle)
        self.tree.bind("<Button-3>", self._on_right_click)
        self.menu = tk.Menu(self.tree, tearoff=0)

    # ── Log ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _line_tag(text):
        if _ERROR_RE.search(text):
            return "error"
        if _WARN_RE.search(text):
            return "warn"
        return None

    def _matches(self, tile, step):
        if self.filter is None:
            return True
        ftile, fstep = self.filter
        if tile != ftile:
            return False
        return fstep is None or step == fstep

    def _insert(self, tile, step, text):
        at_end = self.console.yview()[1] >= 0.999
        prefix = log_prefix(tile, step) if text.strip() else ""
        if prefix:
            self.console.insert(END, prefix, "dim")
        self.console.insert(END, text, self._line_tag(text))
        self.shown += 1
        if self.shown > LOG_SHOWN_MAX:
            drop = self.shown - LOG_SHOWN_MAX + LOG_SHOWN_MAX // 10
            self.console.delete("1.0", f"{drop + 1}.0")
            self.shown -= drop
        return at_end

    def add_lines(self, entries):
        """Append (tile, step, text) lines; text ends with a newline."""
        follow = False
        for tile, step, text in entries:
            self.lines.append((tile, step, text))
            if self._matches(tile, step):
                follow = self._insert(tile, step, text) or follow
        if follow:
            self.console.see(END)

    def clear_log(self):
        self.lines.clear()
        self.console.delete("1.0", END)
        self.shown = 0

    def _set_filter(self, new):
        if new == self.filter:
            return
        self.filter = new
        if new is None:
            self.filter_label.config(text="Log: everything")
            self.show_all_button.pack_forget()
        else:
            tile, step = new
            label = tile + (" " + PROG.STEP_NAMES.get(step, step) if step else "")
            self.filter_label.config(text=f"Log: {label} only")
            self.show_all_button.pack(side="left", padx=6)
        self.console.delete("1.0", END)
        self.shown = 0
        # Consecutive lines with the same tag go in as one insert.
        chunk, chunk_tag = [], None

        def flush():
            if chunk:
                self.console.insert(END, "".join(chunk), chunk_tag)
                chunk.clear()

        kept = [entry for entry in self.lines if self._matches(entry[0], entry[1])]
        for tile, step, text in kept[-LOG_SHOWN_MAX:]:
            for piece, tag in (
                (log_prefix(tile, step) if text.strip() else "", "dim"),
                (text, self._line_tag(text)),
            ):
                if not piece:
                    continue
                if tag != chunk_tag:
                    flush()
                    chunk_tag = tag
                chunk.append(piece)
            self.shown += 1
        flush()
        self.console.see(END)

    def show_all(self):
        self.tree.selection_remove(self.tree.selection())
        self._set_filter(None)

    def _on_select(self, _event=None):
        selection = self.tree.selection()
        if not selection:
            return
        iid = selection[0]
        kind, _, rest = iid.partition(":")
        if kind == "t":
            self._set_filter((rest, None))
        elif kind in ("s", "p", "i"):
            tile, key = rest.split(":")[:2]
            self._set_filter((tile, key))
        else:
            self._set_filter(None)

    # ── Tree ────────────────────────────────────────────────────────────────
    def _on_toggle(self, _event=None):
        iid = self.tree.focus()
        if iid:
            self.user_set.add(iid)

    def refresh(self):
        """Redraw when the run changed, and every second while it runs."""
        now_second = int(time.time())
        version = PROG.version()
        running = bool(self.snap and self.snap.get("run") and self.snap["run"]["state"] == PROG.RUNNING)
        if version == self.last_version and not (running and now_second != self.last_second):
            return
        self.last_version = version
        self.last_second = now_second
        self.snap = PROG.snapshot()
        self._draw(self.snap)

    def _draw(self, snap):
        run = snap.get("run")
        wanted = []   # (iid, parent, text, values, tags, open_by_default)
        if run is not None:
            if run["started"] != self.run_id:
                self._reset()
                self.run_id = run["started"]
            self._run_rows(run, wanted)
        for nbr, pct in sorted(snap.get("loose", {}).items()):
            wanted.append((f"loose:{nbr}", "", "Working", (_measure(None, None, pct), ""), (PROG.RUNNING,), False))
        self._sync(wanted)

    def _reset(self):
        self.tree.delete(*self.tree.get_children(""))
        self.rows.clear()
        self.user_set.clear()

    def _run_rows(self, run, wanted):
        tiles = run["tiles"]
        n = len(tiles)
        finished = sum(1 for t in tiles if t["state"] in (PROG.DONE, PROG.ISSUES, PROG.SKIPPED))
        running = sum(1 for t in tiles if t["state"] == PROG.RUNNING)
        with_issues = sum(1 for t in tiles if t["state"] == PROG.ISSUES)
        failed = sum(1 for t in tiles if t["state"] == PROG.FAILED)
        counts = [f"{finished}/{n} tile{'s' if n != 1 else ''}"]
        if running:
            counts.append(f"{running} running")
        if with_issues:
            counts.append(f"⚠ {with_issues} with issues")
        if failed:
            counts.append(f"✗ {failed} failed")
        if run["state"] == PROG.STOPPED:
            counts.append("stopped")
        elif run["state"] == PROG.DONE:
            counts.append("finished")
        state = run["state"] if run["state"] != PROG.DONE else (
            PROG.FAILED if failed else PROG.ISSUES if with_issues else PROG.DONE
        )
        wanted.append((
            "run", "", run["title"],
            (f"{text_bar(finished / n if n else 1)}  " + " · ".join(counts),
             PROG.format_elapsed(run["elapsed"])),
            (state,), True,
        ))
        for tile in tiles:
            self._tile_rows(tile, wanted)

    def _tile_rows(self, tile, wanted):
        tid = f"t:{tile['name']}"
        chips = " ".join(f"{s['name']}{STATE_MARKS[s['state']]}" for s in tile["steps"])
        issues = sum(len(s["issues"]) for s in tile["steps"])
        if issues:
            chips += f"   ⚠{issues}"
        mark = STATE_MARKS[tile["state"]]
        wanted.append((
            tid, "run", f"{mark} {tile['name']}",
            (chips, PROG.format_elapsed(tile["elapsed"])),
            (tile["state"],), tile["state"] == PROG.RUNNING,
        ))
        for step in tile["steps"]:
            self._step_rows(tile["name"], tid, step, wanted)

    def _step_rows(self, tile_name, parent, step, wanted):
        sid = f"s:{tile_name}:{step['key']}"
        state = step["state"]
        measure = _measure(step["done"], step["total"], step["pct"])
        if state == PROG.RUNNING:
            progress = measure or "running"
        elif state == PROG.QUEUED:
            progress = ""
        elif state == PROG.SKIPPED:
            progress = "done earlier"
        else:
            progress = {PROG.DONE: "done", PROG.ISSUES: "done with issues",
                        PROG.FAILED: "failed", PROG.STOPPED: "stopped"}[state]
            if step["total"]:
                progress += f"  ({step['done'] or 0}/{step['total']})"
        if step["issues"]:
            progress += f"   ⚠{len(step['issues'])}"
        if step["note"] and state == PROG.RUNNING:
            progress += f"   {step['note']}"
        wanted.append((
            sid, parent, f"{STATE_MARKS[state]} {step['name']}",
            (progress, PROG.format_elapsed(step["elapsed"])),
            (state,), state == PROG.RUNNING,
        ))
        if state == PROG.RUNNING:
            for part in step["parts"]:
                wanted.append((
                    f"p:{tile_name}:{step['key']}:{part['label']}", sid, part["label"],
                    (_measure(part["done"], part["total"], part["pct"]), ""), ("note",), False,
                ))
        for index, (item, stage, message) in enumerate(step["issues"]):
            wanted.append((
                f"i:{tile_name}:{step['key']}:{index}", sid, f"⚠ {item}",
                (f"{stage}: {message}", ""), (PROG.ISSUES,), False,
            ))

    def _sync(self, wanted):
        keep = {row[0] for row in wanted}
        for iid in list(self.rows):
            if iid not in keep and self.tree.exists(iid):
                self.tree.delete(iid)
        for iid in list(self.rows):
            if iid not in keep:
                self.rows.pop(iid, None)
                self.user_set.discard(iid)
        index_in = collections.Counter()
        for iid, parent, text, values, tags, opened in wanted:
            index = index_in[parent]
            index_in[parent] += 1
            drawn = (text, values, tags)
            if iid not in self.rows or not self.tree.exists(iid):
                self.tree.insert(parent, index, iid=iid, text=text, values=values, tags=tags)
            else:
                if self.rows[iid] != drawn:
                    self.tree.item(iid, text=text, values=values, tags=tags)
                if self.tree.parent(iid) != parent or self.tree.index(iid) != index:
                    self.tree.move(iid, parent, index)
            self.rows[iid] = drawn
            if iid not in self.user_set and self.tree.item(iid, "open") != opened:
                self.tree.item(iid, open=opened)

    # ── Retry ───────────────────────────────────────────────────────────────
    def _retry_targets(self, iid):
        """{(lat, lon): [step keys]} the row's retry would run, or {}."""
        run = (self.snap or {}).get("run")
        if run is None:
            return {}
        targets = {}
        kind, _, rest = iid.partition(":")
        parts = rest.split(":") if rest else []
        for tile in run["tiles"]:
            if kind in ("t", "s", "i") and tile["name"] != parts[0]:
                continue
            for step in tile["steps"]:
                if kind in ("s", "i") and step["key"] != parts[1]:
                    continue
                if kind == "i" or step["state"] in RETRYABLE:
                    targets.setdefault((tile["lat"], tile["lon"]), []).append(step["key"])
        return targets

    def _on_right_click(self, event):
        iid = self.tree.identify_row(event.y)
        if not iid or iid.startswith(("loose", "p:")):
            return
        self.tree.selection_set(iid)
        targets = self._retry_targets(iid)
        run = (self.snap or {}).get("run") or {}
        busy = run.get("state") == PROG.RUNNING
        n_steps = sum(len(v) for v in targets.values())
        self.menu.delete(0, END)
        if not targets:
            self.menu.add_command(label="Nothing to retry here", state="disabled")
        else:
            label = {
                "run": f"Retry every failed step and issue ({n_steps} step{'s' if n_steps != 1 else ''})",
                "t": f"Retry this tile's failed steps and issues ({n_steps})",
                "s": "Retry this step",
                "i": "Retry this step (redoes this item)",
            }[iid.partition(":")[0]]
            self.menu.add_command(
                label=label + ("  - wait for the build to end" if busy else ""),
                state="disabled" if busy else "normal",
                command=lambda: self.on_retry(targets, dict(run.get("launch") or {})),
            )
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()
