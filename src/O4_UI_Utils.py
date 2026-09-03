import os
import subprocess
import sys
import threading
import time

import O4_File_Names as FNAMES

verbosity = 1
red_flag = False
paused = False
is_working = False
cleaning_level = 1
gui = None
log = True

################################################################################
# Registry of external worker subprocesses.
# Long build steps (Triangle4XP, DSFTool, the SFR .venv python) run as
# external processes which never see red_flag. They register here so that
# the GUI Stop button and window close can terminate them immediately
# instead of letting them run on in the background shell.
_active_subprocesses = set()
_subprocess_lock = threading.Lock()


def register_subprocess(proc):
    with _subprocess_lock:
        _active_subprocesses.add(proc)
    # A worker that got past its pause checkpoint before the pause landed can
    # still reach here; freeze it straight away rather than let it run on.
    if paused:
        _suspend_proc(proc)


def unregister_subprocess(proc):
    with _subprocess_lock:
        _active_subprocesses.discard(proc)


def kill_subprocess(proc):
    """Forcefully terminate ``proc`` and all of its children."""
    if proc.poll() is not None:
        return
    try:
        if sys.platform.startswith("win"):
            # /T kills the whole process tree (e.g. torch dataloader workers)
            subprocess.call(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        else:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    except Exception:
        pass


def kill_all_subprocesses():
    # A paused build has its external workers frozen; a frozen tree never gets
    # around to handling terminate(), so lift the pause before killing.
    _clear_pause()
    with _subprocess_lock:
        procs = list(_active_subprocesses)
    for proc in procs:
        kill_subprocess(proc)

################################################################################
# Pause / resume.
#
# `_resume_event` is set while work may proceed and cleared while the build is
# paused. Worker loops call check_pause() at the same boundaries where they
# test red_flag, so a pause lands between units of work rather than in the
# middle of one and nothing half-written is left behind.
#
# External workers (Triangle4XP, DSFTool, the SFR .venv python) never see the
# flag, so their process trees are suspended instead. That needs psutil, which
# rides along with ultralytics; without it a pause still stops the Python side
# and the external step simply finishes before the pause takes hold.
_resume_event = threading.Event()
_resume_event.set()
_pause_lock = threading.RLock()
_suspended_handles = []


def _psutil():
    try:
        import psutil
        return psutil
    except Exception:
        return None


def _process_tree(proc):
    """psutil handles for ``proc`` and every child it spawned, deepest first."""
    ps = _psutil()
    if ps is None:
        return []
    try:
        parent = ps.Process(proc.pid)
    except Exception:
        return []
    try:
        children = parent.children(recursive=True)
    except Exception:
        children = []
    return children + [parent]


def _suspend_proc(proc):
    if proc.poll() is not None:
        return
    for handle in _process_tree(proc):
        try:
            handle.suspend()
        except Exception:
            continue
        with _pause_lock:
            _suspended_handles.append(handle)


def _suspend_registered():
    with _subprocess_lock:
        procs = list(_active_subprocesses)
    for proc in procs:
        _suspend_proc(proc)


def _resume_registered():
    with _pause_lock:
        handles = list(_suspended_handles)
        del _suspended_handles[:]
    # Resume parents before children so a child is never left running under a
    # frozen parent if one of the calls fails.
    for handle in reversed(handles):
        try:
            handle.resume()
        except Exception:
            pass


def _clear_pause():
    """Drop the pause without logging - used by Stop and by window close."""
    global paused
    with _pause_lock:
        if not paused:
            return
        paused = False
        _resume_registered()
        _resume_event.set()


def request_pause():
    """Suspend the running build. Returns False if it was already paused."""
    global paused
    with _pause_lock:
        if paused:
            return False
        paused = True
        _resume_event.clear()
        _suspend_registered()
    lvprint(0, "Build paused - press Resume to carry on.")
    return True


def request_resume():
    """Let a paused build carry on. Returns False if it was not paused."""
    global paused
    with _pause_lock:
        if not paused:
            return False
        paused = False
        _resume_registered()
        _resume_event.set()
    lvprint(0, "Build resumed.")
    return True


def toggle_pause():
    return request_resume() if paused else request_pause()


def is_paused():
    return paused


def check_pause():
    """Park the calling worker while the build is paused.

    Returns as soon as the build is resumed, or immediately if Stop was hit
    in the meantime - the caller's own red_flag test then aborts it.
    """
    if _resume_event.is_set():
        return
    while not _resume_event.wait(0.25):
        if red_flag:
            return


def stop_requested():
    """check_pause() then report whether the build should abort."""
    check_pause()
    return bool(red_flag)

################################################################################
def progress_bar(nbr, percentage, message=None):
    if gui:
        gui.pgrbv[nbr].set(percentage)


################################################################################
def vprint(min_verbosity, *args):
    if verbosity >= min_verbosity:
        print(*args)


################################################################################
def logprint(*args):
    try:
        f = open(FNAMES.resource_path("Ortho4XP.log"), "a")
        f.write(
            time.strftime("%c")
            + " | "
            + " ".join([str(x) for x in args])
            + "\n"
        )
        f.close()
    except:
        pass


################################################################################
def lvprint(min_verbosity, *args):
    if verbosity >= min_verbosity:
        print(*args)
    if log:
        logprint(*args)


################################################################################
def bug_report(*args):
    logprint(
        "An internal error occured. Please file a bug with lat/lon and cfg"
    )
    if args:
        logprint(*args)


################################################################################
def exit_message_and_bottom_line(*args):
    global is_working
    if not args:
        args = ("Process interrupted",)
    if args[0]:
        logprint(*args)
        print(*args)
    print(
        "_____________________________________________________________"
        + "____________________________________"
    )
    is_working = False


################################################################################
def timings_and_bottom_line(tinit):
    global is_working
    print("\nCompleted in " + nicer_timer(time.time() - tinit) + ".")
    print(
        "_____________________________________________________________"
        + "____________________________________"
    )
    is_working = False


################################################################################
def human_print(num, suffix=""):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1024.0:
            return "{:.1f}{}{}".format(num, unit, suffix)
        num /= 1024.0
    return "{:.1f}{}{}".format(num, "Y", suffix)


################################################################################
def nicer_timer(elapsed):
    out_string = ""
    hours = elapsed // 3600
    if hours:
        elapsed -= 3600 * hours
        out_string += str(int(hours)) + "h"
    minutes = elapsed // 60
    if hours or minutes:
        elapsed -= 60 * minutes
        out_string += str(int(minutes)) + "m"
    elapsed = "{:.2f}".format(elapsed) if not out_string else int(elapsed)
    out_string += str(elapsed) + "sec"
    return out_string
