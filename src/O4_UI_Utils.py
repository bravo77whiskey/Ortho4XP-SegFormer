import os
import subprocess
import sys
import threading
import time

import O4_File_Names as FNAMES

verbosity = 1
red_flag = False
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
    with _subprocess_lock:
        procs = list(_active_subprocesses)
    for proc in procs:
        kill_subprocess(proc)

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
