"""Deploy the checked-out code to the live Ortho4XP install.

The one supported way to update the live app (default G:/Dev/Ortho4XP):

  1. Refuses while the live app or any process from its venv is running
     (a build in progress would pick up half-copied modules).
  2. Runs build.py (PyInstaller -> dist_build -> dist/Ortho4XP).
  3. Overlays dist/Ortho4XP onto the live folder: every _internal entry is
     replaced EXCEPT _internal/Ortho4XP_Data (user config, logs, caches and
     symlinks into another install), then the exe.  Nothing else in the live
     folder is deleted.
  4. Verifies every copied entry byte for byte (runtime __pycache__ ignored),
     checks Ortho4XP_Data's config was not touched, and imports the loose
     overlay modules with the live venv.
  5. Deletes the build leftovers (build/, dist_build/, dist/) unless
     --keep-build, so ~1 GB doesn't pile up between deploys.

usage: .venv/Scripts/python.exe scripts/deploy_live.py [--live DIR] [--skip-build] [--keep-build] [--dry-run]
"""
import argparse
import filecmp
import os
import shutil
import stat
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, "dist", "Ortho4XP")
LEFTOVERS = [os.path.join(ROOT, d) for d in ("build", "dist_build", "dist")]
PROTECTED = "Ortho4XP_Data"          # never written by a deploy
IGNORE = {"__pycache__"}


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def live_processes(live):
    """Running processes whose executable lives under the live folder."""
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and "
          f"$_.ExecutablePath.ToLower().StartsWith('{os.path.join(os.path.normpath(live), '').lower()}') }} | "
          "ForEach-Object { \"$($_.ProcessId) $($_.ExecutablePath)\" }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def _writable(func, path, _exc):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def remove(path):
    if os.path.islink(path) or os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path, onerror=_writable)


def same_tree(a, b):
    """Byte-compare two trees, ignoring runtime __pycache__.  Returns diffs."""
    if os.path.isfile(a):
        return [] if filecmp.cmp(a, b, shallow=False) else [b]
    cmp = filecmp.dircmp(a, b, ignore=list(IGNORE))
    bad = [os.path.join(b, n) for n in cmp.left_only + cmp.right_only + cmp.funny_files]
    _, mismatch, errors = filecmp.cmpfiles(a, b, cmp.common_files, shallow=False)
    bad += [os.path.join(b, n) for n in mismatch + errors]
    for d in cmp.common_dirs:
        bad += same_tree(os.path.join(a, d), os.path.join(b, d))
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", default="G:/Dev/Ortho4XP", help="live install folder")
    ap.add_argument("--skip-build", action="store_true", help="deploy the existing dist/Ortho4XP")
    ap.add_argument("--keep-build", action="store_true", help="keep build/, dist_build/, dist/ afterwards")
    ap.add_argument("--dry-run", action="store_true", help="checks and plan only, copy nothing")
    a = ap.parse_args()
    live = os.path.normpath(a.live)
    live_int = os.path.join(live, "_internal")
    if not os.path.isfile(os.path.join(live, "Ortho4XP.exe")) or not os.path.isdir(live_int):
        sys.exit(f"not a live Ortho4XP install: {live}")

    dirty = git("status", "--porcelain", "--untracked-files=no")
    print(f"[deploy] {git('branch', '--show-current')} @ {git('rev-parse', '--short', 'HEAD')}"
          + ("  (UNCOMMITTED CHANGES ride along)" if dirty else "  (clean)"))
    running = live_processes(live)
    if running:
        print("[deploy] live app / venv processes are running:\n  " + "\n  ".join(running))
        sys.exit("[deploy] refused: wait for them to finish (never kill them by name)")

    if not a.skip_build and not a.dry_run:
        r = subprocess.run([sys.executable, os.path.join(ROOT, "build.py")], cwd=ROOT)
        if r.returncode:
            sys.exit("[deploy] build.py failed; live install untouched")
    dist_int = os.path.join(DIST, "_internal")
    if not os.path.isdir(dist_int):
        sys.exit(f"[deploy] no build at {DIST} (run without --skip-build)")

    entries = sorted(e for e in os.listdir(dist_int) if e != PROTECTED)
    cfg = os.path.join(live_int, PROTECTED, "Ortho4XP.cfg")
    cfg_mtime = os.path.getmtime(cfg) if os.path.exists(cfg) else None
    print(f"[deploy] {len(entries)} _internal entries + Ortho4XP.exe -> {live} "
          f"({PROTECTED} left alone)")
    if a.dry_run:
        print("[deploy] dry run: nothing copied")
        return

    for e in entries:
        src, dst = os.path.join(dist_int, e), os.path.join(live_int, e)
        remove(dst)
        if os.path.isdir(src):
            shutil.copytree(src, dst, symlinks=True)
        else:
            shutil.copy2(src, dst)
    shutil.copy2(os.path.join(DIST, "Ortho4XP.exe"), os.path.join(live, "Ortho4XP.exe"))

    bad = []
    for e in entries:
        bad += same_tree(os.path.join(dist_int, e), os.path.join(live_int, e))
    bad += same_tree(os.path.join(DIST, "Ortho4XP.exe"), os.path.join(live, "Ortho4XP.exe"))
    if bad:
        print("[deploy] MISMATCH after copy:\n  " + "\n  ".join(bad[:50]))
        sys.exit("[deploy] verification failed; build kept for inspection")
    print("[deploy] verified: live matches the build byte for byte")
    if cfg_mtime is not None and os.path.getmtime(cfg) != cfg_mtime:
        sys.exit(f"[deploy] {cfg} changed during the deploy - investigate")

    loose = os.path.join(live_int, "overlay_scripts", "src")
    venv_py = os.path.join(live, ".venv", "Scripts", "python.exe")
    if os.path.isdir(loose) and os.path.isfile(venv_py):
        mods = sorted(f[:-3] for f in os.listdir(loose) if f.startswith("O4_") and f.endswith(".py"))
        code = ("import importlib, sys\nbad = []\n"
                f"for m in {mods!r}:\n"
                "    try:\n        importlib.import_module(m)\n"
                "    except Exception as exc:\n        bad.append(f'{m}: {exc!r}')\n"
                "print('\\n'.join(bad) if bad else f'imported {len(%r)} loose modules')\n"
                "sys.exit(1 if bad else 0)\n" % (mods,))
        r = subprocess.run([venv_py, "-c", code], cwd=loose, capture_output=True, text=True, timeout=600)
        print("[deploy] live venv: " + (r.stdout.strip() or r.stderr.strip()[-2000:]))
        if r.returncode:
            sys.exit("[deploy] a loose module fails to import in the live venv")

    if not a.keep_build:
        for d in LEFTOVERS:
            remove(d)
        print("[deploy] removed build/, dist_build/, dist/")
    print("[deploy] done")


if __name__ == "__main__":
    main()
