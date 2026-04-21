"""
build.py — Safe build script for Ortho4XP.

Use this instead of running pyinstaller directly.

What it does:
  1. Resolves the project venv (checks .venv/, venv/ in repo root).
     If neither exists, creates .venv/ and installs requirements.txt.
  2. Builds the bundle into dist_build/Ortho4XP/ using the venv Python.
  3. Smart-merges into dist/Ortho4XP/, leaving all user data untouched.

User data that is never deleted:
  Tiles, OSM_data, Orthophotos, Masks, yOrtho4XP_Overlays,
  yOrtho4XP_Veg_Overlays, yOrtho4XP_Bld_Overlays, Elevation_data, Geotiffs, tmp,
  Ortho4XP.cfg, .last_gui_params.txt, .venv/

Usage:
  python build.py          (any Python — will locate/create the project venv)
"""

import os
import sys
import shutil
import subprocess
import venv as _venv_mod

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.abspath(__file__))
STAGE_DIR = os.path.join(ROOT, "dist_build", "Ortho4XP")
DIST_DIR  = os.path.join(ROOT, "dist",       "Ortho4XP")

# These names (relative to DIST_DIR) are user data — never overwritten or deleted.
PRESERVE = {
    "Tiles",
    "OSM_data",
    "Orthophotos",
    "Masks",
    "yOrtho4XP_Overlays",
    "yOrtho4XP_Veg_Overlays",
    "yOrtho4XP_Bld_Overlays",
    "Elevation_data",
    "Geotiffs",
    "tmp",
    "SFR_cache",
    "Ortho4XP.cfg",
    ".last_gui_params.txt",
    ".venv",           # shared build+AI venv — preserved so torch/model cache survives rebuilds
}

# ── Resolve project venv ───────────────────────────────────────────────────────

def _venv_python(venv_dir):
    if sys.platform.startswith("win"):
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def _resolve_venv():
    """Return (venv_dir, python_path).  Creates .venv if nothing exists."""
    for candidate in (".venv", "venv"):
        d = os.path.join(ROOT, candidate)
        py = _venv_python(d)
        if os.path.isfile(py):
            return d, py

    # Neither exists — create .venv and install requirements
    venv_dir = os.path.join(ROOT, ".venv")
    print(f"[build] No project venv found — creating {venv_dir} …")
    _venv_mod.create(venv_dir, with_pip=True, clear=False)
    py = _venv_python(venv_dir)
    req = os.path.join(ROOT, "requirements.txt")
    if os.path.isfile(req):
        print("[build] Installing requirements.txt into .venv …")
        subprocess.check_call(
            [py, "-m", "pip", "install", "-r", req],
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )
    print("[build] Installing PyInstaller into .venv …")
    subprocess.check_call(
        [py, "-m", "pip", "install", "pyinstaller"],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    return venv_dir, py


VENV_DIR, VENV_PYTHON = _resolve_venv()
print(f"[build] Using venv: {VENV_DIR}")
print(f"[build] Python:     {VENV_PYTHON}")

# ── Step 1: build into staging ─────────────────────────────────────────────────
print()
print("=" * 70)
print("Step 1/2  Building into dist_build/Ortho4XP/ …")
print("=" * 70)
result = subprocess.run(
    [VENV_PYTHON, "-m", "PyInstaller", "Ortho4XP.spec",
     "--distpath", "dist_build", "--workpath", "build", "-y"],
    cwd=ROOT,
)
if result.returncode != 0:
    print("\nERROR: PyInstaller failed — dist/Ortho4XP/ was NOT modified.")
    sys.exit(result.returncode)

# ── Step 2: smart-merge stage → dist ──────────────────────────────────────────
print()
print("=" * 70)
print("Step 2/2  Updating dist/Ortho4XP/ (preserving user data) …")
print("=" * 70)

os.makedirs(DIST_DIR, exist_ok=True)

copied = []
skipped = []

for item in os.listdir(STAGE_DIR):
    if item in PRESERVE:
        skipped.append(item)
        continue

    src = os.path.join(STAGE_DIR, item)
    dst = os.path.join(DIST_DIR,  item)

    if os.path.isdir(dst):
        shutil.rmtree(dst)
    elif os.path.isfile(dst):
        os.remove(dst)

    if os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)

    copied.append(item)

print(f"\nUpdated  ({len(copied)} items): {', '.join(sorted(copied))}")
if skipped:
    print(f"Skipped  ({len(skipped)} items — user data): {', '.join(sorted(skipped))}")

print(f"\nDone.  Exe ready at:  {os.path.join(DIST_DIR, 'Ortho4XP.exe')}")
