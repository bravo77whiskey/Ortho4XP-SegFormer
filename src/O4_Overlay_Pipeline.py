"""
O4_Overlay_Pipeline.py — Pipeline bridge for the vegetation and building overlays.

Exposes process_veg_tile() / process_bld_tile() entry points that the Ortho4XP
pipeline calls after Build Imagery/DSF, using module-level config vars synced from
the per-tile Tile object.

scripts.generate_veg_overlay and scripts.generate_bld_overlay are imported lazily inside each
process function (after _activate_venv()) because they import torch at the top
level — importing them at module load time would fail in the frozen exe before
.venv has been set up.
"""

import json
import math
import os
import re
import subprocess
import sys
import threading
import time

import O4_Persistent_Cache as PCACHE
import O4_Progress as PROG

# O4_File_Names is only available inside the main process (frozen or source).
# Imported lazily in process_*_tile() so venv subprocesses are unaffected.
def _overlay_cache_dir(lat, lon):
    try:
        import O4_File_Names as FNAMES
        return FNAMES.overlay_cache_dir(lat, lon)
    except Exception:
        # Fallback for any context where FNAMES isn't importable
        return os.path.join(PCACHE.cache_root(_exe_dir),
                            f'{int(lat):+03d}{int(lon):+04d}')


def _overlay_library_cache_dir():
    try:
        import O4_File_Names as FNAMES
        return FNAMES.overlay_library_cache_dir()
    except Exception:
        return os.path.join(PCACHE.cache_root(_exe_dir), '_libraries')


def _overlay_leaf_type_cache_dir():
    try:
        import O4_File_Names as FNAMES
        return FNAMES.overlay_leaf_type_cache_dir()
    except Exception:
        return os.path.join(PCACHE.cache_root(_exe_dir), '_leaf_type')


# Persistent sidecar caches written under the per-tile overlay cache dir by
# PCACHE.load_or_build() / ensure_cached_dsf_text() (parsed OSM, mesh-water
# index, disassembled DSF text, OBJ8 bounds). Unlike the per-DDS inference
# files these ignore the overlay's disable_cache flag, so they must be purged
# here when the cache is disabled — otherwise the tile cache dir is never empty.
_PERSIST_NAMESPACES = (
    "osm_parse",
    "mesh_water",
    "dsf_parse",
    "dsf_disassembly",
    "obj8_bounds",
)
_DERIVED_CACHE_DIRS = _PERSIST_NAMESPACES + (
    "yolo_zl16_analysis",
)
_DERIVED_FILE_PATTERNS = (
    "*_road.pkl",
    "*_vegaux.pkl",
    "*_vegpoly.pkl",
    "*_grasscls.npy",     # class maps the grass step keeps until it is done
    "*_grass1.npz",       # grass masks between its two passes
)
_DISABLED_TILE_FILE_PATTERNS = (
    "*.osm.bz2",
)


def _cleanup_empty_dirs(root_dir):
    """Remove empty cache directories below ``root_dir``, then ``root_dir``."""
    if not root_dir or not os.path.isdir(root_dir):
        return 0

    removed = 0
    for current_root, _dirnames, _filenames in os.walk(root_dir, topdown=False):
        try:
            if os.listdir(current_root):
                continue
        except OSError:
            continue
        try:
            os.rmdir(current_root)
            removed += 1
        except OSError:
            pass
    return removed


def _cleanup_overlay_cache(cache_dir, patterns=(), dir_names=(), label="overlay",
                       reason="cache cleanup"):
    """Remove selected files/directories from a tile overlay cache directory."""
    import glob
    import shutil

    if not cache_dir:
        return 0

    removed = 0
    for pattern in patterns:
        for path in glob.glob(os.path.join(cache_dir, pattern)):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                removed += 1
            except OSError:
                pass
    for name in dir_names:
        path = os.path.join(cache_dir, name)
        if os.path.isdir(path):
            try:
                shutil.rmtree(path)
                removed += 1
            except OSError:
                pass
    removed += _cleanup_empty_dirs(cache_dir)
    if removed:
        print(f"[Overlay] Cleared {label} {reason}.", flush=True)
    return removed


def _purge_overlay_cache(cache_dir, patterns, label):
    """Clear a tile's overlay cache when caching is disabled.

    Removes the disabled overlay's per-DDS files (``patterns``) plus the shared
    persistent sidecar namespaces, so the per-tile cache folder does not
    retain inference maps or parsed OSM/mesh/DSF data between builds.
    """
    return _cleanup_overlay_cache(
        cache_dir,
        tuple(patterns) + _DISABLED_TILE_FILE_PATTERNS + _DERIVED_FILE_PATTERNS,
        _DERIVED_CACHE_DIRS,
        label,
        "cache (caching disabled)",
    )


def _cleanup_derived_cache(cache_dir, label):
    """Drop rebuildable bulky intermediates while keeping reusable overlay caches."""
    return _cleanup_overlay_cache(
        cache_dir,
        _DERIVED_FILE_PATTERNS,
        _DERIVED_CACHE_DIRS,
        label,
        "derived cache",
    )

# ── Locate root dir and make overlay scripts importable ───────────────────────
# In a frozen PyInstaller bundle sys.executable is Ortho4XP.exe and the
# bundled scripts land in overlay_scripts/scripts/ and helper modules land in
# overlay_scripts/src/ inside _MEIPASS (the _internal/ folder next to the exe).
# In source mode _root_dir is one level above src/.
if getattr(sys, 'frozen', False):
    # Keep repo-like structure inside overlay_scripts so subprocess imports match
    # source-mode imports.
    _root_dir = os.path.join(sys._MEIPASS, 'overlay_scripts')
    _exe_dir  = os.path.dirname(sys.executable)   # .venv lives here
    _data_dir = os.path.join(sys._MEIPASS, 'Ortho4XP_Data')
else:
    _root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _exe_dir  = _root_dir
    _data_dir = _root_dir

_src_dir = os.path.join(_root_dir, 'src')
for _path in (_root_dir, _src_dir):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# ── Single shared virtual environment ────────────────────────────────────────
# We use ONE venv for everything — the same .venv that build.py creates to run
# PyInstaller.  It sits next to the exe (frozen) or in the repo root (source).
# No separate .venv; no version-matching gymnastics.
_VENV_DIR = os.path.join(_exe_dir, '.venv')


def _venv_python():
    """Return the Python executable inside the shared .venv."""
    if sys.platform.startswith('win'):
        return os.path.join(_VENV_DIR, 'Scripts', 'python.exe')
    return os.path.join(_VENV_DIR, 'bin', 'python')


def _venv_exists():
    return os.path.isfile(_venv_python())


def _no_window():
    """Return creationflags kwarg dict to suppress console windows on Windows."""
    if sys.platform.startswith('win'):
        return {'creationflags': subprocess.CREATE_NO_WINDOW}
    return {}


def _disable_power_throttling(proc):
    """Opt a worker out of Windows power throttling (EcoQoS). Best effort.

    Windows 11 may run windowless background processes on efficiency cores
    at low clocks.  A throttled bench process took 49 s instead of 22 s on
    one dense texture; opting out costs nothing when Windows would not have
    throttled the worker anyway.
    """
    if not sys.platform.startswith('win'):
        return False
    try:
        import ctypes
        from ctypes import wintypes

        class _PowerThrottlingState(ctypes.Structure):
            _fields_ = [('Version', wintypes.ULONG),
                        ('ControlMask', wintypes.ULONG),
                        ('StateMask', wintypes.ULONG)]

        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.SetProcessInformation.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.SetProcessInformation.restype = wintypes.BOOL
        # PROCESS_POWER_THROTTLING_EXECUTION_SPEED | ..._IGNORE_TIMER_RESOLUTION
        # controlled, both states off; class 4 = ProcessPowerThrottling
        state = _PowerThrottlingState(1, 0x1 | 0x4, 0)
        return bool(kernel32.SetProcessInformation(
            wintypes.HANDLE(int(proc._handle)), 4,
            ctypes.byref(state), ctypes.sizeof(state)))
    except Exception:
        return False


# scripts.generate_veg_overlay / scripts.generate_bld_overlay are run as .venv subprocesses,
# never imported into the frozen exe process.

# ── Subprocess kill-switch ────────────────────────────────────────────────────
# The heavy lifting happens in a detached .venv python; UI.red_flag alone can't
# stop it. Every live Popen is registered with O4_UI_Utils so the GUI Stop
# button and window close can terminate the whole tree immediately.
# O4_UI_Utils is imported lazily: this module must stay importable in contexts
# where the main-process helpers are unavailable (see _overlay_cache_dir above).

def _ui():
    try:
        import O4_UI_Utils as UI
        return UI
    except Exception:
        return None


def _red_flag_set():
    UI = _ui()
    return bool(UI.red_flag) if UI else False

def _veg_tree_env():
    """The tree-rule settings as the O4_VEG_* switches the veg overlay reads."""
    seg = float(veg_canopy_seg_min_h_m)
    return {
        "O4_VEG_CANOPY": "1" if veg_canopy else "0",
        "O4_VEG_CANOPY_MIN_CONF": repr(max(0.0, float(veg_canopy_min_conf))),
        "O4_VEG_CANOPY_SEG_MIN_H": "off" if seg <= 0 else repr(seg),
        "O4_VEG_CANOPY_ADD_MIN_H": "off" if float(veg_canopy_add_min_h_m) <= 0 else repr(float(veg_canopy_add_min_h_m)),
        "O4_VEG_FOREST_HOLES": "1" if veg_forest_holes else "0",
        "O4_VEG_POINT_MAX_CELLS": repr(max(0.0, float(veg_point_tree_max_cells))),
        "O4_VEG_HOLE_CERTAIN_CELLS": repr(max(0.0, float(veg_hole_certain_cells))),
        "O4_VEG_HOLE_UNSURE_CELLS": repr(max(0.0, float(veg_hole_unsure_cells))),
        "O4_VEG_TREE_SIZES": "1" if veg_tree_sizes else "0",
        "O4_VEG_FOREST_HEIGHTS": "1" if veg_forest_heights else "0",
        "O4_VEG_APPROACH_HEIGHTS": "1" if veg_approach_heights else "0",
        "O4_VEG_TROPICS_EVERGREEN": "1" if veg_tropics_evergreen else "0",
        "O4_VEG_UPSCALE_TREES": "1" if veg_upscale_gap_trees else "0",
        "O4_VEG_OSM_RESCUE_RANGELAND": "1" if veg_osm_rescue_rangeland else "0",
    }


# ── Module-level config vars — synced from Tile before each call ──────────────
veg_density       = -1.0    # -1 = auto; 0.0–1.0 = override
veg_close_m       = 10.0
veg_open_m        = 3.0
veg_min_area_m2   = 50.0
veg_simplify_m    = 3.0
veg_excl_buffer_m = 5.0
veg_use_simheaven = True
veg_avoid_simheaven_buildings = True
veg_simheaven_building_buffer_m = 10.0
veg_avoid_simheaven_forests = True
veg_simheaven_forest_buffer_m = 0.0
veg_exclude_lower_forests = False
veg_remove_embedded_trees = False
veg_use_simheaven_asset_proximity = True
veg_use_leaf_type_maps = True
veg_use_xp_forests = True
veg_canopy        = True    # ZL18/19 trees from canopy height on the ortho
veg_canopy_min_conf = 0.2   # 0 = mean height >= 2 m
veg_canopy_seg_min_h_m = 0.25  # 0 = off
veg_canopy_add_min_h_m = 0.1   # ZL18/19 canopy trees added outside airports; 0 = off
veg_forest_holes  = True
veg_point_tree_max_cells = 2.0   # in grid cells of the forest's own spacing
veg_hole_certain_cells = 1.0
veg_hole_unsure_cells = 0.0   # 0 = gaps of unsure ground stay forest
veg_forest_hole_min_m2 = 200.0   # legacy, unused
veg_tree_sizes    = True    # ZL18/19 single trees sized from canopy height
veg_forest_heights = True   # ZL18/19 shorter copies of the climate forests
veg_approach_heights = True  # trees under runway approaches below the approach surface
veg_tropics_evergreen = True  # no autumn colours between the tropics
veg_upscale_gap_trees = True   # ZL16: add the trees SegFormer finds on the texture enlarged 2x
veg_osm_rescue_rangeland = True  # trees on SegFormer rangeland inside OSM forests
veg_avoid_gfv2    = False
veg_gfv2_buffer_m = 0.0
veg_use_gfv2_asset_proximity = False
veg_res_m         = 0.0     # 0 = native DDS resolution
veg_disable_cache = False
veg_verbose_log = False
grass_enabled     = True    # 3-d grass around airports, built by the veg overlay into its own pack
grass_airport_buffer_m = 500.0
grass_full_tile   = False   # grass over the whole tile, not just around airports

bld_spacing_m     = 0.0
bld_close_m       = 30.0
bld_open_m        = 10.0
bld_min_footprint_m2 = 12.0
bld_grid_n        = 16
bld_disable_cache = False
bld_verbose_log   = False
bld_avoid_custom_scenery = True
bld_asset_mode = "both"
bld_roof_color_matching = False
bld_yolo_enabled = True
bld_yolo_checkpoint = r"H:\model_training\runs\yolo_obb_v1\weights\visual_candidate_step_12000.pt"
bld_yolo_conf = 0.18
bld_yolo_min_coverage = 0.80
bld_yolo_iou = 0.5
bld_yolo_stride = 512
bld_yolo_max_det = 100000
bld_height_checkpoint = r"H:\model_training\models\heightnet.pt"

# ── SegFormer inference settings (shared by veg and bld) ─────────────────────
segformer_patch_size        = 512
segformer_overlap           = 64
segformer_batch_size        = 0      # 0 = recommended default in O4_Landcover_Inference



def apply_tile_settings(tile, names):
    """Copy the named overlay settings from ``tile`` onto this module.

    The overlay steps read their settings from module globals (they are baked
    into the subprocess code); call this right before process_*_tile with the
    matching O4_Cfg_Vars.list_*_pipeline_vars list.
    """
    module = sys.modules[__name__]
    for name in names:
        if not hasattr(module, name):
            raise AttributeError(f"O4_Overlay_Pipeline has no setting {name!r}")
        setattr(module, name, getattr(tile, name))

# ── Remote GPU offload (session-only, NOT a config setting) ──────────────────
# Set by the GUI "Remote GPU" checkbox (or O4_REMOTE_GPU=1 for headless runs)
# right before a build starts; never persisted. When set, SegFormer and YOLO
# forward passes for this run execute on the remote host — everything else
# stays local. Probed per tile: if the host is offline the build silently
# continues with local inference.
remote_gpu_host       = ""


def _resolve_remote_host():
    """Return a prepared remote host for this build step, or '' for local."""
    host = (remote_gpu_host or "").strip()
    if not host and os.environ.get("O4_REMOTE_GPU", "").strip() == "1":
        import O4_Remote_GPU as REMOTE
        host = REMOTE.default_host()
        if not host:
            print("[Overlay] O4_REMOTE_GPU=1 but O4_REMOTE_GPU_HOST is not set "
                  "— running inference locally.", flush=True)
    if not host:
        return ""
    import O4_Remote_GPU as REMOTE
    print(f"[Overlay] Remote GPU requested — checking {host} …", flush=True)
    if REMOTE.prepare_remote(host):
        print(f"[Overlay] Model inference will run on {host}.", flush=True)
        return host
    print(
        f"[Overlay] WARNING: remote host {host} is offline or not ready "
        "(after connection retries) — ALL inference for this step will "
        "run locally.",
        flush=True,
    )
    return ""


def _remote_activation_code(remote_host):
    """Lines injected into the overlay subprocess to route inference remotely."""
    if not remote_host:
        return ""
    return (
        f"import O4_Remote_GPU as _REMOTE_GPU\n"
        f"_REMOTE_GPU.activate({remote_host!r})\n"
    )


def _dsf_output_path(lat, lon, folder):
    """Compute X-Plane DSF output path rooted at the runtime data directory.

    e.g. folder='yOrtho4XP_Veg_Overlays' →
         <data_dir>/yOrtho4XP_Veg_Overlays/Earth nav data/+30+100/+36+102.dsf
    """
    lat_i = int(lat); lon_i = int(lon)
    lat_g = int(math.floor(lat / 10)) * 10
    lon_g = int(math.floor(lon / 10)) * 10
    lat_s  = f"{'+' if lat_i >= 0 else '-'}{abs(lat_i):02d}"
    lon_s  = f"{'+' if lon_i >= 0 else '-'}{abs(lon_i):03d}"
    lat_gs = f"{'+' if lat_g >= 0 else '-'}{abs(lat_g):02d}"
    lon_gs = f"{'+' if lon_g >= 0 else '-'}{abs(lon_g):03d}"
    return os.path.join(_data_dir, folder, 'Earth nav data',
                        f'{lat_gs}{lon_gs}', f'{lat_s}{lon_s}.dsf')


def _dsftool_path():
    """Return DSFTool binary path (platform-aware, works frozen and from source)."""
    if getattr(sys, 'frozen', False):
        utils_dir = os.path.join(sys._MEIPASS, 'Ortho4XP_Data', 'Utils')
    else:
        try:
            import O4_Landcover_Inference as _SEG
            return _SEG._dsftool
        except Exception:
            return None
    if sys.platform.startswith('win'):
        return os.path.join(utils_dir, 'win', 'DSFTool.exe')
    if sys.platform.startswith('darwin'):
        return os.path.join(utils_dir, 'mac', 'DSFTool')
    return os.path.join(utils_dir, 'lin', 'DSFTool')


# The venv's python.exe is a launcher that runs the real interpreter as a
# child process, so _disable_power_throttling() on the Popen only reaches the
# launcher. This snippet makes the worker opt itself out of EcoQoS instead
# (measured 2.5x on the trained-YOLO stage). O4_OVERLAY_SELF_NOTHROTTLE=0
# turns it off.
_SELF_UNTHROTTLE_CODE = '''
try:
    import ctypes as _c
    from ctypes import wintypes as _w
    class _PTS(_c.Structure):
        _fields_ = [('Version', _w.ULONG), ('ControlMask', _w.ULONG), ('StateMask', _w.ULONG)]
    _k32 = _c.WinDLL('kernel32', use_last_error=True)
    _k32.GetCurrentProcess.restype = _w.HANDLE
    _k32.SetProcessInformation.argtypes = [_w.HANDLE, _c.c_int, _c.c_void_p, _w.DWORD]
    _k32.SetProcessInformation.restype = _w.BOOL
    _pts = _PTS(1, 0x1 | 0x4, 0)
    _k32.SetProcessInformation(_k32.GetCurrentProcess(), 4, _c.byref(_pts), _c.sizeof(_pts))
    del _c, _w, _PTS, _k32, _pts
except Exception:
    pass
'''


def _run_venv(code):
    """Run Python code in .venv, streaming stdout line by line to this process.

    The subprocess receives PYTHONPATH pointing to _root_dir so it can import
    the bundled overlay modules as loose .py files from overlay_scripts/src.

    Returns the process exit code.
    """
    env = os.environ.copy()
    prev_paths = [
        path for path in env.get('PYTHONPATH', '').split(os.pathsep)
        if path
    ]
    stale_markers = (
        os.path.normcase(os.path.join('Ortho4XP_Data', 'overlay_scripts')),
        os.path.normcase(os.path.join('_internal', '_internal', 'overlay_scripts')),
    )
    filtered_prev = []
    for path in prev_paths:
        norm_path = os.path.normcase(os.path.normpath(path))
        if any(marker in norm_path for marker in stale_markers):
            continue
        filtered_prev.append(path)
    py_paths = [_root_dir, _src_dir]
    env['PYTHONPATH'] = os.pathsep.join(py_paths + filtered_prev)
    # Progress and issues come back as event lines on stdout (O4_Progress).
    env[PROG.EVENT_ENV] = "1"
    if sys.platform.startswith('win') and os.environ.get(
        "O4_OVERLAY_SELF_NOTHROTTLE", ""
    ).strip().lower() not in ("0", "false", "no", "off"):
        code = _SELF_UNTHROTTLE_CODE + code

    proc = subprocess.Popen(
        [_venv_python(), '-u', '-c', code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        **_no_window(),
    )
    _disable_power_throttling(proc)
    UI = _ui()
    if UI:
        UI.register_subprocess(proc)

    # Watchdog: the stdout loop below blocks between lines, so poll red_flag
    # from the side and kill the tree as soon as the user hits Stop.
    def _watchdog():
        while proc.poll() is None:
            if _red_flag_set():
                _ui().kill_subprocess(proc)
                break
            time.sleep(0.5)

    if UI:
        threading.Thread(target=_watchdog, daemon=True).start()

    try:
        for line in proc.stdout:
            if PROG.handle_line(line):
                continue
            print(line, end='', flush=True)
        proc.wait()
    finally:
        if UI:
            UI.unregister_subprocess(proc)
    return proc.returncode


def _orthophoto_tile_dir(tex_dir, lat, lon):
    o4xp_root = os.path.dirname(os.path.dirname(os.path.dirname(tex_dir)))
    lat_i = int(lat); lon_i = int(lon)
    lat_g = int(math.floor(lat / 10)) * 10
    lon_g = int(math.floor(lon / 10)) * 10
    lat_s  = f"{'+' if lat_i >= 0 else '-'}{abs(lat_i):02d}"
    lon_s  = f"{'+' if lon_i >= 0 else '-'}{abs(lon_i):03d}"
    lat_gs = f"{'+' if lat_g >= 0 else '-'}{abs(lat_g):02d}"
    lon_gs = f"{'+' if lon_g >= 0 else '-'}{abs(lon_g):03d}"
    return os.path.join(
        o4xp_root, 'Orthophotos', f'{lat_gs}{lon_gs}', f'{lat_s}{lon_s}')


def _has_orthophotos(ortho_dir):
    if not os.path.isdir(ortho_dir):
        return False
    for _, _, names in os.walk(ortho_dir):
        if any(name.lower().endswith(('.jpg', '.jpeg', '.png')) for name in names):
            return True
    return False


_DDS_STD_RE = re.compile(
    r"^(\d+)_(\d+)_([A-Za-z][A-Za-z0-9_]*)(\d{2})\.dds$",
    re.IGNORECASE,
)
_MASK_TEXTURE_RE = re.compile(r"^\d+_\d+_ZL\d{2}(?:\.[A-Za-z0-9]+)?$", re.IGNORECASE)


def _is_mask_texture_name(name):
    return bool(_MASK_TEXTURE_RE.match(os.path.basename(name)))


def _has_tile_dds_textures(tex_dir):
    if not os.path.isdir(tex_dir):
        return False
    try:
        return any(
            _DDS_STD_RE.match(name) and not _is_mask_texture_name(name)
            for name in os.listdir(tex_dir)
        )
    except OSError:
        return False


def _check_tile_imagery(tex_dir, lat, lon, step_name):
    ortho_dir = _orthophoto_tile_dir(tex_dir, lat, lon)
    if _has_tile_dds_textures(tex_dir):
        return True
    if _has_orthophotos(ortho_dir):
        return True
    print(
        f"{step_name} requires source imagery for tile {int(lat):+03d}{int(lon):+04d}, "
        f"but no DDS textures were found at {tex_dir!r} and no cached orthophotos "
        f"were found at {ortho_dir!r}. "
        "For clean SegFormer benchmarks, clear only the overlay cache tile folder and "
        "keep Orthophotos or Tiles\\zOrtho4XP_*\\textures. Skipping this overlay step.",
        flush=True,
    )
    return False


def _scenery_paths():
    """Return the configured Custom Scenery root needed by the overlay subprocesses."""
    custom_scenery_dir = ""
    try:
        import O4_Config_Utils as CFG
        custom_scenery_dir = getattr(CFG, "custom_scenery_dir", "") or ""
    except Exception:
        pass
    return custom_scenery_dir


def _deps_ready():
    """Return True if all required packages are importable from the .venv Python."""
    if not _venv_exists():
        return False
    try:
        ret = subprocess.call(
            [_venv_python(), '-c', 'import torch, cv2, numpy, PIL, shapely, transformers, ultralytics'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **_no_window(),
        )
        return ret == 0
    except Exception:
        return False


def _auto_setup():
    """Run full model setup automatically if torch is not yet available.

    Called at the start of process_veg_tile() / process_bld_tile() so that
    the first generation run installs everything without a separate setup step.
    """
    print("[Overlay] Checking the overlay environment …", flush=True)
    if _deps_ready():
        print("[Overlay] Environment ready.", flush=True)
        return True
    print("[Overlay] Required packages not found in .venv — running automatic setup …")
    print("[Overlay] (This is a one-time operation; subsequent runs start immediately.)")
    setup_overlay_models()
    return _deps_ready()


def veg_replacement_root():
    """Return the folder holding vegetation-replacement backups and journal."""
    import O4_Veg_Replacement as CLEANUP
    return os.path.join(_data_dir, CLEANUP.MANAGED_DIR_NAME)


def veg_overlay_dsf_path(lat, lon):
    """Return the generated vegetation DSF for a tile."""
    return _dsf_output_path(lat, lon, 'yOrtho4XP_Veg_Overlays')


def _read_veg_status(status_path):
    """Return the overlay subprocess's status file, or None if it wrote none."""
    if not status_path or not os.path.exists(status_path):
        return None
    try:
        with open(status_path, 'r', encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        print(f"[Veg] status file unreadable ({exc})", flush=True)
        return None


def _finish_veg_tile_replacement(lat, lon, status_path, custom_scenery_dir,
                                 out_dsf, exclude_lower_forests,
                                 remove_embedded_trees):
    """Install or undo the tile replacement, once generation has really finished.

    Generation *starting* is not enough to justify suppressing a whole tile's
    forests. This only acts on what the overlay actually wrote: a compiled and
    validated DSF carrying the exclusion, an empty result that has to be
    reconciled, or a failure that must leave everything as it was.
    """
    import O4_Veg_Replacement as CLEANUP

    status = _read_veg_status(status_path)
    tile = CLEANUP.tile_name(lat, lon)
    store = CLEANUP.ManagedStore(veg_replacement_root())

    if status is None:
        if exclude_lower_forests:
            print(
                "[Veg] no status from the overlay run; leaving installed "
                "scenery untouched. Use the restore action if a previous "
                "replacement should be undone.",
                flush=True,
            )
        return

    outcome = status.get('outcome')
    if outcome not in ('ok', 'empty'):
        print(
            f"[Veg] generation ended as {outcome!r}; no exclusion installed "
            "and no embedded-tree cleanup started.",
            flush=True,
        )
        return

    if status.get('needs_reconcile'):
        try:
            result = CLEANUP.reconcile_tile(
                store, tile, overlay_dsf=out_dsf, ignore_busy=True,
                log=lambda message: print(message, flush=True),
            )
        except CLEANUP.ConflictError as exc:
            print(f"[Veg] could not reconcile tile {tile}: {exc}", flush=True)
            return
        if result.get('conflicts'):
            for conflict in result['conflicts']:
                print(f"[Veg] {conflict['path']}: {conflict['reason']}",
                      flush=True)
        return

    if not remove_embedded_trees:
        return
    if not status.get('embedded_tree_cleanup_allowed'):
        print("[Veg] embedded-tree cleanup skipped: the overlay did not "
              "install a tile exclusion.", flush=True)
        return

    conflicts = CLEANUP.validate_scenery_order(custom_scenery_dir, lat, lon)
    for conflict in conflicts:
        print(f"[Veg] scenery order [{conflict['severity']}] "
              f"{conflict['message']}", flush=True)
    if any(conflict['severity'] == 'error' for conflict in conflicts):
        print("[Veg] embedded-tree cleanup skipped until the scenery order "
              "is fixed.", flush=True)
        return

    try:
        inventory = CLEANUP.inventory_tile(
            custom_scenery_dir, lat, lon, rules=store.rules()
        )
        plan = CLEANUP.plan_tile(inventory, store)
        if not plan['sources']:
            print("[Veg] no supported embedded trees to remove in this tile.",
                  flush=True)
        else:
            CLEANUP.apply_plan(
                plan, store, ignore_busy=True,
                log=lambda message: print(message, flush=True),
            )
        print(CLEANUP.format_report(plan, inventory, conflicts,
                                    applied=bool(plan['sources']), store=store),
              flush=True)
    except CLEANUP.ConflictError as exc:
        print(f"[Veg] embedded-tree cleanup stopped: {exc}", flush=True)
    except Exception as exc:
        print(f"[Veg] embedded-tree cleanup failed: {exc}", flush=True)


def has_veg_tile_replacement(lat, lon):
    """True when this tile still has scenery edits recorded against it.

    Lets a caller skip a tile in silence instead of announcing "nothing to
    restore" once per tile across a whole batch.
    """
    try:
        import O4_Veg_Replacement as CLEANUP

        store = CLEANUP.ManagedStore(veg_replacement_root())
        return store.load_tile(CLEANUP.tile_name(lat, lon)) is not None
    except Exception:
        return False


def restore_veg_tile_replacement(lat, lon, keep_exclusion=False):
    """Undo one tile's installed vegetation replacement, both halves of it.

    Exposed for the GUI and the batch path. This is the *only* way installed
    edits come back: changing a configuration checkbox affects future builds,
    never scenery that is already on disk.
    """
    import O4_Veg_Replacement as CLEANUP

    tile = CLEANUP.tile_name(lat, lon)
    store = CLEANUP.ManagedStore(veg_replacement_root())
    overlay = None if keep_exclusion else veg_overlay_dsf_path(lat, lon)
    result = CLEANUP.restore_tile(
        store, tile, overlay_dsf=overlay, ignore_busy=True,
        log=lambda message: print(message, flush=True),
    )
    if result.get('conflicts'):
        for conflict in result['conflicts']:
            print(f"[Veg] {conflict['path']}: {conflict['reason']}",
                  flush=True)
    elif result.get('reason'):
        print(f"[Veg] tile {tile}: {result['reason']}", flush=True)
    else:
        print(f"[Veg] tile {tile}: restored "
              f"{len(result.get('restored', ()))} source DSF(s).", flush=True)
    return result


def process_veg_tile(lat, lon, build_dir, wait_for_bld_flag=None):
    """Run the vegetation overlay for one tile.

    ``wait_for_bld_flag``: the building overlay runs alongside (see
    process_overlays_tile); read its class maps as they appear, and stop
    waiting once this flag file exists.
    """
    print(f"[Veg] Starting for tile +{lat:02d}+{lon:03d} …", flush=True)
    tex_dir   = os.path.join(build_dir, 'textures')
    cache_dir = _overlay_cache_dir(lat, lon)
    if not _check_tile_imagery(tex_dir, lat, lon, "vegetation overlay"):
        return

    if not _auto_setup():
        raise RuntimeError(
            "Overlay setup failed — torch could not be installed into .venv. "
            "Check the log for errors; setup is retried on the next overlay run."
        )

    os.makedirs(cache_dir, exist_ok=True)

    density_override = None if veg_density < 0 else veg_density
    res_m            = None if veg_res_m <= 0 else veg_res_m
    dsftool          = _dsftool_path()
    out_dsf          = _dsf_output_path(lat, lon, 'yOrtho4XP_Veg_Overlays')
    grass_out_dsf    = _dsf_output_path(lat, lon, 'yOrtho4XP_Grass_Overlays')
    custom_scenery_dir = _scenery_paths()
    remote_host      = _resolve_remote_host()
    status_path      = os.path.join(cache_dir, 'veg_overlay_status.json')
    # The overlay runs in the .venv subprocess, so this file is how its outcome
    # comes back: a tile-wide exclusion and the cleanup that goes with it must
    # never be installed just because the subprocess started.
    try:
        if os.path.exists(status_path):
            os.remove(status_path)
    except OSError:
        pass

    exclude_lower_forests = bool(veg_exclude_lower_forests)
    remove_embedded_trees = bool(veg_remove_embedded_trees)
    if remove_embedded_trees and not exclude_lower_forests:
        print(
            "[Veg] embedded-tree removal needs forest replacement; "
            "ignoring it for this tile",
            flush=True,
        )
        remove_embedded_trees = False
    if veg_verbose_log:
        print(
            "[Veg] Effective settings: "
            f"exclude_lower_forests={exclude_lower_forests!r} "
            f"remove_embedded_trees={remove_embedded_trees!r} "
            f"avoid_simheaven_forests={veg_avoid_simheaven_forests!r}"
            + (" (forced off for this run by tile replacement)"
               if exclude_lower_forests and veg_avoid_simheaven_forests else ""),
            flush=True,
        )

    code = (
        (
            f"import os\nos.environ['O4_VEG_WAIT_FOR_BLD_SEG'] = {wait_for_bld_flag!r}\n"
            if wait_for_bld_flag else ""
        ) +
        f"import os\nos.environ.update({_veg_tree_env()!r})\n"
        f"import O4_Landcover_Inference as SEG\n"
        f"SEG.segformer_patch_size = {segformer_patch_size!r}\n"
        f"SEG.segformer_overlap    = {segformer_overlap!r}\n"
        f"SEG.segformer_batch_size = {segformer_batch_size!r}\n"
        f"SEG._dsftool     = {dsftool!r}\n"
        + _remote_activation_code(remote_host) +
        f"import O4_Vegetation_Overlay as veg_overlay\n"
        + (
            f"import O4_Grass_Overlay as _GRASS\n"
            f"_grass = _GRASS.GrassSettings(airport_buffer_m={float(grass_airport_buffer_m)!r}, "
            f"full_tile={bool(grass_full_tile)!r})\n"
            if grass_enabled else "_grass = None\n"
        ) +
        f"veg_overlay.run(\n"
        f"    tex_dir          = {tex_dir!r},\n"
        f"    lat              = {lat!r},\n"
        f"    lon              = {lon!r},\n"
        f"    out_dsf          = {out_dsf!r},\n"
        f"    cache_dir        = {cache_dir!r},\n"
        f"    close_m          = {veg_close_m!r},\n"
        f"    open_m           = {veg_open_m!r},\n"
        f"    min_area_m2      = {veg_min_area_m2!r},\n"
        f"    simplify_m       = {veg_simplify_m!r},\n"
        f"    make_viz         = False,\n"
        f"    density_override = {density_override!r},\n"
        f"    res_m            = {res_m!r},\n"
        f"    disable_cache    = {veg_disable_cache!r},\n"
        f"    verbose_log      = {veg_verbose_log!r},\n"
        f"    excl_buffer_m    = {veg_excl_buffer_m!r},\n"
        f"    use_simheaven    = {veg_use_simheaven!r},\n"
        f"    avoid_simheaven_buildings = {veg_avoid_simheaven_buildings!r},\n"
        f"    simheaven_building_buffer_m = {veg_simheaven_building_buffer_m!r},\n"
        f"    avoid_simheaven_forests = {veg_avoid_simheaven_forests!r},\n"
        f"    simheaven_forest_buffer_m = {veg_simheaven_forest_buffer_m!r},\n"
        f"    use_simheaven_asset_proximity = {veg_use_simheaven_asset_proximity!r},\n"
        f"    use_leaf_type_maps = {veg_use_leaf_type_maps!r},\n"
        f"    leaf_type_cache_dir = {_overlay_leaf_type_cache_dir()!r},\n"
        f"    use_xp_forests = {veg_use_xp_forests!r},\n"
        f"    exclude_lower_forests = {veg_exclude_lower_forests!r},\n"
        f"    status_path      = {status_path!r},\n"
        f"    bld_excl_m       = {0.0 if veg_disable_cache else 10.0!r},\n"
        f"    dsftool_path     = {dsftool!r},\n"
        f"    custom_scenery_dir = {custom_scenery_dir!r},\n"
        f"    grass_settings   = _grass,\n"
        f"    grass_out_dsf    = {grass_out_dsf!r},\n"
        f")\n"
    )
    ret = None
    try:
        ret = _run_venv(code)
        if ret != 0:
            if _red_flag_set():
                print("[Veg] Interrupted by user.", flush=True)
                return
            raise RuntimeError(f"Vegetation overlay subprocess failed (exit {ret})")
    finally:
        if veg_disable_cache:
            _purge_overlay_cache(
                cache_dir,
                ['*_veg.npy', '*_vegtree.npy', '*_vegcanopy.npy', '*_vegaux.pkl', '*_vegpoly.pkl',
                 '*_grasscls.npy', '*_grass1.npz'],
                'veg',
            )
        else:
            _cleanup_derived_cache(cache_dir, 'veg')
    _finish_veg_tile_replacement(
        lat, lon, status_path, custom_scenery_dir, out_dsf,
        exclude_lower_forests, remove_embedded_trees,
    )
    print(f"[Veg] Done for tile +{lat:02d}+{lon:03d}.", flush=True)


def process_bld_tile(lat, lon, build_dir):
    """Run SegFormer+SFD Global building overlay for one tile in the .venv subprocess."""
    print(f"[Bld] Starting for tile +{lat:02d}+{lon:03d} …", flush=True)
    tex_dir     = os.path.join(build_dir, 'textures')
    cache_dir   = _overlay_cache_dir(lat, lon)
    if not _check_tile_imagery(tex_dir, lat, lon, "building overlay"):
        return

    if not _auto_setup():
        raise RuntimeError(
            "Overlay setup failed — torch could not be installed into .venv. "
            "Check the log for errors; setup is retried on the next overlay run."
        )

    os.makedirs(cache_dir, exist_ok=True)
    native_zl16_m_per_px = 2.0
    close_k = max(1, int(round(bld_close_m / native_zl16_m_per_px)))
    open_k = max(1, int(round(bld_open_m / native_zl16_m_per_px)))
    dsftool     = _dsftool_path()
    out_dsf     = _dsf_output_path(lat, lon, 'yOrtho4XP_Bld_Overlays')
    custom_scenery_dir = _scenery_paths()
    remote_host = _resolve_remote_host()
    if bld_verbose_log:
        print(
            "[Bld] Effective settings: "
            f"yolo_enabled={bld_yolo_enabled!r} "
            f"checkpoint={bld_yolo_checkpoint!r} "
            f"height_checkpoint={bld_height_checkpoint!r} "
            f"conf={bld_yolo_conf!r} iou={bld_yolo_iou!r} "
            f"stride={bld_yolo_stride!r} max_det={bld_yolo_max_det!r} "
            f"min_coverage={bld_yolo_min_coverage!r} "
            f"asset_mode={bld_asset_mode!r} "
            f"roof_color_matching={bld_roof_color_matching!r} "
            f"disable_cache={bld_disable_cache!r} "
            f"verbose_log={bld_verbose_log!r} "
            f"out_dsf={out_dsf!r}",
            flush=True,
        )

    code = (
        f"import os\n"
        f"os.environ['O4_BLD_ASSET_MODE'] = {bld_asset_mode!r}\n"
        f"import O4_Landcover_Inference as SEG\n"
        f"SEG.segformer_patch_size = {segformer_patch_size!r}\n"
        f"SEG.segformer_overlap    = {segformer_overlap!r}\n"
        f"SEG.segformer_batch_size = {segformer_batch_size!r}\n"
        f"SEG._dsftool     = {dsftool!r}\n"
        + _remote_activation_code(remote_host) +
        f"import O4_Building_Overlay as bld_overlay\n"
        f"_bld_path = os.path.abspath(getattr(bld_overlay, '__file__', ''))\n"
        f"if {bld_verbose_log!r}:\n"
        f"    print(f'[Bld] bld_overlay.__file__={{_bld_path}}', flush=True)\n"
        f"_bld_norm = os.path.normcase(os.path.normpath(_bld_path))\n"
        f"_stale_markers = (\n"
        f"    os.path.normcase(os.path.join('Ortho4XP_Data', 'overlay_scripts')),\n"
        f"    os.path.normcase(os.path.join('_internal', '_internal', 'overlay_scripts')),\n"
        f")\n"
        f"if any(marker in _bld_norm for marker in _stale_markers):\n"
        f"    raise RuntimeError(f'Stale building overlay module imported: {{_bld_path}}')\n"
        f"bld_overlay.run(\n"
        f"    tex_dir                  = {tex_dir!r},\n"
        f"    lat                      = {lat!r},\n"
        f"    lon                      = {lon!r},\n"
        f"    out_dsf                  = {out_dsf!r},\n"
        f"    spacing_m    = {bld_spacing_m!r},\n"
        f"    close_k      = {close_k!r},\n"
        f"    open_k       = {open_k!r},\n"
        f"    min_footprint_m2 = {bld_min_footprint_m2!r},\n"
        f"    make_viz                 = False,\n"
        f"    disable_cache            = {bld_disable_cache!r},\n"
        f"    verbose_log              = {bld_verbose_log!r},\n"
        f"    avoid_custom_scenery     = {bld_avoid_custom_scenery!r},\n"
        f"    roof_color_matching      = {bld_roof_color_matching!r},\n"
        f"    cache_dir                = {cache_dir!r},\n"
        f"    library_cache_dir        = {_overlay_library_cache_dir()!r},\n"
        f"    grid_n                   = {bld_grid_n!r},\n"
        f"    custom_scenery_dir       = {custom_scenery_dir!r},\n"
        f"    dsftool_path             = {dsftool!r},\n"
        f"    yolo_enabled             = {bld_yolo_enabled!r},\n"
        f"    yolo_checkpoint          = {bld_yolo_checkpoint!r},\n"
        f"    yolo_conf                = {bld_yolo_conf!r},\n"
        f"    yolo_iou                 = {bld_yolo_iou!r},\n"
        f"    yolo_stride              = {bld_yolo_stride!r},\n"
        f"    yolo_max_det             = {bld_yolo_max_det!r},\n"
        f"    yolo_min_coverage        = {bld_yolo_min_coverage!r},\n"
        f"    height_checkpoint        = {bld_height_checkpoint!r},\n"
        f")\n"
    )
    ret = None
    try:
        ret = _run_venv(code)
        if ret != 0:
            if _red_flag_set():
                print("[Bld] Interrupted by user.", flush=True)
                return
            raise RuntimeError(f"Building overlay subprocess failed (exit {ret})")
    finally:
        if bld_disable_cache:
            # Leave the class maps for the vegetation pass, whose own purge
            # removes them (O4_VEG_REUSE_BLD_SEG=0: don't).
            _keep_seg = os.environ.get("O4_VEG_REUSE_BLD_SEG", "").strip().lower() not in ("0", "false", "no", "off")
            _purge_overlay_cache(
                cache_dir,
                ['*_bld.pkl', '*_yolo_obb.pkl', '*_road.pkl'] +
                ([] if _keep_seg else ['*_veg.npy', '*_vegtree.npy']),
                'bld',
            )
        else:
            _cleanup_derived_cache(cache_dir, 'bld')
    print(f"[Bld] Done for tile +{lat:02d}+{lon:03d}.", flush=True)


def _env_on(name):
    return os.environ.get(name, "").strip().lower() not in ("0", "false", "no", "off")


def overlays_run_together():
    """Whether process_overlays_tile runs the two overlays side by side.

    The vegetation overlay then reads the building overlay's SegFormer class
    maps as they are written. Only with caches off: with caches on, the
    vegetation overlay also reads the building placements (_bld.pkl) for its
    clearance, which exist only once the building overlay is done.
    O4_OVERLAYS_TOGETHER=0 always runs them one after the other.
    """
    return (
        _env_on("O4_OVERLAYS_TOGETHER") and _env_on("O4_VEG_REUSE_BLD_SEG")
        and bld_disable_cache and veg_disable_cache
    )


def process_overlays_tile(lat, lon, build_dir, do_bld=True, do_veg=True):
    """Building and vegetation overlays for one tile.

    Apply both overlays' tile settings first. When both are wanted they run
    side by side (see overlays_run_together), otherwise one after the other.
    Returns {"bld": exception or None, "veg": exception or None} for the
    overlays that ran.
    """
    errors = {}
    if not (do_bld and do_veg and overlays_run_together()):
        for key, wanted, step in (("bld", do_bld, process_bld_tile), ("veg", do_veg, process_veg_tile)):
            if not wanted:
                continue
            try:
                with PROG.step(lat, lon, f"{key}_overlay"):
                    step(lat, lon, build_dir)
                errors[key] = None
            except Exception as exc:
                errors[key] = exc
            if _red_flag_set():
                break
        return errors

    import glob
    import shutil
    import tempfile

    cache_dir = _overlay_cache_dir(lat, lon)
    os.makedirs(cache_dir, exist_ok=True)
    seg_patterns = ('*_veg.npy', '*_vegtree.npy', '*_bldseg.skip')

    def _clear_class_maps():
        for pattern in seg_patterns:
            for path in glob.glob(os.path.join(cache_dir, pattern)):
                try:
                    os.remove(path)
                except OSError:
                    pass

    # Only class maps written by this build may be read: clear leftovers.
    _clear_class_maps()
    # Keeps the folder non-empty: the building overlay's cache purge removes
    # an empty tile cache folder, which the vegetation overlay still uses.
    keep_path = os.path.join(cache_dir, 'overlays_running.keep')
    with open(keep_path, 'w'):
        pass
    flag_dir = tempfile.mkdtemp(prefix="o4_overlays_")
    done_flag = os.path.join(flag_dir, 'bld_done.flag')

    def _bld():
        try:
            with PROG.step(lat, lon, "bld_overlay"):
                process_bld_tile(lat, lon, build_dir)
            errors["bld"] = None
        except Exception as exc:
            errors["bld"] = exc
        finally:
            with open(done_flag, 'w'):
                pass

    print("[Overlay] Building and vegetation overlays run side by side.", flush=True)
    bld_thread = threading.Thread(target=_bld, name="bld-overlay", daemon=True)
    bld_thread.start()
    try:
        with PROG.step(lat, lon, "veg_overlay"):
            process_veg_tile(lat, lon, build_dir, wait_for_bld_flag=done_flag)
        errors["veg"] = None
    except Exception as exc:
        errors["veg"] = exc
    finally:
        bld_thread.join()
        _clear_class_maps()
        try:
            os.remove(keep_path)
        except OSError:
            pass
        shutil.rmtree(flag_dir, ignore_errors=True)
        _cleanup_empty_dirs(cache_dir)
    return errors


def rescan_building_libraries():
    """Rebuild the shared scenery-library catalog the building overlay reads.

    Tile builds re-read only packages whose library.txt changed; run this after
    replacing library OBJs in place, or whenever library results look stale.
    """
    print("[Library] Rescanning scenery libraries …", flush=True)
    custom_scenery_dir = _scenery_paths()
    if not custom_scenery_dir:
        print("[Library] No Custom Scenery folder configured.", flush=True)
        return False
    if not _auto_setup():
        raise RuntimeError(
            "Overlay setup failed — torch could not be installed into .venv. "
            "Check the log for errors; setup is retried on the next overlay run."
        )
    code = (
        "import O4_Building_Overlay as bld_overlay\n"
        "bld_overlay.rescan_library_catalog(\n"
        f"    {custom_scenery_dir!r},\n"
        f"    {_overlay_library_cache_dir()!r},\n"
        ")\n"
    )
    ret = _run_venv(code)
    if ret != 0:
        if _red_flag_set():
            print("[Library] Interrupted by user.", flush=True)
            return False
        raise RuntimeError(f"Library rescan subprocess failed (exit {ret})")
    return True


# ── Model setup / dependency installation ────────────────────────────────────

def _find_system_python():
    """Return a usable system Python 3.x command list, or None.

    Tries the Windows Python Launcher (py) first because it is the most
    reliable way to locate a real installation on Windows, then falls back
    to common names on the PATH.  Never returns sys.executable (which is the
    PyInstaller bootloader exe in frozen mode, not a real Python).
    """
    import shutil

    candidates = []
    if sys.platform.startswith('win'):
        # Windows Python Launcher — tries newest first
        for ver in ('3.13', '3.12', '3.11', '3.10', '3.9'):
            candidates.append(('py', f'-{ver}'))
        candidates.append(('py', '-3'))
    candidates += [('python3', None), ('python', None)]

    for exe, flag in candidates:
        path = shutil.which(exe)
        if not path:
            continue
        cmd = [path] + ([flag] if flag else []) + ['--version']
        try:
            out = subprocess.check_output(cmd, text=True,
                                          stderr=subprocess.STDOUT, timeout=10,
                                          **_no_window())
            if 'Python 3.' in out:
                if flag:
                    return [path, flag]
                return [path]
        except Exception:
            continue
    return None


def _ensure_venv():
    """Create the .venv using the system Python if it does not already exist.

    Returns True if the venv is ready, False if creation failed.
    """
    if _venv_exists():
        return True

    print(f"[Overlay Setup] .venv not found — creating {_VENV_DIR} …")
    py_cmd = _find_system_python()
    if py_cmd is None:
        print("[Overlay Setup] ERROR: Could not find a Python 3 installation on this machine.\n"
              "  Install Python 3.10+ from https://python.org and run the overlay again.")
        return False

    try:
        ver = subprocess.check_output(py_cmd + ['--version'], text=True,
                                      stderr=subprocess.STDOUT, timeout=10,
                                      **_no_window()).strip()
        print(f"[Overlay Setup] Using system Python: {ver}  ({' '.join(py_cmd)})")
    except Exception:
        pass

    try:
        subprocess.check_call(py_cmd + ['-m', 'venv', _VENV_DIR],
                               **_no_window())
    except subprocess.CalledProcessError as exc:
        print(f"[Overlay Setup] ERROR: venv creation failed: {exc}")
        return False

    # Upgrade pip so subsequent installs work reliably
    try:
        subprocess.check_call(
            [_venv_python(), '-m', 'pip', 'install', '--upgrade', 'pip'],
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            **_no_window(),
        )
    except Exception:
        pass  # non-fatal

    print("[Overlay Setup] .venv created successfully.")
    return True


def _pip_install(*packages, extra_index_url=None):
    """Install one or more pip packages into .venv."""
    cmd = [_venv_python(), '-m', 'pip', 'install', '--upgrade', *packages]
    if extra_index_url:
        cmd += ['--index-url', extra_index_url]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                          **_no_window())


# Ordered from newest to oldest — we pick the highest version the driver supports.
_CUDA_WHEEL_MAP = [
    ((12, 4), 'cu124'),
    ((12, 1), 'cu121'),
    ((11, 8), 'cu118'),
]


def _detect_torch_index_url():
    """Return the best PyTorch wheel index URL for this machine.

    Runs nvidia-smi to read the CUDA version reported by the driver.
    Returns a CUDA wheel URL if a compatible GPU is found, otherwise the
    CPU-only URL so the install always succeeds without manual intervention.
    """
    import re
    try:
        out = subprocess.check_output(
            ['nvidia-smi'], text=True,
            stderr=subprocess.DEVNULL, timeout=10,
            **_no_window(),
        )
        m = re.search(r'CUDA Version:\s*(\d+)\.(\d+)', out)
        if m:
            driver_cuda = (int(m.group(1)), int(m.group(2)))
            for min_ver, tag in _CUDA_WHEEL_MAP:
                if driver_cuda >= min_ver:
                    return f'https://download.pytorch.org/whl/{tag}', tag
    except Exception:
        pass
    return 'https://download.pytorch.org/whl/cpu', 'cpu'


def setup_overlay_models():
    """Install AI dependencies into .venv and pre-download SegFormer models.

    Called by _auto_setup() on the first overlay run when torch is missing.
    The .venv is created automatically if it does not yet exist.

    Steps:
      1. Create .venv next to the exe if it does not already exist (uses the
         system Python 3 found on PATH via the Windows Python Launcher or
         common names).
      2. Detect NVIDIA GPU / CUDA version via nvidia-smi and install the
         matching torch wheel automatically (CUDA or CPU fallback).
      3. Install transformers and huggingface-hub.
      4. Download and cache both SegFormer models via a .venv subprocess.

    All output goes to stdout so it appears in the Ortho4XP log window.
    """
    if not _ensure_venv():
        return

    print("[Overlay Setup] Starting SegFormer model setup …")
    print(f"[Overlay Setup] Using .venv: {_VENV_DIR}")

    # ── 1. Detect GPU and pick the right torch wheel ──────────────────────────
    index_url, build_tag = _detect_torch_index_url()
    if build_tag == 'cpu':
        print("[Overlay Setup] No NVIDIA GPU detected — installing torch (CPU) …")
    else:
        print(f"[Overlay Setup] NVIDIA GPU detected (CUDA {build_tag[2:]}) — "
              f"installing torch with CUDA support …")
    try:
        _pip_install("torch", "torchvision", extra_index_url=index_url)
    except Exception as exc:
        print(f"[Overlay Setup] ERROR installing torch: {exc}")
        return

    # ── 2. Install overlay script dependencies ───────────────────────────────
    for pkg, label in [
        ("numpy",                  "numpy"),
        ("pillow",                 "Pillow"),
        ("opencv-python",          "opencv-python"),
        ("shapely",                "shapely"),
        ("transformers>=4.30.0",   "transformers"),
        ("huggingface-hub>=1.0.0", "huggingface-hub"),
        ("ultralytics>=8.0.0",     "ultralytics"),
    ]:
        print(f"[Overlay Setup] Installing {label} into .venv …")
        try:
            _pip_install(pkg)
        except Exception as exc:
            print(f"[Overlay Setup] ERROR installing {label}: {exc}")
            return

    # ── 3. Verify torch and download models — all in the venv subprocess ─────
    # We never import torch into the frozen exe process; the venv Python does
    # all the heavy lifting and streams output back line by line.
    ret = _run_venv(
        "import torch\n"
        "cuda_ok = torch.cuda.is_available()\n"
        "print(f'[Overlay Setup] torch {torch.__version__} ready  '\n"
        "      f'({\"CUDA — GPU will be used\" if cuda_ok else \"CPU only\"})')\n"
        "\n"
        "from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, AutoImageProcessor\n"
        "\n"
        "print('[Overlay Setup] Downloading inference model (nave1616/SegFormer-landcover-FT) …')\n"
        "print('[Overlay Setup] (One-time download ~400 MB; subsequent runs load from cache.)')\n"
        "SegformerImageProcessor(\n"
        "    do_resize=True, size={'height': 512, 'width': 512},\n"
        "    do_normalize=True,\n"
        "    image_mean=[0.485, 0.456, 0.406],\n"
        "    image_std=[0.229, 0.224, 0.225],\n"
        ")\n"
        "SegformerForSemanticSegmentation.from_pretrained('nave1616/SegFormer-landcover-FT')\n"
        "print('[Overlay Setup] Inference model ready.')\n"
        "\n"
        "print('[Overlay Setup] Downloading building model (tomascanivari/segformer-b0-finetuned-buildings) …')\n"
        "AutoImageProcessor.from_pretrained('tomascanivari/segformer-b0-finetuned-buildings')\n"
        "SegformerForSemanticSegmentation.from_pretrained('tomascanivari/segformer-b0-finetuned-buildings')\n"
        "print('[Overlay Setup] Building model ready.')\n"
        "print('[Overlay Setup] Setup complete. Both models are cached and ready to use.')\n"
    )
    if ret != 0:
        print(f"[Overlay Setup] ERROR: model download subprocess failed (exit {ret})")
        return

    print("[Overlay Setup] Setup complete. Both models are cached and ready to use.")
