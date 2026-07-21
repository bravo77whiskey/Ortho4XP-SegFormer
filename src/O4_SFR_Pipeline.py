"""
O4_SFR_Pipeline.py — Pipeline bridge for SFR vegetation and building overlays.

Exposes process_veg_tile() / process_bld_tile() entry points that the Ortho4XP
pipeline calls after Build Imagery/DSF, using module-level config vars synced from
the per-tile Tile object.

scripts.generate_veg_overlay and scripts.generate_bld_overlay are imported lazily inside each
process function (after _activate_venv()) because they import torch at the top
level — importing them at module load time would fail in the frozen exe before
.venv has been set up.
"""

import math
import os
import re
import subprocess
import sys
import threading
import time

# O4_File_Names is only available inside the main process (frozen or source).
# Imported lazily in process_*_tile() so venv subprocesses are unaffected.
def _sfr_cache_dir(lat, lon):
    try:
        import O4_File_Names as FNAMES
        return FNAMES.sfr_cache_dir(lat, lon)
    except Exception:
        # Fallback for any context where FNAMES isn't importable
        return os.path.join(_exe_dir, 'SFR_cache',
                            f'{int(lat):+03d}{int(lon):+04d}')


# Persistent sidecar caches written under the per-tile SFR_cache dir by
# PCACHE.load_or_build() / ensure_cached_dsf_text() (parsed OSM, mesh-water
# index, disassembled DSF text, OBJ8 bounds). Unlike the per-DDS inference
# files these ignore the overlay's disable_cache flag, so they must be purged
# here when the cache is disabled — otherwise SFR_cache/<tile> is never empty.
_SFR_PERSIST_NAMESPACES = (
    "osm_parse",
    "mesh_water",
    "dsf_parse",
    "dsf_disassembly",
    "obj8_bounds",
)
_SFR_DERIVED_CACHE_DIRS = _SFR_PERSIST_NAMESPACES + (
    "yolo_zl16_analysis",
)
_SFR_DERIVED_FILE_PATTERNS = (
    "*_road.pkl",
    "*_vegaux.pkl",
    "*_vegpoly.pkl",
)
_SFR_DISABLED_TILE_FILE_PATTERNS = (
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


def _cleanup_sfr_cache(cache_dir, patterns=(), dir_names=(), label="SFR",
                       reason="cache cleanup"):
    """Remove selected files/directories from a tile SFR cache directory."""
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
        print(f"[SFR] Cleared {label} {reason}.", flush=True)
    return removed


def _purge_sfr_cache(cache_dir, patterns, label):
    """Clear a tile's SFR cache when caching is disabled.

    Removes the disabled overlay's per-DDS files (``patterns``) plus the shared
    persistent sidecar namespaces, so the per-tile SFR_cache folder does not
    retain inference maps or parsed OSM/mesh/DSF data between builds.
    """
    return _cleanup_sfr_cache(
        cache_dir,
        tuple(patterns) + _SFR_DISABLED_TILE_FILE_PATTERNS + _SFR_DERIVED_FILE_PATTERNS,
        _SFR_DERIVED_CACHE_DIRS,
        label,
        "cache (caching disabled)",
    )


def _cleanup_sfr_derived_cache(cache_dir, label):
    """Drop rebuildable bulky intermediates while keeping reusable SFR caches."""
    return _cleanup_sfr_cache(
        cache_dir,
        _SFR_DERIVED_FILE_PATTERNS,
        _SFR_DERIVED_CACHE_DIRS,
        label,
        "derived cache",
    )

# ── Locate root dir and make overlay scripts importable ───────────────────────
# In a frozen PyInstaller bundle sys.executable is Ortho4XP.exe and the
# bundled scripts land in sfr_scripts/scripts/ and helper modules land in
# sfr_scripts/src/ inside _MEIPASS (the _internal/ folder next to the exe).
# In source mode _root_dir is one level above src/.
if getattr(sys, 'frozen', False):
    # Keep repo-like structure inside sfr_scripts so subprocess imports match
    # source-mode imports.
    _root_dir = os.path.join(sys._MEIPASS, 'sfr_scripts')
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


# scripts.generate_veg_overlay / scripts.generate_bld_overlay are run as .venv subprocesses,
# never imported into the frozen exe process.

# ── Subprocess kill-switch ────────────────────────────────────────────────────
# The heavy lifting happens in a detached .venv python; UI.red_flag alone can't
# stop it. Every live Popen is registered with O4_UI_Utils so the GUI Stop
# button and window close can terminate the whole tree immediately.
# O4_UI_Utils is imported lazily: this module must stay importable in contexts
# where the main-process helpers are unavailable (see _sfr_cache_dir above).

def _ui():
    try:
        import O4_UI_Utils as UI
        return UI
    except Exception:
        return None


def _red_flag_set():
    UI = _ui()
    return bool(UI.red_flag) if UI else False

# ── Module-level config vars — synced from Tile before each call ──────────────
sfr_veg_density       = -1.0    # -1 = auto; 0.0–1.0 = override
sfr_veg_close_m       = 10.0
sfr_veg_open_m        = 3.0
sfr_veg_min_area_m2   = 50.0
sfr_veg_simplify_m    = 3.0
sfr_veg_excl_buffer_m = 5.0
sfr_veg_use_simheaven = True
sfr_veg_avoid_simheaven_buildings = True
sfr_veg_simheaven_building_buffer_m = 10.0
sfr_veg_avoid_gfv2    = True
sfr_veg_gfv2_buffer_m = 0.0
sfr_veg_use_gfv2_asset_proximity = False
sfr_veg_res_m         = 0.0     # 0 = native DDS resolution
sfr_veg_disable_cache = False

sfr_bld_spacing_m     = 0.0
sfr_bld_close_m       = 30.0
sfr_bld_open_m        = 10.0
sfr_bld_min_footprint_m2 = 12.0
sfr_bld_grid_n        = 16
sfr_bld_disable_cache = False
sfr_bld_verbose_log   = False
sfr_bld_avoid_custom_scenery = True
sfr_bld_asset_mode = "both"
sfr_bld_yolo_enabled = True
sfr_bld_yolo_checkpoint = r"H:\model_training\runs\yolo_obb_v1\weights\visual_candidate_step_12000.pt"
sfr_bld_yolo_conf = 0.18
sfr_bld_yolo_min_coverage = 0.80
sfr_bld_yolo_iou = 0.5
sfr_bld_yolo_stride = 512
sfr_bld_yolo_max_det = 100000
sfr_bld_height_checkpoint = r"H:\model_training\models\heightnet.pt"

# ── SegFormer inference settings (shared by veg and bld) ─────────────────────
sfr_patch_size        = 512
sfr_overlap           = 64
sfr_batch_size        = 0      # 0 = recommended default in O4_SFR_Inference

# ── Remote GPU offload (session-only, NOT a config setting) ──────────────────
# Set by the GUI "Remote GPU" checkbox (or O4_SFR_REMOTE=1 for headless runs)
# right before a build starts; never persisted. When set, SegFormer and YOLO
# forward passes for this run execute on the remote host — everything else
# stays local. Probed per tile: if the host is offline the build silently
# continues with local inference.
sfr_remote_host       = ""


def _resolve_remote_host():
    """Return a prepared remote host for this build step, or '' for local."""
    host = (sfr_remote_host or "").strip()
    if not host and os.environ.get("O4_SFR_REMOTE", "").strip() == "1":
        import O4_SFR_Remote as REMOTE
        host = REMOTE.default_host()
    if not host:
        return ""
    import O4_SFR_Remote as REMOTE
    print(f"[SFR] Remote GPU requested — checking {host} …", flush=True)
    if REMOTE.prepare_remote(host):
        print(f"[SFR] Model inference will run on {host}.", flush=True)
        return host
    print(
        f"[SFR] WARNING: remote host {host} is offline or not ready "
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
        f"import O4_SFR_Remote as _SFR_REMOTE\n"
        f"_SFR_REMOTE.activate({remote_host!r})\n"
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
            import O4_SFR_Inference as _SEG
            return _SEG._dsftool
        except Exception:
            return None
    if sys.platform.startswith('win'):
        return os.path.join(utils_dir, 'win', 'DSFTool.exe')
    if sys.platform.startswith('darwin'):
        return os.path.join(utils_dir, 'mac', 'DSFTool')
    return os.path.join(utils_dir, 'lin', 'DSFTool')


def _run_venv(code):
    """Run Python code in .venv, streaming stdout line by line to this process.

    The subprocess receives PYTHONPATH pointing to _root_dir so it can import
    the bundled SFR/SegFormer modules as loose .py files from sfr_scripts/src.

    Returns the process exit code.
    """
    env = os.environ.copy()
    prev_paths = [
        path for path in env.get('PYTHONPATH', '').split(os.pathsep)
        if path
    ]
    stale_markers = (
        os.path.normcase(os.path.join('Ortho4XP_Data', 'sfr_scripts')),
        os.path.normcase(os.path.join('_internal', '_internal', 'sfr_scripts')),
    )
    filtered_prev = []
    for path in prev_paths:
        norm_path = os.path.normcase(os.path.normpath(path))
        if any(marker in norm_path for marker in stale_markers):
            continue
        filtered_prev.append(path)
    py_paths = [_root_dir, _src_dir]
    env['PYTHONPATH'] = os.pathsep.join(py_paths + filtered_prev)

    proc = subprocess.Popen(
        [_venv_python(), '-u', '-c', code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        **_no_window(),
    )
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
        "For clean SegFormer benchmarks, clear only the SFR_cache tile folder and "
        "keep Orthophotos or Tiles\\zOrtho4XP_*\\textures. Skipping this SFR step.",
        flush=True,
    )
    return False


def _scenery_paths():
    """Return the configured Custom Scenery root needed by the SFR overlay subprocesses."""
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
    the first generation run installs everything without requiring a separate
    Setup SegFormer step.
    """
    print("[SFR] Checking SegFormer environment …", flush=True)
    if _deps_ready():
        print("[SFR] Environment ready.", flush=True)
        return True
    print("[SFR] Required packages not found in .venv — running automatic setup …")
    print("[SFR] (This is a one-time operation; subsequent runs start immediately.)")
    setup_sfr_models()
    return _deps_ready()


def process_veg_tile(lat, lon, build_dir):
    """Run SegFormer+GFv2 vegetation overlay for one tile in the .venv subprocess."""
    print(f"[SFR Veg] Starting for tile +{lat:02d}+{lon:03d} …", flush=True)
    tex_dir   = os.path.join(build_dir, 'textures')
    cache_dir = _sfr_cache_dir(lat, lon)
    if not _check_tile_imagery(tex_dir, lat, lon, "SegFormer vegetation overlay"):
        return

    if not _auto_setup():
        raise RuntimeError(
            "SegFormer setup failed — torch could not be installed into .venv. "
            "Click 'Setup SegFormer' and check the log for errors."
        )

    os.makedirs(cache_dir, exist_ok=True)

    density_override = None if sfr_veg_density < 0 else sfr_veg_density
    res_m            = None if sfr_veg_res_m <= 0 else sfr_veg_res_m
    dsftool          = _dsftool_path()
    out_dsf          = _dsf_output_path(lat, lon, 'yOrtho4XP_Veg_Overlays')
    custom_scenery_dir = _scenery_paths()
    remote_host      = _resolve_remote_host()

    code = (
        f"import O4_SFR_Inference as SEG\n"
        f"SEG.segformer_patch_size = {sfr_patch_size!r}\n"
        f"SEG.segformer_overlap    = {sfr_overlap!r}\n"
        f"SEG.segformer_batch_size = {sfr_batch_size!r}\n"
        f"SEG._dsftool     = {dsftool!r}\n"
        + _remote_activation_code(remote_host) +
        f"import O4_SFR_Vegetation_Overlay as veg_overlay\n"
        f"veg_overlay.run(\n"
        f"    tex_dir          = {tex_dir!r},\n"
        f"    lat              = {lat!r},\n"
        f"    lon              = {lon!r},\n"
        f"    out_dsf          = {out_dsf!r},\n"
        f"    cache_dir        = {cache_dir!r},\n"
        f"    close_m          = {sfr_veg_close_m!r},\n"
        f"    open_m           = {sfr_veg_open_m!r},\n"
        f"    min_area_m2      = {sfr_veg_min_area_m2!r},\n"
        f"    simplify_m       = {sfr_veg_simplify_m!r},\n"
        f"    make_viz         = False,\n"
        f"    density_override = {density_override!r},\n"
        f"    res_m            = {res_m!r},\n"
        f"    disable_cache    = {sfr_veg_disable_cache!r},\n"
        f"    excl_buffer_m    = {sfr_veg_excl_buffer_m!r},\n"
        f"    use_simheaven    = {sfr_veg_use_simheaven!r},\n"
        f"    avoid_simheaven_buildings = {sfr_veg_avoid_simheaven_buildings!r},\n"
        f"    simheaven_building_buffer_m = {sfr_veg_simheaven_building_buffer_m!r},\n"
        f"    avoid_gfv2       = {sfr_veg_avoid_gfv2!r},\n"
        f"    gfv2_buffer_m    = {sfr_veg_gfv2_buffer_m!r},\n"
        f"    use_gfv2_asset_proximity = {sfr_veg_use_gfv2_asset_proximity!r},\n"
        f"    bld_excl_m       = {0.0 if sfr_veg_disable_cache else 10.0!r},\n"
        f"    dsftool_path     = {dsftool!r},\n"
        f"    custom_scenery_dir = {custom_scenery_dir!r},\n"
        f")\n"
    )
    ret = None
    try:
        ret = _run_venv(code)
        if ret != 0:
            if _red_flag_set():
                print("[SFR Veg] Interrupted by user.", flush=True)
                return
            raise RuntimeError(f"SegFormer veg overlay subprocess failed (exit {ret})")
    finally:
        if sfr_veg_disable_cache:
            _purge_sfr_cache(
                cache_dir,
                ['*_veg.npy', '*_vegaux.pkl', '*_vegpoly.pkl'],
                'veg',
            )
        else:
            _cleanup_sfr_derived_cache(cache_dir, 'veg')
    print(f"[SFR Veg] Done for tile +{lat:02d}+{lon:03d}.", flush=True)


def process_bld_tile(lat, lon, build_dir):
    """Run SegFormer+SFD Global building overlay for one tile in the .venv subprocess."""
    print(f"[SFR Bld] Starting for tile +{lat:02d}+{lon:03d} …", flush=True)
    tex_dir     = os.path.join(build_dir, 'textures')
    cache_dir   = _sfr_cache_dir(lat, lon)
    if not _check_tile_imagery(tex_dir, lat, lon, "SegFormer building overlay"):
        return

    if not _auto_setup():
        raise RuntimeError(
            "SegFormer setup failed — torch could not be installed into .venv. "
            "Click 'Setup SegFormer' and check the log for errors."
        )

    os.makedirs(cache_dir, exist_ok=True)
    native_zl16_m_per_px = 2.0
    close_k = max(1, int(round(sfr_bld_close_m / native_zl16_m_per_px)))
    open_k = max(1, int(round(sfr_bld_open_m / native_zl16_m_per_px)))
    dsftool     = _dsftool_path()
    out_dsf     = _dsf_output_path(lat, lon, 'yOrtho4XP_Bld_Overlays')
    custom_scenery_dir = _scenery_paths()
    remote_host = _resolve_remote_host()
    print(
        "[SFR Bld] Effective settings: "
        f"yolo_enabled={sfr_bld_yolo_enabled!r} "
        f"checkpoint={sfr_bld_yolo_checkpoint!r} "
        f"height_checkpoint={sfr_bld_height_checkpoint!r} "
        f"conf={sfr_bld_yolo_conf!r} iou={sfr_bld_yolo_iou!r} "
        f"stride={sfr_bld_yolo_stride!r} max_det={sfr_bld_yolo_max_det!r} "
        f"min_coverage={sfr_bld_yolo_min_coverage!r} "
        f"asset_mode={sfr_bld_asset_mode!r} "
        f"disable_cache={sfr_bld_disable_cache!r} "
        f"verbose_log={sfr_bld_verbose_log!r} "
        f"out_dsf={out_dsf!r}",
        flush=True,
    )

    code = (
        f"import os\n"
        f"os.environ['O4_SFR_BLD_ASSET_MODE'] = {sfr_bld_asset_mode!r}\n"
        f"import O4_SFR_Inference as SEG\n"
        f"SEG.segformer_patch_size = {sfr_patch_size!r}\n"
        f"SEG.segformer_overlap    = {sfr_overlap!r}\n"
        f"SEG.segformer_batch_size = {sfr_batch_size!r}\n"
        f"SEG._dsftool     = {dsftool!r}\n"
        + _remote_activation_code(remote_host) +
        f"import O4_SFR_Building_Overlay as bld_overlay\n"
        f"_bld_path = os.path.abspath(getattr(bld_overlay, '__file__', ''))\n"
        f"print(f'[SFR Bld] bld_overlay.__file__={{_bld_path}}', flush=True)\n"
        f"_bld_norm = os.path.normcase(os.path.normpath(_bld_path))\n"
        f"_stale_markers = (\n"
        f"    os.path.normcase(os.path.join('Ortho4XP_Data', 'sfr_scripts')),\n"
        f"    os.path.normcase(os.path.join('_internal', '_internal', 'sfr_scripts')),\n"
        f")\n"
        f"if any(marker in _bld_norm for marker in _stale_markers):\n"
        f"    raise RuntimeError(f'Stale SFR building overlay module imported: {{_bld_path}}')\n"
        f"bld_overlay.run(\n"
        f"    tex_dir                  = {tex_dir!r},\n"
        f"    lat                      = {lat!r},\n"
        f"    lon                      = {lon!r},\n"
        f"    out_dsf                  = {out_dsf!r},\n"
        f"    spacing_m    = {sfr_bld_spacing_m!r},\n"
        f"    close_k      = {close_k!r},\n"
        f"    open_k       = {open_k!r},\n"
        f"    min_footprint_m2 = {sfr_bld_min_footprint_m2!r},\n"
        f"    make_viz                 = False,\n"
        f"    disable_cache            = {sfr_bld_disable_cache!r},\n"
        f"    verbose_log              = {sfr_bld_verbose_log!r},\n"
        f"    avoid_custom_scenery     = {sfr_bld_avoid_custom_scenery!r},\n"
        f"    cache_dir                = {cache_dir!r},\n"
        f"    grid_n                   = {sfr_bld_grid_n!r},\n"
        f"    custom_scenery_dir       = {custom_scenery_dir!r},\n"
        f"    dsftool_path             = {dsftool!r},\n"
        f"    skip_osm_excl_download   = False,\n"
        f"    yolo_enabled             = {sfr_bld_yolo_enabled!r},\n"
        f"    yolo_checkpoint          = {sfr_bld_yolo_checkpoint!r},\n"
        f"    yolo_conf                = {sfr_bld_yolo_conf!r},\n"
        f"    yolo_iou                 = {sfr_bld_yolo_iou!r},\n"
        f"    yolo_stride              = {sfr_bld_yolo_stride!r},\n"
        f"    yolo_max_det             = {sfr_bld_yolo_max_det!r},\n"
        f"    yolo_min_coverage        = {sfr_bld_yolo_min_coverage!r},\n"
        f"    height_checkpoint        = {sfr_bld_height_checkpoint!r},\n"
        f")\n"
    )
    ret = None
    try:
        ret = _run_venv(code)
        if ret != 0:
            if _red_flag_set():
                print("[SFR Bld] Interrupted by user.", flush=True)
                return
            raise RuntimeError(f"SegFormer bld overlay subprocess failed (exit {ret})")
    finally:
        if sfr_bld_disable_cache:
            _purge_sfr_cache(
                cache_dir,
                ['*_bld.pkl', '*_yolo_obb.pkl', '*_road.pkl', '*_veg.npy'],
                'bld',
            )
        else:
            _cleanup_sfr_derived_cache(cache_dir, 'bld')
    print(f"[SFR Bld] Done for tile +{lat:02d}+{lon:03d}.", flush=True)


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

    print(f"[SFR Setup] .venv not found — creating {_VENV_DIR} …")
    py_cmd = _find_system_python()
    if py_cmd is None:
        print("[SFR Setup] ERROR: Could not find a Python 3 installation on this machine.\n"
              "  Install Python 3.10+ from https://python.org and re-run Setup SegFormer.")
        return False

    try:
        ver = subprocess.check_output(py_cmd + ['--version'], text=True,
                                      stderr=subprocess.STDOUT, timeout=10,
                                      **_no_window()).strip()
        print(f"[SFR Setup] Using system Python: {ver}  ({' '.join(py_cmd)})")
    except Exception:
        pass

    try:
        subprocess.check_call(py_cmd + ['-m', 'venv', _VENV_DIR],
                               **_no_window())
    except subprocess.CalledProcessError as exc:
        print(f"[SFR Setup] ERROR: venv creation failed: {exc}")
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

    print("[SFR Setup] .venv created successfully.")
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


def setup_sfr_models():
    """Install AI dependencies into .venv and pre-download SegFormer models.

    Intended to be called once via the "Setup SegFormer" GUI button before
    running any tile so that the first tile run never stalls on a large download.
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

    print("[SFR Setup] Starting SegFormer model setup …")
    print(f"[SFR Setup] Using .venv: {_VENV_DIR}")

    # ── 1. Detect GPU and pick the right torch wheel ──────────────────────────
    index_url, build_tag = _detect_torch_index_url()
    if build_tag == 'cpu':
        print("[SFR Setup] No NVIDIA GPU detected — installing torch (CPU) …")
    else:
        print(f"[SFR Setup] NVIDIA GPU detected (CUDA {build_tag[2:]}) — "
              f"installing torch with CUDA support …")
    try:
        _pip_install("torch", "torchvision", extra_index_url=index_url)
    except Exception as exc:
        print(f"[SFR Setup] ERROR installing torch: {exc}")
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
        print(f"[SFR Setup] Installing {label} into .venv …")
        try:
            _pip_install(pkg)
        except Exception as exc:
            print(f"[SFR Setup] ERROR installing {label}: {exc}")
            return

    # ── 3. Verify torch and download models — all in the venv subprocess ─────
    # We never import torch into the frozen exe process; the venv Python does
    # all the heavy lifting and streams output back line by line.
    ret = _run_venv(
        "import torch\n"
        "cuda_ok = torch.cuda.is_available()\n"
        "print(f'[SFR Setup] torch {torch.__version__} ready  '\n"
        "      f'({\"CUDA — GPU will be used\" if cuda_ok else \"CPU only\"})')\n"
        "\n"
        "from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, AutoImageProcessor\n"
        "\n"
        "print('[SFR Setup] Downloading inference model (nave1616/SegFormer-landcover-FT) …')\n"
        "print('[SFR Setup] (One-time download ~400 MB; subsequent runs load from cache.)')\n"
        "SegformerImageProcessor(\n"
        "    do_resize=True, size={'height': 512, 'width': 512},\n"
        "    do_normalize=True,\n"
        "    image_mean=[0.485, 0.456, 0.406],\n"
        "    image_std=[0.229, 0.224, 0.225],\n"
        ")\n"
        "SegformerForSemanticSegmentation.from_pretrained('nave1616/SegFormer-landcover-FT')\n"
        "print('[SFR Setup] Inference model ready.')\n"
        "\n"
        "print('[SFR Setup] Downloading building model (tomascanivari/segformer-b0-finetuned-buildings) …')\n"
        "AutoImageProcessor.from_pretrained('tomascanivari/segformer-b0-finetuned-buildings')\n"
        "SegformerForSemanticSegmentation.from_pretrained('tomascanivari/segformer-b0-finetuned-buildings')\n"
        "print('[SFR Setup] Building model ready.')\n"
        "print('[SFR Setup] Setup complete. Both models are cached and ready to use.')\n"
    )
    if ret != 0:
        print(f"[SFR Setup] ERROR: model download subprocess failed (exit {ret})")
        return

    print("[SFR Setup] Setup complete. Both models are cached and ready to use.")
