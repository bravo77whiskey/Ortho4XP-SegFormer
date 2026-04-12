"""
O4_SegFormer_Overlay.py — SegFormer-based vegetation and building overlay generation for Ortho4XP.

Uses the nave1616/SegFormer-landcover-FT model to segment orthophoto DDS textures into:
  - Vegetation (tree, rangeland) → X-Plane .for forest overlays
  - Buildings                   → X-Plane .fac facade overlays

Output is a compiled binary DSF overlay placed in yOrtho4XP_SFR_Overlays/.

Usage (standalone):
    python generate_sfr_overlay.py --lat 45 --lon 7
    python generate_sfr_overlay.py --lat 45 --lon 7 --build-dir /path/to/tile
"""

import os
import random
import re
import sys
import subprocess

_CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
import numpy as np
import cv2
from math import atan, exp, log, pi, tan, cos, floor, sqrt
from shapely import geometry
from shapely.ops import unary_union
from shapely.validation import make_valid

# ── Ortho4XP imports ─────────────────────────────────────────────────────────
# Allow running from repo root or from src/
_src = os.path.dirname(__file__)
if _src not in sys.path:
    sys.path.insert(0, _src)

try:
    import O4_File_Names as FNAMES
    if sys.platform.startswith("darwin"):
        _dsftool = os.path.join(FNAMES.Utils_dir, "mac", "DSFTool")
    elif sys.platform.startswith("win"):
        _dsftool = os.path.join(FNAMES.Utils_dir, "win", "DSFTool.exe")
    else:
        _dsftool = os.path.join(FNAMES.Utils_dir, "lin", "DSFTool")
except ImportError:
    # Running outside the full Ortho4XP stack (e.g. in sfr_venv subprocess).
    # dsftool_path must be passed explicitly to any function that needs it.
    FNAMES = None
    _dsftool = None

# ── Model repositories ───────────────────────────────────────────────────────
# Vegetation model: 9-class landcover SegFormer (tree/rangeland/agriculture/buildings/…)
_MODEL_VEG_REPO = "nave1616/SegFormer-landcover-FT"
# Building model: binary SegFormer-B0 (0=background, 1=building)
# Set segformer_building_model="landcover" to use CLASS_BUILDING from the veg model instead.
_MODEL_BLD_REPO = "tomascanivari/segformer-b0-finetuned-buildings"
# Keep legacy alias so any external code referencing _MODEL_REPO still works
_MODEL_REPO = _MODEL_VEG_REPO

# Class indices in model argmax output (0-indexed)
# 0=background, 1=bareland, 2=rangeland, 3=developed, 4=road,
# 5=tree, 6=water, 7=agriculture, 8=buildings
CLASS_BACKGROUND  = 0
CLASS_BARELAND    = 1
CLASS_RANGELAND   = 2   # brush/grassland → mixed/brushwood forest overlays
CLASS_DEVELOPED   = 3
CLASS_ROAD        = 4
CLASS_TREE        = 5   # forest → deciduous/coniferous overlays by latitude
CLASS_WATER       = 6
CLASS_AGRICULTURE = 7
CLASS_BUILDING    = 8

VEG_CLASSES = (CLASS_TREE, CLASS_RANGELAND, CLASS_AGRICULTURE)

# ── X-Plane asset paths ───────────────────────────────────────────────────────
# Forest .for — XP12 lib paths confirmed from 1200 forests/library.txt.
# Climate variants: cold (<15°N or >55°N), temperate (15–55°N), warm/hot (tropics).
# We pick one per class at DSF-write time based on tile latitude.

def _for_path(class_idx, lat):
    """Return the XP12 lib .for path for a vegetation class at the given latitude."""
    if class_idx == CLASS_TREE:
        # Tree canopy → broadleaf/conifer selection by latitude
        if abs(lat) < 15:
            return "lib/vegetation/forests/broadleaves/hot.for"
        if abs(lat) < 40:
            return "lib/vegetation/forests/broadleaves/warm.for"
        if abs(lat) < 60:
            return "lib/vegetation/forests/broadleaves/temperate.for"
        return "lib/vegetation/forests/conifers/cold.for"
    if class_idx == CLASS_AGRICULTURE:
        # Farmland / crop fields → very sparse ground cover (grass .for)
        if abs(lat) < 40:
            return "lib/vegetation/forests/mixed/warm.for"
        return "lib/vegetation/forests/mixed/temperate.for"
    # CLASS_RANGELAND → mixed/brushwood scrub
    if abs(lat) < 15:
        return "lib/vegetation/forests/mixed/hot.for"
    if abs(lat) < 40:
        return "lib/vegetation/forests/mixed/warm.for"
    if abs(lat) < 60:
        return "lib/vegetation/forests/mixed/temperate.for"
    return "lib/vegetation/forests/mixed/cold.for"


# Per-class DSF density (0–255).  Lower = sparser placement within the .for area.
# Tree:        200 — dense canopy but not wall-to-wall packed
# Rangeland:    80 — sparse shrubs / scrubland
# Agriculture:  30 — very low ground-cover / hedgerow scatter
_FOR_DENSITY_BY_CLASS = {
    CLASS_TREE:        200,
    CLASS_RANGELAND:    80,
    CLASS_AGRICULTURE:  30,
}
_FOR_DENSITY = 200  # fallback for any class not in the dict above

# Facade .fac — XP12 lib paths confirmed from 1000 autogen/library.txt.
# Using generic lib/buildings/facades paths that work globally.
# Only used for large-footprint buildings (≥ segformer_bld_obj_threshold_m2).
_FAC_DEFS = {
    "medium": "lib/buildings/facades/generic/mid_classic_01.fac",
    "large":  "lib/buildings/facades/industrial/warehouse_01_45x45.fac",
}

# SFD Global Autogen suburban object pools by coarse region.
# Virtual library paths used in DSF OBJECT_DEF — resolved by X-Plane at runtime.
_SFD_OBJ_POOLS = {
    # Northern Europe / Scandinavia
    "scandinavia": [f"SFD_Global/Scandinavia/Residential/Suburban_{i}.obj" for i in range(1, 9)],
    # Mediterranean Europe
    "med":         [f"SFD_Global/Med/Residential/Suburban_{i}.obj"         for i in range(1, 9)],
    # North America — eastern / central
    "namerica_e":  [f"SFD_Global/New_England/Residential/Suburban_{i}.obj" for i in range(1, 9)],
    # North America — western
    "namerica_w":  [f"SFD_Global/US_West_Coast/Suburban_{i}.obj"           for i in range(1, 9)],
    # South America
    "samerica":    [f"SFD_Global/South_America/Suburban_{i}.obj"           for i in range(1, 11)],
    # Africa
    "africa":      [f"SFD_Global/Africa/Residential/Suburban_{i}.obj"      for i in range(1, 9)],
    # East / South-East Asia
    "asia":        [f"SFD_Global/Asia/Suburban_{i}.obj"                    for i in range(1, 9)],
    # South Asia (tropical belt)
    "asia_south":  [f"SFD_Global/Asia/Suburban_South_{i}.obj"              for i in range(1, 9)],
}


def _sfd_obj_pool(lat, lon):
    """Return the SFD Global suburban object path list most appropriate for (lat, lon)."""
    if 55 <= lat and 3 <= lon <= 35:          # Scandinavia
        return _SFD_OBJ_POOLS["scandinavia"]
    if 30 <= lat <= 72 and -25 <= lon <= 45:  # Europe
        return _SFD_OBJ_POOLS["scandinavia"] if lat >= 55 else _SFD_OBJ_POOLS["med"]
    if 25 <= lat <= 72 and -170 <= lon <= -50: # North America
        return _SFD_OBJ_POOLS["namerica_w"] if lon <= -100 else _SFD_OBJ_POOLS["namerica_e"]
    if -55 <= lat <= 15 and -82 <= lon <= -34: # South America
        return _SFD_OBJ_POOLS["samerica"]
    if -35 <= lat <= 37 and -17 <= lon <= 51:  # Africa
        return _SFD_OBJ_POOLS["africa"]
    if 0 <= lat <= 55 and 60 <= lon <= 145:    # Asia (mainland)
        return _SFD_OBJ_POOLS["asia"]
    if -10 <= lat < 0 and 60 <= lon <= 145:   # South-East Asia tropical
        return _SFD_OBJ_POOLS["asia_south"]
    return _SFD_OBJ_POOLS["namerica_e"]        # generic fallback

# ── Module-level tunables (overridable from tile config / O4_Cfg_Vars) ────────
sfr_overlay_enabled  = False   # master enable — set True in tile config to run
segformer_do_vegetation    = True    # include vegetation classes in output
segformer_do_buildings     = True    # include buildings class in output
segformer_canvas_px        = 16384   # stitch resolution (0 = legacy per-DDS mode)
segformer_patch_size       = 512     # pixels fed to the model per forward pass
segformer_overlap          = 64      # pixel overlap between adjacent patches (blending)
segformer_max_vram_gb      = 0.0     # 0 = uncapped; otherwise cap auto mode to this many GB
segformer_batch_size       = 0       # 0 = auto-size from available VRAM
segformer_confidence_threshold   = 0.5     # minimum softmax probability for a pixel to be assigned
segformer_min_veg_area_px        = 200   # minimum vegetation blob area in pixels
segformer_min_bld_area_px        = 15    # minimum building blob area in pixels
segformer_bld_obj_threshold_m2   = 200   # footprint below this → SFD object; above → .fac facade
segformer_building_model         = "dedicated"  # "dedicated" | "landcover"
                                  # dedicated → separate binary building model (_MODEL_BLD_REPO)
                                  # landcover → reuse CLASS_BUILDING from the veg model pass
_SIMPLIFY_TOL       = 1e-4    # Douglas-Peucker tolerance (degrees, ~11 m at mid-lat)

_MAX_RING_PTS       = 16000   # hard vertex cap per DSF winding (DSFTool limit ~65535 intervals)
_MAX_HOLES          = 100     # max interior rings kept per polygon

# ── Lazy model handles ────────────────────────────────────────────────────────
_model_veg     = None
_processor_veg = None
_model_bld     = None
_processor_bld = None
# Legacy alias kept for any callers that still reference _model / _processor
_model     = None
_processor = None
_batch_benchmark_cache = {}
_applied_vram_limit_bytes = None


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate helpers (mirror O4_Geo_Utils without importing it to stay
# importable outside Ortho4XP)
# ─────────────────────────────────────────────────────────────────────────────

def _gtile_to_wgs84(til_x, til_y, zl):
    """Top-left corner (lat, lon) of Google-numbered tile (til_x, til_y) at zoom zl."""
    rat_x = til_x / (2 ** (zl - 1)) - 1
    rat_y = 1 - til_y / (2 ** (zl - 1))
    lon = rat_x * 180
    lat = 360 / pi * atan(exp(pi * rat_y)) - 90
    return lat, lon


def _pix_to_wgs84(pix_x, pix_y, zl):
    """WGS84 (lat, lon) of global pixel (pix_x, pix_y) at zoom level zl."""
    rat_x = pix_x / (2 ** (zl + 7)) - 1
    rat_y = 1 - pix_y / (2 ** (zl + 7))
    lon = rat_x * 180
    lat = 360 / pi * atan(exp(pi * rat_y)) - 90
    return lat, lon


def _pixel_to_latlon(px, py, img_w, img_h, til_x_left, til_y_top, zl):
    """
    Convert image pixel (px, py) — origin top-left — to (lat, lon).
    Each Ortho4XP DDS texture covers a 16×16 Google-tile block (4096 global px
    at ZL16, scaled to whatever DDS resolution was chosen).
    """
    block_px = 16 * 256  # 16 tiles × 256 px/tile in global pixel space
    global_px = til_x_left * 256 + px * block_px / img_w
    global_py = til_y_top  * 256 + py * block_px / img_h
    return _pix_to_wgs84(global_px, global_py, zl)


def _latlon_extent(til_x_left, til_y_top, zl):
    """Return (lat_max, lon_min, lat_min, lon_max) for the texture block."""
    lat_max, lon_min = _gtile_to_wgs84(til_x_left,      til_y_top,      zl)
    lat_min, lon_max = _gtile_to_wgs84(til_x_left + 16, til_y_top + 16, zl)
    return lat_max, lon_min, lat_min, lon_max


def _approx_m2_per_px(til_y_top, img_h, zl):
    """Rough ground area (m²) per image pixel — used for minimum-area filtering."""
    # Use the vertical centre of the texture
    lat_top, _  = _pix_to_wgs84(0, til_y_top * 256,       zl)
    lat_bot, _  = _pix_to_wgs84(0, (til_y_top + 16) * 256, zl)
    lat_c = (lat_top + lat_bot) / 2
    metres_per_deg_lat = 111_320
    metres_per_deg_lon = metres_per_deg_lat * cos(lat_c * pi / 180)
    # lat span covered by one pixel
    lat_span = abs(lat_top - lat_bot) / img_h
    lon_span = lat_span  # roughly square pixels
    return lat_span * metres_per_deg_lat * lon_span * metres_per_deg_lon


# ─────────────────────────────────────────────────────────────────────────────
# DDS loading
# ─────────────────────────────────────────────────────────────────────────────

def _stitch_dds_canvas(tex_dir, lat, lon, canvas_size):
    """
    Stitch all DDS textures in tex_dir into a single (H, W, 3) uint8 RGB array
    covering the 1°×1° tile at (lat, lon).  Same coordinate logic as
    the stitched DDS texture canvas.
    Returns numpy array or None if no files could be placed.
    """
    import re as _re
    from PIL import Image as _Image
    _STD_RE   = _re.compile(r"^(\d+)_(\d+)_([A-Za-z][A-Za-z0-9_]*)(\d{2})\.dds$", _re.IGNORECASE)
    _G2XPL_RE = _re.compile(r"^(\d{2})_(\d+)_(\d+)\.dds$", _re.IGNORECASE)

    tile_s, tile_n = float(lat), float(lat + 1)
    tile_w, tile_e = float(lon), float(lon + 1)
    tile_dlon = tile_e - tile_w
    tile_dlat = tile_n - tile_s

    canvas = _Image.new("RGB", (canvas_size, canvas_size), (128, 128, 128))
    pasted = 0

    for fname in sorted(os.listdir(tex_dir)):
        if not fname.lower().endswith(".dds"):
            continue
        m = _STD_RE.match(fname)
        if m:
            til_y_top  = int(m.group(1))
            til_x_left = int(m.group(2))
            zl         = int(m.group(4))
        else:
            m = _G2XPL_RE.match(fname)
            if m:
                zl         = int(m.group(1))
                til_x_left = int(m.group(2))
                inverted_y = int(m.group(3))
                til_y_top  = 2 ** zl - 16 - inverted_y
            else:
                continue

        img_lat_n, img_lon_w = _gtile_to_wgs84(til_x_left,      til_y_top,      zl)
        img_lat_s, img_lon_e = _gtile_to_wgs84(til_x_left + 16, til_y_top + 16, zl)

        if img_lat_n < tile_s or img_lat_s > tile_n:
            continue
        if img_lon_e < tile_w or img_lon_w > tile_e:
            continue

        try:
            piece = _Image.open(os.path.join(tex_dir, fname)).convert("RGB")
        except Exception:
            continue

        col_l = int((img_lon_w - tile_w) / tile_dlon * canvas_size)
        col_r = int((img_lon_e - tile_w) / tile_dlon * canvas_size)
        row_t = int((tile_n - img_lat_n) / tile_dlat * canvas_size)
        row_b = int((tile_n - img_lat_s) / tile_dlat * canvas_size)

        dest_w = max(1, col_r - col_l)
        dest_h = max(1, row_b - row_t)
        src_col = max(0, -col_l)
        src_row = max(0, -row_t)
        paste_col = max(0, col_l)
        paste_row = max(0, row_t)

        if paste_col >= canvas_size or paste_row >= canvas_size:
            continue

        piece_r = piece.resize((dest_w, dest_h), _Image.LANCZOS)
        crop_w = min(canvas_size - paste_col, dest_w - src_col)
        crop_h = min(canvas_size - paste_row, dest_h - src_row)
        if crop_w <= 0 or crop_h <= 0:
            continue
        canvas.paste(piece_r.crop((src_col, src_row, src_col + crop_w, src_row + crop_h)),
                     (paste_col, paste_row))
        pasted += 1

    if pasted == 0:
        return None
    print(f"[SegFormer] Stitched {pasted} DDS textures into {canvas_size}×{canvas_size} canvas")
    return np.array(canvas, dtype=np.uint8)


def _load_dds(path):
    """
    Load a DDS texture file and return an (H, W, 3) uint8 RGB numpy array.
    Uses Pillow (10+) which has native DDS/DXT support.
    """
    from PIL import Image
    with Image.open(path) as img:
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# DDS filename parsing
# ─────────────────────────────────────────────────────────────────────────────

# Standard format:  {til_y_top}_{til_x_left}_{provider_code}{zoomlevel}.dds
#   e.g. 112256_218048_BI18.dds
# g2xpl_16 format: {zoomlevel}_{til_x_left}_{inverted_y}.dds   (all digits, zl first)
#   e.g. 18_218048_131856.dds
#   where inverted_y = 2**zl - 16 - til_y_top  →  til_y_top = 2**zl - 16 - inverted_y
_DDS_STD_RE    = re.compile(r"^(\d+)_(\d+)_([A-Za-z][A-Za-z0-9_]*)(\d{2})\.dds$",
                             re.IGNORECASE)
_DDS_G2XPL_RE  = re.compile(r"^(\d{2})_(\d+)_(\d+)\.dds$", re.IGNORECASE)


def parse_dds_filename(fname):
    """
    Parse an Ortho4XP DDS texture filename.

    Returns (til_y_top, til_x_left, provider_code, zoomlevel) or None.
    provider_code is 'g2xpl_16' for the g2xpl_16 variant.
    """
    base = os.path.basename(fname)

    # Standard provider format
    m = _DDS_STD_RE.match(base)
    if m:
        return int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))

    # g2xpl_16 format: zoomlevel_til_x_left_inverted_y.dds
    m = _DDS_G2XPL_RE.match(base)
    if m:
        zl         = int(m.group(1))
        til_x_left = int(m.group(2))
        inverted_y = int(m.group(3))
        til_y_top  = 2 ** zl - 16 - inverted_y
        return til_y_top, til_x_left, "g2xpl_16", zl

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _get_device(device=None):
    import torch
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _apply_vram_limit(device)
    return device


def _configured_vram_limit_bytes():
    """Return the configured VRAM cap in bytes, or ``None`` when uncapped."""
    if not segformer_max_vram_gb or segformer_max_vram_gb <= 0:
        return None
    return int(float(segformer_max_vram_gb) * (1024 ** 3))


def _apply_vram_limit(device):
    """Apply the configured CUDA per-process memory cap once per process.

    The user-facing setting is optional. When left at 0, SegFormer is allowed
    to use as much VRAM as CUDA makes available. When set, CUDA's allocator is
    capped to the requested fraction of total device memory.
    """
    global _applied_vram_limit_bytes
    import torch

    if not (hasattr(device, 'type') and device.type == 'cuda'):
        return

    limit_bytes = _configured_vram_limit_bytes()
    if limit_bytes == _applied_vram_limit_bytes:
        return

    if limit_bytes is None:
        _applied_vram_limit_bytes = None
        return

    try:
        total_bytes = torch.cuda.get_device_properties(device).total_memory
        fraction = max(1e-3, min(1.0, float(limit_bytes) / float(total_bytes)))
        torch.cuda.set_per_process_memory_fraction(fraction, device=device)
        effective_limit_gb = total_bytes * fraction / (1024 ** 3)
        print(
            f"[SegFormer] VRAM cap: {effective_limit_gb:.2f} GB "
            f"(requested {float(segformer_max_vram_gb):.2f} GB)"
        )
        _applied_vram_limit_bytes = limit_bytes
    except Exception as exc:
        print(f"[SegFormer] VRAM cap could not be applied: {exc}")
        _applied_vram_limit_bytes = limit_bytes


def _vram_budget_bytes(device):
    """Return the VRAM budget available to auto-sized CUDA inference."""
    import torch

    if not (hasattr(device, 'type') and device.type == 'cuda'):
        return None
    try:
        free_bytes = torch.cuda.mem_get_info(device)[0]
    except Exception:
        return None

    limit_bytes = _configured_vram_limit_bytes()
    if limit_bytes is None:
        return free_bytes
    return max(1, min(int(free_bytes), int(limit_bytes)))


def load_vegetation_model(device=None):
    """
    Load (once) the 9-class SegFormer-landcover vegetation model.
    Returns (model, processor, device).
    """
    global _model_veg, _processor_veg, _model, _processor
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    device = _get_device(device)
    if _model_veg is not None:
        return _model_veg, _processor_veg, device

    print(f"[SegFormer] Loading vegetation model ({_MODEL_VEG_REPO}) onto {device} …")
    _processor_veg = SegformerImageProcessor(
        do_resize=True,
        size={"height": segformer_patch_size, "width": segformer_patch_size},
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )
    _model_veg = SegformerForSemanticSegmentation.from_pretrained(_MODEL_VEG_REPO)
    if device.type == 'cuda':
        _model_veg = _model_veg.half()
    _model_veg = _model_veg.to(device).eval()
    # Keep legacy aliases in sync
    _model     = _model_veg
    _processor = _processor_veg
    print("[SegFormer] Vegetation model ready.")
    return _model_veg, _processor_veg, device


def load_building_model(device=None):
    """
    Load (once) the binary SegFormer building segmentation model.
    Classes: 0 = background, 1 = building.
    Returns (model, processor, device).
    """
    global _model_bld, _processor_bld
    from transformers import SegformerForSemanticSegmentation, AutoImageProcessor

    device = _get_device(device)
    if _model_bld is not None:
        return _model_bld, _processor_bld, device

    print(f"[SegFormer] Loading building model ({_MODEL_BLD_REPO}) onto {device} …")
    _processor_bld = AutoImageProcessor.from_pretrained(_MODEL_BLD_REPO)
    _model_bld = SegformerForSemanticSegmentation.from_pretrained(_MODEL_BLD_REPO)
    if device.type == 'cuda':
        _model_bld = _model_bld.half()
    _model_bld = _model_bld.to(device).eval()
    print("[SegFormer] Building model ready.")
    return _model_bld, _processor_bld, device


def load_model(device=None):
    """Legacy entry point — loads vegetation model. Returns (model, device)."""
    model, _, device = load_vegetation_model(device)
    return model, device


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def _infer_batch_size(device, patch_size, num_classes):
    """Return the number of patches to process per forward pass.

    In auto mode, uses the full currently available VRAM budget unless the
    user configured a smaller maximum VRAM limit.

    Falls back to 1 on CPU or if the query fails.
    """
    import torch
    if segformer_batch_size and segformer_batch_size > 0:
        return max(1, int(segformer_batch_size))
    if not (hasattr(device, 'type') and device.type == 'cuda'):
        return 1
    try:
        budget_bytes = _vram_budget_bytes(device)
        if budget_bytes is None:
            return 1
        # Conservative: SegFormer activations run ~40× raw input bytes at fp16.
        bytes_per_patch = 40 * 3 * patch_size * patch_size * 2
        return max(1, min(64, int(budget_bytes / bytes_per_patch)))
    except Exception:
        return 1


def _is_cuda_oom(exc):
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _autotune_batch_size(model, device, sample_patch, max_batch_size, patch_size, num_classes, mean, std):
    """Pick the fastest CUDA batch size up to max_batch_size for this model shape."""
    import time
    import torch
    import torch.nn.functional as F

    if segformer_batch_size and segformer_batch_size > 0:
        print(f"[SegFormer] Inference batch size: {max_batch_size} (configured)")
        return max_batch_size
    if not (hasattr(device, 'type') and device.type == 'cuda'):
        return max_batch_size

    try:
        model_dtype = next(model.parameters()).dtype
    except Exception:
        model_dtype = torch.float16
    # Empirically on the RTX 4080, very large batches can fill VRAM while
    # reducing throughput badly. Keep auto mode below that cliff; explicit
    # segformer_batch_size still lets the user force a larger value.
    max_batch_size = min(int(max_batch_size), 32)
    cache_key = (id(model), str(device), int(patch_size), int(num_classes),
                 str(model_dtype), int(max_batch_size))
    cached = _batch_benchmark_cache.get(cache_key)
    if cached:
        return cached

    base_candidates = (1, 2, 4, 8, 12, 16, 24, 32)
    candidates = [n for n in base_candidates if n <= max_batch_size]
    if max_batch_size not in candidates:
        candidates.append(max_batch_size)
    candidates = sorted(set(max(1, int(n)) for n in candidates))

    sample = np.ascontiguousarray(sample_patch)
    base = torch.from_numpy(sample).permute(2, 0, 1).unsqueeze(0)
    base = base.to(device=device, dtype=torch.float16)
    base = (base / 255.0 - mean) / std

    best_n = 1
    best_rate = 0.0
    results = []

    for n in candidates:
        try:
            print(f"[SegFormer] Autotune batch {n}/{max_batch_size} …", flush=True)
            batch = base.expand(n, -1, -1, -1).contiguous()
            # Warm up this shape once, then time one representative pass. This
            # costs a few seconds once per model, but avoids pathological
            # over-batching when VRAM is full but throughput is worse.
            with torch.inference_mode():
                logits = model(pixel_values=batch).logits
                up = F.interpolate(logits, size=(patch_size, patch_size),
                                   mode="bilinear", align_corners=False)
                _ = torch.softmax(up, dim=1)
            torch.cuda.synchronize(device)

            t0 = time.perf_counter()
            with torch.inference_mode():
                logits = model(pixel_values=batch).logits
                up = F.interpolate(logits, size=(patch_size, patch_size),
                                   mode="bilinear", align_corners=False)
                _ = torch.softmax(up, dim=1)
            torch.cuda.synchronize(device)
            elapsed = max(1e-6, time.perf_counter() - t0)
            rate = n / elapsed
            results.append(f"{n}:{rate:.1f}/s")
            print(f"[SegFormer] Autotune batch {n}: {rate:.1f} patches/s", flush=True)
            if rate > best_rate:
                best_rate = rate
                best_n = n
            del batch, logits, up, _
        except RuntimeError as exc:
            if _is_cuda_oom(exc):
                results.append(f"{n}:oom")
                print(f"[SegFormer] Autotune batch {n}: OOM", flush=True)
                torch.cuda.empty_cache()
                continue
            raise
        except Exception as exc:
            results.append(f"{n}:err")
            print(f"[SegFormer] Batch autotune skipped candidate {n}: {exc}", flush=True)
        finally:
            torch.cuda.empty_cache()

    best_n = max(1, min(best_n, max_batch_size))
    _batch_benchmark_cache[cache_key] = best_n
    if results:
        print(f"[SegFormer] Inference batch autotune: {'  '.join(results)}  -> {best_n}", flush=True)
    return best_n


def run_inference(model, device, img_rgb, processor=None):
    """
    Run patch-wise inference on a full-resolution RGB image.

    Works with any SegformerForSemanticSegmentation model.
    If processor is None, falls back to the legacy global _processor_veg.

    Patches overlap by segformer_overlap pixels on every side; their soft (softmax)
    probabilities are accumulated with a smooth 2-D weight window and averaged
    before argmax.  This eliminates hard square boundaries at patch edges.

    Patches are grouped into batches sized to available GPU VRAM for throughput.
    The model is run in its native dtype (FP16 on CUDA, FP32 on CPU).

    Returns a (H, W) class-index array.
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image as _Image

    if processor is None:
        processor = _processor_veg

    H, W        = img_rgb.shape[:2]
    P           = segformer_patch_size
    ovlp        = segformer_overlap
    stride      = P - 2 * ovlp
    num_classes = model.config.num_labels

    # Smooth 2-D weight window: raised-cosine tapering toward edges so the
    # centre of each patch contributes more than the overlap margins.
    _ramp = np.hanning(P).astype(np.float16)
    _win  = np.outer(_ramp, _ramp)           # (P, P) float16

    # Build grid of top-left corners, ensuring full image coverage
    rows = sorted(set(list(range(0, H - P + 1, stride)) + [max(0, H - P)]))
    cols = sorted(set(list(range(0, W - P + 1, stride)) + [max(0, W - P)]))

    batch_size = _infer_batch_size(device, P, num_classes)

    # Pre-allocate a contiguous uint8 batch buffer — filled in-place each iteration
    # so we avoid per-batch list appends and np.stack allocations.
    _buf = np.empty((batch_size, P, P, 3), dtype=np.uint8)

    # GPU-side ImageNet normalisation constants (float16) — allocated once here so
    # the forward path does a single fused GPU operation instead of calling the
    # SegformerImageProcessor (which allocates float32 on CPU then transfers).
    if device.type == 'cuda':
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        _gpu_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float16,
                                  device=device).view(1, 3, 1, 1)
        _gpu_std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float16,
                                  device=device).view(1, 3, 1, 1)
        try:
            # Keep the weighted soft-probability blend on GPU so each batch only
            # copies the final int8 class map back to CPU.
            _win_gpu = torch.from_numpy(_win).to(device=device, dtype=torch.float16)
            _accum_gpu = torch.zeros((num_classes, H, W), dtype=torch.float16, device=device)
            _wt_sum_gpu = torch.zeros((H, W), dtype=torch.float16, device=device)
            _gpu_accum_enabled = True
        except RuntimeError as exc:
            if _is_cuda_oom(exc):
                print("[SegFormer] GPU blend accumulator OOM; falling back to CPU blending.")
                torch.cuda.empty_cache()
                _gpu_accum_enabled = False
            else:
                raise
    else:
        _cpu_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        _cpu_std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        _gpu_accum_enabled = False

    if not _gpu_accum_enabled:
        # Soft-probability accumulator — float16 keeps memory usage low;
        # precision loss is negligible for argmax.
        accum  = np.zeros((num_classes, H, W), dtype=np.float16)
        wt_sum = np.zeros((H, W), dtype=np.float16)

    def _flush(n, meta):
        """Forward-pass one batch; accumulate soft probabilities."""
        if n == 0:
            return
        # uint8 numpy → GPU float16 → normalize (no PIL, no CPU float32 allocation)
        t = torch.from_numpy(_buf[:n]).permute(0, 3, 1, 2)  # (B,3,P,P) uint8
        if device.type == 'cuda':
            t = t.to(device=device, dtype=torch.float16)     # one transfer: uint8→GPU float16
            t = (t / 255.0 - _gpu_mean) / _gpu_std
        else:
            t = t.to(dtype=torch.float32) / 255.0
            t = (t - _cpu_mean) / _cpu_std
        logits = model(pixel_values=t).logits                # (B, C, Hm, Wm)
        up     = F.interpolate(logits, size=(P, P),
                               mode="bilinear", align_corners=False)
        probs  = torch.softmax(up, dim=1)
        if _gpu_accum_enabled:
            for i, (row, col, ph, pw, resized) in enumerate(meta):
                p = probs[i]
                if resized:
                    p = F.interpolate(p.unsqueeze(0), size=(ph, pw),
                                      mode="bilinear", align_corners=False).squeeze(0)
                    w_crop = _win_gpu[:ph, :pw]
                else:
                    w_crop = _win_gpu
                _accum_gpu[:, row:row + ph, col:col + pw].add_(p * w_crop)
                _wt_sum_gpu[row:row + ph, col:col + pw].add_(w_crop)
        else:
            probs_np = probs.cpu().to(torch.float16).numpy()  # (B,C,P,P) float16
            for i, (row, col, ph, pw, resized) in enumerate(meta):
                p = probs_np[i]
                if resized:
                    p = np.stack([
                        np.array(_Image.fromarray(p[c]).resize((pw, ph), _Image.BILINEAR))
                        for c in range(num_classes)
                    ])
                    w_crop = _win[:ph, :pw]
                else:
                    w_crop = _win
                accum[:, row:row + ph, col:col + pw] += p * w_crop
                wt_sum[row:row + ph, col:col + pw]   += w_crop

    first_patch = img_rgb[rows[0]:rows[0] + P, cols[0]:cols[0] + P]
    if first_patch.shape[0] != P or first_patch.shape[1] != P:
        first_patch = np.array(_Image.fromarray(first_patch).resize((P, P), _Image.BILINEAR))
    if device.type == 'cuda':
        tuned_batch_size = _autotune_batch_size(
            model, device, first_patch, batch_size, P, num_classes, _gpu_mean, _gpu_std
        )
        if tuned_batch_size != batch_size:
            batch_size = tuned_batch_size
            _buf = np.empty((batch_size, P, P, 3), dtype=np.uint8)
    else:
        print(f"[SegFormer] Inference batch size: {batch_size}")

    n_buf    = 0
    meta_buf = []

    with torch.inference_mode():
        for row in rows:
            for col in cols:
                patch = img_rgb[row:row + P, col:col + P]
                ph, pw = patch.shape[:2]
                if ph < 16 or pw < 16:
                    continue
                resized = ph < P or pw < P
                if resized:
                    arr = np.array(_Image.fromarray(patch).resize((P, P), _Image.BILINEAR))
                else:
                    arr = patch  # uint8 view; copied into _buf below
                _buf[n_buf] = arr
                meta_buf.append((row, col, ph, pw, resized))
                n_buf += 1
                if n_buf >= batch_size:
                    _flush(n_buf, meta_buf)
                    n_buf = 0
                    meta_buf.clear()
        _flush(n_buf, meta_buf)

    if _gpu_accum_enabled:
        _wt_sum_gpu.clamp_(min=1e-4)
        avg = _accum_gpu / _wt_sum_gpu.unsqueeze(0)
        max_prob, cls_map = avg.max(dim=0)
        return torch.where(
            max_prob >= segformer_confidence_threshold,
            cls_map,
            torch.full_like(cls_map, -1),
        ).to(torch.int8).cpu().numpy()

    wt_sum = np.maximum(wt_sum, np.float16(1e-4))  # avoid div-by-zero; stay float16
    avg    = accum / wt_sum                          # (C, H, W) float16 — sufficient for argmax
    max_prob = avg.max(axis=0)                       # (H, W)
    # Pixels whose best-class probability is below the threshold are left
    # unclassified (-1) and will be skipped by mask_to_polygons.
    return np.where(
        max_prob >= segformer_confidence_threshold,
        avg.argmax(axis=0),
        -1,
    ).astype(np.int8)   # int8 sufficient for classes -1..8; 4× smaller cache files


PATCH_SIZE = 512  # backwards-compat stub


# ─────────────────────────────────────────────────────────────────────────────
# Mask → polygon vectorisation
# ─────────────────────────────────────────────────────────────────────────────

def _contours_to_shapely(contour_list, hierarchy, latlon_fn, min_area_px):
    """
    Convert OpenCV contours + hierarchy to a list of Shapely Polygons in
    WGS84 lat/lon coordinates.

    latlon_fn(px, py) -> (lat, lon)
    """
    def px_ring_to_latlon(cnt):
        pts = cnt.reshape(-1, 2)
        ring = []
        for px, py in pts:
            lat, lon = latlon_fn(float(px), float(py))
            ring.append((lon, lat))  # Shapely uses (x=lon, y=lat)
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        return ring

    polys = []
    if hierarchy is None:
        return polys

    hier = hierarchy[0]  # shape (N, 4): next, prev, first_child, parent
    N = len(contour_list)

    for i in range(N):
        if hier[i][3] != -1:
            continue  # inner contour; handled as hole of its parent
        if cv2.contourArea(contour_list[i]) < min_area_px:
            continue

        exterior = px_ring_to_latlon(contour_list[i])
        if len(exterior) < 4:
            continue

        holes = []
        child = hier[i][2]
        while child != -1:
            hole_cnt = contour_list[child]
            if cv2.contourArea(hole_cnt) >= min_area_px:
                hole = px_ring_to_latlon(hole_cnt)
                if len(hole) >= 4:
                    holes.append(hole)
            child = hier[child][0]

        try:
            poly = geometry.Polygon(exterior, holes)
            poly = make_valid(poly)
            if poly.is_empty or poly.area == 0:
                continue
            # Expand MultiPolygons from make_valid back into individual entries
            if poly.geom_type == "MultiPolygon":
                polys.extend(poly.geoms)
            else:
                polys.append(poly)
        except Exception:
            continue

    return polys


def mask_to_polygons(class_map, target_classes, latlon_fn, min_area_px):
    """
    Extract polygons for each class in target_classes from a (H, W) class map.

    latlon_fn(px, py) -> (lat, lon)
    Returns dict  {class_idx: [Shapely Polygon, …]}
    """
    result = {}
    for cls in target_classes:
        binary = np.where(class_map == cls, 255, 0).astype(np.uint8)

        # Light morphological clean-up: remove isolated specks, close small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1
        )
        polys = _contours_to_shapely(contours, hierarchy, latlon_fn, min_area_px)

        # Simplify each polygon (Douglas-Peucker)
        simplified = []
        for p in polys:
            s = p.simplify(_SIMPLIFY_TOL, preserve_topology=True)
            if not s.is_empty and s.area > 0:
                simplified.append(s)

        if simplified:
            result[cls] = simplified

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Region / asset helpers
# ─────────────────────────────────────────────────────────────────────────────

def _region_from_latlon(lat, lon):
    """Coarse continent-level region string (kept for future per-region .for tuning)."""
    if 35 <= lat <= 72 and -25 <= lon <= 45:
        return "europe"
    if 15 <= lat <= 72 and -170 <= lon <= -50:
        return "namerica"
    return "default"


def _size_bucket(area_m2):
    """Facade size bucket for buildings at or above the object threshold."""
    if area_m2 < 500:
        return "medium"
    return "large"


def _building_height(area_m2):
    """Heuristic building height (metres) from footprint area."""
    if area_m2 < 50:
        return 3.0
    if area_m2 < 200:
        return 7.0
    if area_m2 < 1000:
        return 12.0
    return 20.0


def _area_m2(poly, lat):
    """Approximate polygon area in m² given its WGS84 geometry."""
    mpd_lat = 111_320.0
    mpd_lon = mpd_lat * cos(lat * pi / 180)
    return poly.area * mpd_lat * mpd_lon


# ─────────────────────────────────────────────────────────────────────────────
# DSF writing
# ─────────────────────────────────────────────────────────────────────────────

def _write_ring(f, ring_coords):
    pts = list(ring_coords)
    # Safety cap: if still over the limit after Shapely simplification, thin linearly
    if len(pts) > _MAX_RING_PTS:
        step = max(1, len(pts) // _MAX_RING_PTS)
        pts = pts[::step]
        if pts[0] != pts[-1]:
            pts.append(pts[0])
    f.write("BEGIN_WINDING\n")
    for lon, lat in pts:
        f.write(f"POLYGON_POINT {lon:.7f} {lat:.7f}\n")
    f.write("END_WINDING\n")


def write_text_dsf(lat, lon, txt_path, veg_polys_by_class, bld_polys):
    """
    Write a text DSF overlay file.

    veg_polys_by_class : {class_idx: [Shapely Polygon, …]}
    bld_polys          : [Shapely Polygon, …]

    Small-footprint buildings (< segformer_bld_obj_threshold_m2) are placed as
    SFD Global Autogen OBJECT entries at the polygon centroid with a
    randomly-assigned heading.  Larger buildings use BEGIN_POLYGON facades.
    """
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)

    tile_lat = lat + 0.5   # centre of the 1° tile
    tile_lon = lon + 0.5
    obj_pool = _sfd_obj_pool(tile_lat, tile_lon)

    # ── Geometry helper ───────────────────────────────────────────────────────
    def _iter_polygons(geom):
        if geom.geom_type == "Polygon":
            yield geom
        elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
            for g in geom.geoms:
                yield from _iter_polygons(g)

    # ── Pre-classify building polygons ────────────────────────────────────────
    # small_objs  : list of (poly, obj_path, heading_deg)  → OBJECT
    # large_facs  : list of (poly, bucket_str)             → BEGIN_POLYGON
    small_objs = []
    large_facs = []

    for raw in bld_polys:
        for poly in _iter_polygons(raw):
            poly = geometry.polygon.orient(poly, sign=1.0)
            if len(poly.exterior.coords) < 4:
                continue
            lat_c = poly.centroid.y
            area  = _area_m2(poly, lat_c)

            if area < segformer_bld_obj_threshold_m2:
                # Deterministic random: seed on centroid so reruns are stable
                random.seed(int(poly.centroid.x * 1e6) ^ int(poly.centroid.y * 1e6))
                obj_path = random.choice(obj_pool)
                heading  = random.uniform(0.0, 360.0)
                small_objs.append((poly, obj_path, heading))
            else:
                large_facs.append((poly, _size_bucket(area)))

    # ── Build separate polygon-def and object-def index tables ────────────────
    poly_defs  = []   # POLYGON_DEF paths (forests + large facades)
    poly_index = {}

    def _ensure_poly_def(path):
        if path not in poly_index:
            poly_index[path] = len(poly_defs)
            poly_defs.append(path)
        return poly_index[path]

    obj_defs  = []    # OBJECT_DEF paths (SFD suburban objects)
    obj_index = {}

    def _ensure_obj_def(path):
        if path not in obj_index:
            obj_index[path] = len(obj_defs)
            obj_defs.append(path)
        return obj_index[path]

    # Forest defs — one path per class, climate-selected by tile lat
    for cls in sorted(veg_polys_by_class.keys()):
        _ensure_poly_def(_for_path(cls, tile_lat))

    # Large facade defs
    for _, bucket in large_facs:
        _ensure_poly_def(_FAC_DEFS[bucket])

    # Small object defs (only paths actually used)
    for _, obj_path, _ in small_objs:
        _ensure_obj_def(obj_path)

    # ── Write file ────────────────────────────────────────────────────────────
    with open(txt_path, "w") as f:
        # Header (DSFTool adds A/800/DSF2 on compile)
        f.write("PROPERTY sim/planet earth\n")
        f.write("PROPERTY sim/overlay 1\n")
        f.write(f"PROPERTY sim/west  {lon}\n")
        f.write(f"PROPERTY sim/east  {lon + 1}\n")
        f.write(f"PROPERTY sim/south {lat}\n")
        f.write(f"PROPERTY sim/north {lat + 1}\n")

        for path in poly_defs:
            f.write(f"POLYGON_DEF {path}\n")
        for path in obj_defs:
            f.write(f"OBJECT_DEF {path}\n")

        # ── Forest polygons ───────────────────────────────────────────────────
        for cls, polys in veg_polys_by_class.items():
            fpath   = _for_path(cls, tile_lat)
            def_idx = poly_index[fpath]
            for raw in polys:
                for poly in _iter_polygons(raw):
                    poly = geometry.polygon.orient(poly, sign=1.0)
                    if len(poly.exterior.coords) < 4:
                        continue
                    density = _FOR_DENSITY_BY_CLASS.get(cls, _FOR_DENSITY)
                    f.write(f"BEGIN_POLYGON {def_idx} {density} 2\n")
                    _write_ring(f, poly.exterior.coords)
                    holes = sorted(poly.interiors,
                                   key=lambda r: abs(geometry.LinearRing(r.coords).length),
                                   reverse=True)
                    for interior in holes[:_MAX_HOLES]:
                        _write_ring(f, interior.coords)
                    f.write("END_POLYGON\n")

        # ── Large building facades ────────────────────────────────────────────
        for poly, bucket in large_facs:
            lat_c   = poly.centroid.y
            height  = _building_height(_area_m2(poly, lat_c))
            def_idx = poly_index[_FAC_DEFS[bucket]]
            f.write(f"BEGIN_POLYGON {def_idx} {height:.1f} 2\n")
            _write_ring(f, poly.exterior.coords)
            holes = sorted(poly.interiors,
                           key=lambda r: abs(geometry.LinearRing(r.coords).length),
                           reverse=True)
            for interior in holes[:_MAX_HOLES]:
                _write_ring(f, interior.coords)
            f.write("END_POLYGON\n")

        # ── Small building objects ────────────────────────────────────────────
        # DSF OBJECT format: OBJECT <def_idx> <lon> <lat> <heading_deg>
        for poly, obj_path, heading in small_objs:
            def_idx = obj_index[obj_path]
            clon    = poly.centroid.x
            clat    = poly.centroid.y
            f.write(f"OBJECT {def_idx} {clon:.7f} {clat:.7f} {heading:.1f}\n")

    return txt_path


# ─────────────────────────────────────────────────────────────────────────────
# DSFTool compilation
# ─────────────────────────────────────────────────────────────────────────────

def compile_dsf(txt_path, dsf_path):
    """Invoke DSFTool to compile txt_path → dsf_path. Returns True on success."""
    os.makedirs(os.path.dirname(dsf_path), exist_ok=True)
    cmd = [_dsftool, "-text2dsf", txt_path, dsf_path]
    result = subprocess.run(cmd, capture_output=True, text=True,
                             creationflags=_CREATE_NO_WINDOW)
    if result.returncode != 0:
        print(f"[SegFormer] DSFTool error:\n{result.stdout}\n{result.stderr}")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main tile pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _run_inference_on_canvas(img_rgb, model, processor, device):
    """Run overlapping patch inference on img_rgb. Returns (H,W) int32 class map."""
    print(f"[SegFormer] Running inference on {img_rgb.shape[1]}×{img_rgb.shape[0]} canvas …")
    return run_inference(model, device, img_rgb, processor)


def process_sfr_tile(lat, lon, build_dir=None, device=None,
                 do_vegetation=None, do_buildings=None):
    """
    Full pipeline for one Ortho4XP tile (lat, lon).

    When segformer_building_model == "dedicated":
      - Vegetation pass: SegFormer-landcover (_MODEL_VEG_REPO) → VEG_CLASSES
      - Building pass:   binary SegFormer    (_MODEL_BLD_REPO)  → class 1 = building
    When segformer_building_model == "landcover":
      - Single pass with SegFormer-landcover → VEG_CLASSES + CLASS_BUILDING

    Returns path to the output .dsf file, or None on failure.
    """
    # Fall back to module-level defaults when caller does not override
    if do_vegetation is None:
        do_vegetation = segformer_do_vegetation
    if do_buildings is None:
        do_buildings = segformer_do_buildings

    use_dedicated_bld = (do_buildings and segformer_building_model == "dedicated")
    use_landcover_bld = (do_buildings and segformer_building_model != "dedicated")

    if build_dir is None:
        build_dir = FNAMES.build_dir(lat, lon, "")

    tex_dir = os.path.join(build_dir, "textures")
    if not os.path.isdir(tex_dir):
        print(f"[SegFormer] No textures directory found: {tex_dir}")
        return None

    dds_files = [f for f in os.listdir(tex_dir) if f.lower().endswith(".dds")]
    if not dds_files:
        print(f"[SegFormer] No DDS files in {tex_dir}")
        return None

    device = _get_device(device)
    all_veg_by_class = {}
    all_bld_polys    = []
    canvas_size      = segformer_canvas_px  # 0 = legacy per-DDS mode

    # ── Load required models ──────────────────────────────────────────────────
    if do_vegetation or use_landcover_bld:
        model_veg, proc_veg, device = load_vegetation_model(device)
    if use_dedicated_bld:
        model_bld, proc_bld, device = load_building_model(device)

    if canvas_size > 0:
        # ── Stitched-canvas mode ──────────────────────────────────────────────
        img_rgb = _stitch_dds_canvas(tex_dir, lat, lon, canvas_size)
        if img_rgb is None:
            print("[SegFormer] Could not stitch DDS textures.")
            return None

        img_h, img_w = img_rgb.shape[:2]
        lat_c     = lat + 0.5
        m2_per_px = (111_320 / img_h) * (111_320 * cos(lat_c * pi / 180) / img_w)

        def latlon_fn(px, py):
            return (lat + 1 - py / img_h, lon + px / img_w)

        # ── Vegetation pass ───────────────────────────────────────────────────
        if do_vegetation or use_landcover_bld:
            print("[SegFormer] Vegetation pass …")
            veg_map = _run_inference_on_canvas(img_rgb, model_veg, proc_veg, device)

            if do_vegetation:
                veg = mask_to_polygons(
                    veg_map, VEG_CLASSES, latlon_fn,
                    min_area_px=max(segformer_min_veg_area_px, int(100 / m2_per_px))
                )
                for cls, polys in veg.items():
                    all_veg_by_class.setdefault(cls, []).extend(polys)

            if use_landcover_bld:
                bld = mask_to_polygons(
                    veg_map, (CLASS_BUILDING,), latlon_fn,
                    min_area_px=max(segformer_min_bld_area_px, int(10 / m2_per_px))
                )
                for polys in bld.values():
                    all_bld_polys.extend(polys)

        # ── Dedicated building pass ───────────────────────────────────────────
        if use_dedicated_bld:
            print("[SegFormer] Building pass …")
            bld_map = _run_inference_on_canvas(img_rgb, model_bld, proc_bld, device)
            bld = mask_to_polygons(
                bld_map, (1,), latlon_fn,   # class 1 = building in binary model
                min_area_px=max(segformer_min_bld_area_px, int(10 / m2_per_px))
            )
            for polys in bld.values():
                all_bld_polys.extend(polys)

    else:
        # ── Legacy per-DDS mode ───────────────────────────────────────────────
        for fname in sorted(dds_files):
            parsed = parse_dds_filename(fname)
            if parsed is None:
                continue
            til_y_top, til_x_left, provider, zl = parsed

            fpath = os.path.join(tex_dir, fname)
            print(f"[SegFormer]   Processing {fname} …")
            try:
                img_rgb = _load_dds(fpath)
            except Exception as e:
                print(f"[SegFormer]   Could not load {fname}: {e}")
                continue

            img_h, img_w = img_rgb.shape[:2]
            m2_per_px    = _approx_m2_per_px(til_y_top, img_h, zl)

            def _latlon_fn(px, py, _w=img_w, _h=img_h, _xl=til_x_left, _yt=til_y_top, _zl=zl):
                return _pixel_to_latlon(px, py, _w, _h, _xl, _yt, _zl)

            if do_vegetation or use_landcover_bld:
                veg_map = run_inference(model_veg, device, img_rgb, proc_veg)

                if do_vegetation:
                    veg = mask_to_polygons(
                        veg_map, VEG_CLASSES, _latlon_fn,
                        min_area_px=max(segformer_min_veg_area_px, int(100 / m2_per_px))
                    )
                    for cls, polys in veg.items():
                        all_veg_by_class.setdefault(cls, []).extend(polys)

                if use_landcover_bld:
                    bld = mask_to_polygons(
                        veg_map, (CLASS_BUILDING,), _latlon_fn,
                        min_area_px=max(segformer_min_bld_area_px, int(10 / m2_per_px))
                    )
                    for polys in bld.values():
                        all_bld_polys.extend(polys)

            if use_dedicated_bld:
                bld_map = run_inference(model_bld, device, img_rgb, proc_bld)
                bld = mask_to_polygons(
                    bld_map, (1,), _latlon_fn,
                    min_area_px=max(segformer_min_bld_area_px, int(10 / m2_per_px))
                )
                for polys in bld.values():
                    all_bld_polys.extend(polys)

    if not all_veg_by_class and not all_bld_polys:
        print("[SegFormer] No features detected — no DSF written.")
        return None

    # ── Output paths ──────────────────────────────────────────────────────────
    dest_dir = os.path.join(
        FNAMES.SFR_Bld_Overlay_dir,
        "Earth nav data",
        FNAMES.round_latlon(lat, lon),
    )
    short    = FNAMES.short_latlon(lat, lon)
    txt_path = os.path.join(dest_dir, short + "_ai.txt")
    dsf_path = os.path.join(dest_dir, short + ".dsf")

    print(f"[SegFormer] Writing DSF text to {txt_path} …")
    write_text_dsf(lat, lon, txt_path, all_veg_by_class, all_bld_polys)

    print(f"[SegFormer] Compiling DSF → {dsf_path} …")
    ok = compile_dsf(txt_path, dsf_path)
    if not ok:
        return None

    try:
        os.remove(txt_path)
    except OSError:
        pass

    print(f"[SegFormer] Done. Overlay at:\n  {dsf_path}")
    return dsf_path

