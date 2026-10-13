"""Shared SegFormer inference helpers for the vegetation and building overlays.

Loads the landcover SegFormer (and the optional dedicated building model),
runs tiled inference over orthophoto DDS textures, and compiles overlay DSF
text with DSFTool. Model forward passes can be offloaded to a remote GPU host
through ``remote_client`` (see O4_Remote_GPU).
"""

from O4_Overlay_Logging import verbose_print
import os
import re
import sys
import subprocess

_CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
import numpy as np
from math import atan, exp, pi, floor

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
    # Running outside the full Ortho4XP stack (e.g. on a remote GPU host).
    # dsftool_path must be passed explicitly to any function that needs it.
    FNAMES = None
    _dsftool = None

# ── Model repositories ───────────────────────────────────────────────────────
# Inference model: 9-class landcover SegFormer (tree/rangeland/agriculture/buildings/…)
_MODEL_VEG_REPO = "nave1616/SegFormer-landcover-FT"

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

# ── X-Plane asset paths ───────────────────────────────────────────────────────
# Facade .fac — XP12 lib paths confirmed from 1000 autogen/library.txt.
# Using generic lib/buildings/facades paths that work globally.
_FAC_DEFS = {
    "medium": "lib/buildings/facades/generic/mid_classic_01.fac",
    "large":  "lib/buildings/facades/industrial/warehouse_01_45x45.fac",
}

# ── Module-level tunables (overridable from tile config / O4_Cfg_Vars) ────────
segformer_patch_size       = 512     # pixels fed to the model per forward pass
segformer_overlap          = 64      # pixel overlap between adjacent patches (blending)
segformer_max_vram_gb      = 0.0     # 0 = uncapped; otherwise cap auto mode to this many GB
segformer_batch_size       = 0       # 0 = recommended default batch size
# Fixed on purpose: a runtime autotune was removed for picking slower batches
# from noisy timings, and raising this changes the batch dimension fed to
# cuDNN, which can flip borderline fp16 argmax pixels in the shared _veg.npy
# cache. Power users can override via the segformer_batch_size tile config.
segformer_default_cuda_batch_size = 8
segformer_use_amp          = os.environ.get("O4_SEGFORMER_AMP", "").strip() == "1"
segformer_channels_last    = os.environ.get("O4_SEGFORMER_CHANNELS_LAST", "").strip() == "1"
segformer_confidence_threshold   = 0.5     # minimum softmax probability for a pixel to be assigned
# Tree support (vegetation only): pixels the class map leaves out although the
# model leans towards tree — tree is the top class but under the confidence
# threshold, or tree is a strong runner-up behind rangeland or water (dark
# canopy at ZL16 reads as water). Kept apart from the class map, which the
# building overlay shares, and cached beside it as <stem>_vegtree.npy.
TREE_SUPPORT_MIN_PROB = 0.30


def tree_support_enabled():
    """O4_VEG_TREE_RULE=0 turns the vegetation tree-support mask off."""
    value = os.environ.get("O4_VEG_TREE_RULE")
    return value is None or value.strip().lower() not in {"0", "false", "no", "off"}


def tree_support_cache_path(veg_cache_path):
    return veg_cache_path[:-len('_veg.npy')] + '_vegtree.npy'


def save_array_atomic(path, array):
    """np.save under a private name, then rename: readers never see half a file.

    The vegetation overlay can read class maps while the building overlay is
    still writing them (overlays run side by side).
    """
    part = f"{path}.{os.getpid()}.part"
    with open(part, 'wb') as handle:
        np.save(handle, array)
    os.replace(part, path)


def save_tree_support(path, mask):
    """Store a boolean mask bit-packed (1/8 of the class map's size)."""
    mask = np.asarray(mask, dtype=bool)
    save_array_atomic(path, np.concatenate([
        np.asarray(mask.shape, dtype=np.uint32).view(np.uint8),
        np.packbits(mask, axis=None),
    ]))


def load_tree_support(path, shape):
    """Return the cached mask for a class map of ``shape``, or None."""
    try:
        raw = np.load(path)
        h, w = (int(v) for v in raw[:8].view(np.uint32))
        if (h, w) != tuple(shape):
            return None
        return np.unpackbits(raw[8:], count=h * w).reshape(h, w).astype(bool)
    except Exception:
        return None

# ── Remote inference offload (optional, session-scoped) ──────────────────────
# Set by O4_Remote_GPU.activate() when the user opted to run model inference
# on a remote GPU host for this run. When set, the loaders hand back a
# lightweight marker instead of loading torch models locally, and
# run_inference() ships the image to the remote worker (which executes this
# very same run_inference code on its own GPU). None = fully local, the
# default — every remote branch below is a no-op then.
remote_client = None


class RemoteModelHandle:
    """Marker returned by the model loaders when inference is offloaded."""

    def __init__(self, kind):
        self.kind = kind


# ── Lazy model handles ────────────────────────────────────────────────────────
_model_veg     = None
_processor_veg = None
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


# ─────────────────────────────────────────────────────────────────────────────
# DDS loading
# ─────────────────────────────────────────────────────────────────────────────


def _load_dds(path):
    """
    Load a DDS texture file and return an (H, W, 3) uint8 RGB numpy array.
    Uses Pillow (10+) which has native DDS/DXT support.
    """
    from PIL import Image
    with Image.open(path) as img:
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
    return arr


def load_dds_or_none(path, log_prefix="[SegFormer]", display_name=None):
    """
    Best-effort DDS loader for batch workflows.

    Returns None when the texture cannot be opened or decoded so callers can
    skip the bad tile instead of aborting the entire overlay run.
    """
    try:
        return _load_dds(path)
    except Exception as exc:
        name = display_name or os.path.basename(path)
        print(f"{log_prefix}   Could not load {name}: {exc} -- skipping")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DDS filename parsing
# ─────────────────────────────────────────────────────────────────────────────

# Standard format:  {til_y_top}_{til_x_left}_{provider_code}{zoomlevel}.dds
#   e.g. 112256_218048_BI18.dds
_DDS_STD_RE    = re.compile(r"^(\d+)_(\d+)_([A-Za-z][A-Za-z0-9_]*)(\d{2})\.dds$",
                             re.IGNORECASE)
_MASK_TEXTURE_RE = re.compile(r"^\d+_\d+_ZL\d{2}(?:\.[A-Za-z0-9]+)?$", re.IGNORECASE)


def is_mask_texture_name(name):
    """
    Return True for Ortho4XP mask texture sidecars such as 123_456_ZL19.png.
    """
    return bool(_MASK_TEXTURE_RE.match(os.path.basename(name)))


def orthophoto_tile_dir_for_tex_dir(tex_dir, lat, lon):
    """Return the cached Orthophotos directory corresponding to a tile textures dir."""
    o4xp_root = os.path.dirname(os.path.dirname(os.path.dirname(tex_dir)))
    lat_i = int(lat)
    lon_i = int(lon)
    lat_g = int(floor(lat / 10)) * 10
    lon_g = int(floor(lon / 10)) * 10
    lat_s = f"{'+' if lat_i >= 0 else '-'}{abs(lat_i):02d}"
    lon_s = f"{'+' if lon_i >= 0 else '-'}{abs(lon_i):03d}"
    lat_gs = f"{'+' if lat_g >= 0 else '-'}{abs(lat_g):02d}"
    lon_gs = f"{'+' if lon_g >= 0 else '-'}{abs(lon_g):03d}"
    return os.path.join(
        o4xp_root, "Orthophotos", f"{lat_gs}{lon_gs}", f"{lat_s}{lon_s}"
    )


def stale_dds_caches(tex_dir, names, cache_paths):
    """Cache files of textures whose DDS was rebuilt after they were cached.

    The per-texture caches are keyed by texture name and settings, not by the
    pixels, so a rebuilt DDS (colour correction, a different provider, cloud
    fill) would otherwise be overlaid from detections on the old imagery.
    Returns (paths to remove, number of textures concerned).
    """
    out, n = [], 0
    for name in names:
        try:
            built = os.stat(os.path.join(tex_dir, name)).st_mtime_ns
        except OSError:
            continue
        paths = [p for p in cache_paths(name) if os.path.exists(p)]
        try:
            if any(os.stat(p).st_mtime_ns < built for p in paths):
                out.extend(paths)
                n += 1
        except OSError:
            continue
    return out, n


def collect_source_texture_files(tex_dir, lat, lon):
    """
    Return source texture names for overlay inference.

    Tile-local DDS files are authoritative. Cached orthophotos are used only as
    a fallback for workflows that cleaned the tile textures but kept downloads.
    """
    dds_files = []
    if os.path.isdir(tex_dir):
        try:
            dds_files = [
                name for name in os.listdir(tex_dir)
                if _DDS_STD_RE.match(name) and not is_mask_texture_name(name)
            ]
        except OSError:
            dds_files = []
    if dds_files:
        verbose_print(f"Using {len(dds_files)} DDS textures from {tex_dir}", flush=True)
        return dds_files, "dds", None

    ortho_dir = orthophoto_tile_dir_for_tex_dir(tex_dir, lat, lon)
    orthophoto_files = []
    if os.path.isdir(ortho_dir):
        for _root, _dirs, names in os.walk(ortho_dir):
            for name in names:
                if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                if is_mask_texture_name(name):
                    continue
                candidate = os.path.splitext(name)[0] + ".dds"
                if _DDS_STD_RE.match(candidate) and not is_mask_texture_name(candidate):
                    orthophoto_files.append(candidate)
    if orthophoto_files:
        print(
            "Tile DDS textures missing; using "
            f"{len(orthophoto_files)} original cached orthophotos from {ortho_dir}",
            flush=True,
        )
        return orthophoto_files, "orthophoto", ortho_dir

    raise FileNotFoundError(
        f"No DDS textures found at {tex_dir!r} and no cached orthophotos "
        f"found at {ortho_dir!r}"
    )


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
        verbose_print(
            f"[SegFormer] VRAM cap: {effective_limit_gb:.2f} GB "
            f"(requested {float(segformer_max_vram_gb):.2f} GB)"
        )
        _applied_vram_limit_bytes = limit_bytes
    except Exception as exc:
        print(f"[SegFormer] VRAM cap could not be applied: {exc}")
        _applied_vram_limit_bytes = limit_bytes


def _should_use_cuda_amp(device):
    """Return True when CUDA autocast should wrap model inference."""
    return (
        segformer_use_amp
        and hasattr(device, 'type')
        and device.type == 'cuda'
    )


def _prepare_cuda_inference_tensor(t, torch):
    """Apply the preferred CUDA memory format to an NCHW tensor."""
    if segformer_channels_last:
        return t.contiguous(memory_format=torch.channels_last)
    return t


def _optimize_model_for_inference(model, device):
    """Apply CUDA inference optimizations in one place."""
    import torch

    model = model.to(device).eval()
    if not (hasattr(device, 'type') and device.type == 'cuda'):
        return model

    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if segformer_channels_last:
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception as exc:
            print(f"[SegFormer] channels-last model layout disabled: {exc}")

    return model


def load_vegetation_model(device=None):
    """
    Load (once) the shared 9-class SegFormer-landcover inference model.
    Returns (model, processor, device).
    """
    global _model_veg, _processor_veg
    if remote_client is not None:
        return RemoteModelHandle('veg'), None, device
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    device = _get_device(device)
    if _model_veg is not None:
        return _model_veg, _processor_veg, device

    print(f"[SegFormer] Loading inference model ({_MODEL_VEG_REPO}) onto {device} …")
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
    _model_veg = _optimize_model_for_inference(_model_veg, device)
    print("[SegFormer] Inference model ready.")
    return _model_veg, _processor_veg, device


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def _infer_batch_size(device, patch_size, num_classes):
    """Return the number of patches to process per forward pass.

    In default mode, use a conservative fixed CUDA batch size. The old
    run-time autotune path was expensive and noisy enough to choose slower
    batches on real tiles.

    Falls back to 1 on CPU or if the query fails.
    """
    if segformer_batch_size and segformer_batch_size > 0:
        return max(1, int(segformer_batch_size))
    if not (hasattr(device, 'type') and device.type == 'cuda'):
        return 1
    return max(1, int(segformer_default_cuda_batch_size))


def _is_cuda_oom(exc):
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def is_cuda_oom(exc):
    """Public alias for callers outside this module."""
    return _is_cuda_oom(exc)


_last_batch_size_message = None


def _print_batch_size_once(message):
    """Print the batch-size line only when it changes, not once per texture."""
    global _last_batch_size_message
    if message != _last_batch_size_message and verbose_print(message):
        _last_batch_size_message = message


def _autotune_batch_size(model, device, sample_patch, max_batch_size, patch_size, num_classes, mean, std):
    """Return the selected batch size.

    This used to benchmark several candidate batches at runtime. On real tiles
    that cost more than it saved and could select slower batches due to noisy
    one-pass timing, so the default path now uses a fixed recommended batch.
    """
    if segformer_batch_size and segformer_batch_size > 0:
        _print_batch_size_once(f"[SegFormer] Inference batch size: {max_batch_size} (configured)")
        return max_batch_size
    _print_batch_size_once(f"[SegFormer] Inference batch size: {max_batch_size} (default)")
    return max_batch_size


def run_inference(model, device, img_rgb, processor=None, tree_support=False):
    """
    Run patch-wise inference on a full-resolution RGB image.

    Works with any SegformerForSemanticSegmentation model.
    If processor is None, falls back to the legacy global _processor_veg.

    Patches overlap by segformer_overlap pixels on every side; their soft (softmax)
    probabilities are accumulated with a smooth 2-D weight window and averaged
    before argmax.  This eliminates hard square boundaries at patch edges.

    Patches are grouped into batches sized to available GPU VRAM for throughput.
    The model is run in its native dtype (FP16 on CUDA, FP32 on CPU).

    Returns a (H, W) class-index array; with ``tree_support`` a tuple of that
    array and the boolean tree-support mask (None when inferred remotely).
    """
    if remote_client is not None and isinstance(model, RemoteModelHandle):
        cls_map = remote_client.segformer_infer(model.kind, img_rgb)
        return (cls_map, None) if tree_support else cls_map
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
            t = _prepare_cuda_inference_tensor(t, torch)
        else:
            t = t.to(dtype=torch.float32) / 255.0
            t = (t - _cpu_mean) / _cpu_std
        if _should_use_cuda_amp(device):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(pixel_values=t).logits        # (B, C, Hm, Wm)
                up     = F.interpolate(logits, size=(P, P),
                                       mode="bilinear", align_corners=False)
                probs  = torch.softmax(up, dim=1)
        else:
            logits = model(pixel_values=t).logits            # (B, C, Hm, Wm)
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
        _print_batch_size_once(f"[SegFormer] Inference batch size: {batch_size}")

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
        out = torch.where(
            max_prob >= segformer_confidence_threshold,
            cls_map,
            torch.full_like(cls_map, -1),
        ).to(torch.int8).cpu().numpy()
        if not tree_support:
            return out
        weak = (
            ((cls_map == CLASS_TREE) & (max_prob < segformer_confidence_threshold))
            | ((avg[CLASS_TREE] >= TREE_SUPPORT_MIN_PROB)
               & ((cls_map == CLASS_RANGELAND) | (cls_map == CLASS_WATER)))
        )
        return out, weak.cpu().numpy()

    wt_sum = np.maximum(wt_sum, np.float16(1e-4))  # avoid div-by-zero; stay float16
    avg    = accum / wt_sum                          # (C, H, W) float16 — sufficient for argmax
    max_prob = avg.max(axis=0)                       # (H, W)
    # Pixels whose best-class probability is below the threshold are left
    # unclassified (-1).
    cls_map = avg.argmax(axis=0)
    out = np.where(
        max_prob >= segformer_confidence_threshold,
        cls_map,
        -1,
    ).astype(np.int8)   # int8 sufficient for classes -1..8; 4× smaller cache files
    if not tree_support:
        return out
    weak = (
        ((cls_map == CLASS_TREE) & (max_prob < segformer_confidence_threshold))
        | ((avg[CLASS_TREE] >= TREE_SUPPORT_MIN_PROB)
           & ((cls_map == CLASS_RANGELAND) | (cls_map == CLASS_WATER)))
    )
    return out, weak


# ─────────────────────────────────────────────────────────────────────────────
# DSFTool compilation
# ─────────────────────────────────────────────────────────────────────────────

def compile_dsf(txt_path, dsf_path):
    """Invoke DSFTool to compile txt_path → dsf_path. Returns True on success."""
    os.makedirs(os.path.dirname(dsf_path), exist_ok=True)
    cmd = [_dsftool, "-text2dsf", txt_path, dsf_path]
    result = subprocess.run(cmd, capture_output=True, text=True,
                             creationflags=_CREATE_NO_WINDOW)
    if result.stdout.strip():
        output = print if result.returncode else verbose_print
        output(f"[SegFormer] DSFTool stdout:\n{result.stdout}")
    if result.stderr.strip():
        print(f"[SegFormer] DSFTool stderr:\n{result.stderr}")
    if result.returncode != 0:
        return False
    return True


