"""Relief from real heights for the terrain material maps (X-Plane 12).

The material maps (O4_Terrain_Materials) carry a relief normal that Step 3
estimates from the fine detail of the orthophoto.  After the overlays, this
stage adds the slopes of the real surface: building footprints at their
HeightNet height (every zoom level) and the canopy-height model's tree canopy
(ZL18/19 textures, where the vegetation overlay runs it).  Each tree stand and
building then gets a lit side toward the sun and a dark side away from it,
plus a broad band on that side which deepens as the sun gets lower
(O4_Terrain_Materials.height_slopes).

The overlays hand their heights over as sidecars in the tile's overlay cache
folder, written whatever their cache settings (O4_RELIEF_SIDECARS=1, set by
O4_Overlay_Pipeline):

  <stem>_relief_dets.pkl     the building overlay's filtered YOLO detections
                            (texture pixels; heights are predicted here for
                            all of them, not only the ones it places)
  <stem>_relief_canopy.npy  the vegetation overlay's canopy height (m)

With overlay caches on, the overlays' own <stem>_yolo_obb.pkl and
<stem>_vegcanopy.npy serve as well.  run() rewrites the material map of every
land texture that has heights, stamps it with HEIGHT_RELIEF_VERSION and
deletes the sidecars.  Step 3 keeps such a map until its texture changes;
the overlays then bring the heights back.

Runs in the overlay .venv (torch, cv2).
"""
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import O4_Terrain_Materials as TMAT

SIDECAR_ENV = "O4_RELIEF_SIDECARS"
BLD_SUFFIX = "_relief_dets.pkl"
CANOPY_SUFFIX = "_relief_canopy.npy"
MIN_HEIGHT_M = 1.0          # textures whose surface never reaches this keep their map


def sidecars_wanted():
    return os.environ.get(SIDECAR_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _stem(texture_file_name):
    return os.path.splitext(os.path.basename(texture_file_name))[0]


def save_building_sidecar(cache_dir, texture_file_name, detections, img_w, img_h):
    """The building overlay's detections of one texture (texture pixel coordinates)."""
    if not (cache_dir and sidecars_wanted()):
        return
    keep = []
    for det in detections or ():
        pts = det.get("points") if isinstance(det, dict) else None
        if pts is None:
            continue
        # Only plain fields: HeightNet's inputs (class, confidence, box) and its result.
        keep.append({k: v for k, v in det.items()
                     if isinstance(v, (int, float, str, bool, list, tuple, np.ndarray, np.generic))})
    path = os.path.join(cache_dir, _stem(texture_file_name) + BLD_SUFFIX)
    try:
        with open(path + ".tmp", "wb") as f:
            pickle.dump({"w": int(img_w), "h": int(img_h), "detections": keep}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(path + ".tmp", path)
    except Exception:            # never let the hand-over break the overlay
        try:
            os.remove(path + ".tmp")
        except OSError:
            pass


def save_canopy_sidecar(cache_dir, texture_file_name, canopy_maps):
    """The vegetation overlay's canopy height of one texture (channel 0 of CanopyModel.maps)."""
    if not (cache_dir and sidecars_wanted()) or canopy_maps is None:
        return
    path = os.path.join(cache_dir, _stem(texture_file_name) + CANOPY_SUFFIX)
    try:
        np.save(path, np.asarray(canopy_maps[0], np.float16))
    except (OSError, IndexError, ValueError):
        pass


def _load_building_detections(cache_dir, stem):
    for name, key in ((stem + BLD_SUFFIX, "detections"), (stem + "_yolo_obb.pkl", "detections")):
        path = os.path.join(cache_dir, name)
        if os.path.isfile(path):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                dets = data.get(key) if isinstance(data, dict) else None
                if isinstance(dets, dict):                     # metadata form
                    dets = dets.get("detections")
                if dets is not None:
                    return [dict(d) for d in dets if isinstance(d, dict) and d.get("points") is not None]
            except Exception:
                pass
    return None


def _load_canopy(cache_dir, stem):
    for name in (stem + CANOPY_SUFFIX, stem + "_vegcanopy.npy"):
        path = os.path.join(cache_dir, name)
        if os.path.isfile(path):
            try:
                arr = np.load(path)
                return np.asarray(arr[0] if arr.ndim == 3 else arr, np.float32)
            except Exception:
                pass
    return None


def building_raster(detections, heights, shape):
    """Footprints filled at their height (taller ones last), float32 ``shape``."""
    import cv2
    raster = np.zeros(shape, np.float32)
    order = sorted(range(len(detections)), key=lambda i: float(heights[i]) if np.isfinite(heights[i]) else -1)
    for i in order:
        h = float(heights[i])
        if not np.isfinite(h) or h <= 0:
            continue
        poly = np.round(np.asarray(detections[i]["points"], np.float64).reshape(-1, 2)).astype(np.int32)
        cv2.fillPoly(raster, [poly], h)
    return raster


def land_textures(build_dir):
    """Texture names of the land .ter files that use our material map, whose map exists."""
    ter_dir, tex_dir = os.path.join(build_dir, "terrain"), os.path.join(build_dir, "textures")
    out = []
    if not os.path.isdir(ter_dir):
        return out
    for name in sorted(os.listdir(ter_dir)):
        if not name.lower().endswith(".ter"):
            continue
        try:
            with open(os.path.join(ter_dir, name), "r", newline="") as f:
                lines, _ = TMAT._split_ter(f.read())
        except OSError:
            continue
        base = TMAT.ter_base_texture(lines)
        if not (TMAT.is_land_ter(name, lines) and base and base.startswith("../textures/")):
            continue
        tex = base[len("../textures/"):]
        if TMAT.ter_lines(tex)[0] not in (l.strip() for l in lines):
            continue
        if tex not in out and os.path.isfile(os.path.join(tex_dir, TMAT.map_file_name(tex))):
            out.append(tex)
    return out


class _HeightNetHeights:
    """Building heights the way the building overlay predicts them (HeightNet + regional priors)."""

    def __init__(self, checkpoint, lat, lon, log):
        self.checkpoint, self.lat, self.lon, self.log = checkpoint, lat, lon, log
        self.model = None
        self.tried = False

    def __call__(self, img, detections, m_per_px):
        """HeightNet heights of all ``detections`` (NaN where there is none): the
        overlay's where it predicted them, predicted here for the rest.  YOLO's own
        height_m is only a per-class placeholder, so height_source decides."""
        missing = [d for d in detections if d.get("height_source") != "heightnet"]
        if missing:
            self._predict(img, missing, m_per_px)
        return np.array([float(d["height_m"]) if d.get("height_source") == "heightnet" else np.nan
                         for d in detections])

    def _predict(self, img, detections, m_per_px):
        import O4_Building_Overlay as B
        import O4_Height_Model as HM
        if not self.tried:
            self.tried = True
            try:
                import torch
                ckpt = self.checkpoint or HM.default_checkpoint_path()
                self.model = HM.load_height_model(ckpt, device="cuda" if torch.cuda.is_available() else "cpu")
                self.region = B._natural_asset_region(self.lat + 0.5, self.lon + 0.5)
            except Exception as exc:
                self.log(f"   [Relief] HeightNet unavailable ({exc}); buildings left out")
                self.model = None
        if self.model is not None:
            B._apply_heightnet_to_detections(self.model, img, detections, m_per_px, region=self.region)


def run(build_dir, cache_dir, lat, lon, relief, converter, height_checkpoint=None,
        workers=4, log=print, height_fn=None, should_stop=None):
    """Rewrite the material maps of a tile's textures with relief from real heights.

    Returns {"made", "flat", "none", "failed"} counts.  ``height_fn(img, detections,
    m_per_px)`` -> heights overrides HeightNet (tests)."""
    import cv2
    from PIL import Image

    stats = dict(made=0, flat=0, none=0, failed=0)
    tex_dir = os.path.join(build_dir, "textures")
    if not os.path.isdir(cache_dir):
        return stats
    handed = sum(1 for n in os.listdir(cache_dir) if n.endswith(BLD_SUFFIX) or n.endswith(CANOPY_SUFFIX))
    height_fn = height_fn or _HeightNetHeights(height_checkpoint, lat, lon, log)
    t0 = time.time()
    size = TMAT.map_size(16)
    workers = max(1, int(workers))
    jobs = []

    def write(tex, rgb, heights):
        try:
            out = os.path.join(tex_dir, TMAT.map_file_name(tex))
            rgba = TMAT.material_map(rgb, relief, heights=heights,
                                     m_per_px=TMAT.texture_m_per_px(tex, size))
            return TMAT.encode_dds(rgba, out, converter, relief, heights=TMAT.HEIGHT_RELIEF_VERSION)
        except Exception:
            return False

    def settle(keep):
        while len(jobs) > keep:                  # bounds the maps waiting in memory
            stats["made" if jobs.pop(0).result() else "failed"] += 1

    textures = land_textures(build_dir)
    with ThreadPoolExecutor(workers) as pool:
        for tex in textures:
            if should_stop and should_stop():
                break
            stem = _stem(tex)
            dets = _load_building_detections(cache_dir, stem)
            canopy = _load_canopy(cache_dir, stem)
            if not dets and canopy is None:
                stats["none"] += 1
                continue
            try:
                img = np.asarray(Image.open(os.path.join(tex_dir, tex)).convert("RGB"))
            except Exception:
                stats["failed"] += 1
                continue
            layers = []
            if dets:
                mpp = TMAT.texture_m_per_px(tex, img.shape[1]) or 1.0
                heights = height_fn(img, dets, mpp)
                layers.append(cv2.resize(building_raster(dets, heights, img.shape[:2]), (size, size),
                                         interpolation=cv2.INTER_AREA))
            if canopy is not None:
                layers.append(cv2.resize(canopy, (size, size), interpolation=cv2.INTER_AREA))
            surface = np.maximum.reduce(layers)
            if float(surface.max()) < MIN_HEIGHT_M:
                stats["flat"] += 1
                continue
            rgb = TMAT.load_rgb(img, size)
            img = None
            jobs.append(pool.submit(write, tex, rgb, surface))
            settle(2 * workers)
        settle(0)
    for name in os.listdir(cache_dir):
        if name.endswith(BLD_SUFFIX) or name.endswith(CANOPY_SUFFIX):
            try:
                os.remove(os.path.join(cache_dir, name))
            except OSError:
                pass
    log(f" *Relief from real heights: {stats['made']} material maps made"
        + (f", {stats['flat']} without anything tall" if stats["flat"] else "")
        + (f", {stats['none']} without heights" if stats["none"] else "")
        + (f", {stats['failed']} failed" if stats["failed"] else "")
        + f" ({handed} height files from the overlays, {len(textures)} textures with a material map,"
        + f" {time.time() - t0:.0f}s).")
    return stats
