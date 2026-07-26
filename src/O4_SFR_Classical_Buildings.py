"""Classical (non-neural) building detection for the SFR building overlay.

Drop-in alternative to the trained YOLO OBB detector: consumes a full
orthophoto array and emits the same rectangular oriented-box detection
dicts that ``O4_SFR_Building_Overlay`` consumes downstream.

Pure numpy + OpenCV + stdlib — deliberately imports no ``O4_*`` module so
it runs standalone (no torch, no ultralytics).

Pipeline (every size derives from metres via ``m_per_px``):
  0. scale-normalise to a working ground-sample distance (downscale only)
  1. priors: vegetation / water / shadow masks, optional SegFormer landcover
  2. evidence: directional-min multi-scale top-hat (MBI-lite) + MSER votes
  3. candidate mask -> seam carving (dark inter-building lanes) ->
     connected components -> strict-rethreshold + watershed split of mats,
     pieces relabelled back into the label map
  4. one vectorised feature pass over all labels (bincount-based), then a
     light per-survivor loop: minAreaRect + gates + confidence in [0, 1]
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

ALGO_VERSION = 4
DEFAULT_CLASSICAL_CONF = 0.30

# ── Tables kept in sync with O4_SFR_Building_Overlay.py ─────────────────────
# Placement classes (BLD_CLASS_*): 1=tiny_residential .. 8=extra_large.
# Thresholds mirror _roof_fragment_class(); heights mirror
# DEFAULT_FACADE_HEIGHT_M.
_ROOF_CLASS_LIMITS = (
    (100.0, 16.0, 1),
    (190.0, 22.0, 2),
    (320.0, 30.0, 3),
    (650.0, 38.0, 4),
    (1100.0, 48.0, 5),
    (2200.0, 65.0, 6),
    (7000.0, 115.0, 7),
)
_EXTRA_LARGE_CLASS = 8
DEFAULT_FACADE_HEIGHT_M = {
    1: 3.5,
    2: 4.0,
    3: 4.0,
    4: 7.0,
    5: 9.0,
    6: 12.0,
    7: 8.0,
    8: 10.0,
}

# Landcover ids, sync with O4_SFR_Inference.CLASS_*.
VEG_BACKGROUND = 0
VEG_BARELAND = 1
VEG_RANGELAND = 2
VEG_DEVELOPED = 3
VEG_ROAD = 4
VEG_TREE = 5
VEG_WATER = 6
VEG_AGRICULTURE = 7
VEG_BUILDING = 8

# Texture stem "til_y_til_x_ProviderZL", sync with STD_RE in the overlay.
_STEM_RE = re.compile(r"^(\d+)_(\d+)_([A-Za-z][A-Za-z0-9_]*?)(\d{2})$")


@dataclass
class ClassicalParams:
    """Every tunable in one place. All metre-denominated; pixels are derived."""

    work_gsd: float = 0.8              # working ground-sample distance (m/px)

    # Evidence: directional top-hat
    tophat_widths_m: tuple = (5.0, 10.0, 20.0, 40.0, 60.0)
    tophat_kernel_cap_px: int = 121
    blackhat_weight: float = 0.8       # dark-roof channel weight
    # Kernels longer than this run on a half-res brightness image and the
    # response is upsampled — large-structure top-hat is smooth, so this is
    # near-lossless and roughly quarters the cost of the big scales.
    tophat_halfres_min_px: int = 25

    # Evidence: MSER votes
    mser_delta: int = 5
    mser_min_area_m2: float = 20.0
    mser_max_area_m2: float = 6000.0
    mser_max_variation: float = 0.5
    mser_min_votes: int = 2
    mser_polarity: str = "both"        # "both" | "bright" | "dark"
    # MSER cost is O(pixels); cap its input dimension and upsample the vote
    # mask. 2048 halves each axis at native ZL16 (4x faster) with negligible
    # recall loss on building-scale blobs.
    mser_max_dim: int = 2048

    # Candidate mask
    evid_floor: float = 0.12           # absolute lower bound on threshold
    evid_otsu_scale: float = 0.5       # t_low = max(floor, scale * otsu)
    open_max_wgsd: float = 1.5         # apply 3x3 opening only below this gsd
    close_m: float = 2.0

    # Seam carving: disconnect buildings from streets / each other along
    # lanes darker than the local mean. The blackhat guard protects compact
    # dark roofs (high directional-min blackhat) from being carved.
    seam_window_m: float = 24.0
    seam_dark_delta: float = 8.0
    seam_dark_guard: float = 10.0
    # Below this wgsd, open the seam mask so only corridors >= ~3 px wide
    # carve; 1-px roof-ridge shading must not shred roofs at ZL18/19.
    seam_open_max_wgsd: float = 1.2
    # Elongation carve for bright roads (max-min directional response at the
    # smallest scale). Disabled by default: it also bites long rowhouses.
    road_elong_delta: float = 0.0      # 0 disables

    # Merged-blob splitting
    split_min_area_m2: float = 800.0
    split_max_solidity: float = 0.80
    split_peak_dilate_m: float = 6.0
    split_min_dist_m: float = 2.5
    split_seed_area_m2: float = 250.0  # ~expected area per seed in big mats
    split_max_seeds: int = 4096
    # Components larger than this go through a strict re-threshold first.
    mat_rethreshold_m2: float = 50000.0
    mat_strict_scale: float = 2.0      # strict threshold = t_low * scale

    # Hard gates
    min_area_m2: float = 20.0
    max_area_m2: float = 9000.0
    min_side_m: float = 2.2
    max_side_m: float = 150.0
    max_aspect: float = 6.0

    # Priors / masks
    veg_exclude_erode_m: float = 8.0   # safety margin on landcover exclusion
    shadow_core_erode_m: float = 3.0
    # Blue-dominant pixels darker than this count as water; brighter ones are
    # blue metal roofs (ubiquitous in KR/CN/TW imagery) and stay candidates.
    water_max_v: float = 110.0
    prior_scores: dict = field(default_factory=lambda: {
        VEG_BUILDING: 1.0,
        VEG_DEVELOPED: 0.65,
        VEG_BARELAND: 0.45,
        VEG_ROAD: 0.45,
        VEG_RANGELAND: 0.30,
    })
    prior_default: float = 0.15

    # Confidence weights (renormalised when no landcover prior is available)
    w_evid: float = 0.30
    w_rect: float = 0.25
    w_edge: float = 0.15
    w_prior: float = 0.20
    w_shadow: float = 0.10
    edge_contrast_div: float = 40.0
    shadow_ring_scale: float = 2.5
    # Candidates whose ring sits in vegetation/water lose confidence:
    # conf *= 1 - ring_veg_penalty * veg_ring_fraction.
    ring_veg_penalty: float = 0.5

    # Orientation: snap boxes to the dominant local building-grid direction
    # (most roofs in a neighbourhood share a street-grid orientation, so the
    # aligned box beats minAreaRect's noise-driven rotation). The aligned box
    # is used unless its fill ratio is worse than minAreaRect by more than
    # ``orient_fill_tolerance`` — then the free rotation is kept.
    orient_window_m: float = 30.0
    orient_min_strength: float = 0.18   # neighbourhood dominant-bin share
    orient_self_min_strength: float = 0.22  # building's-own-edges share
    # When the building's own edges strongly fix the orientation, trust them
    # over minAreaRect even at some fill cost; when they're weak, require the
    # aligned box to fit nearly as well as the free rotation.
    orient_fill_tolerance: float = 0.10
    orient_self_fill_tolerance: float = 0.22

    # Coarse-resolution (ZL16) refinements
    small_blob_px: int = 16
    small_blob_min_wgsd: float = 1.5
    min_box_side_m: float = 4.0

    # Overlap suppression (greedy, occupancy-based). Keeps the highest
    # confidence box; drops a later box when this fraction of its area is
    # already claimed. 0 disables (pipeline runs its own suppression).
    nms_coverage: float = 0.40
    nms_occ_max_dim: int = 2048

    # Safety caps
    max_candidates: int = 60000


# ── Small helpers ────────────────────────────────────────────────────────────
def _px(metres: float, wgsd: float) -> int:
    return max(1, int(round(metres / max(wgsd, 1e-6))))


def _odd(k: int) -> int:
    k = int(k)
    return k if k % 2 == 1 else k + 1


def _ellipse(k: int):
    k = max(3, _odd(k))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _line_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Binary 1-px-wide line of ``length`` px through the kernel centre."""
    length = max(3, _odd(length))
    kernel = np.zeros((length, length), dtype=np.uint8)
    c = length // 2
    rad = math.radians(angle_deg)
    dx = math.cos(rad) * (length // 2)
    dy = math.sin(rad) * (length // 2)
    pt1 = (int(round(c - dx)), int(round(c - dy)))
    pt2 = (int(round(c + dx)), int(round(c + dy)))
    cv2.line(kernel, pt1, pt2, 1, 1)
    return kernel


def _roof_fragment_class(area_m2: float, max_side_m: float) -> int:
    # keep in sync with O4_SFR_Building_Overlay._roof_fragment_class
    for limit_area, limit_side, cls in _ROOF_CLASS_LIMITS:
        if area_m2 <= limit_area and max_side_m <= limit_side:
            return cls
    return _EXTRA_LARGE_CLASS


def _gtile_to_wgs84(til_x: int, til_y: int, zl: int):
    # keep in sync with O4_SFR_Building_Overlay._gtile_to_wgs84
    rat_x = til_x / (2 ** (zl - 1)) - 1
    rat_y = 1 - til_y / (2 ** (zl - 1))
    lon = rat_x * 180
    lat = 360 / math.pi * math.atan(math.exp(math.pi * rat_y)) - 90
    return lat, lon


def dds_bounds_for_stem(stem: str):
    """(lat_n, lat_s, lon_w, lon_e) for a texture stem like 25680_54080_BI16."""
    m = _STEM_RE.match(stem)
    if not m:
        return None
    til_y_top, til_x_left, zl = int(m.group(1)), int(m.group(2)), int(m.group(4))
    lat_n, lon_w = _gtile_to_wgs84(til_x_left, til_y_top, zl)
    lat_s, lon_e = _gtile_to_wgs84(til_x_left + 16, til_y_top + 16, zl)
    return lat_n, lat_s, lon_w, lon_e


def m_per_px_for_texture_stem(stem: str, img_w: int, img_h: int):
    """Approximate metres/pixel for a standard DDS texture stem.

    keep in sync with the mid-latitude maths in O4_SFR_Building_Overlay.run().
    """
    bounds = dds_bounds_for_stem(stem)
    if bounds is None:
        return None
    lat_n, lat_s, lon_w, lon_e = bounds
    mid_lat_rad = math.radians((lat_n + lat_s) / 2)
    lon_span_m = (lon_e - lon_w) * 111320 * math.cos(mid_lat_rad)
    lat_span_m = (lat_n - lat_s) * 110540
    return (lon_span_m / img_w + lat_span_m / img_h) / 2


def _rect_clip_obb_to_image(points, img_w, img_h, min_extent_px=0.5):
    """Clip an oriented box to the image while keeping it a true rectangle.

    keep in sync with O4_SFR_Building_Overlay._rect_clip_obb_to_image: clamping
    the four corners independently shears a rotated rectangle into an irregular
    quad, which downstream extrudes into a non-rectangular facade. Trim along
    the box's own axes instead. Returns clipped 4x2 corners in the input
    winding order, or None when nothing of usable size survives.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(4, 2)
    x_max = float(max(0, int(img_w) - 1))
    y_max = float(max(0, int(img_h) - 1))
    if (
        pts[:, 0].min() >= 0.0 and pts[:, 0].max() <= x_max and
        pts[:, 1].min() >= 0.0 and pts[:, 1].max() <= y_max
    ):
        return pts.astype(np.float32)

    center = pts.mean(axis=0)
    u = pts[1] - pts[0]
    u_len = float(np.linalg.norm(u))
    if u_len < 1e-6:
        return None
    u = u / u_len
    v = pts[3] - pts[0]
    v = v - float(np.dot(v, u)) * u
    v_len = float(np.linalg.norm(v))
    if v_len < 1e-6:
        return None
    v = v / v_len

    rel = pts - center
    s = rel @ u
    t = rel @ v
    s_lo, s_hi = float(s.min()), float(s.max())
    t_lo, t_hi = float(t.min()), float(t.max())
    min_extent = float(min_extent_px)

    constraints = (
        (0, 1.0, x_max), (0, -1.0, 0.0),
        (1, 1.0, y_max), (1, -1.0, 0.0),
    )
    for _ in range(16):
        worst = None
        for axis, sign, bound in constraints:
            a = sign * float(u[axis])
            b = sign * float(v[axis])
            reach = (
                sign * float(center[axis]) +
                max(s_lo * a, s_hi * a) +
                max(t_lo * b, t_hi * b)
            )
            violation = reach - bound
            if violation > 1e-9 and (worst is None or violation > worst[0]):
                worst = (violation, a, b)
        if worst is None:
            break
        violation, a, b = worst
        s_span = s_hi - s_lo
        t_span = t_hi - t_lo
        s_shrink = violation / abs(a) if abs(a) > 1e-9 else math.inf
        t_shrink = violation / abs(b) if abs(b) > 1e-9 else math.inf
        if s_shrink > s_span - min_extent:
            s_shrink = math.inf
        if t_shrink > t_span - min_extent:
            t_shrink = math.inf
        if not math.isfinite(s_shrink) and not math.isfinite(t_shrink):
            return None
        if s_shrink * t_span <= t_shrink * s_span:
            if a > 0.0:
                s_hi -= s_shrink
            else:
                s_lo += s_shrink
        else:
            if b > 0.0:
                t_hi -= t_shrink
            else:
                t_lo += t_shrink
    else:
        return None

    s_mid = (s_lo + s_hi) * 0.5
    t_mid = (t_lo + t_hi) * 0.5
    clipped = np.empty((4, 2), dtype=np.float64)
    for idx in range(4):
        s_idx = s_hi if s[idx] > s_mid else s_lo
        t_idx = t_hi if t[idx] > t_mid else t_lo
        clipped[idx] = center + s_idx * u + t_idx * v
    clipped[:, 0] = np.clip(clipped[:, 0], 0.0, x_max)
    clipped[:, 1] = np.clip(clipped[:, 1], 0.0, y_max)
    return clipped.astype(np.float32)


# ── Detection-dict builder (mirrors _yolo_obb_detection_from_points) ────────
def _detection_from_box_points(points, confidence, img_w, img_h, m_per_px,
                               features=None):
    """Build one pipeline-compatible detection dict from 4 box corners.

    keep in sync with O4_SFR_Building_Overlay._yolo_obb_detection_from_points:
    same clipping, heading and class/height derivation so facades and objects
    place identically downstream.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    valid = (
        (pts[:, 0] >= 0) & (pts[:, 0] < img_w) &
        (pts[:, 1] >= 0) & (pts[:, 1] < img_h)
    )
    center = pts.mean(axis=0)
    if not np.any(valid) and not (0 <= center[0] < img_w and 0 <= center[1] < img_h):
        return None

    clipped = _rect_clip_obb_to_image(pts, img_w, img_h)
    if clipped is None:
        return None
    area_px = abs(float(cv2.contourArea(clipped.astype(np.float32))))
    if area_px <= 1.0:
        return None
    center = clipped.mean(axis=0)

    edges = np.roll(clipped, -1, axis=0) - clipped
    edge_lengths = np.linalg.norm(edges, axis=1)
    long_edge_index = int(np.argmax(edge_lengths))
    long_vec = edges[long_edge_index]
    img_angle = math.degrees(math.atan2(float(long_vec[1]), float(long_vec[0])))
    max_side_px = float(edge_lengths[long_edge_index])
    min_side_px = float(edge_lengths[(long_edge_index + 1) % 4])
    if min_side_px > max_side_px:
        max_side_px, min_side_px = min_side_px, max_side_px
    heading = (img_angle + 90.0) % 180.0
    max_side_m = max_side_px * float(m_per_px)
    min_side_m = min_side_px * float(m_per_px)
    area_m2 = area_px * float(m_per_px) * float(m_per_px)
    placement_cls = _roof_fragment_class(area_m2, max_side_m)
    height_m = float(DEFAULT_FACADE_HEIGHT_M.get(placement_cls, 8.0))

    detection = {
        'points': clipped.tolist(),
        'center': [
            float(np.clip(center[0], 0, max(0, img_w - 1))),
            float(np.clip(center[1], 0, max(0, img_h - 1))),
        ],
        'heading': float(heading),
        'confidence': float(confidence),
        'model_class': int(placement_cls) - 1,
        'area_m2': float(area_m2),
        'max_side_m': float(max_side_m),
        'length_m': float(max_side_m),
        'width_m': float(min_side_m),
        'placement_class': int(placement_cls),
        'height_m': float(height_m),
    }
    if features is not None:
        detection['features'] = {k: round(float(v), 4) for k, v in features.items()}
    return detection


# ── Stage 1: priors ──────────────────────────────────────────────────────────
def _color_fallback_masks(work: np.ndarray, hsv: np.ndarray,
                          p: ClassicalParams):
    """(vegetation, water) uint8 masks from colour indices alone."""
    r = work[:, :, 0].astype(np.int16)
    g = work[:, :, 1].astype(np.int16)
    b = work[:, :, 2].astype(np.int16)
    exg = 2 * g - r - b
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    value = hsv[:, :, 2]
    vegetation = ((exg > 20) | ((hue >= 35) & (hue <= 90) & (sat > 60)))
    # Blue-dominant AND dark: bright blue is metal roofing, not water.
    water = (b > r + 25) & (b > g + 25) & (value < p.water_max_v)
    return vegetation.astype(np.uint8), water.astype(np.uint8)


def _shadow_mask(work: np.ndarray, hsv: np.ndarray) -> np.ndarray:
    value = hsv[:, :, 2]
    thresh = max(35.0, float(np.percentile(value, 20)))
    dark = value < thresh
    cool = work[:, :, 2].astype(np.int16) >= work[:, :, 0].astype(np.int16)
    return (dark & cool).astype(np.uint8)


def _build_priors(work, hsv, wgsd, veg_map, water_mask, p: ClassicalParams):
    """Return (suppress, exclude_raw, prior_score, shadow, has_prior).

    ``exclude_raw`` is the un-eroded vegetation/water exclusion — used for
    the ring-vegetation confidence penalty; ``suppress`` carries the eroded
    safety-margin version that gates candidate pixels.
    """
    h, w = work.shape[:2]
    shadow = _shadow_mask(work, hsv)
    shadow_core = cv2.erode(shadow, _ellipse(_px(p.shadow_core_erode_m, wgsd)))

    if veg_map is not None:
        veg = veg_map
        if veg.shape[:2] != (h, w):
            veg = cv2.resize(veg.astype(np.uint8), (w, h),
                             interpolation=cv2.INTER_NEAREST)
        else:
            veg = veg.astype(np.uint8)
        exclude_raw = ((veg == VEG_WATER) | (veg == VEG_TREE) |
                       (veg == VEG_AGRICULTURE)).astype(np.uint8)
        prior_score = np.full((h, w), p.prior_default, dtype=np.float32)
        for cls_id, score in p.prior_scores.items():
            prior_score[veg == cls_id] = score
        has_prior = True
    else:
        vegetation, water = _color_fallback_masks(work, hsv, p)
        exclude_raw = (vegetation | water).astype(np.uint8)
        prior_score = np.full((h, w), 0.5, dtype=np.float32)
        has_prior = False

    exclude = cv2.erode(exclude_raw, _ellipse(_px(p.veg_exclude_erode_m, wgsd)))

    suppress = exclude.astype(bool) | shadow_core.astype(bool)
    if water_mask is not None:
        wm = water_mask
        if wm.shape[:2] != (h, w):
            wm = cv2.resize((wm > 0).astype(np.uint8), (w, h),
                            interpolation=cv2.INTER_NEAREST)
        wm_bool = wm.astype(bool)
        suppress |= wm_bool
        exclude_raw = (exclude_raw.astype(bool) | wm_bool).astype(np.uint8)
    return suppress, exclude_raw, prior_score, shadow.astype(bool), has_prior


# ── Stage 2: evidence ────────────────────────────────────────────────────────
def _tophat_evidence(bright: np.ndarray, wgsd: float, p: ClassicalParams):
    """Directional-min multi-scale top-hat response.

    Returns (evidence, elong_small, dark_small):
      evidence    — float32, max over scales of min-over-direction responses
                    (bright channel fused with weighted blackhat channel);
      elong_small — float32, max-min directional spread at the smallest
                    scale (high on elongated bright features = roads);
      dark_small  — float32, directional-min blackhat at the smallest scale
                    (high on compact dark blobs = dark roofs).
    """
    angles = (0.0, 45.0, 90.0, 135.0)
    h, w = bright.shape[:2]
    bright_half = None
    evidence = None
    elong_small = None
    dark_small = None
    seen = set()
    for width_m in p.tophat_widths_m:
        length = min(p.tophat_kernel_cap_px, _odd(2 * _px(width_m, wgsd) + 1))
        if length in seen or length < 3:
            continue
        seen.add(length)
        # Large kernels run half-res then upsample (near-lossless, ~4x cheaper).
        use_half = length > p.tophat_halfres_min_px
        if use_half:
            if bright_half is None:
                bright_half = cv2.resize(bright, (w // 2, h // 2),
                                         interpolation=cv2.INTER_AREA)
            src = bright_half
            klen = _odd(length // 2)
        else:
            src = bright
            klen = length
        resp_bright = None
        resp_bright_max = None
        resp_dark = None
        for angle in angles:
            kernel = _line_kernel(klen, angle)
            th = cv2.morphologyEx(src, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
            bh = cv2.morphologyEx(src, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
            resp_bright = th if resp_bright is None else np.minimum(resp_bright, th)
            resp_bright_max = th if resp_bright_max is None else np.maximum(resp_bright_max, th)
            resp_dark = bh if resp_dark is None else np.minimum(resp_dark, bh)
        if use_half:
            resp_bright = cv2.resize(resp_bright, (w, h), interpolation=cv2.INTER_LINEAR)
            resp_dark = cv2.resize(resp_dark, (w, h), interpolation=cv2.INTER_LINEAR)
            resp_bright_max = cv2.resize(resp_bright_max, (w, h),
                                         interpolation=cv2.INTER_LINEAR)
        if elong_small is None:
            elong_small = resp_bright_max - resp_bright
            dark_small = resp_dark
        scale_resp = np.maximum(resp_bright, p.blackhat_weight * resp_dark)
        evidence = scale_resp if evidence is None else np.maximum(evidence, scale_resp)
    if evidence is None:
        shape = bright.shape
        return (np.zeros(shape, np.float32),) * 3
    return evidence, elong_small, dark_small


def _mser_vote_mask(gray: np.ndarray, wgsd: float, p: ClassicalParams):
    """Pixels covered by >= min_votes stable extremal regions (both polarities).

    Runs on a resolution-capped copy (``mser_max_dim``) and upsamples the
    vote mask — MSER is O(pixels), so this is the dominant ZL16 saving.
    """
    full_h, full_w = gray.shape[:2]
    long_side = max(full_h, full_w)
    if long_side > p.mser_max_dim:
        ms = p.mser_max_dim / long_side
        gray_ms = cv2.resize(gray, (max(1, int(full_w * ms)),
                                    max(1, int(full_h * ms))),
                             interpolation=cv2.INTER_AREA)
        wgsd_ms = wgsd / ms
    else:
        gray_ms = gray
        wgsd_ms = wgsd
    h, w = gray_ms.shape[:2]
    min_area = max(4, int(p.mser_min_area_m2 / (wgsd_ms * wgsd_ms)))
    max_area = max(min_area + 1, int(p.mser_max_area_m2 / (wgsd_ms * wgsd_ms)))
    mser = cv2.MSER_create(p.mser_delta, min_area, max_area, p.mser_max_variation)
    votes = np.zeros(h * w, dtype=np.int32)
    if p.mser_polarity == "bright":
        variants = (gray_ms,)
    elif p.mser_polarity == "dark":
        variants = (cv2.bitwise_not(gray_ms),)
    else:
        variants = (gray_ms, cv2.bitwise_not(gray_ms))
    for variant in variants:
        regions, _ = mser.detectRegions(variant)
        if not regions:
            continue
        # Never iterate per region: flatten all region pixels and bincount.
        pts = np.concatenate(regions, axis=0)
        idx = pts[:, 1].astype(np.int64) * w + pts[:, 0].astype(np.int64)
        votes += np.bincount(idx, minlength=h * w).astype(np.int32)
    mask = (votes.reshape(h, w) >= p.mser_min_votes).astype(np.uint8)
    if mask.shape != (full_h, full_w):
        mask = cv2.resize(mask, (full_w, full_h), interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool)


def _color_family_map(hsv: np.ndarray, shadow: np.ndarray) -> np.ndarray:
    """Quantise pixels into roof-colour families (int8 map).

    0 = none (grey mid-tones / ground), 1 = blue, 2 = red, 3 = green,
    4 = bright-white, 5 = dark. Distinctly-coloured roofs form compact
    connected components inside merged candidate mats, while grey ground
    stays family 0 — the basis for the colour-knife mat splitter.
    """
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    shadow_b = shadow.astype(bool)
    blue = (h >= 95) & (h <= 135) & (s >= 60) & (v >= 60)
    red = ((h <= 10) | (h >= 170)) & (s >= 70) & (v >= 60)
    green = (h >= 40) & (h <= 90) & (s >= 60) & (v >= 50)
    white = (v >= 200) & (s <= 50)
    dark = (v <= 90) & ~shadow_b
    return np.select(
        [blue, red, green, white, dark],
        [np.int8(1), np.int8(2), np.int8(3), np.int8(4), np.int8(5)],
        default=np.int8(0),
    )


def _seam_road_mask(bright, elong_small, dark_small, wgsd, p: ClassicalParams):
    """Pixels to carve out of the candidate mask.

    Dark seams: streets/inter-building lanes darker than the local mean,
    excluding compact dark blobs (dark roofs, guarded by blackhat response).
    Optional bright-road carve via directional elongation.
    """
    bright_f = bright.astype(np.float32)
    k = _odd(_px(p.seam_window_m, wgsd))
    local_mean = cv2.blur(bright_f, (k, k))
    seam = (bright_f < local_mean - p.seam_dark_delta) & (dark_small < p.seam_dark_guard)
    if wgsd < p.seam_open_max_wgsd:
        seam = cv2.morphologyEx(seam.astype(np.uint8), cv2.MORPH_OPEN,
                                _ellipse(3)).astype(bool)
    if p.road_elong_delta > 0:
        seam |= elong_small > p.road_elong_delta
    return seam


# ── Stage 3: candidates and splitting ────────────────────────────────────────
def _candidate_mask(evidence01, mser_mask, suppress, seam_road, wgsd,
                    p: ClassicalParams):
    nonzero = evidence01[evidence01 > 0]
    if nonzero.size:
        e8 = np.clip(nonzero * 255.0, 0, 255).astype(np.uint8).reshape(-1, 1)
        otsu_t, _ = cv2.threshold(e8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu01 = float(otsu_t) / 255.0
    else:
        otsu01 = 0.0
    t_low = max(p.evid_floor, p.evid_otsu_scale * otsu01)
    cand = (((evidence01 > t_low) | mser_mask) & ~suppress).astype(np.uint8)
    close_k = _px(p.close_m, wgsd)
    if close_k >= 3:
        cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, _ellipse(close_k))
    cand[seam_road] = 0
    if wgsd < p.open_max_wgsd:
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, _ellipse(3))
    return cand, t_low


def _solidity(mask: np.ndarray, area_px: float) -> float:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    hull_area = max(float(cv2.contourArea(hull)), 1.0)
    return float(area_px) / hull_area


def _watershed_markers(comp_mask, rgb_crop, area_m2, wgsd,
                       p: ClassicalParams):
    """Watershed a merged blob; returns int32 piece markers (>0) or None.

    Seeds: distance-transform peaks (geometric centres). Relief: the RGB
    image itself, so watershed lines follow colour boundaries (roof edges)
    rather than evidence gradients.

    Marker values are arbitrary distinct positive ints over comp pixels;
    0 = not part of any piece. None = no split possible (single seed).
    """
    dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 3)
    peak_kernel = _ellipse(_px(p.split_peak_dilate_m, wgsd))
    local_max = cv2.dilate(dist, peak_kernel)
    min_dist_px = p.split_min_dist_m / wgsd
    peaks = ((dist >= local_max - 1e-3) & (dist > min_dist_px)).astype(np.uint8)
    n_seeds, seed_labels = cv2.connectedComponents(peaks)
    if n_seeds <= 2:
        return None
    max_seeds = min(p.split_max_seeds,
                    max(64, int(area_m2 / max(p.split_seed_area_m2, 1.0))))
    if n_seeds - 1 > max_seeds:
        # keep the strongest seeds only (vectorised per-seed peak height)
        seed_max = np.zeros(n_seeds, dtype=np.float32)
        np.maximum.at(seed_max, seed_labels.ravel(), dist.ravel())
        thresh = np.partition(seed_max[1:], -max_seeds)[-max_seeds]
        weak = seed_max < thresh
        weak[0] = False
        seed_labels[weak[seed_labels]] = 0

    # Seeds keep their (distinct) component ids shifted by +1 so background
    # stays marker 1; watershed only needs distinct positive markers.
    markers = np.where(seed_labels > 0, seed_labels + 1, 0).astype(np.int32)
    markers[comp_mask == 0] = 1

    cv2.watershed(np.ascontiguousarray(rgb_crop), markers)

    pieces = np.where((markers >= 2) & (comp_mask > 0), markers, 0)
    if not pieces.any():
        return None
    return pieces


def _label_bboxes(values: np.ndarray):
    """Vectorised tight bounding boxes for all positive labels in ``values``.

    Returns (ids, x0, y0, w, h) arrays.
    """
    ys, xs = np.nonzero(values)
    if ys.size == 0:
        return None
    ids_px = values[ys, xs]
    order = np.argsort(ids_px, kind='stable')
    ids_sorted = ids_px[order]
    xs_sorted = xs[order]
    ys_sorted = ys[order]
    ids, starts = np.unique(ids_sorted, return_index=True)
    x0 = np.minimum.reduceat(xs_sorted, starts)
    x1 = np.maximum.reduceat(xs_sorted, starts)
    y0 = np.minimum.reduceat(ys_sorted, starts)
    y1 = np.maximum.reduceat(ys_sorted, starts)
    return ids, x0, y0, x1 - x0 + 1, y1 - y0 + 1


def _split_big_components(labels, num, stats, work, evidence01, color_family,
                          wgsd, t_low, p: ClassicalParams):
    """Relabel oversized/low-solidity components into building-scale pieces.

    Mutates ``labels`` in place: split components are erased and their
    pieces painted with fresh ids. Three knives, applied in order as long
    as a blob stays oversized/low-solidity:
      1. strict evidence re-threshold (city-scale mats only),
      2. colour-family split: distinctly-coloured roofs (blue/red/green/
         white/dark) become pieces, grey ground stays residual,
      3. distance-peak watershed over the RGB relief.
    Returns (new_num, piece_boxes) where piece_boxes maps new ids ->
    (x, y, w, h) bounding boxes.
    """
    areas_m2 = stats[:, cv2.CC_STAT_AREA].astype(np.float64) * wgsd * wgsd
    big_ids = np.nonzero((np.arange(num) > 0) &
                         (areas_m2 > p.split_min_area_m2))[0]
    # queue entries: (id, (x, y, w, h, area_px), stage) with stage the next
    # knife to try: 0 = re-threshold, 1 = edge carve, 2 = watershed.
    queue = [(int(i),
              (int(stats[i][0]), int(stats[i][1]), int(stats[i][2]),
               int(stats[i][3]), int(stats[i][4])),
              0)
             for i in big_ids]
    piece_boxes = {}
    next_id = int(num)

    def _register_pieces(values, base_x, base_y, requeue_stage):
        nonlocal next_id
        boxes = _label_bboxes(values)
        if boxes is None:
            return None
        ids, bx0, by0, bw, bh = boxes
        lut = np.zeros(int(values.max()) + 1, dtype=np.int64)
        lut[ids] = np.arange(next_id, next_id + ids.size)
        areas_px = np.bincount(values.ravel(), minlength=int(values.max()) + 1)
        sel = values > 0
        target = labels[base_y:base_y + values.shape[0],
                        base_x:base_x + values.shape[1]]
        target[sel] = lut[values[sel]]
        for k in range(ids.size):
            nid = int(next_id + k)
            box = (base_x + int(bx0[k]), base_y + int(by0[k]),
                   int(bw[k]), int(bh[k]))
            piece_boxes[nid] = box
            piece_area = int(areas_px[ids[k]])
            if (requeue_stage is not None and
                    float(piece_area) * wgsd * wgsd > p.split_min_area_m2):
                queue.append((nid, (*box, piece_area), requeue_stage))
        next_id += ids.size

    while queue:
        label_id, (x, y, w_box, h_box, area_px), stage = queue.pop()
        area_m2 = float(area_px) * wgsd * wgsd
        comp = (labels[y:y + h_box, x:x + w_box] == label_id).astype(np.uint8)

        if stage == 0:
            stage = 1
            if area_m2 > p.mat_rethreshold_m2:
                evid_crop = evidence01[y:y + h_box, x:x + w_box]
                strict_t = min(0.95, t_low * p.mat_strict_scale)
                strict = ((comp > 0) & (evid_crop > strict_t)).astype(np.uint8)
                n2, lab2 = cv2.connectedComponents(strict, 8)
                region = labels[y:y + h_box, x:x + w_box]
                region[comp > 0] = 0
                if n2 > 1:
                    _register_pieces(lab2.astype(np.int64), x, y,
                                     requeue_stage=1)
                continue

        if not (area_m2 > p.max_area_m2 or
                _solidity(comp, area_px) < p.split_max_solidity):
            continue  # solid and within size: keep as one candidate

        if stage == 1:
            fam_crop = color_family[y:y + h_box, x:x + w_box]
            comp_b = comp > 0
            pieces_values = np.zeros(comp.shape, dtype=np.int64)
            offset = 0
            claimed = np.zeros(comp.shape, dtype=bool)
            for fid in (1, 2, 3, 4, 5):
                fam_mask = ((fam_crop == fid) & comp_b).astype(np.uint8)
                if not fam_mask.any():
                    continue
                nf, labf = cv2.connectedComponents(fam_mask, 8)
                if nf <= 1:
                    continue
                sel = labf > 0
                pieces_values[sel] = labf[sel] + offset
                offset += nf - 1
                claimed |= sel
            if offset >= 2:
                residual = (comp_b & ~claimed).astype(np.uint8)
                nr, labr = cv2.connectedComponents(residual, 8)
                if nr > 1:
                    sel = labr > 0
                    pieces_values[sel] = labr[sel] + offset
                region = labels[y:y + h_box, x:x + w_box]
                region[comp_b] = 0
                _register_pieces(pieces_values, x, y, requeue_stage=2)
                continue
            stage = 2

        markers = _watershed_markers(
            comp, work[y:y + h_box, x:x + w_box], area_m2, wgsd, p,
        )
        if markers is None:
            continue
        region = labels[y:y + h_box, x:x + w_box]
        region[comp > 0] = 0
        _register_pieces(markers.astype(np.int64), x, y, requeue_stage=None)

    return next_id, piece_boxes


# ── Stage 4: orientation helpers ─────────────────────────────────────────────
def _build_snap_context(gray: np.ndarray):
    """Precompute the gradient field used for tiny-blob heading snapping."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    bins = np.minimum(
        (((np.degrees(np.arctan2(gy, gx)) + 90.0) % 90.0) / 5.0).astype(np.int32),
        17,
    )
    return mag, bins


def _orientation_from_hist(mag, bins, min_strength):
    """Peak of the mod-90 edge-orientation histogram in degrees, or None.

    ``mag``/``bins`` are flat arrays of gradient magnitude and 5-degree
    orientation bin (already folded into [0, 90)). Folding makes the two
    perpendicular roof-edge sets of a rectilinear building reinforce the
    same peak, so it stays robust on square footprints (unlike a structure
    tensor, which goes degenerate when the two edge sets balance).
    """
    mag_sum = float(mag.sum())
    if mag.size < 16 or mag_sum <= 1e-3:
        return None, 0.0
    hist = np.bincount(bins, weights=mag, minlength=18)
    best = int(np.argmax(hist))
    strength = float(hist[best]) / mag_sum
    if strength < min_strength:
        return None, strength
    return best * 5.0 + 2.5, strength


def _dominant_orientation_masked(snap_ctx, comp_bool, x, y):
    """Edge orientation from a building's OWN pixels (dilated to its edges)."""
    mag_map, bins_map = snap_ctx
    h, w = comp_bool.shape
    band = cv2.dilate(comp_bool.astype(np.uint8), _ellipse(5)).astype(bool)
    mag = mag_map[y:y + h, x:x + w][band]
    bins = bins_map[y:y + h, x:x + w][band]
    return _orientation_from_hist(mag, bins, 0.0)  # caller checks strength


def _dominant_orientation(snap_ctx, cx, cy, half, min_strength):
    """Dominant edge orientation in a fixed neighbourhood window, or None.

    Fallback for blobs whose own edges are too weak/blurry to orient; it
    borrows the local street-grid orientation from neighbours.
    """
    mag_map, bins_map = snap_ctx
    icy, icx = int(round(cy)), int(round(cx))
    ys = slice(max(0, icy - half), icy + half + 1)
    xs = slice(max(0, icx - half), icx + half + 1)
    angle, _ = _orientation_from_hist(
        mag_map[ys, xs].ravel(), bins_map[ys, xs].ravel(), min_strength
    )
    return angle


def _oriented_box(pts_xy, theta_deg):
    """Axis-aligned extent of points in a frame rotated by ``theta_deg``.

    Returns (corners 4x2 float32 in original frame, w, h, fill_denominator).
    ``pts_xy`` is an (N, 2) float array of pixel coordinates.
    """
    rad = math.radians(theta_deg)
    c, s = math.cos(rad), math.sin(rad)
    u = pts_xy[:, 0] * c + pts_xy[:, 1] * s
    v = -pts_xy[:, 0] * s + pts_xy[:, 1] * c
    umin, umax = float(u.min()), float(u.max())
    vmin, vmax = float(v.min()), float(v.max())
    w = umax - umin + 1.0
    h = vmax - vmin + 1.0
    corners_uv = np.array([[umin, vmin], [umax, vmin],
                           [umax, vmax], [umin, vmax]], dtype=np.float32)
    # rotate the (u, v) corners back into image space
    back = np.empty_like(corners_uv)
    back[:, 0] = corners_uv[:, 0] * c - corners_uv[:, 1] * s
    back[:, 1] = corners_uv[:, 0] * s + corners_uv[:, 1] * c
    return back, w, h, w * h


def _suppress_overlapping(detections, img_w, img_h, p: ClassicalParams):
    """Greedy occupancy NMS: keep strong boxes, drop overlapped duplicates.

    Operates on a downscaled occupancy grid so each accepted box stamps its
    footprint; a later (lower-confidence) box is dropped when ``nms_coverage``
    of its area is already claimed. Returns the surviving detections.
    """
    if not detections or p.nms_coverage <= 0:
        return detections
    scale = min(1.0, p.nms_occ_max_dim / max(img_w, img_h))
    occ = np.zeros((max(1, int(img_h * scale)) + 1,
                    max(1, int(img_w * scale)) + 1), dtype=np.uint8)
    occ_h, occ_w = occ.shape
    # Strongest first; ties by larger area so the enclosing box wins.
    order = sorted(range(len(detections)),
                   key=lambda i: (-detections[i]['confidence'],
                                  -detections[i]['area_m2']))
    keep = []
    for i in order:
        pts = np.asarray(detections[i]['points'], dtype=np.float32) * scale
        x0 = max(0, int(np.floor(pts[:, 0].min())))
        y0 = max(0, int(np.floor(pts[:, 1].min())))
        x1 = min(occ_w, int(np.ceil(pts[:, 0].max())) + 1)
        y1 = min(occ_h, int(np.ceil(pts[:, 1].max())) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        local = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillConvexPoly(local, (pts - [x0, y0]).astype(np.int32), 1)
        area = int(local.sum())
        if area <= 0:
            continue
        occ_sub = occ[y0:y1, x0:x1]
        overlap = int(np.count_nonzero(occ_sub & local))
        if overlap / area > p.nms_coverage:
            continue
        occ_sub[local.astype(bool)] = 1
        keep.append(i)
    keep.sort()
    return [detections[i] for i in keep]


def _confidence_weights(p: ClassicalParams, has_prior: bool) -> dict:
    total = p.w_evid + p.w_rect + p.w_edge + p.w_shadow + (
        p.w_prior if has_prior else 0.0
    )
    return {
        'evid': p.w_evid / total,
        'rect': p.w_rect / total,
        'edge': p.w_edge / total,
        'prior': (p.w_prior / total) if has_prior else 0.0,
        'shadow': p.w_shadow / total,
    }


# ── Public entry point ──────────────────────────────────────────────────────
def run_classical_building_inference(
    image,
    m_per_px,
    conf=DEFAULT_CLASSICAL_CONF,
    veg_map=None,
    water_mask=None,
    max_det=4000,
    params=None,
    return_debug=False,
):
    """Detect buildings in a full orthophoto with classical CV only.

    image: (H, W, 3) uint8 RGB array at native resolution.
    m_per_px: ground resolution of ``image`` in metres per pixel.
    conf: minimum confidence kept, in [0, 1].
    veg_map: optional (H, W) landcover map (OpenEarthMap ids, see VEG_*).
    water_mask: optional (H, W) uint8 mask, nonzero = mesh water.
    Returns detection dicts compatible with the YOLO OBB pipeline, sorted by
    (area_m2, -confidence) like _run_yolo_obb_inference.
    """
    p = params if params is not None else ClassicalParams()
    timings = {}
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] < 3:
        return ([], {}) if return_debug else []
    img_h, img_w = image.shape[:2]
    if m_per_px is None or m_per_px <= 0 or img_h < 32 or img_w < 32:
        return ([], {}) if return_debug else []
    image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # Stage 0: scale-normalise (downscale only).
    scale = min(1.0, float(m_per_px) / p.work_gsd)
    if scale < 0.999:
        work = cv2.resize(
            image,
            (max(32, int(round(img_w * scale))), max(32, int(round(img_h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        scale_x = work.shape[1] / img_w
        scale_y = work.shape[0] / img_h
    else:
        scale = 1.0
        work = image
        scale_x = scale_y = 1.0
    wgsd = float(m_per_px) / scale_x

    gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
    bright = work.max(axis=2)
    hsv = cv2.cvtColor(work, cv2.COLOR_RGB2HSV)

    # Stage 1: priors.
    t0 = time.perf_counter()
    suppress, exclude_raw, prior_score, shadow, has_prior = _build_priors(
        work, hsv, wgsd, veg_map, water_mask, p
    )
    timings['priors'] = time.perf_counter() - t0

    # Stage 2: evidence.
    t0 = time.perf_counter()
    evidence, elong_small, dark_small = _tophat_evidence(bright, wgsd, p)
    usable = evidence[~suppress]
    denom = float(np.percentile(usable, 99)) if usable.size else 0.0
    if denom <= 1e-6:
        denom = max(float(evidence.max()), 1e-6)
    evidence01 = np.clip(evidence / denom, 0.0, 1.0)
    timings['tophat'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    mser_mask = _mser_vote_mask(gray, wgsd, p)
    timings['mser'] = time.perf_counter() - t0

    # Stage 3: candidates, then split mats into building-scale pieces.
    t0 = time.perf_counter()
    seam_road = _seam_road_mask(bright, elong_small, dark_small, wgsd, p)
    cand, t_low = _candidate_mask(evidence01, mser_mask, suppress, seam_road,
                                  wgsd, p)
    num, labels, stats, _centroids = cv2.connectedComponentsWithStats(cand, 8)
    timings['candidates'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    piece_boxes = {}
    if num > 1:
        color_family = _color_family_map(hsv, shadow)
        num, piece_boxes = _split_big_components(
            labels, num, stats, work, evidence01, color_family, wgsd, t_low, p
        )
    timings['split'] = time.perf_counter() - t0

    detections = []
    conf_floor = float(conf)
    t0 = time.perf_counter()
    if num > 1:
        weights = _confidence_weights(p, has_prior)

        # Vectorised per-label features (one bincount pass each).
        flat_labels = labels.ravel()
        counts = np.bincount(flat_labels, minlength=num).astype(np.float64)
        counts_safe = np.where(counts == 0, 1.0, counts)
        areas_m2_all = counts * wgsd * wgsd
        mean_evid_v = np.bincount(flat_labels, weights=evidence01.ravel(),
                                  minlength=num) / counts_safe
        mean_prior_v = np.bincount(
            flat_labels, weights=prior_score.ravel().astype(np.float64),
            minlength=num) / counts_safe
        mean_r_v = np.bincount(flat_labels,
                               weights=work[:, :, 0].ravel().astype(np.float64),
                               minlength=num) / counts_safe
        mean_g_v = np.bincount(flat_labels,
                               weights=work[:, :, 1].ravel().astype(np.float64),
                               minlength=num) / counts_safe
        mean_b_v = np.bincount(flat_labels,
                               weights=work[:, :, 2].ravel().astype(np.float64),
                               minlength=num) / counts_safe
        mean_in_v = np.bincount(flat_labels,
                                weights=bright.ravel().astype(np.float64),
                                minlength=num) / counts_safe

        # 1-px outer ring per label via label dilation (ties go to the higher
        # label id — acceptable approximation for ring statistics).
        grow = cv2.dilate(labels.astype(np.float32), _ellipse(3)).ravel()
        ring_sel = (flat_labels == 0) & (grow > 0)
        ring_lab = grow[ring_sel].astype(np.int64)
        ring_counts = np.bincount(ring_lab, minlength=num).astype(np.float64)
        ring_safe = np.where(ring_counts == 0, 1.0, ring_counts)
        ring_bright_v = np.bincount(
            ring_lab, weights=bright.ravel()[ring_sel].astype(np.float64),
            minlength=num) / ring_safe
        ring_shadow_v = np.bincount(
            ring_lab, weights=shadow.ravel()[ring_sel].astype(np.float64),
            minlength=num) / ring_safe
        ring_veg_v = np.bincount(
            ring_lab,
            weights=exclude_raw.ravel()[ring_sel].astype(np.float64),
            minlength=num) / ring_safe
        no_ring = ring_counts == 0
        ring_bright_v[no_ring] = mean_in_v[no_ring]

        f_evid_v = np.clip(mean_evid_v, 0.0, 1.0)
        f_edge_v = np.clip(np.abs(mean_in_v - ring_bright_v) / p.edge_contrast_div,
                           0.0, 1.0)
        f_prior_v = np.clip(mean_prior_v, 0.0, 1.0)
        f_shadow_v = np.clip(ring_shadow_v * p.shadow_ring_scale, 0.0, 1.0)
        f_vegring_v = np.clip(ring_veg_v, 0.0, 1.0)
        veg_mult_v = np.maximum(0.0, 1.0 - p.ring_veg_penalty * f_vegring_v)
        base_v = (weights['evid'] * f_evid_v + weights['edge'] * f_edge_v +
                  weights['prior'] * f_prior_v + weights['shadow'] * f_shadow_v)
        # Upper bound on confidence (f_rect <= 1): lossless early reject.
        bound_v = (base_v + weights['rect']) * veg_mult_v

        # Roof-colour veto, adapted from O4_Building_Overlay._is_roof_color:
        # blue-dominant only counts as water when dark (blue metal roofs are
        # bright), see ClassicalParams.water_max_v.
        exg_v = 2 * mean_g_v - mean_r_v - mean_b_v
        color_ok_v = ~(
            (exg_v > 15) |
            ((mean_b_v > mean_r_v + 25) & (mean_b_v > mean_g_v + 25) &
             (mean_in_v < p.water_max_v)) |
            ((mean_r_v + mean_g_v + mean_b_v) / 3 < 25)
        )

        live = (np.arange(num) > 0) & (counts > 0)
        area_ok = (areas_m2_all >= p.min_area_m2) & (areas_m2_all <= p.max_area_m2)
        keep_sel = live & area_ok & color_ok_v & (bound_v >= conf_floor)
        keep_ids = np.nonzero(keep_sel)[0]
        counters = {
            'labels': int(np.count_nonzero(live)),
            'area_reject': int(np.count_nonzero(live & ~area_ok)),
            'color_reject': int(np.count_nonzero(live & area_ok & ~color_ok_v)),
            'bound_reject': int(np.count_nonzero(
                live & area_ok & color_ok_v & (bound_v < conf_floor))),
            'side_reject': 0,
            'aspect_reject': 0,
            'conf_reject': 0,
            'emitted': 0,
        }
        if keep_ids.size > p.max_candidates:
            order = np.argsort(-bound_v[keep_ids])
            keep_ids = keep_ids[order[:p.max_candidates]]

        min_side_px_small = p.min_box_side_m / wgsd
        orient_half = max(3, _px(p.orient_window_m, wgsd) // 2)
        snap_ctx = _build_snap_context(gray)
        for label_id in keep_ids:
            if label_id in piece_boxes:
                x, y, w_box, h_box = piece_boxes[label_id]
            else:
                x, y, w_box, h_box = stats[label_id][:4]
            area_px = counts[label_id]
            comp = (labels[y:y + h_box, x:x + w_box] == label_id).astype(np.uint8)
            pts = cv2.findNonZero(comp)
            if pts is None:
                continue
            pts_xy = pts.reshape(-1, 2).astype(np.float32)
            pts_xy[:, 0] += x
            pts_xy[:, 1] += y
            (rcx, rcy), (rw, rh), rangle = cv2.minAreaRect(pts)
            # findNonZero yields pixel centres: compensate the half-pixel
            # border so a w*h-pixel blob maps to a w*h-pixel box.
            rw, rh = rw + 1.0, rh + 1.0
            mar_fill = float(area_px) / max(rw * rh, 1.0)
            cx_full, cy_full = rcx + x, rcy + y

            # Orientation, in priority order:
            #  1. the building's OWN edge pixels (most accurate when strong),
            #  2. the local street-grid neighbourhood (for weak/tiny blobs),
            # each accepted only if the aligned box fits acceptably vs the
            # free minAreaRect. Otherwise keep minAreaRect.
            comp_bool = comp.astype(bool)
            self_dom, self_str = _dominant_orientation_masked(
                snap_ctx, comp_bool, x, y)
            box = None
            if self_dom is not None and self_str >= p.orient_self_min_strength:
                obox, ow, oh, odenom = _oriented_box(pts_xy, self_dom)
                if (float(area_px) / max(odenom, 1.0)
                        >= mar_fill - p.orient_self_fill_tolerance):
                    box = obox
                    rw, rh = ow + 1.0, oh + 1.0
            if box is None:
                nb_dom = _dominant_orientation(snap_ctx, cx_full, cy_full,
                                               orient_half, p.orient_min_strength)
                if nb_dom is not None:
                    obox, ow, oh, odenom = _oriented_box(pts_xy, nb_dom)
                    if (float(area_px) / max(odenom, 1.0)
                            >= mar_fill - p.orient_fill_tolerance):
                        box = obox
                        rw, rh = ow + 1.0, oh + 1.0
            if box is None:
                box = cv2.boxPoints(((cx_full, cy_full), (rw, rh), rangle))

            small_blob = (area_px < p.small_blob_px and
                          wgsd >= p.small_blob_min_wgsd)
            if small_blob and (rw < min_side_px_small or rh < min_side_px_small):
                # grow tiny boxes about their centre to a sane minimum
                ctr = box.mean(axis=0)
                grow = max(min_side_px_small / max(min(rw, rh), 1e-3), 1.0)
                box = (box - ctr) * grow + ctr
                rw, rh = max(rw, min_side_px_small), max(rh, min_side_px_small)
            side_long_m = max(rw, rh) * wgsd
            side_short_m = min(rw, rh) * wgsd
            if side_short_m < p.min_side_m or side_long_m > p.max_side_m:
                counters['side_reject'] += 1
                continue
            if side_long_m / max(side_short_m, 1e-6) > p.max_aspect:
                counters['aspect_reject'] += 1
                continue
            fill = float(area_px) / max(rw * rh, 1.0)
            f_rect = min(1.0, max(0.0, (fill - 0.45) / 0.5))
            confidence = float(
                (base_v[label_id] + weights['rect'] * f_rect) * veg_mult_v[label_id]
            )
            if confidence < conf_floor:
                counters['conf_reject'] += 1
                continue
            features = {
                'evid': f_evid_v[label_id],
                'rect': f_rect,
                'edge': f_edge_v[label_id],
                'prior': f_prior_v[label_id],
                'shadow': f_shadow_v[label_id],
                'veg_ring': f_vegring_v[label_id],
            }
            full_box = np.asarray(box, dtype=np.float32)
            full_box[:, 0] /= scale_x
            full_box[:, 1] /= scale_y
            detection = _detection_from_box_points(
                full_box, confidence, img_w, img_h, m_per_px,
                features=features,
            )
            if detection is not None:
                counters['emitted'] += 1
                detections.append(detection)
    else:
        counters = {}
    timings['scoring'] = time.perf_counter() - t0

    detections = [d for d in detections if d['confidence'] >= conf_floor]
    t0 = time.perf_counter()
    n_before = len(detections)
    detections = _suppress_overlapping(detections, img_w, img_h, p)
    timings['nms'] = time.perf_counter() - t0
    if len(detections) > max_det:
        detections.sort(key=lambda d: -d['confidence'])
        detections = detections[:max_det]
    # keep in sync with _run_yolo_obb_inference output ordering
    detections.sort(key=lambda d: (d['area_m2'], -d['confidence']))

    if return_debug:
        debug = {
            'wgsd': wgsd,
            'scale': scale_x,
            't_low': t_low,
            'has_prior': has_prior,
            'counters': {**counters, 'nms_dropped': n_before - len(detections)},
            'timings': {k: round(v, 3) for k, v in timings.items()},
            'evidence01': evidence01,
            'mser_mask': mser_mask.astype(np.uint8) * 255,
            'seam_road': seam_road.astype(np.uint8) * 255,
            'suppress': suppress.astype(np.uint8) * 255,
            'shadow': shadow.astype(np.uint8) * 255,
            'candidates': cand * 255,
            'prior_score': (prior_score * 255).astype(np.uint8),
        }
        return detections, debug
    return detections
