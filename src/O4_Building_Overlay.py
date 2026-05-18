"""
O4_Building_Overlay.py  –  Generate X-Plane 12 building (facade) overlays
                           from Ortho4XP tile imagery.

Pipeline
--------
1. Load the ortho tile image (reuses O4_Veg_Overlay._load_tile_image).
2. Build a vegetation mask (reuses O4_Veg_Overlay._build_vegetation_mask)
   and suppress green pixels so only non-vegetated structure candidates remain.
3. Run Canny edge detection on the non-green, non-water region.
4. Find contours → approximate each to a polygon.
5. Filter by shape:
   - Area within [bld_min_area_px, bld_max_area_px]
   - At least bld_rect_threshold fraction of interior angles close to 90°
   - Aspect ratio not too elongated (not a road or runway)
6. Roof colour filter: reject pixels dominated by vegetation-green or water-blue.
7. Assign a facade .fac definition and height based on footprint area + lat/lon.
8. Write a text DSF overlay (no exclusion zones) and compile with DSFTool.

All processing uses OpenCV + NumPy — no heavy AI.
Requires:  opencv-python, numpy, Pillow.
"""

import os
import sys
import subprocess
import time
from math import cos, pi, sqrt

import numpy as np
from PIL import Image

import O4_File_Names as FNAMES
import O4_UI_Utils as UI
from O4_Veg_Overlay import (
    _load_tile_image,
    _build_vegetation_mask,
    _px_to_latlon,
    _compile_dsf,
    _load_and_downsample,
    _dsf_dest,
)

# ---------------------------------------------------------------------------
# DSFTool path (mirrors O4_Overlay_Utils.py)
# ---------------------------------------------------------------------------
if "dar" in sys.platform:
    _dsftool = os.path.join(FNAMES.Utils_dir, "mac", "DSFTool")
elif "win" in sys.platform:
    _dsftool = os.path.join(FNAMES.Utils_dir, "win", "DSFTool.exe")
else:
    _dsftool = os.path.join(FNAMES.Utils_dir, "lin", "DSFTool")

# ---------------------------------------------------------------------------
# Module-level tunables — set from O4_Cfg_Vars at runtime
# ---------------------------------------------------------------------------
bld_overlay_enabled      = False
bld_min_area_px          = 64      # ignore contours smaller than this (pixels²)
bld_max_area_px          = 40000   # ignore very large blobs (likely fields/car parks)
bld_rect_threshold       = 0.5     # fraction of angles that must be ~90° to count as rect
bld_max_aspect_ratio     = 6.0     # width/height > this → probably a road, skip

# ---------------------------------------------------------------------------
# Facade definitions  –  (fac_path, typical_height_m)
# ---------------------------------------------------------------------------
# Keyed by (region, size_class).  region: 'eu', 'us', 'generic'.
# size_class: 'small' (< 200 m²), 'medium' (200-2000 m²), 'large' (> 2000 m²).
_FAC_DEFS = {
    ("eu",      "small"):   ("lib/g10/residential.fac",  9.0),
    ("eu",      "medium"):  ("lib/g10/commercial.fac",  14.0),
    ("eu",      "large"):   ("lib/g10/industrial.fac",   7.0),
    ("us",      "small"):   ("lib/g10/residential.fac",  8.0),
    ("us",      "medium"):  ("lib/g10/commercial.fac",  12.0),
    ("us",      "large"):   ("lib/g10/industrial.fac",   6.0),
    ("generic", "small"):   ("lib/g10/residential.fac",  8.0),
    ("generic", "medium"):  ("lib/g10/commercial.fac",  12.0),
    ("generic", "large"):   ("lib/g10/industrial.fac",   6.0),
}

_HEIGHT_VARIANCE = 2.0   # ± random offset added to height for variety


def _region_for_latlon(lat: float, lon: float) -> str:
    """Very rough continental region classifier."""
    if 34 <= lat <= 71 and -11 <= lon <= 45:
        return "eu"
    if 24 <= lat <= 60 and -130 <= lon <= -60:
        return "us"
    return "generic"


def _size_class(area_m2: float) -> str:
    if area_m2 < 200:
        return "small"
    if area_m2 < 2000:
        return "medium"
    return "large"


def _area_px_to_m2(area_px: float, img_w: int, img_h: int,
                   tile_lat: float) -> float:
    """Convert pixel area to approximate square metres."""
    deg_per_px_lon = 1.0 / img_w
    deg_per_px_lat = 1.0 / img_h
    m_per_px_lon = deg_per_px_lon * cos(tile_lat * pi / 180) * 111320
    m_per_px_lat = deg_per_px_lat * 111320
    return area_px * m_per_px_lon * m_per_px_lat


# ---------------------------------------------------------------------------
# Building candidate detection
# ---------------------------------------------------------------------------

def _angles_at_vertices(approx: np.ndarray) -> list:
    """
    Return interior angles (degrees) at each vertex of the polygon approximation.
    approx is an (N, 1, 2) OpenCV contour array.
    """
    pts = approx[:, 0, :]   # shape (N, 2)
    n = len(pts)
    angles = []
    for i in range(n):
        p_prev = pts[(i - 1) % n].astype(float)
        p_curr = pts[i].astype(float)
        p_next = pts[(i + 1) % n].astype(float)
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm < 1e-6:
            angles.append(90.0)
            continue
        cos_a = np.dot(v1, v2) / norm
        angles.append(float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))))
    return angles


def _is_roof_color(mean_rgb) -> bool:
    """
    Return True if the mean colour of a contour region looks like a roof.
    Rejects: vegetation green, water blue, near-black (shadow/tree interior).
    """
    r, g, b = float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])
    # Reject vegetation (excess green)
    if 2 * g - r - b > 15:
        return False
    # Reject water / sky (blue-dominant)
    if b > r + 25 and b > g + 25:
        return False
    # Reject pitch black (no signal — inside a tree canopy shadow)
    if (r + g + b) / 3 < 25:
        return False
    return True


def _detect_buildings(img: Image.Image, veg_mask: np.ndarray,
                      img_w: int, img_h: int,
                      tile_lat: float, tile_lon: float) -> list:
    """
    Return a list of building footprint polygons, each as [(lon, lat), ...].

    Steps:
    - Mask out vegetation from the analysis image.
    - Canny edge detection on the non-green region.
    - Contour → polygon approximation → rectangularity + size + colour filter.
    """
    try:
        import cv2
    except ImportError:
        UI.vprint(0, "   ERROR: opencv-python required. pip install opencv-python")
        raise

    img_arr = np.array(img, dtype=np.uint8)

    # Suppress vegetation pixels in the image before edge detection
    non_veg_mask = cv2.bitwise_not(veg_mask)
    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    gray_masked = cv2.bitwise_and(gray, gray, mask=non_veg_mask)

    # Light Gaussian blur to reduce noise before Canny
    blurred = cv2.GaussianBlur(gray_masked, (3, 3), 0)
    edges = cv2.Canny(blurred, 40, 120)

    # Dilate edges slightly to close small gaps in rooflines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    buildings = []

    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if area_px < bld_min_area_px or area_px > bld_max_area_px:
            continue

        # Polygon approximation
        peri = cv2.arcLength(cnt, True)
        if peri < 1:
            continue
        epsilon = 0.025 * peri
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
        n_verts = len(approx)
        if n_verts < 4 or n_verts > 16:
            continue

        # Aspect ratio check via minimum bounding rectangle
        rect = cv2.minAreaRect(cnt)
        w, h = rect[1]
        if min(w, h) < 1:
            continue
        aspect = max(w, h) / min(w, h)
        if aspect > bld_max_aspect_ratio:
            continue

        # Rectangularity: fraction of angles within 20° of 90°
        angles = _angles_at_vertices(approx)
        right_angle_count = sum(
            1 for a in angles if abs(a - 90) < 20 or abs(a - 270) < 20
        )
        if right_angle_count < bld_rect_threshold * n_verts:
            continue

        # Roof colour filter
        contour_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        mean_vals = cv2.mean(img_arr, mask=contour_mask)
        if not _is_roof_color(mean_vals[:3]):
            continue

        # Convert pixel vertices to lat/lon
        pts = [
            _px_to_latlon(p[0][0], p[0][1], img_w, img_h, tile_lat, tile_lon)
            for p in approx
        ]
        if pts[0] != pts[-1]:
            pts.append(pts[0])

        buildings.append((pts, area_px))

    return buildings


# ---------------------------------------------------------------------------
# Data collection (used by both standalone and combined paths)
# ---------------------------------------------------------------------------

def collect_bld_polygons(img, veg_mask, img_w: int, img_h: int,
                         lat: int, lon: int) -> list:
    """
    Detect buildings and return a flat list of (ring, fac_path, height) tuples
    ready to be written to a DSF.  Shared by standalone and combined overlay.
    """
    rng = np.random.default_rng(seed=42)
    region = _region_for_latlon(lat + 0.5, lon + 0.5)

    try:
        raw = _detect_buildings(img, veg_mask, img_w, img_h, lat, lon)
    except ImportError:
        return []

    result = []
    for ring, area_px in raw:
        area_m2 = _area_px_to_m2(area_px, img_w, img_h, lat)
        sc = _size_class(area_m2)
        fac_path, base_h = _FAC_DEFS.get((region, sc), _FAC_DEFS[("generic", sc)])
        height = max(4.0, base_h + float(rng.uniform(-_HEIGHT_VARIANCE, _HEIGHT_VARIANCE)))
        result.append((ring, fac_path, height))
    return result


def _write_building_only_dsf(lat: int, lon: int, txt_path: str,
                              bld_polygons: list) -> None:
    """Write a buildings-only text DSF (no vegetation)."""
    unique_facs = list(dict.fromkeys(fp for _, fp, _ in bld_polygons))
    fac_index = {fp: i for i, fp in enumerate(unique_facs)}

    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "w") as f:
        f.write("PROPERTY sim/planet earth\n")
        f.write("PROPERTY sim/overlay 1\n")
        f.write(f"PROPERTY sim/west {lon}\n")
        f.write(f"PROPERTY sim/east {lon + 1}\n")
        f.write(f"PROPERTY sim/south {lat}\n")
        f.write(f"PROPERTY sim/north {lat + 1}\n")

        for fp in unique_facs:
            f.write(f"POLYGON_DEF {fp}\n")

        for ring, fac_path, height in bld_polygons:
            idx = fac_index[fac_path]
            f.write(f"BEGIN_POLYGON {idx} {int(round(height))} 2\n")
            f.write("BEGIN_WINDING\n")
            for lon_pt, lat_pt in ring:
                f.write(f"POLYGON_POINT {lon_pt:.7f} {lat_pt:.7f}\n")
            f.write("END_WINDING\n")
            f.write("END_POLYGON\n")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_building_overlay(lat: int, lon: int, build_dir: str = "") -> int:
    """Generate a buildings-only overlay DSF for the tile at (lat, lon)."""
    if not build_dir:
        build_dir = FNAMES.build_dir(lat, lon, "")
    timer = time.time()
    UI.vprint(0, f"\nBuilding Overlay : {FNAMES.short_latlon(lat, lon)}\n")

    img, img_w, img_h = _load_and_downsample(lat, lon, build_dir)
    if img is None:
        return 0

    UI.vprint(1, "   Building vegetation mask for suppression...")
    try:
        veg_mask = _build_vegetation_mask(img)
    except ImportError:
        return 0

    UI.vprint(1, "   Detecting building candidates...")
    bld_polygons = collect_bld_polygons(img, veg_mask, img_w, img_h, lat, lon)
    UI.vprint(1, f"   Found {len(bld_polygons)} candidate buildings")

    if not bld_polygons:
        UI.vprint(0, "   No buildings detected — skipping overlay.")
        return 1

    base = FNAMES.short_latlon(lat, lon)
    txt_path = os.path.join(FNAMES.Tmp_dir, base + "_bld.txt")
    UI.vprint(1, "   Writing text DSF...")
    _write_building_only_dsf(lat, lon, txt_path, bld_polygons)

    UI.vprint(1, "   Compiling binary DSF with DSFTool...")
    ok = _compile_dsf(txt_path, _dsf_dest(lat, lon))
    try:
        os.remove(txt_path)
    except OSError:
        pass
    if not ok:
        return 0
    UI.timings_and_bottom_line(timer)
    return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    def _usage():
        print("Usage: python src/O4_Building_Overlay.py <lat> <lon> [build_dir]")

    try:
        _lat = int(sys.argv[1])
        _lon = int(sys.argv[2])
    except (IndexError, ValueError):
        _usage()
        sys.exit(1)

    _build_dir = sys.argv[3] if len(sys.argv) > 3 else ""
    sys.exit(0 if build_building_overlay(_lat, _lon, _build_dir) else 1)
