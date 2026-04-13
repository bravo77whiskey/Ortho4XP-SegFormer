"""
O4_Veg_Overlay.py  –  Generate X-Plane 12 vegetation (forest) overlays
                      from Ortho4XP tile imagery.

Pipeline
--------
1. Load the ortho tile image (DDS → PIL, or any cached PNG/JPEG).
2. Compute an Excess-Green vegetation mask  ExG = 2G − R − B.
3. Separate likely forest from grass/fields using:
   - local texture variance (forests are rough, fields are smooth)
   - connected-component size / shape filtering
4. Vectorise the raster mask into polygons (via OpenCV contours).
5. Classify each polygon as an "area forest" or a "treeline" (high
   length-to-area ratio → line-fill mode, density +256 flag).
6. Pick an appropriate .for definition by latitude.
7. Write a text DSF overlay and compile it with DSFTool.

All image processing is done with OpenCV + NumPy (no heavy AI).
Requires:  opencv-python, numpy, Pillow  (already typical deps).
"""

import os
import sys
import subprocess
import time

_CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
from math import cos, pi, sqrt

import numpy as np
from PIL import Image

import O4_File_Names as FNAMES
import O4_UI_Utils as UI

# Module-level tunables – read from config at call time via the accessors below.
# These defaults are overridden by O4_Cfg_Vars entries when the module is used
# inside Ortho4XP; they remain as fallbacks for standalone CLI use.
veg_overlay_enabled      = False
veg_exg_threshold        = 20
veg_texture_std_threshold = 8
veg_max_analysis_px      = 4096

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
# Forest type lookup by latitude  (simple climate proxy)
# ---------------------------------------------------------------------------
# XP12 lib/ paths for forest overlays (confirmed from 1200 forests/library.txt).
# DSFTool BEGIN_POLYGON format: idx  param  coord_depth
#   coord_depth = 2 means 2D (lon + lat) — required for all ground polygons.
_FOREST_TYPES = [
    # (max_abs_lat, for_file)
    (23,  "lib/vegetation/forests/broadleaves/hot.for"),         # tropical
    (40,  "lib/vegetation/forests/broadleaves/warm.for"),        # temperate broadleaf
    (55,  "lib/vegetation/forests/mixed/temperate.for"),         # mixed temperate
    (75,  "lib/vegetation/forests/conifers/temperate.for"),      # boreal / taiga
    (90,  "lib/vegetation/forests/conifers/cold.for"),           # arctic / tundra
]
_HEDGE_FOR = "lib/vegetation/forests/mixed/temperate.for"

_DEFAULT_DENSITY = 200   # 0-255; 255 = maximum density

# Shape thresholds
_MIN_AREA_PX  = 400     # ignore patches smaller than this (pixels²)
_MAX_AREA_PX  = None    # None = no upper limit
_TREELINE_RATIO = 6.0   # perimeter² / area > ratio → candidate treeline
_TREELINE_MAX_WIDTH_M = 30.0  # broad rough-edged blobs should remain area forest
_LARGE_BLOB_AREA_DEG2 = 0.003   # blobs > this (in degree²) with very low
                                  # internal variance are likely fields – drop

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _for_file_for_lat(lat: float) -> str:
    abs_lat = abs(lat + 0.5)   # centre of the tile
    for max_lat, for_file in _FOREST_TYPES:
        if abs_lat <= max_lat:
            return for_file
    return "vegetation/mixed.for"


def _px_to_latlon(px_col, px_row, img_w, img_h, tile_lat, tile_lon):
    """Convert pixel coordinates to (lon, lat) within the 1°×1° tile."""
    lon = tile_lon + px_col / img_w
    lat = tile_lat + (img_h - px_row) / img_h   # row 0 = north
    return lon, lat


def _area_deg2(contour_latlon):
    """Shoelace area of a polygon given as [(lon, lat), ...] in degrees."""
    pts = contour_latlon
    n = len(pts)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0


def _perimeter_deg(contour_latlon):
    pts = contour_latlon
    n = len(pts)
    if n < 2:
        return 0.0
    total = 0.0
    scalx = cos((contour_latlon[0][1]) * pi / 180) * 111320  # rough m/deg lon
    scaly = 111320  # m/deg lat
    for i in range(n):
        j = (i + 1) % n
        dx = (pts[j][0] - pts[i][0]) * scalx
        dy = (pts[j][1] - pts[i][1]) * scaly
        total += sqrt(dx * dx + dy * dy)
    return total


# ---------------------------------------------------------------------------
# Step 1 – load image
# ---------------------------------------------------------------------------

def _gtile_to_wgs84(til_x, til_y, zoomlevel):
    """Convert web map tile indices to (lat, lon) of the tile's NW corner."""
    from math import atan, exp, pi
    rat_x = til_x / (2 ** (zoomlevel - 1)) - 1
    rat_y = 1 - til_y / (2 ** (zoomlevel - 1))
    lon = rat_x * 180
    lat = 360 / pi * atan(exp(pi * rat_y)) - 90
    return lat, lon


def _stitch_jpegs(lat: int, lon: int, ortho_dir: str,
                  canvas_size: int = 2048) -> Image.Image | None:
    """
    Stitch all JPEG web-map-tile files found under ortho_dir into a single
    RGB image representing (approximately) the 1°×1° tile at (lat, lon).

    Ortho4XP names each JPEG:  {til_y_top}_{til_x_left}_{provider}{zl}.jpg
    Each file is a composite of 16×16 web map tiles stitched at build time.
    We map each JPEG's lat/lon bounding box onto the canvas.

    Returns a PIL Image or None if no usable JPEGs are found.
    """
    tile_s, tile_n = float(lat), float(lat + 1)
    tile_w, tile_e = float(lon), float(lon + 1)
    tile_dlon = tile_e - tile_w
    tile_dlat = tile_n - tile_s

    canvas = Image.new("RGB", (canvas_size, canvas_size), (128, 128, 128))
    pasted = 0

    for subdir in sorted(os.listdir(ortho_dir)):
        subpath = os.path.join(ortho_dir, subdir)
        if not os.path.isdir(subpath):
            continue
        # subdir looks like "BI16" or "Bing_16" — extract zoomlevel from suffix
        try:
            zl = int(subdir.split("_")[-1]) if "_" in subdir else int(subdir[-2:])
        except (ValueError, IndexError):
            continue

        for fname in os.listdir(subpath):
            if not fname.lower().endswith(".jpg"):
                continue
            parts = fname.split("_")
            if len(parts) < 2:
                continue
            try:
                til_y_top  = int(parts[0])
                til_x_left = int(parts[1])
            except (ValueError, IndexError):
                continue

            # Each JPEG covers a 16-tile-wide block in web map tiles
            img_lat_n, img_lon_w = _gtile_to_wgs84(til_x_left,      til_y_top,      zl)
            img_lat_s, img_lon_e = _gtile_to_wgs84(til_x_left + 16, til_y_top + 16, zl)

            # Skip if no overlap with our tile
            if img_lat_n < tile_s or img_lat_s > tile_n:
                continue
            if img_lon_e < tile_w or img_lon_w > tile_e:
                continue

            try:
                piece = Image.open(os.path.join(subpath, fname)).convert("RGB")
            except Exception:
                continue

            pw, ph = piece.size

            # Map image corners to canvas pixel positions
            # x: longitude → pixel column;  y: latitude → pixel row (inverted)
            def lon_to_col(lo):
                return int((lo - tile_w) / tile_dlon * canvas_size)

            def lat_to_row(la):
                return int((tile_n - la) / tile_dlat * canvas_size)

            col_l = lon_to_col(img_lon_w)
            col_r = lon_to_col(img_lon_e)
            row_t = lat_to_row(img_lat_n)
            row_b = lat_to_row(img_lat_s)

            dest_w = max(1, col_r - col_l)
            dest_h = max(1, row_b - row_t)

            # Clip to canvas bounds
            src_col_start = max(0, -col_l)
            src_row_start = max(0, -row_t)
            paste_col = max(0, col_l)
            paste_row = max(0, row_t)

            if paste_col >= canvas_size or paste_row >= canvas_size:
                continue

            piece_resized = piece.resize((dest_w, dest_h), Image.LANCZOS)
            # Crop to only the portion that falls inside the canvas
            crop_w = min(canvas_size - paste_col, dest_w - src_col_start)
            crop_h = min(canvas_size - paste_row, dest_h - src_row_start)
            if crop_w <= 0 or crop_h <= 0:
                continue
            piece_crop = piece_resized.crop(
                (src_col_start, src_row_start,
                 src_col_start + crop_w, src_row_start + crop_h)
            )
            canvas.paste(piece_crop, (paste_col, paste_row))
            pasted += 1

    if pasted == 0:
        return None
    UI.vprint(1, f"   Stitched {pasted} JPEG source tile(s) into {canvas_size}×{canvas_size} canvas")
    return canvas


def _stitch_dds_textures(build_dir: str, lat: int, lon: int,
                         canvas_size: int = 2048) -> Image.Image | None:
    """
    Stitch all DDS texture files in build_dir/textures/ into a single RGB image
    representing the 1°×1° tile at (lat, lon).

    Supports two DDS filename formats used by Ortho4XP:
      Standard:  {til_y_top}_{til_x_left}_{provider}{zl}.dds
      g2xpl_16:  {zl}_{til_x_left}_{inverted_y}.dds
    Each file covers a 16-tile-wide web map tile block.
    """
    import re as _re
    _STD_RE   = _re.compile(r"^(\d+)_(\d+)_([A-Za-z][A-Za-z0-9_]*)(\d{2})\.dds$",
                             _re.IGNORECASE)
    _G2XPL_RE = _re.compile(r"^(\d{2})_(\d+)_(\d+)\.dds$", _re.IGNORECASE)

    textures_dir = os.path.join(build_dir, "textures")
    if not os.path.isdir(textures_dir):
        return None

    tile_s, tile_n = float(lat), float(lat + 1)
    tile_w, tile_e = float(lon), float(lon + 1)
    tile_dlon = tile_e - tile_w
    tile_dlat = tile_n - tile_s

    canvas = Image.new("RGB", (canvas_size, canvas_size), (128, 128, 128))
    pasted = 0

    for fname in sorted(os.listdir(textures_dir)):
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
            piece = Image.open(os.path.join(textures_dir, fname)).convert("RGB")
        except Exception:
            continue

        col_l = int((img_lon_w - tile_w) / tile_dlon * canvas_size)
        col_r = int((img_lon_e - tile_w) / tile_dlon * canvas_size)
        row_t = int((tile_n - img_lat_n) / tile_dlat * canvas_size)
        row_b = int((tile_n - img_lat_s) / tile_dlat * canvas_size)

        dest_w = max(1, col_r - col_l)
        dest_h = max(1, row_b - row_t)

        src_col_start = max(0, -col_l)
        src_row_start = max(0, -row_t)
        paste_col = max(0, col_l)
        paste_row = max(0, row_t)

        if paste_col >= canvas_size or paste_row >= canvas_size:
            continue

        piece_resized = piece.resize((dest_w, dest_h), Image.LANCZOS)
        crop_w = min(canvas_size - paste_col, dest_w - src_col_start)
        crop_h = min(canvas_size - paste_row, dest_h - src_row_start)
        if crop_w <= 0 or crop_h <= 0:
            continue
        piece_crop = piece_resized.crop(
            (src_col_start, src_row_start,
             src_col_start + crop_w, src_row_start + crop_h)
        )
        canvas.paste(piece_crop, (paste_col, paste_row))
        pasted += 1

    if pasted == 0:
        return None
    UI.vprint(1, f"   Stitched {pasted} DDS texture(s) into {canvas_size}×{canvas_size} canvas")
    return canvas


def _load_tile_image(lat: int, lon: int, build_dir: str) -> Image.Image | None:
    """
    Return an RGB PIL Image representing the 1°×1° tile at (lat, lon).

    Search order:
    1. WGS84 GeoTIFFs in FNAMES.Geotiff_dir (written by Ortho4XP's tif export)
    2. JPEG source tiles in FNAMES.Imagery_dir/{latlon}/ → stitched into canvas
    3. All DDS textures in build_dir/textures/ → stitched into full-tile canvas
    """
    latlon = FNAMES.short_latlon(lat, lon)

    # 1. GeoTIFF export directory — stitch any WGS84 tifs that belong to this tile
    if os.path.isdir(FNAMES.Geotiff_dir):
        tifs = [
            f for f in os.listdir(FNAMES.Geotiff_dir)
            if f.endswith("-WGS84.tif")
        ]
        if tifs:
            # Use the first one as a representative (stitching all would need GDAL)
            tif_path = os.path.join(FNAMES.Geotiff_dir, tifs[0])
            try:
                img = Image.open(tif_path).convert("RGB")
                UI.vprint(1, f"   Loaded tile image from GeoTIFF {tif_path}")
                return img
            except Exception as e:
                UI.vprint(2, f"   Could not open GeoTIFF {tif_path}: {e}")

    # 2. Stitch JPEG source tiles from Orthophotos directory
    ortho_dir = os.path.join(FNAMES.Imagery_dir, latlon)
    if os.path.isdir(ortho_dir):
        img = _stitch_jpegs(lat, lon, ortho_dir)
        if img is not None:
            return img

    # 3. Stitch all DDS textures from the tile's textures/ directory
    img = _stitch_dds_textures(build_dir, lat, lon)
    if img is not None:
        return img

    UI.vprint(0, f"   WARNING: No usable tile image found for {latlon}.")
    return None


# ---------------------------------------------------------------------------
# Step 2+3 – vegetation mask
# ---------------------------------------------------------------------------

def _build_vegetation_mask(img: Image.Image) -> np.ndarray:
    """
    Returns a uint8 binary mask (255 = vegetation) of the same H×W as img.

    Method:
    - Compute ExG = 2G − R − B  (Excess Green index, range −255..510).
    - Threshold at ExG > 20 (heuristic; tune with VEG_THRESHOLD env var).
    - Apply morphological opening to remove speckle.
    - Compute local texture variance; suppress low-variance regions
      (likely flat fields / grass) whose ExG might still be "green".
    """
    try:
        import cv2
    except ImportError:
        UI.vprint(0, "   ERROR: opencv-python is required for vegetation overlay generation.")
        UI.vprint(0, "   Install with:  pip install opencv-python")
        raise

    arr = np.array(img, dtype=np.int16)
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    exg = 2 * G.astype(np.int16) - R - B
    green_mask = (exg > veg_exg_threshold).astype(np.uint8) * 255

    # Morphological opening to remove isolated pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

    # Local texture variance – compute on grayscale, then blur to get
    # neighbourhood variance using a larger kernel.
    gray = cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur_sz = 31  # neighbourhood radius
    mean = cv2.blur(gray, (blur_sz, blur_sz))
    mean_sq = cv2.blur(gray * gray, (blur_sz, blur_sz))
    variance = mean_sq - mean * mean   # E[X²] − E[X]²
    std_dev = np.sqrt(np.maximum(variance, 0))

    # Keep only green pixels with sufficient local roughness
    rough_mask = (std_dev > veg_texture_std_threshold).astype(np.uint8) * 255
    veg_mask = cv2.bitwise_and(green_mask, rough_mask)

    # Morphological closing to fill gaps inside canopy blobs
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    veg_mask = cv2.morphologyEx(veg_mask, cv2.MORPH_CLOSE, close_kernel)

    return veg_mask


# ---------------------------------------------------------------------------
# Step 4+5 – vectorise & classify
# ---------------------------------------------------------------------------

def _vectorise_mask(veg_mask: np.ndarray, img_w: int, img_h: int,
                    tile_lat: float, tile_lon: float):
    """
    Returns two lists of lat/lon polygon rings:
        area_polygons  – filled forest blobs
        line_polygons  – narrow treelines / hedgerows
    Each polygon is a list of (lon, lat) tuples, CCW winding.
    """
    try:
        import cv2
    except ImportError:
        raise

    contours, _ = cv2.findContours(
        veg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
    )

    area_polygons = []
    line_polygons = []

    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if area_px < _MIN_AREA_PX:
            continue

        # Simplify contour (Douglas-Peucker)
        epsilon = 2.0 * sqrt(area_px) * 0.01   # ~1% of effective radius
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
        if len(approx) < 3:
            continue

        # Convert pixel coords → (lon, lat)
        pts = [
            _px_to_latlon(p[0][0], p[0][1], img_w, img_h, tile_lat, tile_lon)
            for p in approx
        ]
        # Ensure closed ring
        if pts[0] != pts[-1]:
            pts.append(pts[0])

        area_d2 = _area_deg2(pts)
        perim_m = _perimeter_deg(pts)

        if area_d2 <= 0:
            continue

        # Large, very uniform areas → likely a field, skip
        # (we rely on the texture variance step above, but add a size check)
        if area_d2 > _LARGE_BLOB_AREA_DEG2:
            # Passed texture check but still very large → keep as area forest
            pass

        # Classify: treeline if isoperimetric quotient is very low
        # (long, thin shapes)  iso = 4π·area / perimeter²
        area_m2 = area_d2 * (111320 ** 2)
        if perim_m > 0:
            iso = 4 * pi * area_m2 / (perim_m ** 2)
        else:
            iso = 1.0

        width_m = (2.0 * area_m2 / perim_m) if perim_m > 0 else float("inf")
        if iso < (1.0 / _TREELINE_RATIO) and width_m <= _TREELINE_MAX_WIDTH_M:
            line_polygons.append(pts)
        else:
            area_polygons.append(pts)

    return area_polygons, line_polygons


# ---------------------------------------------------------------------------
# Step 6+7 – write DSF text and compile
# ---------------------------------------------------------------------------

def _write_text_dsf(
    lat: int, lon: int,
    txt_path: str,
    area_polygons: list,
    line_polygons: list,
    for_file: str,
    hedge_for_file: str,
):
    """Write the DSF in human-readable text format."""
    definitions = []
    if area_polygons:
        definitions.append((for_file, _DEFAULT_DENSITY, 2))
    if line_polygons:
        definitions.append((hedge_for_file, _DEFAULT_DENSITY, 2))

    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "w") as f:
        f.write("PROPERTY sim/planet earth\n")
        f.write("PROPERTY sim/overlay 1\n")
        f.write(f"PROPERTY sim/west {lon}\n")
        f.write(f"PROPERTY sim/east {lon + 1}\n")
        f.write(f"PROPERTY sim/south {lat}\n")
        f.write(f"PROPERTY sim/north {lat + 1}\n")
        # No EXCLUSION lines — we intentionally avoid exclusions

        for for_path, _, _ in definitions:
            f.write(f"POLYGON_DEF {for_path}\n")

        def_idx = 0

        def write_polys(polygons, dsf_param):
            nonlocal def_idx
            for ring in polygons:
                f.write(f"BEGIN_POLYGON {def_idx} {dsf_param} 2\n")
                f.write("BEGIN_WINDING\n")
                for lon_pt, lat_pt in ring:
                    f.write(f"POLYGON_POINT {lon_pt:.7f} {lat_pt:.7f}\n")
                f.write("END_WINDING\n")
                f.write("END_POLYGON\n")
            def_idx += 1

        if area_polygons:
            write_polys(area_polygons, _DEFAULT_DENSITY)
        if line_polygons:
            write_polys(line_polygons, _DEFAULT_DENSITY + 256)


def _compile_dsf(txt_path: str, dsf_path: str) -> bool:
    """Compile text DSF to binary using DSFTool. Returns True on success."""
    cmd = [_dsftool, "-text2dsf", txt_path, dsf_path]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             creationflags=_CREATE_NO_WINDOW)
    stdout, stderr = proc.communicate()
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        UI.vprint(2, "   DSFTool: " + line)
    if proc.returncode != 0:
        UI.vprint(0, f"   ERROR: DSFTool failed (exit {proc.returncode})")
        for line in stderr.decode("utf-8", errors="replace").splitlines():
            UI.vprint(0, "   " + line)
        return False
    return True


def _write_combined_dsf(
    lat: int, lon: int, txt_path: str,
    # vegetation data (may be empty)
    area_polygons: list, line_polygons: list,
    for_file: str, hedge_for_file: str,
    # building data (may be empty): list of (ring, fac_path, height)
    bld_polygons: list,
) -> None:
    """
    Write a text DSF overlay that contains both forest polygons and
    building facade polygons in a single file.

    bld_polygons: list of (ring [(lon, lat)], fac_path, height_m)
    """
    # Collect all POLYGON_DEF entries in order (forests first, then facades)
    forest_defs = []
    if area_polygons:
        forest_defs.append((for_file, _DEFAULT_DENSITY, 2))
    if line_polygons:
        forest_defs.append((hedge_for_file, _DEFAULT_DENSITY, 2))

    unique_facs = list(dict.fromkeys(fp for _, fp, _ in bld_polygons))
    fac_index_offset = len(forest_defs)
    fac_index = {fp: fac_index_offset + i for i, fp in enumerate(unique_facs)}

    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "w") as f:
        f.write("PROPERTY sim/planet earth\n")
        f.write("PROPERTY sim/overlay 1\n")
        f.write(f"PROPERTY sim/west {lon}\n")
        f.write(f"PROPERTY sim/east {lon + 1}\n")
        f.write(f"PROPERTY sim/south {lat}\n")
        f.write(f"PROPERTY sim/north {lat + 1}\n")

        for for_path, _, _ in forest_defs:
            f.write(f"POLYGON_DEF {for_path}\n")
        for fp in unique_facs:
            f.write(f"POLYGON_DEF {fp}\n")

        def_idx = 0

        # Forest polygons
        def write_forest(polygons, dsf_param):
            nonlocal def_idx
            for ring in polygons:
                f.write(f"BEGIN_POLYGON {def_idx} {dsf_param} 2\n")
                f.write("BEGIN_WINDING\n")
                for lon_pt, lat_pt in ring:
                    f.write(f"POLYGON_POINT {lon_pt:.7f} {lat_pt:.7f}\n")
                f.write("END_WINDING\n")
                f.write("END_POLYGON\n")
            def_idx += 1

        if area_polygons:
            write_forest(area_polygons, _DEFAULT_DENSITY)
        if line_polygons:
            write_forest(line_polygons, _DEFAULT_DENSITY + 256)

        # Building facade polygons
        for ring, fac_path, height in bld_polygons:
            idx = fac_index[fac_path]
            f.write(f"BEGIN_POLYGON {idx} {height:.1f} 2\n")
            f.write("BEGIN_WINDING\n")
            for lon_pt, lat_pt in ring:
                f.write(f"POLYGON_POINT {lon_pt:.7f} {lat_pt:.7f}\n")
            f.write("END_WINDING\n")
            f.write("END_POLYGON\n")


def _load_and_downsample(lat: int, lon: int, build_dir: str):
    """Load tile image and downsample to analysis resolution. Returns (img, w, h) or (None, 0, 0)."""
    img = _load_tile_image(lat, lon, build_dir)
    if img is None:
        return None, 0, 0
    img_w, img_h = img.size
    UI.vprint(1, f"   Image size: {img_w}×{img_h}")
    if max(img_w, img_h) > veg_max_analysis_px:
        scale = veg_max_analysis_px / max(img_w, img_h)
        img = img.resize((int(img_w * scale), int(img_h * scale)), Image.LANCZOS)
        img_w, img_h = img.size
        UI.vprint(1, f"   Downsampled to {img_w}×{img_h} for analysis")
    return img, img_w, img_h


def _collect_veg_data(img, img_w: int, img_h: int, lat: int, lon: int, veg_mask=None):
    """
    Run vegetation detection on an already-loaded image.
    Returns (area_polys, line_polys, for_file, hedge_for, veg_mask)
    or ([], [], '', '', None) if nothing found.
    veg_mask can be passed in if already computed (avoids recomputation for combined run).
    """
    if veg_mask is None:
        try:
            veg_mask = _build_vegetation_mask(img)
        except ImportError:
            return [], [], "", "", None

    veg_pct = np.count_nonzero(veg_mask) / veg_mask.size * 100
    UI.vprint(1, f"   Vegetation coverage: {veg_pct:.1f}%")

    if veg_pct < 0.5:
        UI.vprint(1, "   No significant vegetation detected.")
        return [], [], "", "", veg_mask

    area_polys, line_polys = _vectorise_mask(veg_mask, img_w, img_h, lat, lon)
    UI.vprint(1, f"   Forest polygons: {len(area_polys)} area, {len(line_polys)} treelines")
    for_file = _for_file_for_lat(lat)
    return area_polys, line_polys, for_file, _HEDGE_FOR, veg_mask


def _dsf_dest(lat: int, lon: int) -> str:
    """Return the canonical DSF output path for an overlay tile."""
    dest_dir = os.path.join(
        FNAMES.Overlay_dir, "Earth nav data", FNAMES.round_latlon(lat, lon)
    )
    os.makedirs(dest_dir, exist_ok=True)
    return os.path.join(dest_dir, FNAMES.short_latlon(lat, lon) + ".dsf")


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def build_veg_overlay(lat: int, lon: int, build_dir: str = "") -> int:
    """Generate a vegetation-only overlay DSF for the tile at (lat, lon)."""
    if not build_dir:
        build_dir = FNAMES.build_dir(lat, lon, "")
    timer = time.time()
    UI.vprint(0, f"\nVeg Overlay : {FNAMES.short_latlon(lat, lon)}\n")

    img, img_w, img_h = _load_and_downsample(lat, lon, build_dir)
    if img is None:
        return 0

    UI.vprint(1, "   Building vegetation mask...")
    area_polys, line_polys, for_file, hedge_for, _ = _collect_veg_data(
        img, img_w, img_h, lat, lon
    )
    if not area_polys and not line_polys:
        UI.vprint(0, "   No vegetation polygons — skipping overlay.")
        return 1

    base = FNAMES.short_latlon(lat, lon)
    txt_path = os.path.join(FNAMES.Tmp_dir, base + "_veg.txt")
    _write_text_dsf(lat, lon, txt_path, area_polys, line_polys, for_file, hedge_for)
    ok = _compile_dsf(txt_path, _dsf_dest(lat, lon))
    try:
        os.remove(txt_path)
    except OSError:
        pass
    if not ok:
        return 0
    UI.timings_and_bottom_line(timer)
    return 1


def build_combined_overlay(lat: int, lon: int, build_dir: str = "",
                           run_veg: bool = True, run_bld: bool = True) -> int:
    """
    Generate a single overlay DSF containing both vegetation forests and
    building facades for the tile at (lat, lon).  Loads the image once.

    run_veg / run_bld control which analyses are performed.
    """
    import O4_Building_Overlay as BLD   # lazy import avoids circular dependency

    if not build_dir:
        build_dir = FNAMES.build_dir(lat, lon, "")
    timer = time.time()
    label = "+".join(filter(None, ["Veg" if run_veg else "", "Bld" if run_bld else ""]))
    UI.vprint(0, f"\n{label} Overlay : {FNAMES.short_latlon(lat, lon)}\n")

    img, img_w, img_h = _load_and_downsample(lat, lon, build_dir)
    if img is None:
        return 0

    # Vegetation
    area_polys, line_polys, for_file, hedge_for, veg_mask = [], [], "", "", None
    if run_veg:
        UI.vprint(1, "   Building vegetation mask...")
        area_polys, line_polys, for_file, hedge_for, veg_mask = _collect_veg_data(
            img, img_w, img_h, lat, lon
        )

    # Buildings (reuse veg_mask so it is not recomputed)
    bld_polygons = []
    if run_bld:
        if veg_mask is None:
            try:
                veg_mask = _build_vegetation_mask(img)
            except ImportError:
                return 0
        UI.vprint(1, "   Detecting building candidates...")
        bld_polygons = BLD.collect_bld_polygons(
            img, veg_mask, img_w, img_h, lat, lon
        )
        UI.vprint(1, f"   Building candidates: {len(bld_polygons)}")

    if not area_polys and not line_polys and not bld_polygons:
        UI.vprint(0, "   Nothing detected — skipping overlay.")
        return 1

    base = FNAMES.short_latlon(lat, lon)
    txt_path = os.path.join(FNAMES.Tmp_dir, base + "_combined.txt")
    UI.vprint(1, "   Writing combined text DSF...")
    _write_combined_dsf(lat, lon, txt_path,
                        area_polys, line_polys, for_file, hedge_for,
                        bld_polygons)
    ok = _compile_dsf(txt_path, _dsf_dest(lat, lon))
    try:
        os.remove(txt_path)
    except OSError:
        pass
    if not ok:
        return 0
    UI.vprint(0, f"   Overlay written to {_dsf_dest(lat, lon)}")
    UI.timings_and_bottom_line(timer)
    return 1


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    def _usage():
        print("Usage: python src/O4_Veg_Overlay.py <lat> <lon> [build_dir]")
        print("Example: python src/O4_Veg_Overlay.py 48 8")

    try:
        _lat = int(sys.argv[1])
        _lon = int(sys.argv[2])
    except (IndexError, ValueError):
        _usage()
        sys.exit(1)

    _build_dir = sys.argv[3] if len(sys.argv) > 3 else ""
    result = build_veg_overlay(_lat, _lon, _build_dir)
    sys.exit(0 if result else 1)
