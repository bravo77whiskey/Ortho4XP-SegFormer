"""Perceptual rooftop colours for X-Plane OBJ8 building assets.

The building overlay places virtual library aliases, not physical OBJ files.
One alias can therefore render any of several physical variants.  This module
keeps the physical descriptors separate so callers can require every variant
to match before preferring an alias.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import O4_SFR_Asset_Inventory as ASSETINV
import O4_SFR_Persistent_Cache as PCACHE


ROOF_COLOR_ANALYZER_VERSION = "obj8-roof-color-v1"
ROOF_NORMAL_Y_MIN = 0.35
ROOF_TOP_CELL_M = 0.5
ROOF_TOP_TOLERANCE_M = 0.25
ROOF_NEUTRAL_CHROMA_MAX = 12.0
ROOF_NEUTRAL_DARK_L_MAX = 35.0
ROOF_NEUTRAL_LIGHT_L_MIN = 72.0

_BARYCENTRIC_SAMPLES = (
    (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    (0.60, 0.20, 0.20),
    (0.20, 0.60, 0.20),
    (0.20, 0.20, 0.60),
    (0.45, 0.45, 0.10),
    (0.45, 0.10, 0.45),
    (0.10, 0.45, 0.45),
)
_ANALYSIS_MEMORY_CACHE = {}


def _lab_family(lab):
    """Return the broad perceptual roof-colour family for one CIE Lab triplet."""
    lightness, a_star, b_star = (float(v) for v in lab)
    chroma = math.hypot(a_star, b_star)
    if chroma < ROOF_NEUTRAL_CHROMA_MAX:
        if lightness < ROOF_NEUTRAL_DARK_L_MAX:
            return "neutral_dark"
        if lightness > ROOF_NEUTRAL_LIGHT_L_MIN:
            return "neutral_light"
        return "neutral_mid"

    hue = math.degrees(math.atan2(b_star, a_star)) % 360.0
    if hue < 105.0 or hue >= 345.0:
        return "warm"
    if hue < 200.0:
        return "green"
    if hue < 315.0:
        return "blue"
    return "violet"


def _rgb_pixels_to_lab(rgb):
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    shape = arr.shape
    converted = cv2.cvtColor(
        np.ascontiguousarray(arr.reshape(-1, 1, 3) / 255.0),
        cv2.COLOR_RGB2LAB,
    )
    return converted.reshape(shape)


def descriptor_from_rgb(rgb):
    """Return a serialisable roof-colour descriptor from an RGB triplet."""
    rgb_arr = np.asarray(rgb, dtype=np.float32).reshape(1, 3)
    lab = _rgb_pixels_to_lab(rgb_arr)[0]
    return {
        "family": _lab_family(lab),
        "lab": tuple(float(v) for v in lab),
        "rgb": tuple(int(np.clip(round(float(v)), 0, 255)) for v in rgb_arr[0]),
    }


def descriptor_from_lab(lab, rgb=None):
    """Return a serialisable descriptor when Lab has already been calculated."""
    lab = tuple(float(v) for v in lab)
    descriptor = {
        "family": _lab_family(lab),
        "lab": lab,
    }
    if rgb is not None:
        descriptor["rgb"] = tuple(
            int(np.clip(round(float(v)), 0, 255)) for v in rgb
        )
    return descriptor


def _weighted_median(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.size == 0 or weights.size != values.size:
        return None
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return None
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    cutoff = float(weights.sum()) * 0.5
    idx = int(np.searchsorted(np.cumsum(weights), cutoff, side="left"))
    return float(values[min(idx, values.size - 1)])


def _weighted_color_descriptor(rgb_samples, weights):
    rgb = np.asarray(rgb_samples, dtype=np.float32).reshape(-1, 3)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if rgb.shape[0] == 0 or weights.size != rgb.shape[0]:
        return None
    lab = _rgb_pixels_to_lab(rgb)
    median_lab = [
        _weighted_median(lab[:, channel], weights)
        for channel in range(3)
    ]
    median_rgb = [
        _weighted_median(rgb[:, channel], weights)
        for channel in range(3)
    ]
    if any(value is None for value in median_lab):
        return None
    return descriptor_from_lab(median_lab, median_rgb)


def sample_rooftop_color(image_rgb, polygon):
    """Sample the median colour inside an inset detected rooftop polygon."""
    if image_rgb is None:
        return None
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        return None
    height, width = image.shape[:2]
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 3 or not np.all(np.isfinite(points)):
        return None

    center = points.mean(axis=0)
    inset = center + (points - center) * 0.80

    def _pixels_for(poly):
        rounded = np.rint(poly).astype(np.int32)
        x0 = max(0, int(rounded[:, 0].min()))
        y0 = max(0, int(rounded[:, 1].min()))
        x1 = min(width - 1, int(rounded[:, 0].max()))
        y1 = min(height - 1, int(rounded[:, 1].max()))
        if x1 < x0 or y1 < y0:
            return np.empty((0, 3), dtype=image.dtype)
        local = rounded - np.asarray((x0, y0), dtype=np.int32)
        mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
        cv2.fillPoly(mask, [local], 1)
        roi = image[y0:y1 + 1, x0:x1 + 1, :3]
        return roi[mask.astype(bool)]

    pixels = _pixels_for(inset)
    if pixels.shape[0] < 9:
        pixels = _pixels_for(points)
    if pixels.shape[0] == 0:
        return None
    median_rgb = np.median(pixels.astype(np.float32), axis=0)
    descriptor = descriptor_from_rgb(median_rgb)
    descriptor["pixel_count"] = int(pixels.shape[0])
    return descriptor


def _obj_texture_name(obj_path):
    try:
        with open(obj_path, "r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                parts = raw_line.strip().split()
                if parts and parts[0].upper() == "TEXTURE" and len(parts) >= 2:
                    value = " ".join(parts[1:]).strip()
                    if value and value.lower() != "none":
                        return value
    except OSError:
        return None
    return None


def _resolve_diffuse_texture(obj_path, texture_name):
    if not texture_name:
        return None
    obj_dir = Path(obj_path).resolve().parent
    raw_path = Path(texture_name.replace("\\", os.sep))
    base = raw_path if raw_path.is_absolute() else obj_dir / raw_path
    candidates = [base]
    for suffix in (".dds", ".png", ".jpg", ".jpeg", ".bmp"):
        candidate = base.with_suffix(suffix)
        if candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def _file_signature(path):
    if not path:
        return None
    try:
        stat = os.stat(path)
    except (OSError, TypeError, ValueError):
        return None
    return (
        os.path.realpath(path),
        int(stat.st_size),
        int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
    )


def _cache_path(obj_path, texture_path, cache_dir):
    if not cache_dir:
        return None
    signature = (
        ROOF_COLOR_ANALYZER_VERSION,
        _file_signature(obj_path),
        _file_signature(texture_path),
    )
    digest = hashlib.sha1(repr(signature).encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, "obj8_roof_color", f"{digest}.pkl")


def _parse_obj8_mesh(obj_path):
    vertices = []
    indices = []
    draw_ranges = []
    try:
        handle = open(obj_path, "r", encoding="utf-8", errors="ignore")
    except OSError:
        return None
    with handle:
        for raw_line in handle:
            parts = raw_line.strip().split()
            if not parts:
                continue
            command = parts[0].upper()
            if command == "VT" and len(parts) >= 9:
                try:
                    vertices.append(tuple(float(value) for value in parts[1:9]))
                except ValueError:
                    continue
            elif command in {"IDX", "IDX10"} and len(parts) >= 2:
                try:
                    indices.extend(int(value) for value in parts[1:])
                except ValueError:
                    continue
            elif command == "TRIS" and len(parts) >= 3:
                try:
                    draw_ranges.append((int(parts[1]), int(parts[2])))
                except ValueError:
                    continue
    if not vertices or len(indices) < 3:
        return None
    if not draw_ranges:
        draw_ranges = [(0, len(indices))]
    return np.asarray(vertices, dtype=np.float64), indices, draw_ranges


def _roof_surface_samples(vertices, indices, draw_ranges):
    samples = []
    top_by_cell = {}
    for offset, count in draw_ranges:
        start = max(0, int(offset))
        stop = min(len(indices), start + max(0, int(count)))
        stop -= (stop - start) % 3
        for pos in range(start, stop, 3):
            tri_indices = indices[pos:pos + 3]
            if len(tri_indices) != 3:
                continue
            if min(tri_indices) < 0 or max(tri_indices) >= len(vertices):
                continue
            tri = vertices[np.asarray(tri_indices, dtype=np.int64)]
            xyz = tri[:, :3]
            normals = tri[:, 3:6]
            normal = normals.mean(axis=0)
            normal_len = float(np.linalg.norm(normal))
            if normal_len <= 1e-9:
                normal = np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])
                normal_len = float(np.linalg.norm(normal))
            if normal_len <= 1e-9:
                continue
            if abs(float(normal[1])) / normal_len < ROOF_NORMAL_Y_MIN:
                continue

            cross = np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])
            projected_area = abs(float(cross[1])) * 0.5
            if projected_area <= 1e-8:
                continue
            weight = projected_area / len(_BARYCENTRIC_SAMPLES)
            uv = tri[:, 6:8]
            for bary in _BARYCENTRIC_SAMPLES:
                bary_arr = np.asarray(bary, dtype=np.float64)
                point = bary_arr @ xyz
                texcoord = bary_arr @ uv
                cell = (
                    int(math.floor(float(point[0]) / ROOF_TOP_CELL_M)),
                    int(math.floor(float(point[2]) / ROOF_TOP_CELL_M)),
                )
                sample = (
                    cell,
                    float(point[1]),
                    float(texcoord[0]),
                    float(texcoord[1]),
                    float(weight),
                )
                samples.append(sample)
                previous = top_by_cell.get(cell)
                if previous is None or point[1] > previous:
                    top_by_cell[cell] = float(point[1])
    return [
        sample
        for sample in samples
        if sample[1] >= top_by_cell.get(sample[0], sample[1]) - ROOF_TOP_TOLERANCE_M
    ]


def _sample_texture(texture_path, surface_samples):
    try:
        with Image.open(texture_path) as source:
            texture = np.asarray(source.convert("RGB"), dtype=np.uint8)
    except Exception:
        return None
    if texture.ndim != 3 or texture.shape[0] <= 0 or texture.shape[1] <= 0:
        return None
    height, width = texture.shape[:2]
    colors = []
    weights = []
    for _cell, _surface_y, u_coord, v_coord, weight in surface_samples:
        u_wrapped = float(u_coord) - math.floor(float(u_coord))
        v_wrapped = float(v_coord) - math.floor(float(v_coord))
        x = int(np.clip(round(u_wrapped * (width - 1)), 0, width - 1))
        y = int(np.clip(round((1.0 - v_wrapped) * (height - 1)), 0, height - 1))
        colors.append(texture[y, x])
        weights.append(float(weight))
    return _weighted_color_descriptor(colors, weights)


def analyze_obj_roof_color(obj_path, cache_dir=None):
    """Return one physical OBJ's roof descriptor, or ``None`` if unsupported."""
    if not obj_path or not os.path.isfile(obj_path):
        return None
    texture_name = _obj_texture_name(obj_path)
    texture_path = _resolve_diffuse_texture(obj_path, texture_name)
    if texture_path is None:
        return None
    analysis_key = (
        ROOF_COLOR_ANALYZER_VERSION,
        _file_signature(obj_path),
        _file_signature(texture_path),
    )
    memory_cached = _ANALYSIS_MEMORY_CACHE.get(analysis_key)
    if isinstance(memory_cached, dict):
        return memory_cached
    cache_path = _cache_path(obj_path, texture_path, cache_dir)
    cached = PCACHE.load(cache_path)
    if isinstance(cached, dict):
        _ANALYSIS_MEMORY_CACHE[analysis_key] = cached
        return cached

    mesh = _parse_obj8_mesh(obj_path)
    if mesh is None:
        return None
    surface_samples = _roof_surface_samples(*mesh)
    if not surface_samples:
        return None
    descriptor = _sample_texture(texture_path, surface_samples)
    if descriptor is None:
        return None
    descriptor.update({
        "obj_path": os.path.abspath(obj_path),
        "texture_path": os.path.abspath(texture_path),
        "sample_count": len(surface_samples),
    })
    PCACHE.save(cache_path, descriptor)
    _ANALYSIS_MEMORY_CACHE[analysis_key] = descriptor
    return descriptor


def _effective_physical_exports(exports):
    """Return active physical variants, ignoring backups when primaries exist."""
    exports = list(exports or ())
    primaries = [
        export for export in exports
        if str(getattr(export, "command", "")).upper() != "EXPORT_BACKUP"
    ]
    chosen = primaries if primaries else exports
    resolved = []
    missing = False
    seen = set()
    for export in chosen:
        path = getattr(export, "resolved_path", None)
        if not path:
            missing = True
            continue
        key = os.path.normcase(os.path.realpath(path))
        if key in seen:
            continue
        seen.add(key)
        resolved.append(path)
    return resolved, missing


def enrich_asset_roof_colors(asset_pools, library_exports, cache_dir=None):
    """Attach strict physical-variant roof descriptors to object assets.

    Returns counters describing how many virtual aliases are safe matches,
    inconsistent across variants, or unavailable.
    """
    grouped = ASSETINV.unique_virtual_exports(library_exports or (), suffix=".obj")
    counters = {
        "safe": 0,
        "inconsistent": 0,
        "unavailable": 0,
    }
    for pool in (asset_pools or {}).values():
        for asset in pool:
            if asset.get("kind") != "object":
                continue
            key = ASSETINV.normalize_library_path(asset.get("path", ""))
            exports = grouped.get(key, ())
            paths, missing_export = _effective_physical_exports(exports)
            if not paths and os.path.isfile(str(asset.get("path") or "")):
                paths = [asset["path"]]
            descriptors = []
            failed = bool(missing_export)
            for path in paths:
                descriptor = analyze_obj_roof_color(path, cache_dir=cache_dir)
                if descriptor is None:
                    failed = True
                    continue
                descriptors.append(descriptor)
            complete = bool(descriptors) and not failed and len(descriptors) == len(paths)
            families = {item.get("family") for item in descriptors}
            consistent = complete and len(families) == 1
            asset["roof_color_variants"] = tuple(descriptors)
            asset["roof_color_complete"] = bool(complete)
            asset["roof_color_consistent"] = bool(consistent)
            if consistent:
                counters["safe"] += 1
            elif complete:
                counters["inconsistent"] += 1
            else:
                counters["unavailable"] += 1
    return counters


def asset_roof_color_distance(asset, target):
    """Return strict worst-variant Lab distance, or ``None`` when not a match."""
    if not target or not asset or not asset.get("roof_color_complete"):
        return None
    target_family = target.get("family")
    target_lab = np.asarray(target.get("lab", ()), dtype=np.float64)
    variants = asset.get("roof_color_variants") or ()
    if target_lab.shape != (3,) or not variants:
        return None
    distances = []
    for variant in variants:
        if variant.get("family") != target_family:
            return None
        variant_lab = np.asarray(variant.get("lab", ()), dtype=np.float64)
        if variant_lab.shape != (3,) or not np.all(np.isfinite(variant_lab)):
            return None
        distances.append(float(np.linalg.norm(variant_lab - target_lab)))
    return max(distances) if distances else None


def color_assets_by_family(asset_pools):
    """Return stable path maps for aliases whose every variant shares a family."""
    result = {}
    for pool in (asset_pools or {}).values():
        for asset in pool:
            if asset.get("kind") != "object" or not asset.get("roof_color_consistent"):
                continue
            variants = asset.get("roof_color_variants") or ()
            if not variants:
                continue
            family = variants[0].get("family")
            path = asset.get("path")
            if family and path:
                result.setdefault(family, {})[path] = asset
    return result


def roof_metadata_signature(asset):
    """Return a compact stable representation for placement cache signatures."""
    variants = []
    for item in asset.get("roof_color_variants") or ():
        variants.append((
            item.get("family"),
            tuple(round(float(value), 3) for value in item.get("lab", ())),
            _file_signature(item.get("obj_path")),
            _file_signature(item.get("texture_path")),
        ))
    return (
        bool(asset.get("roof_color_complete")),
        bool(asset.get("roof_color_consistent")),
        tuple(variants),
    )
