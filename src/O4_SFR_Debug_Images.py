"""Image-first SFR debug previews for explicit DDS textures.

This module deliberately stops before DSF text writing.  It loads one or more
Ortho4XP DDS textures, obtains the current SegFormer class map, and writes PNG
review artifacts that show the source image, masks, procedural zones, proposed
placements, and a composite overview.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import O4_SFR_Building_Overlay as BLD
import O4_SFR_Inference as SEGFORMER


DEFAULT_TEST_FIXTURES = (
    r"H:\XP12 Addons\Applications\Ortho4XP\Tiles\zOrtho4XP_+36+117\textures\25680_54080_BI16.dds",
    r"H:\XP12 Addons\Applications\Ortho4XP\Tiles\zOrtho4XP_+36+117\textures\25680_54096_BI16.dds",
    r"F:\XP12 Addons\Custom Scenery\zOrtho4XP_+22+113\textures\28512_53472_BI16.dds",
    r"F:\XP12 Addons\Custom Scenery\zOrtho4XP_+22+113\textures\28528_53488_BI16.dds",
)

_DDS_RE = re.compile(r"^(\d+)_(\d+)_([A-Za-z][A-Za-z0-9_]*)(\d{2})\.dds$", re.IGNORECASE)
_TILE_RE = re.compile(r"zOrtho4XP_([+-]\d{2})([+-]\d{3})", re.IGNORECASE)

VEG_ZONE_LABELS = {
    0: "none",
    1: "closed_forest",
    2: "scrub_shrub",
    3: "cropland_edge",
}

VEG_ZONE_COLORS = {
    0: (0, 0, 0, 0),
    1: (20, 132, 70, 180),
    2: (189, 169, 69, 180),
    3: (156, 205, 92, 165),
}

BLD_ZONE_LABELS = {
    BLD.BLD_CLASS_COMPACT_RESIDENTIAL: "compact_residential",
    BLD.BLD_CLASS_MEDIUM: "dense_residential",
    BLD.BLD_CLASS_SMALL_APARTMENT: "urban_mid",
    BLD.BLD_CLASS_APARTMENT_BLOCK: "small_apartment",
    BLD.BLD_CLASS_LARGE: "industrial_large",
}

BLD_ZONE_COLORS = {
    BLD.BLD_CLASS_COMPACT_RESIDENTIAL: (64, 200, 88, 190),
    BLD.BLD_CLASS_MEDIUM: (255, 195, 60, 190),
    BLD.BLD_CLASS_SMALL_APARTMENT: (255, 129, 40, 190),
    BLD.BLD_CLASS_APARTMENT_BLOCK: (65, 170, 235, 190),
    BLD.BLD_CLASS_LARGE: (207, 92, 220, 190),
}


@dataclass(frozen=True)
class TextureDebugMetadata:
    path: str
    name: str
    stem: str
    tile_lat: int | None
    tile_lon: int | None
    til_y_top: int
    til_x_left: int
    provider: str
    zoomlevel: int


def parse_texture_metadata(path: str | os.PathLike[str]) -> TextureDebugMetadata:
    """Return Ortho4XP metadata inferred from a standard DDS texture path."""
    texture_path = Path(path)
    match = _DDS_RE.match(texture_path.name)
    if not match:
        raise ValueError(f"Unsupported DDS filename: {texture_path.name!r}")

    tile_lat = tile_lon = None
    for part in texture_path.parts:
        tile_match = _TILE_RE.match(part)
        if tile_match:
            tile_lat = int(tile_match.group(1))
            tile_lon = int(tile_match.group(2))
            break

    return TextureDebugMetadata(
        path=str(texture_path),
        name=texture_path.name,
        stem=texture_path.stem,
        tile_lat=tile_lat,
        tile_lon=tile_lon,
        til_y_top=int(match.group(1)),
        til_x_left=int(match.group(2)),
        provider=match.group(3),
        zoomlevel=int(match.group(4)),
    )


def _load_rgb_image(path: str | os.PathLike[str]) -> np.ndarray:
    """Load DDS/PNG/JPEG imagery as an RGB uint8 array."""
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _resize_image(image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return image
    scale = min(1.0, float(max_side) / float(max(image.size)))
    if scale >= 1.0:
        return image
    size = (max(1, int(round(image.size[0] * scale))), max(1, int(round(image.size[1] * scale))))
    return image.resize(size, Image.Resampling.LANCZOS)


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if mask.shape[:2] == (size[1], size[0]):
        return mask
    return cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)


def _rgba_from_labels(labels: np.ndarray, colors: dict[int, tuple[int, int, int, int]]) -> Image.Image:
    rgba = np.zeros((labels.shape[0], labels.shape[1], 4), dtype=np.uint8)
    for value, color in colors.items():
        if value == 0:
            continue
        rgba[labels == value] = color
    return Image.fromarray(rgba, "RGBA")


def _binary_rgba(mask: np.ndarray, color: tuple[int, int, int, int]) -> Image.Image:
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    rgba[mask.astype(bool)] = color
    return Image.fromarray(rgba, "RGBA")


def _overlay(base_rgb: Image.Image, layer_rgba: Image.Image) -> Image.Image:
    return Image.alpha_composite(base_rgb.convert("RGBA"), layer_rgba).convert("RGB")


def _clean_mask(mask: np.ndarray, close_px: int, open_px: int) -> np.ndarray:
    result = mask.astype(np.uint8)
    if close_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    if open_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px * 2 + 1, open_px * 2 + 1))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    return result


def _dds_m_per_px(lat_n: float, lat_s: float, lon_w: float, lon_e: float,
                  img_h: int, img_w: int) -> float:
    mid_lat = np.deg2rad((float(lat_n) + float(lat_s)) / 2.0)
    lon_m = (float(lon_e) - float(lon_w)) * 111320.0 * float(np.cos(mid_lat))
    lat_m = (float(lat_n) - float(lat_s)) * 110540.0
    return float((lon_m / max(1, img_w) + lat_m / max(1, img_h)) / 2.0)


def _vegetation_zones(class_map: np.ndarray) -> np.ndarray:
    zones = np.zeros(class_map.shape, dtype=np.uint8)
    tree = _clean_mask((class_map == SEGFORMER.CLASS_TREE).astype(np.uint8), 4, 1)
    rangeland = _clean_mask((class_map == SEGFORMER.CLASS_RANGELAND).astype(np.uint8), 3, 1)
    agriculture = _clean_mask((class_map == SEGFORMER.CLASS_AGRICULTURE).astype(np.uint8), 2, 1)
    zones[agriculture != 0] = 3
    zones[rangeland != 0] = 2
    zones[tree != 0] = 1
    return zones


def _building_zones(class_map: np.ndarray, min_zone_px: int = 64) -> np.ndarray:
    bld = _clean_mask((class_map == SEGFORMER.CLASS_BUILDING).astype(np.uint8), 8, 2)
    n_cc, labels, stats, _centroids = cv2.connectedComponentsWithStats(bld, connectivity=8)
    zone_class_by_label = np.zeros(n_cc, dtype=np.uint8)
    if n_cc <= 1:
        return zone_class_by_label[labels]
    areas = stats[:, cv2.CC_STAT_AREA]
    valid = np.flatnonzero((np.arange(n_cc) != 0) & (areas >= min_zone_px))
    if valid.size:
        valid_area = areas[valid]
        zone_class_by_label[valid[valid_area < BLD.ZONE_COMPACT_PX]] = BLD.BLD_CLASS_COMPACT_RESIDENTIAL
        zone_class_by_label[
            valid[(valid_area >= BLD.ZONE_COMPACT_PX) & (valid_area < BLD.ZONE_MEDIUM_PX)]
        ] = BLD.BLD_CLASS_MEDIUM
        zone_class_by_label[
            valid[(valid_area >= BLD.ZONE_MEDIUM_PX) & (valid_area < BLD.ZONE_SMALL_APARTMENT_PX)]
        ] = BLD.BLD_CLASS_SMALL_APARTMENT
        zone_class_by_label[
            valid[(valid_area >= BLD.ZONE_SMALL_APARTMENT_PX) & (valid_area < BLD.ZONE_APARTMENT_BLOCK_PX)]
        ] = BLD.BLD_CLASS_APARTMENT_BLOCK
        zone_class_by_label[valid[valid_area >= BLD.ZONE_APARTMENT_BLOCK_PX]] = BLD.BLD_CLASS_LARGE
    return zone_class_by_label[labels]


def _proposed_building_footprints(
    building_zones: np.ndarray,
    m_per_px: float,
    max_footprints: int = 600,
) -> list[dict]:
    """Create lightweight footprint proposals from procedural building zones."""
    markers: list[dict] = []
    n_cc, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (building_zones != 0).astype(np.uint8), connectivity=8
    )
    for label in range(1, n_cc):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if w <= 0 or h <= 0:
            continue
        zone_values = building_zones[labels == label]
        zone_values = zone_values[zone_values != 0]
        if zone_values.size == 0:
            continue
        zone_class = int(np.bincount(zone_values.astype(np.int32)).argmax())
        span = {
            BLD.BLD_CLASS_COMPACT_RESIDENTIAL: 42,
            BLD.BLD_CLASS_MEDIUM: 56,
            BLD.BLD_CLASS_SMALL_APARTMENT: 74,
            BLD.BLD_CLASS_APARTMENT_BLOCK: 92,
            BLD.BLD_CLASS_LARGE: 130,
        }.get(zone_class, 64)
        step = max(18, span)
        xs = np.arange(x + step // 2, x + w, step)
        ys = np.arange(y + step // 2, y + h, step)
        for cy in ys:
            for cx in xs:
                if len(markers) >= max_footprints:
                    return markers
                if building_zones[int(cy), int(cx)] != zone_class:
                    continue
                bounds_m = BLD.DEFAULT_FACADE_BOUNDS.get(zone_class, BLD.DEFAULT_FACADE_BOUNDS[BLD.BLD_CLASS_MEDIUM])
                xmin, xmax, zmin, zmax = bounds_m
                width_px = max(4, int(round((float(xmax) - float(xmin)) / max(m_per_px, 1e-6))))
                depth_px = max(4, int(round((float(zmax) - float(zmin)) / max(m_per_px, 1e-6))))
                markers.append({
                    "x": int(cx),
                    "y": int(cy),
                    "class": zone_class,
                    "width_px": width_px,
                    "depth_px": depth_px,
                    "bounds_m": tuple(float(v) for v in bounds_m),
                    "label": BLD_ZONE_LABELS.get(zone_class, "unknown"),
                })
    return markers


def _placement_layer(shape: tuple[int, int], markers: list[dict]) -> Image.Image:
    layer = Image.new("RGBA", (shape[1], shape[0]), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for marker in markers:
        color = BLD_ZONE_COLORS.get(int(marker["class"]), (255, 255, 255, 190))
        half_w = max(2, int(marker.get("width_px", 8)) // 2)
        half_h = max(2, int(marker.get("depth_px", 8)) // 2)
        x = int(marker["x"])
        y = int(marker["y"])
        draw.rectangle(
            (x - half_w, y - half_h, x + half_w, y + half_h),
            outline=color[:3] + (245,),
            fill=color,
        )
    return layer


def _lonlat_to_px(lon: float, lat: float, img_w: int, img_h: int,
                  lat_n: float, lat_s: float, lon_w: float, lon_e: float) -> tuple[float, float]:
    px = (float(lon) - float(lon_w)) / (float(lon_e) - float(lon_w)) * float(img_w)
    py = (float(lat_n) - float(lat)) / (float(lat_n) - float(lat_s)) * float(img_h)
    return px, py


def _load_cached_building_placements(
    dds_path: str | os.PathLike[str],
    cache_dir: str | os.PathLike[str] | None,
) -> dict | None:
    if not cache_dir:
        return None
    cache_path = Path(cache_dir) / (Path(dds_path).stem + "_bld.pkl")
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as handle:
            data = pickle.load(handle)
    except Exception:
        return None
    placements = data.get("placements")
    if not isinstance(placements, dict):
        return None
    return {
        "objects": tuple(placements.get("objects", ())),
        "facades": tuple(placements.get("facades", ())),
        "cache_path": str(cache_path),
    }


def _cached_placement_layer(
    shape: tuple[int, int],
    cached_placements: dict,
    metadata: TextureDebugMetadata,
    lat_n: float,
    lat_s: float,
    lon_w: float,
    lon_e: float,
    m_per_px: float,
) -> Image.Image:
    """Draw production cached object/facade footprints in image pixel space."""
    img_h, img_w = shape
    layer = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    for lonlat_ring, facade_path, _height_m in cached_placements.get("facades", ()):
        pts = [
            _lonlat_to_px(lon, lat, img_w, img_h, lat_n, lat_s, lon_w, lon_e)
            for lon, lat in lonlat_ring
        ]
        if len(pts) < 3:
            continue
        color = BLD_ZONE_COLORS.get(BLD.BLD_CLASS_MEDIUM, (255, 180, 60, 190))
        lower_path = (facade_path or "").lower()
        if "warehouse" in lower_path or "industrial" in lower_path:
            color = BLD_ZONE_COLORS.get(BLD.BLD_CLASS_LARGE, color)
        elif "high_" in lower_path:
            color = BLD_ZONE_COLORS.get(BLD.BLD_CLASS_APARTMENT_BLOCK, color)
        elif "low_" in lower_path:
            color = BLD_ZONE_COLORS.get(BLD.BLD_CLASS_COMPACT_RESIDENTIAL, color)
        draw.polygon(pts, fill=color, outline=color[:3] + (245,))

    for lon, lat, heading, obj_path in cached_placements.get("objects", ()):
        cx, cy = _lonlat_to_px(lon, lat, img_w, img_h, lat_n, lat_s, lon_w, lon_e)
        bounds_m = BLD._bounds_for_object_path(obj_path)
        if bounds_m is None:
            dims = BLD._default_object_dims(obj_path) or BLD._simheaven_object_dims(obj_path)
            bounds_m = BLD._bounds_from_dimensions(dims[0], dims[1])
        zone_class = BLD._class_for_object_asset(obj_path, bounds_m)
        pts = BLD._footprint_poly(int(round(cx)), int(round(cy)), bounds_m, float(heading), m_per_px)
        if pts is None or len(pts) < 3:
            continue
        color = BLD_ZONE_COLORS.get(zone_class, (255, 255, 255, 190))
        draw.polygon([(float(x), float(y)) for x, y in pts], fill=color, outline=color[:3] + (245,))

    return layer


def _scale_cached_placement_layer(layer: Image.Image, preview_size: tuple[int, int]) -> Image.Image:
    if layer.size == preview_size:
        return layer
    return layer.resize(preview_size, Image.Resampling.NEAREST)


def _legend_block(counts: dict, width: int = 520) -> Image.Image:
    lines = [
        ("SFR debug preview", (255, 255, 255)),
        (f"DDS: {counts['name']}", (230, 230, 230)),
        (f"zoom={counts['zoomlevel']}  tile={counts.get('tile') or 'unknown'}", (210, 210, 210)),
        ("", (255, 255, 255)),
        ("Vegetation zones", (255, 255, 255)),
    ]
    for value, label in VEG_ZONE_LABELS.items():
        if value == 0:
            continue
        lines.append((f"{label}: {counts['vegetation_zones'].get(label, 0):,} px", VEG_ZONE_COLORS[value][:3]))
    lines.append(("", (255, 255, 255)))
    lines.append(("Building zones", (255, 255, 255)))
    for cls, label in BLD_ZONE_LABELS.items():
        lines.append((f"{label}: {counts['building_zones'].get(label, 0):,} px", BLD_ZONE_COLORS[cls][:3]))
    lines.append(("", (255, 255, 255)))
    source = counts.get("placement_source", "synthetic preview")
    lines.append((f"Proposed footprints: {counts['placements']:,}", (255, 255, 255)))
    lines.append((f"Footprint source: {source}", (210, 210, 210)))
    lines.append(("Existing-overlay exclusions: unavailable in DDS-only mode", (190, 190, 190)))

    font = ImageFont.load_default()
    line_h = 16
    image = Image.new("RGB", (width, max(120, 18 + line_h * len(lines))), (24, 24, 28))
    draw = ImageDraw.Draw(image)
    y = 10
    for text, color in lines:
        if text:
            draw.text((12, y), text, fill=color, font=font)
        y += line_h
    return image


def _count_labels(labels: np.ndarray, mapping: dict[int, str]) -> dict[str, int]:
    result = {label: 0 for label in mapping.values() if label != "none"}
    values, counts = np.unique(labels, return_counts=True)
    for value, count in zip(values, counts):
        label = mapping.get(int(value))
        if label and label != "none":
            result[label] = int(count)
    return result


def _load_cached_class_map(dds_path: str, cache_dir: str | os.PathLike[str] | None) -> np.ndarray | None:
    if not cache_dir:
        return None
    cache_path = Path(cache_dir) / (Path(dds_path).stem + "_veg.npy")
    if not cache_path.exists():
        return None
    arr = np.load(cache_path)
    if arr.ndim != 2:
        raise ValueError(f"Cached class map is not 2-D: {cache_path}")
    return arr.astype(np.int8, copy=False)


def _infer_class_map(image_rgb: np.ndarray, device=None) -> np.ndarray:
    model, processor, device = SEGFORMER.load_vegetation_model(device)
    return SEGFORMER.run_inference(model, device, image_rgb, processor)


def generate_debug_images_for_dds(
    dds_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    class_map: np.ndarray | None = None,
    device=None,
    max_preview_px: int = 1600,
) -> dict:
    """Generate image-only debug artifacts for one DDS texture."""
    metadata = parse_texture_metadata(dds_path)
    image_rgb = _load_rgb_image(dds_path)
    if class_map is None:
        class_map = _load_cached_class_map(str(dds_path), cache_dir)
    if class_map is None:
        class_map = _infer_class_map(image_rgb, device=device)
    if class_map.shape != image_rgb.shape[:2]:
        class_map = cv2.resize(class_map, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    veg_zones = _vegetation_zones(class_map)
    building_zones = _building_zones(class_map)
    _lat_n, _lat_s, _lon_w, _lon_e = BLD.dds_bounds(
        metadata.til_y_top, metadata.til_x_left, metadata.zoomlevel
    )
    m_per_px = _dds_m_per_px(
        _lat_n, _lat_s, _lon_w, _lon_e, image_rgb.shape[0], image_rgb.shape[1]
    )
    cached_placements = _load_cached_building_placements(str(dds_path), cache_dir)
    placements = (
        []
        if cached_placements is not None
        else _proposed_building_footprints(building_zones, m_per_px)
    )

    out_root = Path(output_dir) / metadata.stem
    out_root.mkdir(parents=True, exist_ok=True)

    base_full = Image.fromarray(image_rgb, "RGB")
    base = _resize_image(base_full, max_preview_px)
    preview_size = base.size
    veg_preview = _resize_mask(veg_zones, preview_size)
    bld_preview = _resize_mask(building_zones, preview_size)
    tree_mask_preview = _resize_mask((class_map == SEGFORMER.CLASS_TREE).astype(np.uint8), preview_size)
    bld_mask_preview = _resize_mask((class_map == SEGFORMER.CLASS_BUILDING).astype(np.uint8), preview_size)
    exclusion_preview = np.zeros((preview_size[1], preview_size[0]), dtype=np.uint8)

    source_path = out_root / f"{metadata.stem}_source.png"
    seg_mask_path = out_root / f"{metadata.stem}_segformer_masks.png"
    exclusion_path = out_root / f"{metadata.stem}_existing_exclusions.png"
    veg_zone_path = out_root / f"{metadata.stem}_vegetation_zones.png"
    bld_zone_path = out_root / f"{metadata.stem}_building_zones.png"
    placement_path = out_root / f"{metadata.stem}_proposed_footprints.png"
    composite_path = out_root / f"{metadata.stem}_composite.png"
    counts_path = out_root / f"{metadata.stem}_counts.json"

    base.save(source_path)
    mask_overlay = _overlay(
        base,
        Image.alpha_composite(
            _binary_rgba(tree_mask_preview, (18, 145, 74, 150)),
            _binary_rgba(bld_mask_preview, (242, 91, 69, 150)),
        ),
    )
    mask_overlay.save(seg_mask_path)
    _overlay(base, _binary_rgba(exclusion_preview, (40, 40, 40, 180))).save(exclusion_path)
    _overlay(base, _rgba_from_labels(veg_preview, VEG_ZONE_COLORS)).save(veg_zone_path)
    _overlay(base, _rgba_from_labels(bld_preview, BLD_ZONE_COLORS)).save(bld_zone_path)

    scale_x = preview_size[0] / float(image_rgb.shape[1])
    scale_y = preview_size[1] / float(image_rgb.shape[0])
    if cached_placements is not None:
        placement_layer_full = _cached_placement_layer(
            image_rgb.shape[:2],
            cached_placements,
            metadata,
            _lat_n,
            _lat_s,
            _lon_w,
            _lon_e,
            m_per_px,
        )
        placement_layer = _scale_cached_placement_layer(placement_layer_full, preview_size)
        placement_count = (
            len(cached_placements.get("objects", ())) +
            len(cached_placements.get("facades", ()))
        )
        placement_source = "production building cache"
    else:
        preview_markers = [
            {
                **marker,
                "x": int(round(marker["x"] * scale_x)),
                "y": int(round(marker["y"] * scale_y)),
                "width_px": max(7, int(round(marker["width_px"] * scale_x))),
                "depth_px": max(7, int(round(marker["depth_px"] * scale_y))),
            }
            for marker in placements
        ]
        placement_layer = _placement_layer((preview_size[1], preview_size[0]), preview_markers)
        placement_count = len(placements)
        placement_source = "synthetic fallback"
    _overlay(base, placement_layer).save(placement_path)

    counts = {
        "name": metadata.name,
        "path": metadata.path,
        "tile": (
            f"{metadata.tile_lat:+03d}{metadata.tile_lon:+04d}"
            if metadata.tile_lat is not None and metadata.tile_lon is not None
            else None
        ),
        "zoomlevel": metadata.zoomlevel,
        "m_per_px": float(m_per_px),
        "size": {"width": int(image_rgb.shape[1]), "height": int(image_rgb.shape[0])},
        "vegetation_zones": _count_labels(veg_zones, VEG_ZONE_LABELS),
        "building_zones": _count_labels(building_zones, BLD_ZONE_LABELS),
        "placements": int(placement_count),
        "placement_source": placement_source,
        "placement_cache": cached_placements.get("cache_path") if cached_placements else None,
        "outputs": {
            "source": str(source_path),
            "segformer_masks": str(seg_mask_path),
            "existing_exclusions": str(exclusion_path),
            "vegetation_zones": str(veg_zone_path),
            "building_zones": str(bld_zone_path),
            "proposed_footprints": str(placement_path),
            "composite": str(composite_path),
        },
    }
    counts_path.write_text(json.dumps(counts, indent=2), encoding="utf-8")

    legend = _legend_block(counts)
    composite = Image.new("RGB", (base.width + legend.width, max(base.height, legend.height)), (24, 24, 28))
    composite.paste(_overlay(_overlay(base, _rgba_from_labels(veg_preview, VEG_ZONE_COLORS)), placement_layer), (0, 0))
    composite.paste(legend, (base.width, 0))
    composite.save(composite_path)
    return counts


def generate_debug_images(
    dds_paths: list[str],
    output_dir: str | os.PathLike[str],
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    device=None,
    max_preview_px: int = 1600,
) -> list[dict]:
    results = []
    for dds_path in dds_paths:
        print(f"[SFR debug] {dds_path}", flush=True)
        results.append(
            generate_debug_images_for_dds(
                dds_path,
                output_dir,
                cache_dir=cache_dir,
                device=device,
                max_preview_px=max_preview_px,
            )
        )
    return results


def _group_paths_by_tile(dds_paths: list[str]) -> dict[tuple[str, int, int], list[str]]:
    groups: dict[tuple[str, int, int], list[str]] = {}
    for dds_path in dds_paths:
        metadata = parse_texture_metadata(dds_path)
        if metadata.tile_lat is None or metadata.tile_lon is None:
            raise ValueError(f"Could not infer zOrtho4XP tile from path: {dds_path}")
        key = (str(Path(dds_path).parent), metadata.tile_lat, metadata.tile_lon)
        groups.setdefault(key, []).append(str(Path(dds_path)))
    return groups


def generate_production_building_debug_images(
    dds_paths: list[str],
    output_dir: str | os.PathLike[str],
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    spacing_m: float = 20.0,
    close_k: int = 15,
    open_k: int = 5,
    min_zone_m2: float = 200.0,
    grid_n: int = BLD.HEADING_GRID_N,
    custom_scenery_dir: str | os.PathLike[str] | None = None,
    skip_osm_excl_download: bool = False,
    smart_gap_fill: bool = True,
    viz_size: int = 1600,
    yolo_checkpoint: str | os.PathLike[str] | None = None,
    yolo_conf: float | None = None,
    yolo_iou: float | None = None,
    yolo_stride: int | None = None,
    yolo_max_det: int | None = None,
    yolo_suppress_coverage: float = 0.0,
    yolo_suppress_min_overlap_m2: float = 25.0,
    allow_road_overlap: bool = False,
) -> list[dict]:
    """Run the real building placement pipeline and stop after debug images."""
    output_dir = Path(output_dir)
    runtime_cache = Path(cache_dir) if cache_dir else output_dir / "_runtime_cache"
    runtime_cache.mkdir(parents=True, exist_ok=True)
    results = []
    old_viz_size = os.environ.get("O4_SFR_BLD_VIZ_SIZE")
    os.environ["O4_SFR_BLD_VIZ_SIZE"] = str(int(viz_size))
    try:
        for (tex_dir, lat, lon), paths in _group_paths_by_tile(dds_paths).items():
            tile_label = f"{lat:+03d}{lon:+04d}"
            names = [Path(path).name for path in paths]
            out_dsf = output_dir / "production_building" / tile_label / f"{tile_label}_debug.dsf"
            out_dsf.parent.mkdir(parents=True, exist_ok=True)
            print(
                f"[SFR debug] Production building image pass {tile_label}: "
                f"{', '.join(names)}",
                flush=True,
            )
            placements = BLD.run(
                tex_dir=str(tex_dir),
                lat=lat,
                lon=lon,
                out_dsf=str(out_dsf),
                spacing_m=spacing_m,
                close_k=close_k,
                open_k=open_k,
                min_zone_m2=min_zone_m2,
                make_viz=True,
                cache_dir=str(runtime_cache),
                disable_cache=True,
                grid_n=grid_n,
                custom_scenery_dir=str(custom_scenery_dir) if custom_scenery_dir else None,
                skip_osm_excl_download=skip_osm_excl_download,
                smart_gap_fill=smart_gap_fill,
                debug_image_only=True,
                dds_filter=names,
                ignore_placement_cache=True,
                allow_road_overlap=allow_road_overlap,
                yolo_checkpoint=str(yolo_checkpoint) if yolo_checkpoint else None,
                yolo_conf=yolo_conf,
                yolo_iou=yolo_iou,
                yolo_stride=yolo_stride,
                yolo_max_det=yolo_max_det,
                yolo_suppress_coverage=yolo_suppress_coverage,
                yolo_suppress_min_overlap_m2=yolo_suppress_min_overlap_m2,
            )
            results.append({
                "tile": tile_label,
                "textures": names,
                "placements": int(placements or 0),
                "overview": str(out_dsf).replace(".dsf", "_overview.png"),
                "footprints": str(out_dsf).replace(".dsf", "_footprints.png"),
                "mode": "production_building_image_only",
            })
    finally:
        if old_viz_size is None:
            os.environ.pop("O4_SFR_BLD_VIZ_SIZE", None)
        else:
            os.environ["O4_SFR_BLD_VIZ_SIZE"] = old_viz_size
    summary_path = output_dir / "production_building_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Generate image-only SegFormer/procedural debug PNGs for explicit DDS textures."
    )
    parser.add_argument("dds", nargs="*", help="DDS texture path(s) to preview.")
    parser.add_argument(
        "--default-fixtures",
        action="store_true",
        help="Use the four initial project smoke-test DDS fixtures.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("tmp", "sfr_debug_images"),
        help="Directory where per-DDS PNG outputs will be written.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Optional runtime SFR cache directory. Production mode recomputes per-DDS "
            "class maps, road masks, and placements; the directory is still used for "
            "shared tile context such as extracted overlay text."
        ),
    )
    parser.add_argument("--cpu", action="store_true", help="Run SegFormer inference on CPU.")
    parser.add_argument(
        "--max-preview-px",
        type=int,
        default=1600,
        help="Maximum width/height for review PNGs. Use 0 for full 4096px output if disk space allows.",
    )
    parser.add_argument(
        "--quick-masks-only",
        action="store_true",
        help="Use the lightweight mask renderer instead of the production building-placement pipeline.",
    )
    parser.add_argument("--spacing-m", type=float, default=0.0)
    parser.add_argument("--close-k", type=int, default=15)
    parser.add_argument("--open-k", type=int, default=5)
    parser.add_argument("--min-zone-m2", type=float, default=200.0)
    parser.add_argument("--grid-n", type=int, default=BLD.HEADING_GRID_N)
    parser.add_argument("--custom-scenery-dir", default=None)
    parser.add_argument("--skip-osm-excl-download", action="store_true")
    parser.add_argument("--no-smart-gap-fill", action="store_true")
    parser.add_argument("--yolo-checkpoint", default=None)
    parser.add_argument("--yolo-conf", type=float, default=None)
    parser.add_argument("--yolo-iou", type=float, default=None)
    parser.add_argument("--yolo-stride", type=int, default=None)
    parser.add_argument("--yolo-max-det", type=int, default=None)
    parser.add_argument("--yolo-suppress-coverage", type=float, default=0.0)
    parser.add_argument("--yolo-suppress-min-overlap-m2", type=float, default=25.0)
    parser.add_argument(
        "--allow-road-overlap",
        action="store_true",
        help="Allow generated building footprints to overlap road masks.",
    )
    parser.add_argument(
        "--production-viz-size",
        type=int,
        default=1600,
        help="Per-DDS production overview/footprint image size.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = list(args.dds)
    if args.default_fixtures:
        paths.extend(DEFAULT_TEST_FIXTURES)
    if not paths:
        raise SystemExit("ERROR: provide DDS paths or pass --default-fixtures")

    if args.quick_masks_only:
        device = None
        if args.cpu:
            import torch
            device = torch.device("cpu")
        results = generate_debug_images(
            paths,
            args.output_dir,
            cache_dir=args.cache_dir,
            device=device,
            max_preview_px=args.max_preview_px,
        )
        summary_name = "summary.json"
    else:
        results = generate_production_building_debug_images(
            paths,
            args.output_dir,
            cache_dir=args.cache_dir,
            spacing_m=args.spacing_m,
            close_k=args.close_k,
            open_k=args.open_k,
            min_zone_m2=args.min_zone_m2,
            grid_n=args.grid_n,
            custom_scenery_dir=args.custom_scenery_dir,
            skip_osm_excl_download=args.skip_osm_excl_download,
            smart_gap_fill=not args.no_smart_gap_fill,
            viz_size=args.production_viz_size,
            yolo_checkpoint=args.yolo_checkpoint,
            yolo_conf=args.yolo_conf,
            yolo_iou=args.yolo_iou,
            yolo_stride=args.yolo_stride,
            yolo_max_det=args.yolo_max_det,
            yolo_suppress_coverage=args.yolo_suppress_coverage,
            yolo_suppress_min_overlap_m2=args.yolo_suppress_min_overlap_m2,
            allow_road_overlap=args.allow_road_overlap,
        )
        summary_name = "summary.json"
    summary_path = Path(args.output_dir) / summary_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[SFR debug] Wrote {len(results)} preview set(s) to {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
