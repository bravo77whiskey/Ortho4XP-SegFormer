"""Helpers for locating and caching scenery DSF files used by SFR overlays."""

import hashlib
import os
import subprocess


def resolve_custom_scenery_dir(custom_scenery_dir):
    """Accept either X-Plane root or Custom Scenery and return Custom Scenery."""
    if not custom_scenery_dir:
        return custom_scenery_dir
    custom_scenery_dir = os.path.abspath(custom_scenery_dir)
    if os.path.basename(custom_scenery_dir).lower() == "custom scenery":
        return custom_scenery_dir
    child = os.path.join(custom_scenery_dir, "Custom Scenery")
    if os.path.isdir(child):
        return child
    return custom_scenery_dir


def _tile_dsf_relpath(lat, lon):
    """Return the Earth nav data relative path for a 1x1 tile DSF."""
    lat_int = int(lat)
    lon_int = int(lon)
    lat_group = int(lat_int // 10) * 10
    lon_group = int(lon_int // 10) * 10
    lat_tile = f"{'+' if lat_int >= 0 else '-'}{abs(lat_int):02d}"
    lon_tile = f"{'+' if lon_int >= 0 else '-'}{abs(lon_int):03d}"
    lat_block = f"{'+' if lat_group >= 0 else '-'}{abs(lat_group):02d}"
    lon_block = f"{'+' if lon_group >= 0 else '-'}{abs(lon_group):03d}"
    return os.path.join("Earth nav data", f"{lat_block}{lon_block}", f"{lat_tile}{lon_tile}.dsf")


def _scan_custom_scenery(custom_scenery_dir, tile_dsf_relpath, folder_filter):
    """Yield matching DSF files from a configured Custom Scenery directory."""
    custom_scenery_dir = resolve_custom_scenery_dir(custom_scenery_dir)
    if not custom_scenery_dir or not os.path.isdir(custom_scenery_dir):
        return []

    matches = []
    seen_paths = set()
    for entry in sorted(os.scandir(custom_scenery_dir), key=lambda item: item.name.lower()):
        if not entry.is_dir():
            continue
        if not folder_filter(entry.name.lower()):
            continue
        dsf_path = os.path.join(entry.path, tile_dsf_relpath)
        if not os.path.isfile(dsf_path):
            continue
        resolved_path = os.path.realpath(dsf_path)
        if resolved_path in seen_paths:
            continue
        seen_paths.add(resolved_path)
        matches.append((entry.name, dsf_path))
    return matches


def find_simheaven_network_dsfs(custom_scenery_dir, lat, lon):
    """Return simHeaven network DSFs for a tile from the configured Custom Scenery folder."""
    tile_dsf_relpath = _tile_dsf_relpath(lat, lon)
    return _scan_custom_scenery(
        custom_scenery_dir,
        tile_dsf_relpath,
        lambda folder_name: "simheaven" in folder_name and "network" in folder_name,
    )


def find_simheaven_building_dsfs(custom_scenery_dir, lat, lon):
    """Return simHeaven building-footprint/scenery DSFs for a tile from Custom Scenery."""
    tile_dsf_relpath = _tile_dsf_relpath(lat, lon)
    return _scan_custom_scenery(
        custom_scenery_dir,
        tile_dsf_relpath,
        lambda folder_name: "simheaven" in folder_name
        and ("footprints" in folder_name or "scenery" in folder_name),
    )


def find_global_forests_dsfs(custom_scenery_dir, lat, lon):
    """Return Global Forests v2 DSFs for a tile from Custom Scenery."""
    tile_dsf_relpath = _tile_dsf_relpath(lat, lon)
    return _scan_custom_scenery(
        custom_scenery_dir,
        tile_dsf_relpath,
        lambda folder_name: "global" in folder_name and "forest" in folder_name,
    )


def find_simheaven_vegetation_dsfs(custom_scenery_dir, lat, lon):
    """Return simHeaven DSFs that may contain vegetation overlays for a tile."""
    tile_dsf_relpath = _tile_dsf_relpath(lat, lon)
    return _scan_custom_scenery(
        custom_scenery_dir,
        tile_dsf_relpath,
        lambda folder_name: "simheaven" in folder_name and "network" not in folder_name,
    )


def find_default_overlay_dsfs(custom_overlay_src, lat, lon, alternate_dir=None):
    """Return the configured default overlay-source DSF(s) for a tile."""
    tile_dsf_relpath = _tile_dsf_relpath(lat, lon)
    matches = []
    seen_paths = set()
    for source_dir in (custom_overlay_src, alternate_dir):
        if not source_dir or not os.path.isdir(source_dir):
            continue
        dsf_path = os.path.join(source_dir, tile_dsf_relpath)
        if not os.path.isfile(dsf_path):
            continue
        resolved_path = os.path.realpath(dsf_path)
        if resolved_path in seen_paths:
            continue
        seen_paths.add(resolved_path)
        matches.append((os.path.basename(source_dir.rstrip("\\/")) or source_dir, dsf_path))
    return matches


def cached_dsf_text_path(dsf_path, cache_dir):
    """Return the cache path for the disassembled text version of a DSF file."""
    stat = os.stat(dsf_path)
    cache_key = hashlib.sha1(
        f"{os.path.realpath(dsf_path)}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()
    return os.path.join(cache_dir, "dsf_disassembly", f"{cache_key}.txt")


def ensure_cached_dsf_text(dsf_path, dsftool_path, cache_dir, create_no_window=0):
    """Disassemble a DSF into a persistent cache file and return that file path."""
    text_cache_path = cached_dsf_text_path(dsf_path, cache_dir)
    if os.path.isfile(text_cache_path):
        return text_cache_path

    os.makedirs(os.path.dirname(text_cache_path), exist_ok=True)
    subprocess.run(
        [dsftool_path, "--dsf2text", dsf_path, text_cache_path],
        capture_output=True,
        timeout=120,
        creationflags=create_no_window,
        check=True,
    )
    return text_cache_path
