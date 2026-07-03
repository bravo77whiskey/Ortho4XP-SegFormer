"""Helpers for locating and caching scenery DSF files used by SFR overlays."""

import hashlib
import os
import re
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


def _scenery_pack_path(custom_scenery_dir, pack_path):
    """Resolve one scenery_packs.ini path to an absolute scenery package path."""
    custom_scenery_dir = resolve_custom_scenery_dir(custom_scenery_dir)
    if not custom_scenery_dir:
        return None
    pack_path = (pack_path or "").strip().strip('"')
    if not pack_path:
        return None
    pack_path = pack_path.replace("/", os.sep).replace("\\", os.sep)
    if os.path.isabs(pack_path):
        return os.path.abspath(pack_path)

    custom_scenery_name = "Custom Scenery"
    parts = pack_path.split(os.sep)
    if parts and parts[0].lower() == custom_scenery_name.lower():
        xplane_root = os.path.dirname(custom_scenery_dir)
        return os.path.abspath(os.path.join(xplane_root, pack_path))
    return os.path.abspath(os.path.join(custom_scenery_dir, pack_path))


def active_scenery_pack_dirs(custom_scenery_dir):
    """Return enabled scenery package dirs in scenery_packs.ini order."""
    custom_scenery_dir = resolve_custom_scenery_dir(custom_scenery_dir)
    if not custom_scenery_dir or not os.path.isdir(custom_scenery_dir):
        return []

    ini_path = os.path.join(custom_scenery_dir, "scenery_packs.ini")
    if not os.path.isfile(ini_path):
        return []

    packs = []
    seen = set()
    with open(ini_path, "r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            keyword, pack_path = parts[0].upper(), parts[1]
            if keyword == "SCENERY_PACK_DISABLED":
                continue
            if keyword != "SCENERY_PACK":
                continue
            resolved = _scenery_pack_path(custom_scenery_dir, pack_path)
            if not resolved or not os.path.isdir(resolved):
                continue
            real_path = os.path.realpath(resolved)
            if real_path in seen:
                continue
            seen.add(real_path)
            packs.append((os.path.basename(resolved.rstrip("\\/")) or resolved, resolved))
    return packs


def _scenery_packs_ini_exists(custom_scenery_dir):
    custom_scenery_dir = resolve_custom_scenery_dir(custom_scenery_dir)
    if not custom_scenery_dir or not os.path.isdir(custom_scenery_dir):
        return False
    return os.path.isfile(os.path.join(custom_scenery_dir, "scenery_packs.ini"))


def simheaven_package_region_from_name(folder_name):
    """Return the X-World package family encoded in a simHeaven folder name."""
    normalized = re.sub(
        r"[^a-z0-9]+",
        "-",
        (folder_name or "").strip().lower(),
    ).strip("-")
    if not normalized:
        return None
    if (
        "simheaven" not in normalized
        and "x-world" not in normalized
        and not normalized.startswith("x-")
    ):
        return None
    if "australia-oceania" in normalized or (
        "australia" in normalized and "oceania" in normalized
    ):
        return "australia_oceania"
    if "antarctica" in normalized:
        return "antarctica"
    if "americas" in normalized or "america" in normalized:
        return "america"
    if "europe" in normalized:
        return "europe"
    if "africa" in normalized:
        return "africa"
    if "asia" in normalized:
        return "asia"
    return None


def _scan_custom_scenery(custom_scenery_dir, tile_dsf_relpath, folder_filter):
    """Yield matching DSF files from a configured Custom Scenery directory."""
    custom_scenery_dir = resolve_custom_scenery_dir(custom_scenery_dir)
    if not custom_scenery_dir or not os.path.isdir(custom_scenery_dir):
        return []

    matches = []
    seen_paths = set()
    if _scenery_packs_ini_exists(custom_scenery_dir):
        entries = active_scenery_pack_dirs(custom_scenery_dir)
    else:
        entries = [
            (entry.name, entry.path)
            for entry in sorted(
                os.scandir(custom_scenery_dir),
                key=lambda item: item.name.lower(),
            )
            if entry.is_dir()
        ]
    for folder_name, package_dir in entries:
        if not folder_filter(folder_name.lower()):
            continue
        dsf_path = os.path.join(package_dir, tile_dsf_relpath)
        if not os.path.isfile(dsf_path):
            continue
        resolved_path = os.path.realpath(dsf_path)
        if resolved_path in seen_paths:
            continue
        seen_paths.add(resolved_path)
        matches.append((folder_name, dsf_path))
    return matches


def find_simheaven_package_region_for_tile(custom_scenery_dir, lat, lon):
    """Return ``(region, folder_name)`` for the first matching X-World tile DSF."""
    tile_dsf_relpath = _tile_dsf_relpath(lat, lon)
    matches = _scan_custom_scenery(
        custom_scenery_dir,
        tile_dsf_relpath,
        lambda folder_name: simheaven_package_region_from_name(folder_name) is not None,
    )
    for folder_name, _dsf_path in matches:
        package_region = simheaven_package_region_from_name(folder_name)
        if package_region:
            return package_region, folder_name
    return None, None


def find_active_custom_scenery_dsfs(custom_scenery_dir, tile_dsf_name, skip_dsf_path=None):
    """Return active scenery DSFs with a matching tile basename."""
    custom_scenery_dir = resolve_custom_scenery_dir(custom_scenery_dir)
    if not custom_scenery_dir or not os.path.isdir(custom_scenery_dir):
        return []

    tile_dsf_name = os.path.basename(tile_dsf_name or "")
    if not tile_dsf_name:
        return []
    tile_dsf_name_lower = tile_dsf_name.lower()
    skip_real = os.path.realpath(skip_dsf_path) if skip_dsf_path else None

    matches = []
    seen_paths = set()
    for folder_name, package_dir in active_scenery_pack_dirs(custom_scenery_dir):
        earth_nav_data = os.path.join(package_dir, "Earth nav data")
        if not os.path.isdir(earth_nav_data):
            continue
        for root, _, files in os.walk(earth_nav_data):
            match_name = next((name for name in files if name.lower() == tile_dsf_name_lower), None)
            if match_name is None:
                continue
            dsf_path = os.path.join(root, match_name)
            resolved_path = os.path.realpath(dsf_path)
            if skip_real and resolved_path == skip_real:
                continue
            if resolved_path in seen_paths:
                continue
            seen_paths.add(resolved_path)
            matches.append((folder_name, dsf_path, package_dir))
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
