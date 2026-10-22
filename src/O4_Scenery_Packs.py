"""Helpers for locating and caching scenery DSF files used by the overlays."""

import hashlib
import os
import re
import subprocess
import threading


def _entry_is_dir(entry):
    """``DirEntry.is_dir()`` that treats an unreadable (corrupt) entry as absent."""
    try:
        return entry.is_dir()
    except OSError as exc:
        print(f"Skipping unreadable folder {entry.path}: {exc}", flush=True)
        return False


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
            if _entry_is_dir(entry)
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


def find_simheaven_forest_dsfs(custom_scenery_dir, lat, lon):
    """Return enabled simHeaven X-World ``7-forests`` DSFs for a tile."""
    tile_dsf_relpath = _tile_dsf_relpath(lat, lon)
    return _scan_custom_scenery(
        custom_scenery_dir,
        tile_dsf_relpath,
        lambda folder_name: "simheaven" in folder_name
        and "7-forests" in folder_name,
    )


def cached_dsf_text_path(dsf_path, cache_dir):
    """Return the cache path for the disassembled text version of a DSF file."""
    stat = os.stat(dsf_path)
    cache_key = hashlib.sha1(
        f"{os.path.realpath(dsf_path)}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()
    return os.path.join(cache_dir, "dsf_disassembly", f"{cache_key}.txt")


# Other cache folders whose already-disassembled DSF text may be reused, e.g.
# the parent run's folder for a building-overlay texture helper process.
SHARED_DSF_TEXT_DIRS = []


def ensure_cached_dsf_text(dsf_path, dsftool_path, cache_dir, create_no_window=0):
    """Disassemble a DSF into a persistent cache file and return that file path."""
    text_cache_path = cached_dsf_text_path(dsf_path, cache_dir)
    if os.path.isfile(text_cache_path):
        return text_cache_path
    for shared_dir in SHARED_DSF_TEXT_DIRS:
        shared_path = cached_dsf_text_path(dsf_path, shared_dir)
        if os.path.isfile(shared_path):
            return shared_path

    os.makedirs(os.path.dirname(text_cache_path), exist_ok=True)
    # Write under a private name and rename, so a reader in another thread or
    # process never sees a half-written file.
    part_path = f"{text_cache_path}.{os.getpid()}.{threading.get_ident()}.part"
    try:
        subprocess.run(
            [dsftool_path, "--dsf2text", dsf_path, part_path],
            capture_output=True,
            timeout=120,
            creationflags=create_no_window,
            check=True,
        )
        os.replace(part_path, text_cache_path)
    finally:
        if os.path.exists(part_path):
            try:
                os.remove(part_path)
            except OSError:
                pass
    return text_cache_path


# A pack counts as a custom airport when it defines airports (`Earth nav
# data/apt.dat`) and stays inside a handful of degree tiles: regional VFR
# packs ship an apt.dat too. See O4_Vegetation_Overlay for why only airports'
# exclusion zones are honoured.
AIRPORT_PACK_MAX_DSFS = 8


def is_airport_scenery_pack(package_dir):
    """True when a scenery package is a custom airport rather than area scenery."""
    earth_nav_data = os.path.join(package_dir, "Earth nav data")
    if not os.path.isfile(os.path.join(earth_nav_data, "apt.dat")):
        return False
    n_dsf = 0
    for _root, _dirs, files in os.walk(earth_nav_data):
        for name in files:
            if name.lower().endswith(".dsf"):
                n_dsf += 1
                if n_dsf > AIRPORT_PACK_MAX_DSFS:
                    return False
    return True


def exclusion_zone_ring(value):
    """Parse a ``sim/exclude_*`` property value into an open (lon, lat) ring.

    Two forms exist: the plain rectangle ``west/south/east/north`` and WED's
    polygon zone ``west/south/east/north;lon/lat,lon/lat,...`` (bounding box,
    then the outline). Returns None for anything else.
    """
    head, _, outline = (value or "").partition(";")
    if outline:
        try:
            ring = [tuple(float(c) for c in pt.split("/")) for pt in outline.split(",") if pt]
        except ValueError:
            return None
        if any(len(pt) != 2 for pt in ring):
            return None
        if len(ring) > 1 and ring[0] == ring[-1]:
            ring.pop()
        return ring if len(ring) >= 3 else None
    parts = head.split("/")
    if len(parts) != 4:
        return None
    try:
        west, south, east, north = (float(part) for part in parts)
    except ValueError:
        return None
    west, east = min(west, east), max(west, east)
    south, north = min(south, north), max(south, north)
    if east <= west or north <= south:
        return None
    return [(west, south), (east, south), (east, north), (west, north)]


def read_dsf_binary_properties(dsf_path):
    """Read the header key/value properties straight out of a binary DSF.

    Exclusion zones live in the HEAD/PROP atom at the very start of the file,
    so this reads a few kilobytes instead of asking DSFTool to disassemble a
    whole mesh tile. Returns None when the file is not a plain uncompressed
    DSF (7z-packed global scenery, for instance) so the caller can fall back
    to a text disassembly.
    """
    import struct

    with open(dsf_path, "rb") as handle:
        if handle.read(8) != b"XPLNEDSF":
            return None
        handle.read(4)  # format version
        while True:
            atom_header = handle.read(8)
            if len(atom_header) < 8:
                return None
            atom_id = atom_header[:4]
            atom_len = struct.unpack("<I", atom_header[4:])[0]
            if atom_len < 8:
                return None
            if atom_id != b"DAEH":  # 'HEAD', stored little-endian
                handle.seek(atom_len - 8, os.SEEK_CUR)
                continue
            head = handle.read(atom_len - 8)
            break

    offset = 0
    while offset + 8 <= len(head):
        sub_id = head[offset:offset + 4]
        sub_len = struct.unpack("<I", head[offset + 4:offset + 8])[0]
        if sub_len < 8:
            return None
        payload = head[offset + 8:offset + sub_len]
        offset += sub_len
        if sub_id != b"PORP":  # 'PROP', stored little-endian
            continue
        strings = payload.split(b"\0")
        if strings and strings[-1] == b"":
            strings.pop()
        decoded = [item.decode("utf-8", "replace") for item in strings]
        return list(zip(decoded[0::2], decoded[1::2]))
    return []


def read_dsf_text_properties(text_path):
    """Read header key/value properties from a DSFTool text disassembly."""
    pairs = []
    with open(text_path, "r", encoding="utf-8", errors="ignore") as text_file:
        for raw_line in text_file:
            line = raw_line.strip()
            if not line.startswith("PROPERTY "):
                continue
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            pairs.append((parts[1], parts[2].strip()))
    return pairs


# On-disk (little-endian) atom IDs of the DSF definition tables.
_DSF_DEFINITION_ATOMS = {b"TJBO": "object", b"YLOP": "polygon"}


def _read_dsf_definitions_from(stream, can_seek):
    """Object and polygon definition paths from a binary DSF stream; None if not a DSF."""
    import struct

    if stream.read(12)[:8] != b"XPLNEDSF":
        return None
    while True:
        atom_header = stream.read(8)
        if len(atom_header) < 8:
            return None
        atom_id = atom_header[:4]
        atom_len = struct.unpack("<I", atom_header[4:])[0]
        if atom_len < 8:
            return None
        if atom_id == b"NFED":  # 'DEFN', stored little-endian
            payload = stream.read(atom_len - 8)
            if len(payload) < atom_len - 8:
                return None
            break
        if can_seek:
            stream.seek(atom_len - 8, os.SEEK_CUR)
        else:
            remaining = atom_len - 8
            while remaining:
                chunk = stream.read(min(remaining, 1 << 20))
                if not chunk:
                    return None
                remaining -= len(chunk)

    tables = {"object": [], "polygon": []}
    offset = 0
    while offset + 8 <= len(payload):
        child_id = payload[offset:offset + 4]
        child_len = struct.unpack_from("<I", payload, offset + 4)[0]
        if child_len < 8 or offset + child_len > len(payload):
            return None
        kind = _DSF_DEFINITION_ATOMS.get(child_id)
        if kind:
            strings = payload[offset + 8:offset + child_len].rstrip(b"\0")
            tables[kind] = [s.decode("utf-8", "replace") for s in strings.split(b"\0")] if strings else []
        offset += child_len
    return tables


def read_dsf_definitions(dsf_path, dsftool_path=None):
    """Return ``{'object': [...], 'polygon': [...]}`` from a DSF without disassembling it.

    The definition tables sit near the front of the file. A 7z-packed DSF is
    streamed through the 7-Zip beside DSFTool, stopping once they are read.
    Returns None when the file can't be read this way, so the caller falls
    back to a full disassembly.
    """
    try:
        with open(dsf_path, "rb") as handle:
            magic = handle.read(8)
            if magic == b"XPLNEDSF":
                handle.seek(0)
                return _read_dsf_definitions_from(handle, can_seek=True)
        if not magic.startswith(b"7z\xbc\xaf") or not dsftool_path:
            return None
        sevenzip = os.path.join(os.path.dirname(dsftool_path), "7z.exe" if os.name == "nt" else "7z")
        if not os.path.isfile(sevenzip):
            return None
        proc = subprocess.Popen(
            [sevenzip, "e", "-so", dsf_path],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            return _read_dsf_definitions_from(proc.stdout, can_seek=False)
        finally:
            proc.kill()
            proc.wait()
            proc.stdout.close()
    except (OSError, ValueError):
        return None


def dsf_may_hold_buildings(dsf_path, dsftool_path=None):
    """False only when a DSF provably defines no objects and no facades.

    A base mesh, a forest pack or a vegetation overlay has neither, so its
    disassembly - hundreds of MB of text for an ortho mesh - can be skipped
    by anything that only reads buildings out of it.
    """
    tables = read_dsf_definitions(dsf_path, dsftool_path)
    if tables is None:
        return True
    return bool(tables["object"]) or any(
        path.replace("\\", "/").lower().endswith(".fac") for path in tables["polygon"]
    )
