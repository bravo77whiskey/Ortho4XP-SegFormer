"""Decide which tile textures the overlays may read, and which part of each.

A tile's ``textures/`` directory is not a clean picture of what the sim renders.
Rebuilding a tile with a different provider leaves the previous run's DDS files
behind, so the same footprint and zoomlevel can exist several times over; and a
``zone_list`` boundary that cuts through a footprint legitimately produces two
textures for it, each owning only the part of the footprint on its side of the
boundary.

The tile config is the source of truth for both cases. It is the input that
``O4_DSF_Utils.zone_list_to_ortho_dico`` turned into the per-mesh-cell texture
assignment when the tile was built, so replaying that same rasterisation here
tells us exactly which DDS files the DSF references and which sub-region of its
footprint each one owns. Everything else in ``textures/`` is a leftover and is
skipped: inferencing it wastes time and places buildings and forests from
imagery that is not in the sim.

Without a config (loose texture directories, test fixtures) the legacy
filename-only resolution is used instead: keep the highest zoomlevel available
for each area and mask out the parts of a lower-ZL texture that a kept
higher-ZL texture already covers.
"""

import ast
import math
import os
import re

import numpy as np
from PIL import Image, ImageDraw


TEXTURE_NAME_RE = re.compile(
    r"^(\d+)_(\d+)_([A-Za-z][A-Za-z0-9_]*)(\d{2})\.dds$", re.IGNORECASE
)

# Ortho4XP rasterises zone polygons onto a 4096x4096 mask of the 1x1 degree
# tile; matching that resolution keeps our cell assignment identical to the
# one the DSF was built from.
_MASK_SIZE = 4096
_MASK_MAX = _MASK_SIZE - 1


def compute_covered_fractions(entries):
    """Map texture filename -> regions covered by kept higher-ZL textures.

    ``entries`` yields ``(til_y_top, til_x_left, zl, fname)`` for every KEPT
    texture. Returns ``{fname: ((fx0, fy0, fx1, fy1), ...)}`` where the rects
    are fractions of the texture footprint (x right, y down, matching image
    pixel orientation). Textures with no covered region are absent.
    """
    kept = [tuple(entry) for entry in entries]
    by_zl = {}
    for til_y, til_x, zl, fname in kept:
        by_zl.setdefault(int(zl), []).append((int(til_y), int(til_x), fname))
    covered = {}
    for zl, tiles in by_zl.items():
        higher = [
            (czl, ctiles) for czl, ctiles in by_zl.items() if czl > zl
        ]
        if not higher:
            continue
        for til_y, til_x, fname in tiles:
            rects = []
            for czl, ctiles in higher:
                scale = float(2 ** (czl - zl))
                for c_y, c_x, _cf in ctiles:
                    fy = (c_y / scale - til_y) / 16.0
                    fx = (c_x / scale - til_x) / 16.0
                    fs = 1.0 / scale
                    if fx >= 1.0 or fy >= 1.0 or fx + fs <= 0.0 or fy + fs <= 0.0:
                        continue
                    rects.append((
                        max(0.0, fx), max(0.0, fy),
                        min(1.0, fx + fs), min(1.0, fy + fs),
                    ))
            if rects:
                covered[fname] = tuple(sorted(rects))
    return covered


# ── Tile config ──────────────────────────────────────────────────────────────

def tile_cfg_path(tex_dir, lat, lon):
    """Return the Ortho4XP config next to a tile's textures directory."""
    short = f"{int(lat):+03d}{int(lon):+04d}"
    tex_dir = os.path.abspath(tex_dir)
    candidates = []
    if os.path.basename(tex_dir).lower() == "textures":
        candidates.append(os.path.dirname(tex_dir))
    candidates.append(tex_dir)
    for base in candidates:
        path = os.path.join(base, f"Ortho4XP_{short}.cfg")
        if os.path.isfile(path):
            return path
    return None


def parse_tile_cfg(cfg_path):
    """Parse an Ortho4XP tile config into a plain ``key -> str`` dict."""
    values = {}
    with open(cfg_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _cfg_int(values, key, default):
    try:
        return int(values[key])
    except (KeyError, TypeError, ValueError):
        return default


def _cfg_float(values, key, default):
    try:
        return float(values[key])
    except (KeyError, TypeError, ValueError):
        return default


def _cfg_zone_list(values):
    try:
        zone_list = ast.literal_eval(values.get("zone_list", "[]"))
    except (SyntaxError, ValueError):
        return []
    if not isinstance(zone_list, list):
        return []
    return [
        zone for zone in zone_list
        if isinstance(zone, (list, tuple)) and len(zone) >= 3 and zone[0]
    ]


# ── Web-mercator grid helpers (kept local so the overlays stay pyproj-free) ──

def _gtile_to_wgs84(til_x, til_y, zoomlevel):
    rat_x = til_x / (2 ** (zoomlevel - 1)) - 1
    rat_y = 1 - til_y / (2 ** (zoomlevel - 1))
    lon = rat_x * 180
    lat = 360 / math.pi * math.atan(math.exp(math.pi * rat_y)) - 90
    return lat, lon


def _wgs84_to_orthogrid(lat, lon, zoomlevel):
    ratio_x = lon / 180
    ratio_y = math.log(math.tan((90 + lat) * math.pi / 360)) / math.pi
    mult = 2 ** (zoomlevel - 5)
    til_x = int((ratio_x + 1) * mult) * 16
    til_y = int((1 - ratio_y) * mult) * 16
    return til_x, til_y


# ── Config replay ────────────────────────────────────────────────────────────

def _zone_mask(lat, lon, default_zl, default_website, zone_list):
    """Rasterise the base zone plus zone_list, later zones winning."""
    masks_im = Image.new("L", (_MASK_SIZE, _MASK_SIZE), "black")
    masks_draw = ImageDraw.Draw(masks_im)
    dico_tmp = {}
    index = 1
    base_zone = (
        [
            lat, lon,
            lat, lon + 1,
            lat + 1, lon + 1,
            lat + 1, lon,
            lat, lon,
        ],
        default_zl,
        default_website,
    )
    # zone_list is reversed so its first entry is drawn last and wins, matching
    # O4_DSF_Utils.zone_list_to_ortho_dico.
    for region in [base_zone] + list(zone_list)[::-1]:
        dico_tmp[index] = (int(region[1]), str(region[2]))
        polygon = [
            (
                round((x - lon) * _MASK_MAX),
                round((lat + 1 - y) * _MASK_MAX),
            )
            for (x, y) in zip(region[0][1::2], region[0][::2])
        ]
        masks_draw.polygon(polygon, fill=index)
        index += 1
    return masks_im, dico_tmp


def _airport_array(tex_dir, lat, lon, cover_zl, cover_extent, cover_mode):
    """Rebuild the high-ZL airport upgrade mask, or None when unavailable."""
    if cover_mode not in ("True", "ICAO"):
        return None, None
    short = f"{int(lat):+03d}{int(lon):+04d}"
    tex_dir = os.path.abspath(tex_dir)
    bases = []
    if os.path.basename(tex_dir).lower() == "textures":
        bases.append(os.path.dirname(tex_dir))
    bases.append(tex_dir)
    apt_path = None
    for base in bases:
        candidate = os.path.join(base, f"Data{short}.apt")
        if os.path.isfile(candidate):
            apt_path = candidate
            break
    if apt_path is None:
        return None, (
            f"cover_airports_with_highres={cover_mode} but Data{short}.apt is "
            "missing; airport zoomlevel upgrades cannot be replayed"
        )
    try:
        import pickle

        with open(apt_path, "rb") as handle:
            dico_airports = pickle.load(handle)
    except Exception as exc:  # noqa: BLE001 - any unreadable apt is non-fatal
        return None, f"could not read {os.path.basename(apt_path)} ({exc})"

    # Same constants as O4_Geo_Utils, evaluated at the tile latitude exactly as
    # O4_DSF_Utils.zone_list_to_ortho_dico does.
    m_to_lat = 180 / (math.pi * 6378137)
    m_to_lon = m_to_lat / math.cos(math.pi * lat / 180)
    airports = [
        name for name, info in dico_airports.items()
        if cover_mode != "ICAO" or info.get("key_type") == "icao"
    ]
    array = np.zeros((_MASK_SIZE, _MASK_SIZE), dtype=np.bool_)
    for airport in airports:
        try:
            xmin, ymin, xmax, ymax = dico_airports[airport]["boundary"].bounds
        except (KeyError, AttributeError, TypeError, ValueError):
            continue
        xmin -= 1000 * cover_extent * m_to_lon
        xmax += 1000 * cover_extent * m_to_lon
        ymax += 1000 * cover_extent * m_to_lat
        ymin -= 1000 * cover_extent * m_to_lat
        til_x_left, til_y_top = _wgs84_to_orthogrid(ymax + lat, xmin + lon, cover_zl)
        ymax, xmin = _gtile_to_wgs84(til_x_left, til_y_top, cover_zl)
        ymax -= lat
        xmin -= lon
        til_x_left2, til_y_top2 = _wgs84_to_orthogrid(ymin + lat, xmax + lon, cover_zl)
        ymin, xmax = _gtile_to_wgs84(til_x_left2 + 16, til_y_top2 + 16, cover_zl)
        ymin -= lat
        xmax -= lon
        xmin = max(0, xmin)
        xmax = min(1, xmax)
        ymin = max(0, ymin)
        ymax = min(1, ymax)
        colmin = round(xmin * _MASK_MAX)
        colmax = round(xmax * _MASK_MAX)
        rowmax = round((1 - ymin) * _MASK_MAX)
        rowmin = round((1 - ymax) * _MASK_MAX)
        array[rowmin:rowmax + 1, colmin:colmax + 1] = 1
    return array, None


def cfg_texture_assignment(tex_dir, lat, lon, cfg_path=None):
    """Replay the tile config's per-mesh-cell texture assignment.

    Returns ``(assignment, mesh_zl, notes)`` where ``assignment`` maps
    ``(til_y_text, til_x_text, provider, zl)`` to the set of mesh-grid cells
    ``(til_x, til_y)`` that texture owns, or ``(None, None, notes)`` when the
    tile has no readable config.
    """
    notes = []
    if cfg_path is None:
        cfg_path = tile_cfg_path(tex_dir, lat, lon)
    if cfg_path is None:
        return None, None, notes
    try:
        values = parse_tile_cfg(cfg_path)
    except OSError as exc:
        notes.append(f"could not read {os.path.basename(cfg_path)} ({exc})")
        return None, None, notes

    default_website = values.get("default_website", "")
    if not default_website:
        notes.append(
            f"{os.path.basename(cfg_path)} has no default_website; "
            "falling back to filename-based texture resolution"
        )
        return None, None, notes
    lat = int(lat)
    lon = int(lon)
    default_zl = _cfg_int(values, "default_zl", 16)
    mesh_zl = _cfg_int(values, "mesh_zl", 19)
    cover_zl = _cfg_int(values, "cover_zl", 18)
    cover_extent = _cfg_float(values, "cover_extent", 1.0)
    cover_mode = values.get("cover_airports_with_highres", "False")
    zone_list = _cfg_zone_list(values)

    masks_im, dico_tmp = _zone_mask(
        lat, lon, default_zl, default_website, zone_list
    )
    airport_array, airport_note = _airport_array(
        tex_dir, lat, lon, cover_zl, cover_extent, cover_mode
    )
    if airport_note:
        notes.append(airport_note)

    til_x_min, til_y_min = _wgs84_to_orthogrid(lat + 1, lon, mesh_zl)
    til_x_max, til_y_max = _wgs84_to_orthogrid(lat, lon + 1, mesh_zl)
    assignment = {}
    for til_x in range(til_x_min, til_x_max + 1, 16):
        for til_y in range(til_y_min, til_y_max + 1, 16):
            latp, lonp = _gtile_to_wgs84(til_x + 8, til_y + 8, mesh_zl)
            lonp = max(min(lonp, lon + 1), lon)
            latp = max(min(latp, lat + 1), lat)
            col = round((lonp - lon) * _MASK_MAX)
            row = round((lat + 1 - latp) * _MASK_MAX)
            zoomlevel, provider = dico_tmp[masks_im.getpixel((col, row))]
            if airport_array is not None and airport_array[row, col]:
                zoomlevel = max(zoomlevel, cover_zl)
            scale = 2 ** (mesh_zl - zoomlevel)
            til_x_text = 16 * (int(til_x / scale) // 16)
            til_y_text = 16 * (int(til_y / scale) // 16)
            assignment.setdefault(
                (til_y_text, til_x_text, provider, zoomlevel), set()
            ).add((til_x, til_y))
    return assignment, mesh_zl, notes


def texture_name(til_y_text, til_x_text, provider, zl):
    return f"{til_y_text}_{til_x_text}_{provider}{zl:02d}.dds"


def _grid_position(cell_x, cell_y, til_y_text, til_x_text, scale):
    """Locate a mesh cell within a texture footprint's own scale x scale grid.

    The footprint spans 16 tile units at its own zoomlevel, so one mesh cell is
    1/scale of it on each axis. Returns ``(row, col)``, or None when the cell
    falls outside the footprint.
    """
    if scale <= 1:
        # A texture at or above the mesh zoomlevel is never subdivided: any
        # cell reaching it owns the whole footprint.
        return 0, 0
    col = int(round((cell_x / scale - til_x_text) / 16.0 * scale))
    row = int(round((cell_y / scale - til_y_text) / 16.0 * scale))
    if 0 <= col < scale and 0 <= row < scale:
        return row, col
    return None


def _merge_grid_rects(grid, scale):
    """Turn footprint grid positions into merged footprint-fraction rects."""
    scale = max(1, int(scale))
    step = 1.0 / scale
    # Merge horizontal runs first, then stack vertically identical runs.
    runs = {}
    for row in sorted({r for r, _ in grid}):
        cols = sorted(c for r, c in grid if r == row)
        start = prev = None
        row_runs = []
        for col in cols:
            if start is None:
                start = prev = col
            elif col == prev + 1:
                prev = col
            else:
                row_runs.append((start, prev))
                start = prev = col
        if start is not None:
            row_runs.append((start, prev))
        runs[row] = tuple(row_runs)
    rects = []
    open_runs = {}  # (col0, col1) -> row0
    for row in range(scale + 1):
        current = dict.fromkeys(runs.get(row, ()), row)
        for run, row0 in list(open_runs.items()):
            if run not in current:
                rects.append((run[0], row0, run[1] + 1, row))
                del open_runs[run]
        for run in current:
            open_runs.setdefault(run, row)
    return tuple(
        (col0 * step, row0 * step, col1 * step, row1 * step)
        for col0, row0, col1, row1 in sorted(rects)
    )


def cfg_texture_selection(tex_dir, lat, lon, available, cfg_path=None):
    """Select textures and per-texture excluded regions from the tile config.

    ``available`` is the set of texture filenames actually present. Returns
    ``(selection, notes)`` where ``selection`` is
    ``(files, excluded_fracs_by_file, report)``, or None when the tile has no
    usable config -- ``notes`` explains why and is worth reporting either way.
    """
    assignment, mesh_zl, notes = cfg_texture_assignment(
        tex_dir, lat, lon, cfg_path=cfg_path
    )
    if assignment is None:
        return None, notes

    available = set(available)
    lower_lookup = {name.lower(): name for name in available}
    keys_by_footprint = {}
    resolved = {}
    missing = []
    for key in assignment:
        til_y_text, til_x_text, provider, zl = key
        name = texture_name(til_y_text, til_x_text, provider, zl)
        actual = name if name in available else lower_lookup.get(name.lower())
        if actual is None:
            missing.append(name)
            continue
        resolved[key] = actual
        keys_by_footprint.setdefault((til_y_text, til_x_text, zl), []).append(key)

    if not resolved:
        notes.append(
            "tile config names no texture that exists on disk; "
            "falling back to filename-based texture resolution"
        )
        return None, notes

    zls = sorted({zl for _, _, _, zl in resolved})
    # A cell is foreign to a texture when it lies inside that texture's
    # footprint but the config assigned it to a different texture -- either a
    # different provider at the same zoomlevel (a zone boundary splitting the
    # footprint) or a higher zoomlevel covering part of it.
    foreign = {}
    claimed = {}
    for key, cells in assignment.items():
        for cell_x, cell_y in cells:
            for zl in zls:
                scale = 2 ** (mesh_zl - zl)
                til_x_text = 16 * (int(cell_x / scale) // 16)
                til_y_text = 16 * (int(cell_y / scale) // 16)
                footprint = (til_y_text, til_x_text, zl)
                position = _grid_position(
                    cell_x, cell_y, til_y_text, til_x_text, scale
                )
                if position is None:
                    continue
                claimed.setdefault(footprint, set()).add(position)
                for other in keys_by_footprint.get(footprint, ()):
                    if other != key:
                        foreign.setdefault(other, set()).add(position)

    # A footprint on the tile border reaches into the neighbouring tile, where
    # the config assigns nothing. When two providers share such a footprint,
    # that strip would otherwise be detected once per provider, so it goes to
    # the texture that owns the most of the footprint inside the tile.
    for footprint, keys in keys_by_footprint.items():
        if len(keys) < 2:
            continue
        scale = max(1, int(2 ** (mesh_zl - footprint[2])))
        gaps = {
            (row, col) for row in range(scale) for col in range(scale)
        } - claimed.get(footprint, set())
        if not gaps:
            continue
        ranked = sorted(keys, key=lambda k: (-len(assignment[k]), resolved[k]))
        for key in ranked[1:]:
            foreign.setdefault(key, set()).update(gaps)

    excluded = {}
    for key, grid in foreign.items():
        zl = key[3]
        rects = _merge_grid_rects(grid, 2 ** (mesh_zl - zl))
        if rects:
            excluded[resolved[key]] = rects

    files = sorted(resolved.values())
    report = {
        "cfg_path": cfg_path or tile_cfg_path(tex_dir, lat, lon),
        "selected": len(files),
        "skipped": len(available) - len(set(files)),
        # Config grid cells with no mesh triangles never get a texture written,
        # so a config-named file being absent is routine, not an error.
        "missing": sorted(missing),
        "partial": len(excluded),
        "notes": notes,
    }
    return (files, excluded, report), notes


# ── Legacy (config-less) resolution ──────────────────────────────────────────

def _legacy_selection(source_files):
    """Filename-only resolution: highest zoomlevel wins, ties keep everything."""
    by_zl = {}
    for name in source_files:
        match = TEXTURE_NAME_RE.match(name)
        if not match:
            continue
        by_zl.setdefault(int(match.group(4)), []).append(
            (int(match.group(1)), int(match.group(2)), name)
        )
    if not by_zl:
        return [], {}, {
            "cfg_path": None, "selected": 0, "skipped": 0,
            "missing": [], "partial": 0, "notes": [],
        }
    tiles_at_zl = {
        zl: {(y, x) for y, x, _ in tiles} for zl, tiles in by_zl.items()
    }
    all_zls = sorted(by_zl)
    max_zl = all_zls[-1]
    memo = {}

    def fully_covered(y, x, zl):
        """True iff all 4 children of (y,x,zl) exist or are themselves covered."""
        key = (y, x, zl)
        if key in memo:
            return memo[key]
        if zl >= max_zl:
            memo[key] = False
            return False
        result = all(
            (cy, cx) in tiles_at_zl.get(zl + 1, set()) or fully_covered(cy, cx, zl + 1)
            for cy, cx in (
                (2 * y, 2 * x), (2 * y, 2 * x + 16),
                (2 * y + 16, 2 * x), (2 * y + 16, 2 * x + 16),
            )
        )
        memo[key] = result
        return result

    files = sorted(
        name for zl in all_zls
        for y, x, name in by_zl[zl]
        if not fully_covered(y, x, zl)
    )
    excluded = compute_covered_fractions(
        (
            (int(match.group(1)), int(match.group(2)), int(match.group(4)), name)
            for name in files
            for match in (TEXTURE_NAME_RE.match(name),)
            if match
        ),
    )
    report = {
        "cfg_path": None,
        "selected": len(files),
        "skipped": sum(len(v) for v in by_zl.values()) - len(files),
        "missing": [],
        "partial": len(excluded),
        "notes": [],
    }
    return files, excluded, report


def select_textures(source_files, tex_dir, lat, lon, cfg_path=None):
    """Pick the textures to inference and the region of each one to skip.

    Returns ``(files, excluded_fracs_by_file, report)``. ``files`` are the
    texture names to process; ``excluded_fracs_by_file`` maps a name to the
    ``(fx0, fy0, fx1, fy1)`` footprint fractions that another texture owns and
    that must therefore not be detected or placed from this one.
    """
    source_files = list(source_files)
    available = {
        name for name in source_files if TEXTURE_NAME_RE.match(name)
    }
    selection, notes = (None, [])
    if available:
        selection, notes = cfg_texture_selection(
            tex_dir, lat, lon, available, cfg_path=cfg_path
        )
    if selection is None:
        files, excluded, report = _legacy_selection(source_files)
        report["notes"] = list(notes)
        return files, excluded, report
    return selection


def format_selection_report(report, region_label="detection/placement"):
    """Render the selection outcome as printable lines."""
    lines = []
    for note in report.get("notes", ()):
        lines.append(f"Texture selection: WARNING {note}")
    if report.get("cfg_path"):
        skipped = report.get("skipped", 0)
        lines.append(
            f"Texture selection: {report.get('selected', 0)} texture(s) named by "
            f"{os.path.basename(report['cfg_path'])}"
            + (f"; skipped {skipped} unreferenced texture(s)" if skipped else "")
        )
    elif report.get("skipped"):
        lines.append(
            f"Zoom-level coverage: skipped {report['skipped']} lower-ZL "
            "texture(s) fully covered by higher-ZL textures"
        )
    if report.get("partial"):
        lines.append(
            f"Texture selection: {report['partial']} partially owned texture(s); "
            f"regions owned by another texture excluded from {region_label}"
        )
    return lines
