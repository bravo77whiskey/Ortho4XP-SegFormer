"""Imagery zones drawn automatically around the airports a tile lists.

A tile's ``airport_zone_icaos`` names airports by ICAO code. For each one the
runways are read from apt.dat (the highest-priority scenery pack that defines
the airport, as X-Plane draws it), wrapped in their convex hull, and that hull
is grown by ``airport_zone_zl19_m``, ``airport_zone_zl18_m`` and
``airport_zone_zl17_m`` metres into three nested zones at ZL19, ZL18 and ZL17.
A distance of 0 leaves that zone out.

An airport only counts for the tile its runways sit in; listed elsewhere it is
skipped. Its zones reach into neighbouring tiles of the same batch, so the
zoomlevel does not drop at a tile edge that cuts through them.

Step 3 saves a tile's zones in its config as ``airport_zone_list``, apart from
the hand-drawn ``zone_list``, and rewrites them on every build. The overlays
replay the zone assignment from the config to decide which textures to read,
and the batch that decided which neighbours contributed zones is gone by then,
so that line is the record of what the DSF was actually built with.
"""

import ast
import math
import os
import pickle

import O4_Apt_Pavement as APT

ZONE_ZLS = (19, 18, 17)
DISTANCE_VARS = {zl: f"airport_zone_zl{zl}_m" for zl in ZONE_ZLS}
ICAO_VAR = "airport_zone_icaos"
ZONES_VAR = "airport_zone_list"

_CACHE_VERSION = 1
_BUFFER_RESOLUTION = 16  # segments per quarter circle on the rounded corners


# ── Settings ─────────────────────────────────────────────────────────────────

def parse_icaos(text):
    """'rckh, RCAY RCKH' -> ['RCKH', 'RCAY']: upper case, first mention kept."""
    seen = []
    for code in str(text or "").replace(",", " ").split():
        code = code.strip().upper()
        if code and code not in seen:
            seen.append(code)
    return seen


def distances_from(source):
    """{zl: metres} from an object or dict carrying the airport_zone_zl*_m vars."""
    out = {}
    for zl, var in DISTANCE_VARS.items():
        value = source.get(var, 0) if isinstance(source, dict) else getattr(source, var, 0)
        try:
            out[zl] = max(0.0, float(value))
        except (TypeError, ValueError):
            out[zl] = 0.0
    return out


def _tile_cfg_value(cfg_path, var):
    try:
        with open(cfg_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith(var + "="):
                    return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return None


def read_tile_icaos(cfg_path):
    """The airport_zone_icaos of a tile config, or [] without one."""
    return parse_icaos(_tile_cfg_value(cfg_path, ICAO_VAR))


def read_tile_zones(cfg_path):
    """The airport_zone_list a tile config was last built with, or []."""
    try:
        zones = ast.literal_eval(_tile_cfg_value(cfg_path, ZONES_VAR) or "[]")
    except (SyntaxError, ValueError):
        return []
    if not isinstance(zones, list):
        return []
    return [
        zone for zone in zones
        if isinstance(zone, (list, tuple)) and len(zone) >= 3 and zone[0]
    ]


# ── Runways from apt.dat ─────────────────────────────────────────────────────

def _scan_runways(xplane_dir, wanted, log):
    """{icao: [(width, lat1, lon1, lat2, lon2), ...]} for the wanted codes.

    Scenery packs are read in scenery_packs.ini order and the first definition
    of an airport wins, as in the sim. An airport defined with no land runway
    maps to [].
    """
    try:
        sources = APT._apt_sources(xplane_dir)
    except OSError:
        sources = [("Global Airports", os.path.join(
            xplane_dir, "Global Scenery", "Global Airports", "Earth nav data", "apt.dat"))]
    found = {}
    remaining = set(wanted)
    for _name, apt in sources:
        if not remaining:
            break
        if not os.path.isfile(apt):
            continue
        current = None
        try:
            with open(apt, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    head = line[:4]
                    if head.startswith(("1 ", "16 ", "17 ", "99")):
                        parts = line.split(None, 5)
                        code = parts[4] if len(parts) > 4 and parts[0] != "99" else None
                        if code in remaining:
                            current = code
                            found[code] = []
                            remaining.discard(code)
                        else:
                            current = None
                            if not remaining:
                                break
                        continue
                    if current is not None and head == "100 ":
                        parts = line.split()
                        try:
                            found[current].append((
                                float(parts[1]),
                                float(parts[9]), float(parts[10]),
                                float(parts[18]), float(parts[19]),
                            ))
                        except (IndexError, ValueError):
                            pass
        except OSError as exc:  # unreadable disk sectors: skip, don't abort
            log(f"apt.dat unreadable, skipped: {apt}: {exc}")
    return found


def runways_by_icao(xplane_dir, icaos, cache_dir=None, log=print):
    """Runways of each listed airport; airports apt.dat lacks are left out.

    Reading the global apt.dat takes several seconds, so results (misses
    included) are cached against scenery_packs.ini and the global apt.dat.
    """
    icaos = list(icaos)
    if not icaos or not xplane_dir:
        return {}
    gapt = os.path.join(xplane_dir, "Global Scenery", "Global Airports", "Earth nav data", "apt.dat")
    ini = os.path.join(xplane_dir, "Custom Scenery", "scenery_packs.ini")
    stamp = (_CACHE_VERSION,) + tuple(
        (os.path.getsize(p), int(os.path.getmtime(p))) if os.path.exists(p) else None
        for p in (ini, gapt)
    )
    cpath = os.path.join(cache_dir, "airport_runways.pkl") if cache_dir else None
    known = {}
    if cpath:
        try:
            with open(cpath, "rb") as handle:
                cached = pickle.load(handle)
            if cached.get("stamp") == stamp:
                known = dict(cached["runways"])
        except (OSError, pickle.UnpicklingError, EOFError, KeyError, TypeError):
            pass
    missing = [code for code in icaos if code not in known]
    if missing:
        scanned = _scan_runways(xplane_dir, missing, log)
        for code in missing:
            known[code] = scanned.get(code)
        if cpath:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                tmp = cpath + ".tmp"
                with open(tmp, "wb") as handle:
                    pickle.dump({"stamp": stamp, "runways": known}, handle)
                os.replace(tmp, cpath)
            except OSError:
                pass
    return {code: known[code] for code in icaos if known.get(code) is not None}


# ── Geometry ─────────────────────────────────────────────────────────────────

def _frame(runways):
    """Local metric frame centred on the runway ends: (lat0, lon0, kx, ky)."""
    lat0 = sum(r[1] + r[3] for r in runways) / (2 * len(runways))
    lon0 = sum(r[2] + r[4] for r in runways) / (2 * len(runways))
    return lat0, lon0, APT._m_per_deg_lon(lat0), APT.EARTH_M_PER_DEG_LAT


def runway_hull(runways):
    """Convex hull of the runway rectangles in the local frame, plus the frame."""
    from shapely.geometry import MultiPoint

    lat0, lon0, kx, ky = _frame(runways)
    corners = []
    for width, lat1, lon1, lat2, lon2 in runways:
        x1, y1 = (lon1 - lon0) * kx, (lat1 - lat0) * ky
        x2, y2 = (lon2 - lon0) * kx, (lat2 - lat0) * ky
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 1.0:
            corners.append((x1, y1))
            continue
        px, py = -(y2 - y1) / length * width / 2, (x2 - x1) / length * width / 2
        corners += [(x1 + px, y1 + py), (x1 - px, y1 - py), (x2 + px, y2 + py), (x2 - px, y2 - py)]
    return MultiPoint(corners).convex_hull, (lat0, lon0, kx, ky)


def runway_centre(runways):
    """(lat, lon) of the runway hull's centroid."""
    hull, (lat0, lon0, kx, ky) = runway_hull(runways)
    c = hull.centroid
    return lat0 + c.y / ky, lon0 + c.x / kx


def airport_rings(runways, distances):
    """[(zone_coords, zl)] from the highest zoomlevel down.

    zone_coords is the closed ring flattened as [lat0, lon0, lat1, lon1, ...],
    the form zone_list uses.
    """
    hull, (lat0, lon0, kx, ky) = runway_hull(runways)
    rings = []
    for zl in sorted(distances, reverse=True):
        dist = distances[zl]
        if dist <= 0:
            continue
        ring = hull.buffer(dist, quad_segs=_BUFFER_RESOLUTION).exterior.coords
        coords = []
        for x, y in ring:
            coords += [round(lat0 + y / ky, 7), round(lon0 + x / kx, 7)]
        rings.append((coords, zl))
    return rings


def _ring_polygon(coords):
    from shapely.geometry import Polygon

    return Polygon(list(zip(coords[1::2], coords[::2])))


def _tile_box(lat, lon):
    from shapely.geometry import box

    return box(lon, lat, lon + 1, lat + 1)


# ── Batch plan ───────────────────────────────────────────────────────────────

def plan_zones(tiles, settings_for, runway_lookup, log=print):
    """Decide the airport zones of every tile in a batch.

    tiles: the batch's (lat, lon) tiles.
    settings_for(lat, lon) -> (icaos, {zl: metres}) for one of them.
    runway_lookup(icaos) -> {icao: runways}.

    Returns {(lat, lon): [(zone_coords, zl, icao)]}, zones in zoomlevel order
    (highest first), for the tiles that get any.
    """
    tiles = [(int(lat), int(lon)) for lat, lon in tiles]
    settings = {tile: settings_for(*tile) for tile in tiles}
    wanted = []
    for icaos, _dist in settings.values():
        wanted += [code for code in icaos if code not in wanted]
    runways = runway_lookup(wanted) if wanted else {}
    plan = {}
    boxes = {}
    for home in tiles:
        icaos, distances = settings[home]
        for icao in icaos:
            if not runways.get(icao):
                log(f"   Airport zones: {icao} has no runway in apt.dat, skipped.")
                continue
            lat_c, lon_c = runway_centre(runways[icao])
            if not (home[0] <= lat_c < home[0] + 1 and home[1] <= lon_c < home[1] + 1):
                log(f"   Airport zones: {icao} is not in tile {home[0]:+03d}{home[1]:+04d}, skipped.")
                continue
            rings = airport_rings(runways[icao], distances)
            if not rings:
                log(f"   Airport zones: every distance for {icao} is 0, no zone drawn.")
                continue
            for tile in tiles:
                if tile not in boxes:
                    boxes[tile] = _tile_box(*tile)
                reaching = [
                    (coords, zl, icao) for coords, zl in rings
                    if _ring_polygon(coords).intersects(boxes[tile])
                ]
                if reaching:
                    plan.setdefault(tile, []).extend(reaching)
    for zones in plan.values():
        zones.sort(key=lambda zone: -zone[1])
    return plan


# ── Combining with drawn zones ───────────────────────────────────────────────

def merge_zone_lists(zone_list, airport_zones):
    """Drawn zones with airport zones slotted in by zoomlevel.

    The first zone of a zone_list wins where zones overlap. Drawn zones keep
    their order among themselves; each airport zone goes ahead of the first
    drawn zone with a lower zoomlevel, so it is never undercut by a coarser
    drawn zone yet never overrides a finer one.
    """
    merged = [(zone, False) for zone in zone_list]
    for zone in sorted(airport_zones, key=lambda z: -int(z[1])):
        index = next(
            (i for i, (other, drawn) in enumerate(merged)
             if drawn is False and int(other[1]) < int(zone[1])),
            len(merged),
        )
        merged.insert(index, (zone, True))
    return [zone for zone, _ in merged]
