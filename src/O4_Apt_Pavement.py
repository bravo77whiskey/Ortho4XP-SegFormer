"""Airport layouts from apt.dat: pavement to keep overlays off, and boundaries.

X-Plane draws each airport from the highest-priority pack that defines it, so
apt.dat files are read in scenery_packs.ini order (enabled packs only), the
first definition of an airport wins, and *GLOBAL_AIRPORTS* is read at its own
place in that order.

Per airport:
  * pavement, as [(outer_ring, [hole_rings])] with rings of (lon, lat):
      - row 100 land runways: the full-width rectangle between the two ends,
        extended by each end's blast pad length;
      - row 102 helipads: their rectangle;
      - row 110 pavement (taxiways, aprons, shoulders): its boundary loops,
        Bezier edges sampled; the first loop is the outline, later loops holes;
  * boundaries: the row 130 airport boundary polygons, same form;
  * centre and reach: the mean pavement point and the distance from it to the
    furthest one -- a stand-in footprint for airports without a boundary.
"""

import math
import os
import pickle
from collections import namedtuple

EARTH_M_PER_DEG_LAT = 110574.0
BEZIER_STEPS = 8
_CACHE_VERSION = 1

Airport = namedtuple("Airport", "icao pack pavement boundaries lat lon reach_m")


def _m_per_deg_lon(lat):
    return 111320.0 * math.cos(math.radians(lat))


def _offset(lon, lat, east_m, north_m):
    return lon + east_m / _m_per_deg_lon(lat), lat + north_m / EARTH_M_PER_DEG_LAT


def _runway_rect(width, lat1, lon1, blast1, lat2, lon2, blast2):
    lat_c = 0.5 * (lat1 + lat2)
    kx, ky = _m_per_deg_lon(lat_c), EARTH_M_PER_DEG_LAT
    dx, dy = (lon2 - lon1) * kx, (lat2 - lat1) * ky
    length = math.hypot(dx, dy)
    if length < 1.0:
        return None
    ux, uy = dx / length, dy / length           # along the runway, metres
    px, py = -uy, ux                            # across
    a = (lon1 - ux * blast1 / kx, lat1 - uy * blast1 / ky)
    b = (lon2 + ux * blast2 / kx, lat2 + uy * blast2 / ky)
    hw = 0.5 * width
    return [(a[0] + px * hw / kx, a[1] + py * hw / ky), (b[0] + px * hw / kx, b[1] + py * hw / ky),
            (b[0] - px * hw / kx, b[1] - py * hw / ky), (a[0] - px * hw / kx, a[1] - py * hw / ky)]


def _helipad_rect(lat, lon, heading, length, width):
    h = math.radians(heading)
    ax, ay = math.sin(h), math.cos(h)            # along the heading (east, north)
    px, py = ay, -ax
    return [_offset(lon, lat, sa * ax * length / 2 + sp * px * width / 2,
                    sa * ay * length / 2 + sp * py * width / 2)
            for sa, sp in ((1, 1), (1, -1), (-1, -1), (-1, 1))]


def _bezier_ring(nodes):
    """nodes: (lon, lat, ctrl_lon or None, ctrl_lat or None), closed loop."""
    ring = []
    n = len(nodes)
    for i in range(n):
        x0, y0, cx0, cy0 = nodes[i]
        x3, y3, cx3, cy3 = nodes[(i + 1) % n]
        if cx0 is None and cx3 is None:
            ring.append((x0, y0))
            continue
        p1 = (cx0, cy0) if cx0 is not None else (x0, y0)
        # A node's control point steers the curve leaving it; the curve
        # arriving at it uses the mirror image.
        p2 = (2 * x3 - cx3, 2 * y3 - cy3) if cx3 is not None else (x3, y3)
        for s in range(BEZIER_STEPS):
            t = s / BEZIER_STEPS
            mt = 1 - t
            ring.append((mt ** 3 * x0 + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t ** 3 * x3,
                         mt ** 3 * y0 + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t ** 3 * y3))
    return ring


def parse_apt_dat(path):
    """Yield (icao, pavement, boundaries) for every airport in one apt.dat."""
    icao, pavement, boundaries = None, [], []
    loops, nodes, in_pavement, in_boundary = [], [], False, False

    def close_loops():
        nonlocal loops, nodes, in_pavement, in_boundary
        if in_pavement and loops:
            pavement.append((loops[0], loops[1:]))
        if in_boundary and loops:
            boundaries.append((loops[0], loops[1:]))
        loops, nodes, in_pavement, in_boundary = [], [], False, False

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            code = parts[0]
            if code in ("1", "16", "17", "99"):
                close_loops()
                if icao is not None:
                    yield icao, pavement, boundaries
                icao = parts[4] if len(parts) > 4 and code != "99" else None
                pavement, boundaries = [], []
                continue
            if icao is None:
                continue
            if code in ("111", "112", "113", "114", "115", "116"):
                if not (in_pavement or in_boundary):
                    continue
                lat, lon = float(parts[1]), float(parts[2])
                if code in ("112", "114", "116"):
                    nodes.append((lon, lat, float(parts[4]), float(parts[3])))
                else:
                    nodes.append((lon, lat, None, None))
                if code in ("113", "114", "115", "116"):
                    if len(nodes) >= 3:
                        loops.append(_bezier_ring(nodes))
                    nodes = []
                continue
            close_loops()
            try:
                if code == "110":
                    in_pavement = True
                elif code == "130":
                    in_boundary = True
                elif code == "100":
                    width = float(parts[1])
                    lat1, lon1, blast1 = float(parts[9]), float(parts[10]), float(parts[12])
                    lat2, lon2, blast2 = float(parts[18]), float(parts[19]), float(parts[21])
                    r = _runway_rect(width, lat1, lon1, blast1, lat2, lon2, blast2)
                    if r:
                        pavement.append((r, []))
                elif code == "102":
                    pavement.append((_helipad_rect(float(parts[2]), float(parts[3]), float(parts[4]),
                                                   float(parts[5]), float(parts[6])), []))
            except (IndexError, ValueError):
                pass
    close_loops()
    if icao is not None:
        yield icao, pavement, boundaries


def _apt_sources(xplane_dir):
    """(pack name, apt.dat path) in scenery_packs.ini priority order."""
    ini = os.path.join(xplane_dir, "Custom Scenery", "scenery_packs.ini")
    sources = []
    with open(ini, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("SCENERY_PACK "):
                continue                         # SCENERY_PACK_DISABLED and headers
            entry = line[len("SCENERY_PACK "):].strip()
            if entry == "*GLOBAL_AIRPORTS*":
                sources.append(("Global Airports", os.path.join(
                    xplane_dir, "Global Scenery", "Global Airports", "Earth nav data", "apt.dat")))
                continue
            path = entry if os.path.isabs(entry) else os.path.join(xplane_dir, entry)
            sources.append((os.path.basename(entry.rstrip("/\\")), os.path.join(path, "Earth nav data", "apt.dat")))
    return sources


def _near(rings, lat0, lon0, radius_m):
    kx = _m_per_deg_lon(lat0)
    r2 = radius_m * radius_m
    return any(((lon - lon0) * kx) ** 2 + ((lat - lat0) * EARTH_M_PER_DEG_LAT) ** 2 <= r2
               for outer, _ in rings for lon, lat in outer)


def airports_near(xplane_dir, lat0, lon0, radius_m, log=print):
    """Every airport with pavement within radius_m of the point, using the
    highest-priority definition of each.  Returns a list of Airport."""
    seen, out = set(), []
    for name, apt in _apt_sources(xplane_dir):
        try:
            if not os.path.isfile(apt):
                continue
            for icao, pavement, bounds in parse_apt_dat(apt):
                if icao in seen or not pavement or not _near(pavement, lat0, lon0, radius_m):
                    continue
                seen.add(icao)
                pts = [p for outer, _ in pavement for p in outer]
                clat = sum(p[1] for p in pts) / len(pts)
                clon = sum(p[0] for p in pts) / len(pts)
                kx = _m_per_deg_lon(clat)
                reach = max(math.hypot((p[0] - clon) * kx, (p[1] - clat) * EARTH_M_PER_DEG_LAT) for p in pts)
                out.append(Airport(icao, name, pavement, bounds, clat, clon, reach))
        except OSError as exc:                   # unreadable disk sectors: skip, don't abort
            log(f"apt.dat unreadable, skipped: {apt}: {exc}")
    return out


def airports_for_tile(xplane_dir, tile_lat, tile_lon, cache_dir=None, log=print):
    """Airports whose centre lies in (or within ~5 km of) the one-degree tile.

    Parsing every apt.dat (the global one is ~380 MB) takes 10-25 s, so the
    result is cached per tile against scenery_packs.ini and the global apt.dat
    (size and mtime): any change to either re-reads them.
    """
    gapt = os.path.join(xplane_dir, "Global Scenery", "Global Airports", "Earth nav data", "apt.dat")
    ini = os.path.join(xplane_dir, "Custom Scenery", "scenery_packs.ini")
    stamp = (_CACHE_VERSION,) + tuple(
        (os.path.getsize(p), int(os.path.getmtime(p))) if os.path.exists(p) else None for p in (ini, gapt))
    cpath = None
    if cache_dir:
        cpath = os.path.join(cache_dir, f"airports_{tile_lat:+03d}{tile_lon:+04d}.pkl")
        try:
            with open(cpath, "rb") as f:
                cached = pickle.load(f)
            if cached.get("stamp") == stamp:
                return [Airport(*a) for a in cached["airports"]]
        except (OSError, pickle.UnpicklingError, EOFError, KeyError, TypeError):
            pass
    # 80 km from the tile centre reaches every corner of a one-degree tile.
    found = airports_near(xplane_dir, tile_lat + 0.5, tile_lon + 0.5, 80000.0, log=log)
    margin = 0.05
    airports = [a for a in found if tile_lat - margin <= a.lat <= tile_lat + 1 + margin
                and tile_lon - margin <= a.lon <= tile_lon + 1 + margin]
    if cpath:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            tmp = cpath + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump({"stamp": stamp, "airports": [tuple(a) for a in airports]}, f)
            os.replace(tmp, cpath)
        except OSError:
            pass
    return airports
