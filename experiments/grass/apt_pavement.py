"""Airport pavement from apt.dat, for keeping grass off runways, taxiways and aprons.

X-Plane draws an airport from the highest-priority pack that defines it, so
packs are read in scenery_packs.ini order (enabled entries only) and the first
definition of each airport wins; *GLOBAL_AIRPORTS* is read at its place in that
order.  Only airports with geometry inside the search circle are kept.

Shapes, as (outer_ring, [hole_rings]) with rings of (lon, lat):
  * 100 land runways: the full-width rectangle between the two runway ends,
    extended by each end's blast pad (overrun) length;
  * 102 helipads: their rectangle;
  * 110 pavement (taxiways, aprons, shoulders): the boundary loops, with
    Bezier edges sampled; the first loop is the outline, later loops holes.
"""

import math
import os

EARTH_M_PER_DEG_LAT = 110574.0
BEZIER_STEPS = 8


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
    out = []
    for sa, sp in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        out.append(_offset(lon, lat, sa * ax * length / 2 + sp * px * width / 2,
                           sa * ay * length / 2 + sp * py * width / 2))
    return out


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
    """Yield (icao, [(outer, holes), ...]) for every airport in one apt.dat."""
    icao, shapes = None, []
    loops, nodes, in_pavement = [], [], False

    def close_pavement():
        nonlocal loops, nodes, in_pavement
        if in_pavement and loops:
            shapes.append((loops[0], loops[1:]))
        loops, nodes, in_pavement = [], [], False

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            code = parts[0]
            if code in ("1", "16", "17", "99"):
                close_pavement()
                if icao is not None:
                    yield icao, shapes
                icao, shapes = (parts[4] if len(parts) > 4 and code != "99" else None), []
                continue
            if icao is None:
                continue
            if code in ("111", "112", "113", "114", "115", "116"):
                if not in_pavement:
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
            close_pavement()
            try:
                if code == "110":
                    in_pavement = True
                elif code == "100":
                    width = float(parts[1])
                    lat1, lon1, blast1 = float(parts[9]), float(parts[10]), float(parts[12])
                    lat2, lon2, blast2 = float(parts[18]), float(parts[19]), float(parts[21])
                    r = _runway_rect(width, lat1, lon1, blast1, lat2, lon2, blast2)
                    if r:
                        shapes.append((r, []))
                elif code == "102":
                    r = _helipad_rect(float(parts[2]), float(parts[3]), float(parts[4]),
                                      float(parts[5]), float(parts[6]))
                    shapes.append((r, []))
            except (IndexError, ValueError):
                pass
    close_pavement()
    if icao is not None:
        yield icao, shapes


def _shapes_near(shapes, lat0, lon0, radius_m):
    kx = _m_per_deg_lon(lat0)
    for outer, _ in shapes:
        for lon, lat in outer:
            if ((lon - lon0) * kx) ** 2 + ((lat - lat0) * EARTH_M_PER_DEG_LAT) ** 2 <= radius_m ** 2:
                return True
    return False


def pavement_near(xplane_dir, lat0, lon0, radius_m, log=print):
    """Pavement shapes of every airport touching the circle, highest-priority
    definition per airport.  Returns (shapes, [(icao, source_pack)])."""
    cs = os.path.join(xplane_dir, "Custom Scenery")
    ini = os.path.join(cs, "scenery_packs.ini")
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
            sources.append((os.path.basename(entry.rstrip("/\\")),
                            os.path.join(path, "Earth nav data", "apt.dat")))
    seen, shapes, used = set(), [], []
    margin = radius_m + 5000.0                   # an airport can start outside the circle
    for name, apt in sources:
        try:
            if not os.path.isfile(apt):
                continue
            for icao, ap_shapes in parse_apt_dat(apt):
                if icao in seen or not ap_shapes:
                    continue
                if not _shapes_near(ap_shapes, lat0, lon0, margin):
                    continue
                seen.add(icao)
                shapes.extend(ap_shapes)
                used.append((icao, name, len(ap_shapes)))
        except OSError as exc:                   # bad sectors on H: etc.
            log(f"apt.dat unreadable, skipped: {apt}: {exc}")
    return shapes, used
