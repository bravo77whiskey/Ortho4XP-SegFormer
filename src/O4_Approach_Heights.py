"""Tree height limits under runway approaches.

No tree may reach the runway end's ICAO Annex 14 approach surface (Table 4-1,
first section): a plane starting at the inner edge, a set distance before
the threshold and as wide as the inner edge, widening by the divergence on
each side and rising at the slope; between the inner edge and the threshold
the limit is 0 m. The surface depends on the runway's code number
(its length) and on how the end is flown (its apt.dat markings: precision,
non-precision or visual).

The tallest trees X-Plane plants (MAX_TREE_H_M) clear even the flattest
surface (2 %) from 60 + 45 / 0.02 = 2310 m before the threshold. Nearer
than that the limit falls in a straight line with the distance, to nothing
at the inner edge, so the trees under an approach get gradually shorter
towards the runway, then turn to shrubs; beyond it nothing changes. Where the
limit is under 1 m the overlay still plants 1 m shrubs on vegetation it
finds (O4_Tree_Sizes.SHRUB_FLOOR_M).

Heights are taken above the ground under the tree, not above the threshold:
the terrain is treated as flat.

Buildings (O4_Building_Overlay) are taller, so they also use the second
section of the surfaces that have one (SECOND_SECTIONS): it continues the
first at a steeper slope, to 150 m above the threshold for code 3/4 runways.
The horizontal section beyond it is left out: real cities keep charted towers
above 150 m there.
"""

from __future__ import annotations

import math
from collections import namedtuple

import numpy as np

EARTH_M_PER_DEG_LAT = 110574.0
# Tallest tree in X-Plane 12's 1200 forests library (lib/g8 rainforests).
MAX_TREE_H_M = 45.0
SIGNATURE = "approach-heights-v1"

# Annex 14 Table 4-1, approach surface, first section, per (approach, code
# number): (inner edge length m, distance from threshold m, divergence each
# side, slope, section length m). Instrument inner edges are the 300 m /
# 150 m of the earlier editions (the current 280 m / 140 m is narrower).
SURFACES = {
    ("visual", 1): (60.0, 30.0, 0.10, 0.05, 1600.0),
    ("visual", 2): (80.0, 60.0, 0.10, 0.04, 2500.0),
    ("visual", 3): (150.0, 60.0, 0.10, 1.0 / 30.0, 3000.0),
    ("visual", 4): (150.0, 60.0, 0.10, 0.025, 3000.0),
    ("nonprecision", 1): (150.0, 60.0, 0.15, 1.0 / 30.0, 2500.0),
    ("nonprecision", 2): (150.0, 60.0, 0.15, 1.0 / 30.0, 2500.0),
    ("nonprecision", 3): (300.0, 60.0, 0.15, 0.02, 3000.0),
    ("nonprecision", 4): (300.0, 60.0, 0.15, 0.02, 3000.0),
    ("precision", 1): (150.0, 60.0, 0.15, 0.025, 3000.0),
    ("precision", 2): (150.0, 60.0, 0.15, 0.025, 3000.0),
    ("precision", 3): (300.0, 60.0, 0.15, 0.02, 3000.0),
    ("precision", 4): (300.0, 60.0, 0.15, 0.02, 3000.0),
}
# Annex 14 Table 4-1, second section: (length m, slope), after the first.
SECOND_SECTIONS = {
    ("nonprecision", 3): (3600.0, 0.025),
    ("nonprecision", 4): (3600.0, 0.025),
    ("precision", 1): (12000.0, 0.03),
    ("precision", 2): (12000.0, 0.03),
    ("precision", 3): (3600.0, 0.025),
    ("precision", 4): (3600.0, 0.025),
}
# apt.dat runway marking codes (6 and 7: the EASA non-precision and precision
# markings of X-Plane 12).
_PRECISION_MARKS = (3, 5, 7)
_NONPRECISION_MARKS = (2, 4, 6)

# One approach surface: threshold position, unit vector pointing away from
# the runway (east, north), the Table 4-1 values of the first section and the
# later sections as ((length, slope), ...).
Surface = namedtuple("Surface", "icao end lat lon ux uy inner_half offset divergence slope length later",
                     defaults=((),))


def _kx(lat):
    return 111320.0 * math.cos(math.radians(lat))


def code_number(length_m):
    """Annex 14 aerodrome reference code number from the runway length."""
    if length_m < 800.0:
        return 1
    if length_m < 1200.0:
        return 2
    if length_m < 1800.0:
        return 3
    return 4


def approach_kind(marking):
    if int(marking) in _PRECISION_MARKS:
        return "precision"
    if int(marking) in _NONPRECISION_MARKS:
        return "nonprecision"
    return "visual"


def _sections(surface):
    return ((surface.length, surface.slope),) + tuple(surface.later or ())


def _limited_m(surface, max_h):
    """Distance beyond the inner edge where the surface reaches ``max_h`` (or ends)."""
    along = top = 0.0
    for length, slope in _sections(surface):
        if top + slope * length >= max_h:
            return along + (max_h - top) / slope
        along += length
        top += slope * length
    return along


def reach_m(surface, max_h=MAX_TREE_H_M):
    """Distance before the threshold from which the tallest tree clears the surface."""
    return surface.offset + _limited_m(surface, max_h)


def runway_surfaces(airports):
    """Both approach surfaces of every land runway of ``airports`` (O4_Apt_Pavement.Airport)."""
    out = []
    for airport in airports:
        for rwy in getattr(airport, "runways", None) or ():
            lat_c = 0.5 * (rwy.lat1 + rwy.lat2)
            dx = (rwy.lon2 - rwy.lon1) * _kx(lat_c)
            dy = (rwy.lat2 - rwy.lat1) * EARTH_M_PER_DEG_LAT
            length = math.hypot(dx, dy)
            if length < 1.0:
                continue
            ux, uy = dx / length, dy / length          # from end 1 towards end 2
            code = code_number(length)
            for end, lat, lon, disp, mark, sx, sy in (
                    (1, rwy.lat1, rwy.lon1, rwy.disp1, rwy.mark1, -ux, -uy),
                    (2, rwy.lat2, rwy.lon2, rwy.disp2, rwy.mark2, ux, uy)):
                # The threshold sits the displaced length into the runway.
                t_lon = lon - sx * disp / _kx(lat)
                t_lat = lat - sy * disp / EARTH_M_PER_DEG_LAT
                kind = approach_kind(mark)
                inner, offset, div, slope, sect = SURFACES[(kind, code)]
                later = (SECOND_SECTIONS[(kind, code)],) if (kind, code) in SECOND_SECTIONS else ()
                out.append(Surface(airport.icao, end, t_lat, t_lon, sx, sy,
                                   0.5 * inner, offset, div, slope, sect, later))
    return out


def surface_ring(surface, max_h=MAX_TREE_H_M):
    """(lon, lat) corners of the part of a surface where trees are limited."""
    s_end = _limited_m(surface, max_h)
    near, far = 0.0, surface.offset + s_end           # from the threshold out
    half_far = surface.inner_half + surface.divergence * s_end
    px, py = -surface.uy, surface.ux
    kx = _kx(surface.lat)
    pts = []
    for along, half in ((near, surface.inner_half), (far, half_far), (far, -half_far),
                        (near, -surface.inner_half)):
        e = surface.ux * along + px * half
        n = surface.uy * along + py * half
        pts.append((surface.lon + e / kx, surface.lat + n / EARTH_M_PER_DEG_LAT))
    return pts


def surfaces_in(surfaces, lat_n, lat_s, lon_w, lon_e, max_h=MAX_TREE_H_M):
    """The surfaces whose limited part reaches the box."""
    out = []
    for s in surfaces:
        ring = surface_ring(s, max_h)
        lons = [p[0] for p in ring]
        lats = [p[1] for p in ring]
        if min(lons) <= lon_e and max(lons) >= lon_w and min(lats) <= lat_n and max(lats) >= lat_s:
            out.append(s)
    return out


def _limit(s, e, n, max_h):
    """Height limit (float32, ``inf`` where none) at east/north offsets in metres
    from the threshold of surface ``s``."""
    out_d = e * s.ux + n * s.uy                   # metres beyond the threshold
    side = np.abs(e * -s.uy + n * s.ux)
    beyond = out_d - s.offset                     # metres beyond the inner edge
    limit = np.full(np.broadcast(out_d, side).shape, np.inf, np.float32)
    before_edge = (out_d >= 0.0) & (beyond < 0.0) & (side <= s.inner_half)
    along = top = 0.0
    for length, slope in _sections(s):
        under = ((beyond >= along) & (beyond <= along + length)
                 & (side <= s.inner_half + s.divergence * beyond) & ~np.isfinite(limit))
        limit[under] = (top + slope * (beyond[under] - along)).astype(np.float32)
        along += length
        top += slope * length
    limit[before_edge] = 0.0
    limit[limit >= max_h] = np.inf
    return limit


def cap_at(surfaces, lons, lats, max_h=np.inf):
    """Height limit in metres at each (lon, lat), ``inf`` where no surface limits it."""
    lons = np.asarray(lons, np.float64)
    lats = np.asarray(lats, np.float64)
    cap = np.full(lons.shape, np.inf, np.float32)
    for s in surfaces:
        e = (lons - s.lon) * _kx(s.lat)
        n = (lats - s.lat) * EARTH_M_PER_DEG_LAT
        np.minimum(cap, _limit(s, e, n, max_h), out=cap)
    return cap


def cap_map(surfaces, lat_n, lat_s, lon_w, lon_e, img_h, img_w, max_h=MAX_TREE_H_M):
    """Height limit in metres per pixel (rows linear in latitude, as the overlay maps
    pixels), ``inf`` where no surface limits trees; None when nothing does."""
    cap = None
    lon_px = (lon_e - lon_w) / img_w
    lat_px = (lat_n - lat_s) / img_h
    for s in surfaces_in(surfaces, lat_n, lat_s, lon_w, lon_e, max_h):
        ring = surface_ring(s, max_h)
        x0 = max(0, int(math.floor((min(p[0] for p in ring) - lon_w) / lon_px)) - 1)
        x1 = min(img_w, int(math.ceil((max(p[0] for p in ring) - lon_w) / lon_px)) + 1)
        y0 = max(0, int(math.floor((lat_n - max(p[1] for p in ring)) / lat_px)) - 1)
        y1 = min(img_h, int(math.ceil((lat_n - min(p[1] for p in ring)) / lat_px)) + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        e = ((lon_w + (np.arange(x0, x1) + 0.5) * lon_px) - s.lon) * _kx(s.lat)
        n = ((lat_n - (np.arange(y0, y1) + 0.5) * lat_px) - s.lat) * EARTH_M_PER_DEG_LAT
        limit = _limit(s, e[None, :], n[:, None], max_h)
        if not np.isfinite(limit).any():
            continue
        if cap is None:
            cap = np.full((img_h, img_w), np.inf, np.float32)
        np.minimum(cap[y0:y1, x0:x1], limit, out=cap[y0:y1, x0:x1])
    return cap


def building_signature(surfaces):
    """Cache-key part for building caps (all sections, no height ceiling)."""
    return ("approach-heights-bld-v1", tuple(sorted(
        (s.icao, s.end, round(s.lat, 7), round(s.lon, 7), round(s.ux, 6), round(s.uy, 6),
         s.inner_half, s.offset, s.divergence, round(s.slope, 6), s.length, tuple(s.later or ()))
        for s in surfaces)))


def signature(surfaces):
    """Cache-key part for the surfaces reaching one texture."""
    return (SIGNATURE, MAX_TREE_H_M, tuple(sorted(
        (s.icao, s.end, round(s.lat, 7), round(s.lon, 7), round(s.ux, 6), round(s.uy, 6),
         s.inner_half, s.offset, s.divergence, round(s.slope, 6), s.length) for s in surfaces)))
