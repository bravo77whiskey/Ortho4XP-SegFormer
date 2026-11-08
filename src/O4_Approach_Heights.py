"""Tree and building height limits under runway approaches.

Nothing placed under a runway end may reach its approach surface: the
obstacle free surface (OFS) of ICAO Annex 14 Volume I, Amendment 18
(adopted March 2025, Table 4-1). It is a plane starting at the inner edge, a
set distance before the threshold and as wide as the inner edge, widening by
10 % on each side and rising at the slope (3.33 %, a 3 degree approach less
a margin, on all instrument runways) until it ends; between the inner edge
and the threshold the limit is 0 m. The dimensions depend on the aeroplane
design group the runway serves (guessed from its length and width,
design_group) and on whether the end has an instrument approach (its apt.dat
markings).

Amendment 18 replaced the older approach surface (15 % divergence, 2 % slope,
a second section out to 15 km) with this one; that older shape survives only
as an obstacle evaluation surface, which existing buildings may penetrate
after a study. Real cities keep buildings there, so it is not used here.

The tallest trees X-Plane plants (MAX_TREE_H_M) clear an instrument runway's
surface from 60 + 45 / 0.0333 = 1410 m before the threshold. Nearer than
that the limit falls in a straight line with the distance, to nothing at
the inner edge, so the trees under an approach get gradually shorter towards
the runway, then turn to shrubs; beyond it nothing changes. Where the limit
is under 1 m the overlay still plants 1 m shrubs on vegetation it finds
(O4_Tree_Sizes.SHRUB_FLOOR_M). Buildings (O4_Building_Overlay) use the whole
surface, to 150 m above the threshold 4.5 km out on instrument runways.

The surface is a plane at the threshold's elevation, but trees and
buildings stand on the ground under them, so the limit above that ground is
the surface height plus the threshold elevation minus the ground elevation.
An airport on a raised platform leaves taller trees in the valley under its
approach; a hill under it leaves shorter ones. Elevations come from the
tile's own mesh (mesh_ground), the terrain X-Plane draws. Where the mesh
does not reach (no mesh yet, a threshold or part of the surface in another
tile) the ground counts as level with the threshold.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections import namedtuple

import numpy as np

EARTH_M_PER_DEG_LAT = 110574.0
# Tallest tree in X-Plane 12's 1200 forests library (lib/g8 rainforests).
MAX_TREE_H_M = 45.0
SIGNATURE = "approach-heights-v2"

# Annex 14 (Amendment 18) Table 4-1, approach surface (OFS), per (approach,
# aeroplane design group): (distance from threshold m, inner edge length m,
# length m, slope). Every surface widens by DIVERGENCE on each side.
SURFACES = {
    ("visual", "I"): (30.0, 60.0, 1600.0, 0.05),
    ("visual", "IIA-IIB"): (60.0, 80.0, 2500.0, 0.04),
    ("visual", "IIC"): (60.0, 100.0, 2500.0, 1.0 / 30.0),
    ("visual", "III"): (60.0, 125.0, 2500.0, 1.0 / 30.0),
    ("visual", "IV"): (60.0, 135.0, 2500.0, 1.0 / 30.0),
    ("visual", "V"): (60.0, 150.0, 2500.0, 1.0 / 30.0),
    ("instrument", "I"): (60.0, 110.0, 4500.0, 1.0 / 30.0),
    ("instrument", "IIA-IIB"): (60.0, 125.0, 4500.0, 1.0 / 30.0),
    ("instrument", "IIC"): (60.0, 155.0, 4500.0, 1.0 / 30.0),
    ("instrument", "III"): (60.0, 175.0, 4500.0, 1.0 / 30.0),
    ("instrument", "IV"): (60.0, 185.0, 4500.0, 1.0 / 30.0),
    ("instrument", "V"): (60.0, 200.0, 4500.0, 1.0 / 30.0),
}
DIVERGENCE = 0.10
# apt.dat runway marking codes with an instrument approach: 2/3 non-precision
# and precision, 4/5 the UK ones, 6/7 the EASA ones of X-Plane 12.
_INSTRUMENT_MARKS = (2, 3, 4, 5, 6, 7)

# Ground elevations under a surface are kept on a grid this fine (the mesh
# comes from 30 m DEMs).
GROUND_STEP_M = 15.0

# One approach surface: threshold position, unit vector pointing away from
# the runway (east, north) and its Table 4-1 values; with the mesh, the
# threshold's elevation, the ground under the surface (lon0, lat0, dlon, dlat,
# float32 grid of elevations, NaN off the mesh) and how far that ground rises
# above the threshold at most.
Surface = namedtuple("Surface", "icao end lat lon ux uy inner_half offset divergence slope length "
                                "elev ground rise", defaults=(None, None, 0.0))


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


def design_group(length_m, width_m):
    """Aeroplane design group a runway is built for. apt.dat says nothing of the
    aeroplanes, so short runways take the small groups (by code number) and long
    ones III / IV / V by width (45 m and 60 m runways serve the wide-bodies)."""
    code = code_number(length_m)
    if code < 4:
        return ("I", "IIA-IIB", "IIC")[code - 1]
    if width_m >= 60.0:
        return "V"
    if width_m >= 45.0:
        return "IV"
    return "III"


def approach_kind(marking):
    return "instrument" if int(marking) in _INSTRUMENT_MARKS else "visual"


def inner_edge_m(kind, group, width_m):
    """Table 4-1 inner edge length, with its runway-width footnotes."""
    if kind == "visual":
        if group == "I" and width_m > 30.0:
            return 100.0
        if group == "I" and width_m > 23.0:
            return 80.0
        if group == "IIA-IIB" and width_m > 45.0:
            return 110.0
        if group == "IIA-IIB" and width_m > 30.0:
            return 100.0
        if group == "IIC" and width_m > 45.0:
            return 110.0
    else:
        if group == "I" and width_m > 30.0:
            return 125.0
        if group == "IIA-IIB" and width_m > 30.0:
            return 140.0
        if group == "IIC" and width_m <= 30.0:
            return 140.0
    return SURFACES[(kind, group)][1]


def _limited_m(surface, max_h):
    """Distance beyond the inner edge where the surface reaches ``max_h`` (or ends)."""
    return min(surface.length, max_h / surface.slope)


def reach_m(surface, max_h=MAX_TREE_H_M):
    """Distance before the threshold from which the tallest tree clears the surface
    (on ground rising by the surface's ``rise`` at most)."""
    return surface.offset + _limited_m(surface, max_h + surface.rise)


def runway_surfaces(airports, mesh_path=None):
    """Both approach surfaces of every land runway of ``airports`` (O4_Apt_Pavement.Airport),
    on the ground of the tile mesh at ``mesh_path`` when it exists (mesh_ground)."""
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
            width = float(rwy.width)
            group = design_group(length, width)
            for end, lat, lon, disp, mark, sx, sy in (
                    (1, rwy.lat1, rwy.lon1, rwy.disp1, rwy.mark1, -ux, -uy),
                    (2, rwy.lat2, rwy.lon2, rwy.disp2, rwy.mark2, ux, uy)):
                # The threshold sits the displaced length into the runway.
                t_lon = lon - sx * disp / _kx(lat)
                t_lat = lat - sy * disp / EARTH_M_PER_DEG_LAT
                kind = approach_kind(mark)
                offset, _, sect, slope = SURFACES[(kind, group)]
                inner = inner_edge_m(kind, group, width)
                out.append(Surface(airport.icao, end, t_lat, t_lon, sx, sy,
                                   0.5 * inner, offset, DIVERGENCE, slope, sect))
    if out and mesh_path and os.path.isfile(mesh_path):
        try:
            out = mesh_ground(out, mesh_vertices(mesh_path))
        except Exception as exc:                       # unreadable mesh: level ground
            print(f"[Approach] mesh terrain unreadable ({exc!r}): ground level with the thresholds")
    return out


def mesh_vertices(mesh_path):
    """(lon, lat, elevation m) of every vertex of an Ortho4XP .mesh file."""
    with open(mesh_path, "r", encoding="utf-8", errors="ignore") as f:
        f.readline()                                   # MeshVersionFormatted
        for _ in range(3):
            f.readline()
        n = int(f.readline())
        xyz = np.loadtxt(f, dtype=np.float64, usecols=(0, 1, 2), max_rows=n, ndmin=2)
    xyz[:, 2] *= 100000.0                              # .mesh stores elevations / 100000
    return xyz


def _bilinear(ground, lons, lats):
    """Elevations of ``ground`` (lon0, lat0, dlon, dlat, grid) at the points, NaN off it."""
    lon0, lat0, dlon, dlat, grid = ground
    h, w = grid.shape
    fx = (np.asarray(lons, np.float64) - lon0) / dlon
    fy = (np.asarray(lats, np.float64) - lat0) / dlat
    inside = (fx >= 0.0) & (fx <= w - 1) & (fy >= 0.0) & (fy <= h - 1)
    x0 = np.clip(np.floor(fx).astype(np.int64), 0, max(w - 2, 0))
    y0 = np.clip(np.floor(fy).astype(np.int64), 0, max(h - 2, 0))
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    tx = np.clip(fx - x0, 0.0, 1.0)
    ty = np.clip(fy - y0, 0.0, 1.0)
    out = ((grid[y0, x0] * (1 - tx) + grid[y0, x1] * tx) * (1 - ty)
           + (grid[y1, x0] * (1 - tx) + grid[y1, x1] * tx) * ty)
    return np.where(inside, out, np.nan)


def mesh_ground(surfaces, vertices, step_m=GROUND_STEP_M):
    """``surfaces`` with their threshold elevation and the ground under them,
    interpolated on the mesh triangles (vertices from mesh_vertices). Surfaces
    whose threshold is off the mesh keep level ground."""
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import QhullError

    out = []
    for s in surfaces:
        ring = surface_ring(s, np.inf)
        lons = [p[0] for p in ring] + [s.lon]
        lats = [p[1] for p in ring] + [s.lat]
        kx = _kx(s.lat)
        pad_lon, pad_lat = 100.0 / kx, 100.0 / EARTH_M_PER_DEG_LAT
        w, e = min(lons) - pad_lon, max(lons) + pad_lon
        so, n = min(lats) - pad_lat, max(lats) + pad_lat
        near = vertices[(vertices[:, 0] >= w) & (vertices[:, 0] <= e)
                        & (vertices[:, 1] >= so) & (vertices[:, 1] <= n)]
        if len(near) < 3:
            out.append(s)
            continue
        try:
            interp = LinearNDInterpolator(near[:, :2], near[:, 2])
        except (QhullError, ValueError):
            out.append(s)
            continue
        elev = float(interp(s.lon, s.lat))
        if not math.isfinite(elev):
            out.append(s)
            continue
        dlon, dlat = step_m / kx, step_m / EARTH_M_PER_DEG_LAT
        gx = w + np.arange(int(math.ceil((e - w) / dlon)) + 1) * dlon
        gy = so + np.arange(int(math.ceil((n - so) / dlat)) + 1) * dlat
        glon, glat = np.meshgrid(gx, gy)
        grid = interp(glon, glat).astype(np.float32)
        # Highest ground under the surface itself (not the padded box around it).
        under = np.isfinite(_limit(s, (glon - s.lon) * kx, (glat - s.lat) * EARTH_M_PER_DEG_LAT, np.inf))
        under &= np.isfinite(grid)
        rise = float(grid[under].max() - elev) if under.any() else 0.0
        out.append(s._replace(elev=elev, ground=(w, so, dlon, dlat, grid), rise=max(rise, 0.0)))
    return out


def surface_ring(surface, max_h=MAX_TREE_H_M):
    """(lon, lat) corners of the part of a surface where trees are limited."""
    s_end = _limited_m(surface, max_h + surface.rise)
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
    under = (beyond >= 0.0) & (beyond <= s.length) & (side <= s.inner_half + s.divergence * beyond)
    limit[under] = (s.slope * beyond[under]).astype(np.float32)
    limit[before_edge] = 0.0
    if s.ground is not None:
        # Above the ground, not the threshold: plus threshold minus ground.
        e_b, n_b = np.broadcast_arrays(e, n)
        held = np.isfinite(limit)
        ground = _bilinear(s.ground, s.lon + e_b[held] / _kx(s.lat),
                           s.lat + n_b[held] / EARTH_M_PER_DEG_LAT)
        lift = np.where(np.isfinite(ground), s.elev - ground, 0.0)
        limit[held] = np.maximum(limit[held] + lift, 0.0).astype(np.float32)
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


def _ground_key(s):
    if s.ground is None:
        return None
    grid = np.nan_to_num(np.round(s.ground[4], 1), nan=-9999.0)
    return (round(s.elev, 1), hashlib.md5(grid.tobytes()).hexdigest()[:16])


def building_signature(surfaces):
    """Cache-key part for building caps (the whole surface, no height ceiling)."""
    return ("approach-heights-bld-v2", tuple(sorted(
        (s.icao, s.end, round(s.lat, 7), round(s.lon, 7), round(s.ux, 6), round(s.uy, 6),
         s.inner_half, s.offset, s.divergence, round(s.slope, 6), s.length, _ground_key(s))
        for s in surfaces)))


def signature(surfaces):
    """Cache-key part for the surfaces reaching one texture."""
    return (SIGNATURE, MAX_TREE_H_M, tuple(sorted(
        (s.icao, s.end, round(s.lat, 7), round(s.lon, 7), round(s.ux, 6), round(s.uy, 6),
         s.inner_half, s.offset, s.divergence, round(s.slope, 6), s.length, _ground_key(s))
        for s in surfaces)))
