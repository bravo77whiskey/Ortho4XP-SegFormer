"""Greenhouse meshes in X-Plane OBJ space (x east, y up, z south).

Vertices are (x, y, z, nx, ny, nz, u, v); triangles may come in either
winding -- the writer orients each one to its vertex normals.  UVs come
from ``layout`` so every texture set (Cycles bake, AI images) fits.

Detail that reads from a cockpit, kept cheap:

tunnels   every bay gets its own span, height, sag along its length, roof
          texture variant, texture offset and gable variants; soil berms
          bury the film feet; a door frame stands proud of each gable.
solar     the film sags between joints; insulation quilts are rolled at
          the ridge in uneven lengths and thicknesses; the back wall is
          vertical masonry with a flat coping (brick in variant a, cement
          render in b); gables are thick walls of the same finish that stand
          above the film; a brick anteroom sits at one end.

Three LOD bands share per-bay randomness, so the silhouette does not
change between them.  Bounds are centred and never exceed the declared
footprint (tunnels: bays x pitch by length; solar: length by width).
"""

from __future__ import annotations

import math
import zlib

import numpy as np

import layout as L

LODS = ((0, 600), (600, 1800), (1800, 5000))

TUNNEL_PITCH_M = 8.0
TUNNEL_SPAN_M = 7.0
TUNNEL_HEIGHT_M = 3.2
TUNNEL_EXPONENT = 0.6            # y = H * sin(t) ** e: steep sides
GABLE_INSET_M = 0.15             # gable sits inside the footprint end...
DOOR_PROUD_M = 0.12              # ...so the door frame stays within it


def _rng(*key):
    return np.random.default_rng(zlib.crc32(repr(key).encode("ascii")))


class Mesh:
    def __init__(self):
        self.verts = []
        self.tris = []

    def v(self, p, n, uv):
        self.verts.append((float(p[0]), float(p[1]), float(p[2]),
                           float(n[0]), float(n[1]), float(n[2]),
                           float(uv[0]), float(uv[1])))
        return len(self.verts) - 1

    def quad(self, a, b, c, d):
        self.tris += [a, b, c, a, c, d]

    def fan(self, centre, ring):
        for i in range(len(ring) - 1):
            self.tris += [centre, ring[i], ring[i + 1]]

    def grid(self, rows):
        """rows: list of rows of vertex ids, all the same length."""
        for r0, r1 in zip(rows, rows[1:]):
            for i in range(len(r0) - 1):
                self.quad(r0[i], r0[i + 1], r1[i + 1], r1[i])


def _profile_normals(pts):
    """Outward normals of an (s, y) polyline running left to right."""
    out = []
    for i in range(len(pts)):
        a = pts[max(0, i - 1)]
        b = pts[min(len(pts) - 1, i + 1)]
        ts, ty = b[0] - a[0], b[1] - a[1]
        ln = math.hypot(ts, ty) or 1.0
        out.append((-ty / ln, ts / ln))
    return out


def _arc_fractions(pts):
    arc = [0.0]
    for i in range(1, len(pts)):
        arc.append(arc[-1] + math.dist(pts[i - 1], pts[i]))
    total = arc[-1] or 1.0
    return [a / total for a in arc]


# ── tunnels ─────────────────────────────────────────────────────────────────

def _tunnel_unit_profile(segments):
    pts = []
    for k in range(segments + 1):
        t = math.pi * k / segments
        pts.append((-0.5 * math.cos(t), max(0.0, math.sin(t)) ** TUNNEL_EXPONENT))
    return pts


def tunnel_bay_params(bays, length_m, b):
    rng = _rng("tunnel", bays, int(length_m), b)
    n_joints = max(1, int(round(length_m / 20.0))) + 1
    return {
        "span": TUNNEL_SPAN_M * (1.0 + rng.uniform(-0.04, 0.01)),
        "height": TUNNEL_HEIGHT_M * (1.0 + rng.uniform(-0.07, 0.05)),
        "roof": "roof_b" if rng.random() < 0.35 else "roof_a",
        "voff": float(rng.random()),
        "gables": (L.TUNNEL_GABLES[int(rng.integers(4))],
                   L.TUNNEL_GABLES[int(rng.integers(4))]),
        # sag and sideways wander at the length joints (near LOD only)
        "sag": [1.0 + rng.uniform(-0.035, 0.02) for _ in range(n_joints)],
        "wander": [rng.uniform(-0.05, 0.05) for _ in range(n_joints)],
    }


def tunnel_mesh(bays, length_m, lod):
    m = Mesh()
    segments = (12, 7, 4)[lod]
    unit = _tunnel_unit_profile(segments)
    frac = _arc_fractions(unit)
    z_end = length_m / 2.0 - GABLE_INSET_M
    z0, z1 = -z_end, z_end
    pitch = TUNNEL_PITCH_M
    for b in range(bays):
        p = tunnel_bay_params(bays, length_m, b)
        cx = (b - (bays - 1) / 2.0) * pitch
        span, height = p["span"], p["height"]
        roof = L.TUNNELS[p["roof"]]
        n_j = len(p["sag"]) if lod == 0 else 2
        sag = p["sag"] if lod == 0 else [1.0, 1.0]
        wander = p["wander"] if lod == 0 else [0.0, 0.0]
        rows = []
        for j in range(n_j):
            z = z0 + (z1 - z0) * j / (n_j - 1)
            h = height * sag[j]
            prof = [(s * span, y * h) for s, y in unit]
            nrm = _profile_normals(prof)
            row = []
            for (s, y), (ns, ny), a in zip(prof, nrm, frac):
                bcoord = (z - z0) / L.ALONG_PERIOD_M + p["voff"]
                row.append(m.v((cx + wander[j] + s, y, z), (ns, ny, 0.0),
                               roof.uv(a, bcoord)))
            rows.append(row)
        # grid is (joint, arch) -> quads between consecutive joints
        for r0, r1 in zip(rows, rows[1:]):
            for i in range(segments):
                m.quad(r0[i], r0[i + 1], r1[i + 1], r1[i])
        # gables: fans from the ground centre
        for end, (z, nz) in enumerate(((z0, -1.0), (z1, 1.0))):
            reg = L.TUNNELS[p["gables"][end]]
            h = height * sag[0 if end == 0 else -1]
            w = wander[0 if end == 0 else -1]
            centre = m.v((cx + w, 0.0, z), (0, 0, nz), reg.uv(0.5, 0.0))
            ring = []
            for s, y in unit:
                ring.append(m.v((cx + w + s * span, y * h, z), (0, 0, nz),
                                reg.uv(s + 0.5, y)))
            m.fan(centre, ring)
            door = L.TUNNEL_DOORS[p["gables"][end]]
            if lod == 0 and door is not None:      # None: an open tunnel end
                _door_frame(m, reg, door, cx + w, z, nz, span, h)
        if lod == 0:
            for side in (-1.0, 1.0):
                _berm(m, roof, cx, span, height, side, z0, z1, p["voff"])
    return m


def _door_frame(m, reg, door, cx, z, nz, span, h):
    a0, a1, b0, b1 = door
    x0, x1 = cx + (a0 - 0.5) * span, cx + (a1 - 0.5) * span
    top = min(b1 * TUNNEL_HEIGHT_M, 0.7 * h)
    zf = z + nz * DOOR_PROUD_M
    bt = top / h
    faces = [
        # front
        ([(x0, 0, zf), (x1, 0, zf), (x1, top, zf), (x0, top, zf)], (0, 0, nz),
         [(a0, 0), (a1, 0), (a1, bt), (a0, bt)]),
        # jambs and head, sampling a sliver of the door edge
        ([(x0, 0, z), (x0, 0, zf), (x0, top, zf), (x0, top, z)], (-1, 0, 0),
         [(a0 - 0.01, 0), (a0, 0), (a0, bt), (a0 - 0.01, bt)]),
        ([(x1, 0, zf), (x1, 0, z), (x1, top, z), (x1, top, zf)], (1, 0, 0),
         [(a1, 0), (a1 + 0.01, 0), (a1 + 0.01, bt), (a1, bt)]),
        ([(x0, top, zf), (x1, top, zf), (x1, top, z), (x0, top, z)], (0, 1, 0),
         [(a0, bt), (a1, bt), (a1, bt + 0.02), (a0, bt + 0.02)]),
    ]
    for pts, n, uvs in faces:
        ids = [m.v(p, n, reg.uv(*uv)) for p, uv in zip(pts, uvs)]
        m.quad(*ids)


def _berm(m, roof, cx, span, height, side, z0, z1, voff):
    """Soil heaped over the film foot, outward of the tunnel on ``side``."""
    foot = cx + side * span / 2.0
    limit = cx + side * (TUNNEL_PITCH_M / 2.0 - 0.02)
    inner = (foot - side * 0.03, 0.26)
    top = (foot + side * 0.14, 0.2)
    outer = (limit if abs(limit - foot) < 0.5 else foot + side * 0.5, 0.0)
    band = L.FOOT_SOIL
    a_in, a_top, a_out = (0.7 * band, 0.35 * band, 0.0) if side < 0 else (
        1 - 0.7 * band, 1 - 0.35 * band, 1.0)
    bl = (0.0 + voff, (z1 - z0) / L.ALONG_PERIOD_M + voff)
    for (pa, aa), (pb, ab) in (((inner, a_in), (top, a_top)),
                               ((top, a_top), (outer, a_out))):
        dx, dy = pb[0] - pa[0], pb[1] - pa[1]
        n = (-dy * side, abs(dx), 0.0)
        ln = math.hypot(n[0], n[1]) or 1.0
        n = (n[0] / ln, n[1] / ln, 0.0)
        ids = [m.v((pa[0], pa[1], z0), n, roof.uv(aa, bl[0])),
               m.v((pb[0], pb[1], z0), n, roof.uv(ab, bl[0])),
               m.v((pb[0], pb[1], z1), n, roof.uv(ab, bl[1])),
               m.v((pa[0], pa[1], z1), n, roof.uv(aa, bl[1]))]
        m.quad(*ids)


# ── solar greenhouses ──────────────────────────────────────────────────────

SOLAR_KNEE_M = 0.8
WALL_TOP_M = 0.9                 # masonry back wall, vertical, coped flat
BACK_RUN_M = 1.2                 # horizontal run of the back roof
BACK_DROP_M = 0.9
GABLE_THICK_M = 0.4
GABLE_RISE_M = 0.15              # gable walls stand above the film
ANTE_LEN_M = 3.5
ANTE_DEPTH = (0.3, 4.3)          # d range of the anteroom
ANTE_HEIGHT_M = 2.6


def solar_height_m(width_m):
    return 0.25 * width_m + 1.0


def solar_section(width_m, segments):
    """Cross-section outline pieces in (d, y); d = 0 at the south foot."""
    h = solar_height_m(width_m)
    hw = h - BACK_DROP_M
    d_ridge = width_m - WALL_TOP_M - BACK_RUN_M
    front = [(0.0, 0.0)]
    for k in range(segments + 1):
        t = k / segments
        front.append((d_ridge * t + 0.02 * (k == 0),
                      SOLAR_KNEE_M + (h - SOLAR_KNEE_M) * (1 - (1 - t) ** 1.8)))
    back = [(d_ridge, h), (width_m - WALL_TOP_M, hw)]
    cap = [back[-1], (width_m, hw)]
    wall = [cap[-1], (width_m, 0.0)]
    return {"front": front, "back": back, "cap": cap, "wall": wall,
            "h": h, "hw": hw, "d_ridge": d_ridge}


def _front_y(front, d):
    for (d0, y0), (d1, y1) in zip(front, front[1:]):
        if d0 <= d <= d1 and d1 > d0:
            return y0 + (y1 - y0) * (d - d0) / (d1 - d0)
    return front[-1][1]


def solar_params(length_m, width_m, variant):
    rng = _rng("solar", int(length_m), int(width_m), variant)
    body = length_m - GABLE_THICK_M - (ANTE_LEN_M if length_m >= 30 else GABLE_THICK_M)
    n_j = max(1, int(round(body / 20.0))) + 1
    quilts = []
    x = 0.0
    while x < body - 0.5:
        seg = float(rng.uniform(4.0, 11.0))
        quilts.append((x, min(body, x + seg), float(rng.uniform(0.24, 0.42)),
                       float(rng.uniform(-0.25, 0.25))))
        x += seg
    return {
        "sag": [1.0 + rng.uniform(-0.03, 0.015) for _ in range(n_j)],
        "voff": float(rng.random()),
        "qoff": float(rng.random()),
        "quilts": quilts,
    }


def solar_mesh(length_m, width_m, lod, variant="a"):
    """``variant`` 'b' mirrors the anteroom to the west end."""
    m = Mesh()
    segments = (12, 6, 4)[lod]
    sec = solar_section(width_m, segments)
    p = solar_params(length_m, width_m, variant)
    has_ante = length_m >= 30
    sign = 1.0 if variant == "a" else -1.0
    # body runs from the plain gable to the anteroom (or second gable)
    x_plain = -sign * (length_m / 2.0 - GABLE_THICK_M)
    x_ante = sign * (length_m / 2.0 - (ANTE_LEN_M if has_ante else GABLE_THICK_M))
    xa, xb = min(x_plain, x_ante), max(x_plain, x_ante)

    def P(x, d, y):
        return (x, y, width_m / 2.0 - d)

    def N(nd, ny):       # (d, y) normal -> OBJ normal (z = -d)
        return (0.0, ny, -nd)

    film, quilt = L.SOLAR["film"], L.SOLAR["quilt"]
    wall_name, gable_name = L.SOLAR_FINISH[variant]
    back, wall = L.SOLAR["back"], L.SOLAR[wall_name]
    per = L.ALONG_PERIOD_M

    # the knee: a low masonry kerb in the wall finish, film tucked in on top
    kerb = sec["front"][:2]
    ids = [[m.v(P(x, d, y), N(-1.0, 0.0), wall.uv(y / sec["hw"], (x - xa) / per + p["voff"]))
            for d, y in kerb] for x in (xa, xb)]
    m.quad(ids[0][0], ids[0][1], ids[1][1], ids[1][0])

    # film from the kerb to the ridge, sagging between joints
    front = sec["front"][1:]
    frac = _arc_fractions(front)
    sag = p["sag"] if lod == 0 else [1.0, 1.0]
    rows = []
    for j in range(len(sag)):
        x = xa + (xb - xa) * j / (len(sag) - 1)
        pts = [(d, y if y <= SOLAR_KNEE_M else
                SOLAR_KNEE_M + (y - SOLAR_KNEE_M) * sag[j]) for d, y in front]
        nrm = _profile_normals(pts)
        rows.append([m.v(P(x, d, y), N(nd, ny),
                         film.uv(a, (x - xa) / per + p["voff"]))
                     for (d, y), (nd, ny), a in zip(pts, nrm, frac)])
    for r0, r1 in zip(rows, rows[1:]):
        for i in range(len(front) - 1):
            m.quad(r0[i], r0[i + 1], r1[i + 1], r1[i])

    # back roof, wall coping and wall face (one span each)
    for piece, reg, a_of in (
        (sec["back"], back, lambda k, n: k / (n - 1)),
        (sec["cap"], wall, lambda k, n: 0.97 + 0.03 * k / (n - 1)),
        (sec["wall"], wall, lambda k, n: 1.0 - k / (n - 1)),
    ):
        nrm = _profile_normals(piece)
        ids = []
        for x in (xa, xb):
            ids.append([m.v(P(x, d, y), N(nd, ny),
                            reg.uv(a_of(k, len(piece)), (x - xa) / per + p["voff"]))
                        for k, ((d, y), (nd, ny)) in enumerate(zip(piece, nrm))])
        for i in range(len(piece) - 1):
            m.quad(ids[0][i], ids[0][i + 1], ids[1][i + 1], ids[1][i])

    # quilts rolled at the ridge
    if lod < 2:
        sides = 6 if lod == 0 else 4
        rolls = p["quilts"] if lod == 0 else [(0.0, xb - xa, 0.33, 0.0)]
        for q0, q1, r, shift in rolls:
            d_c = sec["d_ridge"] - 0.55 - r + shift
            y_c = _front_y(front, d_c) + 0.92 * r
            ring = [(math.cos(2 * math.pi * k / sides), math.sin(2 * math.pi * k / sides))
                    for k in range(sides + 1)]
            ends = []
            for x in (xa + q0, xa + q1):
                ends.append([m.v(P(x, d_c + r * c, y_c + r * s), N(c, s),
                                 quilt.uv(k / sides, (x - xa) / per + p["qoff"]))
                             for k, (c, s) in enumerate(ring)])
            for i in range(sides):
                m.quad(ends[0][i], ends[0][i + 1], ends[1][i + 1], ends[1][i])
            for x, ring_ids, nx in ((xa + q0, ends[0], -1.0), (xa + q1, ends[1], 1.0)):
                c_id = m.v(P(x, d_c, y_c), (nx, 0, 0), quilt.uv(0.5, 0.1))
                caps = [m.v(m.verts[k][:3], (nx, 0, 0), quilt.uv(0.5 + 0.4 * c, 0.1 + 0.05 * s))
                        for k, (c, s) in zip(ring_ids, ring)]
                m.fan(c_id, caps)

    # gable walls: outline raised above the film, thickness toward the end
    gable = L.SOLAR[gable_name]
    outline = ([(d, y + GABLE_RISE_M * min(1.0, y / SOLAR_KNEE_M))
                for d, y in sec["front"][1:]] +
               [(d, y + GABLE_RISE_M) for d, y in sec["back"][1:]] +
               [sec["cap"][-1], sec["wall"][-1], (0.0, 0.0)])
    hg = sec["h"] + GABLE_RISE_M
    # (body end, outward sign, slab thickness): the gable behind an
    # anteroom is a bare face, the anteroom stands against it
    ends = [(x_plain, -sign, GABLE_THICK_M),
            (x_ante, sign, 0.0 if has_ante else GABLE_THICK_M)]
    for x_in, out, thick in ends:
        x_out = x_in + out * thick
        cd = sum(d for d, _ in outline) / len(outline)
        cy = sum(y for _, y in outline) / len(outline)
        c_id = m.v(P(x_out, cd, cy), (out, 0, 0), gable.uv(cd / width_m, cy / hg))
        ring = [m.v(P(x_out, d, y), (out, 0, 0), gable.uv(d / width_m, y / hg))
                for d, y in outline + [outline[0]]]
        m.fan(c_id, ring)
        if lod == 0 and x_out != x_in:
            # the slab's exposed edge: south knee, top, back and wall face
            top = [(0.0, 0.0)] + outline[:-1]
            nrm = _profile_normals(top)
            rows = [[m.v(P(x, d, y), N(nd, ny), gable.uv(d / width_m, 0.985 + 0.015 * (x == x_out)))
                     for (d, y), (nd, ny) in zip(top, nrm)] for x in (x_in, x_out)]
            for i in range(len(top) - 1):
                m.quad(rows[0][i], rows[0][i + 1], rows[1][i + 1], rows[1][i])

    if has_ante:
        _anteroom(m, x_ante, sign * length_m / 2.0, width_m)
    return m


def _anteroom(m, x_in, x_out, width_m):
    wall_r, roof_r = L.SOLAR["ante_wall"], L.SOLAR["ante_roof"]
    d0, d1 = ANTE_DEPTH
    z_s, z_n = width_m / 2.0 - d0, width_m / 2.0 - d1
    hh = ANTE_HEIGHT_M
    out = 1.0 if x_out > x_in else -1.0
    depth = d1 - d0
    per = 2 * abs(x_out - x_in) + depth
    run = abs(x_out - x_in)
    # (from, to, normal) along the perimeter: south, end, north
    faces = [
        ((x_in, z_s), (x_out, z_s), (0, 0, 1.0)),
        ((x_out, z_s), (x_out, z_n), (out, 0, 0)),
        ((x_out, z_n), (x_in, z_n), (0, 0, -1.0)),
    ]
    s0 = 0.0
    for (xa, za), (xb, zb), n in faces:
        ln = math.hypot(xb - xa, zb - za)
        a0, a1 = s0 / per, (s0 + ln) / per
        ids = [m.v((xa, 0, za), n, wall_r.uv(a0, 0.0)),
               m.v((xb, 0, zb), n, wall_r.uv(a1, 0.0)),
               m.v((xb, hh, zb), n, wall_r.uv(a1, 1.0)),
               m.v((xa, hh, za), n, wall_r.uv(a0, 1.0))]
        m.quad(*ids)
        s0 += ln
    del run
    # the side against the gable shows above the low front of the film
    n_in = (-out, 0, 0)
    ids = [m.v((x_in, 0, z_n), n_in, wall_r.uv(0.62, 0.0)),
           m.v((x_in, 0, z_s), n_in, wall_r.uv(0.95, 0.0)),
           m.v((x_in, hh, z_s), n_in, wall_r.uv(0.95, 1.0)),
           m.v((x_in, hh, z_n), n_in, wall_r.uv(0.62, 1.0))]
    m.quad(*ids)
    ids = [m.v((x_in, hh, z_s), (0, 1, 0), roof_r.uv(0, 0)),
           m.v((x_out, hh, z_s), (0, 1, 0), roof_r.uv(1, 0)),
           m.v((x_out, hh, z_n), (0, 1, 0), roof_r.uv(1, 1)),
           m.v((x_in, hh, z_n), (0, 1, 0), roof_r.uv(0, 1))]
    m.quad(*ids)


def bounds(mesh):
    a = np.asarray([v[:3] for v in mesh.verts])
    return a.min(axis=0), a.max(axis=0)
