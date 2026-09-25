"""Regional low-poly building archetypes with four LOD bands.

Contract kept from the old library (so the overlay needs no change):
  * the declared footprint L x W is the ROOF outline; every vertex stays
    inside +-L/2 x +-W/2 and the extremes are hit exactly;
  * wall (eave) height = floors x 3.2 m (the ``_LxWxF`` filename token);
  * one texture page (+ its normal map) per OBJ, 4 contiguous ATTR_LOD
    bands, the old band distances and size scaling.

Band content:
  1  full: per-floor facades, eaves with fascia + soffits, barge boards,
     ridge caps, chimneys, dormers, porches/verandas, canopies, arcades;
  2  shell: per-floor facades + bare roof planes (no trims, no add-ons);
  3  box: one plain quad per wall + the roof silhouette;
  4  a single roof-coloured quad.
Band 1 ends at FULL_FRACTION x the old band-1 distance; see budget.py for
why that keeps the frame-cost integral at or below the old library's.
"""

from __future__ import annotations

import hashlib
import math
import random

from layout import bay_period, page_layout
from meshkit import Mesh, Strip, poly_normal
from regions import IND_PAGE, MASSING, REGIONS
import strips as strip_recipes

FLOOR_H = 3.2
TRIM_DROP = 0.22          # fascia board depth below the roof edge
FULL_FRACTION = 0.6       # band 1 = 0 .. 0.6 x old d0
BASE_BANDS = (1100, 2800, 4700, 8500)
_SCALE_REF, _SCALE_MIN, _SCALE_MAX = 30.0, 0.32, 1.65

RES_ARCH = {"gable", "hip", "lshape", "rowhouse", "flatres"}
APT_ARCH = {"aptslab", "aptblock", "flatcom", "shophouse"}
IND_ARCH = {"warehouse", "bigbox"}


# Band 3 of a pitched building keeps its roof silhouette (+4 triangles
# over the old flat-topped box); it hands over to the roof quad 5% sooner
# so its ring costs no more than the old one (see budget.py).
PITCHED_BAND3_END = 0.95


def band_distances(L, W, floors, bands=None):
    diag = math.hypot(math.hypot(L, W), floors * FLOOR_H)
    s = min(_SCALE_MAX, max(_SCALE_MIN, diag / _SCALE_REF))
    d0, d1, d2, d3 = (int(round(b * s)) for b in BASE_BANDS)
    old = (0, d0, d1, d2, d3)
    if bands is not None and bands[2].tri_count() > 12:
        d2 = int(round(d2 * PITCHED_BAND3_END))
    return (0, int(round(d0 * FULL_FRACTION)), d1, d2, d3), old


def stable_seed(parts):
    """Process-independent seed (str hashes are randomised per process)."""
    return int(hashlib.sha1(repr(parts).encode()).hexdigest()[:12], 16)


def strip_seed(bake_page, name):
    return int(hashlib.sha1(f"{bake_page}/{name}".encode()).hexdigest()[:6],
               16)


# ------------------------------------------------------------ context

class Ctx:
    """Everything a builder needs: page, strips, bays, massing, rng."""

    def __init__(self, archetype, region, seed, force=None):
        self.rng = random.Random(seed)
        self.rng_seed = seed
        self.region = region
        self.archetype = archetype
        self.massing = dict(MASSING[region], **(force or {}))
        kit = REGIONS[region]
        if archetype in IND_ARCH:
            self.kind = "ind"
            self.kit = IND_PAGE
            grade = kit["ind"]
            self.page = f"ind_{grade}"
            self.bake_page = "ind"
        elif archetype in APT_ARCH:
            self.kind = "apt"
            self.kit = kit["apt"]
            self.page = f"{region}_apt"
            self.bake_page = self.page
        else:
            self.kind = "res"
            which = self.rng.randrange(len(kit["res"]))
            self.kit = kit["res"][which]
            self.page = f"{region}_res_{'ab'[which]}"
            self.bake_page = self.page
        self.layout = page_layout(self.kind)
        t = self.layout["trim"]["bands"]
        self.trim = {k: Strip(v, k) for k, v in t.items()}

    def strip(self, name):
        return Strip(self.layout[name], name)

    def facade(self, key):
        """Resolved facade for scheme key (s1, a2, c1, i3, ...)."""
        return Facade(self, key)


class Facade:
    def __init__(self, ctx, key):
        self.ctx = ctx
        self.key = key
        self.scheme = ctx.kit[key]
        self.upper = ctx.strip(f"{key}_upper")
        gname = f"{key}_ground"
        self.ground = ctx.strip(gname) if gname in ctx.layout else self.upper
        pname = f"{key}_plain"
        self.plain = ctx.strip(pname) if pname in ctx.layout else \
            ctx.strip(next(k for k in ctx.layout if k.endswith("_plain")))
        self.motif = strip_recipes.motif_len(self.scheme)
        self.p, self.nbays = bay_period(self.upper.world_w,
                                        self.scheme["bay"], self.motif)
        self._tables = {}

    def table(self, level):
        """The exact per-bay table the bake used (kinds + spans)."""
        if level not in self._tables:
            name = f"{self.key}_{level}"
            if name not in self.ctx.layout:
                self._tables[level] = [("win", 1)] * self.nbays
            else:
                rng = random.Random(strip_seed(self.ctx.bake_page, name))
                self._tables[level] = strip_recipes._bay_table(
                    rng, self.nbays, level, self.scheme)
        return self._tables[level]

    def bay_start_with(self, level, kinds, n_bays, rng=None):
        """A bay index so a ``kinds`` bay sits mid-wall, else random."""
        table = self.table(level)
        starts = []
        pos = 0
        for kind, span in table:
            if kind in kinds:
                starts.append(pos)
            pos += span
        rng = rng or self.ctx.rng
        if starts:
            s = rng.choice(starts)
            return (s - n_bays // 2) % self.nbays
        return rng.randrange(self.nbays)


# ------------------------------------------------------------ helpers

def clip_poly_t(pts, axis_o, axis_d, t0, t1):
    """Clip a convex planar polygon to t0 <= dot(p - o, d) <= t1."""
    def t_of(p):
        return sum((p[i] - axis_o[i]) * axis_d[i] for i in range(3))

    def clip(poly, lim, keep_greater):
        out = []
        n = len(poly)
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            ta, tb = t_of(a), t_of(b)
            ina = ta >= lim if keep_greater else ta <= lim
            inb = tb >= lim if keep_greater else tb <= lim
            if ina:
                out.append(a)
            if ina != inb:
                f = (lim - ta) / (tb - ta)
                out.append(tuple(a[k] + (b[k] - a[k]) * f for k in range(3)))
        return out

    poly = clip(pts, t0, True)
    if len(poly) >= 3:
        poly = clip(poly, t1, False)
    return poly if len(poly) >= 3 else []


# How far a roof texture may stretch in v before the face is split into
# another row.  Band 1 keeps the tile scale exact; distant bands trade
# stretch for triangles (a 3 km-away roof never shows its tile courses).
ROOF_STRETCH = {1: 1.0, 2: 1.8, 3: 1e9, 4: 1e9}
_LEVEL = [1]


def set_level(level):
    _LEVEL[0] = level


def roof_face(mesh, pts, up, strip, u0, stretch=None):
    """Roof polygon mapped onto a roof strip, split into rows so the tile
    scale never stretches beyond ROOF_STRETCH for the current band."""
    n = poly_normal(pts)
    ln = math.sqrt(sum(c * c for c in up))
    upn = tuple(c / ln for c in up)
    ts = [sum(p[i] * upn[i] for i in range(3)) for p in pts]
    tmin, tmax = min(ts), max(ts)
    slope = tmax - tmin
    k = ROOF_STRETCH[_LEVEL[0]] if stretch is None else stretch
    strip_h = strip.world_h - 0.05
    rows = max(1, int(math.ceil(slope / (strip_h * k) - 1e-6)))
    row_h = slope / rows
    vs = min(1.0, strip_h / max(row_h, 1e-6))
    right = _cross(upn, n)
    o = (0.0, 0.0, 0.0)
    for r in range(rows):
        t0 = tmin + r * row_h
        t1 = tmax if r == rows - 1 else t0 + row_h
        piece = pts if rows == 1 else clip_poly_t(pts, o, upn, t0, t1)
        if not piece:
            continue
        # anchor u on a fixed world origin so rows line up
        base = piece[0]
        s0 = sum(base[i] * right[i] for i in range(3))
        mesh.planar(piece, strip, upn, u0=u0 + s0 / strip.world_w,
                    origin=base, v_scale=vs,
                    v_off=(sum(base[i] * upn[i] for i in range(3)) - t0) * vs)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def facade_wall(mesh, fac, a, b, z_base, floors, *, level_detail=True,
                door_side=False, start=None, u_start=None, strip_ground=None,
                strip_upper=None):
    """Per-floor wall quads along edge a->b (exterior on the right)."""
    ax, ay = a
    bx, by = b
    length = math.hypot(bx - ax, by - ay)
    if length < 0.05:
        return
    p = fac.p
    n = max(1, int(round(length / p)))
    u_scale = n * p / length
    if start is None:
        kinds = ("door", "garage", "shop", "dock") if door_side else ("win",)
        # seeded by the edge so every LOD band picks the same bays
        erng = random.Random(stable_seed((round(ax, 2), round(ay, 2), round(bx, 2),
                                   round(by, 2), fac.key,
                                   fac.ctx.rng_seed)))
        start = fac.bay_start_with("ground", kinds, n, erng)
    u0 = start * p / fac.upper.world_w if u_start is None else u_start
    # every upper floor slides by whole bay motifs: windows stay stacked,
    # but each floor shows a different stretch of the strip's wear, so no
    # stain repeats floor after floor
    motifs = max(1, fac.nbays // max(1, fac.motif))
    frng = random.Random(stable_seed((round(ax, 2), round(ay, 2), round(bx, 2),
                               round(by, 2), fac.key, fac.ctx.rng_seed,
                               "floors")))
    for f in range(floors):
        z0 = z_base + f * FLOOR_H
        z1 = z0 + FLOOR_H
        st = (strip_ground or fac.ground) if f == 0 else \
            (strip_upper or fac.upper)
        uf = u0
        if st is fac.upper and motifs > 1:
            uf = u0 + frng.randrange(motifs) * fac.motif * p / \
                fac.upper.world_w
        pts = [(ax, ay, z0), (bx, by, z0), (bx, by, z1), (ax, ay, z1)]
        mesh.planar(pts, st, (0, 0, 1), u0=uf, u_scale=u_scale,
                    origin=pts[0])


def plain_wall(mesh, strip, a, b, z0, z1, u0=0.0):
    """One stretched plain quad (far LODs, parapets)."""
    pts = [(a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1),
           (a[0], a[1], z1)]
    h = z1 - z0
    vs = min(1.0, (strip.world_h - 0.05) / max(h, 1e-3))
    mesh.planar(pts, strip, (0, 0, 1), u0=u0, origin=pts[0], v_scale=vs)


def far_quad(mesh, strip, x0, y0, x1, y1, z):
    """Band 4: one roof-coloured quad over the whole footprint."""
    mesh.planar([(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)], strip,
                (0, 1, 0), v_scale=min(1.0, (strip.world_h - 0.05) /
                                       max(y1 - y0, 1e-3)))


def rect_ring(x0, y0, x1, y1):
    """CCW corners seen from above."""
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def ring_walls(mesh, fac, ring, z_base, floors, *, front_edge=0):
    for i in range(len(ring)):
        a, b = ring[i], ring[(i + 1) % len(ring)]
        facade_wall(mesh, fac, a, b, z_base, floors,
                    door_side=(i == front_edge))


def ring_plain(mesh, strip, ring, z0, z1, rng):
    for i in range(len(ring)):
        a, b = ring[i], ring[(i + 1) % len(ring)]
        plain_wall(mesh, strip, a, b, z0, z1, u0=rng.random())


def box_part(mesh, strip, x0, x1, y0, y1, z0, z1, top_strip=None,
             sides=(True, True, True, True), top=True):
    """Chimney/post-style box: 4 sides on ``strip`` + optional top."""
    ring = rect_ring(x0, y0, x1, y1)
    for i, keep in enumerate(sides):
        if not keep:
            continue
        a, b = ring[i], ring[(i + 1) % 4]
        pts = [(a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1),
               (a[0], a[1], z1)]
        vs = min(1.0, (strip.world_h - 0.05) / max(z1 - z0, 1e-3))
        mesh.planar(pts, strip, (0, 0, 1), u0=(i * 0.13 + x0 * 0.01) % 1,
                    origin=pts[0], v_scale=vs)
    if top:
        ts = top_strip or strip
        pts = [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        mesh.planar(pts, ts, (0, 1, 0), origin=pts[0],
                    v_scale=min(1.0, ts.world_h / max(y1 - y0, 1e-3)))


def flat_top(mesh, ctx, roof_strip, x0, y0, x1, y1, z, *, coping=0.3):
    """Coplanar coping ring + roof field (no sunken membrane: the old
    library's sunken roofs rendered see-through in the sim)."""
    c = min(coping, 0.2 * min(x1 - x0, y1 - y0))
    cop = ctx.trim["coping"]
    ix0, iy0, ix1, iy1 = x0 + c, y0 + c, x1 - c, y1 - c
    if c > 0.01:
        for pts in (
            [(x0, y0, z), (x1, y0, z), (ix1, iy0, z), (ix0, iy0, z)],
            [(x1, y0, z), (x1, y1, z), (ix1, iy1, z), (ix1, iy0, z)],
            [(x1, y1, z), (x0, y1, z), (ix0, iy1, z), (ix1, iy1, z)],
            [(x0, y1, z), (x0, y0, z), (ix0, iy0, z), (ix0, iy1, z)],
        ):
            # u along the edge, v across the coping width
            d = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1], 0.0)
            ln = math.hypot(d[0], d[1])
            inward = (-d[1] / ln, d[0] / ln, 0.0)
            mesh.planar(pts, cop, inward, origin=pts[0], v_off=0.05,
                        v_scale=0.3 / c)
    roof_face(mesh, [(ix0, iy0, z), (ix1, iy0, z), (ix1, iy1, z),
                     (ix0, iy1, z)], (0, 1, 0), roof_strip,
              ctx.rng.random())


# ------------------------------------------------------ pitched roofs

class Gable:
    """Gable roof over a rectangle, ridge along x.

    (x0..x1, y0..y1) is the ROOF outline; walls sit ``o`` inside along the
    eaves and ``og`` inside at the gable ends.
    """

    def __init__(self, x0, x1, y0, y1, he, pitch_deg, o, og, max_rise=7.5):
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self.o, self.og = o, og
        self.he = he
        self.yc = (y0 + y1) / 2.0
        ww = (y1 - y0) - 2 * o
        tan = math.tan(math.radians(pitch_deg))
        rise = min(max_rise, ww / 2.0 * tan)
        self.tan = rise / max(ww / 2.0, 1e-3)
        self.ridge = he + rise
        self.ze = he - o * self.tan

    def z_at(self, y):
        return self.ridge - abs(y - self.yc) * self.tan

    def wall_ring(self):
        return rect_ring(self.x0 + self.og, self.y0 + self.o,
                         self.x1 - self.og, self.y1 - self.o)

    def roof(self, mesh, strip, u0):
        x0, x1, y0, y1 = self.x0, self.x1, self.y0, self.y1
        R, ze, yc = self.ridge, self.ze, self.yc
        roof_face(mesh, [(x0, y0, ze), (x1, y0, ze), (x1, yc, R),
                         (x0, yc, R)], (0, yc - y0, R - ze), strip, u0)
        roof_face(mesh, [(x1, y1, ze), (x0, y1, ze), (x0, yc, R),
                         (x1, yc, R)], (0, yc - y1, R - ze), strip, u0 + 0.5)

    def gables(self, mesh, strip, rng):
        wr = self.wall_ring()
        wx0, wy0 = wr[0]
        wx1, wy1 = wr[2]
        for x, a, b in ((wx1, (wx1, wy0), (wx1, wy1)),
                        (wx0, (wx0, wy1), (wx0, wy0))):
            pts = [(a[0], a[1], self.he), (b[0], b[1], self.he),
                   (x, self.yc, self.ridge)]
            rise = self.ridge - self.he
            vs = min(1.0, (strip.world_h - 0.05) / max(rise, 1e-3))
            mesh.planar(pts, strip, (0, 0, 1), u0=rng.random(),
                        origin=pts[0], v_scale=vs)

    def trims(self, mesh, ctx):
        """Fascia + gutter, soffits, barge boards, rake soffits, ridge."""
        x0, x1, y0, y1 = self.x0, self.x1, self.y0, self.y1
        ze, R, yc, d = self.ze, self.ridge, self.yc, TRIM_DROP
        fas, sof = ctx.trim["fascia"], ctx.trim["soffit"]
        # ridge caps sample the roof strip so they match the covering
        rid = ctx.strip(ctx.roof_name)
        wr = self.wall_ring()
        wy0, wy1 = wr[0][1], wr[2][1]
        wx0, wx1 = wr[0][0], wr[1][0]
        # fascia along both eaves (band shows board + gutter)
        for (a, b) in (((x0, y0), (x1, y0)), ((x1, y1), (x0, y1))):
            pts = [(a[0], a[1], ze - d), (b[0], b[1], ze - d),
                   (b[0], b[1], ze), (a[0], a[1], ze)]
            mesh.planar(pts, fas, (0, 0, 1), origin=pts[0],
                        v_scale=0.4 / (d + 0.02))
        # eave soffits (face down), from the fascia back to the wall
        zs = ze - d
        for pts in ([(x0, wy0, zs), (x1, wy0, zs), (x1, y0, zs),
                     (x0, y0, zs)],
                    [(x1, wy1, zs), (x0, wy1, zs), (x0, y1, zs),
                     (x1, y1, zs)]):
            mesh.planar(pts, sof, (1, 0, 0), origin=pts[0],
                        v_scale=0.4 / max(self.o, 0.05))
        # barge boards on the rakes (outer face only) + rake soffits
        for xe, sgn in ((x1, 1), (x0, -1)):
            for (ya, yb) in ((y0, yc), (yc, y1)):
                za, zb = self.z_at(ya), self.z_at(yb)
                if sgn > 0:
                    pts = [(xe, ya, za - d), (xe, yb, zb - d), (xe, yb, zb),
                           (xe, ya, za)]
                else:
                    pts = [(xe, yb, zb - d), (xe, ya, za - d), (xe, ya, za),
                           (xe, yb, zb)]
                mesh.planar(pts, fas, (0, 0, 1), origin=pts[0],
                            v_scale=0.4 / (d + 0.02))
                xw = wx1 if sgn > 0 else wx0
                if abs(xe - xw) > 0.05:
                    if sgn > 0:
                        pts = [(xw, ya, za - d), (xw, yb, zb - d),
                               (xe, yb, zb - d), (xe, ya, za - d)]
                    else:
                        pts = [(xe, ya, za - d), (xe, yb, zb - d),
                               (xw, yb, zb - d), (xw, ya, za - d)]
                    if poly_normal(pts)[2] > 0:
                        pts = pts[::-1]
                    mesh.planar(pts, sof, (1, 0, 0), origin=pts[0],
                                v_scale=0.4 / max(abs(xe - xw), 0.05))
        # ridge cap: two narrow strips over the apex
        w = 0.2
        zc = R + 0.07
        zl = self.z_at(yc - w) + 0.03
        for pts in ([(x0, yc - w, zl), (x1, yc - w, zl), (x1, yc, zc),
                     (x0, yc, zc)],
                    [(x1, yc + w, zl), (x0, yc + w, zl), (x0, yc, zc),
                     (x1, yc, zc)]):
            mesh.planar(pts, rid, (0, 0, 1), origin=pts[0], v_off=0.5,
                        v_scale=2.0)

    def quad(self, mesh, strip):
        z = (self.ze + self.ridge) / 2.0
        mesh.planar([(self.x0, self.y0, z), (self.x1, self.y0, z),
                     (self.x1, self.y1, z), (self.x0, self.y1, z)], strip,
                    (0, 1, 0), v_scale=min(1.0, strip.world_h / max(
                        self.y1 - self.y0, 1e-3)))


class Hip(Gable):
    """Hip roof (pyramid when square); ends slope at the same pitch."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        half_w = (self.y1 - self.y0) / 2.0
        run = (self.ridge - self.ze) / max(self.tan, 1e-3)
        self.run = min(run, half_w)
        cx = (self.x0 + self.x1) / 2.0
        self.xr0 = min(cx, self.x0 + self.run)
        self.xr1 = max(cx, self.x1 - self.run)

    def roof(self, mesh, strip, u0):
        x0, x1, y0, y1 = self.x0, self.x1, self.y0, self.y1
        R, ze, yc = self.ridge, self.ze, self.yc
        xa, xb = self.xr0, self.xr1
        if xb - xa > 0.02:
            roof_face(mesh, [(x0, y0, ze), (x1, y0, ze), (xb, yc, R),
                             (xa, yc, R)], (0, yc - y0, R - ze), strip, u0)
            roof_face(mesh, [(x1, y1, ze), (x0, y1, ze), (xa, yc, R),
                             (xb, yc, R)], (0, yc - y1, R - ze), strip,
                      u0 + 0.5)
        else:
            roof_face(mesh, [(x0, y0, ze), (x1, y0, ze), (xa, yc, R)],
                      (0, yc - y0, R - ze), strip, u0)
            roof_face(mesh, [(x1, y1, ze), (x0, y1, ze), (xa, yc, R)],
                      (0, yc - y1, R - ze), strip, u0 + 0.5)
        roof_face(mesh, [(x1, y0, ze), (x1, y1, ze), (xb, yc, R)],
                  (xb - x1, 0, R - ze), strip, u0 + 0.25)
        roof_face(mesh, [(x0, y1, ze), (x0, y0, ze), (xa, yc, R)],
                  (xa - x0, 0, R - ze), strip, u0 + 0.75)

    def gables(self, mesh, strip, rng):
        return

    def trims(self, mesh, ctx):
        x0, x1, y0, y1 = self.x0, self.x1, self.y0, self.y1
        ze, R, yc, d = self.ze, self.ridge, self.yc, TRIM_DROP
        fas, sof = ctx.trim["fascia"], ctx.trim["soffit"]
        # ridge caps sample the roof strip so they match the covering
        rid = ctx.strip(ctx.roof_name)
        wr = self.wall_ring()
        wx0, wy0 = wr[0]
        wx1, wy1 = wr[2]
        outer = rect_ring(x0, y0, x1, y1)
        inner = rect_ring(wx0, wy0, wx1, wy1)
        zs = ze - d
        for i in range(4):
            a, b = outer[i], outer[(i + 1) % 4]
            pts = [(a[0], a[1], zs), (b[0], b[1], zs), (b[0], b[1], ze),
                   (a[0], a[1], ze)]
            mesh.planar(pts, fas, (0, 0, 1), origin=pts[0],
                        v_scale=0.4 / (d + 0.02))
            ia, ib = inner[i], inner[(i + 1) % 4]
            pts = [(ib[0], ib[1], zs), (ia[0], ia[1], zs), (a[0], a[1], zs),
                   (b[0], b[1], zs)]
            if poly_normal(pts)[2] > 0:
                pts = pts[::-1]
            depth = max(0.05, math.hypot(a[0] - ia[0], a[1] - ia[1]))
            mesh.planar(pts, sof, (1, 0, 0) if i % 2 == 0 else (0, 1, 0),
                        origin=pts[0], v_scale=0.4 / depth)
        if self.xr1 - self.xr0 > 0.02:
            w = 0.2
            zc = R + 0.07
            zl = self.z_at(yc - w) + 0.03
            xa, xb = self.xr0, self.xr1
            for pts in ([(xa, yc - w, zl), (xb, yc - w, zl), (xb, yc, zc),
                         (xa, yc, zc)],
                        [(xb, yc + w, zl), (xa, yc + w, zl), (xa, yc, zc),
                         (xb, yc, zc)]):
                mesh.planar(pts, rid, (0, 0, 1), origin=pts[0], v_off=0.5,
                            v_scale=2.0)


# -------------------------------------------------------- add-ons

def chimney(mesh, ctx, roof, fac, rng):
    wr = roof.wall_ring()
    wx0, wx1 = wr[0][0], wr[1][0]
    if wx1 - wx0 < 3.0:
        return
    cw, cd = rng.uniform(0.5, 0.7), rng.uniform(0.5, 0.8)
    cx = rng.uniform(wx0 + 0.8, wx1 - 0.8 - cw)
    if isinstance(roof, Hip):
        cx = min(max(cx, roof.xr0 + 0.2), max(roof.xr0 + 0.2, roof.xr1 -
                                                cw - 0.2))
    cy = roof.yc + rng.uniform(-1.2, 0.6) - cd / 2
    zlow = min(roof.z_at(cy), roof.z_at(cy + cd)) - 0.1
    ztop = max(roof.ridge + 0.6, max(roof.z_at(cy), roof.z_at(cy + cd)) +
               0.9)
    box_part(mesh, fac.plain, cx, cx + cw, cy, cy + cd, zlow, ztop,
             top_strip=ctx.trim["coping"])


def dormer(mesh, ctx, roof, fac, rng, x_center, width):
    """Gable dormer on the y0 (front) slope with a window from the upper
    facade strip, centred on a real window bay of the bake."""
    tan = roof.tan
    wall_y = roof.y0 + roof.o
    yf = wall_y + 0.35
    zf = roof.z_at(yf)
    hd = 1.75
    zt = zf + hd
    if zt + width / 2 * 0.8 > roof.ridge - 0.25:
        return False
    y_top = roof.yc - (roof.ridge - zt) / tan
    zr = zt + width / 2 * 0.8
    y_meet = roof.yc - (roof.ridge - zr) / tan
    xl, xr = x_center - width / 2, x_center + width / 2
    # front face: rectangle + gable, window from the upper strip
    st = fac.upper
    table = fac.table("upper")
    pos = 0
    wins = []
    for kind, span in table:
        if kind == "win":
            wins.append(pos)
        pos += span
    k = rng.choice(wins) if wins else 0
    win = fac.scheme["window"]
    u_center = (k + 0.5) * fac.p / st.world_w
    v_off = max(0.0, win.get("sill_z", 0.9) - 0.3)
    pts = [(xl, yf, zf), (xr, yf, zf), (xr, yf, zt), (x_center, yf, zr),
           (xl, yf, zt)]
    mesh.planar(pts, st, (0, 0, 1), u0=u_center - (width / 2) / st.world_w,
                origin=pts[0], v_off=v_off)
    # cheeks (triangles bounded below by the main roof)
    plain = fac.plain
    for x, pts in ((xl, [(xl, y_top, zt), (xl, yf, zf), (xl, yf, zt)]),
                   (xr, [(xr, yf, zf), (xr, y_top, zt), (xr, yf, zt)])):
        mesh.planar(pts, plain, (0, 0, 1), origin=(x, yf, zf),
                    u0=rng.random())
    # dormer roof: two planes from a small front overhang back to the roof
    rs = ctx.strip(ctx.roof_name)
    yo = yf - 0.18
    for pts, up in (([(xl - 0.12, yo, zt - 0.1), (x_center, yo, zr),
                      (x_center, y_meet, zr), (xl - 0.12, y_top, zt - 0.1)],
                     (-1, 0, 0.8)),
                    ([(x_center, yo, zr), (xr + 0.12, yo, zt - 0.1),
                      (xr + 0.12, y_top, zt - 0.1), (x_center, y_meet, zr)],
                     (1, 0, 0.8))):
        if poly_normal(pts)[2] < 0:
            pts = pts[::-1]
        roof_face(mesh, pts, (up[0] * -1, 0, 1.0), rs, rng.random())
    return True


def porch(mesh, ctx, x0, x1, y_edge, y_wall, z_top, fac, rng, *,
          pitch=12.0, posts=True):
    """Shed-roofed porch/veranda between the wall line and the footprint
    edge (y_edge < y_wall).  Posts are three-sided (the back never shows)."""
    tan = math.tan(math.radians(pitch))
    depth = y_wall - y_edge
    zw = z_top
    ze = zw - depth * tan
    rs = ctx.strip(ctx.roof_name)
    roof_face(mesh, [(x0, y_edge, ze), (x1, y_edge, ze), (x1, y_wall, zw),
                     (x0, y_wall, zw)], (0, depth, zw - ze), rs,
              rng.random())
    fas, sof = ctx.trim["fascia"], ctx.trim["soffit"]
    d = 0.2
    pts = [(x0, y_edge, ze - d), (x1, y_edge, ze - d), (x1, y_edge, ze),
           (x0, y_edge, ze)]
    mesh.planar(pts, fas, (0, 0, 1), origin=pts[0], v_scale=0.4 / 0.22)
    for xe, flip in ((x0, True), (x1, False)):
        pts = [(xe, y_wall, zw - d), (xe, y_edge, ze - d), (xe, y_edge, ze),
               (xe, y_wall, zw)]
        if flip:
            pts = [pts[1], pts[0], pts[3], pts[2]]
        mesh.planar(pts, fas, (0, 0, 1), origin=pts[0], v_scale=0.4 / 0.22)
    pts = [(x1, y_wall, zw - d), (x0, y_wall, zw - d), (x0, y_edge, ze - d),
           (x1, y_edge, ze - d)]
    if poly_normal(pts)[2] > 0:
        pts = pts[::-1]
    mesh.planar(pts, sof, (1, 0, 0), origin=pts[0], v_scale=0.4 / depth)
    # deck
    deck_h = 0.35
    light = ctx.trim["light"]
    mesh.planar([(x0, y_edge + 0.05, deck_h), (x1, y_edge + 0.05, deck_h),
                 (x1, y_wall, deck_h), (x0, y_wall, deck_h)], ctx.trim["dark"],
                (0, 1, 0), v_scale=0.4 / depth)
    mesh.planar([(x0, y_edge + 0.05, 0.0), (x1, y_edge + 0.05, 0.0),
                 (x1, y_edge + 0.05, deck_h), (x0, y_edge + 0.05, deck_h)],
                ctx.trim["dark"], (0, 0, 1), v_scale=1.0)
    if posts:
        n = max(2, int(round((x1 - x0) / 2.6)) + 1)
        s = 0.16
        for i in range(n):
            px = x0 + 0.12 + (x1 - x0 - 0.24 - s) * i / (n - 1)
            box_part(mesh, light, px, px + s, y_edge + 0.1, y_edge + 0.1 + s,
                     deck_h, ze - d, sides=(True, True, False, True),
                     top=False)


def canopy(mesh, ctx, x0, x1, y_edge, y_wall, z):
    """Flat shop canopy along the front."""
    top = ctx.trim["metal"]
    mesh.planar([(x0, y_edge, z), (x1, y_edge, z), (x1, y_wall, z),
                 (x0, y_wall, z)], top, (0, 1, 0),
                v_scale=0.4 / max(y_wall - y_edge, 0.1))
    d = 0.35
    mesh.planar([(x0, y_edge, z - d), (x1, y_edge, z - d), (x1, y_edge, z),
                 (x0, y_edge, z)], ctx.trim["dark"], (0, 0, 1),
                v_scale=0.4 / d)
    pts = [(x1, y_wall, z - d), (x0, y_wall, z - d), (x0, y_edge, z - d),
           (x1, y_edge, z - d)]
    if poly_normal(pts)[2] > 0:
        pts = pts[::-1]
    mesh.planar(pts, ctx.trim["soffit"], (1, 0, 0),
                v_scale=0.4 / max(y_wall - y_edge, 0.1))


# -------------------------------------------------------------- builders

def _roof_choice(ctx, names=("roof_a", "roof_b", "roof_c")):
    names = [n for n in names if n in ctx.layout]
    ctx.roof_name = ctx.rng.choice(names)
    return ctx.strip(ctx.roof_name)


def _scheme_key(ctx, prefix, keys):
    return ctx.rng.choice([k for k in keys if f"{k}_upper" in ctx.layout])


def build_pitched_house(L, W, F, ctx, hip):
    """gable / hip archetypes."""
    m = ctx.massing
    rng = ctx.rng
    fac = ctx.facade(_scheme_key(ctx, "s", ("s1", "s2", "s3")))
    roof_strip = _roof_choice(ctx)
    pitch = rng.uniform(*(m["hip_pitch"] if hip else m["gable_pitch"]))
    o = min(rng.uniform(*m["overhang"]), 0.12 * W)
    og = o * 0.8 if not hip else o
    he = F * FLOOR_H
    x0, x1, y0, y1 = -L / 2, L / 2, -W / 2, W / 2
    # porch / veranda: pull the front wall back, keep the roof outline
    add_porch = None
    if F <= 2 and W >= 7.0:
        r = rng.random()
        if r < m.get("porch_p", 0):
            add_porch = "porch"
        elif r < m.get("porch_p", 0) + m.get("veranda_p", 0):
            add_porch = "veranda"
    porch_d = 1.9 if add_porch else 0.0
    cls = Hip if hip else Gable
    roof = cls(x0, x1, y0 + porch_d, y1, he, pitch, o, og)
    dormers = (not hip and pitch >= 38 and W >= 8 and L >= 7 and
               rng.random() < m.get("dormer_p", 0))
    has_chimney = rng.random() < m.get("chimney_p", 0)
    u_roof = rng.random()
    bands = []
    for level in (1, 2, 3, 4):
        set_level(level)
        mesh = Mesh()
        ring = roof.wall_ring()
        if level <= 2:
            ring_walls(mesh, fac, ring, 0.0, F, front_edge=0)
            if level == 1 or not hip:
                roof.gables(mesh, fac.plain, random.Random(7))
            if level == 1:
                roof.roof(mesh, roof_strip, u_roof)
                roof.trims(mesh, ctx)
                if has_chimney:
                    chimney(mesh, ctx, roof, fac, random.Random(11))
                if dormers:
                    wr = roof.wall_ring()
                    span = wr[1][0] - wr[0][0]
                    nd = 1 if span < 10 else 2
                    for i in range(nd):
                        xc = wr[0][0] + span * (i + 1) / (nd + 1)
                        dormer(mesh, ctx, roof, fac, random.Random(13 + i),
                               xc, 1.6)
                if add_porch:
                    px0, px1 = (ring[0][0], ring[1][0])
                    if add_porch == "porch":
                        pw = min(px1 - px0, max(3.5, 0.55 * (px1 - px0)))
                        pc = rng.uniform(px0 + pw / 2, px1 - pw / 2)
                        px0, px1 = pc - pw / 2, pc + pw / 2
                    else:
                        px0, px1 = x0 + 0.05, x1 - 0.05
                    porch(mesh, ctx, px0, px1, y0, ring[0][1],
                          min(he, FLOOR_H) - 0.15, fac, random.Random(17))
            else:
                # shell: bare roof planes out to the footprint
                roof.roof(mesh, roof_strip, u_roof)
                if add_porch:
                    rs = roof_strip
                    ze = min(he, FLOOR_H) - 0.15 - porch_d * 0.21
                    roof_face(mesh, [(x0, y0, ze), (x1, y0, ze),
                                     (x1, ring[0][1], ze + porch_d * 0.21),
                                     (x0, ring[0][1], ze + porch_d * 0.21)],
                              (0, 1, 0.21), rs, 0.3)
        elif level == 3:
            ring_plain(mesh, fac.plain, ring, 0.0, he, random.Random(3))
            roof.gables(mesh, fac.plain, random.Random(7))
            roof.roof(mesh, roof_strip, u_roof)
        else:
            full = cls(x0, x1, y0, y1, he, pitch, o, og)
            full.quad(mesh, roof_strip)
        bands.append(mesh)
    return bands


def build_lshape(L, W, F, ctx):
    m = ctx.massing
    rng = ctx.rng
    fac = ctx.facade(_scheme_key(ctx, "s", ("s1", "s2", "s3")))
    roof_strip = _roof_choice(ctx)
    pitch = rng.uniform(*m["gable_pitch"])
    o = min(rng.uniform(*m["overhang"]), 0.1 * W)
    he = F * FLOOR_H
    a = W * rng.uniform(0.5, 0.6)
    b = L * rng.uniform(0.35, 0.45)
    main = Gable(-L / 2, L / 2, -W / 2, -W / 2 + a, he, pitch, o, o * 0.8)
    # wing: ridge along y -> build as a Gable in a rotated frame
    # wing frame: world (x, y) = (y', -x'), a proper rotation
    wing = Gable(-W / 2, W / 2 - a + 1.0, L / 2 - b, L / 2, he, pitch, o,
                 o * 0.8)
    bands = []
    for level in (1, 2, 3, 4):
        set_level(level)
        mesh = Mesh()
        # main block
        ring = main.wall_ring()
        wmesh = Mesh()
        wring = wing.wall_ring()
        if level <= 2:
            ring_walls(mesh, fac, ring, 0.0, F, front_edge=0)
            ring_walls(wmesh, fac, wring, 0.0, F, front_edge=2)
            main.gables(mesh, fac.plain, random.Random(7))
            wing.gables(wmesh, fac.plain, random.Random(8))
            main.roof(mesh, roof_strip, 0.1)
            wing.roof(wmesh, roof_strip, 0.6)
            if level == 1:
                main.trims(mesh, ctx)
                wing.trims(wmesh, ctx)
                if rng.random() < m.get("chimney_p", 0):
                    chimney(mesh, ctx, main, fac, random.Random(11))
        elif level == 3:
            ring_plain(mesh, fac.plain, ring, 0.0, he, random.Random(3))
            ring_plain(wmesh, fac.plain, wring, 0.0, he, random.Random(4))
            main.roof(mesh, roof_strip, 0.1)
            wing.roof(wmesh, roof_strip, 0.6)
        else:
            far_quad(mesh, roof_strip, -L / 2, -W / 2, L / 2, W / 2,
                     (main.ze + main.ridge) / 2)
        # rotate the wing frame (x', y') = (y, -x) -> world
        for v in wmesh.verts:
            x, y, z, nx, ny, nz, u, vv = v
            mesh.verts.append((y, -x, z, ny, -nx, nz, u, vv))
        base = len(mesh.verts) - len(wmesh.verts)
        mesh.tris.extend(i + base for i in wmesh.tris)
        bands.append(mesh)
    return bands


def build_rowhouse(L, W, F, ctx):
    """Terrace: continuous gable roof, per-unit facades and party-wall
    chimneys."""
    m = ctx.massing
    rng = ctx.rng
    keys = [k for k in ("s1", "s2", "s3") if f"{k}_upper" in ctx.layout]
    facs = [ctx.facade(k) for k in rng.sample(keys, 2)]
    roof_strip = _roof_choice(ctx)
    pitch = rng.uniform(*m["gable_pitch"])
    o = min(0.3, 0.08 * W)
    he = F * FLOOR_H
    roof = Gable(-L / 2, L / 2, -W / 2, W / 2, he, pitch, o, 0.05)
    ring = roof.wall_ring()
    wx0, wy0 = ring[0]
    wx1, wy1 = ring[2]
    n_units = max(2, int(round((wx1 - wx0) / rng.uniform(5.0, 6.5))))
    uw = (wx1 - wx0) / n_units
    bands = []
    for level in (1, 2, 3, 4):
        set_level(level)
        mesh = Mesh()
        if level == 2:
            ring_walls(mesh, facs[0], ring, 0.0, F, front_edge=0)
            roof.gables(mesh, facs[0].plain, random.Random(7))
            roof.roof(mesh, roof_strip, 0.2)
        elif level == 1:
            for i in range(n_units):
                fac = facs[i % 2]
                a = wx0 + i * uw
                b = a + uw
                facade_wall(mesh, fac, (a, wy0), (b, wy0), 0.0, F,
                            door_side=True)
                facade_wall(mesh, fac, (b, wy1), (a, wy1), 0.0, F)
            facade_wall(mesh, facs[0], (wx1, wy0), (wx1, wy1), 0.0, F)
            facade_wall(mesh, facs[1], (wx0, wy1), (wx0, wy0), 0.0, F)
            roof.gables(mesh, facs[0].plain, random.Random(7))
            roof.roof(mesh, roof_strip, 0.2)
            if level == 1:
                roof.trims(mesh, ctx)
                if rng.random() < max(0.3, m.get("chimney_p", 0)):
                    for i in range(1, n_units):
                        x = wx0 + i * uw - 0.3
                        box_part(mesh, facs[0].plain, x, x + 0.6,
                                 roof.yc - 0.35, roof.yc + 0.35,
                                 roof.ridge - 0.4, roof.ridge + 0.75,
                                 top_strip=ctx.trim["coping"])
        elif level == 3:
            ring_plain(mesh, facs[0].plain, ring, 0.0, he, random.Random(3))
            roof.gables(mesh, facs[0].plain, random.Random(7))
            roof.roof(mesh, roof_strip, 0.2)
        else:
            roof.quad(mesh, roof_strip)
        bands.append(mesh)
    return bands


def build_flat(L, W, F, ctx, *, keys, front_inset=0.0, canopy_z=None,
               parapet=0.8, roof_names=("roof_flat",), arcade=False):
    """Flat-roofed box with parapet: flatres, aptslab, flatcom, shophouse,
    bigbox.  ``front_inset`` pulls the front wall back (canopy/arcade)."""
    rng = ctx.rng
    fac = ctx.facade(_scheme_key(ctx, "", keys))
    roof_strip = _roof_choice(ctx, roof_names)
    he = F * FLOOR_H
    top = he + parapet
    x0, x1, y0, y1 = -L / 2, L / 2, -W / 2, W / 2
    bands = []
    for level in (1, 2, 3, 4):
        set_level(level)
        mesh = Mesh()
        if level == 1 and arcade and F >= 2:
            # five-foot way: ground floor pulled back, upper floors flush
            yi = y0 + front_inset
            facade_wall(mesh, fac, (x0, yi), (x1, yi), 0.0, 1,
                        door_side=True)
            for a, b in (((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)),
                         ((x0, y1), (x0, y0))):
                facade_wall(mesh, fac, a, b, 0.0, F)
            facade_wall(mesh, fac, (x0, y0), (x1, y0), FLOOR_H, F - 1,
                        strip_ground=fac.upper)
            sof = ctx.trim["soffit"]
            pts = [(x1, yi, FLOOR_H), (x0, yi, FLOOR_H), (x0, y0, FLOOR_H),
                   (x1, y0, FLOOR_H)]
            if poly_normal(pts)[2] > 0:
                pts = pts[::-1]
            mesh.planar(pts, sof, (1, 0, 0), v_scale=0.4 / front_inset)
            n = max(2, int(round(L / 4.5)) + 1)
            s = 0.4
            for i in range(n):
                px = x0 + (L - s) * i / (n - 1)
                box_part(mesh, fac.plain, px, px + s, y0, y0 + s, 0.0,
                         FLOOR_H, sides=(True, True, False, True), top=False)
            # the end walls of the arcade
            for xe, a, b in ((x1, (x1, y0), (x1, yi)),
                             (x0, (x0, yi), (x0, y0))):
                pass
        elif level <= 2:
            yi = y0 + (front_inset if level == 1 else 0.0)
            ring = rect_ring(x0, yi, x1, y1)
            ring_walls(mesh, fac, ring, 0.0, F, front_edge=0)
            if level == 1 and front_inset > 0 and canopy_z is not None:
                canopy(mesh, ctx, x0, x1, y0, yi, canopy_z)
        if level <= 2:
            ring = rect_ring(x0, y0, x1, y1)
            if level == 1 and front_inset and not arcade:
                ring = rect_ring(x0, y0 + front_inset, x1, y1)
            ring_plain(mesh, fac.plain, ring, he, top, random.Random(5))
            if level == 1 and front_inset and not arcade:
                # roof field only over the building; canopy covers the rest
                flat_top(mesh, ctx, roof_strip, x0, y0 + front_inset, x1, y1,
                         top)
            else:
                flat_top(mesh, ctx, roof_strip, x0, y0, x1, y1, top,
                         coping=0.3 if level == 1 else 0.0)
        elif level == 3:
            ring = rect_ring(x0, y0, x1, y1)
            ring_plain(mesh, fac.plain, ring, 0.0, top, random.Random(3))
            flat_top(mesh, ctx, roof_strip, x0, y0, x1, y1, top, coping=0)
        else:
            far_quad(mesh, roof_strip, x0, y0, x1, y1, top)
        bands.append(mesh)
    return bands


def build_apartment(L, W, F, ctx, *, prefer_pitched):
    m = ctx.massing
    rng = ctx.rng
    pitched = rng.random() < (m.get("apt_pitched_p", 0) if prefer_pitched
                              else m.get("apt_pitched_p", 0) * 0.5)
    if not pitched or W > 22:
        return build_flat(L, W, F, ctx, keys=("a1", "a2"),
                          roof_names=("roof_flat", "roof_flat_b"))
    fac = ctx.facade(_scheme_key(ctx, "", ("a1", "a2")))
    ctx.roof_name = "roof_a"
    roof_strip = ctx.strip("roof_a")
    he = F * FLOOR_H
    o = min(0.45, 0.05 * W)
    roof = Hip(-L / 2, L / 2, -W / 2, W / 2, he,
               rng.uniform(*m["hip_pitch"]), o, o, max_rise=6.0)
    bands = []
    for level in (1, 2, 3, 4):
        set_level(level)
        mesh = Mesh()
        ring = roof.wall_ring()
        if level <= 2:
            ring_walls(mesh, fac, ring, 0.0, F, front_edge=0)
            roof.roof(mesh, roof_strip, 0.3)
            if level == 1:
                roof.trims(mesh, ctx)
        elif level == 3:
            ring_plain(mesh, fac.plain, ring, 0.0, he, random.Random(3))
            roof.roof(mesh, roof_strip, 0.3)
        else:
            roof.quad(mesh, roof_strip)
        bands.append(mesh)
    return bands


def build_warehouse(L, W, F, ctx):
    rng = ctx.rng
    fac = ctx.facade(_scheme_key(ctx, "", ("i1", "i2", "i3")))
    roof_strip = _roof_choice(ctx, ("roof_a", "roof_b"))
    he = F * FLOOR_H
    o = 0.35
    roof = Gable(-L / 2, L / 2, -W / 2, W / 2, he, rng.uniform(5, 9), o, o,
                 max_rise=6.0)
    bands = []
    for level in (1, 2, 3, 4):
        set_level(level)
        mesh = Mesh()
        ring = roof.wall_ring()
        if level <= 2:
            ring_walls(mesh, fac, ring, 0.0, F, front_edge=0)
            roof.gables(mesh, fac.plain, random.Random(7))
            roof.roof(mesh, roof_strip, 0.1)
            if level == 1:
                roof.trims(mesh, ctx)
        elif level == 3:
            ring_plain(mesh, fac.plain, ring, 0.0, he, random.Random(3))
            roof.gables(mesh, fac.plain, random.Random(7))
            roof.roof(mesh, roof_strip, 0.1)
        else:
            roof.quad(mesh, roof_strip)
        bands.append(mesh)
    return bands


def build(archetype, L, W, F, seed, region, force=None):
    """-> (page, [mesh_band1..4]).  ``force`` overrides massing keys
    (review renders use it to show a region's signature features)."""
    ctx = Ctx(archetype, region, seed, force)
    m = ctx.massing
    if archetype in ("gable", "hip"):
        hip = archetype == "hip"
        if not hip and ctx.rng.random() < m.get("hip_p", 0) * 0.3:
            hip = True
        bands = build_pitched_house(L, W, F, ctx, hip)
    elif archetype == "lshape":
        bands = build_lshape(L, W, F, ctx)
    elif archetype == "rowhouse":
        bands = build_rowhouse(L, W, F, ctx)
    elif archetype == "flatres":
        bands = build_flat(L, W, F, ctx, keys=("s1", "s2", "s3"),
                           parapet=0.7)
    elif archetype in ("aptslab", "aptblock"):
        bands = build_apartment(L, W, F, ctx,
                                prefer_pitched=archetype == "aptblock")
    elif archetype == "flatcom":
        inset = 1.4 if (W > 9 and ctx.rng.random() < m.get("canopy_p", 0.5)) \
            else 0.0
        bands = build_flat(L, W, F, ctx, keys=("c1", "c2"),
                           front_inset=inset, canopy_z=3.05,
                           roof_names=("roof_flat", "roof_flat_b"))
    elif archetype == "shophouse":
        arcade = m.get("shophouse_arcade", False) and F >= 2 and W > 6
        bands = build_flat(L, W, F, ctx, keys=("c1", "c2"),
                           front_inset=1.8 if arcade else 0.0,
                           canopy_z=None, arcade=arcade,
                           roof_names=("roof_flat", "roof_flat_b"))
    elif archetype == "warehouse":
        bands = build_warehouse(L, W, F, ctx)
    elif archetype == "bigbox":
        bands = build_flat(L, W, F, ctx, keys=("i3", "i4", "i1"),
                           parapet=1.0, roof_names=("roof_flat",),
                           front_inset=2.5 if W > 25 else 0.0,
                           canopy_z=4.2 if F >= 2 else None)
    else:
        raise ValueError(archetype)
    return ctx.page, bands
