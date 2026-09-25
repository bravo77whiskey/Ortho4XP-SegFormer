"""Pure-Python geometry accumulator for the high-poly bake scenes.

Every atlas strip is modelled as a vertical relief facing the bake camera:
x runs along the strip (texture u, periodic with the strip's world width W),
z runs up the strip (texture v), the base surface sits at y = 0, anything
that protrudes toward the camera has y < 0 and recesses have y > 0.  Roof
coverings use the same frame (z = up-slope), so one camera setup, one
tangent frame and one "gravity = -z" rule serve walls and roofs alike.

Faces are grouped per material and carry two per-face random values
(``rnd``/``rnd2``) that the Blender materials read as attributes: that is how
a thousand tiles or curtains get individual colours from one material.
"""

from __future__ import annotations

import math
from collections import defaultdict


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def face_normal(pts):
    n = (0.0, 0.0, 0.0)
    for i in range(1, len(pts) - 1):
        c = _cross(_sub(pts[i], pts[0]), _sub(pts[i + 1], pts[0]))
        n = (n[0] + c[0], n[1] + c[1], n[2] + c[2])
    return n


class Geo:
    """Faces per material, each with (rnd, rnd2) attributes."""

    def __init__(self):
        self.parts = defaultdict(lambda: {"v": [], "f": [], "r": [], "r2": []})

    def face(self, mat, pts, rnd=0.0, rnd2=0.0, facing=None):
        """Add a polygon; if ``facing`` is given, orient it to face there."""
        pts = [tuple(float(c) for c in p) for p in pts]
        if facing is not None and _dot(face_normal(pts), facing) < 0:
            pts = pts[::-1]
        part = self.parts[mat]
        base = len(part["v"])
        part["v"].extend(pts)
        part["f"].append(tuple(range(base, base + len(pts))))
        part["r"].append(float(rnd))
        part["r2"].append(float(rnd2))

    def rect_front(self, mat, x0, x1, z0, z1, y=0.0, rnd=0.0, rnd2=0.0):
        """Camera-facing rectangle (normal -y) in the plane y."""
        if x1 - x0 <= 1e-6 or z1 - z0 <= 1e-6:
            return
        self.face(mat, [(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)],
                  rnd, rnd2, facing=(0, -1, 0))

    def box(self, mat, x0, x1, y0, y1, z0, z1, rnd=0.0, rnd2=0.0,
            back=False, bottom=True, top=True, sides=True, bevel=None):
        """Axis-aligned box; the back (+y) face is skipped by default.

        The camera-facing edges are chamfered (``bevel`` metres, default
        scaled to the part): an orthographic bake only sees faces with a
        -y component, so square edges would vanish from the normal map
        while a chamfer reads as the lit/shaded rim a real profile has.
        """
        if x1 - x0 <= 1e-6 or y1 - y0 <= 1e-6 or z1 - z0 <= 1e-6:
            return
        if bevel is None:
            bevel = min(0.015, 0.3 * min(x1 - x0, y1 - y0, z1 - z0))
        b = max(0.0, min(bevel, 0.45 * min(x1 - x0, z1 - z0), y1 - y0))
        f = self.face
        ys = y0 + b
        if b > 1e-5:
            f(mat, [(x0 + b, y0, z0 + b), (x1 - b, y0, z0 + b),
                    (x1 - b, y0, z1 - b), (x0 + b, y0, z1 - b)],
              rnd, rnd2, facing=(0, -1, 0))
            f(mat, [(x0 + b, y0, z0 + b), (x1 - b, y0, z0 + b),
                    (x1, ys, z0), (x0, ys, z0)], rnd, rnd2,
              facing=(0, -1, -1))
            f(mat, [(x0 + b, y0, z1 - b), (x1 - b, y0, z1 - b),
                    (x1, ys, z1), (x0, ys, z1)], rnd, rnd2,
              facing=(0, -1, 1))
            f(mat, [(x0 + b, y0, z0 + b), (x0 + b, y0, z1 - b),
                    (x0, ys, z1), (x0, ys, z0)], rnd, rnd2,
              facing=(-1, -1, 0))
            f(mat, [(x1 - b, y0, z0 + b), (x1 - b, y0, z1 - b),
                    (x1, ys, z1), (x1, ys, z0)], rnd, rnd2,
              facing=(1, -1, 0))
        else:
            f(mat, [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
              rnd, rnd2, facing=(0, -1, 0))
        if back:
            f(mat, [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
              rnd, rnd2, facing=(0, 1, 0))
        if top:
            f(mat, [(x0, ys, z1), (x1, ys, z1), (x1, y1, z1), (x0, y1, z1)],
              rnd, rnd2, facing=(0, 0, 1))
        if bottom:
            f(mat, [(x0, ys, z0), (x1, ys, z0), (x1, y1, z0), (x0, y1, z0)],
              rnd, rnd2, facing=(0, 0, -1))
        if sides:
            f(mat, [(x0, ys, z0), (x0, y1, z0), (x0, y1, z1), (x0, ys, z1)],
              rnd, rnd2, facing=(-1, 0, 0))
            f(mat, [(x1, ys, z0), (x1, y1, z0), (x1, y1, z1), (x1, ys, z1)],
              rnd, rnd2, facing=(1, 0, 0))

    def tilted_box(self, mat, x0, x1, z0, z1, y_bottom, y_top, thick,
                   rnd=0.0, rnd2=0.0):
        """A plate lying on the relief with its lower edge lifted.

        Used for tiles/slates/shingles: the plate spans x0..x1 and z0..z1
        (up-slope), its exposed face is at y_bottom at z0 and y_top at z1
        (y_bottom < y_top means the lower edge sits proud of the one above
        it, as a lapped roof covering does).
        """
        a = (x0, y_bottom, z0)
        b = (x1, y_bottom, z0)
        c = (x1, y_top, z1)
        d = (x0, y_top, z1)
        self.face(mat, [a, b, c, d], rnd, rnd2, facing=(0, -1, 0.0001))
        # lower edge thickness (the visible butt of the tile), chamfered so
        # the orthographic bake sees it as a shaded course line
        a2 = (x0, y_bottom + thick, z0 - thick * 0.9)
        b2 = (x1, y_bottom + thick, z0 - thick * 0.9)
        self.face(mat, [a, b, b2, a2], rnd, rnd2, facing=(0, 0, -1))
        # side edges
        c2 = (x1, y_top + thick, z1)
        d2 = (x0, y_top + thick, z1)
        self.face(mat, [a, a2, d2, d], rnd, rnd2, facing=(-1, 0, 0))
        self.face(mat, [b, c, c2, b2], rnd, rnd2, facing=(1, 0, 0))

    def profile_sweep(self, mat, xs_ys, x_off, z0, z1, y_off0, y_off1,
                      rnd=0.0, rnd2=0.0, closed_ends=True, lift=0.0,
                      nose=0.0):
        """Sweep a cross-section (list of (x, y)) along z from z0 to z1.

        ``y_off0``/``y_off1`` shift the section toward the camera at the
        two ends (so a tile can tilt), ``lift`` adds a butt face depth.
        The section must run left to right with y measured toward the
        camera as negative.
        """
        n = len(xs_ys)
        lo = [(x_off + x, y + y_off0, z0) for x, y in xs_ys]
        hi = [(x_off + x, y + y_off1, z1) for x, y in xs_ys]
        for i in range(n - 1):
            self.face(mat, [lo[i], lo[i + 1], hi[i + 1], hi[i]], rnd, rnd2,
                      facing=(0, -1, 0.0))
        if closed_ends and lift > 0 and nose > 0:
            # rounded nose: the butt rolls back under the tile over
            # ``nose`` metres, so the bake sees a downward-facing band
            # (the course line a low sun picks out on a real roof)
            mid = [(p[0], p[1] + lift * 0.35, p[2] - nose * 0.6) for p in lo]
            back = [(p[0], p[1] + lift, p[2] - nose) for p in lo]
            for ring_a, ring_b in ((lo, mid), (mid, back)):
                for i in range(n - 1):
                    self.face(mat, [ring_a[i], ring_a[i + 1], ring_b[i + 1],
                                    ring_b[i]], rnd, rnd2,
                              facing=(0, -0.3, -1))
        elif closed_ends and lift > 0:
            # butt face: from the section down to the section pushed back
            back = [(p[0], p[1] + lift, p[2]) for p in lo]
            for i in range(n - 1):
                self.face(mat, [lo[i], lo[i + 1], back[i + 1], back[i]],
                          rnd, rnd2, facing=(0, 0, -1))

    def merge(self, other: "Geo"):
        for mat, part in other.parts.items():
            dst = self.parts[mat]
            base = len(dst["v"])
            dst["v"].extend(part["v"])
            dst["f"].extend(tuple(i + base for i in f) for f in part["f"])
            dst["r"].extend(part["r"])
            dst["r2"].extend(part["r2"])

    def to_json(self):
        return {mat: {"v": part["v"], "f": part["f"], "r": part["r"],
                      "r2": part["r2"]}
                for mat, part in self.parts.items()}

    def face_count(self):
        return sum(len(p["f"]) for p in self.parts.values())


def wall_with_openings(geo: Geo, mat, x0, x1, z0, z1, openings, y=0.0,
                       rnd=0.0):
    """Front wall rectangle with rectangular holes (grid decomposition).

    ``openings`` = list of (ox0, ox1, oz0, oz1); holes may touch but not
    overlap.  Cells are merged per row to keep the face count modest.
    """
    xs = {x0, x1}
    zs = {z0, z1}
    for ox0, ox1, oz0, oz1 in openings:
        xs.update((max(x0, min(x1, ox0)), max(x0, min(x1, ox1))))
        zs.update((max(z0, min(z1, oz0)), max(z0, min(z1, oz1))))
    xs = sorted(xs)
    zs = sorted(zs)

    def inside(cx, cz):
        for ox0, ox1, oz0, oz1 in openings:
            if ox0 < cx < ox1 and oz0 < cz < oz1:
                return True
        return False

    for zi in range(len(zs) - 1):
        za, zb = zs[zi], zs[zi + 1]
        if zb - za < 1e-6:
            continue
        run_start = None
        for xi in range(len(xs) - 1):
            xa, xb = xs[xi], xs[xi + 1]
            solid = xb - xa > 1e-6 and not inside((xa + xb) / 2, (za + zb) / 2)
            if solid and run_start is None:
                run_start = xa
            if not solid and run_start is not None:
                geo.rect_front(mat, run_start, xa, za, zb, y=y, rnd=rnd)
                run_start = None
        if run_start is not None:
            geo.rect_front(mat, run_start, xs[-1], za, zb, y=y, rnd=rnd)


def reveal(geo: Geo, mat, x0, x1, z0, z1, depth, y=0.0, rnd=0.0,
           bottom=True):
    """The four inner faces of an opening from the wall plane to ``depth``."""
    ya, yb = y, y + depth
    geo.face(mat, [(x0, ya, z0), (x0, yb, z0), (x0, yb, z1), (x0, ya, z1)],
             rnd, facing=(1, 0, 0))
    geo.face(mat, [(x1, ya, z0), (x1, yb, z0), (x1, yb, z1), (x1, ya, z1)],
             rnd, facing=(-1, 0, 0))
    geo.face(mat, [(x0, ya, z1), (x1, ya, z1), (x1, yb, z1), (x0, yb, z1)],
             rnd, facing=(0, 0, -1))
    if bottom:
        geo.face(mat, [(x0, ya, z0), (x1, ya, z0), (x1, yb, z0),
                       (x0, yb, z0)], rnd, facing=(0, 0, 1))


def circle_section(radius, start_deg, end_deg, segments):
    """Arc points (x, y) with y toward the camera negative (a convex tile)."""
    pts = []
    for i in range(segments + 1):
        a = math.radians(start_deg + (end_deg - start_deg) * i / segments)
        pts.append((radius * math.cos(a), -radius * math.sin(a)))
    return pts
