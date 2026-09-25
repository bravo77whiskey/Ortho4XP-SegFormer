"""Low-poly mesh accumulation, atlas UV mapping and the OBJ8 writer.

Building space: x = length, y = width, z = up, footprint centred on the
origin (the overlay measures OBJ bounds and requires them centred and
equal to the declared footprint).  Faces are CCW seen from outside; the
writer converts to X-Plane (x, z-up -> y-up, -y -> z) and reverses the
winding, because OBJ8 front faces are CLOCKWISE (verified in-sim for the
old library).
"""

from __future__ import annotations

import math

V_INSET_PX = 1.5
PAGE_PX = 4096


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a):
    n = math.sqrt(_dot(a, a))
    return (a[0] / n, a[1] / n, a[2] / n) if n > 1e-12 else (0.0, 0.0, 1.0)


def poly_normal(pts):
    n = (0.0, 0.0, 0.0)
    for i in range(1, len(pts) - 1):
        c = _cross(_sub(pts[i], pts[0]), _sub(pts[i + 1], pts[0]))
        n = (n[0] + c[0], n[1] + c[1], n[2] + c[2])
    return _norm(n)


class Strip:
    """A v-band of an atlas page: (v0, v1, world_w, world_h)."""

    __slots__ = ("v0", "v1", "world_w", "world_h", "name")

    def __init__(self, row, name=""):
        inset = V_INSET_PX / PAGE_PX
        self.v0 = row["v0"] + inset
        self.v1 = row["v1"] - inset
        self.world_w = row["world_w"]
        self.world_h = row["world_h"]
        self.name = name

    def v(self, h_m):
        """v for a height (metres) above the strip bottom, clamped."""
        t = max(0.0, min(1.0, h_m / self.world_h))
        return self.v0 + (self.v1 - self.v0) * t


class Mesh:
    def __init__(self):
        self.verts = []   # (x, y, z, nx, ny, nz, u, v)
        self.tris = []

    def tri_count(self):
        return len(self.tris) // 3

    def poly(self, pts, uvs):
        if len(pts) < 3:
            return
        n = poly_normal(pts)
        base = len(self.verts)
        for p, (u, v) in zip(pts, uvs):
            self.verts.append((p[0], p[1], p[2], n[0], n[1], n[2], u, v))
        for i in range(1, len(pts) - 1):
            self.tris.extend((base, base + i, base + i + 1))

    def planar(self, pts, strip, up, *, u0=0.0, u_scale=1.0, origin=None,
               v_off=0.0, v_scale=1.0):
        """Map a planar polygon onto a strip: v along ``up`` (in-plane),
        u to the right as seen from outside.  ``u_scale`` stretches world
        distance (bay snapping); ``v_off`` shifts where the strip starts."""
        n = poly_normal(pts)
        up = _norm(up)
        right = _norm(_cross(up, n))
        o = origin or pts[0]
        uvs = []
        for p in pts:
            d = _sub(p, o)
            s = _dot(d, right) * u_scale
            t = _dot(d, up) * v_scale + v_off
            uvs.append((u0 + s / strip.world_w, strip.v(t)))
        self.poly(pts, uvs)

    def bounds(self):
        xs = [v[0] for v in self.verts]
        ys = [v[1] for v in self.verts]
        zs = [v[2] for v in self.verts]
        return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)


def write_obj8(path, bands, texture, normal_tex, *, specular=1.0):
    """bands = [(Mesh, near_m, far_m), ...] contiguous from 0."""
    verts = []
    idx = []
    ranges = []
    for mesh, near, far in bands:
        base = len(verts)
        verts.extend(mesh.verts)
        start = len(idx)
        idx.extend(i + base for i in mesh.tris)
        ranges.append((near, far, start, len(idx) - start))
    lines = ["I", "800", "OBJ", "",
             f"TEXTURE {texture}",
             f"TEXTURE_NORMAL {normal_tex}",
             "NORMAL_METALNESS",
             f"GLOBAL_specular {specular:.2f}",
             f"POINT_COUNTS {len(verts)} 0 0 {len(idx)}", ""]
    for x, y, z, nx, ny, nz, u, v in verts:
        lines.append(f"VT {x:.3f} {z:.3f} {-y:.3f} {nx:.3f} {nz:.3f} "
                     f"{-ny:.3f} {u:.5f} {v:.5f}")
    lines.append("")
    # reverse each triangle (CCW -> OBJ8 clockwise front)
    flipped = []
    for i in range(0, len(idx), 3):
        flipped.extend((idx[i], idx[i + 2], idx[i + 1]))
    for i in range(0, len(flipped) - len(flipped) % 10, 10):
        lines.append("IDX10 " + " ".join(str(k) for k in flipped[i:i + 10]))
    for k in flipped[len(flipped) - len(flipped) % 10:]:
        lines.append(f"IDX {k}")
    lines.append("")
    for near, far, start, count in ranges:
        lines.append(f"ATTR_LOD {int(near)} {int(far)}")
        if count:
            lines.append(f"TRIS {start} {count}")
    lines.append("")
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write("\n".join(lines))
