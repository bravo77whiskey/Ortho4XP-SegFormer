"""Shared mesh primitives for procedural building archetypes.

Pure Python (no bpy) so archetypes are unit-testable outside Blender.
Coordinates are Blender conventions: X = building length, Y = width,
Z = up, base at z=0, footprint centered on the origin.  xplane2blender
maps this to X-Plane's X/Z footprint, which is what the building-overlay
pool measures from OBJ8 ``VT`` lines -- so footprint bounds here MUST be
exactly (+-length/2, +-width/2).

Faces are quads or tris with outward CCW winding (right-hand rule normal)
and carry their own per-loop UVs; vertices are not shared across faces,
which keeps every edge hard (correct for buildings) and the Blender
adapter trivial.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

TRI_BUDGET = 1500


def fmt_dim(value: float) -> str:
    """Format a dimension for filenames: 8 -> '8', 11.20 -> '11.2'."""
    text = f"{float(value):.1f}"
    return text.rstrip("0").rstrip(".")


@dataclass
class MeshSpec:
    """A buildable mesh: flat lists of vertices, faces and per-loop UVs."""

    vertices: list = field(default_factory=list)   # [(x, y, z), ...]
    faces: list = field(default_factory=list)      # [(i0, i1, i2[, i3]), ...]
    uvs: list = field(default_factory=list)        # per face: ((u,v), ...) per loop
    # Silhouette metadata for the far-LOD shell (set by every builder):
    # {"kind": "pitched"|"flat", "ridge_z", "eave_z",
    #  "wall_strip", "roof_strip"}
    meta: dict = field(default_factory=dict)

    def add_face(self, points, uv_points):
        if len(points) != len(uv_points) or len(points) not in (3, 4):
            raise ValueError("faces must be tris or quads with matching UVs")
        base = len(self.vertices)
        self.vertices.extend(tuple(float(c) for c in p) for p in points)
        self.faces.append(tuple(range(base, base + len(points))))
        self.uvs.append(tuple((float(u), float(v)) for u, v in uv_points))

    def add_quad(self, p0, p1, p2, p3, uv0, uv1, uv2, uv3):
        """Add one quad; pass corners CCW when viewed from outside."""
        self.add_face((p0, p1, p2, p3), (uv0, uv1, uv2, uv3))

    def add_tri(self, p0, p1, p2, uv0, uv1, uv2):
        self.add_face((p0, p1, p2), (uv0, uv1, uv2))

    def extend(self, other: "MeshSpec"):
        for face, uv in zip(other.faces, other.uvs):
            self.add_face([other.vertices[i] for i in face], uv)

    def tri_count(self) -> int:
        return sum(len(face) - 2 for face in self.faces)

    def bounds(self):
        """Return (xmin, xmax, ymin, ymax, zmin, zmax)."""
        if not self.vertices:
            raise ValueError("empty mesh")
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        zs = [v[2] for v in self.vertices]
        return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


class BudgetError(ValueError):
    """Raised when an archetype exceeds the per-asset triangle budget."""


def set_shell_meta(spec: "MeshSpec", kind: str, ridge_z: float,
                   eave_z: float, wall_strip: StripUV, roof_strip: StripUV):
    """Record the silhouette the far-LOD shell must reproduce."""
    spec.meta = {
        "kind": kind,
        "ridge_z": float(ridge_z),
        "eave_z": float(eave_z),
        "wall_strip": wall_strip.name,
        "roof_strip": roof_strip.name,
    }


def validate_spec(spec: MeshSpec, length_m: float, width_m: float,
                  tol: float = 0.005) -> MeshSpec:
    """Assert footprint exactness, centering and tri budget; return spec."""
    tris = spec.tri_count()
    if tris > TRI_BUDGET:
        raise BudgetError(f"{tris} tris exceeds budget {TRI_BUDGET}")
    xmin, xmax, ymin, ymax, zmin, zmax = spec.bounds()
    hx, hy = length_m / 2.0, width_m / 2.0
    for got, want, label in (
        (xmin, -hx, "xmin"), (xmax, hx, "xmax"),
        (ymin, -hy, "ymin"), (ymax, hy, "ymax"),
    ):
        if abs(got - want) > tol:
            raise ValueError(
                f"footprint {label}={got:.4f} != {want:.4f} "
                f"(asset must measure exactly {length_m}x{width_m})"
            )
    if abs(zmin) > tol:
        raise ValueError(f"base zmin={zmin:.4f} not grounded at 0")
    return spec


class StripUV:
    """UV helper for one atlas strip.

    Strips span the full atlas width and tile horizontally, so U is free
    (wrap-repeat) while V must stay inside the strip's band.  ``layout``
    rows come from atlas_layout.json:
      {"v0": low V, "v1": high V, "world_w_m": meters per full U repeat,
       "world_h_m": meters the band height represents}
    """

    def __init__(self, layout: dict, name: str):
        try:
            row = layout["strips"][name]
        except KeyError:
            raise KeyError(f"atlas strip {name!r} missing from layout") from None
        self.name = name
        self.v0 = float(row["v0"])
        self.v1 = float(row["v1"])
        self.world_w_m = float(row["world_w_m"])
        self.world_h_m = float(row["world_h_m"])

    def u(self, distance_m: float) -> float:
        return distance_m / self.world_w_m

    def v(self, frac: float) -> float:
        frac = min(1.0, max(0.0, frac))
        return self.v0 + (self.v1 - self.v0) * frac

    def quad_uvs(self, u0_m: float, u1_m: float, frac0: float = 0.0,
                 frac1: float = 1.0):
        """UVs for a quad whose loops run (bottom-left, bottom-right,
        top-right, top-left) in strip space."""
        return (
            (self.u(u0_m), self.v(frac0)),
            (self.u(u1_m), self.v(frac0)),
            (self.u(u1_m), self.v(frac1)),
            (self.u(u0_m), self.v(frac1)),
        )


def wall_quad(spec: MeshSpec, p_bl, p_br, z0: float, z1: float,
              strip: StripUV, u0_m: float = 0.0, u1_m: float = None,
              frac0: float = 0.0, frac1: float = 1.0):
    """Vertical wall quad from bottom edge (p_bl -> p_br) raised z0..z1.

    p_bl/p_br are (x, y) ground-plane points ordered so the outward normal
    follows CCW winding (left-to-right as seen from outside).
    """
    if u1_m is None:
        u1_m = math.dist(p_bl, p_br)
    uvs = strip.quad_uvs(u0_m, u1_m, frac0, frac1)
    spec.add_quad(
        (p_bl[0], p_bl[1], z0), (p_br[0], p_br[1], z0),
        (p_br[0], p_br[1], z1), (p_bl[0], p_bl[1], z1),
        *uvs,
    )


def floor_zs(floors: int, floor_h: float, ground_h: float = None) -> list:
    """Per-floor (z0, z1) spans totalling EXACTLY floors * floor_h.

    The total is the ``_LxWxF`` filename contract (pool height = F * 3.2);
    a combo's taller/shorter ground floor is absorbed by evenly squeezing
    the upper floors, so the building's declared height never drifts.
    """
    total = floors * floor_h
    if floors <= 1 or not ground_h or abs(ground_h - floor_h) < 1e-6:
        return [(i * floor_h, (i + 1) * floor_h) for i in range(floors)]
    # Never squeeze uppers below 70% of the nominal floor: a 2-floor
    # building with a 4.4 m lobby would otherwise get a crushed top row.
    max_ground = total - 0.7 * floor_h * (floors - 1)
    g = max(0.7 * floor_h, min(float(ground_h), max_ground))
    upper = (total - g) / (floors - 1)
    spans = [(0.0, g)]
    for i in range(floors - 1):
        spans.append((g + i * upper, g + (i + 1) * upper))
    return spans


def walls_with_floors(spec: MeshSpec, ring, z0: float, floors: int,
                      floor_h: float, ground_strip: StripUV,
                      upper_strip: StripUV, ground_h: float = None):
    """Extrude a CCW ground-plane ring into per-floor wall quads.

    ``ring`` is a list of (x, y) corners ordered CCW seen from above
    (so each edge's outward normal faces away from the interior).
    Floor 0 maps to ground_strip, floors 1..F-1 to upper_strip; a combo
    ``ground_h`` gives floor 0 its own height (see floor_zs).
    """
    n = len(ring)
    spans = floor_zs(floors, floor_h, ground_h)
    for i in range(n):
        a = ring[i]
        b = ring[(i + 1) % n]
        edge_len = math.dist(a, b)
        for floor, (fz0, fz1) in enumerate(spans):
            strip = ground_strip if floor == 0 else upper_strip
            wall_quad(
                spec, a, b, z0 + fz0, z0 + fz1,
                strip, 0.0, edge_len,
            )


def parapet_flat_roof(spec: MeshSpec, hx: float, hy: float, top_z: float,
                      parapet_h: float, thickness: float, band_strip: StripUV,
                      cap_strip: StripUV, roof_strip: StripUV,
                      roof_drop: float = 0.35, roof_u_m: float = None):
    """Parapet band + ONE full-footprint roof quad at the parapet top.

    Walls below ``top_z`` are the caller's job; this closes the building
    like the top of a simple cuboid (user request 2026-07-08): the old
    mitered cap ring + sunken membrane construction produced see-through
    roof artifacts in-sim, so the roof is now a single quad at parapet_z
    spanning the exact footprint.  ``thickness``/``roof_drop``/
    ``cap_strip`` stay in the signature for call-site compatibility but
    are unused.

    The roof quad deliberately does NOT tile: capping the U/V span kills
    the texture-coordinate derivative that otherwise pushes big roofs into
    deep mip levels at distance, where the strip atlas collapses and
    neighboring strips bleed through as rainbow rings.
    """
    parapet_z = top_z + parapet_h
    ring = ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))
    for i in range(4):
        a, b = ring[i], ring[(i + 1) % 4]
        wall_quad(spec, a, b, top_z, parapet_z, band_strip)
    u_roof = min((2 * hx) / roof_strip.world_w_m, 0.6)
    v_roof = min(2 * hy / roof_strip.world_h_m, 0.6)
    spec.add_quad(
        (-hx, -hy, parapet_z), (hx, -hy, parapet_z),
        (hx, hy, parapet_z), (-hx, hy, parapet_z),
        (roof_strip.u(0.0), roof_strip.v(0.0)),
        (u_roof, roof_strip.v(0.0)),
        (u_roof, roof_strip.v(v_roof)),
        (roof_strip.u(0.0), roof_strip.v(v_roof)),
    )
    return parapet_z


def box(spec: MeshSpec, x0, x1, y0, y1, z0, z1, strip: StripUV,
        top_strip: StripUV = None, bottom: bool = False):
    """Axis-aligned box: 4 side quads + top (+ optional bottom)."""
    ring = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    n = len(ring)
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        wall_quad(spec, a, b, z0, z1, strip)
    ts = top_strip or strip
    spec.add_quad(
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
        *ts.quad_uvs(0.0, x1 - x0),
    )
    if bottom:
        spec.add_quad(
            (x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0),
            *strip.quad_uvs(0.0, x1 - x0),
        )
