"""Far-LOD silhouette shells merged into generated OBJ8s as a second band.

X-Plane autogen sustains 100k+ object instances per tile by carrying a cheap
far mesh inside each OBJ. We do the same: band 1 (0..swap) is the full
Blender-exported model; band 2 (swap..far) is an ~8-face shell synthesized
here in pure Python -- same silhouette, same atlas strips, so the swap is
invisible at distance. Beyond ``far`` the object culls. ATTR_LOD is the one
directive that keeps objects instancing-eligible.

The shell is built from the archetype's MeshSpec bounds + roof family:
  * pitched (gable/hip/lshape/rowhouse): box walls + 2 roof slopes + 2 gable
    tris under the SAME ridge height as the real model;
  * flat (flatres/shophouse/flatcom/bigbox/aptslab): box walls + flat top at
    the parapet height;
  * aptblock/warehouse: pitched with their shallow rises.

Wall UVs stretch one wall strip over the full height -- at 2 km nobody can
count floors. Coordinates are written in OBJ8 axes (Blender x,y,z ->
x, z, -y), matching what xplane2blender emitted for band 1.
"""

from __future__ import annotations

import math

FLOOR_H = 3.2


def shell_mesh(length_m: float, width_m: float, floors: int,
               kind: str, ridge_z: float, eave_z: float,
               wall_band: tuple, roof_band: tuple):
    """Return (vertices, faces): verts as (x, y, z, nx, ny, nz, u, v).

    Blender-convention coordinates (z up); the OBJ8 writer converts.
    ``wall_band``/``roof_band`` are (v0, v1, world_w_m) of atlas strips.
    """
    hx, hy = length_m / 2.0, width_m / 2.0
    wv0, wv1, wall_w = wall_band
    rv0, rv1, roof_w = roof_band
    verts = []
    faces = []

    def quad(p0, p1, p2, p3, uvs):
        # outward CCW; normal from first three points
        ax = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
        bx = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
        n = (
            ax[1] * bx[2] - ax[2] * bx[1],
            ax[2] * bx[0] - ax[0] * bx[2],
            ax[0] * bx[1] - ax[1] * bx[0],
        )
        mag = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2) or 1.0
        n = (n[0] / mag, n[1] / mag, n[2] / mag)
        base = len(verts)
        for p, uv in zip((p0, p1, p2, p3), uvs):
            verts.append((p[0], p[1], p[2], n[0], n[1], n[2], uv[0], uv[1]))
        faces.append((base, base + 1, base + 2))
        faces.append((base, base + 2, base + 3))

    def tri(p0, p1, p2, uvs):
        ax = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
        bx = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
        n = (
            ax[1] * bx[2] - ax[2] * bx[1],
            ax[2] * bx[0] - ax[0] * bx[2],
            ax[0] * bx[1] - ax[1] * bx[0],
        )
        mag = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2) or 1.0
        n = (n[0] / mag, n[1] / mag, n[2] / mag)
        base = len(verts)
        for p, uv in zip((p0, p1, p2), uvs):
            verts.append((p[0], p[1], p[2], n[0], n[1], n[2], uv[0], uv[1]))
        faces.append((base, base + 1, base + 2))

    wall_top = floors * FLOOR_H if kind == "pitched" else ridge_z

    # Far meshes never tile: capped UV spans keep texture-coordinate
    # derivatives small so distant instances sample the top mip levels of
    # the strip atlas instead of deep mips where strips bleed together
    # (the "rainbow roof" artifact).
    def wall_uvs(edge_len):
        u1 = min(edge_len / wall_w, 0.5)
        return ((0.0, wv0), (u1, wv0), (u1, wv1), (0.0, wv1))

    ring = ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))
    for i in range(4):
        a, b = ring[i], ring[(i + 1) % 4]
        edge = math.dist(a, b)
        quad((a[0], a[1], 0.0), (b[0], b[1], 0.0),
             (b[0], b[1], wall_top), (a[0], a[1], wall_top),
             wall_uvs(edge))

    u_roof = min(length_m / roof_w, 0.5)
    if kind == "pitched":
        ez = min(eave_z, wall_top)
        roof_v = min(0.6, math.hypot(hy, ridge_z - ez) / max(roof_w, 1.0))
        ruv = ((0.0, rv0), (u_roof, rv0),
               (u_roof, rv0 + (rv1 - rv0) * roof_v),
               (0.0, rv0 + (rv1 - rv0) * roof_v))
        quad((-hx, -hy, ez), (hx, -hy, ez),
             (hx, 0.0, ridge_z), (-hx, 0.0, ridge_z), ruv)
        quad((hx, hy, ez), (-hx, hy, ez),
             (-hx, 0.0, ridge_z), (hx, 0.0, ridge_z), ruv)
        u_gable = min(width_m / wall_w, 0.5)
        guv = ((0.0, wv0), (u_gable, wv0), (u_gable * 0.5, wv1))
        tri((hx, -hy, wall_top), (hx, hy, wall_top), (hx, 0.0, ridge_z), guv)
        tri((-hx, hy, wall_top), (-hx, -hy, wall_top), (-hx, 0.0, ridge_z), guv)
    else:  # flat: single top quad at the parapet line
        ruv = ((0.0, rv0), (u_roof, rv0), (u_roof, rv1), (0.0, rv1))
        quad((-hx, -hy, wall_top), (hx, -hy, wall_top),
             (hx, hy, wall_top), (-hx, hy, wall_top), ruv)

    return verts, faces


def roof_quad_mesh(length_m: float, width_m: float, top_z: float,
                   roof_band: tuple):
    """Ultra-far band: one roof-colored quad at roof height (2 tris).

    UVs cover a small untiled patch so the most-minified band never reaches
    the deep mips where atlas strips bleed (rainbow artifact).
    """
    hx, hy = length_m / 2.0, width_m / 2.0
    rv0, rv1, _roof_w = roof_band
    v_mid0 = rv0 + (rv1 - rv0) * 0.3
    v_mid1 = rv0 + (rv1 - rv0) * 0.7
    n = (0.0, 0.0, 1.0)
    verts = [
        (-hx, -hy, top_z, *n, 0.05, v_mid0),
        (hx, -hy, top_z, *n, 0.25, v_mid0),
        (hx, hy, top_z, *n, 0.25, v_mid1),
        (-hx, hy, top_z, *n, 0.05, v_mid1),
    ]
    faces = [(0, 1, 2), (0, 2, 3)]
    return verts, faces


def merge_bands_into_obj8(obj_path: str, bands) -> str:
    """Rewrite an OBJ8 as banded LOD: band 1 = the existing mesh.

    ``bands`` is a list of (verts, faces, near_m, far_m) for the far bands;
    band 1 spans 0..bands[0].near. X-Plane allows up to 4 bands total.
    """
    if not bands or len(bands) > 3:
        return "bad-bands"
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as fh:
        lines = fh.read().splitlines()

    header, vt_lines, idx_values = [], [], []
    tris_ranges = []
    for line in lines:
        parts = line.split()
        cmd = parts[0] if parts else ""
        if cmd == "VT":
            vt_lines.append(line)
        elif cmd in ("IDX", "IDX10"):
            idx_values.extend(int(v) for v in parts[1:])
        elif cmd == "TRIS":
            tris_ranges.append((int(parts[1]), int(parts[2])))
        elif cmd == "ATTR_LOD":
            return "already-banded"
        elif cmd == "POINT_COUNTS":
            continue
        elif cmd in ("ANIM_begin", "ANIM_end"):
            return "unsupported"
        elif not vt_lines and not idx_values:
            header.append(line)
    if not tris_ranges or not vt_lines:
        return "no-tris"

    band_ranges = []
    for verts, faces, near_m, far_m in bands:
        base_vt = len(vt_lines)
        # Append band vertices (Blender z-up -> OBJ8: x, z, -y).
        for x, y, z, nx, ny, nz, u, v in verts:
            vt_lines.append(
                f"VT\t{x:.4f} {z:.4f} {-y:.4f}\t"
                f"{nx:.3f} {nz:.3f} {-ny:.3f}\t{u:.5f} {v:.5f}"
            )
        idx_start = len(idx_values)
        for face in faces:
            # X-Plane front faces are clockwise relative to the geometric
            # outward direction (verified against xplane2blender output);
            # our band builders emit CCW-outward, so reverse.
            idx_values.extend(
                (face[0] + base_vt, face[2] + base_vt, face[1] + base_vt)
            )
        band_ranges.append((near_m, far_m, idx_start,
                            len(idx_values) - idx_start))

    out = []
    out.extend(header)
    out.append(f"POINT_COUNTS\t{len(vt_lines)}\t0\t0\t{len(idx_values)}")
    out.extend(vt_lines)
    for i in range(0, len(idx_values), 10):
        chunk = idx_values[i: i + 10]
        if len(chunk) == 10:
            out.append("IDX10\t" + " ".join(str(v) for v in chunk))
        else:
            for value in chunk:
                out.append(f"IDX\t{value}")
    out.append(f"ATTR_LOD\t0 {bands[0][2]}")
    for start, count in tris_ranges:
        out.append(f"TRIS\t{start} {count}")
    for near_m, far_m, idx_start, idx_count in band_ranges:
        out.append(f"ATTR_LOD\t{near_m} {far_m}")
        out.append(f"TRIS\t{idx_start} {idx_count}")
    with open(obj_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out) + "\n")
    return "patched"
