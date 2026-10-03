"""Build the O4SFR greenhouse library.

    python build_greenhouses.py --out <pkg> --pages <dir>

``--pages`` holds the four texture pages (tunnels.png, tunnels_nml.png,
solar.png, solar_nml.png) from one texture set -- ``textures_cycles.py``
(Cycles bake) or ``textures_ai.py`` (AI images); both follow ``layout``.

Writes ``<pkg>/library.txt`` and ``<pkg>/objects/``:

* ``tunnels_<W>x<L>.obj`` -- arrays of arched film tunnels, W = bays x
  pitch across the tunnels (OBJ x), L along them (OBJ z);
* ``solar_<L>x<W>.obj`` and ``solar_<L>x<W>_b.obj`` -- north-China solar
  greenhouses, L along the ridge (OBJ x), W deep (OBJ z), earth wall on
  -z; ``_b`` has its anteroom at the other end.

Virtual paths are ``o4sfr/agri/<name>`` (O4_Agri_Structures).  Bounds
are centred and within the declared footprint.  Geometry: ``geometry.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))
import O4_Agri_Structures as AGRI  # noqa: E402
import geometry as G  # noqa: E402
import layout as L  # noqa: E402

PAGES = ("tunnels.png", "tunnels_nml.png", "solar.png", "solar_nml.png")
NVCOMPRESS = os.path.join(HERE, "..", "..", "..", "Utils", "win", "nvcompress", "nvcompress.exe")


def to_dds(src, dst, normal):
    """PNG page -> DDS with a full mip chain (albedo DXT1; the normal page
    carries base reflectance in B and gloss in A, so DXT5, linear mips)."""
    opts = (["-color", "-alpha", "-no-mip-gamma-correct", "-bc3"] if normal
            else ["-color", "-noalpha", "-bc1"])
    subprocess.run([NVCOMPRESS, "-repeat", "-production", *opts, src, dst],
                   check=True, stdout=subprocess.DEVNULL)


def _fix_winding(verts, tris):
    """Make every triangle CCW seen from its vertex normals' side."""
    v = np.asarray([p[:6] for p in verts], dtype=np.float64)
    t = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    pa, pb, pc = v[t[:, 0], :3], v[t[:, 1], :3], v[t[:, 2], :3]
    n = np.cross(pb - pa, pc - pa)
    vn = v[t[:, 0], 3:] + v[t[:, 1], 3:] + v[t[:, 2], 3:]
    flip = np.einsum("ij,ij->i", n, vn) < 0
    t[flip] = t[flip][:, [0, 2, 1]]
    return t.reshape(-1).tolist()


def write_obj8(path, bands, texture, normal):
    """bands = [(Mesh, near, far)]; X-Plane: x east, y up, z south."""
    verts, idx, ranges = [], [], []
    for mesh, near, far in bands:
        base = len(verts)
        verts += mesh.verts
        start = len(idx)
        idx += [k + base for k in _fix_winding(mesh.verts, mesh.tris)]
        ranges.append((near, far, start, len(idx) - start))
    lines = ["I", "800", "OBJ", "",
             f"TEXTURE {texture}",
             f"TEXTURE_NORMAL {normal}",
             "NORMAL_METALNESS",
             "GLOBAL_specular 0.60",
             f"POINT_COUNTS {len(verts)} 0 0 {len(idx)}", ""]
    for x, y, z, nx, ny, nz, u, v in verts:
        lines.append(f"VT {x:.3f} {y:.3f} {z:.3f} {nx:.3f} {ny:.3f} {nz:.3f} "
                     f"{u:.5f} {v:.5f}")
    lines.append("")
    flipped = []                      # CCW -> OBJ8 clockwise front faces
    for i in range(0, len(idx), 3):
        flipped += [idx[i], idx[i + 2], idx[i + 1]]
    full = len(flipped) - len(flipped) % 10
    for i in range(0, full, 10):
        lines.append("IDX10 " + " ".join(str(k) for k in flipped[i:i + 10]))
    for k in flipped[full:]:
        lines.append(f"IDX {k}")
    lines.append("")
    for near, far, start, count in ranges:
        lines.append(f"ATTR_LOD {int(near)} {int(far)}")
        lines.append(f"TRIS {start} {count}")
    lines.append("")
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write("\n".join(lines))
    return len(idx) // 3


def _check_bounds(mesh, half_x, half_z, name):
    lo, hi = G.bounds(mesh)
    if lo[0] < -half_x - 1e-3 or hi[0] > half_x + 1e-3 or \
            lo[2] < -half_z - 1e-3 or hi[2] > half_z + 1e-3:
        raise ValueError(f"{name}: bounds {lo} {hi} exceed {half_x} x {half_z}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--pages", required=True)
    ap.add_argument("--dds", action="store_true",
                    help="ship DDS pages (nvcompress) instead of PNG")
    ap.add_argument("--only", default=None,
                    help="comma list of object names to build (previews)")
    args = ap.parse_args()
    obj_dir = os.path.join(args.out, "objects")
    os.makedirs(obj_dir, exist_ok=True)
    ext = ".dds" if args.dds else ".png"
    for name in PAGES:
        src = os.path.join(args.pages, name)
        if args.dds:
            to_dds(src, os.path.join(obj_dir, name[:-4] + ".dds"), name.endswith("_nml.png"))
        else:
            shutil.copyfile(src, os.path.join(obj_dir, name))
    only = set(args.only.split(",")) if args.only else None
    doors = os.path.join(args.pages, "doors.json")
    if os.path.isfile(doors):
        with open(doors, "r", encoding="utf-8") as fh:
            L.TUNNEL_DOORS.update({k: tuple(v) for k, v in json.load(fh).items()})
    lines = ["A", "800", "LIBRARY", ""]
    stats = {}

    def emit(virtual, bands, tex, half_x, half_z):
        name = os.path.basename(virtual)
        lines.append(f"EXPORT {virtual} objects/{name}")
        if only is not None and name not in only:
            return
        for mesh, _n, _f in bands:
            _check_bounds(mesh, half_x, half_z, name)
        tris = write_obj8(os.path.join(obj_dir, name), bands,
                          f"{tex}{ext}", f"{tex}_nml{ext}")
        stats.setdefault(tex, []).append(tris)

    for bays in range(1, AGRI.TUNNEL_MAX_BAYS + 1):
        for length in AGRI.TUNNEL_LENGTHS_M:
            bands = [(G.tunnel_mesh(bays, float(length), lod), near, far)
                     for lod, (near, far) in enumerate(G.LODS)]
            emit(AGRI.tunnel_asset_path(bays, length), bands, "tunnels",
                 bays * AGRI.TUNNEL_PITCH_M / 2.0, length / 2.0)
    for width in AGRI.SOLAR_WIDTHS_M:
        for length in AGRI.TUNNEL_LENGTHS_M:
            for variant in AGRI.SOLAR_VARIANTS:
                bands = [(G.solar_mesh(float(length), float(width), lod, variant),
                          near, far) for lod, (near, far) in enumerate(G.LODS)]
                emit(AGRI.solar_asset_path(width, length, variant), bands,
                     "solar", length / 2.0, width / 2.0)
    with open(os.path.join(args.out, "library.txt"), "w", encoding="ascii",
              newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    for tex, tris in stats.items():
        print(f"{tex}: {len(tris)} objects, tris (all LODs) "
              f"median {int(np.median(tris))} max {max(tris)}")
    print(f"{len(lines) - 4} exports -> {args.out}")


if __name__ == "__main__":
    main()
