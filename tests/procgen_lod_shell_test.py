"""Tests for the two-band far-LOD shell patcher (no Blender required)."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROCGEN = ROOT / "scripts" / "asset_pipeline" / "procgen"
if str(PROCGEN) not in sys.path:
    sys.path.insert(0, str(PROCGEN))

from apply_lod_shells import patch_shell  # noqa: E402
from archetypes import ARCHETYPES, build_archetype  # noqa: E402


@pytest.fixture(scope="module")
def layout():
    with open(PROCGEN / "atlas_layout.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_minimal_obj8(path: Path, spec):
    """Emit a valid single-band OBJ8 (VT+IDX+TRIS) from a MeshSpec."""
    verts = []
    indices = []
    for face in spec.faces:
        base = len(verts)
        for vid in face:
            x, y, z = spec.vertices[vid]
            verts.append((x, z, -y))
        for k in range(1, len(face) - 1):
            indices.extend((base, base + k + 1, base + k))
    lines = ["I", "800", "OBJ", "",
             "TEXTURE ../../textures/o4sfr_procgen_atlas_generic.png", "",
             f"POINT_COUNTS {len(verts)} 0 0 {len(indices)}"]
    for x, y, z in verts:
        lines.append(f"VT {x:.4f} {y:.4f} {z:.4f} 0 1 0 0 0")
    for value in indices:
        lines.append(f"IDX {value}")
    lines.append(f"TRIS 0 {len(indices)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse(path: Path):
    lods, tris, n_vt, idx = [], [], 0, 0
    xs, ys = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "ATTR_LOD":
            lods.append((int(parts[1]), int(parts[2])))
        elif parts[0] == "TRIS":
            tris.append((int(parts[1]), int(parts[2])))
        elif parts[0] == "VT":
            n_vt += 1
            xs.append(float(parts[1]))
            ys.append(float(parts[3]))
        elif parts[0] in ("IDX", "IDX10"):
            idx += len(parts) - 1
        elif parts[0] == "POINT_COUNTS":
            declared = (int(parts[1]), int(parts[4]))
    return {"lods": lods, "tris": tris, "n_vt": n_vt, "idx": idx,
            "declared": declared, "xs": xs, "ys": ys}


@pytest.mark.parametrize("archetype,dims", [
    ("gable", (11.5, 9.5, 2)),
    ("hip", (11.5, 9.5, 1)),
    ("flatres", (9.5, 7.8, 2)),
    ("aptslab", (33.4, 14.2, 5)),
    ("warehouse", (53.8, 30.4, 3)),
    ("shophouse", (20.9, 9.5, 3)),
])
def test_lod_bands_added(tmp_path, layout, archetype, dims):
    length, width, floors = dims
    spec = build_archetype(archetype, length, width, floors, 7, layout)
    obj = tmp_path / f"{archetype}_test.obj"
    _write_minimal_obj8(obj, spec)

    bands = (2000, 6000, 11000, 16000)
    status = patch_shell(str(obj), archetype, length, width, floors, 7,
                         "generic", layout, bands)
    assert status == "patched"
    after = _parse(obj)

    # Band distances scale with the 3D bounding diagonal (small buildings
    # cull early, big/tall ones stay visible): expect the scaled boundaries.
    from apply_lod_shells import _band_scale
    scale = _band_scale(length, width, floors * 3.2)
    d0, d1, d2, d3 = (int(round(b * scale)) for b in bands)
    assert 0.44 <= scale <= 1.9
    # Every archetype: full / windowed shell / plain flat box / roof quad.
    assert after["lods"] == [(0, d0), (d0, d1), (d1, d2), (d2, d3)]
    # Far bands get progressively cheaper: the shell carries per-floor wall
    # quads (window rows keep band-1 scale), the box and quad stay minimal.
    far_tris = [count // 3 for _start, count in after["tris"][1:]]
    # Shell walls: one quad per floor per side, plus a partial parapet
    # course on flat archetypes and the roof/gable faces.
    assert far_tris[0] <= 8 * (floors + 1) + 8
    assert far_tris[1] <= 10
    assert far_tris[-1] == 2
    # Bookkeeping stays consistent and the footprint never grows.
    assert after["declared"] == (after["n_vt"], after["idx"])
    assert max(after["xs"]) <= length / 2 + 0.01
    assert min(after["xs"]) >= -length / 2 - 0.01
    assert max(map(abs, after["ys"])) <= width / 2 + 0.01
    # Re-patching re-bands in place and is byte-idempotent.
    first = obj.read_bytes()
    assert patch_shell(str(obj), archetype, length, width, floors, 7,
                       "generic", layout, bands) == "patched"
    assert obj.read_bytes() == first


def test_every_archetype_has_shell_meta(layout):
    for name in sorted(ARCHETYPES):
        spec = build_archetype(name, 20.9, 12.9, 2, 11, layout)
        assert spec.meta.get("kind") in ("pitched", "flat"), name
        assert spec.meta["ridge_z"] > 0, name
        assert spec.meta["wall_strip"] in layout["strips"], name
        assert spec.meta["roof_strip"] in layout["strips"], name