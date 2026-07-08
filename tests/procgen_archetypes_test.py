"""Unit tests for procedural building archetypes (no Blender required).

Every implemented archetype must, across a sweep of dimensions and seeds:
  * stay within the triangle budget;
  * measure exactly the declared footprint, centered, base at z=0;
  * keep UVs inside each strip's V band (U is free -- it tiles);
  * produce identical bounds for every visual variant of one asset.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROCGEN = ROOT / "scripts" / "asset_pipeline" / "procgen"
if str(PROCGEN) not in sys.path:
    sys.path.insert(0, str(PROCGEN))

from archetypes import ARCHETYPES, build_archetype  # noqa: E402
from archetypes.combo_styles import group_for_archetype  # noqa: E402
from archetypes.common import TRI_BUDGET  # noqa: E402


@pytest.fixture(scope="module")
def layout():
    with open(PROCGEN / "atlas_layout.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def _combo(layout, archetype, flavor="generic"):
    group = group_for_archetype(archetype)
    return layout["combos"][f"{flavor}/{group}"]


# Dimension sweep per archetype: (length, width, floors) spanning each
# archetype's band range plus edge cases (square, max aspect, min size).
SWEEPS = {
    "gable": [(3.5, 3.5, 1), (6.2, 5.1, 1), (11.2, 9.2, 2),
              (16.7, 16.7, 2), (20.4, 6.8, 1), (20.4, 13.7, 3)],
    "hip": [(3.5, 3.5, 1), (6.2, 5.1, 1), (11.2, 11.2, 2),
            (16.7, 9.2, 2), (20.4, 6.8, 1), (20.4, 13.7, 3)],
    "lshape": [(7.8, 5.2, 1), (11.5, 9.5, 2), (14.1, 14.1, 2),
               (21.0, 8.9, 1), (21.0, 17.2, 3)],
    "rowhouse": [(11.5, 7.8, 2), (17.2, 9.5, 2), (25.1, 11.7, 3),
                 (36.7, 12.9, 3), (44.5, 10.6, 4)],
    "flatres": [(3.5, 3.5, 1), (6.2, 5.1, 1), (11.2, 9.2, 2),
                (16.7, 16.7, 2), (20.4, 13.7, 3)],
    "shophouse": [(9.5, 6.2, 2), (14.1, 7.8, 2), (20.9, 9.5, 3),
                  (33.4, 11.7, 3), (44.5, 12.9, 4)],
    "aptslab": [(20.7, 11.7, 3), (33.4, 14.2, 4), (48.9, 17.1, 5),
                (65.1, 18.9, 6), (86.7, 22.8, 7)],
    "aptblock": [(17.1, 17.1, 3), (25.1, 20.7, 4), (33.4, 27.6, 5),
                 (40.4, 33.4, 6), (44.5, 36.7, 7)],
    "flatcom": [(17.1, 17.1, 2), (25.1, 12.9, 3), (44.5, 18.9, 4),
                (78.8, 20.7, 5), (104.9, 27.6, 7)],
    "warehouse": [(36.7, 22.8, 2), (53.8, 30.4, 3), (71.6, 36.7, 3),
                  (95.3, 48.9, 4), (104.9, 21.0, 2)],
    "bigbox": [(40.4, 33.4, 2), (59.2, 44.5, 3), (78.8, 53.8, 3),
               (95.3, 65.1, 4), (104.9, 44.5, 3)],
}

SEEDS = (1, 7, 12345, 987654321)


def _cases():
    for name in sorted(ARCHETYPES):
        assert name in SWEEPS, f"add a dimension sweep for archetype {name}"
        for dims in SWEEPS[name]:
            yield name, dims


@pytest.mark.parametrize("name,dims", list(_cases()))
def test_archetype_contract(name, dims, layout):
    length, width, floors = dims
    bounds_seen = set()
    sub = _combo(layout, name)
    for seed in SEEDS:
        spec = build_archetype(name, length, width, floors, seed, sub)
        # validate_spec already ran inside the builder; re-assert the core
        # contract here so a builder that forgets to validate still fails.
        assert spec.tri_count() <= TRI_BUDGET
        xmin, xmax, ymin, ymax, zmin, _zmax = spec.bounds()
        assert abs(xmax - length / 2) < 0.005 and abs(xmin + length / 2) < 0.005
        assert abs(ymax - width / 2) < 0.005 and abs(ymin + width / 2) < 0.005
        assert abs(zmin) < 0.005
        bounds_seen.add(
            tuple(round(v, 4) for v in (xmin, xmax, ymin, ymax))
        )
    # Seeded variants may differ inside, but the footprint never moves.
    assert len(bounds_seen) == 1


@pytest.mark.parametrize("name,dims", list(_cases()))
def test_archetype_uvs_inside_strip_bands(name, dims, layout):
    length, width, floors = dims
    sub = _combo(layout, name)
    bands = sorted(
        (row["v0"], row["v1"]) for row in sub["strips"].values()
    )
    spec = build_archetype(name, length, width, floors, SEEDS[0], sub)
    for face_uvs in spec.uvs:
        for _u, v in face_uvs:
            assert any(v0 - 1e-6 <= v <= v1 + 1e-6 for v0, v1 in bands), (
                f"UV v={v} outside every strip band"
            )


def test_same_seed_same_mesh(layout):
    for name in sorted(ARCHETYPES):
        length, width, floors = SWEEPS[name][1]
        sub = _combo(layout, name)
        a = build_archetype(name, length, width, floors, 42, sub)
        b = build_archetype(name, length, width, floors, 42, sub)
        assert a.vertices == b.vertices
        assert a.faces == b.faces
        assert a.uvs == b.uvs


def test_flavor_changes_massing_not_bounds(layout):
    """Regional massing must vary silhouettes while bounds stay exact."""
    for name in ("gable", "hip"):
        length, width, floors = SWEEPS[name][2]
        heights = {}
        for flavor in ("europe", "mediterranean", "asia"):
            spec = build_archetype(
                name, length, width, floors, 42,
                _combo(layout, name, flavor), flavor
            )
            xmin, xmax, ymin, ymax, _zmin, zmax = spec.bounds()
            assert abs(xmax - length / 2) < 0.005
            assert abs(ymax - width / 2) < 0.005
            heights[flavor] = zmax
        # Steep European roofs must ridge higher than Mediterranean ones.
        assert heights["europe"] > heights["mediterranean"] + 0.3
