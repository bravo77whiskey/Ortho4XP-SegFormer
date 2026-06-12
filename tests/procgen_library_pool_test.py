"""Pool-ingestion smoke test for the O4SFR ProcGen library.

Fabricates a tiny library package (OBJ8s written directly from archetype
MeshSpecs -- no Blender required), emits library.txt via
build_procgen_library, then asserts the REAL building-overlay pool builder
discovers and classifies the assets with the expected dimensions, heights,
regions and context flags.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCGEN = ROOT / "scripts" / "asset_pipeline" / "procgen"
SRC = ROOT / "src"
for path in (str(PROCGEN), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

import build_procgen_library  # noqa: E402
import O4_SFR_Building_Overlay as overlay  # noqa: E402
from archetypes import build_archetype  # noqa: E402
from archetypes.common import fmt_dim  # noqa: E402

PKG_NAME = "o4sfr_procgen_library"

# (region, length, width, floors) -- areas span classes 1-3.
SAMPLE_ASSETS = (
    ("generic", 6.2, 5.1, 1),
    ("generic", 11.2, 9.2, 2),
    ("europe", 7.5, 6.2, 1),
    ("europe", 16.7, 13.7, 2),
    ("north_america", 9.2, 7.5, 2),
)
VARIANTS = 2


def _spec_to_obj8(spec) -> str:
    """Minimal OBJ8 with the xplane2blender axis mapping (y up, z = -y)."""
    lines = ["I", "800", "OBJ", "", "TEXTURE ../../textures/atlas.png", ""]
    lines.append(f"POINT_COUNTS {len(spec.vertices)} 0 0 0")
    for x, y, z in spec.vertices:
        lines.append(f"VT {x:.6f} {z:.6f} {-y:.6f} 0 0 1 0 0")
    return "\n".join(lines) + "\n"


def _layout():
    with open(PROCGEN / "atlas_layout.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_package(tmp_path: Path) -> Path:
    layout = _layout()
    pkg = tmp_path / PKG_NAME
    assets = []
    for region, length, width, floors in SAMPLE_ASSETS:
        dims = f"{fmt_dim(length)}x{fmt_dim(width)}x{floors}"
        stem = f"gable_{dims}"
        asset_id = f"{region}/residential/{stem}"
        variants = []
        for k in range(1, VARIANTS + 1):
            physical = f"{region}/residential/{stem}_v{k}.obj"
            spec = build_archetype("gable", length, width, floors,
                                   1000 + k, layout)
            target = pkg / physical
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(_spec_to_obj8(spec), encoding="utf-8")
            variants.append({"variant": k, "seed": 1000 + k,
                             "physical_path": physical})
        assets.append({
            "id": asset_id,
            "virtual_path": f"o4sfr/{asset_id}.obj",
            "archetype": "gable",
            "length_m": length,
            "width_m": width,
            "floors": floors,
            "height_m": floors * 3.2,
            "region": region,
            "bucket": "residential",
            "band": "test",
            "flavor": "generic",
            "enabled": True,
            "variants": variants,
        })
    manifest = {
        "schema_version": 1,
        "lib_version": 1,
        "config_sha1": "test",
        "assets": assets,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    rc = build_procgen_library.main(
        ["--manifest", str(manifest_path), "--output", str(pkg)]
    )
    assert rc == 0
    assert (pkg / "library.txt").is_file()
    return pkg


def _pool_assets(pools):
    return [asset for pool in pools.values() for asset in pool]


def test_pool_ingests_procgen_package(tmp_path):
    _build_package(tmp_path)
    pools = overlay._build_optional_library_asset_pools(
        custom_scenery_dir=str(tmp_path),
        enabled_library_ids=("o4sfr",),
        cache_dir=str(tmp_path / "cache"),
        asset_region="europe",
    )
    assets = _pool_assets(pools)
    by_path = {asset["path"]: asset for asset in assets}

    # On a europe tile: europe + generic assets pool, north_america doesn't.
    expected_regions = {"generic", "europe"}
    expected_paths = {
        f"o4sfr/{region}/residential/"
        f"gable_{fmt_dim(l)}x{fmt_dim(w)}x{f}.obj"
        for region, l, w, f in SAMPLE_ASSETS if region in expected_regions
    }
    assert set(by_path) == expected_paths

    for region, length, width, floors in SAMPLE_ASSETS:
        if region not in expected_regions:
            continue
        virtual = (
            f"o4sfr/{region}/residential/"
            f"gable_{fmt_dim(length)}x{fmt_dim(width)}x{floors}.obj"
        )
        asset = by_path[virtual]
        # Height comes from the _LxWxF token: floors * 3.2.
        assert abs(asset["height_m"] - floors * 3.2) < 1e-6
        # Measured bounds equal the declared footprint.
        xmin, xmax, zmin, zmax = asset["bounds_m"]
        assert abs((xmax - xmin) - length) < 0.05
        assert abs((zmax - zmin) - width) < 0.05
        assert abs(xmin + xmax) < 0.01 and abs(zmin + zmax) < 0.01
        # Residential bucket => residential-context flag.
        assert asset["requires_residential_context"] is True
        assert asset["kind"] == "object"
        # Region priority: regional match 0, generic wildcard 1.
        assert asset["region_priority"] == (0 if region == "europe" else 1)
        # Footprint class matches the area-derived class.
        expected_class = overlay._class_for_footprint(asset["bounds_m"])
        assert asset["footprint_class"] == expected_class


def test_pool_region_routing_excludes_other_regions(tmp_path):
    _build_package(tmp_path)
    pools = overlay._build_optional_library_asset_pools(
        custom_scenery_dir=str(tmp_path),
        enabled_library_ids=("o4sfr",),
        cache_dir=str(tmp_path / "cache"),
        asset_region="north_america",
    )
    paths = {asset["path"] for asset in _pool_assets(pools)}
    assert not any("/europe/" in path for path in paths)
    assert any("/north_america/" in path for path in paths)
    assert any("/generic/" in path for path in paths)
