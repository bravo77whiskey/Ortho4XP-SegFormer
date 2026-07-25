import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_SFR_Building_Overlay as overlay  # noqa: E402


def test_building_asset_mode_aliases():
    assert overlay._building_asset_mode("obj") == "objects"
    assert overlay._building_asset_mode("fac") == "facades"
    assert overlay._building_asset_mode("all") == "both"
    assert overlay._building_asset_mode("nonsense") == "both"


def test_asset_mode_filters_object_and_facade_pools():
    cls = overlay.BLD_CLASS_TINY_RESIDENTIAL
    pools = {
        c: [] for c in overlay.BLD_PLACEMENT_CLASSES
    }
    pools[cls] = [
        {"kind": "object", "path": "lib/example.obj"},
        {"kind": "facade", "path": "lib/example.fac"},
    ]

    assert [
        asset["kind"]
        for asset in overlay._filter_asset_pools_for_mode(pools, "objects")[cls]
    ] == ["object"]
    assert [
        asset["kind"]
        for asset in overlay._filter_asset_pools_for_mode(pools, "facades")[cls]
    ] == ["facade"]
    assert [
        asset["kind"]
        for asset in overlay._filter_asset_pools_for_mode(pools, "both")[cls]
    ] == ["object", "facade"]


def test_o4sfr_is_not_a_runtime_curated_library():
    assert "o4sfr" not in overlay.CURATED_EXTRA_BUILDING_LIBRARIES
    assert "o4sfr" not in overlay._enabled_extra_library_ids("auto")
    assert overlay._enabled_extra_library_ids("o4sfr") == ()


def test_yolo_facade_fallback_prefers_detection_height():
    rng = overlay.np.random.default_rng(123)
    assert overlay._yolo_facade_height_m(
        {"height_m": 37.5},
        rng,
        {},
        overlay.BLD_CLASS_APARTMENT_BLOCK,
    ) == 37.5

    assert overlay._yolo_facade_height_m(
        {"height_m": float("nan")},
        rng,
        {},
        overlay.BLD_CLASS_TINY_RESIDENTIAL,
    ) == overlay.DEFAULT_FACADE_HEIGHT_M[overlay.BLD_CLASS_TINY_RESIDENTIAL]


def test_large_footprint_height_cap_feeds_facade_height():
    rng = overlay.np.random.default_rng(123)
    detection = {
        "height_m": overlay._capped_detection_height_m(
            {
                "placement_class": overlay.BLD_CLASS_APARTMENT_BLOCK,
                "area_m2": 8_000.0,
                "max_side_m": 120.0,
            },
            80.0,
        ),
    }

    assert detection["height_m"] == overlay.LARGE_FOOTPRINT_HEIGHT_CAP_M
    assert overlay._yolo_facade_height_m(
        detection,
        rng,
        {},
        overlay.BLD_CLASS_LARGE,
    ) == overlay.LARGE_FOOTPRINT_HEIGHT_CAP_M


def test_direct_yolo_facade_rejects_oversized_raw_footprint():
    assert overlay._direct_yolo_facade_footprint_allowed(
        {
            "placement_class": overlay.BLD_CLASS_APARTMENT_BLOCK,
            "area_m2": 1_200.0,
            "max_side_m": 45.0,
        }
    )
    # Warehouse / mega-factory scale is a real building, not an artefact:
    # the gate now sits at the world's largest footprint, so these pass.
    assert overlay._direct_yolo_facade_footprint_allowed(
        {
            "placement_class": overlay.BLD_CLASS_APARTMENT_BLOCK,
            "area_m2": 8_000.0,
            "max_side_m": 120.0,
        }
    )
    assert overlay._direct_yolo_facade_footprint_allowed(
        {
            "placement_class": overlay.BLD_CLASS_MEDIUM,
            "length_m": 140.0,
            "width_m": 70.0,
        }
    )
    # Boeing Everett (~398,000 m², ~1.1 km long) still fits under the gate.
    assert overlay._direct_yolo_facade_footprint_allowed(
        {
            "placement_class": overlay.BLD_CLASS_EXTRA_LARGE,
            "length_m": 1_100.0,
            "width_m": 360.0,
        }
    )
    # Beyond any building on earth — that is an OBB spanning a whole field.
    assert not overlay._direct_yolo_facade_footprint_allowed(
        {
            "placement_class": overlay.BLD_CLASS_EXTRA_LARGE,
            "area_m2": 900_000.0,
            "max_side_m": 1_500.0,
        }
    )
