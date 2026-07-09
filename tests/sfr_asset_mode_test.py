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
