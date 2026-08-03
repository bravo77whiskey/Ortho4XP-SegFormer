"""Large-detection evidence gate: scoped raw YOLO OBBs must be corroborated.

Covers the failure classes that motivated the gate (sports pitches and
featureless farmland extruded as warehouse facades after the world-scale
footprint bound replaced the old 7,000 m2 envelope) and the accept paths that
must keep working (SegFormer-backed roofs, edge-rescued roofs SegFormer
missed, small ungated detections).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_SFR_Building_Overlay as overlay  # noqa: E402
import O4_SFR_Inference as SEGFORMER  # noqa: E402

M_PER_PX = 2.0
IMG_SIDE = 512

# 100x60 px at 2 m/px -> 200x120 m = 24,000 m2: comfortably in gate scope.
BOX_PTS = [[100.0, 100.0], [200.0, 100.0], [200.0, 160.0], [100.0, 160.0]]


def _detection(points=BOX_PTS, area_m2=24_000.0, max_side_m=200.0):
    return {
        "placement_class": overlay.BLD_CLASS_EXTRA_LARGE,
        "points": [list(p) for p in points],
        "area_m2": area_m2,
        "max_side_m": max_side_m,
        "confidence": 0.6,
    }


def _flat_image(rgb):
    image = np.zeros((IMG_SIDE, IMG_SIDE, 3), dtype=np.uint8)
    image[:] = rgb
    return image


def _fill_box(image, rgb, pts=BOX_PTS, pad=0):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, x2 = int(min(xs)) - pad, int(max(xs)) + pad
    y1, y2 = int(min(ys)) - pad, int(max(ys)) + pad
    image[y1:y2, x1:x2] = rgb
    return image


def _ctx(image=None, veg_map=None, image_loader=None):
    ctx = overlay._build_detection_evidence_context(
        image, image_loader, veg_map, M_PER_PX
    )
    assert ctx is not None
    return ctx


def test_missing_metrics_fail_closed():
    assert not overlay._direct_yolo_raw_footprint_allowed({})
    assert not overlay._direct_yolo_raw_footprint_allowed(
        {"area_m2": 0.0, "max_side_m": 0.0}
    )
    assert overlay._direct_yolo_raw_footprint_allowed(
        {"area_m2": 1_000.0, "max_side_m": 40.0}
    )


def test_small_detection_is_not_gated():
    # 400 m2 house on a green lawn: below gate scope, no evidence demanded.
    image = _flat_image((60, 140, 60))
    det = _detection(
        points=[[10, 10], [20, 10], [20, 20], [10, 20]],
        area_m2=400.0,
        max_side_m=20.0,
    )
    allowed, reason = overlay._detection_evidence_allows(det, _ctx(image=image))
    assert allowed and reason is None


def test_green_pitch_is_rejected():
    # Grass pitch on bare soil: crisp boundary edges must NOT rescue it.
    image = _flat_image((150, 120, 90))
    _fill_box(image, (70, 160, 60))
    allowed, reason = overlay._detection_evidence_allows(
        _detection(), _ctx(image=image)
    )
    assert not allowed and reason == "green"


def test_segformer_building_coverage_accepts():
    veg_map = np.zeros((IMG_SIDE, IMG_SIDE), dtype=np.int8)
    xs = [int(p[0]) for p in BOX_PTS]
    ys = [int(p[1]) for p in BOX_PTS]
    veg_map[min(ys):max(ys), min(xs):max(xs)] = SEGFORMER.CLASS_BUILDING
    # Even a green metal roof survives when SegFormer recognises the building.
    image = _flat_image((150, 120, 90))
    _fill_box(image, (60, 150, 70))
    allowed, reason = overlay._detection_evidence_allows(
        _detection(), _ctx(image=image, veg_map=veg_map)
    )
    assert allowed and reason == "segformer"


def test_zone_map_verdict_is_final_even_for_roof_like_pixels():
    # SegFormer says "not built" -> reject, even though the pixels show a
    # crisp grey slab. Perimeter-gradient rescues were measured unsafe on real
    # imagery (row-crop farmland and solar farms are full of crisp edges), so
    # the zone map's word is final when it is present.
    veg_map = np.zeros((IMG_SIDE, IMG_SIDE), dtype=np.int8)
    image = _flat_image((150, 120, 90))
    _fill_box(image, (90, 92, 100))
    allowed, reason = overlay._detection_evidence_allows(
        _detection(), _ctx(image=image, veg_map=veg_map)
    )
    assert not allowed and reason == "uncorroborated"


def test_featureless_field_is_uncorroborated():
    veg_map = np.zeros((IMG_SIDE, IMG_SIDE), dtype=np.int8)
    image = _flat_image((150, 120, 90))  # uniform bare field, no box drawn
    allowed, reason = overlay._detection_evidence_allows(
        _detection(), _ctx(image=image, veg_map=veg_map)
    )
    assert not allowed and reason == "uncorroborated"


def test_without_zone_map_only_green_rejects():
    # Zone-map-less fallback (mesh-water shortcut tiles): grass still rejects,
    # anything else passes — one absent signal must not drop real buildings.
    brown = _flat_image((150, 120, 90))
    allowed, reason = overlay._detection_evidence_allows(
        _detection(), _ctx(image=brown)
    )
    assert allowed and reason is None


def test_no_pixels_and_no_zone_map_allows():
    # Loader fails (missing texture): one absent signal must not reject.
    allowed, reason = overlay._detection_evidence_allows(
        _detection(), _ctx(image_loader=lambda: None)
    )
    assert allowed and reason is None


def test_lazy_image_loader_used_once():
    calls = []

    def loader():
        calls.append(1)
        image = _flat_image((150, 120, 90))
        return _fill_box(image, (70, 160, 60))

    ctx = _ctx(image_loader=loader)
    for _ in range(2):
        allowed, reason = overlay._detection_evidence_allows(_detection(), ctx)
        assert not allowed and reason == "green"
    assert len(calls) == 1


def test_filter_reports_reasons_and_counts():
    image = _flat_image((150, 120, 90))
    _fill_box(image, (70, 160, 60))
    small = _detection(
        points=[[300, 300], [310, 300], [310, 310], [300, 310]],
        area_m2=400.0,
        max_side_m=20.0,
    )
    counts = {}
    kept, dropped = overlay._filter_oversized_direct_yolo_detections(
        [_detection(), small],
        evidence_ctx=_ctx(image=image),
        file_counts=counts,
    )
    assert [d["area_m2"] for d in kept] == [400.0]
    assert dropped == 1
    assert counts == {"yolo_evidence_reject_green": 1}


def test_gate_env_kill_switch(monkeypatch):
    monkeypatch.setenv("O4_SFR_BLD_EVIDENCE_GATE", "0")
    assert (
        overlay._build_detection_evidence_context(None, None, None, M_PER_PX)
        is None
    )


def test_filter_without_context_keeps_legacy_behaviour():
    detections = [
        _detection(),
        {"area_m2": 900_000.0, "max_side_m": 1_500.0, "points": BOX_PTS},
    ]
    kept, dropped = overlay._filter_oversized_direct_yolo_detections(detections)
    assert len(kept) == 1 and dropped == 1
