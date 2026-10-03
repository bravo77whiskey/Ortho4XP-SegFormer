"""Farmland / greenhouse sorting of building detections (O4_Agri_Structures)."""

import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import O4_Agri_Structures as AGRI  # noqa: E402

MPP = 2.0
AGRI_CLS = AGRI.CLASS_AGRICULTURE
BLD_CLS = AGRI.CLASS_BUILDING
ROAD_CLS = AGRI.CLASS_ROAD


def _det(cx, cy, length_m, width_m, angle_deg=0.0):
    a = math.radians(angle_deg)
    ux = np.array([math.cos(a), math.sin(a)])
    uy = np.array([-math.sin(a), math.cos(a)])
    hl, hw = length_m / 2 / MPP, width_m / 2 / MPP
    c = np.array([cx, cy], float)
    pts = [c - ux * hl - uy * hw, c + ux * hl - uy * hw,
           c + ux * hl + uy * hw, c - ux * hl + uy * hw]
    return {
        "points": [p.tolist() for p in pts],
        "center": [cx, cy],
        "area_m2": length_m * width_m,
        "length_m": length_m,
        "width_m": width_m,
        "confidence": 0.1,
    }


def _image(rgb, size=200):
    return np.full((size, size, 3), rgb, np.uint8)


def _veg(cls, size=200):
    return np.full((size, size), cls, np.int8)


def _kind(det, veg, img):
    return AGRI.classify_detection(det, veg, lambda: img, MPP)[0]


GREY_FILM = (200, 202, 198)
GREEN_CROP = (70, 130, 40)


def test_large_unsupported_farmland_grey_is_greenhouse():
    assert _kind(_det(100, 100, 60, 20), _veg(AGRI_CLS), _image(GREY_FILM)) == \
        AGRI.KIND_GREENHOUSE


def test_large_unsupported_farmland_green_is_ground():
    assert _kind(_det(100, 100, 60, 20), _veg(AGRI_CLS), _image(GREEN_CROP)) == \
        AGRI.KIND_GROUND


def test_small_compact_farm_shed_stays_building():
    assert _kind(_det(100, 100, 14, 10), _veg(AGRI_CLS), _image(GREEN_CROP)) == \
        AGRI.KIND_BUILDING


def test_small_but_elongated_is_not_a_shed():
    # 40 x 6 m = 240 m2 but aspect 6.7: a tunnel, not a shed
    assert _kind(_det(100, 100, 40, 6), _veg(AGRI_CLS), _image(GREY_FILM)) == \
        AGRI.KIND_GREENHOUSE


def test_unsupported_outside_farmland_stays_building():
    veg = _veg(3)  # developed
    assert _kind(_det(100, 100, 60, 20), veg, _image(GREEN_CROP)) == AGRI.KIND_BUILDING


def test_unsupported_on_road_is_ground():
    assert _kind(_det(100, 100, 40, 20), _veg(ROAD_CLS), _image(GREY_FILM)) == \
        AGRI.KIND_GROUND


def test_supported_roof_in_town_stays_building():
    assert _kind(_det(100, 100, 60, 12), _veg(BLD_CLS), _image(GREY_FILM)) == \
        AGRI.KIND_BUILDING


def test_supported_long_film_roof_in_farmland_is_greenhouse():
    veg = _veg(AGRI_CLS)
    det = _det(100, 100, 80, 12)
    pts = np.round(np.asarray(det["points"])).astype(np.int32)
    import cv2
    cv2.fillPoly(veg, [pts], BLD_CLS)
    assert _kind(det, veg, _image((215, 215, 212))) == AGRI.KIND_GREENHOUSE
    # the same shape with a coloured metal roof is a livestock shed
    assert _kind(det, veg, _image((60, 90, 170))) == AGRI.KIND_BUILDING


def test_image_is_loaded_lazily_and_once():
    calls = []

    def loader():
        calls.append(1)
        return _image(GREY_FILM)

    dets = [_det(60, 60, 14, 10), _det(140, 140, 60, 20), _det(60, 140, 60, 20)]
    buildings, greenhouses = AGRI.split_detections(dets, _veg(AGRI_CLS), loader, MPP)
    assert len(buildings) == 1 and len(greenhouses) == 2
    assert len(calls) == 1
    assert all(g["structure"] == "greenhouse" for g in greenhouses)

    calls.clear()
    AGRI.split_detections([_det(60, 60, 14, 10)], _veg(AGRI_CLS), loader, MPP)
    assert calls == []


def test_ground_is_dropped_and_counted():
    counts = {}
    b, g = AGRI.split_detections(
        [_det(100, 100, 60, 20)], _veg(AGRI_CLS), lambda: _image(GREEN_CROP), MPP,
        counts=counts)
    assert b == [] and g == []
    assert counts == {"agri_ground_farmland": 1}


def test_pieces_cover_detection_with_ladder_sizes():
    det = _det(500, 500, 100, 40, angle_deg=30)
    det["tunnel_axis"] = "long"
    pieces = AGRI.greenhouse_pieces(det, MPP)
    assert sum(p["bays"] for p in pieces) == 5          # 40 m / 8 m pitch
    assert all(p["length_m"] in AGRI.TUNNEL_LENGTHS_M for p in pieces)
    assert all(p["path"] == AGRI.tunnel_asset_path(p["bays"], p["length_m"])
               for p in pieces)
    # image angle 30 deg (x east, y south) -> bearing 120 deg
    assert pieces[0]["bearing"] == pytest.approx(120.0, abs=1e-3)
    c = np.mean([p["center"] for p in pieces], axis=0)
    assert np.allclose(c, [500, 500], atol=0.5)


def test_pieces_split_long_and_wide_arrays():
    det = _det(1000, 1000, 400, 150)
    det["tunnel_axis"] = "long"
    pieces = AGRI.greenhouse_pieces(det, MPP)
    assert max(p["bays"] for p in pieces) <= AGRI.TUNNEL_MAX_BAYS
    assert all(p["length_m"] <= 150 for p in pieces)
    assert sum(p["bays"] for p in pieces) == 3 * 19   # 19 bays, 3 length segments
    # 400 m -> 3 segments of 133.3 m, snapped down to 130 m
    assert {p["length_m"] for p in pieces} == {130.0}
    # tunnels across the short axis: bearing turns by 90 deg
    det["tunnel_axis"] = "short"
    short = AGRI.greenhouse_pieces(det, MPP)
    assert short[0]["bearing"] % 180 == pytest.approx(0.0, abs=1e-3)


def test_too_short_detection_places_nothing():
    det = _det(100, 100, 9, 8)
    assert AGRI.greenhouse_pieces(det, MPP) == []


def test_tunnel_axis_follows_stripes():
    img = np.full((200, 200, 3), 120, np.uint8)
    img[::4] = 220          # stripes along image x (rows vary)
    det = _det(100, 100, 120, 60)   # long axis along x
    assert AGRI.tunnel_axis(img, det["points"]) == "long"
    assert AGRI.tunnel_axis(img.transpose(1, 0, 2).copy(), det["points"]) == "short"


def _supported(veg, det):
    import cv2
    pts = np.round(np.asarray(det["points"])).astype(np.int32)
    cv2.fillPoly(veg, [pts], BLD_CLS)


def test_greenhouse_field_grows_from_farmland_edge():
    # A dense belt: every greenhouse reads as roof and the belt interior has
    # no farmland ring.  The two rows touching farmland seed the field.
    size = 400
    veg = np.full((size, size), BLD_CLS, np.int8)
    veg[:, :76] = AGRI_CLS
    # the first column's ring reaches the farmland, the others are 70 m on
    dets = [_det(x, 40 + 15 * k, 60, 10) for x in (90, 125, 160) for k in range(20)]
    for d in dets:
        _supported(veg, d)
    img = _image(GREY_FILM, size)
    low = lambda ds: [8.0] * len(ds)
    kinds = [k for k, _ in AGRI.classify_detections(dets, veg, lambda: img, MPP, low)]
    assert kinds.count(AGRI.KIND_GREENHOUSE) == len(dets)
    # tall roofs (an industrial estate) never join the field
    tall = lambda ds: [20.0] * len(ds)
    kinds = [k for k, _ in AGRI.classify_detections(dets, veg, lambda: img, MPP, tall)]
    assert AGRI.KIND_GREENHOUSE not in kinds


def test_heights_asked_only_for_long_grey_roofs():
    asked = []

    def heights(ds):
        asked.extend(ds)
        return [8.0] * len(ds)

    veg = _veg(AGRI_CLS, 300)
    long_roof = _det(80, 80, 80, 12)
    house = _det(200, 200, 12, 10)
    for d in (long_roof, house):
        _supported(veg, d)
    AGRI.classify_detections([long_roof, house], veg, lambda: _image(GREY_FILM, 300),
                             MPP, heights)
    assert asked == [long_roof]


def test_single_east_west_span_is_a_solar_greenhouse_with_wall_north():
    for angle in (0.0, 180.0, 20.0, -20.0):
        det = _det(500, 500, 90, 14, angle_deg=angle)
        det["tunnel_axis"] = "long"
        pieces = AGRI.greenhouse_pieces(det, MPP)
        assert [p["kind"] for p in pieces] == ["solar"]
        p = pieces[0]
        assert p["path"] in {AGRI.solar_asset_path(14, 90, v) for v in AGRI.SOLAR_VARIANTS}
        # the OBJ's -z wall faces bearing == heading: within 90 deg of north
        h = p["bearing"] % 360
        assert h <= 90 or h >= 270
        # image angle a (x east, y south): ridge bearing 90 + a, wall at a
        expected = (((angle + 90.0) % 180.0) - 90.0) % 360.0
        assert min(abs(h - expected), 360 - abs(h - expected)) < 1e-3
    south = AGRI.greenhouse_pieces(_det(500, 500, 90, 14), MPP, northern=False)
    assert south[0]["bearing"] % 360 == pytest.approx(180.0, abs=1e-3)


def test_north_south_or_wide_detections_stay_tunnels():
    det = _det(500, 500, 90, 14, angle_deg=90.0)   # ridge north-south
    det["tunnel_axis"] = "long"
    assert {p["kind"] for p in AGRI.greenhouse_pieces(det, MPP)} == {"tunnels"}
    det = _det(500, 500, 90, 40)                   # several spans wide
    det["tunnel_axis"] = "long"
    assert {p["kind"] for p in AGRI.greenhouse_pieces(det, MPP)} == {"tunnels"}


def test_overlay_places_every_piece_of_a_tiled_detection():
    import O4_Building_Overlay as BLD
    size = 1200
    det = _det(600, 600, 300, 40)            # 300 m long: two 150 m pieces
    det["tunnel_axis"] = "long"
    pieces = AGRI.greenhouse_pieces(det, MPP)
    assert len(pieces) == 2
    zeros = lambda: np.zeros((size, size), np.uint8)
    objects, facades, counts = [], [], {}
    n = BLD._place_greenhouses(
        [det], m_per_px=MPP, img_w=size, img_h=size, lat=27, lon=120,
        bounds=(27.9, 27.8, 120.8, 120.9),
        static_occ_mask=zeros(), route_excl_mask=None,
        building_spacing_mask=zeros(), fit_scratch=zeros(),
        tunnel_paths=frozenset(p["path"] for p in pieces), facade_path=None,
        placed_objects=objects, placed_facades=facades, file_counts=counts,
    )
    assert n == 2 and len(objects) == 2 and facades == []
    assert {o[3] for o in objects} == {p["path"] for p in pieces}


def test_overlay_falls_back_to_facade_or_nothing():
    import O4_Building_Overlay as BLD
    size = 400
    det = _det(200, 200, 60, 20)
    det["tunnel_axis"] = "long"
    zeros = lambda: np.zeros((size, size), np.uint8)
    for facade, expect in (("facades/agricultural/greenhouse_1.fac", 1), (None, 0)):
        objects, facades = [], []
        BLD._place_greenhouses(
            [det], m_per_px=MPP, img_w=size, img_h=size, lat=27, lon=120,
            bounds=(27.9, 27.8, 120.8, 120.9),
            static_occ_mask=zeros(), route_excl_mask=None,
            building_spacing_mask=zeros(), fit_scratch=zeros(),
            tunnel_paths=frozenset(), facade_path=facade,
            placed_objects=objects, placed_facades=facades, file_counts={},
        )
        assert objects == [] and len(facades) == expect
        if expect:
            assert facades[0][1] == facade and facades[0][2] == BLD.GREENHOUSE_HEIGHT_M
