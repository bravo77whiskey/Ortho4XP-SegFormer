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


TEAL_FILM = (70, 120, 125)      # Korean vinyl house film: teal, mid-dark
PALE_BLUE = (150, 200, 230)     # painted steel roof


def _striped(rgb, size=300, pitch_px=4):
    """``rgb`` with darker ridge valleys every ``pitch_px`` rows (8 m at MPP 2)."""
    img = _image(rgb, size).astype(np.int16)
    img[::pitch_px] -= 40
    return np.clip(img, 0, 255).astype(np.uint8)


def _low(ds):
    return [8.0] * len(ds)


def _block_among_fields(det, size=300):
    veg = _veg(AGRI_CLS, size)
    _supported(veg, det)
    return veg


def test_ridge_pattern_scores_stripes_not_flat_or_split_roofs():
    det = _det(150, 150, 60, 40)
    assert AGRI.ridge_pattern(_striped(TEAL_FILM), det["points"], MPP) >= \
        AGRI.RIDGE_MIN_SCORE
    assert AGRI.ridge_pattern(_image(TEAL_FILM, 300), det["points"], MPP) == 0.0
    # a gable roof: one light and one dark half, no repeat
    gable = _image(TEAL_FILM, 300)
    gable[150:] = (40, 70, 75)
    assert AGRI.ridge_pattern(gable, det["points"], MPP) < AGRI.RIDGE_FIELD_SCORE


def test_ridged_teal_block_among_fields_is_greenhouse():
    # squat (aspect 1.5), teal and read as roof: rules 1-4 keep it a building
    det = _det(150, 150, 60, 40)
    veg = _block_among_fields(det)
    img = _striped(TEAL_FILM)
    assert AGRI.classify_detection(det, veg, lambda: img, MPP, _low) == \
        (AGRI.KIND_GREENHOUSE, "ridged_roof")
    assert AGRI.classify_detection(det, veg, lambda: _image(TEAL_FILM, 300), MPP, _low)[0] \
        == AGRI.KIND_BUILDING


def test_ridged_block_needs_film_colour_low_height_and_fields():
    det = _det(150, 150, 60, 40)
    veg = _block_among_fields(det)
    painted = _striped(PALE_BLUE)
    assert AGRI.classify_detection(det, veg, lambda: painted, MPP, _low)[0] == \
        AGRI.KIND_BUILDING
    img = _striped(TEAL_FILM)
    tall = lambda ds: [40.0] * len(ds)     # an apartment slab's balconies
    assert AGRI.classify_detection(det, veg, lambda: img, MPP, tall)[0] == \
        AGRI.KIND_BUILDING
    town = _veg(BLD_CLS, 300)              # skylight strips in an industrial park
    assert AGRI.classify_detection(det, town, lambda: img, MPP, _low)[0] == \
        AGRI.KIND_BUILDING


def _fine_det(cx, cy, length_m, width_m, mpp):
    c = np.array([cx, cy], float)
    hl, hw = length_m / 2 / mpp, width_m / 2 / mpp
    return {"points": [(c + d).tolist() for d in ((-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw))],
            "center": c.tolist(), "area_m2": length_m * width_m,
            "length_m": length_m, "width_m": width_m}


def test_two_tunnels_at_a_block_edge_are_judged_by_the_block():
    # ZL18-like: 16 m across holds two 8 m ridges, too few to score alone;
    # the box sits at the block's edge, fields on its other side
    mpp, size = 0.5, 400
    img = _striped(TEAL_FILM, size, pitch_px=16)
    img[:168] = (150, 120, 80)                             # field above the block
    det = _fine_det(200, 184, 60, 16, mpp)                 # rows 168-200
    veg = _block_among_fields(det, size)
    assert AGRI.ridge_pattern(img, det["points"], mpp) < AGRI.RIDGE_FIELD_SCORE
    assert AGRI.classify_detection(det, veg, lambda: img, mpp, _low) == \
        (AGRI.KIND_GREENHOUSE, "ridged_roof")
    # at ZL16 a block around a small box is as likely a row of houses
    coarse = _det(150, 150, 60, 8)
    assert AGRI.classify_detection(coarse, _block_among_fields(coarse), lambda: _striped(TEAL_FILM),
                                   MPP, _low)[0] == AGRI.KIND_BUILDING


def test_seeds_count_as_field_neighbours():
    # a belt with no farmland in sight: nothing seeds the field by itself
    size = 400
    veg = np.full((size, size), BLD_CLS, np.int8)
    dets = [_det(150, 40 + 15 * k, 60, 10) for k in range(4)]
    img = _image(GREY_FILM, size)
    kinds = [k for k, _ in AGRI.classify_detections(dets, veg, lambda: img, MPP, _low)]
    assert AGRI.KIND_GREENHOUSE not in kinds
    # greenhouses an earlier pass found next to it do
    seeds = [_det(110, 40, 60, 10), _det(110, 55, 60, 10)]
    kinds = [k for k, _ in AGRI.classify_detections(dets, veg, lambda: img, MPP, _low,
                                                    seeds=seeds)]
    assert kinds.count(AGRI.KIND_GREENHOUSE) == len(dets)


def test_no_solar_greenhouses_east_of_china():
    assert AGRI.uses_solar_greenhouses(118.7)        # Shouguang
    assert not AGRI.uses_solar_greenhouses(128.9)    # Gimhae
    det = _det(500, 500, 90, 14)
    det["tunnel_axis"] = "long"
    assert {p["kind"] for p in AGRI.greenhouse_pieces(det, MPP, solar=False)} == \
        {"tunnels"}


def test_fine_textures_accept_an_unbuilt_belt_without_fields():
    # ZL18-like 0.5 m/px: a block between a river and a road, no field in
    # sight; hardly anything built around it is enough, streets are not
    mpp, size = 0.5, 400
    c = np.array([200.0, 200.0])
    hl, hw = 60 / 2 / mpp, 40 / 2 / mpp
    det = {"points": [(c + d).tolist() for d in ((-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw))],
           "center": c.tolist(), "area_m2": 2400.0, "length_m": 60.0, "width_m": 40.0}
    img = _striped(TEAL_FILM, size, pitch_px=16)       # 8 m pitch
    river = np.full((size, size), 6, np.int8)          # water all round
    _supported(river, det)
    assert AGRI.classify_detection(det, river, lambda: img, mpp, _low) == \
        (AGRI.KIND_GREENHOUSE, "ridged_roof")
    town = np.full((size, size), AGRI.CLASS_DEVELOPED, np.int8)
    _supported(town, det)
    assert AGRI.classify_detection(det, town, lambda: img, mpp, _low)[0] == \
        AGRI.KIND_BUILDING
