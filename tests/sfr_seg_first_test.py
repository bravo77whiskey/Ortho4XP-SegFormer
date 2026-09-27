"""Segment-first placement: wall splits, rectangle tests, the guard."""

import sys
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_SFR_Footprint_Polygons as FP  # noqa: E402
import O4_SFR_Building_Overlay as BLD  # noqa: E402

MPP = 1.5
BUILDING = 8


def _veg(size=256):
    return np.zeros((size, size), np.int8)


def _box(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def _det(pts, area_m2=None):
    pts = np.asarray(pts, float)
    d = BLD._yolo_obb_detection_from_points(pts, 0.9, 3, 256, 256, MPP)
    if area_m2 is not None:
        d["area_m2"] = area_m2
    return d


class WallSplitTests(unittest.TestCase):
    def _roof(self, colours):
        rgb = np.zeros((60, 120, 3), np.uint8)
        mask = np.zeros((60, 120), bool)
        mask[10:50, 10:110] = True
        for (x0, x1), c in colours:
            rgb[10:50, x0:x1] = c
        return rgb, mask

    def test_two_roof_colours_split_at_the_line(self):
        rgb, mask = self._roof([((10, 60), (200, 60, 50)), ((60, 110), (90, 90, 95))])
        parts = FP.split_shared_walls(rgb, mask, MPP)
        self.assertEqual(len(parts), 2)
        xs = sorted(int(np.nonzero(p)[1].mean()) for p in parts)
        self.assertLess(abs(xs[0] - 35), 4)
        self.assertLess(abs(xs[1] - 85), 4)

    def test_uniform_roof_stays_whole(self):
        rgb, mask = self._roof([((10, 110), (150, 150, 150))])
        rng = np.random.default_rng(1)
        rgb = np.clip(rgb.astype(int) + rng.integers(-6, 7, rgb.shape), 0, 255).astype(np.uint8)
        self.assertEqual(len(FP.split_shared_walls(rgb, mask, MPP)), 1)

    def test_no_image_no_split(self):
        _rgb, mask = self._roof([((10, 110), (1, 1, 1))])
        self.assertEqual(len(FP.split_shared_walls(None, mask, MPP)), 1)


class TracePieceTests(unittest.TestCase):
    def test_rectangle_detected(self):
        m = np.zeros((100, 100), bool)
        cv2.fillPoly(m.view(np.uint8), [np.int32([[20, 30], [80, 25], [83, 60], [23, 65]])], 1)
        res = FP.trace_piece(m, MPP)
        self.assertIsNotNone(res["rect"])

    def test_l_shape_is_not_a_rectangle(self):
        m = np.zeros((100, 100), bool)
        m[20:80, 20:40] = True
        m[60:80, 20:80] = True
        res = FP.trace_piece(m, MPP)
        self.assertIsNone(res["rect"])
        self.assertEqual(len(res["rings"][0]), 6)


class SegFirstDetectionTests(unittest.TestCase):
    def test_off_is_identity(self):
        dets = [_det(_box(10, 10, 20, 20))]
        self.assertEqual(BLD._seg_first_detections(dets, _veg(), None, MPP, 256, 256, "off"), dets)

    def test_l_building_becomes_polygon_and_drops_nested_boxes(self):
        veg = _veg()
        veg[40:120, 40:70] = BUILDING
        veg[100:120, 40:140] = BUILDING       # L, ~3,500 m2
        nested = _det(_box(42, 42, 68, 70))   # a wing boxed by YOLO
        outside = _det(_box(200, 200, 215, 215))
        out = BLD._seg_first_detections([nested, outside], veg, None, MPP, 256, 256, "guarded")
        kinds = [d.get("seg_first") for d in out]
        self.assertIn("polygon", kinds)
        self.assertNotIn(nested, out)
        self.assertIn(outside, out)
        poly = next(d for d in out if d.get("seg_first") == "polygon")
        self.assertEqual(len(poly["seg_rings"][0]), 6)

    def test_rectangle_matched_by_yolo_keeps_the_yolo_box(self):
        veg = _veg()
        veg[40:100, 40:120] = BUILDING
        yolo = _det(_box(41, 41, 119, 99))
        out = BLD._seg_first_detections([yolo], veg, None, MPP, 256, 256, "guarded")
        self.assertEqual(out, [yolo])

    def test_unmatched_rectangle_becomes_a_detection(self):
        veg = _veg()
        veg[40:100, 40:120] = BUILDING
        out = BLD._seg_first_detections([], veg, None, MPP, 256, 256, "guarded")
        self.assertEqual([d.get("seg_first") for d in out], ["rect"])
        self.assertGreater(out[0]["area_m2"], 9000)

    def test_guard_leaves_terrace_blocks_to_yolo(self):
        veg = _veg()
        veg[40:100, 40:120] = BUILDING
        houses = [_det(_box(42 + 12 * i, 42, 52 + 12 * i, 56), area_m2=120.0) for i in range(5)]
        out = BLD._seg_first_detections(houses, veg, None, MPP, 256, 256, "guarded")
        self.assertEqual(out, houses)
        out_all = BLD._seg_first_detections(houses, veg, None, MPP, 256, 256, "all")
        self.assertEqual([d.get("seg_first") for d in out_all], ["rect"])

    def test_guard_skips_small_components(self):
        veg = _veg()
        veg[40:50, 40:55] = BUILDING          # ~340 m2
        self.assertEqual(BLD._seg_first_detections([], veg, None, MPP, 256, 256, "guarded"), [])
        self.assertEqual(len(BLD._seg_first_detections([], veg, None, MPP, 256, 256, "all")), 1)

    def test_mode_switch(self):
        with mock.patch.dict("os.environ", {"O4_SFR_BLD_SEG_FIRST": "Guarded"}):
            self.assertEqual(BLD._seg_first_mode(), "guarded")
        with mock.patch.dict("os.environ", {"O4_SFR_BLD_SEG_FIRST": "bogus"}):
            self.assertEqual(BLD._seg_first_mode(), "guarded")
        with mock.patch.dict("os.environ", {"O4_SFR_BLD_SEG_FIRST": "off"}):
            self.assertEqual(BLD._seg_first_mode(), "off")

    def test_guarded_is_the_default(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(BLD._seg_first_mode(), "guarded")

    def test_rough_outline_is_left_to_yolo(self):
        veg = _veg(300)
        for i in range(12):                  # ragged: 4.5 m x 6 m steps
            veg[40 + 3 * i:43 + 3 * i, 40:80 + 4 * i] = BUILDING
        res = FP.trace_piece(veg == BUILDING, MPP)
        self.assertGreater(res["irregularity"], FP.MAX_IRREGULARITY)
        yolo = _det(_box(200, 200, 215, 215))
        out = BLD._seg_first_detections([yolo], veg, None, MPP, 300, 300, "guarded")
        self.assertEqual(out, [yolo])

    def test_big_clean_steps_are_regular(self):
        m = np.zeros((300, 300), bool)
        for i in range(4):                   # stepped hall, 12 m x 24 m steps
            m[40 + 8 * i:48 + 8 * i, 40:60 + 16 * i] = True
        m[72:120, 40:108] = True
        res = FP.trace_piece(m, MPP)
        self.assertLessEqual(res["irregularity"], FP.MAX_IRREGULARITY)

    def test_clean_l_is_regular(self):
        m = np.zeros((200, 200), bool)
        m[20:180, 20:70] = True
        m[130:180, 20:180] = True
        res = FP.trace_piece(m, MPP)
        self.assertLessEqual(res["irregularity"], FP.MAX_IRREGULARITY)

    def test_blob_is_irregular(self):
        m = np.zeros((200, 200), np.uint8)
        cv2.ellipse(m, (100, 100), (70, 45), 20, 0, 360, 1, -1)
        res = FP.trace_piece(m.astype(bool), MPP)
        self.assertIsNone(res["rect"])
        self.assertGreater(res["irregularity"], FP.MAX_IRREGULARITY)


class GroundLookalikeTests(unittest.TestCase):
    def _scene(self, inside, outside, noise=0):
        rng = np.random.default_rng(0)
        rgb = np.zeros((80, 80, 3), np.int16)
        rgb[:] = outside
        piece = np.zeros((80, 80), bool)
        piece[25:55, 20:60] = True
        rgb[piece] = inside
        if noise:
            rgb += rng.integers(-noise, noise + 1, rgb.shape).astype(np.int16)
        return np.clip(rgb, 0, 255).astype(np.uint8), piece

    def test_clay_track_is_soil(self):
        rgb, piece = self._scene((150, 110, 80), (70, 110, 60))
        self.assertEqual(FP.looks_like_ground(rgb, piece, piece, MPP), (True, "soil"))

    def test_patch_matching_its_surroundings(self):
        rgb, piece = self._scene((90, 95, 85), (92, 96, 86))
        self.assertEqual(FP.looks_like_ground(rgb, piece, piece, MPP)[1], "no_contrast")

    def test_metal_roof_is_kept(self):
        rgb, piece = self._scene((190, 195, 200), (90, 100, 80))
        self.assertEqual(FP.looks_like_ground(rgb, piece, piece, MPP), (False, None))

    def test_ground_rejected_in_overlay(self):
        veg = _veg()
        veg[40:100, 40:120] = BUILDING
        rgb = np.zeros((256, 256, 3), np.uint8)
        rgb[:] = (70, 110, 60)
        rgb[40:100, 40:120] = (150, 110, 80)
        out = BLD._seg_first_detections([], veg, rgb, MPP, 256, 256, "guarded")
        self.assertEqual(out, [])
        with mock.patch.dict("os.environ", {"O4_SFR_BLD_SEG_FIRST_GROUND": "0"}):
            out = BLD._seg_first_detections([], veg, rgb, MPP, 256, 256, "guarded")
        self.assertEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
