"""Polygon footprints for large irregular roofs (O4_SFR_Footprint_Polygons)."""

import math
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_SFR_Footprint_Polygons as FP  # noqa: E402
import O4_SFR_Building_Overlay as BLD  # noqa: E402

M_PER_PX = 1.5


def _rot(pts, ang_deg, centre):
    a = math.radians(ang_deg)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return (np.asarray(pts, float) - centre) @ r.T + centre


def _mask(polys, size=400):
    m = np.zeros((size, size), np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
    return m


def _rings_mask(rings, size=400):
    return _mask(rings, size)


def _iou(a, b):
    return np.count_nonzero(a & b) / max(1, np.count_nonzero(a | b))


def _edge_angles(ring, ang_deg):
    """Edge directions relative to the OBB axis, folded to [0, 90)."""
    out = []
    ring = np.asarray(ring)
    for i in range(len(ring)):
        d = ring[(i + 1) % len(ring)] - ring[i]
        if np.hypot(*d) < 1e-6:
            continue
        a = (math.degrees(math.atan2(d[1], d[0])) - ang_deg) % 90.0
        out.append(min(a, 90.0 - a))
    return out


class FootprintTests(unittest.TestCase):
    def test_l_shape_is_carved_crisp_and_rotated(self):
        c = np.array([200.0, 200.0])
        ang = 27.0
        # L: 80 x 60 px OBB with a 40 x 30 px corner missing
        l_shape = np.array([[160, 170], [240, 170], [240, 200], [200, 200],
                            [200, 230], [160, 230]], float)
        l_rot = _rot(l_shape, ang, c)
        obb = _rot([[160, 170], [240, 170], [240, 230], [160, 230]], ang, c)
        truth = _mask([l_rot])
        res = FP.extract_footprint(truth, obb, M_PER_PX)
        self.assertIsNotNone(res)
        self.assertEqual(res["kind"], "rectilinear")
        self.assertEqual(len(res["rings"]), 1)
        self.assertEqual(len(res["rings"][0]), 6)
        self.assertGreater(_iou(_rings_mask(res["rings"]), truth), 0.9)
        self.assertLess(res["area_m2"], 0.85 * res["obb_area_m2"])
        # every edge runs along or across the OBB axis
        for a in _edge_angles(res["rings"][0], ang):
            self.assertLess(a, 1.0)

    def test_blobby_mask_edges_become_straight(self):
        rng = np.random.default_rng(3)
        truth = _mask([np.array([[140, 150], [260, 150], [260, 250],
                                 [140, 250]], float)])
        noisy = truth.copy()
        # ragged, rounded border like a 1/4-resolution segmentation
        noisy = cv2.GaussianBlur(noisy.astype(np.float32), (0, 0), 3.0)
        noisy = ((noisy + rng.normal(0, 0.08, noisy.shape)) > 0.5).astype(np.uint8)
        # a notch so the rectangle is irregular
        noisy[150:190, 140:180] = 0
        obb = np.array([[140, 150], [260, 150], [260, 250], [140, 250]], float)
        res = FP.extract_footprint(noisy, obb, M_PER_PX)
        self.assertEqual(res["kind"], "rectilinear")
        ring = res["rings"][0]
        self.assertLessEqual(len(ring), 8)
        for a in _edge_angles(ring, 0.0):
            self.assertLess(a, 1.0)

    def test_courtyard_is_split_into_hole_free_rings(self):
        outer = np.array([[120, 120], [280, 120], [280, 260], [120, 260]], float)
        m = _mask([outer])
        m[160:220, 170:230] = 0            # 60 x 60 px = 8,100 m2 courtyard
        res = FP.extract_footprint(m, outer, M_PER_PX)
        self.assertIsNotNone(res)
        self.assertGreaterEqual(len(res["rings"]), 2)
        drawn = _rings_mask(res["rings"])
        self.assertEqual(drawn[190, 200], 0)    # courtyard stays open
        self.assertGreater(_iou(drawn, m), 0.9)

    def test_missed_wing_is_added_but_facade_strip_is_not(self):
        box = np.array([[150, 150], [250, 150], [250, 200], [150, 200]], float)
        wing = np.array([[150, 200], [185, 200], [185, 214], [150, 214]], float)  # 21 m deep
        strip = np.array([[250, 150], [253, 150], [253, 200], [250, 200]], float)  # 4.5 m
        m = _mask([box, wing, strip])
        res = FP.extract_footprint(m, box, M_PER_PX)
        drawn = _rings_mask(res["rings"])
        self.assertEqual(drawn[208, 165], 1)     # wing kept
        self.assertEqual(drawn[175, 252], 0)     # facade strip dropped

    def test_growth_capped_to_carve_only(self):
        box = np.array([[150, 150], [250, 150], [250, 200], [150, 200]], float)
        bowl = np.array([[150, 150], [250, 150], [250, 330], [150, 330]], float)
        res = FP.extract_footprint(_mask([bowl]), box, M_PER_PX)
        self.assertLessEqual(res["area_m2"], FP.MAX_GROWTH * res["obb_area_m2"])

    def test_no_building_evidence_returns_none(self):
        box = np.array([[150, 150], [250, 150], [250, 200], [150, 200]], float)
        m = np.zeros((400, 400), np.uint8)
        m[150:160, 150:170] = 1
        self.assertIsNone(FP.extract_footprint(m, box, M_PER_PX))


class OverlayDecisionTests(unittest.TestCase):
    def _det(self, pts):
        pts = np.asarray(pts, float)
        e0 = np.hypot(*(pts[1] - pts[0])) * M_PER_PX
        e1 = np.hypot(*(pts[2] - pts[1])) * M_PER_PX
        return {"points": pts.tolist(), "area_m2": e0 * e1,
                "length_m": max(e0, e1), "width_m": min(e0, e1)}

    def test_rectangular_roof_keeps_normal_path(self):
        box = [[150, 150], [250, 150], [250, 200], [150, 200]]
        m = _mask([np.array(box, float)])
        self.assertIsNone(BLD._large_footprint_rings(self._det(box), m, M_PER_PX))

    def test_l_roof_takes_polygon_path(self):
        box = [[150, 150], [250, 150], [250, 230], [150, 230]]
        l_shape = np.array([[150, 150], [250, 150], [250, 180], [190, 180],
                            [190, 230], [150, 230]], float)
        rings = BLD._large_footprint_rings(self._det(box), _mask([l_shape]), M_PER_PX)
        self.assertIsNotNone(rings)

    def test_small_detection_ignored(self):
        box = [[150, 150], [165, 150], [165, 165], [150, 165]]
        m = _mask([np.array(box, float)])
        self.assertIsNone(BLD._large_footprint_rings(self._det(box), m, M_PER_PX))


if __name__ == "__main__":
    unittest.main()
