"""Tests for grazing detections resolved into merged or touching facades
(O4_Bld_Overlap_Facades + the graze_out path of overlap suppression)."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_Building_Overlay as BLD
import O4_Bld_Overlap_Facades as OVL
from shapely.geometry import Polygon


def _rect(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)


def _det(x1, y1, x2, y2, confidence=0.5, det_id=None):
    return {
        "points": _rect(x1, y1, x2, y2).astype(np.float32),
        "area_m2": float(abs(x2 - x1) * abs(y2 - y1)),
        "confidence": float(confidence),
        "id": det_id,
    }


def _node(key, x1, y1, x2, y2, conf=0.5, h=10.0, cls=1):
    return OVL.Node(key, _rect(x1, y1, x2, y2), conf, (x2 - x1) * (y2 - y1), h, cls)


def _area(rings):
    return sum(Polygon(r).area for r in rings)


class SuppressionGrazeTests(unittest.TestCase):
    KW = dict(
        coverage_threshold=0.0, min_overlap_m2=0.0, m_per_px=1.0,
        pair_rule=True, containment_frac=0.35, explain_frac=0.60,
    )

    def test_sliver_keeps_both_and_records_pair(self):
        a = _det(0, 0, 60, 60, det_id="a")
        b = _det(58, 0, 140, 60, det_id="b")
        edges = []
        kept, dropped = BLD._suppress_overlapping_yolo_detections([a, b], graze_out=edges, **self.KW)
        self.assertEqual({d["id"] for d in kept}, {"a", "b"})
        self.assertEqual(dropped, 0)
        self.assertEqual(len(edges), 1)
        self.assertEqual({edges[0][0]["id"], edges[0][1]["id"]}, {"a", "b"})

    def test_without_graze_out_bigger_still_drops(self):
        a = _det(0, 0, 60, 60, det_id="a")
        b = _det(58, 0, 140, 60, det_id="b")
        kept, dropped = BLD._suppress_overlapping_yolo_detections([a, b], **self.KW)
        self.assertEqual({d["id"] for d in kept}, {"a"})
        self.assertEqual(dropped, 1)

    def test_containment_rule_unchanged(self):
        big = _det(0, 0, 60, 60, confidence=0.6, det_id="big")
        small = _det(5, 5, 15, 15, confidence=0.4, det_id="small")
        edges = []
        kept, _ = BLD._suppress_overlapping_yolo_detections([big, small], graze_out=edges, **self.KW)
        self.assertEqual({d["id"] for d in kept}, {"big"})
        self.assertEqual(edges, [])


class ResolveTests(unittest.TestCase):
    def test_similar_heights_merge_into_one_ring(self):
        nodes = {"a": _node("a", 0, 0, 60, 60, h=10.0), "b": _node("b", 58, 0, 140, 60, h=13.0)}
        out = OVL.resolve(nodes, [("a", "b")], 1.0, merge_dh_m=6.0, merge_max=3, conf_tie=0.05)
        self.assertEqual(out["b"].kind, "merged")      # bigger box represents the group
        self.assertEqual(out["a"].kind, "member")
        self.assertEqual(len(out["b"].rings), 1)
        self.assertAlmostEqual(_area(out["b"].rings), 140 * 60, delta=5)

    def test_different_heights_split_higher_conf_keeps_overlap(self):
        nodes = {
            "a": _node("a", 0, 0, 60, 60, conf=0.9, h=10.0),
            "b": _node("b", 50, 0, 140, 60, conf=0.4, h=40.0),
        }
        out = OVL.resolve(nodes, [("a", "b")], 1.0, merge_dh_m=6.0, merge_max=3, conf_tie=0.05)
        self.assertEqual(out["a"].kind, "unchanged")
        self.assertEqual(out["b"].kind, "split")
        self.assertAlmostEqual(_area(out["b"].rings), 80 * 60, delta=5)
        trimmed = Polygon(out["b"].rings[0])
        self.assertLess(trimmed.intersection(Polygon(_rect(0, 0, 60, 60))).area, 1.0)

    def test_near_tie_cuts_down_the_middle(self):
        nodes = {
            "a": _node("a", 0, 0, 60, 60, conf=0.50, h=10.0),
            "b": _node("b", 50, 0, 140, 60, conf=0.52, h=40.0),
        }
        out = OVL.resolve(nodes, [("a", "b")], 1.0, merge_dh_m=6.0, merge_max=3, conf_tie=0.05)
        self.assertEqual(out["a"].kind, "split")
        self.assertEqual(out["b"].kind, "split")
        ra, rb = Polygon(out["a"].rings[0]), Polygon(out["b"].rings[0])
        self.assertAlmostEqual(ra.area, 55 * 60, delta=5)
        self.assertAlmostEqual(rb.area, 85 * 60, delta=5)
        self.assertLess(ra.intersection(rb).area, 1.0)

    def test_unknown_height_never_merges(self):
        nodes = {"a": _node("a", 0, 0, 60, 60, h=None), "b": _node("b", 58, 0, 140, 60, h=10.0)}
        out = OVL.resolve(nodes, [("a", "b")], 1.0, merge_dh_m=6.0, merge_max=3, conf_tie=0.05)
        self.assertNotIn("merged", {o.kind for o in out.values()})

    def test_merge_cap_limits_chain(self):
        nodes = {k: _node(k, 50 * i, 0, 50 * i + 52, 50) for i, k in enumerate("abcde")}
        edges = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")]
        out = OVL.resolve(nodes, edges, 1.0, merge_dh_m=6.0, merge_max=2, conf_tie=0.05)
        for o in out.values():
            if o.kind == "merged":
                self.assertLessEqual(len(o.members), 2)
        # nothing overlaps once resolved
        rings = [Polygon(r) for o in out.values() if o.rings for r in o.rings]
        rings += [Polygon(nodes[k].points) for k, o in out.items() if o.kind == "unchanged"]
        for i in range(len(rings)):
            for j in range(i + 1, len(rings)):
                self.assertLess(rings[i].intersection(rings[j]).area, 1.0)

    def test_too_little_left_drops(self):
        # b loses most of itself to a stronger, much taller a
        nodes = {
            "a": _node("a", 0, 0, 100, 100, conf=0.9, h=10.0),
            "b": _node("b", 80, 0, 110, 100, conf=0.3, h=50.0),
        }
        out = OVL.resolve(nodes, [("a", "b")], 1.0, merge_dh_m=6.0, merge_max=3, conf_tie=0.05)
        self.assertEqual(out["b"].kind, "dropped")
        self.assertEqual(out["a"].kind, "unchanged")


class FacadeSpacingInsetTests(unittest.TestCase):
    def test_shared_edge_is_not_overlap(self):
        import cv2
        mask = np.zeros((100, 100), dtype=np.uint8)
        neighbour = np.int32([[40, 10], [70, 10], [70, 40], [40, 40]])
        cv2.fillPoly(mask, [neighbour], 1)
        box = np.int32([[10, 10], [40, 10], [40, 40], [10, 40]])   # shares x=40
        raw = BLD._poly_mask_overlap_frac(mask, box)
        inset = BLD._poly_mask_overlap_frac(
            mask, BLD._inset_poly_px(box, BLD.DIRECT_FACADE_SPACING_INSET_PX))
        self.assertGreater(raw, BLD.DIRECT_FACADE_SPACING_OVERLAP_FRAC)
        self.assertEqual(inset, 0.0)

    def test_real_intrusion_still_counts(self):
        import cv2
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.fillPoly(mask, [np.int32([[30, 10], [60, 10], [60, 40], [30, 40]])], 1)
        box = np.int32([[10, 10], [40, 10], [40, 40], [10, 40]])
        inset = BLD._poly_mask_overlap_frac(
            mask, BLD._inset_poly_px(box, BLD.DIRECT_FACADE_SPACING_INSET_PX))
        self.assertGreater(inset, BLD.DIRECT_FACADE_SPACING_OVERLAP_FRAC)


class TrimExistingTests(unittest.TestCase):
    @staticmethod
    def _mask(*rects):
        import cv2
        m = np.zeros((200, 200), dtype=np.uint8)
        for r in rects:
            cv2.fillPoly(m, [np.int32(_rect(*r))], 1)
        return m

    def test_edge_graze_is_trimmed_clear_of_scenery(self):
        import cv2
        scenery = self._mask((90, 20, 140, 80))          # overlaps the box's right 10 px
        box = _rect(20, 20, 100, 80)
        rings = BLD._trim_rings_against_mask(box, scenery, None, 1.0)
        self.assertIsNotNone(rings)
        placed = np.zeros_like(scenery)
        for r in rings:
            cv2.fillPoly(placed, [np.rint(r).astype(np.int32)], 1)
        self.assertEqual(int((placed & scenery).sum()), 0)
        self.assertGreater(placed.sum(), 0.8 * 80 * 60)

    def test_scenery_inside_box_still_blocks(self):
        scenery = self._mask((50, 40, 60, 50))
        self.assertIsNone(BLD._trim_rings_against_mask(_rect(20, 20, 100, 80), scenery, None, 1.0))

    def test_heavy_cover_still_blocks(self):
        scenery = self._mask((40, 0, 200, 200))
        self.assertIsNone(BLD._trim_rings_against_mask(_rect(20, 20, 100, 80), scenery, None, 1.0))

    def test_hard_blocker_still_blocks(self):
        scenery = self._mask((90, 20, 140, 80))
        water = self._mask((10, 70, 30, 90))
        self.assertIsNone(BLD._trim_rings_against_mask(_rect(20, 20, 100, 80), scenery, water, 1.0))


if __name__ == "__main__":
    unittest.main()
