"""Tests for coverage-preserving YOLO overlap removal.

Covers the three levers added to keep ground coverage while removing overlaps:
  * marginal-coverage greedy keep  (_suppress_overlapping_yolo_marginal)
  * occupancy-aware facade clipping (_clip_facade_to_free)
  * free-area downsizing at placement (_select_yolo_object_candidate)
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_SFR_Building_Overlay as BLD


def _rect_detection(x1, y1, x2, y2, confidence=0.5, det_id=None):
    """Axis-aligned OBB detection dict with metre == pixel (m_per_px=1)."""
    pts = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
    )
    det = {
        "points": pts,
        "area_m2": float(abs(x2 - x1) * abs(y2 - y1)),
        "confidence": float(confidence),
    }
    if det_id is not None:
        det["id"] = det_id
    return det


class MarginalKeepTests(unittest.TestCase):
    def test_redundant_overlap_dropped_distinct_kept(self):
        det_a = _rect_detection(0, 0, 40, 40, confidence=0.9, det_id="A")
        # B sits almost entirely on top of A -> adds little new ground.
        det_b = _rect_detection(2, 2, 42, 42, confidence=0.8, det_id="B")
        # C covers a separate block -> must be kept.
        det_c = _rect_detection(100, 0, 140, 40, confidence=0.7, det_id="C")

        kept, dropped = BLD._suppress_overlapping_yolo_detections(
            [det_a, det_b, det_c],
            keep_mode="marginal",
            keep_min_new_frac=0.25,
            m_per_px=1.0,
            img_w=200,
            img_h=200,
        )
        kept_ids = {d["id"] for d in kept}
        self.assertEqual(kept_ids, {"A", "C"})
        self.assertEqual(dropped, 1)

    def test_half_overlap_gap_filler_kept(self):
        det_a = _rect_detection(0, 0, 40, 40, confidence=0.9, det_id="A")
        # D overlaps A by ~half -> still adds ~50% new ground, keep it.
        det_d = _rect_detection(20, 0, 60, 40, confidence=0.8, det_id="D")

        kept, dropped = BLD._suppress_overlapping_yolo_detections(
            [det_a, det_d],
            keep_mode="marginal",
            keep_min_new_frac=0.25,
            m_per_px=1.0,
            img_w=200,
            img_h=200,
        )
        self.assertEqual({d["id"] for d in kept}, {"A", "D"})
        self.assertEqual(dropped, 0)

    def test_zero_threshold_keeps_everything(self):
        dets = [
            _rect_detection(0, 0, 40, 40, det_id="A"),
            _rect_detection(1, 1, 41, 41, det_id="B"),
        ]
        kept, dropped = BLD._suppress_overlapping_yolo_detections(
            dets,
            keep_mode="marginal",
            keep_min_new_frac=0.0,
            m_per_px=1.0,
            img_w=100,
            img_h=100,
        )
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, 0)

    def test_drop_mode_is_unchanged_passthrough(self):
        # coverage_threshold <= 0 in legacy drop mode keeps everything.
        dets = [
            _rect_detection(0, 0, 40, 40, det_id="A"),
            _rect_detection(1, 1, 41, 41, det_id="B"),
        ]
        kept, dropped = BLD._suppress_overlapping_yolo_detections(
            dets,
            keep_mode="drop",
            coverage_threshold=0.0,
            m_per_px=1.0,
        )
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, 0)


class FacadeClipTests(unittest.TestCase):
    def test_clip_excludes_occupied_region(self):
        # Detection covers x[0,40] y[0,40]; left half x[0,20] already occupied.
        yolo_poly = np.array(
            [[0, 0], [40, 0], [40, 40], [0, 40]], dtype=np.int32
        )
        occ = np.zeros((100, 100), dtype=np.uint8)
        occ[0:40, 0:20] = 1  # rows, cols -> left half occupied

        ring = BLD._clip_facade_to_free(yolo_poly, (None, occ), min_free_px=10.0)
        self.assertIsNotNone(ring)

        # Rasterise the clipped ring and confirm it does not touch occupancy.
        clip_mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.fillPoly(clip_mask, [np.int32(ring)], 1)
        self.assertEqual(int(cv2.countNonZero(cv2.bitwise_and(clip_mask, occ))), 0)
        # And it still covers a meaningful chunk of the free right half.
        self.assertGreater(int(cv2.countNonZero(clip_mask)), 200)

    def test_fully_occupied_returns_none(self):
        yolo_poly = np.array(
            [[0, 0], [40, 0], [40, 40], [0, 40]], dtype=np.int32
        )
        occ = np.zeros((100, 100), dtype=np.uint8)
        occ[0:41, 0:41] = 1  # whole detection occupied
        ring = BLD._clip_facade_to_free(yolo_poly, (None, occ), min_free_px=10.0)
        self.assertIsNone(ring)


class FreeAreaDownsizeTests(unittest.TestCase):
    def _setup(self):
        big = {
            "kind": "object",
            "path": "big.obj",
            "bounds_m": (-14.0, 14.0, -14.0, 14.0),  # 28 x 28 footprint
            "source": "test",
        }
        small = {
            "kind": "object",
            "path": "small.obj",
            "bounds_m": (-9.0, 9.0, -9.0, 9.0),  # 18 x 18 footprint
            "source": "test",
        }
        table = BLD._build_yolo_object_candidate_index(
            {BLD.BLD_CLASS_MEDIUM: [big, small]}
        )
        # 30 x 30 detection centred at (50, 50) -> spans [35, 65].
        yolo_poly = np.array(
            [[35, 35], [65, 35], [65, 65], [35, 65]], dtype=np.int32
        )
        detection = {
            "length_m": 30.0,
            "width_m": 30.0,
            "area_m2": 900.0,
            "placement_class": BLD.BLD_CLASS_MEDIUM,
        }
        # Occupy the whole detection bbox except a central hole the small asset
        # fits into; the big asset's footprint overlaps the occupied ring.
        spacing = np.zeros((128, 128), dtype=np.uint8)
        spacing[35:65, 35:65] = 1
        spacing[40:60, 40:60] = 0  # 20x20 free hole the small asset fits into
        static = np.zeros((128, 128), dtype=np.uint8)
        scratch = np.zeros((128, 128), dtype=np.uint8)
        static_int = cv2.integral(static)
        spacing_int = cv2.integral(spacing)
        return (table, detection, yolo_poly, static, spacing, scratch,
                static_int, spacing_int)

    def test_downsize_off_places_nothing(self):
        (table, detection, yolo_poly, static, spacing, scratch,
         static_int, spacing_int) = self._setup()
        selected, status = BLD._select_yolo_object_candidate(
            table, detection, yolo_poly, 50, 50, 0.0, 1.0,
            static_occ_mask=static,
            building_spacing_mask=spacing,
            scratch_mask=scratch,
            static_occ_integral=static_int,
            spacing_occ_integral=spacing_int,
            freearea_downsize=False,
        )
        self.assertIsNone(selected)
        self.assertNotEqual(status, "selected")

    def test_downsize_on_places_small_asset(self):
        (table, detection, yolo_poly, static, spacing, scratch,
         static_int, spacing_int) = self._setup()
        selected, status = BLD._select_yolo_object_candidate(
            table, detection, yolo_poly, 50, 50, 0.0, 1.0,
            static_occ_mask=static,
            building_spacing_mask=spacing,
            scratch_mask=scratch,
            static_occ_integral=static_int,
            spacing_occ_integral=spacing_int,
            freearea_downsize=True,
        )
        self.assertEqual(status, "selected")
        self.assertEqual(selected["asset"]["path"], "small.obj")

    def test_skip_occupancy_ignores_occupied_mask(self):
        # With skip_occupancy the dense occupancy mask is ignored entirely, so
        # the highest-coverage asset that fits the detection outline wins even
        # though its footprint overlaps already-placed buildings. This is the
        # fast path used under no_overlap_removal (no per-candidate fillPoly).
        (table, detection, yolo_poly, static, spacing, scratch,
         static_int, spacing_int) = self._setup()
        selected, status = BLD._select_yolo_object_candidate(
            table, detection, yolo_poly, 50, 50, 0.0, 1.0,
            static_occ_mask=static,
            building_spacing_mask=spacing,
            scratch_mask=scratch,
            static_occ_integral=static_int,
            spacing_occ_integral=spacing_int,
            skip_occupancy=True,
        )
        self.assertEqual(status, "selected")
        self.assertEqual(selected["asset"]["path"], "big.obj")


if __name__ == "__main__":
    unittest.main()
