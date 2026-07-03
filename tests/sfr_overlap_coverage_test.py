"""Tests for YOLO overlap removal.

Covers the zero-threshold smallest-first drop (_suppress_overlapping_yolo_detections),
the per-overlap containment rule (pair_rule), the tile-wide cross-texture dedup,
and object selection over occupied detections.
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


class DropModeTests(unittest.TestCase):
    def test_no_min_drops_any_overlap(self):
        # Zero thresholds (coverage_threshold=0, min_overlap_m2=0) drop on ANY
        # overlap: B sits almost on top of A, so the larger of the pair is dropped.
        dets = [
            _rect_detection(0, 0, 40, 40, det_id="A"),
            _rect_detection(1, 1, 41, 41, det_id="B"),
        ]
        kept, dropped = BLD._suppress_overlapping_yolo_detections(
            dets,
            coverage_threshold=0.0,
            min_overlap_m2=0.0,
            m_per_px=1.0,
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, 1)

    def test_non_overlapping_all_kept(self):
        # Disjoint detections are all kept regardless of the zero thresholds.
        dets = [
            _rect_detection(0, 0, 40, 40, det_id="A"),
            _rect_detection(100, 100, 140, 140, det_id="B"),
        ]
        kept, dropped = BLD._suppress_overlapping_yolo_detections(
            dets,
            coverage_threshold=0.0,
            min_overlap_m2=0.0,
            m_per_px=1.0,
        )
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, 0)

    def test_smallest_first_keeps_small(self):
        # Smallest-first: a small detection overlapping a large one survives and
        # the large overlapper is dropped.
        small = _rect_detection(20, 20, 30, 30, det_id="small")
        large = _rect_detection(0, 0, 40, 40, det_id="large")
        kept, dropped = BLD._suppress_overlapping_yolo_detections(
            [large, small],
            coverage_threshold=0.0,
            min_overlap_m2=0.0,
            m_per_px=1.0,
        )
        self.assertEqual([d["id"] for d in kept], ["small"])
        self.assertEqual(dropped, 1)


class CrossTextureDedupTests(unittest.TestCase):
    """Tile-wide vector dedup removes overlaps that per-texture removal misses."""

    def _obj(self, dlon, dlat, path="o4sfr/x/b.obj", heading=0.0):
        # placements are (lon, lat, heading, path); tile origin (24,118).
        return (118.0 + dlon, 24.0 + dlat, heading, path)

    def test_seam_duplicate_dropped(self):
        # Two identical ~10x10 m objects at the same spot (seam duplicate).
        import O4_SFR_Building_Overlay as B
        b = B
        # 10x10 footprint via OBJ_DIMS fallback path: stub bounds resolver.
        orig = b._bounds_for_object_path
        b._bounds_for_object_path = lambda p: (-5.0, 5.0, -5.0, 5.0)
        try:
            a = self._obj(0.0010, 0.0010)
            dup = self._obj(0.0010, 0.0010)  # exact same spot
            far = self._obj(0.0050, 0.0050)  # ~hundreds of m away
            kept, dropped = b._dedupe_overlapping_placements([a, dup, far], 24, 118)
            self.assertEqual(dropped, 1)
            self.assertEqual(len(kept), 2)
        finally:
            b._bounds_for_object_path = orig

    def test_non_overlapping_all_kept(self):
        import O4_SFR_Building_Overlay as b
        orig = b._bounds_for_object_path
        b._bounds_for_object_path = lambda p: (-5.0, 5.0, -5.0, 5.0)
        try:
            objs = [self._obj(0.0010, 0.0010), self._obj(0.0030, 0.0030),
                    self._obj(0.0050, 0.0050)]
            kept, dropped = b._dedupe_overlapping_placements(objs, 24, 118)
            self.assertEqual(dropped, 0)
            self.assertEqual(len(kept), 3)
        finally:
            b._bounds_for_object_path = orig

    def test_corner_graze_survives_partial_stack_drops(self):
        # Adjacent distinct buildings whose 10x10 footprints clip by ~1 m2
        # (1% of the smaller) survive the graze tolerance; a 60% partial
        # stack still drops. frac=0 restores drop-on-any-overlap.
        import O4_SFR_Building_Overlay as b
        orig = b._bounds_for_object_path
        b._bounds_for_object_path = lambda p: (-5.0, 5.0, -5.0, 5.0)
        try:
            mlat = 110540.0
            import math
            mlon = 111320.0 * math.cos(math.radians(24.5))
            base = self._obj(0.0010, 0.0010)
            graze = self._obj(0.0010 + 9.0 / mlon, 0.0010 + 9.0 / mlat)
            kept, dropped = b._dedupe_overlapping_placements(
                [base, graze], 24, 118, overlap_frac=0.20)
            self.assertEqual(dropped, 0)
            stack = self._obj(0.0010 + 4.0 / mlon, 0.0010)  # 60% overlap
            kept, dropped = b._dedupe_overlapping_placements(
                [base, stack], 24, 118, overlap_frac=0.20)
            self.assertEqual(dropped, 1)
            kept, dropped = b._dedupe_overlapping_placements(
                [base, graze], 24, 118, overlap_frac=0.0)
            self.assertEqual(dropped, 1)
        finally:
            b._bounds_for_object_path = orig


class ConfigPropagationTests(unittest.TestCase):
    """Guards the mechanism the GUI single-tile build fix relies on:
    a per-tile config value must override the global default for a bool
    that round-trips through the config file."""

    def test_per_tile_cfg_overrides_global_default(self):
        import tempfile, os
        import O4_Config_Utils as CFG

        old_global = CFG.global_sfr_bld_yolo_enabled
        try:
            CFG.global_sfr_bld_yolo_enabled = True
            tile = CFG.Tile(12, 34, "")
            self.assertTrue(tile.sfr_bld_yolo_enabled)  # global default

            with tempfile.TemporaryDirectory() as tmp:
                cfg_path = os.path.join(tmp, "tile.cfg")
                with open(cfg_path, "w") as f:
                    f.write("sfr_bld_yolo_enabled=False\n")
                tile.read_from_config(config_file=cfg_path)

            self.assertIsInstance(tile.sfr_bld_yolo_enabled, bool)
            self.assertFalse(tile.sfr_bld_yolo_enabled)
        finally:
            CFG.global_sfr_bld_yolo_enabled = old_global


class ObjectSelectionOccupancyTests(unittest.TestCase):
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

    def test_occupied_detection_places_nothing(self):
        # Min coverage is gated against the full detection area: an occupied
        # detection admits no smaller stand-in.
        (table, detection, yolo_poly, static, spacing, scratch,
         static_int, spacing_int) = self._setup()
        selected, status = BLD._select_yolo_object_candidate(
            table, detection, yolo_poly, 50, 50, 0.0, 1.0,
            static_occ_mask=static,
            building_spacing_mask=spacing,
            scratch_mask=scratch,
            static_occ_integral=static_int,
            spacing_occ_integral=spacing_int,
        )
        self.assertIsNone(selected)
        self.assertNotEqual(status, "selected")

    def test_skip_occupancy_ignores_occupied_mask(self):
        # With skip_occupancy the dense occupancy mask is ignored entirely, so
        # the highest-coverage asset that fits the detection outline wins even
        # though its footprint overlaps already-placed buildings. This is the
        # default fast path; tile-wide dedup resolves the overlaps afterwards.
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


class PairOverlapRuleTests(unittest.TestCase):
    """Per-overlap containment rule (pair_rule=True)."""

    KW = dict(
        coverage_threshold=0.0,
        min_overlap_m2=0.0,
        m_per_px=1.0,
        pair_rule=True,
        containment_frac=0.35,
        explain_frac=0.60,
    )

    def test_big_survives_and_evicts_low_conf_contained_small(self):
        big = _rect_detection(0, 0, 60, 60, confidence=0.6, det_id="big")
        # Small box inside the big one explains only ~3% of its ground and
        # does not beat its confidence -> roof furniture, evicted.
        small = _rect_detection(5, 5, 15, 15, confidence=0.4, det_id="small")
        kept, dropped = BLD._suppress_overlapping_yolo_detections(
            [big, small], **self.KW
        )
        self.assertEqual({d["id"] for d in kept}, {"big"})
        self.assertEqual(dropped, 1)

    def test_contained_small_with_higher_conf_survives_alongside(self):
        big = _rect_detection(0, 0, 60, 60, confidence=0.6, det_id="big")
        small = _rect_detection(5, 5, 15, 15, confidence=0.9, det_id="small")
        kept, dropped = BLD._suppress_overlapping_yolo_detections(
            [big, small], **self.KW
        )
        self.assertEqual({d["id"] for d in kept}, {"big", "small"})
        self.assertEqual(dropped, 0)

    def test_cluster_explaining_big_still_drops_big(self):
        # Four smalls tile the big box completely -> they ARE the buildings,
        # the big merged box is dropped exactly as before.
        smalls = [
            _rect_detection(0, 0, 20, 20, confidence=0.5, det_id="s1"),
            _rect_detection(20, 0, 40, 20, confidence=0.5, det_id="s2"),
            _rect_detection(0, 20, 20, 40, confidence=0.5, det_id="s3"),
            _rect_detection(20, 20, 40, 40, confidence=0.5, det_id="s4"),
        ]
        big = _rect_detection(0, 0, 40, 40, confidence=0.9, det_id="big")
        kept, dropped = BLD._suppress_overlapping_yolo_detections(
            smalls + [big], **self.KW
        )
        self.assertEqual({d["id"] for d in kept}, {"s1", "s2", "s3", "s4"})
        self.assertEqual(dropped, 1)

    def test_sliver_contact_keeps_smaller_wins(self):
        # Adjacent buildings whose OBBs overlap by a thin strip: the earlier
        # (smaller) detection still wins, the later one is dropped.
        det_a = _rect_detection(0, 0, 40, 40, confidence=0.5, det_id="A")
        det_b = _rect_detection(38, 0, 80, 42, confidence=0.9, det_id="B")
        kept, dropped = BLD._suppress_overlapping_yolo_detections(
            [det_a, det_b], **self.KW
        )
        self.assertEqual({d["id"] for d in kept}, {"A"})
        self.assertEqual(dropped, 1)

    def test_pair_rule_off_restores_blanket_smaller_wins(self):
        big = _rect_detection(0, 0, 60, 60, confidence=0.6, det_id="big")
        small = _rect_detection(5, 5, 15, 15, confidence=0.4, det_id="small")
        kw = dict(self.KW)
        kw["pair_rule"] = False
        kept, dropped = BLD._suppress_overlapping_yolo_detections(
            [big, small], **kw
        )
        self.assertEqual({d["id"] for d in kept}, {"small"})
        self.assertEqual(dropped, 1)


if __name__ == "__main__":
    unittest.main()
