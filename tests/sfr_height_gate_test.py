"""Height gate on asset selection and the neighbourhood height cap."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_SFR_Building_Overlay as BLD  # noqa: E402
import O4_SFR_Height_Priors as HP  # noqa: E402


def _asset(path, height_m):
    return {
        "kind": "object",
        "path": path,
        "bounds_m": (-14.0, 14.0, -14.0, 14.0),
        "height_m": height_m,
        "source": "test",
    }


class AssetHeightLimitTests(unittest.TestCase):
    def test_one_storey_or_quarter_over_detection(self):
        prior = HP.height_prior("europe", 4)
        self.assertAlmostEqual(HP.asset_height_limit_m(6.0, prior), 9.5)
        self.assertAlmostEqual(HP.asset_height_limit_m(16.0, prior), 20.0)

    def test_never_above_hard_ceiling(self):
        prior = HP.height_prior("europe", 1)       # hard ceiling 15 m
        self.assertEqual(HP.asset_height_limit_m(30.0, prior), 15.0)

    def test_missing_height_falls_back_to_regional_class_prior(self):
        prior = HP.height_prior("africa", 2)
        limit = HP.asset_height_limit_m(None, prior)
        self.assertAlmostEqual(limit, max(prior.p95_m, prior.median_m + 3.5))


class SelectorHeightGateTests(unittest.TestCase):
    def _select(self, detection, assets):
        table = BLD._build_yolo_object_candidate_index(
            {BLD.BLD_CLASS_MEDIUM: assets}
        )
        poly = np.array([[35, 35], [65, 35], [65, 65], [35, 65]], np.int32)
        return BLD._select_yolo_object_candidate(
            table, detection, poly, 50, 50, 0.0, 1.0, skip_occupancy=True
        )

    def _detection(self, **extra):
        det = {"length_m": 30.0, "width_m": 30.0, "area_m2": 900.0,
               "placement_class": BLD.BLD_CLASS_MEDIUM}
        det.update(extra)
        return det

    def test_tall_asset_rejected_short_one_kept(self):
        det = self._detection(height_m=6.0, height_gate_max_m=9.5)
        selected, status = self._select(
            det, [_asset("tall.obj", 19.2), _asset("low.obj", 6.4)])
        self.assertEqual(status, "selected")
        self.assertEqual(selected["asset"]["path"], "low.obj")

    def test_only_tall_assets_reports_height_reject(self):
        det = self._detection(height_m=6.0, height_gate_max_m=9.5)
        selected, status = self._select(det, [_asset("tall.obj", 19.2)])
        self.assertIsNone(selected)
        self.assertEqual(status, "height_reject")

    def test_unknown_asset_height_is_not_gated(self):
        det = self._detection(height_m=6.0, height_gate_max_m=9.5)
        selected, status = self._select(det, [_asset("anon.obj", None)])
        self.assertEqual(status, "selected")

    def test_no_gate_keeps_legacy_choice(self):
        selected, status = self._select(
            self._detection(), [_asset("tall.obj", 19.2)])
        self.assertEqual(status, "selected")

    def test_assign_gates_uses_prior_region(self):
        dets = [self._detection(height_m=6.0), self._detection()]
        BLD._assign_height_gates(dets, "europe")
        self.assertAlmostEqual(dets[0]["height_gate_max_m"], 9.5)
        self.assertIn("height_gate_max_m", dets[1])


class NeighbourhoodCapTests(unittest.TestCase):
    def _street(self, outlier_h, outlier_area):
        pts = [(i * 15.0, 0.0) for i in range(8)]
        heights = [6.0] * 8
        areas = [120.0] * 8
        heights[3] = outlier_h
        areas[3] = outlier_area
        return pts, heights, areas

    def test_small_house_capped_ten_metres_over_neighbourhood(self):
        caps = HP.neighbourhood_context_caps(*self._street(30.0, 120.0))
        self.assertAlmostEqual(caps[3], 16.0)

    def test_large_footprint_not_capped(self):
        caps = HP.neighbourhood_context_caps(*self._street(30.0, 2500.0))
        self.assertIsNone(caps[3])

    def test_sparse_neighbourhood_not_capped(self):
        caps = HP.neighbourhood_context_caps(
            [(0, 0), (500, 0), (1000, 0)], [30.0, 6.0, 6.0], [100] * 3)
        self.assertEqual(caps, [None, None, None])

    def test_overlay_applies_cap_to_heightnet_detections(self):
        dets = []
        for i in range(8):
            dets.append({"center": (i * 15, 0), "area_m2": 120.0,
                         "height_m": 30.0 if i == 3 else 6.0,
                         "height_source": "heightnet"})
        n = BLD._apply_height_context_caps(dets, 1.0)
        self.assertEqual(n, 1)
        self.assertAlmostEqual(dets[3]["height_m"], 16.0)
        self.assertEqual(dets[3]["height_adjustment"], "context_cap")
        self.assertEqual(dets[0]["height_m"], 6.0)


if __name__ == "__main__":
    unittest.main()
