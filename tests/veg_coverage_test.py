"""Vegetation coverage: road widths, edge clipping, tree support, point trees."""

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import torch
except ImportError:  # pragma: no cover - exercised only in minimal envs
    torch = None

import O4_Forest_Assets as FOREST_ASSETS
import O4_Landcover_Inference as INF
import O4_Vegetation_Overlay as VEG

# A 0.01-degree texture, 1000 px square: ~1.1 m per pixel.
LAT_N, LAT_S, LON_W, LON_E = 22.61, 22.60, 120.30, 120.31
SIZE = 1000
MPP = (LAT_N - LAT_S) * 111320.0 / SIZE


def _ll(px, py):
    """(lat, lon) of image pixel (px, py)."""
    return (LAT_N - py / SIZE * (LAT_N - LAT_S), LON_W + px / SIZE * (LON_E - LON_W))


class RoadRasterTests(unittest.TestCase):
    def setUp(self):
        # Production default is no road clearance (scale 0); these tests
        # exercise the widths themselves.
        patcher = mock.patch.dict(os.environ, {"O4_VEG_ROAD_WIDTH_SCALE": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_default_is_no_road_removal(self):
        road = self._road((100, 500), (900, 500), half_m=4)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("O4_VEG_ROAD_WIDTH_SCALE", None)
            mask = VEG._rasterize_roads_typed([road], LAT_N, LAT_S, LON_W, LON_E, SIZE, SIZE, MPP)
        self.assertFalse(mask.any())

    def _road(self, a, b, **extra):
        return dict({"pts": [_ll(*a), _ll(*b)], "type": "network"}, **extra)

    def test_road_crossing_the_edge_keeps_its_direction(self):
        # From far off the left edge to inside: clamping the outside end to
        # x=0 used to bend the line into a steep wedge along the edge.
        road = self._road((-1000, 500), (100, 600), half_m=1)
        with mock.patch.dict(os.environ, {"O4_VEG_ROAD_CLIP": "1"}):
            fixed = VEG._rasterize_roads_typed([road], LAT_N, LAT_S, LON_W, LON_E, SIZE, SIZE, MPP)
        with mock.patch.dict(os.environ, {"O4_VEG_ROAD_CLIP": "0"}):
            clamped = VEG._rasterize_roads_typed([road], LAT_N, LAT_S, LON_W, LON_E, SIZE, SIZE, MPP)
        # True line at x=50 sits near y = 500 + 100 * 1050 / 1100 ≈ 595.
        self.assertTrue(fixed[595, 50])
        self.assertFalse(fixed[550, 50])
        self.assertTrue(clamped[550, 50])

    def test_half_width_comes_from_the_way(self):
        narrow = self._road((100, 500), (900, 500), half_m=2)
        wide = self._road((100, 500), (900, 500))   # table: 'network' = 8 m
        m_narrow = VEG._rasterize_roads_typed([narrow], LAT_N, LAT_S, LON_W, LON_E, SIZE, SIZE, MPP)
        m_wide = VEG._rasterize_roads_typed([wide], LAT_N, LAT_S, LON_W, LON_E, SIZE, SIZE, MPP)
        width = lambda m: int(m[:, 500].sum())
        self.assertAlmostEqual(width(m_narrow) * MPP, 4.0, delta=2.5)
        self.assertAlmostEqual(width(m_wide) * MPP, 16.0, delta=2.5)

    def test_width_scale_zero_removes_no_trees(self):
        road = self._road((100, 500), (900, 500), half_m=4)
        with mock.patch.dict(os.environ, {"O4_VEG_ROAD_WIDTH_SCALE": "0"}):
            none = VEG._rasterize_roads_typed([road], LAT_N, LAT_S, LON_W, LON_E, SIZE, SIZE, MPP)
        with mock.patch.dict(os.environ, {"O4_VEG_ROAD_WIDTH_SCALE": "0.5"}):
            half = VEG._rasterize_roads_typed([road], LAT_N, LAT_S, LON_W, LON_E, SIZE, SIZE, MPP)
        self.assertFalse(none.any())
        self.assertAlmostEqual(half[:, 500].sum() * MPP, 4.0, delta=2.5)

    def test_every_simheaven_class_has_a_width(self):
        import O4_Building_Overlay as BLD
        for _low, _high, road_type in BLD._XP_ROAD_SUBTYPE_RANGES:
            self.assertIn(road_type, VEG.SIMHEAVEN_ROAD_HALF_WIDTH_M)
        self.assertIn("network", VEG.SIMHEAVEN_ROAD_HALF_WIDTH_M)


class TreeSupportCacheTests(unittest.TestCase):
    def test_round_trip_and_shape_check(self):
        mask = np.random.default_rng(1).random((37, 53)) > 0.6
        with tempfile.TemporaryDirectory() as tmp:
            path = INF.tree_support_cache_path(os.path.join(tmp, "1_2_BI16_veg.npy"))
            self.assertTrue(path.endswith("1_2_BI16_vegtree.npy"))
            INF.save_tree_support(path, mask)
            self.assertTrue(np.array_equal(INF.load_tree_support(path, mask.shape), mask))
            self.assertIsNone(INF.load_tree_support(path, (53, 37)))
            self.assertIsNone(INF.load_tree_support(path + "_missing", mask.shape))


@unittest.skipIf(torch is None, "torch is not installed")
class TreeSupportInferenceTests(unittest.TestCase):
    def test_weak_tree_pixels(self):
        def logits_for(probs):
            p = np.full(9, (1.0 - sum(probs.values())) / (9 - len(probs)))
            for cls, v in probs.items():
                p[cls] = v
            return torch.tensor(np.log(p), dtype=torch.float32)

        # Left half: rangeland top (0.55), tree runner-up (0.32) -> support.
        # Right half: tree top but only 0.4 -> unclassified, still support.
        left = logits_for({INF.CLASS_RANGELAND: 0.55, INF.CLASS_TREE: 0.32})
        right = logits_for({INF.CLASS_TREE: 0.4, INF.CLASS_AGRICULTURE: 0.3})

        class Config:
            num_labels = 9

        class Output:
            def __init__(self, logits):
                self.logits = logits

        class FakeModel(torch.nn.Module):
            config = Config()

            def forward(self, pixel_values):
                b, _c, h, w = pixel_values.shape
                out = torch.empty((b, 9, h, w), dtype=pixel_values.dtype)
                out[..., : w // 2] = left.view(1, 9, 1, 1)
                out[..., w // 2:] = right.view(1, 9, 1, 1)
                return Output(out)

        old = (INF.segformer_patch_size, INF.segformer_overlap)
        try:
            INF.segformer_patch_size, INF.segformer_overlap = 32, 0
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            cls_map, weak = INF.run_inference(
                FakeModel(), torch.device("cpu"), image, tree_support=True
            )
            plain = INF.run_inference(FakeModel(), torch.device("cpu"), image)
        finally:
            INF.segformer_patch_size, INF.segformer_overlap = old
        self.assertTrue(np.array_equal(cls_map, plain))
        self.assertTrue((cls_map[2:-2, 2:12] == INF.CLASS_RANGELAND).all())
        self.assertTrue((cls_map[2:-2, 20:30] == -1).all())
        self.assertTrue(weak[2:-2, 2:12].all())
        self.assertTrue(weak[2:-2, 20:30].all())


class PointTreeTests(unittest.TestCase):
    def _kwargs(self):
        return dict(
            img_w=SIZE, img_h=SIZE, lat_n=LAT_N, lat_s=LAT_S, lon_w=LON_W, lon_e=LON_E,
            tile_lat=22, tile_lon=120, m_per_px=MPP, min_area_px=1.0, simplify_px=1.0,
            region=FOREST_ASSETS.climate_region(22.6, 120.3),
            rng=np.random.default_rng(0), density_override=1.0,
        )

    def _disc(self, mask, cx, cy, radius_m):
        import cv2
        cv2.circle(mask, (cx, cy), int(round(radius_m / MPP)), 1, -1)

    def test_small_blobs_become_point_trees(self):
        mask = np.zeros((SIZE, SIZE), np.uint8)
        self._disc(mask, 200, 200, 3.0)     # one street tree, ~28 m²
        self._disc(mask, 500, 500, 12.0)    # small clump, ~450 m²
        self._disc(mask, 800, 800, 30.0)    # park, ~2800 m² -> area fill
        with mock.patch.dict(os.environ, {"O4_VEG_POINT_TREES": "1"}):
            polys = VEG._process_dds_mask(mask, INF.CLASS_TREE, **self._kwargs())
        points = [p for p in polys if p[1] >= VEG.POINT_MODE]
        areas = [p for p in polys if p[1] < VEG.POINT_MODE]
        self.assertEqual(len(points), 2)
        self.assertEqual(len(areas), 1)
        counts = sorted(len(p[2]) for p in points)
        self.assertEqual(counts[0], 1)
        self.assertGreaterEqual(counts[1], 3)
        for _path, param, pts in points:
            self.assertEqual(param, 255 + VEG.POINT_MODE)
            for a, b in zip(pts, pts[1:]):
                d = math.hypot((a[0] - b[0]) * 111320 * math.cos(math.radians(22.6)),
                               (a[1] - b[1]) * 111320)
                self.assertGreater(d, VEG.POINT_TREE_SPACING_M - 1.5)

    def test_switch_off_keeps_area_fill(self):
        mask = np.zeros((SIZE, SIZE), np.uint8)
        self._disc(mask, 500, 500, 12.0)
        with mock.patch.dict(os.environ, {"O4_VEG_POINT_TREES": "0"}):
            polys = VEG._process_dds_mask(mask, INF.CLASS_TREE, **self._kwargs())
        self.assertEqual(len(polys), 1)
        self.assertLess(polys[0][1], VEG.LINE_MODE)

    def test_point_windings_are_not_closed(self):
        import io
        f = io.StringIO()
        VEG._write_point_windings(f, [(120.1, 22.1), (120.2, 22.2)])
        self.assertEqual(f.getvalue().count("POLYGON_POINT"), 2)


if __name__ == "__main__":
    unittest.main()
