import os
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

import O4_Veg_Density as DENSITY
import O4_Vegetation_Overlay as VEG


class CoverTests(unittest.TestCase):
    def _texture(self):
        img = np.full((400, 400, 3), 170, np.uint8)   # open ground
        img[:, :200] = (35, 60, 30)                   # dark canopy
        img[:, 200:300] = (100, 110, 90)              # halfway to the ground
        return img

    def test_confident_tree_is_closed_forest(self):
        mask = np.zeros((400, 400), np.uint8)
        mask[:, :300] = 1
        prob = np.zeros((100, 100), np.uint8)
        prob[:, :50] = 255
        cover = DENSITY.cover_map(self._texture(), prob, mask, 2.0)
        self.assertGreater(cover[200, 40], 0.9)

    def test_forest_the_model_misses_stays_above_a_lawn(self):
        # P(tree) ~0 and no solid block: dark canopy keeps half, brighter mask pixels less.
        mask = np.zeros((400, 400), np.uint8)
        mask[:, :300] = 1
        zeros = np.zeros((100, 100), np.uint8)
        with mock.patch.object(DENSITY, "solidity", side_effect=lambda m, *a, **k: np.zeros(m.shape, np.float32)):
            cover = DENSITY.cover_map(self._texture(), zeros, mask, 2.0)
        self.assertAlmostEqual(float(cover[200, 40]), DENSITY.UNMIX_WEIGHT, delta=0.02)
        self.assertLess(cover[200, 280], cover[200, 40] - 0.1)

    def test_dark_solid_forest_the_model_misses_is_closed(self):
        # 3.2 km across, P(tree) 0: not a lawn.
        img = np.full((800, 800, 3), 170, np.uint8)
        img[:, :400] = (35, 60, 30)
        mask = np.zeros((800, 800), np.uint8)
        mask[:, :400] = 1
        cover = DENSITY.cover_map(img, np.zeros((100, 100), np.uint8), mask, 4.0)
        self.assertGreater(cover[400, 60], 0.9)

    def test_stripes_of_mask_are_not_solid(self):
        mask = np.zeros((800, 800), np.uint8)
        mask[:, ::2] = 1
        self.assertLess(DENSITY.solidity(mask, 4.0)[400, 400], DENSITY.SOLID_LOW)

    def test_water_is_not_the_open_ground(self):
        img = np.full((400, 400, 3), 12, np.uint8)    # sea, darker than the forest
        img[:, :200] = (35, 60, 30)
        mask = np.zeros((400, 400), np.uint8)
        mask[:, :200] = 1
        water = np.zeros((400, 400), bool)
        water[:, 200:] = True
        zeros = np.zeros((100, 100), np.uint8)
        with mock.patch.object(DENSITY, "solidity", side_effect=lambda m, *a, **k: np.zeros(m.shape, np.float32)):
            cover = DENSITY.cover_map(img, zeros, mask, 2.0, not_ground=water)
        self.assertAlmostEqual(float(cover[200, 60]), DENSITY.UNMIX_WEIGHT, delta=0.02)

    def test_a_shore_is_as_solid_as_the_land_behind_it(self):
        mask = np.zeros((400, 400), np.uint8)
        mask[:, :200] = 1
        water = np.zeros((400, 400), bool)
        water[:, 200:] = True
        self.assertGreater(DENSITY.solidity(mask, 2.0, water)[200, 190], 0.95)
        self.assertLess(DENSITY.solidity(mask, 2.0)[200, 190], 0.7)

    def test_veg_res_sized_mask(self):
        mask = np.ones((200, 200), np.uint8)
        cover = DENSITY.cover_map(self._texture(), np.full((100, 100), 128, np.uint8), mask, 4.0)
        self.assertEqual(cover.shape, (200, 200))

    def test_bands_cover_the_mask_and_top_band_is_full_density(self):
        mask = np.ones((1, 7), np.uint8)
        mask[0, 6] = 0
        cover = np.array([[0.0, 0.3, 0.45, 0.6, 0.75, 0.95, 0.95]], np.float32)
        self.assertEqual(DENSITY.density_bands(cover, mask).tolist(), [[1, 2, 3, 4, 5, 6, 0]])
        self.assertEqual(DENSITY.band_density(len(DENSITY.BAND_COVER)), DENSITY.FULL_DENSITY)
        values = [DENSITY.band_density(b) for b in range(1, len(DENSITY.BAND_COVER) + 1)]
        self.assertEqual(values, sorted(values))
        self.assertEqual(len(DENSITY.BAND_COVER), len(DENSITY.BAND_EDGES) + 1)

    def test_coarse_grid_keeps_size_and_bands(self):
        rng = np.random.default_rng(1)
        img = cv2.GaussianBlur(rng.integers(20, 200, (800, 800, 3), dtype=np.uint8), (0, 0), 6)
        mask = (cv2.GaussianBlur(rng.random((800, 800)).astype(np.float32), (0, 0), 20) > 0.5).astype(np.uint8)
        prob = (rng.random((200, 200)) * 255).astype(np.uint8)
        with mock.patch.dict(os.environ, {"O4_VEG_DENSITY_HALF": "0"}):
            full = DENSITY.cover_map(img, prob, mask, 2.0)
            full_sig = DENSITY.signature()
        with mock.patch.dict(os.environ, {"O4_VEG_DENSITY_HALF": "1"}):
            half = DENSITY.cover_map(img, prob, mask, 2.0)
            self.assertNotEqual(DENSITY.signature(), full_sig)
        self.assertEqual(half.shape, mask.shape)
        on = mask > 0
        bands_full = DENSITY.density_bands(full, mask)[on].astype(int)
        bands_half = DENSITY.density_bands(half, mask)[on].astype(int)
        self.assertGreater(np.mean(bands_full == bands_half), 0.9)
        self.assertEqual(int(np.max(np.abs(bands_full - bands_half))), 1)


class PolygonTests(unittest.TestCase):
    def _kwargs(self, mask):
        return dict(img_w=mask.shape[1], img_h=mask.shape[0], lat_n=10.01, lat_s=10.0,
                    lon_w=104.0, lon_e=104.01, tile_lat=10, tile_lon=104, m_per_px=1.0,
                    min_area_px=1, simplify_px=1.0, region="tropical",
                    rng=np.random.default_rng(0), density_override=None, context_masks={})

    def test_each_band_gets_its_own_density(self):
        mask = np.zeros((400, 400), np.uint8)
        mask[20:380, 20:380] = 1
        cover = np.full(mask.shape, 0.95, np.float32)
        cover[:, 200:] = 0.3
        with mock.patch.dict(os.environ, {"O4_VEG_POINT_TREES": "1"}):
            polys = VEG._process_tree_mask(mask, density_cover=cover, **self._kwargs(mask))
        self.assertEqual(sorted(den for _p, den, _r in polys),
                         [DENSITY.band_density(2), DENSITY.band_density(6)])

    def test_veg_density_is_the_ceiling(self):
        mask = np.zeros((400, 400), np.uint8)
        mask[20:380, 20:380] = 1
        cover = np.full(mask.shape, 0.95, np.float32)
        cover[:, 200:] = 0.3
        kwargs = dict(self._kwargs(mask), density_override=1.0)
        with mock.patch.dict(os.environ, {"O4_VEG_POINT_TREES": "1"}):
            polys = VEG._process_tree_mask(mask, density_cover=cover, **kwargs)
        self.assertEqual(sorted(den for _p, den, _r in polys),
                         [DENSITY.band_density(2, 255), 255])

    def test_a_small_band_joins_its_neighbour(self):
        mask = np.zeros((400, 400), np.uint8)
        mask[20:380, 20:380] = 1
        cover = np.full(mask.shape, 0.95, np.float32)
        cover[100:120, 100:120] = 0.3   # 400 m2, under FOREST_ZONE_MIN_M2
        with mock.patch.dict(os.environ, {"O4_VEG_POINT_TREES": "1"}):
            polys = VEG._process_tree_mask(mask, density_cover=cover, **self._kwargs(mask))
        self.assertEqual([den for _p, den, _r in polys], [DENSITY.band_density(6)])

    def test_without_cover_the_fixed_density_stays(self):
        mask = np.zeros((400, 400), np.uint8)
        mask[20:380, 20:380] = 1
        polys = VEG._process_tree_mask(mask, **self._kwargs(mask))
        self.assertEqual(len(polys), 1)
        self.assertIn(polys[0][1], {v for table in VEG._BASE_DENSITY.values() for v in table.values()})


if __name__ == "__main__":
    unittest.main()
