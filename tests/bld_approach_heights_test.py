"""Tests for building height limits under runway approaches."""

import collections
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_Approach_Heights as APPROACH
import O4_Building_Overlay as BLD

Runway = collections.namedtuple("Runway", "lat1 lon1 lat2 lon2 disp1 disp2 mark1 mark2")
Airport = collections.namedtuple("Airport", "icao runways")

# ~3.1 km runway: end 1 precision (code 4, two sections), end 2 visual.
AIRPORT = Airport("TEST", [Runway(22.570, 120.330, 22.578, 120.360, 0, 0, 3, 0)])


def _out(surface, along_m, side_m=0.0):
    """(lon, lat) ``along_m`` beyond the threshold and ``side_m`` to its left."""
    kx = 111320.0 * math.cos(math.radians(surface.lat))
    e = surface.ux * along_m - surface.uy * side_m
    n = surface.uy * along_m + surface.ux * side_m
    return surface.lon + e / kx, surface.lat + n / APPROACH.EARTH_M_PER_DEG_LAT


class SurfaceTest(unittest.TestCase):
    def setUp(self):
        surfaces = APPROACH.runway_surfaces([AIRPORT])
        self.precision = next(s for s in surfaces if s.end == 1)
        self.visual = next(s for s in surfaces if s.end == 2)

    def cap(self, surface, along, side=0.0):
        lon, lat = _out(surface, along, side)
        return float(APPROACH.cap_at([surface], [lon], [lat])[0])

    def test_precision_profile_has_second_section(self):
        s = self.precision
        self.assertEqual(s.later, ((3600.0, 0.025),))
        self.assertEqual(self.cap(s, 30.0), 0.0)            # before the inner edge
        self.assertAlmostEqual(self.cap(s, 1060.0), 20.0, places=2)
        self.assertAlmostEqual(self.cap(s, 3060.0), 60.0, places=2)
        self.assertAlmostEqual(self.cap(s, 4860.0), 105.0, places=2)
        self.assertTrue(math.isinf(self.cap(s, 7000.0)))    # no horizontal section

    def test_visual_end_has_first_section_only(self):
        self.assertEqual(self.visual.later, ())
        self.assertTrue(math.isinf(self.cap(self.visual, 3100.0)))

    def test_surface_widens_by_divergence(self):
        # 60 m beyond the inner edge: half width 150 + 0.15 * 60 = 159 m
        self.assertTrue(math.isfinite(self.cap(self.precision, 120.0, 158.0)))
        self.assertTrue(math.isinf(self.cap(self.precision, 120.0, 170.0)))

    def test_tree_caps_stop_at_tree_height(self):
        cap = APPROACH.cap_map([self.precision], 22.60, 22.52, 120.25, 120.34, 256, 256)
        self.assertLess(float(cap[np.isfinite(cap)].max()), APPROACH.MAX_TREE_H_M)


class BuildingHelpersTest(unittest.TestCase):
    def test_facade_height_clamped_to_cap(self):
        rng = np.random.default_rng(0)
        det = {'height_m': 30.0, 'approach_cap_m': 12.7}
        self.assertEqual(BLD._yolo_facade_height_m(det, rng, {}, BLD.BLD_CLASS_MEDIUM), 12.0)
        det = {'height_m': 8.0, 'approach_cap_m': 12.7}
        self.assertEqual(BLD._yolo_facade_height_m(det, rng, {}, BLD.BLD_CLASS_MEDIUM), 8.0)
        self.assertEqual(BLD._yolo_facade_height_m({'height_m': 30.0}, rng, {}, BLD.BLD_CLASS_MEDIUM), 30.0)

    def test_object_samples_follow_heading(self):
        kx = 111320.0 * math.cos(math.radians(22.0))
        lons, lats = BLD._object_samples(120.0, 22.0, 0.0, (0.0, 100.0, -1.0, 1.0))
        self.assertAlmostEqual((lons.max() - 120.0) * kx, 100.0, places=3)   # +X east
        lons, lats = BLD._object_samples(120.0, 22.0, 90.0, (0.0, 100.0, -1.0, 1.0))
        self.assertAlmostEqual((22.0 - lats.min()) * APPROACH.EARTH_M_PER_DEG_LAT, 100.0, places=3)

    def test_poly_min(self):
        values = np.full((100, 100), np.inf, np.float32)
        values[50:, :] = 5.0
        self.assertEqual(BLD._poly_min(values, [(10, 60), (90, 60), (90, 90), (10, 90)]), 5.0)
        self.assertIsNone(BLD._poly_min(values, [(10, 10), (90, 10), (90, 40), (10, 40)]))

    def test_ring_samples_cover_interior(self):
        ring = [(120.0, 22.0), (120.001, 22.0), (120.001, 22.001), (120.0, 22.001), (120.0, 22.0)]
        lons, lats = BLD._ring_samples(ring)
        self.assertGreater(lons.size, 500)
        inside = (lons > 120.0003) & (lons < 120.0007) & (lats > 22.0003) & (lats < 22.0007)
        self.assertTrue(inside.any())


if __name__ == "__main__":
    unittest.main()
