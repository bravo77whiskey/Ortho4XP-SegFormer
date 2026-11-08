"""Tests for building height limits under runway approaches."""

import collections
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_Approach_Heights as APPROACH
import O4_Building_Overlay as BLD

Runway = collections.namedtuple("Runway", "width lat1 lon1 lat2 lon2 disp1 disp2 mark1 mark2")
Airport = collections.namedtuple("Airport", "icao runways")

# ~3.1 km x 45 m runway (design group IV): end 1 precision, end 2 visual.
AIRPORT = Airport("TEST", [Runway(45.0, 22.570, 120.330, 22.578, 120.360, 0, 0, 3, 0)])


def _out(surface, along_m, side_m=0.0):
    """(lon, lat) ``along_m`` beyond the threshold and ``side_m`` to its left."""
    kx = 111320.0 * math.cos(math.radians(surface.lat))
    e = surface.ux * along_m - surface.uy * side_m
    n = surface.uy * along_m + surface.ux * side_m
    return surface.lon + e / kx, surface.lat + n / APPROACH.EARTH_M_PER_DEG_LAT


def _cap(surface, along, side=0.0):
    lon, lat = _out(surface, along, side)
    return float(APPROACH.cap_at([surface], [lon], [lat])[0])


class SurfaceTest(unittest.TestCase):
    def setUp(self):
        surfaces = APPROACH.runway_surfaces([AIRPORT])
        self.precision = next(s for s in surfaces if s.end == 1)
        self.visual = next(s for s in surfaces if s.end == 2)

    def test_instrument_surface_is_the_amendment_18_ofs(self):
        s = self.precision
        self.assertEqual((s.inner_half, s.offset, s.divergence, s.length), (92.5, 60.0, 0.10, 4500.0))
        self.assertEqual(_cap(s, 30.0), 0.0)                 # before the inner edge
        self.assertAlmostEqual(_cap(s, 1060.0), 1000.0 / 30.0, places=2)
        self.assertAlmostEqual(_cap(s, 4560.0), 150.0, places=2)
        self.assertTrue(math.isinf(_cap(s, 4600.0)))         # surface ends at 150 m

    def test_visual_end_is_shorter(self):
        s = self.visual
        self.assertEqual((s.inner_half, s.length), (67.5, 2500.0))
        self.assertTrue(math.isinf(_cap(s, 2600.0)))

    def test_surface_widens_by_ten_percent(self):
        # 60 m beyond the inner edge: half width 92.5 + 0.10 * 60 = 98.5 m
        self.assertTrue(math.isfinite(_cap(self.precision, 120.0, 97.0)))
        self.assertTrue(math.isinf(_cap(self.precision, 120.0, 100.0)))

    def test_tree_caps_stop_at_tree_height(self):
        cap = APPROACH.cap_map([self.precision], 22.60, 22.52, 120.25, 120.34, 256, 256)
        self.assertLess(float(cap[np.isfinite(cap)].max()), APPROACH.MAX_TREE_H_M)

    def test_design_groups_and_inner_edges(self):
        self.assertEqual(APPROACH.design_group(600.0, 18.0), "I")
        self.assertEqual(APPROACH.design_group(1000.0, 30.0), "IIA-IIB")
        self.assertEqual(APPROACH.design_group(1500.0, 30.0), "IIC")
        self.assertEqual(APPROACH.design_group(3000.0, 40.0), "III")
        self.assertEqual(APPROACH.design_group(3000.0, 45.0), "IV")
        self.assertEqual(APPROACH.design_group(3800.0, 60.0), "V")
        self.assertEqual(APPROACH.inner_edge_m("instrument", "I", 30.0), 110.0)
        self.assertEqual(APPROACH.inner_edge_m("instrument", "I", 45.0), 125.0)
        self.assertEqual(APPROACH.inner_edge_m("instrument", "IIC", 30.0), 140.0)
        self.assertEqual(APPROACH.inner_edge_m("instrument", "IIC", 45.0), 155.0)
        self.assertEqual(APPROACH.inner_edge_m("visual", "I", 25.0), 80.0)
        self.assertEqual(APPROACH.inner_edge_m("visual", "V", 60.0), 150.0)


class TerrainTest(unittest.TestCase):
    """Heights are above the ground: threshold elevation minus ground elevation."""

    def setUp(self):
        self.flat = next(s for s in APPROACH.runway_surfaces([AIRPORT]) if s.end == 1)

    def _on(self, ground_fn):
        s = self.flat
        kx = 111320.0 * math.cos(math.radians(s.lat))
        along, side = np.meshgrid(np.arange(-500.0, 5200.0, 50.0), np.arange(-800.0, 801.0, 50.0))
        e = s.ux * along - s.uy * side
        n = s.uy * along + s.ux * side
        verts = np.column_stack((s.lon + e.ravel() / kx, s.lat + n.ravel() / APPROACH.EARTH_M_PER_DEG_LAT,
                                 ground_fn(along.ravel())))
        return APPROACH.mesh_ground([s], verts)[0]

    def test_platform_allows_taller_below(self):
        # Threshold on a 100 m platform, ground 20 m lower from 550 m out.
        s = self._on(lambda along: np.where(along < 525.0, 100.0, 80.0))
        self.assertAlmostEqual(s.elev, 100.0, places=3)
        self.assertEqual(s.rise, 0.0)
        self.assertAlmostEqual(_cap(s, 1060.0), 1000.0 / 30.0 + 20.0, places=1)
        self.assertLess(APPROACH.reach_m(s), APPROACH.reach_m(self.flat) + 1.0)

    def test_hill_lowers_the_cap(self):
        # Ground rising 1 m per 100 m out from the threshold.
        s = self._on(lambda along: 50.0 + np.maximum(along, 0.0) * 0.01)
        self.assertAlmostEqual(_cap(s, 1060.0), 1000.0 / 30.0 - 10.6, places=1)
        self.assertGreater(s.rise, 40.0)
        self.assertGreater(APPROACH.reach_m(s), APPROACH.reach_m(self.flat) + 100.0)
        self.assertEqual(_cap(s, 30.0), 0.0)                 # never under 0

    def test_off_mesh_threshold_stays_level(self):
        verts = np.array([[0.0, 0.0, 5.0], [0.01, 0.0, 5.0], [0.0, 0.01, 5.0]])
        self.assertIsNone(APPROACH.mesh_ground([self.flat], verts)[0].ground)

    def test_mesh_vertices_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "Data+22+120.mesh")
            with open(path, "w") as f:
                f.write("MeshVersionFormatted 2\nDimension 3\n\nVertices\n2\n"
                        "120.5 22.5 0.00012345 0\n120.6 22.6 0.001 0\n\nNormals\n2\n0 0 0\n0 0 0\n")
            xyz = APPROACH.mesh_vertices(path)
        self.assertEqual(xyz.shape, (2, 3))
        self.assertAlmostEqual(xyz[0, 2], 12.345, places=6)
        self.assertAlmostEqual(xyz[1, 2], 100.0, places=6)

    def test_signature_follows_the_ground(self):
        a = self._on(lambda along: np.full(along.shape, 10.0))
        b = self._on(lambda along: np.where(along < 525.0, 10.0, 0.0))
        self.assertNotEqual(APPROACH.building_signature([a]), APPROACH.building_signature([b]))
        self.assertNotEqual(APPROACH.signature([a]), APPROACH.signature([b]))


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
