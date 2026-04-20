import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_Vector_Map as VMAP


class FakeDEM:
    def __init__(self):
        self.alt_dem = numpy.ones((1, 1), dtype=numpy.float32)
        self.write_calls = []

    def alt_vec(self, way):
        return numpy.zeros((len(way), 1), dtype=numpy.float32)

    def write_to_file(self, filename):
        self.write_calls.append(filename)


class FakeVectorMap:
    def __init__(self):
        self.dico_edges = {}
        self.seeds = {}
        self.line_encodes = 0

    def encode_MultiLineString(self, *args, **kwargs):
        self.line_encodes += 1

    def snap_to_grid(self, *args, **kwargs):
        return None

    def write_node_file(self, *args, **kwargs):
        return None

    def write_poly_file(self, *args, **kwargs):
        return None


class VectorMapTests(unittest.TestCase):
    def test_include_airports_returns_empty_defaults_when_osm_query_fails(self):
        tile = SimpleNamespace(lat=34, lon=-83)

        with mock.patch.object(VMAP.OSM, "OSM_queries_to_OSM_layer", return_value=False):
            apt_array, apt_area = VMAP.include_airports(object(), tile)

        self.assertIsInstance(apt_array, numpy.ndarray)
        self.assertEqual(apt_array.shape, (1001, 1001))
        self.assertFalse(apt_array.any())
        self.assertTrue(apt_area.is_empty)

    def test_build_poly_file_loads_dem_when_airport_query_fails(self):
        fake_vector_map = FakeVectorMap()
        fake_dem = FakeDEM()

        with tempfile.TemporaryDirectory() as tmpdir:
            build_dir = Path(tmpdir) / "build"
            osm_dir = Path(tmpdir) / "osm"
            tile = SimpleNamespace(
                lat=34,
                lon=-83,
                build_dir=str(build_dir),
                mesh_zl=14,
                dem=None,
                custom_dem="",
                fill_nodata="to zero",
                road_level=0,
            )

            with mock.patch.object(VMAP.UI, "is_working", 0), \
                 mock.patch.object(VMAP.UI, "red_flag", 0), \
                 mock.patch.object(VMAP.UI, "logprint"), \
                 mock.patch.object(VMAP.UI, "vprint"), \
                 mock.patch.object(VMAP.UI, "timings_and_bottom_line"), \
                 mock.patch.object(VMAP.UI, "exit_message_and_bottom_line"), \
                 mock.patch.object(VMAP.OSM, "OSM_queries_to_OSM_layer", return_value=False), \
                 mock.patch.object(VMAP.VECT, "Vector_Map", return_value=fake_vector_map), \
                 mock.patch.object(VMAP.DEM, "DEM", return_value=fake_dem) as dem_ctor, \
                 mock.patch.object(VMAP, "include_roads"), \
                 mock.patch.object(VMAP, "include_sea"), \
                 mock.patch.object(VMAP, "include_water"), \
                 mock.patch.object(VMAP.GEO, "wgs84_to_orthogrid", side_effect=[(0, 0), (16, 16)]), \
                 mock.patch.object(VMAP.FNAMES, "short_latlon", return_value="+34-083"), \
                 mock.patch.object(VMAP.FNAMES, "osm_dir", return_value=str(osm_dir)), \
                 mock.patch.object(VMAP.FNAMES, "input_node_file", return_value=str(build_dir / "tile.node")), \
                 mock.patch.object(VMAP.FNAMES, "input_poly_file", return_value=str(build_dir / "tile.poly")):
                result = VMAP.build_poly_file(tile)

        self.assertEqual(result, 1)
        self.assertIs(tile.dem, fake_dem)
        dem_ctor.assert_called_once_with(34, -83, "", "to zero", info_only=False)
        self.assertEqual(fake_vector_map.line_encodes, 2)
        self.assertEqual(fake_dem.write_calls, [str(build_dir / "Data+34-083.alt")])


if __name__ == "__main__":
    unittest.main()
