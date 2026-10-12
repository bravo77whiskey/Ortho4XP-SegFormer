"""Grass around airports: apt.dat reading, the grass rule, forest files, wiring."""

import math
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_Apt_Pavement as APT
import O4_Grass_Overlay as GRASS
import O4_Overlay_Pipeline as PIPELINE

APT_DAT = """I
1200 Generated

1 10 0 0 TEST Test Field
100 30.00 1 0 0.25 0 0 0 09 10.0000000 20.0000000 0 60 0 0 0 0 27 10.0000000 20.0100000 0 60 0 0 0 0
110 1 0.25 0.00 Apron
111 10.0010000 20.0010000
112 10.0010000 20.0030000 10.0015000 20.0035000
111 10.0030000 20.0030000
113 10.0030000 20.0010000
111 10.0015000 20.0015000
111 10.0015000 20.0020000
113 10.0020000 20.0020000
130 Boundary
111 9.9900000 19.9900000
111 9.9900000 20.0200000
111 10.0100000 20.0200000
113 10.0100000 19.9900000

1 5 0 0 NOPV No pavement
1302 city Nowhere

99
"""


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


class AptParseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "apt.dat")
        _write(self.path, APT_DAT)

    def tearDown(self):
        self.tmp.cleanup()

    def test_runway_pavement_and_boundary(self):
        airports = {icao: (pav, bnd) for icao, pav, bnd in APT.parse_apt_dat(self.path)}
        self.assertEqual(set(airports), {"TEST", "NOPV"})
        pavement, boundaries = airports["TEST"]
        self.assertEqual(len(pavement), 2)          # runway + apron
        self.assertEqual(len(boundaries), 1)
        runway = np.array(pavement[0][0])
        # 30 m wide, ~1.1 km long plus a 60 m blast pad at each end.
        width = math.hypot(*(runway[0] - runway[3]) * [111320 * math.cos(math.radians(10)), 110574])
        length = math.hypot(*(runway[1] - runway[0]) * [111320 * math.cos(math.radians(10)), 110574])
        self.assertAlmostEqual(width, 30.0, delta=0.5)
        self.assertAlmostEqual(length, 0.01 * 111320 * math.cos(math.radians(10)) + 120, delta=2)
        apron_outer, apron_holes = pavement[1]
        self.assertEqual(len(apron_holes), 1)
        # The Bezier node is sampled into extra points along its edge.
        self.assertGreater(len(apron_outer), 4)

    def test_airport_without_pavement_is_kept_but_empty(self):
        airports = {icao: (pav, bnd) for icao, pav, bnd in APT.parse_apt_dat(self.path)}
        self.assertEqual(airports["NOPV"], ([], []))


class AirportPriorityTest(unittest.TestCase):
    """The first enabled pack in scenery_packs.ini defining an airport wins."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.xp = self.tmp.name
        cs = os.path.join(self.xp, "Custom Scenery")
        other = APT_DAT.replace("100 30.00", "100 90.00")
        _write(os.path.join(cs, "Airport_A", "Earth nav data", "apt.dat"), APT_DAT)
        _write(os.path.join(cs, "Airport_B", "Earth nav data", "apt.dat"), other)
        _write(os.path.join(cs, "Airport_Off", "Earth nav data", "apt.dat"), other)
        _write(os.path.join(self.xp, "Global Scenery", "Global Airports", "Earth nav data", "apt.dat"), other)
        self.ini = os.path.join(cs, "scenery_packs.ini")
        _write(self.ini, "I\n1000 Version\nSCENERY\n\n"
                         "SCENERY_PACK_DISABLED Custom Scenery/Airport_Off/\n"
                         "SCENERY_PACK Custom Scenery/Airport_A/\n"
                         "SCENERY_PACK *GLOBAL_AIRPORTS*\n"
                         "SCENERY_PACK Custom Scenery/Airport_B/\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_highest_priority_definition_wins(self):
        found = APT.airports_near(self.xp, 10.0, 20.005, 5000.0, log=lambda *_: None)
        self.assertEqual([a.icao for a in found], ["TEST"])
        self.assertEqual(found[0].pack, "Airport_A")
        self.assertEqual(len(found[0].boundaries), 1)
        self.assertGreater(found[0].reach_m, 500)

    def test_tile_cache_follows_scenery_packs_ini(self):
        cache = os.path.join(self.tmp.name, "cache")
        first = APT.airports_for_tile(self.xp, 10, 20, cache, log=lambda *_: None)
        self.assertEqual(first[0].pack, "Airport_A")
        with open(self.ini, "w", encoding="utf-8") as f:
            f.write("I\n1000 Version\nSCENERY\n\nSCENERY_PACK Custom Scenery/Airport_B/\n")
        st = os.stat(self.ini)
        os.utime(self.ini, (st.st_atime, st.st_mtime + 5))
        second = APT.airports_for_tile(self.xp, 10, 20, cache, log=lambda *_: None)
        self.assertEqual(second[0].pack, "Airport_B")


def _lab(l, a, b, shape=(40, 40)):
    out = np.zeros(shape + (3,), np.float32)
    out[...] = (l, a, b)
    return out


class GrassRuleTest(unittest.TestCase):
    MPP = 1.0

    def _grass(self, lab, cls, climate="Cfa"):
        g, s = GRASS.fused_grass(_lab(*lab), np.full((40, 40), cls, np.uint8), self.MPP, climate)
        return bool(g[20, 20]), bool(s[20, 20])

    def test_vegetation_class_grows_grass_unless_it_looks_built(self):
        self.assertEqual(self._grass((45, -8, 20), 2), (True, False))      # straw-ish rangeland
        self.assertEqual(self._grass((60, 0, 2), 2), (False, False))       # grey: built
        self.assertEqual(self._grass((85, -5, 15), 5), (False, False))     # very bright: built

    def test_clearly_green_pixels_are_rescued_from_other_classes(self):
        self.assertEqual(self._grass((45, -25, 20), 3), (True, False))     # green, 'developed'
        self.assertEqual(self._grass((45, 5, 25), 3), (False, False))      # straw, 'developed'
        self.assertEqual(self._grass((45, -25, 20), 8), (False, False))    # green roof: building

    def test_climate_regime(self):
        straw, green = (45, 5, 25), (45, -25, 20)
        self.assertEqual(self._grass(straw, 2, "BWh"), (False, False))     # desert: green only
        self.assertEqual(self._grass(green, 2, "BWh"), (True, False))
        self.assertEqual(self._grass(straw, 2, "BSk"), (True, True))       # steppe: sparse
        self.assertEqual(self._grass(green, 2, "BSk"), (True, False))


class ForestFileTest(unittest.TestCase):
    def test_x_plane_12_forest_with_empty_last_mesh(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c00.for")
            GRASS.write_for(path, "c00.png", 0)
            lines = open(path).read().splitlines()
        self.assertEqual(lines[:3], ["A", "1200", "FOREST"])
        self.assertIn(f"\tTEXTURE {GRASS.BILLBOARD_TEX}", lines)
        self.assertIn("LOD\t230", lines)
        meshes = [l.split("\t") for l in lines if l.startswith("MESH\t")]
        self.assertEqual(len(meshes), 9)                     # 3 tuft sizes x 3 detail levels
        last = [m for m in meshes if m[1].endswith("2")]
        self.assertTrue(all(m[4] == "3" and m[5] == "3" for m in last))   # one zero-area triangle
        self.assertTrue(all(l.split("\t")[2] == "0 1 0" for l in lines if l.startswith("VERTEX")))
        self.assertEqual(sum(l.startswith("MESH_3D") for l in lines), 9)


class PipelineWiringTest(unittest.TestCase):
    def _code(self, enabled):
        captured = {}

        def _capture(code):
            captured["code"] = code
            return 0

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(PIPELINE, "grass_enabled", enabled), \
                mock.patch.object(PIPELINE, "grass_airport_buffer_m", 750.0), \
                mock.patch.object(PIPELINE, "_overlay_cache_dir", return_value=tmp), \
                mock.patch.object(PIPELINE, "_check_tile_imagery", return_value=True), \
                mock.patch.object(PIPELINE, "_auto_setup", return_value=True), \
                mock.patch.object(PIPELINE, "_run_venv", side_effect=_capture), \
                mock.patch.object(PIPELINE, "_dsftool_path", return_value=None), \
                mock.patch.object(PIPELINE, "_scenery_paths", return_value=""), \
                mock.patch.object(PIPELINE, "_resolve_remote_host", return_value=""), \
                mock.patch.object(PIPELINE, "_finish_veg_tile_replacement"):
            PIPELINE.process_veg_tile(10, 20, tmp)
        return captured["code"]

    def test_grass_settings_reach_the_overlay(self):
        code = self._code(True)
        self.assertIn("GrassSettings(airport_buffer_m=750.0)", code)
        self.assertIn("grass_settings   = _grass", code)

    def test_grass_off_passes_none(self):
        code = self._code(False)
        self.assertIn("_grass = None", code)
        self.assertNotIn("import O4_Grass_Overlay", code)


if __name__ == "__main__":
    unittest.main()
