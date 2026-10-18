"""Grass around airports: apt.dat reading, the grass rule, forest files, wiring."""

import math
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
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

    def test_runway_shoulders_widen_the_runway(self):
        self.assertEqual(APT.shoulder_width_m(0), 0.0)
        self.assertEqual(APT.shoulder_width_m(753), 7.0)        # XP12: 7 m of surface 53
        self.assertEqual(APT.shoulder_width_m(1529), 15.0)
        self.assertEqual(APT.shoulder_width_m(1), APT.LEGACY_SHOULDER_M)
        path = os.path.join(self.tmp.name, "shoulders.dat")
        _write(path, APT_DAT.replace("100 30.00 1 0 ", "100 30.00 1 853 "))
        runway = np.array(dict((i, p) for i, p, _ in APT.parse_apt_dat(path))["TEST"][0][0])
        width = math.hypot(*(runway[0] - runway[3]) * [111320 * math.cos(math.radians(10)), 110574])
        self.assertAlmostEqual(width, 30.0 + 2 * 8, delta=0.5)

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


DSF_TEXT = """POLYGON_DEF lib/airport/pavement/asphalt_2L.pol
POLYGON_DEF lib/vegetation/forests/broadleaves/warm.for
POLYGON_DEF local/apron.pol
POLYGON_DEF local/lawn.pol
POLYGON_DEF lib/airport/markings/DrapedDirSigns.pol
BEGIN_POLYGON 0 65535 4
BEGIN_WINDING
POLYGON_POINT 20.0 10.0 0 0
POLYGON_POINT 20.001 10.0 1 0
POLYGON_POINT 20.001 10.001 1 1
END_WINDING
BEGIN_WINDING
POLYGON_POINT 20.0002 10.0002 0 0
POLYGON_POINT 20.0004 10.0002 1 0
POLYGON_POINT 20.0004 10.0004 1 1
END_WINDING
END_POLYGON
BEGIN_POLYGON 1 255 2
BEGIN_WINDING
POLYGON_POINT 21.0 11.0
POLYGON_POINT 21.001 11.0
POLYGON_POINT 21.001 11.001
END_WINDING
END_POLYGON
BEGIN_POLYGON 2 65535 4
BEGIN_WINDING
POLYGON_POINT 22.0 12.0 0 0
POLYGON_POINT 22.001 12.0 1 0
POLYGON_POINT 22.001 12.001 1 1
END_WINDING
END_POLYGON
BEGIN_POLYGON 3 65535 4
BEGIN_WINDING
POLYGON_POINT 23.0 13.0 0 0
POLYGON_POINT 23.001 13.0 1 0
POLYGON_POINT 23.001 13.001 1 1
END_WINDING
END_POLYGON
BEGIN_POLYGON 4 65535 4
BEGIN_WINDING
POLYGON_POINT 24.0 14.0 0 0
POLYGON_POINT 24.001 14.0 1 0
POLYGON_POINT 24.001 14.001 1 1
END_WINDING
END_POLYGON
"""


class PackPavementTest(unittest.TestCase):
    """Draped pavement in an airport pack's own DSF, which apt.dat doesn't list."""

    def test_pavement_polygons_by_path_or_texture(self):
        with tempfile.TemporaryDirectory() as pack:
            _write(os.path.join(pack, "local", "apron.pol"), "A\n850\nDRAPED_POLYGON\nTEXTURE apron_concrete.dds\n")
            _write(os.path.join(pack, "local", "lawn.pol"), "A\n850\nDRAPED_POLYGON\nTEXTURE grass_cut.dds\n")
            text = os.path.join(pack, "dsf.txt")
            _write(text, DSF_TEXT)
            polys = APT.dsf_pavement(text, pack)
        # library pavement (with its hole), the pack's concrete apron, painted signs;
        # not the forest or the pack's lawn
        self.assertEqual([round(o[0][0]) for o, _ in polys], [20, 22, 24])
        self.assertEqual(len(polys[0][1]), 1)
        self.assertEqual(polys[0][0][1], (20.001, 10.0))      # lon, lat; UVs dropped


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


class FullTileTest(unittest.TestCase):
    """Grass away from any airport only with full_tile set."""

    def _pass1(self, full_tile, tmp):
        from PIL import Image
        bbox = GRASS.footprint_bbox(28176, 54800, 16)  # a ZL16 footprint near Taiwan
        lat, lon = math.floor(bbox[0]), math.floor(bbox[1])
        img = np.zeros((256, 256, 3), np.uint8)
        img[...] = (70, 120, 40)                                  # meadow green
        Image.fromarray(img).save(os.path.join(tmp, "t.png"))
        np.save(os.path.join(tmp, "t_cls.npy"), np.full((256, 256), 2, np.uint8))
        GRASS._worker_init(dict(lat=lat, lon=lon, settings=GRASS.GrassSettings(full_tile=full_tile),
                                airports=[], pavement=[], roads_raw=[], spill_dir=tmp))
        return GRASS._pass1(("t", os.path.join(tmp, "t.png"), bbox, (), os.path.join(tmp, "t_cls.npy")))

    def test_no_airport_no_grass(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self._pass1(False, tmp), {"skip": None})

    def test_full_tile_grows_grass_without_airports(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self._pass1(True, tmp)
            self.assertGreater(r["meta"]["grass_km2"], 0)
            self.assertTrue(os.path.exists(r["spill"]))           # masks wait on disk for pass 2
            centres = r["sample"][:1]
            out = GRASS._pass2((r["meta"], r["spill"], centres, 0))
            self.assertGreater(out["n_poly"], 0)

    def test_simplified_outlines_stay_off_pavement(self):
        """A round apron: its hole in the grass, simplified, must not shrink onto it."""
        from PIL import Image
        bbox = GRASS.footprint_bbox(28176 * 8, 54800 * 8, 19)     # ~1.1 km, ~2.2 m/px at 512 px
        lat, lon = math.floor(bbox[0]), math.floor(bbox[1])
        n = 512
        lat_c, lon_c = 0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])
        kx = 111320 * math.cos(math.radians(lat_c))
        ring = [(lon_c + 90 * math.cos(t) / kx, lat_c + 90 * math.sin(t) / 110574)
                for t in np.linspace(0, 2 * math.pi, 90, endpoint=False)]
        with tempfile.TemporaryDirectory() as tmp:
            img = np.zeros((n, n, 3), np.uint8)
            img[...] = (70, 120, 40)
            Image.fromarray(img).save(os.path.join(tmp, "t.png"))
            np.save(os.path.join(tmp, "t_cls.npy"), np.full((n, n), 2, np.uint8))
            # A coarse tolerance, so the plain simplification does cut across.
            settings = GRASS.GrassSettings(full_tile=True, simplify_m=10.0)
            GRASS._worker_init(dict(lat=lat, lon=lon, settings=settings,
                                    airports=[], pavement=[(ring, [])], roads_raw=[], spill_dir=tmp))
            r = GRASS._pass1(("t", os.path.join(tmp, "t.png"), bbox, (), os.path.join(tmp, "t_cls.npy")))
            out = GRASS._pass2((r["meta"], r["spill"], r["sample"][:1], 0))
        drawn = np.zeros((n, n), np.uint8)
        for poly in out["text"].split("BEGIN_POLYGON")[1:]:
            wind = [np.array([[float(v) for v in l.split()[1:3]] for l in w.splitlines() if l.startswith("POLYGON_POINT")])
                    for w in poly.split("BEGIN_WINDING")[1:]]
            px = [GRASS._to_px(w[:, 0], w[:, 1], bbox, n, n) for w in wind]
            one = np.zeros((n, n), np.uint8)
            cv2.fillPoly(one, px[:1], 1)
            cv2.fillPoly(one, px[1:], 0)
            drawn |= one
        mpp = (bbox[3] - bbox[1]) * kx / n
        pave = GRASS.pavement_mask([(ring, [])], bbox, n, n, mpp, 0)
        self.assertGreater(drawn.sum(), 0.5 * n * n)
        self.assertEqual(int((drawn.astype(bool) & pave).sum()), 0)

    def test_class_map_kept_at_grass_resolution(self):
        bbox = GRASS.footprint_bbox(28176 * 8, 54800 * 8, 19)     # ZL19: ~0.27 m/px
        hook = GRASS.GrassHook.__new__(GRASS.GrassHook)
        with tempfile.TemporaryDirectory() as tmp:
            hook.cache_dir, hook.settings, hook.class_maps = tmp, GRASS.GrassSettings(), {}
            hook.sources = {"x.png": (bbox, ())}
            hook.take_class_map("x.png", np.zeros((4096, 4096), np.int8))
            self.assertEqual(np.load(hook.class_maps["x.png"]).shape, (1024, 1024))


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
        self.assertIn("GrassSettings(airport_buffer_m=750.0, full_tile=False)", code)
        self.assertIn("grass_settings   = _grass", code)
        self.assertRegex(code, r"grass_out_dsf    = '[^']*yOrtho4XP_Grass_Overlays[^']*\+10\+020\.dsf'")

    def test_grass_pack_sits_beside_the_tree_pack(self):
        import O4_Vegetation_Overlay as VEG
        veg = os.path.join("X", "yOrtho4XP_Veg_Overlays", "Earth nav data", "+20+120", "+22+120.dsf")
        self.assertEqual(VEG.grass_dsf_path_beside(veg), os.path.abspath(
            os.path.join("X", "yOrtho4XP_Grass_Overlays", "Earth nav data", "+20+120", "+22+120.dsf")))

    def test_grass_off_passes_none(self):
        code = self._code(False)
        self.assertIn("_grass = None", code)
        self.assertNotIn("import O4_Grass_Overlay", code)


if __name__ == "__main__":
    unittest.main()
