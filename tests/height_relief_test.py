"""Relief from real heights (src/O4_Height_Relief.py) on a synthetic tile."""

from pathlib import Path
import os
import struct
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_Height_Relief as HR  # noqa: E402
import O4_Terrain_Materials as TMAT  # noqa: E402

TEX = "27456_54768_BI16.dds"
OTHER = "27472_54768_BI16.dds"


def _ter(tex, ours=True):
    lines = ["A", "800", "TERRAIN", "", "LOAD_CENTER 22.5 120.5 2400 4096",
             f"BASE_TEX_NOWRAP ../textures/{tex}", *(TMAT.ter_lines(tex) if ours else ()), "NO_ALPHA"]
    return "\n".join(lines) + "\n"


class HeightReliefRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.tmp.name
        self.build = os.path.join(root, "tile")
        self.cache = os.path.join(root, "cache")
        self.tex_dir = os.path.join(self.build, "textures")
        for d in (self.cache, self.tex_dir, os.path.join(self.build, "terrain")):
            os.makedirs(d)
        for tex in (TEX, OTHER):
            Image.new("RGB", (256, 256), (110, 110, 100)).save(os.path.join(self.tex_dir, tex), "DDS")
            with open(os.path.join(self.tex_dir, TMAT.map_file_name(tex)), "wb") as f:
                f.write(b"old map")
            with open(os.path.join(self.build, "terrain", tex[:-4] + ".ter"), "w", newline="") as f:
                f.write(_ter(tex))
        self.encoded = {}

        def fake_encode(rgba, out_dds, converter, relief, heights=0):
            self.encoded[os.path.basename(out_dds)] = (rgba, relief, heights)
            return True

        patcher = mock.patch.object(TMAT, "encode_dds", side_effect=fake_encode)
        patcher.start()
        self.addCleanup(patcher.stop)
        env = mock.patch.dict(os.environ, {HR.SIDECAR_ENV: "1"})
        env.start()
        self.addCleanup(env.stop)

    def _building(self, x0=100, y0=100, x1=156, y1=156, **extra):
        det = {"points": np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32),
               "confidence": 0.9, "class_name": "building", **extra}
        return det

    def _run(self, heights=20.0):
        calls = []

        def height_fn(img, dets, mpp):
            calls.append((img.shape, len(dets), mpp))
            return np.full(len(dets), heights, float)

        stats = HR.run(self.build, self.cache, 22, 120, 0.15, "nvcompress.exe",
                       workers=2, log=lambda m: None, height_fn=height_fn)
        return stats, calls

    def test_buildings_make_a_stamped_map_with_relief_around_them(self):
        HR.save_building_sidecar(self.cache, TEX, [self._building()], 256, 256)
        stats, calls = self._run()
        self.assertEqual(stats["made"], 1)
        self.assertEqual(stats["none"], 1)                       # OTHER had no heights
        self.assertEqual(calls[0][:2], ((256, 256, 3), 1))
        self.assertAlmostEqual(calls[0][2], TMAT.texture_m_per_px(TEX, 256))
        rgba, relief, heights = self.encoded[TMAT.map_file_name(TEX)]
        self.assertEqual((relief, heights), (0.15, TMAT.HEIGHT_RELIEF_VERSION))
        self.assertEqual(rgba.shape, (1024, 1024, 4))
        r, g = rgba[512, :, 0].astype(int), rgba[:, 512, 1].astype(int)
        self.assertGreater(r[600:650].max(), 140)               # east wall faces east
        self.assertLess(r[375:425].min(), 116)                  # west wall faces west
        self.assertGreater(g[375:425].max(), 140)               # north wall faces north
        self.assertLess(g[600:650].min(), 116)
        self.assertEqual(os.listdir(self.cache), [])            # sidecars consumed

    def test_canopy_alone_is_enough_and_the_taller_surface_wins(self):
        canopy = np.zeros((1 + 3, 128, 128), np.float32)
        canopy[0, 20:60, 20:60] = 12.0
        HR.save_canopy_sidecar(self.cache, TEX, canopy)
        HR.save_building_sidecar(self.cache, OTHER, [self._building()], 256, 256)
        canopy2 = np.zeros((4, 128, 128), np.float32)
        canopy2[0, 50:78, 50:78] = 30.0                         # trees taller than the building
        HR.save_canopy_sidecar(self.cache, OTHER, canopy2)
        stats, _ = self._run(heights=8.0)
        self.assertEqual(stats["made"], 2)
        self.assertIn(TMAT.map_file_name(TEX), self.encoded)

    def test_low_surfaces_leave_the_map_alone(self):
        HR.save_building_sidecar(self.cache, TEX, [self._building()], 256, 256)
        stats, _ = self._run(heights=0.4)
        self.assertEqual((stats["made"], stats["flat"]), (0, 1))
        with open(os.path.join(self.tex_dir, TMAT.map_file_name(TEX)), "rb") as f:
            self.assertEqual(f.read(), b"old map")

    def test_textures_without_our_map_line_are_skipped(self):
        with open(os.path.join(self.build, "terrain", TEX[:-4] + ".ter"), "w", newline="") as f:
            f.write(_ter(TEX, ours=False))
        HR.save_building_sidecar(self.cache, TEX, [self._building()], 256, 256)
        self.assertEqual(HR.land_textures(self.build), [OTHER])
        stats, _ = self._run()
        self.assertEqual(stats["made"], 0)

    def test_cached_yolo_detections_serve_when_there_is_no_sidecar(self):
        import pickle
        with open(os.path.join(self.cache, TEX[:-4] + "_yolo_obb.pkl"), "wb") as f:
            pickle.dump({"key": "k", "detections": [self._building()]}, f)
        stats, _ = self._run()
        self.assertEqual(stats["made"], 1)
        self.assertTrue(os.path.isfile(os.path.join(self.cache, TEX[:-4] + "_yolo_obb.pkl")))

    def test_sidecars_only_when_asked(self):
        with mock.patch.dict(os.environ, {HR.SIDECAR_ENV: "0"}):
            HR.save_building_sidecar(self.cache, TEX, [self._building()], 256, 256)
            HR.save_canopy_sidecar(self.cache, TEX, np.zeros((4, 8, 8), np.float32))
        self.assertEqual(os.listdir(self.cache), [])

    def test_sidecar_keeps_plain_fields_only(self):
        det = self._building(height_m=17.5, footprint=object())
        HR.save_building_sidecar(self.cache, TEX, [det], 256, 256)
        dets = HR._load_building_detections(self.cache, TEX[:-4])
        self.assertEqual(dets[0]["height_m"], 17.5)
        self.assertNotIn("footprint", dets[0])


class HeightNetHeightsTests(unittest.TestCase):
    def test_overlay_heights_are_kept_and_only_missing_ones_predicted(self):
        hn = HR._HeightNetHeights(None, 22, 120, log=lambda m: None)
        seen = []

        def predict(img, dets, mpp):
            seen.append(len(dets))
            for d in dets[:-1]:                     # the last one HeightNet cannot do
                d["height_m"], d["height_source"] = 9.0, "heightnet"

        hn._predict = predict
        dets = [{"points": [[0, 0]], "height_m": 30.0, "height_source": "heightnet"},
                {"points": [[0, 0]], "height_m": 12.0},       # YOLO's class placeholder
                {"points": [[0, 0]], "height_m": 7.0}]
        out = hn(np.zeros((4, 4, 3), np.uint8), dets, 1.0)
        self.assertEqual(seen, [2])
        self.assertEqual(out[:2].tolist(), [30.0, 9.0])
        self.assertTrue(np.isnan(out[2]))


class StampTests(unittest.TestCase):
    def test_height_version_rides_in_the_stamp_without_changing_the_mark(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.dds")
            with open(path, "wb") as f:
                f.write(b"DDS " + bytes(12) + struct.pack("<I", 1024) + bytes(108))
            TMAT._stamp(path, 0.15)
            self.assertEqual(TMAT.map_heights(path), 0)
            mark = TMAT.map_mark(path)
            TMAT._stamp(path, 0.15, TMAT.HEIGHT_RELIEF_VERSION)
            self.assertEqual(TMAT.map_heights(path), TMAT.HEIGHT_RELIEF_VERSION)
            self.assertEqual(TMAT.map_mark(path), mark)
            self.assertEqual(TMAT.map_heights(os.path.join(d, "missing.dds")), 0)


class SlopeTests(unittest.TestCase):
    def test_metres_per_pixel_of_a_texture(self):
        mpp = TMAT.texture_m_per_px(TEX, 4096)
        self.assertGreater(mpp, 2.0)
        self.assertLess(mpp, 2.39)                               # ZL16 at ~22 N
        self.assertAlmostEqual(TMAT.texture_m_per_px(TEX, 1024), mpp * 4)
        self.assertIsNone(TMAT.texture_m_per_px("not_a_texture.dds", 1024))

    def test_flat_heights_change_nothing(self):
        rgb = np.full((64, 64, 3), 0.5, np.float32)
        plain = TMAT.material_map(rgb, 0.0)
        flat = TMAT.material_map(rgb, 0.0, heights=np.zeros((64, 64), np.float32), m_per_px=1.0)
        self.assertTrue((plain == flat).all())


if __name__ == "__main__":
    unittest.main()
