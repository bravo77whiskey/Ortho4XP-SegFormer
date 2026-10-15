"""XP12 terrain material maps (src/O4_Terrain_Materials.py) on synthetic tiles."""

from pathlib import Path
from types import SimpleNamespace
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

import O4_Terrain_Materials as TMAT  # noqa: E402

NVCOMPRESS = ROOT / "Utils" / "win" / "nvcompress" / "nvcompress.exe"
TEX = "27456_54768_BI16.dds"
TEX17 = "54912_109536_BI17.dds"


def _rgb(colour, size=64):
    return np.full((size, size, 3), np.array(colour, np.float32) / 255, np.float32)


def _fake_encode(rgba, out_dds, converter, relief):
    """encode_dds without nvcompress: a stamped stand-in file."""
    width = rgba.shape[1] if rgba is not None else 1024
    with open(out_dds, "wb") as f:
        f.write(b"DDS " + bytes(12) + struct.pack("<I", width) + bytes(108))
    TMAT._stamp(out_dds, relief)
    return True


def _land_ter(tex, extra=()):
    lines = ["A", "800", "TERRAIN", "", "LOAD_CENTER 22.5 120.5 2400 4096",
             f"BASE_TEX_NOWRAP ../textures/{tex}", *extra, "NO_ALPHA"]
    return "\n".join(lines) + "\n"


class MapTests(unittest.TestCase):
    def test_surface_types_from_colour(self):
        cases = {
            "vegetation": (60, 120, 40),
            "asphalt": (60, 62, 64),
            "concrete": (140, 140, 138),
            "bright": (235, 235, 230),
            "soil": (150, 100, 60),
        }
        names = list(TMAT.MATERIALS)
        for name, colour in cases.items():
            w = TMAT.surface_weights(_rgb(colour))
            self.assertAlmostEqual(float(w.sum(-1).mean()), 1.0, places=5)
            self.assertEqual(names[int(w[0, 0].argmax())], name, colour)

    def test_pure_surfaces_get_their_gloss_and_f0(self):
        for name, colour in (("vegetation", (60, 120, 40)), ("asphalt", (60, 62, 64))):
            m = TMAT.material_map(_rgb(colour), relief=0.15)
            gloss, f0 = TMAT.MATERIALS[name]
            self.assertEqual(m.dtype, np.uint8)
            self.assertEqual(m.shape, (64, 64, 4))
            self.assertEqual(int(m[5, 5, 3]), int(gloss * 255 + 0.5))
            self.assertEqual(int(m[5, 5, 2]), int(f0 * 255 + 0.5))
            # a flat colour has no relief: normal straight up
            self.assertTrue((m[..., :2] == 128).all())

    def test_relief_tilts_the_normal_and_zero_relief_does_not(self):
        rng = np.random.default_rng(0)
        img = rng.random((256, 256, 3)).astype(np.float32)
        flat = TMAT.material_map(img, relief=0.0)
        self.assertTrue((flat[..., :2] == 128).all())
        bumpy = TMAT.material_map(img, relief=0.15)
        self.assertGreater(np.abs(bumpy[..., :2].astype(int) - 128).mean(), 1)

    def test_names_and_sizes(self):
        self.assertEqual(TMAT.map_file_name(TEX), "27456_54768_BI16_mat.dds")
        self.assertTrue(TMAT.is_map_name("27456_54768_BI16_mat.dds"))
        self.assertFalse(TMAT.is_map_name(TEX))
        self.assertEqual(TMAT.map_size(16), 1024)
        self.assertEqual(TMAT.map_size(17), 1024)
        self.assertEqual(TMAT.map_size(19), 1024)
        self.assertEqual(TMAT.ter_lines(TEX), [
            "TEXTURE_NORMAL 1 ../textures/27456_54768_BI16_mat.dds",
            "GLOBAL_specular 1.0",
            "NORMAL_METALNESS",
        ])

    def test_texture_scanners_ignore_maps(self):
        import O4_Texture_Selection as TEXSEL
        import O4_Zone_Utils as ZONE
        name = TMAT.map_file_name(TEX)
        self.assertIsNone(TEXSEL.TEXTURE_NAME_RE.match(name))
        self.assertIsNone(ZONE.DDS_RE.match(name))
        self.assertIsNone(TMAT.TEXTURE_RE.match(name))


class TerFileTests(unittest.TestCase):
    def test_lines_go_right_after_base_tex_and_once(self):
        text = _land_ter(TEX).replace("\n", "\r\n")
        out = TMAT.add_ter_lines(text, TEX)
        lines = out.split("\r\n")
        i = lines.index(f"BASE_TEX_NOWRAP ../textures/{TEX}")
        self.assertEqual(lines[i + 1:i + 4], TMAT.ter_lines(TEX))
        self.assertNotIn("\n", out.replace("\r\n", ""))           # CRLF kept
        self.assertEqual(TMAT.add_ter_lines(out, TEX), out)       # idempotent
        self.assertEqual(TMAT.strip_ter_lines(out, TEX), text)

    def test_land_ter_detection(self):
        land = _land_ter(TEX).splitlines()
        self.assertTrue(TMAT.is_land_ter("27456_54768_BI16.ter", land))
        for name in ("27456_54768_BI16_water.ter", "27456_54768_BI16_sea.ter",
                     "27456_54768_BI16_water_overlay.ter", "27456_54768_BI16_sea_overlay.ter"):
            self.assertFalse(TMAT.is_land_ter(name, land), name)
        self.assertFalse(TMAT.is_land_ter("27456_54768_BI16.ter", land + ["WET"]))


class CreateTerrainFileTests(unittest.TestCase):
    def setUp(self):
        import O4_DSF_Utils as DSF
        self.DSF = DSF
        self.tmp = tempfile.TemporaryDirectory()
        self.tile = SimpleNamespace(
            build_dir=self.tmp.name, use_decal_on_terrain=False, terrain_decal="o4xp",
            terrain_casts_shadows=True, imprint_masks_to_dds=True, mask_zl=14,
            terrain_materials=True)
        os.makedirs(os.path.join(self.tmp.name, "textures"))

    def tearDown(self):
        self.tmp.cleanup()

    def _ter(self, tri_type, is_overlay):
        name = self.DSF.create_terrain_file(self.tile, TEX, 54768, 27456, 16, "BI",
                                            tri_type, is_overlay)
        with open(os.path.join(self.tmp.name, "terrain", name)) as f:
            return name, f.read()

    def test_land_terrain_gets_the_material_lines(self):
        name, text = self._ter(0, False)
        self.assertEqual(name, "27456_54768_BI16.ter")
        lines = text.splitlines()
        i = lines.index(f"BASE_TEX_NOWRAP ../textures/{TEX}")
        self.assertEqual(lines[i + 1:i + 4], TMAT.ter_lines(TEX))
        self.assertNotIn("#", text)

    def test_water_and_overlay_terrain_do_not(self):
        for tri_type, is_overlay in ((1, True), (2, False), (2, True), (1, False)):
            name, text = self._ter(tri_type, is_overlay)
            self.assertNotIn("TEXTURE_NORMAL", text, name)
            self.assertNotIn("NORMAL_METALNESS", text, name)

    def test_off_switch(self):
        self.tile.terrain_materials = False
        _, text = self._ter(0, False)
        self.assertNotIn("TEXTURE_NORMAL", text)


class FinalizeTileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.ter_dir = os.path.join(self.dir, "terrain")
        self.tex_dir = os.path.join(self.dir, "textures")
        os.makedirs(self.ter_dir)
        os.makedirs(self.tex_dir)
        patcher = mock.patch.object(TMAT, "encode_dds", side_effect=_fake_encode)
        self.encode = patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def _texture(self, name, colour=(60, 120, 40)):
        Image.new("RGB", (64, 64), colour).save(os.path.join(self.tex_dir, name), "DDS")

    def _write_ter(self, name, text):
        with open(os.path.join(self.ter_dir, name), "w", newline="") as f:
            f.write(text)

    def _read_ter(self, name):
        with open(os.path.join(self.ter_dir, name), newline="") as f:
            return f.read()

    def test_build_makes_missing_maps_and_is_idempotent(self):
        self._texture(TEX)
        patched = TMAT.add_ter_lines(_land_ter(TEX), TEX)
        self._write_ter("27456_54768_BI16.ter", patched)
        s = TMAT.finalize_tile(self.dir, 0.15, "nvcompress.exe")
        self.assertEqual((s["made"], s["kept"]), (1, 0))
        self.assertEqual(TMAT.map_mark(os.path.join(self.tex_dir, TMAT.map_file_name(TEX))),
                         (TMAT.MARK_VERSION, 1500))
        self.assertEqual(self._read_ter("27456_54768_BI16.ter"), patched)
        s = TMAT.finalize_tile(self.dir, 0.15, "nvcompress.exe")
        self.assertEqual((s["made"], s["kept"]), (0, 1))
        # another relief remakes it
        s = TMAT.finalize_tile(self.dir, 0.3, "nvcompress.exe")
        self.assertEqual(s["made"], 1)

    def test_build_remakes_old_2048_map(self):
        self._texture(TEX17)
        self._write_ter("54912_109536_BI17.ter", TMAT.add_ter_lines(_land_ter(TEX17), TEX17))
        old = os.path.join(self.tex_dir, TMAT.map_file_name(TEX17))
        _fake_encode(np.zeros((2048, 2048, 4), np.uint8), old, None, 0.15)
        s = TMAT.finalize_tile(self.dir, 0.15, "nvcompress.exe")
        self.assertEqual((s["made"], s["kept"]), (1, 0))
        self.assertEqual(TMAT.map_width(old), 1024)

    def test_build_drops_lines_without_texture_and_leaves_plain_ter(self):
        patched = TMAT.add_ter_lines(_land_ter(TEX), TEX)
        self._write_ter("27456_54768_BI16.ter", patched)
        self._write_ter("27456_54784_BI16.ter", _land_ter("27456_54784_BI16.dds"))
        s = TMAT.finalize_tile(self.dir, 0.15, "nvcompress.exe")
        self.assertEqual(s["removed"], 1)
        self.assertEqual(self._read_ter("27456_54768_BI16.ter"), _land_ter(TEX))
        self.assertEqual(self._read_ter("27456_54784_BI16.ter"), _land_ter("27456_54784_BI16.dds"))

    def test_stopped_build_makes_nothing(self):
        self._texture(TEX)
        self._write_ter("27456_54768_BI16.ter", TMAT.add_ter_lines(_land_ter(TEX), TEX))
        s = TMAT.finalize_tile(self.dir, 0.15, "nvcompress.exe", make=False)
        self.assertEqual((s["made"], s["removed"]), (0, 1))
        self.encode.assert_not_called()

    def test_retrofit_patches_land_only_and_is_idempotent(self):
        self._texture(TEX)
        self._texture(TEX17)
        self._write_ter("27456_54768_BI16.ter", _land_ter(TEX))
        self._write_ter("54912_109536_BI17.ter", _land_ter(TEX17).replace("\n", "\r\n"))
        water = _land_ter(TEX).replace("NO_ALPHA", "WET")
        self._write_ter("27456_54768_BI16_water_overlay.ter", water)
        # tile patched by the xp12-lighting prototype: left alone
        proto = _land_ter("27456_54784_BI16.dds",
                          ["TEXTURE_NORMAL 1 ../textures/27456_54784_BI16_mat.png",
                           "GLOBAL_specular 1.0", "NORMAL_METALNESS"])
        self._texture("27456_54784_BI16.dds")
        self._write_ter("27456_54784_BI16.ter", proto)
        s = TMAT.retrofit_tile(self.dir, 0.15, "nvcompress.exe")
        self.assertEqual((s["made"], s["added"], s["skipped"]), (2, 2, 1))
        self.assertEqual(self._read_ter("27456_54768_BI16.ter"),
                         TMAT.add_ter_lines(_land_ter(TEX), TEX))
        crlf = self._read_ter("54912_109536_BI17.ter")
        self.assertIn("\r\nTEXTURE_NORMAL 1 ../textures/54912_109536_BI17_mat.dds\r\n", crlf)
        self.assertEqual(self._read_ter("27456_54768_BI16_water_overlay.ter"), water)
        self.assertEqual(self._read_ter("27456_54784_BI16.ter"), proto)
        sizes = sorted(call.args[0].shape[0] for call in self.encode.call_args_list)
        self.assertEqual(sizes, [1024, 1024])
        s = TMAT.retrofit_tile(self.dir, 0.15, "nvcompress.exe")
        self.assertEqual((s["made"], s["added"], s["kept"]), (0, 0, 2))

    def test_prune_removes_only_our_unreferenced_maps(self):
        self._texture(TEX)
        self._write_ter("27456_54768_BI16.ter", TMAT.add_ter_lines(_land_ter(TEX), TEX))
        TMAT.finalize_tile(self.dir, 0.15, "nvcompress.exe")
        orphan = os.path.join(self.tex_dir, "27456_54800_BI16_mat.dds")
        _fake_encode(None, orphan, None, 0.15)
        foreign = os.path.join(self.tex_dir, "27456_54816_BI16_mat.dds")
        with open(foreign, "wb") as f:
            f.write(b"DDS " + bytes(124))
        self.assertEqual(TMAT.prune_tile(self.dir), 1)
        self.assertFalse(os.path.exists(orphan))
        self.assertTrue(os.path.exists(foreign))
        self.assertTrue(os.path.exists(os.path.join(self.tex_dir, TMAT.map_file_name(TEX))))


@unittest.skipUnless(NVCOMPRESS.is_file(), "nvcompress not available")
class EncodeTests(unittest.TestCase):
    def test_bc3_round_trip_keeps_gloss_and_stamp(self):
        rng = np.random.default_rng(1)
        img = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
        img[:128] = (60, 120, 40)                                  # vegetation
        img[128:] = (60, 62, 64)                                   # asphalt
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "54912_109536_BI17.png")
            Image.fromarray(img).save(src)
            out = os.path.join(tmp, "54912_109536_BI17_mat.dds")
            self.assertTrue(TMAT.write_map(src, out, 17, 0.15, str(NVCOMPRESS)))
            self.assertEqual(TMAT.map_mark(out), (TMAT.MARK_VERSION, 1500))
            dec = np.asarray(Image.open(out).convert("RGBA"))
            self.assertEqual(dec.shape, (1024, 1024, 4))
            self.assertEqual(TMAT.map_width(out), 1024)
            self.assertAlmostEqual(dec[100:400, :, 3].mean(), 0.05 * 255, delta=2)
            self.assertAlmostEqual(dec[600:950, :, 3].mean(), 0.30 * 255, delta=2)
            self.assertEqual(sorted(os.listdir(tmp)), sorted([os.path.basename(src),
                                                              os.path.basename(out)]))


if __name__ == "__main__":
    unittest.main()
