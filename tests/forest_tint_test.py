import math
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_Forest_Heights as FOREST_HEIGHTS
import O4_Forest_Tint as TINT
import O4_Landcover_Inference as SEGFORMER
import O4_Vegetation_Overlay as VEG

_FOR = """A
1200
FOREST
SHADER_2D
\tTEXTURE ../textures/trees_3D1_ALB.png
\tSEASONAL ../textures/trees_3D1_ALB_fa.png
\tTEXTURE_NORMAL 1\t../textures/trees_3D1_NML.png
SPACING\t12 12
GROUP 1 100
\tTREE\t0\t2976\t208\t252\t104\t4\t14\t20\t1\t1\tredoak_04
"""
_SPR = _FOR.replace("trees_3D1_ALB.png", "trees_3D1_ALB_sp.png").replace(
    "trees_3D1_ALB_fa.png", "trees_3D1_ALB.png")


def _dxt5(colours, alpha=b"\xff\x00" + b"\x00" * 6):
    """A one-mip 4 x (4 * len(colours)) DXT5 .dds whose blocks have both endpoints at an RGB colour."""
    w, h = 4 * len(colours), 4
    header = bytearray(128)
    header[:4] = b"DDS "
    struct.pack_into("<IIIII", header, 4, 124, 0x1007, h, w, 0)
    struct.pack_into("<I", header, 28, 1)
    struct.pack_into("<I", header, 76, 32)
    header[84:88] = b"DXT5"
    data = bytearray()
    for r, g, b in colours:
        c = ((r * 31 // 255) << 11) | ((g * 63 // 255) << 5) | (b * 31 // 255)
        data += alpha + struct.pack("<HHI", c, c, 0)
    return bytes(header + data)


def _endpoints(dds):
    n = (len(dds) - 128) // 16
    blocks = np.frombuffer(dds, np.uint8, count=16 * n, offset=128).reshape(n, 16)
    return TINT._unpack565(blocks[:, 8:10].copy().view("<u2").ravel())


def _lab(rgb):
    lab = cv2.cvtColor(np.asarray(rgb, np.float32).reshape(-1, 1, 3) / 255.0, cv2.COLOR_RGB2LAB)
    return lab.reshape(-1, 3)


class BucketTests(unittest.TestCase):
    def test_stock_bucket_is_xplanes_colour(self):
        self.assertEqual(TINT.BUCKETS[TINT.STOCK], (0.0, 1.0))
        a, b = TINT.bucket_ab(TINT.STOCK)
        self.assertAlmostEqual(a, TINT.REFERENCE_AB[0], places=6)
        self.assertAlmostEqual(b, TINT.REFERENCE_AB[1], places=6)

    def test_leaf_colours_turn_and_bark_stays(self):
        leaf, bark = (70, 100, 40), (90, 70, 50)
        k = TINT.BUCKETS.index((24.0, 0.66))
        out = TINT.recolour_rgb([leaf, bark], k)
        before, after = _lab([leaf, bark]), _lab(out)
        hue = lambda lab: math.degrees(math.atan2(lab[2], lab[1]))
        chroma = lambda lab: math.hypot(lab[1], lab[2])
        self.assertAlmostEqual(hue(after[0]) - hue(before[0]), 24.0, delta=0.5)
        self.assertAlmostEqual(chroma(after[0]) / chroma(before[0]), 0.66, delta=0.02)
        self.assertAlmostEqual(after[0][0], before[0][0], delta=0.5)      # lightness kept
        np.testing.assert_allclose(out[1], bark, atol=0.5)                 # bark untouched

    def test_dxt5_endpoints_change_and_alpha_and_indices_stay(self):
        src = _dxt5([(70, 100, 40), (90, 70, 50), (60, 90, 50)])
        k = TINT.BUCKETS.index((12.0, 0.82))
        out = TINT.recolour_dxt5(src, k)
        self.assertEqual(len(out), len(src))
        self.assertEqual(out[:128], src[:128])
        for i in range(3):
            o = 128 + 16 * i
            self.assertEqual(out[o:o + 8], src[o:o + 8])                   # alpha block
            self.assertEqual(out[o + 12:o + 16], src[o + 12:o + 16])       # colour indices
        ends = _endpoints(out)
        for leaf in (0, 2):
            self.assertFalse(np.array_equal(ends[leaf], _endpoints(src)[leaf]))   # leaves moved
        np.testing.assert_array_equal(ends[1], _endpoints(src)[1])         # bark kept
        self.assertEqual(TINT.recolour_dxt5(src, TINT.STOCK), src)

    def test_dxt5_reader_refuses_other_formats(self):
        bad = bytearray(_dxt5([(1, 2, 3)]))
        bad[84:88] = b"DXT1"
        with self.assertRaises(ValueError):
            TINT.recolour_dxt5(bytes(bad), 0)

    def test_for_text_points_leaves_at_the_bucket_and_summer_drops_its_autumn_blend(self):
        k = 3
        summer = TINT.tint_for_text(_FOR, k, "sum")
        self.assertIn("\tTEXTURE ../textures_t3/trees_3D1_ALB.png\n", summer)
        self.assertNotIn("SEASONAL", summer)
        self.assertIn("\tTEXTURE_NORMAL 1\t../textures/trees_3D1_NML.png\n", summer)
        spring = TINT.tint_for_text(_SPR, k, "spr")
        self.assertIn("\tTEXTURE ../textures_t3/trees_3D1_ALB_sp.png\n", spring)
        self.assertIn("\tSEASONAL ../textures_t3/trees_3D1_ALB.png\n", spring)
        autumn = TINT.tint_for_text(_FOR.replace("ALB.png", "ALB_fa.png"), k, "fal")
        self.assertNotIn("textures_t3", autumn)                            # autumn leaves kept
        stock = TINT.tint_for_text(_FOR, TINT.STOCK, "sum")
        self.assertEqual(stock, _FOR.replace("\tSEASONAL ../textures/trees_3D1_ALB_fa.png\n", ""))

    def test_zones_follow_the_canopy_colour(self):
        size = 400
        img = np.zeros((size, size, 3), np.uint8)
        target_green = TINT.BUCKETS.index((24.0, 0.66))
        a, b = TINT.bucket_ab(target_green)
        lab = np.array([[[40.0, a, b]]], np.float32)
        img[:, :200] = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) * 255, 0, 255).astype(np.uint8)
        lab = np.array([[[40.0, *TINT.REFERENCE_AB]]], np.float32)
        img[:, 200:] = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) * 255, 0, 255).astype(np.uint8)
        mask = np.zeros((size, size), np.uint8)
        mask[50:350, 20:380] = 1
        zones = TINT.tint_zones(img, mask, 1.0, sigma_m=10.0)
        self.assertEqual(zones.dtype, np.uint8)
        self.assertTrue((zones[mask == 0] == 0).all())
        self.assertEqual(int(np.median(zones[60:340, 30:150])), target_green + 1)
        self.assertEqual(int(np.median(zones[60:340, 250:370])), TINT.STOCK + 1)


class ZoneCleanupTests(unittest.TestCase):
    def test_small_patch_takes_its_majority_colour_whole(self):
        on = np.zeros((40, 40), bool)
        on[5:15, 5:15] = True
        zones = np.where(on, 3, 0).astype(np.uint8)
        zones[5:15, 12:15] = 7
        out = TINT._whole_small_patches(zones, on, min_px=200)
        self.assertEqual(set(np.unique(out[on])), {3})
        self.assertFalse(out[~on].any())

    def test_sliver_of_a_big_patch_joins_its_neighbour(self):
        zones = np.zeros((60, 60), np.uint8)
        zones[5:55, 5:55] = 2
        zones[5:55, 30:55] = 9
        zones[20:23, 24:27] = 9   # 9-px island of 9 inside the 2 side
        out = TINT._absorb_small_pieces(zones, min_px=100)
        self.assertEqual(out[21, 25], 2)
        self.assertEqual(out[30, 10], 2)
        self.assertEqual(out[30, 40], 9)
        self.assertTrue(np.array_equal(out > 0, zones > 0))


class PathTests(unittest.TestCase):
    def test_tint_combines_with_every_kind_of_copy(self):
        tp = FOREST_HEIGHTS.tinted_virtual_path
        self.assertEqual(tp("lib/g8/broad_tmp_wet.for", 3), "o4xp/forests/t3/broad_tmp_wet.for")
        self.assertEqual(tp("o4xp/forests/h9/broad_tmp_wet.for", 3), "o4xp/forests/h9-t3/broad_tmp_wet.for")
        self.assertEqual(tp("o4xp/forests/c4.5-ev/broad_tmp_wet.for", 0),
                         "o4xp/forests/c4.5-ev-t0/broad_tmp_wet.for")
        self.assertEqual(tp("o4xp/forests/ev/broad_tmp_wet.for", 7), "o4xp/forests/ev-t7/broad_tmp_wet.for")
        self.assertEqual(tp("lib/vegetation/trees/deciduous/shrubs_small.for", 2),
                         "o4xp/forests/t2/trees/deciduous/shrubs_small.for")
        pv = FOREST_HEIGHTS.parse_virtual
        self.assertEqual(pv("o4xp/forests/t3/broad_tmp_wet.for"), ("lib/g8/broad_tmp_wet.for", "t", 0.0))
        self.assertEqual(pv("o4xp/forests/h9-t3/broad_tmp_wet.for"), ("lib/g8/broad_tmp_wet.for", "h", 9.0))
        self.assertEqual(pv("o4xp/forests/ev-t7/broad_tmp_wet.for"), ("lib/g8/broad_tmp_wet.for", "e", 0.0))
        self.assertEqual(FOREST_HEIGHTS.tint_of("o4xp/forests/c4.5-ev-t0/broad_tmp_wet.for"), 0)
        self.assertIsNone(FOREST_HEIGHTS.tint_of("o4xp/forests/c4.5-ev/broad_tmp_wet.for"))
        self.assertTrue(FOREST_HEIGHTS.is_evergreen("o4xp/forests/ev-t7/broad_tmp_wet.for"))
        self.assertFalse(FOREST_HEIGHTS.is_evergreen("o4xp/forests/t7/broad_tmp_wet.for"))
        # Evergreen copies of tinted forests keep their colour.
        self.assertEqual(FOREST_HEIGHTS.evergreen_virtual_path("o4xp/forests/t7/broad_tmp_wet.for"),
                         "o4xp/forests/ev-t7/broad_tmp_wet.for")
        self.assertEqual(FOREST_HEIGHTS.evergreen_virtual_path("o4xp/forests/h9-t7/broad_tmp_wet.for"),
                         "o4xp/forests/h9-ev-t7/broad_tmp_wet.for")
        # Only cap copies drop their trees when the pack is missing.
        self.assertFalse(FOREST_HEIGHTS.is_capped("o4xp/forests/t3/broad_tmp_wet.for"))
        self.assertFalse(FOREST_HEIGHTS.is_capped("o4xp/forests/s0-t3/trees/x/shrubs.for"))
        self.assertTrue(FOREST_HEIGHTS.is_capped("o4xp/forests/s3-t3/trees/x/shrubs.for"))
        self.assertTrue(FOREST_HEIGHTS.is_capped("o4xp/forests/c6-t3/broad_tmp_wet.for"))

    def test_pack_gets_tinted_copies_and_atlases(self):
        with tempfile.TemporaryDirectory() as tmp:
            xp = os.path.join(tmp, "xp")
            cs = os.path.join(tmp, "Custom Scenery")
            os.makedirs(cs)
            for season, text in (("sum", _FOR), ("spr", _SPR)):
                os.makedirs(os.path.join(xp, season))
                with open(os.path.join(xp, season, "broad_tmp.for"), "w", newline="\n") as f:
                    f.write(text)
            os.makedirs(os.path.join(xp, "textures"))
            for atlas in TINT.ATLASES:
                for suffix in TINT.TINTED_SUFFIXES:
                    with open(os.path.join(xp, "textures", f"{atlas}{suffix}.dds"), "wb") as f:
                        f.write(_dxt5([(70, 100, 40)]))
            with open(os.path.join(xp, "library.txt"), "w") as f:
                f.write("A\n1200\nLIBRARY\n"
                        "EXPORT_SEASON sum\tlib/g8/broad_tmp_wet.for\tsum/broad_tmp.for\n"
                        "EXPORT_SEASON spr\tlib/g8/broad_tmp_wet.for\tspr/broad_tmp.for\n")
            fh = FOREST_HEIGHTS.ForestHeights(cs, log=lambda *a: None, xp_dir=xp)
            self.assertEqual(fh.tint_path("lib/g8/broad_tmp_wet.for", 3), "o4xp/forests/t3/broad_tmp_wet.for")
            self.assertEqual(fh.tint_path("simheaven/forests/x.for", 3), "simheaven/forests/x.for")
            self.assertTrue(fh.ensure(["o4xp/forests/t3/broad_tmp_wet.for",
                                       f"o4xp/forests/t{TINT.STOCK}/broad_tmp_wet.for"]))
            pack = os.path.join(cs, FOREST_HEIGHTS.PACK_NAME)
            lib = open(os.path.join(pack, "library.txt")).read()
            self.assertIn("EXPORT_SEASON sum\to4xp/forests/t3/broad_tmp_wet.for\tsum_t3/broad_tmp.for", lib)
            self.assertIn("EXPORT_SEASON spr\to4xp/forests/t3/broad_tmp_wet.for\tspr_t3/broad_tmp.for", lib)
            summer = open(os.path.join(pack, "sum_t3", "broad_tmp.for")).read()
            self.assertIn("../textures_t3/trees_3D1_ALB.png", summer)
            self.assertNotIn("SEASONAL", summer)
            for atlas in TINT.ATLASES:
                for suffix in TINT.TINTED_SUFFIXES:
                    self.assertTrue(os.path.isfile(os.path.join(pack, "textures_t3", f"{atlas}{suffix}.dds")))
            self.assertFalse(os.path.isdir(os.path.join(pack, TINT.texture_dir(TINT.STOCK))))
            stock = open(os.path.join(pack, f"sum_t{TINT.STOCK}", "broad_tmp.for")).read()
            self.assertIn("../textures/trees_3D1_ALB.png", stock)
            self.assertNotIn("SEASONAL", stock)


class OverlayTests(unittest.TestCase):
    def test_tree_mask_splits_by_tint_zone_and_paths_get_the_bucket(self):
        class _Copies:
            def tint_path(self, path, bucket):
                return FOREST_HEIGHTS.tinted_virtual_path(path, bucket)

        class _Mapper:
            def map_path(self, path, lon, lat, veg_type):
                return "lib/g8/broad_tmp_wet.for"

        size = 400
        mask = np.zeros((size, size), np.uint8)
        mask[40:360, 20:380] = 1
        tints = np.zeros_like(mask)
        tints[:, :200] = 4
        tints[:, 200:] = 9
        kw = dict(img_w=size, img_h=size, lat_n=52.004, lat_s=52.0, lon_w=13.0, lon_e=13.006,
                  tile_lat=52, tile_lon=13, m_per_px=1.0, min_area_px=4, simplify_px=1.0,
                  region="northmiddle", rng=np.random.default_rng(0), density_override=None,
                  forest_path_mapper=_Mapper(), tint_zones=tints, forest_copies=_Copies())
        polys = VEG._process_tree_mask(mask, **kw)
        paths = sorted({p for p, _d, _r in polys})
        self.assertEqual(paths, ["o4xp/forests/t3/broad_tmp_wet.for", "o4xp/forests/t8/broad_tmp_wet.for"])

    def test_tint_is_on_by_default(self):
        self.assertTrue(VEG._forest_tint_enabled())


if __name__ == "__main__":
    unittest.main()
