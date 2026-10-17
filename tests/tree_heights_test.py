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

import O4_Forest_Assets as FOREST_ASSETS
import O4_Forest_Heights as FOREST_HEIGHTS
import O4_Landcover_Inference as SEGFORMER
import O4_Tree_Sizes as TREE_SIZES
import O4_Vegetation_Overlay as VEG

XP_FORESTS = Path(os.environ.get(
    "O4_TEST_XP_FORESTS",
    r"D:\SteamLibrary\steamapps\common\X-Plane 12\Resources\default scenery\1200 forests"))

_FOR = """A
1200
FOREST
SHADER_2D
\tTEXTURE ../textures/trees_ALB.png
SPACING\t12 12
GROUP 1 50
\tTREE\t0\t2976\t208\t252\t104\t4\t14\t20\t1\t1\tredoak_04
\tTREE\t398\t2976\t128\t252\t64\t4\t18\t26\t1\t1\tredoak_06
GROUP 2 50
\tTREE2\t452\t2272\t120\t60\t60\t6.0\t1.6\t2.2\t1.6\t5000\t1\t2\tlilac_03
SKIP_SURFACE water
"""


def _fake_xp(root):
    os.makedirs(os.path.join(root, "sum"))
    os.makedirs(os.path.join(root, "win"))
    os.makedirs(os.path.join(root, "textures"))
    for season in ("sum", "win"):
        with open(os.path.join(root, season, "broad_tmp.for"), "w", newline="\n") as f:
            f.write(_FOR)
    with open(os.path.join(root, "sum", "broad_vhot.for"), "w", newline="\n") as f:
        f.write(_FOR)
    with open(os.path.join(root, "library.txt"), "w") as f:
        f.write("A\n1200\nLIBRARY\n"
                "EXPORT_SEASON sum\tlib/g8/broad_tmp_wet.for\t\tsum/broad_tmp.for\n"
                "EXPORT_SEASON win\tlib/g8/broad_tmp_wet.for\t\twin/broad_tmp.for\n"
                "EXPORT\t\t\t\tlib/g8/broad_vhot_wet.for\t\tsum/broad_vhot.for\n")


class TreeSizeTests(unittest.TestCase):
    def test_height_picks_a_size_whose_range_covers_it(self):
        rng = np.random.default_rng(3)
        for family, temp, moist in (("broad", "tmp", "wet"), ("coni", "cld", "wet"),
                                    ("mixed", "wrm", "dry")):
            for height in (5.0, 9.5, 14.0, 20.0):
                path = TREE_SIZES.sized_tree_path(family, temp, moist, height, rng)
                lo, hi = next((lo, hi) for sizes in TREE_SIZES.SIZED_SPECIES.values()
                              for p, lo, hi in sizes if p == path)
                self.assertLessEqual(lo - TREE_SIZES.SPECIES_TOLERANCE_M, height, msg=path)
                self.assertGreaterEqual(hi + TREE_SIZES.SPECIES_TOLERANCE_M, height, msg=path)

    def test_species_follow_the_climate_band(self):
        rng = np.random.default_rng(0)
        coni = {TREE_SIZES.sized_tree_path("coni", "cld", "wet", 14.0, rng) for _ in range(40)}
        self.assertTrue(all("/coniferous/" in p for p in coni), coni)
        tropical = {TREE_SIZES.sized_tree_path("broad", "vhot", "rain", 14.0, rng) for _ in range(40)}
        self.assertTrue(all("/evergreen/" in p for p in tropical), tropical)
        cold = {TREE_SIZES.sized_tree_path("broad", "vcld", "wet", 18.0, rng) for _ in range(40)}
        self.assertFalse(any("oak" in p or "maple" in p for p in cold), cold)

    def test_short_trees_become_shrubs(self):
        self.assertIn("shrubs_small", TREE_SIZES.sized_tree_path("broad", "tmp", "wet", 2.0))
        self.assertIn("shrubs_pines", TREE_SIZES.sized_tree_path("coni", "tmp", "wet", 3.0))
        self.assertEqual(TREE_SIZES.sized_tree_path("broad", "hot", "dry", 3.0),
                         "lib/g8/shrb_hot_dry.for")

    def test_unmeasured_crowns_and_orchards_keep_their_forest(self):
        self.assertIsNone(TREE_SIZES.sized_tree_path("broad", "tmp", "wet", 0.4))
        self.assertIsNone(TREE_SIZES.band_of_path("lib/g8/orch_tmp_wet.for"))
        self.assertEqual(TREE_SIZES.band_of_path("lib/g8/mixed_wrm_sdry.for"), ("mixed", "wrm", "sdry"))

    def test_every_band_has_a_choice(self):
        for family in FOREST_ASSETS.G8_FAMILIES:
            for temp in FOREST_ASSETS.G8_TEMPS:
                for moist in FOREST_ASSETS.G8_MOISTS:
                    for height in (2.0, 6.0, 12.0, 25.0):
                        self.assertTrue(TREE_SIZES.sized_tree_path(family, temp, moist, height),
                                        msg=(family, temp, moist, height))

    def test_crown_height_is_the_tallest_nearby(self):
        hmap = np.zeros((20, 20), np.float32)
        hmap[10, 12] = 9.0
        self.assertEqual(list(TREE_SIZES.crown_heights(hmap, [(10, 10), (2, 2)], 2)), [9.0, 0.0])

    @unittest.skipUnless((XP_FORESTS / "library.txt").is_file(), "X-Plane 12 not installed")
    def test_sized_paths_are_exported_by_xplane_with_these_heights(self):
        exports = FOREST_HEIGHTS.read_exports(str(XP_FORESTS / "library.txt"))
        shrubs = {p for p in exports if p.startswith("lib/g8/shrb_")}
        self.assertEqual(len(shrubs), 24)
        for sizes in TREE_SIZES.SIZED_SPECIES.values():
            for path, lo, hi in sizes:
                self.assertIn(path, exports, msg=path)
                rel = exports[path].get("sum") or exports[path].get("")
                heights = [(float(ln.split()[7]), float(ln.split()[8]))
                           for ln in (XP_FORESTS / rel).read_text(errors="ignore").splitlines()
                           if ln.split()[:1] == ["TREE"]]
                self.assertAlmostEqual(min(h[0] for h in heights), lo, places=1, msg=path)
                self.assertAlmostEqual(max(h[1] for h in heights), hi, places=1, msg=path)


class ForestHeightTests(unittest.TestCase):
    def test_scaling_touches_only_canopy_heights(self):
        self.assertAlmostEqual(FOREST_HEIGHTS.canopy_mean_height(_FOR), 19.5)
        scaled = FOREST_HEIGHTS.scale_canopy(_FOR, 0.5)
        self.assertAlmostEqual(FOREST_HEIGHTS.canopy_mean_height(scaled), 9.75)
        self.assertIn("1.6\t2.2", scaled)                  # understory unchanged
        self.assertIn("TEXTURE ../textures/trees_ALB.png", scaled)
        self.assertEqual(len(scaled.splitlines()), len(_FOR.splitlines()))

    def test_band_choice_and_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            xp = os.path.join(tmp, "xp")
            cs = os.path.join(tmp, "Custom Scenery")
            os.makedirs(cs)
            _fake_xp(xp)
            fh = FOREST_HEIGHTS.ForestHeights(cs, log=lambda *a: None, xp_dir=xp)
            self.assertTrue(fh.available)
            self.assertEqual(fh.path_for("lib/g8/broad_tmp_wet.for", 8.0), "o4xp/forests/h9/broad_tmp_wet.for")
            self.assertEqual(fh.path_for("lib/g8/broad_tmp_wet.for", 18.0), "lib/g8/broad_tmp_wet.for")
            self.assertEqual(fh.path_for("lib/g8/orch_tmp_wet.for", 4.0), "lib/g8/orch_tmp_wet.for")
            self.assertEqual(fh.path_for("lib/g8/broad_tmp_wet.for", None), "lib/g8/broad_tmp_wet.for")
            self.assertEqual(FOREST_HEIGHTS.source_path("o4xp/forests/h9/broad_tmp_wet.for"),
                             "lib/g8/broad_tmp_wet.for")
            self.assertTrue(fh.ensure(["o4xp/forests/h9/broad_tmp_wet.for",
                                       "o4xp/forests/h12/broad_vhot_wet.for", "lib/g8/coni_tmp_wet.for"]))
            pack = os.path.join(cs, FOREST_HEIGHTS.PACK_NAME)
            lib = open(os.path.join(pack, "library.txt")).read()
            self.assertIn("EXPORT_SEASON sum\to4xp/forests/h9/broad_tmp_wet.for\tsum_h9/broad_tmp.for", lib)
            self.assertIn("EXPORT_SEASON win\to4xp/forests/h9/broad_tmp_wet.for\twin_h9/broad_tmp.for", lib)
            self.assertIn("EXPORT\to4xp/forests/h12/broad_vhot_wet.for\tsum_h12/broad_vhot.for", lib)
            text = open(os.path.join(pack, "win_h9", "broad_tmp.for")).read()
            self.assertAlmostEqual(FOREST_HEIGHTS.canopy_mean_height(text), 9.0, places=1)
            self.assertTrue(os.path.isdir(os.path.join(pack, "textures")))
            # A second run adds nothing and keeps the file.
            self.assertTrue(fh.ensure(["o4xp/forests/h9/broad_tmp_wet.for"]))
            self.assertEqual(open(os.path.join(pack, "library.txt")).read(), lib)
            self.assertFalse(fh.ensure(["o4xp/forests/h9/coni_tmp_wet.for"]))


class OverlayHookTests(unittest.TestCase):
    def _blob_mask(self, size=200):
        mask = np.zeros((size, size), np.uint8)
        mask[90:104, 90:104] = 1          # ~14 x 14 px = 49 m2 at 0.5 m/px
        return mask

    def _run(self, mask, canopy_height, forest_heights=None):
        kw = dict(img_w=mask.shape[1], img_h=mask.shape[0],
                  lat_n=52.001, lat_s=52.0, lon_w=13.0, lon_e=13.001,
                  tile_lat=52, tile_lon=13, m_per_px=0.5, min_area_px=4,
                  simplify_px=1.0, region="northmiddle", rng=np.random.default_rng(0),
                  density_override=None, canopy_height=canopy_height,
                  forest_heights=forest_heights,
                  forest_path_mapper=_Mapper())
        return VEG._process_dds_mask(mask, SEGFORMER.CLASS_TREE, **kw)

    def test_point_trees_get_sized_species(self):
        mask = self._blob_mask()
        hmap = np.zeros(mask.shape, np.float32)
        hmap[mask > 0] = 9.5
        with mock.patch.dict(os.environ, {"O4_VEG_TREE_SIZES": "1"}):
            polys = self._run(mask, hmap)
        self.assertTrue(polys)
        for path, den, _pts in polys:
            self.assertGreaterEqual(den, VEG.POINT_MODE)
            self.assertIn(path, TREE_SIZES.SIZED_TREE_PATHS)

    def test_sized_trees_and_shorter_forests_are_off_by_default(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("O4_VEG_TREE_SIZES", "O4_VEG_FOREST_HEIGHTS")}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(VEG._tree_sizes_enabled())
            self.assertFalse(VEG._forest_heights_enabled())
            mask = self._blob_mask()
            hmap = np.zeros(mask.shape, np.float32)
            hmap[mask > 0] = 9.5
            polys = self._run(mask, hmap)
        self.assertEqual({p for p, _d, _r in polys}, {"lib/g8/broad_tmp_wet.for"})

    def test_point_trees_without_canopy_height_keep_the_forest(self):
        mask = self._blob_mask()
        polys = self._run(mask, np.zeros(mask.shape, np.float32))
        self.assertEqual({p for p, _d, _r in polys}, {"lib/g8/broad_tmp_wet.for"})


class ForestZoneTests(unittest.TestCase):
    def test_young_and_old_stands_become_separate_bands(self):
        mpp = 1.0
        mask = np.zeros((300, 400), np.uint8)
        mask[20:280, 20:380] = 1
        hmap = np.zeros(mask.shape, np.float32)
        hmap[:, :200] = 7.0          # young stand
        hmap[:, 200:] = 24.0         # old forest
        fmap = VEG._forest_height_map(hmap, mask, mpp)
        zones = VEG._forest_height_zones(fmap, mask, mpp)
        self.assertLess(zones[150, 60], zones[150, 340])   # young stand in a lower zone
        self.assertTrue((zones[mask == 0] == 0).all())
        self.assertTrue((zones[mask > 0] > 0).all())

    def test_small_zone_patches_join_their_surroundings(self):
        mpp = 1.0
        mask = np.ones((200, 200), np.uint8)
        fmap = np.full(mask.shape, 22.0, np.float32)
        fmap[95:105, 95:105] = 8.0   # 100 m2 < FOREST_ZONE_MIN_M2
        zones = VEG._forest_height_zones(fmap, mask, mpp)
        self.assertEqual(len(np.unique(zones)), 1)


class _Mapper:
    def map_path(self, path, lon, lat, veg_type=None):
        return "lib/g8/broad_tmp_wet.for"


if __name__ == "__main__":
    unittest.main()
