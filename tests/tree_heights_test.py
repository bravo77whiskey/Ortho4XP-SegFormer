import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_Approach_Heights as APPROACH
import O4_Apt_Pavement as APT
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
MESH\t3D_oak\t0\t500\t2\t3\t0.8\t2.0\t10.0\t
VERTEX\t0.5 0 0.25\t0 1 0\t0.1\t0.2\t0\t0\t0.1
VERTEX\t-0.5 14 -0.25\t0 1 0\t0.1\t0.2\t0\t0\t0.1
IDX\t0\t1\t0
MESH\t3D_lilac\t0\t500\t1\t3\t2.0\t0.6\t10.0\t
VERTEX\t0 1.6 0\t0 1 0\t0.1\t0.2\t0\t0\t0.1
IDX\t0\t0\t0
GROUP 1 50
\tTREE\t0\t2976\t208\t252\t104\t4\t14\t20\t1\t1\tredoak_04
\tMESH_3D\t3D_oak
\tTREE\t398\t2976\t128\t252\t64\t4\t18\t26\t1\t1\tredoak_06
GROUP 2 40
\tTREE2\t452\t2272\t120\t60\t60\t6.0\t1.6\t2.2\t1.6\t5000\t1\t2\tlilac_03
\tMESH_3D\t3D_lilac
GROUP 2 10
\tTREE\t0\t3232\t164\t188\t82\t6\t9\t13\t1\t2\toak_understory
\tMESH_3D\t3D_oak
SKIP_SURFACE water
"""


def _mesh_tops(text):
    tops, cur = {}, None
    for line in text.splitlines():
        p = line.split()
        if p[:1] == ["MESH"]:
            cur = p[1]
            tops[cur] = -1e9
        elif p[:1] == ["VERTEX"] and cur:
            tops[cur] = max(tops[cur], float(p[2]))
    return tops


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

    def test_a_cap_keeps_sizes_under_it(self):
        rng = np.random.default_rng(1)
        for _ in range(30):
            path = TREE_SIZES.sized_tree_path("broad", "tmp", "wet", 20.0, rng, max_h=9.0)
            hi = next(hi for sizes in TREE_SIZES.SIZED_SPECIES.values() for p, _lo, hi in sizes if p == path)
            self.assertLessEqual(hi, 9.0, msg=path)
        self.assertIsNone(TREE_SIZES.sized_tree_path("broad", "hot", "wet", 20.0, max_h=3.0))
        self.assertIn("shrubs_small", TREE_SIZES.sized_tree_path("broad", "tmp", "wet", 20.0, max_h=3.0))

    def test_low_caps_turn_forests_into_shrubs_that_fit(self):
        shrub = {cap: TREE_SIZES.shrub_path("lib/g8/broad_tmp_wet.for", cap) for cap in (1, 3, 4.5, 6, 7.5)}
        self.assertTrue(shrub[1].endswith("shrubs_small.for"))
        self.assertTrue(shrub[3].endswith("shrubs_small.for"))
        self.assertTrue(shrub[4.5].endswith("shrubs_medium.for"))
        self.assertTrue(shrub[6].endswith("shrubs_medium.for"))
        self.assertTrue(shrub[7.5].endswith("shrubs_big.for"))
        self.assertTrue(TREE_SIZES.shrub_path("lib/g8/coni_cld_wet.for", 6).endswith("shrubs_pines.for"))
        self.assertEqual(TREE_SIZES.shrub_path("lib/g8/broad_vhot_rain.for", 3), "lib/g8/shrb_vhot_rain.for")
        self.assertEqual(TREE_SIZES.shrub_path("lib/g8/orch_hot_wet.for", 3), "lib/g8/shrb_hot_wet.for")
        self.assertTrue(TREE_SIZES.fits_under(shrub[6], 6.0))
        self.assertFalse(TREE_SIZES.fits_under(shrub[4.5], 4.5))
        self.assertFalse(TREE_SIZES.fits_under("lib/g8/broad_tmp_wet.for", 30.0))

    def test_low_measured_forests_take_the_nearest_shrubs(self):
        self.assertTrue(TREE_SIZES.shrub_for_height("lib/g8/broad_tmp_wet.for", 0.3).endswith("shrubs_small.for"))
        self.assertTrue(TREE_SIZES.shrub_for_height("lib/g8/broad_tmp_wet.for", 4.0).endswith("shrubs_medium.for"))
        self.assertTrue(TREE_SIZES.shrub_for_height("lib/g8/mixed_tmp_wet.for", 3.1).endswith("shrubs_small.for"))
        self.assertTrue(TREE_SIZES.shrub_for_height("lib/g8/coni_cld_wet.for", 4.0).endswith("shrubs_pines.for"))
        self.assertEqual(TREE_SIZES.shrub_for_height("lib/g8/broad_hot_wet.for", 2.0), "lib/g8/shrb_hot_wet.for")

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

    def test_meshes_scale_with_their_trees(self):
        scaled = FOREST_HEIGHTS.scale_canopy(_FOR, 0.5)
        tops = _mesh_tops(scaled)
        # 3D_oak is shared by a canopy tree (x0.5) and the understory (x1):
        # the unscaled one keeps the name, the canopy tree gets a scaled copy.
        self.assertEqual(tops["3D_oak"], 14.0)
        self.assertEqual(tops["3D_oak_o4s1"], 7.0)
        self.assertEqual(tops["3D_lilac"], 1.6)
        self.assertIn("MESH\t3D_oak_o4s1\t0\t500\t2\t3\t0.8\t1\t10.0", scaled)   # wind sway halved
        refs = [ln.split()[1] for ln in scaled.splitlines() if ln.split()[:1] == ["MESH_3D"]]
        self.assertEqual(refs, ["3D_oak_o4s1", "3D_lilac", "3D_oak"])
        self.assertIn("\t7\t10\t1\t1\tredoak_04", scaled)

    def test_cap_copies_cut_every_tree_down_to_the_cap(self):
        capped = FOREST_HEIGHTS.cap_trees(_FOR, 9.0)
        self.assertEqual(FOREST_HEIGHTS.max_tree_height(capped), 9.0)
        self.assertIn("1.6\t2.2", capped)                  # short trees unchanged
        tops = _mesh_tops(capped)
        # The two oaks are cut by 9/20 and 9/13: one keeps the mesh, one a copy.
        self.assertEqual(sorted(round(v, 2) for k, v in tops.items() if k.startswith("3D_oak")),
                         [6.3, 9.69])

    def test_copies_rename_their_meshes_and_close_up_their_grid(self):
        band = FOREST_HEIGHTS.band_trees(_FOR, 6.0, 19.5, "_o4h6")
        tops = _mesh_tops(band)
        self.assertEqual(set(tops), {"3D_oak_o4h6", "3D_oak_o4s1_o4h6", "3D_lilac_o4h6"})
        refs = {ln.split()[1] for ln in band.splitlines() if ln.split()[:1] == ["MESH_3D"]}
        self.assertEqual(refs, set(tops))
        # Canopy x 6/19.5 = 0.31: the 12 m grid would be 3.7 m, held at 4 m.
        self.assertIn("SPACING\t4\t4", band)
        capped = FOREST_HEIGHTS.cap_trees(_FOR, 15.0, "_o4c15")
        self.assertIn("SPACING\t", capped)
        grid = float([ln for ln in capped.splitlines() if ln.startswith("SPACING")][0].split()[1])
        self.assertAlmostEqual(grid, 12 * (15 / 20 + 15 / 26) / 2, places=1)

    def test_numbers_are_written_without_exponents(self):
        # X-Plane reads "-6.9e-09" as -6.9: near-zero vertices became spikes.
        for v in (-6.966618e-09, 2.675e-06, 1e-12, -0.0, 0.0001234567, 12.3456789):
            text = FOREST_HEIGHTS._fmt_fine(v)
            self.assertNotIn("e", text.lower(), msg=text)
            self.assertAlmostEqual(float(text), v, places=6)
        self.assertEqual(FOREST_HEIGHTS._fmt_fine(-6.966618e-09), "0")
        scaled = FOREST_HEIGHTS.scale_canopy(_FOR.replace("0.5 0 0.25", "0.5 0.00000001 0.25"), 0.5)
        self.assertNotRegex(scaled, r"\de-")

    def test_shrub_copies_plant_on_a_close_grid(self):
        shrubs = _FOR.replace("SPACING\t12 12", "SPACING\t10 10\nRANDOM\t4 4")
        dense = FOREST_HEIGHTS.shrub_trees(shrubs, None, "_o4s0")
        self.assertIn("SPACING\t3\t3", dense)
        self.assertIn("RANDOM\t1.2\t1.2", dense)
        self.assertEqual(FOREST_HEIGHTS.max_tree_height(dense), 26.0)      # no cap: heights kept
        # Cut down, the shrubs' crowns shrink and their grid closes up with them.
        capped = FOREST_HEIGHTS.shrub_trees(shrubs, 3.0, "_o4s3")
        self.assertEqual(FOREST_HEIGHTS.max_tree_height(capped), 3.0)
        self.assertIn("SPACING\t1.5\t1.5", capped)                         # held at the floor
        half = FOREST_HEIGHTS.shrub_trees(shrubs, 13.0, "_o4s13")
        grid = float([ln for ln in half.splitlines() if ln.startswith("SPACING")][0].split()[1])
        self.assertAlmostEqual(grid, 3 * (13 / 20 + 13 / 26) / 2, places=1)
        self.assertEqual(FOREST_HEIGHTS.parse_virtual("o4xp/forests/s0/trees/deciduous/shrubs_small.for"),
                         ("lib/vegetation/trees/deciduous/shrubs_small.for", "s", 0.0))

    def test_band_copies_keep_the_understory_under_the_band(self):
        band = FOREST_HEIGHTS.band_trees(_FOR, 6.0, 19.5)
        self.assertAlmostEqual(FOREST_HEIGHTS.canopy_mean_height(band), 6.0, places=1)
        self.assertEqual(FOREST_HEIGHTS.max_tree_height(band), 8.0)   # 26 m oak x 6/19.5
        self.assertIn("\t4.15\t6\t1\t2\toak_understory", band)

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
            # Approach caps: the band copy while its tallest tree fits, else the cap copy.
            self.assertEqual(fh.path_for("lib/g8/broad_tmp_wet.for", None, 40.0), "lib/g8/broad_tmp_wet.for")
            self.assertEqual(fh.path_for("lib/g8/broad_tmp_wet.for", None, 17.0),
                             "o4xp/forests/c15/broad_tmp_wet.for")
            self.assertEqual(fh.path_for("lib/g8/broad_tmp_wet.for", 8.0, 13.0),
                             "o4xp/forests/h9/broad_tmp_wet.for")
            self.assertEqual(fh.path_for("lib/g8/broad_tmp_wet.for", 8.0, 10.0),
                             "o4xp/forests/c9/broad_tmp_wet.for")
            self.assertEqual(fh.path_for("lib/g8/broad_tmp_wet.for", None, 2.0),
                             "o4xp/forests/c2/broad_tmp_wet.for")
            self.assertIsNone(fh.path_for("lib/g8/broad_tmp_wet.for", None, 0.5))
            self.assertIsNone(fh.path_for("simheaven/forests/x.for", None, 20.0))
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
            self.assertTrue(fh.ensure(["o4xp/forests/c6/broad_tmp_wet.for"]))
            text = open(os.path.join(pack, "sum_c6", "broad_tmp.for")).read()
            self.assertEqual(FOREST_HEIGHTS.max_tree_height(text), 6.0)

    def test_copies_from_an_earlier_rule_are_remade_under_their_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            xp = os.path.join(tmp, "xp")
            cs = os.path.join(tmp, "Custom Scenery")
            _fake_xp(xp)
            pack = os.path.join(cs, FOREST_HEIGHTS.PACK_NAME)
            os.makedirs(os.path.join(pack, "sum_h9"))
            old = os.path.join(pack, "sum_h9", "broad_tmp.for")
            with open(old, "w") as f:
                f.write("v1 copy without scaled meshes")
            with open(os.path.join(pack, "library.txt"), "w") as f:
                f.write("A\n1200\nLIBRARY\n\n"
                        "EXPORT_SEASON sum\to4xp/forests/h9/broad_tmp_wet.for\tsum_h9/broad_tmp.for\n")
            fh = FOREST_HEIGHTS.ForestHeights(cs, log=lambda *a: None, xp_dir=xp)
            self.assertTrue(fh.ensure(["o4xp/forests/c6/broad_tmp_wet.for"]))
            remade = open(old).read()
            self.assertAlmostEqual(FOREST_HEIGHTS.canopy_mean_height(remade), 9.0, places=1)
            self.assertAlmostEqual(_mesh_tops(remade)["3D_oak_o4h9"], 6.46, places=2)
            lib = open(os.path.join(pack, "library.txt")).read()
            self.assertIn("o4xp/forests/h9/broad_tmp_wet.for\twin_h9/broad_tmp.for", lib)
            self.assertIn("o4xp/forests/c6/broad_tmp_wet.for", lib)
            self.assertEqual(open(os.path.join(pack, FOREST_HEIGHTS.VERSION_FILE)).read().strip(),
                             FOREST_HEIGHTS.SIGNATURE)

    def test_evergreen_copies_keep_the_summer_forest_all_year(self):
        with tempfile.TemporaryDirectory() as tmp:
            xp = os.path.join(tmp, "xp")
            cs = os.path.join(tmp, "Custom Scenery")
            os.makedirs(cs)
            _fake_xp(xp)
            summer = os.path.join(xp, "sum", "broad_tmp.for")
            with open(summer, "w", newline="\n") as f:
                f.write(_FOR.replace("\tTEXTURE ../textures/trees_ALB.png\n",
                                     "\tTEXTURE ../textures/trees_ALB.png\n"
                                     "\tSEASONAL ../textures/trees_ALB_fa.png\n"))
            fh = FOREST_HEIGHTS.ForestHeights(cs, log=lambda *a: None, xp_dir=xp)
            self.assertTrue(fh.is_seasonal("lib/g8/broad_tmp_wet.for"))
            self.assertTrue(fh.is_seasonal("o4xp/forests/c6/broad_tmp_wet.for"))
            self.assertFalse(fh.is_seasonal("lib/g8/broad_vhot_wet.for"))   # X-Plane's evergreen
            self.assertFalse(fh.is_seasonal("simheaven/forests/x.for"))
            self.assertEqual(fh.evergreen_path("lib/g8/broad_tmp_wet.for"), "o4xp/forests/ev/broad_tmp_wet.for")
            self.assertEqual(fh.evergreen_path("lib/g8/broad_vhot_wet.for"), "lib/g8/broad_vhot_wet.for")
            self.assertEqual(fh.evergreen_path("o4xp/forests/c4.5/broad_tmp_wet.for"),
                             "o4xp/forests/c4.5-ev/broad_tmp_wet.for")
            self.assertEqual(FOREST_HEIGHTS.evergreen_virtual_path(
                "lib/vegetation/trees/deciduous/shrubs_small.for"),
                "o4xp/forests/ev/trees/deciduous/shrubs_small.for")
            self.assertEqual(FOREST_HEIGHTS.parse_virtual("o4xp/forests/c4.5-ev/broad_tmp_wet.for"),
                             ("lib/g8/broad_tmp_wet.for", "c", 4.5))
            self.assertEqual(FOREST_HEIGHTS.parse_virtual("o4xp/forests/ev/trees/deciduous/shrubs_small.for"),
                             ("lib/vegetation/trees/deciduous/shrubs_small.for", "e", 0.0))
            self.assertTrue(fh.ensure(["o4xp/forests/ev/broad_tmp_wet.for",
                                       "o4xp/forests/c6-ev/broad_tmp_wet.for"]))
            pack = os.path.join(cs, FOREST_HEIGHTS.PACK_NAME)
            lib = open(os.path.join(pack, "library.txt")).read()
            # One plain EXPORT of the summer file: no season swaps to the winter one.
            self.assertIn("EXPORT\to4xp/forests/ev/broad_tmp_wet.for\tsum_ev/broad_tmp.for", lib)
            self.assertIn("EXPORT\to4xp/forests/c6-ev/broad_tmp_wet.for\tsum_c6-ev/broad_tmp.for", lib)
            self.assertNotIn("EXPORT_SEASON", lib)
            plain = open(os.path.join(pack, "sum_ev", "broad_tmp.for")).read()
            self.assertEqual(plain, _FOR)                       # the stock forest minus SEASONAL
            capped = open(os.path.join(pack, "sum_c6-ev", "broad_tmp.for")).read()
            self.assertNotIn("SEASONAL", capped)
            self.assertEqual(FOREST_HEIGHTS.max_tree_height(capped), 6.0)

    def test_tropical_polygons_take_evergreen_copies(self):
        class _Copies:
            def is_seasonal(self, path):
                return "tmp" in path

        ring = [(35.3, 0.45), (35.31, 0.45), (35.31, 0.46), (35.3, 0.45)]
        north = [(lon, lat + 30) for lon, lat in ring]
        polys, n = VEG._evergreen_in_tropics(
            [("lib/g8/broad_tmp_wet.for", 255, VEG.HoledRing(ring, [ring[::-1]])),
             ("lib/g8/broad_vhot_wet.for", 255, ring),
             ("o4xp/forests/s0/trees/deciduous/shrubs_tmp.for", 512 + 255, ring[:2]),
             ("lib/g8/broad_tmp_wet.for", 255, north)], _Copies())
        self.assertEqual(n, 2)
        self.assertEqual([p for p, _d, _r in polys], [
            "o4xp/forests/ev/broad_tmp_wet.for", "lib/g8/broad_vhot_wet.for",
            "o4xp/forests/s0-ev/trees/deciduous/shrubs_tmp.for", "lib/g8/broad_tmp_wet.for"])
        self.assertEqual(polys[0][2].holes, [ring[::-1]])   # clearings kept
        self.assertTrue(VEG._tile_reaches_tropics(0) and VEG._tile_reaches_tropics(-24)
                        and VEG._tile_reaches_tropics(23))
        self.assertFalse(VEG._tile_reaches_tropics(24) or VEG._tile_reaches_tropics(-25))


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

    def test_sized_trees_shorter_forests_and_approach_caps_are_on_by_default(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("O4_VEG_TREE_SIZES", "O4_VEG_FOREST_HEIGHTS", "O4_VEG_APPROACH_HEIGHTS")}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(VEG._tree_sizes_enabled())
            self.assertTrue(VEG._forest_heights_enabled())
            self.assertTrue(VEG._approach_heights_enabled())
        with mock.patch.dict(os.environ, {"O4_VEG_TREE_SIZES": "0"}):
            mask = self._blob_mask()
            hmap = np.zeros(mask.shape, np.float32)
            hmap[mask > 0] = 9.5
            polys = self._run(mask, hmap)
        self.assertEqual({p for p, _d, _r in polys}, {"lib/g8/broad_tmp_wet.for"})

    def _capped(self, mask, height_cap, **extra):
        return VEG._process_dds_mask(
            mask, SEGFORMER.CLASS_TREE, img_w=mask.shape[1], img_h=mask.shape[0],
            lat_n=52.001, lat_s=52.0, lon_w=13.0, lon_e=13.001, tile_lat=52, tile_lon=13,
            m_per_px=0.5, min_area_px=4, simplify_px=1.0, region="northmiddle",
            rng=np.random.default_rng(0), density_override=extra.pop("density_override_", None),
            forest_path_mapper=_Mapper(), height_cap=height_cap, **extra)

    def test_capped_single_trees_take_a_size_that_fits(self):
        mask = self._blob_mask()
        hmap = np.zeros(mask.shape, np.float32)
        hmap[mask > 0] = 15.0
        with mock.patch.dict(os.environ, {"O4_VEG_TREE_SIZES": "1"}):
            polys = self._capped(mask, 6.0, canopy_height=hmap, forest_heights=_CapCopies())
        self.assertTrue(polys)
        for path, _den, _pts in polys:
            hi = next(hi for sizes in TREE_SIZES.SIZED_SPECIES.values()
                      for p, _lo, hi in sizes if p == path)
            self.assertLessEqual(hi, 6.0, msg=path)

    def test_short_single_trees_stand_closer(self):
        self.assertEqual(VEG._point_tree_spacing(None), VEG.POINT_TREE_SPACING_M)
        self.assertEqual(VEG._point_tree_spacing(25.0), VEG.POINT_TREE_SPACING_M)
        self.assertAlmostEqual(VEG._point_tree_spacing(6.0), 4.8)
        self.assertEqual(VEG._point_tree_spacing(1.0), VEG.POINT_TREE_MIN_SPACING_M)
        mask = np.zeros((200, 200), np.uint8)
        mask[60:140, 60:140] = 1
        prep = VEG._contour_fill_stats(
            cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0], 0.5,
            include_ring=False)
        wide = VEG._crown_points(mask, prep, 0.5)
        close = VEG._crown_points(mask, prep, 0.5, VEG._point_tree_spacing(5.0))
        self.assertGreater(len(close), len(wide))

    def test_only_narrow_bands_plant_along_their_outline(self):
        def shape(width_px):
            mask = np.zeros((120, 700), np.uint8)
            mask[20:20 + width_px, 20:620] = 1
            cnt = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
            return VEG._contour_shape(cnt, 4, 1.0)[0]
        self.assertEqual(shape(8), "treeline")     # a tree row: one crown a side
        self.assertEqual(shape(22), "area")        # a 22 m belt: filled, not two rows

    def test_shrub_bands_wider_than_two_shrubs_are_filled(self):
        mask = np.zeros((60, 700), np.uint8)
        mask[20:40, 20:680] = 1                   # 10 m x 330 m at 0.5 m/px: a treeline
        hmap = np.full(mask.shape, 3.0, np.float32)
        polys = self._capped(mask, None, forest_heights=_CapCopies(), forest_height_map=hmap)
        self.assertTrue(polys)
        for path, den, _r in polys:
            self.assertIn("/s0/", path)
            self.assertLess(den, VEG.LINE_MODE)   # filled shrubs, not a ring

    def test_capped_forests_take_the_cap_copy(self):
        mask = np.zeros((200, 200), np.uint8)
        mask[20:180, 20:180] = 1
        polys = self._capped(mask, 9.0, forest_heights=_CapCopies())
        self.assertEqual({p for p, _d, _r in polys}, {"o4xp/forests/c9/broad_tmp_wet.for"})

    def test_without_copies_nothing_is_planted_under_a_cap(self):
        mask = np.zeros((200, 200), np.uint8)
        mask[20:180, 20:180] = 1
        self.assertEqual(self._capped(mask, 9.0), [])

    def test_forests_measured_under_the_smallest_tree_become_shrubs(self):
        mask = np.zeros((200, 200), np.uint8)
        mask[20:180, 20:180] = 1
        hmap = np.full(mask.shape, 3.8, np.float32)
        polys = self._capped(mask, None, forest_heights=_CapCopies(), forest_height_map=hmap)
        self.assertEqual({p for p, _d, _r in polys}, {"o4xp/forests/s0/trees/deciduous/shrubs_medium.for"})
        # Shrubs plant at the full-forest density, whatever the woodland thinning.
        self.assertEqual({d for _p, d, _r in polys}, {VEG._BASE_DENSITY['tree'][100]})
        hmap[:] = 8.0                         # a tree band: not shrubs
        polys = self._capped(mask, None, forest_heights=_CapCopies(), forest_height_map=hmap)
        self.assertEqual({p for p, _d, _r in polys}, {"lib/g8/broad_tmp_wet.for"})

    def test_capped_forests_under_the_shrub_limit_become_shrubs(self):
        mask = np.zeros((200, 200), np.uint8)
        mask[20:180, 20:180] = 1
        polys = self._capped(mask, 6.0)          # shrubs_medium fits even without copies
        self.assertEqual({p for p, _d, _r in polys}, {"lib/vegetation/trees/deciduous/shrubs_medium.for"})
        polys = self._capped(mask, 2.0, forest_heights=_CapCopies())
        self.assertEqual({p for p, _d, _r in polys}, {"o4xp/forests/s2/trees/deciduous/shrubs_small.for"})
        polys = self._capped(mask, 2.0, forest_heights=_CapCopies(), density_override_=0.4)
        self.assertEqual({d for _p, d, _r in polys}, {102})

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

    def test_flat_canopy_outside_airports_has_no_height(self):
        mpp = 1.0
        mask = np.zeros((300, 600), np.uint8)
        mask[20:280, 20:580] = 1
        hmap = np.zeros(mask.shape, np.float32)     # the network sees no tree...
        hmap[:, 400:] = 12.0                        # ...except in the east
        inside = VEG._forest_height_map(hmap, mask, mpp)
        self.assertLess(inside[150, 100], 1.0)       # inside airports: flat = low (shrubs)
        outside = VEG._forest_height_map(hmap, mask, mpp, flat_unknown=np.ones(mask.shape, bool))
        self.assertTrue(np.isnan(outside[150, 100]))   # outside: unknown, full-height forest
        self.assertAlmostEqual(float(outside[150, 500]), 12.0, delta=1.0)
        zones = VEG._forest_height_zones(outside, mask, mpp)
        self.assertEqual(zones[150, 100], len(VEG.FOREST_ZONE_EDGES_M) + 2)   # the unknown zone


class _Mapper:
    def map_path(self, path, lon, lat, veg_type=None):
        return "lib/g8/broad_tmp_wet.for"


class _CapCopies:
    """path_for of a 20 m forest: the cap copy whenever a cap is given."""

    def path_for(self, path, measured_h, cap_m=None):
        if cap_m is None:
            return path
        return FOREST_HEIGHTS.cap_virtual_path(path, FOREST_HEIGHTS.cap_level(cap_m))

    def shrub_path_for(self, path, cap_m=None):
        level = FOREST_HEIGHTS.cap_level(cap_m) if cap_m is not None else 0.0
        return FOREST_HEIGHTS.band_virtual_path(path, level, "s")


class CapZoneTests(unittest.TestCase):
    def test_caps_split_the_forest_and_the_inner_edge_gets_floor_shrubs(self):
        mask = np.zeros((300, 300), np.uint8)
        mask[10:290, 10:290] = 1
        cap = np.full(mask.shape, np.inf, np.float32)
        cap[:, :100] = 0.0       # before the inner edge: 1 m shrubs
        cap[:, 100:200] = 10.0   # 9 m level
        seen = []

        def fake(part, veg_cls, **kw):
            seen.append((kw.get('height_cap'), int(part[:, :100].sum())))
            return []

        with mock.patch.object(VEG, "_process_dds_mask", side_effect=fake):
            VEG._process_tree_mask(mask, m_per_px=1.0, height_cap_map=cap)
        self.assertEqual(sorted(seen, key=lambda r: (r[0] is None, r[0])),
                         [(1.0, 280 * 90), (9.0, 0), (None, 0)])


class ApproachSurfaceTests(unittest.TestCase):
    def _airport(self, mark=3, length_deg=0.03):
        rwy = APT.Runway(45.0, 0.0, 10.0, 0.0, mark, 0.0, 10.0 + length_deg, 0.0, mark)
        return APT.Airport("TEST", "pack", [], [], 0.0, 10.0, 1000.0, None, [rwy])

    def test_surface_rises_from_the_inner_edge(self):
        surfaces = APPROACH.runway_surfaces([self._airport()])
        self.assertEqual(len(surfaces), 2)
        west = next(s for s in surfaces if s.end == 1)
        self.assertAlmostEqual(west.ux, -1.0, places=6)
        self.assertEqual((west.inner_half, west.offset, west.slope), (150.0, 60.0, 0.02))
        # A texture west of the runway, 0.05 deg square, 1000 px.
        lat_n, lat_s, lon_w, lon_e = 0.025, -0.025, 9.95, 10.0
        cap = APPROACH.cap_map(surfaces, lat_n, lat_s, lon_w, lon_e, 1000, 1000)
        kx = 111320.0

        def at(dist_m, side_m=0.0):
            x = int(((10.0 - dist_m / kx) - lon_w) / (lon_e - lon_w) * 1000)
            y = int((lat_n - side_m / APPROACH.EARTH_M_PER_DEG_LAT) / (lat_n - lat_s) * 1000)
            return float(cap[y, x])

        self.assertEqual(at(30.0), 0.0)                        # between threshold and inner edge
        self.assertAlmostEqual(at(1060.0), 20.0, delta=0.3)    # 1000 m beyond the inner edge
        self.assertAlmostEqual(at(560.0), 10.0, delta=0.3)
        self.assertTrue(np.isinf(at(2500.0)))                  # the tallest tree clears it
        self.assertTrue(np.isinf(at(560.0, side_m=400.0)))     # beside the surface
        self.assertAlmostEqual(at(1060.0, side_m=290.0), 20.0, delta=0.3)   # 150 m + 15 % of 1000

    def test_short_visual_strips_get_steeper_narrower_surfaces(self):
        strip = APPROACH.runway_surfaces([self._airport(mark=0, length_deg=0.006)])[0]
        self.assertEqual((strip.inner_half, strip.offset, strip.slope), (30.0, 30.0, 0.05))
        self.assertAlmostEqual(APPROACH.reach_m(strip), 30.0 + 45.0 / 0.05)

    def test_runways_are_read_from_apt_dat(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "apt.dat")
            with open(path, "w") as f:
                f.write("I\n1200 Generated\n\n1 100 0 0 TEST Test\n"
                        "100 45.00 1 0 0.25 1 1 1 09 0.00000000 10.00000000 150.00 0.00 3 0 0 0 "
                        "27 0.00000000 10.03000000 0.00 60.00 1 0 0 0\n99\n")
            airports = list(APT.parse_airports(path))
        runways = airports[0][3]
        self.assertEqual(runways, [APT.Runway(45.0, 0.0, 10.0, 150.0, 3, 0.0, 10.03, 0.0, 1)])
        first = APPROACH.runway_surfaces([self._airport()._replace(runways=runways)])[0]
        self.assertAlmostEqual((first.lon - 10.0) * 111320.0, 150.0, delta=0.5)   # displaced


class AirportInsideTests(unittest.TestCase):
    def test_boundary_and_reach_circle(self):
        ring = [(10.0, 0.0), (10.01, 0.0), (10.01, 0.01), (10.0, 0.01), (10.0, 0.0)]
        with_boundary = APT.Airport("A", "p", [], [(ring, [])], 0.005, 10.005, 500.0, None, [])
        inside = VEG._airport_inside_mask([with_boundary], 0.02, 0.0, 10.0, 10.02, 200, 200)
        self.assertTrue(inside[150, 50])
        self.assertFalse(inside[50, 150])
        no_boundary = APT.Airport("B", "p", [], [], 0.015, 10.015, 300.0, None, [])
        inside = VEG._airport_inside_mask([no_boundary], 0.02, 0.0, 10.0, 10.02, 200, 200)
        self.assertTrue(inside[50, 150])
        self.assertFalse(inside[150, 50])
        self.assertIsNone(VEG._airport_inside_mask([no_boundary], 1.02, 1.0, 10.0, 10.02, 200, 200))


if __name__ == "__main__":
    unittest.main()
