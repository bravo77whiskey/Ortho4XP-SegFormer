"""Summer forests without X-Plane's autumn blend (2026-11-16, replaced the leaf colour copies)."""

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

import O4_Config_Utils as CFG
import O4_Forest_Heights as FH
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
G8 = "lib/g8/broad_tmp_wet.for"


class PathTests(unittest.TestCase):
    def test_summer_combines_with_every_kind_of_copy(self):
        sp = FH.summer_virtual_path
        self.assertEqual(sp(G8), "o4xp/forests/su/broad_tmp_wet.for")
        self.assertEqual(sp("o4xp/forests/h9/broad_tmp_wet.for"), "o4xp/forests/h9-su/broad_tmp_wet.for")
        self.assertEqual(sp("o4xp/forests/m7.5/broad_tmp_wet.for"), "o4xp/forests/m7.5-su/broad_tmp_wet.for")
        self.assertEqual(sp("lib/vegetation/trees/deciduous/shrubs_small.for"),
                         "o4xp/forests/su/trees/deciduous/shrubs_small.for")
        self.assertEqual(sp("o4xp/forests/h9-su/broad_tmp_wet.for"), "o4xp/forests/h9-su/broad_tmp_wet.for")
        pv = FH.parse_virtual
        self.assertEqual(pv("o4xp/forests/su/broad_tmp_wet.for"), (G8, FH.SUMMER_KIND, 0.0))
        self.assertEqual(pv("o4xp/forests/c4.5-su/broad_tmp_wet.for"), (G8, "c", 4.5))
        self.assertTrue(FH.is_summer("o4xp/forests/c4.5-su/broad_tmp_wet.for"))
        self.assertFalse(FH.is_summer("o4xp/forests/c4.5/broad_tmp_wet.for"))
        self.assertTrue(FH.is_capped("o4xp/forests/c6-su/broad_tmp_wet.for"))
        self.assertFalse(FH.is_capped("o4xp/forests/su/broad_tmp_wet.for"))

    def test_evergreen_copies_already_have_no_autumn_blend(self):
        self.assertEqual(FH.summer_virtual_path("o4xp/forests/ev/broad_tmp_wet.for"),
                         "o4xp/forests/ev/broad_tmp_wet.for")
        self.assertEqual(FH.evergreen_virtual_path("o4xp/forests/h9-su/broad_tmp_wet.for"),
                         "o4xp/forests/h9-ev/broad_tmp_wet.for")
        self.assertEqual(FH.evergreen_virtual_path("o4xp/forests/su/broad_tmp_wet.for"),
                         "o4xp/forests/ev/broad_tmp_wet.for")

    def test_leaf_colour_paths_are_gone(self):
        self.assertIsNone(FH.parse_virtual("o4xp/forests/t3/broad_tmp_wet.for"))
        self.assertFalse(hasattr(FH, "tint_of"))


class PackTests(unittest.TestCase):
    def test_summer_file_loses_its_autumn_blend_other_seasons_stay(self):
        with tempfile.TemporaryDirectory() as tmp:
            xp = os.path.join(tmp, "xp")
            cs = os.path.join(tmp, "Custom Scenery")
            os.makedirs(cs)
            for season, text in (("sum", _FOR), ("spr", _SPR)):
                os.makedirs(os.path.join(xp, season))
                with open(os.path.join(xp, season, "broad_tmp.for"), "w", newline="\n") as f:
                    f.write(text)
            os.makedirs(os.path.join(xp, "textures"))
            with open(os.path.join(xp, "library.txt"), "w") as f:
                f.write("A\n1200\nLIBRARY\n"
                        f"EXPORT_SEASON sum\t{G8}\tsum/broad_tmp.for\n"
                        f"EXPORT_SEASON spr\t{G8}\tspr/broad_tmp.for\n"
                        "EXPORT\tlib/g8/broad_hot_wet.for\tsum/broad_tmp.for\n")
            fh = FH.ForestHeights(cs, log=lambda *a: None, xp_dir=xp)
            self.assertEqual(fh.summer_path(G8), "o4xp/forests/su/broad_tmp_wet.for")
            # Forests X-Plane does not change with the seasons, and other packs' forests, stay.
            self.assertEqual(fh.summer_path("lib/g8/broad_hot_wet.for"), "lib/g8/broad_hot_wet.for")
            self.assertEqual(fh.summer_path("simheaven/forests/x.for"), "simheaven/forests/x.for")
            path = "o4xp/forests/su/broad_tmp_wet.for"
            self.assertTrue(fh.ensure([path]))
            pack = os.path.join(cs, FH.PACK_NAME)
            lib = open(os.path.join(pack, "library.txt")).read()
            self.assertIn(f"EXPORT_SEASON sum\t{path}\tsum_su/broad_tmp.for", lib)
            self.assertIn(f"EXPORT_SEASON spr\t{path}\tspr_su/broad_tmp.for", lib)
            summer = open(os.path.join(pack, "sum_su", "broad_tmp.for")).read()
            self.assertNotIn("SEASONAL", summer)
            self.assertIn("../textures/trees_3D1_ALB.png", summer)   # X-Plane's own leaves
            self.assertEqual(open(os.path.join(pack, "spr_su", "broad_tmp.for")).read(), _SPR)
            self.assertFalse([d for d in os.listdir(pack) if d.startswith("textures_")])


class OverlayTests(unittest.TestCase):
    def test_tree_mask_paths_get_the_summer_copy(self):
        class _Copies:
            def summer_path(self, path):
                return FH.summer_virtual_path(path)

        class _Mapper:
            def map_path(self, path, lon, lat, veg_type):
                return G8

        size = 400
        mask = np.zeros((size, size), np.uint8)
        mask[40:360, 20:380] = 1
        kw = dict(img_w=size, img_h=size, lat_n=52.004, lat_s=52.0, lon_w=13.0, lon_e=13.006,
                  tile_lat=52, tile_lon=13, m_per_px=1.0, min_area_px=4, simplify_px=1.0,
                  region="northmiddle", rng=np.random.default_rng(0), density_override=None,
                  forest_path_mapper=_Mapper())
        plain = VEG._process_tree_mask(mask, **kw)
        summer = VEG._process_tree_mask(mask, summer_leaves=_Copies(), **kw)
        self.assertEqual({p for p, _d, _r in plain}, {G8})
        self.assertEqual({p for p, _d, _r in summer}, {"o4xp/forests/su/broad_tmp_wet.for"})
        # Same polygons: the copy only changes the path (no colour zones cut forests up).
        self.assertEqual([(d, len(r)) for _p, d, r in plain], [(d, len(r)) for _p, d, r in summer])

    def test_on_by_default_and_the_old_setting_carries_over(self):
        self.assertTrue(VEG._summer_leaves_enabled())
        self.assertEqual(CFG.normalize_config_entry("veg_forest_tint", "False"),
                         ("veg_summer_leaves", "False"))


if __name__ == "__main__":
    unittest.main()
