"""Forest edge thickets: the mantle copies (O4_Forest_Heights) and the edge bands
cut from each forest polygon (O4_Vegetation_Overlay)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import O4_Forest_Heights as FOREST_HEIGHTS
import O4_Landcover_Inference as SEGFORMER
import O4_Vegetation_Overlay as VEG
from tree_heights_test import _FOR, _fake_xp, _mesh_tops

FH = FOREST_HEIGHTS


def _trees(text):
    out = []
    for line in text.splitlines():
        p = line.split()
        if p[:1] in (["TREE"], ["TREE2"]):
            lo_i, hi_i, _nom, lay_i = FH._tree_fields(p)
            out.append((p[-1], int(p[lay_i]), float(p[lo_i]), float(p[hi_i])))
    return out


class MantleCopyTests(unittest.TestCase):
    def test_young_trees_mixed_with_shrubs_under_the_top(self):
        text = FH.mantle_trees(_FOR, 9.0, "_o4m9")
        trees = _trees(text)
        layer1 = [t for t in trees if t[1] == 1]
        # Canopy trees topped at 9 m; the shrubs (lilac, the oak understory cut to 9 m) join them.
        self.assertEqual({t[3] for t in layer1 if t[0].startswith("redoak")}, {9.0})
        self.assertIn("lilac_03", {t[0] for t in layer1})
        self.assertIn("oak_understory", {t[0] for t in layer1})
        self.assertEqual({t[0] for t in trees if t[1] == 2}, {"lilac_03", "oak_understory"})
        # Young trees are MANTLE_YOUNG_SHARE of the canopy layer's frequency.
        freq = {}
        for line in text.splitlines():
            p = line.split()
            if p[:1] in (["TREE"], ["TREE2"]) and FH._tree_fields(p) and p[FH._tree_fields(p)[3]] == "1":
                freq[p[-1]] = freq.get(p[-1], 0.0) + float(p[6])
        young = freq["redoak_04"] + freq["redoak_06"]
        self.assertAlmostEqual(young / sum(freq.values()), FH.MANTLE_YOUNG_SHARE, places=2)
        self.assertEqual(FH.max_tree_height(text), 9.0)
        lilac = [ln for ln in text.splitlines() if ln.split()[-1:] == ["lilac_03"]]
        self.assertTrue(all(float(ln.split()[10]) == FH.MANTLE_LOD_M for ln in lilac))
        self.assertIn(f"SPACING	{FH._fmt(FH.MANTLE_SPACING_M)}	{FH._fmt(FH.MANTLE_SPACING_M)}", text)
        self.assertIn(f"RANDOM	{FH._fmt(FH.MANTLE_RANDOM_M)}", text)

    def test_meshes_are_scaled_with_their_trees(self):
        text = FH.mantle_trees(_FOR, 9.0, "_o4m9")
        tops = _mesh_tops(text)
        # The oak mesh is modelled at 14 m; young redoak_04 is 14/20 * 9 = 6.3 m at its minimum,
        # the 9-13 m oak understory shares the 14 m mesh, scaled by 9/13.
        self.assertTrue(any(abs(top - 6.3) < 0.01 for name, top in tops.items() if "3D_oak" in name), tops)
        self.assertTrue(any(abs(top - 14 * 9 / 13) < 0.01 for name, top in tops.items() if "3D_oak" in name), tops)
        refs = {ln.split()[1] for ln in text.splitlines() if ln.split()[:1] == ["MESH_3D"]}
        self.assertTrue(refs <= set(tops), (refs, set(tops)))

    def test_clutter_and_empty_groups_are_dropped(self):
        clutter = _FOR.replace("lilac_03", "ff_branch_01")
        text = FH.mantle_trees(clutter, 9.0)
        self.assertNotIn("ff_branch_01", text)
        self.assertNotIn("GROUP 2 40", text)        # its only tree went
        self.assertNotIn("MESH\t3D_lilac", text)    # and the mesh only it used
        self.assertIn("GROUP 2 10", text)

    def test_top_is_a_share_of_the_canopy(self):
        self.assertEqual(FH.mantle_top(19.5), 9.0)
        self.assertEqual(FH.mantle_top(13.0), 6.0)
        self.assertEqual(FH.mantle_top(6.0), 3.0)
        self.assertEqual(FH.mantle_top(19.5, cap_m=5.0), 4.5)
        self.assertEqual(FH.mantle_top(19.5, cap_m=2.0), 0.0)

    def test_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            xp = os.path.join(tmp, "xp")
            cs = os.path.join(tmp, "Custom Scenery")
            os.makedirs(cs)
            _fake_xp(xp)
            fh = FH.ForestHeights(cs, log=lambda *a: None, xp_dir=xp)
            self.assertEqual(fh.mantle_path_for("lib/g8/broad_tmp_wet.for"),
                             "o4xp/forests/m9/broad_tmp_wet.for")
            self.assertEqual(fh.mantle_path_for("o4xp/forests/h9/broad_tmp_wet.for"),
                             "o4xp/forests/m4.5/broad_tmp_wet.for")
            self.assertEqual(fh.mantle_path_for("o4xp/forests/c6/broad_tmp_wet.for"),
                             "o4xp/forests/m3/broad_tmp_wet.for")
            self.assertEqual(fh.mantle_path_for("lib/g8/broad_tmp_wet.for", 5.0),
                             "o4xp/forests/m4.5/broad_tmp_wet.for")
            self.assertIsNone(fh.mantle_path_for("lib/g8/broad_tmp_wet.for", 2.0))
            self.assertEqual(fh.mantle_path_for("o4xp/forests/ev-t3/broad_tmp_wet.for"),
                             "o4xp/forests/m9-ev-t3/broad_tmp_wet.for")
            self.assertIsNone(fh.mantle_path_for("o4xp/forests/s0/trees/x/shrubs.for"))
            self.assertIsNone(fh.mantle_path_for("simheaven/forests/x.for"))
            path = "o4xp/forests/m9/broad_tmp_wet.for"
            self.assertEqual(FH.parse_virtual(path), ("lib/g8/broad_tmp_wet.for", "m", 9.0))
            self.assertTrue(FH.is_mantle(path) and FH.needs_pack(path) and not FH.is_capped(path))
            tinted = fh.tint_path(path, 3)
            self.assertEqual(tinted, "o4xp/forests/m9-t3/broad_tmp_wet.for")
            self.assertEqual(FH.evergreen_virtual_path(tinted), "o4xp/forests/m9-ev-t3/broad_tmp_wet.for")
            # The pack writes it like the other copies.
            self.assertTrue(fh.ensure([path]))
            lib = open(os.path.join(fh.pack_dir, "library.txt")).read()
            self.assertIn(f"EXPORT_SEASON sum\t{path}\tsum_m9/broad_tmp.for", lib)
            written = open(os.path.join(fh.pack_dir, "sum_m9", "broad_tmp.for")).read()
            self.assertEqual(FH.max_tree_height(written), 9.0)

    def test_spacing_of_forests_and_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            xp = os.path.join(tmp, "xp")
            cs = os.path.join(tmp, "Custom Scenery")
            os.makedirs(cs)
            _fake_xp(xp)
            fh = FH.ForestHeights(cs, log=lambda *a: None, xp_dir=xp)
            self.assertEqual(fh.spacing("lib/g8/broad_tmp_wet.for"), 12.0)
            self.assertEqual(fh.spacing("o4xp/forests/ev-t3/broad_tmp_wet.for"), 12.0)
            self.assertEqual(fh.spacing("o4xp/forests/m9/broad_tmp_wet.for"), FH.MANTLE_SPACING_M)
            self.assertLess(fh.spacing("o4xp/forests/h9-t3/broad_tmp_wet.for"), 12.0)  # shorter, closer
            self.assertIsNone(fh.spacing("simheaven/forests/x.for"))


class EdgeDensityTests(unittest.TestCase):
    def test_thicket_thinned_next_to_sparse_forests(self):
        self.assertEqual(VEG._edge_density(None), 255)
        self.assertEqual(VEG._edge_density(6.0), 255)          # 3 m: the grid's own 4 m
        self.assertEqual(VEG._edge_density(8.0), 255)
        self.assertEqual(VEG._edge_density(10.0), round(255 * 16 / 25))
        self.assertEqual(VEG._edge_density(12.0), round(255 * 16 / 36))
        self.assertEqual(VEG._edge_density(20.0), round(255 * 16 / 36))   # 6 m at most
        self.assertEqual(VEG._edge_density(12.0, 0.5), round(255 * 16 / 36 * 0.5))
        self.assertEqual(VEG._edge_density(None, 0.0), 1)

    def test_bands_simplified_to_their_width(self):
        mask = np.zeros((400, 400), np.uint8)
        yy, xx = np.mgrid[0:400, 0:400]
        mask[(yy - 200) ** 2 + (xx - 200) ** 2 < 150 ** 2] = 1  # a round wood
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        def pts(simplify_px, mpp, width=6.0):
            rings, _core = VEG._edge_band_rings(
                cnts[0], [], mask > 0, np.full((400, 400), width, np.float32), mpp,
                400, 400, 1.0, 0.0, 0.0, 1.0, core_simplify_px=simplify_px)
            area = _band_area(rings, 1.0, 400, 400, (1.0, 0.0, 0.0, 1.0)).sum(dtype=np.int64)
            return sum(len(r) + sum(len(h) for h in getattr(r, "holes", ())) for r in rings), int(area)
        # Fine pixels: simplified like the forest, up to 0.8 x the band's
        # narrowest width (3 m = 10 px -> 8 px, so 5 px), and the band keeps its area.
        fine, fine_area = pts(5.0, 0.3)
        exact, exact_area = pts(0.6, 0.3)
        self.assertLess(fine, 0.6 * exact)
        self.assertLess(abs(fine_area - exact_area), 0.1 * exact_area)
        # Coarse pixels (a narrowest 1.5 m band is 0.6 px): FOREST_EDGE_SIMPLIFY_PX.
        self.assertEqual(pts(5.0, 2.4, 1.5), pts(0.6, 2.4, 1.5))


def _band_area(rings, m_per_px, img_w, img_h, bounds):
    """Band area in pixels, rasterised back from (lon, lat) rings."""
    lat_n, lat_s, lon_w, lon_e = bounds
    out = np.zeros((img_h, img_w), np.uint8)
    for ring in rings:
        def px(r):
            return np.array([[[(lo - lon_w) / (lon_e - lon_w) * img_w,
                               (lat_n - la) / (lat_n - lat_s) * img_h]] for lo, la in r]).round().astype(np.int32)
        one = np.zeros_like(out)
        cv2.fillPoly(one, [px(ring)], 1)
        for hole in getattr(ring, "holes", ()):
            cv2.fillPoly(one, [px(hole)], 0)
        out |= one
    return out


class EdgeBandTests(unittest.TestCase):
    W = H = 200
    MPP = 1.0
    BOUNDS = (1.0, 0.0, 0.0, 1.0)   # lat_n, lat_s, lon_w, lon_e

    def _bands(self, poly_mask, extent, holes=(), width=6.0, inset=0.0):
        cnts, _ = cv2.findContours(poly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        widths = np.full((self.H, self.W), width, np.float32)
        lat_n, lat_s, lon_w, lon_e = self.BOUNDS
        rings, core = VEG._edge_band_rings(cnts[0], list(holes), extent, widths, self.MPP,
                                           self.W, self.H, lat_n, lat_s, lon_w, lon_e,
                                           inset_m=inset, core_simplify_px=0.5)
        if inset:
            return rings, _band_area(rings, self.MPP, self.W, self.H, self.BOUNDS), core
        self.assertIsNone(core)
        return rings, _band_area(rings, self.MPP, self.W, self.H, self.BOUNDS)

    def test_inset_moves_the_tall_trees_back_from_open_edges_only(self):
        mask = np.zeros((self.H, self.W), np.uint8)
        mask[0:100, 50:150] = 1                                # touches the texture's top
        _rings, band, core = self._bands(mask, mask > 0, width=3.0, inset=5.0)
        core_px = _band_area(core, self.MPP, self.W, self.H, self.BOUNDS)
        self.assertTrue(core_px[50, 100])
        self.assertFalse(core_px[50, 52])                       # open west edge: inset
        self.assertFalse(core_px[97, 100])                      # open south edge: inset
        self.assertTrue(core_px[1, 100])                        # texture edge: no inset
        self.assertTrue(band[50, 54])                           # band at least the inset wide
        # A strip narrower than twice the inset keeps its trees.
        strip = np.zeros((self.H, self.W), np.uint8)
        strip[50:58, 20:180] = 1
        _r, _b, none = self._bands(strip, strip > 0, inset=5.0)
        self.assertIsNone(none)

    def test_open_edges_get_a_band_and_the_inside_stays_bare(self):
        mask = np.zeros((self.H, self.W), np.uint8)
        mask[50:150, 50:150] = 1
        rings, band = self._bands(mask, mask > 0)
        self.assertEqual(len(rings), 1)
        self.assertTrue(band[52, 100] and band[100, 52])     # along the edge
        self.assertFalse(band[100, 100])                      # the middle
        self.assertFalse(band[100, 70])                       # 20 m in
        self.assertLess(abs(band.sum() - (100 * 100 - 88 * 88)), 400)

    def test_texture_edge_and_neighbouring_forest_are_not_edges(self):
        mask = np.zeros((self.H, self.W), np.uint8)
        mask[0:100, 0:100] = 1                                 # touches the texture's top and left
        extent = np.zeros_like(mask, dtype=bool)
        extent[0:100, 0:200] = True                            # another zone carries on to the east
        _rings, band = self._bands(mask, extent)
        self.assertFalse(band[1, 50])                          # top: texture edge
        self.assertFalse(band[50, 1])                          # left: texture edge
        self.assertFalse(band[50, 98])                         # east: more forest
        self.assertTrue(band[97, 50])                          # south: open ground

    def test_clearings_get_a_band(self):
        mask = np.zeros((self.H, self.W), np.uint8)
        mask[20:180, 20:180] = 1
        hole = np.array([[[80, 80]], [[120, 80]], [[120, 120]], [[80, 120]]], np.int32)
        cv2.fillPoly(mask, [hole], 0)
        _rings, band = self._bands(mask, mask > 0, holes=[hole])
        self.assertTrue(band[78, 100])                         # just outside the clearing
        self.assertFalse(band[100, 100])                       # in it
        self.assertFalse(band[60, 60])

    def test_width_follows_the_map(self):
        mask = np.zeros((self.H, self.W), np.uint8)
        mask[50:150, 50:150] = 1
        _r, thin = self._bands(mask, mask > 0, width=3.0)
        _r, wide = self._bands(mask, mask > 0, width=9.0)
        self.assertGreater(wide.sum(), 2.5 * thin.sum())

    def test_width_map_varies_smoothly_around_the_mean(self):
        with mock.patch.dict(os.environ, {"O4_VEG_FOREST_EDGE_M": "6"}):
            w = VEG._edge_width_map((400, 400), 1.0, 7)
        self.assertAlmostEqual(float(w.mean()), 6.0, delta=0.6)
        self.assertGreaterEqual(float(w.min()), 6.0 * (1 - VEG.FOREST_EDGE_WIDTH_SPREAD) - 1e-3)
        self.assertLessEqual(float(w.max()), 6.0 * (1 + VEG.FOREST_EDGE_WIDTH_SPREAD) + 1e-3)
        self.assertLess(float(np.abs(np.diff(w, axis=1)).max()), 0.5)


class EdgePolygonTests(unittest.TestCase):
    def test_area_forests_get_a_thicket_polygon(self):
        with tempfile.TemporaryDirectory() as tmp:
            xp = os.path.join(tmp, "xp")
            cs = os.path.join(tmp, "Custom Scenery")
            os.makedirs(cs)
            _fake_xp(xp)
            fh = FH.ForestHeights(cs, log=lambda *a: None, xp_dir=xp)
            mask = np.zeros((400, 400), np.uint8)
            mask[100:300, 100:300] = 1
            mask[20:24, 20:24] = 1                             # a single tree
            mapper = mock.Mock()
            mapper.map_path.side_effect = lambda p, *_a: "lib/g8/broad_tmp_wet.for"
            kw = dict(img_w=400, img_h=400, lat_n=1.0, lat_s=0.0, lon_w=0.0, lon_e=1.0,
                      tile_lat=0, tile_lon=0, m_per_px=1.0, min_area_px=4, simplify_px=1.0,
                      region="northmiddle", rng=np.random.default_rng(0), density_override=1.0,
                      forest_path_mapper=mapper)
            env = {"O4_VEG_FOREST_EDGES": "1"}
            with mock.patch.dict(os.environ, env):
                plain = VEG._process_tree_mask(mask, **kw)
                edged = VEG._process_tree_mask(
                    mask, forest_edges=fh, edge_width=np.full((400, 400), 6.0, np.float32), **kw)
            extra = [p for p in edged if p not in plain]
            self.assertEqual(len(edged), len(plain) + 1)
            with mock.patch.dict(os.environ, dict(env, O4_VEG_FOREST_EDGE_INSET_M="5")):
                inset = VEG._process_tree_mask(
                    mask, forest_edges=fh, edge_width=np.full((400, 400), 6.0, np.float32), **kw)
            forest = [r for p, d, r in inset if p == "lib/g8/broad_tmp_wet.for" and d < VEG.POINT_MODE]
            lons = [lo for lo, _la in forest[0]]
            self.assertAlmostEqual(min(lons) * 400, 105, delta=1.5)   # 5 m in from x = 100
            self.assertEqual(extra[0][0], "o4xp/forests/m9/broad_tmp_wet.for")
            # X-Plane's 12 m forest grid: thinned to a 6 m thicket.
            self.assertEqual(extra[0][1], round(255 * (4 / 6) ** 2))
            self.assertFalse(any(FH.is_mantle(p) for p, d, _r in edged if d >= VEG.POINT_MODE))


class EdgeSettingTests(unittest.TestCase):
    def test_settings_reach_the_overlay(self):
        import O4_Cfg_Vars as CFGVARS
        import O4_Overlay_Pipeline as PIPELINE
        for name in ("veg_forest_edges", "veg_forest_edge_m", "veg_forest_edge_inset_m"):
            self.assertIn(name, CFGVARS.list_veg_vars)
            self.assertEqual(getattr(PIPELINE, name), CFGVARS.cfg_vars[name]["default"])
        self.assertEqual(CFGVARS.cfg_vars["veg_forest_edge_m"]["default"], VEG.FOREST_EDGE_WIDTH_M)
        self.assertEqual(CFGVARS.cfg_vars["veg_forest_edge_inset_m"]["default"], VEG.FOREST_EDGE_INSET_M)
        saved = (PIPELINE.veg_forest_edges, PIPELINE.veg_forest_edge_m, PIPELINE.veg_forest_edge_inset_m)
        try:
            PIPELINE.veg_forest_edges, PIPELINE.veg_forest_edge_m = False, 9.0
            PIPELINE.veg_forest_edge_inset_m = 4.0
            env = PIPELINE._veg_tree_env()
        finally:
            (PIPELINE.veg_forest_edges, PIPELINE.veg_forest_edge_m,
             PIPELINE.veg_forest_edge_inset_m) = saved
        with mock.patch.dict(os.environ, env):
            self.assertFalse(VEG._forest_edges_enabled())
            self.assertEqual(VEG._forest_edge_width_m(), 9.0)
            self.assertEqual(VEG._forest_edge_inset_m(), 4.0)
        self.assertNotEqual(VEG._forest_edge_sig(), ())


if __name__ == "__main__":
    unittest.main()
