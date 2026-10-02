import os
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_Cfg_Vars as CFG
import O4_SFR_Leaf_Type as LEAF


def _write_tiled_tiff(path, bands, tile, compression, west, north, px):
    """Minimal classic tiled GeoTIFF writer (8-bit, pixel-interleaved)."""
    spp, h, w = bands.shape
    across, down = -(-w // tile), -(-h // tile)
    blobs = []
    for tr in range(down):
        for tc in range(across):
            t = np.zeros((tile, tile, spp), np.uint8)
            part = bands[:, tr * tile:(tr + 1) * tile, tc * tile:(tc + 1) * tile].transpose(1, 2, 0)
            t[:part.shape[0], :part.shape[1]] = part
            raw = t.tobytes()
            blobs.append(zlib.compress(raw) if compression == 8 else raw)
    ext = bytearray()
    entries = []

    def add(tag, typ, values):
        fmt = {3: "H", 4: "I", 12: "d"}[typ]
        data = struct.pack("<" + fmt * len(values), *values)
        entries.append((tag, typ, len(values), data))

    add(256, 4, [w]); add(257, 4, [h]); add(258, 3, [8] * spp); add(259, 3, [compression])
    add(262, 3, [1]); add(277, 3, [spp]); add(284, 3, [1]); add(322, 3, [tile]); add(323, 3, [tile])
    add(33550, 12, [px, px, 0.0]); add(33922, 12, [0.0, 0.0, 0.0, west, north, 0.0])
    n_entries = len(entries) + 2
    ifd_off = 8
    ext_off = ifd_off + 2 + n_entries * 12 + 4
    offsets_pos = ext_off
    data_pos = ext_off + 4096 + 8 * len(blobs)   # room for the external tag data
    tile_offsets, pos = [], data_pos
    for b in blobs:
        tile_offsets.append(pos)
        pos += len(b)
    add(324, 4, tile_offsets)
    add(325, 4, [len(b) for b in blobs])
    entries.sort(key=lambda e: e[0])
    ifd = bytearray(struct.pack("<H", len(entries)))
    for tag, typ, count, data in entries:
        if len(data) <= 4:
            ifd += struct.pack("<HHI", tag, typ, count) + data.ljust(4, b"\0")
        else:
            ifd += struct.pack("<HHII", tag, typ, count, offsets_pos + len(ext))
            ext += data
    ifd += struct.pack("<I", 0)
    with open(path, "wb") as f:
        f.write(b"II*\x00" + struct.pack("<I", ifd_off))
        f.write(ifd)
        # the external tag data replaces the reserved offsets area
        f.write(ext)
        pad = data_pos - (ifd_off + len(ifd) + len(ext))
        assert pad >= 0
        f.write(b"\0" * pad)
        for b in blobs:
            f.write(b)


def _grid(name, codes, west=0.0, north=1.0):
    codes = np.asarray(codes, np.uint8)
    return LEAF.LeafGrid(name, codes, west, north, 1.0 / codes.shape[1], 1.0 / codes.shape[0])


class GlcNamingTests(unittest.TestCase):
    def test_block_labels_are_top_left_corners(self):
        self.assertEqual(LEAF.glc_block_for(62.5, 2.0)[:2], ("E0N65", "E0-E5"))
        self.assertEqual(LEAF.glc_block_for(44.4, -72.3)[:2], ("W75N45", "W75-W80"))
        self.assertEqual(LEAF.glc_block_for(44.4, -77.0)[:2], ("W80N45", "W75-W80"))
        self.assertEqual(LEAF.glc_block_for(-37.3, -72.7)[:2], ("W75S35", "W75-W80"))
        self.assertEqual(LEAF.glc_block_for(-38.4, 176.6)[:2], ("E175S35", "E170-E175"))
        self.assertEqual(LEAF.glc_block_for(0.4, 32.9)[:2], ("E30N5", "E30-E35"))
        self.assertEqual(LEAF.glc_block_for(-0.5, -3.0)[:2], ("W5N0", "W5-W10"))


class GeoTiffReaderTests(unittest.TestCase):
    def _roundtrip(self, compression):
        rng = np.random.default_rng(1)
        bands = rng.integers(0, 6, size=(3, 70, 90), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.tif")
            _write_tiled_tiff(path, bands, 32, compression, 10.0, 50.0, 0.01)
            src = LEAF._FileBytes(path)
            try:
                tif = LEAF.GeoTiffTiles(src)
                self.assertEqual((tif.width, tif.height, tif.spp), (90, 70, 3))
                self.assertAlmostEqual(tif.west, 10.0)
                self.assertAlmostEqual(tif.north, 50.0)
                win = tif.read_window(5, 20, 50, 60, sample=2)
                np.testing.assert_array_equal(win, bands[2, 5:55, 20:80])
                edge = tif.read_window(60, 80, 20, 20, sample=1)
                np.testing.assert_array_equal(edge[:10, :10], bands[1, 60:70, 80:90])
                self.assertFalse(edge[10:].any() or edge[:, 10:].any())
            finally:
                src.close()

    def test_deflate_tiles(self):
        self._roundtrip(8)

    def test_uncompressed_tiles(self):
        self._roundtrip(1)

    def test_lzw_decode_matches_opencv_encoder(self):
        img = np.random.default_rng(2).integers(0, 255, size=(8, 96), dtype=np.uint8)
        ok, buf = cv2.imencode(".tiff", img, [cv2.IMWRITE_TIFF_COMPRESSION, 5])
        self.assertTrue(ok)
        data = buf.tobytes()
        ifd = struct.unpack("<I", data[4:8])[0]
        n = struct.unpack("<H", data[ifd:ifd + 2])[0]
        tags = {}
        for i in range(n):
            tag, typ, count, val = struct.unpack("<HHII", data[ifd + 2 + 12 * i: ifd + 14 + 12 * i])
            if typ == 3 and count == 1:
                val &= 0xFFFF
            tags[tag] = (count, val)
        self.assertEqual(tags[259][1], 5)
        self.assertEqual(tags[273][0], 1, "test needs a single strip")
        strip = data[tags[273][1]:tags[273][1] + tags[279][1]]
        decoded = LEAF._lzw_decode(strip, 96, 8)
        if tags.get(317, (1, 1))[1] == 2:
            # OpenCV writes horizontal differencing; GLC tiles use none.
            decoded = (np.cumsum(decoded.astype(np.int64), axis=1) % 256).astype(np.uint8)
        np.testing.assert_array_equal(decoded, img)


class LeafGridTests(unittest.TestCase):
    def test_filled_takes_nearest_typed_pixel_within_limit(self):
        codes = np.zeros((1, 200), np.uint8)
        codes[0, 0] = LEAF.CONI
        codes[0, 199] = LEAF.BROAD
        grid = LEAF.LeafGrid("t", codes, 0.0, 0.001, 0.0001, 0.0001)   # ~11 m pixels
        filled = grid.filled(500.0).codes[0]
        self.assertEqual(filled[10], LEAF.CONI)
        self.assertEqual(filled[190], LEAF.BROAD)
        self.assertEqual(filled[100], LEAF.NONE)   # ~1.1 km from either: beyond 500 m

    def test_window_and_sample(self):
        g = _grid("t", [[1, 2], [3, 0]])
        self.assertEqual(int(g.sample(0.75, 0.25)), 1)
        self.assertEqual(int(g.sample(0.25, 0.25)), 3)
        self.assertEqual(int(g.sample(5.0, 5.0)), LEAF.NONE)
        w = g.window(0.0, 0.5, 0.5, 1.0)
        self.assertEqual(w.codes.tolist(), [[0]])


class TileLeafTypesTests(unittest.TestCase):
    def _resolver(self, grids, forest_records=()):
        sources = tuple((name, (lambda g: lambda *a, **k: g)(g)) for name, g in grids.items())
        return LEAF.TileLeafTypes(0.0, 0.0, "unused", forest_records=forest_records,
                                  log=lambda *a: None, grid_sources=sources)

    def test_priority_follows_chain(self):
        hrl = _grid("HRL-DLT-2018", np.full((50, 50), LEAF.CONI))
        glc = _grid("GLC_FCS30D", np.full((50, 50), LEAF.BROAD))
        r = self._resolver({"HRL-DLT-2018": hrl, "GLC_FCS30D": glc})
        zone = r.zone_for_texture(0.6, 0.4, 0.4, 0.6, 64, 64)
        self.assertTrue((zone == LEAF.CONI).all())

    def test_untyped_pixels_fall_through_to_lower_sources(self):
        hrl = np.zeros((100, 100), np.uint8)
        hrl[:, :50] = LEAF.CONI
        r = self._resolver({
            "HRL-DLT-2018": _grid("HRL-DLT-2018", hrl),
            "GLC_FCS30D": _grid("GLC_FCS30D", np.full((100, 100), LEAF.BROAD)),
        })
        zone = r.zone_for_texture(0.52, 0.48, 0.3, 0.7, 50, 400)
        self.assertEqual(int(zone[25, 10]), LEAF.CONI)
        self.assertEqual(int(zone[25, 390]), LEAF.BROAD)
        self.assertIn(LEAF.MIXED, set(np.unique(zone[25, 150:250]).tolist()))

    def test_explicit_osm_overrides_global_maps(self):
        rec = {"path": "simheaven/forests/coni.for",
               "pts": [(0.45, 0.45), (0.55, 0.45), (0.55, 0.55), (0.45, 0.55), (0.45, 0.45)],
               "_bounds": (0.45, 0.55, 0.45, 0.55)}
        mixed_rec = dict(rec, path="simheaven/forests/mixed.for")
        r = self._resolver({"GLC_FCS30D": _grid("GLC_FCS30D", np.full((50, 50), LEAF.BROAD))},
                           forest_records=[rec, mixed_rec])
        self.assertEqual(len(r.osm), 1)  # mixed is the untagged default: ignored
        zone = r.zone_for_texture(0.505, 0.495, 0.495, 0.505, 32, 32)
        self.assertTrue((zone == LEAF.CONI).all())

    def test_no_coverage_is_unavailable(self):
        r = self._resolver({})
        self.assertFalse(r.available)
        self.assertIsNone(r.zone_for_texture(0.6, 0.4, 0.4, 0.6, 8, 8))

    def test_signature_tracks_map_content(self):
        a = self._resolver({"GLC_FCS30D": _grid("GLC_FCS30D", np.full((10, 10), LEAF.BROAD))})
        b = self._resolver({"GLC_FCS30D": _grid("GLC_FCS30D", np.full((10, 10), LEAF.CONI))})
        self.assertNotEqual(a.signature, b.signature)

    def test_failing_source_is_skipped(self):
        def boom(*a, **k):
            raise IOError("offline")
        glc = _grid("GLC_FCS30D", np.full((10, 10), LEAF.MIXED))
        r = LEAF.TileLeafTypes(0, 0, "unused", log=lambda *a: None,
                               grid_sources=(("HRL-DLT-2018", boom),
                                             ("GLC_FCS30D", lambda *a, **k: glc)))
        self.assertEqual(r.chain, ["GLC_FCS30D"])


class XPForestTests(unittest.TestCase):
    def test_band_parsing(self):
        self.assertEqual(LEAF.band_name(LEAF.g8_band_of_path("lib/g8/coni_tmp_wet.for")), ("tmp", "wet"))
        self.assertEqual(LEAF.band_name(LEAF.g8_band_of_path("lib/g8/broad_sp_vhot_rain.for")), ("vhot", "rain"))
        self.assertEqual(LEAF.band_name(LEAF.g8_band_of_path("lib/g8/wetl_shrb_cld_sdry.for")), ("cld", "sdry"))
        self.assertEqual(LEAF.g8_band_of_path("lib/g12/beaches.bch"), 0)
        self.assertEqual(LEAF.g8_band_of_path("lib/g8/grass_rainforest.for"), 0)
        for bid in range(1, 25):
            self.assertEqual(LEAF.band_id(*LEAF.band_name(bid)), bid)

    def test_rasterize_streamed_dsf_text(self):
        text = b"\n".join([
            b"PROPERTY sim/west 8",
            b"POLYGON_DEF lib/g12/beaches.bch",
            b"POLYGON_DEF lib/g8/coni_tmp_wet.for",
            b"POLYGON_DEF lib/g8/broad_wrm_sdry.for",
            b"BEGIN_POLYGON 1 255 2",
            b"BEGIN_WINDING",
            b"POLYGON_POINT 8.0 49.0", b"POLYGON_POINT 8.5 49.0",
            b"POLYGON_POINT 8.5 48.5", b"POLYGON_POINT 8.0 48.5",
            b"END_WINDING", b"END_POLYGON",
            b"BEGIN_POLYGON 0 0 2",
            b"BEGIN_WINDING", b"POLYGON_POINT 8.5 48.5", b"POLYGON_POINT 9.0 48.5",
            b"POLYGON_POINT 9.0 48.0", b"END_WINDING", b"END_POLYGON",
            b"PATCH_VERTEX 8.1 48.1 0 0 0",
        ])
        img = LEAF.rasterize_g8_bands(text.splitlines(), 8.0, 49.0, 0.1, 0.1, 10, 10)
        self.assertEqual(int(img[2, 2]), LEAF.band_id("tmp", "wet"))
        self.assertEqual(int(img[8, 8]), 0)  # the beach polygon is not a forest band

    def test_koppen_fallback_covers_every_class(self):
        import O4_Forest_Assets as FOREST_ASSETS

        for class_id in range(1, 31):
            temp, moist = LEAF.KOPPEN_BANDS[class_id]
            path = FOREST_ASSETS.g8_forest_path("mixed", temp, moist)
            self.assertIn(path, FOREST_ASSETS.G8_FOREST_PATHS)

    def test_mapper_uses_nearest_global_band_and_orchards(self):
        import O4_Forest_Assets as FOREST_ASSETS

        codes = np.zeros((20, 20), np.uint8)
        codes[:, :10] = LEAF.band_id("cld", "wet")
        codes[:, 15:] = LEAF.band_id("wrm", "sdry")
        mapper = LEAF.XPForestMapper.__new__(LEAF.XPForestMapper)
        mapper.fallback_band = LEAF.band_id("tmp", "wet")
        mapper.grid = LEAF.LeafGrid("XP-global", codes, 0.0, 1.0, 0.05, 0.05).filled(float("inf"))
        self.assertEqual(mapper.map_path("simheaven/forests/coni.for", 0.1, 0.5), "lib/g8/coni_cld_wet.for")
        self.assertEqual(mapper.map_path("simheaven/forests/broad.for", 0.9, 0.5), "lib/g8/broad_wrm_sdry.for")
        self.assertEqual(mapper.map_path("simheaven/forests/broad.for", 0.9, 0.5, "orchard_or_plantation"),
                         "lib/g8/olive_wrm_sdry.for")
        self.assertEqual(mapper.map_path("simheaven/forests/mixed.for", 0.1, 0.5, "orchard_or_plantation"),
                         "lib/g8/mixed_cld_wet.for")      # no orchard set for cold bands
        self.assertEqual(mapper.map_path("lib/vegetation/x.for", 0.5, 0.5), "lib/vegetation/x.for")
        for path in ("lib/g8/coni_cld_wet.for", "lib/g8/broad_wrm_sdry.for", "lib/g8/olive_wrm_sdry.for"):
            self.assertTrue(FOREST_ASSETS.is_approved_generated_forest_path(path))

    def test_mapper_without_global_scenery_uses_koppen(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapper = LEAF.XPForestMapper(48, 8, tmp, os.path.join(tmp, "Custom Scenery"), None,
                                         log=lambda *a: None)
        self.assertIsNone(mapper.grid)
        self.assertTrue(mapper.map_path("simheaven/forests/coni.for", 8.5, 48.5).startswith("lib/g8/coni_"))


class ConfigTests(unittest.TestCase):
    def test_leaf_type_maps_setting_defaults_on(self):
        self.assertIs(CFG.cfg_tile_vars["sfr_veg_use_leaf_type_maps"]["default"], True)
        self.assertIn("sfr_veg_use_leaf_type_maps", CFG.list_sfr_veg_vars)

    def test_paths_match_simheaven_assets(self):
        import O4_Forest_Assets as FOREST_ASSETS
        for _, path in LEAF.CODE_PATHS:
            self.assertEqual(FOREST_ASSETS.normalize_simheaven_tree_path(path), path)


if __name__ == "__main__":
    unittest.main()
