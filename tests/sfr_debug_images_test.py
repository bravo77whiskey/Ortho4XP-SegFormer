import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_SFR_Debug_Images as DBG
import O4_SFR_Building_Overlay as BLD
import O4_SFR_Inference as SEGFORMER


class SfrDebugImagesTests(unittest.TestCase):
    def test_roof_color_matching_debug_flag_is_opt_in(self):
        disabled = DBG.parse_args(["example.dds"])
        enabled = DBG.parse_args(["example.dds", "--roof-color-matching"])

        self.assertFalse(disabled.roof_color_matching)
        self.assertTrue(enabled.roof_color_matching)

    def test_parse_texture_metadata_from_fixture_style_path(self):
        meta = DBG.parse_texture_metadata(
            r"H:\XP12 Addons\Applications\Ortho4XP\Tiles\zOrtho4XP_+36+117\textures\25680_54080_BI16.dds"
        )

        self.assertEqual(meta.tile_lat, 36)
        self.assertEqual(meta.tile_lon, 117)
        self.assertEqual(meta.til_y_top, 25680)
        self.assertEqual(meta.til_x_left, 54080)
        self.assertEqual(meta.provider, "BI")
        self.assertEqual(meta.zoomlevel, 16)

    def test_generate_debug_images_draws_building_footprints(self):
        with tempfile.TemporaryDirectory() as tmp:
            texture_dir = Path(tmp) / "zOrtho4XP_+36+117" / "textures"
            texture_dir.mkdir(parents=True)
            dds_path = texture_dir / "25680_54080_BI16.dds"
            Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8), "RGB").save(
                dds_path,
                format="PNG",
            )

            class_map = np.full((128, 128), -1, dtype=np.int8)
            class_map[20:60, 20:60] = SEGFORMER.CLASS_TREE
            class_map[70:100, 70:100] = SEGFORMER.CLASS_BUILDING

            result = DBG.generate_debug_images_for_dds(
                dds_path,
                Path(tmp) / "debug",
                class_map=class_map,
                max_preview_px=0,
            )

            outputs = result["outputs"]
            self.assertTrue(Path(outputs["source"]).exists())
            self.assertTrue(Path(outputs["segformer_masks"]).exists())
            self.assertTrue(Path(outputs["vegetation_zones"]).exists())
            self.assertTrue(Path(outputs["building_zones"]).exists())
            self.assertTrue(Path(outputs["proposed_footprints"]).exists())
            self.assertTrue(Path(outputs["composite"]).exists())
            self.assertGreater(result["placements"], 0)

            counts_path = Path(outputs["composite"]).with_name("25680_54080_BI16_counts.json")
            counts = json.loads(counts_path.read_text(encoding="utf-8"))
            self.assertIn("closed_forest", counts["vegetation_zones"])
            self.assertGreater(counts["building_zones"]["compact_residential"], 0)

            footprint_img = np.asarray(Image.open(outputs["proposed_footprints"]).convert("RGB"))
            colored_pixels = np.count_nonzero(np.any(footprint_img != 0, axis=2))
            self.assertGreater(colored_pixels, 10)

    def test_generate_debug_images_prefers_cached_production_footprints(self):
        with tempfile.TemporaryDirectory() as tmp:
            texture_dir = Path(tmp) / "zOrtho4XP_+36+117" / "textures"
            cache_dir = Path(tmp) / "cache"
            texture_dir.mkdir(parents=True)
            cache_dir.mkdir()
            dds_path = texture_dir / "25680_54080_BI16.dds"
            Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8), "RGB").save(
                dds_path,
                format="PNG",
            )
            class_map = np.full((128, 128), -1, dtype=np.int8)
            class_map[70:100, 70:100] = SEGFORMER.CLASS_BUILDING
            lat_n, lat_s, lon_w, lon_e = BLD.dds_bounds(25680, 54080, 16)
            lon_c = (lon_w + lon_e) / 2.0
            lat_c = (lat_n + lat_s) / 2.0
            d_lon = (lon_e - lon_w) / 12.0
            d_lat = (lat_n - lat_s) / 12.0
            with (cache_dir / "25680_54080_BI16_bld.pkl").open("wb") as handle:
                pickle.dump(
                    {
                        "placements": {
                            "objects": (),
                            "facades": (
                                (
                                    [
                                        (lon_c - d_lon, lat_c + d_lat),
                                        (lon_c + d_lon, lat_c + d_lat),
                                        (lon_c + d_lon, lat_c - d_lat),
                                        (lon_c - d_lon, lat_c - d_lat),
                                        (lon_c - d_lon, lat_c + d_lat),
                                    ],
                                    "lib/buildings/facades/generic/mid_modern_01.fac",
                                    7.0,
                                ),
                            ),
                        }
                    },
                    handle,
                )

            result = DBG.generate_debug_images_for_dds(
                dds_path,
                Path(tmp) / "debug",
                cache_dir=cache_dir,
                class_map=class_map,
                max_preview_px=0,
            )

            self.assertEqual(result["placements"], 1)
            self.assertEqual(result["placement_source"], "production building cache")
            footprint_img = np.asarray(Image.open(result["outputs"]["proposed_footprints"]).convert("RGB"))
            colored_pixels = np.count_nonzero(np.any(footprint_img != 0, axis=2))
            self.assertGreater(colored_pixels, 100)

    def test_zl16_zone_classifier_distinguishes_large_roofs_from_fine_grain(self):
        labels = np.zeros((160, 220), dtype=np.int32)
        labels[10:110, 10:110] = 1
        labels[20:140, 130:210] = 2
        stats = np.zeros((3, 5), dtype=np.int32)
        stats[1] = (10, 10, 100, 100, 10_000)
        stats[2] = (130, 20, 80, 120, 9_600)

        bld_raw = np.zeros_like(labels, dtype=np.uint8)
        for y in range(14, 104, 12):
            for x in range(14, 104, 12):
                bld_raw[y:y + 5, x:x + 5] = 1
        bld_raw[35:105, 145:195] = 1
        residential = np.zeros_like(labels, dtype=np.uint8)
        residential[labels == 1] = 1

        classes, counts = BLD._classify_building_zones_zl16(
            labels,
            stats,
            np.array([1, 2], dtype=np.int32),
            bld_raw,
            None,
            residential,
            2.0,
        )

        self.assertEqual(classes[1], BLD.BLD_CLASS_SMALL_RESIDENTIAL)
        self.assertEqual(classes[2], BLD.BLD_CLASS_EXTRA_LARGE)
        self.assertGreater(counts["fine_grain_zones"], 0)
        self.assertGreater(counts["large_roof_zones"], 0)

    def test_local_roof_evidence_refines_candidate_class_and_heading(self):
        bld_raw = np.zeros((120, 120), dtype=np.uint8)
        bld_zone = np.zeros_like(bld_raw)
        bld_raw[15:25, 15:25] = 1
        bld_raw[15:80, 75:90] = 1
        bld_zone[8:35, 8:35] = 1
        bld_zone[10:90, 68:98] = 1

        evidence = BLD._build_local_roof_evidence(bld_raw, bld_zone, 2.0)
        cand_x = np.array([20, 82], dtype=np.int32)
        cand_y = np.array([20, 45], dtype=np.int32)
        cand_cls = np.array(
            [BLD.BLD_CLASS_SMALL_APARTMENT, BLD.BLD_CLASS_SMALL_APARTMENT],
            dtype=np.uint8,
        )

        refined, changed = BLD._refine_candidate_classes_from_roofs(
            cand_x, cand_y, cand_cls, evidence, 2.0
        )

        self.assertEqual(changed, 1)
        self.assertLess(refined[0], BLD.BLD_CLASS_SMALL_APARTMENT)
        self.assertEqual(refined[1], BLD.BLD_CLASS_SMALL_APARTMENT)
        heading = BLD._roof_heading_for_candidate(evidence, 82, 45, 2.0)
        self.assertTrue(np.isfinite(heading))
        self.assertLess(min(abs(heading), abs(heading - 180.0)), 10.0)


if __name__ == "__main__":
    unittest.main()
