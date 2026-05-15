import sys
import tempfile
import unittest
import bz2
from unittest import mock
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_SFR_Building_Overlay as BLD


def _paths_for_classes(pools, classes):
    return {
        asset["path"]
        for cls in classes
        for asset in pools.get(cls, ())
    }


class SfdBuildingAssetTests(unittest.TestCase):
    def test_osm_tile_peer_path_handles_all_road_sources(self):
        base = r"C:\O4XP\OSM_data\+30+110\+36+117\+36+117_big_roads.osm.bz2"

        self.assertEqual(
            BLD._osm_tile_peer_path(base, "_all_roads.osm.bz2"),
            r"C:\O4XP\OSM_data\+30+110\+36+117\+36+117_all_roads.osm.bz2",
        )
        self.assertEqual(
            BLD._osm_tile_peer_path(
                r"C:\O4XP\OSM_data\+30+110\+36+117\+36+117_all_roads.osm.bz2",
                "_small_roads.osm.bz2",
            ),
            r"C:\O4XP\OSM_data\+30+110\+36+117\+36+117_small_roads.osm.bz2",
        )

    def test_load_osm_roads_parses_generic_highway_extract(self):
        xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<osm>
  <node id="1" lat="36.0" lon="117.0" />
  <node id="2" lat="36.1" lon="117.1" />
  <way id="10">
    <nd ref="1" />
    <nd ref="2" />
    <tag k="highway" v="residential" />
  </way>
</osm>"""
        with tempfile.TemporaryDirectory() as tmpdir:
            osm_path = Path(tmpdir) / "+36+117_all_roads.osm.bz2"
            with bz2.open(osm_path, "wb") as handle:
                handle.write(xml)

            roads = BLD._load_osm_roads(str(osm_path))

        self.assertEqual(len(roads), 1)
        self.assertEqual(roads[0]["type"], "residential")
        self.assertEqual(roads[0]["pts"], [(36.0, 117.0), (36.1, 117.1)])

    def test_medium_is_an_explicit_placement_class(self):
        self.assertIn(BLD.BLD_CLASS_MEDIUM, BLD.BLD_PLACEMENT_CLASSES)
        self.assertEqual(BLD.BLD_CLASS_LABELS[BLD.BLD_CLASS_MEDIUM], "medium footprint")
        self.assertEqual(BLD.BLD_CLASS_STANDARD_RESIDENTIAL, BLD.BLD_CLASS_MEDIUM)

    def test_building_spacing_is_edge_gap_plus_class_span(self):
        class_spans = {
            BLD.BLD_CLASS_TINY_RESIDENTIAL: 6.0,
            BLD.BLD_CLASS_SMALL_RESIDENTIAL: 10.0,
            BLD.BLD_CLASS_COMPACT_RESIDENTIAL: 8.0,
            BLD.BLD_CLASS_MEDIUM: 20.0,
            BLD.BLD_CLASS_SMALL_APARTMENT: 30.0,
            BLD.BLD_CLASS_APARTMENT_BLOCK: 40.0,
            BLD.BLD_CLASS_LARGE: 50.0,
            BLD.BLD_CLASS_EXTRA_LARGE: 80.0,
        }

        self.assertEqual(
            {
                cls: BLD._spacing_for_zone_class_m(20.0, cls, class_spans)
                for cls in BLD.BLD_PLACEMENT_CLASSES
            },
            {
                BLD.BLD_CLASS_TINY_RESIDENTIAL: 26.0,
                BLD.BLD_CLASS_SMALL_RESIDENTIAL: 30.0,
                BLD.BLD_CLASS_COMPACT_RESIDENTIAL: 28.0,
                BLD.BLD_CLASS_MEDIUM: 40.0,
                BLD.BLD_CLASS_SMALL_APARTMENT: 50.0,
                BLD.BLD_CLASS_APARTMENT_BLOCK: 60.0,
                BLD.BLD_CLASS_LARGE: 70.0,
                BLD.BLD_CLASS_EXTRA_LARGE: 100.0,
            },
        )

    def test_orientation_order_aligns_long_footprint_side_first(self):
        self.assertEqual(
            BLD._orientation_angles_for_bounds((-5.0, 5.0, -20.0, 20.0), 90.0),
            (90.0, 0.0),
        )
        self.assertEqual(
            BLD._orientation_angles_for_bounds((-20.0, 20.0, -5.0, 5.0), 90.0),
            (0.0, 90.0),
        )

    def test_yolo_obb_detection_conversion_extracts_heading_and_class(self):
        detection = BLD._yolo_obb_detection_from_points(
            np.array([
                [10.0, 10.0],
                [30.0, 10.0],
                [30.0, 20.0],
                [10.0, 20.0],
            ]),
            confidence=0.8,
            cls=0,
            img_w=64,
            img_h=64,
            m_per_px=1.0,
        )

        self.assertIsNotNone(detection)
        self.assertAlmostEqual(detection["center"][0], 20.0)
        self.assertAlmostEqual(detection["center"][1], 15.0)
        self.assertAlmostEqual(detection["heading"], 90.0)
        self.assertEqual(detection["placement_class"], BLD.BLD_CLASS_COMPACT_RESIDENTIAL)

    def test_yolo_obb_detection_clips_bounds(self):
        detection = BLD._yolo_obb_detection_from_points(
            np.array([
                [-5.0, -5.0],
                [20.0, -5.0],
                [20.0, 10.0],
                [-5.0, 10.0],
            ]),
            confidence=0.9,
            cls=0,
            img_w=16,
            img_h=16,
            m_per_px=1.0,
        )

        self.assertIsNotNone(detection)
        points = np.asarray(detection["points"])
        self.assertGreaterEqual(points.min(), 0.0)
        self.assertLessEqual(points[:, 0].max(), 15.0)
        self.assertLessEqual(points[:, 1].max(), 15.0)

    def test_yolo_guidance_prefers_nearby_detection_heading_and_class(self):
        guidance = BLD._build_yolo_guidance(
            [
                {
                    "center": [8.0, 8.0],
                    "heading": 45.0,
                    "confidence": 0.7,
                    "area_m2": 220.0,
                    "placement_class": BLD.BLD_CLASS_COMPACT_RESIDENTIAL,
                    "points": [
                        [4.0, 6.0],
                        [12.0, 6.0],
                        [12.0, 10.0],
                        [4.0, 10.0],
                    ],
                }
            ],
            img_h=32,
            img_w=32,
            m_per_px=1.0,
        )
        cand_cls = np.array([BLD.BLD_CLASS_MEDIUM], dtype=np.uint8)

        refined, changed = BLD._refine_candidate_classes_from_yolo(
            np.array([10], dtype=np.int32),
            np.array([8], dtype=np.int32),
            cand_cls,
            guidance,
        )

        self.assertEqual(changed, 1)
        self.assertEqual(int(refined[0]), BLD.BLD_CLASS_COMPACT_RESIDENTIAL)
        self.assertAlmostEqual(BLD._yolo_heading_for_candidate(guidance, 10, 8), 45.0)

    def test_yolo_template_translates_nearest_obb_shape(self):
        guidance = BLD._build_yolo_guidance(
            [
                {
                    "center": [8.0, 8.0],
                    "heading": 45.0,
                    "confidence": 0.7,
                    "area_m2": 220.0,
                    "placement_class": BLD.BLD_CLASS_COMPACT_RESIDENTIAL,
                    "points": [
                        [4.0, 6.0],
                        [12.0, 6.0],
                        [12.0, 10.0],
                        [4.0, 10.0],
                    ],
                }
            ],
            img_h=32,
            img_w=32,
            m_per_px=1.0,
        )

        template = BLD._yolo_template_for_candidate(guidance, 18, 18)

        self.assertIsNotNone(template)
        points, cls, heading = template
        np.testing.assert_array_equal(
            points,
            np.array([[14, 16], [22, 16], [22, 20], [14, 20]], dtype=np.int32),
        )
        self.assertEqual(cls, BLD.BLD_CLASS_COMPACT_RESIDENTIAL)
        self.assertAlmostEqual(heading, 45.0)

    def test_yolo_consensus_template_uses_fuzzy_majority_heading_and_shape(self):
        def tpl(center, heading, long_len, short_len, confidence=0.8):
            return {
                "center": np.asarray(center, dtype=np.float32),
                "points": BLD._points_from_yolo_heading(center, long_len, short_len, heading),
                "class": BLD.BLD_CLASS_COMPACT_RESIDENTIAL,
                "heading": heading,
                "confidence": confidence,
            }

        consensus = BLD._consensus_yolo_template(
            [
                tpl((10, 10), 88.0, 20.0, 10.0),
                tpl((30, 10), 91.0, 21.0, 9.5),
                tpl((50, 10), 94.0, 19.0, 10.5),
                tpl((70, 10), 25.0, 48.0, 12.0),
                tpl((90, 10), 27.0, 50.0, 12.0),
            ],
            heading_tol_deg=8.0,
            shape_rel_tol=0.15,
            shape_abs_tol_px=3.0,
        )

        self.assertIsNotNone(consensus)
        self.assertLessEqual(BLD._angle_delta_180(consensus["heading"], 91.0), 3.0)
        metrics = BLD._yolo_template_metrics(consensus)
        self.assertIsNotNone(metrics)
        self.assertLessEqual(abs(metrics["long_len"] - 20.0), 1.0)
        self.assertLessEqual(abs(metrics["short_len"] - 10.0), 1.0)
        self.assertEqual(consensus["consensus_heading_votes"], 3)
        self.assertEqual(consensus["consensus_shape_votes"], 3)

    def test_yolo_consensus_heading_wraps_at_180_degrees(self):
        templates = []
        for center, heading in [((10, 10), 178.0), ((30, 10), 1.0), ((50, 10), 3.0)]:
            templates.append({
                "center": np.asarray(center, dtype=np.float32),
                "points": BLD._points_from_yolo_heading(center, 18.0, 8.0, heading),
                "class": BLD.BLD_CLASS_SMALL_RESIDENTIAL,
                "heading": heading,
                "confidence": 0.8,
            })

        consensus = BLD._consensus_yolo_template(templates, heading_tol_deg=6.0)

        self.assertIsNotNone(consensus)
        self.assertLessEqual(
            min(
                BLD._angle_delta_180(consensus["heading"], 0.0),
                BLD._angle_delta_180(consensus["heading"], 180.0),
            ),
            2.0,
        )

    def test_neighbor_yolo_templates_only_supply_zones_without_obb(self):
        labels = np.zeros((20, 30), dtype=np.int32)
        labels[3:17, 2:12] = 1
        labels[3:17, 15:27] = 2
        stats = np.zeros((3, 5), dtype=np.int32)
        stats[1, BLD.cv2.CC_STAT_LEFT] = 2
        stats[1, BLD.cv2.CC_STAT_TOP] = 3
        stats[1, BLD.cv2.CC_STAT_WIDTH] = 10
        stats[1, BLD.cv2.CC_STAT_HEIGHT] = 14
        stats[1, BLD.cv2.CC_STAT_AREA] = 140
        stats[2, BLD.cv2.CC_STAT_LEFT] = 15
        stats[2, BLD.cv2.CC_STAT_TOP] = 3
        stats[2, BLD.cv2.CC_STAT_WIDTH] = 12
        stats[2, BLD.cv2.CC_STAT_HEIGHT] = 14
        stats[2, BLD.cv2.CC_STAT_AREA] = 168
        yolo_template = {
            "center": np.asarray([7.0, 10.0], dtype=np.float32),
            "points": BLD._points_from_yolo_heading((7.0, 10.0), 8.0, 4.0, 90.0),
            "class": BLD.BLD_CLASS_COMPACT_RESIDENTIAL,
            "heading": 90.0,
            "confidence": 0.9,
        }

        neighbors = BLD._neighbor_yolo_templates_by_zone(
            labels,
            stats,
            np.array([1, 2], dtype=np.int32),
            {1: [yolo_template]},
            radius_px=4,
        )

        self.assertNotIn(1, neighbors)
        self.assertIn(2, neighbors)
        self.assertIs(neighbors[2][0], yolo_template)

    def test_neighbor_yolo_templates_choose_closest_obb_zone(self):
        labels = np.zeros((20, 50), dtype=np.int32)
        labels[3:17, 2:10] = 1
        labels[3:17, 20:28] = 2
        labels[3:17, 34:42] = 3
        stats = np.zeros((4, 5), dtype=np.int32)
        for label, left in ((1, 2), (2, 20), (3, 34)):
            stats[label, BLD.cv2.CC_STAT_LEFT] = left
            stats[label, BLD.cv2.CC_STAT_TOP] = 3
            stats[label, BLD.cv2.CC_STAT_WIDTH] = 8
            stats[label, BLD.cv2.CC_STAT_HEIGHT] = 14
            stats[label, BLD.cv2.CC_STAT_AREA] = 112
        west_template = {
            "center": np.asarray([6.0, 10.0], dtype=np.float32),
            "points": BLD._points_from_yolo_heading((6.0, 10.0), 8.0, 4.0, 0.0),
            "class": BLD.BLD_CLASS_SMALL_RESIDENTIAL,
            "heading": 0.0,
            "confidence": 0.8,
        }
        east_template = {
            "center": np.asarray([38.0, 10.0], dtype=np.float32),
            "points": BLD._points_from_yolo_heading((38.0, 10.0), 8.0, 4.0, 90.0),
            "class": BLD.BLD_CLASS_EXTRA_LARGE,
            "heading": 90.0,
            "confidence": 0.8,
        }

        neighbors = BLD._neighbor_yolo_templates_by_zone(
            labels,
            stats,
            np.array([1, 2, 3], dtype=np.int32),
            {1: [west_template], 3: [east_template]},
            radius_px=100,
        )

        self.assertNotIn(1, neighbors)
        self.assertNotIn(3, neighbors)
        self.assertEqual(neighbors[2], [east_template])

    def test_nearest_obb_zone_heading_ignores_class(self):
        stats = np.zeros((4, 5), dtype=np.int32)
        centroids = np.zeros((4, 2), dtype=np.float32)
        # Source OBB zone, intentionally a different placement class.
        stats[1, BLD.cv2.CC_STAT_LEFT] = 0
        stats[1, BLD.cv2.CC_STAT_TOP] = 0
        stats[1, BLD.cv2.CC_STAT_WIDTH] = 10
        stats[1, BLD.cv2.CC_STAT_HEIGHT] = 10
        centroids[1] = (5.0, 5.0)
        # Non-OBB target zone next to source.
        stats[2, BLD.cv2.CC_STAT_LEFT] = 15
        stats[2, BLD.cv2.CC_STAT_TOP] = 0
        stats[2, BLD.cv2.CC_STAT_WIDTH] = 8
        stats[2, BLD.cv2.CC_STAT_HEIGHT] = 8
        centroids[2] = (19.0, 4.0)
        # Farther non-OBB target zone also receives the same nearest source.
        stats[3, BLD.cv2.CC_STAT_LEFT] = 80
        stats[3, BLD.cv2.CC_STAT_TOP] = 0
        stats[3, BLD.cv2.CC_STAT_WIDTH] = 8
        stats[3, BLD.cv2.CC_STAT_HEIGHT] = 8
        centroids[3] = (84.0, 4.0)
        template = {
            "center": np.asarray([5.0, 5.0], dtype=np.float32),
            "points": BLD._points_from_yolo_heading((5.0, 5.0), 12.0, 4.0, 37.0),
            "class": BLD.BLD_CLASS_EXTRA_LARGE,
            "heading": 37.0,
            "confidence": 0.9,
        }

        headings, counts, source_labels, distances = BLD._nearest_obb_zone_headings(
            stats,
            centroids,
            np.array([1, 2, 3], dtype=np.int32),
            {1: [template]},
        )

        self.assertTrue(np.isnan(headings[1]))
        self.assertAlmostEqual(float(headings[2]), 37.0)
        self.assertEqual(int(counts[2]), 1)
        self.assertEqual(int(source_labels[2]), 1)
        self.assertAlmostEqual(float(distances[2]), 6.0)
        self.assertAlmostEqual(float(headings[3]), 37.0)

    def test_retarget_yolo_template_heading_preserves_shape(self):
        template = {
            "center": np.asarray([20.0, 20.0], dtype=np.float32),
            "points": BLD._points_from_yolo_heading((20.0, 20.0), 12.0, 6.0, 10.0),
            "class": BLD.BLD_CLASS_MEDIUM,
            "heading": 10.0,
            "confidence": 0.7,
        }

        retargeted = BLD._retarget_yolo_template_heading(template, 85.0)

        self.assertIsNotNone(retargeted)
        self.assertAlmostEqual(retargeted["heading"], 85.0)
        old_metrics = BLD._yolo_template_metrics(template)
        new_metrics = BLD._yolo_template_metrics(retargeted)
        self.assertAlmostEqual(old_metrics["long_len"], new_metrics["long_len"], places=4)
        self.assertAlmostEqual(old_metrics["short_len"], new_metrics["short_len"], places=4)
        self.assertEqual(retargeted["class"], BLD.BLD_CLASS_MEDIUM)

    def test_largest_fitting_yolo_template_shrinks_only_too_large_dimension(self):
        template = {
            "center": np.asarray([15.0, 30.0], dtype=np.float32),
            "points": BLD._points_from_yolo_heading((15.0, 30.0), 30.0, 12.0, 90.0),
            "class": BLD.BLD_CLASS_MEDIUM,
            "heading": 90.0,
            "confidence": 0.8,
        }
        static_occ_mask = np.zeros((60, 24), dtype=np.uint8)
        spacing_mask = np.zeros_like(static_occ_mask)
        scratch = np.zeros_like(static_occ_mask)

        poly, scales = BLD._largest_fitting_yolo_template_poly(
            template,
            15,
            30,
            img_w=24,
            img_h=60,
            static_occ_mask=static_occ_mask,
            building_spacing_mask=spacing_mask,
            scratch_mask=scratch,
        )

        self.assertIsNotNone(poly)
        long_scale, short_scale = scales
        self.assertLess(long_scale, 1.0)
        self.assertAlmostEqual(short_scale, 1.0)
        metrics = BLD._yolo_template_metrics({"points": poly})
        self.assertLess(metrics["long_len"], 30.0)
        self.assertGreaterEqual(metrics["short_len"], 11.0)

    def test_smart_gap_fill_is_retired(self):
        self.assertFalse(BLD.BLD_SMART_GAP_FILL_ENABLED)

    def test_building_zone_cell_mask_tracks_only_occupied_grid_cells(self):
        bld_zone = np.zeros((8, 8), dtype=np.uint8)
        bld_zone[1, 1] = 1
        bld_zone[6, 7] = 1

        cell_mask = BLD._building_zone_cell_mask(bld_zone, 8, 8, 4)

        self.assertEqual(int(cell_mask.sum()), 2)
        self.assertTrue(cell_mask[0, 0])
        self.assertTrue(cell_mask[3, 3])
        self.assertFalse(cell_mask[0, 3])

    def test_image_heading_grid_can_be_limited_to_building_cells(self):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        cell_mask = np.zeros((4, 4), dtype=bool)
        cell_mask[2, 1] = True
        calls = []

        def fake_edge_hist(_patch, bin_deg=5.0):
            calls.append(_patch.shape)
            hist = np.zeros(int(180 / bin_deg), dtype=np.float32)
            hist[0] = 1.0
            return hist

        with mock.patch.object(BLD, "_cell_edge_hist", side_effect=fake_edge_hist):
            hgrid = BLD._image_heading_grid(img, 8, 8, 4, cell_mask=cell_mask)

        self.assertEqual(len(calls), 1)
        self.assertFalse(np.isnan(hgrid[2, 1]))
        self.assertTrue(np.isnan(hgrid[0, 0]))

    def test_oversized_polygon_rasterization_fills_covered_tile(self):
        poly = [
            (-1.0, -1.0),
            (-1.0, 2.0),
            (2.0, 2.0),
            (2.0, -1.0),
            (-1.0, -1.0),
        ]

        mask = BLD._rasterize_polygons(
            [poly],
            lat_n=1.0,
            lat_s=0.0,
            lon_w=0.0,
            lon_e=1.0,
            img_h=32,
            img_w=32,
        )

        self.assertTrue(np.all(mask == 1))

    def test_oversized_polygon_rasterization_clips_partial_water(self):
        poly = [
            (-1.0, -1.0),
            (-1.0, 0.5),
            (2.0, 0.5),
            (2.0, -1.0),
            (-1.0, -1.0),
        ]

        mask = BLD._rasterize_polygons(
            [poly],
            lat_n=1.0,
            lat_s=0.0,
            lon_w=0.0,
            lon_e=1.0,
            img_h=32,
            img_w=32,
        )

        self.assertTrue(np.all(mask[:, :16] == 1))
        self.assertTrue(np.all(mask[:, 18:] == 0))

    def test_mesh_water_reader_does_not_treat_land_type_as_water(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mesh_path = Path(tmpdir) / "Data+00+000.mesh"
            mesh_path.write_text(
                "\n".join(
                    [
                        "MeshVersionFormatted 1.3",
                        "Dimension 3",
                        "Vertices",
                        "unused",
                        "4",
                        "0.0 0.0 0.0",
                        "1.0 0.0 0.0",
                        "1.0 1.0 0.0",
                        "0.0 1.0 0.0",
                        "Normals",
                        "unused",
                        "unused",
                        "0.0 0.0",
                        "0.0 0.0",
                        "0.0 0.0",
                        "0.0 0.0",
                        "Triangles",
                        "unused",
                        "2",
                        "1 2 3 0",
                        "1 3 4 2",
                    ]
                ),
                encoding="utf-8",
            )

            tris = BLD._read_mesh_water_triangles(str(mesh_path))

        self.assertEqual(tris.shape, (1, 3, 2))
        self.assertTrue(np.allclose(tris[0], [(0.0, 0.0), (1.0, 1.0), (1.0, 0.0)]))

    def test_component_side_heading_counts_touched_sides_not_road_length(self):
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[30:70, 30:70] = 1
        stats = np.zeros((2, 5), dtype=np.int32)
        stats[1, BLD.cv2.CC_STAT_LEFT] = 30
        stats[1, BLD.cv2.CC_STAT_TOP] = 30
        stats[1, BLD.cv2.CC_STAT_WIDTH] = 40
        stats[1, BLD.cv2.CC_STAT_HEIGHT] = 40
        stats[1, BLD.cv2.CC_STAT_AREA] = 1600

        def ll(px, py):
            return (1.0 - py / 100.0, px / 100.0)

        roads = [
            {"pts": [ll(31 + i * 4, 28), ll(33 + i * 4, 28)]}
            for i in range(10)
        ] + [
            {"pts": [ll(28, 20), ll(28, 80)]},
            {"pts": [ll(72, 20), ll(72, 80)]},
        ]

        headings, side_counts = BLD._component_side_touch_headings(
            labels,
            stats,
            np.array([1], dtype=np.int32),
            roads,
            lat_n=1.0,
            lat_s=0.0,
            lon_w=0.0,
            lon_e=1.0,
            img_h=100,
            img_w=100,
            band_px=4,
        )

        self.assertEqual(int(side_counts[1]), 3)
        self.assertLessEqual(min(abs(float(headings[1])), abs(float(headings[1]) - 360.0)), 5.0)

    def test_dead_end_road_contact_can_supply_heading(self):
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[30:70, 30:70] = 1
        stats = np.zeros((2, 5), dtype=np.int32)
        stats[1, BLD.cv2.CC_STAT_LEFT] = 30
        stats[1, BLD.cv2.CC_STAT_TOP] = 30
        stats[1, BLD.cv2.CC_STAT_WIDTH] = 40
        stats[1, BLD.cv2.CC_STAT_HEIGHT] = 40
        stats[1, BLD.cv2.CC_STAT_AREA] = 1600

        def ll(px, py):
            return (1.0 - py / 100.0, px / 100.0)

        headings, contact_counts = BLD._component_side_touch_headings(
            labels,
            stats,
            np.array([1], dtype=np.int32),
            [{"pts": [ll(50, 10), ll(50, 32)]}],
            lat_n=1.0,
            lat_s=0.0,
            lon_w=0.0,
            lon_e=1.0,
            img_h=100,
            img_w=100,
            band_px=4,
        )

        self.assertGreaterEqual(int(contact_counts[1]), 1)
        self.assertLessEqual(min(abs(float(headings[1])), abs(float(headings[1]) - 360.0)), 5.0)

    def test_touching_local_road_beats_nearby_large_road_for_zone_heading(self):
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[30:70, 30:70] = 1
        stats = np.zeros((2, 5), dtype=np.int32)
        stats[1, BLD.cv2.CC_STAT_LEFT] = 30
        stats[1, BLD.cv2.CC_STAT_TOP] = 30
        stats[1, BLD.cv2.CC_STAT_WIDTH] = 40
        stats[1, BLD.cv2.CC_STAT_HEIGHT] = 40
        stats[1, BLD.cv2.CC_STAT_AREA] = 1600

        def ll(px, py):
            return (1.0 - py / 100.0, px / 100.0)

        roads = [
            {"pts": [ll(28, 20), ll(28, 80)]},
            {"pts": [ll(0, 22), ll(99, 22)]},
        ]

        headings, contact_counts = BLD._component_side_touch_headings(
            labels,
            stats,
            np.array([1], dtype=np.int32),
            roads,
            lat_n=1.0,
            lat_s=0.0,
            lon_w=0.0,
            lon_e=1.0,
            img_h=100,
            img_w=100,
            band_px=12,
        )

        self.assertEqual(int(contact_counts[1]), 1)
        self.assertLessEqual(min(abs(float(headings[1])), abs(float(headings[1]) - 360.0)), 5.0)

    def test_road_mask_contact_ignores_non_touching_road_in_heading_band(self):
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[30:70, 30:70] = 1
        stats = np.zeros((2, 5), dtype=np.int32)
        stats[1, BLD.cv2.CC_STAT_LEFT] = 30
        stats[1, BLD.cv2.CC_STAT_TOP] = 30
        stats[1, BLD.cv2.CC_STAT_WIDTH] = 40
        stats[1, BLD.cv2.CC_STAT_HEIGHT] = 40
        stats[1, BLD.cv2.CC_STAT_AREA] = 1600
        road_mask = np.zeros((100, 100), dtype=np.uint8)
        road_mask[20:24, :] = 1
        road_mask[:, 24:31] = 1

        def ll(px, py):
            return (1.0 - py / 100.0, px / 100.0)

        roads = [
            {"pts": [ll(24, 20), ll(24, 80)]},
            {"pts": [ll(0, 22), ll(99, 22)]},
        ]

        headings, contact_counts = BLD._component_side_touch_headings(
            labels,
            stats,
            np.array([1], dtype=np.int32),
            roads,
            lat_n=1.0,
            lat_s=0.0,
            lon_w=0.0,
            lon_e=1.0,
            img_h=100,
            img_w=100,
            band_px=12,
            contact_px=5,
            road_mask=road_mask,
        )

        self.assertEqual(int(contact_counts[1]), 1)
        self.assertLessEqual(min(abs(float(headings[1])), abs(float(headings[1]) - 360.0)), 5.0)

    def test_straight_contact_patches_are_preferred_over_noisy_fragments(self):
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[30:70, 30:70] = 1
        stats = np.zeros((2, 5), dtype=np.int32)
        stats[1, BLD.cv2.CC_STAT_LEFT] = 30
        stats[1, BLD.cv2.CC_STAT_TOP] = 30
        stats[1, BLD.cv2.CC_STAT_WIDTH] = 40
        stats[1, BLD.cv2.CC_STAT_HEIGHT] = 40
        stats[1, BLD.cv2.CC_STAT_AREA] = 1600

        def ll(px, py):
            return (1.0 - py / 100.0, px / 100.0)

        roads = [{"pts": [ll(28, 20), ll(28, 80)]}]
        roads.extend(
            {"pts": [ll(35 + i * 4, 28), ll(36 + i * 4, 31)]}
            for i in range(7)
        )

        headings, contact_counts = BLD._component_side_touch_headings(
            labels,
            stats,
            np.array([1], dtype=np.int32),
            roads,
            lat_n=1.0,
            lat_s=0.0,
            lon_w=0.0,
            lon_e=1.0,
            img_h=100,
            img_w=100,
            band_px=4,
        )

        self.assertGreaterEqual(int(contact_counts[1]), 1)
        self.assertLessEqual(min(abs(float(headings[1])), abs(float(headings[1]) - 360.0)), 5.0)

    def test_simheaven_buildings_supply_zone_heading_after_split(self):
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[20:80, 20:80] = 1
        objects = {
            "lat": np.array([0.70, 0.60, 0.50], dtype=np.float32),
            "lon": np.array([0.30, 0.40, 0.50], dtype=np.float32),
            "heading": np.array([30.0, 32.0, 120.0], dtype=np.float32),
            "w_m": np.array([10.0, 10.0, 20.0], dtype=np.float32),
            "h_m": np.array([20.0, 20.0, 20.0], dtype=np.float32),
        }

        headings, counts = BLD._simheaven_building_zone_headings(
            objects,
            labels,
            np.array([1], dtype=np.int32),
            lat_n=1.0,
            lat_s=0.0,
            lon_w=0.0,
            lon_e=1.0,
            img_h=100,
            img_w=100,
        )

        self.assertEqual(int(counts[1]), 2)
        self.assertLessEqual(abs(float(headings[1]) - 32.5), 5.0)

    def test_asset_regions_use_non_rectangular_boundaries(self):
        self.assertEqual(BLD._asset_region(45.0, -75.0), "north_america_ne")
        self.assertEqual(BLD._asset_region(35.0, -120.0), "north_america_west")
        self.assertEqual(BLD._asset_region(42.0, 12.0), "mediterranean")
        self.assertEqual(BLD._asset_region(10.0, 25.0), "africa")
        self.assertEqual(BLD._asset_region(35.0, -40.0), "generic")

    def test_residential_pools_have_expanded_regional_suburban_variety(self):
        cases = (
            (65.0, 20.0, "SFD_Global/Scandinavia/Residential/Suburban_", 8),
            (10.0, 25.0, "SFD_Global/Africa/Residential/Suburban_", 8),
            (42.0, 12.0, "SFD_Global/Med/Residential/Suburban_", 8),
            (45.0, -75.0, "SFD_Global/New_England/Residential/Suburban_", 8),
            (35.0, -120.0, "SFD_Global/US_West_Coast/Suburban_", 8),
            (-20.0, -60.0, "SFD_Global/South_America/Suburban_", 10),
            (35.5, 139.5, "SFD_Global/Asia/Suburban_", 10),
        )

        for lat, lon, prefix, expected_count in cases:
            with self.subTest(prefix=prefix):
                pools = BLD._build_sfd_asset_pools(lat, lon)
                residential_paths = _paths_for_classes(
                    pools,
                    (
                        BLD.BLD_CLASS_TINY_RESIDENTIAL,
                        BLD.BLD_CLASS_SMALL_RESIDENTIAL,
                        BLD.BLD_CLASS_COMPACT_RESIDENTIAL,
                        BLD.BLD_CLASS_MEDIUM,
                        BLD.BLD_CLASS_SMALL_APARTMENT,
                    ),
                )
                regional_suburban = [
                    path for path in residential_paths
                    if path.startswith(prefix)
                ]

                self.assertGreaterEqual(len(regional_suburban), expected_count)

    def test_regional_apartment_like_assets_expand_where_sfd_exports_them(self):
        med_pools = BLD._build_sfd_asset_pools(42.0, 12.0)
        med_apartments = _paths_for_classes(
            med_pools,
            (
                BLD.BLD_CLASS_MEDIUM,
                BLD.BLD_CLASS_SMALL_APARTMENT,
                BLD.BLD_CLASS_APARTMENT_BLOCK,
            ),
        )
        south_america_pools = BLD._build_sfd_asset_pools(-20.0, -60.0)
        south_america_medium = _paths_for_classes(
            south_america_pools,
            (
                BLD.BLD_CLASS_COMPACT_RESIDENTIAL,
                BLD.BLD_CLASS_MEDIUM,
                BLD.BLD_CLASS_SMALL_APARTMENT,
                BLD.BLD_CLASS_APARTMENT_BLOCK,
            ),
        )

        self.assertGreaterEqual(
            len([
                path for path in med_apartments
                if path.startswith("SFD_Global/Med/Residential/Apartment_North_")
            ]),
            8,
        )
        self.assertGreaterEqual(
            len([
                path for path in south_america_medium
                if path.startswith("SFD_Global/South_America/Med_")
            ]),
            8,
        )

    def test_urban_residential_assets_are_included_by_footprint(self):
        pools = BLD._build_sfd_asset_pools(42.0, 12.0)
        residential_and_apartment_paths = _paths_for_classes(
            pools,
            (
                BLD.BLD_CLASS_COMPACT_RESIDENTIAL,
                BLD.BLD_CLASS_MEDIUM,
                BLD.BLD_CLASS_SMALL_APARTMENT,
                BLD.BLD_CLASS_APARTMENT_BLOCK,
            ),
        )
        urban_paths = [
            path for path in residential_and_apartment_paths
            if path.startswith("SFD_Global/Med/Residential/Urban_Mid_")
        ]

        self.assertGreaterEqual(len(urban_paths), 16)
        self.assertIn("SFD_Global/Med/Residential/Urban_Mid_30m.obj", urban_paths)
        self.assertEqual(
            BLD._class_for_footprint(
                BLD._bounds_for_object_path(
                    "SFD_Global/Med/Residential/Urban_Mid_30m.obj"
                )
            ),
            BLD.BLD_CLASS_SMALL_APARTMENT,
        )

    def test_compact_residential_excludes_midrise_row_blocks(self):
        pools = BLD._build_sfd_asset_pools(42.0, 12.0)
        compact_paths = _paths_for_classes(
            pools,
            (BLD.BLD_CLASS_COMPACT_RESIDENTIAL,),
        )
        medium_paths = _paths_for_classes(
            pools,
            (BLD.BLD_CLASS_MEDIUM,),
        )

        self.assertNotIn("SFD_Global/Med/Residential/Urban_Mid_7m.obj", compact_paths)
        self.assertIn("SFD_Global/Med/Residential/Urban_Mid_7m.obj", medium_paths)

    def test_compact_simheaven_residential_stays_one_or_two_floor(self):
        pools = BLD._build_simheaven_asset_pools([], 45.0, 7.0)
        compact_paths = _paths_for_classes(
            pools,
            (BLD.BLD_CLASS_COMPACT_RESIDENTIAL,),
        )
        small_paths = _paths_for_classes(
            pools,
            (BLD.BLD_CLASS_SMALL_RESIDENTIAL,),
        )
        medium_paths = _paths_for_classes(
            pools,
            (BLD.BLD_CLASS_MEDIUM,),
        )

        self.assertIn("simheaven/houses/house_09x12x2.obj", small_paths)
        self.assertIn("simheaven/houses/house_12x15x2.obj", compact_paths)
        self.assertNotIn("simheaven/residential/residential_10x10x3.obj", compact_paths)
        self.assertIn("simheaven/residential/residential_10x10x3.obj", medium_paths)

    def test_fit_selection_does_not_retry_after_selected_asset_fails(self):
        pool = [
            {
                "kind": "object",
                "path": "too-large.obj",
                "bounds_m": (-8.0, 8.0, -8.0, 8.0),
                "fit_bounds_m": (-8.0, 8.0, -8.0, 8.0),
                "mark_bounds_m": (-8.0, 8.0, -8.0, 8.0),
            },
            {
                "kind": "object",
                "path": "small-enough.obj",
                "bounds_m": (-2.0, 2.0, -2.0, 2.0),
                "fit_bounds_m": (-2.0, 2.0, -2.0, 2.0),
                "mark_bounds_m": (-2.0, 2.0, -2.0, 2.0),
            },
        ]
        occ_mask = np.zeros((40, 40), dtype=np.uint8)
        occ_mask[20, 27] = 1
        scratch = np.zeros_like(occ_mask)
        counts = {}

        building_spacing_mask = np.zeros_like(occ_mask)

        asset, final_h, final_poly, spacing_poly, skipped = BLD._find_fitting_asset(
            pool,
            np.random.default_rng(1),
            20,
            20,
            0.0,
            1.0,
            occ_mask,
            building_spacing_mask,
            scratch,
            counts,
        )

        self.assertIsNone(asset)
        self.assertIsNone(final_h)
        self.assertEqual(skipped, 0)
        self.assertEqual(counts["fit_checks"], 2)
        self.assertIsNone(final_poly)
        self.assertIsNone(spacing_poly)

    def test_largest_fit_mode_selects_largest_asset_that_fits(self):
        pool = [
            {
                "kind": "object",
                "path": "small.obj",
                "bounds_m": (-2.0, 2.0, -2.0, 2.0),
                "fit_bounds_m": (-2.0, 2.0, -2.0, 2.0),
                "mark_bounds_m": (-2.0, 2.0, -2.0, 2.0),
                "footprint_area_m2": 16.0,
                "footprint_max_side_m": 4.0,
            },
            {
                "kind": "object",
                "path": "large.obj",
                "bounds_m": (-6.0, 6.0, -4.0, 4.0),
                "fit_bounds_m": (-6.0, 6.0, -4.0, 4.0),
                "mark_bounds_m": (-6.0, 6.0, -4.0, 4.0),
                "footprint_area_m2": 96.0,
                "footprint_max_side_m": 12.0,
            },
        ]
        occ_mask = np.zeros((40, 40), dtype=np.uint8)
        spacing_mask = np.zeros_like(occ_mask)
        counts = {}

        asset, final_h, final_poly, spacing_poly, skipped = BLD._find_fitting_asset(
            pool,
            np.random.default_rng(1),
            20,
            20,
            0.0,
            1.0,
            occ_mask,
            spacing_mask,
            np.zeros_like(occ_mask),
            counts,
            prefer_largest_fit=True,
        )

        self.assertEqual(asset["path"], "large.obj")
        self.assertIsNotNone(final_h)
        self.assertIsNotNone(final_poly)
        self.assertIsNotNone(spacing_poly)
        self.assertEqual(skipped, 0)
        self.assertEqual(counts["largest_fit_asset_selected"], 1)

    def test_static_integral_fit_path_matches_standard_fit_path(self):
        pool = [
            {
                "kind": "object",
                "path": "small.obj",
                "bounds_m": (-2.0, 2.0, -2.0, 2.0),
                "fit_bounds_m": (-6.0, 6.0, -6.0, 6.0),
                "mark_bounds_m": (-10.0, 10.0, -10.0, 10.0),
            },
        ]
        static_occ_mask = np.zeros((50, 50), dtype=np.uint8)
        static_occ_mask[20, 25] = 1
        building_spacing_mask = np.zeros_like(static_occ_mask)

        standard_counts = {}
        standard = BLD._find_fitting_asset(
            pool,
            np.random.default_rng(1),
            20,
            20,
            0.0,
            1.0,
            static_occ_mask,
            building_spacing_mask,
            np.zeros_like(static_occ_mask),
            standard_counts,
        )

        integral_counts = {}
        integral = BLD._find_fitting_asset(
            pool,
            np.random.default_rng(1),
            20,
            20,
            0.0,
            1.0,
            static_occ_mask,
            building_spacing_mask,
            np.zeros_like(static_occ_mask),
            integral_counts,
            static_occ_integral=BLD.cv2.integral(static_occ_mask, sdepth=BLD.cv2.CV_32S),
        )

        self.assertEqual(standard[0]["path"], integral[0]["path"])
        self.assertEqual(standard[1], integral[1])
        np.testing.assert_array_equal(standard[2], integral[2])
        np.testing.assert_array_equal(standard[3], integral[3])
        self.assertEqual(standard[4], integral[4])
        self.assertEqual(standard_counts, integral_counts)

    def test_fit_selection_checks_only_selected_asset_orientations(self):
        pool = [
            {
                "kind": "object",
                "path": "too-large-a.obj",
                "bounds_m": (-8.0, 8.0, -8.0, 8.0),
                "fit_bounds_m": (-8.0, 8.0, -8.0, 8.0),
                "mark_bounds_m": (-8.0, 8.0, -8.0, 8.0),
            },
            {
                "kind": "object",
                "path": "too-large-b.obj",
                "bounds_m": (-8.0, 8.0, -8.0, 8.0),
                "fit_bounds_m": (-8.0, 8.0, -8.0, 8.0),
                "mark_bounds_m": (-8.0, 8.0, -8.0, 8.0),
            },
            {
                "kind": "object",
                "path": "small-enough.obj",
                "bounds_m": (-2.0, 2.0, -2.0, 2.0),
                "fit_bounds_m": (-2.0, 2.0, -2.0, 2.0),
                "mark_bounds_m": (-2.0, 2.0, -2.0, 2.0),
            },
        ]
        occ_mask = np.zeros((40, 40), dtype=np.uint8)
        occ_mask[20, 27] = 1
        building_spacing_mask = np.zeros_like(occ_mask)
        counts = {}

        with mock.patch.object(
            BLD, "_footprint_poly_with_bbox_basis", wraps=BLD._footprint_poly_with_bbox_basis
        ) as footprint_poly:
            asset, final_h, final_poly, spacing_poly, skipped = BLD._find_fitting_asset(
                pool,
                np.random.default_rng(11),
                20,
                20,
                0.0,
                1.0,
                occ_mask,
                building_spacing_mask,
                np.zeros_like(occ_mask),
                counts,
            )

        self.assertIsNone(asset)
        self.assertIsNone(final_h)
        self.assertEqual(skipped, 0)
        self.assertEqual(counts["fit_checks"], 2)
        self.assertIsNone(final_poly)
        self.assertIsNone(spacing_poly)
        self.assertEqual(footprint_poly.call_count, 4)

    def test_static_blockers_use_raw_footprint_not_spacing_pad(self):
        pool = [
            {
                "kind": "object",
                "path": "small.obj",
                "bounds_m": (-2.0, 2.0, -2.0, 2.0),
                "fit_bounds_m": (-6.0, 6.0, -6.0, 6.0),
                "mark_bounds_m": (-10.0, 10.0, -10.0, 10.0),
            },
        ]
        static_occ_mask = np.zeros((50, 50), dtype=np.uint8)
        static_occ_mask[20, 25] = 1
        building_spacing_mask = np.zeros_like(static_occ_mask)
        scratch = np.zeros_like(static_occ_mask)
        counts = {}

        asset, final_h, final_poly, spacing_poly, skipped = BLD._find_fitting_asset(
            pool,
            np.random.default_rng(1),
            20,
            20,
            0.0,
            1.0,
            static_occ_mask,
            building_spacing_mask,
            scratch,
            counts,
        )

        self.assertEqual(asset["path"], "small.obj")
        self.assertEqual(final_h, 0.0)
        self.assertEqual(skipped, 0)
        self.assertIsNotNone(final_poly)
        self.assertIsNotNone(spacing_poly)

    def test_generated_building_spacing_still_uses_padded_footprint(self):
        pool = [
            {
                "kind": "object",
                "path": "small.obj",
                "bounds_m": (-2.0, 2.0, -2.0, 2.0),
                "fit_bounds_m": (-6.0, 6.0, -6.0, 6.0),
                "mark_bounds_m": (-10.0, 10.0, -10.0, 10.0),
            },
        ]
        static_occ_mask = np.zeros((50, 50), dtype=np.uint8)
        building_spacing_mask = np.zeros_like(static_occ_mask)
        building_spacing_mask[20, 25] = 1
        scratch = np.zeros_like(static_occ_mask)
        counts = {}

        asset, final_h, final_poly, spacing_poly, skipped = BLD._find_fitting_asset(
            pool,
            np.random.default_rng(1),
            20,
            20,
            0.0,
            1.0,
            static_occ_mask,
            building_spacing_mask,
            scratch,
            counts,
        )

        self.assertIsNone(asset)
        self.assertIsNone(final_h)
        self.assertIsNone(final_poly)
        self.assertIsNone(spacing_poly)
        self.assertEqual(skipped, 0)

    def test_fit_selection_tries_rotated_asset_before_next_asset(self):
        pool = [
            {
                "kind": "object",
                "path": "rotates-to-fit.obj",
                "bounds_m": (-2.0, 2.0, -8.0, 8.0),
                "fit_bounds_m": (-2.0, 2.0, -8.0, 8.0),
                "mark_bounds_m": (-2.0, 2.0, -8.0, 8.0),
            },
            {
                "kind": "object",
                "path": "fallback.obj",
                "bounds_m": (-1.0, 1.0, -1.0, 1.0),
                "fit_bounds_m": (-1.0, 1.0, -1.0, 1.0),
                "mark_bounds_m": (-1.0, 1.0, -1.0, 1.0),
            },
        ]
        static_occ_mask = np.zeros((50, 50), dtype=np.uint8)
        static_occ_mask[24, 20] = 1
        building_spacing_mask = np.zeros_like(static_occ_mask)
        scratch = np.zeros_like(static_occ_mask)
        counts = {}

        asset, final_h, final_poly, spacing_poly, skipped = BLD._find_fitting_asset(
            pool,
            np.random.default_rng(1),
            20,
            20,
            0.0,
            1.0,
            static_occ_mask,
            building_spacing_mask,
            scratch,
            counts,
        )

        self.assertEqual(asset["path"], "rotates-to-fit.obj")
        self.assertIn(final_h, (90.0, 270.0))
        self.assertIsNotNone(final_poly)
        self.assertIsNotNone(spacing_poly)
        self.assertEqual(skipped, 0)

    def test_nonresidential_context_skips_house_like_assets_not_facades(self):
        pool = [
            {
                "kind": "object",
                "path": "simheaven/houses/house_09x12x2.obj",
                "bounds_m": (-4.5, 4.5, -6.0, 6.0),
                "fit_bounds_m": (-4.5, 4.5, -6.0, 6.0),
                "mark_bounds_m": (-4.5, 4.5, -6.0, 6.0),
            },
            {
                "kind": "facade",
                "path": "lib/buildings/facades/commercial/low_commercial_01.fac",
                "bounds_m": (-5.0, 5.0, -5.0, 5.0),
                "fit_bounds_m": (-5.0, 5.0, -5.0, 5.0),
                "mark_bounds_m": (-5.0, 5.0, -5.0, 5.0),
            },
        ]
        static_occ_mask = np.zeros((40, 40), dtype=np.uint8)
        building_spacing_mask = np.zeros_like(static_occ_mask)
        scratch = np.zeros_like(static_occ_mask)
        counts = {}

        asset, final_h, final_poly, spacing_poly, skipped = BLD._find_fitting_asset(
            pool,
            np.random.default_rng(1),
            20,
            20,
            0.0,
            1.0,
            static_occ_mask,
            building_spacing_mask,
            scratch,
            counts,
            residential_context=False,
        )

        self.assertEqual(asset["kind"], "facade")
        self.assertEqual(counts["residential_asset_skipped"], 1)
        self.assertEqual(skipped, 0)

    def test_nonresidential_retry_context_preserves_selection_and_skip_counts(self):
        pool = [
            {
                "kind": "object",
                "path": "simheaven/houses/house_09x12x2.obj",
                "bounds_m": (-4.5, 4.5, -6.0, 6.0),
                "fit_bounds_m": (-4.5, 4.5, -6.0, 6.0),
                "mark_bounds_m": (-4.5, 4.5, -6.0, 6.0),
            },
            {
                "kind": "facade",
                "path": "too-large.fac",
                "bounds_m": (-8.0, 8.0, -8.0, 8.0),
                "fit_bounds_m": (-8.0, 8.0, -8.0, 8.0),
                "mark_bounds_m": (-8.0, 8.0, -8.0, 8.0),
            },
            {
                "kind": "object",
                "path": "SFD_Global/Asia/Suburban_1.obj",
                "bounds_m": (-5.0, 5.0, -5.0, 5.0),
                "fit_bounds_m": (-5.0, 5.0, -5.0, 5.0),
                "mark_bounds_m": (-5.0, 5.0, -5.0, 5.0),
            },
            {
                "kind": "facade",
                "path": "small.fac",
                "bounds_m": (-2.0, 2.0, -2.0, 2.0),
                "fit_bounds_m": (-2.0, 2.0, -2.0, 2.0),
                "mark_bounds_m": (-2.0, 2.0, -2.0, 2.0),
            },
        ]
        static_occ_mask = np.zeros((50, 50), dtype=np.uint8)
        static_occ_mask[20, 27] = 1
        building_spacing_mask = np.zeros_like(static_occ_mask)

        standard_counts = {}
        standard = BLD._find_fitting_asset(
            pool,
            np.random.default_rng(0),
            20,
            20,
            0.0,
            1.0,
            static_occ_mask,
            building_spacing_mask,
            np.zeros_like(static_occ_mask),
            standard_counts,
            residential_context=False,
        )

        context_counts = {}
        context = BLD._find_fitting_asset(
            pool,
            np.random.default_rng(0),
            20,
            20,
            0.0,
            1.0,
            static_occ_mask,
            building_spacing_mask,
            np.zeros_like(static_occ_mask),
            context_counts,
            residential_context=False,
            retry_context=BLD._asset_retry_context(pool),
        )

        self.assertEqual(standard[0]["path"], context[0]["path"])
        self.assertEqual(standard[1], context[1])
        np.testing.assert_array_equal(standard[2], context[2])
        np.testing.assert_array_equal(standard[3], context[3])
        self.assertEqual(standard[4], context[4])
        self.assertEqual(standard_counts, context_counts)

    def test_asset_retry_order_prefers_smaller_footprints(self):
        pools = {
            cls: []
            for cls in BLD.BLD_PLACEMENT_CLASSES
        }
        pools[BLD.BLD_CLASS_COMPACT_RESIDENTIAL] = [
            {
                "path": "larger.obj",
                "bounds_m": (-5.0, 5.0, -5.0, 5.0),
                "footprint_area_m2": 100.0,
                "footprint_max_side_m": 10.0,
            },
            {
                "path": "smaller.obj",
                "bounds_m": (-2.0, 2.0, -2.0, 2.0),
                "footprint_area_m2": 16.0,
                "footprint_max_side_m": 4.0,
            },
        ]

        BLD._sort_asset_pools_for_retry(pools)

        self.assertEqual(
            [asset["path"] for asset in pools[BLD.BLD_CLASS_COMPACT_RESIDENTIAL]],
            ["smaller.obj", "larger.obj"],
        )

    def test_keep_smallest_asset_per_class_keeps_one_minimal_asset(self):
        pools = {
            cls: []
            for cls in BLD.BLD_PLACEMENT_CLASSES
        }
        pools[BLD.BLD_CLASS_MEDIUM] = [
            {
                "path": "larger.obj",
                "bounds_m": (-5.0, 5.0, -5.0, 5.0),
                "footprint_area_m2": 100.0,
                "footprint_max_side_m": 10.0,
            },
            {
                "path": "smaller.obj",
                "bounds_m": (-2.0, 2.0, -2.0, 2.0),
                "footprint_area_m2": 16.0,
                "footprint_max_side_m": 4.0,
            },
        ]

        BLD._keep_smallest_asset_per_class(pools)

        self.assertEqual(
            [asset["path"] for asset in pools[BLD.BLD_CLASS_MEDIUM]],
            ["smaller.obj"],
        )

    def test_class_min_footprint_span_uses_smallest_raw_asset_span(self):
        pools = {
            cls: []
            for cls in BLD.BLD_PLACEMENT_CLASSES
        }
        pools[BLD.BLD_CLASS_COMPACT_RESIDENTIAL] = [
            {
                "path": "wider.obj",
                "bounds_m": (-5.0, 5.0, -3.0, 3.0),
            },
            {
                "path": "narrower.obj",
                "bounds_m": (-2.0, 2.0, -4.0, 4.0),
            },
        ]

        min_span = BLD._class_min_footprint_span_m(pools)

        self.assertEqual(min_span[BLD.BLD_CLASS_COMPACT_RESIDENTIAL], 8.0)

    def test_mark_pad_for_edge_spacing_accounts_for_fit_pad(self):
        self.assertEqual(BLD._mark_pad_for_edge_spacing_m(20.0), 16.0)

    def test_asset_retry_sequence_wraps_from_random_offset(self):
        pool = [
            {"path": "a.obj"},
            {"path": "b.obj"},
            {"path": "c.obj"},
            {"path": "d.obj"},
        ]

        sequence = list(BLD._asset_retry_sequence(pool, np.random.default_rng(1)))

        self.assertEqual(
            [asset["path"] for asset in sequence],
            ["b.obj", "c.obj", "d.obj", "a.obj"],
        )

    def test_class_min_fit_inradius_uses_smallest_available_asset(self):
        pools = {
            cls: []
            for cls in BLD.BLD_PLACEMENT_CLASSES
        }
        pools[BLD.BLD_CLASS_COMPACT_RESIDENTIAL] = [
            {
                "path": "larger.obj",
                "fit_bounds_m": (-7.0, 7.0, -5.0, 5.0),
            },
            {
                "path": "smaller.obj",
                "fit_bounds_m": (-3.0, 3.0, -2.0, 2.0),
            },
        ]

        min_radius = BLD._class_min_fit_inradius_m(pools)

        self.assertEqual(min_radius[BLD.BLD_CLASS_COMPACT_RESIDENTIAL], 2.0)

    def test_dynamic_center_blocker_marks_future_impossible_centers(self):
        masks = {
            cls: np.zeros((50, 50), dtype=np.uint8)
            for cls in BLD.BLD_PLACEMENT_CLASSES
        }
        min_radius = {
            cls: 0.0
            for cls in BLD.BLD_PLACEMENT_CLASSES
        }
        min_radius[BLD.BLD_CLASS_COMPACT_RESIDENTIAL] = 3.0

        BLD._mark_dynamic_center_blockers(
            masks,
            25,
            25,
            0.0,
            (-2.0, 2.0, -2.0, 2.0),
            1.0,
            min_radius,
        )

        self.assertEqual(masks[BLD.BLD_CLASS_COMPACT_RESIDENTIAL][25, 25], 1)
        self.assertEqual(masks[BLD.BLD_CLASS_COMPACT_RESIDENTIAL][25, 29], 1)
        self.assertEqual(masks[BLD.BLD_CLASS_COMPACT_RESIDENTIAL][25, 32], 0)
        self.assertEqual(masks[BLD.BLD_CLASS_MEDIUM][25, 25], 0)

    def test_leftover_gap_candidates_are_component_capped(self):
        leftover = np.zeros((30, 30), dtype=np.uint8)
        leftover[4:14, 4:14] = 1
        leftover[18:28, 18:28] = 1
        zone_class = np.zeros_like(leftover, dtype=np.uint8)
        zone_class[leftover != 0] = BLD.BLD_CLASS_MEDIUM

        gap_x, gap_y, gap_cls, _, n_components, n_dropped = BLD._leftover_gap_candidates(
            leftover,
            zone_class,
            max_candidates=3,
            rng=np.random.default_rng(1),
            max_per_component=2,
        )

        self.assertEqual(n_components, 2)
        self.assertEqual(gap_x.size, 3)
        self.assertEqual(gap_y.size, 3)
        self.assertGreaterEqual(n_dropped, 1)
        self.assertTrue(np.all(gap_cls == BLD.BLD_CLASS_MEDIUM))

    def test_tall_apartments_are_allowed_when_their_footprint_fits(self):
        pools = BLD._build_sfd_asset_pools(35.5, 139.5)
        apartment_paths = _paths_for_classes(
            pools,
            (
                BLD.BLD_CLASS_SMALL_APARTMENT,
                BLD.BLD_CLASS_APARTMENT_BLOCK,
            ),
        )

        self.assertIn("SFD_Global/Buildings/Apartment_30m_1.obj", apartment_paths)
        self.assertIn("SFD_Global/Buildings/Apartment_30m_2.obj", apartment_paths)
        self.assertEqual(
            BLD._class_for_footprint(
                BLD._bounds_for_object_path("SFD_Global/Buildings/Apartment_30m_2.obj")
            ),
            BLD.BLD_CLASS_SMALL_APARTMENT,
        )

    def test_tiny_fillers_are_excluded_from_building_pools(self):
        asia_paths = _paths_for_classes(
            BLD._build_sfd_asset_pools(35.5, 139.5),
            (BLD.BLD_CLASS_COMPACT_RESIDENTIAL,),
        )
        simheaven_paths = _paths_for_classes(
            BLD._build_simheaven_asset_pools(tile_lat=35.5, tile_lon=139.5),
            BLD.BLD_PLACEMENT_CLASSES,
        )

        self.assertNotIn("SFD_Global/Asia/Carport_1.obj", asia_paths)
        self.assertNotIn("SFD_Global/Asia/Carport_2.obj", asia_paths)
        self.assertNotIn("SFD_Global/Asia/Shed_1.obj", asia_paths)
        self.assertNotIn("simheaven/sheds/shed_02x03x1.obj", simheaven_paths)

    def test_small_accessory_building_classes_are_included_when_reasonable(self):
        north_america_paths = _paths_for_classes(
            BLD._build_sfd_asset_pools(45.0, -75.0),
            BLD.BLD_PLACEMENT_CLASSES,
        )
        australia_paths = _paths_for_classes(
            BLD._build_sfd_asset_pools(-33.0, 151.0),
            BLD.BLD_PLACEMENT_CLASSES,
        )

        self.assertIn("SFD_Global/New_England/Residential/Garage.obj", north_america_paths)
        self.assertIn("SFD_Global/Australia/Shed.obj", australia_paths)
        self.assertNotIn("SFD_Global/Australia/Carport.obj", australia_paths)

    def test_default_assets_include_facade_and_rectangular_object_variety(self):
        pools = BLD._build_default_asset_pools(45.0, -75.0)
        medium_paths = _paths_for_classes(pools, (BLD.BLD_CLASS_MEDIUM,))
        large_paths = _paths_for_classes(pools, (BLD.BLD_CLASS_LARGE,))

        self.assertIn("lib/buildings/facades/generic/mid_modern_05.fac", medium_paths)
        self.assertIn(
            "lib/buildings/facades/industrial/warehouse_07_90x40.fac",
            large_paths,
        )
        self.assertIn("/lib/global8/us/feat_Building_50_40_600r40.obj", large_paths)
        self.assertEqual(
            BLD._class_for_footprint(BLD._bounds_from_dimensions(50.0, 40.0)),
            BLD.BLD_CLASS_LARGE,
        )

    def test_default_object_catalog_keeps_regional_context(self):
        north_america = _paths_for_classes(
            BLD._build_default_asset_pools(45.0, -75.0),
            BLD.BLD_PLACEMENT_CLASSES,
        )
        europe = _paths_for_classes(
            BLD._build_default_asset_pools(45.0, 7.0),
            BLD.BLD_PLACEMENT_CLASSES,
        )
        asia = _paths_for_classes(
            BLD._build_default_asset_pools(35.5, 139.5),
            BLD.BLD_PLACEMENT_CLASSES,
        )

        self.assertIn("/lib/global8/us/feat_Building_50_40_600r40.obj", north_america)
        self.assertNotIn("/lib/global8/us/feat_Building_50_40_600r40.obj", europe)
        self.assertIn("/lib/global8/us/hill_sq_30_30r.obj", europe)
        self.assertNotIn("/lib/global8/us/hill_sq_30_30r.obj", asia)

    def test_simheaven_catalog_expands_all_footprint_classes(self):
        pools = BLD._build_simheaven_asset_pools([], 45.0, 7.0)
        small_paths = _paths_for_classes(pools, (BLD.BLD_CLASS_SMALL_RESIDENTIAL,))
        compact_paths = _paths_for_classes(pools, (BLD.BLD_CLASS_COMPACT_RESIDENTIAL,))
        medium_paths = _paths_for_classes(pools, (BLD.BLD_CLASS_MEDIUM,))
        apartment_paths = _paths_for_classes(
            pools,
            (
                BLD.BLD_CLASS_SMALL_APARTMENT,
                BLD.BLD_CLASS_APARTMENT_BLOCK,
            ),
        )
        large_paths = _paths_for_classes(pools, (BLD.BLD_CLASS_LARGE,))

        self.assertIn("simheaven/houses/house_09x12x1.obj", small_paths)
        self.assertIn("simheaven/houses/house_12x15x2.obj", compact_paths)
        self.assertIn("simheaven/residential/residential_15x20x4.obj", medium_paths)
        self.assertIn("simheaven/residential/residential_20x30x3.obj", apartment_paths)
        self.assertIn("simheaven/industrial/industrial_30x60.obj", large_paths)

    def test_simheaven_catalog_keeps_regional_context(self):
        asia_paths = _paths_for_classes(
            BLD._build_simheaven_asset_pools([], 35.5, 139.5),
            BLD.BLD_PLACEMENT_CLASSES,
        )
        europe_paths = _paths_for_classes(
            BLD._build_simheaven_asset_pools([], 45.0, 7.0),
            BLD.BLD_PLACEMENT_CLASSES,
        )

        self.assertIn("simheaven/houses/house_09x12x1.obj", asia_paths)
        self.assertIn("simheaven/commercial/commercial_18x42.obj", asia_paths)
        self.assertIn("simheaven/industrial/industrial_30x60.obj", asia_paths)
        self.assertIn("simheaven/industrial/industrial_30x60.obj", europe_paths)

    def test_simheaven_special_landmarks_are_not_reused_as_assets(self):
        special_objects = [
            {
                "path": "simheaven/landmarks/church_20x30.obj",
                "w_m": 20.0,
                "h_m": 30.0,
            },
            {
                "path": "simheaven/landmarks/cathedral_40x60.obj",
                "w_m": 40.0,
                "h_m": 60.0,
            },
            {
                "path": "simheaven/commercial/school_30x40.obj",
                "w_m": 30.0,
                "h_m": 40.0,
            },
            {
                "path": "simheaven/commercial/petrol_18x18.obj",
                "w_m": 18.0,
                "h_m": 18.0,
            },
            {
                "path": "simheaven/commercial/supermarket_30x24.obj",
                "w_m": 30.0,
                "h_m": 24.0,
            },
            {
                "path": "simheaven/commercial/hospital_30x40.obj",
                "w_m": 30.0,
                "h_m": 40.0,
            },
            {
                "path": "simheaven/commercial/townhall_20x20.obj",
                "w_m": 20.0,
                "h_m": 20.0,
            },
            {
                "path": "simheaven/commercial/bank_15x20.obj",
                "w_m": 15.0,
                "h_m": 20.0,
            },
            {
                "path": "simheaven/residential/residential_15x20x4.obj",
                "w_m": 15.0,
                "h_m": 20.0,
            },
        ]
        paths = _paths_for_classes(
            BLD._build_simheaven_asset_pools(special_objects, 45.0, 7.0),
            BLD.BLD_PLACEMENT_CLASSES,
        )

        self.assertTrue(
            BLD._is_simheaven_building_object("simheaven/landmarks/church_20x30.obj")
        )
        self.assertNotIn("simheaven/landmarks/church_20x30.obj", paths)
        self.assertNotIn("simheaven/landmarks/cathedral_40x60.obj", paths)
        self.assertNotIn("simheaven/commercial/school_30x40.obj", paths)
        self.assertNotIn("simheaven/commercial/petrol_18x18.obj", paths)
        self.assertNotIn("simheaven/commercial/supermarket_30x24.obj", paths)
        self.assertNotIn("simheaven/commercial/hospital_30x40.obj", paths)
        self.assertNotIn("simheaven/commercial/townhall_20x20.obj", paths)
        self.assertNotIn("simheaven/commercial/bank_15x20.obj", paths)
        self.assertIn("simheaven/residential/residential_15x20x4.obj", paths)

        catalog_paths = _paths_for_classes(
            BLD._build_simheaven_asset_pools([], 45.0, 7.0),
            BLD.BLD_PLACEMENT_CLASSES,
        )
        self.assertNotIn("simheaven/commercial/petrol_18x18.obj", catalog_paths)
        self.assertNotIn("simheaven/commercial/supermarket_30x24.obj", catalog_paths)

    def test_long_slab_and_industrial_assets_stay_out_of_residential_pools(self):
        pools = BLD._build_sfd_asset_pools(35.5, 139.5)
        residential_and_apartment_paths = _paths_for_classes(
            pools,
            (
                BLD.BLD_CLASS_COMPACT_RESIDENTIAL,
                BLD.BLD_CLASS_MEDIUM,
                BLD.BLD_CLASS_SMALL_APARTMENT,
                BLD.BLD_CLASS_APARTMENT_BLOCK,
            ),
        )

        self.assertNotIn("SFD_Global/Asia/Apartment_2.obj", residential_and_apartment_paths)
        self.assertTrue(
            all("/Industry_" not in path for path in residential_and_apartment_paths)
        )
        self.assertEqual(
            BLD._class_for_footprint(
                BLD._bounds_for_object_path("SFD_Global/Asia/Apartment_2.obj")
            ),
            BLD.BLD_CLASS_LARGE,
        )

    def test_sfd_catalog_is_a_strict_allowlist(self):
        pools = BLD._build_sfd_asset_pools(3.0, 102.0)
        all_paths = _paths_for_classes(pools, BLD.BLD_PLACEMENT_CLASSES)

        self.assertNotIn("SFD_Global/Asia/Suburban_South_Test.obj", all_paths)
        self.assertNotIn("SFD_Global/Asia/blank.obj", all_paths)
        self.assertNotIn("SFD_Global/Asia/Store.obj", all_paths)
        self.assertNotIn("SFD_Global/Asia/Sign_Drugstore.obj", all_paths)
        self.assertNotIn("SFD_Global/Australia/Pool.obj", all_paths)
        self.assertNotIn("SFD_Global/Australia/Trampoline.obj", all_paths)
        self.assertIsNone(
            BLD._bounds_for_object_path("SFD_Global/Asia/Suburban_South_Test.obj")
        )


if __name__ == "__main__":
    unittest.main()
