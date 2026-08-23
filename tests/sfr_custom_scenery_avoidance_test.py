import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_SFR_DSF_Utils as DSF
import O4_SFR_Building_Overlay as BLD
import O4_SFR_Vegetation_Overlay as VEG


def _write_obj(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "I",
                "800",
                "OBJ",
                "VT -1.0 0.0 -2.0 0 1 0 0 0",
                "VT 3.0 0.0 4.0 0 1 0 1 1",
            ]
        ),
        encoding="utf-8",
    )


class CustomSceneryAvoidanceTests(unittest.TestCase):
    def test_scenery_packs_parser_keeps_enabled_existing_unique_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            custom = Path(tmpdir) / "Custom Scenery"
            active = custom / "Active Pack"
            disabled = custom / "Disabled Pack"
            space = custom / "Pack With Spaces"
            active.mkdir(parents=True)
            disabled.mkdir()
            space.mkdir()
            (custom / "scenery_packs.ini").write_text(
                "\n".join(
                    [
                        "I",
                        "1000 Version",
                        "SCENERY",
                        "SCENERY_PACK Custom Scenery/Active Pack/",
                        "SCENERY_PACK_DISABLED Custom Scenery/Disabled Pack/",
                        "SCENERY_PACK Custom Scenery/Pack With Spaces/",
                        "SCENERY_PACK Custom Scenery/Active Pack/",
                    ]
                ),
                encoding="utf-8",
            )

            packs = DSF.active_scenery_pack_dirs(custom)

        self.assertEqual([name for name, _ in packs], ["Active Pack", "Pack With Spaces"])

    def test_active_custom_scenery_dsf_search_matches_basename_and_skips_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            custom = Path(tmpdir) / "Custom Scenery"
            active = custom / "Active Pack"
            disabled = custom / "Disabled Pack"
            output_pack = custom / "yOrtho4XP_Bld_Overlays"
            for pack in (active, disabled, output_pack):
                (pack / "Earth nav data" / "+20+120").mkdir(parents=True)
                (pack / "Earth nav data" / "+20+120" / "+22+120.dsf").write_text("", encoding="utf-8")
            output_dsf = output_pack / "Earth nav data" / "+20+120" / "+22+120.dsf"
            (custom / "scenery_packs.ini").write_text(
                "\n".join(
                    [
                        "SCENERY_PACK Custom Scenery/Active Pack/",
                        "SCENERY_PACK_DISABLED Custom Scenery/Disabled Pack/",
                        "SCENERY_PACK Custom Scenery/yOrtho4XP_Bld_Overlays/",
                    ]
                ),
                encoding="utf-8",
            )

            matches = DSF.find_active_custom_scenery_dsfs(
                custom,
                "+22+120.dsf",
                skip_dsf_path=output_dsf,
            )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0], "Active Pack")
        self.assertTrue(matches[0][1].endswith(os.path.join("+20+120", "+22+120.dsf")))

    def test_simheaven_forest_search_uses_only_enabled_7_forests_pack(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            custom = Path(tmpdir) / "Custom Scenery"
            enabled = custom / "simHeaven_X-World_Asia-7-forests"
            disabled = custom / "simHeaven_X-World_Europe-7-forests"
            for package in (enabled, disabled):
                dsf = package / "Earth nav data" / "+20+120" / "+22+120.dsf"
                dsf.parent.mkdir(parents=True)
                dsf.write_bytes(b"dsf")
            (custom / "scenery_packs.ini").write_text(
                "\n".join(
                    [
                        "SCENERY_PACK Custom Scenery/"
                        "simHeaven_X-World_Asia-7-forests/",
                        "SCENERY_PACK_DISABLED Custom Scenery/"
                        "simHeaven_X-World_Europe-7-forests/",
                    ]
                ),
                encoding="utf-8",
            )

            matches = DSF.find_simheaven_forest_dsfs(custom, 22, 120)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0], enabled.name)

    def test_simheaven_forest_parser_keeps_all_families_for_overlap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dsf = tmp / "+22+120.dsf"
            dsf.write_bytes(b"dsf")
            text = tmp / "+22+120.txt"
            text.write_text(
                "\n".join(
                    [
                        "POLYGON_DEF simheaven/forests/broad.for",
                        "POLYGON_DEF simheaven/forests/orchard.for",
                        "BEGIN_POLYGON 0 255 2",
                        "BEGIN_WINDING",
                        "POLYGON_POINT 120.1 22.1",
                        "POLYGON_POINT 120.2 22.1",
                        "POLYGON_POINT 120.2 22.2",
                        "END_WINDING",
                        "END_POLYGON",
                        "BEGIN_POLYGON 1 180 2",
                        "BEGIN_WINDING",
                        "POLYGON_POINT 120.3 22.3",
                        "POLYGON_POINT 120.4 22.3",
                        "POLYGON_POINT 120.4 22.4",
                        "END_WINDING",
                        "END_POLYGON",
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch.object(
                VEG,
                "ensure_cached_dsf_text",
                return_value=str(text),
            ), mock.patch.object(
                VEG.PCACHE,
                "load_or_build",
                side_effect=lambda _path, _cache, _kind, build, **_kwargs: build(),
            ):
                records = VEG._load_forest_polygons(
                    "simHeaven X-World",
                    [("simHeaven_X-World_Asia-7-forests", str(dsf))],
                    "DSFTool.exe",
                    tmpdir,
                )

        self.assertEqual(
            [record["path"] for record in records],
            [
                "simheaven/forests/broad.for",
                "simheaven/forests/orchard.for",
            ],
        )

    def test_obj8_bounds_and_library_resolution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            custom = Path(tmpdir) / "Custom Scenery"
            lib = custom / "Library Pack"
            obj = lib / "objects" / "building.obj"
            _write_obj(obj)
            lib.mkdir(parents=True, exist_ok=True)
            (lib / "library.txt").write_text(
                "EXPORT lib/custom/building.obj objects/building.obj\n",
                encoding="utf-8",
            )
            (custom / "scenery_packs.ini").write_text(
                "SCENERY_PACK Custom Scenery/Library Pack/\n",
                encoding="utf-8",
            )
            index = BLD._active_custom_library_index(custom)

            resolved = BLD._resolve_custom_object_path(
                "lib/custom/building.obj",
                None,
                index,
            )
            bounds = BLD._read_obj8_bounds(resolved, tmpdir)

        self.assertEqual(Path(resolved), obj)
        self.assertEqual(bounds, (-1.0, 3.0, -2.0, 4.0))

    def test_generic_custom_dsf_parser_extracts_objects_and_facades(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            package = tmp / "Custom Scenery" / "Active Pack"
            _write_obj(package / "objects" / "building.obj")
            dsf = package / "Earth nav data" / "+20+120" / "+22+120.dsf"
            dsf.parent.mkdir(parents=True)
            dsf.write_text("", encoding="utf-8")
            text = tmp / "source.txt"
            text.write_text(
                "\n".join(
                    [
                        "OBJECT_DEF objects/building.obj",
                        "OBJECT_DEF objects/missing.obj",
                        "POLYGON_DEF lib/custom/building.fac",
                        "POLYGON_DEF lib/custom/forest.for",
                        "OBJECT 0 120.5000000 22.5000000 45.0",
                        "OBJECT 1 120.6000000 22.6000000 0.0",
                        "BEGIN_POLYGON 0 0 2",
                        "BEGIN_WINDING",
                        "POLYGON_POINT 120.1 22.1",
                        "POLYGON_POINT 120.2 22.1",
                        "POLYGON_POINT 120.2 22.2",
                        "END_WINDING",
                        "END_POLYGON",
                        "BEGIN_POLYGON 1 0 2",
                        "BEGIN_WINDING",
                        "POLYGON_POINT 120.3 22.3",
                        "POLYGON_POINT 120.4 22.3",
                        "POLYGON_POINT 120.4 22.4",
                        "END_WINDING",
                        "END_POLYGON",
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch.object(BLD, "ensure_cached_dsf_text", return_value=str(text)), \
                    mock.patch.object(
                        BLD,
                        "find_active_custom_scenery_dsfs",
                        return_value=[("Active Pack", str(dsf), str(package))],
                    ), \
                    mock.patch.object(BLD, "_active_custom_library_index", return_value={}):
                polys, objects, skipped, layers = BLD._load_custom_scenery_building_exclusions(
                    str(tmp / "Custom Scenery"),
                    22,
                    120,
                    str(tmp / "out.dsf"),
                    "DSFTool.exe",
                    tmpdir,
                )

        self.assertEqual(layers, 1)
        self.assertEqual(len(objects), 1)
        self.assertEqual((objects[0]["w_m"], objects[0]["h_m"]), (4.0, 6.0))
        self.assertEqual(len(polys), 1)
        self.assertEqual(polys[0][0], (22.1, 120.1))
        self.assertEqual(skipped, 1)

    def test_generic_custom_dsf_parser_skips_street_light_clutter(self):
        # Name-token matches (street lights, lamps, lanterns, light poles).
        self.assertTrue(BLD._is_nonblocking_clutter_object("objects/Street_Light_3.obj"))
        self.assertTrue(BLD._is_nonblocking_clutter_object("props/streetlight.obj"))
        self.assertTrue(BLD._is_nonblocking_clutter_object("lib/lamppost_01.obj"))
        # Real-world names seen in installed packs (ChudobaDesign, aericaps, MisterX).
        self.assertTrue(BLD._is_nonblocking_clutter_object("ChudobaDesign_Library/lampa.obj"))
        self.assertTrue(BLD._is_nonblocking_clutter_object("aericaps_collection/Street/street_lamp_single-11m.obj"))
        self.assertTrue(BLD._is_nonblocking_clutter_object("aericaps_collection/Building/lightpole.obj"))
        self.assertTrue(BLD._is_nonblocking_clutter_object("ruscenery/houses/lantern.obj"))
        # Dedicated lighting folders catch whole packs (default X-Plane g10 set,
        # MisterX airport lights) regardless of the per-object filename.
        self.assertTrue(BLD._is_nonblocking_clutter_object("lib/g10/streetlights/ResLt1.obj"))
        self.assertTrue(BLD._is_nonblocking_clutter_object("MisterX_Library/Airport/Lights/Airport_Light_1.obj"))
        # A lighthouse is a real landmark building, not street furniture.
        self.assertFalse(BLD._is_nonblocking_clutter_object("landmarks/Lighthouse.obj"))
        self.assertFalse(BLD._is_nonblocking_clutter_object("objects/building.obj"))
        # A place name in a parent folder must not trigger the "lamp" token.
        self.assertFalse(BLD._is_nonblocking_clutter_object("Lampedusa/objects/terminal.obj"))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            package = tmp / "Custom Scenery" / "Active Pack"
            _write_obj(package / "objects" / "building.obj")
            dsf = package / "Earth nav data" / "+20+120" / "+22+120.dsf"
            dsf.parent.mkdir(parents=True)
            dsf.write_text("", encoding="utf-8")
            text = tmp / "source.txt"
            text.write_text(
                "\n".join(
                    [
                        "OBJECT_DEF objects/building.obj",
                        "OBJECT_DEF props/Street_Light_3.obj",
                        "OBJECT 0 120.5000000 22.5000000 45.0",
                        "OBJECT 1 120.6000000 22.6000000 0.0",
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch.object(BLD, "ensure_cached_dsf_text", return_value=str(text)), \
                    mock.patch.object(
                        BLD,
                        "find_active_custom_scenery_dsfs",
                        return_value=[("Active Pack", str(dsf), str(package))],
                    ), \
                    mock.patch.object(BLD, "_active_custom_library_index", return_value={}):
                polys, objects, skipped, layers = BLD._load_custom_scenery_building_exclusions(
                    str(tmp / "Custom Scenery"),
                    22,
                    120,
                    str(tmp / "out.dsf"),
                    "DSFTool.exe",
                    tmpdir,
                )

        # Only the building is kept as an occupancy blocker; the street light is
        # dropped (and not counted as a parse failure).
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["path"], "objects/building.obj")
        self.assertEqual(skipped, 0)

    def test_custom_object_mask_rejects_overlapping_yolo_footprint(self):
        custom_objects = {
            "lat": BLD.np.array([0.5], dtype=BLD.np.float32),
            "lon": BLD.np.array([0.5], dtype=BLD.np.float32),
            "heading": BLD.np.array([0.0], dtype=BLD.np.float32),
            "w_m": BLD.np.array([80.0], dtype=BLD.np.float32),
            "h_m": BLD.np.array([80.0], dtype=BLD.np.float32),
        }
        static_occ_mask = BLD._rasterize_simheaven_objects(
            custom_objects,
            lat_n=1.0,
            lat_s=0.0,
            lon_w=0.0,
            lon_e=1.0,
            img_h=100,
            img_w=100,
            m_per_px=2.0,
            margin_m=0.0,
        )
        scratch = BLD.np.zeros_like(static_occ_mask)
        yolo_poly = BLD.np.array(
            [[40, 40], [60, 40], [60, 60], [40, 60]],
            dtype=BLD.np.int32,
        )

        self.assertFalse(
            BLD._direct_yolo_poly_fits(
                static_occ_mask,
                BLD.np.zeros_like(static_occ_mask),
                yolo_poly,
                scratch_mask=scratch,
                static_occ_integral=BLD.cv2.integral(static_occ_mask, sdepth=BLD.cv2.CV_32S),
            )
        )


if __name__ == "__main__":
    unittest.main()
