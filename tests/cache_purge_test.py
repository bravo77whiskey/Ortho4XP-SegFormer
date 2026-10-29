"""Guards for overlay tile-cache cleanup behavior."""

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

import O4_Overlay_Pipeline as PIPELINE


class CachePurgeTests(unittest.TestCase):
    def _make(self, cache_dir, rel):
        path = os.path.join(cache_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(b"x")
        return path

    def test_disabled_purge_removes_per_dds_tile_files_and_sidecars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "SFR_cache", "+12+034")
            per_dds = self._make(cache_dir, "+12+034_bld.pkl")
            yolo = self._make(cache_dir, "+12+034_yolo_obb.pkl")
            roads = self._make(cache_dir, "+12+034_all_roads.osm.bz2")
            osm = self._make(cache_dir, "osm_parse/abc.pkl")
            mesh = self._make(cache_dir, "mesh_water/def.pkl")
            parsed = self._make(cache_dir, "dsf_parse/jkl.pkl")
            dsf = self._make(cache_dir, "dsf_disassembly/ghi.txt")
            obj = self._make(cache_dir, "obj8_bounds/mno.pkl")
            analysis = self._make(cache_dir, "yolo_zl16_analysis/a.png")

            PIPELINE._purge_overlay_cache(
                cache_dir,
                ['*_bld.pkl', '*_yolo_obb.pkl', '*_road.pkl', '*_veg.npy'],
                'bld',
            )

            for path in (per_dds, yolo, roads, osm, mesh, parsed, dsf, obj, analysis):
                self.assertFalse(os.path.exists(path), f"still present: {path}")
            self.assertFalse(os.path.exists(cache_dir))

    def test_enabled_cleanup_drops_derived_but_keeps_primary_reusable_files(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            veg = self._make(cache_dir, "+12+034_veg.npy")
            bld = self._make(cache_dir, "+12+034_bld.pkl")
            yolo = self._make(cache_dir, "+12+034_yolo_obb.pkl")
            road = self._make(cache_dir, "+12+034_road.pkl")
            aux = self._make(cache_dir, "+12+034_vegaux.pkl")
            poly = self._make(cache_dir, "+12+034_vegpoly.pkl")
            osm = self._make(cache_dir, "osm_parse/abc.pkl")
            mesh = self._make(cache_dir, "mesh_water/def.pkl")
            parsed = self._make(cache_dir, "dsf_parse/jkl.pkl")
            dsf = self._make(cache_dir, "dsf_disassembly/ghi.txt")
            obj = self._make(cache_dir, "obj8_bounds/mno.pkl")
            analysis = self._make(cache_dir, "yolo_zl16_analysis/a.png")

            PIPELINE._cleanup_derived_cache(cache_dir, 'bld')

            for path in (veg, bld, yolo):
                self.assertTrue(os.path.exists(path), f"primary cache removed: {path}")
            for path in (road, aux, poly, osm, mesh, parsed, dsf, obj, analysis):
                self.assertFalse(os.path.exists(path), f"derived cache still present: {path}")

    def test_purge_keeps_unrelated_files(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            keep = self._make(cache_dir, "notes.txt")

            PIPELINE._purge_overlay_cache(
                cache_dir,
                ['*_bld.pkl', '*_yolo_obb.pkl', '*_road.pkl'],
                'bld',
            )

            self.assertTrue(os.path.exists(keep))

    def test_disabled_bld_process_purges_after_subprocess_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "SFR_cache", "+12+034")

            def _fail_after_writing_cache(_code):
                self._make(cache_dir, "+12+034_bld.pkl")
                self._make(cache_dir, "+12+034_all_roads.osm.bz2")
                self._make(cache_dir, "osm_parse/abc.pkl")
                return 3

            old_disable_cache = PIPELINE.bld_disable_cache
            PIPELINE.bld_disable_cache = True
            try:
                with mock.patch.object(PIPELINE, "_overlay_cache_dir", return_value=cache_dir), \
                        mock.patch.object(PIPELINE, "_check_tile_imagery", return_value=True), \
                        mock.patch.object(PIPELINE, "_auto_setup", return_value=True), \
                        mock.patch.object(PIPELINE, "_run_venv", side_effect=_fail_after_writing_cache), \
                        mock.patch.object(PIPELINE, "_dsftool_path", return_value=None), \
                        mock.patch.object(PIPELINE, "_scenery_paths", return_value=("", "", "")):
                    with self.assertRaisesRegex(RuntimeError, "exit 3"):
                        PIPELINE.process_bld_tile(12, 34, tmpdir)
            finally:
                PIPELINE.bld_disable_cache = old_disable_cache

            self.assertFalse(os.path.exists(cache_dir))

    def test_side_by_side_bld_cleanup_leaves_the_running_veg_grass_files(self):
        # Building overlay done while the vegetation overlay still runs: its
        # cache purge used to delete the grass class maps, and the tile got no grass.
        import re
        import threading
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "SFR_cache", "+12+034")
            os.makedirs(cache_dir)
            grass_cls = os.path.join(cache_dir, "1_2_BI18_grasscls.npy")
            veg_wrote = threading.Event()
            seen = {}

            def _venv(code):
                if "veg_overlay.run" not in code:          # the building overlay
                    veg_wrote.wait(5)
                    return 0
                self._make(cache_dir, os.path.basename(grass_cls))
                veg_wrote.set()
                flag = re.search(r"O4_VEG_WAIT_FOR_BLD_SEG'\] = '([^']+)'", code).group(1)
                deadline = time.time() + 5
                while not os.path.exists(flag) and time.time() < deadline:
                    time.sleep(0.01)
                seen["bld_done"] = os.path.exists(flag)
                seen["grass_kept"] = os.path.exists(grass_cls)
                return 0

            with mock.patch.object(PIPELINE, "bld_disable_cache", True), \
                    mock.patch.object(PIPELINE, "veg_disable_cache", True), \
                    mock.patch.object(PIPELINE, "overlays_run_together", return_value=True), \
                    mock.patch.object(PIPELINE, "_overlay_cache_dir", return_value=cache_dir), \
                    mock.patch.object(PIPELINE, "_check_tile_imagery", return_value=True), \
                    mock.patch.object(PIPELINE, "_auto_setup", return_value=True), \
                    mock.patch.object(PIPELINE, "_run_venv", side_effect=_venv), \
                    mock.patch.object(PIPELINE, "_dsftool_path", return_value=None), \
                    mock.patch.object(PIPELINE, "_scenery_paths", return_value=""), \
                    mock.patch.object(PIPELINE, "_resolve_remote_host", return_value=""), \
                    mock.patch.object(PIPELINE, "_finish_veg_tile_replacement"):
                errors = PIPELINE.process_overlays_tile(12, 34, tmpdir)

            self.assertEqual(errors, {"bld": None, "veg": None})
            self.assertTrue(seen["bld_done"])
            self.assertTrue(seen["grass_kept"], "building overlay purged the running grass class maps")
            self.assertFalse(os.path.exists(cache_dir), "tile cache not cleaned up after both overlays")


if __name__ == "__main__":
    unittest.main()
