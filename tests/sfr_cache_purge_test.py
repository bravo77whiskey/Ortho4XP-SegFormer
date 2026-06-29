"""Guards for SFR tile-cache cleanup behavior."""

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

import O4_SFR_Pipeline as SFR


class SfrCachePurgeTests(unittest.TestCase):
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

            SFR._purge_sfr_cache(
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

            SFR._cleanup_sfr_derived_cache(cache_dir, 'bld')

            for path in (veg, bld, yolo):
                self.assertTrue(os.path.exists(path), f"primary cache removed: {path}")
            for path in (road, aux, poly, osm, mesh, parsed, dsf, obj, analysis):
                self.assertFalse(os.path.exists(path), f"derived cache still present: {path}")

    def test_purge_keeps_unrelated_files(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            keep = self._make(cache_dir, "notes.txt")

            SFR._purge_sfr_cache(
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

            old_disable_cache = SFR.sfr_bld_disable_cache
            SFR.sfr_bld_disable_cache = True
            try:
                with mock.patch.object(SFR, "_sfr_cache_dir", return_value=cache_dir), \
                        mock.patch.object(SFR, "_check_tile_imagery", return_value=True), \
                        mock.patch.object(SFR, "_auto_setup", return_value=True), \
                        mock.patch.object(SFR, "_run_venv", side_effect=_fail_after_writing_cache), \
                        mock.patch.object(SFR, "_dsftool_path", return_value=None), \
                        mock.patch.object(SFR, "_scenery_paths", return_value=("", "", "")):
                    with self.assertRaisesRegex(RuntimeError, "exit 3"):
                        SFR.process_bld_tile(12, 34, tmpdir)
            finally:
                SFR.sfr_bld_disable_cache = old_disable_cache

            self.assertFalse(os.path.exists(cache_dir))


if __name__ == "__main__":
    unittest.main()
