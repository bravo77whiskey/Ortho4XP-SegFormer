"""Guard: disabling the SFR cache must leave the per-tile cache folder empty.

The per-DDS inference/placement files respect the overlay ``disable_cache``
flag, but the persistent sidecar caches written by ``PCACHE.load_or_build``
(parsed OSM, mesh-water index, disassembled DSF text, OBJ8 bounds) ignore it.
``O4_SFR_Pipeline._purge_sfr_cache`` clears both when caching is disabled so
``SFR_cache/<tile>`` is not retained between builds.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

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

    def test_purge_removes_per_dds_and_persistent_sidecars(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            per_dds = self._make(cache_dir, "+12+034_bld.pkl")
            yolo = self._make(cache_dir, "+12+034_yolo_obb.pkl")
            osm = self._make(cache_dir, "osm_parse/abc.pkl")
            mesh = self._make(cache_dir, "mesh_water/def.pkl")
            dsf = self._make(cache_dir, "dsf_disassembly/ghi.txt")

            SFR._purge_sfr_cache(
                cache_dir,
                ['*_bld.pkl', '*_yolo_obb.pkl', '*_road.pkl', '*_veg.npy'],
                'bld',
            )

            for path in (per_dds, yolo, osm, mesh, dsf):
                self.assertFalse(os.path.exists(path), f"still present: {path}")
            self.assertFalse(os.path.isdir(os.path.join(cache_dir, "osm_parse")))

    def test_purge_keeps_unrelated_per_dds_files(self):
        # The bld purge must not delete another overlay's per-DDS outputs that
        # are not in its pattern list (e.g. vegetation polygon caches).
        with tempfile.TemporaryDirectory() as cache_dir:
            keep = self._make(cache_dir, "+12+034_vegpoly.pkl")

            SFR._purge_sfr_cache(
                cache_dir,
                ['*_bld.pkl', '*_yolo_obb.pkl', '*_road.pkl'],
                'bld',
            )

            self.assertTrue(os.path.exists(keep))


if __name__ == "__main__":
    unittest.main()
