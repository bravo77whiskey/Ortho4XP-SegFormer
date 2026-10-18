import sys
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_Building_Overlay as BLD


def _lake(n_side, south=0.0, west=0.0, size=0.01):
    """A square lake meshed into 2 * n_side**2 triangles, as the mesh water index holds them."""
    step = size / n_side
    tris = []
    for i in range(n_side):
        for j in range(n_side):
            la, lo = south + i * step, west + j * step
            a, b, c, d = (la, lo), (la, lo + step), (la + step, lo + step), (la + step, lo)
            tris += [(a, b, c), (a, c, d)]
    tris = np.asarray(tris, np.float64)
    return {"tris": tris, "south": tris[:, :, 0].min(1), "north": tris[:, :, 0].max(1),
            "west": tris[:, :, 1].min(1), "east": tris[:, :, 1].max(1)}


class MeshWaterRasterTests(unittest.TestCase):
    def test_a_finely_meshed_lake_fills_its_area(self):
        # Lake over the west half of the texture, 0.01 deg square.
        idx = _lake(40, south=0.0, west=0.0, size=0.01)
        mask = BLD._rasterize_mesh_water_mask(idx, 0.01, 0.0, 0.0, 0.02, 500, 1000)
        self.assertTrue(mask[:, :495].all())
        self.assertFalse(mask[:, 505:].any())

    def test_hundreds_of_thousands_of_triangles_stay_fast(self):
        # +30+120's Taihu shore texture has 974k water triangles; one fillPoly
        # over all of them took 854 s.
        idx = _lake(300)                       # 180,000 triangles
        t = time.perf_counter()
        mask = BLD._rasterize_mesh_water_mask(idx, 0.01, 0.0, 0.0, 0.01, 2048, 2048)
        self.assertLess(time.perf_counter() - t, 5.0)
        self.assertTrue(mask.all())


if __name__ == "__main__":
    unittest.main()
