import io
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import torch
except ImportError:  # pragma: no cover - exercised only in minimal envs
    torch = None

import O4_Canopy_Height as CANOPY
import O4_Vegetation_Overlay as VEG


class CanopySwitchTests(unittest.TestCase):
    def test_canopy_on_by_default_and_in_feature_sig(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("O4_VEG_CANOPY", None)
            self.assertTrue(VEG._canopy_enabled())
            on = VEG._veg_feature_sig()
            os.environ["O4_VEG_CANOPY"] = "0"
            self.assertFalse(VEG._canopy_enabled())
            self.assertNotEqual(on, VEG._veg_feature_sig())
            os.environ.pop("O4_VEG_CANOPY", None)

    def test_narrow_rows_keep_tree_lines_and_drop_blobs(self):
        mpp = 0.2
        mask = np.zeros((1000, 1000), np.uint8)
        mask[100:900, 100:120] = 1      # 160 m x 4 m roadside row
        mask[300:700, 400:800] = 1      # 80 m x 80 m blob on grass
        rows = VEG._narrow_rows(mask, mpp)
        self.assertTrue(rows[500, 110])
        self.assertFalse(rows[500, 600])
        self.assertLess(rows[300:700, 400:800].mean(), 0.05)

    def test_only_textures_above_zl17_are_shrunk(self):
        self.assertEqual(VEG._veg_infer_factor(16), 1)
        self.assertEqual(VEG._veg_infer_factor(17), 1)
        self.assertEqual(VEG._veg_infer_factor(18), 2)
        self.assertEqual(VEG._veg_infer_factor(19), 4)


class CanopyHelperTests(unittest.TestCase):
    def test_tree_mask_by_height_or_by_confidence(self):
        maps = np.zeros((1 + len(CANOPY.LEVELS_M), 4, 4), np.float16)
        maps[0, :2, :2] = 10.0                                   # tall, top-left
        maps[1 + CANOPY.LEVELS_M.index(2.0), 2:, 2:] = 0.6       # likely >= 2 m, bottom-right
        by_h = CANOPY.tree_mask(maps, (16, 16), 2.0)
        self.assertEqual(by_h.shape, (16, 16))
        self.assertTrue(by_h[0, 0])
        self.assertFalse(by_h[15, 15])
        by_conf = CANOPY.tree_mask(maps, (16, 16), 2.0, 0.5)
        self.assertTrue(by_conf[15, 15])
        self.assertFalse(by_conf[0, 0])

    def test_forest_clearings_become_holes(self):
        mpp = 0.5
        mask = np.zeros((400, 400), np.uint8)
        mask[50:350, 50:350] = 1          # 150 m x 150 m forest
        mask[150:250, 150:250] = 0        # 50 m x 50 m clearing -> hole
        mask[80:86, 80:86] = 0            # 3 m x 3 m gap -> stays forest
        kwargs = dict(img_w=400, img_h=400, lat_n=10.002, lat_s=10.0, lon_w=104.0, lon_e=104.002,
                      tile_lat=10, tile_lon=104, m_per_px=mpp, min_area_px=1, simplify_px=1.0,
                      region="tropical", rng=np.random.default_rng(0), density_override=1.0,
                      context_masks={})
        with mock.patch.dict(os.environ, {"O4_VEG_FOREST_HOLES": "1", "O4_VEG_POINT_TREES": "1"}):
            polys = VEG._process_dds_mask(mask, VEG.SEGFORMER.CLASS_TREE, **kwargs)
        area = [ring for _p, den, ring in polys if not den & VEG.POINT_MODE]
        self.assertEqual(len(area), 1)
        holes = getattr(area[0], "holes", [])
        self.assertEqual(len(holes), 1)
        self.assertNotEqual(VEG._signed_area(list(area[0])[:-1]) >= 0,
                            VEG._signed_area(holes[0][:-1]) >= 0)
        with mock.patch.dict(os.environ, {"O4_VEG_FOREST_HOLES": "0"}):
            plain = VEG._process_dds_mask(mask, VEG.SEGFORMER.CLASS_TREE, **kwargs)
        self.assertEqual(getattr(plain[0][2], "holes", []), [])
        import pickle
        back = pickle.loads(pickle.dumps(area[0]))
        self.assertEqual(back.holes, holes)

    def test_segformer_height_gate_switch(self):
        with mock.patch.dict(os.environ, {"O4_VEG_CANOPY_SEG_MIN_H": ""}):
            self.assertEqual(VEG._canopy_seg_min_h(), VEG.CANOPY_SEG_MIN_H_M)
            on = VEG._veg_feature_sig()
        with mock.patch.dict(os.environ, {"O4_VEG_CANOPY_SEG_MIN_H": "off"}):
            self.assertIsNone(VEG._canopy_seg_min_h())
            self.assertNotEqual(on, VEG._veg_feature_sig())
        with mock.patch.dict(os.environ, {"O4_VEG_CANOPY_SEG_MIN_H": "0.5"}):
            self.assertEqual(VEG._canopy_seg_min_h(), 0.5)

    def test_canopy_rule_env_and_level_snapping(self):
        with mock.patch.dict(os.environ, {"O4_VEG_CANOPY_MIN_H": "2.2",
                                          "O4_VEG_CANOPY_MIN_CONF": "0.2"}):
            self.assertEqual(VEG._canopy_rule(), (2.0, 0.2))
        with mock.patch.dict(os.environ, {"O4_VEG_CANOPY_MIN_H": "",
                                          "O4_VEG_CANOPY_MIN_CONF": ""}):
            self.assertEqual(VEG._canopy_rule(), (VEG.CANOPY_MIN_TREE_H_M, VEG.CANOPY_MIN_CONFIDENCE))

    def test_model_dir_is_shared_beside_the_tile_caches(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("O4_CANOPY_MODEL_DIR", None)
            tile = os.path.join("root", "cache", "+50+010", "+52+013")
            self.assertEqual(
                CANOPY.default_model_dir(tile),
                os.path.join(os.path.abspath(os.path.join("root", "cache")), "_canopy_height"),
            )
            os.environ["O4_CANOPY_MODEL_DIR"] = "X:/models"
            self.assertEqual(CANOPY.default_model_dir(tile), "X:/models")

    def test_norm_unpickler_stubs_lightning_and_refuses_others(self):
        class Evil:
            def __reduce__(self):
                return (os.system, ("echo pwned",))

        data = pickle.dumps(Evil())
        with self.assertRaises(pickle.UnpicklingError):
            CANOPY._NormUnpickler(io.BytesIO(data)).load()
        stub_payload = b"cpytorch_lightning.utilities\nAttributeDict\n."
        self.assertIs(CANOPY._NormUnpickler(io.BytesIO(stub_payload)).load(), CANOPY._Stub)


@unittest.skipIf(torch is None, "torch is not installed")
class CanopyInferenceTests(unittest.TestCase):
    def test_heights_run_at_zl17_scale_with_aerial_normalisation(self):
        seen = {}

        class FakeNet(torch.nn.Module):
            def forward(self, x):
                seen["shape"] = tuple(x.shape)
                # 15 m mean height, then one exceedance probability per level.
                out = torch.full((x.shape[0], 1 + len(CANOPY.LEVELS_M), x.shape[2], x.shape[3]), 0.8)
                out[:, 0] = 15.0
                return out

        class FakeNorm(torch.nn.Module):
            def forward(self, x):
                q = torch.tensor([0.1, 0.1, 0.1, 0.6, 0.6, 0.6])
                return q.expand(x.shape[0], 6)

        model = CANOPY.CanopyModel(torch, torch.device("cpu"), FakeNet(), FakeNorm())
        img = np.random.default_rng(0).integers(0, 255, (1024, 1024, 3), dtype=np.uint8)
        with mock.patch.object(CANOPY.GPUB, "call", lambda label, fn, *a, **k: fn(*a)):
            maps = model.maps(img, 19)
        side = 1024 // 2 ** (19 - CANOPY.INFER_ZL)
        self.assertEqual(maps.shape, (1 + len(CANOPY.LEVELS_M), side, side))
        self.assertEqual(seen["shape"][1:], (3, CANOPY.PATCH, CANOPY.PATCH))
        self.assertTrue(np.allclose(maps[0].astype(np.float32), 15.0, atol=0.1))
        self.assertTrue(np.allclose(maps[1:].astype(np.float32), 0.8, atol=0.01))

    def test_load_returns_none_without_cuda(self):
        logs = []
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with tempfile.TemporaryDirectory() as d:
                self.assertIsNone(CANOPY.CanopyModel.load(d, log=logs.append))
        self.assertTrue(any("no CUDA" in line for line in logs))


if __name__ == "__main__":
    unittest.main()
