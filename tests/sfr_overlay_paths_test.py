import os
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_SFR_Pipeline as SFR
import O4_File_Names as FNAMES


class SfrOverlayPathTests(unittest.TestCase):
    def test_building_and_vegetation_overlays_use_separate_folders(self):
        data_dir = str(ROOT)
        with mock.patch.object(SFR, "_data_dir", data_dir):
            bld_path = SFR._dsf_output_path(38, -122, "yOrtho4XP_Bld_Overlays")
            veg_path = SFR._dsf_output_path(38, -122, "yOrtho4XP_Veg_Overlays")

        self.assertEqual(
            bld_path,
            os.path.join(
                data_dir,
                "yOrtho4XP_Bld_Overlays",
                "Earth nav data",
                "+30-130",
                "+38-122.dsf",
            ),
        )
        self.assertEqual(
            veg_path,
            os.path.join(
                data_dir,
                "yOrtho4XP_Veg_Overlays",
                "Earth nav data",
                "+30-130",
                "+38-122.dsf",
            ),
        )

    def test_file_names_exposes_building_overlay_dir(self):
        self.assertTrue(
            FNAMES.SFR_Bld_Overlay_dir.endswith(
                os.path.join("yOrtho4XP_Bld_Overlays")
            )
        )


if __name__ == "__main__":
    unittest.main()
