"""Finding a tile's DSF in the overlay source / Global Scenery directories."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_Overlay_Utils as OVL  # noqa: E402


def _make_dsf(root):
    path = Path(root) / "Earth nav data" / "+20+120" / "+22+120.dsf"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"dsf")
    return path


class FindSourceDsfTests(unittest.TestCase):
    def test_prefers_custom_overlay_src(self):
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as alt:
            expected = _make_dsf(main)
            _make_dsf(alt)
            with mock.patch.multiple(OVL, custom_overlay_src=main,
                                     custom_overlay_src_alternate=alt):
                path, report = OVL.find_source_dsf(22, 120)
        self.assertEqual(Path(path), expected)
        self.assertIsNone(report)

    def test_falls_back_to_the_alternate(self):
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as alt:
            expected = _make_dsf(alt)
            with mock.patch.multiple(OVL, custom_overlay_src=main,
                                     custom_overlay_src_alternate=alt):
                path, _ = OVL.find_source_dsf(22, 120)
        self.assertEqual(Path(path), expected)

    def test_report_names_each_setting_and_path(self):
        with tempfile.TemporaryDirectory() as main:
            with mock.patch.multiple(OVL, custom_overlay_src=main,
                                     custom_overlay_src_alternate=""):
                path, report = OVL.find_source_dsf(22, 120)
        self.assertIsNone(path)
        self.assertIn(f"custom_overlay_src = {main!r}", report)
        self.assertIn("custom_overlay_src_alternate = ''", report)
        self.assertIn("+22+120.dsf: not found", report)

    def test_report_keeps_the_os_error(self):
        with tempfile.TemporaryDirectory() as main:
            error = OSError(1392, "The file or directory is corrupted and unreadable")
            with mock.patch.multiple(OVL, custom_overlay_src=main,
                                     custom_overlay_src_alternate=""), \
                    mock.patch.object(OVL.os, "stat", side_effect=error):
                path, report = OVL.find_source_dsf(22, 120)
        self.assertIsNone(path)
        self.assertIn("corrupted and unreadable", report)


if __name__ == "__main__":
    unittest.main()
