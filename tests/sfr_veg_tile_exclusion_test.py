"""The tile-wide forest exclusion written into the generated vegetation DSF."""

import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_SFR_Vegetation_Overlay as VEG


def _dsf(properties):
    """A binary DSF carrying a HEAD/PROP atom plus filler.

    The filler only exists so these fixtures clear the validator's
    minimum-size floor; a real compiled tile is orders of magnitude larger.
    """
    prop = b"".join(
        f"{key}\0{value}\0".encode("utf-8") for key, value in properties
    )
    head = b"PORP" + struct.pack("<I", 8 + len(prop)) + prop
    filler = b"SDMC" + struct.pack("<I", 8 + 128) + bytes(128)
    body = (
        b"XPLNEDSF"
        + struct.pack("<I", 1)
        + b"DAEH"
        + struct.pack("<I", 8 + len(head))
        + head
        + filler
    )
    return body + hashlib.md5(body).digest()


class ExclusionValueTests(unittest.TestCase):
    def test_zurich(self):
        self.assertEqual(VEG.tile_exclusion_property_value(47, 8), "8/47/9/48")

    def test_negative_latitude_and_longitude(self):
        self.assertEqual(
            VEG.tile_exclusion_property_value(-34, -71), "-71/-34/-70/-33"
        )

    def test_fractional_coordinates_floor_to_the_containing_tile(self):
        # int() would truncate toward zero and name the wrong degree square.
        self.assertEqual(
            VEG.tile_exclusion_property_value(-33.8, -70.2), "-71/-34/-70/-33"
        )
        self.assertEqual(
            VEG.tile_exclusion_property_value(47.9, 8.9), "8/47/9/48"
        )

    def test_the_value_round_trips_through_the_overlay_rect_parser(self):
        ring = VEG._parse_exclusion_rect(
            VEG.tile_exclusion_property_value(-34, -71)
        )
        self.assertEqual(ring, [(-71.0, -34.0), (-70.0, -34.0),
                                (-70.0, -33.0), (-71.0, -33.0)])


class OverlapConflictTests(unittest.TestCase):
    def test_replacement_forces_forest_overlap_avoidance_off(self):
        self.assertFalse(
            VEG.effective_avoid_simheaven_forests(True, True)
        )

    def test_the_saved_setting_resumes_when_replacement_is_off(self):
        self.assertTrue(VEG.effective_avoid_simheaven_forests(True, False))
        self.assertFalse(VEG.effective_avoid_simheaven_forests(False, False))


class CompiledOverlayValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "+47+008.dsf"

    def _write(self, properties):
        self.path.write_bytes(_dsf(properties))
        return str(self.path)

    def test_an_overlay_carrying_the_expected_exclusion_passes(self):
        path = self._write([
            ("sim/overlay", "1"), ("sim/exclude_for", "8/47/9/48"),
        ])
        self.assertIsNone(VEG.validate_compiled_overlay(path, "8/47/9/48"))

    def test_a_missing_exclusion_is_rejected(self):
        path = self._write([("sim/overlay", "1")])
        self.assertIn("missing", VEG.validate_compiled_overlay(path, "8/47/9/48"))

    def test_a_wrong_exclusion_is_rejected(self):
        path = self._write([
            ("sim/overlay", "1"), ("sim/exclude_for", "7/46/8/47"),
        ])
        problem = VEG.validate_compiled_overlay(path, "8/47/9/48")
        self.assertIn("expected 8/47/9/48", problem)

    def test_an_unexpected_exclusion_is_rejected(self):
        path = self._write([
            ("sim/overlay", "1"), ("sim/exclude_for", "8/47/9/48"),
        ])
        self.assertIn("unexpected", VEG.validate_compiled_overlay(path, None))

    def test_a_non_overlay_dsf_is_rejected(self):
        path = self._write([("sim/overlay", "0")])
        self.assertIn("sim/overlay", VEG.validate_compiled_overlay(path, None))

    def test_a_truncated_file_is_rejected(self):
        self.path.write_bytes(b"XPLNEDSF")
        self.assertIn("bytes", VEG.validate_compiled_overlay(str(self.path), None))

    def test_a_missing_file_is_rejected(self):
        self.assertIn(
            "missing",
            VEG.validate_compiled_overlay(
                str(self.path.parent / "absent.dsf"), None
            ),
        )


class GenerationStatusTests(unittest.TestCase):
    def test_status_is_written_atomically_and_reads_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "nested" / "status.json")
            VEG.write_generation_status(path, {"outcome": "ok", "polygons": 12})
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle),
                                 {"outcome": "ok", "polygons": 12})
            self.assertFalse(Path(path + ".tmp").exists())

    def test_no_status_path_is_a_no_op(self):
        VEG.write_generation_status(None, {"outcome": "ok"})

    def test_an_unwritable_path_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "blocker"
            blocker.write_text("not a directory", encoding="utf-8")
            VEG.write_generation_status(str(blocker / "status.json"),
                                        {"outcome": "ok"})


class OverlayWriterTests(unittest.TestCase):
    """The DSF text the overlay hands to DSFTool."""

    def _source(self):
        return (SRC / "O4_SFR_Vegetation_Overlay.py").read_text(encoding="utf-8")

    def test_the_exclusion_is_written_only_when_replacement_is_on(self):
        source = self._source()
        self.assertIn("if exclude_lower_forests:\n", source)
        self.assertIn(
            'f"PROPERTY {FOREST_EXCLUSION_PROPERTY_KEYS[0]} {exclusion_value}',
            source,
        )

    def test_compilation_goes_to_a_temporary_file_first(self):
        source = self._source()
        self.assertIn("tmp_dsf = out_dsf + '.tmp'", source)
        self.assertIn("SEGFORMER.compile_dsf(txt_path, tmp_dsf)", source)
        self.assertIn("os.replace(tmp_dsf, out_dsf)", source)

    def test_an_empty_result_drops_a_stale_overlay_and_asks_for_reconciliation(self):
        source = self._source()
        self.assertIn("status['outcome'] = 'empty'", source)
        self.assertIn("status['needs_reconcile'] = True", source)


if __name__ == "__main__":
    unittest.main()
