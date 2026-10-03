"""Round-trip the DSF edits through DSFTool, on a real compiled tile.

The other tests assert against hand-built atoms, which proves the reader and
writer agree with each other. This one compiles a DSF with the same tool
X-Plane's ecosystem uses, edits it, and decodes both versions back to text, so a
disagreement between this module and the real format shows up as a diff rather
than as a passing test.

Skipped when DSFTool is not on this machine.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_DSF_Edit as DSFEDIT
import O4_Veg_Replacement as CLEANUP


def _dsftool():
    if sys.platform.startswith("win"):
        candidate = ROOT / "Utils" / "win" / "DSFTool.exe"
    elif sys.platform == "darwin":
        candidate = ROOT / "Utils" / "mac" / "DSFTool"
    else:
        candidate = ROOT / "Utils" / "lin" / "DSFTool"
    if candidate.is_file():
        return str(candidate)
    return shutil.which("DSFTool")


DSFTOOL = _dsftool()

# A minimal but genuine overlay: two object definitions, one forest, and one
# placement of each, over tile +47+008.
OVERLAY_TEXT = """PROPERTY sim/planet earth
PROPERTY sim/overlay 1
PROPERTY sim/west  8
PROPERTY sim/east  9
PROPERTY sim/south 47
PROPERTY sim/north 48

OBJECT_DEF trees/oak.obj
OBJECT_DEF buildings/hangar.obj
POLYGON_DEF forests/mixed.for

OBJECT 0 8.25 47.25 0.000000
OBJECT 1 8.75 47.75 90.000000

BEGIN_POLYGON 0 100 2
BEGIN_WINDING
POLYGON_POINT 8.10 47.10
POLYGON_POINT 8.20 47.10
POLYGON_POINT 8.20 47.20
END_WINDING
END_POLYGON
"""


def _compile(text_path, dsf_path):
    result = subprocess.run(
        [DSFTOOL, "-text2dsf", str(text_path), str(dsf_path)],
        capture_output=True, text=True, timeout=120,
    )
    return result


def _decode(dsf_path, text_path):
    result = subprocess.run(
        [DSFTOOL, "--dsf2text", str(dsf_path), str(text_path)],
        capture_output=True, text=True, timeout=120,
    )
    return result


def _significant_lines(text):
    """Drop DSFTool's own comments and blank lines from a disassembly."""
    return [
        line.strip() for line in text.splitlines()
        if line.strip() and not line.startswith(("#", "I", "800", "1000"))
    ]


@unittest.skipUnless(DSFTOOL and os.path.isfile(DSFTOOL),
                     "DSFTool is not available on this machine")
class DsfToolRoundTripTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        text = self.base / "source.txt"
        text.write_text(OVERLAY_TEXT, encoding="utf-8")
        self.dsf = self.base / "+47+008.dsf"
        result = _compile(text, self.dsf)
        if result.returncode != 0 or not self.dsf.is_file():
            self.skipTest(f"DSFTool could not compile the fixture: "
                          f"{result.stdout}{result.stderr}")
        self.original = self.dsf.read_bytes()

    def _decode_text(self, path):
        out = self.base / (Path(path).stem + "_decoded.txt")
        result = _decode(path, out)
        self.assertEqual(result.returncode, 0,
                         f"{result.stdout}{result.stderr}")
        return out.read_text(encoding="utf-8", errors="ignore")

    def test_reading_a_real_dsf_agrees_with_its_disassembly(self):
        parsed = DSFEDIT.DsfFile(self.original)
        self.assertEqual(parsed.definitions("object"),
                         ["trees/oak.obj", "buildings/hangar.obj"])
        self.assertEqual(parsed.definitions("polygon"), ["forests/mixed.for"])
        self.assertIn(("sim/overlay", "1"), parsed.properties())

    def test_an_untouched_rewrite_is_byte_identical(self):
        self.assertEqual(DSFEDIT.DsfFile(self.original).to_bytes(),
                         self.original)

    def test_dsftool_reads_back_an_added_exclusion(self):
        parsed = DSFEDIT.DsfFile(self.original)
        parsed.set_properties(DSFEDIT.upsert_property(
            parsed.properties(), CLEANUP.EXCLUSION_PROPERTY, "8/47/9/48"))
        self.dsf.write_bytes(parsed.to_bytes())
        decoded = self._decode_text(self.dsf)
        self.assertIn("PROPERTY sim/exclude_for 8/47/9/48", decoded)

    def test_dsftool_reads_back_a_repointed_definition(self):
        parsed = DSFEDIT.DsfFile(self.original)
        parsed.set_definitions(
            "object", ["o4xp_veg_replacement/empty_v1.obj",
                       "buildings/hangar.obj"])
        self.dsf.write_bytes(parsed.to_bytes())
        decoded = self._decode_text(self.dsf)
        self.assertIn("OBJECT_DEF o4xp_veg_replacement/empty_v1.obj", decoded)
        self.assertIn("OBJECT_DEF buildings/hangar.obj", decoded)
        self.assertNotIn("trees/oak.obj", decoded)

    def test_only_the_planned_lines_differ_in_the_disassembly(self):
        before = self._significant(self.original)
        parsed = DSFEDIT.DsfFile(self.original)
        parsed.set_definitions(
            "object", ["o4xp_veg_replacement/empty_v1.obj",
                       "buildings/hangar.obj"])
        parsed.set_properties(DSFEDIT.upsert_property(
            parsed.properties(), CLEANUP.EXCLUSION_PROPERTY, "8/47/9/48"))
        after = self._significant(parsed.to_bytes())

        added = [line for line in after if line not in before]
        removed = [line for line in before if line not in after]
        self.assertEqual(sorted(removed), ["OBJECT_DEF trees/oak.obj"])
        self.assertEqual(
            sorted(added),
            ["OBJECT_DEF o4xp_veg_replacement/empty_v1.obj",
             "PROPERTY sim/exclude_for 8/47/9/48"],
        )
        # Every placement record survives, coordinates and all.
        placements = [line for line in before if line.startswith(
            ("OBJECT ", "POLYGON_POINT", "BEGIN_POLYGON", "END_POLYGON",
             "BEGIN_WINDING", "END_WINDING"))]
        self.assertTrue(placements)
        for line in placements:
            self.assertIn(line, after)

    def _significant(self, data):
        path = self.base / "variant.dsf"
        path.write_bytes(data)
        return _significant_lines(self._decode_text(path))

    def test_the_usage_counter_matches_the_placements_in_the_text(self):
        counts = DSFEDIT.count_definition_usage(self.original)
        self.assertEqual(counts["object"].get(0), 1)
        self.assertEqual(counts["object"].get(1), 1)
        self.assertEqual(counts["polygon"].get(0), 1)


if __name__ == "__main__":
    unittest.main()
