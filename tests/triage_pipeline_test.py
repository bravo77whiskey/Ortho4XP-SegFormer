"""Tests for the triage pipeline (extract -> probe -> manifest assembly)."""

import csv
import os
import shutil
import sys
import tempfile
import textwrap
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PIPELINE = ROOT / "scripts" / "asset_pipeline"
for path in (SRC, PIPELINE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import triage_extract as EXTRACT  # noqa: E402
import triage_probe as PROBE  # noqa: E402
import manifest_from_triage as MANIFEST  # noqa: E402
import build_custom_library as PIPE  # noqa: E402


def _write_obj_box(path: Path, width=12.0, depth=8.0, height=6.0):
    """Write a tiny axis-aligned box .obj at the given dimensions."""
    w = width / 2.0
    d = depth / 2.0
    path.write_text(
        "\n".join(
            [
                f"v {-w:.4f} 0.0 {-d:.4f}",
                f"v { w:.4f} 0.0 {-d:.4f}",
                f"v { w:.4f} 0.0 { d:.4f}",
                f"v {-w:.4f} 0.0 { d:.4f}",
                f"v {-w:.4f} {height:.4f} {-d:.4f}",
                f"v { w:.4f} {height:.4f} {-d:.4f}",
                f"v { w:.4f} {height:.4f} { d:.4f}",
                f"v {-w:.4f} {height:.4f} { d:.4f}",
                "f 1 2 3 4",
                "f 5 6 7 8",
                "f 1 2 6 5",
                "f 2 3 7 6",
                "f 3 4 8 7",
                "f 4 1 5 8",
                "",
            ]
        ),
        encoding="utf-8",
    )


class ObjBboxParserTest(unittest.TestCase):
    def test_parses_box_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj = Path(tmp) / "box.obj"
            _write_obj_box(obj, width=12.0, depth=8.0, height=6.0)
            dx, dy, dz, verts, tris = PROBE.parse_obj_bbox(obj)
            self.assertAlmostEqual(dx, 12.0, places=4)
            self.assertAlmostEqual(dy, 6.0, places=4)  # y is height in our box
            self.assertAlmostEqual(dz, 8.0, places=4)
            self.assertEqual(verts, 8)
            # 6 quads -> 12 tris
            self.assertEqual(tris, 12)

    def test_empty_obj_returns_zeros(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj = Path(tmp) / "empty.obj"
            obj.write_text("# comment only\n", encoding="utf-8")
            dx, dy, dz, verts, tris = PROBE.parse_obj_bbox(obj)
            self.assertEqual((dx, dy, dz, verts, tris), (0.0, 0.0, 0.0, 0, 0))


class ExtractWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.workdir = Path(tempfile.mkdtemp())
        self.downloads = self.workdir / "downloads"
        self.sources = self.workdir / "sources"
        self.triage = self.workdir / "triage"
        self.downloads.mkdir()
        # Build a zip mimicking a Sketchfab layout: source/<model> + textures/
        archive = self.downloads / "japanese-residential-home-01.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("source/JapaneseResidentialHome_01.fbx", b"fake fbx")
            zf.writestr("textures/HouseLayout_CLR.png",
                        b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
            zf.writestr("textures/HouseLayout_NRM.png",
                        b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        # And a zip whose source is a nested archive -- should be flagged.
        nested = self.downloads / "cottage.zip"
        with zipfile.ZipFile(nested, "w") as zf:
            zf.writestr("source/Cottage.rar", b"fake rar")
            zf.writestr("textures/Base.jpeg",
                        b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 32)

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def test_extract_records_format_and_picks_color_preview(self):
        # Patch the thumbnail step -- the test zip's PNG isn't a real image.
        original_thumb = EXTRACT.make_thumbnail
        def stub_thumb(src, dst, size_px):
            dst.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")  # JPEG header
            return True
        EXTRACT.make_thumbnail = stub_thumb
        try:
            rc = EXTRACT.main([
                "--downloads", str(self.downloads),
                "--sources", str(self.sources),
                "--triage", str(self.triage),
                "--default-license", "CC-BY-4.0",
            ])
        finally:
            EXTRACT.make_thumbnail = original_thumb
        self.assertEqual(rc, 0)
        inventory = self.triage / "inventory.csv"
        self.assertTrue(inventory.exists())
        with open(inventory, "r", encoding="utf-8") as fh:
            rows = {r["slug"]: r for r in csv.DictReader(fh)}
        self.assertIn("japanese-residential-home-01", rows)
        jrh = rows["japanese-residential-home-01"]
        self.assertEqual(jrh["format"], "fbx")
        self.assertEqual(jrh["license"], "CC-BY-4.0")
        self.assertEqual(jrh["preview_path"], "japanese-residential-home-01.jpg")
        self.assertTrue(jrh["source_url"].endswith("japanese-residential-home-01"))

        self.assertIn("cottage", rows)
        cottage = rows["cottage"]
        self.assertEqual(cottage["format"], "nested-archive")
        self.assertEqual(cottage["model_path"], "")


class FootprintHeuristicTest(unittest.TestCase):
    def test_probe_in_range_passes_through(self):
        self.assertEqual(
            MANIFEST.choose_footprint("residential", "12.0", "8.0"),
            (12.0, 8.0),
        )

    def test_probe_out_of_range_falls_back_to_residential_default(self):
        self.assertEqual(
            MANIFEST.choose_footprint("residential", "0.013", "0.008"),
            (10.0, 8.0),
        )

    def test_probe_too_large_falls_back_to_commercial_default(self):
        self.assertEqual(
            MANIFEST.choose_footprint("commercial", "1300", "800"),
            (15.0, 12.0),
        )

    def test_probe_blank_falls_back_to_industrial_default(self):
        self.assertEqual(
            MANIFEST.choose_footprint("industrial", "", ""),
            (25.0, 18.0),
        )

    def test_unknown_bucket_falls_back_to_residential(self):
        self.assertEqual(
            MANIFEST.choose_footprint("warehouse", "9999", "9999"),
            (10.0, 8.0),
        )


class LicenseGateTest(unittest.TestCase):
    def test_cc0_passes(self):
        self.assertTrue(MANIFEST.license_passes("CC0"))
    def test_cc_by_passes(self):
        self.assertTrue(MANIFEST.license_passes("CC-BY-4.0"))
    def test_lowercase_passes(self):
        self.assertTrue(MANIFEST.license_passes("cc0"))
    def test_proprietary_rejected(self):
        self.assertFalse(MANIFEST.license_passes("PROPRIETARY"))
    def test_blank_rejected(self):
        self.assertFalse(MANIFEST.license_passes(""))


class ManifestAssemblyTest(unittest.TestCase):
    def setUp(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML not installed")
        self.workdir = Path(tempfile.mkdtemp())
        self.triage = self.workdir / "triage"
        self.triage.mkdir()
        # inventory
        with open(self.triage / "inventory.csv", "w",
                  newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "slug", "archive", "format", "model_path",
                "preview_path", "author", "license", "source_url",
            ])
            writer.writeheader()
            writer.writerow({
                "slug": "japanese-residential-home-01",
                "archive": "japanese-residential-home-01.zip",
                "format": "fbx",
                "model_path": "japanese-residential-home-01/source/JRH.fbx",
                "preview_path": "japanese-residential-home-01.jpg",
                "author": "Test",
                "license": "CC-BY-4.0",
                "source_url": "https://sketchfab.com/3d-models/japanese-residential-home-01",
            })
            writer.writerow({
                "slug": "ranch_na",
                "archive": "ranch_na.zip",
                "format": "obj",
                "model_path": "ranch_na/source/ranch.obj",
                "preview_path": "ranch_na.jpg",
                "author": "Test",
                "license": "CC0",
                "source_url": "https://sketchfab.com/3d-models/ranch-na",
            })
            writer.writerow({
                "slug": "payware_house",
                "archive": "payware_house.zip",
                "format": "fbx",
                "model_path": "payware_house/source/p.fbx",
                "preview_path": "payware_house.jpg",
                "author": "ACME",
                "license": "PROPRIETARY",
                "source_url": "https://example.com/proprietary",
            })
        # probe
        with open(self.triage / "probe.csv", "w",
                  newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "slug", "format", "bbox_x", "bbox_y", "bbox_z",
                "vertex_count", "triangle_count", "error",
            ])
            writer.writeheader()
            writer.writerow({
                "slug": "japanese-residential-home-01",
                "format": "fbx",
                "bbox_x": "9.0", "bbox_y": "12.0", "bbox_z": "6.0",
                "vertex_count": "1200", "triangle_count": "1800",
                "error": "",
            })
            writer.writerow({
                "slug": "ranch_na",
                "format": "obj",
                "bbox_x": "20.0", "bbox_y": "14.0", "bbox_z": "5.0",
                "vertex_count": "80000", "triangle_count": "150000",
                "error": "",
            })
            writer.writerow({
                "slug": "payware_house",
                "format": "fbx",
                "bbox_x": "", "bbox_y": "", "bbox_z": "",
                "vertex_count": "", "triangle_count": "",
                "error": "skipped",
            })
        # decisions
        (self.triage / "decisions.yaml").write_text(
            textwrap.dedent(
                """
                version: 1
                decisions:
                  - slug: japanese-residential-home-01
                    region: asia
                    bucket: residential
                    confidence: high
                    notes: traditional minka roof + shoji
                  - slug: ranch_na
                    region: north_america
                    bucket: residential
                    confidence: high
                  - slug: payware_house
                    region: europe
                    bucket: residential
                    confidence: high
                """
            ).strip(),
            encoding="utf-8",
        )
        self.manifest_out = self.workdir / "sources.yaml"

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def test_assembles_valid_manifest_and_drops_proprietary(self):
        rc = MANIFEST.main([
            "--triage", str(self.triage),
            "--output", str(self.manifest_out),
        ])
        self.assertEqual(rc, 0)
        assets = PIPE.load_manifest(str(self.manifest_out))
        ids = {a.id for a in assets}
        # Proprietary entry dropped, both CC entries pass through.
        self.assertEqual(
            ids,
            {"japanese-residential-home-01", "ranch_na"},
        )
        by_id = {a.id: a for a in assets}
        # Probe in range -> trusted as footprint.
        self.assertEqual(
            tuple(by_id["japanese-residential-home-01"].footprint_m),
            (9.0, 12.0),
        )
        # High-poly photogrammetry gets decimate target.
        self.assertEqual(by_id["ranch_na"].decimate_target, 2000)
        # Low-poly minka keeps the default decimate target.
        self.assertEqual(
            by_id["japanese-residential-home-01"].decimate_target, 2000
        )

    def test_include_unknown_coerces_license(self):
        rc = MANIFEST.main([
            "--triage", str(self.triage),
            "--output", str(self.manifest_out),
            "--include-unknown",
        ])
        self.assertEqual(rc, 0)
        assets = PIPE.load_manifest(str(self.manifest_out))
        ids = {a.id for a in assets}
        self.assertIn("payware_house", ids)


if __name__ == "__main__":
    unittest.main()
