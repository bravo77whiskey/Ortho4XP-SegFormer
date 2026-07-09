"""Tests for the O4SFR_Library asset pipeline.

Covers:
- Manifest schema round-trips through build_custom_library.load_manifest()
- Generated library.txt uses the o4sfr/<region>/<bucket>/<id>.obj scheme
- Every supported region remains in the overlay region taxonomy
- License gate refuses non-CC0 / non-CC-BY entries
"""

import os
import shutil
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PIPELINE = ROOT / "scripts" / "asset_pipeline"
for path in (SRC, PIPELINE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import O4_SFR_Building_Overlay as BLD
import build_custom_library as PIPE


class ManifestRoutingTest(unittest.TestCase):
    def setUp(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML not installed")
        self.workdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def _write_region_manifest(self):
        manifest = self.workdir / "regions.yaml"
        rows = []
        for region in sorted(PIPE.VALID_REGIONS):
            rows.append(
                textwrap.dedent(
                    f"""
                    - id: sample_{region}
                      region: {region}
                      bucket: residential
                      license: CC0
                      author: Test
                      source_url: https://example.com/{region}
                      source_file: {region}/sample.fbx
                      footprint_m: [8.0, 10.0]
                    """
                ).rstrip()
            )
        manifest.write_text(
            "version: 1\nassets:\n" + "\n".join(rows),
            encoding="utf-8",
        )
        return manifest

    def test_manifest_loads_and_is_non_empty(self):
        assets = PIPE.load_manifest(str(self._write_region_manifest()))
        self.assertGreater(len(assets), 0)

    def test_manifest_regions_match_overlay_taxonomy(self):
        assets = PIPE.load_manifest(str(self._write_region_manifest()))
        valid = set(BLD.OPTIONAL_ASSET_REGION_ALIASES) | {"generic"}
        for asset in assets:
            with self.subTest(asset=asset.id):
                self.assertIn(asset.region, valid)

    def test_manifest_licenses_are_permissive(self):
        assets = PIPE.load_manifest(str(self._write_region_manifest()))
        for asset in assets:
            with self.subTest(asset=asset.id):
                self.assertTrue(
                    asset.license.startswith(("CC0", "CC-BY", "PUBLIC DOMAIN")),
                    msg=f"unexpected license: {asset.license!r}",
                )


class BuildOutputTest(unittest.TestCase):
    def setUp(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML not installed")
        self.workdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def _write_minimal_manifest(self):
        manifest = self.workdir / "manifest.yaml"
        manifest.write_text(
            textwrap.dedent(
                """
                version: 1
                assets:
                  - id: townhouse_eu_01
                    region: europe
                    bucket: residential
                    license: CC0
                    author: Kenney
                    source_url: https://kenney.nl/assets/modular-buildings
                    source_file: kenney/town_01.fbx
                    footprint_m: [8.0, 11.0]
                    height_m: 8.5
                  - id: ranch_na_01
                    region: north_america
                    bucket: residential
                    license: CC-BY-4.0
                    author: Test
                    source_url: https://example.com/ranch
                    source_file: third/ranch_01.glb
                    footprint_m: [12.0, 18.0]
                    height_m: 5.5
                """
            ).strip(),
            encoding="utf-8",
        )
        return manifest

    def _stub_obj(self, output_dir: Path, region: str, bucket: str, asset_id: str):
        path = output_dir / region / bucket / f"{asset_id}.obj"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("OBJ stub", encoding="utf-8")

    def test_build_writes_library_txt_with_export_lines(self):
        manifest = self._write_minimal_manifest()
        out = self.workdir / "out"
        out.mkdir()
        self._stub_obj(out, "europe", "residential", "townhouse_eu_01")
        self._stub_obj(out, "north_america", "residential", "ranch_na_01")
        rc = PIPE.main(["--manifest", str(manifest), "--output", str(out)])
        self.assertEqual(rc, 0)
        library_txt = (out / "library.txt").read_text(encoding="utf-8")
        self.assertIn(
            "EXPORT o4sfr/europe/residential/townhouse_eu_01.obj "
            "europe/residential/townhouse_eu_01.obj",
            library_txt,
        )
        self.assertIn(
            "EXPORT o4sfr/north_america/residential/ranch_na_01.obj "
            "north_america/residential/ranch_na_01.obj",
            library_txt,
        )
        self.assertIn("LIBRARY", library_txt.splitlines()[:5])

    def test_build_emits_license_and_attributions(self):
        manifest = self._write_minimal_manifest()
        out = self.workdir / "out"
        out.mkdir()
        self._stub_obj(out, "europe", "residential", "townhouse_eu_01")
        self._stub_obj(out, "north_america", "residential", "ranch_na_01")
        PIPE.main(["--manifest", str(manifest), "--output", str(out)])
        attributions = (out / "ATTRIBUTIONS.md").read_text(encoding="utf-8")
        self.assertIn("CC-BY-4.0", attributions)
        self.assertIn("CC0", attributions)
        self.assertIn(
            "o4sfr/europe/residential/townhouse_eu_01.obj", attributions
        )
        license_file = (out / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("ATTRIBUTIONS.md", license_file)

    def test_build_fails_on_missing_objs_without_skip(self):
        manifest = self._write_minimal_manifest()
        out = self.workdir / "out"
        out.mkdir()
        # No stub .obj files written.
        with self.assertRaises(SystemExit):
            PIPE.main(["--manifest", str(manifest), "--output", str(out)])

    def test_build_skip_missing_drops_unbuilt_entries(self):
        manifest = self._write_minimal_manifest()
        out = self.workdir / "out"
        out.mkdir()
        # Stub only the europe .obj; the north_america one stays missing.
        self._stub_obj(out, "europe", "residential", "townhouse_eu_01")
        rc = PIPE.main(
            ["--manifest", str(manifest), "--output", str(out), "--skip-missing"]
        )
        self.assertEqual(rc, 0)
        library_txt = (out / "library.txt").read_text(encoding="utf-8")
        self.assertIn("townhouse_eu_01.obj", library_txt)
        self.assertNotIn(
            "ranch_na_01.obj", library_txt,
            msg="missing .obj entries must be dropped from library.txt",
        )

    def test_build_rejects_non_open_license(self):
        manifest = self.workdir / "manifest.yaml"
        manifest.write_text(
            textwrap.dedent(
                """
                version: 1
                assets:
                  - id: payware_house
                    region: europe
                    bucket: residential
                    license: PROPRIETARY
                    author: ACME
                    source_url: https://example.com/proprietary
                    source_file: prop/house.fbx
                    footprint_m: [8.0, 10.0]
                    height_m: 6.0
                """
            ).strip(),
            encoding="utf-8",
        )
        out = self.workdir / "out"
        out.mkdir()
        with self.assertRaises(SystemExit) as ctx:
            PIPE.main(["--manifest", str(manifest), "--output", str(out)])
        self.assertIn("PROPRIETARY", str(ctx.exception))

    def test_build_rejects_duplicate_ids(self):
        manifest = self.workdir / "manifest.yaml"
        manifest.write_text(
            textwrap.dedent(
                """
                version: 1
                assets:
                  - id: house_dup
                    region: europe
                    bucket: residential
                    license: CC0
                    author: A
                    source_url: https://example.com/a
                    source_file: a.fbx
                    footprint_m: [8.0, 10.0]
                  - id: house_dup
                    region: north_america
                    bucket: residential
                    license: CC0
                    author: B
                    source_url: https://example.com/b
                    source_file: b.fbx
                    footprint_m: [9.0, 11.0]
                """
            ).strip(),
            encoding="utf-8",
        )
        with self.assertRaises(SystemExit) as ctx:
            PIPE.load_manifest(str(manifest))
        self.assertIn("Duplicate", str(ctx.exception))

    def test_manifest_decimate_and_texture_defaults(self):
        manifest = self._write_minimal_manifest()
        assets = PIPE.load_manifest(str(manifest))
        for asset in assets:
            with self.subTest(asset=asset.id):
                self.assertEqual(asset.decimate_target, 2000)
                self.assertEqual(asset.texture_max_px, 1024)

    def test_manifest_per_entry_decimate_and_texture_override(self):
        manifest = self.workdir / "manifest.yaml"
        manifest.write_text(
            textwrap.dedent(
                """
                version: 1
                assets:
                  - id: scan_house_01
                    region: europe
                    bucket: residential
                    license: CC-BY-4.0
                    author: Photogrammetrist
                    source_url: https://example.com/scan
                    source_file: scans/scan_house_01.glb
                    footprint_m: [10.0, 12.0]
                    decimate_target: 1500
                    texture_max_px: 512
                """
            ).strip(),
            encoding="utf-8",
        )
        (asset,) = PIPE.load_manifest(str(manifest))
        self.assertEqual(asset.decimate_target, 1500)
        self.assertEqual(asset.texture_max_px, 512)

    def test_manifest_rejects_decimate_target_too_low(self):
        manifest = self.workdir / "manifest.yaml"
        manifest.write_text(
            textwrap.dedent(
                """
                version: 1
                assets:
                  - id: too_low
                    region: europe
                    bucket: residential
                    license: CC0
                    author: A
                    source_url: https://example.com/
                    source_file: a.fbx
                    footprint_m: [8.0, 10.0]
                    decimate_target: 50
                """
            ).strip(),
            encoding="utf-8",
        )
        with self.assertRaises(SystemExit) as ctx:
            PIPE.load_manifest(str(manifest))
        self.assertIn("decimate_target", str(ctx.exception))

    def test_build_rejects_unknown_region(self):
        manifest = self.workdir / "manifest.yaml"
        manifest.write_text(
            textwrap.dedent(
                """
                version: 1
                assets:
                  - id: weird
                    region: atlantis
                    bucket: residential
                    license: CC0
                    author: A
                    source_url: https://example.com/
                    source_file: a.fbx
                    footprint_m: [8.0, 10.0]
                """
            ).strip(),
            encoding="utf-8",
        )
        with self.assertRaises(SystemExit) as ctx:
            PIPE.load_manifest(str(manifest))
        self.assertIn("Invalid region", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
