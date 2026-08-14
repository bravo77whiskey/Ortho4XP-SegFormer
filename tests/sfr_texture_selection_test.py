"""Config-driven texture selection for the SFR overlays.

A tile's textures/ directory accumulates files from every build it has ever
had, so the same footprint and zoomlevel can exist under several providers,
and a zone_list boundary can legitimately split one footprint between two.
These tests pin that the tile config decides which files are read and which
part of each one is theirs.
"""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_SFR_Texture_Selection as TEXSEL


LAT, LON = 12, 34

BASE_CFG = {
    "default_website": "BI",
    "default_zl": "16",
    "mesh_zl": "19",
    "cover_airports_with_highres": "False",
    "cover_zl": "18",
    "cover_extent": "1.0",
}


def _write_tile(tmp, cfg_values, texture_names=()):
    """Create a tile directory with a config and the given texture files."""
    tile_dir = Path(tmp) / f"zOrtho4XP_{LAT:+03d}{LON:+04d}"
    tex_dir = tile_dir / "textures"
    tex_dir.mkdir(parents=True)
    cfg = dict(BASE_CFG)
    cfg.update(cfg_values)
    (tile_dir / f"Ortho4XP_{LAT:+03d}{LON:+04d}.cfg").write_text(
        "\n".join(f"{key}={value}" for key, value in cfg.items()) + "\n",
        encoding="utf-8",
    )
    for name in texture_names:
        (tex_dir / name).write_bytes(b"")
    return str(tex_dir)


def _config_named_textures(tex_dir, cfg_values=None):
    """Names the config asks for, so a fixture can write exactly those."""
    assignment, _mesh_zl, _notes = TEXSEL.cfg_texture_assignment(tex_dir, LAT, LON)
    return sorted(TEXSEL.texture_name(*key) for key in assignment)


def _owned_area(name, excluded):
    return 1.0 - sum(
        (x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in excluded.get(name, ())
    )


def _half_tile_zone(zl, provider):
    """A zone covering the western half of the tile."""
    return [
        [
            LAT, LON,
            LAT, LON + 0.5,
            LAT + 1, LON + 0.5,
            LAT + 1, LON,
            LAT, LON,
        ],
        zl,
        provider,
    ]


class CfgTextureSelectionTest(unittest.TestCase):
    def test_leftover_provider_textures_are_dropped(self):
        # A rebuild under a new provider leaves the old files behind. The
        # config names only BI16, so the GO2 leftovers must not be inferenced
        # -- and must not displace BI16 the way filename-only resolution does,
        # which prefers any higher-ZL file it finds.
        with tempfile.TemporaryDirectory() as tmp:
            probe = _write_tile(tmp, {})
            wanted = _config_named_textures(probe)
            self.assertTrue(wanted)

        leftovers = []
        for name in wanted[:4]:
            match = TEXSEL.TEXTURE_NAME_RE.match(name)
            til_y, til_x = int(match.group(1)), int(match.group(2))
            leftovers.append(TEXSEL.texture_name(til_y, til_x, "GO2", 16))
            for child_y, child_x in (
                (2 * til_y, 2 * til_x), (2 * til_y, 2 * til_x + 16),
                (2 * til_y + 16, 2 * til_x), (2 * til_y + 16, 2 * til_x + 16),
            ):
                leftovers.append(TEXSEL.texture_name(child_y, child_x, "GO2", 17))

        with tempfile.TemporaryDirectory() as tmp:
            tex_dir = _write_tile(tmp, {}, wanted + leftovers)
            files, excluded, report = TEXSEL.select_textures(
                wanted + leftovers, tex_dir, LAT, LON
            )

        self.assertEqual(files, wanted)
        self.assertEqual(report["skipped"], len(leftovers))
        self.assertEqual(excluded, {})

    def test_zone_boundary_splits_a_footprint_between_providers(self):
        # Where the zone edge cuts through a footprint both providers get a
        # texture, and each owns only its side.
        zone = _half_tile_zone(16, "Arc")
        cfg = {"zone_list": repr([zone])}
        with tempfile.TemporaryDirectory() as tmp:
            probe = _write_tile(tmp, cfg)
            wanted = _config_named_textures(probe)
        with tempfile.TemporaryDirectory() as tmp:
            tex_dir = _write_tile(tmp, cfg, wanted)
            files, excluded, _report = TEXSEL.select_textures(
                wanted, tex_dir, LAT, LON
            )

        by_footprint = {}
        for name in files:
            match = TEXSEL.TEXTURE_NAME_RE.match(name)
            key = (match.group(1), match.group(2), match.group(4))
            by_footprint.setdefault(key, []).append(name)
        split = [group for group in by_footprint.values() if len(group) > 1]
        self.assertTrue(split, "the zone edge should split at least one footprint")
        for group in split:
            self.assertEqual(len(group), 2)
            self.assertAlmostEqual(
                sum(_owned_area(name, excluded) for name in group), 1.0, places=9
            )
            for name in group:
                self.assertGreater(_owned_area(name, excluded), 0.0)
                self.assertLess(_owned_area(name, excluded), 1.0)

    def test_higher_zl_zone_is_excluded_from_the_lower_zl_texture(self):
        zone = _half_tile_zone(18, "BI")
        cfg = {"zone_list": repr([zone])}
        with tempfile.TemporaryDirectory() as tmp:
            probe = _write_tile(tmp, cfg)
            wanted = _config_named_textures(probe)
        with tempfile.TemporaryDirectory() as tmp:
            tex_dir = _write_tile(tmp, cfg, wanted)
            files, excluded, _report = TEXSEL.select_textures(
                wanted, tex_dir, LAT, LON
            )

        zls = {int(TEXSEL.TEXTURE_NAME_RE.match(n).group(4)) for n in files}
        self.assertEqual(zls, {16, 18})
        partial_16 = [
            name for name in excluded
            if TEXSEL.TEXTURE_NAME_RE.match(name).group(4) == "16"
        ]
        self.assertTrue(partial_16, "ZL16 textures over the zone must be masked")
        for name in excluded:
            self.assertEqual(
                TEXSEL.TEXTURE_NAME_RE.match(name).group(4), "16",
                "only the lower-ZL side of a cross-ZL overlap is excluded",
            )

    def test_shared_footprint_never_leaves_a_region_to_both_providers(self):
        # Footprints on the tile border reach into the neighbouring tile, where
        # the config assigns nothing; that strip must still go to one texture.
        zone = _half_tile_zone(16, "Arc")
        cfg = {"zone_list": repr([zone])}
        with tempfile.TemporaryDirectory() as tmp:
            probe = _write_tile(tmp, cfg)
            wanted = _config_named_textures(probe)
        with tempfile.TemporaryDirectory() as tmp:
            tex_dir = _write_tile(tmp, cfg, wanted)
            files, excluded, _report = TEXSEL.select_textures(
                wanted, tex_dir, LAT, LON
            )

        by_footprint = {}
        for name in files:
            match = TEXSEL.TEXTURE_NAME_RE.match(name)
            key = (match.group(1), match.group(2), match.group(4))
            by_footprint.setdefault(key, []).append(name)
        for key, group in by_footprint.items():
            if len(group) < 2:
                continue
            self.assertAlmostEqual(
                sum(_owned_area(name, excluded) for name in group), 1.0, places=9,
                msg=f"footprint {key} is owned {group} times over",
            )

    def test_config_named_texture_missing_from_disk_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            probe = _write_tile(tmp, {})
            wanted = _config_named_textures(probe)
        present = wanted[1:]
        with tempfile.TemporaryDirectory() as tmp:
            tex_dir = _write_tile(tmp, {}, present)
            files, _excluded, report = TEXSEL.select_textures(
                present, tex_dir, LAT, LON
            )

        self.assertEqual(files, present)
        self.assertIn(wanted[0], report["missing"])


class NoConfigFallbackTest(unittest.TestCase):
    def test_filename_resolution_is_used_without_a_config(self):
        # Loose texture directories keep the old behaviour: the highest
        # zoomlevel available wins, and partially covered lower-ZL textures
        # are masked where a kept higher-ZL one overlaps them.
        names = [
            "112256_218048_BI16.dds",
            "224512_436096_BI17.dds",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tex_dir = Path(tmp) / "textures"
            tex_dir.mkdir()
            files, excluded, report = TEXSEL.select_textures(
                names, str(tex_dir), LAT, LON
            )

        self.assertIsNone(report["cfg_path"])
        self.assertEqual(files, sorted(names))
        self.assertEqual(
            excluded["112256_218048_BI16.dds"], ((0.0, 0.0, 0.5, 0.5),)
        )

    def test_config_naming_nothing_on_disk_falls_back(self):
        # A config for a different tile position must not select nothing.
        names = ["112256_218048_BI16.dds"]
        with tempfile.TemporaryDirectory() as tmp:
            tex_dir = _write_tile(tmp, {}, names)
            files, _excluded, report = TEXSEL.select_textures(
                names, tex_dir, LAT, LON
            )

        self.assertIsNone(report["cfg_path"])
        self.assertEqual(files, names)
        self.assertTrue(report["notes"])


class SharedHelperTest(unittest.TestCase):
    def test_building_overlay_still_exports_compute_covered_fractions(self):
        import O4_SFR_Building_Overlay as BLD

        self.assertIs(BLD.compute_covered_fractions, TEXSEL.compute_covered_fractions)


if __name__ == "__main__":
    unittest.main()
