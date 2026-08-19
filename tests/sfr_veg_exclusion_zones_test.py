"""Custom-scenery forest exclusion zones honoured by the vegetation overlay."""

import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_SFR_Vegetation_Overlay as VEG


# Only the presence of Earth nav data/apt.dat marks a pack as an airport,
# so a marker line is enough here.
_APT_DAT = "I 1100 Generated test apt.dat"


def _dsf_bytes(properties):
    """Build a minimal binary DSF carrying only a HEAD/PROP atom."""
    prop = b"".join(
        f"{key}\0{value}\0".encode("utf-8") for key, value in properties
    )
    head = b"PORP" + struct.pack("<I", 8 + len(prop)) + prop
    return (
        b"XPLNEDSF"
        + struct.pack("<I", 1)
        + b"DAEH"
        + struct.pack("<I", 8 + len(head))
        + head
        # A second atom after HEAD, so a reader that keeps walking still stops
        # at the right place.
        + b"NFED"
        + struct.pack("<I", 8)
        + b"\0" * 16  # MD5 footer
    )


def _text_dsf(properties):
    lines = ["PROPERTY sim/planet earth", "PROPERTY sim/overlay 1"]
    lines += [f"PROPERTY {key} {value}" for key, value in properties]
    return "\n".join(lines) + "\n"


# One WED exclusion rectangle with forest + object + facade ticked, plus a
# second rectangle that excludes objects only.
MIXED_PROPERTIES = (
    ("sim/planet", "earth"),
    ("sim/overlay", "1"),
    ("sim/exclude_for", "12.20/50.10/12.30/50.20"),
    ("sim/exclude_obj", "12.20/50.10/12.30/50.20"),
    ("sim/exclude_fac", "12.20/50.10/12.30/50.20"),
    ("sim/exclude_obj", "12.60/50.60/12.70/50.70"),
    ("sim/exclude_net", "12.60/50.60/12.70/50.70"),
)


class ExclusionRectParsingTests(unittest.TestCase):
    def test_rect_parses_as_west_south_east_north_ring(self):
        ring = VEG._parse_exclusion_rect("12.2/50.1/12.3/50.2")
        self.assertEqual(
            ring,
            [(12.2, 50.1), (12.3, 50.1), (12.3, 50.2), (12.2, 50.2)],
        )

    def test_malformed_and_degenerate_rects_are_dropped(self):
        for value in ("", "1/2/3", "a/b/c/d", "12.2/50.1/12.2/50.2",
                      "12.2/50.1/12.3/50.1"):
            self.assertIsNone(VEG._parse_exclusion_rect(value), value)

    def test_mixed_zone_keeps_only_the_forest_type(self):
        zones = VEG._zones_from_property_pairs(MIXED_PROPERTIES)
        self.assertEqual(
            sorted(zones),
            ["sim/exclude_fac", "sim/exclude_for",
             "sim/exclude_net", "sim/exclude_obj"],
        )
        self.assertEqual(len(zones["sim/exclude_for"]), 1)
        self.assertEqual(len(zones["sim/exclude_obj"]), 2)
        self.assertEqual(VEG.FOREST_EXCLUSION_PROPERTY_KEYS, ("sim/exclude_for",))


class DsfPropertyReaderTests(unittest.TestCase):
    def test_binary_header_reader_returns_property_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dsf_path = Path(tmpdir) / "+50+012.dsf"
            dsf_path.write_bytes(_dsf_bytes(MIXED_PROPERTIES))
            pairs = VEG._read_dsf_binary_properties(str(dsf_path))
        self.assertEqual(pairs, list(MIXED_PROPERTIES))

    def test_binary_reader_returns_none_for_non_dsf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            packed = Path(tmpdir) / "+50+012.dsf"
            packed.write_bytes(b"7z\xbc\xaf\x27\x1c" + b"\0" * 64)
            self.assertIsNone(VEG._read_dsf_binary_properties(str(packed)))

    def test_text_reader_matches_binary_reader(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            text_path = Path(tmpdir) / "dsf.txt"
            text_path.write_text(_text_dsf(MIXED_PROPERTIES), encoding="utf-8")
            pairs = VEG._read_dsf_text_properties(str(text_path))
        self.assertEqual(
            VEG._zones_from_property_pairs(pairs),
            VEG._zones_from_property_pairs(MIXED_PROPERTIES),
        )


class ForestExclusionLoaderTests(unittest.TestCase):
    def _custom_scenery(self, tmpdir):
        custom = Path(tmpdir) / "Custom Scenery"
        packs = {
            "Airport Pack": MIXED_PROPERTIES,
            "Object Only Pack": (("sim/exclude_obj", "13.0/51.0/13.1/51.1"),),
            # Same rectangle as Airport Pack — must not be counted twice.
            "Duplicate Pack": (("sim/exclude_for", "12.20/50.10/12.30/50.20"),),
            "yOrtho4XP_Veg_Overlays": (("sim/exclude_for", "12.0/50.0/12.9/50.9"),),
        }
        for name, properties in packs.items():
            nav = custom / name / "Earth nav data" / "+50+012"
            nav.mkdir(parents=True)
            (nav / "+50+012.dsf").write_bytes(_dsf_bytes(properties))
            # These stand in for custom airports; only those are consulted.
            (nav.parent / "apt.dat").write_text(_APT_DAT, encoding="utf-8")
        (custom / "scenery_packs.ini").write_text(
            "\n".join(
                f"SCENERY_PACK Custom Scenery/{name}/" for name in packs
            ),
            encoding="utf-8",
        )
        return custom

    def test_forest_zones_deduped_and_output_pack_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            custom = self._custom_scenery(tmpdir)
            out_dsf = (custom / "yOrtho4XP_Veg_Overlays" / "Earth nav data"
                       / "+50+012" / "+50+012.dsf")
            rings, type_counts = VEG._load_custom_scenery_forest_exclusions(
                str(custom), 50, 12, str(out_dsf), "dsftool-not-used", tmpdir
            )

        self.assertEqual(len(rings), 1)
        self.assertEqual(
            rings[0],
            [(12.2, 50.1), (12.3, 50.1), (12.3, 50.2), (12.2, 50.2)],
        )
        # The mixed zone's other types are reported but do not exclude trees.
        self.assertEqual(type_counts["sim/exclude_for"], 2)
        self.assertEqual(type_counts["sim/exclude_obj"], 3)
        self.assertEqual(type_counts["sim/exclude_fac"], 1)

    def test_no_zones_when_only_other_types_are_excluded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            custom = Path(tmpdir) / "Custom Scenery"
            nav = custom / "Roads Pack" / "Earth nav data" / "+50+012"
            nav.mkdir(parents=True)
            (nav / "+50+012.dsf").write_bytes(
                _dsf_bytes((("sim/exclude_net", "12.2/50.1/12.3/50.2"),))
            )
            (nav.parent / "apt.dat").write_text(_APT_DAT, encoding="utf-8")
            (custom / "scenery_packs.ini").write_text(
                "SCENERY_PACK Custom Scenery/Roads Pack/\n", encoding="utf-8"
            )
            rings, type_counts = VEG._load_custom_scenery_forest_exclusions(
                str(custom), 50, 12, "", "dsftool-not-used", tmpdir
            )

        self.assertEqual(rings, [])
        self.assertEqual(type_counts, {"sim/exclude_net": 1})


class AirportPackOnlyTests(unittest.TestCase):
    """Only custom airports may veto vegetation — not regional/global scenery."""

    WHOLE_TILE = (("sim/exclude_for", "12.0/50.0/13.0/51.0"),)

    def _pack(self, custom, name, properties, apt_dat, n_dsf=1):
        nav = custom / name / "Earth nav data"
        (nav / "+50+012").mkdir(parents=True)
        (nav / "+50+012" / "+50+012.dsf").write_bytes(_dsf_bytes(properties))
        # Extra tiles decide regional-vs-airport; their contents are irrelevant.
        for i in range(1, n_dsf):
            grid = nav / "+50+013"
            grid.mkdir(parents=True, exist_ok=True)
            (grid / f"+50+{13 + i:03d}.dsf").write_bytes(_dsf_bytes(()))
        if apt_dat:
            (nav / "apt.dat").write_text(_APT_DAT, encoding="utf-8")

    def _load(self, tmpdir, name, properties, apt_dat, n_dsf=1):
        custom = Path(tmpdir) / "Custom Scenery"
        self._pack(custom, name, properties, apt_dat, n_dsf)
        (custom / "scenery_packs.ini").write_text(
            f"SCENERY_PACK Custom Scenery/{name}/\n", encoding="utf-8"
        )
        return VEG._load_custom_scenery_forest_exclusions(
            str(custom), 50, 12, "", "dsftool-not-used", tmpdir
        )

    def test_global_forest_pack_without_apt_dat_is_ignored(self):
        # Global Forests v2 stamps a whole-tile sim/exclude_for on every tile to
        # take over the stock forests. Honouring it wipes the whole overlay.
        with tempfile.TemporaryDirectory() as tmpdir:
            rings, type_counts = self._load(
                tmpdir, "Global_Forests_v2", self.WHOLE_TILE, apt_dat=False
            )
        self.assertEqual(rings, [])
        self.assertEqual(type_counts, {})

    def test_regional_pack_with_apt_dat_is_ignored(self):
        # A regional VFR pack ships an apt.dat too, so the tile footprint is
        # what separates it from an airport (Bhutan_VFR spans 15 tiles).
        with tempfile.TemporaryDirectory() as tmpdir:
            rings, _ = self._load(
                tmpdir,
                "Bhutan_VFR",
                self.WHOLE_TILE,
                apt_dat=True,
                n_dsf=VEG.AIRPORT_PACK_MAX_DSFS + 1,
            )
        self.assertEqual(rings, [])

    def test_airport_pack_at_the_tile_limit_is_honoured(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rings, _ = self._load(
                tmpdir,
                "HADC_Scenery_Pack",
                (("sim/exclude_for", "12.20/50.10/12.21/50.11"),),
                apt_dat=True,
                n_dsf=VEG.AIRPORT_PACK_MAX_DSFS,
            )
        self.assertEqual(len(rings), 1)

    def test_pack_without_earth_nav_data_is_not_an_airport(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertFalse(VEG._is_airport_scenery_pack(tmpdir))


def _repo_dsftool():
    for rel in ("Utils/win/DSFTool.exe", "Utils/lin/DSFTool", "Utils/mac/DSFTool"):
        candidate = ROOT / rel
        if candidate.is_file():
            return candidate
    return None


class RealDsfRoundTripTests(unittest.TestCase):
    """The synthetic DSFs above encode the layout by hand — check it against a
    file DSFTool actually produced, so a wrong atom convention can't pass."""

    def test_dsftool_compiled_header_reads_back(self):
        dsftool = _repo_dsftool()
        if dsftool is None:
            self.skipTest("DSFTool not bundled in this checkout")
        import subprocess

        with tempfile.TemporaryDirectory() as tmpdir:
            text_path = Path(tmpdir) / "excl.txt"
            text_path.write_text(
                "\n".join(
                    [
                        "PROPERTY sim/planet earth",
                        "PROPERTY sim/overlay 1",
                        "PROPERTY sim/west 12",
                        "PROPERTY sim/east 13",
                        "PROPERTY sim/south 50",
                        "PROPERTY sim/north 51",
                    ]
                    + [f"PROPERTY {k} {v}" for k, v in MIXED_PROPERTIES[2:]]
                )
                + "\n",
                encoding="utf-8",
            )
            dsf_path = Path(tmpdir) / "+50+012.dsf"
            proc = subprocess.run(
                [str(dsftool), "--text2dsf", str(text_path), str(dsf_path)],
                capture_output=True, timeout=120,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr[:400])
            pairs = VEG._read_dsf_binary_properties(str(dsf_path))

        self.assertIsNotNone(pairs, "binary reader rejected a real DSF")
        zones = VEG._zones_from_property_pairs(pairs)
        self.assertEqual(
            zones["sim/exclude_for"],
            [[(12.2, 50.1), (12.3, 50.1), (12.3, 50.2), (12.2, 50.2)]],
        )
        self.assertEqual(len(zones["sim/exclude_obj"]), 2)


class ExclusionZoneMaskTests(unittest.TestCase):
    def test_zone_clears_trees_inside_it_only(self):
        lat_n, lat_s, lon_w, lon_e = 50.3, 50.0, 12.0, 12.4
        img_h = img_w = 40
        ring = VEG._parse_exclusion_rect("12.20/50.10/12.30/50.20")
        prepared = VEG._prepare_polygons([ring])
        index = VEG.BBOX.build_bounds_index(prepared)
        local = VEG._polys_for_bounds(index, lat_n, lat_s, lon_w, lon_e)
        mask = VEG._rasterize_polygons(
            local, lat_n, lat_s, lon_w, lon_e, img_h, img_w
        )

        self.assertEqual(len(local), 1)
        # Inside the zone (12.25 E, 50.15 N) is masked, outside is untouched.
        inside_x = int((12.25 - lon_w) / (lon_e - lon_w) * img_w)
        inside_y = int((lat_n - 50.15) / (lat_n - lat_s) * img_h)
        outside_x = int((12.05 - lon_w) / (lon_e - lon_w) * img_w)
        outside_y = int((lat_n - 50.05) / (lat_n - lat_s) * img_h)
        self.assertEqual(mask[inside_y, inside_x], 1)
        self.assertEqual(mask[outside_y, outside_x], 0)

        trees = np.ones((img_h, img_w), dtype=np.uint8)
        trees = VEG.cv2.bitwise_and(trees, VEG.cv2.bitwise_not(mask))
        self.assertEqual(trees[inside_y, inside_x], 0)
        self.assertEqual(trees[outside_y, outside_x], 1)


class MaskCacheKeyTests(unittest.TestCase):
    def test_cache_key_changes_with_exclusion_zones(self):
        base = dict(
            fname="1_2_BI16.dds", lat_n=50.3, lat_s=50.0, lon_w=12.0, lon_e=12.4,
            img_h=40, img_w=40, mpp=2.0, road_sig=(0, 0, 0.0),
            res_road_sig=(0, 0, 0.0), tree_row_sig=(0, 0, 0.0),
            context_poly_sigs={}, forest_layer_sigs=(), sh_bld_poly_sig=(0, 0, 0.0),
            sh_bld_obj_sig=(0, 0.0), bld_excl_m=10.0, bld_cache_stat=None,
            simheaven_building_buffer_m=10.0,
        )
        without = VEG._dds_mask_cache_key(**base, excl_zone_sig=(0, 0, 0.0, 0))
        with_zone = VEG._dds_mask_cache_key(**base, excl_zone_sig=(1, 4, 3.5, 7))
        self.assertNotEqual(without, with_zone)


if __name__ == "__main__":
    unittest.main()
