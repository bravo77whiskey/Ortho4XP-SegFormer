"""simHeaven-derived road classes and forest-asset context (no OSM downloads)."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_Building_Overlay as BLD  # noqa: E402
import O4_Vegetation_Overlay as VEG  # noqa: E402

_NETWORK_DSF_TEXT = """\
NETWORK_DEF lib/g10/roads.net
NETWORK_DEF objects/road.net
BEGIN_SEGMENT 0 51 1 120.10 22.10 0.0
END_SEGMENT 2 120.11 22.11 0.0
BEGIN_SEGMENT 0 152 3 120.20 22.20 0.0
SHAPE_POINT 120.21 22.21 0.0
END_SEGMENT 4 120.22 22.22 0.0
BEGIN_SEGMENT 0 100 5 120.30 22.30 0.0
END_SEGMENT 6 120.31 22.31 0.0
BEGIN_SEGMENT 1 250 7 120.40 22.40 0.0
END_SEGMENT 8 120.41 22.41 0.0
"""


class SimHeavenSegmentTypeTest(unittest.TestCase):
    def test_roads_net_subtypes_map_to_osm_classes(self):
        cases = {
            12: 'primary', 24: 'primary', 33: 'secondary', 43: 'secondary',
            51: 'residential', 63: 'residential', 70: 'service',
            100: 'motorway', 110: 'trunk', 120: 'motorway_link',
            151: 'rail', 153: 'rail', 154: 'rail', 220: 'network',
        }
        for subtype, expected in cases.items():
            with self.subTest(subtype=subtype):
                self.assertEqual(
                    BLD._simheaven_segment_type('lib/g10/roads.net', subtype), expected
                )

    def test_other_network_definitions_stay_generic(self):
        self.assertEqual(BLD._simheaven_segment_type('objects/road.net', 51), 'network')

    def test_network_parse_tags_each_segment(self):
        with tempfile.TemporaryDirectory() as tmp:
            text_path = Path(tmp) / "net.txt"
            text_path.write_text(_NETWORK_DSF_TEXT, encoding="utf-8")
            with mock.patch.object(
                BLD, "find_simheaven_network_dsfs",
                return_value=[("simHeaven_X-World_Asia-8-network", str(text_path))],
            ), mock.patch.object(BLD, "ensure_cached_dsf_text", return_value=str(text_path)):
                ways = BLD._load_simheaven_network(tmp, 22, 120, "dsftool", tmp)
        self.assertEqual(
            [way['type'] for way in ways], ['residential', 'rail', 'motorway', 'network']
        )
        # (lat, lon) points, shape points kept
        self.assertEqual(ways[1]['pts'], [(22.20, 120.20), (22.21, 120.21), (22.22, 120.22)])


class SimHeavenForestContextTest(unittest.TestCase):
    def test_forest_families_map_to_context_keys(self):
        square = [(120.0, 22.0), (120.1, 22.0), (120.1, 22.1), (120.0, 22.0)]
        records = [
            {'pts': square, 'path': 'simheaven/forests/mixed.for'},
            {'pts': square, 'path': 'simheaven/forests/broad.for'},
            {'pts': square, 'path': 'simheaven/forests/orchard.for'},
            {'pts': square, 'path': 'simheaven/forests/shrubs.for'},
            {'pts': square, 'path': 'simheaven/forests/wetland.for'},
            {'pts': square, 'path': 'simheaven/forests/hedge.for'},
        ]
        context = VEG._veg_context_from_simheaven(records)
        counts = {key: len(value) for key, value in context.items()}
        self.assertEqual(counts, {
            'forest_polys': 2,
            'woodland_hint_polys': 1,
            'orchard_polys': 1,
            'managed_polys': 0,
            'residential_polys': 0,
            'farmland_polys': 0,
            'water_polys': 1,
            'tree_rows': 0,
        })

    def test_no_simheaven_gives_empty_context(self):
        context = VEG._veg_context_from_simheaven([])
        self.assertTrue(all(not value for value in context.values()))


if __name__ == "__main__":
    unittest.main()
