import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_SFR_Bounds_Index as BBOX


class BoundsIndexTests(unittest.TestCase):
    def test_query_returns_intersections_in_input_order(self):
        items = [
            {"name": "west", "_bounds": (0.0, 1.0, 0.0, 1.0)},
            {"name": "middle", "_bounds": (0.4, 1.4, 0.4, 1.4)},
            {"name": "east", "_bounds": (0.0, 1.0, 2.0, 3.0)},
        ]

        index = BBOX.build_bounds_index(items)
        result = BBOX.query_bounds(index, 0.5, 0.9, 0.5, 1.2)

        self.assertEqual([item["name"] for item in result], ["west", "middle"])

    def test_query_returns_empty_list_for_empty_or_miss(self):
        self.assertIsNone(BBOX.build_bounds_index([]))
        self.assertEqual(BBOX.query_bounds(None, 0.0, 1.0, 0.0, 1.0), [])

        items = [{"name": "far", "_bounds": (10.0, 11.0, 10.0, 11.0)}]
        index = BBOX.build_bounds_index(items)

        self.assertEqual(BBOX.query_bounds(index, 0.0, 1.0, 0.0, 1.0), [])


if __name__ == "__main__":
    unittest.main()
