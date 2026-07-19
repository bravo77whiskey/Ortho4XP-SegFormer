import math
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts import compare_sfr_height_models as COMPARE
from scripts import merge_sfr_height_comparisons as MERGE


class HeightModelComparisonTests(unittest.TestCase):
    def test_final_heights_apply_floor_and_large_footprint_cap(self):
        detections = [
            {"placement_class": 0, "area_m2": 80.0, "max_side_m": 12.0},
            {"placement_class": 4, "area_m2": 4_000.0, "max_side_m": 90.0},
        ]

        final, floored, capped = COMPARE.finalize_heights(
            detections,
            np.asarray([1.0, 80.0]),
        )

        np.testing.assert_allclose(
            final, [2.5, COMPARE.BLD.LARGE_FOOTPRINT_HEIGHT_CAP_M]
        )
        np.testing.assert_array_equal(floored, [True, False])
        np.testing.assert_array_equal(capped, [False, True])

    def test_summarize_predictions_ignores_nan_and_reports_bands(self):
        summary = COMPARE.summarize_predictions(
            np.asarray([3.0, 6.0, 10.0, 20.0, 45.0, math.nan]),
            inference_seconds=2.0,
            floor_count=1,
            cap_count=2,
        )

        self.assertEqual(summary["count"], 5)
        self.assertEqual(summary["floor_count"], 1)
        self.assertEqual(summary["cap_count"], 2)
        self.assertEqual(summary["throughput_buildings_per_s"], 2.5)
        self.assertEqual(
            summary["bands"],
            {"0-4m": 1, "4-8m": 1, "8-15m": 1, "15-30m": 1, "30m+": 1},
        )

    def test_summarize_differences_uses_aligned_finite_pairs(self):
        summary = COMPARE.summarize_differences(
            np.asarray([4.0, math.nan, 12.0]),
            np.asarray([5.0, 8.0, 9.0]),
        )

        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["mean_delta_m"], -1.0)
        self.assertEqual(summary["median_abs_delta_m"], 2.0)
        self.assertEqual(summary["new_taller_count"], 1)
        self.assertEqual(summary["old_taller_count"], 1)

    def test_resolve_inputs_can_resume_after_sorted_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("c.dds", "a.dds", "b.dds"):
                (root / name).touch()

            selected = COMPARE._resolve_inputs([str(root)], skip=1)

        self.assertEqual([path.name for path in selected], ["b.dds", "c.dds"])

    def test_merge_summary_accepts_disjoint_shards_and_rejects_overlap(self):
        def write_shard(path, source, delta):
            row = {field: 0 for field in COMPARE.CSV_FIELDS}
            row.update({
                "source": source,
                "old_raw_m": 10.0,
                "old_final_m": 10.0,
                "new_raw_m": 10.0 + delta,
                "new_final_m": 10.0 + delta,
                "old_floored": False,
                "old_capped": False,
                "new_floored": False,
                "new_capped": False,
            })
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=COMPARE.CSV_FIELDS)
                writer.writeheader()
                writer.writerow(row)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.csv"
            second = root / "second.csv"
            write_shard(first, "zOrtho4XP_-01+036_a", -2.0)
            write_shard(second, "zOrtho4XP_+35+139_b", 3.0)

            result = MERGE._summarize_shards([first, second])
            self.assertEqual(result[4], 2)
            self.assertEqual(set(result[0]), {
                "zOrtho4XP_-01+036", "zOrtho4XP_+35+139",
            })

            with self.assertRaisesRegex(ValueError, "overlap"):
                MERGE._summarize_shards([first, first])

    def test_merge_main_preserves_measured_shard_throughput(self):
        def write_shard(path, source):
            row = {field: 0 for field in COMPARE.CSV_FIELDS}
            row.update({
                "source": source,
                "old_raw_m": 10.0,
                "old_final_m": 10.0,
                "new_raw_m": 8.0,
                "new_final_m": 8.0,
                "old_floored": False,
                "old_capped": False,
                "new_floored": False,
                "new_capped": False,
            })
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=COMPARE.CSV_FIELDS)
                writer.writeheader()
                writer.writerow(row)

        def model_summary(arch, crop, seconds, throughput):
            return {
                "metadata": {"arch": arch, "crop_px": crop},
                "raw": {
                    "inference_seconds": seconds,
                    "throughput_buildings_per_s": throughput,
                },
                "final": {
                    "inference_seconds": seconds,
                    "throughput_buildings_per_s": throughput,
                },
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.csv"
            second = root / "second.csv"
            output = root / "output"
            timing_path = root / "timing.json"
            write_shard(first, "zOrtho4XP_-01+036_a")
            write_shard(second, "zOrtho4XP_+35+139_b")
            timing_path.write_text(json.dumps({
                "inputs": 1,
                "device": "cuda",
                "yolo_checkpoint": "yolo.pt",
                "models": {
                    "old": model_summary("v1-legacy", 96, 2.0, 123.0),
                    "new": model_summary("v2s", 128, 3.0, 45.0),
                },
            }), encoding="utf-8")

            result = MERGE.main([
                str(first), str(second),
                "--output-dir", str(output),
                "--timing-summary", str(timing_path),
                "--input-count", "3",
            ])
            summary = json.loads((output / "summary.json").read_text())

        self.assertEqual(result, 0)
        self.assertEqual(summary["inputs"], 3)
        self.assertEqual(summary["inputs_with_detections"], 2)
        self.assertEqual(
            summary["models"]["old"]["final"]["throughput_buildings_per_s"],
            123.0,
        )
        self.assertEqual(
            summary["models"]["new"]["final"]["throughput_buildings_per_s"],
            45.0,
        )


if __name__ == "__main__":
    unittest.main()
