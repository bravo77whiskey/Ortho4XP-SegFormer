from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts import install_gfv2_simheaven_redirects as REDIRECTS


def sample_library() -> str:
    rows = ["A", "800", "LIBRARY", "REGION_DEFINE test", "REGION_ALL", "REGION test"]
    for season in ("sum", "spr", "fal", "win"):
        rows.extend(
            (
                f"EXPORT_SEASON {season} simheaven/forests/broad.for "
                f"1200 forests/{season}/broad.for",
                f"EXPORT_SEASON {season} simheaven/forests/mixed.for "
                f"1200 forests/{season}/mixed.for",
                f"EXPORT_SEASON {season} simheaven/forests/coni.for "
                f"1200 forests/{season}/coni.for",
            )
        )
    return "\n".join(rows) + "\n"


class GlobalForestsSimheavenRedirectTests(unittest.TestCase):
    def test_build_redirects_preserves_seasonal_region_exports(self):
        legacy_paths = (
            "forests/tropical/woodland/tropical_woodland_75_y1.for",
            "forests/northnorth/mixed/northnorth_mixed_75_y1.for",
            "forests/northnorth/decidbroad/northnorth_decidbroad_75_y1.for",
        )

        patched, counts = REDIRECTS.build_redirected_library(
            sample_library(),
            legacy_paths,
        )

        REDIRECTS.redirect_season_coverage(patched, legacy_paths)
        self.assertEqual(counts[legacy_paths[0]], 4)
        self.assertEqual(counts[legacy_paths[1]], 4)
        self.assertEqual(counts[legacy_paths[2]], 4)
        self.assertIn(
            "EXPORT_SEASON\tsum\t"
            "forests/northnorth/mixed/northnorth_mixed_75_y1.for\t"
            "1200 forests/sum/mixed.for",
            patched,
        )

    def test_reinstall_replaces_old_generated_blocks(self):
        legacy_paths = (
            "forests/tropical/woodland/tropical_woodland_75_y1.for",
        )
        first, _counts = REDIRECTS.build_redirected_library(
            sample_library(),
            legacy_paths,
        )

        second, _counts = REDIRECTS.build_redirected_library(
            first,
            legacy_paths,
        )

        self.assertEqual(first, second)
        self.assertEqual(first.count(REDIRECTS.BEGIN_MARKER), 4)
        self.assertEqual(first.count(REDIRECTS.END_MARKER), 4)

    def test_family_mapping_uses_simheaven_categories(self):
        self.assertEqual(
            REDIRECTS.source_kind_for_legacy_path(
                "forests/northnorth/mixed/northnorth_mixed_75_y1.for"
            ),
            "mixed",
        )
        self.assertEqual(
            REDIRECTS.source_kind_for_legacy_path(
                "forests/northnorth/decidbroad/northnorth_decidbroad_75_y1.for"
            ),
            "broad",
        )
        self.assertEqual(
            REDIRECTS.source_kind_for_legacy_path(
                "forests/northnorth/conifer/northnorth_conifer_75_y1.for"
            ),
            "coni",
        )


if __name__ == "__main__":
    unittest.main()
