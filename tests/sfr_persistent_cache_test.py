import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_SFR_Persistent_Cache as PCACHE


class PersistentCacheTests(unittest.TestCase):
    def test_load_or_build_reuses_cached_payload_until_source_changes(self):
        builds = []

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "source.txt")
            cache_dir = os.path.join(tmpdir, "cache")

            with open(source_path, "w", encoding="utf-8") as handle:
                handle.write("v1")

            def _builder():
                builds.append(len(builds) + 1)
                return {"build": builds[-1]}

            first = PCACHE.load_or_build(
                source_path,
                cache_dir,
                "unit",
                _builder,
                version="test-v1",
            )
            second = PCACHE.load_or_build(
                source_path,
                cache_dir,
                "unit",
                _builder,
                version="test-v1",
            )

            self.assertEqual(first, {"build": 1})
            self.assertEqual(second, {"build": 1})
            self.assertEqual(builds, [1])

            with open(source_path, "w", encoding="utf-8") as handle:
                handle.write("v2 with different size")
            os.utime(source_path, None)

            third = PCACHE.load_or_build(
                source_path,
                cache_dir,
                "unit",
                _builder,
                version="test-v1",
            )

            self.assertEqual(third, {"build": 2})
            self.assertEqual(builds, [1, 2])


if __name__ == "__main__":
    unittest.main()
