"""O4_SFR_Overpass: server rotation, answer validation, atomic cache writes."""

import bz2
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import O4_SFR_Overpass as OVERPASS  # noqa: E402

GOOD = b'<?xml version="1.0"?>\n<osm version="0.6"><way id="1"/></osm>\n'
PARTIAL = (
    b'<?xml version="1.0"?>\n<osm version="0.6">'
    b'<remark> runtime error: Query timed out in "query" at line 1 </remark>'
    b'<way id="1"/></osm>\n'
)


def _http_error(code, retry_after=None):
    headers = {"Retry-After": str(retry_after)} if retry_after is not None else {}
    return urllib.error.HTTPError("url", code, "err", headers, None)


class SfrOverpassTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = os.path.join(self.tmp.name, "sub", "tile.osm.bz2")
        sleep = mock.patch.object(OVERPASS.time, "sleep")
        self.sleep = sleep.start()
        self.addCleanup(sleep.stop)
        self.addCleanup(self.tmp.cleanup)

    def _run(self, answers):
        calls = []

        def fake_post(server, query, timeout):
            calls.append(server)
            answer = answers[len(calls) - 1]
            if isinstance(answer, Exception):
                raise answer
            return answer

        with mock.patch.object(OVERPASS, "_post", side_effect=fake_post):
            ok = OVERPASS.download_to_cache("q", self.cache, "test")
        return ok, calls

    def test_rotates_past_rejection_to_next_server(self):
        ok, calls = self._run([_http_error(406), GOOD])
        self.assertTrue(ok)
        self.assertEqual(calls, list(OVERPASS.SERVERS[:2]))
        with bz2.open(self.cache, "rb") as handle:
            self.assertEqual(handle.read(), GOOD)
        self.assertFalse(os.path.exists(self.cache + ".part"))

    def test_partial_and_truncated_answers_are_not_cached(self):
        ok, calls = self._run([PARTIAL, GOOD[:-8], GOOD])
        self.assertTrue(ok)
        self.assertEqual(len(calls), 3)

    def test_all_rounds_fail_leaves_no_cache(self):
        n = len(OVERPASS.SERVERS) * OVERPASS.MAX_ROUNDS
        ok, calls = self._run([_http_error(504, retry_after=30)] * n)
        self.assertFalse(ok)
        self.assertEqual(len(calls), n)
        self.assertFalse(os.path.exists(self.cache))
        # Retry-After is honoured (capped) between rounds.
        self.sleep.assert_called_once_with(30)

    def test_existing_cache_skips_network(self):
        os.makedirs(os.path.dirname(self.cache))
        Path(self.cache).write_bytes(b"x")
        ok, calls = self._run([])
        self.assertTrue(ok)
        self.assertEqual(calls, [])

    def test_tile_query_and_user_agent(self):
        query = OVERPASS.tile_query(22, 120, ('way["highway"]',), timeout=10)
        self.assertEqual(
            query, '[out:xml][timeout:10];(way["highway"](22,120,23,121););(._;>;);out body;'
        )
        self.assertNotIn("Python-urllib", OVERPASS.USER_AGENT)
        self.assertTrue(OVERPASS.USER_AGENT.startswith("Ortho4XP/"))


if __name__ == "__main__":
    unittest.main()
