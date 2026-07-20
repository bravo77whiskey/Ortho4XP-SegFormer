from pathlib import Path
import sys
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_OSM_Utils as OSM


OSM_OK_BODY = b'<?xml version="1.0"?>\n<osm version="0.6">\n</osm>'


class FakeResponse:
    def __init__(self, status_code=200, content=OSM_OK_BODY, headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}


class FakeSessionFactory:
    """Stands in for requests.Session; serves scripted responses and records calls."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self):
        return _FakeSession(self)


class _FakeSession:
    def __init__(self, factory):
        self.factory = factory

    def post(self, url, data=None, headers=None, timeout=None):
        self.factory.calls.append(
            {"url": url, "data": data, "headers": headers, "timeout": timeout}
        )
        item = self.factory.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def run_query(factory, query='way["highway"="primary"]', server_code=None,
              choice="KU"):
    with mock.patch.object(OSM.requests, "Session", factory), \
         mock.patch.object(OSM, "overpass_server_choice", choice), \
         mock.patch.object(OSM.time, "sleep") as fake_sleep:
        result = OSM.get_overpass_data(query, (0, 0, 1, 1), server_code)
    return result, fake_sleep


def test_success_uses_post_with_identifying_user_agent():
    factory = FakeSessionFactory([FakeResponse()])

    result, _ = run_query(factory, server_code="DE")

    assert result == OSM_OK_BODY
    assert len(factory.calls) == 1
    call = factory.calls[0]
    assert call["url"] == OSM.overpass_servers["DE"]
    assert call["data"]["data"].startswith("(")
    assert "(0, 0, 1, 1)" in call["data"]["data"]
    assert call["headers"]["User-Agent"] == OSM.overpass_user_agent
    assert "Ortho4XP" in call["headers"]["User-Agent"]


def test_tuple_query_is_batched_into_one_request():
    factory = FakeSessionFactory([FakeResponse()])
    query = (
        'way["highway"="motorway"]',
        'way["highway"="trunk"]',
        'way["railway"="rail"]',
    )

    result, _ = run_query(factory, query=query, server_code="DE")

    assert result == OSM_OK_BODY
    assert len(factory.calls) == 1
    payload = factory.calls[0]["data"]["data"]
    for fragment in query:
        assert fragment + "(0, 0, 1, 1);" in payload


def test_rotation_moves_to_next_server_on_rejection():
    factory = FakeSessionFactory([FakeResponse(status_code=400), FakeResponse()])

    result, _ = run_query(factory, choice="KU")

    assert result == OSM_OK_BODY
    codes = list(OSM.overpass_servers.keys())
    start = codes.index("KU")
    assert factory.calls[0]["url"] == OSM.overpass_servers[codes[start]]
    expected_second = codes[(start + 1) % len(codes)]
    assert factory.calls[1]["url"] == OSM.overpass_servers[expected_second]


def test_rotation_moves_to_next_server_on_exception():
    factory = FakeSessionFactory([ConnectionError("boom"), FakeResponse()])

    result, _ = run_query(factory, choice="KU")

    assert result == OSM_OK_BODY
    assert factory.calls[0]["url"] != factory.calls[1]["url"]


def test_retry_after_header_is_honored_but_capped():
    factory = FakeSessionFactory(
        [
            FakeResponse(status_code=429, headers={"Retry-After": "120"}),
            FakeResponse(),
        ]
    )

    result, fake_sleep = run_query(factory)

    assert result == OSM_OK_BODY
    assert fake_sleep.call_args_list[0] == mock.call(OSM.max_retry_after_wait)


def test_small_retry_after_is_used_verbatim():
    factory = FakeSessionFactory(
        [
            FakeResponse(status_code=429, headers={"Retry-After": "5"}),
            FakeResponse(),
        ]
    )

    result, fake_sleep = run_query(factory)

    assert result == OSM_OK_BODY
    assert fake_sleep.call_args_list[0] == mock.call(5)


def test_gives_up_after_max_tentatives_having_tried_distinct_servers():
    rejections = [
        FakeResponse(status_code=504, headers={})
        for _ in range(OSM.max_osm_tentatives)
    ]
    factory = FakeSessionFactory(rejections)

    result, fake_sleep = run_query(factory)

    assert result == 0
    assert len(factory.calls) == OSM.max_osm_tentatives
    urls = [call["url"] for call in factory.calls]
    assert len(set(urls)) == min(
        OSM.max_osm_tentatives, len(OSM.overpass_servers)
    )
    # no sleep after the final tentative
    assert fake_sleep.call_count == OSM.max_osm_tentatives - 1


def test_corrupted_body_triggers_retry_on_next_server():
    factory = FakeSessionFactory(
        [FakeResponse(content=b"<osm>truncated"), FakeResponse()]
    )

    result, _ = run_query(factory)

    assert result == OSM_OK_BODY
    assert factory.calls[0]["url"] != factory.calls[1]["url"]


def test_explicit_server_code_pins_first_tentative_only():
    factory = FakeSessionFactory([FakeResponse(status_code=400), FakeResponse()])

    result, _ = run_query(factory, server_code="JP", choice="random")

    assert result == OSM_OK_BODY
    assert factory.calls[0]["url"] == OSM.overpass_servers["JP"]
    assert factory.calls[1]["url"] != OSM.overpass_servers["JP"]
