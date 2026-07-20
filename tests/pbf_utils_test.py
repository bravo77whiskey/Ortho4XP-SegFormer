import datetime
from pathlib import Path
import sys
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_PBF_Utils as PBF
import O4_UI_Utils as UI


UTC = datetime.timezone.utc


def parse_filter_expressions(expressions):
    """osmium filter expr list -> set of (types, key, value) triples."""
    covered = set()
    for expression in expressions:
        types, keyval = expression.split("/", 1)
        if "=" in keyval:
            key, values = keyval.split("=", 1)
            for value in values.split(","):
                covered.add((types, key, value))
        else:
            covered.add((types, keyval, None))
    return covered


# Ground truth: the Overpass queries issued by O4_Vector_Map / O4_Mesh_Utils.
OVERPASS_QUERY_TAGS = {
    "airports": [("nwr", "aeroway", None)],
    "big_roads": [
        ("w", "highway", "motorway"),
        ("w", "highway", "trunk"),
        ("w", "highway", "primary"),
        ("w", "highway", "secondary"),
        ("w", "railway", "rail"),
        ("w", "railway", "narrow_gauge"),
    ],
    "small_roads": [
        ("w", "highway", "tertiary"),
        ("w", "highway", "unclassified"),
        ("w", "highway", "residential"),
        ("w", "highway", "service"),
        ("w", "highway", "track"),
    ],
    "coastline": [("w", "natural", "coastline")],
    "water": [
        ("wr", "natural", "water"),
        ("wr", "waterway", "riverbank"),
        ("w", "waterway", "dock"),
    ],
}


def covers(covered, types, key, value):
    for candidate_types, candidate_key, candidate_value in covered:
        if key != candidate_key:
            continue
        if candidate_value is not None and candidate_value != value:
            continue
        if all(t in candidate_types for t in types):
            return True
    return False


def test_layer_filters_cover_all_overpass_queries():
    for suffix, tags in OVERPASS_QUERY_TAGS.items():
        covered = parse_filter_expressions(PBF.LAYER_FILTERS[suffix])
        for types, key, value in tags:
            assert covers(covered, types, key, value), (suffix, key, value)


def test_scenery_filter_covers_every_layer():
    covered = parse_filter_expressions(PBF.SCENERY_FILTER)
    for suffix, tags in OVERPASS_QUERY_TAGS.items():
        for types, key, value in tags:
            assert covers(covered, types, key, value), (suffix, key, value)


def test_parse_state_text_unescapes_colons():
    seq, timestamp = PBF._parse_state_text(
        "#comment\nsequenceNumber=5058\n"
        "timestamp=2026-07-19T00\\:00\\:00Z\n"
    )
    assert seq == 5058
    assert timestamp == datetime.datetime(2026, 7, 19, tzinfo=UTC)


def test_seq_url_formatting():
    assert PBF._seq_url(5058).endswith("/000/005/058")
    assert PBF._seq_url(1234567).endswith("/001/234/567")


def test_find_start_sequence_binary_search():
    base = datetime.datetime(2026, 1, 1, tzinfo=UTC)
    calls = []

    def fake_fetch_state(seq=None):
        calls.append(seq)
        return (seq, base + datetime.timedelta(days=seq))

    local_ts = base + datetime.timedelta(days=150, hours=1)
    with mock.patch.object(PBF, "fetch_state", side_effect=fake_fetch_state):
        start = PBF.find_start_sequence(local_ts, 200)

    # target = day 150 - 1h; the largest state <= target is day 149.
    assert start == 150
    assert len(calls) <= 14


class FakeDownloadResponse:
    def __init__(self, status_code, chunks, headers=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._chunks = chunks

    def iter_content(self, chunk_size):
        for chunk in self._chunks:
            yield chunk

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_download_resumes_from_part_file(tmp_path):
    dest = tmp_path / "file.bin"
    part = tmp_path / "file.bin.part"
    part.write_bytes(b"abc")
    recorded = {}

    def fake_get(url, headers=None, stream=True, timeout=None):
        recorded["headers"] = headers
        return FakeDownloadResponse(
            206,
            [b"def", b"gh"],
            headers={"Content-Range": "bytes 3-7/8"},
        )

    with mock.patch.object(PBF.requests, "get", side_effect=fake_get), \
         mock.patch.object(PBF.time, "sleep"):
        ok = PBF._download_file("http://example.org/file.bin", str(dest))

    assert ok is True
    assert recorded["headers"]["Range"] == "bytes=3-"
    assert "Ortho4XP" in recorded["headers"]["User-Agent"]
    assert dest.read_bytes() == b"abcdefgh"
    assert not part.exists()


def test_download_red_flag_keeps_part_file(tmp_path):
    dest = tmp_path / "file.bin"

    def chunks():
        yield b"1234"
        UI.red_flag = True
        yield b"5678"

    def fake_get(url, headers=None, stream=True, timeout=None):
        return FakeDownloadResponse(
            200, chunks(), headers={"Content-Length": "12"}
        )

    try:
        with mock.patch.object(PBF.requests, "get", side_effect=fake_get), \
             mock.patch.object(PBF.time, "sleep"):
            ok = PBF._download_file("http://example.org/file.bin", str(dest))
    finally:
        UI.red_flag = False

    assert ok is False
    assert not dest.exists()
    assert (tmp_path / "file.bin.part").exists()


def test_download_retries_on_dropped_connection(tmp_path):
    dest = tmp_path / "file.bin"
    responses = [
        FakeDownloadResponse(200, [b"12"], headers={"Content-Length": "4"}),
        FakeDownloadResponse(
            206, [b"34"], headers={"Content-Range": "bytes 2-3/4"}
        ),
    ]

    def fake_get(url, headers=None, stream=True, timeout=None):
        return responses.pop(0)

    with mock.patch.object(PBF.requests, "get", side_effect=fake_get), \
         mock.patch.object(PBF.time, "sleep"):
        ok = PBF._download_file("http://example.org/file.bin", str(dest))

    assert ok is True
    assert dest.read_bytes() == b"1234"


def test_download_sha256_mismatch_discards_file(tmp_path):
    dest = tmp_path / "file.bin"

    def fake_get(url, headers=None, stream=True, timeout=None):
        return FakeDownloadResponse(
            200, [b"data"], headers={"Content-Length": "4"}
        )

    with mock.patch.object(PBF.requests, "get", side_effect=fake_get), \
         mock.patch.object(PBF.time, "sleep"):
        ok = PBF._download_file(
            "http://example.org/file.bin",
            str(dest),
            expected_sha256="0" * 64,
        )

    assert ok is False
    assert not dest.exists()
    assert not (tmp_path / "file.bin.part").exists()


def test_preflight_disk_refuses_when_short(tmp_path):
    usage = mock.Mock(free=10)
    with mock.patch.object(PBF.shutil, "disk_usage", return_value=usage):
        assert PBF._preflight_disk(str(tmp_path), 100) is False
    usage = mock.Mock(free=200)
    with mock.patch.object(PBF.shutil, "disk_usage", return_value=usage):
        assert PBF._preflight_disk(str(tmp_path), 100) is True


def test_install_updates_refuses_while_tile_build_running():
    with mock.patch.object(PBF.UI, "is_working", True), \
         mock.patch.object(
             PBF, "fetch_state", side_effect=AssertionError("no network")
         ):
        assert PBF.install_updates() == 0


def test_replace_with_retries_retries_then_succeeds():
    attempts = []

    def flaky_replace(src, dst):
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError()

    with mock.patch.object(PBF.os, "replace", side_effect=flaky_replace), \
         mock.patch.object(PBF.time, "sleep"):
        assert PBF._replace_with_retries("a", "b") is True
    assert len(attempts) == 3


def test_replace_with_retries_gives_up():
    with mock.patch.object(
             PBF.os, "replace", side_effect=PermissionError()
         ), \
         mock.patch.object(PBF.time, "sleep"):
        assert PBF._replace_with_retries("a", "b") is False


def test_pbf_ready_false_without_configuration():
    with mock.patch.object(PBF, "osm_pbf_dir", ""):
        assert PBF.pbf_ready() is False


def test_pbf_ready_false_during_maintenance(tmp_path):
    extract = tmp_path / "scenery.osm.pbf"
    extract.write_bytes(b"x")
    with mock.patch.object(PBF, "osm_pbf_dir", str(tmp_path)), \
         mock.patch.object(PBF, "maintenance_in_progress", True), \
         mock.patch.object(PBF, "tools_available", return_value=True):
        assert PBF.pbf_ready() is False


def test_pbf_ready_true_when_extract_and_tools_present(tmp_path):
    extract = tmp_path / "scenery.osm.pbf"
    extract.write_bytes(b"x")
    with mock.patch.object(PBF, "osm_pbf_dir", str(tmp_path)), \
         mock.patch.object(PBF, "tools_available", return_value=True):
        assert PBF.pbf_ready() is True
