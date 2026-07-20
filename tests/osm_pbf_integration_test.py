import bz2
import os
from pathlib import Path
import sys
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_OSM_Utils as OSM
import O4_PBF_Utils as PBF


MINI_OSM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="osmium/1.19.1">
  <node id="1" version="2" lat="0.5000000" lon="0.5000000"/>
  <node id="2" version="1" lat="0.6000000" lon="0.6000000"/>
  <way id="10" version="3">
    <nd ref="1"/>
    <nd ref="2"/>
    <tag k="highway" v="primary"/>
  </way>
</osm>
"""


def fake_run_tool_factory(recorded):
    def fake_run_tool(cmd, log_prefix="      "):
        recorded.append(cmd)
        out_path = cmd[cmd.index("-o") + 1]
        if "tags-filter" in cmd:
            with open(out_path, "wb") as f:
                f.write(bz2.compress(MINI_OSM_XML.encode("utf-8")))
        else:
            with open(out_path, "wb") as f:
                f.write(b"fake pbf")
        return 0

    return fake_run_tool


def test_slice_tile_layer_command_shapes_and_cache_placement(tmp_path):
    recorded = []
    cache_file = tmp_path / "OSM_data" / "+47+009_big_roads.osm.bz2"
    extract = tmp_path / "scenery.osm.pbf"
    extract.write_bytes(b"x")

    with mock.patch.object(PBF, "osm_pbf_dir", str(tmp_path)), \
         mock.patch.object(PBF, "tools_available", return_value=True), \
         mock.patch.object(PBF, "osmium_path", return_value="osmium"), \
         mock.patch.object(PBF, "_run_tool", side_effect=fake_run_tool_factory(recorded)), \
         mock.patch.object(PBF, "_tile_extract_cache", {}), \
         mock.patch.object(PBF.FNAMES, "Tmp_dir", str(tmp_path / "tmp")), \
         mock.patch.object(
             PBF.FNAMES, "osm_cached", return_value=str(cache_file)
         ):
        result = PBF.slice_tile_layer(47, 9, "big_roads")

    assert result == str(cache_file)
    assert cache_file.is_file()
    with bz2.open(cache_file, "rt", encoding="utf-8") as f:
        assert "<osm " in f.read()

    extract_cmd = recorded[0]
    assert "extract" in extract_cmd
    assert extract_cmd[extract_cmd.index("-b") + 1] == "9,47,10,48"
    assert extract_cmd[extract_cmd.index("-s") + 1] == "smart"
    assert str(extract) in extract_cmd

    filter_cmd = recorded[1]
    assert "tags-filter" in filter_cmd
    for expression in PBF.LAYER_FILTERS["big_roads"]:
        assert expression in filter_cmd
    # osmium tags-filter includes referenced objects by default; the option
    # that exists is -R (omit them), which must NOT be passed.
    assert "-r" not in filter_cmd
    assert "-R" not in filter_cmd
    assert filter_cmd[filter_cmd.index("-f") + 1] == "osm.bz2"


def test_tile_extract_is_memoized_across_layers(tmp_path):
    recorded = []
    extract = tmp_path / "scenery.osm.pbf"
    extract.write_bytes(b"x")

    def cache_path(lat, lon, suffix):
        return str(tmp_path / "OSM_data" / (suffix + ".osm.bz2"))

    with mock.patch.object(PBF, "osm_pbf_dir", str(tmp_path)), \
         mock.patch.object(PBF, "tools_available", return_value=True), \
         mock.patch.object(PBF, "osmium_path", return_value="osmium"), \
         mock.patch.object(PBF, "_run_tool", side_effect=fake_run_tool_factory(recorded)), \
         mock.patch.object(PBF, "_tile_extract_cache", {}), \
         mock.patch.object(PBF.FNAMES, "Tmp_dir", str(tmp_path / "tmp")), \
         mock.patch.object(PBF.FNAMES, "osm_cached", side_effect=cache_path):
        assert PBF.slice_tile_layer(47, 9, "big_roads")
        assert PBF.slice_tile_layer(47, 9, "water")

    extract_calls = [cmd for cmd in recorded if "extract" in cmd]
    filter_calls = [cmd for cmd in recorded if "tags-filter" in cmd]
    assert len(extract_calls) == 1
    assert len(filter_calls) == 2


def test_queries_layer_uses_pbf_and_never_calls_overpass(tmp_path):
    layer = OSM.OSM_layer()
    cache_file = tmp_path / "+47+009_big_roads.osm.bz2"

    def fake_slice(lat, lon, cached_suffix):
        assert (lat, lon, cached_suffix) == (47, 9, "big_roads")
        with open(cache_file, "wb") as f:
            f.write(bz2.compress(MINI_OSM_XML.encode("utf-8")))
        return str(cache_file)

    with mock.patch.object(OSM.PBF, "pbf_ready", return_value=True), \
         mock.patch.object(OSM.PBF, "slice_tile_layer", side_effect=fake_slice), \
         mock.patch.object(
             OSM,
             "get_overpass_data",
             side_effect=AssertionError("Overpass must not be called"),
         ), \
         mock.patch.object(
             OSM.FNAMES, "osm_cached", return_value=str(cache_file)
         ):
        ok = OSM.OSM_queries_to_OSM_layer(
            ['way["highway"="primary"]'],
            layer,
            47,
            9,
            [],
            cached_suffix="big_roads",
        )

    assert ok == 1
    assert len(layer.dicosmfirst["w"]) == 1
    wayid = next(iter(layer.dicosmfirst["w"]))
    assert layer.dicosmtags["w"][wayid]["highway"] == "primary"


def test_failed_slice_falls_back_to_overpass(tmp_path):
    layer = OSM.OSM_layer()
    cache_file = tmp_path / "+47+009_coastline.osm.bz2"
    overpass = mock.Mock(return_value=0)

    with mock.patch.object(OSM.PBF, "pbf_ready", return_value=True), \
         mock.patch.object(OSM.PBF, "slice_tile_layer", return_value=None), \
         mock.patch.object(OSM, "get_overpass_data", overpass), \
         mock.patch.object(
             OSM.FNAMES, "osm_cached", return_value=str(cache_file)
         ):
        ok = OSM.OSM_queries_to_OSM_layer(
            ['way["natural"="coastline"]'],
            layer,
            47,
            9,
            [],
            cached_suffix="coastline",
        )

    assert ok == 1
    assert overpass.called
    assert not cache_file.exists()


def test_existing_cache_still_wins_over_pbf(tmp_path):
    layer = OSM.OSM_layer()
    cache_file = tmp_path / "+47+009_big_roads.osm.bz2"
    cache_file.write_bytes(bz2.compress(MINI_OSM_XML.encode("utf-8")))

    with mock.patch.object(
             OSM.PBF,
             "pbf_ready",
             side_effect=AssertionError("PBF must not be consulted"),
         ), \
         mock.patch.object(
             OSM.FNAMES, "osm_cached", return_value=str(cache_file)
         ):
        ok = OSM.OSM_queries_to_OSM_layer(
            ['way["highway"="primary"]'],
            layer,
            47,
            9,
            [],
            cached_suffix="big_roads",
        )

    assert ok == 1
    assert len(layer.dicosmfirst["w"]) == 1
