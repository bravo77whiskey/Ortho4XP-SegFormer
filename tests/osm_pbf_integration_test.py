import bz2
import contextlib
import json
import os
from pathlib import Path
import sys
import time
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


def fake_run_tool_factory(recorded, configs=None):
    def fake_run_tool(cmd, log_prefix="      "):
        recorded.append(cmd)
        if "tags-filter" in cmd:
            out_path = cmd[cmd.index("-o") + 1]
            with open(out_path, "wb") as f:
                f.write(bz2.compress(MINI_OSM_XML.encode("utf-8")))
            return 0
        with open(cmd[cmd.index("-c") + 1], "r", encoding="utf-8") as f:
            config = json.load(f)
        if configs is not None:
            configs.append(config)
        for extract in config["extracts"]:
            with open(extract["output"], "wb") as f:
                f.write(b"fake pbf")
        return 0

    return fake_run_tool


def make_old_extract(tmp_path):
    extract = tmp_path / "scenery.osm.pbf"
    extract.write_bytes(b"x")
    old = time.time() - 3600
    os.utime(extract, (old, old))
    return extract


def pbf_patches(tmp_path, recorded, configs=None, cache_path=None):
    return [
        mock.patch.object(PBF, "osm_pbf_dir", str(tmp_path)),
        mock.patch.object(PBF, "tools_available", return_value=True),
        mock.patch.object(PBF, "osmium_path", return_value="osmium"),
        mock.patch.object(
            PBF,
            "_run_tool",
            side_effect=fake_run_tool_factory(recorded, configs),
        ),
        mock.patch.object(PBF.FNAMES, "Tmp_dir", str(tmp_path / "tmp")),
        mock.patch.object(
            PBF.FNAMES,
            "osm_cached",
            side_effect=cache_path
            or (lambda lat, lon, suffix: str(
                tmp_path / "OSM_data" / (suffix + ".osm.bz2")
            )),
        ),
    ]


def test_slice_tile_layer_command_shapes_and_cache_placement(tmp_path):
    recorded = []
    configs = []
    cache_file = tmp_path / "OSM_data" / "+47+009_big_roads.osm.bz2"
    extract = make_old_extract(tmp_path)

    with contextlib.ExitStack() as stack, \
         mock.patch.object(PBF, "block_degrees", []):
        for patch in pbf_patches(
            tmp_path, recorded, configs, lambda *a: str(cache_file)
        ):
            stack.enter_context(patch)
        result = PBF.slice_tile_layer(47, 9, "big_roads")

    assert result == str(cache_file)
    assert cache_file.is_file()
    with bz2.open(cache_file, "rt", encoding="utf-8") as f:
        assert "<osm " in f.read()

    extract_cmd = recorded[0]
    assert "extract" in extract_cmd
    assert extract_cmd[extract_cmd.index("-s") + 1] == "smart"
    assert str(extract) in extract_cmd
    (tile,) = configs[0]["extracts"]
    assert tile["bbox"] == [9, 47, 10, 48]
    assert tile["output_format"] == "pbf"

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
    make_old_extract(tmp_path)

    with contextlib.ExitStack() as stack:
        for patch in pbf_patches(tmp_path, recorded):
            stack.enter_context(patch)
        assert PBF.slice_tile_layer(47, 9, "big_roads")
        assert PBF.slice_tile_layer(47, 9, "water")

    extract_calls = [cmd for cmd in recorded if "extract" in cmd]
    filter_calls = [cmd for cmd in recorded if "tags-filter" in cmd]
    assert len(extract_calls) == 1
    assert len(filter_calls) == 2


def test_first_tile_of_a_region_writes_its_blocks_in_one_pass(tmp_path):
    recorded = []
    configs = []
    extract = make_old_extract(tmp_path)

    with contextlib.ExitStack() as stack, \
         mock.patch.object(PBF, "block_degrees", [30, 5]):
        for patch in pbf_patches(tmp_path, recorded, configs):
            stack.enter_context(patch)
        assert PBF.slice_tile_layer(-30, -54, "airports")

    (extract_cmd,) = [cmd for cmd in recorded if "extract" in cmd]
    assert str(extract) in extract_cmd
    bboxes = [e["bbox"] for e in configs[0]["extracts"]]
    assert bboxes == [[-60, -30, -30, 0], [-55, -30, -50, -25], [-54, -30, -53, -29]]
    blocks = tmp_path / "blocks"
    assert (blocks / "30deg_-30-060.osm.pbf").is_file()
    assert (blocks / "5deg_-30-055.osm.pbf").is_file()
    assert not list(blocks.glob("*.part"))


def test_next_tile_reads_the_smallest_current_block(tmp_path):
    recorded = []
    configs = []
    make_old_extract(tmp_path)

    with contextlib.ExitStack() as stack, \
         mock.patch.object(PBF, "block_degrees", [30, 5]):
        for patch in pbf_patches(tmp_path, recorded, configs):
            stack.enter_context(patch)
        assert PBF.slice_tile_layer(-30, -54, "airports")
        # Same 5 degree block.
        assert PBF.slice_tile_layer(-29, -51, "airports")
        # Same 30 degree block, new 5 degree block.
        assert PBF.slice_tile_layer(-21, -45, "airports")

    extract_cmds = [cmd for cmd in recorded if "extract" in cmd]
    blocks = tmp_path / "blocks"
    assert str(blocks / "5deg_-30-055.osm.pbf") in extract_cmds[1]
    assert [e["bbox"] for e in configs[1]["extracts"]] == [[-51, -29, -50, -28]]
    assert str(blocks / "30deg_-30-060.osm.pbf") in extract_cmds[2]
    assert [e["bbox"] for e in configs[2]["extracts"]] == [
        [-45, -25, -40, -20],
        [-45, -21, -44, -20],
    ]


def test_blocks_older_than_the_extract_are_ignored(tmp_path):
    recorded = []
    configs = []
    extract = make_old_extract(tmp_path)

    with contextlib.ExitStack() as stack, \
         mock.patch.object(PBF, "block_degrees", [30, 5]):
        for patch in pbf_patches(tmp_path, recorded, configs):
            stack.enter_context(patch)
        assert PBF.slice_tile_layer(-30, -54, "airports")
        future = time.time() + 60
        os.utime(extract, (future, future))
        assert PBF.slice_tile_layer(-29, -51, "airports")

    extract_cmds = [cmd for cmd in recorded if "extract" in cmd]
    assert str(extract) in extract_cmds[1]
    assert len(configs[1]["extracts"]) == 3


def test_failed_cut_leaves_no_blocks(tmp_path):
    make_old_extract(tmp_path)

    def failing_run_tool(cmd, log_prefix="      "):
        with open(cmd[cmd.index("-c") + 1], "r", encoding="utf-8") as f:
            first = json.load(f)["extracts"][0]
        with open(first["output"], "wb") as f:
            f.write(b"partial")
        return 1

    with contextlib.ExitStack() as stack, \
         mock.patch.object(PBF, "block_degrees", [30, 5]):
        for patch in pbf_patches(tmp_path, []):
            stack.enter_context(patch)
        stack.enter_context(
            mock.patch.object(PBF, "_run_tool", side_effect=failing_run_tool)
        )
        assert PBF.slice_tile_layer(-30, -54, "airports") is None

    assert not list((tmp_path / "blocks").iterdir())


def test_block_levels_drop_sizes_that_do_not_nest():
    with mock.patch.object(PBF, "block_degrees", [5, 30, 7, 10]):
        assert PBF._block_levels() == [30, 10, 5]


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
