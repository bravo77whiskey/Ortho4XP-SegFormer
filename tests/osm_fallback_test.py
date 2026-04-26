from pathlib import Path
import sys
import types
from unittest import mock

from shapely import geometry


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_OSM_Utils as OSM
import O4_Overlay_Utils as OVL


def test_max_osm_tentatives_is_reduced():
    assert OSM.max_osm_tentatives == 4


def test_osm_downloads_are_enabled_by_default():
    assert OSM.disable_osm_downloads is False


def test_failed_big_roads_download_uses_fallback_without_cache_write(tmp_path):
    layer = OSM.OSM_layer()
    cache_path = tmp_path / "+00+000_big_roads.osm.bz2"

    def add_fallback(osm_layer, lat, lon, cached_suffix):
        assert cached_suffix == "big_roads"
        return OSM._add_synthetic_way(
            osm_layer,
            [(0.1, 0.1), (0.2, 0.2)],
            {"highway": "primary"},
        )

    with mock.patch.object(OSM, "get_overpass_data", return_value=0), \
         mock.patch.object(OSM.FNAMES, "osm_cached", return_value=str(cache_path)), \
         mock.patch.object(OSM, "_simulator_network_fallback", side_effect=add_fallback):
        ok = OSM.OSM_queries_to_OSM_layer(
            ['way["highway"="primary"]'],
            layer,
            0,
            0,
            [],
            cached_suffix="big_roads",
        )

    assert ok == 1
    assert len(layer.dicosmfirst["w"]) == 1
    assert not cache_path.exists()


def test_failed_coastline_download_keeps_empty_soft_failure(tmp_path):
    layer = OSM.OSM_layer()
    cache_path = tmp_path / "+00+000_coastline.osm.bz2"

    with mock.patch.object(OSM, "get_overpass_data", return_value=0), \
         mock.patch.object(OSM, "_configured_overlay_dsf", return_value=(None, None)), \
         mock.patch.object(OSM.FNAMES, "osm_cached", return_value=str(cache_path)):
        ok = OSM.OSM_queries_to_OSM_layer(
            ['way["natural"="coastline"]'],
            layer,
            0,
            0,
            [],
            cached_suffix="coastline",
        )

    assert ok == 1
    assert len(layer.dicosmfirst["w"]) == 0
    assert not cache_path.exists()


def test_water_geometry_from_dsf_text_uses_water_patch():
    lines = [
        "TERRAIN_DEF terrain_Water\n",
        "TERRAIN_DEF terrain/land.ter\n",
        "BEGIN_PATCH 0 0.000000 -1.000000 1 7\n",
        "BEGIN_PRIMITIVE 0\n",
        "PATCH_VERTEX 0.100000 0.100000 0 0 0 1 1\n",
        "PATCH_VERTEX 0.400000 0.100000 0 0 0 1 1\n",
        "PATCH_VERTEX 0.100000 0.400000 0 0 0 1 1\n",
        "END_PRIMITIVE\n",
        "END_PATCH\n",
        "BEGIN_PATCH 1 0.000000 -1.000000 1 7\n",
        "BEGIN_PRIMITIVE 0\n",
        "PATCH_VERTEX 0.600000 0.600000 0 0 0\n",
        "PATCH_VERTEX 0.800000 0.600000 0 0 0\n",
        "PATCH_VERTEX 0.600000 0.800000 0 0 0\n",
        "END_PRIMITIVE\n",
        "END_PATCH\n",
    ]

    water = OSM._water_geometry_from_dsf_text(lines, 0, 0)

    assert len(water.geoms) == 1
    assert water.area > 0
    assert water.contains(geometry.Point(0.2, 0.2))
    assert not water.contains(geometry.Point(0.7, 0.7))


def test_configured_overlay_dsf_finds_tile_from_xplane_root(tmp_path):
    xplane_root = tmp_path / "X-Plane 12"
    dsf_dir = (
        xplane_root
        / "Global Scenery"
        / "X-Plane Global Scenery"
        / "Earth nav data"
        / "+50-120"
    )
    dsf_dir.mkdir(parents=True)
    dsf_path = dsf_dir / "+55-119.dsf"
    dsf_path.write_bytes(b"XPLNEDSF")

    with mock.patch.object(OVL, "custom_overlay_src", str(xplane_root)), \
         mock.patch.object(OVL, "custom_overlay_src_alternate", ""):
        found_path, ovl_module = OSM._configured_overlay_dsf(55, -119)

    assert found_path == str(dsf_path)
    assert ovl_module is OVL


def test_configured_overlay_dsf_derives_xplane_root_from_custom_scenery(tmp_path):
    xplane_root = tmp_path / "X-Plane 12"
    custom_scenery = xplane_root / "Custom Scenery"
    custom_scenery.mkdir(parents=True)
    dsf_dir = (
        xplane_root
        / "Global Scenery"
        / "X-Plane 12 Global Scenery"
        / "Earth nav data"
        / "+50-120"
    )
    dsf_dir.mkdir(parents=True)
    dsf_path = dsf_dir / "+55-119.dsf"
    dsf_path.write_bytes(b"XPLNEDSF")
    fake_cfg = types.SimpleNamespace(custom_scenery_dir=str(custom_scenery))

    with mock.patch.object(OVL, "custom_overlay_src", ""), \
         mock.patch.object(OVL, "custom_overlay_src_alternate", ""), \
         mock.patch.dict(sys.modules, {"O4_Config_Utils": fake_cfg}):
        found_path, ovl_module = OSM._configured_overlay_dsf(55, -119)

    assert found_path == str(dsf_path)
    assert ovl_module is OVL


def test_failed_water_download_uses_default_dsf_water_without_cache_write(tmp_path):
    layer = OSM.OSM_layer()
    cache_path = tmp_path / "+00+000_water.osm.bz2"
    water = geometry.MultiPolygon(
        [geometry.Polygon([(0.1, 0.1), (0.4, 0.1), (0.1, 0.4), (0.1, 0.1)])]
    )

    with mock.patch.object(OSM, "get_overpass_data", return_value=0), \
         mock.patch.object(OSM, "_configured_overlay_dsf", return_value=("default.dsf", object())), \
         mock.patch.object(OSM, "_parse_simulator_water_dsf", return_value=water), \
         mock.patch.object(OSM.FNAMES, "osm_cached", return_value=str(cache_path)):
        ok = OSM.OSM_queries_to_OSM_layer(
            ['way["natural"="water"]'],
            layer,
            0,
            0,
            [],
            cached_suffix="water",
        )

    assert ok == 1
    assert len(layer.dicosmfirst["r"]) == 1
    rel_id = next(iter(layer.dicosmfirst["r"]))
    assert layer.dicosmtags["r"][rel_id]["source"] == "xplane_default_scenery"
    assert not cache_path.exists()


def test_failed_coastline_download_skips_fallback(tmp_path):
    layer = OSM.OSM_layer()
    cache_path = tmp_path / "+00+000_coastline.osm.bz2"

    with mock.patch.object(OSM, "get_overpass_data", return_value=0), \
         mock.patch.object(OSM.FNAMES, "osm_cached", return_value=str(cache_path)):
        ok = OSM.OSM_queries_to_OSM_layer(
            ['way["natural"="coastline"]'],
            layer,
            0,
            0,
            [],
            cached_suffix="coastline",
        )

    assert ok == 1
    assert len(layer.dicosmfirst["w"]) == 0
    assert not cache_path.exists()


def test_failed_airport_download_uses_apt_dat_without_cache_write(tmp_path):
    apt_dat = tmp_path / "apt.dat"
    apt_dat.write_text(
        "\n".join(
            [
                "I",
                "1000 Version",
                "1 100 0 0 TEST Test Airport",
                (
                    "100 45.0 1 0 0.25 1 2 1 16 "
                    "0.250000 0.250000 0 0 0 0 34 "
                    "0.750000 0.750000 0 0 0 0"
                ),
                "99",
            ]
        ),
        encoding="utf-8",
    )
    layer = OSM.OSM_layer()
    cache_path = tmp_path / "+00+000_airports.osm.bz2"

    with mock.patch.object(OSM, "get_overpass_data", return_value=0), \
         mock.patch.object(OSM, "_candidate_apt_dat_files", return_value=[str(apt_dat)]), \
         mock.patch.object(OSM.FNAMES, "osm_cached", return_value=str(cache_path)):
        ok = OSM.OSM_queries_to_OSM_layer(
            [('node["aeroway"]', 'way["aeroway"]', 'rel["aeroway"]')],
            layer,
            0,
            0,
            ["all"],
            cached_suffix="airports",
        )

    assert ok == 1
    assert len(layer.dicosmfirst["n"]) == 1
    assert len(layer.dicosmfirst["w"]) == 1
    runway_id = next(iter(layer.dicosmfirst["w"]))
    assert layer.dicosmtags["w"][runway_id]["aeroway"] == "runway"
    assert layer.dicosmtags["w"][runway_id]["width"] == "45.0"
    assert not cache_path.exists()
