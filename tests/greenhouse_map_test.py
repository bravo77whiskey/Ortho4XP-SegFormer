"""Greenhouse map prior (O4_Greenhouse_Map) and its use by the field fill."""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

import O4_Agri_Structures as AGRI  # noqa: E402
import O4_Greenhouse_Map as GHMAP  # noqa: E402


def _write_tif(path, data, west, north, res):
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
                       count=1, dtype="uint8", crs="EPSG:4326", nodata=255,
                       transform=from_origin(west, north, res, res)) as ds:
        ds.write(data, 1)


def test_texture_mask_samples_the_map(tmp_path):
    data = np.zeros((100, 100), np.uint8)
    data[:50, :50] = 1      # NW quarter of the 1x1 deg tile is greenhouse
    data[90:, 90:] = 255    # nodata
    _write_tif(str(tmp_path / "a_PCG_Result.tif"), data, 118.0, 37.0, 0.01)
    GHMAP._INDEX_CACHE.clear()
    # texture over the middle of the tile: its NW half lies in the mapped quarter
    m = GHMAP.texture_mask(36.75, 36.25, 118.25, 118.75, 200, 200, root=str(tmp_path))
    assert m is not None and m.shape == (200, 200)
    assert m[:100, :100].all() and not m[100:, :].any() and not m[:, 100:].any()
    assert os.path.exists(tmp_path / GHMAP.INDEX_NAME)
    # outside every map: None
    assert GHMAP.texture_mask(30.5, 30.0, 100.0, 100.5, 10, 10, root=str(tmp_path)) is None


def test_index_picks_up_tiles_added_later(tmp_path):
    data = np.ones((10, 10), np.uint8)
    _write_tif(str(tmp_path / "a.tif"), data, 118.0, 37.0, 0.1)
    GHMAP._INDEX_CACHE.clear()
    assert GHMAP.texture_mask(30.5, 30.0, 100.0, 100.5, 4, 4, root=str(tmp_path)) is None
    # a second tile appears after the index was written (a download still running)
    _write_tif(str(tmp_path / "b.tif"), data, 100.0, 31.0, 0.1)
    (tmp_path / "junk.tif").write_bytes(b"not a tiff")
    GHMAP._INDEX_CACHE.clear()
    m = GHMAP.texture_mask(30.5, 30.0, 100.0, 100.5, 4, 4, root=str(tmp_path))
    assert m is not None and m.all()


def _gh(cx, cy, length, width):
    hl, hw = length / 2, width / 2
    return {"points": [[cx - hl, cy - hw], [cx + hl, cy - hw], [cx + hl, cy + hw], [cx - hl, cy + hw]],
            "center": [cx, cy], "area_m2": length * width, "length_m": length,
            "width_m": width, "confidence": 0.5, "structure": "greenhouse"}


def test_mapped_land_fills_around_one_greenhouse():
    img = np.zeros((400, 400, 3), np.uint8)
    img[:] = (150, 110, 70)
    img[50:350, 50:350] = (150, 160, 160)   # one big film block, one YOLO box
    veg = np.full((400, 400), AGRI.CLASS_AGRICULTURE, np.uint8)
    prior = np.zeros((400, 400), np.uint8)
    prior[40:360, 40:360] = 1
    one = [_gh(200, 70, 280, 30)]
    kept, members, fills = AGRI.greenhouse_fields(one, [], veg, img, 1.0, solar=False, prior=prior)
    assert fills
    area = sum(d["area_m2"] for d in fills) + one[0]["area_m2"]
    assert 0.7 * 300 * 300 <= area <= 1.05 * 300 * 300
    # one box without the map is too few for a field
    assert AGRI.greenhouse_fields(one, [], veg, img, 1.0, solar=False)[2] == []


def test_mapped_land_with_no_greenhouse_today_stays_empty():
    # the map is from 2020: film-grey land it marks but where nothing is
    # detected now (a drained lagoon, cleared plots) fills nothing
    img = np.zeros((400, 400, 3), np.uint8)
    img[:] = (150, 110, 70)
    img[50:350, 50:350] = (150, 160, 160)
    veg = np.full((400, 400), AGRI.CLASS_AGRICULTURE, np.uint8)
    prior = np.zeros((400, 400), np.uint8)
    prior[40:360, 40:360] = 1
    assert AGRI.greenhouse_fields([], [], veg, img, 1.0, solar=False, prior=prior)[2] == []
