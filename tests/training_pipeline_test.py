import json
import gzip
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import Polygon, box

import sys

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

import building_training_lib as lib
import build_yolo_obb_dataset as yolo_build


def test_orthogrid_round_trip_bounds_contains_point():
    lat, lon = 37.7749, -122.4194
    tile_x, tile_y = lib.wgs84_to_orthogrid(lat, lon)
    west, south, east, north = lib.texture_bounds(tile_x, tile_y)
    assert west <= lon <= east
    assert south <= lat <= north


def test_rasterize_feature_preserves_individual_footprint_shape():
    bounds = (0.0, 0.0, 1.0, 1.0)
    feature = lib.Feature("test", box(0.25, 0.25, 0.75, 0.75))
    mask = lib.rasterize_features([feature], bounds, chip_size=100)
    arr = np.asarray(mask)
    assert arr[50, 50] == 255
    assert arr[5, 5] == 0
    assert 2400 <= (arr > 0).sum() <= 2700


def test_orientation_error_uses_minimum_rotated_rectangle():
    poly = Polygon([(0, 0), (10, 0), (10, 2), (0, 2)])
    rotated = lib.affinity.rotate(poly, 30, origin="centroid")
    angle_a = lib.dominant_angle_degrees(poly)
    angle_b = lib.dominant_angle_degrees(rotated)
    assert angle_a is not None
    assert angle_b is not None
    assert lib.rotation_error_degrees(angle_a, angle_b) == pytest.approx(30.0)


def test_temporary_texture_cleanup_only_removes_temp_file(tmp_path):
    temp_path = tmp_path / "temp.jpg"
    keep_path = tmp_path / "existing.jpg"
    Image.new("RGB", (8, 8)).save(temp_path)
    Image.new("RGB", (8, 8)).save(keep_path)
    lib.cleanup_texture(lib.TextureRef(Image.new("RGB", (8, 8)), temp_path, True))
    lib.cleanup_texture(lib.TextureRef(Image.new("RGB", (8, 8)), keep_path, False))
    assert not temp_path.exists()
    assert keep_path.exists()


def test_append_manifest_and_load_completed_ids(tmp_path):
    manifest = tmp_path / "dataset.jsonl"
    lib.append_jsonl(manifest, {"chip_id": "abc", "split": "train"})
    lib.append_jsonl(manifest, {"chip_id": "def", "split": "val"})
    assert lib.load_completed_ids(manifest) == {"abc", "def"}


@pytest.mark.skipif(lib.rasterio_missing if hasattr(lib, "rasterio_missing") else False, reason="rasterio unavailable")
def test_cbra_pixel_policy_can_be_represented_in_mask():
    arr = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    mask = Image.fromarray((arr == 255).astype(np.uint8) * 255, mode="L")
    assert np.asarray(mask).tolist() == [[0, 255], [255, 0]]


def test_microsoft_geojsonl_csv_gz_parts_are_streamed(tmp_path):
    part = tmp_path / "part.csv.gz"
    record = {
        "type": "Feature",
        "properties": {"height": 12.5, "confidence": 0.91},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [100.0, 1.0],
                [100.001, 1.0],
                [100.001, 1.001],
                [100.0, 1.001],
                [100.0, 1.0],
            ]],
        },
    }
    with gzip.open(part, "wt", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    features = list(lib._stream_microsoft_part(part, "Singapore"))
    assert len(features) == 1
    assert features[0].source == "microsoft"
    assert features[0].height == pytest.approx(12.5)
    assert features[0].confidence == pytest.approx(0.91)
    assert features[0].properties["location"] == "Singapore"


def test_asia_dense_helpers_parse_size_and_bounds():
    assert lib.parse_size_bytes("1.5MB") == pytest.approx(1.5 * 1024 * 1024)
    feature = lib.Feature("test", box(103.0, 1.0, 103.01, 1.01))
    assert lib.feature_centroid_in_bounds(feature, lib.ASIA_DENSE_BOUNDS)


def test_holdout_regions_are_split_away_from_training(tmp_path):
    for name in (
        "zOrtho4XP_-10-060",
        "zOrtho4XP_+35+139",
        "zOrtho4XP_+45+007",
    ):
        (tmp_path / name).mkdir()

    records = lib.discover_tiles(tmp_path, seed=1, val_fraction=0.0, test_fraction=0.0)
    by_name = {record.name: record for record in records}

    assert by_name["zOrtho4XP_-10-060"].split == "holdout"
    assert by_name["zOrtho4XP_-10-060"].region == "south_america"
    assert by_name["zOrtho4XP_+35+139"].split == "holdout"
    assert by_name["zOrtho4XP_+35+139"].region == "east_asia"
    assert by_name["zOrtho4XP_+45+007"].split == "train"


def test_simheaven_dsf_text_parser_extracts_objects_and_facades(tmp_path):
    text = tmp_path / "tile.txt"
    text.write_text(
        "\n".join(
            [
                "OBJECT_DEF simheaven/houses/house_09x12x2.obj",
                "POLYGON_DEF simheaven/facades/residential.fac",
                "OBJECT 0 7.5000000 45.5000000 30.0",
                "BEGIN_POLYGON 0 8.0 2",
                "BEGIN_WINDING",
                "POLYGON_POINT 7.6000000 45.6000000",
                "POLYGON_POINT 7.6010000 45.6000000",
                "POLYGON_POINT 7.6010000 45.6010000",
                "POLYGON_POINT 7.6000000 45.6010000",
                "POLYGON_POINT 7.6000000 45.6000000",
                "END_WINDING",
                "END_POLYGON",
            ]
        ),
        encoding="utf-8",
    )

    features = lib.parse_simheaven_text(text)

    assert len(features) == 2
    assert features[0].asset_path == "simheaven/houses/house_09x12x2.obj"
    assert features[0].width_m == pytest.approx(9.0)
    assert features[0].depth_m == pytest.approx(12.0)
    assert features[0].height == pytest.approx(6.4)
    assert features[1].asset_path == "simheaven/facades/residential.fac"
    assert features[1].height == pytest.approx(8.0)
    assert features[1].geometry.area > 0


def test_yolo_height_bins_encode_raw_heights_without_max_generated_clamp():
    tall_class = lib.encode_yolo_obb_class(
        lib.CLASS_APARTMENT_BLOCK,
        300.0,
        height_labels=lib.YOLO_HEIGHT_LABEL_BINS,
    )
    placement, height_m = lib.decode_yolo_obb_class(
        tall_class,
        height_labels=lib.YOLO_HEIGHT_LABEL_BINS,
    )

    assert placement == lib.CLASS_APARTMENT_BLOCK
    assert height_m == pytest.approx(120.0)
    assert lib.yolo_obb_class_names(lib.YOLO_HEIGHT_LABEL_BINS)[
        lib.encode_yolo_obb_class(
            lib.CLASS_SMALL_RESIDENTIAL,
            6.0,
            height_labels=lib.YOLO_HEIGHT_LABEL_BINS,
        )
    ] == "small_residential__h06"


def test_build_chip_labels_preserve_dimensions_and_heading():
    feature = lib.Feature(
        source="simheaven",
        geometry=lib.oriented_footprint(0.5, 0.5, 10.0, 30.0, 90.0),
        width_m=10.0,
        depth_m=30.0,
        heading_deg=90.0,
        placement_class=lib.CLASS_SMALL_RESIDENTIAL,
    )

    labels = lib.build_chip_labels([feature], (0.0, 0.0, 1.0, 1.0), chip_size=64)
    dims = np.asarray(labels["dims"])
    heading = np.asarray(labels["heading"])
    klass = np.asarray(labels["placement_class"])

    occupied = np.asarray(labels["mask"]) > 0
    assert occupied.any()
    assert int(np.median(dims[occupied, 0])) == 20
    assert int(np.median(dims[occupied, 1])) == 60
    assert int(np.median(heading[occupied, 0])) > 250
    assert int(np.median(klass[occupied])) == lib.CLASS_SMALL_RESIDENTIAL


def test_build_chip_labels_can_use_native_texture_shape():
    feature = lib.Feature(
        source="simheaven",
        geometry=box(0.25, 0.25, 0.75, 0.75),
        width_m=20.0,
        depth_m=40.0,
        heading_deg=0.0,
        placement_class=lib.CLASS_MEDIUM,
    )

    labels = lib.build_chip_labels([feature], (0.0, 0.0, 1.0, 1.0), chip_size=(96, 64))

    assert labels["mask"].size == (96, 64)
    assert labels["dims"].size == (96, 64)


def test_vector_labels_are_compact_and_rasterized_per_training_crop(tmp_path):
    torch = pytest.importorskip("torch")
    image = tmp_path / "native.png"
    Image.new("RGB", (128, 96), color=(30, 40, 50)).save(image)
    feature = lib.Feature(
        source="simheaven",
        geometry=box(0.35, 0.35, 0.65, 0.65),
        height=18.5,
        width_m=16.0,
        depth_m=24.0,
        heading_deg=45.0,
        placement_class=lib.CLASS_MEDIUM,
    )
    label_path = lib._save_vector_labels(
        [feature],
        tmp_path / "labels",
        "native",
        (0.0, 0.0, 1.0, 1.0),
        (128, 96),
    )
    row = {
        "chip_id": "native",
        "image": str(image),
        "labels": str(label_path),
        "west": 0.0,
        "south": 0.0,
        "east": 1.0,
        "north": 1.0,
    }

    dataset = lib.PlacementDataset([row], chip_size=64, random_crop=False)
    item = dataset[0]

    assert label_path.name.endswith(".json.gz")
    assert item["image"].shape == (3, 64, 64)
    assert item["mask"].shape == (1, 64, 64)
    assert float(item["mask"].sum()) > 0


def test_sqlite_labels_are_indexed_and_rasterized_per_training_crop(tmp_path):
    torch = pytest.importorskip("torch")
    image = tmp_path / "native.png"
    Image.new("RGB", (128, 96), color=(30, 40, 50)).save(image)
    feature = lib.Feature(
        source="simheaven",
        geometry=box(0.35, 0.35, 0.65, 0.65),
        height=18.5,
        width_m=16.0,
        depth_m=24.0,
        heading_deg=45.0,
        placement_class=lib.CLASS_MEDIUM,
    )
    label_path = lib._save_sqlite_labels(
        [feature],
        tmp_path / "labels",
        "native",
        (0.0, 0.0, 1.0, 1.0),
    )
    row = {
        "chip_id": "native",
        "image": str(image),
        "labels": str(label_path),
        "west": 0.0,
        "south": 0.0,
        "east": 1.0,
        "north": 1.0,
    }

    assert label_path.name.endswith(".sqlite")
    assert lib._sqlite_has_features(label_path, (0.0, 0.0, 1.0, 1.0))
    loaded = lib._features_from_sqlite_for_bounds(label_path, (0.0, 0.0, 1.0, 1.0))
    assert loaded[0].height == pytest.approx(18.5)

    dataset = lib.PlacementDataset([row], chip_size=64, random_crop=False)
    item = dataset[0]

    assert item["image"].shape == (3, 64, 64)
    assert item["mask"].shape == (1, 64, 64)
    assert float(item["mask"].sum()) > 0


def test_v1_sqlite_labels_remain_readable_without_height(tmp_path):
    label_path = tmp_path / "legacy.sqlite"
    connection = sqlite3.connect(label_path)
    try:
        connection.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute(
            """
            CREATE TABLE features (
                id INTEGER PRIMARY KEY,
                asset_path TEXT,
                heading REAL,
                width REAL,
                depth REAL,
                class INTEGER NOT NULL,
                geom BLOB NOT NULL
            )
            """
        )
        connection.execute("CREATE VIRTUAL TABLE idx USING rtree(id, minx, maxx, miny, maxy)")
        geom = box(0.25, 0.25, 0.75, 0.75)
        connection.executemany(
            "INSERT INTO meta(key, value) VALUES (?, ?)",
            [
                ("count", "1"),
                ("format", "simheaven-placement-sqlite-v1"),
            ],
        )
        connection.execute(
            "INSERT INTO features(id, asset_path, heading, width, depth, class, geom) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (1, "simheaven/facades/residential.fac", 0.0, 20.0, 20.0, lib.CLASS_MEDIUM, sqlite3.Binary(geom.wkb)),
        )
        connection.execute("INSERT INTO idx(id, minx, maxx, miny, maxy) VALUES (?, ?, ?, ?, ?)", (1, 0.25, 0.75, 0.25, 0.75))
        connection.commit()
    finally:
        connection.close()

    assert lib._sqlite_label_cache_is_valid(label_path)
    assert not lib._sqlite_label_cache_is_valid(label_path, require_height=True)
    loaded = lib._features_from_sqlite_for_bounds(label_path, (0.0, 0.0, 1.0, 1.0))
    assert len(loaded) == 1
    assert loaded[0].height is None


def test_obb_labels_can_expand_classes_with_height_bins():
    feature = lib.Feature(
        source="simheaven",
        geometry=box(0.25, 0.25, 0.75, 0.75),
        height=15.0,
        placement_class=lib.CLASS_MEDIUM,
    )

    labels = yolo_build._obb_labels_for_crop(
        [feature],
        (0.0, 0.0, 1.0, 1.0),
        64,
        height_labels=lib.YOLO_HEIGHT_LABEL_BINS,
    )

    expected_class = lib.encode_yolo_obb_class(
        lib.CLASS_MEDIUM,
        15.0,
        height_labels=lib.YOLO_HEIGHT_LABEL_BINS,
    )
    assert labels
    assert labels[0].split()[0] == str(expected_class)


def test_training_can_pause_and_resume_from_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    manifest = tmp_path / "dataset.jsonl"
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()

    for split, idx in (("train", 0), ("val", 1)):
        image = image_dir / f"{split}.png"
        Image.new("RGB", (32, 32), color=(40 + idx, 50, 60)).save(image)
        labels = lib.build_chip_labels(
            [
                lib.Feature(
                    source="simheaven",
                    geometry=box(0.25, 0.25, 0.75, 0.75),
                    width_m=12.0,
                    depth_m=18.0,
                    heading_deg=0.0,
                    placement_class=lib.CLASS_COMPACT_RESIDENTIAL,
                )
            ],
            (0.0, 0.0, 1.0, 1.0),
            chip_size=32,
        )
        paths = {}
        for name, label in labels.items():
            path = label_dir / f"{split}_{name}.png"
            label.save(path)
            paths[name] = path
        lib.append_jsonl(
            manifest,
            {
                "chip_id": split,
                "split": split,
                "tile": "synthetic",
                "region": "standard",
                "image": str(image),
                "center": str(paths["center"]),
                "mask": str(paths["mask"]),
                "dims": str(paths["dims"]),
                "heading": str(paths["heading"]),
                "placement_class": str(paths["placement_class"]),
            },
        )

    run_dir = tmp_path / "run"
    pause_file = run_dir / "PAUSE"
    run_dir.mkdir()
    pause_file.write_text("pause after first epoch", encoding="utf-8")
    first = lib.train(
        lib.TrainConfig(
            dataset_manifest=manifest,
            run_dir=run_dir,
            epochs=2,
            batch_size=1,
            chip_size=32,
            device="cpu",
            pause_file=pause_file,
        )
    )
    assert first["start_epoch"] == 1
    assert (run_dir / "last.pt").exists()

    pause_file.unlink()
    second = lib.train(
        lib.TrainConfig(
            dataset_manifest=manifest,
            run_dir=run_dir,
            epochs=2,
            batch_size=1,
            chip_size=32,
            device="cpu",
        )
    )
    assert second["start_epoch"] == 2
    assert (run_dir / "best.pt").exists()
