import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REMOTE = ROOT / "training" / "remote"
if str(REMOTE) not in sys.path:
    sys.path.insert(0, str(REMOTE))

import build_nobara_subset as subset


def _write(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def test_build_subset_rewrites_paths_and_emits_rsync_lists(tmp_path):
    tiles_root = tmp_path / "Tiles"
    training_root = tmp_path / "model_training"
    manifest = training_root / "manifests" / "dataset.jsonl"

    small_image = _write(tiles_root / "zOrtho4XP_+01+001" / "textures" / "1_1_BI16.dds", 100)
    small_label = _write(training_root / "labels" / "tiles" / "zOrtho4XP_+01+001_labels.sqlite", 100)
    large_image = _write(tiles_root / "zOrtho4XP_+02+002" / "textures" / "2_2_BI16.dds", 900)
    large_label = _write(training_root / "labels" / "tiles" / "zOrtho4XP_+02+002_labels.sqlite", 200)

    _append_jsonl(
        manifest,
        {
            "chip_id": "small_train",
            "tile": "zOrtho4XP_+01+001",
            "split": "train",
            "region": "standard",
            "image": str(small_image),
            "source_texture": str(small_image),
            "labels": str(small_label),
        },
    )
    _append_jsonl(
        manifest,
        {
            "chip_id": "small_val",
            "tile": "zOrtho4XP_+01+001",
            "split": "val",
            "region": "standard",
            "image": str(small_image),
            "source_texture": str(small_image),
            "labels": str(small_label),
        },
    )
    _append_jsonl(
        manifest,
        {
            "chip_id": "large_train",
            "tile": "zOrtho4XP_+02+002",
            "split": "train",
            "region": "standard",
            "image": str(large_image),
            "source_texture": str(large_image),
            "labels": str(large_label),
        },
    )

    output_dir = tmp_path / "subset"
    report = subset.build_subset(
        manifest,
        output_dir,
        budget_gb=500 / (1024**3),
        mappings=[
            (str(tiles_root), "~/model_training/tiles"),
            (str(training_root), "~/model_training"),
        ],
    )

    assert report["selected_tiles"] == 1
    assert report["selected_rows"] == 2
    rows = [json.loads(line) for line in (output_dir / "dataset.nobara_subset.jsonl").read_text().splitlines()]
    assert {row["chip_id"] for row in rows} == {"small_train", "small_val"}
    assert rows[0]["image"].startswith("~/model_training/tiles/")
    assert rows[0]["source_texture"].startswith("~/model_training/tiles/")
    assert rows[0]["labels"].startswith("~/model_training/labels/")
    assert (output_dir / "tiles_files_from.txt").read_text(encoding="utf-8").strip() == (
        "zOrtho4XP_+01+001/textures/1_1_BI16.dds"
    )
    assert (output_dir / "training_files_from.txt").read_text(encoding="utf-8").strip() == (
        "labels/tiles/zOrtho4XP_+01+001_labels.sqlite"
    )


def test_translate_path_requires_known_mapping():
    try:
        subset.translate_path(r"Z:\elsewhere\file.dds", [(r"H:\model_training", "~/model_training")])
    except ValueError as exc:
        assert "No remote path mapping" in str(exc)
    else:
        raise AssertionError("translate_path should reject unmapped paths")
