"""Build a size-capped Nobara training subset from the Windows manifest.

The current full texture library is too large for the Nobara PC, so this tool
selects whole Ortho4XP tiles until a byte budget is reached, rewrites manifest
paths to Linux remote paths, and writes file-pair lists for later rsync/copy
helpers. It does not copy files.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


DEFAULT_WINDOWS_TILES_ROOT = r"H:\XP12 Addons\Applications\Ortho4XP\Tiles"
DEFAULT_WINDOWS_TRAINING_ROOT = r"H:\model_training"
DEFAULT_REMOTE_TILES_ROOT = "~/model_training/tiles"
DEFAULT_REMOTE_TRAINING_ROOT = "~/model_training"


@dataclass
class TileGroup:
    tile: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    images: dict[str, int] = field(default_factory=dict)
    labels: dict[str, int] = field(default_factory=dict)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def train_count(self) -> int:
        return sum(1 for row in self.rows if row.get("split") == "train")

    @property
    def val_count(self) -> int:
        return sum(1 for row in self.rows if row.get("split") == "val")

    @property
    def byte_count(self) -> int:
        return sum(self.images.values()) + sum(self.labels.values())

    @property
    def useful(self) -> bool:
        return self.train_count > 0 or self.val_count > 0

    @property
    def density(self) -> float:
        if self.byte_count <= 0:
            return 0.0
        return self.row_count / (self.byte_count / (1024 * 1024))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def normalized_path(value: str) -> str:
    return str(value).replace("\\", "/").rstrip("/")


def translate_path(path: str, mappings: list[tuple[str, str]]) -> str:
    normalized = normalized_path(path)
    for local_root, remote_root in sorted(mappings, key=lambda item: len(item[0]), reverse=True):
        local = normalized_path(local_root)
        if normalized == local:
            return normalized_path(remote_root)
        if normalized.lower().startswith((local + "/").lower()):
            suffix = normalized[len(local) :].lstrip("/")
            return normalized_path(remote_root) + "/" + suffix
    raise ValueError(f"No remote path mapping for {path!r}")


def file_size(path: str) -> int:
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def file_exists(path: str) -> bool:
    try:
        return Path(path).is_file()
    except OSError:
        return False


def read_groups(manifest: Path) -> dict[str, TileGroup]:
    groups: dict[str, TileGroup] = {}
    for row in iter_jsonl(manifest):
        tile = str(row.get("tile") or "")
        if not tile:
            continue
        image = row.get("image")
        labels = row.get("labels")
        if not image or not labels or not file_exists(str(image)) or not file_exists(str(labels)):
            continue
        group = groups.setdefault(tile, TileGroup(tile=tile))
        group.rows.append(row)
        if str(image) not in group.images:
            group.images[str(image)] = file_size(str(image))
        if str(labels) not in group.labels:
            group.labels[str(labels)] = file_size(str(labels))
    return groups


def split_density(group: TileGroup, split: str) -> float:
    if group.byte_count <= 0:
        return 0.0
    rows = group.val_count if split == "val" else group.train_count
    return rows / (group.byte_count / (1024 * 1024))


def selection_key(group: TileGroup, split: str) -> tuple[float, int, int, str]:
    score = split_density(group, split)
    return (score, group.val_count, group.train_count, group.tile)


def select_groups(groups: dict[str, TileGroup], budget_bytes: int) -> list[TileGroup]:
    selected: list[TileGroup] = []
    used = 0
    selected_tiles: set[str] = set()

    def add_candidates(candidates: list[TileGroup], limit: int, split: str) -> None:
        nonlocal used
        for group in sorted(candidates, key=lambda item: selection_key(item, split), reverse=True):
            if group.tile in selected_tiles:
                continue
            if used + group.byte_count > budget_bytes or used + group.byte_count > limit:
                continue
            selected.append(group)
            selected_tiles.add(group.tile)
            used += group.byte_count

    candidates = [group for group in groups.values() if group.useful and group.byte_count > 0]
    val_budget = min(int(budget_bytes * 0.15), 10 * 1024 * 1024 * 1024)
    add_candidates([group for group in candidates if group.val_count > 0], val_budget, "val")

    for group in sorted(
        [group for group in candidates if group.train_count > 0 and group.tile not in selected_tiles],
        key=lambda item: selection_key(item, "train"),
        reverse=True,
    ):
        if used + group.byte_count > budget_bytes:
            continue
        selected.append(group)
        selected_tiles.add(group.tile)
        used += group.byte_count
    return sorted(selected, key=lambda group: group.tile)


def rewrite_row(row: dict[str, Any], mappings: list[tuple[str, str]]) -> dict[str, Any]:
    rewritten = dict(row)
    for key in ("image", "source_texture", "labels"):
        if rewritten.get(key):
            rewritten[key] = translate_path(str(rewritten[key]), mappings)
    return rewritten


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def file_pairs(groups: list[TileGroup], mappings: list[tuple[str, str]]) -> list[dict[str, Any]]:
    pairs: dict[str, dict[str, Any]] = {}
    for group in groups:
        for kind, paths in (("image", group.images), ("labels", group.labels)):
            for local_path, size in paths.items():
                pairs[local_path] = {
                    "kind": kind,
                    "tile": group.tile,
                    "local_path": local_path,
                    "remote_path": translate_path(local_path, mappings),
                    "bytes": size,
                }
    return sorted(pairs.values(), key=lambda row: (row["kind"], row["local_path"]))


def relative_to_mapping(path: str, local_root: str) -> str | None:
    normalized = normalized_path(path)
    root = normalized_path(local_root)
    if normalized.lower() == root.lower():
        return ""
    prefix = root + "/"
    if normalized.lower().startswith(prefix.lower()):
        return normalized[len(prefix) :]
    return None


def write_files_from_list(path: Path, values: Iterable[str]) -> None:
    unique = sorted({value.replace("\\", "/") for value in values if value})
    path.write_text("\n".join(unique) + ("\n" if unique else ""), encoding="utf-8")


def build_subset(
    manifest: Path,
    output_dir: Path,
    budget_gb: float,
    mappings: list[tuple[str, str]],
) -> dict[str, Any]:
    groups = read_groups(manifest)
    selected = select_groups(groups, int(budget_gb * 1024 * 1024 * 1024))
    selected_tiles = {group.tile for group in selected}
    rows = [
        rewrite_row(row, mappings)
        for group in selected
        for row in group.rows
        if group.tile in selected_tiles
    ]
    pairs = file_pairs(selected, mappings)

    subset_manifest = output_dir / "dataset.nobara_subset.jsonl"
    write_jsonl(subset_manifest, rows)
    write_jsonl(output_dir / "file_pairs.jsonl", pairs)
    (output_dir / "windows_files.txt").write_text(
        "\n".join(pair["local_path"] for pair in pairs) + ("\n" if pairs else ""),
        encoding="utf-8",
    )
    (output_dir / "remote_files.txt").write_text(
        "\n".join(pair["remote_path"] for pair in pairs) + ("\n" if pairs else ""),
        encoding="utf-8",
    )
    for local_root, remote_root in mappings:
        name = "tiles" if normalized_path(remote_root).endswith("/tiles") else "training"
        write_files_from_list(
            output_dir / f"{name}_files_from.txt",
            (
                rel
                for pair in pairs
                if (rel := relative_to_mapping(pair["local_path"], local_root)) is not None
            ),
        )

    report = {
        "budget_gb": budget_gb,
        "used_gb": round(sum(group.byte_count for group in selected) / (1024**3), 3),
        "selected_tiles": len(selected),
        "selected_rows": len(rows),
        "selected_train_rows": sum(group.train_count for group in selected),
        "selected_val_rows": sum(group.val_count for group in selected),
        "selected_images": sum(len(group.images) for group in selected),
        "selected_label_files": sum(len(group.labels) for group in selected),
        "subset_manifest": str(subset_manifest),
        "file_pairs": str(output_dir / "file_pairs.jsonl"),
        "tiles": [
            {
                "tile": group.tile,
                "rows": group.row_count,
                "train_rows": group.train_count,
                "val_rows": group.val_count,
                "images": len(group.images),
                "bytes": group.byte_count,
                "gb": round(group.byte_count / (1024**3), 3),
            }
            for group in selected
        ],
    }
    (output_dir / "subset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(DEFAULT_WINDOWS_TRAINING_ROOT) / "manifests" / "dataset.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_WINDOWS_TRAINING_ROOT) / "remote_subsets" / "nobara_80gb")
    parser.add_argument("--budget-gb", type=float, default=80.0)
    parser.add_argument("--windows-tiles-root", default=DEFAULT_WINDOWS_TILES_ROOT)
    parser.add_argument("--windows-training-root", default=DEFAULT_WINDOWS_TRAINING_ROOT)
    parser.add_argument("--remote-tiles-root", default=DEFAULT_REMOTE_TILES_ROOT)
    parser.add_argument("--remote-training-root", default=DEFAULT_REMOTE_TRAINING_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mappings = [
        (args.windows_tiles_root, args.remote_tiles_root),
        (args.windows_training_root, args.remote_training_root),
    ]
    report = build_subset(args.manifest, args.output_dir, args.budget_gb, mappings)
    print(json.dumps({key: report[key] for key in report if key != "tiles"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
