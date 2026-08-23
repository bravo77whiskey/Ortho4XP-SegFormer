"""Add Global Forests compatibility aliases to simHeaven's library.

The tool leaves vegetation DSFs untouched. It duplicates simHeaven's own
seasonal and climate-regional forest exports under the legacy paths found in
an overlay package. Dry-run is the default; --write creates a timestamped
backup and atomically replaces library.txt.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import tempfile

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import O4_Forest_Assets as FOREST_ASSETS


BEGIN_MARKER = "# BEGIN ORTHO4XP GFV2 TO SIMHEAVEN REDIRECTS"
END_MARKER = "# END ORTHO4XP GFV2 TO SIMHEAVEN REDIRECTS"
SOURCE_VIRTUALS = {
    "broad": "simheaven/forests/broad.for",
    "coni": "simheaven/forests/coni.for",
    "mixed": "simheaven/forests/mixed.for",
}
GFV2_BINARY_RE = re.compile(
    rb"(?<!lib/vegetation/)forests/[A-Za-z0-9_./-]+\.for",
    re.IGNORECASE,
)
EXPORT_RE = re.compile(
    r"^(?P<command>EXPORT_SEASON)\s+"
    r"(?P<season>spr|sum|fal|win)\s+"
    r"(?P<virtual>\S+)\s+"
    r"(?P<real>.+?)\s*$",
    re.IGNORECASE,
)
BACKUP_FOLDER = "_asset_rewrite_backups"


def active_dsfs(overlay_root: Path) -> list[Path]:
    return sorted(
        path
        for path in overlay_root.rglob("*.dsf")
        if BACKUP_FOLDER not in path.relative_to(overlay_root).parts
    )


def legacy_paths_in_overlay(overlay_root: Path) -> tuple[str, ...]:
    paths = set()
    for dsf_path in active_dsfs(overlay_root):
        data = dsf_path.read_bytes()
        paths.update(
            match.group(0).decode("utf-8", errors="strict")
            for match in GFV2_BINARY_RE.finditer(data)
            if FOREST_ASSETS.parse_gfv2_path(
                match.group(0).decode("utf-8", errors="strict")
            )
            is not None
        )
    return tuple(sorted(paths))


def source_kind_for_legacy_path(path: str) -> str:
    metadata = FOREST_ASSETS.parse_gfv2_path(path)
    if metadata is None:
        raise ValueError(f"Unsupported Global Forests path: {path}")
    family = metadata["family"]
    if any(
        token in family
        for token in ("conifer", "needle", "pine", "spruce", "fir")
    ):
        return "coni"
    if family in {"mixed", "woodland"}:
        return "mixed"
    return "broad"


def remove_existing_redirect_blocks(text: str) -> str:
    lines = text.splitlines()
    cleaned: list[str] = []
    inside = False
    for line in lines:
        if line == BEGIN_MARKER:
            if inside:
                raise RuntimeError("Nested Ortho4XP redirect markers found")
            inside = True
            continue
        if line == END_MARKER:
            if not inside:
                raise RuntimeError("Unmatched Ortho4XP redirect end marker found")
            inside = False
            continue
        if not inside:
            cleaned.append(line)
    if inside:
        raise RuntimeError("Unclosed Ortho4XP redirect marker found")
    return "\n".join(cleaned).rstrip() + "\n"


def parse_export(line: str) -> tuple[str, str, str, str] | None:
    match = EXPORT_RE.match(line)
    if match is None:
        return None
    return (
        match.group("command").upper(),
        match.group("season").lower(),
        match.group("virtual").replace("\\", "/").lower(),
        match.group("real").strip(),
    )


def build_redirected_library(
    source_text: str,
    legacy_paths: tuple[str, ...],
) -> tuple[str, dict[str, int]]:
    clean_text = remove_existing_redirect_blocks(source_text)
    aliases_by_virtual: dict[str, tuple[str, ...]] = {}
    grouped: dict[str, list[str]] = defaultdict(list)
    for legacy_path in legacy_paths:
        grouped[source_kind_for_legacy_path(legacy_path)].append(legacy_path)
    for kind, paths in grouped.items():
        aliases_by_virtual[SOURCE_VIRTUALS[kind]] = tuple(sorted(paths))

    output: list[str] = []
    source_counts: Counter[str] = Counter()
    alias_counts: Counter[str] = Counter()
    for line in clean_text.splitlines():
        output.append(line)
        parsed = parse_export(line)
        if parsed is None:
            continue
        command, season, virtual, real_path = parsed
        aliases = aliases_by_virtual.get(virtual)
        if not aliases:
            continue
        source_counts[virtual] += 1
        output.append(BEGIN_MARKER)
        for alias in aliases:
            output.append(f"{command}\t{season}\t{alias}\t{real_path}")
            alias_counts[alias] += 1
        output.append(END_MARKER)

    missing_sources = sorted(
        virtual for virtual in aliases_by_virtual if source_counts[virtual] == 0
    )
    if missing_sources:
        raise RuntimeError(
            "simHeaven source exports were not found: " + ", ".join(missing_sources)
        )
    for virtual, aliases in aliases_by_virtual.items():
        expected = source_counts[virtual]
        for alias in aliases:
            if alias_counts[alias] != expected:
                raise RuntimeError(
                    f"Incomplete redirect set for {alias}: "
                    f"{alias_counts[alias]} of {expected}"
                )

    return "\n".join(output).rstrip() + "\n", dict(alias_counts)


def validate_real_paths(library_root: Path, text: str) -> int:
    checked: set[str] = set()
    missing: list[str] = []
    for line in text.splitlines():
        parsed = parse_export(line)
        if parsed is None:
            continue
        _command, _season, virtual, real_path = parsed
        if virtual not in SOURCE_VIRTUALS.values():
            continue
        normalized = real_path.replace("\\", "/")
        if normalized in checked:
            continue
        checked.add(normalized)
        if not (library_root / Path(normalized)).is_file():
            missing.append(normalized)
    if missing:
        raise RuntimeError(
            "simHeaven forest targets are missing: " + ", ".join(sorted(missing))
        )
    return len(checked)


def redirect_season_coverage(text: str, legacy_paths: tuple[str, ...]) -> None:
    coverage: dict[str, set[str]] = defaultdict(set)
    for line in text.splitlines():
        parsed = parse_export(line)
        if parsed is None:
            continue
        _command, season, virtual, _real_path = parsed
        if virtual in legacy_paths:
            coverage[virtual].add(season)
    required = {"spr", "sum", "fal", "win"}
    incomplete = {
        path: sorted(required - coverage[path])
        for path in legacy_paths
        if coverage[path] != required
    }
    if incomplete:
        details = ", ".join(
            f"{path}: missing {seasons}" for path, seasons in sorted(incomplete.items())
        )
        raise RuntimeError(f"Incomplete seasonal redirect coverage: {details}")


def install_redirects(
    library_path: Path,
    overlay_root: Path,
    write: bool,
) -> dict:
    source_text = library_path.read_text(encoding="utf-8-sig", errors="strict")
    legacy_paths = legacy_paths_in_overlay(overlay_root)
    if not legacy_paths:
        raise RuntimeError(f"No Global Forests paths found under {overlay_root}")

    patched_text, alias_counts = build_redirected_library(source_text, legacy_paths)
    redirect_season_coverage(patched_text, legacy_paths)
    physical_targets = validate_real_paths(library_path.parent, patched_text)
    report = {
        "library": str(library_path),
        "overlay_root": str(overlay_root),
        "legacy_paths": len(legacy_paths),
        "redirect_rows": sum(alias_counts.values()),
        "physical_targets_checked": physical_targets,
        "source_kinds": dict(
            sorted(Counter(source_kind_for_legacy_path(path) for path in legacy_paths).items())
        ),
        "write": write,
    }
    if not write:
        return report

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = library_path.with_name(
        f"{library_path.name}.before-o4xp-gfv2-redirects-{timestamp}"
    )
    backup_path.write_bytes(library_path.read_bytes())

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=".library-o4xp-",
            suffix=".txt",
            dir=library_path.parent,
            delete=False,
        ) as stream:
            stream.write(patched_text)
            stream.flush()
            os.fsync(stream.fileno())
            temp_path = Path(stream.name)
        os.replace(temp_path, library_path)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        if backup_path.is_file():
            library_path.write_bytes(backup_path.read_bytes())
        raise

    installed = library_path.read_text(encoding="utf-8", errors="strict")
    redirect_season_coverage(installed, legacy_paths)
    report["backup"] = str(backup_path)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add Global Forests compatibility aliases to simHeaven's "
            "X-World Vegetation Library without changing DSFs."
        )
    )
    parser.add_argument("library", type=Path, help="simHeaven library.txt path")
    parser.add_argument("overlay_root", type=Path, help="vegetation overlay root")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Install redirects. The default is a read-only validation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    library_path = args.library.resolve()
    overlay_root = args.overlay_root.resolve()
    if not library_path.is_file():
        raise FileNotFoundError(f"simHeaven library not found: {library_path}")
    if not overlay_root.is_dir():
        raise FileNotFoundError(f"Overlay root not found: {overlay_root}")

    report = install_redirects(library_path, overlay_root, args.write)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.write:
        print("Dry run only. Re-run with --write to install the redirects.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
