"""Inventory helpers for X-Plane building asset library exports."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable

from O4_SFR_DSF_Utils import active_scenery_pack_dirs, resolve_custom_scenery_dir


@dataclass(frozen=True)
class LibraryExport:
    """One resolved object/facade export from an X-Plane ``library.txt``."""

    package_name: str
    package_dir: str
    library_txt: str
    command: str
    virtual_path: str
    physical_path: str
    resolved_path: str | None


def normalize_library_path(path):
    """Return a normalized, case-insensitive library path key."""
    return (path or "").replace("\\", "/").strip().lower().lstrip("/")


def _package_dirs(custom_scenery_dir, package_name_patterns=None):
    custom_scenery_dir = resolve_custom_scenery_dir(custom_scenery_dir)
    if not custom_scenery_dir or not os.path.isdir(custom_scenery_dir):
        return []

    entries = active_scenery_pack_dirs(custom_scenery_dir)
    if not entries:
        entries = [
            (entry.name, entry.path)
            for entry in sorted(
                os.scandir(custom_scenery_dir),
                key=lambda item: item.name.lower(),
            )
            if entry.is_dir()
        ]

    patterns = [
        pattern.strip().lower()
        for pattern in (package_name_patterns or ())
        if pattern and pattern.strip()
    ]
    if not patterns:
        return entries
    return [
        (name, path)
        for name, path in entries
        if any(pattern in name.lower() for pattern in patterns)
    ]


def iter_library_txt_files(custom_scenery_dir=None, xplane_root=None,
                           include_default=False, package_name_patterns=None,
                           recursive_custom=False):
    """Yield ``(package_name, package_dir, library_txt)`` library files."""
    if custom_scenery_dir:
        for package_name, package_dir in _package_dirs(
            custom_scenery_dir,
            package_name_patterns=package_name_patterns,
        ):
            if recursive_custom:
                for root, _, files in os.walk(package_dir):
                    match = next(
                        (name for name in files if name.lower() == "library.txt"),
                        None,
                    )
                    if match is not None:
                        yield package_name, root, os.path.join(root, match)
            else:
                library_txt = os.path.join(package_dir, "library.txt")
                if os.path.isfile(library_txt):
                    yield package_name, package_dir, library_txt

    if include_default:
        if xplane_root is None and custom_scenery_dir:
            resolved = resolve_custom_scenery_dir(custom_scenery_dir)
            if resolved and os.path.basename(resolved).lower() == "custom scenery":
                xplane_root = os.path.dirname(resolved)
        if not xplane_root:
            return
        default_root = os.path.join(xplane_root, "Resources", "default scenery")
        if not os.path.isdir(default_root):
            return
        for root, _, files in os.walk(default_root):
            match = next((name for name in files if name.lower() == "library.txt"), None)
            if match is not None:
                yield "XP12 default scenery", root, os.path.join(root, match)


def parse_library_exports(library_txt, package_name=None, package_dir=None,
                          suffixes=(".obj", ".fac")):
    """Parse X-Plane library exports from one ``library.txt`` file."""
    package_dir = os.path.abspath(package_dir or os.path.dirname(library_txt))
    package_name = package_name or os.path.basename(package_dir.rstrip("\\/"))
    suffixes = tuple(suffix.lower() for suffix in suffixes)

    try:
        handle = open(library_txt, "r", encoding="utf-8", errors="ignore")
    except OSError:
        return []

    exports = []
    with handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith(("#", "//")):
                continue
            parts = line.split()
            if not parts:
                continue
            command = parts[0].upper()
            virtual_path = physical_path = None
            if command in {"EXPORT", "EXPORT_BACKUP", "EXPORT_EXCLUDE"} and len(parts) >= 3:
                virtual_path, physical_path = parts[1], parts[2]
            elif command == "EXPORT_RATIO" and len(parts) >= 4:
                virtual_path, physical_path = parts[2], parts[3]
            elif command == "EXPORT_EXTEND" and len(parts) >= 3:
                virtual_path, physical_path = parts[1], parts[2]
            if not virtual_path or not physical_path:
                continue

            virt = virtual_path.replace("\\", "/")
            phys = physical_path.replace("\\", "/")
            if suffixes and not (
                virt.lower().endswith(suffixes) or phys.lower().endswith(suffixes)
            ):
                continue
            resolved = os.path.abspath(os.path.join(package_dir, physical_path))
            exports.append(
                LibraryExport(
                    package_name=package_name,
                    package_dir=package_dir,
                    library_txt=library_txt,
                    command=command,
                    virtual_path=virt,
                    physical_path=phys,
                    resolved_path=resolved if os.path.isfile(resolved) else None,
                )
            )
    return exports


def scan_library_exports(custom_scenery_dir=None, xplane_root=None,
                         include_default=False, package_name_patterns=None,
                         suffixes=(".obj", ".fac"), recursive_custom=False):
    """Return parsed exports from Custom Scenery and optionally default scenery."""
    exports = []
    for package_name, package_dir, library_txt in iter_library_txt_files(
        custom_scenery_dir=custom_scenery_dir,
        xplane_root=xplane_root,
        include_default=include_default,
        package_name_patterns=package_name_patterns,
        recursive_custom=recursive_custom,
    ):
        exports.extend(
            parse_library_exports(
                library_txt,
                package_name=package_name,
                package_dir=package_dir,
                suffixes=suffixes,
            )
        )
    return exports


def unique_virtual_exports(exports: Iterable[LibraryExport], suffix=".obj"):
    """Group exports by virtual path, preserving all physical variants."""
    suffix = suffix.lower()
    grouped = {}
    for export in exports:
        if suffix and not export.virtual_path.lower().endswith(suffix):
            continue
        key = normalize_library_path(export.virtual_path)
        grouped.setdefault(key, []).append(export)
    return grouped
