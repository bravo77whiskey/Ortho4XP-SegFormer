"""Extract Sketchfab .zip downloads into a triage workspace.

For each archive under --downloads:
  1. Unzip into --sources/<slug>/ (skipped if already populated).
  2. Locate the primary 3D model (.fbx > .blend > .obj > .glb/.gltf by preference).
  3. Pick a "diffuse-looking" texture from the archive and write a 256x256 JPEG
     preview to --triage/previews/<slug>.jpg (the AI-visible thumbnail).
  4. Append a row to --triage/inventory.csv with slug, archive, format,
     model_path, preview_path, author, license, source_url.

Caveat: Sketchfab auto-download zips do NOT carry license/author metadata —
that lives on the model page. We populate license = --default-license
(CC-BY-4.0 by default) and source_url = the assumed Sketchfab URL
``https://sketchfab.com/3d-models/<slug>``. Verify and correct each row in
inventory.csv before running build_custom_library.py.

Nested archives (a .rar/.zip inside a zip's ``source/``) are flagged with
``format = nested-archive`` so the user can extract them manually.

Run:
    python scripts/asset_pipeline/triage_extract.py \
        --downloads G:/Downloads/sketchfab_assets \
        --sources   scripts/asset_pipeline/sources/sketchfab \
        --triage    scripts/asset_pipeline/triage
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import zipfile
from pathlib import Path

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit(
        "Pillow is required: pip install pillow"
    ) from exc


DEFAULT_THUMB_PX = 256
PREFERRED_MODEL_EXTS = (".fbx", ".blend", ".obj", ".glb", ".gltf")
COLOR_TEXTURE_PRIORITY_TOKENS = (
    "base", "diffuse", "albedo", "color", "_d.", "_clr", "_col", "rgb",
)
NESTED_ARCHIVE_EXTS = (".rar", ".7z", ".zip")
MAX_NESTED_DEPTH = 4


def _seven_zip_path() -> str | None:
    """Return the path to 7-Zip if discoverable, else None."""
    import shutil as _sh
    for cand in ("7z", "7z.exe", "7za", "7za.exe"):
        p = _sh.which(cand)
        if p:
            return p
    return None


def unpack_nested_archives(slug_dir: Path) -> int:
    """Recursively unpack nested archives (.zip / .rar / .7z) under slug_dir.

    Returns the number of archives unpacked.
    """
    seven = _seven_zip_path()
    unpacked = 0
    for depth in range(MAX_NESTED_DEPTH):
        nested = []
        for p in slug_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in NESTED_ARCHIVE_EXTS:
                nested.append(p)
        if not nested:
            break
        for archive in nested:
            ext = archive.suffix.lower()
            target = archive.parent / f"_extracted_{archive.stem}"
            target.mkdir(parents=True, exist_ok=True)
            ok = False
            if ext == ".zip":
                try:
                    with zipfile.ZipFile(archive) as zf:
                        zf.extractall(target)
                    ok = True
                except (zipfile.BadZipFile, OSError) as exc:
                    print(f"  ! nested .zip failed {archive.name}: {exc}")
            elif ext in (".rar", ".7z"):
                if not seven:
                    print(f"  ! cannot unpack {archive.name}: no 7-Zip available")
                    # Leave the file in place so the format ends up as nested-archive.
                    continue
                try:
                    import subprocess
                    result = subprocess.run(
                        [seven, "x", "-y", str(archive), f"-o{target}"],
                        capture_output=True, text=True, check=False, timeout=300,
                    )
                    ok = result.returncode == 0
                    if not ok:
                        print(f"  ! 7z failed for {archive.name} (rc={result.returncode})")
                except Exception as exc:
                    print(f"  ! 7z error for {archive.name}: {exc}")
            if ok:
                unpacked += 1
                # Delete the source archive so the next pass doesn't re-unpack it.
                try:
                    archive.unlink()
                except OSError:
                    pass
    return unpacked


def slug_from_archive(path: Path) -> str:
    """Lowercase, underscore-separated slug derived from the zip filename."""
    return path.stem.lower().replace(" ", "_")


def find_primary_model(extracted_dir: Path) -> tuple[Path | None, str]:
    """Return (model_path, format_label). format_label is empty / 'nested-archive'
    when no usable model is found."""
    by_ext: dict[str, list[Path]] = {ext: [] for ext in PREFERRED_MODEL_EXTS}
    nested: list[Path] = []
    for p in extracted_dir.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in by_ext:
            by_ext[ext].append(p)
        elif ext in NESTED_ARCHIVE_EXTS:
            nested.append(p)
    for ext in PREFERRED_MODEL_EXTS:
        if by_ext[ext]:
            best = max(by_ext[ext], key=lambda p: p.stat().st_size)
            return best, ext.lstrip(".")
    if nested:
        return None, "nested-archive"
    return None, ""


def pick_primary_image(extracted_dir: Path) -> Path | None:
    """Find the most diffuse-looking texture in the archive.

    Heuristic: prefer files whose name contains 'base'/'color'/'diffuse'/etc.;
    within each priority tier, prefer larger files (likely 4K diffuse over
    1K detail map).
    """
    candidates: list[Path] = []
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        for p in extracted_dir.rglob(f"*{ext}"):
            if p.is_file():
                candidates.append(p)
    if not candidates:
        return None

    def score(p: Path) -> tuple[int, int]:
        n = p.name.lower()
        for i, token in enumerate(COLOR_TEXTURE_PRIORITY_TOKENS):
            if token in n:
                return (i, -p.stat().st_size)
        return (len(COLOR_TEXTURE_PRIORITY_TOKENS), -p.stat().st_size)

    candidates.sort(key=score)
    return candidates[0]


def make_thumbnail(src: Path, dst: Path, size_px: int) -> bool:
    """Write a JPEG thumbnail. Returns True on success."""
    try:
        with Image.open(src) as im:
            im.thumbnail((size_px, size_px), Image.LANCZOS)
            im.convert("RGB").save(dst, "JPEG", quality=82)
        return True
    except Exception as exc:
        print(f"  ! thumbnail failed for {src.name}: {exc}")
        return False


def extract_archive(archive: Path, target: Path) -> bool:
    """Unzip into target. Skip if target already contains files."""
    if target.exists() and any(target.iterdir()):
        return True
    target.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
        return True
    except (zipfile.BadZipFile, OSError) as exc:
        print(f"  ! bad zip {archive.name}: {exc}")
        return False


def process_archive(archive: Path, sources: Path, previews: Path,
                    default_license: str) -> dict | None:
    slug = slug_from_archive(archive)
    target = sources / slug
    print(f"- {slug}")
    if not extract_archive(archive, target):
        return None

    model_path, fmt = find_primary_model(target)
    if model_path is None:
        unpacked = unpack_nested_archives(target)
        if unpacked:
            print(f"  + unpacked {unpacked} nested archive(s)")
            model_path, fmt = find_primary_model(target)
    if model_path is None:
        if fmt == "nested-archive":
            print("  ! nested archive remains in source/; install 7-Zip to auto-extract")
        else:
            print("  ! no model file located")
        model_path_rel = ""
    else:
        model_path_rel = str(
            model_path.relative_to(sources)
        ).replace("\\", "/")

    preview_src = pick_primary_image(target)
    preview_path_rel = ""
    if preview_src is not None:
        preview_dst = previews / f"{slug}.jpg"
        if make_thumbnail(preview_src, preview_dst, DEFAULT_THUMB_PX):
            preview_path_rel = preview_dst.name
    else:
        print("  ! no image found for preview")

    return {
        "slug": slug,
        "archive": archive.name,
        "format": fmt,
        "model_path": model_path_rel,
        "preview_path": preview_path_rel,
        "author": "",
        "license": default_license,
        "source_url": f"https://sketchfab.com/3d-models/{slug}",
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", required=True,
                        help="Directory containing the Sketchfab .zip downloads.")
    parser.add_argument("--sources", required=True,
                        help="Where to extract each zip (one folder per slug).")
    parser.add_argument("--triage", required=True,
                        help="Triage workspace; previews/ and inventory.csv land here.")
    parser.add_argument("--default-license", default="CC-BY-4.0",
                        help="License assumed for all rows (verify per-slug afterwards).")
    parser.add_argument("--only", default=None,
                        help="Optional substring filter on archive name; useful for re-running a subset.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    downloads = Path(args.downloads)
    sources = Path(args.sources)
    triage = Path(args.triage)
    previews = triage / "previews"
    inventory_path = triage / "inventory.csv"

    if not downloads.is_dir():
        raise SystemExit(f"--downloads path not found: {downloads}")
    sources.mkdir(parents=True, exist_ok=True)
    previews.mkdir(parents=True, exist_ok=True)

    archives = sorted(downloads.glob("*.zip"))
    if args.only:
        needle = args.only.lower()
        archives = [a for a in archives if needle in a.name.lower()]
    if not archives:
        raise SystemExit(f"No zip files matched under {downloads}")

    rows = []
    for archive in archives:
        row = process_archive(archive, sources, previews, args.default_license)
        if row is not None:
            rows.append(row)

    fieldnames = ["slug", "archive", "format", "model_path",
                  "preview_path", "author", "license", "source_url"]
    with open(inventory_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"inventory: {inventory_path} ({len(rows)} rows)")
    print(f"sources:   {sources}")
    print(f"previews:  {previews}")
    print()
    print("LICENSE NOTE: Sketchfab download zips carry no per-asset license/")
    print("author metadata. inventory.csv was populated with the default")
    print("license value and a guessed source_url. Confirm or correct each")
    print("row before producing the final library.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
