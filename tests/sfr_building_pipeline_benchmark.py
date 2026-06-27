"""Run the real SFR building overlay pipeline as a speed benchmark.

This is intentionally not a unit test. It calls O4_SFR_Building_Overlay.run()
with cache disabled by default so timings include the production DDS load,
model inference, road/exclusion masks, heading grid, placement, DSF text, and
DSFTool compile when available.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
DEFAULT_DEPLOYED_PYTHON = Path(os.environ.get(
    "O4_SFR_BENCH_PYTHON",
    r"G:\Dev\Ortho4XP\.venv\Scripts\python.exe",
))


TILE_RE = re.compile(r"zOrtho4XP_([+-]\d{2})([+-]\d{3})", re.IGNORECASE)


def _cfg_custom_scenery_dir() -> Path | None:
    """Return custom_scenery_dir from the repo's Ortho4XP.cfg, if usable.

    This is what production tile builds pass to the overlay
    (O4_SFR_Pipeline._scenery_paths), so the benchmark should default to the
    same directory instead of inferring one from the tile path -- tile
    folders are commonly junctions whose resolved parent is NOT the real
    Custom Scenery (no scenery_packs.ini, no libraries).
    """
    cfg = ROOT / "Ortho4XP.cfg"
    if not cfg.is_file():
        return None
    try:
        for raw_line in cfg.read_text(encoding="utf-8", errors="ignore").splitlines():
            key, sep, value = raw_line.partition("=")
            if sep and key.strip() == "custom_scenery_dir":
                value = value.strip()
                if value:
                    path = Path(value)
                    if path.is_dir():
                        return path
                return None
    except OSError:
        return None
    return None


def _infer_tile(dds_path: Path) -> tuple[int, int, Path, Path | None]:
    tex_dir = dds_path.parent
    for parent in [tex_dir, *tex_dir.parents]:
        match = TILE_RE.match(parent.name)
        if match:
            custom_scenery_dir = parent.parent if parent.parent.exists() else None
            return int(match.group(1)), int(match.group(2)), tex_dir, custom_scenery_dir
    raise ValueError(
        "Could not infer lat/lon from a parent folder named zOrtho4XP_+22+113. "
        "Pass --lat and --lon."
    )


def _default_dsftool() -> Path | None:
    if sys.platform.startswith("win"):
        candidate = ROOT / "Utils" / "win" / "DSFTool.exe"
    elif sys.platform.startswith("darwin"):
        candidate = ROOT / "Utils" / "mac" / "DSFTool"
    else:
        candidate = ROOT / "Utils" / "lin" / "DSFTool"
    return candidate if candidate.exists() else None


def _tile_name(lat: int, lon: int) -> str:
    return f"{lat:+03d}{lon:+04d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the real SFR building generation path."
    )
    parser.add_argument(
        "dds_or_tex_dir",
        help="A DDS file to benchmark, or a textures directory when --full-tile is set.",
    )
    parser.add_argument("--lat", type=int, default=None)
    parser.add_argument("--lon", type=int, default=None)
    parser.add_argument("--full-tile", action="store_true")
    parser.add_argument(
        "--file-filter",
        default=None,
        help="Comma-separated DDS filename/glob filters, e.g. 28448_53424_BI16.dds,28448_53440_BI16.dds",
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--out-dsf", default=None)
    parser.add_argument("--osm-roads", default=None)
    parser.add_argument("--custom-scenery-dir", default=None)
    parser.add_argument("--dsftool", default=None)
    parser.add_argument("--spacing", type=float, default=0.0)
    parser.add_argument("--close-k", type=int, default=15)
    parser.add_argument("--open-k", type=int, default=5)
    parser.add_argument("--min-zone-m2", type=float, default=200.0)
    parser.add_argument("--grid-n", type=int, default=16)
    parser.add_argument("--yolo-conf", type=float, default=None)
    parser.add_argument("--yolo-iou", type=float, default=None)
    parser.add_argument("--yolo-stride", type=int, default=None)
    parser.add_argument("--yolo-max-det", type=int, default=None)
    parser.add_argument("--yolo-suppress-coverage", type=float, default=0.0)
    parser.add_argument("--yolo-suppress-min-overlap-m2", type=float, default=25.0)
    parser.add_argument(
        "--legacy-overlap-removal",
        action="store_true",
        help="Run the legacy avoidance path (yolo_no_overlap_removal=False) "
             "instead of the max-coverage default, to reproduce the slow path.",
    )
    parser.add_argument("--yolo-outline-tolerance", type=float, default=None,
                        help="Fraction of an object footprint required inside "
                             "the YOLO polygon (1.0 = strict containment).")
    parser.add_argument(
        "--no-custom-scenery-avoidance",
        action="store_true",
        help="Disable avoidance of existing custom-scenery objects/facades "
             "(simulates a fresh tile with no prior yOrtho4XP_Bld_Overlays).",
    )
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument(
        "--cold-cache",
        action="store_true",
        help="Clear the benchmark cache and run with cache disabled.",
    )
    parser.add_argument("--skip-osm-download", action="store_true")
    parser.add_argument(
        "--allow-current-python",
        action="store_true",
        help="Do not re-exec through the deployed CUDA runtime.",
    )
    parser.add_argument(
        "--deployed-python",
        default=str(DEFAULT_DEPLOYED_PYTHON),
        help="Python used for deployed-runtime benchmarks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    deployed_python = Path(args.deployed_python).resolve()
    current_python = Path(sys.executable).resolve()
    if (
        not args.allow_current_python
        and deployed_python.exists()
        and current_python != deployed_python
    ):
        cmd = [
            str(deployed_python),
            str(Path(__file__).resolve()),
            *sys.argv[1:],
            "--allow-current-python",
        ]
        print(f"[bench] re-exec deployed CUDA runtime: {deployed_python}", flush=True)
        return subprocess.call(cmd)

    if args.cold_cache:
        args.use_cache = False
        args.clear_cache = True

    # abspath, NOT resolve(): tile folders are often junctions and following
    # them would move every inferred parent off the real Custom Scenery.
    input_path = Path(os.path.abspath(args.dds_or_tex_dir))

    selected_file = None
    if input_path.is_file():
        lat, lon, tex_dir, inferred_scenery = _infer_tile(input_path)
        selected_file = input_path.name
    else:
        tex_dir = input_path
        inferred_scenery = tex_dir.parent.parent if tex_dir.name.lower() == "textures" else None
        if args.lat is None or args.lon is None:
            raise ValueError("--lat and --lon are required when passing a textures directory")
        lat, lon = args.lat, args.lon

    if args.lat is not None:
        lat = args.lat
    if args.lon is not None:
        lon = args.lon

    bench_root = ROOT / "tmp" / "bench_sfr_building" / _tile_name(lat, lon)
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else bench_root / "cache"
    out_dsf = Path(args.out_dsf).resolve() if args.out_dsf else bench_root / "out" / f"{_tile_name(lat, lon)}.dsf"
    if args.custom_scenery_dir:
        custom_scenery_dir = Path(args.custom_scenery_dir)
        scenery_source = "cli"
    else:
        custom_scenery_dir = _cfg_custom_scenery_dir()
        scenery_source = "Ortho4XP.cfg"
        if custom_scenery_dir is None:
            custom_scenery_dir = inferred_scenery
            scenery_source = "inferred-from-tile-path"
    dsftool = Path(args.dsftool).resolve() if args.dsftool else _default_dsftool()

    if args.clear_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dsf.parent.mkdir(parents=True, exist_ok=True)

    old_detail = os.environ.get("O4_SFR_TIMING_DETAIL")
    old_slow = os.environ.get("O4_SFR_TIMING_SLOW")
    old_filter = os.environ.get("O4_SFR_FILE_FILTER")
    os.environ["O4_SFR_TIMING_DETAIL"] = "1"
    os.environ["O4_SFR_TIMING_SLOW"] = "0"
    if args.file_filter:
        os.environ["O4_SFR_FILE_FILTER"] = args.file_filter
    elif selected_file and not args.full_tile:
        os.environ["O4_SFR_FILE_FILTER"] = selected_file
    elif "O4_SFR_FILE_FILTER" in os.environ:
        del os.environ["O4_SFR_FILE_FILTER"]

    try:
        import O4_SFR_Building_Overlay as bld

        print("[bench] real building pipeline")
        print(f"[bench] python={sys.executable}")
        try:
            import torch
            cuda_state = (
                f"cuda={torch.cuda.is_available()}"
                + (
                    f" device={torch.cuda.get_device_name(0)}"
                    if torch.cuda.is_available() else ""
                )
            )
        except Exception as exc:
            cuda_state = f"cuda_check_failed={exc}"
        print(f"[bench] {cuda_state}")
        print(f"[bench] tex_dir={tex_dir}")
        if selected_file and not args.full_tile:
            print(f"[bench] dds={selected_file}")
        print(f"[bench] tile={_tile_name(lat, lon)} cache={'on' if args.use_cache else 'off'}")
        print(f"[bench] cache_dir={cache_dir}")
        print(f"[bench] out_dsf={out_dsf}")
        print(f"[bench] custom_scenery_dir={custom_scenery_dir or ''} "
              f"(source={scenery_source})")
        print(f"[bench] dsftool={dsftool or ''}")

        t0 = time.perf_counter()
        bld.run(
            tex_dir=str(tex_dir),
            lat=lat,
            lon=lon,
            out_dsf=str(out_dsf),
            spacing_m=args.spacing,
            close_k=args.close_k,
            open_k=args.open_k,
            min_zone_m2=args.min_zone_m2,
            make_viz=False,
            cache_dir=str(cache_dir),
            disable_cache=not args.use_cache,
            grid_n=args.grid_n,
            osm_roads_path=args.osm_roads,
            dsftool_path=str(dsftool) if dsftool else None,
            skip_osm_excl_download=args.skip_osm_download,
            custom_scenery_dir=str(custom_scenery_dir) if custom_scenery_dir else None,
            avoid_custom_scenery=not args.no_custom_scenery_avoidance,
            yolo_conf=args.yolo_conf,
            yolo_iou=args.yolo_iou,
            yolo_stride=args.yolo_stride,
            yolo_max_det=args.yolo_max_det,
            yolo_suppress_coverage=args.yolo_suppress_coverage,
            yolo_suppress_min_overlap_m2=args.yolo_suppress_min_overlap_m2,
            yolo_no_overlap_removal=not args.legacy_overlap_removal,
            yolo_outline_tolerance=args.yolo_outline_tolerance,
        )
        print(f"[bench] total wall={time.perf_counter() - t0:.2f}s")
    finally:
        if old_detail is None:
            os.environ.pop("O4_SFR_TIMING_DETAIL", None)
        else:
            os.environ["O4_SFR_TIMING_DETAIL"] = old_detail
        if old_slow is None:
            os.environ.pop("O4_SFR_TIMING_SLOW", None)
        else:
            os.environ["O4_SFR_TIMING_SLOW"] = old_slow
        if old_filter is None:
            os.environ.pop("O4_SFR_FILE_FILTER", None)
        else:
            os.environ["O4_SFR_FILE_FILTER"] = old_filter

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
