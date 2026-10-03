# Changelog

## Unreleased: project cleanup

The overlays started as a SegFormer ("SFR") experiment and outgrew it. This
release renames everything that still said SFR, removes dead code and sets up
the project for development and distribution. The generated overlays are
unchanged.

### Renamed

Modules in `src/`:

| Before | After |
| --- | --- |
| `O4_SFR_Pipeline` | `O4_Overlay_Pipeline` |
| `O4_SFR_Building_Overlay` | `O4_Building_Overlay` |
| `O4_SFR_Vegetation_Overlay` | `O4_Vegetation_Overlay` |
| `O4_SFR_Inference` | `O4_Landcover_Inference` |
| `O4_SFR_Remote` | `O4_Remote_GPU` |
| `O4_SFR_DSF_Utils` | `O4_Scenery_Packs` |
| `O4_SFR_Veg_Cleanup` | `O4_Veg_Replacement` |
| `O4_SFR_Debug_Images` | `O4_Overlay_Debug_Images` |
| `O4_SFR_<Name>` (all others) | `O4_<Name>` |
| `O4_SFR_Region_Overrides.json` | `O4_Region_Overrides.json` |

Settings. Old config files keep loading: the old keys are translated when read
and saved under the new names. A build from before this release does not know
the new names and falls back to defaults for them.

| Before | After |
| --- | --- |
| `sfr_veg_*` | `veg_*` |
| `sfr_bld_*` | `bld_*` |
| `sfr_patch_size`, `sfr_overlap`, `sfr_batch_size` | `segformer_patch_size`, `segformer_overlap`, `segformer_batch_size` |
| `global_` variants of the above | same rule |

Environment variables (see [docs/environment_variables.md](docs/environment_variables.md));
the old names are no longer read:

| Before | After |
| --- | --- |
| `O4_SFR_BLD_*` | `O4_BLD_*` |
| `O4_SFR_STOCK_YOLO_*` | `O4_STOCK_YOLO_*` |
| `O4_SFR_REMOTE`, `O4_SFR_REMOTE_HOST` | `O4_REMOTE_GPU`, `O4_REMOTE_GPU_HOST` |
| `O4_SFR_USE_AMP`, `O4_SFR_CHANNELS_LAST` | `O4_SEGFORMER_AMP`, `O4_SEGFORMER_CHANNELS_LAST` |
| `O4_SFR_TIMING_DETAIL`, `O4_SFR_TIMING_SLOW`, `O4_SFR_FILE_FILTER` | `O4_OVERLAY_TIMING_DETAIL`, `O4_OVERLAY_TIMING_SLOW`, `O4_OVERLAY_FILE_FILTER` |
| `O4_SFR_BENCH_PYTHON` | `O4_BENCH_PYTHON` |

Folders:

- `SFR_cache/` is now `Overlay_cache/`. An existing `SFR_cache/` keeps being
  used; rename it by hand (with no build running) to switch.
- The packaged build ships the overlay modules in `_internal/overlay_scripts`
  instead of `_internal/sfr_scripts`; `build.py` removes the old folder from
  `dist/`.
- The remote GPU environment lives in `~/.ortho4xp_remote`. A host set up
  before keeps its environment: `~/.ortho4xp_remote` is created as a link to
  the existing `~/.ortho4xp_sfr` on the next connection, so nothing is
  reinstalled.

GUI and logs:

- The **SegFormer Bld** / **SegFormer Veg** buttons and batch steps are now
  **Building Overlay** / **Vegetation Overlay**. Unfinished batch journals
  written with the old step names still resume.
- Log prefixes `[SFR …]` are now `[Overlay]`, `[Bld]`, `[Veg]`,
  `[Overlay Setup]`, `[Remote GPU]`, `[Library]` and `[Debug]`.

Scripts: `src/scripts/sfr_remote_worker.py` is `remote_gpu_worker.py` and
`audit_sfr_building_assets.py` is `audit_building_assets.py`. Developer-only
tools moved from `src/scripts/` to `scripts/` (`compare_height_models.py`,
`merge_height_comparisons.py`, `generate_debug_images.py`,
`generate_roof_color_visuals.py`, `run_classical_bld_regression.py`), as did
`tests/sfr_building_pipeline_benchmark.py` (now
`scripts/building_pipeline_benchmark.py`). Test files lost their `sfr_`
prefix.

Unchanged on purpose: the `o4sfr/...` virtual paths and the `O4SFR_*` library
folders, which installed libraries and generated DSFs refer to.

### Removed

- Prototype modules nothing used any more: `O4_AI_Overlay` and
  `src/scripts/generate_overlay.py`, the excess-green `O4_Veg_Overlay` and
  `O4_Building_Overlay` prototypes (the name now belongs to the real building
  overlay), the combined `process_sfr_tile` path with
  `src/scripts/generate_sfr_overlay.py`, and `src/Unused/`.
- Unreferenced upstream modules `O4_Geotag`, `O4_Recut_Water` and
  `O4_Scenery_Utils`.
- About 50 functions with no callers, the loader for the unused binary
  building SegFormer, and the never-read `bld_close_k` / `bld_open_k`
  settings (old values are still converted to `bld_close_m` / `bld_open_m`).
- The built-in default remote GPU host; headless runs set
  `O4_REMOTE_GPU_HOST`.
- The building overlay's retired candidate-grid placement (zone templates,
  gap fill, road-heading grids, dynamic centre blockers), which no longer
  placed anything but still ran on every texture, and the never-read
  `yolo_guidance` distance transforms (about 2,600 lines in all). Its
  environment variables (`O4_BLD_MAX_CANDIDATES`, `O4_BLD_GAP_*`,
  `O4_BLD_YOLO_TEMPLATE_*`, `O4_BLD_DISABLE_CENTER_BLOCKERS`) are no longer
  read. Existing `*_bld.pkl` placement caches stay valid.
- The reader for per-texture `*_road.pkl` caches, which nothing has written
  since May 2026.
- `segmentation-models-pytorch`, `Wand` and `torch` from `requirements.txt`
  (the first two were unused; see the file for installing PyTorch).

### Fixed

- Building water masks with a custom mask extent crashed with a `NameError`.
- A leftover copy of the mask-deletion loop in `record_water_tris` failed on
  every run (silently) and printed duplicate progress lines.
- `scripts/asset_pipeline/procgen/build_procgen_library.py` crashed at exit
  (missing `import sys`).
- Overlay setup errors pointed at a "Setup SegFormer" button that does not
  exist.
- `install_windows.bat` referenced a GDAL wheel that is not in `Utils/win`, and
  the launch scripts used `venv/` while the overlays and `build.py` use
  `.venv/`.
- Tests that picked the wrong DSFTool binary on Linux or assumed Windows paths.

### Changed

- The GUI and batch builds copy overlay settings to the pipeline through one
  function, `O4_Overlay_Pipeline.apply_tile_settings`, instead of two
  hand-maintained lists.
- `Ortho4XP.spec` lists the loose overlay modules once (`OVERLAY_MODULES`)
  and derives both the data files and the hidden imports from it.
- The `training/` code is now part of the repository.
- `O4_Building_Overlay.run()` (3,500 lines) is split into the stages of
  `_BuildingOverlayRun` and the per-texture stages of `_TexturePass`.
  `scripts/building_overlay_parity_harness.py` runs the real overlay on a
  synthetic tile with stand-in models and compares the outputs of two source
  trees byte for byte; the restructuring was verified with it.

### Added

- `README.md` describing this project, `docs/architecture.md`,
  `docs/environment_variables.md`, this changelog.
- `pyproject.toml` with pytest and ruff settings.
- `.env` and log files are ignored by git.
