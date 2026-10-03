# Environment variables

User-facing options are config settings (`bld_*`, `veg_*`, `segformer_*` in the
configuration window). The variables below are developer switches: A/B tests,
diagnostics and escape hatches. They are read by the overlay subprocess, which
inherits the environment of the Ortho4XP process that starts it.

Boolean switches accept `1/true/yes/on` and `0/false/no/off`.

Until the cleanup in [CHANGELOG.md](../CHANGELOG.md) these were all named
`O4_SFR_<NAME>`; the old names are no longer read.

## Remote GPU

| Variable | Default | Effect |
| --- | --- | --- |
| `O4_REMOTE_GPU` | off | `1` sends model inference to `O4_REMOTE_GPU_HOST` for headless runs (the GUI has its own Remote GPU option) |
| `O4_REMOTE_GPU_HOST` | none | SSH host (an `~/.ssh/config` alias) used with `O4_REMOTE_GPU=1` |

## Models and inference

| Variable | Default | Effect |
| --- | --- | --- |
| `O4_BLD_YOLO_CHECKPOINT` | built-in path | Building detector checkpoint when the `bld_yolo_checkpoint` setting is not passed |
| `O4_BLD_HEIGHT_CHECKPOINT` | built-in path | HeightNet checkpoint when the `bld_height_checkpoint` setting is not passed |
| `O4_STOCK_YOLO_CHECKPOINT` | `yolo26x-obb.pt` in the repository root | Stock DOTA YOLO checkpoint (downloaded there when missing) |
| `O4_BLD_YOLO_ENABLED` | setting | Overrides `bld_yolo_enabled` |
| `O4_BLD_YOLO_ALLOW_MISSING` | off | Carry on without the building detector when its checkpoint is missing |
| `O4_BLD_HEIGHT_ENABLED` | on | `0` skips HeightNet and uses class-default heights |
| `O4_BLD_HEIGHT_FP16` | on | `0` runs HeightNet in fp32 (exact numbers for parity A/B runs) |
| `O4_BLD_YOLO_BATCH` | 1 | Building detector batch size; values above 1 change borderline detections |
| `O4_STOCK_YOLO_BATCH` | 1 | Stock YOLO batch size; same caveat |
| `O4_BLD_YOLO_FUSE` | off | Fuse detector layers before inference; changes borderline detections |
| `O4_BLD_YOLO_ANALYSIS_ZL` | 0 | Run the detector on a copy downsampled to this zoom level (0 = native resolution) |
| `O4_BLD_YOLO_CUDA_CLEANUP_EVERY` | 0 | Empty the CUDA cache every N textures (0 = never) |
| `O4_BLD_PREFETCH` | on | `0` disables the one-texture-ahead inference prefetch thread |
| `O4_SEGFORMER_AMP` | off | `1` runs SegFormer under CUDA autocast |
| `O4_SEGFORMER_CHANNELS_LAST` | off | `1` uses channels-last memory layout for SegFormer |

## Building overlay behaviour

| Variable | Default | Effect |
| --- | --- | --- |
| `O4_BLD_ASSET_MODE` | setting | Overrides `bld_asset_mode` (`objects`, `facades` or `both`); `O4_BLD_O4SFR_LIBRARY_ASSETS` and `O4_BLD_EXTRA_LIBRARY_ASSETS` are older spellings |
| `O4_BLD_EXTRA_LIBRARIES` | `auto` | Which optional third-party building libraries may be used |
| `O4_BLD_ROOF_COLOR_MATCHING` | setting | Overrides `bld_roof_color_matching` |
| `O4_BLD_SEG_FIRST` | `guarded` | Landcover-first footprints: `off`, `guarded` (regular outlines only) or `all` |
| `O4_BLD_SEG_FIRST_GROUND` | on | Ground test for landcover-first footprints |
| `O4_BLD_FOOTPRINT_POLYGONS` | on | Polygon footprints for large irregular buildings |
| `O4_BLD_AGRI` | `greenhouse` | Farmland handling: `greenhouse`, `drop` or `off` |
| `O4_BLD_HEIGHT_GATE` | on | Drop detections taller than their regional height limit |
| `O4_BLD_HEIGHT_CONTEXT` | on | Lower heights that stand out from their neighbourhood |
| `O4_BLD_EVIDENCE_GATE` | 1 | `0` disables the evidence gate for very large detections |
| `O4_BLD_EVIDENCE_MIN_AREA_M2`, `O4_BLD_EVIDENCE_MIN_SIDE_M`, `O4_BLD_EVIDENCE_SEGFORMER_FRAC`, `O4_BLD_EVIDENCE_GREEN_FRAC` | module constants | Evidence gate thresholds |
| `O4_BLD_PAIR_OVERLAP_RULE` | on | `0` restores the blanket smaller-wins overlap drop |
| `O4_BLD_SUPPRESS_CONTAINMENT_FRAC` | 0.35 | Overlap suppression: containment fraction |
| `O4_BLD_SUPPRESS_EXPLAIN_FRAC` | 0.60 | Overlap suppression: explained-area fraction |
| `O4_BLD_TILE_DEDUP` | on | Tile-wide duplicate removal after placement |
| `O4_BLD_DEDUP_OVERLAP_FRAC` | 0.20 | Overlap below which adjacent buildings both survive the tile-wide dedup |
| `O4_BLD_OBJECT_SELF_AVOID` | off | Per-candidate self-avoidance between placed objects |
| `O4_BLD_ALLOW_OVERSIZED_ALIASES` | off | Keep library aliases whose mesh is far bigger than their name declares |
| `O4_BLD_SMALLEST_ASSET_ONLY` | off | Keep only the smallest asset of each class |
| `O4_BLD_STRICT_FIT` | off | Strict asset-fit test |
| `O4_BLD_FOOTPRINT_PAD_M` | module constant | Padding around detected footprints |

## Diagnostics

| Variable | Default | Effect |
| --- | --- | --- |
| `O4_OVERLAY_FILE_FILTER` | none | Comma-separated texture name patterns; only matching textures are processed |
| `O4_OVERLAY_TIMING_DETAIL` | off | Per-stage timing lines for every texture |
| `O4_OVERLAY_TIMING_SLOW` | 0 (3 with `bld_verbose_log`) | Print timing lines for textures slower than this many seconds |
| `O4_BLD_PLACE_PROFILE` | off | Break the placement loop timing into sub-phases |
| `O4_BLD_BLOCK_DEBUG` | off | Dump `<texture>_blockdebug.pkl` with why each detection was not placed |
| `O4_BLD_VIZ_SIZE` | 512 | Tile size in pixels of the overview images |
| `O4_BENCH_PYTHON` | deployed `.venv` | Interpreter for `scripts/building_pipeline_benchmark.py` |

## Other

| Variable | Default | Effect |
| --- | --- | --- |
| `O4_PBF_DIFF_BATCH` | 10 | Daily diffs applied per pass when updating a local planet file |
