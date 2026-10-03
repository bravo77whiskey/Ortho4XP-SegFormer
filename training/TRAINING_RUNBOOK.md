# Building Placement Training Runbook

This file captures the current SimHeaven-supervised building placement training work so a later Codex/session can resume without relying on chat history.

## Goal

Replace brittle post-zone procedural building placement with a learned placement model trained from local Ortho4XP textures and SimHeaven object/facade placements.

The current active model path is the YOLO OBB alternate model. It predicts oriented building boxes directly from texture chips and is currently much more promising visually than the first custom multi-head placement model.

## Important Paths

- Project: `G:\Dev\Ortho4XP-SegFormer`
- CUDA PyTorch environment: `G:\Dev\Ortho4XP\.venv`
- Ortho4XP texture source: `H:\XP12 Addons\Applications\Ortho4XP\Tiles`
- Extra tested Ortho4XP textures: `G:\XP12 Addons2\Custom Scenery`
- SimHeaven labels source: `D:\SteamLibrary\steamapps\common\X-Plane 12\Custom Scenery`
- Training artifacts root: `H:\model_training`
- Tile manifest: `H:\model_training\manifests\tiles.jsonl`
- Training sample manifest: `H:\model_training\manifests\dataset.jsonl`
- SQLite label cache: `H:\model_training\labels\tiles`
- Active YOLO run: `H:\model_training\runs\yolo_obb_v1`
- Active progress stream: `H:\model_training\runs\yolo_obb_v1\yolo_progress.jsonl`
- Active stdout log: `H:\model_training\runs\yolo_obb_v1\stdout_512_batch8_vramfix.log`
- Active stderr log: `H:\model_training\runs\yolo_obb_v1\stderr_512_batch8_vramfix.log`
- Active weights: `H:\model_training\runs\yolo_obb_v1\weights`
- Paused height-bin run: `H:\model_training\runs\yolo_obb_height_v1` (indefinitely paused; legacy/non-height is active)

## What Was Implemented

The project now has a `training/` package with the first placement training pipeline:

- `training/building_training_lib.py`
  - Tile discovery.
  - SimHeaven DSF disassembly/cache/parsing.
  - Coordinate conversion helpers.
  - Texture chip and label helpers.
  - Holdout filtering for South America and eastern Asia.
  - Initial custom placement model/data utilities.
- `training/discover_tiles.py`
  - CLI entry point for deterministic tile discovery.
- `training/extract_dataset.py`
  - CLI entry point for dataset extraction/caching.
- `training/train_placement_model.py`
  - Initial custom multi-head placement trainer.
- `training/export_model.py`
  - Initial export entry point.
- `training/build_yolo_obb_dataset.py`
  - Builds a materialized YOLO OBB dataset if needed.
  - Avoid this for the main run unless intentionally rebuilding, because it duplicates image/label data.
- `training/train_yolo_obb.py`
  - YOLO OBB trainer with step checkpoints.
- `training/train_yolo_obb_stream.py`
  - Current preferred trainer.
  - Streams local DDS/PNG texture chips and SQLite labels directly from `dataset.jsonl`.
  - Avoids materializing another huge YOLO image dataset.

The YOLO stream trainer was patched to reduce VRAM creep:

- `--cuda-memory-fraction` limits this process's CUDA allocator.
- `--cuda-empty-cache-interval` periodically releases unused cached CUDA blocks.
- Step callbacks write progress and checkpoint files.
- `latest_step.pt` is saved every checkpoint interval.
- `best_step.pt` is saved when the callback loss improves by `--best-step-min-delta`.

## Data Policy

Training uses local texture files and SimHeaven-derived SQLite labels. It does not use OSM, Microsoft, CBRA, or downloaded footprint datasets for v1.

South America and eastern Asia are excluded from the training split and kept as holdout/problem regions:

- South America default holdout: lat `-60..15`, lon `-90..-30`
- Eastern Asia default holdout: lat `-10..55`, lon `100..150`

At the time the manifest was built:

- `tiles.jsonl` contains `671` tile entries.
- `dataset.jsonl` contains `97,959` training sample entries.

Do not rebuild the dataset unless needed. The SQLite cache already exists and is many GB. The current YOLO stream trainer reuses it.

## Current Active Run

Run directory:

```powershell
H:\model_training\runs\yolo_obb_v1
```

The active run was started on `2026-05-06 02:43:56 +03:00`.

Recorded launch metadata:

```powershell
H:\model_training\runs\yolo_obb_v1\process_512_batch8_vramfix.json
```

Launch args from that file:

```powershell
& 'G:\Dev\Ortho4XP\.venv\Scripts\python.exe' `
  'training\train_yolo_obb_stream.py' `
  --manifest 'H:\model_training\manifests\dataset.jsonl' `
  --model 'H:\model_training\runs\yolo_obb_v1\weights\latest_step.pt' `
  --run-dir 'H:\model_training\runs\yolo_obb_v1' `
  --epochs 20 `
  --imgsz 512 `
  --batch 8 `
  --workers 2 `
  --device 0 `
  --max-train 0 `
  --max-val 0 `
  --max-instances 1000 `
  --checkpoint-interval 1000 `
  --progress-interval 25 `
  --best-step-min-delta 0.0001 `
  --cuda-empty-cache-interval 25 `
  --cuda-memory-fraction 0.90 `
  --no-resume
```

Important: `--no-resume` was used for that launch because we intentionally restarted from `latest_step.pt` with the new 512/batch-8 settings. For crash recovery or normal continuation, use `--resume` or omit the resume flag, because resume is the default.

As of the `2026-06-11` visual-test check:

- Latest progress row seen: step `20425`, epoch `2`, loss around `30.3886`.
- `latest_step.pt` was last written at step `20000`.
- `last.pt` was also written at step `20000`.
- `best_step.pt` is still step `9574`; there was no new best checkpoint.
- Height-bin training is indefinitely paused; do not use `--height-labels bins` unless explicitly resuming that experiment.

The progress will be newer if training continued after this file was written.

## Check Progress

From PowerShell:

```powershell
Set-Location 'G:\Dev\Ortho4XP-SegFormer'

Get-Content -LiteralPath 'H:\model_training\runs\yolo_obb_v1\yolo_progress.jsonl' -Tail 10

Get-Item -LiteralPath `
  'H:\model_training\runs\yolo_obb_v1\weights\latest_step.pt', `
  'H:\model_training\runs\yolo_obb_v1\weights\last.pt', `
  'H:\model_training\runs\yolo_obb_v1\weights\best_step.pt' |
  Select-Object FullName,Length,LastWriteTime

nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

Get-Process python -ErrorAction SilentlyContinue |
  Select-Object Id,CPU,StartTime,Path |
  Sort-Object StartTime
```

The most useful training files are:

- `yolo_progress.jsonl`: step-level loss/progress plus checkpoint events.
- `stdout_512_batch8_vramfix.log`: Ultralytics progress rows.
- `stderr_512_batch8_vramfix.log`: should normally be empty.
- `weights\latest_step.pt`: latest periodic step checkpoint.
- `weights\last.pt`: Ultralytics resume checkpoint.
- `weights\best_step.pt`: callback best-by-loss checkpoint.
- `weights\epoch0.pt`, `epoch1.pt`, etc.: epoch checkpoints.

## Pause, Stop, and Resume

There is no special in-model pause state. Pause means stop the Python process after a checkpoint exists, then resume from `last.pt`.

Best practice before stopping:

1. Check that `weights\latest_step.pt` and `weights\last.pt` exist.
2. Prefer stopping shortly after a checkpoint row such as `"saved_latest_step"` appears in `yolo_progress.jsonl`.
3. Stop the process.
4. Restart with `--resume`.

To find the current process:

```powershell
Get-Process python -ErrorAction SilentlyContinue |
  Select-Object Id,CPU,StartTime,Path |
  Sort-Object StartTime
```

The wrapper PID is also recorded in:

```powershell
H:\model_training\runs\yolo_obb_v1\process_512_batch8_vramfix.json
```

To stop a known PID:

```powershell
Stop-Process -Id <PID>
```

If there are worker child processes still alive after stopping the main process, inspect `Get-Process python` again and stop only the run-related Python processes.

## Resume Current Run

Use this for crash recovery or normal continuation. `--resume` is optional because it defaults to true, but keep it explicit for clarity.

```powershell
Set-Location 'G:\Dev\Ortho4XP-SegFormer'

& 'G:\Dev\Ortho4XP\.venv\Scripts\python.exe' `
  'training\train_yolo_obb_stream.py' `
  --manifest 'H:\model_training\manifests\dataset.jsonl' `
  --model 'H:\model_training\runs\yolo_obb_v1\weights\latest_step.pt' `
  --run-dir 'H:\model_training\runs\yolo_obb_v1' `
  --epochs 20 `
  --imgsz 512 `
  --batch 8 `
  --workers 2 `
  --device 0 `
  --max-train 0 `
  --max-val 0 `
  --max-instances 1000 `
  --checkpoint-interval 1000 `
  --progress-interval 25 `
  --best-step-min-delta 0.0001 `
  --cuda-empty-cache-interval 25 `
  --cuda-memory-fraction 0.90 `
  --resume
```

Resume behavior:

- If `H:\model_training\runs\yolo_obb_v1\weights\last.pt` exists, the trainer loads `last.pt`.
- If `last.pt` does not exist, it loads the `--model` path.
- `--model` is therefore mostly a fallback during resume.

To continue past 20 epochs, restart with a larger total epoch count, for example `--epochs 40`, and keep `--resume`.

## Start in Background

For a long run that can be checked later, start it with redirected logs and save the process metadata. Adjust the process JSON and log names if starting a different run.

```powershell
Set-Location 'G:\Dev\Ortho4XP-SegFormer'

$runDir = 'H:\model_training\runs\yolo_obb_v1'
$stdout = Join-Path $runDir 'stdout_512_batch8_vramfix_resume.log'
$stderr = Join-Path $runDir 'stderr_512_batch8_vramfix_resume.log'

$args = @(
  'training\train_yolo_obb_stream.py',
  '--manifest', 'H:\model_training\manifests\dataset.jsonl',
  '--model', 'H:\model_training\runs\yolo_obb_v1\weights\latest_step.pt',
  '--run-dir', 'H:\model_training\runs\yolo_obb_v1',
  '--epochs', '20',
  '--imgsz', '512',
  '--batch', '8',
  '--workers', '2',
  '--device', '0',
  '--max-train', '0',
  '--max-val', '0',
  '--max-instances', '1000',
  '--checkpoint-interval', '1000',
  '--progress-interval', '25',
  '--best-step-min-delta', '0.0001',
  '--cuda-empty-cache-interval', '25',
  '--cuda-memory-fraction', '0.90',
  '--resume'
)

$process = Start-Process `
  -FilePath 'G:\Dev\Ortho4XP\.venv\Scripts\python.exe' `
  -ArgumentList $args `
  -WorkingDirectory 'G:\Dev\Ortho4XP-SegFormer' `
  -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr `
  -WindowStyle Hidden `
  -PassThru

[pscustomobject]@{
  pid = $process.Id
  started = (Get-Date).ToString('o')
  cwd = 'G:\Dev\Ortho4XP-SegFormer'
  stdout = $stdout
  stderr = $stderr
  args = $args
} | ConvertTo-Json -Depth 4 |
  Set-Content -LiteralPath (Join-Path $runDir 'process_resume.json') -Encoding UTF8
```

## VRAM Notes

The current settings are a compromise that worked better than earlier attempts:

- Image size: `512`
- Batch size: `8`
- Workers: `2`
- CUDA memory fraction: `0.90`
- Empty cache interval: every `25` steps

YOLO uses more VRAM than the first custom trainer. Some temporary spikes happen during checkpoint saves. If VRAM gets too close to full again, first try reducing `--cuda-memory-fraction` to `0.80` or `0.85`. If that is not enough, reduce `--batch` before reducing image size further.

Batch size does affect optimization stability but not model architecture. Larger batches usually make gradients less noisy; smaller batches can still train, but the run may need more steps and careful learning-rate behavior.

## Overlay Test History

Overlay tests were run to judge visual quality because the callback best loss did not always match the most useful visual checkpoint.

Recent test output folders:

- `G:\Dev\Ortho4XP-SegFormer\tmp\yolo_obb_tests\latest_step_3000`
- `G:\Dev\Ortho4XP-SegFormer\tmp\yolo_obb_tests\latest_step_3000_earlier`
- `G:\Dev\Ortho4XP-SegFormer\tmp\yolo_obb_tests\latest_step_4000`
- `G:\Dev\Ortho4XP-SegFormer\tmp\yolo_obb_tests\latest_step_5000`
- `G:\Dev\Ortho4XP-SegFormer\tmp\yolo_obb_tests\latest_step_20000_nonheight`

The regression helper now defaults to no text labels on detection boxes and class-based colors. Height labels and the height legend are opt-in via `--overlay-label-mode height` or `--overlay-color-mode height`.

Step 5000 test summary:

- `28512_53472_BI16`: `4446` detections.
- `25680_54080_BI16`: `1567` detections.
- `25680_54096_BI16`: `1755` detections.
- `28528_54656_Arc16`: `703` detections.
- `28032_54832_Arc16`: `304` detections.
- `28048_54880_Arc16`: `3115` detections.
- `203056_446832_Arc19`: `129` detections.
- `203024_446816_Arc19`: `1418` detections.

The step 5000 overlay summary is:

```powershell
G:\Dev\Ortho4XP-SegFormer\tmp\yolo_obb_tests\latest_step_5000\batch_summary.json
```

Qualitative notes:

- Step 3000 was very trigger-happy in some ZL16 regions.
- Step 4000 was less noisy but sometimes under-detected dense urban areas.
- Step 5000 recovered more detections in the original dense test image and improved some Arc19 results, while becoming much more conservative on a few Taiwan ZL16 textures.

## Best Step Caveat

`best_step.pt` is currently best according to the step-loss callback state, but visual quality has not always tracked that loss. Earlier, an old `best_step.pt` produced obvious water false positives, so it was manually replaced with the then-current `latest_step.pt`.

Relevant files:

- Current callback state: `H:\model_training\runs\yolo_obb_v1\best_step_state.json`
- Backup made before manual replacement: `H:\model_training\runs\yolo_obb_v1\weights\best_step_before_latest_replace_20260506_033009.pt`
- Manual replacement note: `H:\model_training\runs\yolo_obb_v1\manual_best_step_replacement_20260506_033009.json`

When choosing a checkpoint for integration, compare `latest_step.pt`, `last.pt`, and `best_step.pt` with real overlays, not only the loss number.

## Do Not Accidentally Rebuild Huge Data

For normal training, use:

```powershell
training\train_yolo_obb_stream.py
```

Avoid this unless intentionally creating a duplicate YOLO dataset:

```powershell
training\build_yolo_obb_dataset.py
```

The stream trainer is the important one because it reads local DDS textures and existing SQLite labels directly. That avoids another multi-day extraction and avoids duplicating more than 1 TB of texture imagery.
