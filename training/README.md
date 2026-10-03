# SimHeaven Building Placement Training

This package builds a v1 supervised training set from local Ortho4XP textures and
SimHeaven X-World building placements, then trains a placement-field model for
building centers, footprint dimensions, headings, masks, and coarse classes.

## Paths

- Textures: `H:\XP12 Addons\Applications\Ortho4XP\Tiles`
- SimHeaven: `D:\SteamLibrary\steamapps\common\X-Plane 12\Custom Scenery`
- Artifacts: `H:\model_training`
- GPU Python: `G:\Dev\Ortho4XP\.venv\Scripts\python.exe`

## Commands

Create deterministic tile splits:

```powershell
& "G:\Dev\Ortho4XP\.venv\Scripts\python.exe" training\discover_tiles.py `
  --tiles-root "H:\XP12 Addons\Applications\Ortho4XP\Tiles" `
  --output "H:\model_training\manifests\tiles.jsonl"
```

Extract a small dry-run dataset. By default this references the existing DDS
textures directly and writes compact vector label JSON/metadata under the
output root:

```powershell
& "G:\Dev\Ortho4XP\.venv\Scripts\python.exe" training\extract_dataset.py `
  --splits "H:\model_training\manifests\tiles.jsonl" `
  --output-root "H:\model_training\scratch" `
  --max-tiles 2
```

Extract the full dataset:

```powershell
& "G:\Dev\Ortho4XP\.venv\Scripts\python.exe" training\extract_dataset.py `
  --splits "H:\model_training\manifests\tiles.jsonl" `
  --output-root "H:\model_training"
```

The default extractor keeps dense tiles. If you need a one-off emergency cap,
`--max-tile-features` can skip tiles above a feature count and records them in
`H:\model_training\manifests\skipped_tiles.jsonl`, but the normal training path
does not use that cap.

Use `--materialize-images` only if you intentionally want resized PNG copies in
the training root for portable experiments. For the local 1 TB+ texture library,
leave it off.

Extraction writes one SQLite/RTree vector label cache per tile, not per texture
and not as full-resolution raster label images. Manifest rows for each DDS
reference the shared tile label cache. The training `--chip-size` is only the
crop size sampled from the native DDS and rasterized in memory, so training
matches the coordinate system used later by overlay generation without
duplicating the image or label library.

Train:

```powershell
& "G:\Dev\Ortho4XP\.venv\Scripts\python.exe" training\train_placement_model.py `
  --dataset-manifest "H:\model_training\manifests\dataset.jsonl" `
  --run-dir "H:\model_training\runs\placement_v1" `
  --epochs 20 `
  --batch-size 4
```

## Pause And Resume

Training writes `last.pt`, `best.pt`, `metrics.jsonl`, and `failures.jsonl` after
every epoch. If the process crashes, rerun the same command and it resumes from
`last.pt` by default.

To pause cleanly after the current epoch, create:

```powershell
New-Item "H:\model_training\runs\placement_v1\PAUSE" -ItemType File
```

Remove that file and rerun the same training command to continue.

Use `--no-resume` only when intentionally starting a fresh run in an existing
directory.
