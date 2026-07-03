# YOLO OBB ZL16 Analysis Texture Plan

Created: 2026-06-25

> **Status 2026-07-02: reverted to native-ZL inference by default.** The
> ZL16-analysis downsample is no longer the default path; trained YOLO runs on
> the native-resolution texture at every zoom level. The downsample machinery
> is kept behind `O4_SFR_BLD_YOLO_ANALYSIS_ZL=<zl>` for A/B comparisons. In the
> same change, cross-ZL double coverage (lower-ZL textures also covered by
> higher-ZL textures) is excluded at the source, which removes the dominant
> feed of the tile-wide placement dedup pass.

## Goal

Make Ortho4XP-SegFormer run the building YOLO OBB model on consistent ZL16-scale imagery, even when the scenery tile itself uses higher zoom textures such as ZL17/ZL18/ZL19.

The model currently performs much better at ZL16 scale. Native ZL19 inference sees too little ground per 512px crop, while an 8× downscaled ZL19 copy gives building-scale detections that look much more like the training distribution.

## Confirmed findings

1. Ortho4XP does not reliably keep lower-ZL parent textures when higher-ZL child textures fully cover the same area.
   - `src/O4_DSF_Utils.py::zone_list_to_ortho_dico()` assigns each mesh cell to exactly one texture attribute.
   - Higher-ZL zones overwrite the base/default ZL in the mask.
   - DSF generation queues/downloads only terrain-referenced textures.
   - `src/O4_Tile_Utils.py::remove_unwanted_textures()` removes DDS files not referenced by `.ter`.
   - Therefore a fully covered ZL16 parent should not be assumed to exist.

2. Current YOLO inference in Ortho4XP-SegFormer has an RGB/BGR bug.
   - `run_texture_regression.py` and `O4_SFR_Building_Overlay.py` pass RGB NumPy arrays to Ultralytics.
   - Ultralytics treats NumPy arrays as OpenCV/BGR.
   - The training project uses PIL/path input, so it feeds RGB correctly.
   - Fix: either pass PIL images, or convert RGB arrays to BGR before `model.predict()`.

3. Production YOLO cap is too low for dense building textures.
   - `src/O4_SFR_Building_Overlay.py` has `DEFAULT_YOLO_OBB_MAX_DET = 1000`.
   - Dense ZL16-equivalent city chips can exceed this.
   - Raise default to at least `3000`; for high-recall `conf=0.05`, consider `100000` or expose a clear setting.

## Desired design

Add a YOLO analysis texture cache separate from Ortho4XP production DDS files.

For each building-overlay source area, create or reuse a ZL16-scale analysis image:

- If an actual ZL16 texture exists for the area, use it.
- If only higher-ZL children exist, stitch/downsample them into a ZL16-equivalent image.
- Never overwrite or delete Ortho4XP scenery textures.
- Store derived images under a cache path such as:
  - `tmp/sfr_yolo_zl16_cache/<tile_label>/...`
  - or a project/tile cache directory if preferred for production reuse.

The production scenery can remain high-ZL. Only the YOLO building detector should see the ZL16-scale analysis image.

## Implementation plan

### 1. Fix YOLO input color handling first

Patch these paths:

- `.agents/skills/ortho4xp-yolo-obb-training/scripts/run_texture_regression.py`
- `src/O4_SFR_Building_Overlay.py`

Where a PIL/RGB crop is converted to NumPy and passed to `model.predict()`, use one of:

```python
source=np.ascontiguousarray(np.asarray(crop)[..., ::-1])
```

or keep the crop as a PIL image.

For production code that already has `image` as an RGB NumPy array, convert each crop:

```python
crop_bgr = np.ascontiguousarray(crop[..., ::-1])
```

Then pass `crop_bgr` to Ultralytics.

### 2. Raise the production max-det default

Patch:

- `src/O4_SFR_Building_Overlay.py`

Change:

```python
DEFAULT_YOLO_OBB_MAX_DET = 1000
```

to at least:

```python
DEFAULT_YOLO_OBB_MAX_DET = 3000
```

If the production run is intentionally high recall at `conf=0.05`, prefer exposing/using `100000` for diagnostics and dense areas.

### 3. Add ZL16 analysis texture builder

Add a small helper module or functions near the building overlay code.

Suggested function:

```python
def build_yolo_zl16_analysis_image(
    tex_dir: str | Path,
    til_y_top: int,
    til_x_left: int,
    provider: str,
    source_zl: int,
    target_zl: int = 16,
    cache_dir: str | Path | None = None,
) -> Path:
    ...
```

Expected behavior:

- Determine the ZL16 parent orthogrid for the requested texture footprint.
- Find child DDS textures at ZL16/ZL17/ZL18/ZL19 that cover that parent.
- If the parent ZL16 DDS exists, convert/copy it to RGB PNG in cache.
- Else stitch child textures into a parent canvas and downsample to 4096×4096 or 512×512 depending on chosen inference mode.

Recommended inference mode:

- Use 512×512 ZL16-equivalent images for each 4096×4096 high-ZL DDS footprint when testing one source DDS.
- For production, a 4096×4096 ZL16 parent image still maps naturally to existing 512-window inference with 8×8 crops.

Important: keep georeferencing metadata so detections can map back to the original DDS/tile coordinate system.

### 4. Integrate into building overlay inference

Patch:

- `src/O4_SFR_Building_Overlay.py`

When `yolo_enabled` is true:

1. Build/resolve a ZL16 analysis image for each DDS or parent area.
2. Run YOLO on that ZL16 analysis image.
3. Scale detections back to the production texture coordinate system before placement.

For a direct ZL19 DDS downscaled to ZL16:

```python
scale = 2 ** (source_zl - 16)
points_original = points_zl16 * scale
```

But if using parent-level cache/stitching, map through geographic bounds rather than assuming one source DDS equals one ZL16 parent.

### 5. Keep native high-ZL path optional

Eventually support two-pass inference:

- ZL16 analysis pass for normal/large building footprints.
- Optional native-ZL pass for tiny details if needed.

Do not implement this first. The first production fix should be simple and stable: ZL16-only YOLO analysis.

## Verification checklist

Run visual tests after patching:

1. Re-run corrected YOLO-only tests at `conf=0.05`.
2. Confirm counts match the training project/PIL-style inference.
3. Re-run Arc19 ZL16 downscale comparison.
4. Run both-layer overlay:
   - cyan = SegFormer zones
   - yellow = YOLO detections
5. Confirm no rectangular no-detection artifacts from RGB/BGR or per-crop cap.
6. Confirm original DDS files are untouched.

Useful existing artifacts from the investigation:

- `tmp/yolo_obb_tests/obb_combined_best_conf005_bgr_fixed_diagnostic/bgr_fixed_yolo_conf005_contact_sheet.jpg`
- `tmp/yolo_obb_tests/arc19_downscaled_to_zl16_quality/arc19_original_vs_downscaled_zl16_yolo_comparison.jpg`
- `tmp/yolo_obb_tests/obb_combined_best_conf005_segformer_zones_close24_both_layers_clear/close24_both_segformer_zones_and_yolo_contact_sheet.jpg`

## Open questions

- Should the ZL16 analysis cache be per tile, per DDS, or per parent ZL16 orthogrid cell?
- Should production default to `conf=0.05`, or should placement use a higher confidence after the ZL16/BGR fixes?
- Should generated ZL16 analysis images be retained permanently in the tile build directory or only under `tmp/`?

