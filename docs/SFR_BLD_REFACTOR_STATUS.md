# SFR Building Overlay refactor — status & resume notes

Snapshot of in-progress work on the **facade-only YOLO + stock-YOLO pre-step + SegFormer-context facade variants** refactor.

Approved plan file: `C:\Users\oliver\.claude\plans\do-not-make-any-wiggly-pancake.md`.

---

## Headline state

Pipeline behaviour after the changes (build + redeploy required):

```
Per tile:
  1. SegFormer landcover inference  →  9-class veg_map  (always runs; bypass removed)
  2. Stock YOLO-OBB (DOTAv1 yolo26x-obb.pt) pre-step
       ├─ storage tank   → simheaven/facades/tank.fac     (facade, 10 m)
       └─ harbor (crane) → simheaven/landmarks/gantry-crane.obj  (OBJ)
       footprints marked into static_occ_mask + building_spacing_mask
  3. Trained YOLO-OBB facade pass  (the existing model)
       ├─ model class → placement class (heights come from model)
       └─ facade variant picked by SegFormer landcover context around each detection
  4. DSF text emit  (facades + objects ; draped polys disabled by user)
```

All five source files modified parse cleanly. Standalone smoke tests for the new module pass.

---

## What landed (chronological)

### A. DSFTool height bug (the original problem)
- **Root cause**: DSFTool 2.4.0-b1 silently drops `.fac` facade polygons whose `BEGIN_POLYGON` height parameter contains a decimal point (e.g. `3.5`, `10.0`). Forest `.for` polygons accept floats fine; facades require **integer** heights.
- Confirmed with hand-crafted test files: `BEGIN_POLYGON 0 3.5 2` produces an empty DSF; `BEGIN_POLYGON 0 4 2` round-trips correctly.
- **Fix**: `{height_m:.1f}` → `{int(round(height_m))}` at five facade-writer sites:
  - [src/O4_SFR_Building_Overlay.py:7671](src/O4_SFR_Building_Overlay.py:7671)
  - [src/O4_SFR_Inference.py:1142](src/O4_SFR_Inference.py:1142)
  - [src/O4_AI_Overlay.py:1016](src/O4_AI_Overlay.py:1016)
  - [src/O4_Veg_Overlay.py:618](src/O4_Veg_Overlay.py:618)
  - [src/O4_Building_Overlay.py:292](src/O4_Building_Overlay.py:292)
- DSFTool also emits useful warnings on stderr that were swallowed by `compile_dsf`. Now printed even on success — see [src/O4_SFR_Inference.py:1166](src/O4_SFR_Inference.py:1166).
- The `_bld.txt` intermediate is **preserved on disk** after a successful compile for debugging (delete commented out at [src/O4_SFR_Building_Overlay.py:7716](src/O4_SFR_Building_Overlay.py:7716)). Re-enable the delete once the pipeline is stable.

### B. Removed `direct_yolo_only` / `yolo_fit_objects` flags
- **Pipeline / config / GUI surface** stripped:
  - [src/O4_SFR_Pipeline.py](src/O4_SFR_Pipeline.py) — both module-level defaults removed; subprocess code-string no longer passes either kwarg.
  - [src/O4_Cfg_Vars.py](src/O4_Cfg_Vars.py) — both UI entries removed.
  - [src/O4_GUI_Utils.py:748](src/O4_GUI_Utils.py:748) and [src/O4_Tile_Utils.py:411](src/O4_Tile_Utils.py:411) — tile-config propagation lines deleted.
- **In Building Overlay**, the parameters still exist on `run()` for backwards compat but are **forced to their effective values** at the top of the function:
  - `direct_yolo_only = True` (locked)
  - `yolo_fit_objects = False` (locked)
  - Env-var overrides removed.
- Behavior: pipeline is permanently facade-first, SegFormer-zone fill paths are unreachable.
- **Caveat (carry-over work)**: ~500–1000 lines of now-unreachable code remains (`_maybe_fit_yolo_object_asset`, `_find_largest_object_asset_for_yolo`, every `if not direct_yolo_only:` branch). Compiles + runs fine; cosmetic cleanup is a follow-up.

### C. SegFormer landcover always runs
- The `direct_yolo_only` branch that zeroed `veg_map` at [src/O4_SFR_Building_Overlay.py:5720](src/O4_SFR_Building_Overlay.py:5720) is gone. SegFormer inference now runs on every DDS and the per-DDS `_veg.npy` cache writes always happen (was previously skipped in direct-YOLO mode).

### D. Context-aware facade variant picker
- New `_facade_for_detection()` at [src/O4_SFR_Building_Overlay.py:2548](src/O4_SFR_Building_Overlay.py:2548) and accompanying `CONTEXT_FACADE_VARIANTS` table (29 entries).
- For each YOLO facade detection, samples a ~50 m window of the SegFormer landcover map around the detection center, picks the **dominant non-building class** (DEVELOPED / ROAD / AGRICULTURE / BARELAND / TREE / WATER / RANGELAND), and looks up a variant pool keyed by `(placement_class, dominant_landcover)`.
- Variant within the pool selected by a stable hash of `(lat, lon, jx, jy)` so identical detections always pick the same path (reruns deterministic).
- Pools mix XP12 lib paths (`lib/buildings/facades/generic/*`, `lib/buildings/facades/commercial/*`, `lib/buildings/facades/industrial/*`) with **simHeaven library group exports** (`simheaven/facades/residential.fac`, `building.fac`, `commercial.fac`, `industrial.fac`) — sim auto-randomizes within those at load time.
- Wired in at both facade-placement sites: direct YOLO ([line ~6442](src/O4_SFR_Building_Overlay.py:6442)) and YOLO-template ([line ~6896](src/O4_SFR_Building_Overlay.py:6896)).

### E. Heights now come from the trained YOLO model
- Trained `yolo26x v1` checkpoint at `H:\model_training\runs\yolo_obb_v1\weights\visual_candidate_step_12000.pt` has 8 classes that align with `BLD_PLACEMENT_CLASSES` (model emits 0..7; codebase uses 1..8 — shift by +1).
- [src/O4_SFR_Building_Overlay.py:3724](src/O4_SFR_Building_Overlay.py:3724) (`_yolo_obb_detection_from_points`) now sets `placement_class = model_class + 1` when in range, falling back to the area heuristic only if out of range. Heights are looked up from `DEFAULT_FACADE_HEIGHT_M` keyed by the model's class.
- **Heights table re-balanced** for warehouse reality (big footprint ≠ tall):
  - `BLD_CLASS_LARGE`: 14 m → **8 m**  (single-story high-bay warehouse)
  - `BLD_CLASS_EXTRA_LARGE`: 16 m → **10 m**  (big-box / distribution center)
  - Other tiers unchanged.

### F. Cache schema bump
- The per-DDS `_bld.pkl` cache tuple now includes `"schema=v2-facades-only-stockyolo"` at [src/O4_SFR_Building_Overlay.py:5805](src/O4_SFR_Building_Overlay.py:5805). Old caches auto-invalidate on next run.

### G. `placed_objects` → `placed_stock_objects` rename
- Global Edit replace_all in the building overlay. The list is now populated by the stock-YOLO pre-step (E below), not by the (now-dead) building-as-OBJ fitter.

### H. Depth-Anything-V2 height-prototype experiment — **failed (documented)**
- Standalone prototype at [scripts/depth_height_prototype.py](scripts/depth_height_prototype.py) ran Depth-Anything-V2-Small + SegFormer-anchored linear calibration on Arc18 and Arc16 DDS tiles of `+22+120`.
- **ZL18 result**: every class median collapsed to ~8.3 m with p25-p75 spread of 0.05-0.1 m — flat constant signal, no usable per-building height information.
- **ZL16 result**: wider spread but the linear fit was essentially fitting the priors we were trying to replace (scale=50). Within-class spread larger than between-class spread.
- **Cause**: orthorectification removes the parallax/displacement cues monocular depth models rely on. The geometric signal Depth Anything is trained to exploit is largely absent in nadir orthos. Foundation depth models are wrong-domain for this.
- **Conclusion**: per-pixel monocular depth is **not** a viable height source for this pipeline. The viable paths remaining are: (i) re-label YOLO training data with finer height-correlated classes, or (ii) add a height regression head to the YOLO. Both require training-data work.

### I. New stock YOLO-OBB pre-step
- New module: [src/O4_SFR_Stock_Yolo_Objects.py](src/O4_SFR_Stock_Yolo_Objects.py).
- Uses `G:\Dev\Ortho4XP-SegFormer\yolo26x-obb.pt` (DOTAv1, 15 classes, 127 MB).
- **Asset map** (final, after the "no draped polys" pruning):
  | DOTA class | Placement type | Asset |
  |---|---|---|
  | 2 storage tank | facade (10 m) | `simheaven/facades/tank.fac` |
  | 7 harbor       | object        | `simheaven/landmarks/gantry-crane.obj` |
- Sports fields, swimming pools temporarily **disabled** — they require draped `.pol` polygons which user wants to avoid. Re-add when proper OBJ assets are mapped.
- All "moving" classes (plane, ship, helicopter, vehicles) excluded by design.
- Bridge and roundabout excluded (X-Plane mesh / road network already renders them).
- Per-class size sanity filters (`_MIN_LONG_SIDE_M` / `_MAX_LONG_SIDE_M`).
- Standalone runner: `python -m O4_SFR_Stock_Yolo_Objects <image> --lat 22 --lon 120 --m-per-px 2.4 [--out-dsf out.txt]`.
- **Standalone test results**:
  - `114176_218672_Arc18.dds` (`+22+120`, ZL18, 2.4 m/px): **29 storage tank detections** + 1 tennis court (now filtered out by the static class list), ~25.6 s CPU.
  - `28480_54656_Arc16.dds` (`+22+120`, ZL16, 9.6 m/px): **0 detections** — see resolution note below.
  - `112160_219360_Arc18.dds` (`+25+121`, ZL18, 2.4 m/px): **230 storage tank detections**, ~21 s CPU.
  - `28032_54832_Arc16.dds` (`+25+121`, ZL16, 9.6 m/px): **0 detections**.
- **Resolution sensitivity** (important): the DOTAv1 stock model was trained on imagery at ~0.5-2 m/px. At ZL16 (9.6 m/px) storage tanks are only ~2-6 pixels across and the model finds nothing. At ZL17+ (≤4.8 m/px) detection works well. **Suggested optimization (not yet done):** in the building-overlay integration, skip the stock-YOLO call when the DDS is ZL ≤ 16 to save ~25 s/file. Detect via the `Arc<N>` suffix in `fname` or via `zl` already parsed at [src/O4_SFR_Building_Overlay.py:5901](src/O4_SFR_Building_Overlay.py:5901): `if zl <= 16: stock_yolo_model = None  # locally skipped`. Add this when ready.

### J. Integration into Building Overlay
- Stock-YOLO module imported as `STOCKYOLO` at the top of `O4_SFR_Building_Overlay.py`.
- Model loaded once per `run()` invocation, parallel to the trained-YOLO model load, with graceful fallback if the checkpoint isn't on disk.
- Per-DDS: stock-YOLO inference fires immediately after the trained-YOLO inference (both share the loaded `img`), recording OBB pixel quads in `stock_yolo_occupied_polys`.
- OBB quads are `cv2.fillPoly`-marked into **both** `static_occ_mask` and `building_spacing_mask` before the trained-YOLO facade loop runs — so the loop excludes those pixels and facades don't overlap the static objects.
- Stock-YOLO placements append to the same global `placed_facades` / `placed_stock_objects` / `placed_draped` lists used by the DSF writer. (`placed_draped` is currently unused at runtime because the asset map has no draped entries, but the plumbing is in place for when it's re-enabled.)
- DSF text writer extended to emit `.pol` draped polygons alongside `.fac` facades (`BEGIN_POLYGON idx 0 2`), sharing the same `POLYGON_DEF` table.
- Per-DDS console line now shows e.g. `[Bld stage] X.dds stock YOLO: storage tank=29`.
- Total-line updated: `Total: N placements (K stock-yolo objects, F facades, D draped polys) trained_yolo_facades=…`.

---

## Files touched

| File | Nature of changes |
|---|---|
| [src/O4_SFR_Building_Overlay.py](src/O4_SFR_Building_Overlay.py) | Picker, stock-YOLO integration, height map, cache schema, rename, DSFTool fix, locked flags |
| [src/O4_SFR_Inference.py](src/O4_SFR_Inference.py) | DSFTool stdout/stderr surfacing, facade-height int formatting |
| [src/O4_AI_Overlay.py](src/O4_AI_Overlay.py) | Facade-height int formatting (same DSFTool bug) |
| [src/O4_Veg_Overlay.py](src/O4_Veg_Overlay.py) | Facade-height int formatting |
| [src/O4_Building_Overlay.py](src/O4_Building_Overlay.py) | Facade-height int formatting |
| [src/O4_SFR_Pipeline.py](src/O4_SFR_Pipeline.py) | Drop direct_yolo_only / yolo_fit_objects vars + subprocess args |
| [src/O4_Cfg_Vars.py](src/O4_Cfg_Vars.py) | Drop UI entries for removed flags |
| [src/O4_GUI_Utils.py](src/O4_GUI_Utils.py) | Drop GUI propagation |
| [src/O4_Tile_Utils.py](src/O4_Tile_Utils.py) | Drop tile-config propagation |
| **NEW** [src/O4_SFR_Stock_Yolo_Objects.py](src/O4_SFR_Stock_Yolo_Objects.py) | DOTAv1 stock-YOLO pre-step module |
| **NEW** [scripts/depth_height_prototype.py](scripts/depth_height_prototype.py) | Diagnostic only — kept as evidence the depth route is unviable |

---

## What's NOT done (resume queue)

In suggested order:

1. **Pick a test image and run the new stock-YOLO module standalone.**
   The user wants to verify storage tank detection on a coastal/agricultural tile with visible white tank fields. Action item:
   - `python -m O4_SFR_Stock_Yolo_Objects <image_path> --lat <lat> --lon <lon> --m-per-px <gsd>`
   - Expect to see `storage tank=N` in the by-class summary.

2. **`python build.py` + redeploy to runtime** at `G:\Dev\Ortho4XP\_internal\sfr_scripts\src\`, re-run `+22+120` in facade-only YOLO mode end-to-end. Expected console signals:
   - `Stock YOLO OBB (DOTAv1): loaded ... ; keeping classes (2, 7)` at startup.
   - SegFormer inference timings non-zero (was previously 0.0s in direct-YOLO mode).
   - Per-DDS: `[Bld stage] X.dds stock YOLO: storage tank=N` lines where applicable.
   - Final total line: `Total: N placements (K stock-yolo objects, F facades, 0 draped polys)`.
   - Compiled DSF should contain `POLYGON_DEF simheaven/facades/tank.fac` and `OBJECT_DEF simheaven/landmarks/gantry-crane.obj` where detected.

3. **Dead-code cleanup** (cosmetic). Touch points all flagged in the plan file; ~500–1000 lines:
   - Delete `_maybe_fit_yolo_object_asset` and `_find_largest_object_asset_for_yolo`.
   - Delete every `if not direct_yolo_only:` branch (all are now unreachable).
   - Delete `if yolo_fit_objects:` branches.
   - Delete the locked `direct_yolo_only` / `yolo_fit_objects` parameters from `run()` once no internal references remain.
   - Drop the `direct_yolo_object_placements` counter and the obsolete `if direct_yolo_only:` assertion block at the end of `run()` (lines ~7611-7622).

4. **Test suite update**:
   - [tests/sfr_sfd_building_assets_test.py](tests/sfr_sfd_building_assets_test.py) may reference removed config keys.
   - Add a test for `_facade_for_detection` covering each `(placement_class, dominant_landcover)` pair returns a non-empty path.
   - Add a smoke test for `O4_SFR_Stock_Yolo_Objects.STOCK_YOLO_ASSET_MAP` invariants (all keys in `STATIC_DOTA_CLASSES`; all paths look like `simheaven/...` or `lib/...`).

5. **Re-enable sports / pools** (optional, when ready):
   - Find/source OBJ assets (or stadium-style facades) for tennis courts, soccer fields, pools, basketball courts, baseball diamonds.
   - Add entries to `STOCK_YOLO_ASSET_MAP` and re-list the classes in `STATIC_DOTA_CLASSES`.
   - The `placed_draped` plumbing already exists in the building overlay if `.pol` becomes acceptable again.

6. **Re-enable `_bld.txt` cleanup** ([line 7716](src/O4_SFR_Building_Overlay.py:7716)) after the pipeline is stable. The debug-preserve was left on while the DSFTool bug investigation was active.

7. **Height refinement** (deferred, no quick win):
   - Monocular depth foundation models (Depth Anything V2 etc.) **proven ineffective** — see Section H.
   - Realistic next options if heights still need improvement:
     - **(a) Finer YOLO classes.** Split `large` → `large_warehouse` / `large_office`; split `apartment_block` → `walk_up` / `mid_rise`. Annotate a subset of training data with the finer labels and retrain. Easier than regression.
     - **(b) Height regression head on YOLO.** Add a continuous height output. Requires training-data labels — bootstrap from LIDAR DSM where available (USGS 3DEP for US, EU national LIDAR for some countries).
     - OSM `building:height` was rejected by the user because coverage is exactly the problem this project exists to solve.

---

## Quick reproducer commands

```powershell
# Parse-check all modified files
$py = 'G:\Dev\Ortho4XP-SegFormer\.venv\Scripts\python.exe'
foreach ($f in @(
  'G:\Dev\Ortho4XP-SegFormer\src\O4_SFR_Building_Overlay.py',
  'G:\Dev\Ortho4XP-SegFormer\src\O4_SFR_Stock_Yolo_Objects.py',
  'G:\Dev\Ortho4XP-SegFormer\src\O4_SFR_Inference.py',
  'G:\Dev\Ortho4XP-SegFormer\src\O4_SFR_Pipeline.py',
  'G:\Dev\Ortho4XP-SegFormer\src\O4_Cfg_Vars.py',
  'G:\Dev\Ortho4XP-SegFormer\src\O4_GUI_Utils.py',
  'G:\Dev\Ortho4XP-SegFormer\src\O4_Tile_Utils.py'
)) { & $py -m py_compile $f }

# Standalone stock-YOLO smoke run
cd G:\Dev\Ortho4XP-SegFormer\src
$py = 'G:\Dev\Ortho4XP-SegFormer\.venv\Scripts\python.exe'
& $py -m O4_SFR_Stock_Yolo_Objects `
    'G:\Dev\Ortho4XP\_internal\Ortho4XP_Data\Tiles\zOrtho4XP_+22+120\textures\114176_218672_Arc18.dds' `
    --lat 22 --lon 120 --m-per-px 2.4 --conf 0.20

# Full tile rebuild (from clean repo state)
cd G:\Dev\Ortho4XP-SegFormer
& $py build.py
# Then copy src/O4_SFR_Building_Overlay.py and src/O4_SFR_Stock_Yolo_Objects.py
# to G:\Dev\Ortho4XP\_internal\sfr_scripts\src\, then run the tile from the UI.
```

---

## Key invariants to preserve

- **Facade heights must serialize as integers** in DSF text. The DSFTool 2.4.0-b1 bug is real; any new facade writer must use `{int(round(h))}` not `{h:.1f}`.
- **`BLD_PLACEMENT_CLASSES` is 1..8** but the trained YOLO emits **0..7**. Always shift `model_class + 1` when mapping.
- **simHeaven facade library group paths** (`simheaven/facades/{residential,building,commercial,industrial,greenhouse}.fac`) auto-randomize at sim load — referencing them in DSF gives instant variety without enumerating individual files.
- **Cache schema is `v2-facades-only-stockyolo`** — bump again on any schema-breaking change.
- **Stock-YOLO classes are locked to (2, 7)** until non-draped assets exist for sports/pools.
