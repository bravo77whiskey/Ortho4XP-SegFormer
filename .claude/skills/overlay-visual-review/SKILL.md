---
name: overlay-visual-review
description: Test and compare changes to the building and vegetation overlays (YOLO thresholds and models, placement rules, asset selection, de-dup, SegFormer, greenhouse sorting, any O4_Building_Overlay / O4_Vegetation_Overlay change) with the visual-review harness - run the review set, open the full-resolution A/B flip viewer, explain what its layers mean. Use this whenever the user wants to try, test, A/B, compare, review or judge a detection or placement change, asks why a building was or wasn't placed, wants to see results "on the textures", or mentions the review viewer, review runs, baseline runs, Kaohsiung / Shenzhen / Wenzhou / Shouguang test textures - even if they don't say "skill" or "harness".
---

# Overlay visual review

The user judges overlay quality **only by eye**: placed buildings and trees drawn over the source texture, flipped between two runs. Don't propose metric benchmarks (OSM matching, precision/recall); imagery dates differ between providers and fast-growing areas, so the picture is the ground truth. Your job is to make a run, make it easy to look at, and describe what changed - not to declare a winner.

## Where things are

| What | Where |
|---|---|
| Code (committed) | `scripts/visual_review/` - `run_review.py`, `fetch_set.py`, `index.html` (viewer) |
| Data (never on C:) | `E:\Ortho4XP_Review` (override with env `O4_REVIEW_DATA`): `sets/`, `runs/`, `_cache/w0,w1`, `_tmp/`, `runs_index.js`, served `index.html`, `_tools/` |
| Viewer server | `.claude/launch.json` config **overlay-review** -> `http://localhost:8791` (port 8765 is the user's own meet_recorder - never use it) |
| Settings used | The **installed** global config `G:\Dev\Ortho4XP\_internal\Ortho4XP_Data\Ortho4XP.cfg`, read-only, plus per-run `--cfg` overrides |
| Overlay GPU env | `G:\Dev\Ortho4XP\.venv` (CUDA). Run the scripts themselves with the repo `.venv` |

Everything a run writes stays under the data root. Runs disable Custom Scenery auto-linking and vegetation tile replacement, so nothing touches X-Plane or the user's real tiles. Keep it that way.

## The test set

All assessment uses these four places, 23 Bing textures in set `hotspots` (the user's choice - don't add other places unless asked):

- **Kaohsiung** NW->SW of KHH: `khh_nw`, `khh_n`, `khh_airport` (ZL16) - OSM-poor city, port, steel works
- **Shenzhen** N->E of SZX: `szx_n`, `szx_ne`, `szx_e` (ZL16) - OSM-poor urban villages
- **Wenzhou** N->SW of WNZ: `wnz_n`, `wnz_airport`, `wnz_w`, `wnz_sw` (ZL16) - greenhouses, farmland
- **Shouguang**: `shouguang` (ZL16) - greenhouse fields
- Plus `<loc>_z17`, `_z18`, `_z19` per location: one nested higher-zoom texture each.

`fetch_set.py [PROVIDER]` downloads or refreshes the set (it skips files already present). A/B is always **technique vs technique on the same imagery**, never Bing vs Google.

### Whole-tile sets

Changes that act across textures or tiles (seams, tile edges, tile-wide passes) need complete tiles, not samples. `make_tile_set.py SET TILE [TILE ...]` copies finished tiles read-only from the H: Tiles folder (textures, cfg, apt, mesh) to `sets/SET/tiles/` and writes a JPG per rendered texture. Runs on such a set build each tile complete, every zoom level together like a real build, one tile per worker; textures on a tile edge show both tiles' placements. The viewer's Location is the tile. Existing sets (user-approved, 2026-10-06): `gran-canaria` (+27-016/+28-016, 84 textures, ~16 min per run) and `taipei` (+25+121/+24+121, 250 textures, ~1 h per run). Use `--skip-veg` unless trees are being judged. Ask before adding tiles: the user picks them, and never read a tile folder while a build may be running.

## Making a comparison

1. **Make sure the reference exists.** Run B is compared against `hotspots-baseline` ("Your current installed settings"). The baseline is a snapshot of the production code and settings. Re-run it (same name, so it is replaced) whenever production behaviour changes: a committed improvement, a new model, or the user changing their installed config.
2. **Express the change.**
   - Setting only -> `--cfg KEY=VALUE` (any key from `src/O4_Cfg_Vars.py`, e.g. `--cfg bld_yolo_conf=0.02`). Nothing else to do.
   - Env-flag experiment -> `--env KEY=VALUE` (e.g. `--env O4_BLD_SEAM_OWNER=0`), passed to the overlay processes.
   - Code change -> edit `src/` directly; workers import the repo's `src`, so the run tests the working tree. Run the baseline *before* editing if it isn't current, because once the code changes, a fresh "baseline" would include the change. Experimental code paths behind an env flag are the safest way to keep the baseline reproducible.
3. **Run it** from the repo root, in the background (about 15 min for all 23 textures with 2 workers):
   ```
   .venv/Scripts/python.exe scripts/visual_review/run_review.py <run-name> --note "<what changed, in plain words>" [--cfg KEY=VALUE ...]
   ```
   - Always pass `--note`; it is what the user reads in the A/B dropdowns.
   - Use short kebab-case names that say what changed: `conf-0.02`, `overlap-tiles-25pct`, `dedup-smaller-asset`.
   - `--samples key ...` re-runs part of a run and keeps the rest.
   - `--workers N` sets the parallelism (default 2, GPU-shared).
4. **While it runs**, check the worker logs at `E:\Ortho4XP_Review\runs\<run>\_logs\worker_*.log` if anything looks stuck. The orchestrator prints per-texture counts and `done in N s` at the end.
5. **Show it**: start the `overlay-review` preview server (or tell the user to reload `http://localhost:8791`). The viewer opens with A = newest baseline and B = newest other run. Tell the user which run is B, what it changed, and a short per-location placement count comparison. Then point them at where to look: the textures where counts moved most, and the layers that explain the change.

Why the runner works the way it does (don't "simplify" it away):
- **Textures sharing a tile and zoom level are built together**, like a real tile build.
- **Different zoom levels never share a build.** The overlay drops lower-ZL detections that lie under higher-ZL textures, which would corrupt results.
- **Each worker has its own cache folder** so parallel workers never write the same file.
- **`TMP`/`TEMP` point at `E:\...\_tmp`** so transient caches stay off C:.

## What the viewer shows

Pick **Location**, then a texture. Space flips A/B, holding H shows the bare texture, 1-9 picks a texture, F fits, wheel zooms (native pixels past 100%). Hover any shape for details.

- **Placed output** (solid borders): Objects (orange), Facades (pink), Trees (green). "Detected box (before asset fit)" (white outline, off by default) is YOLO's box for placed buildings, for judging asset fit. "Kept with a smaller asset" (green box): buildings the de-dup would have dropped, kept with a smaller asset instead (production default since 2026-10-05; runs with `O4_BLD_DEDUP_FALLBACK=0` and runs from before then don't have it).
- **Why not placed** (from the review instrumentation; each candidate has exactly one outcome):
  - **Below confidence** (dashed grey): near misses down to 0.01, only where nothing above the threshold was found.
  - **Filtered**: smaller than `bld_min_footprint_m2`, oversized, overlapping a kept detection, or under a higher-ZL texture. Off by default; mostly YOLO duplicates.
  - **Blocked by existing scenery** (yellow): centre or footprint on another pack's object or facade.
  - **Centre on a road.**
  - **No asset fits.**
  - **Removed: asset overlapped a neighbour** (dashed orange box, with the removed asset filled orange): placed, then dropped by the end-of-run tile de-dup (>20% overlap with a smaller kept placement).
  - **Removed: smaller piece of a split building** (dashed blue): a building cut at a YOLO window edge or texture edge was detected as pieces; only the biggest piece keeps its placement (biggest piece wins, production default since 2026-10-06; off with `O4_BLD_SEAM_OWNER=0`).
  - **Grazing pairs** (production default since 2026-10-06; off with `O4_BLD_OVERLAP_FACADES=0`): two boxes that only graze both stay. Similar heights (within 6 m) merge into one facade at the biggest box's height; otherwise the higher-confidence box keeps the overlap and the other is trimmed to meet it. Hover a pink facade for "merged with grazing neighbours" or "trimmed back to meet a grazing neighbour"; absorbed boxes are in "Detected box (before asset fit)", and boxes with too little left after trimming are under Filtered.
  - **Trimmed back from existing scenery** (production default since 2026-10-06; off with `O4_BLD_TRIM_EXISTING=0`): a box whose footprint grazes only existing building scenery (at most 35% covered, not wholly inside) is cut back from it and placed as a polygon facade. Street furniture (bus stops, hydrants, toilets, construction markers...) and props under 25 m2 never block and are not drawn as blockers.
  - **Farmland / greenhouse / split** (off by default): open-farmland drops, greenhouse placements, boxes replaced by segmented outlines.
- **Trees** (vegetation coverage changes, default on since 2026-10-07, uncommitted until the user approves):
  - **Single trees (one per crown)** (lime octagons): canopy blobs under 600 m2 are planted as point-mode forests, one tree per crown at least 8 m apart, because X-Plane's 12 m forest grid gives such small polygons no tree most of the time. Off with `O4_VEG_POINT_TREES=0`.
  - **Trees removed: road exclusion** (magenta, fill only): canopy SegFormer found inside a road's clearance. simHeaven network roads now clear their carriageway by type (local 8 m wide, primary 16 m, motorway 26 m, powerlines 6 m) instead of a flat 16 m (`O4_VEG_ROAD_WIDTHS=0` restores the flat width), and roads crossing the texture edge are no longer bent into wedges along it (`O4_VEG_ROAD_CLIP=0` restores the bug).
  - **Trees removed: other exclusions** (orange, fill only, off by default): SegFormer road/building/developed buffer, existing buildings, mesh water, forests and exclusion zones.
  - Tree pixels also include weak tree evidence: tree as the top class under the 0.5 confidence cut, or tree at 0.30+ behind rangeland or water (`O4_VEG_TREE_RULE=0` off). It is cached as `<stem>_vegtree.npy` beside `_veg.npy`.
- **Existing scenery** (cyan, fill only, includes the 3 m clearance): simHeaven and other Custom Scenery objects and facades that block placement. Hover names the scenery pack and asset.
- **Display toggles**: *Texture & tile boundaries* (on): the texture's edge, other-ZL textures overlapping it (dashed, labelled), and whole-degree tile edges with the tile names either side. *YOLO window seams*: the 512 px grid YOLO reads each texture through, where buildings get split.

Display rules the user set: blockers are fill only with no border, and placed output has borders. Keep any new layer consistent with that.

## Changing the instrumentation or viewer

- The review data comes from `O4_BLD_REVIEW_DUMP=1` (set by the runner), plus `O4_BLD_REVIEW_CONF` for the near-miss floor.
- The data is written per texture as `<stem>_review.json`, only after the tile-wide de-dup has run.
- It must never change what is placed. After touching `src/O4_Building_Overlay.py`, prove that with the parity harness against the last commit, keeping its work files on E::
  ```
  mkdir -p /e/Ortho4XP_Review/_parity/old && git archive HEAD src | tar -x -C /e/Ortho4XP_Review/_parity/old   # Git Bash: tar reads "E:" as a remote host, so use /e/
  .venv/Scripts/python.exe scripts/building_overlay_parity_harness.py run --src E:/Ortho4XP_Review/_parity/old/src --out E:/Ortho4XP_Review/_parity/out_old --work E:/Ortho4XP_Review/_parity/work --dsftool G:/Dev/Ortho4XP-SegFormer/Utils/win/DSFTool.exe
  .venv/Scripts/python.exe scripts/building_overlay_parity_harness.py run --src src --out E:/Ortho4XP_Review/_parity/out_new --work E:/Ortho4XP_Review/_parity/work --dsftool G:/Dev/Ortho4XP-SegFormer/Utils/win/DSFTool.exe
  .venv/Scripts/python.exe scripts/building_overlay_parity_harness.py compare E:/Ortho4XP_Review/_parity/out_old E:/Ortho4XP_Review/_parity/out_new
  ```
  All scenarios must report IDENTICAL. `--work` must be the same path for both runs. `--dsftool` must be the absolute Windows path.
- Edit the viewer in `scripts/visual_review/index.html`. The runner copies it to the data root on every run; copy it by hand to see an edit without running.

## Gotchas

- **Never stop processes by name.** The user's own Ortho4XP builds use the same CUDA python. Stop by PID or process tree only, after identifying it from its command line.
- **Review runs and a real build compete for the GPU.** If the user has a build going, run with `--workers 1` or wait.
- **An old page needs a reload.** The viewer cache-busts its data on each load.
- **Pass Windows paths (`G:/...`, `E:/...`) to Python, not Git Bash `/g/...` paths.**
- **Run deletion:** removing a run means deleting `E:\Ortho4XP_Review\runs\<run>` and rebuilding `runs_index.js`. The next `run_review.py` call rewrites the index, or rebuild it from the `data_*.js` files.
