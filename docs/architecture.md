# Architecture

Ortho4XP-SegFormer is the Ortho4XP application with two overlay generators
added. This page describes how the overlays are wired into it; the upstream
build steps (vector data, mesh, masks, imagery/DSF) work as in Ortho4XP.

## Two Python processes

The app itself (the GUI, or `Ortho4XP.exe` when packaged) never imports
PyTorch. Each overlay step runs in a separate **`.venv` python** that has
torch, transformers and ultralytics installed:

```
GUI / batch build (main process)
  O4_Tile_Utils / O4_GUI_Utils
    -> O4_Overlay_Pipeline.apply_tile_settings(tile, ...)   copy the tile's bld_/veg_/segformer_ settings
    -> O4_Overlay_Pipeline.process_bld_tile / process_veg_tile
         builds a short Python program and runs it with the .venv python
              import O4_Building_Overlay as bld_overlay
              bld_overlay.run(tex_dir=..., lat=..., ...)
```

In a source checkout the `.venv` is the one in the repository root and the
subprocess imports the modules straight from `src/`. In the packaged build the
`.venv` sits next to the exe and the overlay modules are shipped as loose
files in `_internal/overlay_scripts/src` (listed in `Ortho4XP.spec` as
`OVERLAY_MODULES`), because the frozen exe cannot hand its bundled modules to
another interpreter. `O4_Overlay_Pipeline` creates the `.venv` and installs
the dependencies on the first overlay run.

### Settings flow

1. `O4_Cfg_Vars` declares every setting (type, default, GUI hint).
2. `O4_Config_Utils` loads the global and per-tile `.cfg` files into a `Tile`
   object, translating old key names through `normalize_config_entry`.
3. Before an overlay step, `O4_Overlay_Pipeline.apply_tile_settings` copies the
   settings named in `O4_Cfg_Vars.list_bld_pipeline_vars` or
   `list_veg_pipeline_vars` onto module globals of the pipeline.
4. The pipeline writes them as keyword arguments into the subprocess program.

Developer switches that are not user settings are read from `O4_*`
environment variables; see [environment_variables.md](environment_variables.md).

### Remote GPU

When the session's Remote GPU option is on, only the model forward passes
leave the machine. `O4_Remote_GPU` syncs `src/` to the remote host over SSH,
starts `src/scripts/remote_gpu_worker.py` there, and swaps the model handles
in `O4_Landcover_Inference`, `O4_Building_Overlay` and `O4_Stock_Yolo_Objects`
for proxies that send images and receive class maps or detections. Everything
else (caching, placement, DSF writing) stays local.

## Module map

### Application and upstream build steps

| Module | Purpose |
| --- | --- |
| `Ortho4XP.py` | Entry point: GUI, or a command-line build of one tile |
| `O4_GUI_Utils`, `O4_GUI_Theme`, `O4_Config_Utils`, `O4_Cfg_Vars` | Main window, config window, settings |
| `O4_Tile_Utils` | Tile and batch builds; runs the steps in order |
| `O4_Build_State` | Journals for resumable batch builds |
| `O4_Vector_Map`, `O4_OSM_Utils`, `O4_PBF_Utils`, `O4_Airport_Utils` | Vector data from OSM (Overpass or a local planet file) |
| `O4_Mesh_Utils`, `O4_DEM_Utils`, `O4_Mask_Utils`, `O4_Bathymetry` | Mesh, elevation, water masks |
| `O4_Imagery_Utils`, `O4_DSF_Utils`, `O4_Overlay_Utils` | Orthophoto download, terrain DSF, upstream overlay extraction |
| `O4_Scenery_Links` | Custom Scenery links to built tiles |
| `O4_File_Names`, `O4_Geo_Utils`, `O4_Vector_Utils`, `O4_UI_Utils`, `O4_Parallel_Utils`, `O4_Zone_Utils` | Shared helpers |

### Overlay orchestration (main process)

| Module | Purpose |
| --- | --- |
| `O4_Overlay_Pipeline` | Settings, `.venv` setup, launching the overlay subprocesses, cache cleanup |
| `O4_Remote_GPU` | SSH remote-inference client and host discovery |

### Shared overlay infrastructure (`.venv` process)

| Module | Purpose |
| --- | --- |
| `O4_Landcover_Inference` | SegFormer landcover inference, texture discovery, DSFTool compile |
| `O4_Texture_Selection` | Which textures (and which part of each) an overlay may read |
| `O4_Scenery_Packs` | Finding active scenery packs and their DSFs, cached DSF disassembly |
| `O4_Library_Catalog`, `O4_Asset_Inventory` | Catalog of installed X-Plane libraries and their exported assets |
| `O4_Region_Boundaries`, `O4_Region_Overrides.json`, `O4_Climate_Regions` | Region and climate lookups for asset choice |
| `O4_Persistent_Cache`, `O4_Bounds_Index` | On-disk sidecar caches, fast bounding-box queries |

### Building overlay

| Module | Purpose |
| --- | --- |
| `O4_Building_Overlay` | The building pass: detections, filtering, asset choice, placement, DSF output. `run()` drives `_BuildingOverlayRun` (tile-wide stages) and one `_TexturePass` per texture |
| `O4_Stock_Yolo_Objects` | Stock DOTA YOLO pass for tanks, pools, courts, cranes |
| `O4_Height_Model`, `O4_Height_Priors` | HeightNet building heights and regional limits |
| `O4_Footprint_Polygons` | Polygon footprints for large non-rectangular buildings |
| `O4_Agri_Structures` | Greenhouses and farmland false positives |
| `O4_Roof_Color` | Roof colour matching between imagery and assets |
| `O4_Classical_Buildings` | Experimental non-neural detector (regression harness only) |
| `O4_Overlay_Debug_Images` | PNG previews of masks and placements for chosen textures (developer tool) |

### Vegetation overlay and tile vegetation replacement

| Module | Purpose |
| --- | --- |
| `O4_Vegetation_Overlay` | Forest polygons from the landcover map, forest choice, DSF output |
| `O4_Forest_Assets`, `O4_Leaf_Type` | simHeaven / X-Plane 12 forest assets, leaf-type maps |
| `O4_Veg_Replacement`, `O4_DSF_Edit` | Removing other scenery's trees from a tile, with backups (see [tile_vegetation_replacement.md](tile_vegetation_replacement.md)) |

## Files on disk

| Location | Contents |
| --- | --- |
| `Overlay_cache/<grid>/<tile>/` | Per-texture inference and placement caches (`*_veg.npy`, `*_yolo_obb.pkl`, `*_bld.pkl`, ...) plus parsed OSM/DSF sidecars. Installs from before the rename keep using `SFR_cache/`. |
| `Overlay_cache/_libraries/library_catalog.pkl` | Scan of the installed scenery libraries (Libraries button) |
| `Overlay_cache/_leaf_type/` | Downloaded leaf-type map tiles |
| `yOrtho4XP_Bld_Overlays/`, `yOrtho4XP_Veg_Overlays/` | Generated overlay scenery packs |
| `yOrtho4XP_Veg_Replacement/` | Originals and journal of tile vegetation replacement; the only copy of edited DSFs |
| `build_state/` | Journals of unfinished batch builds |

In the packaged build these live under `_internal/Ortho4XP_Data/`.

## Names that must not change

Some identifiers are written into files outside this repository and are kept
as they are even though they predate the current naming:

- `o4sfr/...` virtual paths, the `O4SFR_Library` / `O4SFR_ProcGen_Library`
  scenery folders and the `o4sfr` library key: they are referenced from
  installed libraries and from every generated building DSF.
- The `sfr_*` setting names, the `sfr_bld` / `sfr_veg` journal steps, the
  `SFR_cache` folder and the remote `~/.ortho4xp_sfr` folder are still read so
  older configs, unfinished batches, caches and remote setups keep working
  (see [CHANGELOG.md](../CHANGELOG.md)).

## Checking a refactor of the building overlay

`scripts/building_overlay_parity_harness.py` runs `O4_Building_Overlay.run()`
on a generated tile (textures, OSM, scenery packs, mesh water) with stand-in
models, over a matrix of settings, and records hashes of every output. Run it
on the old and the new source tree and compare; it needs no GPU and no real
scenery. It complements, not replaces, a byte-diff of real tiles.
