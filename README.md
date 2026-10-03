# Ortho4XP-SegFormer

Ortho4XP-SegFormer is a fork of [Ortho4XP](https://github.com/oscarpilote/Ortho4XP),
the X-Plane scenery generator that builds the base mesh and photo texture layer
from orthophotos. On top of everything Ortho4XP does, it reads the tile's own
orthophotos with machine-learning models and generates two extra overlays:

- **Building overlay** (`yOrtho4XP_Bld_Overlays`): buildings detected in the
  imagery are placed as X-Plane objects and facades from the scenery libraries
  you have installed, matched by region, footprint size, shape and roof colour,
  with heights predicted from the image.
- **Vegetation overlay** (`yOrtho4XP_Veg_Overlays`): forest and scrub mapped
  from the imagery become X-Plane 12 forests, picked by climate and by
  broadleaf/conifer leaf-type maps, kept clear of existing buildings and
  (optionally) replacing the forests of lower-priority scenery.

The project started as a SegFormer experiment, hence the name. The overlays now
combine several models:

| Model | Role | Source |
| --- | --- | --- |
| SegFormer landcover (`nave1616/SegFormer-landcover-FT`) | vegetation mask, landcover context for buildings | Hugging Face, downloaded on first use |
| Trained YOLO OBB building detector | primary building detections | your own checkpoint (see [Models](#models)) |
| Stock YOLO OBB (`yolo26x-obb.pt`, DOTAv1) | storage tanks, pools, courts, cranes | Ultralytics, downloaded on first use |
| HeightNet | building heights | your own checkpoint |

## Requirements

- Windows 10 or 11. The overlay pipeline is developed and tested on Windows;
  the upstream Ortho4XP tools also ship macOS and Linux binaries.
- Python 3.12 when running from source.
- An NVIDIA GPU is strongly recommended for the overlays. They run on the CPU
  too, much more slowly, or on another machine's GPU (see
  [Remote GPU](#remote-gpu)).
- X-Plane 12. The overlays reference assets from X-Plane's own library and
  from the simHeaven X-World libraries, so install those for the placed
  objects to show up.

## Running from source

`install_windows.bat` does the following steps; to do them by hand, create
the virtual environment in `.venv` (the overlay steps look for it there):

```bash
python -m venv .venv
```

Install PyTorch in the build that matches your GPU before the other
requirements, otherwise `ultralytics` pulls in the CPU-only build:

```bash
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

```bash
.venv\Scripts\python -m pip install -r requirements.txt
```

Then start the GUI (or run `start_windows.bat`):

```bash
.venv\Scripts\python Ortho4XP.py
```

If PyTorch is missing when you first run an overlay, Ortho4XP installs it into
`.venv` itself, picking the CUDA build that matches your NVIDIA driver, and
downloads the SegFormer weights.

GDAL is optional; it is only needed for GeoTIFF elevation sources. On Windows
install it from a pre-built wheel (`Utils/win` has one for Python 3.13).

## Using the overlays

1. Build a tile as usual (Assemble vector data, Triangulate, Draw masks,
   Build Imagery/DSF). The overlays read that tile's textures.
2. Click **Building Overlay** and/or **Vegetation Overlay**. In the Tiles
   Collection window the same two steps can be ticked for batch builds, and
   interrupted batches can be resumed.
3. Tune the overlays in the configuration window: settings starting with
   `bld_` drive the building overlay, `veg_` the vegetation overlay and
   `segformer_` the landcover inference shared by both.

Inference results are cached per texture under `Overlay_cache/`, so changing
placement settings and regenerating is fast. Installs that already have an
`SFR_cache` folder keep using it.

The **Libraries** button rescans the installed scenery libraries the building
overlay chooses assets from; run it after installing or removing libraries.

Optional features with their own documentation:

- [Tile vegetation replacement](docs/tile_vegetation_replacement.md): remove
  the trees other scenery draws over a tile so only the generated forests
  remain, with verified backups and restore.

### Models

The trained building detector and HeightNet are not distributed with the
code. Point the `bld_yolo_checkpoint` and `bld_height_checkpoint` settings
(Global config tab) at your checkpoint files. The code that trains them is in
[`training/`](training/README.md).

### Remote GPU

The **Remote GPU** checkbox sends the model forward passes (SegFormer and
YOLO) to another machine over SSH for the current session; everything else
stays local. The picker lists the hosts in `~/.ssh/config`. The first
connection sets up a Python environment under `~/.ortho4xp_remote` on the
remote machine. If the host goes offline, the build carries on locally.

## Project layout

| Path | Contents |
| --- | --- |
| `Ortho4XP.py` | Entry point (GUI, or a command-line build for one tile) |
| `src/O4_*.py` | The application. Upstream Ortho4XP modules plus the overlay modules; see [docs/architecture.md](docs/architecture.md) |
| `src/scripts/` | Entry points the app runs in its `.venv` subprocess and ships with the build |
| `scripts/` | Developer tools: benchmarks, parity harnesses, the asset-library pipeline |
| `training/` | Training and dataset code for the building detector and HeightNet |
| `tests/` | Automated tests (pytest) |
| `docs/` | Design notes and references |
| `Providers/`, `Extents/`, `Filters/`, `Utils/`, `Previews/`, `Patches/` | Upstream Ortho4XP data and bundled tools |

## Development

Run the tests and the linter:

```bash
.venv\Scripts\python -m pytest
```

```bash
ruff check .
```

Test and lint settings live in `pyproject.toml`. Tests that need files outside
the repository (the generated asset manifest, pandas, a DSFTool binary for the
current platform) skip themselves when those are missing.

Build the Windows package (PyInstaller, configured in `Ortho4XP.spec`):

```bash
python build.py
```

The overlay modules run in the `.venv` python rather than inside the frozen
exe, so they are shipped as loose files. A new module imported by an overlay
module must be added to `OVERLAY_MODULES` in `Ortho4XP.spec`;
`tests/spec_bundles_modules_test.py` fails when one is missing.

More references:

- [docs/architecture.md](docs/architecture.md): how the pieces fit together.
- [docs/environment_variables.md](docs/environment_variables.md): developer
  switches read from the environment.
- [CHANGELOG.md](CHANGELOG.md): notable changes, including renamed settings.

## Credits and licence

Ortho4XP is © Oscar Pilote and contributors, licensed under the GNU GPL v3
(see `Licence/`); this fork is distributed under the same licence. It builds
on [shred86/Ortho4XP](https://github.com/shred86/Ortho4XP), whose changes are
listed in [docs/upstream_fork_changes.md](docs/upstream_fork_changes.md).

Bundled data: Köppen-Geiger climate classes from Beck et al. (2018), CC BY 4.0;
administrative boundaries derived from Natural Earth (public domain).
