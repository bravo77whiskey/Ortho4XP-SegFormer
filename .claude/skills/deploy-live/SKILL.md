---
name: deploy-live
description: Deploy (sync) the repo's code to the user's live Ortho4XP app at G:\Dev\Ortho4XP with scripts/deploy_live.py - build, overlay everything but Ortho4XP_Data, verify byte for byte, import-check the live venv, clean up build leftovers. Use whenever the user says deploy, sync live, install the build, "so I can test it in the sim/app", or ships a change to the live app - and never copy files into the live folder by hand.
---

# Deploy to the live app

The live app is `G:\Dev\Ortho4XP` (exe + `_internal`, CUDA venv in `.venv`). The user runs real tile builds from it, so a deploy is the step that puts code in front of them. **Deploy only when the user asks.** Committing is a separate request.

## The step

1. **Decide what goes out.** The deploy builds whatever is checked out, normally `dev`. Uncommitted changes ride along, and the script prints that. If the user asked to deploy a feature branch, check it out first and say so. Don't deploy a branch the user didn't name.
2. **Run the tests** (`.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider`) unless they ran green on this exact tree in this session.
3. **Run the script in the background.** PyInstaller takes several minutes:
   ```
   .venv/Scripts/python.exe scripts/deploy_live.py
   ```
   Options:
   - `--dry-run`: checks and plan only.
   - `--skip-build`: reuse the current `dist/`.
   - `--keep-build`: keep `build/`, `dist_build/` and `dist/`. By default they are deleted, about 1 GB.
   - `--live DIR`: deploy to a different install.
4. **Read the result.** It must end with `[deploy] done`. Any `refused`, `MISMATCH` or import failure means the deploy is not complete. Report it, don't work around it.
5. **Record it.** Add one line to memory `deployment-sync-policy.md`: date, branch @ commit, what changed, and anything unusual.
6. **Tell the user** what is now live, and which batch steps or tile settings they need to re-run to see it.

## What the script guarantees (don't "simplify" it away)

- **It refuses while the live app or any process from its venv is running.** The user's builds use that python, and a half-copied module would break them. Wait. Never kill processes by name, only by PID, and only if the user says so.
- **`_internal/Ortho4XP_Data` is never written.** It holds `Ortho4XP.cfg`, logs, `build_state`, caches, the stock YOLO/DIOR models, and symlinks (Tiles, Providers, Patches...) into the H: install. Copying into it writes into another install. The script also fails if `Ortho4XP.cfg` changed during the deploy.
- **Nothing else in the live folder is deleted.** It keeps about 4 GB of legacy packages from older builds. Only entries present in the build are replaced.
- **Verification** is a byte compare of every copied entry plus the exe, ignoring runtime `__pycache__`. It also imports every loose `overlay_scripts/src/O4_*.py` with the live venv.

## Gotchas

- **A new module that loose overlay files import must be in `Ortho4XP.spec`, in both `added_datas` and `hiddenimports`.** Otherwise every live run dies with ModuleNotFoundError. `tests/spec_bundles_sfr_modules_test.py` guards this.
- **Some modules are frozen inside the exe PYZ** (e.g. `O4_Cfg_Vars`, `O4_Overlay_Pipeline`, `O4_DSF_Utils`, `O4_Tile_Utils`). Changes to them reach live only through a full build, so never copy a single loose file as a "quick deploy".
- **Live `Ortho4XP_Data\Providers` is a symlink into H:.** Provider `.lay` edits don't reach live through a deploy. Editing the H: file needs the user's OK.
- **Settings in tile cfgs** (`H:\...\Tiles\zOrtho4XP_*\*.cfg`) are user data. A deploy never changes them. Ask before editing one.
