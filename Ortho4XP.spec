# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Ortho4XP (with veg/building/AI overlay extensions).

Build command (run from repo root with the project venv active):
    python build.py          ← preferred (safe smart-merge into dist/)
    pyinstaller Ortho4XP.spec -y   ← raw PyInstaller (output in dist_build/)

Output:  dist/Ortho4XP/Ortho4XP.exe  (one-dir bundle)

NOTE: AI packages (torch, transformers, huggingface_hub) are NOT bundled into
the exe.  They live in sfr_venv/ next to the exe and are added to sys.path at
runtime by O4_SFR_Pipeline._activate_venv().  This keeps the bundle small
(<500 MB) and lets users upgrade to a CUDA torch wheel independently.

User data preserved on rebuild (never overwritten by build.py):
    Tiles, OSM_data, Orthophotos, Masks, yOrtho4XP_Overlays,
    yOrtho4XP_Veg_Overlays, yOrtho4XP_Bld_Overlays, Elevation_data, Geotiffs, tmp,
    Ortho4XP.cfg, .last_gui_params.txt, sfr_venv/
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# ── Paths ──────────────────────────────────────────────────────────────────────
SPEC_DIR = os.path.dirname(os.path.abspath(SPEC))

# ── Data files to bundle ───────────────────────────────────────────────────────
added_datas = [
    # Application resource directories (read-only; bundled inside _MEIPASS)
    (os.path.join(SPEC_DIR, "Utils"),     "Ortho4XP_Data/Utils"),
    (os.path.join(SPEC_DIR, "Providers"), "Ortho4XP_Data/Providers"),
    (os.path.join(SPEC_DIR, "Extents"),   "Ortho4XP_Data/Extents"),
    (os.path.join(SPEC_DIR, "Filters"),   "Ortho4XP_Data/Filters"),
    (os.path.join(SPEC_DIR, "Licence"),   "Ortho4XP_Data/Licence"),
    (os.path.join(SPEC_DIR, "community_server.txt"), "Ortho4XP_Data"),
    (os.path.join(SPEC_DIR, "Previews"),  "Ortho4XP_Data/Previews"),
    (os.path.join(SPEC_DIR, "Patches"),   "Ortho4XP_Data/Patches"),
    # SFR overlay scripts + AI inference module — run as venv subprocesses.
    # Bundled as loose .py files so the .venv Python can import them directly.
    # Keep a repo-like layout inside sfr_scripts/src so imports work the same
    # in source and packaged runs.
    (os.path.join(SPEC_DIR, "src", "scripts", "__init__.py"),            "sfr_scripts/src/scripts"),
    (os.path.join(SPEC_DIR, "src", "scripts", "generate_overlay.py"),    "sfr_scripts/src/scripts"),
    (os.path.join(SPEC_DIR, "src", "scripts", "generate_veg_overlay.py"),"sfr_scripts/src/scripts"),
    (os.path.join(SPEC_DIR, "src", "scripts", "generate_bld_overlay.py"),"sfr_scripts/src/scripts"),
    (os.path.join(SPEC_DIR, "src", "scripts", "generate_sfr_overlay.py"),"sfr_scripts/src/scripts"),
    (os.path.join(SPEC_DIR, "src", "scripts", "audit_sfr_building_assets.py"),"sfr_scripts/src/scripts"),
    # Remote-GPU offload: worker ships as a loose file because the packaged
    # app tar-syncs sfr_scripts/src to the remote host, which then runs it.
    (os.path.join(SPEC_DIR, "src", "scripts", "sfr_remote_worker.py"),   "sfr_scripts/src/scripts"),
    (os.path.join(SPEC_DIR, "src", "O4_AI_Overlay.py"),                  "sfr_scripts/src"),
    (os.path.join(SPEC_DIR, "src", "O4_Forest_Assets.py"),               "sfr_scripts/src"),
    (os.path.join(SPEC_DIR, "src", "O4_SFR_Asset_Inventory.py"),         "sfr_scripts/src"),
    (os.path.join(SPEC_DIR, "src", "O4_SFR_Bounds_Index.py"),            "sfr_scripts/src"),
    (os.path.join(SPEC_DIR, "src", "O4_SFR_Climate_Regions.py"),         "sfr_scripts/src"),
    (os.path.join(SPEC_DIR, "src", "O4_SFR_Persistent_Cache.py"),        "sfr_scripts/src"),
    (os.path.join(SPEC_DIR, "src", "O4_SFR_Region_Overrides.json"),      "sfr_scripts/src"),
    (os.path.join(SPEC_DIR, "src", "O4_SFR_Region_Boundaries.py"),       "sfr_scripts/src"),
    (os.path.join(SPEC_DIR, "src", "O4_SFR_Texture_Selection.py"),       "sfr_scripts/src"),
    (os.path.join(SPEC_DIR, "src", "O4_SFR_Remote.py"),             "sfr_scripts/src"),
    (os.path.join(SPEC_DIR, "src", "O4_SFR_Inference.py"),          "sfr_scripts/src"),
    (os.path.join(SPEC_DIR, "src", "O4_SFR_Building_Overlay.py"),   "sfr_scripts/src"),
    (os.path.join(SPEC_DIR, "src", "O4_SFR_Roof_Color.py"),         "sfr_scripts/src"),
    (os.path.join(SPEC_DIR, "src", "O4_SFR_Height_Model.py"),       "sfr_scripts/src"),
    (os.path.join(SPEC_DIR, "src", "O4_SFR_Stock_Yolo_Objects.py"), "sfr_scripts/src"),
    (os.path.join(SPEC_DIR, "src", "O4_SFR_DSF_Utils.py"),          "sfr_scripts/src"),
    (os.path.join(SPEC_DIR, "src", "O4_SFR_Vegetation_Overlay.py"), "sfr_scripts/src"),
]
# pyproj CRS data is collected automatically by the pyinstaller pyproj hook

# shapely and rtree ship their own DLLs; collect them as data so PyInstaller
# picks them up even if it misses them during the automatic analysis pass.
added_datas += collect_data_files("shapely")
added_datas += collect_data_files("rtree")

# ── Binaries (native DLLs needed at runtime) ──────────────────────────────────
# AI packages (torch, transformers, etc.) are NOT bundled — they live in
# sfr_venv/ and are loaded at runtime via _activate_venv().
added_binaries = []
added_binaries += collect_dynamic_libs("shapely")
added_binaries += collect_dynamic_libs("rtree")

# ── Hidden imports PyInstaller may miss ───────────────────────────────────────
hidden = [
    # tkinter (sometimes missed on Windows)
    "tkinter",
    "tkinter.ttk",
    "tkinter.filedialog",
    "tkinter.messagebox",
    # PIL / Pillow
    "PIL._tkinter_finder",
    # pyproj internals (hook handles CRS data automatically)
    "pyproj.datadir",
    # Shapely geometry types
    "shapely.geometry",
    "shapely.ops",
    "shapely.validation",
    # Our source modules
    "O4_UI_Utils",
    "O4_File_Names",
    "O4_Imagery_Utils",
    "O4_Vector_Map",
    "O4_Mesh_Utils",
    "O4_Mask_Utils",
    "O4_Tile_Utils",
    "O4_GUI_Utils",
    "O4_Config_Utils",
    "O4_Cfg_Vars",
    "O4_DSF_Utils",
    "O4_Overlay_Utils",
    "O4_Veg_Overlay",
    "O4_Building_Overlay",
    "O4_AI_Overlay",
    "O4_Forest_Assets",
    "O4_SFR_Asset_Inventory",
    "O4_SFR_Bounds_Index",
    "O4_SFR_Climate_Regions",
    "O4_SFR_Persistent_Cache",
    "O4_SFR_Pipeline",
    "O4_SFR_Region_Boundaries",
    "O4_SFR_Remote",
    "O4_SFR_Texture_Selection",
    "O4_SFR_Inference",
    "O4_SFR_Building_Overlay",
    "O4_SFR_Roof_Color",
    "O4_SFR_Height_Model",
    "O4_SFR_Stock_Yolo_Objects",
    "O4_SFR_DSF_Utils",
    "O4_SFR_Vegetation_Overlay",
    "O4_Parallel_Utils",
    "O4_OSM_Utils",
    "O4_Geo_Utils",
    "O4_DEM_Utils",
    "O4_Vector_Utils",
    "O4_Version",
    "O4_Zone_Utils",
    # SFR scripts are bundled as .py data files (see added_datas) and imported
    # lazily at runtime after _activate_venv() — not via the import graph.
]

# ── Modules to exclude ────────────────────────────────────────────────────────
# Explicitly exclude AI packages — they live in sfr_venv, not the bundle.
# NOTE: do NOT exclude setuptools/distutils — PyInstaller hooks them internally.
excludes = [
    "torch",
    "torchvision",
    "transformers",
    "huggingface_hub",
    "timm",
    "safetensors",
    "wand",
    "matplotlib",
    "IPython",
    "notebook",
    "pytest",
]

# ── Analysis ──────────────────────────────────────────────────────────────────
a = Analysis(
    ["Ortho4XP.py"],
    pathex=[SPEC_DIR, os.path.join(SPEC_DIR, "src")],
    binaries=added_binaries,
    datas=added_datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,       # onedir: binaries go in the bundle folder
    name="Ortho4XP",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,                # match master: keep one attached console for the whole app session
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Ortho4XP",
)
