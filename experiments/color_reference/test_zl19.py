"""ZL19 group starting halfway through a ZL14 reference tile (regression test)."""
import os, sys, shutil
from types import SimpleNamespace
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
os.chdir(REPO); sys.path.insert(0, os.path.join(REPO, "src"))
DATA = os.environ.get("O4_REVIEW_DATA", r"E:\Ortho4XP_Review")
import O4_File_Names as FNAMES
FNAMES.Imagery_dir = os.path.join(DATA, "color", "imagery")
sys.path.append(FNAMES.Provider_dir)
import O4_Imagery_Utils as IMG, O4_Geo_Utils as GEO, O4_Overlay_Pipeline as PIPELINE, O4_Color_Build as CCB
PIPELINE._venv_python = lambda: r"G:\Dev\Ortho4XP\.venv\Scripts\python.exe"
IMG.initialize_extents_dict(); IMG.initialize_color_filters_dict(); IMG.initialize_providers_dict()
IMG.local_combined_providers_dict = {}
lat, lon = 29.7580, -95.3670
tx, ty = GEO.wgs84_to_orthogrid(lat, lon, 19)
if (tx // 16) % 2 == 0:
    tx += 16                                    # start mid ZL14 tile
build = os.path.join(DATA, "color", "_tilebuild", "zl19")
shutil.rmtree(build, ignore_errors=True); os.makedirs(os.path.join(build, "textures"))
tile = SimpleNamespace(lat=29, lon=-96, build_dir=build, imprint_masks_to_dds=False,
                       color_correction=True, color_fill_clouds=True)
attrs = [(tx + 16 * i, ty, 19, "BI") for i in range(3)]
for a in attrs:
    IMG.build_jpeg_ortho(tile, *a)
done = CCB.run(tile, attrs)
print(f"ZL19 group starting mid-tile (x0 % 32 = {tx % 32}): {len(done)}/{len(attrs)} converted")
