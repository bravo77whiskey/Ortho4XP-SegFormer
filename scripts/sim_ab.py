"""In-sim A/B of one tile: swap a variant, then restart X-Plane at the same
saved situation (Reload Scenery is not trusted to pick up every change).

    python scripts/sim_ab.py build     --tile +22+120 [--out E:/Ortho4XP_Review/sim_ab]
    python scripts/sim_ab.py install   --tile +22+120      (X-Plane closed)
    python scripts/sim_ab.py grass on|off      --tile +22+120
    python scripts/sim_ab.py mat 2048|1024|off --tile +22+120
    python scripts/sim_ab.py mat-build --tile +22+120   (after build)
    python scripts/sim_ab.py mat strong|max|dome --tile +22+120
    python scripts/sim_ab.py canopy-heights --tile +22+120   (CUDA, X-Plane closed; then mat-build)
    python scripts/sim_ab.py building-heights --tile +22+120 (CUDA, X-Plane closed; then mat-build)
    python scripts/sim_ab.py mat relief-fix|canopy|canopy-half|long|longer --tile +22+120
    python scripts/sim_ab.py trees-build --tile +22+120  (after build)
    python scripts/sim_ab.py trees off|on      --tile +22+120
    python scripts/sim_ab.py tint-build [off|on] --tile +22+120   (after build)
    python scripts/sim_ab.py tint off|on       --tile +22+120
    python scripts/sim_ab.py status    --tile +22+120
    python scripts/sim_ab.py uninstall --tile +22+120      (X-Plane closed)

build copies the tile and its vegetation-overlay DSF + grass assets to --out
(the real tile and veg pack are only read):

  zOrtho4XP_<tile>_AB         ortho tile copy; ZL17+ land textures also get a
                              1024 px material map <stem>_mat1k.dds
  yOrtho4XP_Veg_Overlays_AB   the tile's veg DSF and grass/<tile>/

and keeps the variants under _ab/<tile>/: the .ter sets (ZL17+ maps at 2048 =
as built, at 1024, or no maps at all) and the veg DSF with and without the
grass polygons.  A swap only copies those files into the _AB packs.

install links both packs into Custom Scenery, puts each above its real pack
in scenery_packs.ini and disables the real veg pack (trees would be drawn
twice otherwise; tiles other than <tile> have no trees while it is
installed).  uninstall undoes exactly that.  scenery_packs.ini is backed up
to _ab/ first.  Several tiles can be installed at once: they share the _AB
veg pack and forest library, which the last uninstall takes down.

The grass swap expects the grass inside the veg DSF, i.e. a tile built before
the grass got its own pack (yOrtho4XP_Grass_Overlays, 890d523b).

trees-build regenerates the trees of the tile's ZL18/19 textures (the
airport zones) with THIS checkout's vegetation overlay, twice: 'off' = the
canopy-height model decides trees on the whole texture and single trees and
forests are X-Plane's full-height climate forests, 'on' = the canopy model
decides trees only inside airport boundaries, sized single-species trees and
shorter forest copies from canopy height, and trees under runway approaches
capped by the approach surface (veg_tree_sizes / veg_forest_heights /
veg_approach_heights).  Each result
replaces the copied veg DSF's polygons centred inside those textures; the
rest of the tile, grass included, stays as built.  The shorter forests come
from an O4XP_Forest_Heights library written under --out; install links it
into Custom Scenery too (uninstall removes it).  A later 'grass' swap
replaces the trees variant and the other way round.

mat-build adds three test sets of material maps for every land texture:
'strong' (gloss x3, F0 x2), 'max' (gloss 1, F0 at least 0.04) and 'dome'
(as-built gloss with a grid of 80 m domes as the relief, R = +east,
G = +north, to find the sign of X-Plane's terrain tangent frame).  'mat 2048'
goes back to the maps as built.  'relief-fix' is the as-built relief with its
north-south sign fixed; after canopy-heights, 'canopy' and 'canopy-half' add
the slopes of the canopy-height model's surface on ZL18/19 textures.

tint-build regenerates the whole tile's trees with THIS checkout's vegetation
overlay, twice: 'off' = X-Plane's one leaf colour (veg_forest_tint off), 'on'
= each forest in the leaf colour nearest the canopy on the ortho, summer
forests without the autumn blend (O4_Forest_Tint).  Everything else is the
installed settings, so the two differ only in leaf colour.  The tinted
copies and their atlases go into the same O4XP_Forest_Heights library under
--out.  A tint swap replaces the whole veg DSF (grass inside an old tile's
veg DSF is not rebuilt).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402

import O4_Terrain_Materials as TMAT  # noqa: E402

DEFAULT_CS = r"D:\SteamLibrary\steamapps\common\X-Plane 12\Custom Scenery"
DEFAULT_OUT = r"G:\Ortho4XP_Review\sim_ab"   # E: retired 2026-11-08
CUDA_PYTHON = r"G:\Dev\Ortho4XP\.venv\Scripts\python.exe"
INSTALLED_CFG = r"G:\Dev\Ortho4XP\_internal\Ortho4XP_Data\Ortho4XP.cfg"
# The live app's canopy model (read only); leaf-type downloads go to the review cache.
CANOPY_MODEL_DIR = r"H:\XP12 Addons\Applications\Ortho4XP 1.4\_internal\Ortho4XP_Data\SFR_cache\_canopy_height"
LEAF_TYPE_CACHE = r"G:\Ortho4XP_Review\_cache\w0\_leaf_type"
FOREST_PACK = "O4XP_Forest_Heights"
# A real O4XP_Forest_Heights written by tile builds is moved here while the
# A/B is installed (the real veg pack, its only user, is disabled meanwhile).
FOREST_HIDDEN = FOREST_PACK + ".hidden_by_sim_ab"
VEG_PACK = "yOrtho4XP_Veg_Overlays"
NVCOMPRESS = os.path.join(ROOT, "Utils", "win", "nvcompress", "nvcompress.exe")
DSFTOOL = os.path.join(ROOT, "Utils", "win", "DSFTool.exe")
MAT_1K = "_mat1k.dds"


# ── names ────────────────────────────────────────────────────────────────────
class Names:
    def __init__(self, args):
        self.tile = args.tile
        lat, lon = int(args.tile[:3]), int(args.tile[3:])
        self.dsf_dir = f"{lat - lat % 10:+03d}{lon - lon % 10:+04d}"
        self.cs = args.custom_scenery
        self.out = args.out
        self.real_tile_pack = f"zOrtho4XP_{args.tile}"
        self.tile_pack = f"zOrtho4XP_{args.tile}_AB"
        self.veg_pack = f"{VEG_PACK}_AB"
        self.tile_dir = os.path.join(self.out, self.tile_pack)
        self.veg_dir = os.path.join(self.out, self.veg_pack)
        self.ab = os.path.join(self.out, "_ab", args.tile)
        self.state_file = os.path.join(self.ab, "state.json")
        self.forest_dir = os.path.join(self.out, FOREST_PACK)

    def real(self, pack):
        return os.path.realpath(os.path.join(self.cs, pack))

    def veg_dsf(self, base):
        return os.path.join(base, "Earth nav data", self.dsf_dir, f"{self.tile}.dsf")

    def ini_line(self, pack, enabled=True):
        return f"SCENERY_PACK{'' if enabled else '_DISABLED'} Custom Scenery/{pack}/"


def load_state(n):
    try:
        with open(n.state_file) as f:
            return json.load(f)
    except OSError:
        return {}


def save_state(n, state):
    with open(n.state_file, "w") as f:
        json.dump(state, f, indent=1)


def xplane_running():
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq X-Plane.exe"],
                         capture_output=True, text=True).stdout
    return "X-Plane.exe" in out


# ── build ────────────────────────────────────────────────────────────────────
def copy_tree(src, dst):
    """Copy what is missing or a different size (re-runs resume)."""
    n = 0
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for f in files:
            s, d = os.path.join(root, f), os.path.join(dst, rel, f)
            if not os.path.isfile(d) or os.path.getsize(d) != os.path.getsize(s):
                shutil.copy2(s, d)
                n += 1
    return n


def encode_bc3_cpu(rgba, out_dds, relief):
    """TMAT.encode_dds with -nocuda (stays off the GPU a build may be using)."""
    tga, tmp = out_dds + ".mat.tga", out_dds + ".mat.tmp"
    TMAT.save_tga(rgba, tga)
    try:
        rc = subprocess.call([NVCOMPRESS, "-nocuda", "-bc3", "-fast", "-no-mip-gamma-correct", tga, tmp],
                             stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if rc != 0 or not os.path.isfile(tmp):
            return False
        TMAT._stamp(tmp, relief)
        os.replace(tmp, out_dds)
        return True
    finally:
        for f in (tga, tmp):
            if os.path.exists(f):
                os.remove(f)


def build_materials(n):
    """.ter sets 2048 (as built) / 1024 / off; 1024 maps for ZL17+ textures."""
    ter_dir, tex_dir = os.path.join(n.tile_dir, "terrain"), os.path.join(n.tile_dir, "textures")
    sets = {m: os.path.join(n.ab, f"ter_{m}") for m in ("2048", "1024", "off")}
    for d in sets.values():
        os.makedirs(d, exist_ok=True)
    jobs = []
    for name in sorted(os.listdir(ter_dir)):
        if not name.lower().endswith(".ter"):
            continue
        with open(os.path.join(ter_dir, name), newline="") as f:
            text = f.read()
        lines, _ = TMAT._split_ter(text)
        base = TMAT.ter_base_texture(lines)
        if not (TMAT.is_land_ter(name, lines) and base and base.startswith("../textures/")):
            continue
        tex = base[len("../textures/"):]
        ours = TMAT.ter_lines(tex)[0]
        if ours not in (l.strip() for l in lines):
            continue
        zl = TMAT._zoomlevel(tex)
        variants = {"2048": text, "off": TMAT.strip_ter_lines(text, tex), "1024": text}
        if zl >= 17:
            stem = TMAT.map_file_name(tex)[:-len("_mat.dds")]
            variants["1024"] = text.replace(ours, f"TEXTURE_NORMAL 1 ../textures/{stem}{MAT_1K}")
            jobs.append((tex, stem))
        for m, t in variants.items():
            with open(os.path.join(sets[m], name), "w", newline="") as f:
                f.write(t)

    def one(job):
        tex, stem = job
        out = os.path.join(tex_dir, stem + MAT_1K)
        mark = TMAT.map_mark(os.path.join(tex_dir, stem + "_mat.dds"))
        relief = mark[1] / 10000 if mark else TMAT.RELIEF_DEFAULT
        if TMAT.map_mark(out) == (TMAT.MARK_VERSION, TMAT._relief_code(relief)):
            return "kept"
        rgb = TMAT.load_rgb(os.path.join(tex_dir, tex), 1024)
        return "made" if encode_bc3_cpu(TMAT.material_map(rgb, relief), out, relief) else "FAILED"

    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(one, jobs))
    print(f"  material maps 1024: {res.count('made')} made, {res.count('kept')} kept, "
          f"{res.count('FAILED')} failed, of {len(jobs)} ZL17+ textures")
    if "FAILED" in res:
        sys.exit("material map encoding failed")


# ── material test maps ───────────────────────────────────────────────────────
# Each test set points every land .ter at its own map <stem><suffix>.dds:
#   strong  gloss x3 (capped at 1) and F0 x2: is gloss visible at all in clear weather?
#   max     gloss 1 and F0 >= 0.04 everywhere: the most X-Plane can show.
#   dome    the as-built gloss/F0 with the relief replaced by a grid of domes,
#           encoded as R = +east, G = +north.  Which side of a dome faces the
#           sun settles the sign of X-Plane's terrain tangent frame (it uses
#           per-draw uniforms, not UV derivatives) before real heights go in.
MAT_TESTS = {"strong": "_matS.dds", "max": "_matM.dds", "dome": "_matD.dds", "dome-s": "_matDs.dds",
             "relief-fix": "_matF.dds", "canopy": "_matC.dds", "canopy-half": "_matH.dds",
             "long": "_matL.dds", "longer": "_matLL.dds"}
# long, longer  trees (canopy, ZL18/19) and buildings (building-heights, every ZL):
#              the 'canopy' slopes plus a broad copy of the height field (smoothed
#              over LONG_VARIANTS metres, slopes x gain).  The side away from the sun
#              gets a wide dark band that turns darker as the sun gets lower: the
#              closest a static map gets to long shadows (it cannot cast one).
LONG_VARIANTS = {"long": (10.0, 1.5), "longer": (16.0, 2.5)}
# relief-fix   the as-built luminance relief with its north-south sign fixed (MARK_VERSION 2)
# canopy       relief-fix plus the true slopes of the canopy-height model's surface
#              (ZL18/19 textures with canopy-heights run; others = relief-fix)
# canopy-half  the same with the canopy slopes halved
CANOPY_SMOOTH_M = 1.5
# 2026-11-09 in-sim (RCKH, facing west, 30 June 4 pm = sun 283 deg, 36 deg up): 'dome'
# was lit from the WNW, i.e. R = +east, G = +north read right.  'dome-s' (G = +south)
# is the deliberately wrong twin for a north-south check.
DOME_RADIUS_M, DOME_SPACING_M = 80.0, 260.0


def _map_metres_per_px(n, tex, size=1024):
    zl = TMAT._zoomlevel(tex)
    lat = int(n.tile[:3]) + 0.5
    # A texture is 16 x 16 web-mercator tiles of its ZL.
    return 40075016.686 * np.cos(np.radians(lat)) / (2 ** zl) * 16 / size


def dome_relief(size, m_per_px, g_south=False):
    """RG normal bytes of a dome grid: R = +east (columns), G = +north (up the image),
    or +south (down the image) with ``g_south``."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) * m_per_px
    cx = (xx % DOME_SPACING_M) - DOME_SPACING_M / 2
    cy = (yy % DOME_SPACING_M) - DOME_SPACING_M / 2
    d = np.hypot(cx, cy)
    inside = d < DOME_RADIUS_M
    # Spherical cap normal: (x, y, z) / r; north is up the image, so -row.
    nx = np.where(inside, cx / DOME_RADIUS_M, 0.0)
    ny = np.where(inside, (cy if g_south else -cy) / DOME_RADIUS_M, 0.0)
    nx, ny = nx * 0.8, ny * 0.8                     # keep z > 0 at the rim
    return (np.clip(np.stack([nx, ny], -1) * 127.5 + 127.5, 0, 255) + 0.5).astype(np.uint8)


def relief_heights(n, stem, size=1024):
    """Tree + building height field (m) at ``size`` px: the higher of the canopy model's
    canopy and the building-heights raster, or None when neither was computed."""
    import cv2
    layers = []
    for sub in ("canopy", "buildings"):
        p = os.path.join(n.ab, sub, stem + ".npy")
        if os.path.isfile(p):
            layers.append(cv2.resize(np.load(p).astype(np.float32), (size, size),
                                     interpolation=cv2.INTER_AREA))
    return np.maximum.reduce(layers) if layers else None


def canopy_slopes(heights, size, m_per_px, smooth_m=CANOPY_SMOOTH_M):
    """East and north slopes (dz/dx, dz/dy) of a height map at ``size`` px."""
    import cv2
    h = cv2.resize(np.asarray(heights, np.float32), (size, size), interpolation=cv2.INTER_AREA)
    h = cv2.GaussianBlur(h, (0, 0), max(0.5, smooth_m / m_per_px))
    gy, gx = np.gradient(h, m_per_px)                 # gy: per metre down the image (south)
    return gx, -gy


def with_slopes(rg, sx, sy):
    """RG normal bytes with extra surface slopes (dz/dx east, dz/dy north) added."""
    nx = (rg[..., 0].astype(np.float32) - 127.5) / 127.5
    ny = (rg[..., 1].astype(np.float32) - 127.5) / 127.5
    nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 1e-4, 1))
    px, py = -nx / nz + sx, -ny / nz + sy             # normal (nx, ny, nz) = slope (-nx/nz, -ny/nz); add
    nz = 1 / np.sqrt(1 + px * px + py * py)
    nx, ny = -px * nz, -py * nz
    return (np.clip(np.stack([nx, ny], -1) * 127.5 + 127.5, 0, 255) + 0.5).astype(np.uint8)


def build_material_tests(n):
    """ter_strong / ter_max / ter_dome sets and their maps, from the as-built .ter set."""
    ter_src = os.path.join(n.ab, "ter_2048")
    tex_dir = os.path.join(n.tile_dir, "textures")
    sets = {m: os.path.join(n.ab, f"ter_{m}") for m in MAT_TESTS}
    for d in sets.values():
        os.makedirs(d, exist_ok=True)
    jobs = []
    for name in sorted(os.listdir(ter_src)):
        with open(os.path.join(ter_src, name), newline="") as f:
            text = f.read()
        lines, _ = TMAT._split_ter(text)
        base = TMAT.ter_base_texture(lines)
        if not (TMAT.is_land_ter(name, lines) and base and base.startswith("../textures/")):
            continue
        tex = base[len("../textures/"):]
        ours = TMAT.ter_lines(tex)[0]
        stem = TMAT.map_file_name(tex)[:-len("_mat.dds")]
        for m, suffix in MAT_TESTS.items():
            t = text.replace(ours, f"TEXTURE_NORMAL 1 ../textures/{stem}{suffix}")
            if ours not in (l.strip() for l in lines):   # no map as built: add one for the test
                t = TMAT.add_ter_lines(text, tex).replace(ours, f"TEXTURE_NORMAL 1 ../textures/{stem}{suffix}")
            with open(os.path.join(sets[m], name), "w", newline="") as f:
                f.write(t)
        jobs.append((tex, stem))

    def one(job):
        tex, stem = job
        built = TMAT.map_mark(os.path.join(tex_dir, stem + "_mat.dds"))
        relief = built[1] / 10000 if built else TMAT.RELIEF_DEFAULT
        rgb = TMAT.load_rgb(os.path.join(tex_dir, tex), 1024)
        base = TMAT.material_map(rgb, relief)
        f0, gloss = base[..., 2].astype(np.float32), base[..., 3].astype(np.float32)
        variants = {
            "strong": (base[..., :2], np.minimum(f0 * 2, 255), np.minimum(gloss * 3, 255)),
            "max": (base[..., :2], np.maximum(f0, 0.04 * 255), np.full_like(gloss, 255)),
            "dome": (dome_relief(1024, _map_metres_per_px(n, tex)), f0, gloss),
            "dome-s": (dome_relief(1024, _map_metres_per_px(n, tex), g_south=True), f0, gloss),
            "relief-fix": (base[..., :2], f0, gloss),
        }
        canopy = os.path.join(n.ab, "canopy", stem + ".npy")
        if os.path.isfile(canopy):
            sx, sy = canopy_slopes(np.load(canopy), 1024, _map_metres_per_px(n, tex))
            variants["canopy"] = (with_slopes(base[..., :2], sx, sy), f0, gloss)
            variants["canopy-half"] = (with_slopes(base[..., :2], sx * 0.5, sy * 0.5), f0, gloss)
        else:
            variants["canopy"] = variants["canopy-half"] = variants["relief-fix"]
        heights = relief_heights(n, stem)
        mpp = _map_metres_per_px(n, tex)
        if heights is not None:
            fx, fy = canopy_slopes(heights, 1024, mpp)
            for m, (smooth_m, gain) in LONG_VARIANTS.items():
                bx, by = canopy_slopes(heights, 1024, mpp, smooth_m)
                variants[m] = (with_slopes(base[..., :2], fx + bx * gain, fy + by * gain), f0, gloss)
        else:
            for m in LONG_VARIANTS:
                variants[m] = variants["relief-fix"]
        ok = True
        for m, (rg, b, a) in variants.items():
            out = os.path.join(tex_dir, stem + MAT_TESTS[m])
            rgba = np.dstack([rg, (b + 0.5).astype(np.uint8), (a + 0.5).astype(np.uint8)])
            ok &= encode_bc3_cpu(np.ascontiguousarray(rgba), out, relief)
        return ok

    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(one, jobs))
    print(f"  material test maps: {res.count(True)} textures x {len(MAT_TESTS)} made, "
          f"{res.count(False)} failed")
    if not all(res):
        sys.exit("material test map encoding failed")


def cmd_canopy_heights(n, args):
    """Canopy height (m) of every ZL18/19 texture into _ab/<tile>/canopy/<stem>.npy (CUDA)."""
    if not load_state(n):
        sys.exit("build first")
    if xplane_running():
        sys.exit("quit X-Plane first (the canopy model needs ~4 GB of GPU memory)")
    out = os.path.join(n.ab, "canopy")
    os.makedirs(out, exist_ok=True)
    code = "\n".join([
        "import os, re, sys, time",
        f"sys.path.insert(0, {os.path.join(ROOT, 'src')!r})",
        "import numpy as np",
        "from PIL import Image",
        "import O4_Canopy_Height as C",
        f"tex, out = {os.path.join(n.tile_dir, 'textures')!r}, {out!r}",
        f"model = C.CanopyModel.load({CANOPY_MODEL_DIR!r})",
        "if model is None: sys.exit('canopy model unavailable')",
        "names = sorted(f for f in os.listdir(tex) if re.match('^[0-9]+_[0-9]+_[A-Za-z]+1[89][.]dds$', f))",
        "t0 = time.time()",
        "for i, f in enumerate(names):",
        "    dst = os.path.join(out, f[:-4] + '.npy')",
        "    if os.path.isfile(dst): continue",
        "    rgb = np.asarray(Image.open(os.path.join(tex, f)).convert('RGB'))",
        "    np.save(dst, model.maps(rgb, int(f[-6:-4]))[0])",
        "    print(f'  [{i + 1}/{len(names)}] {f}', flush=True)",
        "print(f'canopy heights: {len(names)} textures in {time.time() - t0:.0f}s')",
    ]) + "\n"
    rc = subprocess.call([CUDA_PYTHON, "-c", code], cwd=os.path.join(ROOT, "src"))
    if rc != 0:
        sys.exit("canopy heights failed")


def cmd_building_heights(n, args):
    """Building heights (YOLO + HeightNet, live cfg models) into _ab/<tile>/buildings/ (CUDA)."""
    if not load_state(n):
        sys.exit("build first")
    if xplane_running():
        sys.exit("quit X-Plane first (YOLO + HeightNet need the GPU)")
    cfg = {}
    with open(INSTALLED_CFG, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    yolo, height = cfg.get("bld_yolo_checkpoint"), cfg.get("bld_height_checkpoint")
    if not (yolo and height and os.path.isfile(yolo) and os.path.isfile(height)):
        sys.exit(f"building models not found in {INSTALLED_CFG}: {yolo!r}, {height!r}")
    script = os.path.join(ROOT, "experiments", "xp12_realism_20261108", "building_heights.py")
    rc = subprocess.call([CUDA_PYTHON, script, os.path.join(n.tile_dir, "textures"),
                          os.path.join(n.ab, "buildings"), yolo, height,
                          cfg.get("bld_yolo_conf", "0.05"), cfg.get("bld_yolo_iou", "0.5"),
                          cfg.get("bld_yolo_stride", "512"), "512", "8", "asia"],
                         cwd=os.path.join(ROOT, "src"))
    if rc != 0:
        sys.exit("building heights failed")


def cmd_mat_build(n, args):
    if not load_state(n):
        sys.exit("build first")
    t0 = time.time()
    build_material_tests(n)
    state = load_state(n)
    state["mat_tests_built"] = time.strftime("%Y-%m-%d %H:%M")
    save_state(n, state)
    print(f"material tests built in {time.time() - t0:.0f}s; swap with: mat {'|'.join(MAT_TESTS)}|2048|off")


def build_veg(n, real_veg):
    os.makedirs(os.path.dirname(n.veg_dsf(n.veg_dir)), exist_ok=True)
    copy_tree(os.path.join(real_veg, "grass", n.tile), os.path.join(n.veg_dir, "grass", n.tile))
    on, off = os.path.join(n.ab, "veg_grass_on.dsf"), os.path.join(n.ab, "veg_grass_off.dsf")
    shutil.copy2(n.veg_dsf(real_veg), on)
    txt = os.path.join(n.ab, "veg.txt")
    subprocess.run([DSFTOOL, "--dsf2text", on, txt], check=True, stdout=subprocess.DEVNULL)
    defs, out, skip, dropped = 0, [], False, 0
    grass = set()
    with open(txt) as f:
        for line in f:
            if line.startswith("POLYGON_DEF"):
                if line.split(None, 1)[1].startswith("grass/"):
                    grass.add(defs)
                defs += 1
            elif line.startswith("BEGIN_POLYGON"):
                skip = int(line.split()[1]) in grass
                dropped += skip
            if not skip:
                out.append(line)
            if line.startswith("END_POLYGON"):
                skip = False
    with open(txt, "w") as f:
        f.writelines(out)
    subprocess.run([DSFTOOL, "--text2dsf", txt, off], check=True, stdout=subprocess.DEVNULL)
    os.remove(txt)
    print(f"  veg DSF: {len(grass)} grass forests, {dropped} grass polygons dropped for 'off'")


def cmd_build(n, args):
    real_tile, real_veg = n.real(n.real_tile_pack), n.real(VEG_PACK)
    for p in (real_tile, n.veg_dsf(real_veg), NVCOMPRESS, DSFTOOL):
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")
    os.makedirs(n.ab, exist_ok=True)
    t0 = time.time()
    print(f"copying {real_tile} -> {n.tile_dir}")
    for sub in ("Earth nav data", "terrain", "textures"):
        print(f"  {sub}: {copy_tree(os.path.join(real_tile, sub), os.path.join(n.tile_dir, sub))} files copied")
    terrain_2048 = os.path.join(n.ab, "ter_2048")
    if os.path.isdir(terrain_2048):              # re-run: start from the as-built .ter set
        copy_tree(terrain_2048, os.path.join(n.tile_dir, "terrain"))
    build_materials(n)
    build_veg(n, real_veg)
    state = load_state(n)
    state.update(built=time.strftime("%Y-%m-%d %H:%M"), real_tile=real_tile, real_veg=real_veg)
    save_state(n, state)
    set_grass(n, "on")
    set_mat(n, "2048")
    print(f"built in {time.time() - t0:.0f}s")


# ── trees ────────────────────────────────────────────────────────────────────
TREE_VARIANTS = {"off": {"O4_VEG_TREE_SIZES": "0", "O4_VEG_FOREST_HEIGHTS": "0",
                         "O4_VEG_APPROACH_HEIGHTS": "0", "O4_VEG_CANOPY_AIRPORTS_ONLY": "0"},
                 "on": {"O4_VEG_TREE_SIZES": "1", "O4_VEG_FOREST_HEIGHTS": "1",
                        "O4_VEG_APPROACH_HEIGHTS": "1", "O4_VEG_CANOPY_AIRPORTS_ONLY": "1"}}


def installed_veg_settings():
    """The installed config's veg_* values as run() keywords, the way a tile build sets them."""
    vals = {}
    with open(INSTALLED_CFG, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "=" in line and line.startswith("veg_"):
                k, v = line.strip().split("=", 1)
                vals[k] = v

    def num(k, d):
        return float(vals.get(k, d))

    def yes(k, d):
        return vals.get(k, str(d)) == "True"
    density = num("veg_density", -1.0)
    return dict(
        close_m=num("veg_close_m", 10.0), open_m=num("veg_open_m", 3.0),
        min_area_m2=num("veg_min_area_m2", 50.0), simplify_m=num("veg_simplify_m", 3.0),
        density_override=density if density >= 0 else None,
        res_m=num("veg_res_m", 0.0) or None, excl_buffer_m=num("veg_excl_buffer_m", 5.0),
        use_simheaven=yes("veg_use_simheaven", True),
        avoid_simheaven_buildings=yes("veg_avoid_simheaven_buildings", True),
        simheaven_building_buffer_m=num("veg_simheaven_building_buffer_m", 10.0),
        avoid_simheaven_forests=yes("veg_avoid_simheaven_forests", True),
        simheaven_forest_buffer_m=num("veg_simheaven_forest_buffer_m", 0.0),
        use_simheaven_asset_proximity=yes("veg_use_simheaven_asset_proximity", True),
        exclude_lower_forests=yes("veg_exclude_lower_forests", False),
        avoid_gfv2=yes("veg_avoid_gfv2", False), gfv2_buffer_m=num("veg_gfv2_buffer_m", 0.0),
        use_gfv2_asset_proximity=yes("veg_use_gfv2_asset_proximity", False),
        use_leaf_type_maps=yes("veg_use_leaf_type_maps", True),
        use_xp_forests=yes("veg_use_xp_forests", True),
        bld_excl_m=0.0 if yes("veg_disable_cache", False) else 10.0,
    )


def zone_textures(n):
    """(name, (lat_n, lat_s, lon_w, lon_e)) of the copy's ZL18+ textures."""
    import re
    import O4_Vegetation_Overlay as VEG
    out = []
    for name in sorted(os.listdir(os.path.join(n.tile_dir, "textures"))):
        m = re.match(r"^(\d+)_(\d+)_([A-Za-z][A-Za-z0-9_]*?)(\d{2})\.dds$", name)
        if m and int(m.group(4)) >= 18:
            out.append((name, VEG.dds_bounds(int(m.group(1)), int(m.group(2)), int(m.group(4)))))
    return out


def read_dsf_text(path):
    """(PROPERTY lines, polygon defs, [(def index, block lines, centroid lon/lat)])."""
    header, defs, polys, block = [], [], [], None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("POLYGON_DEF "):
                defs.append(line.split(None, 1)[1].strip())
            elif line.startswith("BEGIN_POLYGON"):
                block = [line]
            elif block is not None:
                block.append(line)
                if line.startswith("END_POLYGON"):
                    pts = [tuple(map(float, ln.split()[1:3])) for ln in block if ln.startswith("POLYGON_POINT")]
                    c = ((sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
                         if pts else (0.0, 0.0))
                    polys.append((int(block[0].split()[1]), block, c))
                    block = None
            elif line.startswith("PROPERTY"):
                header.append(line)
    return header, defs, polys


def merge_tree_dsf(base_dsf, new_dsf, out_dsf, boxes):
    """base_dsf with its tree polygons centred inside ``boxes`` replaced by new_dsf's."""
    def inside(c):
        return any(s <= c[1] <= nn and w <= c[0] <= e for nn, s, w, e in boxes)
    tmp = []
    try:
        texts = []
        for dsf in (base_dsf, new_dsf):
            txt = f"{out_dsf}.{len(texts)}.txt"
            tmp.append(txt)
            subprocess.run([DSFTOOL, "--dsf2text", dsf, txt], check=True, stdout=subprocess.DEVNULL)
            texts.append(read_dsf_text(txt))
        (header, bdefs, bpolys), (_h, ndefs, npolys) = texts
        defs, index, keep, dropped = [], {}, [], 0

        def remap(defs_list, i):
            path = defs_list[i]
            if path not in index:
                index[path] = len(defs)
                defs.append(path)
            return index[path]
        for i, block, c in bpolys:
            if bdefs[i].startswith("grass/") or not inside(c):
                keep.append((remap(bdefs, i), block))
            else:
                dropped += 1
        for i, block, _c in npolys:
            keep.append((remap(ndefs, i), block))
        txt = out_dsf + ".txt"
        tmp.append(txt)
        with open(txt, "w", encoding="utf-8") as f:
            f.writelines(header)
            f.write("\n")
            for path in defs:
                f.write(f"POLYGON_DEF {path}\n")
            f.write("\n")
            for i, block in keep:
                parts = block[0].split()
                parts[1] = str(i)
                f.write(" ".join(parts) + "\n")
                f.writelines(block[1:])
        subprocess.run([DSFTOOL, "--text2dsf", txt, out_dsf], check=True, stdout=subprocess.DEVNULL)
        return dropped, len(npolys)
    finally:
        for t in tmp:
            if os.path.exists(t):
                os.remove(t)


def cmd_trees_build(n, args):
    if not load_state(n):
        sys.exit("build first")
    zones = zone_textures(n)
    if not zones:
        sys.exit("the tile copy has no ZL18/19 textures")
    lat, lon = int(n.tile[:3]), int(n.tile[3:])
    tex = os.path.join(n.ab, "trees_textures")
    os.makedirs(tex, exist_ok=True)
    for name, _b in zones:                           # the zone textures only, as hard links
        dst = os.path.join(tex, name)
        if not os.path.exists(dst):
            try:
                os.link(os.path.join(n.tile_dir, "textures", name), dst)
            except OSError:
                shutil.copy2(os.path.join(n.tile_dir, "textures", name), dst)
    settings = installed_veg_settings()
    base = os.path.join(n.ab, "veg_grass_on.dsf")
    t0 = time.time()
    for variant, env in TREE_VARIANTS.items():
        out = os.path.join(n.ab, f"trees_{variant}_zones.dsf")
        code = (
            "import sys\n"
            f"sys.path.insert(0, {os.path.join(ROOT, 'src')!r})\n"
            "import O4_Landcover_Inference as SEG\n"
            f"SEG._dsftool = {DSFTOOL!r}\n"
            "import O4_Vegetation_Overlay as V\n"
            f"V.run(tex_dir={tex!r}, lat={lat}, lon={lon}, out_dsf={out!r},\n"
            f"      cache_dir={os.path.join(n.ab, 'trees_cache')!r}, make_viz=False,\n"
            f"      dsftool_path={DSFTOOL!r}, custom_scenery_dir={n.cs!r},\n"
            f"      leaf_type_cache_dir={LEAF_TYPE_CACHE!r}, grass_settings=None,\n"
            f"      **{settings!r})\n"
        )
        run_env = dict(os.environ, O4_CANOPY_MODEL_DIR=CANOPY_MODEL_DIR,
                       O4_FOREST_HEIGHTS_PACK=n.forest_dir, **env)
        print(f"trees {variant}: building {len(zones)} ZL18/19 textures ...", flush=True)
        rc = subprocess.call([CUDA_PYTHON, "-c", code], env=run_env, cwd=os.path.join(ROOT, "src"))
        if rc != 0 or not os.path.isfile(out):
            sys.exit(f"vegetation overlay failed for trees {variant}")
        dropped, added = merge_tree_dsf(base, out, os.path.join(n.ab, f"veg_trees_{variant}.dsf"),
                                        [b for _n, b in zones])
        print(f"  trees {variant}: {dropped} built polygons in the zones replaced by {added}")
    s = load_state(n)
    s["trees_built"] = time.strftime("%Y-%m-%d %H:%M")
    save_state(n, s)
    set_trees(n, "off")
    print(f"trees built in {time.time() - t0:.0f}s; swapped to 'trees off'")


TINT_VARIANTS = {"off": {"O4_VEG_FOREST_TINT": "0"}, "on": {"O4_VEG_FOREST_TINT": "1"}}


def installed_veg_env():
    """The O4_VEG_* switches a tile build sets from the installed config (tree rules,
    canopy, heights, density ...), so a variant differs from a real build only in
    what it swaps."""
    import O4_Cfg_Vars as CFGVARS
    import O4_Overlay_Pipeline as PIPELINE
    with open(INSTALLED_CFG, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "=" not in line:
                continue
            k, v = line.strip().split("=", 1)
            if not hasattr(PIPELINE, k) or k not in CFGVARS.cfg_vars:
                continue
            typ = CFGVARS.cfg_vars[k]["type"]
            try:
                val = (v == "True") if typ is bool else typ(v) if typ in (int, float) else v
            except ValueError:
                continue
            setattr(PIPELINE, k, val)
    return {k: v for k, v in PIPELINE._veg_tree_env().items() if k.startswith(("O4_VEG_", "O4_CANOPY_"))}


def cmd_tint_build(n, args):
    if not load_state(n):
        sys.exit("build first")
    lat, lon = int(n.tile[:3]), int(n.tile[3:])
    tex = os.path.join(n.tile_dir, "textures")
    settings = installed_veg_settings()
    t0 = time.time()
    if args.mode and args.mode not in TINT_VARIANTS:
        sys.exit("tint-build takes no mode, or one of: off on (rebuild only that variant)")
    for variant, env in TINT_VARIANTS.items():
        if args.mode and variant != args.mode:
            continue
        out = os.path.join(n.ab, f"veg_tint_{variant}.dsf")
        code = (
            "import sys\n"
            f"sys.path.insert(0, {os.path.join(ROOT, 'src')!r})\n"
            "import O4_Landcover_Inference as SEG\n"
            f"SEG._dsftool = {DSFTOOL!r}\n"
            "import O4_Vegetation_Overlay as V\n"
            f"V.run(tex_dir={tex!r}, lat={lat}, lon={lon}, out_dsf={out!r},\n"
            f"      cache_dir={os.path.join(n.ab, 'tint_cache')!r}, make_viz=False,\n"
            f"      dsftool_path={DSFTOOL!r}, custom_scenery_dir={n.cs!r},\n"
            f"      leaf_type_cache_dir={LEAF_TYPE_CACHE!r}, grass_settings=None,\n"
            f"      **{settings!r})\n"
        )
        run_env = dict(os.environ, **installed_veg_env())
        run_env.update(O4_CANOPY_MODEL_DIR=CANOPY_MODEL_DIR, O4_FOREST_HEIGHTS_PACK=n.forest_dir, **env)
        print(f"tint {variant}: building the tile's trees ...", flush=True)
        rc = subprocess.call([CUDA_PYTHON, "-c", code], env=run_env, cwd=os.path.join(ROOT, "src"))
        if rc != 0 or not os.path.isfile(out):
            sys.exit(f"vegetation overlay failed for tint {variant}")
    s = load_state(n)
    s["tint_built"] = time.strftime("%Y-%m-%d %H:%M")
    save_state(n, s)
    set_tint(n, "off")
    print(f"tint variants built in {time.time() - t0:.0f}s; swapped to 'tint off'")


def set_tint(n, mode):
    shutil.copyfile(os.path.join(n.ab, f"veg_tint_{mode}.dsf"), n.veg_dsf(n.veg_dir))
    state = load_state(n)
    state["tint"] = mode
    state.pop("trees", None)                         # the tint variants carry the whole tile's trees
    state.pop("grass", None)
    save_state(n, state)


def set_trees(n, mode):
    shutil.copyfile(os.path.join(n.ab, f"veg_trees_{mode}.dsf"), n.veg_dsf(n.veg_dir))
    state = load_state(n)
    state["trees"] = mode
    state["grass"] = "on"                            # the trees variants keep the built grass
    state.pop("tint", None)
    save_state(n, state)


# ── swaps ────────────────────────────────────────────────────────────────────
def set_grass(n, mode):
    shutil.copyfile(os.path.join(n.ab, f"veg_grass_{mode}.dsf"), n.veg_dsf(n.veg_dir))
    state = load_state(n)
    state["grass"] = mode
    state.pop("trees", None)                         # the grass variants carry the trees as built
    state.pop("tint", None)
    save_state(n, state)


def set_mat(n, mode):
    src = os.path.join(n.ab, f"ter_{mode}")
    for name in os.listdir(src):
        shutil.copyfile(os.path.join(src, name), os.path.join(n.tile_dir, "terrain", name))
    state = load_state(n)
    state["mat"] = mode
    save_state(n, state)


def gpu_memory():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        used, total = (int(x) for x in out.split(",")[:2])
        return f"{used / 1024:.1f} / {total / 1024:.1f} GB"
    except (OSError, ValueError):
        return "n/a"


def cmd_status(n, args):
    s = load_state(n)
    if not s:
        sys.exit("not built")
    print(f"tile {n.tile}: grass {s.get('grass')}, material maps {s.get('mat')}, "
          f"trees {s.get('trees', 'as built')}, tint {s.get('tint', 'as built')}"
          f" ({'installed' if s.get('installed') else 'NOT installed'}); GPU memory used {gpu_memory()}")


# ── install ──────────────────────────────────────────────────────────────────
def read_ini(n):
    path = os.path.join(n.cs, "scenery_packs.ini")
    with open(path, newline="") as f:
        raw = f.read()
    n.ini_eol = "\r\n" if "\r\n" in raw else "\n"
    return path, raw.splitlines()


def write_ini(n, path, lines):
    shutil.copy2(path, os.path.join(n.ab, f"scenery_packs.ini.{time.strftime('%Y%m%d-%H%M%S')}.bak"))
    with open(path, "w", newline="") as f:
        f.write(n.ini_eol.join(lines) + n.ini_eol)


_SHARED_FLAGS = ("veg_disabled_by_us", "forest_real_hidden", "forest_line_existed", "forest_pack_installed")


def other_installed(n):
    """Tiles other than n.tile installed from the same --out: they share the _AB veg
    pack and the forest library, so only the last uninstall takes those down."""
    root = os.path.join(n.out, "_ab")
    found = []
    for tile in sorted(os.listdir(root)) if os.path.isdir(root) else ():
        if tile == n.tile:
            continue
        try:
            with open(os.path.join(root, tile, "state.json")) as f:
                if json.load(f).get("installed"):
                    found.append(tile)
        except (OSError, ValueError):
            pass
    return found


def cmd_install(n, args):
    if xplane_running():
        sys.exit("quit X-Plane first (it reads scenery_packs.ini at start)")
    s = load_state(n)
    if not s:
        sys.exit("build first")
    forest = os.path.isfile(os.path.join(n.forest_dir, "library.txt"))
    others = other_installed(n)   # their veg pack line is already above the disabled real one
    path, lines = read_ini(n)
    if forest:
        real = os.path.join(n.cs, FOREST_PACK)
        if os.path.isdir(real) and os.path.realpath(real) == os.path.abspath(real):
            os.rename(real, os.path.join(n.cs, FOREST_HIDDEN))
            s["forest_real_hidden"] = True
            s["forest_line_existed"] = n.ini_line(FOREST_PACK) in [ln.strip() for ln in lines]
            save_state(n, s)
            print(f"moved the real {FOREST_PACK} aside to {FOREST_HIDDEN} until uninstall")
    links = [(n.tile_pack, n.tile_dir), (n.veg_pack, n.veg_dir)]
    if forest:
        links.append((FOREST_PACK, n.forest_dir))
    for pack, target in links:
        link = os.path.join(n.cs, pack)
        if not os.path.exists(link):
            subprocess.run(["cmd", "/c", "mklink", "/J", link, target], check=True, stdout=subprocess.DEVNULL)
    out = []
    for line in lines:
        if line.strip() in (n.ini_line(n.tile_pack), n.ini_line(n.veg_pack), n.ini_line(FOREST_PACK)):
            continue                                   # re-install: placed again below
        if line.strip() == n.ini_line(n.real_tile_pack):
            out.append(n.ini_line(n.tile_pack))
        elif line.strip() == n.ini_line(VEG_PACK) or (
                others and line.strip() == n.ini_line(VEG_PACK, enabled=False)):
            if forest:
                out.append(n.ini_line(FOREST_PACK))    # a library: its place does not matter
            out.append(n.ini_line(n.veg_pack))
            if line.strip() == n.ini_line(VEG_PACK):
                s["veg_disabled_by_us"] = True
            line = n.ini_line(VEG_PACK, enabled=False)
        out.append(line)
    if n.ini_line(n.tile_pack) not in out or n.ini_line(n.veg_pack) not in out:
        sys.exit(f"{n.real_tile_pack} or {VEG_PACK} not found enabled in scenery_packs.ini; nothing changed")
    write_ini(n, path, out)
    s["installed"] = True
    s["forest_pack_installed"] = forest
    save_state(n, s)
    print(f"installed: {n.tile_pack} above {n.real_tile_pack}, {n.veg_pack} above {VEG_PACK} (disabled)"
          + (f", {FOREST_PACK} library" if forest else ""))


def cmd_uninstall(n, args):
    if xplane_running():
        sys.exit("quit X-Plane first")
    s = load_state(n)
    others = other_installed(n)
    if others:
        # The veg pack, forest library and disabled real veg pack stay for the other
        # tile(s); hand over what has to be undone at the last uninstall.
        path, lines = read_ini(n)
        out = [ln for ln in lines if ln.strip() != n.ini_line(n.tile_pack)]
        if out != lines:
            write_ini(n, path, out)
        link = os.path.join(n.cs, n.tile_pack)
        if os.path.isdir(link) and os.path.realpath(link) != os.path.abspath(link):
            os.rmdir(link)
        heir = Names(argparse.Namespace(tile=others[0], custom_scenery=n.cs, out=n.out))
        hs = load_state(heir)
        for k in _SHARED_FLAGS:
            hs[k] = bool(hs.get(k) or s.get(k))
        save_state(heir, hs)
        s.update(installed=False, **{k: False for k in _SHARED_FLAGS})
        save_state(n, s)
        print(f"uninstalled {n.tile_pack}; the shared veg pack and forest library stay for {', '.join(others)}")
        return
    path, lines = read_ini(n)
    out = []
    for line in lines:
        if line.strip() in (n.ini_line(n.tile_pack), n.ini_line(n.veg_pack), n.ini_line(FOREST_HIDDEN)):
            continue                                   # X-Plane lists the moved-aside folder as a pack
        if (line.strip() == n.ini_line(FOREST_PACK) and s.get("forest_pack_installed")
                and not s.get("forest_line_existed")):
            continue
        if line.strip() == n.ini_line(VEG_PACK, enabled=False) and s.get("veg_disabled_by_us"):
            line = n.ini_line(VEG_PACK)
        out.append(line)
    if out != lines:
        write_ini(n, path, out)
    packs = (n.tile_pack, n.veg_pack) + ((FOREST_PACK,) if s.get("forest_pack_installed") else ())
    for pack in packs:
        link = os.path.join(n.cs, pack)
        if os.path.isdir(link) and os.path.realpath(link) != os.path.abspath(link):
            os.rmdir(link)                              # removes the junction only
    hidden = os.path.join(n.cs, FOREST_HIDDEN)
    if s.get("forest_real_hidden") and os.path.isdir(hidden) and not os.path.exists(
            os.path.join(n.cs, FOREST_PACK)):
        os.rename(hidden, os.path.join(n.cs, FOREST_PACK))
        print(f"put the real {FOREST_PACK} back")
    s.update(installed=False, veg_disabled_by_us=False, forest_pack_installed=False,
             forest_real_hidden=False, forest_line_existed=False)
    save_state(n, s)
    print(f"uninstalled; {VEG_PACK} re-enabled. Test data stays in {n.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["build", "install", "uninstall", "status", "grass", "mat", "mat-build",
                                        "canopy-heights", "building-heights",
                                        "trees-build", "trees", "tint-build", "tint"])
    ap.add_argument("mode", nargs="?")
    ap.add_argument("--tile", required=True, help="e.g. +22+120")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--custom-scenery", default=DEFAULT_CS)
    args = ap.parse_args()
    n = Names(args)
    if args.command in ("grass", "mat", "trees", "tint"):
        modes = {"grass": ("on", "off"), "mat": ("2048", "1024", "off") + tuple(MAT_TESTS), "trees": ("off", "on"),
                 "tint": ("off", "on")}[args.command]
        if args.mode not in modes:
            sys.exit(f"{args.command} needs one of: {' '.join(modes)}")
        if not load_state(n):
            sys.exit("build first")
        if args.command == "trees" and not os.path.isfile(os.path.join(n.ab, "veg_trees_on.dsf")):
            sys.exit("trees-build first")
        if args.command == "tint" and not os.path.isfile(os.path.join(n.ab, "veg_tint_on.dsf")):
            sys.exit("tint-build first")
        if args.command == "mat" and not os.path.isdir(os.path.join(n.ab, f"ter_{args.mode}")):
            sys.exit("mat-build first")
        {"grass": set_grass, "mat": set_mat, "trees": set_trees, "tint": set_tint}[args.command](n, args.mode)
        cmd_status(n, args)
        print("now restart X-Plane (or start it) to load this variant")
        return
    {"build": cmd_build, "install": cmd_install, "uninstall": cmd_uninstall, "status": cmd_status,
     "trees-build": cmd_trees_build, "tint-build": cmd_tint_build,
     "mat-build": cmd_mat_build, "canopy-heights": cmd_canopy_heights,
     "building-heights": cmd_building_heights}[args.command](n, args)


if __name__ == "__main__":
    main()
