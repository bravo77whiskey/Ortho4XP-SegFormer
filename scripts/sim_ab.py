"""In-sim A/B of one tile: swap a variant, then restart X-Plane at the same
saved situation (Reload Scenery is not trusted to pick up every change).

    python scripts/sim_ab.py build     --tile +22+120 [--out E:/Ortho4XP_Review/sim_ab]
    python scripts/sim_ab.py install   --tile +22+120      (X-Plane closed)
    python scripts/sim_ab.py grass on|off      --tile +22+120
    python scripts/sim_ab.py mat 2048|1024|off --tile +22+120
    python scripts/sim_ab.py trees-build --tile +22+120  (after build)
    python scripts/sim_ab.py trees off|on      --tile +22+120
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
to _ab/ first.

The grass swap expects the grass inside the veg DSF, i.e. a tile built before
the grass got its own pack (yOrtho4XP_Grass_Overlays, 890d523b).

trees-build regenerates the trees of the tile's ZL18/19 textures (the
airport zones, where trees come from the canopy-height model) with THIS
checkout's vegetation overlay, twice: 'off' = single trees and forests as
X-Plane's full-height climate forests, 'on' = sized single-species trees and
shorter forest copies (veg_tree_sizes / veg_forest_heights).  Each result
replaces the copied veg DSF's polygons centred inside those textures; the
rest of the tile, grass included, stays as built.  The shorter forests come
from an O4XP_Forest_Heights library written under --out; install links it
into Custom Scenery too (uninstall removes it).  A later 'grass' swap
replaces the trees variant and the other way round.
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

import O4_Terrain_Materials as TMAT  # noqa: E402

DEFAULT_CS = r"D:\SteamLibrary\steamapps\common\X-Plane 12\Custom Scenery"
DEFAULT_OUT = r"E:\Ortho4XP_Review\sim_ab"
CUDA_PYTHON = r"G:\Dev\Ortho4XP\.venv\Scripts\python.exe"
INSTALLED_CFG = r"G:\Dev\Ortho4XP\_internal\Ortho4XP_Data\Ortho4XP.cfg"
CANOPY_MODEL_DIR = r"E:\Ortho4XP_Review\veg_alt\chm_model\ckpt"
LEAF_TYPE_CACHE = r"E:\Ortho4XP_Review\_cache\w0\_leaf_type"
FOREST_PACK = "O4XP_Forest_Heights"
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
TREE_VARIANTS = {"off": {"O4_VEG_TREE_SIZES": "0", "O4_VEG_FOREST_HEIGHTS": "0"},
                 "on": {"O4_VEG_TREE_SIZES": "1", "O4_VEG_FOREST_HEIGHTS": "1"}}


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


def set_trees(n, mode):
    shutil.copyfile(os.path.join(n.ab, f"veg_trees_{mode}.dsf"), n.veg_dsf(n.veg_dir))
    state = load_state(n)
    state["trees"] = mode
    state["grass"] = "on"                            # the trees variants keep the built grass
    save_state(n, state)


# ── swaps ────────────────────────────────────────────────────────────────────
def set_grass(n, mode):
    shutil.copyfile(os.path.join(n.ab, f"veg_grass_{mode}.dsf"), n.veg_dsf(n.veg_dir))
    state = load_state(n)
    state["grass"] = mode
    state.pop("trees", None)                         # the grass variants carry the trees as built
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
          f"trees {s.get('trees', 'as built')}"
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


def cmd_install(n, args):
    if xplane_running():
        sys.exit("quit X-Plane first (it reads scenery_packs.ini at start)")
    s = load_state(n)
    if not s:
        sys.exit("build first")
    forest = os.path.isfile(os.path.join(n.forest_dir, "library.txt"))
    links = [(n.tile_pack, n.tile_dir), (n.veg_pack, n.veg_dir)]
    if forest:
        links.append((FOREST_PACK, n.forest_dir))
    for pack, target in links:
        link = os.path.join(n.cs, pack)
        if not os.path.exists(link):
            subprocess.run(["cmd", "/c", "mklink", "/J", link, target], check=True, stdout=subprocess.DEVNULL)
    path, lines = read_ini(n)
    out = []
    for line in lines:
        if line.strip() in (n.ini_line(n.tile_pack), n.ini_line(n.veg_pack), n.ini_line(FOREST_PACK)):
            continue                                   # re-install: placed again below
        if line.strip() == n.ini_line(n.real_tile_pack):
            out.append(n.ini_line(n.tile_pack))
        elif line.strip() == n.ini_line(VEG_PACK):
            if forest:
                out.append(n.ini_line(FOREST_PACK))    # a library: its place does not matter
            out.append(n.ini_line(n.veg_pack))
            line = n.ini_line(VEG_PACK, enabled=False)
            s["veg_disabled_by_us"] = True
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
    path, lines = read_ini(n)
    out = []
    for line in lines:
        if line.strip() in (n.ini_line(n.tile_pack), n.ini_line(n.veg_pack)):
            continue
        if line.strip() == n.ini_line(FOREST_PACK) and s.get("forest_pack_installed"):
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
    s.update(installed=False, veg_disabled_by_us=False, forest_pack_installed=False)
    save_state(n, s)
    print(f"uninstalled; {VEG_PACK} re-enabled. Test data stays in {n.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["build", "install", "uninstall", "status", "grass", "mat",
                                        "trees-build", "trees"])
    ap.add_argument("mode", nargs="?")
    ap.add_argument("--tile", required=True, help="e.g. +22+120")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--custom-scenery", default=DEFAULT_CS)
    args = ap.parse_args()
    n = Names(args)
    if args.command in ("grass", "mat", "trees"):
        modes = {"grass": ("on", "off"), "mat": ("2048", "1024", "off"), "trees": ("off", "on")}[args.command]
        if args.mode not in modes:
            sys.exit(f"{args.command} needs one of: {' '.join(modes)}")
        if not load_state(n):
            sys.exit("build first")
        if args.command == "trees" and not os.path.isfile(os.path.join(n.ab, "veg_trees_on.dsf")):
            sys.exit("trees-build first")
        {"grass": set_grass, "mat": set_mat, "trees": set_trees}[args.command](n, args.mode)
        cmd_status(n, args)
        print("now restart X-Plane (or start it) to load this variant")
        return
    {"build": cmd_build, "install": cmd_install, "uninstall": cmd_uninstall, "status": cmd_status,
     "trees-build": cmd_trees_build}[args.command](n, args)


if __name__ == "__main__":
    main()
