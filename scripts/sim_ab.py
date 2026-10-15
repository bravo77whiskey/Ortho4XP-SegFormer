"""In-sim A/B of one tile: swap a variant, then restart X-Plane at the same
saved situation (Reload Scenery is not trusted to pick up every change).

    python scripts/sim_ab.py build     --tile +22+120 [--out E:/Ortho4XP_Review/sim_ab]
    python scripts/sim_ab.py install   --tile +22+120      (X-Plane closed)
    python scripts/sim_ab.py grass on|off      --tile +22+120
    python scripts/sim_ab.py mat 2048|1024|off --tile +22+120
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


# ── swaps ────────────────────────────────────────────────────────────────────
def set_grass(n, mode):
    shutil.copyfile(os.path.join(n.ab, f"veg_grass_{mode}.dsf"), n.veg_dsf(n.veg_dir))
    state = load_state(n)
    state["grass"] = mode
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
    print(f"tile {n.tile}: grass {s.get('grass')}, material maps {s.get('mat')}"
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
    for pack, target in ((n.tile_pack, n.tile_dir), (n.veg_pack, n.veg_dir)):
        link = os.path.join(n.cs, pack)
        if not os.path.exists(link):
            subprocess.run(["cmd", "/c", "mklink", "/J", link, target], check=True, stdout=subprocess.DEVNULL)
    path, lines = read_ini(n)
    out = []
    for line in lines:
        if line.strip() in (n.ini_line(n.tile_pack), n.ini_line(n.veg_pack)):
            continue                                   # re-install: placed again below
        if line.strip() == n.ini_line(n.real_tile_pack):
            out.append(n.ini_line(n.tile_pack))
        elif line.strip() == n.ini_line(VEG_PACK):
            out.append(n.ini_line(n.veg_pack))
            line = n.ini_line(VEG_PACK, enabled=False)
            s["veg_disabled_by_us"] = True
        out.append(line)
    if n.ini_line(n.tile_pack) not in out or n.ini_line(n.veg_pack) not in out:
        sys.exit(f"{n.real_tile_pack} or {VEG_PACK} not found enabled in scenery_packs.ini; nothing changed")
    write_ini(n, path, out)
    s["installed"] = True
    save_state(n, s)
    print(f"installed: {n.tile_pack} above {n.real_tile_pack}, {n.veg_pack} above {VEG_PACK} (disabled)")


def cmd_uninstall(n, args):
    if xplane_running():
        sys.exit("quit X-Plane first")
    s = load_state(n)
    path, lines = read_ini(n)
    out = []
    for line in lines:
        if line.strip() in (n.ini_line(n.tile_pack), n.ini_line(n.veg_pack)):
            continue
        if line.strip() == n.ini_line(VEG_PACK, enabled=False) and s.get("veg_disabled_by_us"):
            line = n.ini_line(VEG_PACK)
        out.append(line)
    if out != lines:
        write_ini(n, path, out)
    for pack in (n.tile_pack, n.veg_pack):
        link = os.path.join(n.cs, pack)
        if os.path.isdir(link) and os.path.realpath(link) != os.path.abspath(link):
            os.rmdir(link)                              # removes the junction only
    s.update(installed=False, veg_disabled_by_us=False)
    save_state(n, s)
    print(f"uninstalled; {VEG_PACK} re-enabled. Test data stays in {n.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["build", "install", "uninstall", "status", "grass", "mat"])
    ap.add_argument("mode", nargs="?")
    ap.add_argument("--tile", required=True, help="e.g. +22+120")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--custom-scenery", default=DEFAULT_CS)
    args = ap.parse_args()
    n = Names(args)
    if args.command in ("grass", "mat"):
        modes = {"grass": ("on", "off"), "mat": ("2048", "1024", "off")}[args.command]
        if args.mode not in modes:
            sys.exit(f"{args.command} needs one of: {' '.join(modes)}")
        if not load_state(n):
            sys.exit("build first")
        (set_grass if args.command == "grass" else set_mat)(n, args.mode)
        cmd_status(n, args)
        print("now restart X-Plane (or start it) to load this variant")
        return
    {"build": cmd_build, "install": cmd_install, "uninstall": cmd_uninstall, "status": cmd_status}[args.command](n, args)


if __name__ == "__main__":
    main()
