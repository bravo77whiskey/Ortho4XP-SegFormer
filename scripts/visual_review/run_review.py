"""Run the production building + vegetation overlays on the review samples.

Usage (from the repo, with the repo .venv):
    python scripts/visual_review/run_review.py RUN_NAME --note "what changed"
        [--cfg KEY=VALUE ...] [--samples key1 key2 ...] [--workers 2]

Textures that share a tile and a zoom level are built together as one overlay
run, the way a real tile build sees them; different zoom levels never share a
run (the overlay drops lower-ZL detections under higher-ZL textures, which
would change their results). Groups are spread over --workers parallel worker
processes, each with its own cache folder. Settings come from the installed
Ortho4XP global config (read-only); nothing is linked into or changed in
Custom Scenery. Output goes under DATA (default E:\\Ortho4XP_Review, override
with O4_REVIEW_DATA): runs/RUN_NAME/<provider>/<sample>.js per texture,
runs/RUN_NAME/data_<provider>.js, runs_index.js and a copy of index.html, the
viewer, which is served from DATA.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
# Review data (textures, runs, caches, logs, the served viewer) lives outside
# the repo and off C:. Override with the O4_REVIEW_DATA environment variable.
DATA = os.environ.get("O4_REVIEW_DATA", r"E:\Ortho4XP_Review")
LIVE_CFG = r"G:\Dev\Ortho4XP\_internal\Ortho4XP_Data\Ortho4XP.cfg"
CUDA_VENV = r"G:\Dev\Ortho4XP\.venv"

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--providers", nargs="+", default=["BI"])
ap.add_argument("--samples", nargs="*")
ap.add_argument("--set", default="hotspots", help="sample set folder under sets/")
ap.add_argument("--note", help="what this run changes (shown in the viewer's A/B lists)")
ap.add_argument("--cfg", action="append", default=[], metavar="KEY=VALUE",
                help="override one Ortho4XP setting on top of the installed config (repeatable)")
ap.add_argument("--workers", type=int, default=2, help="parallel worker processes (GPU shared)")
ap.add_argument("--skip-veg", action="store_true")
ap.add_argument("--skip-bld", action="store_true")
# Internal: a worker process runs the groups listed in this JSON file.
ap.add_argument("--worker", help=argparse.SUPPRESS)
ap.add_argument("--cache-root", help=argparse.SUPPRESS)
args = ap.parse_args()

run_dir = os.path.join(DATA, "runs", args.run)


def set_dir(provider):
    """Folder holding a sample set's textures and samples.json, relative to DATA."""
    return f"sets/{args.set}/{provider}"


def load_samples(provider):
    samples = json.load(open(os.path.join(DATA, set_dir(provider), "samples.json")))
    if args.samples:
        samples = [s for s in samples if s["key"] in args.samples]
    return samples


def tile_of(sample):
    lat_s, lon_w, lat_n, lon_e = sample["bbox"]
    return int((lat_s + lat_n) // 2), int((lon_w + lon_e) // 2)


# ── Orchestrator: plan groups, start workers, write the manifests ────────────
def plan_groups():
    groups = {}
    for provider in args.providers:
        for s in load_samples(provider):
            lat, lon = tile_of(s)
            key = (provider, lat, lon, s["zl"])
            groups.setdefault(key, {"provider": provider, "lat": lat, "lon": lon,
                                    "zl": s["zl"], "keys": []})["keys"].append(s["key"])
    return list(groups.values())


def assign(groups, n):
    """Largest groups first, each to the worker with the least work so far."""
    loads = [[0, []] for _ in range(max(1, n))]
    for g in sorted(groups, key=lambda g: -len(g["keys"])):
        slot = min(loads, key=lambda l: l[0])
        slot[0] += len(g["keys"])
        slot[1].append(g)
    return [groups for _, groups in loads if groups]


def write_manifest(provider):
    """List every sample this run has for a provider (re-runs of a subset keep the rest)."""
    rows = json.load(open(os.path.join(DATA, set_dir(provider), "samples.json")))
    manifest = {}
    for r in rows:
        if os.path.exists(os.path.join(run_dir, provider, r["key"] + ".js")):
            manifest[r["key"]] = {"file": r["file"], "label": r["label"], "zl": r["zl"],
                                  "location": r.get("location", ""),
                                  "src": f"runs/{args.run}/{provider}/{r['key']}.js",
                                  "img": f"{set_dir(provider)}/{r['file']}"}
    with open(os.path.join(run_dir, f"data_{provider}.js"), "w") as f:
        f.write("window.REVIEW = window.REVIEW || {};\n")
        f.write(f"window.REVIEW[{json.dumps(args.run + '/' + provider)}] = {json.dumps(manifest)};\n")


def write_index():
    index = []
    for r in sorted(os.listdir(os.path.join(DATA, "runs"))):
        if not os.path.isdir(os.path.join(DATA, "runs", r)):
            continue
        for f in sorted(os.listdir(os.path.join(DATA, "runs", r))):
            m = re.match(r"data_(.+)\.js$", f)
            if m:
                note_path = os.path.join(DATA, "runs", r, "note.txt")
                index.append({"run": r, "provider": m.group(1), "src": f"runs/{r}/{f}",
                              "mtime": int(os.path.getmtime(os.path.join(DATA, "runs", r, f))),
                              "note": open(note_path, encoding="utf-8").read() if os.path.exists(note_path) else ""})
    with open(os.path.join(DATA, "runs_index.js"), "w") as f:
        f.write("window.REVIEW_INDEX = " + json.dumps(index, indent=1) + ";\n")
    # The viewer is served from DATA next to the runs it reads.
    shutil.copy(os.path.join(HERE, "index.html"), os.path.join(DATA, "index.html"))


def orchestrate():
    os.makedirs(os.path.join(run_dir, "_logs"), exist_ok=True)
    if args.note:
        with open(os.path.join(run_dir, "note.txt"), "w", encoding="utf-8") as f:
            f.write(args.note.strip())
    with open(os.path.join(run_dir, "overrides.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(args.cfg))
    groups = plan_groups()
    plan = assign(groups, args.workers)
    t0 = time.time()
    print(f"{sum(len(g['keys']) for g in groups)} textures in {len(groups)} builds "
          f"on {len(plan)} worker(s)", flush=True)
    procs = []
    for i, worker_groups in enumerate(plan):
        plan_path = os.path.join(run_dir, "_logs", f"worker_{i}.json")
        json.dump(worker_groups, open(plan_path, "w"))
        cmd = [sys.executable, os.path.abspath(__file__), args.run, "--set", args.set,
               "--worker", plan_path, "--cache-root", os.path.join(DATA, "_cache", f"w{i}")]
        for override in args.cfg:
            cmd += ["--cfg", override]
        if args.skip_veg:
            cmd.append("--skip-veg")
        if args.skip_bld:
            cmd.append("--skip-bld")
        log = open(os.path.join(run_dir, "_logs", f"worker_{i}.log"), "w")
        # TMP/TEMP on the data drive: the overlays' transient caches go there, not C:.
        tmp = os.path.join(DATA, "_tmp")
        os.makedirs(tmp, exist_ok=True)
        procs.append((i, subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                          env=dict(os.environ, PYTHONUNBUFFERED="1", TMP=tmp, TEMP=tmp)), log))
    failed = 0
    for i, proc, log in procs:
        rc = proc.wait()
        log.close()
        failed += rc != 0
        for line in open(os.path.join(run_dir, "_logs", f"worker_{i}.log"), encoding="utf-8", errors="replace"):
            if line.startswith(("== ", "   ")) and ("textures" in line or "buildings," in line or line.startswith("== ")):
                print(f"[w{i}] {line.rstrip()}", flush=True)
            elif "Traceback" in line:
                print(f"[w{i}] {line.rstrip()}", flush=True)
        print(f"[w{i}] exit {rc}", flush=True)
    for provider in args.providers:
        write_manifest(provider)
    write_index()
    print(f"done in {time.time() - t0:.0f} s" + (f" ({failed} worker(s) failed)" if failed else ""))
    return 1 if failed else 0


if not args.worker:
    sys.exit(orchestrate())


# ── Worker: run the overlays for each assigned group ─────────────────────────
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
import O4_File_Names as FNAMES  # noqa: E402
sys.path.append(FNAMES.Provider_dir)
import O4_Imagery_Utils as IMG  # noqa: E402
import O4_Geo_Utils as GEO  # noqa: E402
import O4_Tile_Utils as TILE  # noqa: E402,F401
import O4_GUI_Utils as GUI  # noqa: E402,F401
import O4_Config_Utils as CFG  # noqa: E402
import O4_Cfg_Vars as CFGVARS  # noqa: E402
import O4_Scenery_Links as SLINK  # noqa: E402
import O4_Overlay_Pipeline as PIPELINE  # noqa: E402
import O4_Building_Overlay as BLD  # noqa: E402

# The user's installed settings, applied over the repo defaults.
CFG.global_cfg_file = LIVE_CFG
for line in open(LIVE_CFG):
    line = line.strip()
    if not line or line[0] == "#" or "=" not in line:
        continue
    var, value = line.split("=", 1)
    var, value = CFG.normalize_config_entry(var, CFG.config_compatibility(value))
    try:
        CFG.set_global_variables(var, value)
        CFG.set_global_variables(CFG.global_prefix + var, value)
    except Exception:
        pass
# Never touch the real X-Plane install.
CFG.auto_link_custom_scenery = False
SLINK.auto_link_tile = lambda tile: None
PIPELINE._finish_veg_tile_replacement = lambda *a, **kw: None
PIPELINE._VENV_DIR = CUDA_VENV
# Candidate outcomes + blocking scenery per texture (never changes placements).
os.environ["O4_BLD_REVIEW_DUMP"] = "1"
os.environ.setdefault("O4_BLD_REVIEW_CONF", "0.01")
# This worker's own cache folder: input caches (library scans, leaf-type
# maps, simHeaven parses) stay warm across runs without two workers writing
# the same files. Overlay result caches stay off per the user's config.
FNAMES.Overlay_cache_dir = args.cache_root

IMG.initialize_extents_dict()
IMG.initialize_color_filters_dict()
IMG.initialize_providers_dict()


def dds_from_jpg(jpg, dds):
    # Same conversion Ortho4XP applies to a downloaded orthophoto (BC1, fast).
    subprocess.run([IMG.dds_convert_cmd, "-bc1", "-fast", jpg, dds],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def parse_dsf_text(path):
    """Return (objects, polygons) from a DSF text file."""
    obj_defs, poly_defs, objects, polys = [], [], [], []
    cur = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            k = p[0]
            if k == "OBJECT_DEF":
                obj_defs.append(p[1])
            elif k == "POLYGON_DEF":
                poly_defs.append(p[1])
            elif k == "OBJECT":
                objects.append((obj_defs[int(p[1])], float(p[2]), float(p[3]), float(p[4])))
            elif k == "BEGIN_POLYGON":
                cur = {"def": poly_defs[int(p[1])], "param": float(p[2]), "rings": []}
            elif k == "BEGIN_WINDING":
                cur["rings"].append([])
            elif k == "POLYGON_POINT":
                cur["rings"][-1].append((float(p[1]), float(p[2])))
            elif k == "END_POLYGON":
                polys.append(cur)
                cur = None
    return objects, polys


def to_px(lon, lat, bbox):
    """Lon/lat to texture pixels (Web Mercator rows, like the DDS itself)."""
    lat_s, lon_w, lat_n, lon_e = bbox
    x = (lon - lon_w) / (lon_e - lon_w) * 4096.0
    y_n, y_s = GEO.geo_to_webm(lon_w, lat_n)[1], GEO.geo_to_webm(lon_w, lat_s)[1]
    y = (y_n - GEO.geo_to_webm(lon, lat)[1]) / (y_n - y_s) * 4096.0
    return round(x, 1), round(y, 1)


def object_footprint(obj_path, lon, lat, heading, bbox, m_per_px):
    bounds = BLD._bounds_for_object_path(obj_path)
    if bounds is None:
        dims = BLD._default_object_dims(obj_path) or BLD._simheaven_object_dims(obj_path)
        bounds = BLD._bounds_from_dimensions(dims[0], dims[1])
    cx, cy = to_px(lon, lat, bbox)
    # Same basis as the overlay's own footprint test, kept sub-pixel.
    pts = BLD._footprint_poly_metres(0.0, 0.0, bounds, heading)
    return [[round(cx + float(x) / m_per_px, 1), round(cy + float(y) / m_per_px, 1)] for x, y in pts]


def read_outputs(out_dir):
    """Decompile every DSF a group's overlays wrote: [(kind, objects, polys)]."""
    parsed = []
    for d, _, files in os.walk(out_dir):
        for f in sorted(files):
            if f.endswith(".dsf"):
                dsf = os.path.join(d, f)
                txt = dsf[:-4] + "_review.txt"
                subprocess.run([PIPELINE._dsftool_path(), "--dsf2text", dsf, txt], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                parsed.append(parse_dsf_text(txt))
    return parsed


def inside(px):
    return 0.0 <= px[0] < 4096.0 and 0.0 <= px[1] < 4096.0


def collect(parsed, sample, m_per_px):
    """The group's placements that belong to this texture, in its pixels."""
    bld, veg = [], []
    bbox = sample["bbox"]
    for objects, polys in parsed:
        for obj_path, lon, lat, hdg in objects:
            if not inside(to_px(lon, lat, bbox)):
                continue
            bld.append({"t": "obj", "a": obj_path,
                        "p": object_footprint(obj_path, lon, lat, hdg, bbox, m_per_px)})
        for poly in polys:
            rings = [[list(to_px(lon, lat, bbox)) for lon, lat in r] for r in poly["rings"]]
            outer = rings[0] if rings and rings[0] else None
            if not outer:
                continue
            centre = (sum(p[0] for p in outer) / len(outer), sum(p[1] for p in outer) / len(outer))
            if not inside(centre):
                continue
            if poly["def"].lower().endswith(".for"):
                veg.append({"a": poly["def"], "d": poly["param"], "r": rings})
            else:
                bld.append({"t": "fac", "a": poly["def"], "h": poly["param"], "r": rings})
    return bld, veg


def write_sample(provider, key, data):
    """One JS file per sample, so the viewer only loads what it shows."""
    path = os.path.join(run_dir, provider, key + ".js")
    with open(path, "w") as f:
        f.write("window.REVIEW_SAMPLE = window.REVIEW_SAMPLE || {};\n")
        f.write(f"window.REVIEW_SAMPLE[{json.dumps(args.run + '/' + provider + '/' + key)}] = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")


def run_group(group):
    t0 = time.time()
    provider, lat, lon, zl = group["provider"], group["lat"], group["lon"], group["zl"]
    by_key = {s["key"]: s for s in json.load(open(os.path.join(DATA, set_dir(provider), "samples.json")))}
    samples = [by_key[k] for k in group["keys"]]
    gdir = os.path.join(run_dir, provider, "_builds", f"{lat:+03d}{lon:+04d}_zl{zl}")
    shutil.rmtree(gdir, ignore_errors=True)
    os.makedirs(os.path.join(gdir, "textures"))
    os.makedirs(os.path.join(run_dir, provider), exist_ok=True)
    for s in samples:
        dds_from_jpg(os.path.join(DATA, set_dir(provider), s["file"]),
                     os.path.join(gdir, "textures", s["file"][:-4] + ".dds"))
    PIPELINE._data_dir = os.path.join(gdir, "out")
    tile = CFG.Tile(lat, lon, gdir)
    tile.read_from_config(use_global=True)
    for override in args.cfg:
        key, value = override.split("=", 1)
        key, value = key.strip(), value.strip()
        kind = CFGVARS.cfg_vars[key]["type"]
        setattr(tile, key, (value.lower() in ("1", "true", "yes")) if kind is bool else kind(value))
    print(f"== {provider} {' '.join(group['keys'])} (tile {lat:+03d}{lon:+04d}, ZL{zl})", flush=True)
    cache_dir = FNAMES.overlay_cache_dir(lat, lon)
    review_json = {s["key"]: os.path.join(cache_dir, s["file"][:-4] + "_review.json") for s in samples}
    for path in review_json.values():
        if os.path.exists(path):
            os.remove(path)
    if not args.skip_bld:
        PIPELINE.apply_tile_settings(tile, CFGVARS.list_bld_pipeline_vars)
        PIPELINE.process_bld_tile(lat, lon, gdir)
    if not args.skip_veg:
        PIPELINE.apply_tile_settings(tile, CFGVARS.list_veg_pipeline_vars)
        PIPELINE.process_veg_tile(lat, lon, gdir)
    parsed = read_outputs(os.path.join(gdir, "out"))
    from math import cos, radians
    for s in samples:
        lat_s, lon_w, lat_n, lon_e = s["bbox"]
        m_per_px = (lon_e - lon_w) * 111320.0 * cos(radians((lat_n + lat_s) / 2)) / 4096.0
        bld, veg = collect(parsed, s, m_per_px)
        diag = None
        if os.path.exists(review_json[s["key"]]):
            shutil.copy(review_json[s["key"]], os.path.join(gdir, s["key"] + "_review.json"))
            diag = json.load(open(review_json[s["key"]], encoding="utf-8"))
        write_sample(provider, s["key"], {
            "file": s["file"], "label": s["label"], "zl": s["zl"],
            "bld": bld, "veg": veg, "diag": diag})
        print(f"   {s['key']}: {len(bld)} buildings, {len(veg)} tree polygons", flush=True)
    print(f"   group done in {time.time() - t0:.0f} s", flush=True)


for group in json.load(open(args.worker)):
    run_group(group)
