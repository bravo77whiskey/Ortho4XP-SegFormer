"""End-to-end parity harness for the building overlay, without GPU or real data.

Runs the real ``O4_Building_Overlay.run()`` on a synthetic tile: generated
DDS textures, OSM roads and water, a generated O4SFR ProcGen library and a
simHeaven-style library in a throwaway Custom Scenery folder. The models are
replaced by deterministic fakes that "see" marker colours painted into the
textures, so the real crop iteration, suppression, placement and DSF writing
all run unchanged:

* trained YOLO OBB and stock DOTA YOLO: connected components of marker colours
  in each crop they are given, returned as ultralytics-style OBB results;
* landcover SegFormer: a colour-to-class lookup of the painted texture;
* HeightNet: a height derived from each detection's footprint.

Every scenario runs in a fresh process. The outputs (DSF text, compiled DSF,
overview images, per-texture caches) are hashed into ``manifest.json`` so two
source trees can be compared byte for byte::

    python scripts/building_overlay_parity_harness.py run --src old/src --out out_old
    python scripts/building_overlay_parity_harness.py run --src new/src --out out_new
    python scripts/building_overlay_parity_harness.py compare out_old out_new

``--work`` must be the same path for both runs: absolute paths end up in
cache keys. Pass ``--coverage`` to record line coverage of the overlay module.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PROCGEN = REPO / "scripts" / "asset_pipeline" / "procgen"

LAT, LON = 45, 7
IMG_PX = 4096
FIXED_MTIME = 1_700_000_000
TEXTURES = (
    # (zl, point inside the texture)
    (16, (45.990, 7.020)),   # tile north-west corner: edge margins, out-of-tile
    (16, (45.500, 7.500)),
    (16, (45.500, 7.590)),
    (17, (45.505, 7.505)),   # covers part of the second texture
)

# ── Scenarios ────────────────────────────────────────────────────────────────
# name -> (run() keyword overrides, environment, keep caches from previous).
# Most make_viz runs use the ZL16 textures only, so they stay comparable with
# trees from before the mixed-zoom overview fix; viz_mixed_zl covers the rest.
SCENARIOS = {
    "default": ({}, {}, False),
    "warm_placement_cache": ({}, {}, True),
    "warm_inference_only": ({"ignore_placement_cache": True, "verbose_log": True}, {}, True),
    "self_avoid": ({"verbose_log": True}, {"O4_BLD_OBJECT_SELF_AVOID": "1"}, False),
    "objects_only": ({}, {"O4_BLD_ASSET_MODE": "objects"}, False),
    "viz_zl16": ({"make_viz": True, "dds_filter": "ZL16"}, {}, False),
    "viz_mixed_zl": ({"make_viz": True}, {}, False),
    "facades_only": ({"make_viz": True, "dds_filter": "ZL16"},
                     {"O4_BLD_ASSET_MODE": "facades"}, False),
    "seg_all_agri_drop": ({}, {"O4_BLD_SEG_FIRST": "all", "O4_BLD_AGRI": "drop"}, False),
    "seg_off_agri_off": (
        {}, {"O4_BLD_SEG_FIRST": "off", "O4_BLD_AGRI": "off",
             "O4_BLD_FOOTPRINT_POLYGONS": "0"}, False),
    "roof_color": ({"roof_color_matching": True, "make_viz": True, "dds_filter": "ZL16"},
                   {}, False),
    "roof_color_warm": ({"roof_color_matching": True, "ignore_placement_cache": True,
                         "dds_filter": "ZL16"}, {}, True),
    "smallest_assets": (
        {}, {"O4_BLD_SMALLEST_ASSET_ONLY": "1", "O4_BLD_ALLOW_OVERSIZED_ALIASES": "1",
             "O4_BLD_EXTRA_LIBRARIES": "off"}, False),
    "no_cache_sequential": (
        {"disable_cache": True},
        {"O4_BLD_PREFETCH": "0", "O4_OVERLAY_TIMING_DETAIL": "1",
         "O4_BLD_PLACE_PROFILE": "1", "O4_BLD_BLOCK_DEBUG": "1"}, False),
    "roads_allowed_spacing": (
        {"allow_road_overlap": True, "spacing_m": 5.0}, {"O4_BLD_STRICT_FIT": "1"}, False),
    "heights_off_blanket_drop": (
        {}, {"O4_BLD_HEIGHT_ENABLED": "0", "O4_BLD_PAIR_OVERLAP_RULE": "0",
             "O4_BLD_TILE_DEDUP": "0", "O4_BLD_EVIDENCE_GATE": "0"}, False),
    "yolo_disabled": ({"yolo_enabled": False}, {}, False),
    "debug_image_only": ({"debug_image_only": True, "make_viz": True, "dds_filter": "ZL16"},
                         {}, False),
}


# ── Geometry helpers (independent of the code under test) ───────────────────
def _orthogrid(lat, lon, zl):
    n = 2 ** zl
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return (x // 16) * 16, (y // 16) * 16


def _tile_lat(y, zl):
    return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / 2 ** zl))))


def _texture_bounds(til_x, til_y, zl):
    n = 2 ** zl
    lon_w = til_x / n * 360.0 - 180.0
    lon_e = (til_x + 16) / n * 360.0 - 180.0
    return _tile_lat(til_y, zl), _tile_lat(til_y + 16, zl), lon_w, lon_e


def _px_to_latlon(px, py, bounds):
    lat_n, lat_s, lon_w, lon_e = bounds
    return (lat_n + (lat_s - lat_n) * py / IMG_PX,
            lon_w + (lon_e - lon_w) * px / IMG_PX)


def _rng(*parts):
    seed = int.from_bytes(hashlib.sha1("|".join(map(str, parts)).encode()).digest()[:8], "big")
    return np.random.default_rng(seed)


# ── Marker colours ───────────────────────────────────────────────────────────
# Trained-YOLO building markers: colour -> model class (8-class model).
BUILDING_MARKERS = {
    (181, 71, 52): 0, (176, 92, 70): 1, (150, 60, 50): 2, (160, 150, 140): 3,
    (120, 120, 130): 4, (140, 140, 160): 5, (110, 120, 140): 6, (100, 105, 115): 7,
}
GHOST_MARKER = (171, 163, 112)      # YOLO sees a building, SegFormer sees farmland
GREENHOUSE_MARKER = (236, 238, 232)  # YOLO building on agricultural land
SEG_ONLY_ROOF = (190, 80, 60)        # SegFormer building the detector misses
STOCK_MARKERS = {(40, 200, 220): 14, (200, 60, 200): 2}  # pool, storage tank
COLOURS = {
    "agri": (170, 160, 110), "range": (150, 150, 90), "tree": (60, 110, 50),
    "road": (110, 110, 110), "water": (40, 70, 140), "bare": (180, 170, 150),
}
CLASS_OF_COLOUR = {
    COLOURS["agri"]: 7, COLOURS["range"]: 2, COLOURS["tree"]: 5,
    COLOURS["road"]: 4, COLOURS["water"]: 6, COLOURS["bare"]: 1,
    GHOST_MARKER: 7, GREENHOUSE_MARKER: 7, SEG_ONLY_ROOF: 8,
    **{c: 8 for c in BUILDING_MARKERS},
    **{c: 3 for c in STOCK_MARKERS},
}


# ── Fixture ──────────────────────────────────────────────────────────────────
def _rect(cx, cy, length, width, heading_deg):
    h = math.radians(heading_deg)
    ux, uy = math.cos(h), math.sin(h)
    vx, vy = -uy, ux
    pts = []
    for sl, sw in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        pts.append((cx + sl * ux * length / 2 + sw * vx * width / 2,
                    cy + sl * uy * length / 2 + sw * vy * width / 2))
    return np.round(np.array(pts)).astype(np.int32)


def _paint_texture(name, zl):
    """Paint one texture; returns (image, roads, water rings) in pixel space."""
    import cv2
    rng = _rng("paint", name)
    img = np.empty((IMG_PX, IMG_PX, 3), dtype=np.uint8)
    img[:] = COLOURS["agri"]
    roads, water = [], []
    scale = 2 if zl == 17 else 1                       # px per ZL16 px
    for _ in range(6):                                  # rangeland / bare patches
        cx, cy = rng.integers(0, IMG_PX, 2)
        cv2.ellipse(img, (int(cx), int(cy)), (int(rng.integers(150, 500)),) * 2, 0, 0, 360,
                    COLOURS["range" if rng.random() < 0.6 else "bare"], -1)
    for _ in range(5):                                  # forest blobs
        cx, cy = rng.integers(0, IMG_PX, 2)
        cv2.ellipse(img, (int(cx), int(cy)), (int(rng.integers(80, 300)), int(rng.integers(80, 300))),
                    float(rng.integers(0, 180)), 0, 360, COLOURS["tree"], -1)
    pond = (int(rng.integers(500, 3500)), int(rng.integers(500, 3500)))
    cv2.ellipse(img, pond, (120, 80), 20, 0, 360, COLOURS["water"], -1)
    ring = cv2.ellipse2Poly(pond, (120, 80), 20, 0, 360, 15)
    water.append([(int(x), int(y)) for x, y in ring])

    buildings = []   # (cx, cy, length_px) of every painted building

    def building(cx, cy, length, width, heading, colour):
        cv2.fillPoly(img, [_rect(cx, cy, length, width, heading)], colour)
        buildings.append((cx, cy, length))

    markers = list(BUILDING_MARKERS)
    # Residential neighbourhoods: a road grid with houses in the blocks.
    for n in range(3):
        ox, oy = (int(v) for v in rng.integers(300, IMG_PX - 900, 2))
        size, step = 560 * scale // 2 + 280, 70 * scale // 2 + 35
        heading = float(rng.integers(0, 40))
        for k in range(0, size + 1, step):
            for seg in (((ox, oy + k), (ox + size, oy + k)), ((ox + k, oy), (ox + k, oy + size))):
                cv2.line(img, seg[0], seg[1], COLOURS["road"], 5)
                roads.append(("residential" if k else "primary", [seg[0], seg[1]]))
        for bx in range(ox, ox + size - step + 1, step):
            for by in range(oy, oy + size - step + 1, step):
                for _ in range(int(rng.integers(0, 4))):
                    cx = bx + int(rng.integers(12, step - 12))
                    cy = by + int(rng.integers(12, step - 12))
                    length = float(rng.uniform(5, 14)) * scale
                    width = length * float(rng.uniform(0.55, 0.95))
                    colour = (SEG_ONLY_ROOF if rng.random() < 0.08
                              else markers[int(rng.integers(0, 3))])
                    building(cx, cy, length, width, heading + float(rng.normal(0, 6)), colour)
                if rng.random() < 0.25:                 # garden pool
                    cv2.fillPoly(img, [_rect(bx + step // 2, by + step // 2, 6 * scale, 3 * scale,
                                             heading)], (40, 200, 220))
    # Apartment blocks and an industrial estate with one L-shaped hall.
    ox, oy = (int(v) for v in rng.integers(400, IMG_PX - 1200, 2))
    for i in range(6):
        building(ox + 60 * i, oy, float(rng.uniform(18, 35)) * scale,
                 float(rng.uniform(9, 14)) * scale, 12.0, markers[3 + i % 3])
    ix, iy = ox + 40, oy + 160
    for i in range(4):
        building(ix + 110 * i, iy, float(rng.uniform(40, 70)) * scale,
                 float(rng.uniform(25, 45)) * scale, 3.0, markers[6 + i % 2])
    cv2.fillPoly(img, [_rect(ix + 120, iy + 150, 120 * scale, 35 * scale, 0.0)], markers[7])
    cv2.fillPoly(img, [_rect(ix + 75, iy + 200, 30 * scale, 120 * scale, 0.0)], markers[7])
    for i in range(3):                                  # storage tanks
        cv2.circle(img, (ix + 450 + 18 * i, iy + 60), 6 * scale, (200, 60, 200), -1)
    building(ix + 300, iy + 300, 90 * scale, 70 * scale, 30.0, GHOST_MARKER)
    # Greenhouse rows on farmland.
    gx, gy = (int(v) for v in rng.integers(300, IMG_PX - 600, 2))
    for i in range(5):
        building(gx + 22 * i * scale, gy, 15 * scale, 50 * scale, 0.0, GREENHOUSE_MARKER)
    roads.append(("track", [(gx - 30, gy - 60), (gx + 150, gy - 60)]))
    cv2.line(img, (gx - 30, gy - 60), (gx + 150, gy - 60), COLOURS["road"], 3)
    return img, roads, water, buildings


def _osm_file(path, ways, closed):
    """Write a minimal OSM XML file (bz2) with the given ways."""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6">']
    node_id, way_lines = 1, []
    for way_id, (tags, pts) in enumerate(ways, 1):
        refs = []
        for lat, lon in pts:
            lines.append(f'<node id="{node_id}" lat="{lat:.7f}" lon="{lon:.7f}"/>')
            refs.append(node_id)
            node_id += 1
        if closed:
            refs.append(refs[0])
        way_lines.append(f'<way id="{way_id}">' + "".join(f'<nd ref="{r}"/>' for r in refs)
                         + "".join(f'<tag k="{k}" v="{v}"/>' for k, v in tags.items()) + "</way>")
    lines += way_lines + ["</osm>"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with bz2.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _obj8(path, length, width, height):
    hl, hw = length / 2, width / 2
    verts = []
    for y in (0.0, height):
        for x, z in ((-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)):
            verts.append(f"VT {x:.3f} {y:.3f} {z:.3f} 0 1 0 0.5 0.5")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("I\n800\nOBJ\n\nTEXTURE roof.png\nPOINT_COUNTS 8 0 0 0\n"
                    + "\n".join(verts) + "\n", encoding="utf-8")


def _procgen_library(pkg):
    sys.path.insert(0, str(PROCGEN))
    import build_procgen_library
    from archetypes import build_archetype
    from archetypes.common import fmt_dim
    layout = json.loads((PROCGEN / "atlas_layout.json").read_text(encoding="utf-8"))
    specs = []
    for region in ("generic", "europe"):
        for archetype, bucket, combo, sizes in (
            ("gable", "residential", "res", ((6.2, 5.1, 1), (8.0, 6.5, 1), (11.2, 9.2, 2),
                                              (14.0, 10.0, 2), (18.0, 12.0, 2))),
            ("aptblock", "apartment", "apt", ((24.0, 12.0, 4), (34.0, 13.0, 6))),
            ("flatcom", "commercial", "com", ((40.0, 28.0, 2),)),
            ("warehouse", "industrial", "ind", ((60.0, 40.0, 2), (90.0, 55.0, 2))),
        ):
            for length, width, floors in sizes:
                specs.append((region, archetype, bucket, combo, length, width, floors))
    assets = []
    for region, archetype, bucket, combo, length, width, floors in specs:
        stem = f"{archetype}_{fmt_dim(length)}x{fmt_dim(width)}x{floors}"
        asset_id = f"{region}/{bucket}/{stem}"
        spec = build_archetype(archetype, length, width, floors, 1001,
                               layout["combos"][f"{region}/{combo}"], region)
        physical = f"{asset_id}_v1.obj"
        target = pkg / physical
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = ["I", "800", "OBJ", "", "TEXTURE ../../textures/atlas.png", "",
                 f"POINT_COUNTS {len(spec.vertices)} 0 0 0"]
        lines += [f"VT {x:.6f} {z:.6f} {-y:.6f} 0 0 1 0 0" for x, y, z in spec.vertices]
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        assets.append({
            "id": asset_id, "virtual_path": f"o4sfr/{asset_id}.obj", "archetype": archetype,
            "length_m": length, "width_m": width, "floors": floors,
            "height_m": floors * 3.2, "region": region, "bucket": bucket, "band": "test",
            "flavor": "generic", "enabled": True,
            "variants": [{"variant": 1, "seed": 1001, "physical_path": physical}],
        })
    manifest = pkg.parent / "procgen_manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "lib_version": 1,
                                    "config_sha1": "harness", "assets": assets}))
    with open(os.devnull, "w") as devnull:
        stdout, sys.stdout = sys.stdout, devnull
        try:
            rc = build_procgen_library.main(["--manifest", str(manifest), "--output", str(pkg)])
        finally:
            sys.stdout = stdout
    manifest.unlink()
    assert rc == 0, "build_procgen_library failed"


def _compile_dsf(dsftool, text, dsf_path):
    dsf_path.parent.mkdir(parents=True, exist_ok=True)
    txt = dsf_path.with_suffix(".txt")
    header = "\n".join([
        "PROPERTY sim/planet earth", "PROPERTY sim/overlay 1",
        f"PROPERTY sim/west {LON}", f"PROPERTY sim/east {LON + 1}",
        f"PROPERTY sim/south {LAT}", f"PROPERTY sim/north {LAT + 1}", "", "",
    ])
    txt.write_text(header + text, encoding="utf-8")
    subprocess.run([dsftool, "-text2dsf", str(txt), str(dsf_path)], check=True,
                   capture_output=True)
    txt.unlink()


def _scenery_dsfs(custom, textures, dsftool):
    """simHeaven scenery/network packs and a third-party overlay over the tile."""
    tile = f"{LAT:+03d}{LON:+04d}"
    grid = f"{LAT // 10 * 10:+03d}{LON // 10 * 10:+04d}"
    rel = Path("Earth nav data") / grid / f"{tile}.dsf"
    name, _zl, bounds, buildings = textures[1]
    rng = _rng("scenery", name)
    picks = [buildings[int(i)] for i in rng.choice(len(buildings), 40, replace=False)]
    lines = ["POLYGON_DEF simheaven/facades/residential.fac"]
    lines += [f"OBJECT_DEF simheaven/houses/europe/house_{i}.obj" for i in range(3)]
    lines.append("")
    for cx, cy, length in picks[30:34]:
        lines += ["BEGIN_POLYGON 0 8 2", "BEGIN_WINDING"]
        for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            lat, lon = _px_to_latlon(cx + dx * length, cy + dy * length, bounds)
            lines.append(f"POLYGON_POINT {lon:.7f} {lat:.7f}")
        lines += ["END_WINDING", "END_POLYGON"]
    for k, (cx, cy, _length) in enumerate(picks[:30]):
        lat, lon = _px_to_latlon(cx, cy, bounds)
        lines.append(f"OBJECT {k % 3} {lon:.7f} {lat:.7f} {(k * 37) % 360}.0")
    _compile_dsf(dsftool, "\n".join(lines) + "\n",
                 custom / "simHeaven_X-World_Europe-6-scenery" / rel)

    # A railway and a road across the second texture.
    (la0, lo0), (la1, lo1) = _px_to_latlon(300, 2000, bounds), _px_to_latlon(3800, 2100, bounds)
    (la2, lo2), (la3, lo3) = _px_to_latlon(2000, 300, bounds), _px_to_latlon(2050, 3800, bounds)
    network = "\n".join([
        "NETWORK_DEF lib/g10/roads.net", "",
        f"BEGIN_SEGMENT 0 151 1 {lo0:.7f} {la0:.7f} 0",
        f"END_SEGMENT 2 {lo1:.7f} {la1:.7f} 0",
        f"BEGIN_SEGMENT 0 50 3 {lo2:.7f} {la2:.7f} 0",
        f"SHAPE_POINT {(lo2 + lo3) / 2:.7f} {(la2 + la3) / 2 + 0.002:.7f} 0",
        f"END_SEGMENT 4 {lo3:.7f} {la3:.7f} 0", "",
    ])
    _compile_dsf(dsftool, network, custom / "simHeaven_X-World_Europe-4-network" / rel)

    # A third-party overlay with its own objects (custom-scenery avoidance).
    _name3, _zl3, bounds3, buildings3 = textures[2]
    overlay = custom / "Custom_Test_Overlay"
    _obj8(overlay / "objects" / "hangar.obj", 30, 20, 9)
    rows = ["OBJECT_DEF objects/hangar.obj", ""]
    for cx, cy, _length in buildings3[5:25]:
        lat, lon = _px_to_latlon(cx, cy, bounds3)
        rows.append(f"OBJECT 0 {lon:.7f} {lat:.7f} 0.0")
    _compile_dsf(dsftool, "\n".join(rows) + "\n", overlay / rel)

    packs = sorted(p.name for p in custom.iterdir() if p.is_dir())
    ini = ["I", "1000 Version", "SCENERY", ""]
    ini += [f"SCENERY_PACK Custom Scenery/{p}/" for p in packs]
    (custom / "scenery_packs.ini").write_text("\n".join(ini) + "\n", encoding="utf-8")


def _mesh_with_water(path, textures):
    """An Ortho4XP mesh whose water covers the first texture and a lake in the third."""
    nodes, tris = [], []

    def quad(lat_n, lat_s, lon_w, lon_e):
        base = len(nodes)
        nodes.extend([(lon_w, lat_n), (lon_e, lat_n), (lon_e, lat_s), (lon_w, lat_s)])
        tris.extend([(base + 1, base + 2, base + 3), (base + 1, base + 3, base + 4)])

    lat_n, lat_s, lon_w, lon_e = textures[0][2]
    quad(lat_n + 0.001, lat_s - 0.001, lon_w - 0.001, lon_e + 0.001)
    lat_n, lat_s, lon_w, lon_e = textures[2][2]
    quad(lat_n - (lat_n - lat_s) * 0.10, lat_n - (lat_n - lat_s) * 0.25,
         lon_w + (lon_e - lon_w) * 0.60, lon_w + (lon_e - lon_w) * 0.80)
    out = ["MeshVersionFormatted 1.3", "Dimension 3", "", "Vertices", str(len(nodes))]
    out += [f"{lon:.7f} {lat:.7f} 0 0" for lon, lat in nodes]
    out += ["", "", "Normals"] + ["0 0" for _ in nodes]
    out += ["", "Triangles", str(len(tris))]
    out += [f"{a} {b} {c} 1" for a, b, c in tris]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def build_fixture(work, dsftool):
    from PIL import Image
    if work.exists():
        shutil.rmtree(work)
    tile = f"{LAT:+03d}{LON:+04d}"
    grid = f"{LAT // 10 * 10:+03d}{LON // 10 * 10:+04d}"
    tex_dir = work / "Tiles" / f"zOrtho4XP_{tile}" / "textures"
    tex_dir.mkdir(parents=True)
    all_roads, all_water, textures = [], [], []
    for zl, (plat, plon) in TEXTURES:
        til_x, til_y = _orthogrid(plat, plon, zl)
        name = f"{til_y}_{til_x}_BI{zl}.dds"
        img, roads, water, buildings = _paint_texture(name, zl)
        Image.fromarray(img).save(tex_dir / name)
        bounds = _texture_bounds(til_x, til_y, zl)
        for kind, pts in roads:
            all_roads.append(({"highway": kind}, [_px_to_latlon(x, y, bounds) for x, y in pts]))
        for ring in water:
            all_water.append(({"natural": "water"}, [_px_to_latlon(x, y, bounds) for x, y in ring]))
        textures.append((name, zl, bounds, buildings))
    osm = work / "OSM_data" / grid / tile
    big = [w for w in all_roads if w[0]["highway"] == "primary"]
    small = [w for w in all_roads if w[0]["highway"] != "primary"]
    _osm_file(osm / f"{tile}_big_roads.osm.bz2", big, closed=False)
    _osm_file(osm / f"{tile}_small_roads.osm.bz2", small, closed=False)
    _osm_file(osm / f"{tile}_water.osm.bz2", all_water, closed=True)
    _osm_file(osm / f"{tile}_airports.osm.bz2", [], closed=True)

    custom = work / "X-Plane 12" / "Custom Scenery"
    _procgen_library(custom / "o4sfr_procgen_library")
    sh = custom / "simHeaven_X-World_Europe-5-footprints"
    exports = ["A", "800", "LIBRARY", ""]
    (sh / "facades").mkdir(parents=True)
    (sh / "facades" / "tank.fac").write_text("A\n800\nFACADE\n", encoding="utf-8")
    exports.append("EXPORT simheaven/facades/tank.fac facades/tank.fac")
    for i, (length, width, height) in enumerate(((9, 7, 6), (12, 9, 7), (16, 11, 9))):
        rel = f"objects/house_{i}.obj"
        _obj8(sh / rel, length, width, height)
        exports.append(f"EXPORT simheaven/houses/europe/house_{i}.obj {rel}")
    (sh / "facades" / "greenhouse_1.fac").write_text("A\n800\nFACADE\n", encoding="utf-8")
    exports.append("EXPORT facades/agricultural/greenhouse_1.fac facades/greenhouse_1.fac")
    (sh / "library.txt").write_text("\n".join(exports) + "\n", encoding="utf-8")
    # O4SFR greenhouse tunnel arrays: every width/length the splitter can ask for.
    gh = custom / "o4sfr_greenhouse_library"
    _obj8(gh / "tunnel.obj", 8, 10, 4)
    gh_exports = ["A", "800", "LIBRARY", ""]
    for bays in range(1, 13):
        for length in range(10, 155, 5):
            gh_exports.append(f"EXPORT o4sfr/agri/tunnels_{bays * 8}x{length}.obj tunnel.obj")
    (gh / "library.txt").write_text("\n".join(gh_exports) + "\n", encoding="utf-8")

    _scenery_dsfs(custom, textures, dsftool)
    _mesh_with_water(tex_dir.parent / f"Data{tile}.mesh", textures)

    models = work / "models"
    models.mkdir()
    for name in ("building_obb.pt", "heightnet.pt"):
        (models / name).write_bytes(b"harness checkpoint")
    for root, _dirs, files in os.walk(work):
        for name in files:
            os.utime(os.path.join(root, name), (FIXED_MTIME, FIXED_MTIME))
    return tex_dir


# ── Fake models ──────────────────────────────────────────────────────────────
class FakeObbModel:
    """ultralytics-like OBB model that detects marker colours in each crop."""

    def __init__(self, markers, names, extra_inner=False):
        self.markers = markers
        self.names = names
        self.extra_inner = extra_inner

    def predict(self, source=None, conf=0.25, max_det=300, **_kwargs):
        single = not isinstance(source, (list, tuple))
        crops = [source] if single else list(source)
        return [self._detect(np.ascontiguousarray(crop[..., ::-1]), conf, max_det)
                for crop in crops]

    def _detect(self, rgb, conf, max_det):
        import cv2
        from O4_Remote_GPU import _ShimOBB, _ShimResult
        corners, xywhr, confs, classes = [], [], [], []
        for colour, cls in self.markers.items():
            mask = np.all(rgb == np.array(colour, dtype=np.uint8), axis=2).astype(np.uint8)
            if not mask.any():
                continue
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for label in range(1, n):
                if stats[label, cv2.CC_STAT_AREA] < 6:
                    continue
                pts = np.column_stack(np.nonzero(labels == label)[::-1]).astype(np.float32)
                boxes = [cv2.minAreaRect(pts)]
                if self.extra_inner and stats[label, cv2.CC_STAT_AREA] > 900:
                    (cx, cy), (w, h), a = boxes[0]
                    boxes.append(((cx + 2.0, cy + 1.0), (w * 0.6, h * 0.7), a))
                for k, box in enumerate(boxes):
                    (cx, cy), (w, h), a = box
                    score = 0.35 + (int(cx * 7 + cy * 13 + k * 5) % 60) / 100.0
                    if score < conf:
                        continue
                    corners.append(cv2.boxPoints(box))
                    xywhr.append((cx, cy, w, h, math.radians(a)))
                    confs.append(score)
                    classes.append(cls)
        if not corners:
            return _ShimResult(None)
        order = np.argsort(-np.asarray(confs), kind="stable")[:max_det]
        return _ShimResult(_ShimOBB(
            np.asarray(corners, dtype=np.float32)[order],
            np.asarray(xywhr, dtype=np.float32)[order],
            np.asarray(confs, dtype=np.float32)[order],
            np.asarray(classes, dtype=np.float32)[order],
        ))


def _install_fakes():
    import O4_Building_Overlay as BLD
    import O4_Height_Model as HEIGHT
    import O4_Landcover_Inference as SEG
    import O4_Stock_Yolo_Objects as STOCK

    lut = np.zeros(256 ** 3, dtype=np.int8)
    for colour, cls in CLASS_OF_COLOUR.items():
        lut[(colour[0] << 16) | (colour[1] << 8) | colour[2]] = cls

    def run_inference(model, device, img_rgb, processor=None):
        key = ((img_rgb[..., 0].astype(np.int32) << 16)
               | (img_rgb[..., 1].astype(np.int32) << 8) | img_rgb[..., 2])
        return lut[key]

    SEG.load_vegetation_model = lambda device=None: ("fake-segformer", None, device)
    SEG.run_inference = run_inference

    building = {**BUILDING_MARKERS, GHOST_MARKER: 7, GREENHOUSE_MARKER: 3}
    names = {i: f"class_{i}" for i in range(8)}
    BLD._load_yolo_obb_model = lambda checkpoint, fuse=False: FakeObbModel(
        building, names, extra_inner=True)
    STOCK.load_stock_yolo_model = lambda checkpoint=None: FakeObbModel(
        STOCK_MARKERS, {i: STOCK.DOTA_CLASS_NAMES.get(i, str(i)) for i in range(15)})

    def load_height_model(path, device="cpu"):
        return types.SimpleNamespace(
            _o4_heightnet_arch="fake", _o4_crop_px=96, _o4_window_m=64.0,
            _o4_heightnet_head_layout="fake", _o4_scalar_layout="fake", _o4_device=device,
        )

    def predict_detection_heights(model, image, detections, m_per_px):
        out = []
        for det in detections:
            area = float(det.get("area_m2", 0.0))
            cx, cy = (float(v) for v in det.get("center", (0.0, 0.0)))
            out.append(3.0 + math.sqrt(max(area, 0.0)) / 2.5 + (int(cx + 3 * cy) % 7))
        return out

    HEIGHT.load_height_model = load_height_model
    HEIGHT.predict_detection_heights = predict_detection_heights


# ── Child: one scenario ──────────────────────────────────────────────────────
def run_scenario(args):
    sys.path.insert(0, str(Path(args.src).resolve()))
    import O4_Building_Overlay as BLD
    _install_fakes()
    work = Path(args.work)
    kwargs, _env, _keep = SCENARIOS[args.scenario]
    tile = f"{LAT:+03d}{LON:+04d}"
    grid = f"{LAT // 10 * 10:+03d}{LON // 10 * 10:+04d}"
    params = dict(
        tex_dir=str(work / "Tiles" / f"zOrtho4XP_{tile}" / "textures"),
        lat=LAT, lon=LON,
        out_dsf=str(work / "yOrtho4XP_Bld_Overlays" / "Earth nav data" / grid / f"{tile}.dsf"),
        spacing_m=0.0, close_k=15, open_k=5, min_footprint_m2=12.0,
        cache_dir=str(work / "Overlay_cache" / grid / tile),
        library_cache_dir=str(work / "Overlay_cache" / "_libraries"),
        dsftool_path=args.dsftool,
        custom_scenery_dir=str(work / "X-Plane 12" / "Custom Scenery"),
        yolo_checkpoint=str(work / "models" / "building_obb.pt"),
        height_checkpoint=str(work / "models" / "heightnet.pt"),
        yolo_conf=0.2,
    )
    params.update(kwargs)
    if params.get("dds_filter") == "ZL16":
        params["dds_filter"] = sorted(
            n for n in os.listdir(params["tex_dir"]) if n.endswith("16.dds"))
    os.makedirs(params["cache_dir"], exist_ok=True)
    result = BLD.run(**params)
    print(f"[harness] run() returned {result!r}")


# ── Parent ───────────────────────────────────────────────────────────────────
_VOLATILE = (
    re.compile(r"\d+\.\d+s\b"), re.compile(r"\(\d+\.\dmin\)"),
    re.compile(r"\d+(\.\d+)?ms\b"), re.compile(r"pid \d+"),
)


def _normalise_log(text, work, repo):
    out = []
    for line in text.replace(str(work), "<WORK>").replace(str(repo), "<REPO>").splitlines():
        if line.startswith(("[Bld timing", "    [Bld DDS timing")) or "<frozen" in line:
            continue
        for rx in _VOLATILE:
            line = rx.sub("<t>", line)
        out.append(line)
    return "\n".join(out) + "\n"


def _collect(work, dest):
    files = {}
    roots = [work / "yOrtho4XP_Bld_Overlays", work / "Overlay_cache"]
    for root in roots:
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(work).as_posix()
            if not path.is_file() or "/_libraries/" in f"/{rel}" or re.search(
                    r"/(osm_parse|mesh_water|dsf_parse|dsf_disassembly|obj8_bounds|roof_color)/", rel):
                continue
            files[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    return files


def run_all(args):
    work, out = Path(args.work).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    build_fixture(work, args.dsftool)
    names = args.scenarios.split(",") if args.scenarios else list(SCENARIOS)
    manifest = {}
    for name in names:
        _kwargs, env, keep = SCENARIOS[name]
        if not keep:
            for sub in ("yOrtho4XP_Bld_Overlays", "Overlay_cache"):
                shutil.rmtree(work / sub, ignore_errors=True)
        child_env = {k: v for k, v in os.environ.items() if not k.startswith("O4_")}
        child_env.update(env)
        child_env["PYTHONHASHSEED"] = "0"
        cmd = [sys.executable, "-W", "ignore"]
        if args.coverage:
            cmd += ["-m", "coverage", "run", "-p", f"--data-file={out / '.coverage'}",
                    "--include=*O4_Building_Overlay.py"]
        cmd += [__file__, "child", "--src", args.src, "--work", str(work),
                "--dsftool", args.dsftool, "--scenario", name]
        proc = subprocess.run(cmd, env=child_env, capture_output=True, text=True)
        log = proc.stdout + proc.stderr
        (out / f"{name}.log").write_text(log, encoding="utf-8")
        (out / f"{name}.normalised.log").write_text(
            _normalise_log(log, work, Path(args.src).resolve().parent), encoding="utf-8")
        files = _collect(work, out / name)
        manifest[name] = {"returncode": proc.returncode, "files": files}
        placed = re.search(r"Total: ([\d,]+) placements", log)
        print(f"{name:26s} rc={proc.returncode}  files={len(files):3d}  "
              f"placements={placed.group(1) if placed else '-'}", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))


def compare(args):
    a, b = Path(args.a), Path(args.b)
    ma = json.loads((a / "manifest.json").read_text())
    mb = json.loads((b / "manifest.json").read_text())
    ok = True
    for name in sorted(set(ma) | set(mb)):
        ra, rb = ma.get(name), mb.get(name)
        if ra is None or rb is None:
            print(f"{name}: only in one run")
            ok = False
            continue
        problems = []
        if ra["returncode"] != rb["returncode"]:
            problems.append(f"returncode {ra['returncode']} vs {rb['returncode']}")
        for rel in sorted(set(ra["files"]) | set(rb["files"])):
            if ra["files"].get(rel) != rb["files"].get(rel):
                problems.append(f"differs: {rel}")
        log_a = (a / f"{name}.normalised.log").read_text(encoding="utf-8")
        log_b = (b / f"{name}.normalised.log").read_text(encoding="utf-8")
        if log_a != log_b:
            problems.append("log differs (see diff below)")
        print(f"{name:26s} {'IDENTICAL' if not problems else 'DIFFERENT'}"
              f"  ({len(ra['files'])} files)")
        for line in problems:
            print("    " + line)
        if log_a != log_b:
            import difflib
            for line in list(difflib.unified_diff(log_a.splitlines(), log_b.splitlines(),
                                                  lineterm="", n=0))[:40]:
                print("      " + line)
        ok = ok and not problems
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--src", required=True, help="the src/ folder to test")
    r.add_argument("--out", required=True)
    r.add_argument("--work", default="/tmp/bld_parity_work")
    r.add_argument("--dsftool", default=str(REPO / "Utils" / "lin" / "DSFTool"))
    r.add_argument("--scenarios", default="")
    r.add_argument("--coverage", action="store_true")
    c = sub.add_parser("child")
    c.add_argument("--src", required=True)
    c.add_argument("--work", required=True)
    c.add_argument("--dsftool", required=True)
    c.add_argument("--scenario", required=True)
    m = sub.add_parser("compare")
    m.add_argument("a")
    m.add_argument("b")
    args = ap.parse_args()
    if args.cmd == "run":
        run_all(args)
        return 0
    if args.cmd == "child":
        run_scenario(args)
        return 0
    return compare(args)


if __name__ == "__main__":
    sys.exit(main())
