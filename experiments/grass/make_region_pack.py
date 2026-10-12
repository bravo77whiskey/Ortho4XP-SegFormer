"""Grass spike, round 2: one shared colour palette over a radius around a point,
multi-tier LOD tufts, one X-Plane 12 test pack.

Usage (CUDA venv):
  python make_region_pack.py --center 22.5772 120.3500 --radius-nm 2 \
      --textures 28528_54656 28528_54672 28544_54656 28544_54672

Colour: Lab, normalised-Gaussian smoothed over grass pixels only (SIGMA_M),
k-means palette pooled over every grass pixel in the circle (K), mode filter
(WIN_M), polygons >= MIN_M2.  Geometry: --tiers "far:blades:segs:width,..."
nearest first; each farther tier reuses the first N blades of the nearest
tuft, widened, so the silhouette stays put across the switch.
"""
import argparse, glob, json, math, os, shutil, subprocess, sys
import numpy as np, cv2
from PIL import Image

SPIKE = r"E:\Ortho4XP_Review\_grass_spike"
SETS = [r"E:\Ortho4XP_Review\sets\hotspots\BI", os.path.join(SPIKE, "extra")]
DSFTOOL = r"G:\Dev\Ortho4XP-SegFormer\Utils\win\DSFTool.exe"
PACK_NAME = "O4XP_Grass_Spike"
RANGELAND = 2

ap = argparse.ArgumentParser()
ap.add_argument("--center", nargs=2, type=float, required=True)
ap.add_argument("--radius-nm", type=float, default=2.0)
ap.add_argument("--textures", nargs="+", default=[])
ap.add_argument("--tile-dir")
ap.add_argument("--airports", choices=("auto", "none"), default="auto",
                help="auto: keep grass off apt.dat runways/taxiways/aprons of every airport in the "
                     "circle, highest-priority pack per airport (needs --tile-dir inside Custom Scenery)")
ap.add_argument("--roads", choices=("auto", "none"), default="auto",
                help="auto: keep grass off the simHeaven road + rail network X-Plane draws "
                     "for this tile (needs --tile-dir inside Custom Scenery)")
ap.add_argument("--road-extra-m", type=float, default=0.0,
                help="added to each road type's half-width")
ap.add_argument("--pave-buffer-m", type=float, default=2.0,
                help="grow airport pavement by this much before removing it from the grass")
ap.add_argument("--k", type=int, default=48)
ap.add_argument("--fine-m", type=float, default=1.2,
                help="grass mask resolution: finer textures are downsampled to about this")
ap.add_argument("--coarse-m", type=float, default=4.0,
                help="grid for the colour field, palette assignment and clean-up")
ap.add_argument("--workers", type=int, default=0, help="threads (0 = min(8, cores))")
ap.add_argument("--simplify-m", type=float, default=1.5,
                help="Douglas-Peucker tolerance for grass outlines, metres")
ap.add_argument("--green-chroma", type=float, default=10.0,
                help="fused: Lab chroma a pixel needs (hue 95-170 deg) to count as clearly green")
ap.add_argument("--sparse-density", type=float, default=0.25,
                help="fused: DSF density for sparse grass (steppe climates), 0..1")
ap.add_argument("--mask", choices=("segformer", "colour", "fused"), default="segformer",
                help="segformer: rangeland class only; colour: grass everywhere except grey / bright / roof-coloured pixels (no model)")
ap.add_argument("--sigma-m", type=float, default=20.0)
ap.add_argument("--win-m", type=float, default=41.0)
ap.add_argument("--min-m2", type=float, default=400.0)
ap.add_argument("--spacing", type=float, default=1.0)
ap.add_argument("--tiers", default="40:24:3:1.0,200:24:2:1.0,800:24:1:1.0")
ap.add_argument("--px-mrad", type=float, default=0.55,
                help="one screen pixel in milliradians (2560 px across ~70 deg = 0.55); "
                     "0 disables the pixel-width rule")
ap.add_argument("--min-px", type=float, default=1.0,
                help="blade width in pixels at the far end of each farther detail level")
ap.add_argument("--tier-contrast", default="1,0.4,0",
                help="per level: 1 = full root/tip gradient, 0 = flat ground colour")
ap.add_argument("--tier-wind", default="1,0.3,0", help="per level wind scale")
ap.add_argument("--tier-height", default="1,1,1",
                help="per level blade height scale (a short last level sinks the grass into the ground at the draw-distance edge)")
ap.add_argument("--tier-tip", default="0.25,0.7,0.7",
                help="per level blade width at the tip as a fraction of the base")
ap.add_argument("--shadows", action="store_true")
ap.add_argument("--tint-tiers", action="store_true",
                help="diagnostic: 2nd detail level red, 3rd blue, to see where each starts")
ap.add_argument("--name", default="region")
ap.add_argument("--out", default=os.path.join(SPIKE, "pack"))
ap.add_argument("--near-layer", default="",
                help="FAR:FADE_END:BLADES[:COPIES] adds COPIES (default 1, max 3) extra forest layers of "
                     "near-only tufts (e.g. 50:60:6:2), each offset half a cell, ending on an empty mesh at FADE_END")
ap.add_argument("--only-for", action="store_true",
                help="keep the existing pack in --out (DSF, palette); rewrite only the "
                     ".for files and textures with the current geometry options")
args = ap.parse_args()
TIERS = [tuple(float(x) if i in (0, 3) else int(x) for i, x in enumerate(t.split(":")))
         for t in args.tiers.split(",")]
LOD = TIERS[-1][0]
BLADE_W = 0.011                 # mean near-blade width (m), tuft() draws 0.008..0.014
CONTRAST = [float(x) for x in args.tier_contrast.split(",")]
WIND = [float(x) for x in args.tier_wind.split(",")]
TIP = [float(x) for x in args.tier_tip.split(",")]
HEIGHT = [float(x) for x in args.tier_height.split(",")]
# Pixel-width rule: a blade thinner than a pixel lands on it in one frame and
# misses it in the next, which is the shimmer.  Every farther level gets
# blades at least --min-px wide at its far end, and correspondingly fewer of
# them so the tuft covers about the same ground (near tuft = 24 x 1.1 cm).
if args.px_mrad > 0:
    for i in range(1, len(TIERS)):
        far, blades, segs, wscale = TIERS[i]
        need = far * args.px_mrad / 1000.0 * args.min_px / BLADE_W
        wscale = max(wscale, need)
        blades = max(1, min(blades, int(round(24 / wscale))))
        TIERS[i] = (far, blades, segs, wscale)
print("detail levels (far m, blades, segments, width x):",
      [(f, b, sg, round(w, 1)) for f, b, sg, w in TIERS])
# One small opaque texture per palette colour (TEX_W x TEX_H per block; a
# block per detail level with --tint-tiers).  A shared atlas was tried first:
# at coarse mips, bilinear sampling near cell edges pulled in neighbouring
# colours and the empty transparent-black area (roots up to -50 L, alpha
# down to 143), which showed in the sim as dark speckle beyond ~150 m.  A
# single-colour texture's mips just converge on that colour.
TEX_W, TEX_H = 16, 64
TINTS = [(0.0, 0.0, 0.0), (0.0, 45.0, 25.0), (0.0, 20.0, -55.0)]   # Lab offsets per detail level


# ── helpers ──────────────────────────────────────────────────────────────────
def merc_y(lat):
    return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def inv_merc(y):
    return math.degrees(2 * math.atan(math.exp(y)) - math.pi / 2)


def lab2rgb(lab):
    lab = np.asarray(lab, np.float32).reshape(-1, 1, 3)
    return (np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1) * 255).round().astype(np.uint8).reshape(-1, 3)


def smooth_masked(lab_img, mask, sigma_px):
    m = mask.astype(np.float32)
    w = cv2.GaussianBlur(m, (0, 0), sigma_px)
    out = np.empty_like(lab_img)
    for c in range(3):
        out[..., c] = cv2.GaussianBlur(lab_img[..., c] * m, (0, 0), sigma_px) / np.maximum(w, 1e-6)
    return out


def assign(px, centres):
    best = np.full(len(px), np.inf, np.float32); lbl = np.zeros(len(px), np.int32)
    for i, c in enumerate(centres):
        d = ((px - c) ** 2).sum(1); b = d < best; best[b] = d[b]; lbl[b] = i
    return lbl


def mode_filter(labels, mask, k, win):
    best_n = np.full(labels.shape, -1, np.float32); out = labels.copy()
    for i in range(k):
        n = cv2.boxFilter(((labels == i) & mask).astype(np.float32), -1, (win, win), normalize=False)
        b = n > best_n; out[b] = i; best_n[b] = n[b]
    out[~mask] = -1
    return out


def is_grass_colour(L, a, b):
    """Spike rule: drop pinkish/brown (soil, granite) and pale grey (bare rock)."""
    C = math.hypot(a, b); h = math.degrees(math.atan2(b, a)) % 360
    return h >= 80 and not (L > 55 and C < 18)


# ── atlas + .for ─────────────────────────────────────────────────────────────
def build_texture(lab_colour, path):
    """Opaque root->tip ramp around one palette colour (row 0 = tip), with a
    +-4 L ramp across so blades sampling different u vary a little."""
    L, a, b = lab_colour
    blocks = len(TIERS) if args.tint_tiers else 1
    yy, xx = np.mgrid[0:TEX_H, 0:TEX_W].astype(np.float32)
    up = 1.0 - yy / (TEX_H - 1)
    side = xx / (TEX_W - 1) * 2 - 1
    cols = []
    for t in range(blocks):
        dL, da, db = TINTS[t] if args.tint_tiers else TINTS[0]
        lab = np.stack([L + dL - 7 + 14 * up + 4 * side, a + da + 0.75 - 1.5 * up, b + db - 1.5 + 3 * up], -1)
        cols.append(lab2rgb(lab.reshape(-1, 3)).reshape(TEX_H, TEX_W, 3))
    Image.fromarray(np.hstack(cols), "RGB").save(path)


def cell_uv(block, fx, fy):
    """fx across the block, fy 0 = root.  X-Plane t=0 is the image bottom."""
    blocks = len(TIERS) if args.tint_tiers else 1
    px = block * TEX_W + 2 + fx * (TEX_W - 4)
    py = 2 + (1 - fy) * (TEX_H - 4)
    return px / (TEX_W * blocks), 1 - py / TEX_H


def tuft(seed, block, height, n_blades, segs, width_scale, contrast=1.0, tip=0.25, offset=(0.0, 0.0)):
    rng = np.random.default_rng(seed)
    blades = []
    for _ in range(24 if n_blades <= 24 else n_blades):   # same draw order for every tier
        blades.append(dict(ang=rng.uniform(0, 2 * math.pi), r0=rng.uniform(0, 0.18),
                           lean=rng.uniform(0.15, 0.45) * height, face=rng.uniform(0, 2 * math.pi),
                           h=height * rng.uniform(0.6, 1.0), w=rng.uniform(0.008, 0.014),
                           fx=rng.uniform(0, 1), phase=rng.uniform(0, 1)))
    verts, idx = [], []
    for bl in blades[:n_blades]:
        x0 = offset[0] + bl["r0"] * math.cos(bl["ang"])
        z0 = offset[1] + bl["r0"] * math.sin(bl["ang"])
        w = bl["w"] * width_scale
        dx, dz = math.cos(bl["face"]) * w, math.sin(bl["face"]) * w
        lx, lz = math.cos(bl["ang"]), math.sin(bl["ang"])
        base = len(verts)
        for s in range(segs + 1):
            f = s / segs
            cx, cz = x0 + lx * bl["lean"] * f * f, z0 + lz * bl["lean"] * f * f
            cy = bl["h"] * f * (1 - 0.2 * f)
            # contrast squeezes the sampled ramp toward the texture's middle,
            # which is the palette colour: 0 = flat ground colour
            u, v = cell_uv(block, 0.5 + (bl["fx"] - 0.5) * contrast, 0.5 + (f - 0.5) * contrast)
            wgt = (f * f, 0.5 * f, bl["phase"])
            t = 1 - (1 - tip) * f
            if s < segs or tip > 0.3:      # wide-tipped blades end in an edge, not a point
                verts.append((cx - dx * t, cy, cz - dz * t, u, v, wgt))
                verts.append((cx + dx * t, cy, cz + dz * t, u, v, wgt))
            else:
                verts.append((cx, cy, cz, u, v, wgt))
        tris = []
        quads = segs if tip > 0.3 else segs - 1
        for s in range(quads):
            a0 = base + 2 * s
            tris += [(a0, a0 + 2, a0 + 1), (a0 + 1, a0 + 2, a0 + 3)]
        if tip <= 0.3:
            a0 = base + 2 * (segs - 1)
            tris.append((a0, a0 + 2, a0 + 1))
        for t in tris:
            idx.extend(t); idx.extend((t[0], t[2], t[1]))
    if n_blades == 0:
        # Empty last level: one zero-area triangle.  X-Plane cross-fades mesh
        # into mesh almost invisibly, but the last mesh into the billboard
        # showed as a band; ending on an empty mesh makes the draw-distance
        # edge a mesh-to-mesh fade too.
        u, v = cell_uv(block, 0.5, 0.5)
        verts = [(0.0, 0.0, 0.0, u, v, (0.0, 0.0, 0.0))] * 3
        idx = [0, 1, 2]
    return verts, idx


BILLBOARD_TEX = "bb_empty.png"


def write_empty_billboard(gdir):
    """Fully transparent texture for the mandatory billboard LOD step, the
    same size as a blade texture so the TREE rectangle stays valid."""
    w = TEX_W * (len(TIERS) if args.tint_tiers else 1)
    Image.fromarray(np.zeros((TEX_H, w, 4), np.uint8), "RGBA").save(os.path.join(gdir, BILLBOARD_TEX))


def write_for(path, tex_name, k):
    variants = [("a", 0.35, 34), ("b", 0.50, 33), ("c", 0.70, 33)]
    out = ["A", "1200", "FOREST", "",
           # The billboard X-Plane insists on as every tree's last LOD step gets
           # its own fully transparent texture.  With the blade texture it was a
           # camera-facing card, lit far brighter than up-normal blades, that the
           # last mesh cross-faded into: a whitish band at the draw-distance edge.
           # Matte (SUPER_ROUGHNESS 1.0) as Laminar's own grass.
           "SHADER_2D", f"\tTEXTURE {BILLBOARD_TEX}", "\tSUPER_ROUGHNESS 1.0", "",
           "SHADER_3D", f"\tTEXTURE {tex_name}", "\tSUPER_ROUGHNESS 1.0", "",
           f"LOD\t{LOD:g}", f"SCALE_X\t{TEX_W * (len(TIERS) if args.tint_tiers else 1)}", f"SCALE_Y\t{TEX_H}",
           f"SPACING\t{args.spacing:g} {args.spacing:g}",
           f"RANDOM\t{args.spacing * 0.5:g} {args.spacing * 0.5:g}", ""]
    flag = "" if args.shadows else "\tNO_SHADOW"
    # Layers: X-Plane plants one tree per layer per grid cell (Laminar's grass
    # has tree percentages summing to 100 within each layer), so a near-only
    # second layer doubles tuft density close up without touching SPACING.
    # Its geometry sits half a cell off the grid point in case both layers
    # share placement positions.
    layers = [dict(layer=0, tiers=list(TIERS), seed=0, offset=(0.0, 0.0),
                   contrast=CONTRAST, wind=WIND, tip=TIP, height=HEIGHT)]
    if args.near_layer:
        parts = args.near_layer.split(":")
        nfar, nfade, nblades = parts[:3]
        copies = int(parts[3]) if len(parts) > 3 else 1
        half = args.spacing * 0.5
        # One extra layer per copy, each with its own tufts and its own
        # half-cell offset, so they fill the gaps between the base layer's tufts.
        offsets = [(half, half)] if copies == 1 else [(half, 0.0), (0.0, half), (half, half)][:copies]
        for c, off in enumerate(offsets):
            layers.append(dict(layer=1 + c, tiers=[(float(nfar), int(nblades), 3, 1.0), (float(nfade), 0, 1, 1.0)],
                               seed=500 * (c + 1), offset=off, contrast=[CONTRAST[0], 0.0],
                               wind=[WIND[0], 0.0], tip=[TIP[0], TIP[0]], height=[HEIGHT[0], 1.0]))
    for ly in layers:
        for vi, (name, height, _) in enumerate(variants):
            near = 0.0
            for ti, (far, blades, segs, wscale) in enumerate(ly["tiers"]):
                verts, idx = tuft(1000 * k + vi + ly["seed"], ti if args.tint_tiers else 0,
                                  height * ly["height"][ti], blades, segs, wscale,
                                  contrast=ly["contrast"][ti], tip=ly["tip"][ti], offset=ly["offset"])
                wind = ly["wind"][ti]
                out.append(f"MESH\tg{ly['layer']}_{name}{ti}\t{near:g}\t{far:g}\t{len(verts)}\t{len(idx)}"
                           f"\t{1.0 * wind:g}\t{0.15 * wind:g}\t10.0{flag}")
                for x, y, z, u, v, (w1, w2, w3) in verts:
                    out.append(f"VERTEX\t{x:.4f} {y:.4f} {z:.4f}\t0 1 0\t{u:.5f} {v:.5f}\t{w1:.3f} {w2:.3f} {w3:.3f}")
                for i in range(0, len(idx), 10):
                    out.append("IDX\t" + " ".join(map(str, idx[i:i + 10])))
                near = far
    out.append("")
    for ly in layers:
        for name, height, freq in variants:
            out.append(f"TREE\t0\t0\t{TEX_W}\t{TEX_H}\t{TEX_W // 2}\t{freq}\t{height}\t{height}"
                       f"\t1\t{ly['layer']}\tgrass{ly['layer']}_{name}")
            for ti in range(len(ly["tiers"])):
                out.append(f"MESH_3D\tg{ly['layer']}_{name}{ti}")
    out += ["", "SKIP_SURFACE water", ""]
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(out))


if args.only_for:
    pack = os.path.join(args.out, PACK_NAME)
    info_path = os.path.join(pack, "spike_info.json")
    info = json.load(open(info_path))
    centres = np.array(info["palette_lab"], np.float32)
    gdirs = [d for d in glob.glob(os.path.join(pack, "grass", "*")) if os.path.isdir(d)]
    n = 0
    for gdir in gdirs:
        write_empty_billboard(gdir)
        for old in glob.glob(os.path.join(gdir, "c*.for")):
            k = int(os.path.basename(old)[1:3])
            build_texture(centres[k], os.path.join(gdir, f"c{k:02d}.png"))
            write_for(old, f"c{k:02d}.png", k)
            n += 1
    info["geometry"] = {"tiers": TIERS, "px_mrad": args.px_mrad, "tier_contrast": args.tier_contrast,
                        "tier_wind": args.tier_wind, "tier_tip": args.tier_tip, "tier_height": args.tier_height,
                        "near_layer": args.near_layer, "tint_tiers": args.tint_tiers}
    json.dump(info, open(info_path, "w"), indent=1)
    print("rewrote", n, ".for files + textures in", pack)
    sys.exit(0)


# ── texture sources ──────────────────────────────────────────────────────────
# --tile-dir: the installed Ortho4XP tile, textures chosen by the overlays' own
# cfg replay (O4_Texture_Selection) so we read exactly what X-Plane draws,
# with the parts of a footprint that another texture owns masked out.
# Otherwise: review-set / extra JPEGs named with --textures.
sys.path.insert(0, r"G:\Dev\Ortho4XP-SegFormer\src")
lat0, lon0 = args.center
radius_m = args.radius_nm * 1852.0
cos0 = math.cos(math.radians(lat0))


def footprint_bbox(til_y, til_x, zl):
    def g2w(tx, ty):
        rx = tx / (2 ** (zl - 1)) - 1; ry = 1 - ty / (2 ** (zl - 1))
        return 360 / math.pi * math.atan(math.exp(math.pi * ry)) - 90, rx * 180
    lat_n, lon_w = g2w(til_x, til_y)
    lat_s, lon_e = g2w(til_x + 16, til_y + 16)
    return [lat_s, lon_w, lat_n, lon_e]


def bbox_hits_circle(b):
    lat_s, lon_w, lat_n, lon_e = b
    cy = min(max(lat0, lat_s), lat_n); cx = min(max(lon0, lon_w), lon_e)
    return ((cy - lat0) * 110574) ** 2 + ((cx - lon0) * 111320 * cos0) ** 2 <= radius_m ** 2


sources = []      # (key, path, bbox, zl, excluded_fracs)
if args.tile_dir:
    import O4_Texture_Selection as TEXSEL
    tex_dir = os.path.join(args.tile_dir, "textures")
    tlat, tlon = math.floor(lat0), math.floor(lon0)
    files, excluded, rep = TEXSEL.select_textures(os.listdir(tex_dir), tex_dir, tlat, tlon)
    print(f"texture selection: {rep.get('selected')} selected, {rep.get('skipped')} skipped, "
          f"{rep.get('partial')} partial; notes={rep.get('notes')}")
    for f in files:
        m = TEXSEL.TEXTURE_NAME_RE.match(f)
        til_y, til_x, zl = int(m.group(1)), int(m.group(2)), int(m.group(4))
        b = footprint_bbox(til_y, til_x, zl)
        if bbox_hits_circle(b):
            sources.append((f[:-4], os.path.join(tex_dir, f), b, zl, excluded.get(f, ())))
else:
    samples = {}
    for d in SETS:
        p = os.path.join(d, "samples.json")
        if os.path.isfile(p):
            for s in json.load(open(p)):
                samples[s["file"]] = dict(s, dir=d)
    for t in args.textures:
        s = samples[f"{t}_BI16.jpg"]
        sources.append((s["file"][:-4], os.path.join(s["dir"], s["file"]), s["bbox"], 16, ()))
sources.sort(key=lambda s: s[3])      # low ZL first: mosaic lets higher ZL paint over
print(f"{len(sources)} textures touch the {args.radius_nm} nm circle:",
      {zl: sum(1 for s in sources if s[3] == zl) for zl in sorted({s[3] for s in sources})})

# Mosaic canvas for the preview: Web-Mercator-linear, ~MOSAIC_M m/px, over the circle.
MOSAIC_M = 2.0
dlat = radius_m / 110574; dlon = radius_m / (111320 * cos0)
mz_n, mz_s = merc_y(lat0 + dlat), merc_y(lat0 - dlat)
mlon_w, mlon_e = lon0 - dlon, lon0 + dlon
MW = int(round(2 * radius_m / MOSAIC_M)); MH = MW
mosaic_img = np.zeros((MH, MW, 3), np.uint8)
mosaic_lbl = np.full((MH, MW), -1, np.int16)
mosaic_excl = np.zeros((MH, MW), np.uint8)


def mosaic_rect(b):
    lat_s, lon_w, lat_n, lon_e = b
    x0 = (lon_w - mlon_w) / (mlon_e - mlon_w) * MW; x1 = (lon_e - mlon_w) / (mlon_e - mlon_w) * MW
    y0 = (mz_n - merc_y(lat_n)) / (mz_n - mz_s) * MH; y1 = (mz_n - merc_y(lat_s)) / (mz_n - mz_s) * MH
    return x0, y0, x1, y1


def paste(canvas, tile_arr, b, interp, keep_mask=None):
    """Paste a texture-sized array into the mosaic at its footprint."""
    x0, y0, x1, y1 = mosaic_rect(b)
    ix0, iy0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
    ix1, iy1 = min(MW, int(math.ceil(x1))), min(MH, int(math.ceil(y1)))
    if ix1 <= ix0 or iy1 <= iy0:
        return
    h, w = tile_arr.shape[:2]
    xs = ((np.arange(ix0, ix1) + 0.5 - x0) / (x1 - x0) * w).astype(int).clip(0, w - 1)
    ys = ((np.arange(iy0, iy1) + 0.5 - y0) / (y1 - y0) * h).astype(int).clip(0, h - 1)
    sub = tile_arr[ys][:, xs]
    if keep_mask is None:
        canvas[iy0:iy1, ix0:ix1] = sub
    else:
        km = keep_mask[ys][:, xs]
        region = canvas[iy0:iy1, ix0:ix1]
        region[km] = sub[km]


def pavement_mask(shapes, bbox, h, w, mpp):
    """Rasterise apt.dat pavement (outer rings filled, holes cleared) onto one
    texture's Web-Mercator pixel grid, grown by --pave-buffer-m."""
    lat_s, lon_w, lat_n, lon_e = bbox
    ym_n, ym_s = merc_y(lat_n), merc_y(lat_s)
    mask = np.zeros((h, w), np.uint8)

    def px(ring):
        return np.array([[(lon - lon_w) / (lon_e - lon_w) * w,
                          (ym_n - merc_y(lat)) / (ym_n - ym_s) * h] for lon, lat in ring],
                        np.float64).round().astype(np.int32)

    for outer, holes in shapes:
        o = px(outer)
        if o[:, 0].max() < 0 or o[:, 0].min() >= w or o[:, 1].max() < 0 or o[:, 1].min() >= h:
            continue
        one = np.zeros((h, w), np.uint8)
        cv2.fillPoly(one, [o], 1)
        if holes:
            cv2.fillPoly(one, [px(r) for r in holes], 0)
        mask |= one
    if mask.any() and args.pave_buffer_m > 0:
        r = max(1, int(round(args.pave_buffer_m / mpp)))
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return mask.astype(bool)


# Half-widths (m) per simHeaven network type: the carriageway X-Plane draws
# plus about a metre -- the same table the vegetation overlay uses for trees.
# Untyped 'network' segments are power lines and cableways: not excluded.
ROAD_HALF_WIDTH_M = {
    'motorway': 13, 'trunk': 10, 'motorway_link': 6,
    'primary': 8, 'secondary': 6, 'residential': 4, 'service': 3, 'rail': 4,
}


def road_mask(roads, bbox, h, w):
    """Rasterise road/rail polylines at their type's width onto one texture's
    Web-Mercator pixel grid."""
    lat_s, lon_w, lat_n, lon_e = bbox
    ym_n, ym_s = merc_y(lat_n), merc_y(lat_s)
    mpp_x = (lon_e - lon_w) * 111320 * math.cos(math.radians(0.5 * (lat_n + lat_s))) / w
    mask = np.zeros((h, w), np.uint8)
    for road in roads:
        lats = [p[0] for p in road["pts"]]; lons = [p[1] for p in road["pts"]]
        if max(lons) < lon_w or min(lons) > lon_e or max(lats) < lat_s or min(lats) > lat_n:
            continue
        half = ROAD_HALF_WIDTH_M[road["type"]] + args.road_extra_m
        pts = np.array([[(lo - lon_w) / (lon_e - lon_w) * w, (ym_n - merc_y(la)) / (ym_n - ym_s) * h]
                        for la, lo in road["pts"]]).round().astype(np.int32)
        cv2.polylines(mask, [pts], False, 1, thickness=max(1, int(round(2 * half / mpp_x))))
    return mask.astype(bool)


def green_mask(L, mpp):
    """Clearly green pixels: Lab hue 95-170 deg, chroma >= --green-chroma, not black."""
    Ls = cv2.GaussianBlur(L, (0, 0), max(0.5, 1.0 / mpp))
    C = np.hypot(Ls[..., 1], Ls[..., 2])
    hue = np.degrees(np.arctan2(Ls[..., 2], Ls[..., 1])) % 360
    return (hue >= 95) & (hue <= 170) & (C >= args.green_chroma) & (Ls[..., 0] >= 20)


def colour_ground_mask(L, mpp):
    """Programmatic 'not obviously built' mask from the ortho alone.

    Lab on a ~1 m blur: grey low-chroma (asphalt, concrete, grey roofs),
    very bright (white roofs, apron), and roof hues (red/orange tiles below
    60 deg, blue metal above 200 deg) are built; everything else -- grass,
    canopy, soil, shadow -- counts as ground that can carry grass.
    """
    Ls = cv2.GaussianBlur(L, (0, 0), max(0.5, 1.0 / mpp))
    l, a, b = Ls[..., 0], Ls[..., 1], Ls[..., 2]
    C = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a)) % 360
    built = ((C < 8) & (l > 25)) | (l > 72) | ((C > 14) & ((hue < 60) | (hue > 200)))
    r = max(1, int(round(1.5 / mpp)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    built = cv2.morphologyEx(built.astype(np.uint8), cv2.MORPH_CLOSE, ker)
    ground = cv2.morphologyEx(1 - built, cv2.MORPH_OPEN, ker)   # drop slivers between buildings
    return ground.astype(bool)


# ── load textures, class maps, circle ────────────────────────────────────────
PAVEMENT = []
if args.airports == "auto" and args.tile_dir:
    import apt_pavement
    xplane_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.tile_dir)))
    # Parsing every apt.dat (the global one is ~380 MB) takes ~20 s; cache the
    # result against scenery_packs.ini and the global file's size and mtime.
    import pickle
    gapt = os.path.join(xplane_dir, "Global Scenery", "Global Airports", "Earth nav data", "apt.dat")
    ini = os.path.join(xplane_dir, "Custom Scenery", "scenery_packs.ini")
    stamp = tuple((os.path.getsize(p), int(os.path.getmtime(p))) if os.path.exists(p) else None for p in (ini, gapt))
    ckey = f"{lat0:.4f}_{lon0:.4f}_{radius_m:.0f}"
    cpath = os.path.join(SPIKE, "_apt_cache", ckey + ".pkl")
    cached = pickle.load(open(cpath, "rb")) if os.path.isfile(cpath) else None
    if cached and cached["stamp"] == stamp:
        PAVEMENT, used = cached["shapes"], cached["used"]
    else:
        PAVEMENT, used = apt_pavement.pavement_near(xplane_dir, lat0, lon0, radius_m)
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        pickle.dump({"stamp": stamp, "shapes": PAVEMENT, "used": used}, open(cpath, "wb"))
    print("airport pavement:", used or "none in the circle", flush=True)

if args.mask == "fused":
    import O4_Climate_Regions as CLIM

ROADS = []
if args.roads == "auto" and args.tile_dir:
    import O4_Building_Overlay as BLD
    cs_dir = os.path.dirname(os.path.abspath(args.tile_dir))
    dsf_cache = os.path.join(SPIKE, "_dsf_cache")
    os.makedirs(dsf_cache, exist_ok=True)
    nets = BLD._load_simheaven_network(cs_dir, math.floor(lat0), math.floor(lon0),
                                       os.path.join(os.path.dirname(DSFTOOL), "DSFTool.exe"), dsf_cache)
    ROADS = [r for r in nets if r.get("type") in ROAD_HALF_WIDTH_M]
    counts = {}
    for r in ROADS:
        counts[r["type"]] = counts.get(r["type"], 0) + 1
    print("road/rail segments kept:", counts, flush=True)

# ── per texture, pass 1: masks at working resolution, colour on a coarse grid ─
# Grass edges need about a metre; the colour is smoothed over --sigma-m (8 m)
# anyway.  So the mask runs at --fine-m (textures finer than that are area-
# downsampled first: ZL18 /2, ZL19 /4) and the colour field, the palette
# assignment and the clean-up run on a --coarse-m grid.  Both passes run on a
# thread pool: OpenCV and the heavy numpy calls release the GIL, and threads
# avoid Windows' spawn re-running this script in every worker.
import concurrent.futures as CF
import threading
import time

GPU_LOCK = threading.Lock()
_MODEL = {}
T_START = time.time()


def classes_for(key, img):
    cls_path = os.path.join(SPIKE, "classmaps_tile" if args.tile_dir else "classmaps", key + "_cls.npy")
    if not os.path.isfile(cls_path):
        with GPU_LOCK:                    # one model, one GPU: inference stays serial
            if not os.path.isfile(cls_path):
                import O4_Landcover_Inference as SEG
                if not _MODEL:
                    _MODEL["m"] = SEG.load_vegetation_model()
                model, proc, dev = _MODEL["m"]
                os.makedirs(os.path.dirname(cls_path), exist_ok=True)
                np.save(cls_path, SEG.run_inference(model, dev, img, proc).astype(np.uint8))
    return np.load(cls_path)


def block_sum(a, c, hc, wc):
    """Sum c x c blocks of a (h, w[, ch]) array padded to (hc*c, wc*c)."""
    h, w = a.shape[:2]
    pad = [(0, hc * c - h), (0, wc * c - w)] + [(0, 0)] * (a.ndim - 2)
    a = np.pad(a, pad)
    return a.reshape(hc, c, wc, c, *a.shape[2:]).sum(axis=(1, 3))


def pass1(src):
    key, path, bbox, zl, excl = src
    t0 = time.time()
    try:
        img = np.asarray(Image.open(path).convert("RGB"))
    except Exception as exc:              # H: has bad sectors: skip, don't abort
        print(f"SKIP {key}: {exc}", flush=True)
        return None
    cls = classes_for(key, img) if args.mask in ("segformer", "fused") else None
    lat_s, lon_w, lat_n, lon_e = bbox
    mpp0 = (lon_e - lon_w) * 111320 * cos0 / img.shape[1]
    f = 2 ** max(0, int(round(math.log2(max(1.0, args.fine_m / mpp0)))))
    if f > 1:
        img = cv2.resize(img, (img.shape[1] // f, img.shape[0] // f), interpolation=cv2.INTER_AREA)
        if cls is not None:
            cls = cls[f // 2::f, f // 2::f][:img.shape[0], :img.shape[1]]
    h, w = img.shape[:2]
    mpp = mpp0 * f
    ym_n, ym_s = merc_y(lat_n), merc_y(lat_s)
    lons = lon_w + (np.arange(w) + 0.5) / w * (lon_e - lon_w)
    yy = ym_n + (np.arange(h) + 0.5) / h * (ym_s - ym_n)
    lats = np.degrees(2 * np.arctan(np.exp(yy)) - np.pi / 2)
    inside = (((lats - lat0) * 110574)[:, None] ** 2 + (((lons - lon0) * 111320 * cos0)[None, :]) ** 2) <= radius_m ** 2
    if not inside.any():
        return None
    owned = np.ones((h, w), bool)
    for fx0, fy0, fx1, fy1 in excl:       # footprint parts another texture owns
        owned[int(round(fy0 * h)):int(round(fy1 * h)), int(round(fx0 * w)):int(round(fx1 * w))] = False
    L = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    sparse = np.zeros((h, w), bool)
    climate = ""
    if args.mask == "colour":
        grass = colour_ground_mask(L, mpp)
    elif args.mask == "fused":
        # SegFormer decides vegetation vs bare/built ground (it tells straw
        # grass from sand), the colour rule vetoes built-looking pixels and
        # rescues clearly green ones SegFormer put in a non-vegetation class,
        # and the climate regime -- static, so it cannot go out of date --
        # sets how much the ortho must show: in deserts only green pixels,
        # in steppe non-green vegetation is sparse.  Every parcel-scale
        # decision comes from the ortho on screen.
        ground = colour_ground_mask(L, mpp)
        green = green_mask(L, mpp)
        veg = np.isin(cls, (2, 5, 7))                 # rangeland, tree, agriculture
        soft = np.isin(cls, (0, 1, 3, 255))           # background, bareland, developed, none
        grass = ground & (veg | (soft & green))
        climate = CLIM.koppen_code(0.5 * (lat_n + lat_s), 0.5 * (lon_w + lon_e)) or ""
        if climate.startswith("BW"):
            grass &= green
        elif climate.startswith("BS"):
            sparse = grass & ~green
    else:
        grass = cv2.morphologyEx((cls == RANGELAND).astype(np.uint8), cv2.MORPH_OPEN,
                                 np.ones((3, 3), np.uint8)).astype(bool)
    if PAVEMENT:
        # Off the pavement before the colour field, so concrete never tints it.
        grass &= ~pavement_mask(PAVEMENT, bbox, h, w, mpp)
    if ROADS:
        grass &= ~road_mask(ROADS, bbox, h, w)
    sel = grass & inside & owned

    # Coarse colour field: grass-weighted block means, then a normalised
    # Gaussian over grass only (smoothed before the circle/ownership cuts, so
    # no edge bias).
    c = max(1, int(round(args.coarse_m / mpp)))
    hc, wc = -(-h // c), -(-w // c)
    gm = grass.astype(np.float32)
    lsum = block_sum(L * gm[..., None], c, hc, wc)
    wsum = block_sum(gm, c, hc, wc)
    sig = args.sigma_m / (mpp * c)
    wb = cv2.GaussianBlur(wsum, (0, 0), sig)
    S_c = np.stack([cv2.GaussianBlur(lsum[..., i], (0, 0), sig) for i in range(3)], -1) / np.maximum(wb, 1e-6)[..., None]
    sel_c = block_sum(sel.astype(np.uint8), c, hc, wc) > 0
    prev = cv2.resize(img, (wc, hc), interpolation=cv2.INTER_AREA)
    excl_c = block_sum((inside & ~grass).astype(np.float32), c, hc, wc) > 0.5 * c * c
    owned_c = block_sum(owned.astype(np.uint8), c, hc, wc) > 0
    out = dict(key=key, zl=zl, bbox=bbox, mpp=mpp, h=h, w=w, c=c, hc=hc, wc=wc, ym=(ym_n, ym_s),
               climate=climate, sel=np.packbits(sel), sparse=np.packbits(sparse & sel),
               S_c=S_c.astype(np.float32), sel_c=sel_c, prev=prev, excl_c=excl_c, owned_c=owned_c,
               grass_km2=float(sel.sum()) * mpp * mpp / 1e6,
               sparse_km2=float((sparse & sel).sum()) * mpp * mpp / 1e6)
    print(f"{key} ZL{zl} {climate} /{f} {w}x{h} coarse {wc}x{hc}: grass {out['grass_km2']:.3f} km2"
          f" (sparse {out['sparse_km2']:.3f})  {time.time() - t0:.1f}s", flush=True)
    return out


def unpack(t, name):
    return np.unpackbits(t[name])[:t["h"] * t["w"]].reshape(t["h"], t["w"]).astype(bool)


WORKERS = args.workers or min(8, os.cpu_count() or 4)
with CF.ThreadPoolExecutor(WORKERS) as ex:
    tex = [t for t in ex.map(pass1, sources) if t is not None]
T_PASS1 = time.time()
for t in tex:                              # mosaic preview, low ZL first so higher ZL paints over
    paste(mosaic_img, t["prev"], t["bbox"], None, keep_mask=t["owned_c"])
    paste(mosaic_excl, t["excl_c"].astype(np.uint8), t["bbox"], None, keep_mask=t["owned_c"])

# ── shared palette ───────────────────────────────────────────────────────────
pool = np.concatenate([t["S_c"][t["sel_c"]] for t in tex])
rng = np.random.default_rng(0)
samp = pool[rng.choice(len(pool), min(300000, len(pool)), replace=False)].astype(np.float32)
cv2.setRNGSeed(0)
_, _, centres = cv2.kmeans(samp, args.k, None,
                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.05),
                           3, cv2.KMEANS_PP_CENTERS)
# Colour and fused modes decide where grass grows before the palette: no colour veto.
keep = list(range(args.k)) if args.mask in ("colour", "fused") else \
    [k for k, cc in enumerate(centres) if is_grass_colour(*cc)]
KEEP = set(keep)

pack = os.path.join(args.out, PACK_NAME)
if os.path.isdir(pack):
    shutil.rmtree(pack)
gdir = os.path.join(pack, "grass", args.name)
os.makedirs(gdir)
write_empty_billboard(gdir)
for k in keep:
    build_texture(centres[k], os.path.join(gdir, f"c{k:02d}.png"))
    write_for(os.path.join(gdir, f"c{k:02d}.for"), f"c{k:02d}.png", k)


# ── per texture, pass 2: labels -> polygons ──────────────────────────────────
def pass2(t):
    sel_c = t["sel_c"]
    lbl_c = np.full(sel_c.shape, -1, np.int32)
    lbl_c[sel_c] = assign(t["S_c"][sel_c], centres)
    win = int(round(args.win_m / (t["mpp"] * t["c"])))
    if win >= 2:
        lbl_c = mode_filter(lbl_c, sel_c, args.k, win | 1)
    c, h, w = t["c"], t["h"], t["w"]
    sel = unpack(t, "sel")
    sparse = unpack(t, "sparse")
    lbl = np.repeat(np.repeat(lbl_c, c, 0), c, 1)[:h, :w]
    lat_s, lon_w, lat_n, lon_e = t["bbox"]; ym_n, ym_s = t["ym"]
    min_px = args.min_m2 / t["mpp"] ** 2
    eps = max(0.5, args.simplify_m / t["mpp"])   # outline simplification
    sparse_den = max(1, min(255, int(round(args.sparse_density * 255))))
    xs_scale, ys_span = (lon_e - lon_w) / w, (ym_s - ym_n) / h

    def to_ll(r):
        p = r.reshape(-1, 2).astype(np.float64)
        lo = lon_w + p[:, 0] * xs_scale
        la = np.degrees(2 * np.arctan(np.exp(ym_n + p[:, 1] * ys_span)) - np.pi / 2)
        return list(zip(lo.tolist(), la.tolist()))

    polys = []
    for k in np.unique(lbl_c[lbl_c >= 0]).tolist():
        if k not in KEEP:
            continue
        ys, xs = np.nonzero(lbl_c == k)            # coarse bbox -> crop the fine grid
        y0, y1 = ys.min() * c, min(h, (ys.max() + 1) * c)
        x0, x1 = xs.min() * c, min(w, (xs.max() + 1) * c)
        lk = (lbl[y0:y1, x0:x1] == k) & sel[y0:y1, x0:x1]
        sp = sparse[y0:y1, x0:x1]
        for den, part in ((255, lk & ~sp), (sparse_den, lk & sp)):
            if not part.any():
                continue
            cnts, hier = cv2.findContours(part.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE,
                                          offset=(int(x0), int(y0)))
            for j, cn in enumerate(cnts):
                if hier[0][j][3] >= 0 or cv2.contourArea(cn) < min_px:
                    continue
                rings = [cv2.approxPolyDP(cn, eps, True)]
                ch = hier[0][j][2]
                while ch >= 0:
                    if cv2.contourArea(cnts[ch]) >= min_px:
                        rings.append(cv2.approxPolyDP(cnts[ch], eps, True))
                    ch = hier[0][ch][0]
                rings = [to_ll(r) for r in rings if len(r) >= 3]
                if rings:
                    polys.append((f"grass/{args.name}/c{k:02d}.for", rings, den))
    planted = np.where(np.isin(lbl_c, keep) & sel_c, lbl_c, -1).astype(np.int16)
    rep = {"texture": t["key"], "zl": t["zl"], "climate": t["climate"], "polygons": len(polys),
           "sparse_km2": round(t["sparse_km2"], 3), "grass_km2_in_circle": round(t["grass_km2"], 3)}
    return (math.floor(lat_s), math.floor(lon_w)), polys, planted, rep


tiles, report = {}, []
with CF.ThreadPoolExecutor(WORKERS) as ex:
    results = list(ex.map(pass2, tex))
for t, (tile, polys, planted, rep) in zip(tex, results):
    tiles.setdefault(tile, []).extend(polys)
    paste(mosaic_lbl, planted, t["bbox"], None, keep_mask=t["owned_c"])
    report.append(rep)
T_PASS2 = time.time()
print(f"timing: pass1 {T_PASS1 - T_START:.1f}s, palette+pass2 {T_PASS2 - T_PASS1:.1f}s, workers {WORKERS}", flush=True)

# Mosaic preview: ortho | ortho with planted grass in its palette colour, circle drawn.
rgbc = lab2rgb(centres)
filled = mosaic_img.copy(); g = mosaic_lbl >= 0
filled[g] = rgbc[mosaic_lbl[g]]
outline = mosaic_img.copy()
if args.mask in ("colour", "fused"):
    ex = mosaic_excl > 0
    outline[ex] = (0.45 * outline[ex] + 0.55 * np.array([255, 0, 64])).astype(np.uint8)
else:
    cnts, _ = cv2.findContours(g.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(outline, cnts, -1, (255, 0, 255), 2)
for im in (outline, filled):
    cv2.circle(im, (MW // 2, MH // 2), MW // 2 - 2, (255, 255, 0), 3)
pv = np.hstack([outline, filled])
sc = min(1.0, 3200 / pv.shape[1])
Image.fromarray(cv2.resize(pv, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)).save(
    os.path.join(SPIKE, f"{args.name}_mosaic_preview.jpg"), quality=88)

for (lat, lon), polys in tiles.items():
    defs = sorted({p for p, _, _ in polys}); di = {p: i for i, p in enumerate(defs)}
    folder = f"{math.floor(lat / 10) * 10:+03d}{math.floor(lon / 10) * 10:+04d}"
    name = f"{lat:+03d}{lon:+04d}"
    os.makedirs(os.path.join(pack, "Earth nav data", folder), exist_ok=True)
    txt = os.path.join(args.out, name + ".txt")
    with open(txt, "w", newline="\n") as f:
        f.write("PROPERTY sim/planet earth\nPROPERTY sim/overlay 1\nPROPERTY sim/creation_agent O4XP_grass_spike\n")
        f.write(f"PROPERTY sim/west {lon}\nPROPERTY sim/east {lon + 1}\nPROPERTY sim/south {lat}\nPROPERTY sim/north {lat + 1}\n\n")
        for p in defs:
            f.write(f"POLYGON_DEF {p}\n")
        f.write("\n")
        for p, rings, den in polys:
            f.write(f"BEGIN_POLYGON {di[p]} {den} 2\n")
            for r in rings:
                f.write("BEGIN_WINDING\n")
                for lo, la in r + [r[0]]:
                    f.write(f"POLYGON_POINT {lo:.8f} {la:.8f}\n")
                f.write("END_WINDING\n")
            f.write("END_POLYGON\n")
    dsf = os.path.join(pack, "Earth nav data", folder, name + ".dsf")
    r = subprocess.run([DSFTOOL, "-text2dsf", txt, dsf], capture_output=True, text=True)
    print("DSFTool", name, "rc", r.returncode)

tri = {}
for far, blades, segs, ws in TIERS:
    tri[f"<= {far:g} m"] = blades * (2 * (segs - 1) + 1)   # front-face triangles per tuft
info = {"params": vars(args), "tiers": TIERS, "tris_per_tuft": tri,
        "colours": args.k, "colours_kept": len(keep), "textures": report,
        "palette_lab": centres.round(2).tolist()}
json.dump(info, open(os.path.join(pack, "spike_info.json"), "w"), indent=1)
print(json.dumps({k: info[k] for k in ("tiers", "tris_per_tuft", "colours_kept", "textures")}, indent=1))
