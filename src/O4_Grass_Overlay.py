"""3-d grass around airports, coloured from the ortho underneath.

Runs inside the vegetation overlay (O4_Vegetation_Overlay.run): GrassHook
collects the SegFormer class maps the overlay already has for the textures an
airport zone reaches, then builds the grass and hands its polygons back, so
they go into the same DSF as the trees -- one DSFTool compile, one validation,
and the overlay's tile-wide forest exclusion (which applies only to lower
packs) can never hide them.

Where grass grows, per pixel:
  * SegFormer vegetation (rangeland, tree, agriculture) unless the ortho looks
    built (grey, very bright or roof-coloured), plus clearly green pixels that
    SegFormer put in a non-vegetation class;
  * the Koppen regime decides how much the ortho must show -- desert: green
    only; steppe: non-green vegetation becomes sparse grass;
  * never on apt.dat pavement (runways with blast pads, helipads, taxiways,
    aprons, shoulders) or on the simHeaven road/rail network;
  * only inside airport zones: each airport's apt.dat boundary, or without
    one a circle to its furthest pavement point, grown by a buffer.

Colour: one k-means palette per tile over a 4 m grid of the ortho's Lab,
smoothed over grass only; each palette colour is its own .for with its own
small texture (a shared atlas bled neighbouring colours into the blades at
coarse mips).  Geometry: 24-blade tufts to 110 m, 10 wider blades to 220 m,
then an empty mesh to 230 m, so the draw-distance edge is a mesh-to-mesh fade;
the mandatory billboard is fully transparent.

Per-texture work runs in a process pool (the overlay subprocess is started
with ``python -c``, so spawned workers import only this module).
"""

import concurrent.futures as CF
import io
import math
import os
import shutil
import time
import zlib
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

import O4_Apt_Pavement as APT

VEG_CLASSES = (2, 5, 7)                 # rangeland, tree, agriculture
SOFT_CLASSES = (0, 1, 3, 255)           # background, bareland, developed, no class (-1 as uint8)
MAX_RING_PTS = 8000                     # same per-winding cap as the vegetation overlay
MAX_HOLES = 253                         # DSF: a polygon holds at most 254 windings
TEX_W, TEX_H = 16, 64
BILLBOARD_TEX = "bb_empty.png"

# Half-widths (m) per simHeaven network type: the carriageway X-Plane draws
# plus about a metre -- the vegetation overlay's table for trees.  Untyped
# 'network' segments are power lines and cableways: not excluded.
ROAD_HALF_WIDTH_M = {
    'motorway': 13, 'trunk': 10, 'motorway_link': 6,
    'primary': 8, 'secondary': 6, 'residential': 4, 'service': 3, 'rail': 4,
}


@dataclass
class GrassSettings:
    airport_buffer_m: float = 500.0     # zone = boundary (or reach circle) grown by this
    sigma_m: float = 8.0                # colour smoothing over grass
    win_m: float = 9.0                  # colour clean-up (majority) window
    min_m2: float = 50.0                # smallest grass patch
    simplify_m: float = 3.0             # outline tolerance
    fine_m: float = 1.2                 # mask resolution (finer textures are downsampled)
    coarse_m: float = 4.0               # colour grid
    k: int = 64                         # palette size
    green_chroma: float = 10.0
    sparse_density: float = 0.25        # DSF density of sparse (steppe) grass
    pave_buffer_m: float = 2.0
    workers: int = 0                    # 0 = max(2, min(8, cores - 4))


@dataclass
class Geometry:
    """Detail levels: (far_m, blades, segments, width_scale); a 0-blade level
    is an empty mesh."""
    tiers: list = field(default_factory=lambda: [(110.0, 24, 3, 1.0), (220.0, 10, 2, 2.2), (230.0, 0, 1, 1.0)])
    contrast: list = field(default_factory=lambda: [1.0, 0.6, 0.0])
    wind: list = field(default_factory=lambda: [1.0, 1.0, 0.0])
    tip: list = field(default_factory=lambda: [0.25, 0.25, 0.25])
    height: list = field(default_factory=lambda: [1.0, 1.0, 1.0])
    spacing: float = 1.0
    shadows: bool = False


# ── projection ───────────────────────────────────────────────────────────────
def merc_y(lat):
    return np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def inv_merc(y):
    return np.degrees(2 * np.arctan(np.exp(y)) - np.pi / 2)


def footprint_bbox(til_y, til_x, zl):
    """(lat_s, lon_w, lat_n, lon_e) of an Ortho4XP 16x16-tile texture."""
    def g2w(tx, ty):
        rx = tx / (2 ** (zl - 1)) - 1
        ry = 1 - ty / (2 ** (zl - 1))
        return 360 / math.pi * math.atan(math.exp(math.pi * ry)) - 90, rx * 180
    lat_n, lon_w = g2w(til_x, til_y)
    lat_s, lon_e = g2w(til_x + 16, til_y + 16)
    return [lat_s, lon_w, lat_n, lon_e]


def _to_px(lons, lats, bbox, h, w):
    lat_s, lon_w, lat_n, lon_e = bbox
    ym_n, ym_s = merc_y(lat_n), merc_y(lat_s)
    x = (np.asarray(lons) - lon_w) / (lon_e - lon_w) * w
    y = (ym_n - merc_y(np.asarray(lats))) / (ym_n - ym_s) * h
    return np.stack([x, y], -1).round().astype(np.int32)


def lab2rgb(lab):
    lab = np.asarray(lab, np.float32).reshape(-1, 1, 3)
    return (np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1) * 255).round().astype(np.uint8).reshape(-1, 3)


# ── where grass grows ────────────────────────────────────────────────────────
def _lab_blur(L, mpp):
    """The ortho's Lab on a ~1 m blur, as (l, a, b) planes."""
    Ls = cv2.GaussianBlur(L, (0, 0), max(0.5, 1.0 / mpp))
    return Ls[..., 0], Ls[..., 1], Ls[..., 2]


def _hue_between(a, b, lo_deg, hi_deg):
    """Lab hue in [lo, hi] (span under 180 deg) without arctan: the colour
    vector lies counter-clockwise of the lo direction and clockwise of hi."""
    lo, hi = math.radians(lo_deg), math.radians(hi_deg)
    return ((math.cos(lo) * b - math.sin(lo) * a) >= 0) & ((a * math.sin(hi) - b * math.cos(hi)) >= 0)


def colour_ground_mask(L, mpp, lab=None):
    """'Not obviously built' from the ortho alone: grey low-chroma, very bright
    and roof-coloured (red/orange < 60 deg, blue > 200 deg) pixels are built."""
    l, a, b = lab if lab is not None else _lab_blur(L, mpp)
    c2 = a * a + b * b
    built = ((c2 < 64) & (l > 25)) | (l > 72) | ((c2 > 196) & ~_hue_between(a, b, 60, 200))
    r = max(1, int(round(1.5 / mpp)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    built = cv2.morphologyEx(built.astype(np.uint8), cv2.MORPH_CLOSE, ker)
    return cv2.morphologyEx(1 - built, cv2.MORPH_OPEN, ker).astype(bool)


def green_mask(L, mpp, chroma=10.0, lab=None):
    """Clearly green: Lab hue 95-170 deg, chroma >= chroma, not black."""
    l, a, b = lab if lab is not None else _lab_blur(L, mpp)
    return _hue_between(a, b, 95, 170) & ((a * a + b * b) >= chroma * chroma) & (l >= 20)


_VEG_LUT = np.zeros(256, bool)
_VEG_LUT[list(VEG_CLASSES)] = True
_SOFT_LUT = np.zeros(256, bool)
_SOFT_LUT[list(SOFT_CLASSES)] = True


def fused_grass(L, cls, mpp, climate, green_chroma=10.0):
    """Returns (grass, sparse) masks; cls is the uint8 SegFormer class map."""
    lab = _lab_blur(L, mpp)
    ground = colour_ground_mask(L, mpp, lab)
    green = green_mask(L, mpp, green_chroma, lab)
    del lab
    grass = ground & (_VEG_LUT[cls] | (_SOFT_LUT[cls] & green))
    sparse = np.zeros_like(grass)
    if climate.startswith("BW"):
        grass &= green
    elif climate.startswith("BS"):
        sparse = grass & ~green
    return grass, sparse


# ── colour field ─────────────────────────────────────────────────────────────
def block_mean(a, c, hc, wc):
    """Mean of c x c blocks (zero-padded to hc*c x wc*c), via OpenCV area
    resampling (exact for an integer factor).  float32; up to 4 channels."""
    a = np.asarray(a)
    if a.dtype != np.float32:
        a = a.astype(np.float32)
    h, w = a.shape[:2]
    if hc * c != h or wc * c != w:
        a = cv2.copyMakeBorder(a, 0, hc * c - h, 0, wc * c - w, cv2.BORDER_CONSTANT, value=0)
    if c == 1:
        return a
    return cv2.resize(a, (wc, hc), interpolation=cv2.INTER_AREA)


def coarse_colour(L, grass, c, hc, wc, sigma_px):
    """Grass-weighted block means of Lab, then a normalised Gaussian over grass."""
    gm = grass.astype(np.float32)
    lsum, wsum = block_mean(L * gm[..., None], c, hc, wc), block_mean(gm, c, hc, wc)
    wb = cv2.GaussianBlur(wsum, (0, 0), sigma_px)
    return cv2.GaussianBlur(lsum, (0, 0), sigma_px) / np.maximum(wb, 1e-6)[..., None]


def assign(px, centres, chunk=400000):
    """Nearest palette colour via |x|^2 - 2 x.c + |c|^2, in row chunks."""
    c = np.asarray(centres, np.float32)
    cc = (c * c).sum(1)
    out = np.empty(len(px), np.int32)
    for i in range(0, len(px), chunk):
        out[i:i + chunk] = np.argmin(cc[None, :] - 2.0 * (px[i:i + chunk] @ c.T), axis=1)
    return out


def mode_filter(labels, mask, win):
    """Majority label in a win x win window, counting on 8/16-bit images."""
    best_n = np.zeros(labels.shape, np.uint16)
    out = labels.copy()
    depth = cv2.CV_16U if win * win > 255 else -1
    one = np.empty(labels.shape, bool)
    gt = np.empty(labels.shape, bool)
    n16 = np.empty(labels.shape, np.uint16)
    for i in np.unique(labels[labels >= 0]).tolist():
        np.equal(labels, i, out=one)
        one &= mask
        n = cv2.boxFilter(one.view(np.uint8), depth, (win, win), normalize=False, borderType=cv2.BORDER_CONSTANT)
        if n.dtype == np.uint16:
            n16 = n
        else:
            np.copyto(n16, n)
        np.greater(n16, best_n, out=gt)
        np.copyto(out, i, where=gt)
        np.maximum(best_n, n16, out=best_n)
    out[~mask] = -1
    return out


# ── exclusions ───────────────────────────────────────────────────────────────
def pavement_mask(pavement, bbox, h, w, mpp, buffer_m=2.0):
    """apt.dat pavement (outer rings filled, holes cleared), grown by buffer_m."""
    lat_s, lon_w, lat_n, lon_e = bbox
    mask = np.zeros((h, w), np.uint8)
    for outer, holes in pavement:
        o = np.asarray(outer)
        if o[:, 0].max() < lon_w or o[:, 0].min() > lon_e or o[:, 1].max() < lat_s or o[:, 1].min() > lat_n:
            continue
        one = np.zeros((h, w), np.uint8)
        cv2.fillPoly(one, [_to_px(o[:, 0], o[:, 1], bbox, h, w)], 1)
        if holes:
            cv2.fillPoly(one, [_to_px(np.asarray(r)[:, 0], np.asarray(r)[:, 1], bbox, h, w) for r in holes], 0)
        mask |= one
    if mask.any() and buffer_m > 0:
        r = max(1, int(round(buffer_m / mpp)))
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return mask.astype(bool)


class RoadIndex:
    """simHeaven road/rail polylines with bounding boxes, so a texture only
    touches the roads that overlap it."""

    def __init__(self, roads):
        self.pts = [np.asarray(r["pts"], np.float64) for r in roads if r.get("type") in ROAD_HALF_WIDTH_M]
        self.half = np.array([ROAD_HALF_WIDTH_M[r["type"]] for r in roads if r.get("type") in ROAD_HALF_WIDTH_M])
        if self.pts:
            allp = np.concatenate(self.pts)
            starts = np.cumsum([0] + [len(p) for p in self.pts[:-1]])
            self.bb = np.stack([np.minimum.reduceat(allp[:, 0], starts), np.minimum.reduceat(allp[:, 1], starts),
                                np.maximum.reduceat(allp[:, 0], starts), np.maximum.reduceat(allp[:, 1], starts)], 1)
        else:
            self.bb = np.zeros((0, 4))

    def mask(self, bbox, h, w):
        lat_s, lon_w, lat_n, lon_e = bbox
        out = np.zeros((h, w), np.uint8)
        if len(self.bb):
            hit = np.nonzero((self.bb[:, 2] >= lat_s) & (self.bb[:, 0] <= lat_n) &
                             (self.bb[:, 3] >= lon_w) & (self.bb[:, 1] <= lon_e))[0]
            mpp_x = (lon_e - lon_w) * 111320 * math.cos(math.radians(0.5 * (lat_n + lat_s))) / w
            for i in hit.tolist():
                p = self.pts[i]
                cv2.polylines(out, [_to_px(p[:, 1], p[:, 0], bbox, h, w)], False, 1,
                              thickness=max(1, int(round(2 * self.half[i] / mpp_x))))
        return out.astype(bool)


def airport_zone(airports, bbox, hc, wc, mpp_c, buffer_m):
    """Airport zones on a texture's coarse grid: each boundary (or reach circle)
    grown by buffer_m.  Rasterised on a grid padded by the buffer so an airport
    just outside the texture still reaches in."""
    m = int(math.ceil(buffer_m / mpp_c)) + 2
    lat_s, lon_w, lat_n, lon_e = bbox
    seed = np.zeros((hc + 2 * m, wc + 2 * m), np.uint8)
    near_deg = (buffer_m + 20000.0) / 111000.0
    hit = False
    for a in airports:
        if a.lat < lat_s - near_deg or a.lat > lat_n + near_deg or a.lon < lon_w - near_deg or a.lon > lon_e + near_deg:
            continue
        hit = True
        if a.boundaries:
            for outer, _ in a.boundaries:
                o = np.asarray(outer)
                cv2.fillPoly(seed, [_to_px(o[:, 0], o[:, 1], bbox, hc, wc) + m], 1)
        else:
            c = _to_px(np.array([a.lon]), np.array([a.lat]), bbox, hc, wc)[0] + m
            cv2.circle(seed, (int(c[0]), int(c[1])), int(round(a.reach_m / mpp_c)), 1, -1)
    if not hit:
        return np.zeros((hc, wc), bool)
    dist = cv2.distanceTransform((1 - seed).astype(np.uint8), cv2.DIST_L2, 5) * mpp_c
    return (dist <= buffer_m)[m:m + hc, m:m + wc]


def zone_box(a, buffer_m):
    """(lat_s, lon_w, lat_n, lon_e) around an airport zone."""
    pad = buffer_m
    if a.boundaries:
        pts = np.array([p for outer, _ in a.boundaries for p in outer])
        lo0, lo1, la0, la1 = pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()
    else:
        lo0 = lo1 = a.lon
        la0 = la1 = a.lat
        pad += a.reach_m
    dla = pad / 110574.0
    dlo = pad / (111320.0 * math.cos(math.radians(a.lat)))
    return la0 - dla, lo0 - dlo, la1 + dla, lo1 + dlo


# ── X-Plane 12 forest files ──────────────────────────────────────────────────
def build_texture(lab_colour, path):
    """Opaque root->tip ramp centred on one palette colour (row 0 = tip)."""
    L, a, b = lab_colour
    yy, xx = np.mgrid[0:TEX_H, 0:TEX_W].astype(np.float32)
    up = 1.0 - yy / (TEX_H - 1)
    side = xx / (TEX_W - 1) * 2 - 1
    lab = np.stack([L - 7 + 14 * up + 4 * side, a + 0.75 - 1.5 * up, b - 1.5 + 3 * up], -1)
    Image.fromarray(lab2rgb(lab.reshape(-1, 3)).reshape(TEX_H, TEX_W, 3), "RGB").save(path)


def write_empty_billboard(gdir):
    """X-Plane ends every tree on a billboard; a camera-facing card with the
    blade texture lit brighter than the blades (a whitish band at the edge)."""
    Image.fromarray(np.zeros((TEX_H, TEX_W, 4), np.uint8), "RGBA").save(os.path.join(gdir, BILLBOARD_TEX))


def _cell_uv(fx, fy):
    return (2 + fx * (TEX_W - 4)) / TEX_W, 1 - (2 + (1 - fy) * (TEX_H - 4)) / TEX_H


def _tuft(seed, height, n_blades, segs, width_scale, contrast, tip):
    rng = np.random.default_rng(seed)
    blades = [dict(ang=rng.uniform(0, 2 * math.pi), r0=rng.uniform(0, 0.18),
                   lean=rng.uniform(0.15, 0.45) * height, face=rng.uniform(0, 2 * math.pi),
                   h=height * rng.uniform(0.6, 1.0), w=rng.uniform(0.008, 0.014),
                   fx=rng.uniform(0, 1), phase=rng.uniform(0, 1)) for _ in range(max(24, n_blades))]
    verts, idx = [], []
    for bl in blades[:n_blades]:
        x0, z0 = bl["r0"] * math.cos(bl["ang"]), bl["r0"] * math.sin(bl["ang"])
        w = bl["w"] * width_scale
        dx, dz = math.cos(bl["face"]) * w, math.sin(bl["face"]) * w
        lx, lz = math.cos(bl["ang"]), math.sin(bl["ang"])
        base = len(verts)
        for s in range(segs + 1):
            f = s / segs
            cx, cz = x0 + lx * bl["lean"] * f * f, z0 + lz * bl["lean"] * f * f
            cy = bl["h"] * f * (1 - 0.2 * f)
            u, v = _cell_uv(0.5 + (bl["fx"] - 0.5) * contrast, 0.5 + (f - 0.5) * contrast)
            wgt = (f * f, 0.5 * f, bl["phase"])
            t = 1 - (1 - tip) * f
            if s < segs or tip > 0.3:
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
        for t in tris:                      # both windings: the 3-d shader culls back faces
            idx.extend(t)
            idx.extend((t[0], t[2], t[1]))
    if n_blades == 0:                       # empty mesh: one zero-area triangle
        u, v = _cell_uv(0.5, 0.5)
        verts = [(0.0, 0.0, 0.0, u, v, (0.0, 0.0, 0.0))] * 3
        idx = [0, 1, 2]
    return verts, idx


def write_for(path, tex_name, k, geo=None):
    """One palette colour's forest: up-facing normals (lit like the ground),
    matte, no shadows, empty last mesh, transparent billboard."""
    geo = geo or Geometry()
    variants = [("a", 0.35, 34), ("b", 0.50, 33), ("c", 0.70, 33)]
    out = ["A", "1200", "FOREST", "",
           "SHADER_2D", f"\tTEXTURE {BILLBOARD_TEX}", "\tSUPER_ROUGHNESS 1.0", "",
           "SHADER_3D", f"\tTEXTURE {tex_name}", "\tSUPER_ROUGHNESS 1.0", "",
           f"LOD\t{geo.tiers[-1][0]:g}", f"SCALE_X\t{TEX_W}", f"SCALE_Y\t{TEX_H}",
           f"SPACING\t{geo.spacing:g} {geo.spacing:g}",
           f"RANDOM\t{geo.spacing * 0.5:g} {geo.spacing * 0.5:g}", ""]
    flag = "" if geo.shadows else "\tNO_SHADOW"
    for vi, (name, height, _) in enumerate(variants):
        near = 0.0
        for ti, (far, blades, segs, wscale) in enumerate(geo.tiers):
            verts, idx = _tuft(1000 * k + vi, height * geo.height[ti], blades, segs, wscale,
                               geo.contrast[ti], geo.tip[ti])
            wind = geo.wind[ti]
            out.append(f"MESH\tg0_{name}{ti}\t{near:g}\t{far:g}\t{len(verts)}\t{len(idx)}"
                       f"\t{1.0 * wind:g}\t{0.15 * wind:g}\t10.0{flag}")
            for x, y, z, u, v, (w1, w2, w3) in verts:
                out.append(f"VERTEX\t{x:.4f} {y:.4f} {z:.4f}\t0 1 0\t{u:.5f} {v:.5f}\t{w1:.3f} {w2:.3f} {w3:.3f}")
            for i in range(0, len(idx), 10):
                out.append("IDX\t" + " ".join(map(str, idx[i:i + 10])))
            near = far
    out.append("")
    for name, height, freq in variants:
        out.append(f"TREE\t0\t0\t{TEX_W}\t{TEX_H}\t{TEX_W // 2}\t{freq}\t{height}\t{height}\t1\t0\tgrass0_{name}")
        for ti in range(len(geo.tiers)):
            out.append(f"MESH_3D\tg0_{name}{ti}")
    out += ["", "SKIP_SURFACE water", ""]
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(out))


# ── worker processes ─────────────────────────────────────────────────────────
_W = {}


def _worker_init(cfg):
    _W.clear()
    _W.update(cfg)
    _W["roads"] = RoadIndex(cfg.pop("roads_raw"))
    _W.pop("roads_raw", None)
    cv2.setNumThreads(2)


def _climate(lat, lon):
    import O4_Climate_Regions as CLIM
    return CLIM.koppen_code(lat, lon) or ""


def _pass1(src):
    """Masks and the colour field of one texture (kept in memory, packed)."""
    key, path, bbox, excl, cls_path = src
    s = _W["settings"]
    try:
        img = np.asarray(Image.open(path).convert("RGB"))
        cls = np.load(cls_path)
    except Exception as exc:              # unreadable disk sectors etc.: skip, don't abort
        return {"skip": f"{key}: {exc}"}
    if cls.shape[:2] != img.shape[:2]:
        cls = cv2.resize(cls, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    tlat, tlon = _W["lat"], _W["lon"]
    lat_s, lon_w, lat_n, lon_e = bbox
    mpp0 = (lon_e - lon_w) * 111320 * math.cos(math.radians(0.5 * (lat_s + lat_n))) / img.shape[1]
    f = 2 ** max(0, int(round(math.log2(max(1.0, s.fine_m / mpp0)))))
    if f > 1:
        img = cv2.resize(img, (img.shape[1] // f, img.shape[0] // f), interpolation=cv2.INTER_AREA)
        cls = cls[f // 2::f, f // 2::f][:img.shape[0], :img.shape[1]]
    h, w = img.shape[:2]
    mpp = mpp0 * f
    ym_n, ym_s = merc_y(lat_n), merc_y(lat_s)
    lons = lon_w + (np.arange(w) + 0.5) / w * (lon_e - lon_w)
    lats = inv_merc(ym_n + (np.arange(h) + 0.5) / h * (ym_s - ym_n))
    inside = ((lats >= tlat) & (lats < tlat + 1))[:, None] & ((lons >= tlon) & (lons < tlon + 1))[None, :]
    owned = np.ones((h, w), bool)
    for fx0, fy0, fx1, fy1 in excl:       # parts of the footprint another texture owns
        owned[int(round(fy0 * h)):int(round(fy1 * h)), int(round(fx0 * w)):int(round(fx1 * w))] = False
    c = max(1, int(round(s.coarse_m / mpp)))
    hc, wc = -(-h // c), -(-w // c)
    zone_c = airport_zone(_W["airports"], bbox, hc, wc, mpp * c, s.airport_buffer_m)
    if not zone_c.any():
        return {"skip": None}
    climate = _climate(0.5 * (lat_n + lat_s), 0.5 * (lon_w + lon_e))
    L = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    grass, sparse = fused_grass(L, cls, mpp, climate, s.green_chroma)
    del cls
    if _W["pavement"]:
        grass &= ~pavement_mask(_W["pavement"], bbox, h, w, mpp, s.pave_buffer_m)
    grass &= ~_W["roads"].mask(bbox, h, w)
    sel = grass & inside & owned & np.repeat(np.repeat(zone_c, c, 0), c, 1)[:h, :w]
    sparse &= sel
    if not sel.any():
        return {"skip": None}
    S_c = coarse_colour(L, grass, c, hc, wc, s.sigma_m / (mpp * c))
    del L
    sel_c = block_mean(sel, c, hc, wc) > 0
    vals = S_c[sel_c].astype(np.float16)
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    sample = vals[rng.choice(len(vals), min(4000, len(vals)), replace=False)].astype(np.float32)
    meta = dict(key=key, bbox=bbox, mpp=mpp, h=h, w=w, c=c, hc=hc, wc=wc, climate=climate,
                grass_km2=float(sel.sum()) * mpp * mpp / 1e6, sparse_km2=float(sparse.sum()) * mpp * mpp / 1e6)
    arrays = dict(sel=np.packbits(sel), sparse=np.packbits(sparse), S=vals, sel_c=np.packbits(sel_c))
    return {"meta": meta, "arrays": arrays, "sample": sample}


def _pass2(job):
    """Palette labels -> DSF polygon text for one texture."""
    t, z, centres, def_offset = job
    s = _W["settings"]
    tlat, tlon = _W["lat"], _W["lon"]
    h, w, c, hc, wc = t["h"], t["w"], t["c"], t["hc"], t["wc"]

    def unpack(a, n, shape):
        return np.unpackbits(a)[:n].reshape(shape).astype(bool)

    sel = unpack(z["sel"], h * w, (h, w))
    sparse = unpack(z["sparse"], h * w, (h, w))
    sel_c = unpack(z["sel_c"], hc * wc, (hc, wc))
    lbl_c = np.full((hc, wc), -1, np.int32)
    lbl_c[sel_c] = assign(z["S"].astype(np.float32), centres)
    win = int(round(s.win_m / (t["mpp"] * c)))
    if win >= 2:
        lbl_c = mode_filter(lbl_c, sel_c, win | 1)
    lat_s, lon_w, lat_n, lon_e = t["bbox"]
    ym_n, ym_s = merc_y(lat_n), merc_y(lat_s)
    eps = max(0.5, s.simplify_m / t["mpp"])
    min_px = s.min_m2 / t["mpp"] ** 2
    min_cells = s.min_m2 / (t["mpp"] * c) ** 2
    sparse_den = max(1, min(255, int(round(s.sparse_density * 255))))
    lon_scale, y_scale = (lon_e - lon_w) / w, (ym_s - ym_n) / h
    out = io.StringIO()
    n_poly = n_pts = 0

    def winding(r):
        p = r.reshape(-1, 2).astype(np.float64)
        lo = np.clip(lon_w + p[:, 0] * lon_scale, tlon, tlon + 1)
        la = np.clip(inv_merc(ym_n + p[:, 1] * y_scale), tlat, tlat + 1)
        pts = [f"POLYGON_POINT {a:.7f} {b:.7f}" for a, b in zip(lo.tolist(), la.tolist())]
        pts.append(pts[0])
        return "BEGIN_WINDING\n" + "\n".join(pts) + "\nEND_WINDING\n", len(pts)

    for k in np.unique(lbl_c[lbl_c >= 0]).tolist():
        # Each colour is scattered over the texture: split it into connected
        # pieces on the coarse grid and work at full resolution only inside
        # each piece's crop.
        n_cc, cc, stats, _ = cv2.connectedComponentsWithStats((lbl_c == k).astype(np.uint8), connectivity=8)
        for i in range(1, n_cc):
            cx0, cy0, cw, chh, cells = stats[i]
            if cells < min_cells:
                continue
            y0, y1 = cy0 * c, min(h, (cy0 + chh) * c)
            x0, x1 = cx0 * c, min(w, (cx0 + cw) * c)
            piece = cc[cy0:cy0 + chh, cx0:cx0 + cw] == i
            lk = np.repeat(np.repeat(piece, c, 0), c, 1)[:y1 - y0, :x1 - x0] & sel[y0:y1, x0:x1]
            sp = sparse[y0:y1, x0:x1]
            for den, part in ((255, lk & ~sp), (sparse_den, lk & sp)):
                if not part.any():
                    continue
                cnts, hier = cv2.findContours(part.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE,
                                              offset=(int(x0), int(y0)))
                for j, cn in enumerate(cnts):
                    if hier[0][j][3] >= 0 or cv2.contourArea(cn) < min_px:
                        continue
                    holes = []
                    chn = hier[0][j][2]
                    while chn >= 0:
                        a = cv2.contourArea(cnts[chn])
                        if a >= min_px:
                            holes.append((a, cnts[chn]))
                        chn = hier[0][chn][0]
                    holes.sort(key=lambda x: -x[0])
                    body = []
                    for r in [cn] + [hh for _, hh in holes[:MAX_HOLES]]:
                        e = eps
                        simple = cv2.approxPolyDP(r, e, True)
                        while len(simple) > MAX_RING_PTS:
                            e *= 2
                            simple = cv2.approxPolyDP(r, e, True)
                        if len(simple) >= 3:
                            txt, n = winding(simple)
                            body.append(txt)
                            n_pts += n
                    if body:
                        out.write(f"BEGIN_POLYGON {def_offset + k} {den} 2\n" + "".join(body) + "END_POLYGON\n")
                        n_poly += 1
    return {"text": out.getvalue(), "n_poly": n_poly, "n_pts": n_pts}


# ── the hook the vegetation overlay calls ────────────────────────────────────
def tile_name(lat, lon):
    lat, lon = int(math.floor(lat)), int(math.floor(lon))
    return f"{lat:+03d}{lon:+04d}"


def remove_tile_assets(pack_dir, lat, lon):
    """Delete a tile's grass forest files (rebuild, or grass turned off)."""
    shutil.rmtree(os.path.join(pack_dir, "grass", tile_name(lat, lon)), ignore_errors=True)


class GrassHook:
    """Collects class maps during the vegetation overlay, then builds grass.

    wants(fname) / take_class_map(fname, cls): called per texture while the
    overlay has its class map.  build(def_offset): grass polygons as DSF text
    with POLYGON_DEF indices starting at def_offset; returns
    (polygon_def_paths, text, stats) or None.  cleanup(): drop the class maps.
    """

    def __init__(self, tex_dir, lat, lon, cache_dir, custom_scenery_dir, pack_dir, dsftool_path,
                 settings=None, log=print):
        import O4_Texture_Selection as TEXSEL
        self.lat, self.lon = int(math.floor(lat)), int(math.floor(lon))
        self.tex_dir, self.cache_dir, self.pack_dir = tex_dir, cache_dir, pack_dir
        self.custom_scenery_dir, self.dsftool_path = custom_scenery_dir, dsftool_path
        self.settings = settings or GrassSettings()
        self.log = log
        self.sources = {}
        self.class_maps = {}
        self.airports = []
        t0 = time.time()
        if not custom_scenery_dir:
            log("[Grass] no Custom Scenery folder: no airports to read, grass skipped")
            return
        xplane_dir = os.path.dirname(os.path.abspath(custom_scenery_dir))
        self.airports = APT.airports_for_tile(xplane_dir, self.lat, self.lon, cache_dir, log=log)
        if not self.airports:
            log(f"[Grass] no airports in the tile ({time.time() - t0:.1f}s)")
            return
        boxes = [zone_box(a, self.settings.airport_buffer_m) for a in self.airports]
        files, excluded, _ = TEXSEL.select_textures(os.listdir(tex_dir), tex_dir, self.lat, self.lon)
        for f in files:
            m = TEXSEL.TEXTURE_NAME_RE.match(f)
            if not m:
                continue
            bbox = footprint_bbox(int(m.group(1)), int(m.group(2)), int(m.group(4)))
            if any(b[0] <= bbox[2] and b[2] >= bbox[0] and b[1] <= bbox[3] and b[3] >= bbox[1] for b in boxes):
                self.sources[f] = (bbox, excluded.get(f, ()))
        log(f"[Grass] {len(self.airports)} airports ("
            + ", ".join(f"{a.icao}: {'boundary' if a.boundaries else f'{a.reach_m:.0f} m reach'}"
                        for a in self.airports)
            + f"), {len(self.sources)} textures in their zones ({time.time() - t0:.1f}s)")

    def wants(self, fname):
        return fname in self.sources and fname not in self.class_maps

    def take_class_map(self, fname, cls):
        if not self.wants(fname) or cls is None:
            return
        path = os.path.join(self.cache_dir, fname[:-4] + "_grasscls.npy")
        np.save(path, np.asarray(cls).astype(np.uint8))      # -1 (no class) becomes 255
        self.class_maps[fname] = path

    def cleanup(self):
        for path in self.class_maps.values():
            try:
                os.remove(path)
            except OSError:
                pass
        self.class_maps = {}

    def build(self, def_offset):
        if not self.class_maps:
            remove_tile_assets(self.pack_dir, self.lat, self.lon)
            return None
        t0 = time.time()
        s = self.settings
        missing = len(self.sources) - len(self.class_maps)
        if missing:
            self.log(f"[Grass] {missing} zone textures had no class map from the overlay; skipped")
        import O4_Building_Overlay as BLD
        roads = BLD._load_simheaven_network(self.custom_scenery_dir, self.lat, self.lon,
                                            self.dsftool_path, self.cache_dir) if self.dsftool_path else []
        pavement = [p for a in self.airports for p in a.pavement]
        cfg = dict(lat=self.lat, lon=self.lon, settings=s, airports=self.airports, pavement=pavement,
                   roads_raw=[r for r in roads if r.get("type") in ROAD_HALF_WIDTH_M])
        workers = s.workers or max(2, min(8, (os.cpu_count() or 4) - 4))
        # One BLAS/OpenMP thread per worker: many processes each starting a
        # thread per core oversubscribe the CPU.  Spawned workers inherit it.
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        tasks = [(f[:-4], os.path.join(self.tex_dir, f), self.sources[f][0], self.sources[f][1], path)
                 for f, path in sorted(self.class_maps.items())]
        with CF.ProcessPoolExecutor(workers, initializer=_worker_init, initargs=(cfg,)) as pool:
            results = []
            for r in pool.map(_pass1, tasks):
                if r.get("skip"):
                    self.log(f"[Grass] skipped {r['skip']}")
                elif "meta" in r:
                    results.append(r)
            if not results:
                remove_tile_assets(self.pack_dir, self.lat, self.lon)
                self.log(f"[Grass] no grass in the airport zones ({time.time() - t0:.1f}s)")
                return None
            results.sort(key=lambda r: r["meta"]["key"])     # reproducible palette sample
            samples = np.concatenate([r["sample"] for r in results])
            samp = samples[np.random.default_rng(0).choice(len(samples), min(300000, len(samples)), replace=False)]
            k = min(s.k, len(samp))
            cv2.setRNGSeed(0)
            _, _, centres = cv2.kmeans(samp, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.05),
                                       3, cv2.KMEANS_PP_CENTERS)
            jobs = [(r["meta"], r["arrays"], centres, def_offset) for r in results]
            parts = list(pool.map(_pass2, jobs))
        name = tile_name(self.lat, self.lon)
        remove_tile_assets(self.pack_dir, self.lat, self.lon)
        gdir = os.path.join(self.pack_dir, "grass", name)
        os.makedirs(gdir, exist_ok=True)
        write_empty_billboard(gdir)
        geo = Geometry()
        defs = []
        for i in range(k):
            build_texture(centres[i], os.path.join(gdir, f"c{i:02d}.png"))
            write_for(os.path.join(gdir, f"c{i:02d}.for"), f"c{i:02d}.png", i, geo)
            defs.append(f"grass/{name}/c{i:02d}.for")
        stats = dict(textures=len(results), polygons=sum(p["n_poly"] for p in parts),
                     points=sum(p["n_pts"] for p in parts),
                     grass_km2=round(sum(r["meta"]["grass_km2"] for r in results), 2),
                     sparse_km2=round(sum(r["meta"]["sparse_km2"] for r in results), 2),
                     airports=[a.icao for a in self.airports], seconds=round(time.time() - t0, 1))
        self.log(f"[Grass] {stats['polygons']} polygons, {stats['grass_km2']} km2 around "
                 f"{len(self.airports)} airports from {stats['textures']} textures ({stats['seconds']}s)")
        return defs, "".join(p["text"] for p in parts), stats
