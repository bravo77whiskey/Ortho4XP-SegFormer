"""Forest leaf colours matched to the ortho under the forest.

All of X-Plane 12's forests (``1200 forests``) draw their leaves from two
albedo atlases, ``trees_3D1_ALB`` (3-d trees) and ``trees_bb1_ALB``
(billboards), with ``_sp`` / ``_fa`` / ``_wi`` versions for the other seasons.
Their summer leaves sit at one colour everywhere (CIELAB a* -11, b* 16:
chroma 19.4, hue 124.5 deg), while the canopy in the imagery ranges over
hue ~115-160 deg and chroma ~11-22 (review hotspots, 2026-11-09): close to
X-Plane's leaves around Kaohsiung and Shouguang, ~25 deg greener and 1.3-1.5x
less saturated around Shenzhen. A .for file has no per-instance tint, so a
forest takes the colour of one of a fixed set of recoloured atlas copies
(BUCKETS: a hue step and a chroma scale applied to X-Plane's leaves), the
one nearest the canopy colour measured on the ortho around it.

The atlases are recoloured block by block in their DXT5 data: only the two
RGB565 endpoint colours of every 4x4 colour block change, so Laminar's own
mip chain and alpha (the leaf cut-outs) stay exactly as they are and nothing
is re-encoded. Only leaf colours move (a weight on the endpoint's hue keeps
bark, trunks and dry leaves); lightness is left alone, because the ortho's
lightness carries shadows and exposure that the sim adds itself.

Spring leaves get the same shift as summer leaves; autumn and winter keep
X-Plane's textures. A tinted copy's summer file also drops its SEASONAL
autumn blend (X-Plane's summer forests blend toward the autumn texture as the
season goes on: strongly autumnal by mid-July in the sim).
"""

from __future__ import annotations

import math
import os
import re
import struct

import numpy as np

SIGNATURE = "forest-tint-v1"
# Bump when the zone map changes (it only moves forest paths, not the textures).
ZONES_VERSION = 2

# X-Plane's leaf atlases, and the season suffixes recoloured (summer, spring).
ATLASES = ("trees_3D1_ALB", "trees_bb1_ALB")
TINTED_SUFFIXES = ("", "_sp")

# Hue steps (deg, + = greener / bluer) and chroma scales of the buckets. About
# 3 dE apart at the canopy's chroma: under that a step is barely visible.
HUE_STEPS_DEG = (-12.0, 0.0, 12.0, 24.0)
CHROMA_SCALES = (1.15, 1.0, 0.82, 0.66)
BUCKETS = tuple((dh, s) for dh in HUE_STEPS_DEG for s in CHROMA_SCALES)
STOCK = BUCKETS.index((0.0, 1.0))

# Median leaf colour (CIELAB a*, b*) of X-Plane's summer 3-d tree atlas
# (alpha > 200, green-dominant pixels), measured 2026-11-09.
REFERENCE_AB = (-11.0, 16.0)

# Canopy colour is averaged over this radius (Gaussian sigma, metres) of
# crown interiors before a bucket is picked; tint zones smaller than
# MIN_ZONE_M2 join their neighbour. Each colour in view costs ~14 MB of
# VRAM, and at 60 m / 0.5 ha a ZL16 texture took 8-13 colours for shading
# and haze noise; at 120 m / 2 ha it takes 3-9 (hotspots, 2026-11-09).
SIGMA_M = 120.0
MIN_ZONE_M2 = 20000.0

# Leaf weight by CIELAB hue of a colour: full between the inner limits.
_LEAF_HUE = (80.0, 100.0, 175.0, 195.0)

_TEX_RE = re.compile(r"^(\s*(?:TEXTURE|SEASONAL)\s+)\.\./textures/("
                     + "|".join(ATLASES) + r")(_sp)?(\.\w+)(\s*)$")
_SEASONAL_RE = re.compile(r"^\s*SEASONAL\s")


def bucket_ab(k):
    """CIELAB (a*, b*) of bucket ``k``'s summer leaves (X-Plane's median leaf, recoloured)."""
    dh, s = BUCKETS[k]
    a0, b0 = REFERENCE_AB
    c = math.hypot(a0, b0) * s
    h = math.atan2(b0, a0) + math.radians(dh)
    return c * math.cos(h), c * math.sin(h)


def texture_dir(k):
    """Folder (in the forest copies' pack) of bucket ``k``'s atlases."""
    return f"textures_t{k}"


# ── Picking a bucket from the ortho ──────────────────────────────────────────

def _lab_ab(rgb):
    import cv2
    lab = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]).astype(np.float32) / 255.0,
                       cv2.COLOR_RGB2LAB)
    return lab[..., 1], lab[..., 2]


def _merge_small(idx, valid, min_px):
    """Zones (values of ``idx`` on ``valid``) whose connected pieces are under ``min_px``
    take the zone of the nearest bigger piece."""
    import cv2
    small = np.zeros(idx.shape, bool)
    for z in np.unique(idx[valid]):
        n, lab, st, _ = cv2.connectedComponentsWithStats(((idx == z) & valid).astype(np.uint8), connectivity=8)
        tiny = np.nonzero(st[1:, cv2.CC_STAT_AREA] < min_px)[0] + 1
        if len(tiny):
            small |= np.isin(lab, tiny)
    keep = valid & ~small
    if small.any() and keep.any():
        _d, near = cv2.distanceTransformWithLabels(
            (~keep).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(keep)
        lut = np.zeros(int(near.max()) + 1, np.uint8)
        lut[near[ys, xs]] = idx[ys, xs]
        idx = idx.copy()
        idx[small] = lut[near[small]]
    return idx


# Grid the colour zones are worked out on, metres: the colour is averaged
# over SIGMA_M anyway, and at the texture's own resolution a ZL16 texture
# took ~3 s (2026-11-09).
GRID_M = 8.0


def tint_zones(rgb, tree_mask, m_per_px, sigma_m=SIGMA_M, min_zone_m2=MIN_ZONE_M2):
    """Per pixel of ``tree_mask``: 1 + the bucket nearest the canopy colour around it
    (crown interiors within ``sigma_m``), 0 off the mask; zones under ``min_zone_m2``
    joined to their neighbour. uint8, mask-sized."""
    import cv2
    on = np.asarray(tree_mask) > 0
    h, w = on.shape
    if rgb is None or not on.any():
        return None
    if rgb.shape[:2] != (h, w):
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    shrink = max(1, int(round(GRID_M / max(m_per_px, 1e-6))))
    sw, sh = max(1, w // shrink), max(1, h // shrink)
    cell_m = m_per_px * w / sw
    crowns = cv2.erode(on.astype(np.uint8), np.ones((5, 5), np.uint8))
    if crowns.sum() < 0.05 * on.sum():
        crowns = on.astype(np.uint8)   # thin rows only: their own pixels
    a, b = _lab_ab(rgb)
    cf = crowns.astype(np.float32)
    wgt = cv2.resize(cf, (sw, sh), interpolation=cv2.INTER_AREA)
    sa = cv2.resize(a * cf, (sw, sh), interpolation=cv2.INTER_AREA)
    sb = cv2.resize(b * cf, (sw, sh), interpolation=cv2.INTER_AREA)
    sig = sigma_m / cell_m
    wgt = cv2.GaussianBlur(wgt, (0, 0), sig)
    sa = cv2.GaussianBlur(sa, (0, 0), sig) / np.maximum(wgt, 1e-6)
    sb = cv2.GaussianBlur(sb, (0, 0), sig) / np.maximum(wgt, 1e-6)
    targets = np.array([bucket_ab(k) for k in range(len(BUCKETS))], np.float32)
    d = (sa[..., None] - targets[:, 0]) ** 2 + (sb[..., None] - targets[:, 1]) ** 2
    idx = np.argmin(d, axis=-1).astype(np.uint8)
    idx[wgt < 1e-4] = STOCK   # no canopy anywhere near: X-Plane's colour
    valid = cv2.resize(on.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_AREA) > 0
    if min_zone_m2:
        idx = _merge_small(idx, valid, min_zone_m2 / (cell_m * cell_m))
    idx = cv2.resize(idx, (w, h), interpolation=cv2.INTER_NEAREST)
    zones = np.where(on, idx + 1, 0).astype(np.uint8)
    if min_zone_m2:
        min_px = min_zone_m2 / (m_per_px * m_per_px)
        zones = _whole_small_patches(zones, on, min_px)
        zones = _absorb_small_pieces(zones, min_px)
    return zones


def _whole_small_patches(zones, on, min_px):
    """Tree patches under ``min_px`` take their majority zone whole. A zone edge
    through a small patch only cut it into pieces that became extra single-tree
    groups and tree rows (Kaohsiung tile: point groups +70 %, rows x6, 2026-11-09)."""
    import cv2
    n, lab, st, _ = cv2.connectedComponentsWithStats(on.astype(np.uint8), connectivity=8)
    if n <= 1:
        return zones
    small = st[:, cv2.CC_STAT_AREA] < min_px
    small[0] = False
    sel = small[lab]
    if not sel.any():
        return zones
    nz = len(BUCKETS) + 1
    counts = np.bincount(lab[sel].astype(np.int64) * nz + zones[sel], minlength=n * nz).reshape(n, nz)
    counts[:, 0] = 0
    major = counts.argmax(axis=1).astype(np.uint8)
    out = zones.copy()
    out[sel] = major[lab[sel]]
    return out


_NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _absorb_small_pieces(zones, min_px, rounds=2):
    """A piece of one zone under ``min_px`` that touches another zone takes the zone it
    shares the most border with, so a zone edge never leaves slivers of a big patch
    behind (they became single trees and tree rows of their own). Two rounds: extra
    forest pieces +29/+57 % -> +3/+6 % on szx_e/khh_n, ~0.6 s each (2026-11-09)."""
    import cv2
    nz = len(BUCKETS) + 1
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(rounds):
        on = zones > 0
        z255 = np.where(on, zones, 255).astype(np.uint8)
        edge = on & ((cv2.dilate(zones, k3) != zones) | (cv2.erode(z255, k3) != z255))
        ys, xs = np.nonzero(edge)
        if not len(ys):
            break
        pid = np.zeros(zones.shape, np.int32)
        sizes = [0]
        for z in np.unique(zones[ys, xs]):
            mz = (zones == z).astype(np.uint8)
            x0, y0, bw, bh = cv2.boundingRect(mz)
            n, lab, st, _ = cv2.connectedComponentsWithStats(mz[y0:y0 + bh, x0:x0 + bw], connectivity=8)
            if n > 1:
                sub = pid[y0:y0 + bh, x0:x0 + bw]
                m = lab > 0
                sub[m] = lab[m] + len(sizes) - 1
                sizes.extend(st[1:, cv2.CC_STAT_AREA].tolist())
        small = np.asarray(sizes) < min_px
        small[0] = False
        keep = small[pid[ys, xs]]
        ys, xs = ys[keep], xs[keep]
        if not len(ys):
            break
        pad_z = np.pad(zones, 1)
        own = zones[ys, xs]
        own_pid = pid[ys, xs].astype(np.int64)
        idx = []
        for dy, dx in _NEIGHBOURS:
            nb = pad_z[ys + 1 + dy, xs + 1 + dx]
            ok = (nb > 0) & (nb != own)
            idx.append(own_pid[ok] * nz + nb[ok])
        tally = np.bincount(np.concatenate(idx), minlength=len(sizes) * nz).reshape(len(sizes), nz)
        best = tally.argmax(axis=1).astype(np.uint8)
        moved = small & (tally.max(axis=1) > 0)
        if not moved.any():
            break
        m = moved[pid]
        zones = zones.copy()
        zones[m] = best[pid[m]]
    return zones


# ── Recoloured copies of X-Plane's atlases ───────────────────────────────────

def _leaf_weight(hue_deg):
    lo0, lo1, hi0, hi1 = _LEAF_HUE
    up = np.clip((hue_deg - lo0) / (lo1 - lo0), 0.0, 1.0)
    down = np.clip((hi1 - hue_deg) / (hi1 - hi0), 0.0, 1.0)
    return np.minimum(up, down)


def recolour_rgb(rgb, k):
    """Float RGB (0-255, N x 3) recoloured for bucket ``k``: leaf hues turned by its
    hue step and their chroma scaled, lightness kept."""
    import cv2
    dh, s = BUCKETS[k]
    rgb = np.asarray(rgb, np.float32).reshape(-1, 1, 3)
    if dh == 0.0 and s == 1.0:
        return rgb.reshape(-1, 3).copy()
    lab = cv2.cvtColor(rgb / 255.0, cv2.COLOR_RGB2LAB)
    a, b = lab[..., 1], lab[..., 2]
    hue = np.degrees(np.arctan2(b, a)) % 360.0
    w = _leaf_weight(hue)
    t = math.radians(dh)
    na = (a * math.cos(t) - b * math.sin(t)) * s
    nb = (a * math.sin(t) + b * math.cos(t)) * s
    lab[..., 1] = a + w * (na - a)
    lab[..., 2] = b + w * (nb - b)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) * 255.0
    return np.clip(out, 0.0, 255.0).reshape(-1, 3)


def _dxt5_levels(header):
    """(width, height, mip count) of a DXT5 .dds header, else ValueError."""
    if header[:4] != b"DDS " or len(header) < 128:
        raise ValueError("not a DDS file")
    h, w = struct.unpack("<II", header[12:20])
    mips = max(1, struct.unpack("<I", header[28:32])[0])
    if header[84:88] != b"DXT5":
        raise ValueError(f"not DXT5 ({header[84:88]!r})")
    return w, h, mips


def _block_count(w, h, mips):
    n = 0
    for _ in range(mips):
        n += max(1, (w + 3) // 4) * max(1, (h + 3) // 4)
        w, h = max(1, w // 2), max(1, h // 2)
    return n


def _unpack565(v):
    v = v.astype(np.uint32)
    r, g, b = (v >> 11) & 31, (v >> 5) & 63, v & 31
    return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], -1).astype(np.float32)


def _pack565(rgb):
    r = np.clip(np.rint(rgb[:, 0] * 31.0 / 255.0), 0, 31).astype(np.uint16)
    g = np.clip(np.rint(rgb[:, 1] * 63.0 / 255.0), 0, 63).astype(np.uint16)
    b = np.clip(np.rint(rgb[:, 2] * 31.0 / 255.0), 0, 31).astype(np.uint16)
    return (r << 11) | (g << 5) | b


def recolour_dxt5(data, k):
    """A DXT5 .dds (bytes) with every colour block's endpoints recoloured for bucket ``k``.

    DXT5 colour blocks always decode in 4-colour mode, so endpoint order does
    not matter and the indices and alpha blocks are kept as they are.
    """
    w, h, mips = _dxt5_levels(data[:128])
    n = _block_count(w, h, mips)
    if 128 + 16 * n > len(data):
        raise ValueError("DDS shorter than its mip chain")
    out = bytearray(data)
    blocks = np.frombuffer(out, dtype=np.uint8, count=16 * n, offset=128).reshape(n, 16)
    ends = blocks[:, 8:12].copy().view("<u2").reshape(n, 2)
    uniq, inv = np.unique(ends.reshape(-1), return_inverse=True)
    new = _pack565(recolour_rgb(_unpack565(uniq), k))[inv].reshape(n, 2)
    view = np.frombuffer(out, dtype=np.uint8, count=16 * n, offset=128).reshape(n, 16)
    view[:, 8:12] = new.astype("<u2").view(np.uint8).reshape(n, 4)
    return bytes(out)


def write_textures(xp_textures_dir, dst_dir, k):
    """Bucket ``k``'s atlases (summer and spring, 3-d and billboard) in ``dst_dir``.

    Kept when its version file matches; True when the folder is complete.
    """
    stamp = os.path.join(dst_dir, "o4xp_tint.txt")
    want = f"{SIGNATURE} {BUCKETS[k][0]:g} {BUCKETS[k][1]:g}\n"
    names = [f"{atlas}{suffix}.dds" for atlas in ATLASES for suffix in TINTED_SUFFIXES]
    try:
        with open(stamp, "r", encoding="utf-8") as f:
            if f.read() == want and all(os.path.isfile(os.path.join(dst_dir, n)) for n in names):
                return True
    except OSError:
        pass
    os.makedirs(dst_dir, exist_ok=True)
    for name in names:
        with open(os.path.join(xp_textures_dir, name), "rb") as f:
            data = recolour_dxt5(f.read(), k)
        tmp = os.path.join(dst_dir, name + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, os.path.join(dst_dir, name))
    with open(stamp, "w", encoding="utf-8") as f:
        f.write(want)
    return True


# ── .for files ───────────────────────────────────────────────────────────────

def tint_for_text(text, k, season):
    """A .for's text pointing its summer/spring leaf atlases at bucket ``k``'s copies;
    a summer file (``season`` "sum") also loses its SEASONAL autumn blend."""
    out = []
    for line in text.splitlines(keepends=True):
        if season == "sum" and _SEASONAL_RE.match(line):
            continue
        if k != STOCK:
            body = line.rstrip("\r\n")
            m = _TEX_RE.match(body)
            if m:
                line = (f"{m.group(1)}../{texture_dir(k)}/{m.group(2)}{m.group(3) or ''}"
                        f"{m.group(4)}{m.group(5)}" + line[len(body):])
        out.append(line)
    return "".join(out)
