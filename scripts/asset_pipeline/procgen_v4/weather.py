"""Post-bake pass: occlusion normalisation, glass, climate wear, packing.

Input is one strip's .npz from bake_blender.py; output is the strip's final
albedo (sRGB uint8) plus the X-Plane normal map (R,G = tangent normal,
B = base reflectance for NORMAL_METALNESS, A = gloss).

Wear is driven by the bake's own masks and by climate, never painted as a
uniform overlay:

walls
  * an EXPOSURE field along u (metres-scale) scales every layer, so the
    stretch of strip one building samples can be grimy while its
    neighbour's is fresh -- no two walls age alike;
  * run-off from above: water sheeting down from roof edges and upper
    floors, in threads of varying length (mould-green in humid climates);
  * sill streaks leaving from the sill ENDS, rust runs below bars, rails and
    AC brackets, grime settling in occluded reveals;
  * mould blooms near the ground and under sills, patchy repaint blocks on
    rendered walls, peeling paint exposing the render beneath, brick
    efflorescence, sun fade, soil splash-back on ground floors.
roofs
  * individual pieces faded or replaced (per-tile random from the bake),
    dirt collecting toward the eave, black algae streaks running down-slope,
    lichen rosettes, moss in the laps, rust runs on metal, dust.

Every noise field wraps in u, so strips still tile seamlessly.  All
regions weather, asia included (the old "asia stays clean" rule was
lifted by the user on 2026-09-25 for this library).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from strips import (CLS_GLASS, CLS_METAL, CLS_OBJECT, CLS_PLINTH, CLS_ROOF,
                    CLS_SHUTTER, CLS_SIGN, CLS_SILL, CLS_TRIM, CLS_WALL)

CLEAN_REGIONS = frozenset()

_BASE = dict(streak=0.55, splash=0.5, grime=0.45, lichen=0.4, moss=0.25,
             rust=0.35, dust=0.25, fade=0.35, humid=0.5, mould=0.35,
             rust_stain=0.4, peel=0.3, repaint=0.35, efflor=0.2, algae=0.4,
             fade_tiles=0.08, soil="#6b5d4c")

WEAR = {
    "generic": dict(_BASE),
    "europe": dict(_BASE, streak=0.7, splash=0.55, grime=0.5, lichen=0.7,
                   moss=0.5, rust=0.25, dust=0.1, fade=0.2, humid=0.65,
                   mould=0.4, rust_stain=0.35, peel=0.2, repaint=0.35,
                   efflor=0.4, algae=0.45, fade_tiles=0.09, soil="#5c5246"),
    "north_america": dict(_BASE, streak=0.45, splash=0.4, grime=0.35,
                          lichen=0.3, moss=0.35, rust=0.25, dust=0.2,
                          fade=0.3, humid=0.55, mould=0.3, rust_stain=0.3,
                          peel=0.15, repaint=0.2, efflor=0.2, algae=0.75,
                          fade_tiles=0.06, soil="#5f5446"),
    "mediterranean": dict(_BASE, streak=0.6, splash=0.45, grime=0.45,
                          lichen=0.6, moss=0.1, rust=0.35, dust=0.4,
                          fade=0.6, humid=0.3, mould=0.2, rust_stain=0.45,
                          peel=0.5, repaint=0.5, efflor=0.2, algae=0.2,
                          fade_tiles=0.12, soil="#8a7358"),
    "asia": dict(_BASE, streak=0.8, splash=0.5, grime=0.5, lichen=0.15,
                 moss=0.3, rust=0.45, dust=0.3, fade=0.35, humid=0.95,
                 mould=0.75, rust_stain=0.7, peel=0.35, repaint=0.45,
                 efflor=0.1, algae=0.55, fade_tiles=0.07, soil="#6a5e50"),
    "africa": dict(_BASE, streak=0.65, splash=0.9, grime=0.55, lichen=0.15,
                   moss=0.05, rust=0.75, dust=0.65, fade=0.6, humid=0.45,
                   mould=0.35, rust_stain=0.55, peel=0.55, repaint=0.6,
                   efflor=0.1, algae=0.25, fade_tiles=0.09, soil="#9a5a36"),
    "south_america": dict(_BASE, streak=0.7, splash=0.7, grime=0.55,
                          lichen=0.35, moss=0.3, rust=0.6, dust=0.4,
                          fade=0.5, humid=0.8, mould=0.6, rust_stain=0.55,
                          peel=0.55, repaint=0.55, efflor=0.35, algae=0.5,
                          fade_tiles=0.11, soil="#83553a"),
    "australia_oceania": dict(_BASE, streak=0.4, splash=0.45, grime=0.35,
                              lichen=0.5, moss=0.1, rust=0.35, dust=0.45,
                              fade=0.5, humid=0.4, mould=0.2,
                              rust_stain=0.3, peel=0.25, repaint=0.25,
                              efflor=0.15, algae=0.3, fade_tiles=0.07,
                              soil="#8a6a4a"),
}

# wall materials that are painted (repaint blocks / peeling make sense)
PAINTED = {"render", "panels", "block", "siding"}


def wear_for(region):
    if region in CLEAN_REGIONS:
        return {k: (0 if not isinstance(v, str) else v)
                for k, v in _BASE.items()}
    return dict(WEAR.get(region, WEAR["generic"]))


def _hex(value):
    value = value.lstrip("#")
    return np.array([int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4)],
                    dtype=np.float32)


def srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def _lin(hexcol):
    return srgb_to_lin(_hex(hexcol)).astype(np.float32)


def _fft_blur(n, sy, sx):
    """Gaussian blur by FFT: periodic in both axes (u wrap for free)."""
    h, w = n.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    g = np.exp(-2.0 * (np.pi ** 2) * ((fy * sy) ** 2 + (fx * sx) ** 2))
    return np.fft.irfft2(np.fft.rfft2(n) * g, s=(h, w)).astype(np.float32)


def periodic_noise(shape, sigma_px, rng, *, octaves=1, aniso=1.0):
    """Smooth noise in [0, 1] that wraps in u (axis 1).

    ``aniso`` > 1 stretches features vertically (rain runs, rust tails).
    """
    acc = np.zeros(shape, np.float32)
    amp = 1.0
    total = 0.0
    s = float(sigma_px)
    for _ in range(octaves):
        n = rng.standard_normal(shape).astype(np.float32)
        n = _fft_blur(n, s * aniso, s)
        n /= (n.std() + 1e-6)
        acc += amp * n
        total += amp
        amp *= 0.5
        s *= 0.5
    acc /= total
    return np.clip(0.5 + 0.22 * acc, 0.0, 1.0)


def _smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _classes(cls):
    return np.rint(cls * 16.0).astype(np.int16)


def _mix(out, col, amount, mask=None):
    """out <- lerp(out, col, amount) (amount [h,w], col rgb or [h,w,3])."""
    a = np.clip(amount, 0.0, 1.0)
    if mask is not None:
        a = a * mask
    a = a[..., None]
    return out * (1 - a) + col * a


def _darken(out, amount, tint=None, mask=None):
    """Multiply toward a (tinted) shadow: out * lerp(1, tint, amount)."""
    a = np.clip(amount, 0.0, 1.0)
    if mask is not None:
        a = a * mask
    t = np.array([0.35, 0.33, 0.31], np.float32) if tint is None else tint
    return out * (1 - a[..., None] * (1 - t))


def occlusion(shaded, albedo, cls_i, ref_classes):
    lum_s = shaded.mean(-1)
    lum_a = np.maximum(albedo.mean(-1), 1e-3)
    occ = lum_s / lum_a
    mask = np.isin(cls_i, ref_classes)
    ref = np.percentile(occ[mask], 92) if mask.sum() > 100 else \
        np.percentile(occ, 92)
    return np.clip(occ / max(ref, 1e-3), 0.0, 1.08)


def _run_down(source, px_per_m, rng, run_m, block=None, block_keep=0.2):
    """Propagate a source field downward (rows are top-first); each column
    runs its own distance in ``run_m`` = (min, max) metres."""
    h, w = source.shape
    lo, hi = run_m
    runs = lo + (hi - lo) * periodic_noise((1, w), max(1.0, 0.05 * px_per_m),
                                           rng)[0]
    decay = np.exp(-1.0 / np.maximum(1.0, runs * px_per_m)).astype(
        np.float32)
    out = np.empty_like(source)
    acc = np.zeros(w, np.float32)
    for r in range(h):
        acc = acc * decay + source[r]
        if block is not None:
            acc = np.where(block[r], acc * block_keep, acc)
        out[r] = acc
    return out


def _bottom_edges(mask):
    """Rows just below a mask region (where water leaves it)."""
    below = np.zeros_like(mask)
    below[1:] = mask[:-1]
    return np.clip(below.astype(np.float32) - mask.astype(np.float32), 0, 1)


def sill_streaks(drip, cls_i, px_per_m, rng, strength):
    """Rain runs leaving drip sources (sills, copings, AC units) at their
    ENDS, as water actually does."""
    src = drip > 0.5
    edge = _bottom_edges(src)
    dist = ndimage.distance_transform_edt(edge > 0, sampling=(1e3, 1.0))
    ends = edge * np.exp(-dist / max(1.0, 0.07 * px_per_m))
    ends = ndimage.gaussian_filter1d(ends, max(1.0, 0.035 * px_per_m),
                                     axis=1, mode="wrap") * 2.5
    source = 0.05 * edge + ends
    colnoise = periodic_noise((1, source.shape[1]),
                              max(1.0, 0.02 * px_per_m), rng)[0]
    fine = periodic_noise((1, source.shape[1]),
                          max(0.7, 0.006 * px_per_m), rng)[0]
    source *= (0.15 + 1.5 * colnoise * (0.4 + fine))[None, :]
    out = _run_down(source, px_per_m, rng, (0.3, 1.6),
                    block=cls_i == CLS_GLASS)
    out = ndimage.gaussian_filter(out, (0.6, max(0.6, 0.012 * px_per_m)),
                                  mode=("nearest", "wrap"))
    return np.clip(out * strength, 0, 1)


def top_runoff(shape, cls_i, px_per_m, rng, strength):
    """Water sheeting down from the roof edge / floor above: threads that
    enter at the top of the strip and fade at random heights."""
    h, w = shape
    col = periodic_noise((1, w), max(1.0, 0.18 * px_per_m), rng)[0]
    thread = periodic_noise((1, w), max(0.7, 0.012 * px_per_m), rng)[0]
    src = np.zeros(shape, np.float32)
    start = periodic_noise((1, w), max(1.0, 0.3 * px_per_m), rng)[0]
    rows = np.clip((start * 0.45 * h).astype(int), 0, h - 1)
    src[rows, np.arange(w)] = (_smoothstep(0.5, 0.82, col) *
                               (0.3 + 1.4 * thread) * 1.6)
    out = _run_down(src, px_per_m, rng, (0.6, 3.5),
                    block=cls_i == CLS_GLASS, block_keep=0.5)
    mod = periodic_noise(shape, 0.3 * px_per_m, rng, aniso=4.0)
    out = out * (0.4 + 1.2 * mod)
    return np.clip(out * strength, 0, 1)


def rust_runs(cls_i, metal, px_per_m, rng, strength):
    """Orange-brown runs below bars, rails and AC brackets."""
    fix = (cls_i == CLS_METAL) | ((metal > 0.35) & (cls_i != CLS_GLASS))
    edge = _bottom_edges(fix)
    edge = ndimage.gaussian_filter1d(edge, 0.8, axis=1, mode="wrap")
    jitter = periodic_noise((1, edge.shape[1]), max(0.8, 0.01 * px_per_m),
                            rng)[0]
    src = edge * _smoothstep(0.35, 0.7, jitter)[None, :] * 2.0
    out = _run_down(src, px_per_m, rng, (0.12, 0.7),
                    block=cls_i == CLS_GLASS)
    return np.clip(out * strength, 0, 1)


def repaint_blocks(shape, px_per_m, rng, amount):
    """Patchy repainting: soft-edged rectangles a shade off the wall."""
    h, w = shape
    field = np.ones(shape, np.float32)
    n = int(w / px_per_m * h / px_per_m / 12.0 * amount)
    for _ in range(n):
        bw = int(rng.uniform(0.6, 4.5) * px_per_m)
        bh = int(rng.uniform(0.5, 3.2) * px_per_m)
        x0 = int(rng.integers(0, w))
        y0 = int(rng.integers(-bh // 2, h))
        f = 1.0 + rng.choice((-1, 1)) * rng.uniform(0.03, 0.09)
        ys = slice(max(0, y0), min(h, y0 + bh))
        xs = (np.arange(x0, x0 + bw) % w)
        field[ys, xs] *= f
    return ndimage.gaussian_filter(field, (1.5, 1.5), mode=("nearest",
                                                            "wrap"))


def peeling(shape, px_per_m, rng, strength):
    """Flaking paint: small ragged islands, clustered in weak areas."""
    fine = periodic_noise(shape, max(0.7, 0.025 * px_per_m), rng, octaves=3)
    zone = periodic_noise(shape, 1.4 * px_per_m, rng)
    m = _smoothstep(0.74, 0.78, fine) * _smoothstep(0.62, 0.8, zone)
    return np.clip(m * strength * 1.4, 0, 1)


def process(npz, *, region, kind, level=None, px_per_m=80.0, seed=0,
            wear_scale=1.0, wall_kind="render", roof_flat=False):
    """Return (albedo_srgb_u8 [H,W,3], normal_rgba_u8 [H,W,4])."""
    d = {k: npz[k].astype(np.float32) for k in npz.files
         if k not in ("W", "H")}
    rng = np.random.default_rng(seed)
    shaded, albedo = d["shaded"], d["albedo"]
    cls_i = _classes(d["cls"])
    h, w = cls_i.shape
    wear = wear_for(region)
    k = wear_scale

    ref = (CLS_ROOF,) if kind == "roof" else (CLS_WALL,)
    occ = occlusion(shaded, albedo, cls_i, ref)
    # keep most of the baked occlusion: it is what reads as depth
    out = albedo * (occ ** 0.9)[..., None]

    glass = cls_i == CLS_GLASS
    if glass.any():
        # interior as seen through glass: dimmer, plus a soft sky reflection
        # that brightens toward the top of each pane
        lab, nlab = ndimage.label(glass)
        rows = np.arange(h, dtype=np.float32)[:, None].repeat(w, 1)
        refl = np.zeros((h, w), np.float32)
        if nlab:
            for i, s in enumerate(ndimage.find_objects(lab), start=1):
                if s is None:
                    continue
                r0, r1 = s[0].start, s[0].stop
                m = lab[s] == i
                t = (rows[s] - r0) / max(1.0, r1 - r0)
                refl[s][m] = (1.0 - t[m]) * 0.5 + 0.5
        sky = np.array([0.30, 0.36, 0.44], np.float32)
        tint = periodic_noise((h, w), 0.4 * px_per_m, rng)
        ref_amt = (0.16 + 0.10 * tint) * refl
        inner = shaded * 0.55
        out = np.where(glass[..., None],
                       inner * (1 - ref_amt[..., None]) +
                       sky * ref_amt[..., None], out)

    if kind != "roof":
        out = _walls(out, d, cls_i, occ, wear, k, level, px_per_m, rng,
                     wall_kind)
    else:
        out = _roofs(out, d, cls_i, occ, wear, k, px_per_m, rng,
                     flat=roof_flat)

    albedo_u8 = (lin_to_srgb(out) * 255.0 + 0.5).astype(np.uint8)
    nrm = d["normal"]
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-4)
    gloss = np.clip(d["gloss"], 0, 1)
    metal = np.clip(d["metal"], 0, 1)
    # base reflectance: dielectrics ~0.04 -> a small blue value
    base_refl = 0.04 + 0.96 * metal
    normal_u8 = np.stack([
        nrm[..., 0] * 0.5 + 0.5,
        nrm[..., 1] * 0.5 + 0.5,
        base_refl,
        gloss,
    ], axis=-1)
    normal_u8 = (np.clip(normal_u8, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    return albedo_u8, normal_u8


def _exposure(shape, px_per_m, rng, scale_m=6.0):
    """How weathered each stretch of the strip is: varies along u at the
    scale of a building, plus a softer 2-D component."""
    h, w = shape
    along = periodic_noise((1, w), max(2.0, scale_m * px_per_m), rng)[0]
    along = 0.25 + 1.35 * _smoothstep(0.2, 0.8, along)
    local = 0.7 + 0.6 * periodic_noise(shape, 1.2 * px_per_m, rng, octaves=2)
    return (along[None, :] * local).astype(np.float32)


def _walls(out, d, cls_i, occ, wear, k, level, px_per_m, rng, wall_kind):
    h, w = cls_i.shape
    if not any(wear[x] for x in ("streak", "splash", "grime", "mould")):
        return out
    wall = np.isin(cls_i, (CLS_WALL, CLS_PLINTH)).astype(np.float32)
    trim = np.isin(cls_i, (CLS_TRIM, CLS_SILL, CLS_SHUTTER, CLS_SIGN,
                           CLS_METAL)).astype(np.float32)
    solid = np.clip(wall + trim, 0, 1)
    E = _exposure((h, w), px_per_m, rng) * k
    humid = wear["humid"]
    dirt = _lin("#4a4238")
    mould_col = _lin("#27301f")
    streak_tint = (1 - humid) * np.array([0.42, 0.39, 0.35], np.float32) + \
        humid * np.array([0.33, 0.37, 0.30], np.float32)
    rust_col = _lin("#7a4424")

    # 0) paint colour: fade, repaint blocks (painted walls only)
    if wear["fade"] > 0:
        f = periodic_noise((h, w), 2.0 * px_per_m, rng) * wear["fade"] * E
        lum = out.mean(-1, keepdims=True)
        chalk = lum * 0.8 + 0.2 * np.array([0.8, 0.78, 0.74], np.float32)
        out = _mix(out, chalk, f * 0.4, solid)
    if wall_kind in PAINTED and wear["repaint"] > 0 and k > 0:
        field = repaint_blocks((h, w), px_per_m, rng, wear["repaint"])
        out = out * np.where(wall[..., None] > 0, field[..., None], 1.0)

    # 1) grime settles where the bake is occluded (reveals, under ledges)
    g = np.clip((0.95 - occ) / 0.45, 0, 1) * wear["grime"] * E
    g *= 0.4 + periodic_noise((h, w), 0.2 * px_per_m, rng, octaves=3)
    out = _mix(out, out * 0.55 + dirt * 0.12, g * 0.8, solid)

    # 2) large, soft weathering blotches (the wall "breathing" dirt)
    blot = periodic_noise((h, w), 1.0 * px_per_m, rng, octaves=3,
                          aniso=1.6)
    bl = _smoothstep(0.42, 0.85, blot) * wear["grime"] * E * 0.45
    out = _darken(out, bl, streak_tint, wall)

    # 3) run-off from above and 4) sill-end streaks
    if wear["streak"] > 0:
        s_top = top_runoff((h, w), cls_i, px_per_m, rng,
                           wear["streak"] * 0.9)
        s_sill = sill_streaks(d["drip"], cls_i, px_per_m, rng,
                              wear["streak"] * 1.6)
        s = np.clip((s_top + s_sill) * np.minimum(E, 1.6), 0, 1)
        out = _darken(out, s * 0.62, streak_tint, solid)

    # 5) rust runs below metal fixings
    if wear["rust_stain"] > 0:
        r = rust_runs(cls_i, d["metal"], px_per_m, rng,
                      wear["rust_stain"] * 1.2) * np.minimum(E, 1.5)
        out = _mix(out, out * 0.4 + rust_col * 0.6, r * 0.85, wall)

    # 6) mould / algae blooms: near the ground, under sills, in patches
    if wear["mould"] > 0:
        z_from_bottom = (h - 1 - np.arange(h, dtype=np.float32)) / px_per_m
        base = np.exp(-z_from_bottom / 0.9)[:, None] if level == "ground" \
            else np.full((h, 1), 0.25, np.float32)
        near_drip = ndimage.gaussian_filter(
            (d["drip"] > 0.5).astype(np.float32),
            (0.25 * px_per_m, 0.12 * px_per_m), mode=("nearest", "wrap"))
        near_drip = np.roll(near_drip, int(0.2 * px_per_m), axis=0)
        patch = periodic_noise((h, w), 0.35 * px_per_m, rng, octaves=3)
        m = (_smoothstep(0.5, 0.78, patch) * (base + near_drip * 3.0 + 0.15)
             * wear["mould"] * humid * E * 1.4)
        out = _mix(out, out * 0.5 + mould_col * 0.5, np.clip(m, 0, 0.85),
                   wall)

    # 7) peeling paint on painted walls
    if wall_kind in PAINTED and wear["peel"] > 0:
        p = peeling((h, w), px_per_m, rng, wear["peel"]) * np.minimum(E, 1.4)
        under = out * 0.55 + _lin("#8f8b84") * 0.45
        rim = ndimage.grey_dilation(p, size=(3, 3)) - p
        out = _mix(out, under, p * 0.9, wall)
        out = _darken(out, np.clip(rim, 0, 1) * 0.5, None, wall)

    # 8) efflorescence on brick
    if wall_kind == "brick" and wear["efflor"] > 0:
        z_from_bottom = (h - 1 - np.arange(h, dtype=np.float32)) / px_per_m
        base = np.exp(-z_from_bottom / 0.8)[:, None] if level == "ground" \
            else np.full((h, 1), 0.2, np.float32)
        near_drip = ndimage.gaussian_filter(
            (d["drip"] > 0.5).astype(np.float32),
            (0.3 * px_per_m, 0.1 * px_per_m), mode=("nearest", "wrap"))
        patch = periodic_noise((h, w), 0.25 * px_per_m, rng, octaves=2)
        m = _smoothstep(0.55, 0.8, patch) * (base + near_drip * 2) * \
            wear["efflor"] * E
        out = _mix(out, _lin("#d8d3c6"), np.clip(m * 0.6, 0, 0.6), wall)

    # 9) splash-back at the foot of ground floors (height varies per column)
    if level == "ground" and wear["splash"] > 0:
        z_from_bottom = (h - 1 - np.arange(h, dtype=np.float32)) / px_per_m
        reach = 0.18 + 0.5 * periodic_noise((1, w), 0.4 * px_per_m, rng)[0]
        prof = np.exp(-np.clip(z_from_bottom[:, None] - 0.03, 0, None) /
                      reach[None, :])
        sn = periodic_noise((h, w), 0.05 * px_per_m, rng, octaves=2)
        sp = np.clip(prof * (0.35 + sn) * wear["splash"] * E * 1.2, 0, 0.9)
        soil = _lin(wear["soil"])
        out = _mix(out, out * 0.45 + soil * 0.55, sp,
                   (cls_i != CLS_GLASS).astype(np.float32))
    return out


def _roofs(out, d, cls_i, occ, wear, k, px_per_m, rng, flat=False):
    h, w = cls_i.shape
    roof = (cls_i == CLS_ROOF).astype(np.float32)
    obj = (cls_i == CLS_OBJECT).astype(np.float32)
    E = (0.35 + 1.3 * periodic_noise((h, w), 3.0 * px_per_m, rng)) * k
    metal_px = np.clip((d["metal"] - 0.15) / 0.4, 0, 1)
    humid = wear["humid"]
    z_from_bottom = (h - 1 - np.arange(h, dtype=np.float32)) / px_per_m

    # 1) individual pieces: sun-faded / greyed, and the odd replacement
    piece = d["rnd"]
    hsh = (piece * 7.137 + 0.311) % 1.0
    faded = (hsh < wear["fade_tiles"]).astype(np.float32)
    faded = ndimage.grey_erosion(faded, size=(2, 2))
    lum = out.mean(-1, keepdims=True)
    # a sun-faded piece keeps its hue but loses saturation and darkens less
    # in the laps: half-way toward its own luminance, a touch lighter
    pale = (out * 0.55 + lum * 0.45) * 1.08
    out = _mix(out, pale, faded * 0.5 * np.minimum(E, 1.2), roof)
    replaced = (((piece * 13.71 + 0.77) % 1.0) < 0.025).astype(np.float32)
    replaced = ndimage.grey_erosion(replaced, size=(2, 2))
    out = _mix(out, out * np.array([1.12, 1.0, 0.92], np.float32) * 1.05,
               replaced * 0.8, roof)

    # 2) dirt drifting down to the eave (bottom rows of a roof strip)
    eave = np.exp(-z_from_bottom / 1.4)[:, None] * (0.0 if flat else 1.0)
    en = periodic_noise((h, w), 0.35 * px_per_m, rng, octaves=2)
    dirt_amt = eave * (0.3 + en) * (0.3 + wear["grime"]) * E * 0.45
    out = _darken(out, np.clip(dirt_amt, 0, 0.6),
                  np.array([0.45, 0.42, 0.38], np.float32), roof)

    if flat:
        # ponding: dried puddles leave dark tide marks; mould blooms where
        # water stands in humid climates; general grime in blotches
        pond = periodic_noise((h, w), 1.2 * px_per_m, rng, octaves=2)
        inside = _smoothstep(0.6, 0.66, pond)
        tide = inside - _smoothstep(0.66, 0.74, pond)
        out = _darken(out, np.clip((inside * 0.14 + tide * 0.16) *
                                   np.minimum(E, 1.4) * (0.4 + humid),
                                   0, 0.6),
                      np.array([0.5, 0.48, 0.45], np.float32), roof)
        blot = periodic_noise((h, w), 0.5 * px_per_m, rng, octaves=3)
        mb = _smoothstep(0.58, 0.8, blot) * wear["mould"] * humid * E
        out = _mix(out, out * 0.45 + _lin("#2c302a") * 0.55,
                   np.clip(mb, 0, 0.7), roof)
        grime = periodic_noise((h, w), 0.25 * px_per_m, rng, octaves=3)
        out = _darken(out, np.clip(grime * wear["grime"] * E * 0.35, 0,
                                   0.4), None, roof)
    # 3) black algae streaks running down-slope (Gloeocapsa etc.)
    if wear["algae"] > 0 and not flat:
        col = periodic_noise((1, w), max(1.0, 0.35 * px_per_m), rng,
                             octaves=2)[0]
        start = periodic_noise((1, w), max(1.0, 0.8 * px_per_m), rng)[0]
        start_m = (0.3 + 0.9 * start) * (h / px_per_m)
        on = _smoothstep(0.46, 0.7, col)[None, :]
        ramp = _smoothstep(0.0, 1.5, (start_m[None, :] -
                                      z_from_bottom[:, None]))
        body = periodic_noise((h, w), 0.25 * px_per_m, rng, aniso=5.0)
        a = on * ramp * (0.35 + body) * wear["algae"] * humid * E
        a = a * (1 - 0.6 * metal_px)
        out = _darken(out, np.clip(a * 0.8, 0, 0.7),
                      np.array([0.3, 0.3, 0.29], np.float32), roof)

    # 4) lichen rosettes: grey-white and yellow-orange colonies
    if wear["lichen"] > 0:
        cl = _smoothstep(0.4, 0.75, periodic_noise((h, w), 0.8 * px_per_m,
                                                   rng))
        for colour, dens in (("#c8c5b6", 0.0016), ("#c9a24a", 0.0007)):
            spots = rng.random((h, w)) < dens * wear["lichen"] * \
                (1 - 0.8 * np.mean(metal_px))
            r = max(1, int(0.025 * px_per_m))
            spots = ndimage.binary_dilation(spots, iterations=r)
            spots = ndimage.gaussian_filter(spots.astype(np.float32), 0.8,
                                            mode=("nearest", "wrap"))
            out = _mix(out, _lin(colour), np.clip(spots * cl * 1.5, 0, 0.85)
                       * np.minimum(E, 1.2), roof)

    # 5) moss in the laps and along the eave
    if wear["moss"] > 0:
        lap = np.clip(0.9 - occ, 0, 1) * 2.0
        clump = _smoothstep(0.55, 0.8, periodic_noise((h, w),
                                                      0.2 * px_per_m, rng,
                                                      octaves=2))
        m = (lap * 0.5 + clump * (0.3 + eave)) * wear["moss"] * humid * E
        m = m * (1 - metal_px)
        out = _mix(out, _lin("#4b5a2c"), np.clip(m * 0.7, 0, 0.75), roof)

    # 6) rust on metal: spots at fasteners and laps, bleeding down-slope
    if wear["rust"] > 0 and metal_px.mean() > 0.05:
        seeds = periodic_noise((h, w), 0.05 * px_per_m, rng, octaves=2)
        spots = _smoothstep(0.64, 0.82, seeds)
        laps = np.clip(0.8 - occ, 0, 1) * 1.5
        src = np.clip(spots + laps * 0.6, 0, 1) * wear["rust"] * np.minimum(
            E, 1.5)
        run = np.zeros_like(src)
        acc = np.zeros(w, np.float32)
        dec = np.exp(-1.0 / max(1.0, 0.35 * px_per_m))
        for r_ in range(h):
            acc = np.maximum(acc * dec, src[r_])
            run[r_] = acc
        patchy = periodic_noise((h, w), 1.2 * px_per_m, rng)
        rust = np.clip(run * (0.35 + patchy), 0, 1) * metal_px
        tone = periodic_noise((h, w), 0.15 * px_per_m, rng)
        col = (_lin("#5e3219")[None, None, :] * (1 - tone[..., None]) +
               _lin("#8e4f24")[None, None, :] * tone[..., None])
        out = _mix(out, col, 0.8 * rust)
        d["gloss"] = d["gloss"] * (1 - 0.8 * rust)
        d["metal"] = d["metal"] * (1 - 0.9 * rust)

    # 7) dust
    if wear["dust"] > 0:
        dn = periodic_noise((h, w), 1.5 * px_per_m, rng, octaves=2)
        du = dn * wear["dust"] * np.minimum(E, 1.4) * 0.35
        out = _mix(out, _lin(wear["soil"]) * 1.3, du,
                   np.clip(roof + obj * 0.5, 0, 1))

    # 8) rooftop equipment: grime, and rust on metal kit in wet/worn places
    og = periodic_noise((h, w), 0.3 * px_per_m, rng, octaves=2)
    out = _darken(out, np.clip((0.3 + og) * 0.25 * (0.4 + wear["grime"]) +
                               np.clip(0.9 - occ, 0, 1) * 0.4, 0, 0.6),
                  None, obj)
    if wear["rust"] > 0:
        ro = _smoothstep(0.6, 0.85, og) * metal_px * wear["rust"]
        out = _mix(out, _lin("#6e3a1c"), ro * 0.7, obj)

    # gentle large-scale tonal drift so a big roof isn't one flat tone
    drift = periodic_noise((h, w), 3.0 * px_per_m, rng)
    out = out * (0.92 + 0.16 * drift)[..., None]
    return out
