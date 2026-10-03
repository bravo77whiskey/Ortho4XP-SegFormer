"""Relief scenes for the Cycles texture set (host side, numpy).

Each atlas region is modelled as a small 3-D scene seen straight on:
height-field layers (soil and crops under the film, the film itself, earth,
brick, quilts) carrying per-vertex colour, roughness and transparency, plus
tubes (hoops, ropes, poles, rolled film, tyres, hoses) and boxes (bricks,
sandbags).  ``bake_relief.py`` renders them in Cycles under an overcast
dome, so the film's translucency, the hoops showing through it, rope
grooves, sagging, soil heaps and dirt all come out of real light transport.

Scene space: x = the region's ``across`` axis, y = ``along`` (up the image),
z = out of the surface toward the camera; metres.  Tiling regions wrap in
y: every noise field is periodic and every feature is placed modulo the
period.

Wear is layered from the scene's own structure, never as a flat overlay:
dust settles on flat film and in rope grooves, soil splashes up from the
feet, rain carries grime down the slope in streaks, algae grows where
water sits, lime shading is sprayed in blotches, film yellows and is
patched with newer film and tape, and earth walls erode in gullies.
"""

from __future__ import annotations

import math
import os

import numpy as np

# linear-light colours
def srgb(h):
    h = h.lstrip("#")
    c = np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


SOIL = srgb("#6d5c48")
SOIL_DRY = srgb("#9b8a70")
LOESS = srgb("#8f7c62")
CROP = srgb("#3f5a2a")
CROP_DARK = srgb("#27381b")
FILM = srgb("#c4c8c1")
FILM_OLD = srgb("#c2bc9f")
DUST = srgb("#8e8470")
GRIME = srgb("#565a45")
LIME = srgb("#eeeeea")
ALGAE = srgb("#6f7a55")
STEEL = srgb("#8a8d8c")
ROPE = srgb("#1c1c1c")
BAMBOO = srgb("#a58f62")


class Scene:
    def __init__(self, name, W, H, px, view="top", period=None, seed=0):
        self.name, self.W, self.H, self.px = name, float(W), float(H), px
        self.view = view            # 'top' (sky above) or 'wall' (sky toward +y)
        self.period = period        # y period (tiling) or None
        self.rng = np.random.default_rng(seed)
        self.layers = []
        self.tubes = []
        self.boxes = []

    # grid helpers -----------------------------------------------------
    def grid(self, cell):
        nx = max(8, int(round(self.W / cell)) + 1)
        ny = max(8, int(round(self.H / cell)) + (0 if self.period else 1))
        xs = np.linspace(0.0, self.W, nx)
        if self.period:
            ys = np.arange(ny) * (self.H / ny)
        else:
            ys = np.linspace(0.0, self.H, ny)
        X, Y = np.meshgrid(xs, ys)
        return X, Y

    def noise(self, shape, cell, scale_m, aniso=(1.0, 1.0)):
        """Gaussian-filtered noise, unit std, periodic in both axes.

        ``aniso`` = (x, y): the smaller a value, the longer features run
        along that axis -- (0.1, 1.0) makes them ten times longer in x.
        """
        ny, nx = shape
        w = self.rng.standard_normal((ny, nx))
        fy = np.fft.fftfreq(ny, d=cell)[:, None] / aniso[1]
        fx = np.fft.fftfreq(nx, d=cell)[None, :] / aniso[0]
        g = np.exp(-2 * (math.pi * scale_m) ** 2 * (fx ** 2 + fy ** 2))
        out = np.real(np.fft.ifft2(np.fft.fft2(w) * g))
        s = out.std()
        return out / s if s > 0 else out

    def fbm(self, shape, cell, scale_m, octaves=4, aniso=(1.0, 1.0)):
        out = np.zeros(shape)
        amp, tot = 1.0, 0.0
        for o in range(octaves):
            out += amp * self.noise(shape, cell, scale_m / 2 ** o, aniso)
            tot += amp
            amp *= 0.5
        return out / tot

    def dy_wrap(self, Y, y0):
        d = Y - y0
        if self.period:
            d = (d + self.period / 2) % self.period - self.period / 2
        return d

    def layer(self, X, Y, z, col, trans=0.0, rough=0.8, metal=0.0, mask=None):
        shape = X.shape
        full = lambda v: np.broadcast_to(np.asarray(v, dtype=np.float64), shape).copy()
        self.layers.append({
            "x": X, "y": Y, "z": full(z),
            "col": np.broadcast_to(col, shape + (3,)).astype(np.float32).copy()
            if np.ndim(col) == 1 else np.asarray(col, np.float32),
            "trans": full(trans).astype(np.float32),
            "rough": full(rough).astype(np.float32),
            "metal": full(metal).astype(np.float32),
            "mask": np.ones(shape, bool) if mask is None else np.asarray(mask, bool),
        })

    def tube(self, pts, r, col, rough=0.6, metal=0.0, sides=8):
        self.tubes.append({"pts": [list(map(float, p)) for p in pts], "r": float(r),
                           "col": [float(c) for c in col], "rough": rough,
                           "metal": metal, "sides": sides})

    def box(self, c, size, rot, col, rough=0.8):
        self.boxes.append({"c": [float(v) for v in c], "size": [float(v) for v in size],
                           "rot": float(rot), "col": [float(v) for v in col],
                           "rough": rough})

    def save(self, out_dir):
        # render at the scene's own aspect (ortho camera spans max(W, H));
        # the page region stretches it afterwards
        self.px = (int(self.px[0]), max(8, int(round(self.px[0] * self.H / self.W))))
        os.makedirs(out_dir, exist_ok=True)
        files = []
        for i, lay in enumerate(self.layers):
            f = os.path.join(out_dir, f"{self.name}_L{i}.npz")
            np.savez_compressed(f, **lay)
            files.append(f)
        return {"name": self.name, "W": self.W, "H": self.H, "px": list(self.px),
                "view": self.view, "period": self.period, "layers": files,
                "tubes": self.tubes, "boxes": self.boxes}


def _mix(a, b, t):
    t = np.clip(t, 0.0, 1.0)[..., None]
    return a * (1 - t) + b * t


def _smooth(x, e0, e1):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _blobs(sc, shape, cell, scale, coverage, soft=0.35):
    """Random soft blobs covering about ``coverage`` of the area."""
    n = sc.fbm(shape, cell, scale, 3)
    thr = np.quantile(n, 1.0 - coverage)
    return _smooth(n, thr - soft * n.std(), thr + soft * n.std())


def _rect_mask(X, Y, sc, x0, y0, w, h):
    dy = sc.dy_wrap(Y, y0 + h / 2)
    return (np.abs(X - (x0 + w / 2)) <= w / 2) & (np.abs(dy) <= h / 2)


def weather_film(sc, X, Y, cell, col, trans, z, strength=1.0, slope=None):
    """Irregular film wear on top of a scene's own dust and streaks.

    ``slope`` (0 flat .. 1 steep, same shape) steers what settles and what
    runs off.  Adds: mottled grime at three scales, drip-dried water spots,
    sharp creases, leaf and straw debris, bird droppings, and a dusty side
    from the prevailing wind.  Returns (col, trans, z).
    """
    sh = X.shape
    flat = 1.0 - (slope if slope is not None else np.zeros(sh))
    mott = (0.6 * sc.fbm(sh, cell, 2.5, 2) + 0.4 * sc.fbm(sh, cell, 0.5, 3))
    grime = _smooth(mott, -0.2, 1.4) * strength
    col = _mix(col, GRIME, 0.45 * grime)
    trans = trans - 0.2 * grime
    # dried drip spots: rings of dirt where condensation fell back
    spots = _smooth(sc.noise(sh, cell, 0.03), 2.2, 2.9) * (0.4 + 0.6 * flat)
    col = _mix(col, DUST * 0.7, 0.7 * spots * strength)
    # creases: thin sharp lines at random angles
    for _ in range(int(sc.rng.integers(4, 9) * strength)):
        cx, cy = sc.rng.uniform(0, sc.W), sc.rng.uniform(0, sc.H)
        ang = sc.rng.uniform(0, math.pi)
        ln = sc.rng.uniform(0.3, 1.5)
        dx, dy = X - cx, sc.dy_wrap(Y, cy)
        along = dx * math.cos(ang) + dy * math.sin(ang)
        across = -dx * math.sin(ang) + dy * math.cos(ang)
        line = np.exp(-(across / 0.012) ** 2) * (np.abs(along) < ln / 2)
        z = z + 0.008 * line
        col = _mix(col, col * 1.15, 0.6 * line)
    # debris: leaves, straw, a few droppings
    debris = _smooth(sc.noise(sh, cell, 0.05), 2.4, 2.7) * flat
    col = _mix(col, srgb("#4b3b24"), 0.9 * debris * strength)
    trans = trans - 0.6 * debris
    drop = _smooth(sc.noise(sh, cell, 0.02), 3.0, 3.3) * flat
    col = _mix(col, LIME, 0.9 * drop)
    # wind side: one flank dustier than the other (tiles along y)
    side = 0.5 + 0.5 * np.tanh((X - sc.W * 0.5) / (0.3 * sc.W))
    col = _mix(col, DUST, 0.25 * side * strength)
    return col, np.clip(trans, 0.0, 0.85), z


# ── polytunnel roof ─────────────────────────────────────────────────────────

def tunnel_roof(name, arc_m, period, px, variant, seed):
    """Film over hoops, unrolled: x = arc from the left foot, y = along."""
    sc = Scene(name, arc_m, period, px, "top", period, seed)
    cell = 0.025
    X, Y = sc.grid(cell)
    sh = X.shape
    a = X / arc_m
    old = variant == "b"
    # soil and crops beneath: rows along the tunnel
    rows = 0.5 + 0.5 * np.cos(2 * math.pi * (X - 0.4) / 0.9)
    plants = np.clip(rows * 1.3 - 0.3 + 0.35 * sc.fbm(sh, cell, 0.4), 0, 1)
    plants *= _smooth(np.minimum(X, arc_m - X), 0.5, 1.3)
    floor_col = _mix(SOIL * (0.8 + 0.2 * sc.fbm(sh, cell, 1.0)[..., None]),
                     CROP * (0.7 + 0.5 * sc.fbm(sh, cell, 0.15)[..., None]), plants)
    floor_z = -0.9 + 0.25 * plants + 0.04 * sc.fbm(sh, cell, 0.08)
    sc.layer(X, Y, floor_z, floor_col, rough=0.95)
    # hoops every metre under the film, ropes between them over it
    hoop_y = np.arange(0.0, period, 1.0) + 0.5
    rope_y = np.arange(0.0, period, 1.0)
    z = np.zeros(sh)
    for y in hoop_y:
        z += 0.012 * np.exp(-(sc.dy_wrap(Y, y) / 0.06) ** 2)
    rope_pull = np.zeros(sh)
    for y in rope_y:
        rope_pull += np.exp(-(sc.dy_wrap(Y, y) / 0.16) ** 2)
    # ropes pull hardest on the sides, the crown bulges between them
    side = 1.0 - np.sin(math.pi * a) ** 2
    z -= (0.025 + 0.02 * side) * rope_pull
    # wrinkles run around the arch (film stretched over the hoops)
    z += 0.006 * sc.fbm(sh, cell, 0.12, 3, aniso=(0.25, 1.0))
    z += 0.012 * sc.fbm(sh, cell, 1.2, 2)
    # film appearance
    dust = np.clip(0.45 + 0.35 * sc.fbm(sh, cell, 1.5, 3), 0, 1)
    dust = dust * (0.55 + 0.45 * np.sin(math.pi * a))          # flat crown holds dust
    dust += 0.5 * rope_pull * np.sin(math.pi * a)              # dirt lines at ropes
    streak = _smooth(sc.fbm(sh, cell, 0.35, 3, aniso=(0.12, 1.0)), 0.4, 1.4)
    streak *= 1.0 - np.sin(math.pi * a) ** 4                   # runs down the sides
    col = _mix(FILM_OLD if old else FILM, DUST, 0.6 * np.clip(dust, 0, 1))
    col = _mix(col, GRIME, (0.6 if old else 0.4) * streak)
    trans = 0.7 - 0.3 * np.clip(dust, 0, 1) - 0.2 * streak
    if old:
        trans -= 0.12
        # lime shading is sprayed from the ground: bands down the slope,
        # thick near the crown, thinning and speckled toward the sides
        spray = _smooth(sc.fbm(sh, cell, 0.6, 3, aniso=(0.25, 1.0)), -0.3, 1.0)
        reach = _smooth(np.sin(math.pi * a), 0.25, 0.8)
        speck = _smooth(sc.noise(sh, cell, 0.015), -0.5, 1.5)
        lime = np.clip(spray * reach * (0.45 + 0.55 * speck), 0, 1)
        lime *= _blobs(sc, sh, cell, 2.0, 0.7, 0.5)                 # sprayed in sessions
        col = _mix(col, LIME, 0.75 * lime)
        trans -= 0.45 * lime
    # repair patches of newer film, edged with tape
    mask = np.ones(sh, bool)
    tape = np.zeros(sh)
    for _ in range(int(sc.rng.integers(3, 6) + (3 if old else 0))):
        w, h = sc.rng.uniform(0.3, 1.4), sc.rng.uniform(0.3, 1.2)
        x0 = sc.rng.uniform(0.12 * arc_m, 0.88 * arc_m - w)
        y0 = sc.rng.uniform(0, period)
        inner = _rect_mask(X, Y, sc, x0, y0, w, h)
        outer = _rect_mask(X, Y, sc, x0 - 0.04, y0 - 0.04, w + 0.08, h + 0.08)
        col[inner] = _mix(col[inner], FILM * 0.9, 0.8)
        trans[inner] = 0.55
        tape[outer & ~inner] = 1.0
    col = _mix(col, srgb("#b9b8ae"), 0.85 * tape)
    trans = np.where(tape > 0, 0.15, trans)
    # tears (old film only)
    if old:
        for _ in range(int(sc.rng.integers(2, 5))):
            cx, cy = sc.rng.uniform(0.15, 0.85) * arc_m, sc.rng.uniform(0, period)
            rr = sc.rng.uniform(0.05, 0.18)
            d = np.hypot((X - cx) / (1 + sc.rng.random()), sc.dy_wrap(Y, cy))
            rag = 0.03 * sc.noise(sh, cell, 0.04)
            mask &= d > rr + rag
    # soil over the feet, mud splashed above them
    foot = np.minimum(X, arc_m - X)
    buried = foot < 0.03 * arc_m
    splash = (1 - _smooth(foot, 0.1, 0.8)) * (0.6 + 0.4 * sc.fbm(sh, cell, 0.05))
    col = _mix(col, SOIL, 0.8 * splash)
    trans = trans - 0.5 * splash
    z += 0.02 * splash * sc.fbm(sh, cell, 0.03)
    col[buried] = (SOIL * (0.8 + 0.3 * sc.fbm(sh, cell, 0.1)[..., None]))[buried]
    trans[buried] = 0.0
    z[buried] += 0.08 + 0.04 * sc.fbm(sh, cell, 0.1)[buried]
    # side vents rolled up (old film): the film ends above the vent
    if old:
        for lo, hi in ((0.03, 0.065), (0.935, 0.97)):
            gap = (a > lo) & (a < hi)
            mask &= ~gap
            x_roll = (hi if lo < 0.5 else lo) * arc_m
            sc.tube([[x_roll, -0.5, 0.03], [x_roll, period + 0.5, 0.03]], 0.055,
                    FILM_OLD * 0.8, rough=0.5)
    col, trans, z = weather_film(sc, X, Y, cell, col, trans, z,
                                 1.3 if old else 1.0, slope=np.abs(np.cos(math.pi * a)))
    rough = 0.35 + 0.5 * np.clip(dust, 0, 1)
    sc.layer(X, Y, z, col, trans=np.clip(trans, 0, 0.85), rough=rough, mask=mask)
    for y in hoop_y:
        sc.tube([[0.0, y, -0.012], [arc_m, y, -0.012]], 0.012, STEEL, rough=0.4, metal=0.8)
    for y in rope_y:
        wob = sc.rng.uniform(-0.03, 0.03)
        pts = [[x, y + wob * math.sin(x * 1.3),
                -0.018 - 0.02 * (1 - math.sin(math.pi * x / arc_m) ** 2)]
               for x in np.linspace(0.05 * arc_m, 0.95 * arc_m, 16)]
        sc.tube(pts, 0.005, ROPE, rough=0.7)
    return sc


# ── polytunnel gable ───────────────────────────────────────────────────────

def tunnel_gable(name, span, height, px, variant, seed, door):
    sc = Scene(name, span, height, px, "wall", None, seed)
    cell = 0.02
    X, Y = sc.grid(cell)
    sh = X.shape
    # dark interior with plants
    plants = _blobs(sc, sh, cell, 0.25, 0.45) * _smooth(1.6 - Y, 0.0, 0.8)
    inside = _mix(CROP_DARK * 0.5, CROP * 0.8, plants)
    sc.layer(X, Y, -1.2 + 0.4 * plants, inside, rough=0.95)
    a0, a1, b0, b1 = door
    dx0, dx1, dtop = a0 * span, a1 * span, b1 * 3.2
    # frame: posts and a bar over the door
    for x in (0.18 * span, 0.3 * span, dx0 - 0.03, dx1 + 0.03, 0.7 * span, 0.82 * span):
        top = height * max(0.0, math.sin(math.pi * x / span)) ** 0.6
        sc.tube([[x, 0.0, -0.05], [x, top, -0.05]], 0.02, STEEL, rough=0.45, metal=0.8)
    sc.tube([[dx0 - 0.05, dtop + 0.03, -0.05], [dx1 + 0.05, dtop + 0.03, -0.05]], 0.02,
            STEEL, rough=0.45, metal=0.8)
    # film: folds hang down, bulges between posts
    z = 0.01 * sc.fbm(sh, cell, 0.1, 3, aniso=(1.0, 0.2)) + 0.02 * sc.fbm(sh, cell, 0.8, 2)
    dust = np.clip(0.4 + 0.3 * sc.fbm(sh, cell, 0.8), 0, 1)
    splash = (1 - _smooth(Y, 0.05, 0.9)) * (0.5 + 0.5 * sc.fbm(sh, cell, 0.04))
    col = _mix(FILM_OLD if variant in (1, 3) else FILM, DUST, 0.55 * dust)
    col = _mix(col, SOIL, 0.85 * splash)
    trans = np.clip(0.55 - 0.3 * dust - 0.6 * splash, 0, 0.85)
    rain = _smooth(sc.fbm(sh, cell, 0.25, 3, aniso=(1.0, 0.1)), 0.3, 1.5)
    col = _mix(col, GRIME, 0.5 * rain)
    trans = trans - 0.2 * rain
    algae = (1 - _smooth(Y, 0.2, 1.0)) * _blobs(sc, sh, cell, 0.3, 0.5)
    col = _mix(col, ALGAE * 0.8, 0.6 * algae)
    hands = np.exp(-((X - (dx0 + dx1) / 2) / 0.9) ** 2 - ((Y - 1.1) / 0.5) ** 2)
    col = _mix(col, SOIL * 0.8, 0.5 * hands * (0.5 + 0.5 * sc.fbm(sh, cell, 0.08)))
    col, trans, z = weather_film(sc, X, Y, cell, col, trans, z, 1.0, slope=np.ones(sh) * 0.8)
    mask = np.ones(sh, bool)
    door_area = (X > dx0) & (X < dx1) & (Y < dtop)
    mask &= ~door_area
    if variant in (1, 2):
        # torn corner flapping open
        cx = span * (0.25 if variant == 1 else 0.72)
        d = np.hypot(X - cx, Y - 1.6)
        mask &= d > 0.25 + 0.06 * sc.noise(sh, cell, 0.05)
    # patches and tape
    for _ in range(int(sc.rng.integers(2, 5))):
        w, h = sc.rng.uniform(0.3, 0.9), sc.rng.uniform(0.3, 0.8)
        x0, y0 = sc.rng.uniform(0.1, 0.8) * span, sc.rng.uniform(0.2, 2.2)
        inner = (np.abs(X - x0 - w / 2) < w / 2) & (np.abs(Y - y0 - h / 2) < h / 2)
        col[inner] = _mix(col[inner], FILM, 0.6)
        trans[inner] = 0.45
    sc.layer(X, Y, z, col, trans=trans, rough=0.4 + 0.4 * dust, mask=mask)
    # the door itself
    dmask = (X > dx0 - 0.02) & (X < dx1 + 0.02) & (Y < dtop + 0.02)
    if variant == 0:          # film door on a steel frame
        dcol = _mix(FILM, SOIL, 0.6 * (1 - _smooth(Y, 0.0, 0.7)))
        sc.layer(X, Y, 0.04 + 0.005 * sc.fbm(sh, cell, 0.1), dcol, trans=0.45,
                 rough=0.45, mask=dmask)
        for x in (dx0, dx1):
            sc.tube([[x, 0.0, 0.05], [x, dtop, 0.05]], 0.018, STEEL, 0.4, 0.8)
        for y in (0.05, dtop / 2, dtop):
            sc.tube([[dx0, y, 0.05], [dx1, y, 0.05]], 0.015, STEEL, 0.4, 0.8)
    elif variant == 1:        # plank door
        plank = np.floor((X - dx0) / 0.14)
        tone = 0.75 + 0.25 * np.sin(plank * 12.9898) ** 2
        grain = 0.08 * sc.fbm(sh, cell, 0.05, 2, aniso=(1.0, 0.1))
        dcol = srgb("#6e5a44") * (tone + grain)[..., None]
        dcol = _mix(dcol, SOIL, 0.6 * (1 - _smooth(Y, 0.0, 0.6)))
        gap = (((X - dx0) / 0.14) % 1.0) < 0.06
        sc.layer(X, Y, 0.05 - 0.01 * gap, dcol, rough=0.85, mask=dmask)
        sc.tube([[dx0, 0.4, 0.07], [dx1, 0.4, 0.07]], 0.02, srgb("#5a4a38"), 0.9)
        sc.tube([[dx0, dtop - 0.4, 0.07], [dx1, dtop - 0.4, 0.07]], 0.02, srgb("#5a4a38"), 0.9)
    elif variant == 2:        # film door hanging slack, tied with rope
        dcol = _mix(FILM_OLD, SOIL, 0.5 * dust + 0.5 * (1 - _smooth(Y, 0.0, 0.8)))
        sc.layer(X, Y, 0.03 + 0.03 * sc.fbm(sh, cell, 0.2, 2, aniso=(1.0, 0.3)), dcol,
                 trans=0.35, rough=0.5, mask=dmask)
        sc.tube([[dx0 - 0.1, 1.0, 0.07], [(dx0 + dx1) / 2, 0.95, 0.08], [dx1 + 0.1, 1.02, 0.07]],
                0.008, ROPE, 0.8)
    else:                     # quilt hung as a door curtain
        stitch = ((np.abs(((X - dx0) / 0.25) % 1 - 0.5) < 0.04) |
                  (np.abs((Y / 0.25) % 1 - 0.5) < 0.04))
        puff = 0.02 * (np.sin(math.pi * ((X - dx0) / 0.25 % 1)) *
                       np.sin(math.pi * (Y / 0.25 % 1)))
        dcol = srgb("#3f4a35") * (0.8 + 0.25 * sc.fbm(sh, cell, 0.3))[..., None]
        dcol = _mix(dcol, SOIL_DRY, 0.35 * dust + 0.6 * (1 - _smooth(Y, 0.0, 0.5)))
        dcol[stitch] *= 0.7
        sc.layer(X, Y, 0.05 + puff, dcol, rough=0.95, mask=dmask)
    # sandbags weighing the film down
    if variant != 1:
        x = 0.1
        while x < span - 0.3:
            if not (dx0 - 0.2 < x < dx1 + 0.1):
                bag = srgb(("#a49c86", "#8c9aa6", "#9a8f6a", "#b1ab98")[int(sc.rng.integers(4))])
                sc.box([x + 0.25, 0.08, 0.12], [0.5, 0.16, 0.25], sc.rng.uniform(-8, 8),
                       bag * sc.rng.uniform(0.45, 0.75), 0.95)
            x += sc.rng.uniform(0.55, 0.9)
    return sc


# ── solar greenhouse ───────────────────────────────────────────────────────

def solar_film(name, arc_m, period, px, knee_frac, seed):
    sc = Scene(name, arc_m, period, px, "top", period, seed)
    cell = 0.03
    X, Y = sc.grid(cell)
    sh = X.shape
    a = X / arc_m
    # crop rows run north-south, i.e. across the film
    rows = 0.5 + 0.5 * np.cos(2 * math.pi * Y / 0.8)
    plants = np.clip(rows * 1.4 - 0.3 + 0.3 * sc.fbm(sh, cell, 0.3), 0, 1)
    plants *= _smooth(a, knee_frac + 0.02, knee_frac + 0.1) * (1 - _smooth(a, 0.8, 0.95))
    floor_col = _mix(SOIL * 0.8, CROP * (0.7 + 0.4 * sc.fbm(sh, cell, 0.15)[..., None]), plants)
    sc.layer(X, Y, -1.2 + 0.5 * plants, floor_col, rough=0.95)
    poles = np.arange(0.0, period, 1.0) + 0.5
    ropes = np.arange(0.0, period, 1.0)
    z = np.zeros(sh)
    for y in poles:
        z += 0.02 * np.exp(-(sc.dy_wrap(Y, y) / 0.07) ** 2)
    pull = np.zeros(sh)
    for y in ropes:
        pull += np.exp(-(sc.dy_wrap(Y, y) / 0.18) ** 2)
    z -= 0.03 * pull
    z += 0.007 * sc.fbm(sh, cell, 0.15, 3, aniso=(0.25, 1.0))
    # heavy yellow dust, drip lines down the slope, dark stains under the quilts
    dust = np.clip(0.55 + 0.35 * sc.fbm(sh, cell, 2.0, 3) + 0.3 * pull, 0, 1)
    drip = _smooth(sc.fbm(sh, cell, 0.3, 3, aniso=(0.1, 1.0)), 0.5, 1.5)
    under_quilt = _smooth(a, 0.82, 0.9) * (0.6 + 0.4 * sc.fbm(sh, cell, 0.5))
    col = _mix(FILM_OLD, DUST, 0.65 * dust)
    col = _mix(col, GRIME, 0.45 * drip)
    col = _mix(col, srgb("#4d4a42"), 0.55 * under_quilt)
    trans = np.clip(0.6 - 0.35 * dust - 0.2 * drip - 0.4 * under_quilt, 0.05, 0.8)
    # knee: straw and soil banked against the film
    knee = a < knee_frac
    straw = _mix(srgb("#8d7a55"), SOIL, 0.5 + 0.25 * sc.fbm(sh, cell, 0.05))
    col[knee] = straw[knee]
    trans[knee] = 0.0
    z[knee] += 0.06 * (0.5 + 0.5 * sc.fbm(sh, cell, 0.04)[knee])
    splash = (1 - _smooth(a, knee_frac, knee_frac + 0.08)) * (1 - knee)
    col = _mix(col, SOIL, 0.7 * splash)
    # patches
    for _ in range(int(sc.rng.integers(3, 7))):
        w, h = sc.rng.uniform(0.4, 1.8), sc.rng.uniform(0.4, 1.4)
        x0 = sc.rng.uniform(knee_frac + 0.05, 0.8) * arc_m
        inner = _rect_mask(X, Y, sc, x0, sc.rng.uniform(0, period), w, h)
        col[inner] = _mix(col[inner], FILM, 0.55)
        trans[inner] = 0.55
    col, trans, z = weather_film(sc, X, Y, cell, col, trans, z, 1.3,
                                 slope=1.0 - _smooth(a, 0.2, 0.9))
    sc.layer(X, Y, z, col, trans=trans, rough=0.4 + 0.45 * dust)
    for y in poles:
        sc.tube([[knee_frac * arc_m, y, -0.02], [arc_m, y, -0.02]], 0.02, BAMBOO, 0.7)
    for y in ropes:
        pts = [[x, y + 0.03 * math.sin(x), -0.035] for x in np.linspace(knee_frac * arc_m, arc_m, 10)]
        sc.tube(pts, 0.005, ROPE, 0.8)
    return sc


QUILT_COLS = ("#3b3d3a", "#4a5337", "#26282a", "#5b6068", "#6b6a5a", "#2f3a45")


def solar_quilt(name, circ_m, period, px, seed):
    sc = Scene(name, circ_m, period, px, "top", period, seed)
    cell = 0.02
    X, Y = sc.grid(cell)
    sh = X.shape
    col = np.zeros(sh + (3,))
    z = np.zeros(sh)
    y = 0.0
    edges = []
    while y < period - 0.8:
        w = sc.rng.uniform(1.6, 3.2)
        w = min(w, period - y)
        edges.append((y, y + w))
        y += w
    if edges:
        edges[-1] = (edges[-1][0], period)
    for (y0, y1) in edges:
        band = (Y >= y0) & (Y < y1)
        c = srgb(QUILT_COLS[int(sc.rng.integers(len(QUILT_COLS)))])
        col[band] = c
        # a roll shows its wraps: spiral edge lines around it
        wraps = (X / circ_m * sc.rng.uniform(2.5, 4.0) + (Y - y0) * 0.02) % 1.0
        z[band] += (0.02 * np.sin(math.pi * wraps))[band]
        dy = np.minimum(Y - y0, y1 - Y)
        z[band] -= (0.04 * (1 - _smooth(dy, 0.0, 0.08)))[band]     # gap between rolls
    stitch = (np.abs((Y / 0.3) % 1 - 0.5) < 0.05)
    z -= 0.005 * stitch
    dust = np.clip(0.5 + 0.4 * sc.fbm(sh, cell, 0.8, 3), 0, 1) * (0.5 + 0.5 * np.sin(math.pi * X / circ_m))
    col = col * (0.85 + 0.25 * sc.fbm(sh, cell, 0.2)[..., None])
    col = _mix(col, LOESS, 0.5 * dust)
    col[stitch] *= 0.8
    sc.layer(X, Y, z + 0.004 * sc.fbm(sh, cell, 0.05), col, rough=0.95)
    # straps around the rolls
    for _ in range(int(period / 2.5)):
        yy = sc.rng.uniform(0, period)
        sc.tube([[0, yy, 0.02], [circ_m, yy + 0.02, 0.02]], 0.008, srgb("#1d1d1d"), 0.8)
    return sc


def solar_back(name, run_m, period, px, seed):
    sc = Scene(name, run_m, period, px, "top", period, seed)
    cell = 0.02
    X, Y = sc.grid(cell)
    sh = X.shape
    straw_fibres = sc.fbm(sh, cell, 0.04, 2, aniso=(1.0, 0.08))
    straw = srgb("#8f7b54") * (0.75 + 0.2 * straw_fibres[..., None])
    quilt = _blobs(sc, sh, cell, 0.8, 0.35, 0.2)
    qcol = srgb("#3d4236") * (0.9 + 0.2 * sc.fbm(sh, cell, 0.3)[..., None])
    soil = _blobs(sc, sh, cell, 0.4, 0.25, 0.3)
    weeds = _blobs(sc, sh, cell, 0.12, 0.08, 0.2) * _smooth(X, run_m * 0.6, run_m)
    col = _mix(straw, qcol, quilt)
    col = _mix(col, SOIL_DRY, 0.7 * soil)
    col = _mix(col, CROP, 0.9 * weeds)
    col = _mix(col, LOESS, 0.3 * np.clip(0.5 + 0.4 * sc.fbm(sh, cell, 1.0), 0, 1))
    z = 0.03 * quilt + 0.015 * straw_fibres + 0.05 * weeds + 0.01 * sc.fbm(sh, cell, 0.2)
    sc.layer(X, Y, z, col, rough=0.95)
    for _ in range(int(period / 1.2)):
        sc.box([sc.rng.uniform(0.2, run_m - 0.2), sc.rng.uniform(0, period), 0.08],
               [0.24, 0.115, 0.053], sc.rng.uniform(0, 180),
               srgb("#7a4a38") * sc.rng.uniform(0.7, 1.0), 0.9)
    for _ in range(int(period / 5)):
        cx, cy, r = sc.rng.uniform(0.4, run_m - 0.4), sc.rng.uniform(0, period), 0.3
        pts = [[cx + r * math.cos(t), cy + r * math.sin(t), 0.1]
               for t in np.linspace(0, 2 * math.pi, 17)]
        sc.tube(pts, 0.09, srgb("#1b1b1b"), 0.8, sides=8)
    return sc


def _earth(sc, X, Y, cell, height, layers=0.35, gully_dir="y"):
    """Rammed earth: compaction layers, erosion gullies, crumbled edges."""
    sh = X.shape
    hcoord = Y if gully_dir == "y" else X
    base = LOESS * (0.78 + 0.22 * sc.fbm(sh, cell, 1.5)[..., None])
    base = _mix(base, srgb("#6f6a5e"), 0.35 * _blobs(sc, sh, cell, 1.2, 0.5, 0.6))
    band = (hcoord / layers) % 1.0
    ledge = -0.012 * _smooth(band, 0.85, 1.0)
    tone = 0.93 + 0.08 * np.sin(np.floor(hcoord / layers) * 7.3)
    col = base * tone[..., None]
    # gullies run down the wall
    along = X if gully_dir == "y" else Y
    g = sc.fbm(sh, cell, 0.25, 3, aniso=(1.0, 0.1) if gully_dir == "y" else (0.1, 1.0))
    gully = _smooth(g, 0.8, 1.8) * (0.4 + 0.6 * _smooth(hcoord, 0.0, height))
    z = ledge - 0.04 * gully + 0.004 * sc.fbm(sh, cell, 0.06, 3)
    col = _mix(col, SOIL * 0.7, 0.55 * gully)
    # rain-darkened runs and pale salt bloom
    runs = _smooth(sc.fbm(sh, cell, 0.15, 3, aniso=(1.0, 0.12) if gully_dir == "y" else (0.12, 1.0)), 0.2, 1.6)
    col = _mix(col, GRIME, 0.35 * runs)
    bloom = _blobs(sc, sh, cell, 0.12, 0.12, 0.4) * _smooth(sc.fbm(sh, cell, 1.5), 0.0, 1.0)
    col = _mix(col, srgb("#aaa392"), 0.25 * bloom)
    damp = (1 - _smooth(hcoord, 0.0, 0.5)) * (0.6 + 0.4 * sc.fbm(sh, cell, 0.3))
    col = _mix(col, SOIL * 0.6, 0.5 * damp)
    weeds = _blobs(sc, sh, cell, 0.12, 0.35, 0.2) * (
        (1 - _smooth(hcoord, 0.05, 0.4)) + _smooth(hcoord, height - 0.25, height))
    col = _mix(col, _mix(CROP, SOIL_DRY, 0.4 * sc.fbm(sh, cell, 0.1)), 0.9 * weeds)
    z += 0.06 * weeds
    del along
    return z, col


def solar_wall(name, slant_m, period, px, seed):
    """x = height up the battered face, y = along."""
    sc = Scene(name, slant_m, period, px, "wall_x", period, seed)
    cell = 0.02
    X, Y = sc.grid(cell)
    z, col = _earth(sc, X, Y, cell, slant_m, gully_dir="x")
    cap = X > 0.97 * slant_m
    col[cap] = _mix(col[cap], CROP, 0.3)
    sc.layer(X, Y, z, col, rough=0.95)
    return sc


def solar_gable(name, width_m, height_m, px, seed):
    sc = Scene(name, width_m, height_m, px, "wall", None, seed)
    cell = 0.025
    X, Y = sc.grid(cell)
    sh = X.shape
    z, col = _earth(sc, X, Y, cell, height_m)
    # small wooden door and a vent hole
    dx0, dx1, dtop = 0.18 * width_m, 0.18 * width_m + 0.8, 1.7
    door = (X > dx0) & (X < dx1) & (Y < dtop)
    plank = np.floor((X - dx0) / 0.12)
    dcol = srgb("#5d4a36") * (0.7 + 0.3 * np.sin(plank * 7.1) ** 2)[..., None]
    col[door] = _mix(dcol, SOIL, 0.5 * (1 - _smooth(Y, 0.0, 0.5)))[door]
    z[door] = -0.1
    frame = door & ~((X > dx0 + 0.06) & (X < dx1 - 0.06) & (Y < dtop - 0.06))
    col[frame] = srgb("#3e3326")
    vent = np.hypot(X - 0.55 * width_m, Y - 0.8 * height_m) < 0.2
    col[vent] = srgb("#1a1a18")
    z[vent] = -0.25
    sc.layer(X, Y, z, col, rough=0.95)
    return sc


def ante_wall(name, perim_m, height_m, px, seed, door, faces):
    """Brick walls: x = distance round the room, y = height."""
    sc = Scene(name, perim_m, height_m, px, "wall", None, seed)
    cell = 0.008
    X, Y = sc.grid(cell)
    sh = X.shape
    course = np.floor(Y / 0.063)
    off = (course % 2) * 0.125
    bx = (X + off) / 0.25
    bid = (np.floor(bx).astype(np.int64) * 73856093) ^ (course.astype(np.int64) * 19349663)
    h1 = ((bid * 2654435761) % 1000003) / 1000003.0
    h2 = ((bid * 40503 + 12345) % 999983) / 999983.0
    mortar = (((bx % 1.0) < 0.04) | (((Y / 0.063) % 1.0) < 0.16))
    tone = 0.72 + 0.28 * h1
    brick = _mix(srgb("#7a5445"), srgb("#80796f"), (h2 > 0.55) * 0.8)
    col = brick * tone[..., None]
    col[mortar] = srgb("#9c978d")
    z = np.where(mortar, -0.006, 0.0) + 0.002 * sc.fbm(sh, cell, 0.02)
    smear = _blobs(sc, sh, cell, 0.2, 0.1, 0.4) * _smooth(sc.fbm(sh, cell, 1.0), -0.2, 1.0)
    col = _mix(col, srgb("#a19c90"), 0.6 * smear)
    z += 0.004 * smear
    splash = (1 - _smooth(Y, 0.0, 0.5)) * (0.5 + 0.5 * sc.fbm(sh, cell, 0.05))
    col = _mix(col, SOIL, 0.75 * splash)
    col = _mix(col, LOESS, 0.3 * np.clip(0.5 + 0.4 * sc.fbm(sh, cell, 1.0), 0, 1))
    for fx in faces[1:]:
        corner = np.abs(X - fx) < 0.03
        col[corner] *= 0.75
    a0, a1, b0, b1 = door
    dx0, dx1, dtop = a0 * perim_m, a1 * perim_m, b1 * height_m
    dm = (X > dx0) & (X < dx1) & (Y < dtop)
    dent = sc.fbm(sh, cell, 0.15, 2)
    dcol = srgb("#4a6173") * (0.8 + 0.15 * dent)[..., None]
    dcol = _mix(dcol, GRIME, 0.4 * _smooth(sc.fbm(sh, cell, 0.3), -0.5, 1.5))
    rust = _smooth(sc.fbm(sh, cell, 0.08, 2), 0.6, 1.5) * (1 - _smooth(Y, 0.0, 0.6))
    dcol = _mix(dcol, srgb("#6b3a1f"), 0.8 * rust)
    dcol = _mix(dcol, SOIL, 0.5 * (1 - _smooth(Y, 0.0, 0.4)))
    col[dm] = dcol[dm]
    z[dm] = -0.03 + 0.004 * dent[dm]
    # window on the end face
    wx0 = faces[1] + 1.4
    wm = (X > wx0) & (X < wx0 + 0.7) & (Y > 1.2) & (Y < 1.8)
    glass = _mix(srgb("#2a2f2e"), srgb("#8a8a80"), 0.5 * _blobs(sc, sh, cell, 0.1, 0.4))
    col[wm] = glass[wm]
    z[wm] = -0.06
    sc.layer(X, Y, z, col, rough=np.where(wm, 0.2, 0.9))
    sc.tube([[dx1 - 0.08, 1.0, 0.02], [dx1 - 0.02, 1.0, 0.02]], 0.012, STEEL, 0.4, 0.8)
    return sc


def ante_roof(name, w, h, px, seed):
    sc = Scene(name, w, h, px, "top", None, seed)
    cell = 0.01
    X, Y = sc.grid(cell)
    sh = X.shape
    col = srgb("#8f8c85") * (0.85 + 0.15 * sc.fbm(sh, cell, 0.5)[..., None])
    tar = _blobs(sc, sh, cell, 0.3, 0.2, 0.15)
    col = _mix(col, srgb("#222222"), 0.9 * tar)
    stain = _blobs(sc, sh, cell, 0.8, 0.4, 0.4)
    col = _mix(col, SOIL * 0.7, 0.4 * stain)
    col = _mix(col, LOESS, 0.3 * np.clip(0.5 + 0.4 * sc.fbm(sh, cell, 1.0), 0, 1))
    z = 0.004 * tar + 0.003 * sc.fbm(sh, cell, 0.03)
    parapet = (np.minimum(np.minimum(X, w - X), np.minimum(Y, h - Y)) < 0.12)
    z[parapet] += 0.1
    col[parapet] = srgb("#77716a")
    sc.layer(X, Y, z, col, rough=np.where(tar > 0.5, 0.4, 0.9))
    for _ in range(5):
        sc.box([sc.rng.uniform(0.4, w - 0.4), sc.rng.uniform(0.4, h - 0.4), 0.03],
               [0.24, 0.115, 0.053], sc.rng.uniform(0, 180), srgb("#7a4a38"), 0.9)
    cx, cy = w * 0.62, h * 0.35
    pts = [[cx + (0.15 + 0.02 * t) * math.cos(t), cy + (0.15 + 0.02 * t) * math.sin(t), 0.02]
           for t in np.linspace(0, 6 * math.pi, 60)]
    sc.tube(pts, 0.012, srgb("#2f5a3a"), 0.6, sides=6)
    return sc
