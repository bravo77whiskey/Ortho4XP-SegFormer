"""Offline model of X-Plane 12's terrain decal stage, for judging decals
before they go into the sim.

Per decal layer (as compiled in XP12's terrain shader):

* RGB:   albedo += (decal.rgb - mean(decal.rgb)) * clamp(dot(albedo, rgb_key) + m + k)
* alpha: v = mix(0.5, decal.a, clamp(dot(albedo, a_key) + m + k)), then an
         overlay blend of the albedo with v (v > 0.5 brightens, < 0.5 darkens)

The second layer keys on the albedo the first one produced.  Projected
decals repeat every ``scale_x`` x ``scale_y`` metres in world space; the
model samples them box-filtered to the render resolution, which is what
mipmapping does on screen.
"""

import numpy as np
from PIL import Image


class Layer:
    """One DECAL_PARAMS_PROJ line."""

    def __init__(self, texture, period_m, rgb_key, rgb_mk, a_key, a_mk):
        self.texture = np.asarray(texture, np.float32)
        self.period_m = float(period_m)
        self.rgb_key = np.asarray(rgb_key, np.float32)
        self.rgb_k = float(sum(rgb_mk))
        self.a_key = np.asarray(a_key, np.float32)
        self.a_k = float(sum(a_mk))

    @classmethod
    def from_dcl_line(cls, line, texture):
        f = line.split()
        assert f[0] == "DECAL_PARAMS_PROJ", line
        v = [float(t) for t in f[1:16]]
        return cls(texture, v[0], v[3:7], v[7:9], v[9:13], v[13:15])

    def sample(self, height, width, m_per_px, origin=(0.0, 0.0)):
        """Decal texels covering a height x width render, box-filtered."""
        n = max(4, int(round(self.period_m / m_per_px)))
        chans = []
        for c in range(self.texture.shape[2]):
            im = Image.fromarray(self.texture[..., c], mode="F")
            chans.append(np.asarray(im.resize((n, n), Image.BOX), np.float32))
        tex = np.stack(chans, -1)
        oy = int(round(origin[0] / m_per_px))
        ox = int(round(origin[1] / m_per_px))
        rows = (oy + np.arange(height)) % n
        cols = (ox + np.arange(width)) % n
        return tex[rows[:, None], cols[None, :]]


def apply_layers(albedo, layers, m_per_px, origin=(0.0, 0.0)):
    """Albedo (H x W x 3, 0..1) after the decal layers."""
    out = np.asarray(albedo, np.float32).copy()
    h, w, _ = out.shape
    for layer in layers:
        d = layer.sample(h, w, m_per_px, origin)
        mean = layer.texture.reshape(-1, layer.texture.shape[2]).mean(0)
        alb4 = np.concatenate([out, np.ones((h, w, 1), np.float32)], -1)
        lev_rgb = np.clip(alb4 @ layer.rgb_key + layer.rgb_k, 0, 1)[..., None]
        out = out + (d[..., :3] - mean[:3]) * lev_rgb
        lev_a = np.clip(alb4 @ layer.a_key + layer.a_k, 0, 1)[..., None]
        v = 0.5 + (d[..., 3:4] - 0.5) * lev_a
        out = np.where(v > 0.5, 1 - (1 - out) * (1 - 2 * (v - 0.5)), out * 2 * v)
        out = np.clip(out, 0, 1)
    return out


def luma(img):
    return img[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def repetition_score(detail, m_per_px, min_lag_m=4.0):
    """Highest normalised autocorrelation of ``detail`` at any lag of at
    least ``min_lag_m``: ~1 for a pattern that repeats exactly, ~0 for one
    that never does.  Returns (score, (dy_m, dx_m))."""
    d = detail - detail.mean()
    h, w = d.shape
    fh, fw = 2 * h, 2 * w
    F = np.fft.rfft2(d, (fh, fw))
    ac = np.fft.irfft2(F * np.conj(F), (fh, fw))
    ones = np.fft.rfft2(np.ones_like(d), (fh, fw))
    overlap = np.fft.irfft2(ones * np.conj(ones), (fh, fw))
    ac = ac / np.maximum(overlap, 1) / d.var()
    ys = np.fft.fftfreq(fh, 1 / fh)[:, None]
    xs = np.fft.fftfreq(fw, 1 / fw)[None, :]
    lag = np.hypot(ys, xs) * m_per_px
    # Only lags where at least half the window overlaps carry a usable estimate.
    valid = (lag >= min_lag_m) & (np.abs(ys) <= h / 2) & (np.abs(xs) <= w / 2)
    ac = np.where(valid, ac, -np.inf)
    i = np.unravel_index(np.argmax(ac), ac.shape)
    return float(ac[i]), (float(ys[i[0], 0] * m_per_px), float(xs[0, i[1]] * m_per_px))
