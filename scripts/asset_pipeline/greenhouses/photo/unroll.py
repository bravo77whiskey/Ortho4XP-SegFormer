"""Unroll polytunnel film from an oblique photo into a flat, tiling strip.

The photo shows a tunnel from its end and above.  Its two feet are the
dark gaps beside it, which are straight lines on the ground, so a
homography from the band between them to a rectangle removes the
ground-plane perspective.  The film stands above the ground, so each hoop
still bows in that rectangle; every hoop is traced across the strip and
the median bow is shifted out, column by column.  Hoops are then placed at
an even spacing and a window of whole hoops is cut out, so the strip
repeats every ``hoops`` hoops.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

LUM = np.array([0.3, 0.59, 0.11], np.float32)


def gap_mask(img, level=110 / 255):
    """Dark gaps (crop rows, shadow) between bright film."""
    lum = img @ LUM
    return ndimage.binary_opening(ndimage.uniform_filter(lum, 9) < level, iterations=2)


def edge_line(mask, p0, p1, side, win=80):
    """Fit x = a*y + b to the edge of the dark gap that runs near p0 -> p1.

    ``side`` 'right' takes the gap's right edge (the tunnel lies to its
    right), 'left' its left edge.
    """
    (x0, y0), (x1, y1) = p0, p1
    ys, xs = [], []
    for y in range(int(max(0, min(y0, y1))), int(min(mask.shape[0], max(y0, y1))), 4):
        xc = x0 + (x1 - x0) * (y - y0) / (y1 - y0)
        lo, hi = int(max(0, xc - win)), int(min(mask.shape[1], xc + win))
        idx = np.flatnonzero(mask[y, lo:hi])
        if len(idx) < 6:
            continue
        xs.append(lo + (idx.max() if side == "right" else idx.min()))
        ys.append(y)
    ys, xs = np.asarray(ys, float), np.asarray(xs, float)
    for _ in range(3):
        a, b = np.polyfit(ys, xs, 1)
        r = np.abs(xs - (a * ys + b))
        keep = r < max(6.0, 2.5 * float(np.median(r)))
        ys, xs = ys[keep], xs[keep]
    return np.polyfit(ys, xs, 1)


def homography(src, dst):
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.asarray(rows, float))
    return vt[-1].reshape(3, 3) / vt[-1, -1]


def warp_quad(img, quad, w, h):
    """Photo quad (TL, TR, BR, BL) -> w x h image, bilinear."""
    H = homography([(0, 0), (w, 0), (w, h), (0, h)], quad)
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    p = H @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
    x, y = p[0] / p[2], p[1] / p[2]
    return np.stack([ndimage.map_coordinates(img[..., c], [y, x], order=1, mode="nearest")
                     .reshape(h, w) for c in range(img.shape[2])], -1)


def bright_ridge(img):
    """Thin bright lines running across (the hoops' highlights)."""
    lum = img @ LUM
    hp = lum - ndimage.gaussian_filter(lum, (5, 1))
    return ndimage.gaussian_filter(hp, (1.2, 2))


def trace_bow(img, step=8):
    """Median vertical offset of the hoops per column, relative to the
    centre column, from tracing every hoop that crosses the centre."""
    r = bright_ridge(img)
    h, w = r.shape
    c = w // 2
    prof = ndimage.gaussian_filter1d(r[:, c - 6:c + 6].mean(1), 1.5)
    thr = np.percentile(prof, 80)
    peaks = [y for y in range(13, h - 13)
             if prof[y] == prof[y - 12:y + 13].max() and prof[y] > thr]
    curves = []
    for y0 in peaks:
        ys = np.full(w, np.nan)
        ys[c] = y0
        for d in (1, -1):
            y, x = y0, c
            while 0 <= x + d < w:
                x += d
                lo = max(0, y - step)
                y = lo + int(np.argmax(r[lo:y + step + 1, x]))
                ys[x] = y
        curves.append(ys - y0)
    return ndimage.gaussian_filter1d(np.nanmedian(np.asarray(curves), 0), 4)


def shift_columns(img, shift):
    h, w = img.shape[:2]
    yy = np.arange(h)[:, None] + shift[None, :]
    xx = np.broadcast_to(np.arange(w)[None, :], (h, w))
    return np.stack([ndimage.map_coordinates(img[..., c], [yy, xx], order=1, mode="nearest")
                     for c in range(img.shape[2])], -1)


def hoop_rows(img):
    """Rows of the hoops (row 0 = far end), evenly enough spaced to trust."""
    prof = ndimage.gaussian_filter1d(bright_ridge(img).mean(1), 2)
    p = prof - prof.mean()
    ac = np.correlate(p, p, "full")[len(p) - 1:]
    # a strip holds 10-40 hoops: look for the period in that range
    lo, hi = len(p) // 40, len(p) // 10
    period = lo + int(np.argmax(ac[lo:hi]))
    half = max(4, int(period * 0.35))
    peaks = [y for y in range(half, len(prof) - half)
             if prof[y] == prof[y - half:y + half + 1].max()]
    return np.asarray(peaks), period


def even_hoops(img, peaks):
    """Resample rows so the hoops at ``peaks`` are evenly spaced, first hoop
    on row 0; returns the image, the new hoop rows and the spacing."""
    period = float(np.median(np.diff(peaks)))
    target = period * np.arange(len(peaks))
    w = img.shape[1]
    rows = np.arange(int(target[-1]) + 1, dtype=float)
    src = np.interp(rows, target, peaks)
    yy = np.broadcast_to(src[:, None], (len(rows), w))
    xx = np.broadcast_to(np.arange(w)[None, :], (len(rows), w))
    out = np.stack([ndimage.map_coordinates(img[..., c], [yy, xx], order=1, mode="nearest")
                    for c in range(img.shape[2])], -1)
    return out, target, period


def flatten(img, sigma):
    """Divide out shading wider than ``sigma`` px (keeps the mean colour)."""
    lum = img @ LUM
    low = ndimage.gaussian_filter(lum, sigma, mode="wrap")
    return np.clip(img * (lum.mean() / np.maximum(low, 1e-3))[..., None], 0, 1)


def unroll(img, gap_left, gap_right, y_range, columns, width=768, height=4000):
    """The tunnel's film between two gap lines as an even-hoop strip.

    Returns (strip, hoop rows, hoop period); strip row 0 is the far end.
    """
    mask = gap_mask(img)
    gl = edge_line(mask, *gap_left, "right")
    gr = edge_line(mask, *gap_right, "left")
    y_far, y_near = y_range
    quad = [(gl[0] * y_far + gl[1], y_far), (gr[0] * y_far + gr[1], y_far),
            (gr[0] * y_near + gr[1], y_near), (gl[0] * y_near + gl[1], y_near)]
    band = warp_quad(img, quad, width, height)
    band = shift_columns(band, trace_bow(band))
    band = band[:, int(width * columns[0]):int(width * columns[1])]
    peaks, _ = hoop_rows(band)
    strip, rows, period = even_hoops(band, peaks)
    return strip, rows, period
