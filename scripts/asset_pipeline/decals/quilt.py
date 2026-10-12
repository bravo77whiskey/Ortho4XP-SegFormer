"""Toroidal image quilting: grow a small tileable texture into a large one
that still tiles seamlessly but has no internal repetition.

Efros & Freeman (2001) quilting with every index taken modulo the output
size, so the last row and column of patches are cut against the first and
the result wraps like the source does.  Patches come from random positions
in the (tileable) source; among ``candidates`` random picks the one that
best matches the pixels already placed under its overlap wins (ties broken
at random within ``tolerance``), and a minimum-error cut through each
overlap band decides which pixels it replaces.
"""

import numpy as np


def _cut_vertical(err):
    """Column index per row of the minimum-error top-to-bottom path."""
    h, w = err.shape
    cost = err.copy()
    for r in range(1, h):
        prev = cost[r - 1]
        left = np.concatenate(([np.inf], prev[:-1]))
        right = np.concatenate((prev[1:], [np.inf]))
        cost[r] += np.minimum(np.minimum(left, prev), right)
    path = np.empty(h, dtype=np.int64)
    path[-1] = int(np.argmin(cost[-1]))
    for r in range(h - 2, -1, -1):
        c = path[r + 1]
        lo, hi = max(c - 1, 0), min(c + 2, w)
        path[r] = lo + int(np.argmin(cost[r, lo:hi]))
    return path


def _side_mask(err, filled, overlap, side):
    """Pixels the new patch may write, judged against one overlap band."""
    size = err.shape[0]
    keep = np.ones(err.shape, dtype=bool)
    if side in ("left", "right"):
        band = slice(0, overlap) if side == "left" else slice(size - overlap, size)
        if not filled[:, band].any():
            return keep
        path = _cut_vertical(np.where(filled[:, band], err[:, band], 0.0))
        cols = np.arange(overlap)[None, :]
        new = cols >= path[:, None] if side == "left" else cols <= path[:, None]
        keep[:, band] = new
    else:
        band = slice(0, overlap) if side == "top" else slice(size - overlap, size)
        if not filled[band, :].any():
            return keep
        path = _cut_vertical(np.where(filled[band, :], err[band, :], 0.0).T)
        rows = np.arange(overlap)[None, :]
        new = rows >= path[:, None] if side == "top" else rows <= path[:, None]
        keep[band, :] = new.T
    return keep


def quilt(source, out_size, patch=80, overlap=16, candidates=160,
          tolerance=0.1, seed=0, weights=None):
    """Return an ``out_size`` square texture quilted from ``source``.

    ``source`` is H x W x C float32 and must itself tile.  ``out_size`` must
    be a multiple of ``patch - overlap``.  ``weights`` scales each channel's
    contribution to the matching error.
    """
    rng = np.random.default_rng(seed)
    src = np.asarray(source, dtype=np.float32)
    sh, sw, ch = src.shape
    step = patch - overlap
    if out_size % step:
        raise ValueError(f"out_size {out_size} is not a multiple of {step}")
    w = np.ones(ch, np.float32) if weights is None else np.asarray(weights, np.float32)
    out = np.zeros((out_size, out_size, ch), np.float32)
    filled = np.zeros((out_size, out_size), bool)
    ar = np.arange(patch)
    for y in range(0, out_size, step):
        rows = (y + ar) % out_size
        for x in range(0, out_size, step):
            cols = (x + ar) % out_size
            target = out[rows][:, cols]
            have = filled[rows][:, cols]
            sy = rng.integers(0, sh, candidates)
            sx = rng.integers(0, sw, candidates)
            if have.any():
                prow = (sy[:, None] + ar) % sh
                pcol = (sx[:, None] + ar) % sw
                cand = src[prow[:, :, None], pcol[:, None, :]]
                diff = ((cand - target) ** 2 * w).sum(-1)
                score = (diff * have).sum((1, 2))
                ok = np.flatnonzero(score <= score.min() * (1 + tolerance) + 1e-9)
                pick = cand[rng.choice(ok)]
                err = ((pick - target) ** 2 * w).sum(-1)
                write = ~have
                cut = np.ones_like(have)
                for side in ("left", "top", "right", "bottom"):
                    cut &= _side_mask(err, have, overlap, side)
                write |= cut
            else:
                pick = src[((sy[0] + ar) % sh)[:, None], ((sx[0] + ar) % sw)[None, :]]
                write = np.ones((patch, patch), bool)
            block = out[rows[:, None], cols[None, :]]
            block[write] = pick[write]
            out[rows[:, None], cols[None, :]] = block
            filled[rows[:, None], cols[None, :]] = True
    return out
