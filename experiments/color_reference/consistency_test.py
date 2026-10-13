"""Fit two overlapping halves of a block independently and compare them.

Simulates two neighbouring tiles each fitted only on their own textures: A sees
texture columns 0..1, B sees 1..2.  Reports Delta-E76 between A's and B's
corrected colour over the shared column (coarse grid), which is what a tile
border running through that column would show.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refnorm as RN

def run(key, prov, look, axis="cols"):
    b = RN.Block(key, prov)
    O, R, valid = b.coarse()
    c, m = RN.TEX // RN.COARSE, b.margin
    def sub(lo, hi):
        v = np.zeros_like(valid)
        if axis == "cols": v[:, m + lo * c:m + hi * c] = valid[:, m + lo * c:m + hi * c]
        else: v[m + lo * c:m + hi * c] = valid[m + lo * c:m + hi * c]
        g, o, s, t, w = RN.fit_fields(O, R, v, b.px_m)
        g, o, s, t = RN.compose_look((g, o, s, t), look)
        return RN.apply(O, g, o, s, t)
    A, B = sub(0, 2), sub(1, 3)
    sl = (slice(m, m + 3 * c), slice(m + c, m + 2 * c)) if axis == "cols" else (slice(m + c, m + 2 * c), slice(m, m + 3 * c))
    dE = np.linalg.norm(A[sl] - B[sl], axis=-1)[valid[sl]]
    mid = (slice(m, m + 3 * c), m + c + c // 2) if axis == "cols" else (m + c + c // 2, slice(m, m + 3 * c))
    dmid = np.linalg.norm(A[mid] - B[mid], axis=-1)
    print(f"{key} {prov} {axis}: shared column dE p50 {np.median(dE):.2f} p90 {np.percentile(dE, 90):.2f} "
          f"p99 {np.percentile(dE, 99):.2f} | centre line p50 {np.median(dmid):.2f} p90 {np.percentile(dmid, 90):.2f} max {dmid.max():.2f}")

look = json.load(open(os.path.join(RN.DATA, "color", "runs", "v1", "look.json")))["look"]
for k, pvs in [("houston", ["BI", "GO2"]), ("boston_sw", ["GO2", "BI"])]:
    for pv in pvs:
        for ax in ("cols", "rows"):
            run(k, pv, look, ax)
