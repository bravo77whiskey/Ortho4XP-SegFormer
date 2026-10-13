"""Compare refnorm_gpu against refnorm (CPU) and time the GPU pipeline.

Usage: python test_gpu.py [key ...]   (CUDA venv)
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refnorm as RN  # noqa: E402
import refnorm_gpu as RG  # noqa: E402

look = json.load(open(os.path.join(RN.DATA, "color", "runs", "v1", "look.json")))["look"]
KEYS = sys.argv[1:] or ["houston"]


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


for key in KEYS:
    meta = json.load(open(os.path.join(RN.SET, key, "meta.json")))
    for prov in meta.get("providers", ["BI", "GO2"]):
        # --- CPU reference --------------------------------------------------
        t0 = time.perf_counter()
        b = RN.Block(key, prov)
        O, R, valid = b.coarse()
        excl = RN.water_mask(R, b.rdet, b.odet, valid)
        cg, co, cs, ct, cw = RN.fit_fields(O, R, valid, b.px_m, exclude=excl)
        cfields = RN.compose_look((cg, co, cs, ct), look)
        ij0 = sorted(b.paths)[len(b.paths) // 2]
        cpu_out = RN.correct_texture(b.texture(ij0), cfields, b, ij0)
        t_cpu = time.perf_counter() - t0

        # --- GPU, warm-up then timed ----------------------------------------
        RG.run_block(key, prov, look)
        t0 = sync()
        out, (g, o, s, t, w), gb = RG.run_block(key, prov, look)
        host = {ij: img.permute(1, 2, 0).cpu().numpy() for ij, img in out.items()}
        t_gpu = sync() - t0

        # Stage timing.
        t0 = sync()
        images = {ij: RG.decode(p) for ij, p in gb.paths.items()}
        t1 = sync()
        Og, Rg, vg, odet, rdet = gb.coarse(images)
        ex = RG.water_mask(Rg, rdet, odet, vg)
        t2 = sync()
        fit = RG.fit_fields(Og, Rg, RG._np(vg), gb.px_m, RG.P, ex)
        t3 = sync()
        fl = RG.compose_look(fit[:4], look)
        for ij in gb.paths:
            RG.correct_texture(images[ij], fl, gb, ij)
        t4 = sync()
        for ij in gb.paths:
            out[ij].cpu()
        t5 = sync()

        d = lambda a, c: np.abs(RG._np(a) - c)
        print(f"{key} {prov}: {len(gb.paths)} textures  CPU(1 texture applied) {t_cpu:.1f}s  "
              f"GPU all {t_gpu:.2f}s = decode {t1 - t0:.2f} + stats {t2 - t1:.2f} + fit {t3 - t2:.2f} "
              f"+ apply {t4 - t3:.2f} + download {t5 - t4:.2f}")
        print(f"   field max|diff|  g {d(g, cg).max():.4f}  o {d(o, co).max():.3f}  s {d(s, cs).max():.4f}  "
              f"t {np.abs(RG._np(t).transpose(1, 2, 0) - ct).max():.3f}   "
              f"(p99: o {np.percentile(d(o, co), 99):.3f})")
        px = np.abs(host[ij0].astype(int) - cpu_out.astype(int))
        print(f"   pixels vs CPU  mean {px.mean():.3f}  p99 {np.percentile(px, 99):.0f}  max {px.max()}  "
              f"identical {np.mean(px.max(-1) == 0):.1%}")
