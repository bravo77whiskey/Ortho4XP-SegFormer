"""Cycles texture set: model each atlas region, render it, pack the pages.

    python textures_cycles.py --out <pages dir> [--work <dir>] [--samples 96]
                              [--blender <exe>] [--only name,...]

Scenes come from ``relief.py``, rendering from ``bake_relief.py`` (Blender,
OptiX).  Each region renders at twice its page size and is filtered down.
Pages: tunnels.png / solar.png (albedo, sRGB) and *_nml.png (R,G tangent
normal, B base reflectance for NORMAL_METALNESS, A gloss).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import geometry as G  # noqa: E402
import layout as L  # noqa: E402
import relief as R  # noqa: E402

DEFAULT_BLENDER = r"F:\SteamLibrary\steamapps\common\Blender\blender.exe"
OVERSAMPLE = 2
# the overcast dome lights a white surface to about 0.45: expose up to this
EXPOSURE_MAX = 2.0


def _arc(pts):
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))


def dimensions():
    unit = G._tunnel_unit_profile(48)
    tunnel_arc = _arc([(s * G.TUNNEL_SPAN_M, y * G.TUNNEL_HEIGHT_M) for s, y in unit])
    sec = G.solar_section(14.0, 48)
    film_arc = _arc(sec["front"])
    return {
        "tunnel_arc": tunnel_arc,
        "solar_arc": film_arc,
        "knee_frac": G.SOLAR_KNEE_M / film_arc,
        "back_run": _arc(sec["back"]),
        "wall_slant": _arc(sec["wall"]) + _arc(sec["cap"]),
        "gable_w": 14.0,
        "gable_h": sec["h"] + G.GABLE_RISE_M,
        "ante_perim": 2 * G.ANTE_LEN_M + (G.ANTE_DEPTH[1] - G.ANTE_DEPTH[0]),
        "ante_faces": (0.0, G.ANTE_LEN_M, G.ANTE_LEN_M + G.ANTE_DEPTH[1] - G.ANTE_DEPTH[0]),
        "quilt_circ": 2 * math.pi * 0.33,
    }


def scenes():
    d = dimensions()
    P = L.ALONG_PERIOD_M

    def px(region):
        a, b = region.size_px
        return (a * OVERSAMPLE, b * OVERSAMPLE)

    T, S = L.TUNNELS, L.SOLAR
    out = [
        ("tunnels", "roof_a", R.tunnel_roof("roof_a", d["tunnel_arc"], P, px(T["roof_a"]), "a", 11)),
        ("tunnels", "roof_b", R.tunnel_roof("roof_b", d["tunnel_arc"], P, px(T["roof_b"]), "b", 12)),
    ]
    for k, name in enumerate(L.TUNNEL_GABLES):
        out.append(("tunnels", name, R.tunnel_gable(
            name, G.TUNNEL_SPAN_M, G.TUNNEL_HEIGHT_M, px(T[name]), k, 20 + k, L.TUNNEL_DOOR)))
    out += [
        ("solar", "film", R.solar_film("film", d["solar_arc"], P, px(S["film"]), d["knee_frac"], 31)),
        ("solar", "quilt", R.solar_quilt("quilt", d["quilt_circ"], P, px(S["quilt"]), 32)),
        ("solar", "back", R.solar_back("back", d["back_run"], P, px(S["back"]), 33)),
        ("solar", "wall_a", R.solar_wall("wall_a", d["wall_slant"], P, px(S["wall_a"]), 34)),
        ("solar", "wall_b", R.solar_wall("wall_b", d["wall_slant"], P, px(S["wall_b"]), 38)),
        ("solar", "gable_a", R.solar_gable("gable_a", d["gable_w"], d["gable_h"], px(S["gable_a"]), 35)),
        ("solar", "gable_b", R.solar_gable("gable_b", d["gable_w"], d["gable_h"], px(S["gable_b"]), 39)),
        ("solar", "ante_wall", R.ante_wall("ante_wall", d["ante_perim"], G.ANTE_HEIGHT_M,
                                           px(S["ante_wall"]), 36, L.ANTE_DOOR, d["ante_faces"])),
        ("solar", "ante_roof", R.ante_roof("ante_roof", G.ANTE_LEN_M,
                                           G.ANTE_DEPTH[1] - G.ANTE_DEPTH[0], px(S["ante_roof"]), 37)),
    ]
    return out


def _to_srgb(lin):
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)


def finish(npz_path, region):
    """Rendered passes -> (albedo RGB uint8, normal RGBA uint8) at page size."""
    d = np.load(npz_path)
    shaded = d["shaded"].astype(np.float32)
    # the dome is not a white furnace: bring the brightest film to ~0.92
    scale = 0.85 / max(1e-3, float(np.quantile(shaded.max(axis=-1), 0.995)))
    albedo = (_to_srgb(shaded * min(scale, EXPOSURE_MAX)) * 255 + 0.5).astype(np.uint8)
    n = d["normal"].astype(np.float32)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6
    rough = np.clip(d["rough"].astype(np.float32), 0, 1)
    metal = np.clip(d["metal"].astype(np.float32), 0, 1)
    nml = np.stack([
        (n[..., 0] * 0.5 + 0.5) * 255,
        (n[..., 1] * 0.5 + 0.5) * 255,
        (10 + 190 * metal),
        (1.0 - rough) * 255,
    ], axis=-1).clip(0, 255).astype(np.uint8)
    size = region.size_px
    a_img = Image.fromarray(albedo).resize(size, Image.LANCZOS)
    n_img = Image.fromarray(nml, "RGBA").resize(size, Image.LANCZOS)
    if region.rotated:
        a_img = a_img.transpose(Image.ROTATE_270)
        arr = np.asarray(n_img.transpose(Image.ROTATE_270)).copy()
        # clockwise turn: old +y -> new +x, old +x -> new -y
        nx = arr[..., 0].astype(np.int16) - 128
        ny = arr[..., 1].astype(np.int16) - 128
        arr[..., 0] = np.clip(ny + 128, 0, 255)
        arr[..., 1] = np.clip(-nx + 128, 0, 255)
        n_img = Image.fromarray(arr, "RGBA")
    return a_img, n_img


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=None)
    ap.add_argument("--samples", type=int, default=96)
    ap.add_argument("--blender", default=DEFAULT_BLENDER)
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    args.out = os.path.abspath(args.out)
    work = os.path.abspath(args.work or os.path.join(args.out, "_work"))
    os.makedirs(work, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None
    items = scenes()
    jobs = []
    for page, name, sc in items:
        if only and name not in only:
            continue
        job = sc.save(os.path.join(work, "layers"))
        job["out"] = os.path.join(work, f"{page}_{name}.npz")
        jobs.append(job)
    spec = os.path.join(work, "jobs.json")
    with open(spec, "w", encoding="utf-8") as fh:
        json.dump({"samples": args.samples, "jobs": jobs}, fh)
    if not args.no_render:
        subprocess.run([args.blender, "--background", "--factory-startup", "--python",
                        os.path.join(HERE, "bake_relief.py"), "--", spec], check=True)
    for page, regions in (("tunnels", L.TUNNELS), ("solar", L.SOLAR)):
        alb = Image.new("RGB", (L.PAGE_PX, L.PAGE_PX), (128, 128, 128))
        nml = Image.new("RGBA", (L.PAGE_PX, L.PAGE_PX), (128, 128, 10, 60))
        for name, region in regions.items():
            f = os.path.join(work, f"{page}_{name}.npz")
            if not os.path.isfile(f):
                print(f"missing render {f}")
                continue
            a_img, n_img = finish(f, region)
            alb.paste(a_img, (region.x0, region.y0))
            nml.paste(n_img, (region.x0, region.y0))
        alb.save(os.path.join(args.out, f"{page}.png"))
        nml.save(os.path.join(args.out, f"{page}_nml.png"))
        print(f"page {page} -> {args.out}")


if __name__ == "__main__":
    main()
