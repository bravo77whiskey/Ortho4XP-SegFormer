"""Build the O4XP ortho decal library: X-Plane's shrub/dirt detail decal
without the 25 m grid.

    python build_decals.py --out <pkg> [--xplane <X-Plane 12 dir>] [--png]

X-Plane's ``maquify_2_green_key.dcl`` (the decal Ortho4XP puts on ortho
terrain) repeats one 1024 px texture every 25 m, and from a few hundred
feet the repeat reads as a regular lattice over every field.  This library
keeps the same texture pixels and the same colour keys but:

* re-synthesises the texture by toroidal quilting (``quilt.py``) into
  4096 px textures that tile seamlessly and contain no repeat inside;
* uses both decal slots X-Plane allows: layer A at the source pixel
  density (period 100 m), layer B ``PHI`` times coarser (period 161.8 m).
  The periods have an irrational ratio, so the sum of the two never
  repeats; each carries 1/sqrt(2) of the source contrast so the sum keeps
  the source's overall contrast.

The source texture is read from the local X-Plane install; its pixels are
Laminar's and never enter this repository, only the generated package.

Writes ``<pkg>/library.txt``, ``<pkg>/ortho_detail.dcl`` and
``<pkg>/textures/``.  Tiles refer to the decal as ``DECAL_LIB
o4xp/decals/ortho_detail.dcl`` (``O4_DSF_Utils``).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from quilt import quilt  # noqa: E402

NVCOMPRESS = os.path.join(HERE, "..", "..", "..", "Utils", "win", "nvcompress", "nvcompress.exe")
DEFAULT_XPLANE = r"D:\SteamLibrary\steamapps\common\X-Plane 12"
DECALS = os.path.join("Resources", "default scenery", "1000 decals")
SOURCE_DCL = "maquify_2_green_key.dcl"

VIRTUAL = "o4xp/decals/ortho_detail.dcl"
PHI = (1 + 5 ** 0.5) / 2
SIZE = 4096
# (name, metres per source period multiplier, quilting seed)
LAYERS = (("ortho_detail_a", 1.0, 1), ("ortho_detail_b", PHI, 2))


def read_source(xplane):
    """The source .dcl line's fields and its RGBA texture (0..1 floats)."""
    folder = os.path.join(xplane, DECALS)
    with open(os.path.join(folder, SOURCE_DCL), encoding="utf-8") as f:
        line = next(ln for ln in f if ln.startswith("DECAL_PARAMS_PROJ"))
    fields = line.split()
    tex = Image.open(os.path.join(folder, fields[-1].replace("/", os.sep))).convert("RGBA")
    return fields, np.asarray(tex, np.float32) / 255


def rescale_tileable(tex, factor):
    """Resample a tileable texture by ``factor`` keeping it tileable."""
    h, w, c = tex.shape
    nh, nw = int(round(h * factor)), int(round(w * factor))
    big = np.tile(tex, (3, 3, 1))
    out = []
    for i in range(c):
        im = Image.fromarray(big[..., i], mode="F").resize((3 * nw, 3 * nh), Image.LANCZOS)
        out.append(np.asarray(im, np.float32)[nh:2 * nh, nw:2 * nw])
    return np.clip(np.stack(out, -1), 0, 1)


def soften(tex, gain):
    """Scale the contrast of the RGB detail around its mean and of the alpha
    overlay around neutral 0.5 -- the two quantities the shader uses."""
    out = tex.copy()
    mean = tex[..., :3].reshape(-1, 3).mean(0)
    out[..., :3] = mean + (tex[..., :3] - mean) * gain
    out[..., 3] = 0.5 + (tex[..., 3] - 0.5) * gain
    return np.clip(out, 0, 1)


def build_layer(src, coarser, seed, size=SIZE):
    """One layer texture: ``coarser`` times the source texel size."""
    base = src if coarser == 1.0 else rescale_tileable(src, 1.0 / coarser)
    return soften(quilt(base, size, seed=seed), 2 ** -0.5)


def to_dds(src, dst):
    # Linear mips: the shader subtracts the 1x1 mip as the texture mean.
    subprocess.run([NVCOMPRESS, "-repeat", "-production", "-color", "-alpha",
                    "-no-mip-gamma-correct", "-bc3", src, dst],
                   check=True, stdout=subprocess.DEVNULL)


def dcl_line(fields, period_m, texture):
    """The source line with a new period and texture."""
    p = f"{period_m:.3f}"
    return "\t".join(["DECAL_PARAMS_PROJ", p, p, *fields[3:-1], texture])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="library package directory")
    ap.add_argument("--xplane", default=DEFAULT_XPLANE, help="X-Plane 12 directory")
    ap.add_argument("--png", action="store_true", help="ship PNG instead of DDS")
    ap.add_argument("--size", type=int, default=SIZE, help="texture size (px)")
    args = ap.parse_args()

    fields, src = read_source(args.xplane)
    src_period = float(fields[1])
    texel = src_period / src.shape[0]
    os.makedirs(os.path.join(args.out, "textures"), exist_ok=True)
    lines = []
    for name, coarser, seed in LAYERS:
        print(f"{name}: quilting {args.size} px ...", flush=True)
        tex = build_layer(src, coarser, seed, args.size)
        png = os.path.join(args.out, "textures", name + ".png")
        Image.fromarray((tex * 255 + 0.5).astype(np.uint8), "RGBA").save(png)
        rel = "textures/" + name + ".png"
        if not args.png:
            to_dds(png, png[:-4] + ".dds")
            os.remove(png)
            rel = rel[:-4] + ".dds"
        lines.append(dcl_line(fields, args.size * texel * coarser, rel))

    with open(os.path.join(args.out, "ortho_detail.dcl"), "w", encoding="ascii", newline="\n") as f:
        f.write("A\n1000\nDECAL\n\n" + "\n".join(lines) + "\n")
    with open(os.path.join(args.out, "library.txt"), "w", encoding="ascii", newline="\n") as f:
        f.write(f"A\n800\nLIBRARY\n\nEXPORT {VIRTUAL} ortho_detail.dcl\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
