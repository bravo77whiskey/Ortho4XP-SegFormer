"""Render the europe street with several texture packages and stack them.

    python photo/render_photo_compare.py --out <dir> \
        --pkg "Cycles bake=<pkg>" "AI image-gen=<pkg>" "CC0 photo=<pkg>"
        [--skip-render]

Identical OBJs per package (samples.py seeds), so only the pages differ.
Writes ``<dir>/<label slug>/<scene>.png`` and ``<dir>/compare_<scene>.png``
(one row per package, top to bottom in the order given).
"""

from __future__ import annotations

import argparse
import os
import re
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(V4, "image-gen"))
sys.path.insert(0, V4)

import render_compare  # noqa: E402
import samples  # noqa: E402


def slug(label):
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def stack(paths, labels, out):
    ims = [Image.open(p).convert("RGB") for p in paths]
    w, h = ims[0].size
    bar = 48
    sheet = Image.new("RGB", (w, len(ims) * (h + bar)), (24, 26, 30))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
    for i, (im, lab) in enumerate(zip(ims, labels)):
        y = i * (h + bar)
        d.text((16, y + 8), lab, fill=(235, 236, 240), font=font)
        sheet.paste(im, (0, y + bar))
    sheet.save(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pkg", nargs="+", required=True,
                    help='"label=package dir"')
    ap.add_argument("--region", default="europe")
    ap.add_argument("--skip-render", nargs="*", default=[],
                    help="labels whose renders already exist")
    ap.add_argument("--blender", default=samples.DEFAULT_BLENDER)
    args = ap.parse_args(argv)
    pkgs = [p.split("=", 1) for p in args.pkg]
    names = None
    for label, pkg in pkgs:
        d = os.path.join(args.out, slug(label))
        if label in args.skip_render:
            continue
        names = render_compare.render(pkg, args.region, d, args.blender)
    if names is None:
        d = os.path.join(args.out, slug(pkgs[0][0]))
        names = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    for n in names:
        paths = [os.path.join(args.out, slug(lab), n) for lab, _ in pkgs]
        if all(os.path.isfile(p) for p in paths):
            stack(paths, [lab for lab, _ in pkgs],
                  os.path.join(args.out, f"compare_{n}"))
            print("wrote", f"compare_{n}", flush=True)


if __name__ == "__main__":
    main()
