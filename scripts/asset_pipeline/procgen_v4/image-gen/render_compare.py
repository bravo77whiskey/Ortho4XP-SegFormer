"""Render the same europe street with two texture packages and pair them.

    python image-gen/render_compare.py --before <pkg> --after <pkg> --out <dir>
                                [--region europe] [--render before after]

Both packages get identical OBJs from samples.py (same seeds), so the only
difference between the paired images is the texture pages.  Adds two
close-up cameras (a house facade, the apartment/shop end) to the three
standard scenes, then writes ``compare_<scene>.png`` (before on top).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.dirname(HERE)
sys.path.insert(0, V4)

import samples  # noqa: E402

EXTRA = {
    "detail": {"loc": [-43, -4, 5.5], "look_at": [-33.8, 12, 3.5],
               "lens": 38},
    "apts": {"loc": [66, -3, 9], "look_at": [92, 10, 8], "lens": 28},
    "ind": {"loc": [12, 38, 9], "look_at": [-22, 62, 5], "lens": 30},
}


def scenes(pkg, region, out_dir):
    sc, _stats = samples.scenes_for(pkg, region, out_dir)
    base = dict(sc[0])
    for name, cam in EXTRA.items():
        sc.append(dict(base, camera=cam, res=[1920, 1080],
                       out=os.path.join(out_dir, f"{region}_{name}.png")))
    return sc


def render(pkg, region, out_dir, blender):
    os.makedirs(out_dir, exist_ok=True)
    sc = scenes(os.path.abspath(pkg), region, os.path.abspath(out_dir))
    path = os.path.join(out_dir, "scenes.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(sc, fh)
    subprocess.run([blender, "--background", "--factory-startup",
                    "--python-exit-code", "1", "--python",
                    os.path.join(V4, "render_preview.py"), "--", path],
                   check=True)
    return [os.path.basename(s["out"]) for s in sc]


def pair(before, after, out, labels=("Before: Cycles bake",
                                     "After: AI re-render")):
    a, b = Image.open(before).convert("RGB"), Image.open(after).convert("RGB")
    w, h = a.size
    sheet = Image.new("RGB", (w, 2 * h + 2 * 48), (24, 26, 30))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
    for i, (im, lab) in enumerate(((a, labels[0]), (b, labels[1]))):
        y = i * (h + 48)
        d.text((16, y + 8), lab, fill=(235, 236, 240), font=font)
        sheet.paste(im, (0, y + 48))
    sheet.save(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--region", default="europe")
    ap.add_argument("--render", nargs="*", default=["before", "after"])
    ap.add_argument("--blender", default=samples.DEFAULT_BLENDER)
    args = ap.parse_args(argv)
    names = None
    for tag, pkg in (("before", args.before), ("after", args.after)):
        d = os.path.join(args.out, tag)
        if tag in args.render:
            names = render(pkg, args.region, d, args.blender)
    names = names or sorted(f for f in os.listdir(os.path.join(args.out,
                                                                "after"))
                            if f.endswith(".png"))
    for n in names:
        bp = os.path.join(args.out, "before", n)
        apath = os.path.join(args.out, "after", n)
        if os.path.isfile(bp) and os.path.isfile(apath):
            pair(bp, apath, os.path.join(args.out, f"compare_{n}"))
            print("wrote", f"compare_{n}")


if __name__ == "__main__":
    main()
