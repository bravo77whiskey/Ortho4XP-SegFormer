"""Build DDS atlases with strip-clamped mip chains (rainbow-roof fix).

X-Plane generates mipmaps for PNG textures by plain box-filtering; at deep
mip levels a horizontal strip atlas collapses and neighboring strips bleed
into each other. On large roofs in perspective the mip level varies across
the surface, so each mip-level contour renders as a differently-colored
ring -- pastel rainbow rectangles following the roof outline.

Fix: pre-build the mip chain so every level is downsampled WITHIN its strip
only (each output row belongs to exactly one strip and averages only that
strip's pixels). X-Plane prefers <name>.dds over <name>.png automatically,
so dropping these files next to the PNGs fixes every distance ring without
touching a single OBJ.

The DDS is uncompressed BGRA8 with a full mip chain (~5.6 MB per flavor).

Run:  python atlas_dds.py --output <pkg>
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

DDSD_FLAGS = (0x1 | 0x2 | 0x4 | 0x8 | 0x1000 | 0x20000)  # caps|h|w|pitch|pf|mips
DDPF_RGB_ALPHA = 0x40 | 0x1
DDSCAPS = 0x1000 | 0x400000 | 0x8  # texture | mipmap | complex


def _dds_header(width: int, height: int, mip_count: int) -> bytes:
    pf = struct.pack(
        "<2I4s5I",
        32, DDPF_RGB_ALPHA, b"\x00\x00\x00\x00", 32,
        0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000,
    )
    header = struct.pack(
        "<4s7I44x", b"DDS ", 124, DDSD_FLAGS, height, width,
        width * 4, 0, mip_count,
    )
    return header + pf + struct.pack("<5I", DDSCAPS, 0, 0, 0, 0)


def _strip_rows(layout: dict, size: int):
    """Return per-strip (y0, y1) pixel rows at the base level."""
    return sorted(
        tuple(strip["px"]) for strip in layout["strips"].values()
    )


def build_mips(base: np.ndarray, strips):
    """Mip chain where each output row averages only its own strip."""
    h, w = base.shape[:2]
    mips = [base]
    level = 1
    while max(h >> level, 1) >= 1 and max(w >> level, 1) >= 1:
        mh, mw = max(h >> level, 1), max(w >> level, 1)
        mip = np.zeros((mh, mw, 4), dtype=np.uint8)
        scale = h / mh
        x_idx = (np.arange(mw + 1) * (w / mw)).astype(int)
        for row in range(mh):
            yc = (row + 0.5) * scale
            # strip whose base range contains this row's center
            y0, y1 = strips[0]
            for s0, s1 in strips:
                if s0 <= yc < s1:
                    y0, y1 = s0, s1
                    break
            else:
                # past the last strip edge: clamp to nearest
                y0, y1 = strips[-1] if yc >= strips[-1][1] else strips[0]
            band = base[y0:y1].astype(np.float64)
            for col in range(mw):
                window = band[:, x_idx[col]: max(x_idx[col + 1], x_idx[col] + 1)]
                mip[row, col] = window.mean(axis=(0, 1)).round()
        mips.append(mip)
        if mh == 1 and mw == 1:
            break
        level += 1
    return mips


def write_dds(png_path: str, layout: dict) -> str:
    img = Image.open(png_path).convert("RGBA")
    base = np.asarray(img)
    h, w = base.shape[:2]
    strips = _strip_rows(layout, h)
    mips = build_mips(base, strips)
    dds_path = os.path.splitext(png_path)[0] + ".dds"
    with open(dds_path, "wb") as fh:
        fh.write(_dds_header(w, h, len(mips)))
        for mip in mips:
            bgra = mip[..., [2, 1, 0, 3]]
            fh.write(bgra.tobytes())
    return dds_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout", default=os.path.join(HERE, "atlas_layout.json")
    )
    parser.add_argument("--output", required=True,
                        help="Library package folder (textures/*.png).")
    args = parser.parse_args(argv)

    with open(args.layout, "r", encoding="utf-8") as fh:
        layout = json.load(fh)
    textures_dir = os.path.join(args.output, "textures")
    count = 0
    for name in sorted(os.listdir(textures_dir)):
        if name.lower().endswith(".png"):
            dds = write_dds(os.path.join(textures_dir, name), layout)
            print(f"wrote {dds}")
            count += 1
    print(f"{count} DDS atlases with strip-clamped mips")
    return 0


if __name__ == "__main__":
    sys.exit(main())
