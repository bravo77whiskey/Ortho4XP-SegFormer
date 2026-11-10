"""Stack preview renders into one labelled comparison image.

  python compose.py OUT.jpg "label=path.png" ["label=path.png" ...] [--crop-top PX]
"""

import sys

from PIL import Image, ImageDraw, ImageFont


def main():
    args = sys.argv[1:]
    crop_top = 0
    if "--crop-top" in args:
        i = args.index("--crop-top")
        crop_top = int(args[i + 1])
        del args[i:i + 2]
    out, items = args[0], [a.split("=", 1) for a in args[1:]]
    imgs = [(label, Image.open(path).convert("RGB")) for label, path in items]
    imgs = [(label, im.crop((0, crop_top, im.width, im.height))) for label, im in imgs]
    w = max(im.width for _l, im in imgs)
    h = sum(im.height for _l, im in imgs)
    sheet = Image.new("RGB", (w, h), "white")
    try:
        font = ImageFont.truetype("arialbd.ttf", 34)
    except OSError:
        font = ImageFont.load_default()
    y = 0
    for label, im in imgs:
        sheet.paste(im, (0, y))
        d = ImageDraw.Draw(sheet)
        d.rectangle((0, y, d.textlength(label, font=font) + 24, y + 50), fill=(0, 0, 0))
        d.text((12, y + 7), label, fill=(255, 255, 255), font=font)
        y += im.height
    sheet.save(out, quality=88)
    print("wrote", out)


if __name__ == "__main__":
    main()
