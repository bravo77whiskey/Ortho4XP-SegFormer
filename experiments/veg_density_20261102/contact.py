"""Contact sheet of a review set's textures, to pick test textures with mixed tree density."""
import glob, os, sys
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
folder, out, pattern = sys.argv[1], sys.argv[2], sys.argv[3]
files = sorted(glob.glob(os.path.join(folder, pattern)))
n = 10
s = 170
rows = (len(files) + n - 1) // n
sheet = Image.new("RGB", (n * s, rows * s), (20, 20, 20))
d = ImageDraw.Draw(sheet)
for i, f in enumerate(files):
    im = Image.open(f)
    im.draft("RGB", (s, s))
    im = im.convert("RGB").resize((s - 4, s - 4))
    x, y = (i % n) * s, (i // n) * s
    sheet.paste(im, (x + 2, y + 2))
    d.text((x + 4, y + 4), os.path.basename(f).replace("_BI", " ").replace(".jpg", ""), fill=(255, 255, 0))
sheet.save(out, quality=88)
print(len(files))
