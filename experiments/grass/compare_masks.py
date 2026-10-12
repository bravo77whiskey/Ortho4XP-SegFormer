"""Cross-tabulate the colour-rule ground mask against SegFormer classes.

Usage: python compare_masks.py --center LAT LON --radius-nm R --tile-dir DIR --out NAME
Uses classmaps_tile/<texture>_cls.npy (cached SegFormer output) for the same
textures make_region_pack.py selects.  Writes <NAME>_compare.json and a
mosaic preview <NAME>_compare.jpg:
  green   both say ground/vegetation
  yellow  colour rule plants, SegFormer says bare ground
  magenta colour rule plants, SegFormer says developed / building / road
  blue    SegFormer says vegetation, colour rule excludes it
"""
import argparse, json, math, os, re, sys
import numpy as np, cv2
from PIL import Image

sys.path.insert(0, r"G:\Dev\Ortho4XP-SegFormer\src")
import O4_Texture_Selection as TEXSEL

SPIKE = r"E:\Ortho4XP_Review\_grass_spike"
NAMES = {255: "none", 0: "background", 1: "bareland", 2: "rangeland", 3: "developed", 4: "road",
         5: "tree", 6: "water", 7: "agriculture", 8: "building"}
VEG = (2, 5, 7)
BUILT = (3, 4, 8)

ap = argparse.ArgumentParser()
ap.add_argument("--center", nargs=2, type=float, required=True)
ap.add_argument("--radius-nm", type=float, default=2.0)
ap.add_argument("--tile-dir", help="installed Ortho4XP tile: textures chosen by its cfg")
ap.add_argument("--texture-dir", help="plain folder of <y>_<x>_<PROV><ZL>.jpg/.dds textures")
ap.add_argument("--crop-m", type=float, default=1600.0,
                help="half-size of the centred close-up written to <out>_closeup.jpg")
ap.add_argument("--out", required=True)
ap.add_argument("--fused", action="store_true",
                help="map the combined rule: green = from SegFormer vegetation, "
                     "blue = rescued green pixels, magenta = colour-only pixels it drops")
ap.add_argument("--green-chroma", type=float, default=10.0)
ap.add_argument("--climate", action="store_true",
                help="with --fused: Koppen regime (static, cannot go stale) sets how much the "
                     "ortho must show -- desert: green pixels only; steppe: non-green vegetation sparse")
ap.add_argument("--wc-only", action="store_true",
                help="map WorldCover alone: green grass (tree/grassland/cropland/wetland), "
                     "yellow sparse (shrubland/moss), orange bare/snow, magenta built-up, cyan water")
ap.add_argument("--worldcover", action="store_true",
                help="with --fused: ESA WorldCover gate -- bare/snow/water no grass (orange), "
                     "shrubland/moss sparse grass (yellow); other classes leave the rule alone")
args = ap.parse_args()
lat0, lon0 = args.center
R = args.radius_nm * 1852
cos0 = math.cos(math.radians(lat0))


def my(la):
    return math.log(math.tan(math.pi / 4 + math.radians(la) / 2))


def inv_my(y):
    return math.degrees(2 * math.atan(math.exp(y)) - math.pi / 2)


def colour_ground_mask(L, mpp):          # identical to make_region_pack.py
    Ls = cv2.GaussianBlur(L, (0, 0), max(0.5, 1.0 / mpp))
    l, a, b = Ls[..., 0], Ls[..., 1], Ls[..., 2]
    C = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a)) % 360
    built = ((C < 8) & (l > 25)) | (l > 72) | ((C > 14) & ((hue < 60) | (hue > 200)))
    r = max(1, int(round(1.5 / mpp)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    built = cv2.morphologyEx(built.astype(np.uint8), cv2.MORPH_CLOSE, ker)
    return cv2.morphologyEx(1 - built, cv2.MORPH_OPEN, ker).astype(bool)


NAME_RE = re.compile(r"^(\d+)_(\d+)_([A-Za-z][A-Za-z0-9_]*)(\d{2})\.(dds|jpg)$", re.IGNORECASE)
if args.tile_dir:
    tex_dir = os.path.join(args.tile_dir, "textures")
    files, excluded, _ = TEXSEL.select_textures(os.listdir(tex_dir), tex_dir, math.floor(lat0), math.floor(lon0))
else:
    tex_dir = args.texture_dir
    files, excluded = sorted(f for f in os.listdir(tex_dir) if NAME_RE.match(f)), {}
model = None
dlat, dlon = R / 110574, R / (111320 * cos0)
WC = None
if args.worldcover or args.wc_only:
    import worldcover
    WC = worldcover.window(lat0 - dlat - 0.002, lat0 + dlat + 0.002, lon0 - dlon - 0.002, lon0 + dlon + 0.002,
                           os.path.join(SPIKE, "_worldcover"))
    u, n = np.unique(WC[0], return_counts=True)
    print("WorldCover classes in window (%):", {int(a): round(100 * b / n.sum(), 1) for a, b in zip(u, n)})
mz_n, mz_s, mw, me = my(lat0 + dlat), my(lat0 - dlat), lon0 - dlon, lon0 + dlon
MW = MH = int(2 * R / 4.0)                      # 4 m/px preview
canvas = np.zeros((MH, MW, 3), np.uint8)
tint = np.zeros((MH, MW), np.int8) - 1
area = {}                                        # (colour_ground, class) -> m2
for f in files:
    m = NAME_RE.match(f)
    ty, tx, zl = int(m.group(1)), int(m.group(2)), int(m.group(4))

    def g2w(x, y):
        rx = x / 2 ** (zl - 1) - 1; ry = 1 - y / 2 ** (zl - 1)
        return 360 / math.pi * math.atan(math.exp(math.pi * ry)) - 90, rx * 180
    lat_n, lon_w = g2w(tx, ty); lat_s, lon_e = g2w(tx + 16, ty + 16)
    cy = min(max(lat0, lat_s), lat_n); cx = min(max(lon0, lon_w), lon_e)
    if ((cy - lat0) * 110574) ** 2 + ((cx - lon0) * 111320 * cos0) ** 2 > R * R:
        continue
    cls_path = os.path.join(SPIKE, "classmaps_tile", f[:-4] + "_cls.npy")
    img = np.asarray(Image.open(os.path.join(tex_dir, f)).convert("RGB"))
    if args.wc_only:
        pass                                       # WorldCover only: no SegFormer
    elif not os.path.isfile(cls_path):
        import O4_Landcover_Inference as SEG
        if model is None:
            model, proc, dev = SEG.load_vegetation_model()
        np.save(cls_path, SEG.run_inference(model, dev, img, proc).astype(np.uint8))
    cls = np.zeros(img.shape[:2], np.uint8) if args.wc_only else np.load(cls_path)
    h, w = cls.shape
    mpp = (lon_e - lon_w) * 111320 * cos0 / w
    L = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    ground = colour_ground_mask(L, mpp)
    lons = lon_w + (np.arange(w) + .5) / w * (lon_e - lon_w)
    lats = np.array([inv_my(my(lat_n) + (y + .5) / h * (my(lat_s) - my(lat_n))) for y in range(h)])
    inside = (((lats - lat0) * 110574)[:, None] ** 2 + (((lons - lon0) * 111320 * cos0)[None, :]) ** 2) <= R * R
    owned = np.ones((h, w), bool)
    for fx0, fy0, fx1, fy1 in excluded.get(f, ()):
        owned[int(round(fy0 * h)):int(round(fy1 * h)), int(round(fx0 * w)):int(round(fx1 * w))] = False
    sel = inside & owned
    px_m2 = mpp * mpp
    for g in (True, False):
        mm = sel & (ground == g)
        u, n = np.unique(cls[mm], return_counts=True)
        for c, k in zip(u.tolist(), n.tolist()):
            area[(g, NAMES.get(c, str(c)))] = area.get((g, NAMES.get(c, str(c))), 0) + k * px_m2
    cat = np.full((h, w), -1, np.int8)
    veg = np.isin(cls, VEG)
    if args.fused:
        # SegFormer decides vegetation vs bare/built ground (it can tell
        # straw grass from sand); the colour rule vetoes built-looking pixels
        # and rescues clearly green ones SegFormer lumped into a non-veg class.
        Ls = cv2.GaussianBlur(L, (0, 0), max(0.5, 1.0 / mpp))
        Cc = np.hypot(Ls[..., 1], Ls[..., 2])
        hue = np.degrees(np.arctan2(Ls[..., 2], Ls[..., 1])) % 360
        green = (hue >= 95) & (hue <= 170) & (Cc >= args.green_chroma) & (Ls[..., 0] >= 20)
        soft = np.isin(cls, (0, 1, 3, 255))           # background, bareland, developed, no class
        from_veg = sel & ground & veg
        rescued = sel & ground & soft & green & ~veg
        dropped = sel & ground & ~from_veg & ~rescued
        for name, mm in (("fused: from SegFormer vegetation", from_veg),
                         ("fused: rescued green", rescued),
                         ("fused: colour-only, dropped", dropped)):
            area[(True, name)] = area.get((True, name), 0) + mm.sum() * px_m2
        cat[from_veg] = 0
        cat[rescued] = 3
        cat[dropped] = 2
        if args.climate:
            import O4_Climate_Regions as CLIM
            code = (CLIM.koppen_code(0.5 * (lat_n + lat_s), 0.5 * (lon_w + lon_e)) or "")
            if code.startswith("BW") or code.startswith("BS"):
                grass = from_veg | rescued
                not_green = grass & ~green
                if code.startswith("BW"):     # desert: only what the ortho shows as green
                    name, cval = "climate: desert, non-green removed", 5
                else:                         # steppe: non-green vegetation is sparse
                    name, cval = "climate: steppe, non-green sparse", 4
                area[(True, name)] = area.get((True, name), 0) + not_green.sum() * px_m2
                cat[not_green] = cval
            area[(True, f"climate zone {code}")] = area.get((True, f"climate zone {code}"), 0) + sel.sum() * px_m2
        if WC is not None:
            # Regional gate from Sentinel-2-based land cover: it sees
            # near-infrared, which tells living vegetation from soil where
            # RGB colour and SegFormer both fail (arid scrub, dry fields).
            wc = worldcover.sample(*WC, lats, lons)
            grass = from_veg | rescued
            wc_none = grass & np.isin(wc, (60, 70, 80, 95))
            wc_sparse = grass & np.isin(wc, (20, 100)) & ~wc_none
            for name, mm in (("worldcover: removed (bare/snow/water)", wc_none),
                             ("worldcover: sparse (shrubland/moss)", wc_sparse)):
                area[(True, name)] = area.get((True, name), 0) + mm.sum() * px_m2
            cat[wc_sparse] = 4
            cat[wc_none] = 5
    else:
        cat[sel & ground & veg] = 0
        cat[sel & ground & (cls == 1)] = 1
        cat[sel & ground & np.isin(cls, BUILT)] = 2
        cat[sel & ~ground & veg] = 3
    if args.wc_only:
        wc = worldcover.sample(*WC, lats, lons)
        cat = np.full((h, w), -1, np.int8)
        for code, codes, name in ((0, (10, 30, 40, 90), "wc-only: grass (tree/grassland/cropland/wetland)"),
                                  (4, (20, 100), "wc-only: sparse (shrubland/moss)"),
                                  (5, (60, 70), "wc-only: none (bare/snow)"),
                                  (2, (50,), "wc-only: built-up"),
                                  (6, (80, 95), "wc-only: water/mangrove")):
            mm = sel & np.isin(wc, codes)
            cat[mm] = code
            area[(True, name)] = area.get((True, name), 0) + mm.sum() * px_m2
    # paste into the 4 m/px canvas
    x0 = (lon_w - mw) / (me - mw) * MW; x1 = (lon_e - mw) / (me - mw) * MW
    y0 = (mz_n - my(lat_n)) / (mz_n - mz_s) * MH; y1 = (mz_n - my(lat_s)) / (mz_n - mz_s) * MH
    ix0, iy0, ix1, iy1 = max(0, int(x0)), max(0, int(y0)), min(MW, int(math.ceil(x1))), min(MH, int(math.ceil(y1)))
    if ix1 <= ix0 or iy1 <= iy0:
        continue
    xs = ((np.arange(ix0, ix1) + .5 - x0) / (x1 - x0) * w).astype(int).clip(0, w - 1)
    ys = ((np.arange(iy0, iy1) + .5 - y0) / (y1 - y0) * h).astype(int).clip(0, h - 1)
    own = owned[ys][:, xs]
    canvas[iy0:iy1, ix0:ix1][own] = img[ys][:, xs][own]
    tint[iy0:iy1, ix0:ix1][own] = cat[ys][:, xs][own]

colours = {0: (40, 200, 60), 1: (255, 210, 0), 2: (255, 0, 200), 3: (30, 120, 255),
           4: (255, 220, 0), 5: (255, 90, 0), 6: (0, 200, 255)}
out = canvas.copy()
for k, c in colours.items():
    mk = tint == k
    out[mk] = (0.45 * out[mk] + 0.55 * np.array(c)).astype(np.uint8)
Image.fromarray(np.hstack([canvas, out])).save(os.path.join(SPIKE, f"{args.out}_compare.jpg"), quality=88)
c = int(args.crop_m / 4.0)
y0c, y1c, x0c, x1c = max(0, MH // 2 - c), min(MH, MH // 2 + c), max(0, MW // 2 - c), min(MW, MW // 2 + c)
close = np.vstack([canvas[y0c:y1c, x0c:x1c], out[y0c:y1c, x0c:x1c]])
sc = 2000 / close.shape[1]
Image.fromarray(cv2.resize(close, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)).save(
    os.path.join(SPIKE, f"{args.out}_closeup.jpg"), quality=90)
rows = sorted(((g, c, round(a / 1e6, 3)) for (g, c), a in area.items()), key=lambda r: (not r[0], -r[2]))
json.dump(rows, open(os.path.join(SPIKE, f"{args.out}_compare.json"), "w"), indent=1)
print("colour rule PLANTS:")
for g, c, a in rows:
    if g: print(f"   {c:12s} {a:7.3f} km2")
print("colour rule EXCLUDES:")
for g, c, a in rows:
    if not g: print(f"   {c:12s} {a:7.3f} km2")
