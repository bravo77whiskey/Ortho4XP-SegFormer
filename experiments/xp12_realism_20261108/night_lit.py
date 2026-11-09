"""Prototype: a night-light (TEXTURE_LIT) map for one Ortho4XP texture.

Why: Laminar's own urban terrain carries a far night texture
(TEXTURE_LIT .../uw9_FAR_LIT.dds, LIT_FADE_PARAMS 8192 7000 15000,
LUMINANCE 600), so distant cities glow where X-Plane's 3-d road lights
are no longer drawn. Ortho4XP .ter files have no TEXTURE_LIT, so an
ortho city goes dark beyond the 3-d light range.

This builds the map from data we already have: OSM roads (the tile's
cached *_all_roads.osm.bz2) and our building detections (review dump),
with the brightness statistics matched to Laminar's far LIT texture so
nothing is "turned up" relative to the stock look.

Usage: python night_lit.py <texture.jpg> <all_roads.osm.bz2> <review.json> <out_dir>
"""
import bz2, json, math, os, sys
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from PIL import Image

SIZE = 4096          # work resolution (texture pixels)
OUT = 1024           # LIT map size written (Laminar's far LIT is 1024)

# street-light "pool" per highway class: (half-width of lit band m, relative intensity)
ROAD = {
    "motorway": (18, 1.0), "trunk": (16, 1.0), "primary": (14, 0.9),
    "secondary": (12, 0.8), "tertiary": (10, 0.7),
    "motorway_link": (10, 0.8), "trunk_link": (10, 0.8), "primary_link": (9, 0.7),
    "secondary_link": (8, 0.6), "tertiary_link": (8, 0.6),
    "residential": (7, 0.45), "unclassified": (7, 0.4), "living_street": (6, 0.4),
    "service": (5, 0.25), "pedestrian": (6, 0.35),
}
# Laminar's uw9_FAR_LIT.dds, measured: mean (7.2, 5.2, 2.4), p99 (63, 44, 27) /255
TARGET_MEAN_R = 7.2
TARGET_P99_R = 63.0


def tex_geo(name):
    y, x, rest = os.path.basename(name).split("_")
    zl = int("".join(c for c in rest.split(".")[0] if c.isdigit())[-2:])
    return int(y), int(x), zl


def to_px(lat, lon, til_y, til_x, zl):
    n = 2 ** zl
    xt = (lon + 180.0) / 360.0 * n
    lr = math.radians(lat)
    yt = (1 - math.log(math.tan(lr) + 1 / math.cos(lr)) / math.pi) / 2 * n
    return (xt - til_x) * 256 * SIZE / 4096, (yt - til_y) * 256 * SIZE / 4096


def m_per_px(lat, zl):
    return 156543.03 * math.cos(math.radians(lat)) / 2 ** zl * 4096 / SIZE


def load_roads(path):
    nodes, ways = {}, []
    with bz2.open(path) as f:
        for ev, el in ET.iterparse(f, events=("end",)):
            if el.tag == "node":
                nodes[el.get("id")] = (float(el.get("lat")), float(el.get("lon")))
                el.clear()
            elif el.tag == "way":
                tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
                hw = tags.get("highway")
                if hw in ROAD:
                    ways.append((hw, [nd.get("ref") for nd in el.findall("nd")]))
                el.clear()
    return nodes, ways


def main(tex, osm, review, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    til_y, til_x, zl = tex_geo(tex)
    rgb = np.asarray(Image.open(tex).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR))
    lat_c = None
    nodes, ways = load_roads(osm)

    lum = np.zeros((SIZE, SIZE), np.float32)       # light pools (energy)
    core = np.zeros((SIZE, SIZE), np.float32)      # lamp rows (sharp)
    n_in = 0
    for hw, refs in ways:
        pts = [nodes[r] for r in refs if r in nodes]
        if len(pts) < 2:
            continue
        if lat_c is None:
            lat_c = pts[0][0]
        xy = np.array([to_px(la, lo, til_y, til_x, zl) for la, lo in pts], np.float32)
        if (xy[:, 0].max() < -50 or xy[:, 0].min() > SIZE + 50 or
                xy[:, 1].max() < -50 or xy[:, 1].min() > SIZE + 50):
            continue
        n_in += 1
        half_m, inten = ROAD[hw]
        mpp = m_per_px(lat_c, zl)
        w = max(1, int(round(2 * half_m / mpp)))
        poly = np.round(xy * 16).astype(np.int32)
        cv2.polylines(lum, [poly], False, inten, w, cv2.LINE_AA, shift=4)
        cv2.polylines(core, [poly], False, inten, max(1, int(round(3 / mpp))), cv2.LINE_AA, shift=4)
    mpp = m_per_px(lat_c, zl)

    # lamps every ~30 m along the core: modulate the core with a dotted pattern
    rng = np.random.default_rng(7)
    dots = (rng.random((SIZE // 8, SIZE // 8)) < 0.35).astype(np.float32)
    dots = cv2.resize(dots, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    core *= 0.5 + dots

    # building density: lit windows/signage spill + yards. Detections are in 4096 texture px.
    bld = np.zeros((SIZE, SIZE), np.float32)
    try:
        dets = json.load(open(review))["detections"]
        for d in dets:
            if d.get("f"):      # filtered out (below footprint, etc.)
                continue
            q = np.array(d["q"], np.float32) * SIZE / 4096
            cv2.fillPoly(bld, [np.round(q * 16).astype(np.int32)], 1.0, cv2.LINE_AA, shift=4)
    except Exception as e:
        print("no buildings:", e)
    sig = 120 / mpp
    density = cv2.GaussianBlur(bld, (0, 0), sig)
    density /= max(1e-6, np.percentile(density, 99.5))
    density = np.clip(density, 0, 1)

    # street-light pools: soft falloff; in dense areas streets are brighter
    pools = cv2.GaussianBlur(lum, (0, 0), 6 / mpp)
    # real night imagery: brightness follows activity, not road length. Rural/hill
    # roads keep only sparse lamps; dense districts glow (windows, signage, yards).
    act = np.clip(density, 0, 1) ** 0.7
    energy = (pools * (0.12 + 1.3 * act) + core * (0.15 + 0.8 * act)
              + 0.25 * act * cv2.GaussianBlur(bld, (0, 0), 4 / mpp))

    # water stays dark: ortho blue-dark + low texture → no light (harbour)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    water = ((hsv[..., 2] < 90) & (cv2.GaussianBlur(cv2.Laplacian(rgb[..., 1].astype(np.float32), cv2.CV_32F) ** 2, (0, 0), 6) < 30))
    energy[water & (lum < 0.05)] *= 0.2

    # colour: mix of sodium (2000 K) on minor roads, LED/metal-halide (4000 K) on major roads
    sodium = np.array([1.00, 0.62, 0.25], np.float32)
    led = np.array([1.00, 0.86, 0.68], np.float32)
    major = cv2.GaussianBlur((core > 0.6).astype(np.float32), (0, 0), 10 / mpp)
    mix = np.clip(major * 2, 0, 1)[..., None]
    col = energy[..., None] * ((1 - mix) * sodium + mix * led)

    # calibrate to Laminar's far LIT statistics (mean & p99 of red, over the whole texture)
    r = col[..., 0]
    # solve r' = a * r^g so mean and p99 match the target
    best = None
    for g in np.linspace(0.6, 1.6, 41):
        rg = np.power(np.maximum(r, 0), g)
        p99 = np.percentile(rg, 99)
        a = TARGET_P99_R / max(p99, 1e-6)
        err = abs(a * rg.mean() - TARGET_MEAN_R)
        if best is None or err < best[0]:
            best = (err, g, a)
    _, g, a = best
    lit = a * np.power(np.maximum(col, 0), g)
    lit = np.clip(lit, 0, 255)
    small = cv2.resize(lit, (OUT, OUT), interpolation=cv2.INTER_AREA)
    print(f"roads in texture: {n_in}  m/px(work) {mpp:.2f}  gamma {g:.2f}  "
          f"mean {small.reshape(-1,3).mean(0).round(1)}  p99 {np.percentile(small.reshape(-1,3),99,axis=0).round(0)}")
    stem = os.path.splitext(os.path.basename(tex))[0]
    Image.fromarray(small.astype(np.uint8)).save(os.path.join(out_dir, stem + "_LIT.png"))

    # preview: a crude night composite (moonlit ortho + emissive with bloom), and day ortho
    moon = (rgb.astype(np.float32) / 255) ** 2.2 * 0.015
    em = (lit / 255) ** 2.2 * 6.0
    bloom = cv2.GaussianBlur(em, (0, 0), 25 / mpp) * 0.6
    hdr = moon + em + bloom
    disp = np.clip(1.0 - np.exp(-hdr * 2.0), 0, 1) ** (1 / 2.2)
    prev = cv2.resize((disp * 255).astype(np.uint8), (2048, 2048), interpolation=cv2.INTER_AREA)
    Image.fromarray(prev).save(os.path.join(out_dir, stem + "_night_preview.jpg"), quality=90)
    Image.fromarray(cv2.resize(rgb, (2048, 2048), interpolation=cv2.INTER_AREA)).save(
        os.path.join(out_dir, stem + "_day.jpg"), quality=88)


if __name__ == "__main__":
    main(*sys.argv[1:5])
