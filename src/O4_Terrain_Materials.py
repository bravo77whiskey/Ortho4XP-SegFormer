"""Material maps for the orthophoto terrain (X-Plane 12).

X-Plane 12 lights an orthophoto as plain colour: no normal map, maximum
roughness and no base reflectance, so sunlight can only brighten it.  Every
land texture gets a companion map in X-Plane's normal-map layout, referenced
from its .ter file:

  R, G   tangent-space normal X, Y (128 = flat) - micro-relief from a
         luminance high-pass of the orthophoto
  B      base reflectance F0 (the .ter says NORMAL_METALNESS)
  A      gloss (255 = smooth, 0 = rough)

Surface types are estimated from the orthophoto colours with soft weights (no
hard class edges): vegetation (green excess), asphalt (dark grey), concrete
(mid grey), bright roofs and markings (bright, unsaturated) and soil
(everything else).

The map is <texture stem>_mat.dds next to the texture (BC3, so the gloss in
alpha survives), 2048 px for ZL17 and up, 1024 px below.  The land .ter gets,
right after its BASE_TEX_NOWRAP line:

    TEXTURE_NORMAL 1 ../textures/<texture stem>_mat.dds
    GLOBAL_specular 1.0
    NORMAL_METALNESS

Step 3 writes the map whenever it converts a texture (from the RGB it has in
hand) and finalize_tile() fills in the maps of the textures it kept.
retrofit_tile() adds maps and .ter lines to a tile built before this existed
(scripts/add_terrain_materials.py).

Imports nothing from Ortho4XP: the colour-correction worker uses it from the
overlay .venv python.
"""
import os
import re
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageFilter

RELIEF_DEFAULT = 0.15
MAT_SUFFIX = "_mat"

# per surface type: (gloss 0..1, F0)
MATERIALS = {
    "vegetation": (0.05, 0.00),
    "soil":       (0.10, 0.02),
    "asphalt":    (0.30, 0.04),
    "concrete":   (0.25, 0.04),
    "bright":     (0.40, 0.04),
}
_GLOSS = np.array([v[0] for v in MATERIALS.values()], np.float32)
_F0 = np.array([v[1] for v in MATERIALS.values()], np.float32)

# Maps carry a stamp in DDS_HEADER.dwReserved1[0..2] (like the colour
# correction stamp on the textures): version and relief, so a rebuild knows
# whether an existing map matches the settings.  Bump MARK_VERSION when the
# map itself changes.
MARK = b"O4MT"
MARK_VERSION = 1
_MARK_OFFSET = 32

TEXTURE_RE = re.compile(r"^(\d+)_(\d+)_([A-Za-z][A-Za-z0-9_]*?)(\d{2})\.dds$", re.IGNORECASE)
# Land .ter only: water / sea / overlay .ter names end in a suffix, not the ZL.
LAND_TER_RE = re.compile(r"^(\d+)_(\d+)_([A-Za-z][A-Za-z0-9_]*?)(\d{2})\.ter$", re.IGNORECASE)

_CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0


# ---------------------------------------------------------------------------
# names and .ter lines
# ---------------------------------------------------------------------------
def enabled(tile):
    return bool(getattr(tile, "terrain_materials", True))


def relief_of(tile):
    try:
        return max(0.0, float(getattr(tile, "terrain_materials_relief", RELIEF_DEFAULT)))
    except (TypeError, ValueError):
        return RELIEF_DEFAULT


def map_size(zoomlevel):
    return 2048 if int(zoomlevel) >= 17 else 1024


def map_file_name(texture_file_name):
    return os.path.splitext(texture_file_name)[0] + MAT_SUFFIX + ".dds"


def is_map_name(name):
    return name.lower().endswith(MAT_SUFFIX + ".dds")


def ter_lines(texture_file_name):
    return [
        "TEXTURE_NORMAL 1 ../textures/" + map_file_name(texture_file_name),
        "GLOBAL_specular 1.0",
        "NORMAL_METALNESS",
    ]


def _zoomlevel(texture_file_name):
    m = TEXTURE_RE.match(os.path.basename(texture_file_name))
    return int(m.group(4)) if m else 17


# ---------------------------------------------------------------------------
# the map
# ---------------------------------------------------------------------------
def _weights(rgb):
    """Unnormalised soft weights (veg, soil, asphalt, concrete, bright) and their sum."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1e-3)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    exg = (2 * g - r - b) / np.maximum(r + g + b, 1e-3)    # excess green, ~0 grey, >0.1 vegetation

    def ramp(x, a, b_):
        return np.clip((x - a) / (b_ - a), 0, 1)

    veg = ramp(exg, 0.03, 0.12)
    grey = (1 - ramp(sat, 0.16, 0.32)) * (1 - veg)
    bright = grey * ramp(lum, 0.62, 0.80)
    asphalt = grey * (1 - ramp(lum, 0.30, 0.45))
    concrete = np.clip(grey - bright - asphalt, 0, 1)
    soil = np.clip(1 - veg - grey, 0, 1)
    ws = [veg, soil, asphalt, concrete, bright]
    return ws, np.maximum(veg + soil + asphalt + concrete + bright, 1e-6)


def surface_weights(rgb):
    """Soft per-pixel weights for each surface type; rgb float 0..1, HxWx3."""
    ws, total = _weights(np.asarray(rgb, np.float32))
    return np.stack(ws, -1) / total[..., None]


def material_map(rgb, relief=RELIEF_DEFAULT, blur=1.5):
    """RGBA uint8 material map from an orthophoto (float 0..1, HxWx3)."""
    rgb = np.asarray(rgb, np.float32)
    ws, total = _weights(rgb)
    gloss = sum(w * v for w, v in zip(ws, _GLOSS)) / total
    f0 = sum(w * v for w, v in zip(ws, _F0)) / total
    del ws, total
    h, wd = rgb.shape[:2]
    if relief > 0:
        lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        img = Image.fromarray((lum * 255).astype(np.uint8))
        base = np.asarray(img.filter(ImageFilter.GaussianBlur(8)), np.float32) / 255
        fine = np.asarray(img.filter(ImageFilter.GaussianBlur(blur)), np.float32) / 255
        height = fine - base                                 # high-pass: local relief only
        gy, gx = np.gradient(height)
        nx, ny = -gx * (relief * 20), -gy * (relief * 20)
    else:
        nx = ny = np.zeros((h, wd), np.float32)
    nz = 1 / np.sqrt(1 + nx * nx + ny * ny)
    out = np.empty((h, wd, 4), np.uint8)
    for i, ch in enumerate(((nx * nz) * 0.5 + 0.5, (ny * nz) * 0.5 + 0.5, f0, gloss)):
        out[..., i] = np.clip(ch * 255 + 0.5, 0, 255)
    return out


def load_rgb(source, size):
    """float32 RGB at size x size from a path (jpg/png/dds), PIL image or uint8 array."""
    if isinstance(source, np.ndarray):
        im = Image.fromarray(np.ascontiguousarray(source[..., :3] if source.ndim == 3 else source))
    elif isinstance(source, Image.Image):
        im = source
    else:
        im = Image.open(source)
    if im.mode != "RGB":
        im = im.convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.LANCZOS)
    return np.asarray(im, np.float32) / 255


# ---------------------------------------------------------------------------
# DDS
# ---------------------------------------------------------------------------
def save_tga(arr, path):
    """Uncompressed 24/32-bit TGA, top-left origin (nvcompress honours it)."""
    h, w, c = arr.shape
    hdr = bytearray(18)
    hdr[2] = 2
    hdr[12:14] = w.to_bytes(2, "little")
    hdr[14:16] = h.to_bytes(2, "little")
    hdr[16] = 8 * c
    hdr[17] = 0x20 | (8 if c == 4 else 0)
    px = arr[..., [2, 1, 0, 3] if c == 4 else [2, 1, 0]]
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(np.ascontiguousarray(px).tobytes())


def _relief_code(relief):
    return int(round(float(relief) * 10000))


def map_mark(path):
    """(version, relief code) stamped into a material map, or None."""
    try:
        with open(path, "rb") as f:
            head = f.read(_MARK_OFFSET + 12)
    except OSError:
        return None
    if len(head) < _MARK_OFFSET + 12 or head[_MARK_OFFSET:_MARK_OFFSET + 4] != MARK:
        return None
    return struct.unpack_from("<II", head, _MARK_OFFSET + 4)


def _stamp(path, relief):
    with open(path, "r+b") as f:
        f.seek(_MARK_OFFSET)
        f.write(struct.pack("<4sII", MARK, MARK_VERSION, _relief_code(relief)))


def encode_dds(rgba, out_dds, converter, relief):
    """RGBA uint8 -> BC3 DDS with mipmaps, written atomically; True on success."""
    tool = os.path.basename(converter).lower()
    tmp_dds = out_dds + ".mat.tmp"
    if tool.startswith("nvcompress"):
        src = out_dds + ".mat.tga"
        save_tga(rgba, src)
        # Data, not colour: average the mipmaps linearly.
        cmd = [converter, "-bc3", "-fast", "-no-mip-gamma-correct", src, tmp_dds]
    else:                                               # macOS DDSTool
        src = out_dds + ".mat.png"
        Image.fromarray(rgba, "RGBA").save(src)
        cmd = [converter, "--png2dxt5", src, tmp_dds]
    ok = False
    try:
        for _ in range(3):
            rc = subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                                 creationflags=_CREATE_NO_WINDOW)
            if rc == 0 and os.path.isfile(tmp_dds):
                _stamp(tmp_dds, relief)
                os.replace(tmp_dds, out_dds)
                ok = True
                break
    finally:
        for f in (src, tmp_dds):
            try:
                os.remove(f)
            except OSError:
                pass
    return ok


def write_map(source, out_dds, zoomlevel, relief, converter):
    """Make the material map of one texture; True on success, never raises."""
    try:
        rgb = load_rgb(source, map_size(zoomlevel))
        return encode_dds(material_map(rgb, relief), out_dds, converter, relief)
    except Exception:
        for f in (out_dds + ".mat.tmp", out_dds + ".mat.tga", out_dds + ".mat.png"):
            try:
                os.remove(f)
            except OSError:
                pass
        return False


def map_is_current(map_path, texture_path, relief):
    """True when the map exists, matches the settings and is not older than its texture."""
    if map_mark(map_path) != (MARK_VERSION, _relief_code(relief)):
        return False
    try:
        return os.path.getmtime(map_path) >= os.path.getmtime(texture_path)
    except OSError:
        return False


# ---------------------------------------------------------------------------
# .ter files
# ---------------------------------------------------------------------------
def _split_ter(text):
    nl = "\r\n" if "\r\n" in text else "\n"
    return text.splitlines(), nl


def ter_base_texture(lines):
    for line in lines:
        if line.startswith("BASE_TEX_NOWRAP "):
            return line.split(None, 1)[1].strip()
    return None


def is_land_ter(name, lines):
    return bool(LAND_TER_RE.match(name)) and "WET" not in (l.strip() for l in lines)


def add_ter_lines(text, texture_file_name):
    """.ter text with the material lines after BASE_TEX_NOWRAP (unchanged if already there)."""
    lines, nl = _split_ter(text)
    if any(l.startswith("TEXTURE_NORMAL ") for l in lines):
        return text
    out = []
    for line in lines:
        out.append(line)
        if line.startswith("BASE_TEX_NOWRAP "):
            out.extend(ter_lines(texture_file_name))
    return nl.join(out) + nl


def strip_ter_lines(text, texture_file_name):
    """.ter text without the material lines this module writes."""
    lines, nl = _split_ter(text)
    ours = set(ter_lines(texture_file_name))
    return nl.join(l for l in lines if l.strip() not in ours) + nl


def _write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        f.write(text)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# whole tile
# ---------------------------------------------------------------------------
def finalize_tile(build_dir, relief, converter, workers=4, add_missing=False, prune=False,
                  make=True, should_stop=None, log=None):
    """Make every land .ter of a tile and its material map agree.

    A .ter with our TEXTURE_NORMAL line gets its map made when the map is
    missing, stale (older than the texture) or made with other settings; if
    that is impossible (no texture) the lines are taken out again.  With
    add_missing, a land .ter without any TEXTURE_NORMAL line gets a map and
    the lines (retrofit of tiles built before).  A .ter whose TEXTURE_NORMAL
    points elsewhere is left alone.  Idempotent.

    If a stale map cannot be remade the old one stays in use ("failed").

    With make=False (an interrupted build) nothing is made: any existing map
    is kept and the lines of a .ter without one are taken out.  Once
    should_stop() is true the remaining .ter files are handled that way.

    With prune, maps this module made that no .ter references any more (a
    texture used for water only, a texture gone) are deleted.

    Returns {"made", "kept", "added", "removed", "skipped", "failed",
    "pruned"} counts.
    """
    stats = dict(made=0, kept=0, added=0, removed=0, skipped=0, failed=0, pruned=0)
    ter_dir = os.path.join(build_dir, "terrain")
    tex_dir = os.path.join(build_dir, "textures")
    if not os.path.isdir(ter_dir):
        return stats
    jobs = []
    for name in sorted(os.listdir(ter_dir)):
        if not name.lower().endswith(".ter"):
            continue
        path = os.path.join(ter_dir, name)
        try:
            with open(path, "r", newline="") as f:
                text = f.read()
        except OSError:
            continue
        lines, _ = _split_ter(text)
        if not is_land_ter(name, lines):
            continue
        base = ter_base_texture(lines)
        if not base or not base.startswith("../textures/"):
            continue
        tex_name = base[len("../textures/"):]
        normal = [l for l in lines if l.startswith("TEXTURE_NORMAL ")]
        ours = ter_lines(tex_name)[0]
        if normal and normal[0].strip() != ours:
            stats["skipped"] += 1                    # someone else's map
            continue
        if not normal and not add_missing:
            continue
        jobs.append((path, text, tex_name, bool(normal)))

    def one(job):
        path, text, tex_name, has_lines = job
        tex_path = os.path.join(tex_dir, tex_name)
        map_path = os.path.join(tex_dir, map_file_name(tex_name))
        making = make and not (should_stop and should_stop())
        if map_is_current(map_path, tex_path, relief) or (not making and os.path.isfile(map_path)):
            result = "kept"
        elif making and os.path.isfile(tex_path) and write_map(tex_path, map_path, _zoomlevel(tex_name),
                                                     relief, converter):
            result = "made"
        elif os.path.isfile(map_path):
            return "failed"                          # an older map is better than none
        else:
            if has_lines:                            # never point X-Plane at a missing file
                _write_text(path, strip_ter_lines(text, tex_name))
                return "removed"
            return "failed"
        if not has_lines:
            _write_text(path, add_ter_lines(text, tex_name))
            return result + "+added"
        return result

    if jobs:
        with ThreadPoolExecutor(max(1, int(workers))) as ex:
            for i, res in enumerate(ex.map(one, jobs), 1):
                for part in res.split("+"):
                    stats[part] += 1
                if log and (i % 50 == 0 and i < len(jobs)):
                    log(f"   Terrain material maps: {i}/{len(jobs)} textures checked")
    if prune:
        stats["pruned"] = prune_tile(build_dir)
    return stats


def prune_tile(build_dir):
    """Delete the maps made here that no .ter of the tile references; returns the count."""
    ter_dir = os.path.join(build_dir, "terrain")
    tex_dir = os.path.join(build_dir, "textures")
    if not os.path.isdir(tex_dir):
        return 0
    referenced = set()
    if os.path.isdir(ter_dir):
        for name in os.listdir(ter_dir):
            if not name.lower().endswith(".ter"):
                continue
            try:
                with open(os.path.join(ter_dir, name), "r") as f:
                    for line in f:
                        if line.startswith("TEXTURE_NORMAL "):
                            referenced.add(os.path.basename(line.split()[-1]).lower())
            except OSError:
                continue
    n = 0
    for name in os.listdir(tex_dir):
        if not is_map_name(name) or name.lower() in referenced:
            continue
        path = os.path.join(tex_dir, name)
        if map_mark(path) is None:                       # not ours
            continue
        try:
            os.remove(path)
            n += 1
        except OSError:
            pass
    return n


def retrofit_tile(build_dir, relief=RELIEF_DEFAULT, converter=None, workers=4, log=None):
    """Add material maps and .ter lines to an already built tile, in place."""
    return finalize_tile(build_dir, relief, converter, workers=workers, add_missing=True, log=log)
