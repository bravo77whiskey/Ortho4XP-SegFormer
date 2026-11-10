"""Oblique / top-down preview of a vegetation overlay over its ortho texture.

Not X-Plane: an approximation to judge forest edges by eye without the sim.
It reads the overlay's DSF text, plants every forest polygon the way a .for
says (SPACING grid, RANDOM jitter, one tree per layer per cell kept with
probability density/255, group by percentage, tree by frequency, height
uniform in [min, max], 3-d mesh scaled by height / nominal), draws X-Plane's
own 3-d tree meshes and albedo atlases (alpha tested, sun + shadow map) on a
ground mesh textured with the ortho, and saves a PNG.

Run with a Python that has moderngl, numpy, pillow, opencv:
  python forest_preview.py --dsf-txt X_review.txt --texture T.jpg --bbox lat_s lon_w lat_n lon_e
      --window x0 y0 x1 y1 --xp "<X-Plane>/Resources/default scenery/1200 forests"
      --pack <O4XP_Forest_Heights> --out out.png [--az 200 --elev 12 --dist 260]
"""

import argparse
import math
import os
import re
import sys

import cv2
import moderngl
import numpy as np
from PIL import Image

R_EARTH = 6378137.0
FLIP_V = os.environ.get("PREVIEW_FLIP_V", "1") == "1"


def merc_y(lat):
    return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def inv_merc_y(y):
    return math.degrees(2 * math.atan(math.exp(y)) - math.pi / 2)


# ── DSF text ──────────────────────────────────────────────────────────────────
def parse_dsf_text(path):
    defs, polys, cur = [], [], None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            if p[0] == "POLYGON_DEF":
                defs.append(p[1])
            elif p[0] == "BEGIN_POLYGON":
                cur = {"def": defs[int(p[1])], "param": int(float(p[2])), "rings": []}
            elif p[0] == "BEGIN_WINDING":
                cur["rings"].append([])
            elif p[0] == "POLYGON_POINT":
                cur["rings"][-1].append((float(p[1]), float(p[2])))
            elif p[0] == "END_POLYGON":
                polys.append(cur)
                cur = None
    return polys


# ── Library resolution ────────────────────────────────────────────────────────
def read_library(lib_dir, season="sum"):
    out = {}
    path = os.path.join(lib_dir, "library.txt")
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.split()
            if len(p) == 4 and p[0] == "EXPORT_SEASON" and p[1] == season:
                out[p[2]] = os.path.join(lib_dir, p[3])
            elif len(p) == 3 and p[0] == "EXPORT":
                out.setdefault(p[1], os.path.join(lib_dir, p[2]))
    return out


# ── .for parsing ──────────────────────────────────────────────────────────────
class Forest:
    def __init__(self, path):
        self.path = path
        self.spacing = (10.0, 10.0)
        self.random = (0.0, 0.0)
        self.meshes = {}          # name -> (verts float32 [n, 8], idx uint32)
        self.layers = {}          # layer -> [(pct, [tree, ...])]
        self.texture = None
        shader3d = False
        cur_mesh, verts, idx = None, [], []
        group = None
        tree = None
        with open(path, encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()

        def close_mesh():
            if cur_mesh is not None:
                self.meshes[cur_mesh] = (np.array(verts, np.float32).reshape(-1, 8),
                                         np.array(idx, np.uint32))

        for line in lines:
            p = line.split()
            if not p:
                continue
            k = p[0]
            if k == "VERTEX" and cur_mesh is not None:
                verts.extend(float(v) for v in p[1:9])
                continue
            if k == "IDX" and cur_mesh is not None:
                idx.extend(int(v) for v in p[1:])
                continue
            if k == "MESH":
                close_mesh()
                cur_mesh, verts, idx = p[1], [], []
                continue
            close_mesh()
            cur_mesh = None
            if k == "SHADER_3D":
                shader3d = True
            elif k == "SHADER_2D":
                shader3d = False
            elif k == "TEXTURE" and shader3d and self.texture is None:
                self.texture = os.path.normpath(os.path.join(os.path.dirname(path), p[1]))
            elif k == "SPACING":
                self.spacing = (float(p[1]), float(p[2]))
            elif k == "RANDOM":
                self.random = (float(p[1]), float(p[2]))
            elif k == "GROUP":
                group = (float(p[2]), [])
                self.layers.setdefault(int(p[1]), []).append(group)
            elif k in ("TREE", "TREE2"):
                if k == "TREE":
                    freq, lo, hi, layer, nom = float(p[6]), float(p[7]), float(p[8]), int(p[10]), None
                else:
                    freq, lo, hi, nom, layer = float(p[6]), float(p[7]), float(p[8]), float(p[9]), int(p[12])
                tree = {"freq": abs(freq), "lo": lo, "hi": hi, "nom": nom or lo, "mesh": None,
                        "name": p[-1]}
                if group is None:
                    group = (100.0, [])
                    self.layers.setdefault(layer, []).append(group)
                group[1].append(tree)
            elif k == "MESH_3D" and tree is not None:
                if tree["mesh"] is None and not p[1].startswith("lod_"):
                    tree["mesh"] = p[1]
        close_mesh()

    def pick(self, layer, rng):
        groups = self.layers.get(layer) or []
        if not groups:
            return None
        w = np.array([g[0] for g in groups], float)
        g = groups[rng.choice(len(groups), p=w / w.sum())]
        trees = [t for t in g[1] if t["mesh"]]
        if not trees:
            return None
        w = np.array([t["freq"] for t in trees], float) + 1e-6
        return trees[rng.choice(len(trees), p=w / w.sum())]


def load_texture_rgba(path):
    for cand in (path, os.path.splitext(path)[0] + ".dds", os.path.splitext(path)[0] + ".png"):
        if os.path.isfile(cand):
            img = Image.open(cand).convert("RGBA")
            return np.asarray(img)
    raise FileNotFoundError(path)


# ── Geometry helpers ──────────────────────────────────────────────────────────
def point_in_poly(xs, ys, ring):
    """Even-odd test of points (xs, ys) against ring [(x, y), ...]."""
    inside = np.zeros(xs.shape, bool)
    n = len(ring)
    for i in range(n):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % n]
        cond = (y0 > ys) != (y1 > ys)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = x0 + (ys - y0) * (x1 - x0) / (y1 - y0)
        inside ^= cond & (xs < xint)
    return inside


# ── Shaders ───────────────────────────────────────────────────────────────────
TREE_VS = """
#version 330
in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv;
in vec4 i_pos_scale; in float i_rot;
uniform mat4 mvp; uniform mat4 light_mvp;
out vec3 v_nrm; out vec2 v_uv; out vec4 v_light; out vec3 v_world;
void main() {
    float c = cos(i_rot), s = sin(i_rot);
    vec3 p = in_pos * i_pos_scale.w;
    p = vec3(c * p.x - s * p.z, p.y, s * p.x + c * p.z) + i_pos_scale.xyz;
    vec3 n = vec3(c * in_nrm.x - s * in_nrm.z, in_nrm.y, s * in_nrm.x + c * in_nrm.z);
    v_nrm = n; v_uv = in_uv; v_world = p;
    v_light = light_mvp * vec4(p, 1.0);
    gl_Position = mvp * vec4(p, 1.0);
}
"""

TREE_FS = """
#version 330
in vec3 v_nrm; in vec2 v_uv; in vec4 v_light; in vec3 v_world;
uniform sampler2D tex; uniform sampler2D shadow; uniform vec3 sun; uniform vec3 cam;
uniform int depth_only;
out vec4 f;
float lit(vec4 lp) {
    vec3 q = lp.xyz / lp.w * 0.5 + 0.5;
    if (q.x < 0.0 || q.x > 1.0 || q.y < 0.0 || q.y > 1.0) return 1.0;
    float s = 0.0;
    for (int dx = -1; dx <= 1; dx++) for (int dy = -1; dy <= 1; dy++) {
        float d = texture(shadow, q.xy + vec2(dx, dy) / 4096.0).r;
        s += (q.z - 0.0015 > d) ? 0.0 : 1.0;
    }
    return s / 9.0;
}
void main() {
    vec4 c = texture(tex, v_uv);
    if (c.a < 0.5) discard;
    if (depth_only == 1) { f = vec4(1.0); return; }
    vec3 n = normalize(v_nrm);
    float ndl = abs(dot(n, sun));           // leaves: lit through both faces
    float sh = lit(v_light);
    vec3 col = c.rgb * (0.38 + 0.72 * ndl * sh);
    float fog = clamp(length(v_world - cam) / 6000.0, 0.0, 0.5);
    f = vec4(mix(col, vec3(0.70, 0.76, 0.84), fog), 1.0);
}
"""

GROUND_VS = """
#version 330
in vec3 in_pos; in vec2 in_uv;
uniform mat4 mvp; uniform mat4 light_mvp;
out vec2 v_uv; out vec4 v_light; out vec3 v_world;
void main() { v_uv = in_uv; v_world = in_pos; v_light = light_mvp * vec4(in_pos, 1.0);
              gl_Position = mvp * vec4(in_pos, 1.0); }
"""

GROUND_FS = """
#version 330
in vec2 v_uv; in vec4 v_light; in vec3 v_world;
uniform sampler2D tex; uniform sampler2D shadow; uniform vec3 sun; uniform vec3 cam;
out vec4 f;
void main() {
    vec3 q = v_light.xyz / v_light.w * 0.5 + 0.5;
    float s = 0.0;
    for (int dx = -1; dx <= 1; dx++) for (int dy = -1; dy <= 1; dy++) {
        float d = texture(shadow, q.xy + vec2(dx, dy) / 4096.0).r;
        s += (q.z - 0.0015 > d) ? 0.0 : 1.0;
    }
    s /= 9.0;
    vec3 col = texture(tex, v_uv).rgb * (0.55 + 0.45 * s);
    float fog = clamp(length(v_world - cam) / 6000.0, 0.0, 0.5);
    f = vec4(mix(col, vec3(0.70, 0.76, 0.84), fog), 1.0);
}
"""


def perspective(fovy, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy) / 2)
    m = np.zeros((4, 4), np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


def ortho(l, r, b, t, n, f):
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = 2 / (r - l); m[1, 1] = 2 / (t - b); m[2, 2] = -2 / (f - n)
    m[0, 3] = -(r + l) / (r - l); m[1, 3] = -(t + b) / (t - b); m[2, 3] = -(f + n) / (f - n)
    return m


def look_at(eye, target, up=(0, 1, 0)):
    eye, target, up = (np.asarray(v, np.float32) for v in (eye, target, up))
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    if abs(float(np.dot(fwd, up))) > 0.999:
        up = np.array([0, 0, -1], np.float32)
    side = np.cross(fwd, up)
    side /= np.linalg.norm(side)
    u = np.cross(side, fwd)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = side, u, -fwd
    m[:3, 3] = -m[:3, :3] @ eye
    return m


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsf-txt", required=True)
    ap.add_argument("--texture", required=True)
    ap.add_argument("--bbox", nargs=4, type=float, required=True, help="lat_s lon_w lat_n lon_e")
    ap.add_argument("--window", nargs=4, type=float, required=True, help="x0 y0 x1 y1 texture px")
    ap.add_argument("--xp", required=True)
    ap.add_argument("--pack", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--az", type=float, default=200.0, help="camera bearing FROM target, degrees")
    ap.add_argument("--elev", type=float, default=12.0)
    ap.add_argument("--dist", type=float, default=260.0)
    ap.add_argument("--fov", type=float, default=45.0)
    ap.add_argument("--size", nargs=2, type=int, default=[1600, 900])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--skip", nargs="*", default=[], help="path substrings to leave out")
    ap.add_argument("--only-target-zoom", type=float, default=0.0,
                    help="frame the target at this many metres across instead of --dist (top views)")
    args = ap.parse_args()

    lat_s, lon_w, lat_n, lon_e = args.bbox
    tex_img = cv2.imread(args.texture)[:, :, ::-1]
    th, tw = tex_img.shape[:2]
    x0, y0, x1, y1 = args.window
    ym_n, ym_s = merc_y(lat_n), merc_y(lat_s)

    def px_to_ll(x, y):
        lon = lon_w + x / tw * (lon_e - lon_w)
        lat = inv_merc_y(ym_n - y / th * (ym_n - ym_s))
        return lon, lat

    lon0, lat0 = px_to_ll((x0 + x1) / 2, (y0 + y1) / 2)
    kx = math.cos(math.radians(lat0)) * math.pi / 180 * R_EARTH
    ky = math.pi / 180 * R_EARTH

    def ll_to_xz(lon, lat):   # x east, z south (OpenGL: -z is north)
        return (lon - lon0) * kx, -(lat - lat0) * ky

    wlon0, wlat0 = px_to_ll(x0, y1)
    wlon1, wlat1 = px_to_ll(x1, y0)
    wx0, wz1 = ll_to_xz(wlon0, wlat0)
    wx1, wz0 = ll_to_xz(wlon1, wlat1)

    libs = read_library(args.xp)
    if args.pack:
        libs.update(read_library(args.pack))

    rng = np.random.default_rng(args.seed)
    forests = {}
    instances = {}   # (texture, mesh key) -> list of [x, y, z, scale, rot]
    mesh_data = {}
    missing = set()
    n_trees = {}
    for poly in parse_dsf_text(args.dsf_txt):
        path = poly["def"]
        if not path.endswith(".for") or any(s in path for s in args.skip):
            continue
        rings = [[ll_to_xz(lo, la) for lo, la in r] for r in poly["rings"] if r]
        if not rings:
            continue
        allp = np.array([p for r in rings for p in r])
        if (allp[:, 0].max() < wx0 - 20 or allp[:, 0].min() > wx1 + 20
                or allp[:, 1].max() < wz0 - 20 or allp[:, 1].min() > wz1 + 20):
            continue
        fpath = libs.get(path)
        if not fpath or not os.path.isfile(fpath):
            missing.add(path)
            continue
        if fpath not in forests:
            forests[fpath] = Forest(fpath)
        fo = forests[fpath]
        dens = (poly["param"] % 256) / 255.0
        mode = poly["param"] // 256
        pts_by_layer = {}
        if mode == 2:     # point mode: one tree per point
            pts = np.array([p for r in rings for p in r])
            pts_by_layer[1] = pts
        elif mode == 1:   # line mode: trees along every winding
            pts = []
            for r in rings:
                for (ax, az), (bx, bz) in zip(r, r[1:]):
                    seg = math.hypot(bx - ax, bz - az)
                    n = max(1, int(seg / fo.spacing[0]))
                    for t in (np.arange(n) + rng.random(n)) / n:
                        pts.append((ax + (bx - ax) * t, az + (bz - az) * t))
            for layer in fo.layers:
                pts_by_layer[layer] = np.array(pts) if pts else np.zeros((0, 2))
        else:             # area: grid cells inside the outline, outside its holes
            sx, sz = fo.spacing
            gx = np.arange(math.floor(max(allp[:, 0].min(), wx0 - 30) / sx) * sx,
                           min(allp[:, 0].max(), wx1 + 30), sx)
            gz = np.arange(math.floor(max(allp[:, 1].min(), wz0 - 30) / sz) * sz,
                           min(allp[:, 1].max(), wz1 + 30), sz)
            if not len(gx) or not len(gz):
                continue
            GX, GZ = np.meshgrid(gx + sx / 2, gz + sz / 2)
            GX, GZ = GX.ravel(), GZ.ravel()
            for layer in fo.layers:
                jx = GX + (rng.random(GX.shape) - 0.5) * fo.random[0]
                jz = GZ + (rng.random(GZ.shape) - 0.5) * fo.random[1]
                keep = rng.random(GX.shape) < dens
                ins = point_in_poly(jx, jz, rings[0])
                for hole in rings[1:]:
                    ins &= ~point_in_poly(jx, jz, hole)
                keep &= ins
                pts_by_layer[layer] = np.stack([jx[keep], jz[keep]], 1)
        for layer, pts in pts_by_layer.items():
            for x, z in pts:
                if not (wx0 - 30 <= x <= wx1 + 30 and wz0 - 30 <= z <= wz1 + 30):
                    continue
                tree = fo.pick(layer, rng)
                if tree is None:
                    continue
                h = rng.uniform(tree["lo"], tree["hi"])
                key = (fo.texture, fpath, tree["mesh"])
                if key not in mesh_data:
                    mesh_data[key] = fo.meshes.get(tree["mesh"])
                if mesh_data[key] is None:
                    continue
                instances.setdefault(key, []).append((x, 0.0, z, h / max(tree["nom"], 0.1),
                                                      rng.uniform(0, 2 * math.pi)))
                n_trees[path] = n_trees.get(path, 0) + 1
    if missing:
        print("not found:", ", ".join(sorted(missing)[:8]), file=sys.stderr)
    print("trees:", sum(n_trees.values()), " by path:",
          ", ".join(f"{k.split('/')[-2]}/{k.split('/')[-1]}={v}" for k, v in sorted(n_trees.items())))

    # ── GL ──
    ctx = moderngl.create_standalone_context()
    W, H = args.size
    fbo = ctx.simple_framebuffer((W, H), samples=4)
    out_fbo = ctx.simple_framebuffer((W, H))
    shadow_tex = ctx.depth_texture((4096, 4096))
    shadow_tex.compare_func = ""
    shadow_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    shadow_fbo = ctx.framebuffer(depth_attachment=shadow_tex)
    ctx.enable(moderngl.DEPTH_TEST)

    tree_prog = ctx.program(vertex_shader=TREE_VS, fragment_shader=TREE_FS)
    ground_prog = ctx.program(vertex_shader=GROUND_VS, fragment_shader=GROUND_FS)

    # Ground: a grid over the window plus margin, ortho UVs by Web Mercator.
    gx0, gx1, gz0, gz1 = wx0 - 400, wx1 + 400, wz0 - 400, wz1 + 400
    N = 96
    verts = []
    for j in range(N + 1):
        for i in range(N + 1):
            x = gx0 + (gx1 - gx0) * i / N
            z = gz0 + (gz1 - gz0) * j / N
            lon = lon0 + x / kx
            lat = lat0 - z / ky
            u = (lon - lon_w) / (lon_e - lon_w)
            v = (ym_n - merc_y(lat)) / (ym_n - ym_s)
            verts.append((x, 0.0, z, u, v))
    verts = np.array(verts, np.float32)
    idx = []
    for j in range(N):
        for i in range(N):
            a = j * (N + 1) + i
            idx += [a, a + 1, a + N + 1, a + 1, a + N + 2, a + N + 1]
    gvbo = ctx.buffer(verts.tobytes())
    gibo = ctx.buffer(np.array(idx, np.uint32).tobytes())
    gvao = ctx.vertex_array(ground_prog, [(gvbo, "3f 2f", "in_pos", "in_uv")], gibo)
    gtex = ctx.texture((tw, th), 3, np.ascontiguousarray(tex_img).tobytes())
    gtex.build_mipmaps()
    gtex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    gtex.anisotropy = 16.0
    gtex.repeat_x = gtex.repeat_y = False

    tex_cache = {}

    def tree_texture(path):
        if path not in tex_cache:
            rgba = (load_texture_rgba(path)[::-1] if FLIP_V else load_texture_rgba(path)).copy()
            t = ctx.texture((rgba.shape[1], rgba.shape[0]), 4, rgba.tobytes())
            t.build_mipmaps()
            t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            tex_cache[path] = t
        return tex_cache[path]

    draws = []
    for key, inst in instances.items():
        texpath, _f, _m = key
        v, ix = mesh_data[key]
        vbo = ctx.buffer(np.ascontiguousarray(v[:, :8]).tobytes())
        ibo = ctx.buffer(ix.tobytes())
        ibuf = ctx.buffer(np.array(inst, np.float32).tobytes())
        vao = ctx.vertex_array(tree_prog, [(vbo, "3f 3f 2f", "in_pos", "in_nrm", "in_uv"),
                                           (ibuf, "4f 1f/i", "i_pos_scale", "i_rot")], ibo)
        draws.append((vao, tree_texture(texpath), len(inst)))

    # Sun: from the south-west-ish, 45 degrees up.
    sun = np.array([-0.45, 0.72, 0.53], np.float32)
    sun /= np.linalg.norm(sun)
    centre = np.array([(wx0 + wx1) / 2, 0.0, (wz0 + wz1) / 2], np.float32)
    span = max(wx1 - wx0, wz1 - wz0) + 900
    light_view = look_at(centre + sun * 1500, centre)
    light_mvp = ortho(-span / 2, span / 2, -span / 2, span / 2, 1.0, 3500.0) @ light_view

    az = math.radians(args.az)
    el = math.radians(args.elev)
    target = centre.copy()
    if args.only_target_zoom:
        dist = args.only_target_zoom / 2 / math.tan(math.radians(args.fov) / 2)
    else:
        dist = args.dist
    eye = target + dist * np.array([math.sin(az) * math.cos(el), math.sin(el),
                                    -math.cos(az) * math.cos(el)], np.float32)
    view = look_at(eye, target)
    proj = perspective(args.fov, W / H, 1.0, 8000.0)
    mvp = proj @ view

    def set_common(prog, depth_only):
        prog["mvp"].write((light_mvp if depth_only else mvp).T.copy().tobytes())
        prog["light_mvp"].write(light_mvp.T.copy().tobytes())
        if "sun" in prog:
            prog["sun"].value = tuple(sun)
        if "cam" in prog:
            prog["cam"].value = tuple(eye)

    # Shadow pass.
    shadow_fbo.use()
    shadow_fbo.clear(depth=1.0)
    set_common(tree_prog, True)
    tree_prog["depth_only"].value = 1
    for vao, tex, n in draws:
        tex.use(0)
        tree_prog["tex"].value = 0
        vao.render(instances=n)

    # Colour pass.
    fbo.use()
    fbo.clear(0.70, 0.76, 0.84, 1.0)
    shadow_tex.use(1)
    set_common(ground_prog, False)
    gtex.use(0)
    ground_prog["tex"].value = 0
    ground_prog["shadow"].value = 1
    gvao.render()
    set_common(tree_prog, False)
    tree_prog["depth_only"].value = 0
    tree_prog["shadow"].value = 1
    for vao, tex, n in draws:
        tex.use(0)
        tree_prog["tex"].value = 0
        vao.render(instances=n)
    ctx.copy_framebuffer(out_fbo, fbo)
    img = np.frombuffer(out_fbo.read(components=3), np.uint8).reshape(H, W, 3)[::-1]
    Image.fromarray(img).save(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
