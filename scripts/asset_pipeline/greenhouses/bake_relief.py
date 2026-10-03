"""Render relief scenes from ``relief.py`` (runs inside Blender).

    blender --background --factory-startup --python bake_relief.py -- jobs.json

jobs.json = {"samples": int, "jobs": [scene dict from Scene.save + "out"]}.
Each scene is seen by an orthographic camera straight down its z axis
(image up = +y) under an overcast dome whose bright side follows the
scene's ``view`` ('top': overhead; 'wall': toward +y; 'wall_x': toward +x).
Film layers mix a diffuse/glossy surface with a transparent one by their
``trans`` value, so light and the view reach the soil, crops and hoops
under the film.  Tiling scenes are extended past both ends by a period
margin so edge shading wraps.  Output: ``out``.npz with shaded (linear
RGB), normal (x, y, z), rough, metal.
"""

import json
import math
import os
import sys
import time

import bpy
import numpy as np
import OpenImageIO as oiio

MARGIN_M = 0.8


def _reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for dev in prefs.devices:
        dev.use = dev.type == "OPTIX"
    scene.cycles.device = "GPU"
    return scene


def _sky(scene, up, low=0.3):
    world = bpy.data.worlds.new("sky")
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes["Background"]
    bg.inputs["Color"].default_value = (1, 1, 1, 1)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    dot = nt.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    dot.inputs[1].default_value = up
    nt.links.new(tc.outputs["Generated"], dot.inputs[0])
    ma = nt.nodes.new("ShaderNodeMath")
    ma.operation = "MULTIPLY_ADD"
    ma.use_clamp = True
    ma.inputs[1].default_value = 0.75
    ma.inputs[2].default_value = 0.55
    nt.links.new(dot.outputs["Value"], ma.inputs[0])
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["To Min"].default_value = low
    mr.inputs["To Max"].default_value = 1.0
    nt.links.new(ma.outputs[0], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], bg.inputs["Strength"])
    scene.world = world


def _setup(scene, job, samples):
    W, H = job["W"], job["H"]
    cyc = scene.cycles
    cyc.samples = samples
    cyc.use_adaptive_sampling = True
    cyc.adaptive_threshold = 0.01
    cyc.use_denoising = True
    cyc.denoiser = "OPENIMAGEDENOISE"
    cyc.max_bounces = 6
    cyc.diffuse_bounces = 3
    cyc.glossy_bounces = 1
    cyc.transparent_max_bounces = 8
    cyc.filter_width = 1.0
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.render.resolution_x, scene.render.resolution_y = job["px"]
    scene.render.resolution_percentage = 100
    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = max(W, H)
    cam_data.clip_start = 0.1
    cam_data.clip_end = 200.0
    cam = bpy.data.objects.new("cam", cam_data)
    cam.location = (W / 2.0, H / 2.0, 40.0)
    scene.collection.objects.link(cam)
    scene.camera = cam
    layer = scene.view_layers[0]
    layer.use_pass_combined = True
    layer.use_pass_normal = True
    for name in ("rough", "metal"):
        aov = layer.aovs.add()
        aov.name = name
        aov.type = "VALUE"
    img = scene.render.image_settings
    img.file_format = "OPEN_EXR_MULTILAYER"
    img.color_depth = "32"
    img.exr_codec = "ZIP"
    up = {"top": (0.0, 0.0, 1.0), "wall": (0.0, 0.6, 0.8),
          "wall_x": (0.6, 0.0, 0.8)}[job["view"]]
    _sky(scene, up)


def _attr(nt, name):
    n = nt.nodes.new("ShaderNodeAttribute")
    n.attribute_name = name
    return n


def _aov(nt, name, socket=None, value=None):
    node = nt.nodes.new("ShaderNodeOutputAOV")
    node.aov_name = name
    if socket is not None:
        nt.links.new(socket, node.inputs["Value"])
    else:
        node.inputs["Value"].default_value = value


def _layer_material():
    mat = bpy.data.materials.new("layer")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    col, rough, metal, trans = (_attr(nt, n) for n in ("col", "rough", "metal", "trans"))
    nt.links.new(col.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(rough.outputs["Fac"], bsdf.inputs["Roughness"])
    nt.links.new(metal.outputs["Fac"], bsdf.inputs["Metallic"])
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    tint = nt.nodes.new("ShaderNodeMix")
    tint.data_type = "RGBA"
    tint.inputs["Factor"].default_value = 0.25
    tint.inputs["A"].default_value = (1, 1, 1, 1)
    nt.links.new(col.outputs["Color"], tint.inputs["B"])
    nt.links.new(tint.outputs["Result"], tr.inputs["Color"])
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(trans.outputs["Fac"], mix.inputs["Fac"])
    nt.links.new(bsdf.outputs["BSDF"], mix.inputs[1])
    nt.links.new(tr.outputs["BSDF"], mix.inputs[2])
    out = nt.nodes["Material Output"]
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    _aov(nt, "rough", rough.outputs["Fac"])
    _aov(nt, "metal", metal.outputs["Fac"])
    return mat


def _flat_material(col, rough, metal):
    mat = bpy.data.materials.new("flat")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = tuple(col) + (1.0,)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    _aov(nt, "rough", value=rough)
    _aov(nt, "metal", value=metal)
    return mat


def _extend(arr, rows, period_rows, axis=0):
    """Wrap ``rows`` rows onto both ends (tiling scenes)."""
    return np.concatenate([arr[-rows:], arr, arr[:rows]], axis=axis)


def _layer_mesh(scene, job, i, path, mat):
    d = np.load(path)
    X, Y, Z = d["x"], d["y"], d["z"]
    col, trans, rough, metal, mask = (d[k] for k in ("col", "trans", "rough", "metal", "mask"))
    if job["period"]:
        P = job["period"]
        dy = Y[1, 0] - Y[0, 0]
        k = max(1, int(MARGIN_M / dy))
        Y = np.concatenate([Y[-k:] - P, Y, Y[:k] + P])
        X, Z, col, trans, rough, metal, mask = (
            _extend(a, k, None) for a in (X, Z, col, trans, rough, metal, mask))
    ny, nx = Z.shape
    verts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    idx = np.arange(ny * nx).reshape(ny, nx)
    q = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], axis=-1)
    m = mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, 1:] & mask[1:, :-1]
    faces = q[m].reshape(-1, 4)
    mesh = bpy.data.meshes.new(f"L{i}")
    mesh.vertices.add(len(verts))
    mesh.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    mesh.loops.add(faces.size)
    mesh.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
    mesh.polygons.add(len(faces))
    mesh.polygons.foreach_set("loop_start", (np.arange(len(faces)) * 4).astype(np.int32))
    mesh.polygons.foreach_set("loop_total", np.full(len(faces), 4, np.int32))
    mesh.update()
    mesh.validate()
    c = mesh.color_attributes.new("col", "FLOAT_COLOR", "POINT")
    rgba = np.concatenate([col.reshape(-1, 3), np.ones((ny * nx, 1))], axis=1)
    c.data.foreach_set("color", rgba.astype(np.float32).ravel())
    for name, arr in (("trans", trans), ("rough", rough), ("metal", metal)):
        a = mesh.attributes.new(name, "FLOAT", "POINT")
        a.data.foreach_set("value", arr.astype(np.float32).ravel())
    mesh.polygons.foreach_set("use_smooth", np.ones(len(faces), bool))
    mesh.materials.append(mat)
    obj = bpy.data.objects.new(f"L{i}", mesh)
    scene.collection.objects.link(obj)


def _tube_mesh(scene, t, shift, mats):
    pts = np.asarray(t["pts"], dtype=np.float64) + np.array([0.0, shift, 0.0])
    r, sides = t["r"], t["sides"]
    verts, faces = [], []
    for i, p in enumerate(pts):
        a = pts[max(0, i - 1)]
        b = pts[min(len(pts) - 1, i + 1)]
        tg = b - a
        tg /= np.linalg.norm(tg) or 1.0
        ref = np.array([0.0, 0.0, 1.0]) if abs(tg[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        n1 = np.cross(tg, ref)
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(tg, n1)
        for k in range(sides):
            ang = 2 * math.pi * k / sides
            verts.append(tuple(p + r * (math.cos(ang) * n1 + math.sin(ang) * n2)))
    for i in range(len(pts) - 1):
        for k in range(sides):
            a0, a1 = i * sides + k, i * sides + (k + 1) % sides
            faces.append((a0, a1, a1 + sides, a0 + sides))
    mesh = bpy.data.meshes.new("tube")
    mesh.from_pydata(verts, [], faces)
    mesh.polygons.foreach_set("use_smooth", [True] * len(faces))
    key = (tuple(t["col"]), t["rough"], t["metal"])
    if key not in mats:
        mats[key] = _flat_material(t["col"], t["rough"], t["metal"])
    mesh.materials.append(mats[key])
    obj = bpy.data.objects.new("tube", mesh)
    scene.collection.objects.link(obj)


def _box(scene, b, shift, mats):
    sx, sy, sz = b["size"]
    c = np.array(b["c"]) + np.array([0.0, shift, 0.0])
    rot = math.radians(b["rot"])
    cr, sr = math.cos(rot), math.sin(rot)
    verts = []
    for dz in (-sz / 2, sz / 2):
        for dx, dy in ((-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2)):
            # slightly rounded top: pull top corners in
            k = 0.92 if dz > 0 else 1.0
            verts.append((c[0] + k * (dx * cr - dy * sr), c[1] + k * (dx * sr + dy * cr), c[2] + dz))
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    mesh = bpy.data.meshes.new("box")
    mesh.from_pydata(verts, [], faces)
    key = (tuple(b["col"]), b["rough"], 0.0)
    if key not in mats:
        mats[key] = _flat_material(b["col"], b["rough"], 0.0)
    mesh.materials.append(mats[key])
    obj = bpy.data.objects.new("box", mesh)
    scene.collection.objects.link(obj)


def _read_exr(path):
    inp = oiio.ImageInput.open(path)
    spec = inp.spec()
    data = np.asarray(inp.read_image(0, 0, 0, spec.nchannels, oiio.FLOAT))
    inp.close()
    data = data.reshape(spec.height, spec.width, spec.nchannels)
    return {name: data[:, :, i] for i, name in enumerate(spec.channelnames)}


def _pick(ch, pass_name, comps):
    out = []
    for comp in comps:
        hits = [k for k in ch if f".{pass_name}." in k and k.endswith("." + comp)]
        if not hits:
            raise KeyError(f"{pass_name}.{comp} not in {sorted(ch)}")
        out.append(ch[hits[0]])
    return np.stack(out, axis=-1)


def run_job(job, samples):
    scene = _reset()
    _setup(scene, job, samples)
    mat = _layer_material()
    for i, path in enumerate(job["layers"]):
        _layer_mesh(scene, job, i, path, mat)
    mats = {}
    shifts = (0.0,) if not job["period"] else (-job["period"], 0.0, job["period"])
    for s in shifts:
        for t in job["tubes"]:
            _tube_mesh(scene, t, s, mats)
        for b in job["boxes"]:
            _box(scene, b, s, mats)
    tmp = job["out"] + ".exr"
    scene.render.filepath = tmp
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    ch = _read_exr(tmp)
    arrays = {
        "shaded": _pick(ch, "Combined", "RGB"),
        "normal": _pick(ch, "Normal", "XYZ"),
        "rough": _pick(ch, "rough", ("V",))[..., 0] if any(k.endswith(".rough.V") for k in ch)
        else _pick(ch, "rough", ("X",))[..., 0],
        "metal": _pick(ch, "metal", ("V",))[..., 0] if any(k.endswith(".metal.V") for k in ch)
        else _pick(ch, "metal", ("X",))[..., 0],
    }
    np.savez_compressed(job["out"], **{k: v.astype(np.float16) for k, v in arrays.items()})
    os.remove(tmp)
    print(f"[bake] {job['name']}: {time.time() - t0:.1f}s", flush=True)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    with open(argv[0], "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    only = set(argv[1:])
    for job in spec["jobs"]:
        if only and job["name"] not in only:
            continue
        run_job(job, spec.get("samples", 96))


if __name__ == "__main__":
    main()
