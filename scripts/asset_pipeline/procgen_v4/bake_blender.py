"""Render atlas strips from high-poly relief scenes (runs inside Blender).

    blender --background --factory-startup --python bake_blender.py -- job.json

``job.json`` = {"samples": int, "jobs": [{"name", "kind": facade|plain|roof,
"level", "scheme", "roof", "W", "H", "px": [w, h], "seed", "mats", "out"}]}.

Each strip is rendered on its own (other strips would otherwise shadow it)
with an orthographic camera looking at the relief, under a uniform white
sky and no lamps, so the Combined pass is albedo x ambient occlusion with
real inter-reflection.  Passes are written to ``out`` as a float16 .npz:
shaded, albedo, normal (tangent space: u, v, out), gloss, metal, drip, cls,
rnd, depth.  Rows are top-first (row 0 = top of the strip).
"""

import json
import math
import os
import sys
import time

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import OpenImageIO as oiio  # noqa: E402

import strips  # noqa: E402
from bake_materials import build_material  # noqa: E402

AOVS = ("gloss", "metal", "drip", "cls", "rnd", "depth")


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


def _sky(scene, up, low):
    """Overcast dome: full brightness toward ``up``, ``low`` from below.

    Soft directional sky light gives the contact shadows under sills,
    lintels and tile noses that make a photographed facade read as solid,
    without the hard sun direction the simulator adds itself.
    """
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


def _setup(scene, W, H, px, samples):
    cyc = scene.cycles
    cyc.samples = samples
    cyc.use_adaptive_sampling = True
    cyc.adaptive_threshold = 0.01
    cyc.use_denoising = True
    cyc.denoiser = "OPENIMAGEDENOISE"
    cyc.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    try:
        cyc.denoising_use_gpu = True
    except AttributeError:
        pass
    cyc.max_bounces = 4
    cyc.diffuse_bounces = 3
    cyc.glossy_bounces = 0
    cyc.transmission_bounces = 0
    cyc.filter_width = 1.0
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.render.resolution_x = px[0]
    scene.render.resolution_y = px[1]
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = W
    cam_data.clip_start = 0.1
    cam_data.clip_end = 200.0
    cam = bpy.data.objects.new("cam", cam_data)
    cam.location = (W / 2.0, -30.0, H / 2.0)
    cam.rotation_euler = (math.pi / 2, 0.0, 0.0)
    scene.collection.objects.link(cam)
    scene.camera = cam
    layer = scene.view_layers[0]
    layer.use_pass_combined = True
    layer.use_pass_diffuse_color = True
    layer.use_pass_normal = True
    for name in AOVS:
        aov = layer.aovs.add()
        aov.name = name
        aov.type = "VALUE"
    img = scene.render.image_settings
    img.file_format = "OPEN_EXR_MULTILAYER"
    img.color_depth = "32"
    img.exr_codec = "ZIP"


def _mesh(scene, name, part, material):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(part["v"], [], part["f"])
    mesh.validate(clean_customdata=False)
    for attr in ("r", "r2"):
        a = mesh.attributes.new("rnd" if attr == "r" else "rnd2", "FLOAT",
                                "FACE")
        values = part[attr]
        if len(values) != len(mesh.polygons):
            values = (values + [0.0] * len(mesh.polygons))[:len(mesh.polygons)]
        a.data.foreach_set("value", values)
    mesh.materials.append(material)
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    return obj


def _read_exr(path):
    inp = oiio.ImageInput.open(path)
    if inp is None:
        raise RuntimeError(f"cannot open {path}: {oiio.geterror()}")
    spec = inp.spec()
    data = inp.read_image(0, 0, 0, spec.nchannels, oiio.FLOAT)
    inp.close()
    data = np.asarray(data).reshape(spec.height, spec.width, spec.nchannels)
    return {name: data[:, :, i] for i, name in enumerate(spec.channelnames)}


def _pick(channels, pass_name, comps):
    out = []
    for comp in comps:
        hits = [k for k in channels
                if k.split(".")[-2:-1] == [pass_name] and k.endswith("." + comp)]
        if not hits:
            hits = [k for k in channels if f".{pass_name}." in k
                    and k.endswith(comp)]
        if not hits:
            raise KeyError(f"{pass_name}.{comp} not in {sorted(channels)}")
        out.append(channels[hits[0]])
    return np.stack(out, axis=-1)


def run_job(job, samples):
    scene = _reset()
    W, H = float(job["W"]), float(job["H"])
    _setup(scene, W, H, job["px"], samples)
    kind = job["kind"]
    if kind == "roof":
        _sky(scene, (0.0, -0.55, 0.83), 0.3)
    else:
        _sky(scene, (0.0, -0.25, 0.97), 0.25)
    if kind == "facade":
        geo, _info = strips.facade_strip(job["scheme"], job["level"], W, H,
                                         job["seed"])
    elif kind == "plain":
        geo, _info = strips.plain_strip(job["scheme"], W, H, job["seed"],
                                        vents=job.get("vents", True))
    elif kind == "roof":
        geo, _info = strips.roof_strip(job["roof"], W, H, job["seed"])
    elif kind == "band":
        geo = strips.trim_band(job["band"], W, H, job["seed"])
    else:
        raise ValueError(kind)
    mats = job["mats"]
    seed_w = (job["seed"] % 997) / 97.0
    for name, part in geo.parts.items():
        if not part["f"]:
            continue
        spec = mats.get(name)
        if spec is None:
            raise KeyError(f"job {job['name']}: no material '{name}'")
        material = build_material(name, spec, W, seed_w)
        _mesh(scene, name, part, material)
    tmp = job["out"] + ".exr"
    scene.render.filepath = tmp
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    ch = _read_exr(tmp)
    arrays = {
        "shaded": _pick(ch, "Combined", "RGB"),
        "albedo": _pick(ch, "DiffCol", "RGB"),
        "normal_world": _pick(ch, "Normal", "XYZ"),
    }
    for name in AOVS:
        arrays[name] = _pick(ch, name, ("V",))[..., 0] if any(
            k.endswith(f".{name}.V") for k in ch) else _pick(
                ch, name, ("X",))[..., 0]
    n = arrays.pop("normal_world")
    # camera looks along +y at a relief facing -y: tangent u = +x,
    # bitangent v = +z, outward = -y
    arrays["normal"] = np.stack([n[..., 0], n[..., 2], -n[..., 1]], axis=-1)
    np.savez_compressed(job["out"], **{k: v.astype(np.float16)
                                       for k, v in arrays.items()},
                        W=W, H=H)
    os.remove(tmp)
    print(f"[bake] {job['name']}: {geo.face_count()} faces, "
          f"{time.time() - t0:.1f}s", flush=True)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    with open(argv[0], "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    only = set(argv[1:])
    for job in spec["jobs"]:
        if only and job["name"] not in only:
            continue
        if spec.get("skip_existing") and os.path.isfile(job["out"]):
            continue
        run_job(job, spec.get("samples", 64))


if __name__ == "__main__":
    main()
