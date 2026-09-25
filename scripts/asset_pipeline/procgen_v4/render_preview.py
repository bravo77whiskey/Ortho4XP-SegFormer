"""Render review images of the EXPORTED OBJ8 files (runs inside Blender).

    blender --background --factory-startup --python render_preview.py -- scene.json

The importer reads the OBJ8 text itself (VT / IDX / ATTR_LOD / TRIS), picks
one LOD band, and shades it with the shipped textures the way X-Plane's
PBR model reads them: albedo, tangent normal from RG, base reflectance
from B (NORMAL_METALNESS), gloss from A.  No subdivision, bevels or extra
geometry are added -- what renders is what the sim gets.  Ground, roads,
sky and sun are presentation only.

scene.json: {"out", "res": [w, h], "samples", "camera": {"loc", "look_at",
"lens"}, "sun": {"elev", "azim", "strength"}, "ground": "#hex",
"roads": [[x0, y0, x1, y1, width], ...], "objects": [{"obj", "loc": [x, y],
"rot", "band"}]}
"""

import json
import math
import os
import sys

import bpy
from mathutils import Vector

_MATS = {}


def _srgb(h):
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple((x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4)
                 for x in c) + (1.0,)


def read_obj8(path, band):
    verts, idx, lods = [], [], []
    tex = nml = None
    cur = None
    with open(path, "r", encoding="ascii") as fh:
        for line in fh:
            p = line.split()
            if not p:
                continue
            k = p[0]
            if k == "VT":
                verts.append(tuple(float(x) for x in p[1:9]))
            elif k == "IDX10" or k == "IDX":
                idx.extend(int(x) for x in p[1:])
            elif k == "TEXTURE":
                tex = p[1]
            elif k == "TEXTURE_NORMAL":
                nml = p[1]
            elif k == "ATTR_LOD":
                cur = [float(p[1]), float(p[2]), []]
                lods.append(cur)
            elif k == "TRIS":
                cur[2].append((int(p[1]), int(p[2])))
    base = os.path.dirname(path)
    rng = lods[min(band, len(lods) - 1)][2]
    tris = []
    for start, count in rng:
        for i in range(start, start + count, 3):
            # OBJ8 is clockwise-front: reverse back to CCW
            tris.append((idx[i], idx[i + 2], idx[i + 1]))
    return verts, tris, os.path.normpath(os.path.join(base, tex)), \
        os.path.normpath(os.path.join(base, nml))


def material(tex, nml):
    key = (tex, nml)
    if key in _MATS:
        return _MATS[key]
    m = bpy.data.materials.new(os.path.basename(tex))
    m.use_nodes = True
    nt = m.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf = nodes["Principled BSDF"]
    ia = nodes.new("ShaderNodeTexImage")
    ia.image = bpy.data.images.load(tex, check_existing=True)
    ia.interpolation = "Cubic"
    links.new(ia.outputs["Color"], bsdf.inputs["Base Color"])
    inn = nodes.new("ShaderNodeTexImage")
    img = bpy.data.images.load(nml, check_existing=True)
    img.colorspace_settings.name = "Non-Color"
    img.alpha_mode = "CHANNEL_PACKED"
    inn.image = img
    sep = nodes.new("ShaderNodeSeparateColor")
    links.new(inn.outputs["Color"], sep.inputs["Color"])

    def mth(op, a, b=None):
        n = nodes.new("ShaderNodeMath")
        n.operation = op
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = v
            else:
                links.new(v, n.inputs[i])
        return n.outputs[0]

    x = mth("SUBTRACT", mth("MULTIPLY", sep.outputs[0], 2.0), 1.0)
    y = mth("SUBTRACT", mth("MULTIPLY", sep.outputs[1], 2.0), 1.0)
    z = mth("SQRT", mth("MAXIMUM", mth("SUBTRACT", 1.0, mth(
        "ADD", mth("MULTIPLY", x, x), mth("MULTIPLY", y, y))), 0.0))
    comb = nodes.new("ShaderNodeCombineColor")
    links.new(mth("ADD", mth("MULTIPLY", x, 0.5), 0.5), comb.inputs[0])
    links.new(mth("ADD", mth("MULTIPLY", y, 0.5), 0.5), comb.inputs[1])
    links.new(mth("ADD", mth("MULTIPLY", z, 0.5), 0.5), comb.inputs[2])
    nm = nodes.new("ShaderNodeNormalMap")
    links.new(comb.outputs[0], nm.inputs["Color"])
    links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(mth("SUBTRACT", 1.0, inn.outputs["Alpha"]),
              bsdf.inputs["Roughness"])
    links.new(mth("MULTIPLY", mth("SUBTRACT", sep.outputs[2], 0.04), 1.04),
              bsdf.inputs["Metallic"])
    _MATS[key] = m
    return m


def add_obj(spec):
    verts, tris, tex, nml = read_obj8(spec["obj"], spec.get("band", 0))
    name = os.path.basename(spec["obj"])
    mesh = bpy.data.meshes.new(name)
    # X-Plane (x, y up, z south) -> Blender (x, -z, y)
    co = [(v[0], -v[2], v[1]) for v in verts]
    mesh.from_pydata(co, [], tris)
    uv = mesh.uv_layers.new(name="UV")
    for poly in mesh.polygons:
        for li in poly.loop_indices:
            vi = mesh.loops[li].vertex_index
            uv.data[li].uv = (verts[vi][6], verts[vi][7])
    mesh.polygons.foreach_set("use_smooth", [False] * len(mesh.polygons))
    mesh.materials.append(material(tex, nml))
    obj = bpy.data.objects.new(name, mesh)
    obj.location = (spec["loc"][0], spec["loc"][1], 0.0)
    obj.rotation_euler = (0, 0, math.radians(spec.get("rot", 0.0)))
    bpy.context.scene.collection.objects.link(obj)
    return obj


def ground(scene, colour, roads):
    bpy.ops.mesh.primitive_plane_add(size=4000, location=(0, 0, -0.01))
    g = bpy.context.active_object
    m = bpy.data.materials.new("ground")
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 0.08
    noise.inputs["Detail"].default_value = 8
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    base = _srgb(colour)
    ramp.color_ramp.elements[0].color = tuple(c * 0.7 for c in base[:3]) + (1,)
    ramp.color_ramp.elements[1].color = tuple(min(1, c * 1.25)
                                              for c in base[:3]) + (1,)
    nt.links.new(noise.outputs["Fac"], ramp.inputs[0])
    nt.links.new(ramp.outputs[0], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.95
    g.data.materials.append(m)
    am = bpy.data.materials.new("asphalt")
    am.use_nodes = True
    ab = am.node_tree.nodes["Principled BSDF"]
    ab.inputs["Base Color"].default_value = _srgb("#4a4a48")
    ab.inputs["Roughness"].default_value = 0.9
    sm = bpy.data.materials.new("kerb")
    sm.use_nodes = True
    sb = sm.node_tree.nodes["Principled BSDF"]
    sb.inputs["Base Color"].default_value = _srgb("#9c9a94")
    sb.inputs["Roughness"].default_value = 0.9
    for x0, y0, x1, y1, w in roads:
        for mat_, width, z in ((sm, w + 3.0, 0.0), (am, w, 0.02)):
            dx, dy = x1 - x0, y1 - y0
            ln = math.hypot(dx, dy)
            bpy.ops.mesh.primitive_plane_add(size=1, location=(
                (x0 + x1) / 2, (y0 + y1) / 2, z))
            r = bpy.context.active_object
            r.scale = (ln, width, 1)
            r.rotation_euler = (0, 0, math.atan2(dy, dx))
            r.data.materials.append(mat_)


def setup(scene, spec):
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "OPTIX"
    scene.cycles.device = "GPU"
    scene.cycles.samples = spec.get("samples", 96)
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = spec.get(
        "res", [1920, 1080])
    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass
    sun = spec.get("sun", {})
    elev = math.radians(sun.get("elev", 38))
    azim = math.radians(sun.get("azim", 135))
    world = bpy.data.worlds.new("sky")
    world.use_nodes = True
    sky = world.node_tree.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA"
    sky.sun_disc = False
    sky.sun_elevation = elev
    sky.sun_rotation = azim
    sky.altitude = 200
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Strength"].default_value = sun.get("sky", 0.22)
    world.node_tree.links.new(sky.outputs["Color"], bg.inputs["Color"])
    scene.world = world
    ld = bpy.data.lights.new("sun", "SUN")
    ld.energy = sun.get("strength", 4.2)
    ld.angle = math.radians(0.8)
    ld.color = (1.0, 0.96, 0.9)
    lo = bpy.data.objects.new("sun", ld)
    # direction the light travels: from the sun toward the ground
    d = Vector((-math.cos(elev) * math.sin(azim),
                -math.cos(elev) * math.cos(azim), -math.sin(elev)))
    lo.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(lo)
    cam = spec["camera"]
    cd = bpy.data.cameras.new("cam")
    cd.lens = cam.get("lens", 50)
    cd.clip_end = 20000
    co = bpy.data.objects.new("cam", cd)
    co.location = cam["loc"]
    look = Vector(cam["look_at"]) - Vector(cam["loc"])
    co.rotation_euler = look.to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(co)
    scene.camera = co


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    with open(argv[0], "r", encoding="utf-8") as fh:
        specs = json.load(fh)
    if isinstance(specs, dict):
        specs = [specs]
    for spec in specs:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        _MATS.clear()
        scene = bpy.context.scene
        setup(scene, spec)
        ground(scene, spec.get("ground", "#5f6b45"), spec.get("roads", []))
        for o in spec["objects"]:
            add_obj(o)
        scene.render.filepath = spec["out"]
        bpy.ops.render.render(write_still=True)
        print(f"[render] {spec['out']}", flush=True)


if __name__ == "__main__":
    main()
