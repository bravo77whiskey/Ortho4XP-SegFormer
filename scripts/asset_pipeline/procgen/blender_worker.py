"""Blender-side worker: build archetype meshes and export X-Plane OBJ8s.

Runs INSIDE Blender (never import from host code):

    blender --background --factory-startup --python blender_worker.py -- \
        <chunk.json> <output_root> <atlas_layout.json>

``chunk.json`` holds flattened per-OBJ rows prepared by
generate_procedural_buildings.py:

    {"assets": [{"stem", "physical_path", "archetype", "length_m",
                 "width_m", "floors", "seed", "flavor", "group"}, ...]}

``atlas_layout.json`` is the v2 registry: one strip layout per
(flavor, group) combo under "combos"; each row builds against its own
combo sub-layout.

The export recipe (fresh exportable collection per asset, layer.name = stem,
export_type "scenery") mirrors the proven flow in convert_to_xplane_obj.py.
One Blender process handles a whole chunk; scene objects are torn down after
every asset so consecutive exports never leak into each other.
"""

import json
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from archetypes import build_archetype  # noqa: E402


def _enable_xplane2blender():
    for module in ("io_xplane2blender", "xplane2blender", "XPlane2Blender"):
        try:
            bpy.ops.preferences.addon_enable(module=module)
            return
        except Exception:
            continue
    raise SystemExit(
        "xplane2blender addon not found. Install it via Edit > Preferences > "
        "Add-ons > Install... and Save Preferences before running this script."
    )


def _texture_relpath(flavor: str, group: str) -> str:
    # Physical objs live at <root>/<region>/<bucket>/, atlases at
    # <root>/textures/ -- two levels up.
    return f"../../textures/o4sfr_procgen_atlas_{flavor}_{group}.png"


def _material_for_flavor(flavor: str, group: str, output_root: str):
    name = f"o4sfr_procgen_{flavor}_{group}"
    material = bpy.data.materials.get(name)
    if material is not None:
        return material
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    image_path = os.path.join(
        output_root, "textures",
        f"o4sfr_procgen_atlas_{flavor}_{group}.png"
    )
    if os.path.isfile(image_path) and bsdf is not None:
        image = bpy.data.images.get(os.path.basename(image_path))
        if image is None:
            image = bpy.data.images.load(image_path)
        node = material.node_tree.nodes.new("ShaderNodeTexImage")
        node.image = image
        material.node_tree.links.new(
            node.outputs["Color"], bsdf.inputs["Base Color"]
        )
    return material


def _build_mesh_object(stem: str, spec, material):
    mesh = bpy.data.meshes.new(stem)
    mesh.from_pydata(spec.vertices, [], spec.faces)
    mesh.validate()
    uv_layer = mesh.uv_layers.new(name="UVMap")
    loop_index = 0
    for face_uvs in spec.uvs:
        for uv in face_uvs:
            uv_layer.data[loop_index].uv = uv
            loop_index += 1
    if loop_index != len(mesh.loops):
        raise RuntimeError(
            f"{stem}: UV loop count {loop_index} != mesh loops {len(mesh.loops)}"
        )
    mesh.materials.append(material)
    return bpy.data.objects.new(stem, mesh)


def _export_one(row: dict, output_root: str, layout: dict) -> str | None:
    """Build + export one OBJ; return error string or None on success."""
    stem = row["stem"]
    target_dir = os.path.join(
        output_root, os.path.dirname(row["physical_path"])
    )
    os.makedirs(target_dir, exist_ok=True)
    final_obj = os.path.join(output_root, row["physical_path"])

    combo = layout["combos"][f"{row['flavor']}/{row['group']}"]
    spec = build_archetype(
        row["archetype"], float(row["length_m"]), float(row["width_m"]),
        int(row["floors"]), int(row["seed"]), combo, row["flavor"],
    )
    material = _material_for_flavor(row["flavor"], row["group"],
                                    output_root)
    obj = _build_mesh_object(stem, spec, material)

    export_coll = bpy.data.collections.new(name=f"O4SFR_{stem}")
    bpy.context.scene.collection.children.link(export_coll)
    export_coll.objects.link(obj)
    bpy.context.view_layer.update()

    xp = getattr(export_coll, "xplane", None)
    if xp is None:
        raise SystemExit(
            "collection.xplane property group missing -- xplane2blender "
            "not fully registered"
        )
    error = None
    try:
        xp.is_exportable_collection = True
        xp.layer.name = stem
        try:
            xp.layer.export_type = "scenery"
        except Exception:
            pass
        try:
            # Pin the texture so the exporter writes a deterministic
            # TEXTURE directive; the host driver re-checks it either way.
            xp.layer.autodetectTextures = False
            xp.layer.texture = _texture_relpath(row["flavor"],
                                                row["group"])
        except Exception:
            pass
        try:
            bpy.ops.export.xplane_obj(filepath=target_dir + os.sep)
        except Exception:
            bpy.ops.xplane.export_objects(filepath=target_dir + os.sep)
        exported = os.path.join(target_dir, stem + ".obj")
        if not os.path.isfile(exported):
            error = f"exporter produced no file at {exported}"
        elif exported != final_obj:
            os.replace(exported, final_obj)
    except Exception as exc:  # keep the batch going; report per asset
        error = f"{type(exc).__name__}: {exc}"
    finally:
        # Tear down so the next export call sees exactly one exportable
        # collection (xplane2blender exports ALL marked collections).
        try:
            xp.is_exportable_collection = False
        except Exception:
            pass
        for parent in list(obj.users_collection):
            parent.objects.unlink(obj)
        mesh = obj.data
        bpy.data.objects.remove(obj)
        bpy.data.meshes.remove(mesh)
        bpy.context.scene.collection.children.unlink(export_coll)
        bpy.data.collections.remove(export_coll)
    return error


def main() -> int:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    chunk_path, output_root, layout_path = argv[0], argv[1], argv[2]

    bpy.ops.wm.read_factory_settings(use_empty=True)
    _enable_xplane2blender()

    with open(chunk_path, "r", encoding="utf-8") as fh:
        chunk = json.load(fh)
    with open(layout_path, "r", encoding="utf-8") as fh:
        layout = json.load(fh)

    ok = 0
    failures = []
    for row in chunk["assets"]:
        try:
            error = _export_one(row, output_root, layout)
        except SystemExit:
            raise
        except Exception as exc:  # archetype bugs etc.
            error = f"{type(exc).__name__}: {exc}"
        if error is None:
            ok += 1
            print(f"[ok] {row['physical_path']}")
        else:
            failures.append((row["physical_path"], error))
            print(f"[fail] {row['physical_path']}: {error}")

    print(f"[chunk-done] ok={ok} fail={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
