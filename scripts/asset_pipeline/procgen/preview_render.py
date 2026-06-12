"""Render preview contact sheets of procgen archetypes for visual QA.

Dual-mode file (same pattern as convert_to_xplane_obj.py):
  * Run OUTSIDE Blender, it samples the manifest (a spread of dimensions per
    archetype), spawns headless Blender on itself, then montages the renders
    into one contact-sheet PNG per archetype with PIL.
  * Inside Blender it rebuilds each sampled asset's mesh (same code path as
    the exporter) and renders it with the Workbench engine in TEXTURE color
    mode -- reliable headless and honest about what the atlas looks like.

Run:  python preview_render.py --output <pkg> --out-dir previews [--per-archetype 6]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import bpy  # noqa: F401
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False


# ----------------------------- Blender side -----------------------------

def _blender_main() -> int:
    import math

    import bpy
    from mathutils import Vector

    from archetypes import build_archetype

    argv = sys.argv[sys.argv.index("--") + 1:]
    plan_path, output_root, layout_path, render_dir = argv[:4]

    bpy.ops.wm.read_factory_settings(use_empty=True)
    with open(plan_path, "r", encoding="utf-8") as fh:
        plan = json.load(fh)
    with open(layout_path, "r", encoding="utf-8") as fh:
        layout = json.load(fh)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "TEXTURE"
    scene.render.resolution_x = 512
    scene.render.resolution_y = 384
    scene.render.film_transparent = True

    cam_data = bpy.data.cameras.new("preview_cam")
    cam = bpy.data.objects.new("preview_cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    materials = {}

    def _material(flavor):
        if flavor in materials:
            return materials[flavor]
        material = bpy.data.materials.new(name=f"preview_{flavor}")
        material.use_nodes = True
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        image_path = os.path.join(
            output_root, "textures", f"o4sfr_procgen_atlas_{flavor}.png"
        )
        if bsdf is not None and os.path.isfile(image_path):
            image = bpy.data.images.load(image_path, check_existing=True)
            node = material.node_tree.nodes.new("ShaderNodeTexImage")
            node.image = image
            material.node_tree.links.new(
                node.outputs["Color"], bsdf.inputs["Base Color"]
            )
        materials[flavor] = material
        return material

    os.makedirs(render_dir, exist_ok=True)
    for row in plan["assets"]:
        spec = build_archetype(
            row["archetype"], float(row["length_m"]), float(row["width_m"]),
            int(row["floors"]), int(row["seed"]), layout, row["flavor"],
        )
        mesh = bpy.data.meshes.new(row["stem"])
        mesh.from_pydata(spec.vertices, [], spec.faces)
        mesh.validate()
        uv_layer = mesh.uv_layers.new(name="UVMap")
        loop_index = 0
        for face_uvs in spec.uvs:
            for uv in face_uvs:
                uv_layer.data[loop_index].uv = uv
                loop_index += 1
        mesh.materials.append(_material(row["flavor"]))
        obj = bpy.data.objects.new(row["stem"], mesh)
        scene.collection.objects.link(obj)

        # 3/4 aerial view framing the bounding sphere.
        xmin, xmax, ymin, ymax, zmin, zmax = (
            min(v[0] for v in spec.vertices),
            max(v[0] for v in spec.vertices),
            min(v[1] for v in spec.vertices),
            max(v[1] for v in spec.vertices),
            min(v[2] for v in spec.vertices),
            max(v[2] for v in spec.vertices),
        )
        center = Vector(((xmin + xmax) / 2, (ymin + ymax) / 2,
                         (zmin + zmax) / 2))
        radius = max(xmax - xmin, ymax - ymin, zmax - zmin)
        distance = radius * 1.9 + 4.0
        azimuth, elevation = math.radians(35.0), math.radians(32.0)
        cam.location = center + Vector((
            distance * math.cos(elevation) * math.cos(azimuth),
            -distance * math.cos(elevation) * math.sin(azimuth),
            distance * math.sin(elevation),
        ))
        direction = center - cam.location
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

        scene.render.filepath = os.path.join(render_dir, row["stem"] + ".png")
        bpy.ops.render.render(write_still=True)
        print(f"[ok] rendered {row['stem']}")

        scene.collection.objects.unlink(obj)
        bpy.data.objects.remove(obj)
        bpy.data.meshes.remove(mesh)

    print("[render-done]")
    return 0


# ------------------------------ host side -------------------------------

def _host_main() -> int:
    import subprocess
    import tempfile

    from PIL import Image, ImageDraw

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default=os.path.join(HERE, "procgen_manifest.json")
    )
    parser.add_argument(
        "--layout", default=os.path.join(HERE, "atlas_layout.json")
    )
    parser.add_argument("--output", required=True,
                        help="Library package folder (for the atlas PNGs).")
    parser.add_argument("--out-dir", default=os.path.join(HERE, "previews"))
    parser.add_argument("--per-archetype", type=int, default=6)
    parser.add_argument("--blender",
                        default=os.environ.get("O4SFR_BLENDER", "blender"))
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    # Spread sample: per archetype, evenly step through its variant rows
    # sorted by footprint area so the sheet spans small -> large.
    by_archetype = {}
    for asset in manifest["assets"]:
        if not asset.get("enabled"):
            continue
        for variant in asset["variants"]:
            by_archetype.setdefault(variant["archetype"], []).append(
                (asset, variant)
            )
    plan_rows = []
    for archetype, rows in sorted(by_archetype.items()):
        rows.sort(key=lambda item: item[0]["length_m"] * item[0]["width_m"])
        step = max(1, len(rows) // args.per_archetype)
        for asset, variant in rows[::step][: args.per_archetype]:
            plan_rows.append({
                "stem": os.path.splitext(
                    os.path.basename(variant["physical_path"])
                )[0],
                "region": asset["region"],
                "archetype": archetype,
                "length_m": asset["length_m"],
                "width_m": asset["width_m"],
                "floors": asset["floors"],
                "seed": variant["seed"],
                "flavor": asset["flavor"],
            })

    render_dir = tempfile.mkdtemp(prefix="o4sfr_previews_")
    plan_path = os.path.join(render_dir, "plan.json")
    with open(plan_path, "w", encoding="utf-8") as fh:
        json.dump({"assets": plan_rows}, fh)

    cmd = [
        args.blender, "--background", "--factory-startup",
        "--python", os.path.abspath(__file__), "--",
        plan_path, args.output, args.layout, render_dir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if "[render-done]" not in (result.stdout or ""):
        tail = "\n".join((result.stdout or "").splitlines()[-15:])
        raise SystemExit(f"Blender render failed:\n{tail}\n{result.stderr}")

    os.makedirs(args.out_dir, exist_ok=True)
    tile_w, tile_h, caption_h = 512, 384, 24
    for archetype in sorted(by_archetype):
        rows = [r for r in plan_rows if r["archetype"] == archetype]
        cols = min(3, len(rows))
        grid_rows = (len(rows) + cols - 1) // cols
        sheet = Image.new(
            "RGB", (cols * tile_w, grid_rows * (tile_h + caption_h)),
            (28, 28, 30),
        )
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(rows):
            png = os.path.join(render_dir, row["stem"] + ".png")
            if not os.path.isfile(png):
                continue
            tile = Image.open(png).convert("RGB")
            x = (i % cols) * tile_w
            y = (i // cols) * (tile_h + caption_h)
            sheet.paste(tile, (x, y))
            caption = f"{row.get('region', '?')}/{row['stem']}"
            draw.text((x + 8, y + tile_h + 5), caption,
                      fill=(220, 220, 220))
        sheet_path = os.path.join(args.out_dir, f"sheet_{archetype}.png")
        sheet.save(sheet_path)
        print(f"wrote {sheet_path}")
    return 0


if __name__ == "__main__":
    sys.exit(_blender_main() if IN_BLENDER else _host_main())
