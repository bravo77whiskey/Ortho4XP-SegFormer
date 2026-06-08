"""Render preview thumbnails for X-Plane .obj building objects.

X-Plane OBJ8 isn't a Wavefront OBJ -- Blender's stock importer won't load it.
We parse the OBJ8 file ourselves (VT/IDX/IDX10/TRIS), build a Blender mesh
from the verts + faces, optionally load the TEXTURE-referenced image as the
diffuse map, and render a 3/4-view JPEG to the triage workspace.

Run:
    python scripts/asset_pipeline/track_a_render.py \\
        --inventory scripts/asset_pipeline/track_a_triage/inventory.csv \\
        --triage    scripts/asset_pipeline/track_a_triage \\
        --blender   "F:/SteamLibrary/steamapps/common/Blender/blender.exe" \\
        --size      512
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


def _slug(lib_id: str, family_prefix: str) -> str:
    """Compose a filesystem-safe preview filename from lib + family prefix."""
    clean = re.sub(r"[^a-z0-9]+", "_", family_prefix.lower()).strip("_")
    return f"{lib_id}__{clean}"


def _blender_render_snippet() -> str:
    return textwrap.dedent(
        '''
        import bpy
        import math
        import os
        import sys
        from math import inf
        from mathutils import Vector

        argv = sys.argv
        if "--" in argv:
            argv = argv[argv.index("--") + 1:]
        source, output_image, render_px = argv[0], argv[1], int(argv[2])

        bpy.ops.wm.read_factory_settings(use_empty=True)

        def parse_xplane_obj(path):
            verts = []
            indices = []
            tris = []
            texture_rel = None
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    parts = line.split()
                    if not parts:
                        continue
                    tag = parts[0]
                    if tag == "TEXTURE" and len(parts) > 1:
                        texture_rel = parts[1]
                    elif tag == "VT" and len(parts) >= 4:
                        # X-Plane: Y up, Z forward.
                        # Blender: Z up, Y forward.  Map (x, y, z) -> (x, z, y).
                        try:
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        except ValueError:
                            continue
                        verts.append((x, z, y))
                    elif tag in ("IDX", "IDX10"):
                        for tok in parts[1:]:
                            try:
                                indices.append(int(tok))
                            except ValueError:
                                pass
                    elif tag == "TRIS" and len(parts) >= 3:
                        try:
                            tris.append((int(parts[1]), int(parts[2])))
                        except ValueError:
                            pass
            faces = []
            for offset, count in tris:
                for i in range(0, count, 3):
                    if offset + i + 2 >= len(indices):
                        break
                    faces.append((
                        indices[offset + i],
                        indices[offset + i + 1],
                        indices[offset + i + 2],
                    ))
            return verts, faces, texture_rel

        verts, faces, texture_rel = parse_xplane_obj(source)
        if not verts or not faces:
            print(f"[err] no geometry parsed from {source}")
            sys.exit(3)

        mesh = bpy.data.meshes.new("XPlaneMesh")
        mesh.from_pydata(verts, [], faces)
        mesh.update()
        obj = bpy.data.objects.new("XPlaneObj", mesh)
        bpy.context.scene.collection.objects.link(obj)

        # Neutral material -- we never need PBR for triage.
        mat = bpy.data.materials.new(name="O4SFR_Triage_Neutral")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            try:
                bsdf.inputs["Roughness"].default_value = 0.85
            except KeyError:
                pass
            # Try to load the X-Plane texture as the diffuse map.
            if texture_rel:
                tex_path = os.path.normpath(
                    os.path.join(os.path.dirname(source), texture_rel)
                )
                if os.path.isfile(tex_path):
                    try:
                        img = bpy.data.images.load(tex_path, check_existing=True)
                        tex_node = mat.node_tree.nodes.new("ShaderNodeTexImage")
                        tex_node.image = img
                        mat.node_tree.links.new(
                            tex_node.outputs["Color"],
                            bsdf.inputs["Base Color"],
                        )
                    except Exception as exc:
                        print(f"[warn] texture load failed: {exc}")
            if not any(
                link.to_node is bsdf and link.to_socket.name == "Base Color"
                for link in mat.node_tree.links
            ):
                bsdf.inputs["Base Color"].default_value = (0.82, 0.78, 0.72, 1.0)
        obj.data.materials.append(mat)
        if not obj.data.uv_layers:
            obj.data.uv_layers.new(name="UVMap")

        # Frame the camera on the mesh.
        mn_x = mn_y = mn_z = inf
        mx_x = mx_y = mx_z = -inf
        for v in obj.bound_box:
            w = obj.matrix_world @ Vector(v)
            if w.x < mn_x: mn_x = w.x
            if w.x > mx_x: mx_x = w.x
            if w.y < mn_y: mn_y = w.y
            if w.y > mx_y: mx_y = w.y
            if w.z < mn_z: mn_z = w.z
            if w.z > mx_z: mx_z = w.z
        center = Vector(((mn_x + mx_x) / 2, (mn_y + mx_y) / 2, (mn_z + mx_z) / 2))
        size = max(mx_x - mn_x, mx_y - mn_y, mx_z - mn_z) or 1.0

        cam_dist = size * 2.5
        cam_loc = center + Vector((cam_dist * 0.7, -cam_dist * 0.7, cam_dist * 0.55))
        bpy.ops.object.camera_add(location=cam_loc)
        cam = bpy.context.object
        cam.data.lens = 35.0
        direction = center - cam.location
        cam.rotation_mode = "QUATERNION"
        cam.rotation_quaternion = direction.to_track_quat("-Z", "Z")
        bpy.context.scene.camera = cam

        bpy.ops.object.light_add(
            type="SUN", location=(center.x, center.y, mx_z + size)
        )
        sun = bpy.context.object
        sun.data.energy = 3.0
        sun.rotation_euler = (math.radians(50), 0, math.radians(45))

        world = bpy.context.scene.world or bpy.data.worlds.new("World")
        bpy.context.scene.world = world
        world.use_nodes = True
        bg = world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs[0].default_value = (0.7, 0.75, 0.8, 1.0)
            bg.inputs[1].default_value = 1.0

        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        try:
            scene.cycles.device = "CPU"
        except Exception:
            pass
        scene.cycles.samples = 16
        scene.cycles.use_denoising = False
        try:
            scene.view_settings.exposure = -0.5
        except Exception:
            pass
        scene.render.resolution_x = render_px
        scene.render.resolution_y = render_px
        scene.render.image_settings.file_format = "JPEG"
        scene.render.image_settings.quality = 85
        scene.render.filepath = output_image

        bpy.ops.render.render(write_still=True)
        '''
    ).strip() + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--triage", required=True)
    parser.add_argument("--blender",
                        default=os.environ.get("O4SFR_BLENDER", "blender"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--only-lib", default=None,
                        help="Restrict to one lib_id.")
    parser.add_argument("--force", action="store_true",
                        help="Re-render existing previews.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    triage = Path(args.triage)
    previews = triage / "previews"
    previews.mkdir(parents=True, exist_ok=True)
    with open(args.inventory, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.only_lib:
        rows = [r for r in rows if r["lib_id"] == args.only_lib]
    snippet_dir = Path(tempfile.mkdtemp(prefix="o4sfr_tra_"))
    snippet_path = snippet_dir / "render.py"
    snippet_path.write_text(_blender_render_snippet(), encoding="utf-8")
    ok = failed = skipped = 0
    try:
        for row in rows:
            slug = _slug(row["lib_id"], row["family_prefix"])
            out = previews / f"{slug}.jpg"
            if out.exists() and not args.force and out.stat().st_size > 4096:
                skipped += 1
                continue
            source = Path(row["representative_physical_path"])
            if not source.is_file():
                print(f"- {slug}: missing physical at {source}")
                failed += 1
                continue
            print(f"- {slug}")
            cmd = [
                args.blender, "--background", "--factory-startup",
                "--python", str(snippet_path.resolve()),
                "--",
                str(source.resolve()),
                str(out.resolve()),
                str(args.size),
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True,
                                        check=False, timeout=180)
            except subprocess.TimeoutExpired:
                print("  ! timeout")
                failed += 1
                continue
            if result.returncode == 0 and out.exists():
                ok += 1
            else:
                print(f"  ! render failed rc={result.returncode}")
                stderr_tail = (result.stderr or "").splitlines()[-3:]
                for line in stderr_tail:
                    print(f"     {line}")
                failed += 1
    finally:
        shutil.rmtree(snippet_dir, ignore_errors=True)
    print()
    print(f"rendered: {ok}   skipped: {skipped}   failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
