"""Render a real 3/4-view preview image of every triaged model.

Sketchfab download zips ship only the model + raw textures -- they do NOT
include the model's preview render. The extractor's texture-priority
heuristic therefore picks a tileable diffuse map, which is useless for
identifying the building's region/style.

This script fixes that by importing each model into headless Blender,
framing it with a 3/4 view camera, adding a sun light, and saving a JPEG
preview. The new image overwrites the texture-tile preview from
triage_extract.py.

Run:
    python scripts/asset_pipeline/triage_render_previews.py \\
        --triage  scripts/asset_pipeline/triage \\
        --sources scripts/asset_pipeline/sources/sketchfab \\
        --blender "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" \\
        --size 512
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


def _blender_render_snippet() -> str:
    return textwrap.dedent(
        """
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

        ext = os.path.splitext(source)[1].lower()
        try:
            if ext == ".fbx":
                bpy.ops.import_scene.fbx(filepath=source)
            elif ext in (".gltf", ".glb"):
                bpy.ops.import_scene.gltf(filepath=source)
            elif ext == ".obj":
                bpy.ops.wm.obj_import(filepath=source)
            elif ext == ".blend":
                with bpy.data.libraries.load(source, link=False) as (src, dst):
                    dst.objects = list(src.objects)
                for obj in dst.objects:
                    if obj is not None:
                        bpy.context.collection.objects.link(obj)
            else:
                sys.exit(2)
        except Exception as exc:
            print(f"[err] import failed: {exc}", file=sys.stderr)
            sys.exit(3)

        # Delete non-mesh leftovers from import (cameras, lights, empties),
        # which otherwise inflate scene bounds and steal the camera frame.
        for obj in list(bpy.context.scene.objects):
            if obj.type not in ("MESH",):
                bpy.data.objects.remove(obj, do_unlink=True)

        meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
        if not meshes:
            print("[err] no mesh objects", file=sys.stderr)
            sys.exit(4)

        # Apply transforms so each mesh's bound_box is in world coords.
        bpy.ops.object.select_all(action="DESELECT")
        for o in meshes:
            o.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        try:
            bpy.ops.object.transform_apply(
                location=True, rotation=True, scale=True
            )
        except Exception:
            pass

        def bbox_of(objs):
            mn = Vector((inf, inf, inf))
            mx = Vector((-inf, -inf, -inf))
            for o in objs:
                for v in o.bound_box:
                    w = o.matrix_world @ Vector(v)
                    if w.x < mn.x: mn.x = w.x
                    if w.x > mx.x: mx.x = w.x
                    if w.y < mn.y: mn.y = w.y
                    if w.y > mx.y: mx.y = w.y
                    if w.z < mn.z: mn.z = w.z
                    if w.z > mx.z: mx.z = w.z
            return mn, mx

        # Pick "the building": the mesh with the most vertices. Sketchfab
        # downloads frequently ship a backdrop / pedestal / ground plane that
        # dominates the bounding box; counting vertices instead of volume is
        # the most robust way to identify the actual model.
        primary = max(meshes, key=lambda o: len(o.data.vertices))
        p_min, p_max = bbox_of([primary])
        p_size_xy = max(p_max.x - p_min.x, p_max.y - p_min.y) or 1.0
        p_size_z = (p_max.z - p_min.z) or 1.0

        # Include any sibling mesh whose bbox overlaps the primary's expanded
        # bbox -- this keeps balconies / detached chimneys / fences attached
        # to the same physical building, while still dropping huge backdrops.
        pad = max(p_size_xy, p_size_z)
        keep = []
        for o in meshes:
            mn, mx = bbox_of([o])
            ox = (mx.x + mn.x) / 2
            oy = (mx.y + mn.y) / 2
            if (abs(ox - (p_min.x + p_max.x) / 2) <= pad and
                    abs(oy - (p_min.y + p_max.y) / 2) <= pad):
                keep.append(o)
        s_min, s_max = bbox_of(keep)
        center = Vector((
            (s_min.x + s_max.x) / 2.0,
            (s_min.y + s_max.y) / 2.0,
            (s_min.z + s_max.z) / 2.0,
        ))
        size = max(
            s_max.x - s_min.x,
            s_max.y - s_min.y,
            s_max.z - s_min.z,
        ) or 1.0

        # 3/4 hero shot, framed so the building fills ~70% of the frame.
        # Wide-ish 35mm lens gives a satisfying perspective for buildings.
        cam_dist = size * 2.5
        cam_loc = center + Vector((
            cam_dist * 0.7,
            -cam_dist * 0.7,
            cam_dist * 0.55,
        ))
        bpy.ops.object.camera_add(location=cam_loc)
        cam = bpy.context.object
        cam.data.lens = 35.0
        direction = center - cam.location
        cam.rotation_mode = "QUATERNION"
        cam.rotation_quaternion = direction.to_track_quat("-Z", "Z")
        bpy.context.scene.camera = cam

        bpy.ops.object.light_add(
            type="SUN", location=(center.x, center.y, s_max.z + size)
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
        # Workbench renders through the camera correctly in --background,
        # is fast (no shaders), and immune to FBX material/texture import
        # quirks. Studio matcap + cavity shading gives high-contrast
        # geometric detail that's ideal for region/style triage.
        # Sketchfab FBXs ship materials whose Image Texture nodes lose their
        # texture references during import, leaving meshes pink. Clear all
        # materials and assign a single neutral one so the geometry is what
        # the assistant sees, not a missing-texture magenta.
        neutral = bpy.data.materials.new(name="O4SFR_Triage_Neutral")
        neutral.use_nodes = True
        principled = neutral.node_tree.nodes.get("Principled BSDF")
        if principled:
            principled.inputs["Base Color"].default_value = (0.78, 0.74, 0.68, 1.0)
            try:
                principled.inputs["Roughness"].default_value = 0.85
            except KeyError:
                pass
            try:
                principled.inputs["Specular IOR Level"].default_value = 0.3
            except KeyError:
                try:
                    principled.inputs["Specular"].default_value = 0.3
                except KeyError:
                    pass
        for o in [obj for obj in scene.objects if obj.type == "MESH"]:
            o.data.materials.clear()
            o.data.materials.append(neutral)

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
        """
    ).strip() + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triage", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--blender",
                        default=os.environ.get("O4SFR_BLENDER", "blender"))
    parser.add_argument("--size", type=int, default=512,
                        help="Square render resolution.")
    parser.add_argument("--only", default=None,
                        help="Optional substring filter on slug.")
    parser.add_argument("--force", action="store_true",
                        help="Re-render even if a preview already exists.")
    parser.add_argument("--skip-format", action="append", default=[],
                        help="Skip model formats (repeatable). e.g. --skip-format blend")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    triage = Path(args.triage)
    sources = Path(args.sources)
    previews = triage / "previews"
    previews.mkdir(parents=True, exist_ok=True)
    inventory = triage / "inventory.csv"
    if not inventory.is_file():
        raise SystemExit(f"inventory.csv missing: {inventory}")

    with open(inventory, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.only:
        needle = args.only.lower()
        rows = [r for r in rows if needle in r["slug"].lower()]
    skip_fmts = {f.lower().lstrip(".") for f in args.skip_format}

    snippet_dir = Path(tempfile.mkdtemp(prefix="o4sfr_render_"))
    snippet_path = snippet_dir / "render.py"
    snippet_path.write_text(_blender_render_snippet(), encoding="utf-8")

    ok = failed = skipped = 0
    try:
        for row in rows:
            slug = row["slug"]
            model_rel = row.get("model_path") or ""
            fmt = (row.get("format") or "").lower()
            preview_path = previews / f"{slug}.jpg"

            if not model_rel:
                print(f"- {slug}: no model_path, skipping")
                skipped += 1
                continue
            if fmt in skip_fmts:
                print(f"- {slug}: skipping (format={fmt})")
                skipped += 1
                continue
            source = sources / model_rel
            if not source.is_file():
                print(f"- {slug}: model missing on disk, skipping")
                skipped += 1
                continue
            # Marker file for re-runs: an existing render exists if file is > 4 KB
            # (texture-tile thumbnails the extractor wrote can be smaller).
            if preview_path.exists() and not args.force:
                if preview_path.stat().st_size > 4096:
                    print(f"- {slug}: already rendered, skipping")
                    ok += 1
                    continue

            print(f"- {slug} [{fmt}]")
            # Pass absolute paths to Blender; in --background mode it resolves
            # relative paths against an unpredictable working directory (often
            # the user's home), not Python's CWD.
            cmd = [
                args.blender, "--background", "--factory-startup",
                "--python", str(snippet_path.resolve()),
                "--",
                str(source.resolve()),
                str(preview_path.resolve()),
                str(args.size),
            ]
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True,
                    check=False, timeout=240,
                )
            except subprocess.TimeoutExpired:
                print(f"  ! timeout")
                failed += 1
                continue
            except FileNotFoundError:
                raise SystemExit(f"Blender not found at {args.blender!r}")
            if result.returncode == 0 and preview_path.exists():
                ok += 1
            else:
                print(f"  ! render failed rc={result.returncode} "
                      f"exists={preview_path.exists()}")
                stdout_tail = (result.stdout or "").splitlines()[-8:]
                stderr_tail = (result.stderr or "").splitlines()[-8:]
                for line in stdout_tail:
                    print(f"     out: {line}")
                for line in stderr_tail:
                    print(f"     err: {line}")
                failed += 1
    finally:
        shutil.rmtree(snippet_dir, ignore_errors=True)

    print()
    print(f"rendered: {ok}   failed: {failed}   skipped: {skipped}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
