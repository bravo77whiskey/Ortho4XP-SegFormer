"""Headless-Blender batch converter: arbitrary 3D models -> X-Plane .obj.

Reads the same sources.yaml manifest used by build_custom_library.py and, for
each entry whose physical .obj is missing under --output, invokes Blender in
``--background`` mode with a small Python snippet that:

  1. Imports the source file (.fbx / .glb / .gltf / .obj / .blend).
  2. Re-centers the model on the origin and aligns the base to z=0.
  3. Scales to the manifest's footprint_m if --resize is given.
  4. Ensures every mesh has a UV map + a material (xplane2blender requires both).
  5. Exports via the official xplane2blender addon (File > Export > XPlane Object).

The Blender snippet itself lives in ``_blender_convert_snippet`` below; we
write it to a temp file and call::

    blender --background --python <snippet>.py -- \
        <source_file> <output_obj> <width_m> <depth_m>

This file is therefore safe to import outside Blender (its top-level only
parses args and shells out to Blender). The snippet is best-effort -- some
source models will need manual cleanup before they convert cleanly. Failures
are reported and the script moves on to the next entry.

Prerequisites on the host:
  * Blender 3.6+ (4.x preferred).
  * The xplane2blender addon installed and enabled in Blender's user prefs.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

# Reuse the manifest schema from the sibling script.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from build_custom_library import load_manifest, Asset  # noqa: E402


def _blender_convert_snippet() -> str:
    """Return the Python snippet executed inside Blender."""
    return textwrap.dedent(
        """
        import bpy
        import sys
        import os
        from math import inf
        from mathutils import Vector

        argv = sys.argv
        if "--" in argv:
            argv = argv[argv.index("--") + 1:]
        (source, output_obj, width_m, depth_m,
         decimate_target, texture_max_px) = (
            argv[0], argv[1],
            float(argv[2]), float(argv[3]),
            int(argv[4]), int(argv[5]),
        )

        bpy.ops.wm.read_factory_settings(use_empty=True)

        # --factory-startup disables every user addon. xplane2blender is what
        # we need for the export call below, so re-enable it explicitly. The
        # module name varies between releases; try the common ones in order.
        _xp_enabled = False
        for _mod in ("io_xplane2blender", "xplane2blender", "XPlane2Blender"):
            try:
                bpy.ops.preferences.addon_enable(module=_mod)
                _xp_enabled = True
                break
            except Exception:
                continue
        if not _xp_enabled:
            raise SystemExit(
                "xplane2blender addon not found. Install it via "
                "Edit > Preferences > Add-ons > Install... and Save Preferences "
                "before running this script."
            )

        ext = os.path.splitext(source)[1].lower()
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
            raise SystemExit(f"Unsupported source extension: {ext}")

        meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
        if not meshes:
            raise SystemExit("Source contains no mesh objects")

        # Bake imported transforms (FBX/glTF importers leave 90deg rotations
        # and 0.01 scale unapplied -- xplane2blender exports raw vertices and
        # ignores the object transform, so we apply here).
        bpy.ops.object.select_all(action="DESELECT")
        for o in meshes:
            o.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.transform_apply(
            location=True, rotation=True, scale=True
        )

        # Walk world bbox.  Vertices == world coords after transform_apply
        # above so o.matrix_world is the identity, but we still apply it for
        # safety in case Blender re-normalised something.
        def _world_bbox():
            mn_x = mn_y = mn_z = inf
            mx_x = mx_y = mx_z = -inf
            for o in meshes:
                for v in o.bound_box:
                    w = o.matrix_world @ Vector(v)
                    if w.x < mn_x: mn_x = w.x
                    if w.x > mx_x: mx_x = w.x
                    if w.y < mn_y: mn_y = w.y
                    if w.y > mx_y: mx_y = w.y
                    if w.z < mn_z: mn_z = w.z
                    if w.z > mx_z: mx_z = w.z
            return mn_x, mn_y, mn_z, mx_x, mx_y, mx_z

        # SCALE FIRST, then centre.  Earlier versions did this the other way
        # round; the un-baked location offset survived the scale step and the
        # building ended up hovering or off to the side of the marker.
        if width_m > 0 and depth_m > 0:
            mn_x, mn_y, mn_z, mx_x, mx_y, mx_z = _world_bbox()
            cur_w = (mx_x - mn_x) or 1.0
            cur_d = (mx_y - mn_y) or 1.0
            sx = width_m / cur_w
            sy = depth_m / cur_d
            sz = (sx + sy) / 2.0
            for o in meshes:
                o.scale.x *= sx
                o.scale.y *= sy
                o.scale.z *= sz
            bpy.ops.object.transform_apply(
                location=False, rotation=False, scale=True
            )

        # Now centre on the origin (XY) and ground the base (Z), using the
        # post-scale bbox, and bake the translation so vertices are at the
        # final positions xplane2blender will see.
        mn_x, mn_y, mn_z, mx_x, mx_y, mx_z = _world_bbox()
        dx = -(mn_x + mx_x) / 2.0
        dy = -(mn_y + mx_y) / 2.0
        dz = -mn_z
        for o in meshes:
            o.location.x += dx
            o.location.y += dy
            o.location.z += dz
        bpy.ops.object.transform_apply(
            location=True, rotation=False, scale=False
        )

        # Auto-decimate any mesh that's denser than decimate_target tris.
        # Photogrammetry scans frequently land at 100k-1M+ triangles; X-Plane
        # instances 100s of these per tile so they must be lean.
        for o in meshes:
            tri_count = sum(len(p.vertices) - 2 for p in o.data.polygons)
            if tri_count <= decimate_target:
                continue
            ratio = decimate_target / float(tri_count)
            modifier = o.modifiers.new(name="O4SFR_Decimate", type="DECIMATE")
            modifier.decimate_type = "COLLAPSE"
            modifier.ratio = max(0.01, min(1.0, ratio))
            modifier.use_collapse_triangulate = True
            bpy.context.view_layer.objects.active = o
            try:
                bpy.ops.object.modifier_apply(modifier=modifier.name)
            except Exception as exc:  # pragma: no cover -- Blender-internal
                print(f"[warn] decimate failed for {o.name}: {exc}")

        # Downsize any image textures whose larger dimension exceeds
        # texture_max_px. 4K photogrammetry albedos are common; X-Plane's
        # VRAM budget per instance dictates we cap at ~1K for residential.
        seen_images = set()
        for image in bpy.data.images:
            if image.name in seen_images or not image.has_data:
                continue
            seen_images.add(image.name)
            w, h = image.size[0], image.size[1]
            if max(w, h) <= texture_max_px:
                continue
            scale = texture_max_px / float(max(w, h))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            image.scale(new_w, new_h)

        # Guarantee material + UV for xplane2blender.
        for o in meshes:
            if not o.data.materials:
                mat = bpy.data.materials.new(name=f"{o.name}_mat")
                o.data.materials.append(mat)
            if not o.data.uv_layers:
                o.data.uv_layers.new(name="UVMap")

        os.makedirs(os.path.dirname(output_obj), exist_ok=True)

        # Save downsized images next to the .obj so xplane2blender's TEXTURE
        # directive resolves at runtime.
        output_dir = os.path.dirname(output_obj)
        output_stem = os.path.splitext(os.path.basename(output_obj))[0]
        for image in bpy.data.images:
            if not image.has_data or image.size[0] == 0:
                continue
            target_path = os.path.join(output_dir, output_stem + ".png")
            image.filepath_raw = target_path
            image.file_format = "PNG"
            try:
                image.save()
                break  # Only the first/primary texture lands beside the obj.
            except Exception as exc:  # pragma: no cover
                print(f"[warn] texture save failed: {exc}")

        # Configure xplane2blender to treat a dedicated collection as a
        # Scenery OBJ8 exportable root.  Create a fresh "O4SFR_Export"
        # collection, move every mesh into it, mark it exportable, and set
        # the layer name + export type.  Doing it on a brand-new collection
        # avoids xplane2blender quirks with the Master ("Scene") collection
        # or with FBX-importer-created sub-collections that the addon's
        # property hooks may not have decorated yet.
        export_coll = bpy.data.collections.new(name="O4SFR_Export")
        bpy.context.scene.collection.children.link(export_coll)
        # Move every mesh from wherever it lives into the export collection.
        for o in [obj for obj in bpy.data.objects if obj.type == "MESH"]:
            for parent_coll in list(o.users_collection):
                try:
                    parent_coll.objects.unlink(o)
                except RuntimeError:
                    pass
            export_coll.objects.link(o)
        # Force a depsgraph update so xplane2blender's property hooks see
        # the new collection before we touch its .xplane sub-tree.
        bpy.context.view_layer.update()
        xp = getattr(export_coll, "xplane", None)
        if xp is None:
            print("[err] export_coll.xplane property group missing -- "
                  "addon not fully registered")
            raise SystemExit(5)
        xp.is_exportable_collection = True
        xp.layer.name = output_stem
        try:
            xp.layer.export_type = "scenery"
        except Exception:
            pass
        print(f"[ok] marked O4SFR_Export as exportable, "
              f"{len(export_coll.all_objects)} objects, "
              f"layer.name={output_stem!r}")

        # Invoke xplane2blender. filepath is the output directory.
        try:
            bpy.ops.export.xplane_obj(filepath=output_dir + os.sep)
        except Exception:
            bpy.ops.xplane.export_objects(filepath=output_dir + os.sep)

        # Post-process: inject a TEXTURE directive when xplane2blender omitted
        # one.  Sketchfab FBXs commonly import with detached Image Texture
        # nodes; even after we sidestep them with a neutral material for
        # rendering, the exporter has no albedo image bound to write a
        # TEXTURE line.  We already saved the source's primary PNG next to
        # the .obj as <stem>.png -- splice that into the .obj header.
        final_obj = os.path.join(output_dir, output_stem + ".obj")
        texture_basename = output_stem + ".png"
        texture_path = os.path.join(output_dir, texture_basename)
        if os.path.isfile(final_obj) and os.path.isfile(texture_path):
            try:
                with open(final_obj, "r", encoding="utf-8") as fh:
                    obj_text = fh.read()
                has_texture = any(
                    line.lstrip().startswith("TEXTURE")
                    and not line.lstrip().startswith("TEXTURE_")
                    for line in obj_text.splitlines()
                )
                if not has_texture:
                    new_lines = []
                    inserted = False
                    for line in obj_text.splitlines():
                        new_lines.append(line)
                        if (not inserted
                                and line.strip() in ("OBJ", "OBJ8")):
                            new_lines.append("")
                            new_lines.append(f"TEXTURE {texture_basename}")
                            inserted = True
                    if not inserted:
                        # Couldn't find header marker; prepend at top.
                        new_lines = (
                            ["A", "800", "OBJ", "",
                             f"TEXTURE {texture_basename}", ""]
                            + obj_text.splitlines()
                        )
                    with open(final_obj, "w", encoding="utf-8") as fh:
                        fh.write("\n".join(new_lines) + "\n")
                    print(f"[ok] injected TEXTURE {texture_basename}")
            except Exception as exc:
                print(f"[warn] TEXTURE inject failed: {exc}")
        """
    ).strip() + "\n"


_DIFFUSE_PRIORITY_TOKENS = (
    "base", "diffuse", "albedo", "color", "_d.", "_clr", "_col", "rgb",
)


def _find_source_diffuse_texture(source_path: str) -> str | None:
    """Locate the primary diffuse texture next to the source model.

    Sketchfab archives ship textures in a sibling ``textures/`` directory;
    pick the file whose name most strongly suggests it's the diffuse map and
    is the largest among ties.
    """
    source_dir = os.path.dirname(os.path.dirname(source_path))
    candidates: list[str] = []
    for root, _dirs, files in os.walk(source_dir):
        for name in files:
            lower = name.lower()
            if lower.endswith((".png", ".jpg", ".jpeg", ".tga", ".webp")):
                candidates.append(os.path.join(root, name))
    if not candidates:
        return None

    def score(p):
        n = os.path.basename(p).lower()
        for i, token in enumerate(_DIFFUSE_PRIORITY_TOKENS):
            if token in n:
                return (i, -os.path.getsize(p))
        return (len(_DIFFUSE_PRIORITY_TOKENS), -os.path.getsize(p))

    candidates.sort(key=score)
    return candidates[0]


def _ensure_texture_alongside_obj(asset: Asset, source_path: str,
                                  output_obj: str) -> str | None:
    """Copy the source's primary diffuse texture next to the .obj.

    Returns the basename of the texture file on success, else None.
    """
    src = _find_source_diffuse_texture(source_path)
    if src is None:
        return None
    output_dir = os.path.dirname(output_obj)
    stem = os.path.splitext(os.path.basename(output_obj))[0]
    # Always emit as .png for X-Plane simplicity; we leave non-PNG sources
    # alone and trust X-Plane to load .jpg/.tga if that's what we copy.
    ext = os.path.splitext(src)[1].lower()
    if ext == ".jpeg":
        ext = ".jpg"
    dst = os.path.join(output_dir, stem + ext)
    try:
        shutil.copyfile(src, dst)
        return os.path.basename(dst)
    except OSError:
        return None


def _ensure_obj_texture_directive(output_obj: str, texture_basename: str):
    """Splice a ``TEXTURE`` line into the .obj if xplane2blender omitted one."""
    try:
        with open(output_obj, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return
    for line in text.splitlines():
        s = line.lstrip()
        if s.startswith("TEXTURE") and not s.startswith("TEXTURE_"):
            return  # Already present.
    new_lines: list[str] = []
    inserted = False
    for line in text.splitlines():
        new_lines.append(line)
        if not inserted and line.strip() in ("OBJ", "OBJ8"):
            new_lines.append("")
            new_lines.append(f"TEXTURE {texture_basename}")
            inserted = True
    if not inserted:
        new_lines = (
            ["A", "800", "OBJ", "", f"TEXTURE {texture_basename}", ""]
            + text.splitlines()
        )
    try:
        with open(output_obj, "w", encoding="utf-8") as fh:
            fh.write("\n".join(new_lines) + "\n")
    except OSError:
        pass


def _run_blender(blender: str, snippet_path: str, asset: Asset,
                 source_root: str, output_dir: str) -> bool:
    source_path = os.path.join(source_root, asset.source_file)
    if not os.path.isfile(source_path):
        print(f"  ! source missing: {source_path}")
        return False
    output_obj = os.path.join(output_dir, asset.physical_path)
    cmd = [
        blender, "--background", "--factory-startup",
        "--python", snippet_path,
        "--",
        source_path, output_obj,
        str(asset.footprint_m[0]), str(asset.footprint_m[1]),
        str(asset.decimate_target), str(asset.texture_max_px),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise SystemExit(f"Blender not found at {blender!r}: {exc}") from exc
    if result.returncode != 0 or not os.path.isfile(output_obj):
        print(f"  ! conversion failed for {asset.id}")
        if result.stderr:
            for line in result.stderr.splitlines()[-10:]:
                print(f"      {line}")
        return False

    # Copy a diffuse texture from the source archive next to the .obj and
    # make sure the .obj's TEXTURE directive points at it.  We do this in
    # Python (not in the Blender snippet) because image.save() inside
    # --background mode silently fails when no image data has been loaded
    # into bpy.data.images, which is common for FBX imports.
    texture_basename = _ensure_texture_alongside_obj(
        asset, source_path, output_obj
    )
    if texture_basename:
        _ensure_obj_texture_directive(output_obj, texture_basename)
    return True


def convert(args):
    assets = load_manifest(args.manifest)
    if not assets:
        raise SystemExit("Manifest contains no assets.")
    os.makedirs(args.output, exist_ok=True)
    snippet_dir = tempfile.mkdtemp(prefix="o4sfr_blender_")
    snippet_path = os.path.join(snippet_dir, "convert.py")
    try:
        with open(snippet_path, "w", encoding="utf-8") as fh:
            fh.write(_blender_convert_snippet())
        ok = 0
        failed: list[Asset] = []
        for asset in assets:
            output_obj = os.path.join(args.output, asset.physical_path)
            if os.path.isfile(output_obj) and not args.force:
                ok += 1
                continue
            print(f"convert {asset.id} ({asset.region}/{asset.bucket})")
            if _run_blender(args.blender, snippet_path, asset,
                            args.source_root, args.output):
                ok += 1
            else:
                failed.append(asset)
        print(f"converted {ok}/{len(assets)}; failed={len(failed)}")
        if failed:
            for asset in failed:
                print(f"  - {asset.id}: {asset.source_file}")
            return 1
        return 0
    finally:
        shutil.rmtree(snippet_dir, ignore_errors=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True,
                        help="Path to sources.yaml.")
    parser.add_argument("--output", required=True,
                        help="Custom Scenery folder to build into "
                             "(physical assets land under <output>/<region>/<bucket>/).")
    parser.add_argument("--source-root", default=".",
                        help="Directory the manifest's source_file paths are relative to.")
    parser.add_argument("--blender",
                        default=os.environ.get("O4SFR_BLENDER", "blender"),
                        help="Path to the Blender executable.")
    parser.add_argument("--force", action="store_true",
                        help="Re-convert assets whose .obj already exists.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return convert(args)


if __name__ == "__main__":
    sys.exit(main())
