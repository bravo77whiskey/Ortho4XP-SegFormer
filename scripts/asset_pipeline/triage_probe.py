"""Compute bounding-box dimensions for each triaged 3D asset.

Reads --triage/inventory.csv, opens the model file referenced by each row,
extracts (bbox_x, bbox_y, bbox_z) in source units, and emits probe.csv.

Wavefront ``.obj`` is parsed in Python (text format, fast). FBX, BLEND, and
glTF require Blender; if --blender is given they are processed via a headless
snippet, otherwise rows are recorded with empty bbox fields and the
manifest_from_triage step falls back to the bucket-default footprint.

Source units are intentionally NOT guessed — downstream Blender conversion
rescales geometry to the manifest's footprint_m. The probe just gives the
triage step a real-vs-tiny-vs-huge signal so the footprint heuristic can
decide whether to trust the probe or fall back to a default.

Run:
    python scripts/asset_pipeline/triage_probe.py \\
        --triage scripts/asset_pipeline/triage \\
        --sources scripts/asset_pipeline/sources/sketchfab \\
        --blender "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe"
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


def parse_obj_bbox(obj_path: Path) -> tuple[float, float, float, int, int]:
    """Stream the .obj file once, returning (dx, dy, dz, verts, tris).

    Triangles are estimated as (sum(len(face) - 2) for face in f-lines).
    """
    min_x = min_y = min_z = float("inf")
    max_x = max_y = max_z = float("-inf")
    verts = 0
    tris = 0
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    except ValueError:
                        continue
                    if x < min_x: min_x = x
                    if x > max_x: max_x = x
                    if y < min_y: min_y = y
                    if y > max_y: max_y = y
                    if z < min_z: min_z = z
                    if z > max_z: max_z = z
                    verts += 1
            elif line.startswith("f "):
                n = len(line.split()) - 1
                if n >= 3:
                    tris += n - 2
    if verts == 0:
        return 0.0, 0.0, 0.0, 0, 0
    return (
        max_x - min_x,
        max_y - min_y,
        max_z - min_z,
        verts,
        tris,
    )


def _blender_probe_snippet() -> str:
    return textwrap.dedent(
        """
        import bpy
        import json
        import os
        import sys
        from math import inf
        from mathutils import Vector

        argv = sys.argv
        if "--" in argv:
            argv = argv[argv.index("--") + 1:]
        source, output_json = argv[0], argv[1]

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
                raise SystemExit(f"unsupported ext: {ext}")
        except Exception as exc:
            with open(output_json, "w", encoding="utf-8") as fh:
                json.dump({"error": str(exc)}, fh)
            sys.exit(0)

        meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
        if not meshes:
            with open(output_json, "w", encoding="utf-8") as fh:
                json.dump({"error": "no mesh"}, fh)
            sys.exit(0)

        # Apply transforms so bounding box reflects vertex world positions.
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

        min_x = min_y = min_z = inf
        max_x = max_y = max_z = -inf
        verts = 0
        tris = 0
        for o in meshes:
            verts += len(o.data.vertices)
            for poly in o.data.polygons:
                vlen = len(poly.vertices)
                if vlen >= 3:
                    tris += vlen - 2
            for v in o.bound_box:
                w = o.matrix_world @ Vector(v)
                if w.x < min_x: min_x = w.x
                if w.x > max_x: max_x = w.x
                if w.y < min_y: min_y = w.y
                if w.y > max_y: max_y = w.y
                if w.z < min_z: min_z = w.z
                if w.z > max_z: max_z = w.z

        result = {
            "bbox_x": max_x - min_x,
            "bbox_y": max_y - min_y,
            "bbox_z": max_z - min_z,
            "vertex_count": verts,
            "triangle_count": tris,
        }
        with open(output_json, "w", encoding="utf-8") as fh:
            json.dump(result, fh)
        """
    ).strip() + "\n"


def probe_via_blender(blender: str, snippet_path: str, source: Path) -> dict | None:
    out_json = source.parent / f".probe_{source.stem}.json"
    cmd = [
        blender, "--background", "--factory-startup",
        "--python", snippet_path,
        "--",
        str(source),
        str(out_json),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=180
        )
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except FileNotFoundError:
        return {"error": "blender-not-found"}
    if not out_json.exists():
        return {"error": f"no output (rc={result.returncode})"}
    try:
        with open(out_json, "r", encoding="utf-8") as fh:
            import json
            data = json.load(fh)
    except Exception as exc:
        data = {"error": f"json read: {exc}"}
    finally:
        out_json.unlink(missing_ok=True)
    return data


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triage", required=True,
                        help="Triage workspace (contains inventory.csv).")
    parser.add_argument("--sources", required=True,
                        help="Sources root; model_path in inventory is relative to this.")
    parser.add_argument("--blender", default=os.environ.get("O4SFR_BLENDER"),
                        help="Blender executable. If omitted, only .obj rows are probed.")
    parser.add_argument("--only", default=None,
                        help="Optional substring filter on slug.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    triage = Path(args.triage)
    sources = Path(args.sources)
    inventory_path = triage / "inventory.csv"
    probe_path = triage / "probe.csv"
    if not inventory_path.is_file():
        raise SystemExit(f"inventory not found: {inventory_path}")

    with open(inventory_path, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.only:
        needle = args.only.lower()
        rows = [r for r in rows if needle in r["slug"].lower()]

    snippet_dir = None
    snippet_path = None
    if args.blender:
        snippet_dir = Path(tempfile.mkdtemp(prefix="o4sfr_probe_"))
        snippet_path = snippet_dir / "probe.py"
        snippet_path.write_text(_blender_probe_snippet(), encoding="utf-8")

    results = []
    try:
        for row in rows:
            slug = row["slug"]
            model_rel = row.get("model_path") or ""
            fmt = (row.get("format") or "").lower()
            print(f"- {slug} [{fmt or 'unknown'}]")
            if not model_rel:
                results.append({
                    "slug": slug, "format": fmt,
                    "bbox_x": "", "bbox_y": "", "bbox_z": "",
                    "vertex_count": "", "triangle_count": "",
                    "error": "no model_path",
                })
                continue
            source = sources / model_rel
            if not source.is_file():
                results.append({
                    "slug": slug, "format": fmt,
                    "bbox_x": "", "bbox_y": "", "bbox_z": "",
                    "vertex_count": "", "triangle_count": "",
                    "error": f"missing file: {model_rel}",
                })
                continue
            if fmt == "obj":
                dx, dy, dz, verts, tris = parse_obj_bbox(source)
                results.append({
                    "slug": slug, "format": fmt,
                    "bbox_x": f"{dx:.4f}", "bbox_y": f"{dy:.4f}",
                    "bbox_z": f"{dz:.4f}",
                    "vertex_count": verts, "triangle_count": tris,
                    "error": "",
                })
                continue
            if not args.blender:
                results.append({
                    "slug": slug, "format": fmt,
                    "bbox_x": "", "bbox_y": "", "bbox_z": "",
                    "vertex_count": "", "triangle_count": "",
                    "error": "skipped (no --blender)",
                })
                continue
            probe = probe_via_blender(args.blender, str(snippet_path), source)
            if probe is None or "error" in probe:
                err = (probe or {}).get("error", "unknown")
                print(f"  ! probe failed: {err}")
                results.append({
                    "slug": slug, "format": fmt,
                    "bbox_x": "", "bbox_y": "", "bbox_z": "",
                    "vertex_count": "", "triangle_count": "",
                    "error": err,
                })
                continue
            results.append({
                "slug": slug, "format": fmt,
                "bbox_x": f"{probe['bbox_x']:.4f}",
                "bbox_y": f"{probe['bbox_y']:.4f}",
                "bbox_z": f"{probe['bbox_z']:.4f}",
                "vertex_count": probe["vertex_count"],
                "triangle_count": probe["triangle_count"],
                "error": "",
            })
    finally:
        if snippet_dir is not None:
            shutil.rmtree(snippet_dir, ignore_errors=True)

    fields = ["slug", "format", "bbox_x", "bbox_y", "bbox_z",
              "vertex_count", "triangle_count", "error"]
    with open(probe_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print()
    print(f"probe: {probe_path} ({len(results)} rows)")
    failed = [r for r in results if r["error"]]
    if failed:
        print(f"  {len(failed)} rows had errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
