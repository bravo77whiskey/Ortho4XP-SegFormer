"""Procedural X-Plane building-asset generation (O4SFR ProcGen Library).

Modules:
  grid       -- enumerate the dimension grid into procgen_manifest.json
  atlas      -- PIL-generated facade/roof texture atlas + layout JSON
  archetypes -- pure-Python parametric building mesh builders (no bpy)
  blender_worker -- runs inside Blender; meshes -> xplane2blender OBJ8
  generate_procedural_buildings -- host-side batch driver
"""
