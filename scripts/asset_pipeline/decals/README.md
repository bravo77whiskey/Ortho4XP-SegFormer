# O4XP ortho decal library

The detail decal that `use_decal_on_terrain` puts on ortho terrain, without
the grid.

X-Plane's `maquify_2_green_key.dcl` repeats one 1024 px shrub/dirt texture
every 25 m. Over any even field the repeat reads as a lattice. Measured with
`shader_sim.py` on 12 windows of a ZL16 and a ZL18 hotspot ortho (20 cm/px),
the decal's contribution correlates 0.77-0.90 with itself at a 25 m shift
(1.0 = exact repeat). With this library the highest correlation at any shift
is 0.37-0.47, at 162 m, and the detail contrast is unchanged. The naive
alternative, the same 25 m texture in both slots, still scores 0.52-0.64 and
doubles the contrast.

This library keeps Laminar's pixels and colour keys, but:

* `quilt.py` re-synthesises the texture by toroidal image quilting into
  4096 px textures that tile seamlessly with no repeat inside;
* both decal slots are used: layer A at the source density (100 m period),
  layer B 1.618x coarser (161.8 m). The irrational ratio means the sum never
  repeats. Each layer carries 1/sqrt(2) of the source contrast.

| Step | File |
| --- | --- |
| Texture synthesis | `quilt.py` |
| Offline model of XP12's decal shader + repetition score | `shader_sim.py` |
| Library package (DDS via nvcompress, `.dcl`, `library.txt`) | `build_decals.py` |
| Switch built tiles' `.ter` files between decals | `set_terrain_decal.py` |

```powershell
python build_decals.py --out "<Custom Scenery>\O4XP_Decal_Library"
python set_terrain_decal.py <tile dir> --decal o4xp
```

Tiles reference `DECAL_LIB o4xp/decals/ortho_detail.dcl` when
`terrain_decal = o4xp` (the default). The pack must be installed, or X-Plane
can't resolve the decal. `terrain_decal = xplane` goes back to Laminar's
decal.

The source texture is read from the local X-Plane install. Its pixels never
enter this repository, only the generated package.
