# ProcGen v4: regional building library, baked from 3-D detail

A drop-in replacement for the old `O4SFR_ProcGen_Library`. It keeps the same
virtual paths, footprints, floors, regions, seeds and physical paths as the
old manifest, so the overlay and existing DSFs need no change.

There is no image generation and no photo textures. Every texture pixel is
rendered in Blender Cycles from real 3-D relief. Each window is a recess with
a reveal, a bevelled frame, sashes, a sill, shutters, blinds and a room behind
the glass. Bricks and tiles are individual pieces with gaps, and roof tiles
overlap with rounded noses. A post pass then adds wear driven by the bake's
own masks.

## Pipeline

| Step | File | Runs in |
| --- | --- | --- |
| Regional data: facade schemes, roofs, trims, massing | `regions.py`, `kits.py` | host |
| Strip relief recipes (windows, doors, loggias, docks, tiles, sheets) | `strips.py`, `kitgeo.py` | host + Blender |
| Bake one strip: orthographic Cycles render under an overcast dome | `bake_blender.py`, `bake_materials.py` | Blender 4.5, OptiX |
| Wear, glass, packing into albedo and normal/metal/gloss | `weather.py` | host (numpy) |
| Page assembly, cache, multi-process | `bake_pages.py` | host |
| Low-poly archetypes, 4 LOD bands, OBJ8 writer | `buildings.py`, `meshkit.py` | host |
| Full library + `library.txt` | `build_library.py` | host |
| Performance contract against the old library | `budget.py` | host |
| Review renders of the exported OBJ8s | `samples.py`, `render_preview.py` | host + Blender |

```powershell
python scripts/asset_pipeline/procgen_v4/bake_pages.py --out <pkg>          # ~30 min, cached per strip
python scripts/asset_pipeline/procgen_v4/build_library.py --out <pkg>       # ~1 min, 104,466 OBJs
python scripts/asset_pipeline/procgen/verify_procgen_library.py --output <pkg>
python scripts/asset_pipeline/procgen_v4/budget.py --all
python scripts/asset_pipeline/procgen_v4/samples.py --pkg <pkg> --out <dir> --regions europe asia --render
python scripts/asset_pipeline/procgen_v4/samples.py --pkg <pkg> --out <dir> --sheets --render --regions europe north_america mediterranean asia africa south_america australia_oceania generic
```

## Textures

Per region there are two residential pages (colourways A/B) and one
apartment/commercial page. One industrial page is shared in three wear
grades: clean, mild and worn. Wear is a post step, so the three grades come
from one bake. Every page is 4096² albedo plus a 4096² normal map (RG tangent
normal, B base reflectance for `NORMAL_METALNESS`, A gloss).

Every strip spans the full page width and wraps in u. Strips are stacked
tallest-first, so each one starts on a multiple of its own power-of-two
height. Box-filtered mips then never mix two strips until a strip is a single
texel. The old atlas's arbitrary strip heights caused its "rainbow roof"
rings; this layout removes that cause instead of working around it.

A tile loads at most two residential pages, one apartment page and one
industrial page. That is 4 albedo and 4 normal pages, 134 Mpx in all. The old
library loaded 12 albedo 4096² pages and 12 normal 2048² pages, 251 Mpx. The
new library uses about 47% less texture memory per tile.

### Wear and repetition

Wear (`weather.py`) is layered from the bake's own masks and a per-region
climate profile (`WEAR`), and every region weathers (Asia included since
2026-09-25).

* **Walls:** an exposure field along the strip, so one building's stretch
  can be grimy while the next is fresh. On top of that: run-off threads from
  above, streaks leaving sill ends, rust runs below bars, rails and AC
  brackets, grime in occluded reveals, and mould near the ground and under
  sills. Rendered walls also get repaint blocks and peeling, and brick gets
  efflorescence. Every wall gets sun fade, and ground floors get soil
  splash-back.
* **Pitched roofs:** faded and replaced pieces (per-tile random from the
  bake), dirt toward the eave, black algae streaks down-slope, lichen, moss
  in the laps, rust runs on metal, and dust.
* **Flat roofs:** ponding tide marks, mould blooms and blotchy grime.
* **Rooftop equipment** is baked flat into the roof strips, clustered so
  some roofs stay bare, with no geometry cost (`strips.roof_objects`,
  `regions.ROOF_OBJECTS`). Examples: tanks, AC units and banks, hatches,
  sheds, solar water heaters, panels, skylights, vents and dishes.

Upper floors repeat a short bay motif (`strips.motif_len`). Each floor of a
wall is shifted by a whole number of motifs (`buildings.facade_wall`), so
windows stay stacked but no stain repeats floor after floor.

## Performance contract (`budget.py`)

Each new asset is checked against the old asset built from the same manifest
row: same archetype, footprint, floors, region and seed.

* 4 contiguous `ATTR_LOD` bands. The cull distance and the old size scaling
  (`diag/30 m`, clamped 0.32–1.65) are identical.
* One texture page and one material per OBJ, with no alpha blending, no
  animation and no lights, so instancing and batching behave exactly as
  before. Variant counts are unchanged (same manifest).
* No rooftop clutter cubes, no protruding balconies (loggias are recessed in
  the texture), and flat roofs have a coplanar coping ring. All three were
  rejected before.
* Frame load is modelled as `Σ tris_band × (far² − near²)`, because instance
  count grows with ring area.
* Smooth LOD (2026-09-27): no band switch may move, recolour or drop
  anything a viewer can see at that distance. Band 1 (full model) ends at
  0.8× the old full-detail range. Band 2 keeps band 1's walls, roof tile
  scale and offset, chimneys, dormers, porch roofs, shop canopies and
  recessed fronts, coping, eave boards and per-unit terrace colours, and
  drops only features a pixel or two wide at the switch (soffits, barge
  boards, ridge caps, porch and arcade posts). Band 3 keeps the window
  texture: one quad per floor up to two floors, otherwise the ground floor
  plus one stretched quad for the upper floors, so the facade's mean tone
  no longer brightens into a plain slab. Band 4 is still one roof quad.
  Rendered at each switch distance (2560 px, 60° view), the first switch
  now changes 0.0–0.6% of pixels in a building crop, down from 3–7%.
* Cost: the modelled load is **1.36× the old library (median 1.28)**.
  Terraces carry most of it (up to 4.4×, about 6% of OBJs). Instance
  counts, texture pages and batching are unchanged.

## Not done yet

* No in-sim frame-time measurement. The contract is a model, not a
  benchmark; the smooth-LOD round made that check more important.
* No night lighting (`TEXTURE_LIT`).
* No normal-map mip tuning: the old "shimmer" suspect was `GLOBAL_specular`
  with high gloss. v4 keeps `GLOBAL_specular 1.0` but walls carry
  gloss ≈ 0.1 in the alpha channel.
* Sub-regional routing. The overlay's region aliases let a `europe` tile draw
  from `scandinavia` and `mediterranean` too, so Nordic timber or a British
  terrace family would need overlay routing first. The kits keep each
  region to its mainstream vernacular.
