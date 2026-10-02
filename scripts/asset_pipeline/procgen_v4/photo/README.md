# ProcGen v4: CC0 photo-material pages (prototype)

This is the image-gen idea with real photographs instead of generated
pixels. Wall, plinth, lintel and roof surfaces sample photoscanned textures
from [Poly Haven](https://polyhaven.com/license). Those are CC0: commercial
use, modification and redistribution are allowed, with no attribution
required. Everything else is the normal v4 Cycles bake: the window, door
and shutter relief, the room behind the glass, the occlusion and the page
layout. So every window sits exactly where the OBJ UVs expect it, with no
snapping step. Geometry, UVs, bay tables and the perf contract are
unchanged.

| Step | File | What |
| --- | --- | --- |
| Sources | `photo_kits.py` | `ASSETS` (Poly Haven id → resolution) and the per-material mapping (`EUROPE_RES_A`): asset, palette target, tint, AO exponent, bump distance, `rot90`, `flatten`. |
| Download | `fetch_cc0.py` | Colour, displacement and AO maps per asset, md5-checked, plus `LICENSES.json` (asset page, authors, real size, URLs, hashes). |
| Bake | `build_photo_pages.py` | Patches the europe res A kit with `photo` blocks and runs `bake_pages.py` for that page. Default wear is 0, because the photos are real, already-weathered surfaces. `--base` first copies another package's pages. |
| Review | `render_photo_compare.py` | The europe street per package, stacked into one sheet per camera. |

```powershell
python photo/fetch_cc0.py
python photo/build_photo_pages.py --out <pkg> --base <pkg with the other pages>
python photo/render_photo_compare.py --out <dir> --pkg "Cycles bake=<pkg>" "AI image-gen=<pkg>" "CC0 photo=<pkg>"
```

## How a photo material bakes

`bake_materials.build_material` sees `spec["photo"]` and replaces the
palette, noise and pattern branch with these steps:

* colour = photo × `gain` × AO^`ao_k`. `gain` pulls the photo's mean
  colour toward the scheme's palette by `tint`, so colourways A/B survive
  while the detail stays photographic;
* bump from the displacement map (`bump` = distance in metres);
* repeat sizes snapped to whole fits: u to the **bay period** for facades,
  so floors that slide by whole motifs stay continuous, and v to the
  strip height, so stacked floors and split roof rows join.

A photo roof is one flat sheet (`strips.roof_strip`), because the photo
already has its tiles, overlaps and shadows. Rooftop equipment is still
modelled.

`flatten` (metres) divides a render photo by a wrap-around blur of itself.
Its large-scale shading otherwise repeats once per bay and reads as
banding.

## Findings (2026-10-01, europe_res_a only)

* `weather._walls` applied repaint blocks even at wear 0, which left a
  pale square patch on clean render walls. It is now skipped at k = 0.
  That also changes the clean strips `image-gen/prep_ai.py` cuts.
  Outputs at k > 0 are unchanged.
* Poly Haven's API size is the photo's **width**. `beige_wall_002` is a
  2048×512 map listed as 3×3 m, so height comes from the pixel aspect
  (`photo_kits._size_m`). It still repeated four times per floor and was
  replaced by `beige_wall_001`.
* `clay_roof_tiles_02` shows tiles about 6 cm wide at its listed 2.5 m, so
  `clay_roof_tiles_03` is used instead (about 20 cm interlocking tiles).
* Not photo yet: the trim bands (chimney, coping, ridge). Their 0.4 m
  bands would need a photo that repeats every 0.4 m. Window frames, doors,
  rollers and interiors also stay procedural.

## Real windows and doors (2026-10-01)

Poly Haven and ambientCG have no CC0 residential windows: ambientCG's
"Facade" sets are procedural, and Poly Haven only has roller-shutter
models. The windows and doors here are cut from CC0 photos on Wikimedia
Commons.

| Step | File | What |
| --- | --- | --- |
| Cut-outs | `window_cutouts.py` | `SOURCES`: per scheme, one near-frontal Commons photo for its windows and one for its door, with a rough box per opening. The license is checked through the API (CC0/PD only), and every side snaps to the strongest edge within 22 px. A keystone is kept only when both halves of a side agree. The quad is warped to a rectangle at 200 px/m. It writes `textures_cc0/windows/*.png` and `SOURCES.json` (page, author, license, corners, size); `--review` draws the quads. |
| Paste | `window_compose.py` | Re-runs `strips.facade_strip` with the bake's own scheme/level/W/H/seed to get the exact opening rectangles, and pastes the cut-outs there with fractional-pixel coverage. Exposure (p95) and colour cast (25% gray-world) are evened out between photos. In the normal page the opening is made flat, with gloss on dark, unsaturated glass. |
| Build | `build_photo_pages.py --windows` | Sets each scheme's window and door size to the photos' real proportions (texture-only, so OBJs and UVs do not change), bakes, then pastes. |

Only the opening is replaced. The wall, sill, lintel and the bake's
shadow around the opening stay. One source photo per scheme keeps a
facade's windows matching, as on a real house. Instances from the same
photo (s2 has three) are shuffled across the bays.

Known limits: s1's source window has its roller shutter almost closed on
every instance. The s3 photo is dark and cool-graded; normalising helps,
but the frames stay grey-brown. The pasted opening has no bake reveal
inside it, so recess depth comes only from the photo.

## Asia (2026-10-01, the in-sim test region)

`build_photo_pages.py --page asia_res_a` (now the default) and
`window_cutouts.py --page asia_res_a`. Mapping `ASIA_RES_A` in
`photo_kits.py`: s1 siding = `exterior_wall_cladding_03`, s2 tiles =
`rectangular_facade_tiles`, s3 render = `white_stucco`, kawara =
`grey_roof_tiles`, blue/green painted metal = `corrugated_iron_03` /
`box_profile_metal_sheet` recoloured (`desat` greys the photo before the
palette gain), flat slab = `concrete_floor_worn_001`.

Windows: the white uPVC sliders of an apartment block in Parand, Iran
(Commons, CC0, Mostafameraji) for s1 and s3; a Taipei double door
(Commons, CC0, 丘崈) for the s1 door. s2's barred cage windows have no
CC0 match among about 400 Asia-themed Commons hits (mostly temples, food,
stained glass), so they stay baked, as does the s3 door. A per-source
`inset` trims wall the snapped quad kept. Staged at 512 px:
`test_output/procgen_v4_photo/stage_asia_512`.

## All regions and classes (2026-10-01)

```powershell
python photo/fetch_cc0.py
python photo/window_cutouts.py --review <dir>
python photo/build_photo_pages.py --out <pkg> --base <pkg with the AI pages> --windows --size 512
```

With no `--pages`, every page is built: 8 regions × (res A, res B, apt)
plus `ind_clean/mild/worn`.

* **Materials** (`photo_kits.auto_mapping`): pages without a hand table
  are mapped by family. Brick → `red_brick` / `yellow_brick` (by hue).
  Siding → `white_planks_clean` turned horizontal. Tiles →
  `rectangular_facade_tiles`. Panels → `concrete_slab_wall_02`. Boards
  (industrial cladding) → `box_profile_metal_sheet`. Render → one of
  `white_stucco` / `beige_wall_001` / `plaster_grey_04`, hashed per
  scheme. Fair-faced concrete → `plastered_wall_04`. Roofs by kind
  (`ROOF_PHOTOS`). Each photo is tinted to the scheme's own palette, and
  `desat` greys it first where the photo has its own colour.
  **Shingle roofs stay procedural** (North America): neither CC0 library
  has a photographed asphalt shingle.
* **Windows and doors** (`window_cutouts.LIBRARY` + `window_compose.assign`):
  one shared library of cut-outs tagged by type (`casement`,
  `double_hung`, `slider`, `door_eu`, `door_na`, `door_asia`). Each scheme
  gets one window source and one door source, chosen deterministically,
  with different sources across a page where the pool allows. Doors follow
  the region (`DOOR_TYPES`). These stay baked: barred windows (Africa,
  South America, Asian grilles; the bars are relief in front of the
  glass), office glazing bands (w ≥ 2 m), shops, loggias, garages, docks
  and the whole industrial page. Window widths follow the photo's
  proportions at the kit's own heights, and doors are capped at
  0.55 × bay.
* `window_assignment.json` in the package records which source covers
  which scheme. `SOURCES_cc0_windows.json` and `LICENSES_cc0_photo.json`
  record the provenance of every pixel source.

### Source review and roof objects (2026-10-01)

* Dropped as oblique shots, where a jamb side face, reveal or slanted
  floor shows: `loppersum`, `torgau_brown`, `pitt_tall`, `pitt_maroon`.
  The other sources trim side faces with per-box `inset`s. Check every
  new cut-out for a strip of side face before adding it to `LIBRARY`.
* Photovoltaic panels are removed from `regions.ROOF_OBJECTS` in every
  region. Densities were scaled by the removed weight, so vents,
  skylights, tanks and solar water heaters keep their counts. This
  changes the roof strips of the plain v4 bake too.
