# Making X-Plane 12 scenery look less flat: where the remaining realism is

2026-11-08. Research into what still makes XP12 + Ortho4XP look flat next to MSFS 2024, limited to
changes that are physically motivated rather than "turn it up". It builds on work already done:

- sim lighting (tone curve, haze, multi-scatter, AO, adaptation): `G:/Dev/xp12-lighting/docs/research.md`
- ortho materials (gloss/F0/relief `_mat.dds`), no-grid decals, colour correction, grass/soil clutter,
  canopy heights and forest stands: see the project memory and the commits that shipped them.

Everything below was read from the installed X-Plane 12 (shaders, `X-Plane.exe`, Laminar's own `.ter`/`.for`
files) or measured on the review hotspots. Nothing in the X-Plane install or any tile was changed.
Prototype code: `experiments/xp12_realism_20261108/`.

## Summary

**2026-11-09 update:** the user flies in daytime only, so the two night items (#1, #3) are **parked**.
The daytime order is #2 → #4 → #5; #6 stays research only.

| # | Lever | Kind | Evidence | Effort | Recommendation |
|---|---|---|---|---|---|
| 1 | Night lights baked into ortho terrain (`TEXTURE_LIT`) | scenery | Laminar's urban terrain has it, Ortho4XP `.ter` doesn't; prototype built | small–medium | **Parked** (daytime-only user) |
| 2 | Forest colour matched to the imagery (own forest library) | scenery | canopy hue 115–160°, chroma 11–22 across the hotspots; XP's one leaf colour (124.5°, 19.4) matches Kaohsiung/Shouguang, not Shenzhen (corrected 2026-11-09) | medium | **Built 2026-11-09** (`veg_forest_tint`), needs in-sim A/B |
| 3 | Night windows on our ProcGen v4 buildings | asset | v4 README: "No night lighting" | medium | **Parked** (daytime-only user) |
| 4 | Material classes from SegFormer instead of colour rules; weather hooks | scenery | terrain shader reads a weather texture and snow coefficients we never set | medium | After the weather round of the lighting work |
| 5 | Relief normals from real heights (high-res DEM, canopy/building height rasters) | scenery | the 3" DEM in use is already mostly in the mesh | medium | Only where a ≤10 m DEM or our height rasters exist |
| 6 | Baked ortho shadows vs. sim sun | scenery/ML | no released aerial de-shadowing model | large, risky | Research only |
| — | Long shadows, terrain shadows, tone curve | engine | tested 2026-10-12/13 | — | Leave; wait for the tile engine |

## Why it looks flat: the causes that are still open

The sim-wide lighting causes (toe-less tone curve, 1 km shadow range, strong sky ambient) are covered in the
lighting research. What's left is mostly about the **content** XP is given to light:

1. **An ortho is a photo, not a material.** XP lights it as matte albedo. The shading baked into the photo
   (sun direction, crown shadows, building shadows at capture time) never changes. The `_mat.dds` gloss and
   relief maps are the first step away from that.
2. **3-d content doesn't match the photo it stands on.** X-Plane has one leaf colour for every
   forest, while the canopy in the imagery varies by region (measured in §2): where they differ, the
   eye sees two layers, not one forest.
   MSFS tints its autogen vegetation toward the ground colour; community reports agree on that, but Asobo
   has not documented the mechanism.
3. **At night, ortho tiles lose Laminar's distant city glow** (§1). This is a regression relative to stock
   scenery, not just a missing extra.
4. **Weather never reaches the ortho as a material.** The terrain shader has rain/snow inputs (§4), but our
   `.ter` files give it nothing to work with.

MSFS 2024's lead at low level also comes from 3-d ground cover, tessellated ground materials and
photogrammetry. The first two are already covered by the grass/soil and decal work. Photogrammetry was
studied separately (memory: xplane-photogrammetry-viability) and is capped by OBJ8/DSF until the tile engine.

## 1. Night lights for ortho terrain (`TEXTURE_LIT`) (parked: user flies daytime only)

**Finding.** Laminar's XP12 urban terrain (`1000 world terrain/terrain12/*/urban_*.ter`, 49 files) carries:

```
TEXTURE_LIT	../../textures12/urban_FAR/uw9_FAR_LIT.dds
LIT_FADE_PARAMS 8192 7000 15000
LUMINANCE 600
```

`uw9_FAR_LIT.dds` is a 1024 px night aerial of a generic street grid (`img/laminar_uw9_FAR_LIT.png`).
It fades in between ~7 and 15 km, where X-Plane stops drawing 3-d road and object lights. That is how stock
cities still glow from cruise altitude. Ortho4XP's `create_terrain_file` writes no LIT line, so an ortho city
goes dark beyond the 3-d light range.

**Verified in the shader** (`terrain.xsa`, variant 1548 and siblings): the night texture `tex_nite` is
sampled with the **same UVs as the base texture**, scaled by two per-draw factors, and written to the
emissive G-buffer target. A per-ortho-texture LIT map therefore lines up with the photo exactly, with no
projection tricks needed.

**Prototype** (`night_lit.py`, Kaohsiung north ZL16, `img/khh_n_night_lit.jpg` vs `img/khh_n_day.jpg`):
- light pools from the tile's cached OSM roads (width and intensity by highway class), lamp rows
  along each road, plus spill and signage glow driven by **our own building detections** (density at a 120 m scale).
  Brightness follows activity: the dense core glows, hill roads keep sparse lamps, and water and forest stay dark.
- **Calibrated, not cranked**: the map is fitted so its mean and 99th percentile match Laminar's own
  `uw9_FAR_LIT.dds` (red mean 7.1 vs 7.2, p99 55 vs 63). It is exactly as bright as stock night cities,
  only correct for the place.
- Cost: 1024 px BC1 ≈ 0.7 MB per ZL16 texture, only for textures with roads/buildings. Build time ~30 s per
  texture in Python (one-off, CPU), and it could be vectorised.

**Open before shipping (one in-sim test, one tile, user's go-ahead):**
- `LIT_FADE_PARAMS` meaning: the reading is "<scale> <fade-in start m> <fade-in end m>"; the first value
  varies with `#scale` (3072/4096/8192). Test by copying Laminar's line.
- `LUMINANCE 600` (cd/m²) is Laminar's value for these maps; keep it.
- PNG vs DDS for LIT (Laminar uses DDS here).
- Colour: the prototype mixes sodium on minor roads with 4000 K LED on major ones. Taiwan and China are mostly LED
  now, so the palette should be a per-region setting.
- Better input: VIIRS night radiance (NASA Black Marble, 500 m) would calibrate brightness per district. It
  needs an Earthdata login, so it was not downloaded.

Integration: in `create_terrain_file`, add the three lines for land textures that have a `_LIT.dds`. Make the map
in Step 3 next to `_mat.dds` (same hooks as `O4_Terrain_Materials`).

## 2. Forest colour vs. the imagery under it

**Correction (2026-11-09).** The first measurement (`tree_colour.py`) read the bit-packed tree-support
file as if it were the tree mask, so its ortho numbers (chroma ~11, hue ~109°, "XP 1.6–1.8× more
saturated and 15° bluer") were wrong. Re-measured with the SegFormer tree class (`tree_colour2.py`,
`canopy_zones.py`), CIELAB medians of a* and b*:

| | L* | a* | b* | chroma | hue |
|---|---|---|---|---|---|
| XP `trees_3D1_ALB` (3-d trees) | 35.3 | −11 | 16 | **19.4** | **124.5°** |
| XP `trees_bb1_ALB` (billboards) | 28.2 | −10 | 14 | 17.2 | 125.5° |
| Shenzhen canopy (3 textures) | 16–26 | −11…−13 | 5–9 | 12.5–14.8 | 141–159° |
| Shouguang canopy (2) | 26–30 | −9…−10 | 14 | 16.6–17.2 | 123–126° |
| Kaohsiung canopy (60 m smoothed, p50) | | | | 17–20 | 128–133° |
| Wenzhou canopy (60 m smoothed, p50) | | | | 13–16 | 134–140° |
| all hotspots, 60 m smoothed, p5–p95 | | | | 10.9–22.4 | 116–156° |

So X-Plane's single leaf colour is close to the canopy around Kaohsiung and Shouguang and ~25° too yellow
and 1.3–1.5× too saturated around Shenzhen. The gap is regional, not a global offset: the fix has to pick
the colour locally. The photo canopy is also much darker (L* 16–30 vs 35), but that includes crown
shadows and exposure, which the sim adds itself, so lightness is left alone.

**Constraint:** OBJ8/.for has no per-instance tint (`u_imm_a_instance_tint` exists in the shader but isn't
exposed for forests), so the fix is texture variants.

**Built (2026-11-09, uncommitted): `src/O4_Forest_Tint.py`, setting `veg_forest_tint` (default on).**
- All 175 `1200 forests` files use the same two leaf atlases, so a colour costs 4 textures (3-d +
  billboard, summer + spring): ~14 MB of VRAM per colour in use, written once into `O4XP_Forest_Heights/textures_t<k>`.
- 16 buckets: hue steps −12/0/+12/+24° × chroma ×1.15/1/0.82/0.66 of X-Plane's leaf colour, ~3 ΔE apart
  (`img/forest_tint_buckets.jpg`). Lightness unchanged; bark and dry leaves (hue outside ~80–195°) untouched.
- Recoloured in the DXT5 data itself: only each block's two RGB565 endpoints change, so Laminar's mip chain
  and leaf alpha are kept bit for bit (no re-encode, 0.1 s per atlas). Measured on the same pixels, the
  result lands within 1 ΔE of each bucket's target.
- The veg overlay averages the canopy colour of crown interiors over 120 m, picks the nearest bucket per
  pixel, merges colour zones under 2 ha and splits forests by colour zone (3–9 colours per ZL16 hotspot
  texture; at 60 m / 0.5 ha it was 8–13, mostly shading and haze noise). Every tree path becomes a
  `o4xp/forests/<kind>[-ev]-t<k>/…` copy; it combines with the height, cap, shrub and evergreen copies.
- Colour must not change how trees are planted. On the full RCKH tile, the first version raised forest
  polygons from 302k to 471k: zone edges cut patches into slivers, which became extra single-tree groups
  (196k → 333k) and tree rows (2.8k → 17k). Now a patch under 2 ha takes its majority colour whole, and a
  colour piece under 2 ha inside a bigger patch joins the neighbouring colour it shares the most border
  with. On szx_e/khh_n that cuts the extra pieces from +29/+57 % to +3/+6 %, at ~1–2 s per ZL16 texture.
- Tinted summer files drop `SEASONAL` (the summer→autumn blend). Autumn and winter keep X-Plane's textures.
  Unverified: whether this jumps from green to full autumn at the start of X-Plane's autumn season.
- Unverified in the sim: the colour gain itself (the A/B is the test), and that XP picks up the `.dds` behind
  a `.png` reference in another pack (it does so for its own atlases).

## 3. Night windows on ProcGen v4 buildings (parked: user flies daytime only)

`scripts/asset_pipeline/procgen_v4/README.md`: "No night lighting (`TEXTURE_LIT`)". Wherever v4 buildings
replace simHeaven/SFD assets, the city goes darker at night. A LIT atlas from the existing window cut-outs
(random lit fraction per window, warm/cool mix) is a contained asset-pipeline task. It pairs with #1: near
the aircraft, windows; far away, the terrain LIT.

## 4. Material classes and weather on ortho terrain

What the terrain shader reads that our `.ter` files never set:
- `tex_weather_texture` / `u_weather_mode`: weather variants sample a per-surface weather texture with the
  base UVs (Laminar's forests use `WEATHER .._SM.png`). Rain wetness and puddles on ortho ground depend on this.
- `u_material_snow_luma_coef/mix`: Laminar terrain sets `SNOW_ALBEDO_LUMA 1.2 1.5 0 0`. Ortho terrain uses
  defaults, so snow cover on orthos follows a generic rule.
- `SUPER_ROUGHNESS 0.5` is on 6,960 of Laminar's 9,848 terrain files. We don't set it. Its effect on
  terrain (probably a roughness floor/scale) is unverified.

The `_mat.dds` gloss/F0 currently comes from colour rules (veg/soil/asphalt/concrete/bright). The landcover
SegFormer already knows water, roofs, roads and vegetation. Class-driven gloss (wet paddies, metal and
greenhouse roofs, asphalt vs. concrete) would be more correct than colour. The user found gloss alone
"almost negligible" in clear weather, so this mostly matters **in rain and low sun**. That is exactly where
the lighting work stopped ("next: weather"). Do it as part of that round, not before.

## 5. Relief normals from real heights

Today's relief channel is a luminance high-pass, a brightness-as-height proxy that the user liked at 0.15.
Two refinements, with limits:

- **Tangent frame is fixed, not UV-derived.** In `terrain.xsa` the normal map is applied as
  `T*nx + B*ny + N_vertex*nz`, with T and B coming from two per-draw uniform vectors, not from UV
  derivatives. So a height-derived normal map is lit from the correct side only if we know the signs of T
  and B. A one-tile in-sim check settles it: a texture with a strong east-facing ramp at morning sun. The
  luminance relief didn't need this because it has no true direction.
- **The DEM in use is too coarse to add anything.** `Elevation_data` holds 3" (~90 m) HGT tiles, and Ortho4XP's
  mesh already carries most of that detail. DEM-residual normals only pay off with ≤10 m DEMs (Taiwan 20 m and
  better national DTMs, EU LiDAR, USGS 3DEP).
- **Our own height rasters are the interesting source:** canopy height (Meta 1 m, already used for tree
  sizes) and HeightNet building heights. Normals from them give crowns and roof edges shading that follows
  the sim sun. Expect it to matter at low altitude only; mips average it out with distance.

## 6. Baked shadows in the ortho

Building and tree shadows in the photo point to the capture-time sun and never move. Literature check
(2024–2026): aerial shadow **detection** is maturing (ISPRS 2026), but removal models are thesis-level,
need DSM/NIR inputs, and no weights are released. Partial removal also risks a flat, milky photo. One
idea worth recording: a custom mip chain that softens shadows only in the top mips (near the camera, where
our 3-d buildings and trees cast real shadows) and keeps them in low mips (beyond the 1 km shadow range,
where baked shadows are the only depth cue). It is speculative and not prototyped. Revisit only if shadow
mismatch becomes a named complaint.

## What not to spend effort on

- **Global tone curve, contrast, saturation:** all rejected in-sim (2026-10-12).
- **Longer shadow distance:** the switch-on edge stays visible on trees (2026-10-13). Long terrain shadows
  (mountains at low sun) are an engine feature that MSFS has and XP12 lacks. No scenery workaround exists,
  because a sun-dependent shadow needs per-frame shader data.
- **DSF/OBJ8 workarounds for streaming or photogrammetry:** wait for the tile engine (memory:
  xp-tile-engine-research).

## Suggested order

Daytime only (night items parked 2026-11-09):

1. Forest leaf colours (built 2026-11-09, `veg_forest_tint`): A/B on one hotspot with `sim_ab.py`.
2. Weather round of the lighting work, with class-driven materials and the weather/snow `.ter` inputs.
3. Height-derived relief from the canopy/HeightNet rasters, after the one-tile tangent-sign check.

Parked: night LIT terrain (§1) and ProcGen v4 LIT windows (§3). The prototype stays in
`experiments/xp12_realism_20261108/night_lit.py` if night flying ever matters.

## Sources

- Laminar `.ter` spec (older; no XP12 commands): https://developer.x-plane.com/article/terrain-type-ter-file-format-specification/
- X-Plane normal maps: https://developer.x-plane.com/article/normal-maps
- Additive (LIT) lighting: https://developer.x-plane.com/article/additive-lighting/
- AVSIM, XP extended night lighting (textures far, 3-d lights near; Laminar won't adapt it for orthos): https://www.avsim.com/forums/topic/533768-new-extended-night-lighting-in-beta/
- MSFS 2024 terrain changes (3-d ground materials, imagery-driven clutter): https://fsnews.eu/microsoft-flightsim-development-update-at-fsexpo-2024/ , https://flyawaysimulation.com/ask/answers/msfs-2020-vs-2024-terrain/
- MSFS autogen colour vs imagery (community observation, not documented by Asobo): https://forums.flightsimulator.com/t/color-of-autogen-trees/446668
- Aerial shadow detection/removal: https://isprs-archives.copernicus.org/articles/XLIX-B3-2026/141/2026/ , https://www.cg.tuwien.ac.at/research/publications/2024/staats-2024-atr
- Local reads: `X-Plane.exe` keyword strings; `Resources/shaders/bin/spv/terrain.xsa` (4,614 variants, names
  tallied, variant 1548 disassembled with `G:/Dev/xp12-lighting/spvdis.py`); `1000 world terrain` and
  `1200 forests` files.
