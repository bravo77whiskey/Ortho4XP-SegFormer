# 3-d grass coloured from the ortho

An experiment, not wired into the overlays.  It plants X-Plane 12 forest-file
grass around a point, coloured from the tile textures under it, and writes a
self-contained scenery pack.  Tested in the sim around RCKH (Kaohsiung,
`zOrtho4XP_+22+120`, Bing ZL17–19).

## Checkpoint 2026-10-12: the best result so far

Built with:

```
python make_region_pack.py --center 22.5772 120.3500 --radius-nm 2 --name rckh \
    --tile-dir "D:/SteamLibrary/steamapps/common/X-Plane 12/Custom Scenery/zOrtho4XP_+22+120" \
    --mask fused --k 64 --sigma-m 8 --win-m 9 --min-m2 50 \
    --tiers "110:24:3:1.0,220:10:2:2.2,230:0:1:1.0" --px-mrad 0 \
    --tier-contrast "1,0.6,0" --tier-wind "1,1,0" --tier-tip "0.25,0.25,0.25" \
    --tier-height "1,1,1" --out <dir>
```

The blades are full height: the tallest in each tuft size are 0.27, 0.40
and 0.55 m.  A version with blades 25 % shorter (`--tier-height
0.75,0.75,1`) was tried and then put back.

Run it in the CUDA venv (`G:\Dev\Ortho4XP\.venv`); the full build reads 35
textures and takes a while.  `--only-for` rewrites only the `.for` files and
textures of an existing pack in `--out`, which takes seconds; use it for any
geometry change.  A built copy of this checkpoint is at
`E:\Ortho4XP_Review\_grass_spike\checkpoints\2026-10-12_best`.

The parts:

* **Where (`--mask fused`, the current rule):** each source decides what
  it is good at, and every decision at the scale of a single field comes
  from the ortho on screen:
  * SegFormer vegetation (rangeland, tree, agriculture) gets grass unless
    the colour rule says the pixel looks built.  SegFormer is what tells
    dry straw grass from sand or soil.
  * Pixels that SegFormer calls bareland, developed or unclassified get
    grass only where the ortho is clearly green (Lab hue 95–170°, chroma at
    least 10).  This catches lawns and yards.
  * SegFormer building, road and water never get grass.  The water class
    catches some of the canals.
  * The climate regime (Köppen, `O4_Climate_Regions`; static, so it cannot
    go out of date) sets how much the ortho must show.  In desert (BW*) only
    green pixels get grass.  In steppe (BS*) non-green vegetation becomes
    sparse grass at `--sparse-density`, written as a lower DSF polygon
    density.
  * The colour still comes from the ortho palette, so SegFormer's large,
    uniform segments never flatten it.
* `--mask colour` is the model-free first version.  It plants everywhere
  except grey, very bright or roof-coloured pixels, and that puts grass in
  deserts and in city shadows.
* **Exclusions from map data:** both are cut out before the colour is smoothed.
  * `--airports auto`: runways with their blast pads, helipads, and taxiway,
    apron and shoulder pavement from apt.dat (`apt_pavement.py`, curved
    edges included), plus a 2 m margin.  For each airport it uses the
    highest-priority pack, in `scenery_packs.ini` order, and falls back to
    Global Airports.
  * `--roads auto`: the simHeaven road and rail network X-Plane draws for the
    tile, at the vegetation overlay's widths for each road type.  Power lines
    and cableways are not cut.
* **Colour:** Lab, smoothed over grass pixels only (8 m), one k-means palette
  (64 colours) shared by every texture in the circle, a 9 m mode filter, then
  one polygon per colour region.  Each colour is its own `.for` with its own
  16x64 opaque texture.
* **Geometry:** 24-blade tufts at 1 m spacing out to 110 m, 10 wider blades
  (60 % contrast) to 220 m, then an **empty mesh** to 230 m.  An empty mesh is
  one zero-area triangle; a tier with 0 blades in `--tiers` writes one.  The draw-distance
  edge is therefore a mesh-to-mesh cross-fade, which X-Plane blends almost
  invisibly.  Blades have up-facing normals so they light like the ground, are
  matte (`SUPER_ROUGHNESS 1.0`), cast no shadows, and the mandatory billboard
  uses a fully transparent texture.

## What did not work, and why

* **One atlas shared by all colours:** at coarse mips, bilinear sampling near
  cell edges pulled in neighbouring colours and the empty transparent-black
  area.  This showed as dark speckle beyond ~150 m.  Use one texture per colour.
* **Pointed far blades:** most of each blade's area is the dark root end of
  the ramp, so the far levels went dark.
* **A pixel-width rule** (fewer, much wider blades and flat colour far out):
  the shimmer went, but the far grass looked blocky.  Rejected.
* **Ending on the billboard:** the last mesh cross-fades into a camera-facing
  card that is lit brighter than up-normal blades, which showed as a whitish
  band at the edge.  Fixed by the transparent billboard and ending on an
  empty mesh.
* **Denser near grass as extra forest layers** (`--near-layer`; kept only as
  a record): X-Plane plants every layer out to the file's full draw distance,
  whatever that layer's own meshes cover.  Two near-only layers ending at
  30 m left about 300k invisible tufts out to 230 m, each still drawing its
  empty mesh and transparent billboard.  The frame-rate cost was too high.
  Near-only density belongs in a separate `.for` with its own short `LOD`.
* **Shorter blades / a 6-blade mid level:** both tried and put back.
* **ESA WorldCover as a gate** (`worldcover.py`, `compare_masks.py
  --worldcover`; kept as a record): it fixed Garowe, but it is a 2021 map.
  Any dated outside layer that decides individual fields can contradict the
  ortho under it, so it was rejected.  The 10 m pixels also left blocky
  edges.  The climate regime replaced it.
* **SegFormer rangeland as the grass mask:** it misses a lot, including green
  areas and the dried airport infield.  It also gives no grass under trees.

## Known gaps

* Dark building shadows pass the colour rules.  Building footprints would be
  an exact, model-free exclusion.
* Narrow canals and drains get grass.  OSM maps them as `waterway=*` lines,
  which never become mesh water.  Ortho4XP's cached water file holds only
  water areas, so the lines need fetching (Overpass or planet mode).
* SegFormer calls some dried grass "developed".  At RCKH that leaves a patch
  at the east end of the infield bare.
* The climate grid is 0.5°, so it lumps wet and dry halves of islands such
  as Gran Canaria together.  The 1 km Beck map would fix that.  The steppe
  case is untested.
* `compare_masks.py` maps the masks against each other around a point
  (`--fused`, `--climate`, `--worldcover`, `--wc-only`).  `fetch_circle.py`
  downloads review imagery for places without a tile.
* About 79k polygons per 2 nm circle.  This is fine for a test but needs
  measuring at tile scale.
* Zooming in extends X-Plane's draw distance, so the grass cost rises when zoomed.
* Placeholder blade colours are plain ramps.  The blades still need AI-imagery
  texture detail.

`grass_palette.py` is the first-round analysis: polygon count and colour error
against palette size and smoothing on single textures.  `regen_textures.py`
rewrites a pack's colour textures from its saved palette.
