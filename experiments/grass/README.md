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
    --mask colour --k 64 --sigma-m 8 --win-m 9 --min-m2 50 \
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

* **Where:** `--mask colour` plants everywhere except pixels that are clearly
  not ground: grey low-chroma, very bright, or roof-coloured (red/orange,
  blue).  There is no model.  Water is left to `SKIP_SURFACE water`.
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
* **SegFormer rangeland as the grass mask:** it misses a lot, including green
  areas and the dried airport infield.  It also gives no grass under trees.

## Known gaps

* Warm-tinted apron concrete and dark building shadows pass the colour rules.
  The apt.dat pavement, simHeaven roads and our building footprints would be
  exact, model-free exclusions.
* About 79k polygons per 2 nm circle.  This is fine for a test but needs
  measuring at tile scale.
* Zooming in extends X-Plane's draw distance, so the grass cost rises when zoomed.
* Placeholder blade colours are plain ramps.  The blades still need AI-imagery
  texture detail.

`grass_palette.py` is the first-round analysis: polygon count and colour error
against palette size and smoothing on single textures.  `regen_textures.py`
rewrites a pack's colour textures from its saved palette.
