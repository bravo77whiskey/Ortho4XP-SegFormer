# Vegetation VRAM in the sim (2026-11-15)

Two changes, measured on the hotspots review set (`--skip-bld`, vegetation only):

1. **Shrubs follow the tree rules.**
   - Shrub copies (`s` kind) had their own 3 m planting grid, where X-Plane's own shrub forests use 4–10 m. They were also planted at full forest density, skipping the woodland and settlement thinning that trees get.
   - Now a shrub forest that fits under the height cap is planted from X-Plane's own forest. Only cut shrubs get a copy, made like a `c` (cap) copy.
   - Shrubs take the same DSF density as the trees they replace.
   - `KIND_REVISIONS = {"s": 2}` remakes only the existing `s` copies in the pack. The other ~4,300 copies are kept.
2. **Rare forest copies merge into a close one** (`veg_merge_rare_copies`, default on).
   - X-Plane loads each distinct `.for` with its own inline 3-d tree meshes, about 2.5–4 MB each. Leaf tint (16 buckets) and edge thickets (5 heights) multiply the distinct files a tile loads.
   - A copy that plants under 5 % of its forest's trees on the tile takes the nearest leaf colour the tile keeps (at most 3.6 ΔE, one step), or a thicket height one step lower (never higher, so runway caps hold).

## Leaf colour matching retired (2026-11-16)

The user retired the leaf colours matched to the ortho: some imagery has unnatural colours, and the trees copied them. `O4_Forest_Tint`, its 16 buckets and the ortho colour zones are gone. Summer forests still drop X-Plane's autumn blend, which was part of the tint copies, as `su` copies in X-Plane's own leaf colour (`veg_summer_leaves`, default on; the old `veg_forest_tint` key carries over). Forests X-Plane doesn't change with the seasons (very-hot band, single tropical trees) stay as they are. The rare-copy merge now only merges thicket heights.

Measured with run `veg-no-tint` (the shrub rules and merge from below, tint retired):

| Location | Forest files | Mesh (est.) | Leaf colour textures |
|---|---|---|---|
| Kaohsiung | 203 → **102** | 294.6 → **78.3 MB** | ~200 MB → 0 |
| Shenzhen | 191 → **96** | 269.9 → **64.6 MB** | ~173 MB → 0 |
| Wenzhou | 158 → **83** | 178.0 → **26.1 MB** | ~173 MB → 0 |
| Shouguang | 120 → **28** | 210.1 → **47.7 MB** | ~133 MB → 0 |

"Before" is HEAD 9ac6ac15 (`veg-shrubs-before`). Plant counts by kind stay the same within 3 % (forest plants −0.2 to +1.4 %, single trees −0.8 to −2.7 %). Colour zones no longer cut forests into extra pieces, so all 23 textures together have 4.6 % fewer tree polygons (86,922 → 82,883). Of the files left at Kaohsiung and Shenzhen, 64 are grass with tiny meshes.

Everything below was measured before the retirement.

## Numbers

`vram_estimate.py` counts the distinct forest files per location (all of a location's textures together, a stand-in for one scene). It estimates their mesh memory at 44 B per vertex and 4 B per index. These are assumed sizes, so use the numbers to compare runs, not as real VRAM.

| Location | Run | Forest files | Mesh (est.) | Tint textures | Shrub plants |
|---|---|---|---|---|---|
| Kaohsiung | before (HEAD 9ac6ac15) | 203 | 294.6 MB | 15 (~200 MB) | 15,584 |
| | shrubs follow trees | 198 | 290.2 MB | 15 | 2,306 |
| | + merge | 162 | **214.7 MB** | 15 | 2,341 |
| Shenzhen | before | 191 | 269.9 MB | 13 (~173 MB) | 14,229 |
| | shrubs follow trees | 186 | 264.5 MB | 13 | 1,459 |
| | + merge | 151 | **190.3 MB** | 13 | 1,482 |
| Wenzhou | before | 158 | 178.0 MB | 13 (~173 MB) | 4,797 |
| | shrubs follow trees | 149 | 166.4 MB | 13 | 2,565 |
| | + merge | 141 | **155.2 MB** | 13 | 2,384 |
| Shouguang | before | 120 | 210.1 MB | 10 (~133 MB) | 4,520 |
| | shrubs follow trees | 110 | 207.6 MB | 10 | 2,405 |
| | + merge | 99 | **174.6 MB** | 9 (~120 MB) | 2,461 |

- Forest, thicket, single-tree and grass plant counts stay the same within 0.4 % across all three runs (run-to-run noise). No trees are removed: the only differences are the shrub grid and which copy a few polygons point at.
- Grass counts as 64 files, but each has a tiny mesh (663 vertices) and a 230 m LOD. It is not where the memory goes.
- The merge decides per build. A real tile build has all of a tile's zoom levels in one pass, so it merges at least as well as the review builds.

## Looked at, not changed

- **Merge threshold.** At 10 % or 20 %, Wenzhou and Shouguang save a little more, but Shenzhen gets slightly worse. More trees would change colour, so it stays at 5 %.
- **Tint textures.** Each bucket costs about 14 MB (trees_3D1_ALB plus trees_bb1_ALB, DXT5 with mips). That is the same format X-Plane's own textures use, and normal maps are already shared through `textures/`. The only saving is fewer colour buckets, which changes the look the user approved in-sim on 2026-11-09.
- **Thickets.** They use a 4 m grid (X-Plane's tree_shr_dec_1 also uses 4 m) and a 6 km LOD (X-Plane's shrubs use 8–12 km). This is in line with stock and was approved in-sim on 2026-11-10.
- **Sharing meshes between copies.** The .for format keeps meshes inline and has no cross-file reference, so each copy loads its own.

## Review runs

`G:/Ortho4XP_Review/runs/`:

- `veg-shrubs-before`: HEAD code, from the G:/Dev/o4xp-vram-base worktree.
- `veg-shrubs-tree-rules`: new shrub rules, merge off.
- `veg-vram-merge`: new shrub rules, merge on.

Estimator JSON files: `G:/Ortho4XP_Review/_tmp/vram_{before,tree_rules,merge}.json`.

Neither change has been checked in the sim.
