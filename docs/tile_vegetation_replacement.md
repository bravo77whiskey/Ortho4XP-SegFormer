# Tile-wide vegetation replacement

Replaces the vegetation of **selected generated tiles** with the SegFormer
vegetation overlay, and leaves every other tile alone. Off by default.

Two settings drive it, both per tile with a saved global default:

| Setting | Short name | Effect |
|---|---|---|
| `sfr_veg_exclude_lower_forests` | `sfr_veg_excl_for` | Replace lower-priority forests throughout the tile |
| `sfr_veg_remove_embedded_trees` | `sfr_veg_rm_trees` | Also remove supported embedded trees from lower-priority scenery |

They appear in the **SegFormer Veg** column of both the tile config and the
saved global config, and propagate through the GUI single-tile build, the batch
build and the CLI. Embedded-tree removal needs forest replacement; on its own it
is ignored, with a line in the log saying so.

## What the exclusion does

The generated overlay for the tile gets one extra header property:

```text
PROPERTY sim/exclude_for 8/47/9/48
```

for `+47+008`. That is `west/south/east/north` over the **whole one-degree
tile** — every forest a lower-priority package places anywhere in the tile
disappears, including where the model detected no trees. There is no separate
exclusion package; a tile without a generated overlay gets no exclusion.

X-Plane applies an exclusion to the DSFs *below* the one carrying it and never
to that DSF itself, so the generated trees survive their own exclusion. Keep
`yOrtho4XP_Veg_Overlays` above the regional scenery being replaced (simHeaven,
SwissUHD, …) and below any airport whose authored scenery should win.
`validate_scenery_order` checks the installed order and reports conflicts; it
never rearranges `scenery_packs.ini`.

### Forest-overlap avoidance is forced off

`sfr_veg_avoid_simheaven_forests` keeps generated trees out of existing
simHeaven forests. With replacement on, those forests are then excluded, which
would leave a bare hole exactly where each one was. For a replacement run the
effective value is therefore `False`; the saved setting is not rewritten and
resumes as soon as replacement is off. Tree-family selection from simHeaven
assets, road and railway avoidance, building avoidance, water masking and
authored airport forest exclusions all keep working unchanged.

## What the exclusion cannot do

`sim/exclude_for` only suppresses *forest polygons*. Trees a package draws some
other way survive it: standalone tree objects, trees attached to road networks,
trees baked into autogen blocks and strings. `sfr_veg_remove_embedded_trees`
deals with those.

### How embedded trees are removed

One kind of edit, at two levels:

> a reference to a **proven** tree object is repointed at an empty object

* in the tile's source DSF, by substituting one entry of the `DEFN/OBJT` table;
* inside a referencing text asset (`.net`, `.ags`, `.agb`, `.agp`, `.str`,
  `.lin`), by substituting the path on the line that declares it, in a private
  copy that only the processed tile's DSF points at.

Nothing is deleted, reordered or reindexed, so buildings, facades, ground
surfaces, roads, lights and unrelated objects survive by construction, and no
coordinate is requantized. Only the enclosing atom lengths and the trailing MD5
footer are recomputed.

### What counts as "proven"

`classify_object` requires an object to do nothing a tree would not do, *and*
for all of its artwork to be vegetation artwork, *and* for it to be either filed
under a vegetation folder or built like a two-sided billboard. A filename
containing "tree" is never evidence on its own. An object that drapes, is hard,
animates, emits light or exposes a cockpit or manipulator is rejected whatever
it is called.

Anything that does not clear that bar is left exactly as it is and named in the
report. To override a decision after looking at the model yourself, add it to
`<managed root>/asset_rules.json`:

```json
{"tree": ["props/tree_stand.obj"], "not_tree": ["trees/oak.obj"]}
```

`not_tree` always wins.

### Treatment table

| Source | Treatment |
|---|---|
| Ordinary forest placements | Suppressed by the tile exclusion |
| Autogen tree annotations | Repointed in a private copy of the asset |
| Separate tree-object placements | Repointed to an empty object in this tile's DSF |
| Roadside vegetation | Repointed inside a private `.net` copy; the road definition is kept |
| Trees within mixed object geometry | Not touched — reported |
| Ambiguous or unsupported content | Not touched — reported |

Higher-priority airport packs are out of scope and are never inventoried as
sources.

## Private assets

A private copy is written **beside its original**, named
`<stem>_vegfree_<hash><suffix>`, so every relative reference inside it —
textures, sibling assets — resolves exactly as the original's did. The original
is never modified. The name is content-addressed on the source bytes plus the
transformation version, so two tiles needing the same modified asset share one
copy, and a package update produces a new one instead of silently reusing the
old.

An asset owned by a *different* package than the DSF pointing at it cannot be
redirected for one tile without affecting every other tile that uses it, so
those are reported as unsupported rather than edited.

Shared empty objects live in `<package>/o4xp_veg_replacement/`. That folder only
ever gains files; deleting it returns the package to its shipped content.

## Storage

Exactly two persistent copies of any modified DSF exist:

| File | Retention |
|---|---|
| Verified original backup | Persistent, content-addressed under `backups/` |
| Modified DSF at its installed path | Persistent, used by X-Plane |
| Temporary modified DSF | Only while that file is being processed |

There is **no third directory holding another full set of stripped DSFs**.
Shared assets are not copied into a backup directory at all, because their
originals are never overwritten.

Before anything is modified, `estimate_storage` adds up backup space, private
asset space and peak temporary space from the **actual file sizes**, checks the
free space on every volume involved, and stops if it is short. Pass
`--force-space` to override.

## Safety

* Every destructive step verifies the installed file's SHA-256 first, stores and
  re-hashes a backup, rewrites to a temporary file, re-parses and compares the
  result atom by atom against the original, then replaces atomically and
  verifies what landed.
* A backup is never overwritten. A changed source is a new version with its own
  backup; older recovery data is never deleted to make space.
* `journal.jsonl` records every completed change; `pending.json` marks an
  operation as in flight and is removed only when it finishes or is restored, so
  an interrupted run is detected on the next one. A tile is never reported
  complete while only part of its workflow succeeded.
* Installing refuses to start while X-Plane or a generation run is active.

## Lifecycle

| Generation outcome | Result |
|---|---|
| Compiled, validated, replacement on | Exclusion installed; embedded-tree cleanup may run |
| Compiled, validated, replacement off | Any previous managed replacement for the tile is reconciled away |
| No vegetation polygons | Stale generated overlay removed; previous replacement reconciled away |
| DSFTool failure | Previous overlay left untouched; no cleanup starts |
| Imagery missing or incomplete | Generation never reaches the write; nothing is installed |

The overlay subprocess writes `veg_overlay_status.json` into the tile's SFR
cache folder; that file, not the exit code, is what decides whether a tile-wide
exclusion is installed.

## Existing tiles, without re-running inference

```bash
python src/scripts/veg_tile_replacement.py migrate --tile +47+008 --custom-scenery-dir "D:/X-Plane 12" --preview
```

`migrate` adds or removes the exclusion on an already compiled overlay by
rewriting only its header property table, then runs the same inventory and
cleanup. `--preview` changes nothing. `--disable` removes the exclusion.
`--exclusion-only` skips the embedded-tree work. The overlay must be an overlay
DSF, cover the right tile and define at least one forest, otherwise replacing
the layers below it would leave the tile bare.

Verified originals left by the older `strip_simheaven_vegetation` run can be
reused:

```bash
python src/scripts/veg_tile_replacement.py adopt --output <that run's --output folder>
```

A stored file is adopted only when it hashes to a recorded *original*, never to
a recorded stripped version. (Cleaning up that old archive is a separate task.)

## Restoring

Changing a configuration checkbox affects the next build and cannot undo edits
already on disk. Restoring is a separate action:

```bash
python src/scripts/veg_tile_replacement.py restore --tile +47+008
```

or, in the GUI, **Restore Veg Scenery** in the tile-management window (under
"Installed veg scenery", separate from the build steps). It restores every
modified source DSF from its verified backup, removes the exclusion from the
generated overlay, and deletes private assets no remaining processed tile still
references. Installed hashes are checked first: if another tool or a scenery
update changed a file, it stops and names the conflict instead of overwriting
it. Recovery records are kept until restoration has been verified.

**Veg Replace Status** lists the tiles with an installed replacement and any
interrupted operation.

### Restoring when a tile is erased

The Tiles window's Batch Delete restores a tile's replacement before erasing it,
so a package never keeps its trees removed with nothing of ours left to replace
them. This happens automatically whenever **Tile (whole)** is selected, and can
be done on its own with the **Veg replacement** box in the erase list. Tiles
with nothing recorded against them are passed over in silence; a tile whose
installed DSF is neither the original nor what this tool wrote is reported in a
dialog and left alone, so the record survives for a manual recovery.

## Command line

```bash
python src/scripts/veg_tile_replacement.py inventory --tile +47+008 --custom-scenery-dir "D:/X-Plane 12"
python src/scripts/veg_tile_replacement.py preview   --tile +47+008 --custom-scenery-dir "D:/X-Plane 12"
python src/scripts/veg_tile_replacement.py apply     --tile +47+008 --custom-scenery-dir "D:/X-Plane 12" --exclusion
python src/scripts/veg_tile_replacement.py status
python src/scripts/veg_tile_replacement.py restore   --tile +47+008
```

The overlay generator itself takes `--exclude-lower-forests` and
`--status-path`.

## Reports

A report lists the tiles processed, the forest polygons the exclusion
suppresses, the embedded tree placements removed, the unsupported sources left
visible, the private assets, the backups and journal locations, the storage
consumed, and the restore command. It also names neighbouring tiles that still
draw an asset this tile repoints, since a tree at a border renders into the next
tile and would otherwise show up as a seam.

The report describes **file-level changes verified by hash and by re-parsing
each DSF**. It does not state what the tile looks like in X-Plane.

## Modules

| File | Role |
|---|---|
| `src/O4_Cfg_Vars.py` | Settings, defaults, labels, help text, persistence |
| `src/O4_GUI_Utils.py` | Passes tile settings into generation; restore and status actions |
| `src/O4_Tile_Utils.py` | Carries the settings through batch generation |
| `src/O4_SFR_Pipeline.py` | Coordinates generation, validation and cleanup |
| `src/O4_SFR_Vegetation_Overlay.py` | Exclusion property, effective overlap behaviour, output lifecycle |
| `src/O4_SFR_DSF_Edit.py` | Byte-level DSF header/definition edits and usage counting |
| `src/O4_SFR_Veg_Cleanup.py` | Inventory, classification, private assets, backups, journal, restore |
| `src/scripts/veg_tile_replacement.py` | CLI |

Tests: `tests/veg_tile_replacement_test.py`,
`tests/sfr_veg_tile_exclusion_test.py`,
`tests/sfr_settings_propagation_test.py`.
