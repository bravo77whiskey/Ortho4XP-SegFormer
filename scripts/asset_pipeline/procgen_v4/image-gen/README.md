# ProcGen v4: AI re-rendered texture pages (experiment)

The same idea as the greenhouse library: texture pixels come from image
generation (codex CLI's built-in image tool). Here the model does not
invent layouts. It re-renders the v4 Cycles bakes as photographs, and code
puts every window back where the OBJ UVs expect it. Geometry, UVs, page
layout and the perf contract do not change. Only the albedo pages change,
plus the normals of roof strips.

| Step | File | What |
| --- | --- | --- |
| Inputs | `prep_ai.py` | Clean (wear 0) strips from the bake cache. Facades: upper stacked on ground (6.4 m), cut into 4 chunks of 16 m plus 1.6 m overlap each side, wrapping. Roofs: same chunks. Plain walls, porch and 256 px flat roofs: 4 chunks stacked into one image with black bars. Also writes class maps (`*_cls.png`) and one prompt per group. |
| Generate | `run_codex.py` | One `codex exec` per group, N in parallel. It skips groups that are already done, so a rerun fills the gaps. |
| Compose | `compose_ai.py` | Resize, ECC affine fit (blurred luminance; skipped on periodic pitched roofs), per-window snapping (template match on the bake's glass / rooftop-object classes plus a smooth displacement field), split stacked bands at the detected black bars, even out colour across the group, level tone steps at the joins, min-error-cut stitching, paste into a copy of the page. Roof normals are derived from the AI albedo. `--size 2048` writes half-size pages. |
| Review | `render_compare.py` | Renders the europe street plus two close-ups with identical OBJs for two packages and pairs them. |

```powershell
python image-gen/prep_ai.py --pkg <v4 pkg> --pages europe_res_a europe_res_b europe_apt
python image-gen/run_codex.py --pages europe_res_a europe_res_b europe_apt --jobs 4
python image-gen/compose_ai.py --pkg <v4 pkg> --out <new pkg> --pages ... [--size 2048]
python image-gen/render_compare.py --before <pkg> --after <new pkg> --out <dir>
```

`image-gen/work/` holds the inputs and outputs. It is untracked, and regenerating
it costs codex image calls. `out_round1/` keeps the first plain-wall
stacks, where the model exaggerated the bake's faint tone blocks into
patches. The round-2 prompt forbids that.

## Findings (2026-09-28, europe pages)

* Layout: the first chunk of a group comes back within 1–2 px. Later
  chunks, generated with chunk 0 as a style reference, drift 10–20 px and
  about 5% in vertical scale. After ECC and snapping, the median window
  error is ≤1 px (1.25 cm), with a worst case around 5 px.
* Codex sometimes refuses to save a result it judges misaligned. The
  prompt now says that compose corrects small shifts.
* The model sometimes drops a separator bar in stacked images. The band
  detector then fails and the whole-image fallback is used; regenerate
  those images.
* The AI weathering is milder than the v4 wear pass. Raise it in the
  prompt (`COMMON` in prep_ai.py) if you want more.

## Level 2: free designs on the bay grid

Level 1 keeps the bake's design. Level 2 lets the model design the facade
and keeps only what the 3-D models depend on:

* 3.2 m floors;
* the scheme's exact bay period, with one centred opening per bay;
* the bake's bay tables (door, window, small window, blank, loggia, shop),
  so `Facade.bay_start_with` still puts a door mid-front;
* a repeating upper-floor pattern.

| Step | File | What |
| --- | --- | --- |
| Briefs | `design_briefs.py` | One concrete building type per scheme (e.g. "1930s Dutch terraced house, dark red wire-cut brick..."). |
| Sketches + prompts | `design_prep.py` | Per scheme, two 3-storey × 6-bay zone sketches whose bay kinds are windows of the bake's own tables (coloured zone = maximum opening extent). The prompt asks for elevation A, elevation B (A as style reference) and a blank side wall. |
| Generate | `run_codex.py --work design_work` | One codex session per scheme (3 images). |
| Conform + rebuild | `design_compose.py --base <level-1 pkg>` | Opening mask (Lab distance + edges, Otsu), zone matching, per-floor piecewise-linear x warp onto bay centres, y scale from the upper-row spacing, B colour-matched to A. Strips are rebuilt by covering the bay table with runs of matching source bays (reuse penalised), with min-cut joins and tone leveling at the joins. Normals come from the albedo, and gloss from dark pixels inside openings. The plain wall is made seamless in u and tiled. `--diag` writes detection and conform overlays. |

Tested on synthetic elevations (level-1 strips, randomly shifted and
scaled per floor): all openings found; residual 0.1–0.4 px median,
≤2.2 px worst.

### Round 2 (2026-09-28): window size, side walls

* The model fills whatever box it is given ("maximum" is ignored), and v1
  drew windows 0.7–0.8 of a bay wide. Zones now come from each scheme's
  kit (`scheme_zones`: window w/h/sill, door, loggia pier), capped at half
  a bay wide and ≥0.7 m of wall above the lintel (`MIN_WALL_ABOVE`). The
  prompt states the target size in metres.
* The model still draws windows ~0.4 m taller than asked. Compose aligns
  the upper floors by the **window head** to the kit's head height
  (the roof fascia hides the top ~0.3 m of the top floor). The top
  floor's missing wall above comes from one floor down.
* Side walls: the prompt forbids downpipes and other vertical elements
  (they repeated every 12 m when tiled).
* Gloss: only the darker pixels inside the opening box, trimmed above
  the sill (a glossy brick sill sparkled white).
* v1 kept in `design_work/*/out_v1`, `test_output/procgen_v4_ai/design_v1`.

### All regions (2026-09-28, installed 2026-09-29)

Every region plus the industrial page now uses level 2. The installed
library holds these pages. The previous Cycles-bake pages are kept in
`O4XP_ProcGen_Library/textures_backup_cycles_bake`.

* Briefs for all 8 regions and industrial are in `design_briefs.py`. Wear
  wording per region is in `WEAR`, and roof wear is in `ROOF_WEAR`
  (`design_prep.py`).
* Garages and loading docks span two bays: the second bay's kind is
  `<kind>+`. Upper-floor-only schemes (industrial office wing) are
  handled. Industrial bays are 5.7 m, so its sketches have 4 bays
  (`K_BY_KIND`).
* Roofs (every region except Europe, whose roofs come from level 1): a
  2.5:1 crop of each clean roof bake is re-rendered as a photo that tiles
  left-to-right (`prep_roofs`). `design_compose.roof_strip` makes it
  seamless and repeats it across the strip.
* `--pages ind` builds one design and writes it to `ind_clean`,
  `ind_mild` and `ind_worn`. Each grade keeps its own trim strip.
* The sketch picker makes sure every bay kind of a table appears in at
  least one sketch, so the rebuild never lacks a source bay.
* Compose drops an elevation whose median residual exceeds
  `MAX_RESIDUAL` (5 px), provided the other elevation covers every bay
  kind. One Mediterranean apartment elevation came back with its loggias
  half a bay off the grid.

About 330 images in all. Every opening was found in 157 of the 160
elevations; each of the other three missed a single opening. Median
residual after conforming is under 2 px almost everywhere. In-sim: no
performance regression at 4096.

```powershell
python image-gen/design_prep.py --pkg <v4 pkg> --pages asia_res_a asia_res_b asia_apt ind
python image-gen/run_codex.py --work design_work --pages asia_res_a asia_res_b asia_apt ind --jobs 4
python image-gen/design_compose.py --base <pkg> --out <new pkg> --pages asia_res_a asia_res_b asia_apt ind [--size 2048] [--diag]
python image-gen/render_compare.py --region asia --before <pkg> --after <new pkg> --out <dir>
```

Codex's image tool has a usage limit (about 40 images per window). Rerun
`run_codex.py` after the reset: it only fills in missing outputs.
