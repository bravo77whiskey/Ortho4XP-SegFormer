# O4XP greenhouse library

Objects the building overlay places on greenhouse detections
(`src/O4_Agri_Structures.py`, `O4_BLD_AGRI`):

* `o4xp/agri/tunnels_<W>x<L>.obj` -- south-China arrays of arched film
  tunnels, 8 m pitch, 1-12 bays, 10-150 m long;
* `o4xp/agri/solar_<L>x<W>[_b].obj` -- north-China solar greenhouses, 8-20 m
  deep, brick (a) or cement-render (b) back wall and gables, entry room at
  either end.

| Step | File |
| --- | --- |
| Shared UV atlas (two 1024 px pages) | `layout.py` |
| Meshes, 3 LOD bands, per-bay variation | `geometry.py` |
| Texture pages from real photographs | `textures_photo.py` (+ `photo/`) |
| Texture pages from the AI source images | `textures_ai.py` |
| OBJ8 + library.txt (+ DDS via nvcompress) | `build_greenhouses.py` |
| UV debug pages | `uvcheck.py` |

```powershell
python photo/fetch_sources.py
python textures_photo.py --out <pages>
python build_greenhouses.py --out <pkg> --pages <pages> --dds
```

## Photo texture set

Every pixel comes from a public-domain or CC0 photograph
(`photo/sources.py`; `fetch_sources.py` refuses any file whose license
field says otherwise and writes `LICENSES.json`, which the build ships in
the library):

* tunnel and solar film: a USDA photo of Californian raspberry tunnels
  (U.S. government work, public domain), unrolled by `photo/unroll.py`:
  homography between the tunnel's two feet, the hoops' bow traced and
  shifted out, hoops resampled to an even spacing, a window of whole hoops
  cut so it tiles;
* tunnel gables: open tunnel ends from the same USDA set, so the photo
  set's `doors.json` has no door frames (`null`);
* back walls, gables, anteroom, back roof, rolled quilts / straw mats,
  soil, anteroom roof: Poly Haven CC0 scans tiled at their measured size;
  CC0 Commons door photos pasted where the layout has a door.

Commons and Openverse have no license-clean photos of Chinese solar
greenhouses, their quilts, or a polytunnel end wall with a door, so those
parts use photos of the same materials elsewhere.  `photo/sources/` (33 MB)
is not tracked; rerun `fetch_sources.py`.

## AI texture set

```powershell
python textures_ai.py --out <pages>
```

The texture pixels are image-generation output (codex CLI), produced from
`ai/PROMPT_IMAGES.md` and `ai/PROMPT_WALLS.md` into `ai/sources/` (48 MB,
not tracked -- regenerate with `codex exec < ai/PROMPT_*.md`, run from
`ai/`).  Code only crops, scales, tiles and blends them; along-greenhouse
tiling is closed with a minimum-error cut, never mirrored or cross-faded.
