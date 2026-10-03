# O4SFR greenhouse library

Objects the building overlay places on greenhouse detections
(`src/O4_Agri_Structures.py`, `O4_BLD_AGRI`):

* `o4sfr/agri/tunnels_<W>x<L>.obj` -- south-China arrays of arched film
  tunnels, 8 m pitch, 1-12 bays, 10-150 m long;
* `o4sfr/agri/solar_<L>x<W>[_b].obj` -- north-China solar greenhouses, 8-20 m
  deep, brick (a) or cement-render (b) back wall and gables, entry room at
  either end.

| Step | File |
| --- | --- |
| Shared UV atlas (two 1024 px pages) | `layout.py` |
| Meshes, 3 LOD bands, per-bay variation | `geometry.py` |
| Texture pages from the AI source images | `textures_ai.py` |
| OBJ8 + library.txt (+ DDS via nvcompress) | `build_greenhouses.py` |
| UV debug pages | `uvcheck.py` |

```powershell
python textures_ai.py --out <pages>
python build_greenhouses.py --out <pkg> --pages <pages> --dds
```

The texture pixels are image-generation output (codex CLI), produced from
`ai/PROMPT_IMAGES.md` and `ai/PROMPT_WALLS.md` into `ai/sources/` (48 MB,
not tracked -- regenerate with `codex exec < ai/PROMPT_*.md`, run from
`ai/`).  Code only crops, scales, tiles and blends them; along-greenhouse
tiling is closed with a minimum-error cut, never mirrored or cross-faded.
