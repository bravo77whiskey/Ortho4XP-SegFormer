# O4SFR_Library asset triage runbook

Canonical workflow for sorting downloaded 3D model archives into the shipped
`o4sfr` X-Plane library. Designed for resumability — every stage's state lives
on disk so you can stop mid-triage and resume tomorrow without losing
progress.

## Cast of files

| File | Owner | Purpose |
|---|---|---|
| `G:\Downloads\sketchfab_assets\*.zip` | User | Raw downloads |
| `scripts/asset_pipeline/sources/sketchfab/<slug>/` | `triage_extract.py` | Unzipped archives |
| `scripts/asset_pipeline/triage/inventory.csv` | `triage_extract.py` | Per-slug archive + license + model path |
| `scripts/asset_pipeline/triage/previews/<slug>.jpg` | `triage_extract.py` | 256×256 thumbnail for AI visual triage |
| `scripts/asset_pipeline/triage/probe.csv` | `triage_probe.py` | Bounding-box dimensions, tri count |
| `scripts/asset_pipeline/triage/decisions.yaml` | Assistant | Region + bucket per slug |
| `scripts/asset_pipeline/sources.yaml` | `manifest_from_triage.py` | Final manifest the converter consumes |
| `D:\SteamLibrary\steamapps\common\X-Plane 12\Custom Scenery\O4SFR_Library\` | Build output | Loadable X-Plane library |

The assistant skill `.agents/skills/xplane-asset-triage/SKILL.md` references
this runbook.

## Region taxonomy

Match the keys in `_O4SFR_PATH_REGION_TOKENS` (see
[src/O4_SFR_Building_Overlay.py](../../src/O4_SFR_Building_Overlay.py)):

`europe`, `scandinavia`, `mediterranean`,
`north_america`, `south_america`,
`asia`, `se_asia`, `africa`, `australia_oceania`,
`generic` (wildcard — overlay serves to any tile).

Rule: a model must **strongly fit a region** before being locked exclusively
to it. Anything ambiguous goes to `generic`. Filename is a hint only; visual
cues are authoritative.

## Workflow

### 0. Setup once

```
pip install pyyaml pillow
```

### 1. Extract zips + generate previews

```
python scripts/asset_pipeline/triage_extract.py \
    --downloads G:/Downloads/sketchfab_assets \
    --sources   scripts/asset_pipeline/sources/sketchfab \
    --triage    scripts/asset_pipeline/triage
```

Idempotent — already-extracted slugs are skipped. Re-run after downloading
more zips.

**License caveat**: Sketchfab downloads carry no per-asset license metadata
inside the zip. `inventory.csv` is populated with `license = CC-BY-4.0` and
`source_url = https://sketchfab.com/3d-models/<slug>` as defaults. Verify
those columns against the original Sketchfab page before final build.

### 2. Probe bounding boxes

```
python scripts/asset_pipeline/triage_probe.py \
    --triage  scripts/asset_pipeline/triage \
    --sources scripts/asset_pipeline/sources/sketchfab \
    --blender "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe"
```

`.obj` rows are probed in Python (fast). `.fbx`/`.blend`/`.glb`/`.gltf` rows
need Blender; if you omit `--blender` those rows get blank dimensions and the
manifest assembly falls back to per-bucket footprint defaults.

### 3. Visual triage (assistant-driven)

Open `triage/previews/*.jpg`. For each slug not yet present in
`decisions.yaml`, decide:

- `region`: a single value from the taxonomy. Strong fit only → otherwise `generic`.
- `bucket`: `residential | commercial | industrial | farm | accessory`.
- `confidence`: `high | medium | low`.
- `notes`: a short rationale.

Append to `triage/decisions.yaml`:

```yaml
version: 1
decisions:
  - slug: <slug>
    region: <region>
    bucket: <bucket>
    confidence: <high|medium|low>
    notes: "<short rationale>"
```

The assistant skill `xplane-asset-triage` automates this — it diffs
inventory.csv against decisions.yaml to find un-triaged slugs and processes
them in batches.

### "Where am I?" recipe

```
python -c "import csv; from pathlib import Path; \
inv = {r['slug'] for r in csv.DictReader(open('scripts/asset_pipeline/triage/inventory.csv', encoding='utf-8'))}; \
import yaml; \
dec = {d['slug'] for d in (yaml.safe_load(Path('scripts/asset_pipeline/triage/decisions.yaml').read_text(encoding='utf-8')) or {}).get('decisions', [])} if Path('scripts/asset_pipeline/triage/decisions.yaml').exists() else set(); \
print(f'inventory: {len(inv)}'); print(f'decisions: {len(dec)}'); print(f'remaining: {len(inv - dec)}'); \
[print(f'  - {s}') for s in sorted(inv - dec)[:20]]"
```

### 4. Assemble the manifest

```
python scripts/asset_pipeline/manifest_from_triage.py \
    --triage scripts/asset_pipeline/triage \
    --output scripts/asset_pipeline/sources.yaml
```

Drops entries with rejected licenses and unsupported formats; prints the
skipped list with reasons. Pass `--include-unknown` only to push past
license gaps for a test build.

### 5. Convert + build the library

```
python scripts/asset_pipeline/convert_to_xplane_obj.py \
    --manifest    scripts/asset_pipeline/sources.yaml \
    --source-root scripts/asset_pipeline/sources/sketchfab \
    --output      "D:/SteamLibrary/steamapps/common/X-Plane 12/Custom Scenery/O4SFR_Library" \
    --blender     "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe"

python scripts/asset_pipeline/build_custom_library.py \
    --manifest scripts/asset_pipeline/sources.yaml \
    --output   "D:/SteamLibrary/steamapps/common/X-Plane 12/Custom Scenery/O4SFR_Library"
```

`convert_to_xplane_obj.py` auto-decimates models above `decimate_target`
(default 2000 tris) and downsizes textures above `texture_max_px`
(default 1024).

### 6. Verify

```
# Region routing + library inventory check
python src/scripts/audit_sfr_building_assets.py --lat <lat> --lon <lon>

# Smoke / regression tests
pytest tests/sfr_optional_library_regions_test.py \
       tests/custom_library_manifest_test.py \
       tests/triage_pipeline_test.py
```

Then load a tile in X-Plane 12 (or open WED) and confirm the `o4sfr/` library
tree resolves and previews render.

## Resume protocol

Coming back later? Read this runbook, glance at the "Where am I?" recipe to
find the next un-triaged slugs, then continue from Step 3. All prior state
is on disk.
