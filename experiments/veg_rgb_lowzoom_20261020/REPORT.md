# RGB-only vegetation at ZL16/17

20 October 2026

**Production integration:** option G is now the default tree mask for ZL16 and
ZL17 in `O4_Veg_Lowzoom.py`. ZL18/19 retain their existing detector. G runs on
native RGB before optional polygon-resolution scaling and is not expanded by
morphological closing, weak tree support, enlarged-image inference, or OSM
forest rescue. Existing exclusions, asset selection, and polygonization still
apply. The polygon cache includes the algorithm version and native G mask hash.
Changing the detector does not regenerate previously built scenery; the next
vegetation build uses G. `validate_production.py` compares the production mask
with the frozen preview masks without running any models or downloading data.

Production validation found zero differing pixels on all 27 native 4096 × 4096
images (18 ZL16 and 9 ZL17); results are in `production-validation.json`.
The focused source suite passed 214 tests plus 20 subtests, including G routing,
building exclusions, OSM-rescue bypass, RGB cache invalidation, native-before-
coarse-resolution processing, and preservation of the ZL18/19 path. These checks
establish implementation parity, not ground-truth detection accuracy.

**Preview update:** the gallery now supports synchronized wheel zoom, drag panning, a zoom slider, Fit, and 1:1 viewing. It uses the original 4096 × 4096 JPEGs and native binary masks. Changing methods retains the view; changing samples resets it.

**Requested hybrid option G:** `F AND (SegFormer forest OR rangeland)`. SegFormer class 5 and class 2 define the allowed zones, and F determines the marked pixels inside them. The separate "SegFormer forest + rangeland zones" option displays the zone mask before F. All 27 images have both new masks; every G mask was checked against that exact intersection.

The hybrid uses the already-installed `nave1616/SegFormer-landcover-FT` snapshot `e5a8046f89268dd56a328ff0ad99c47c52cc4ae6`, with offline loading enforced. It runs at native resolution with the existing 512-pixel patches, 64-pixel overlap, and 0.5 class-confidence threshold. There is no tree-support expansion, enlarged-image pass, or added OSM context. Class maps, image hashes, inference-code hashes, and provenance are saved per sample. No downloads were needed.

To regenerate G after producing F:

```powershell
& G:\Dev\Ortho4XP\.venv\Scripts\python.exe experiments\veg_rgb_lowzoom_20261020\combine_segformer.py
```

To rebuild only the preview:

```powershell
& .\.venv\Scripts\python.exe experiments\veg_rgb_lowzoom_20261020\build_gallery.py
```

The research findings below describe the original RGB-only A–F experiment. Its reported accuracy caveats and timing measurements do not include G. G can remove false positives outside the model zones, but also loses real trees when SegFormer misses a vegetation zone.

The most promising approach in this experiment is **colour plus texture measured in metres, with optional rejection of large areas of parallel edges**. It detects substantial wooded areas and rejects much of the green water that defeats colour thresholds. It still plants some crops and misses some smooth canopy. It is a research candidate, not a validated replacement for the current detector.

[Open the interactive comparison](http://127.0.0.1:8791/rgb_lowzoom_20261020/), or open the [local HTML file](E:/Ortho4XP_Review/rgb_lowzoom_20261020/index.html) directly. Select method C for the initial texture candidate or F for the crop-row refinement. All six alternatives remain available.

![RGB, initial texture candidate, and crop-row refinement](G:/Dev/Ortho4XP-SegFormer/experiments/veg_rgb_lowzoom_20261020/comparison.jpg)

## Scope and current pipeline

The initial experiment examined committed source at `625c6a47`, without using
existing uncommitted changes or altering scenery, the installed application,
configuration, or model caches. The production integration described above was
added afterward at the user's request, with unrelated changes stashed separately
during packaging.

The committed vegetation pipeline loads SegFormer when it needs a class map, optionally performs an enlarged pass at ZL16, and uses canopy-model output on ZL18/19. Its own comments describe the low-resolution canopy-height problem. This experiment did not rerun or measure those models.

The existing `_canopy_look` colour rule is a useful clue, but it is gated by forest context and model classes. Running that sort of rule everywhere is a different problem: green water and fields also pass it.

The experiments below consume only local RGB images and their metres-per-pixel scale. No models, existing inference masks, OSM masks, pretrained weights, package installation, or image downloads were used. Web reading was limited to method references.

## What was tested

27 existing 4096 × 4096 textures: 18 at ZL16 and 9 at ZL17, covering Kaohsiung, Shenzhen, Wenzhou, Shouguang, Taipei, and Gran Canaria. Ground sampling spans 0.956–2.206 m/pixel. The selection includes urban areas, crop fields, water, wooded hills, and dry terrain, but is not a geographically balanced benchmark.

| Candidate | Operation | Observed tradeoff |
|---|---|---|
| A: green | Normalized excess green, channel ratios, component cleanup | Broad vegetation candidates; crops and some water also pass |
| B: dark green | Add local brightness threshold | Dark water/crops remain; bright canopy can be lost |
| C: texture | Colour support, local contrast, fine texture, edge-direction filtering | Strongest initial balance; removes much smooth water but still accepts crop beds |
| D: seed growth | Stronger texture seeds, growth limited to 8 m inside weaker support | Little improvement on the spot checks; loses smooth canopy in Taipei |
| E: dark rows | Black-hat morphology and directional structure | Finds dark linear objects, not necessarily trees; unsuitable as a standalone tree detector |
| F: crop-row refinement | C with parallel-edge rejection over 50 m and 80 m windows | Reduces field responses; can also reject tree rows and plantations |

C uses roughly 12–22 m neighbourhoods for colour and texture, rather than fixed pixel sizes. Its detail measure compares fine intensity variation with local contrast, and its structure tensor rejects strongly directional edges. F rejects pixels with direction coherence above 0.55 at either wider scale. Components smaller than 60 m² are removed. These are heuristic choices, not learned parameters or calibrated confidence scores.

F was developed after examining the initial patch results. Its threshold is therefore informed by the same examples used below. The initial C results remain available unchanged.

## Measurements and limits

I visually inspected 17 small RGB patches, six judged to be predominantly tree canopy and eleven predominantly non-tree. Reference images and coordinates are saved. Some patches contain small mixed boundaries, so these are spot checks, not pixel-accurate labels.

| Candidate | Mean marked area in six tree patches | Mean marked area in eleven non-tree patches |
|---|---:|---:|
| A: green | 95.2% | 43.7% |
| B: dark green | 95.2% | 36.8% |
| C: texture | 94.3% | 15.6% |
| D: seed growth | 93.6% | 14.3% |
| E: dark rows | 6.7% | 7.3% |
| F: crop-row refinement | 93.2% | 4.0% |

Each patch has equal weight. These numbers are not precision, recall, or an estimate of general accuracy. The low E response on non-tree patches is not a success: it also misses almost all of the checked tree interiors.

Concrete examples:

- A Shenzhen reservoir-water patch fell from 92.4% marked by A to 1.1% by C.
- A Wenzhou dark crop patch fell from 55.7% with C to 6.7% with F.
- A second crop-bed patch still had **27.4%** marked with F, down from 53.6%. Agriculture remains unresolved.
- The roadside tree patch fell from 87.6% with C to 82.9% with F.
- On one Taipei texture, C marked 55.8% of the image while D marked 36.5%. Visual inspection shows smooth woodland being lost. Those percentages describe mask coverage, not forest accuracy.

For resolution sensitivity, I reduced all nine native ZL17 images by 2× with area averaging and reran the detector using twice the metres-per-pixel value. Median mask intersection-over-union was 0.860 for C, 0.847 for D, and 0.821 for F. F ranged from 0.726 to 0.873. These measure consistency on the same imagery, not accuracy; separately acquired ZL16 imagery may differ.

On this i9-13900K with four OpenCV threads, A–E together took a median 1.94 seconds per image. F added a median 2.25 seconds, giving a median combined processing time of 4.20 seconds. Timings exclude loading and saving. This is a whole-image prototype; peak RAM and full-tile throughput were not measured.

## Recommendation

Use C as the starting detector and keep F as a separately reviewable conservative option. F is useful where false trees in fields matter, but its wider rejection can erase real vegetation beside roads or in regular plantations. D should not be the default; its stronger seeds do not solve the ambiguity of smooth forests versus smooth fields.

For a production experiment, return separate masks for tree candidates, other green vegetation, and uncertain areas. Do not label every green non-tree pixel as grass: crops, algae, and shrubs remain possible. RGB at these scales does not establish tree height, species, or reliable individual-tree positions. Upsampling alone cannot recover missing crown detail.

A production integration needs an explicit ZL16/17 branch before `load_vegetation_model` and before the extra enlarged inference pass, separate cache keys, and haloed processing to avoid artificial texture seams. Model-derived building/water/class exclusions must also be removed or replaced in that branch. Existing local pavement, mesh-water, or road masks could be optional exclusions if a later task permits non-image inputs; none were used here.

Grass needs its own decision. `O4_Grass_Overlay.fused_grass` currently consumes model classes. Replacing only the tree mask would not make the full vegetation pipeline model-free.

Before enabling either candidate, label independent scenes with smooth forest, tree plantations, crops, leaf-off trees, dry scrub, shadows, and green water. Assess tree/non-tree errors separately and review resulting scenery. There is no simulator validation or comparison against current production predictions in this report.

## Reproduce and inspect

Run in `G:\Dev\Ortho4XP-SegFormer`, using the already-installed environment:

```powershell
& .\.venv\Scripts\python.exe -W error experiments\veg_rgb_lowzoom_20261020\research.py
& .\.venv\Scripts\python.exe -W error experiments\veg_rgb_lowzoom_20261020\refine_direction.py
& .\.venv\Scripts\python.exe -W error experiments\veg_rgb_lowzoom_20261020\evaluate.py
```

The initial research script accepts `--root` and `--out`; the refinement and evaluation scripts use the default output path in `evaluate.py`. The default output is `E:\Ortho4XP_Review\rgb_lowzoom_20261020`. The HTML works locally without fetching a JavaScript library or loading an external service.

- [Input hashes, per-image coverage, versions, and timings](E:/Ortho4XP_Review/rgb_lowzoom_20261020/metrics.json)
- [Patch coordinates, measurements, and resolution checks](E:/Ortho4XP_Review/rgb_lowzoom_20261020/evaluation.json)
- [Visually inspected reference patches](G:/Dev/Ortho4XP-SegFormer/experiments/veg_rgb_lowzoom_20261020/reference_patches.jpg)
- Each sample folder contains native binary PNG masks and reduced preview overlays.

Validation: the final batch and evaluation passed with runtime warnings treated as errors; flat black, white, grey, green, and water-coloured negative controls passed; all 27 input hashes matched after processing. Gallery sample selection, method selection, and overlay toggling were checked in the browser.

Method references: [Woebbecke et al., colour indices for vegetation/background separation](https://pure.psu.edu/en/publications/color-indices-for-weed-identification-under-various-soil-residue-/) supports the use of visible-band colour indices, not tree recognition; [OpenCV connected-component documentation](https://docs.opencv.org/4.5.4/d3/dc0/group__imgproc__shape.html) describes the component statistics used for area cleanup. All performance and coverage numbers above come from this local experiment.
