# Regional building-height prior research

Generated 2026-08-29T11:46:30.012825+00:00.

## Result

The sample contains 681,511 joined buildings from 33 five-degree tiles. It transfers at most 1.42 GB of byte ranges rather than downloading the 36 TB archive.

The tables use the exact eight footprint classes in `O4_SFR_Building_Overlay.py`. Mean is the requested statistic; median is the safer fallback prior for the skewed height distribution. P99 plus the published continental GBA height RMSE, rounded up to 5 m, is shown as a conservative hard ceiling candidate.

### Mean height in metres

| Region | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| north_america | 3.9 | 4.6 | 5.3 | 6.0 | 5.9 | 6.9 | 8.8 | 8.5 |
| north_america_ne | 5.8 | 6.3 | 6.7 | 6.3 | 6.3 | 7.0 | 7.3 | 10.0 |
| north_america_west | 3.8 | 4.9 | 5.0 | 5.5 | 6.2 | 7.8 | 8.8 | 9.6 |
| europe | 3.2 | 4.1 | 4.6 | 4.9 | 4.9 | 5.2 | 5.7 | 7.1 |
| scandinavia | 4.2 | 4.9 | 5.1 | 5.9 | 6.1 | 6.8 | 7.8 | 9.2 |
| mediterranean | 3.0 | 3.5 | 4.3 | 5.0 | 5.0 | 5.4 | 5.7 | 5.8 |
| asia | 2.9 | 3.6 | 4.3 | 5.4 | 7.9 | 8.7 | 10.2 | 9.5 |
| se_asia | 4.4 | 5.1 | 5.5 | 5.9 | 6.6 | 7.2 | 8.1 | 9.3 |
| africa | 2.6 | 3.1 | 3.3 | 3.6 | 3.6 | 4.3 | 4.4 | 4.8 |
| australia_oceania | 3.9 | 4.3 | 4.5 | 4.6 | 4.8 | 5.7 | 5.9 | 7.1 |
| south_america | 3.5 | 3.7 | 4.3 | 4.5 | 4.6 | 6.0 | 6.4 | 6.8 |
| generic | 3.6 | 4.6 | 5.1 | 5.6 | 5.9 | 7.0 | 8.2 | 8.9 |

### Median height in metres

| Region | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| north_america | 3.5 | 4.3 | 5.0 | 5.8 | 5.6 | 6.3 | 7.0 | 7.1 |
| north_america_ne | 4.4 | 4.9 | 5.3 | 5.2 | 5.4 | 6.1 | 6.2 | 7.6 |
| north_america_west | 3.1 | 4.3 | 4.6 | 5.0 | 5.6 | 6.6 | 7.0 | 8.0 |
| europe | 3.0 | 3.9 | 4.3 | 4.6 | 4.7 | 5.0 | 5.5 | 6.7 |
| scandinavia | 3.5 | 4.5 | 4.8 | 5.4 | 5.7 | 6.4 | 7.2 | 8.1 |
| mediterranean | 2.7 | 3.3 | 4.2 | 4.9 | 4.8 | 5.3 | 5.4 | 5.7 |
| asia | 2.8 | 3.5 | 3.9 | 4.5 | 5.3 | 6.3 | 7.4 | 8.3 |
| se_asia | 4.2 | 5.0 | 5.4 | 5.7 | 6.4 | 7.0 | 7.7 | 8.8 |
| africa | 2.3 | 2.8 | 3.0 | 3.2 | 3.3 | 3.8 | 4.2 | 5.2 |
| australia_oceania | 3.3 | 4.0 | 4.3 | 4.4 | 4.6 | 5.3 | 5.4 | 5.5 |
| south_america | 2.9 | 3.2 | 3.7 | 4.1 | 4.3 | 4.5 | 6.5 | 6.1 |
| generic | 3.1 | 4.1 | 4.6 | 5.2 | 5.2 | 6.0 | 6.7 | 7.7 |

### P99 plus GBA error margin, in metres

| Region | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| north_america | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 25.0 | 50.0 | 35.0 |
| north_america_ne | 30.0 | 30.0 | 30.0 | 30.0 | 30.0 | 30.0 | 30.0 | 40.0 |
| north_america_west | 20.0 | 20.0 | 20.0 | 20.0 | 25.0 | 40.0 | 50.0 | 50.0 |
| europe | 15.0 | 15.0 | 20.0 | 20.0 | 15.0 | 20.0 | 20.0 | 20.0 |
| scandinavia | 20.0 | 20.0 | 20.0 | 25.0 | 20.0 | 20.0 | 25.0 | 35.0 |
| mediterranean | 15.0 | 15.0 | 15.0 | 15.0 | 20.0 | 20.0 | 20.0 | 20.0 |
| asia | 15.0 | 15.0 | 20.0 | 30.0 | 50.0 | 45.0 | 50.0 | 40.0 |
| se_asia | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 30.0 | 35.0 |
| africa | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 |
| australia_oceania | 20.0 | 15.0 | 15.0 | 15.0 | 15.0 | 20.0 | 20.0 | 30.0 |
| south_america | 25.0 | 25.0 | 25.0 | 25.0 | 25.0 | 40.0 | 25.0 | 25.0 |
| generic | 25.0 | 25.0 | 25.0 | 25.0 | 30.0 | 40.0 | 45.0 | 40.0 |

## Interpretation

These are footprint classes, not use or storey classes. A tower can have the same footprint as a house, and a hospital can have the same footprint as a warehouse. Region and footprint therefore make a useful prior, but they do not justify unconditional low hard caps.

Use the mean or median only when HeightNet is missing or weak. Use the soft ceiling for shrinkage of uncertain predictions. Reserve the hard ceiling for obvious outliers, and bypass it when another source supplies height or floor count.

The `generic` row pools every sampled region because the project's `generic` boundary key has no coherent real-world building stock.

## Comparison with the current runtime

The current generic fallback heights for C1 through C8 are 3.5, 4, 4, 7, 9, 12, 8, and 10 m. The measured regional means show that C5 and C6 are usually much lower than the current 9 and 12 m fallbacks. That happens because the runtime names imply apartments, while the classifier only knows footprint dimensions.

The current 16 m cap for C7 and C8 is near or above the sampled P95 in many regions. It is still too low for the minority of large-footprint apartment, office, hospital, and tower buildings. Raising every large building to the regional P99 ceiling would reintroduce tall warehouse errors. The cap needs one more signal, such as developed versus industrial landcover or a selected asset family.

A safe first experiment is to use the regional median when HeightNet is missing, softly pull predictions above P95 toward P95, and clip only above the reported hard ceiling. Keep the 16 m industrial cap for large footprints on bareland or agricultural context. Evaluate this on shared detections before changing the production default.

## Sampling and limits

GlobalBuildingAtlas supplies predicted, not surveyed, heights. The paper reports continental LoD1 height RMSE from 1.5 m in Oceania to 8.9 m in South America, with no direct African validation. This report adds those errors to the proposed ceilings.

The sample is deterministic. It reads the aligned start of each ODbL polygon tile and its LoD1 height index. Those ODbL files contain OSM, Microsoft, or Google-derived footprints depending on the tile. Three metropolitan or mixed tiles represent each project region. It is broad enough for a practical prior, but it is not a census-weighted regional statistic.

The separate non-ODbL polygon files do not share byte order with the height index, so partial range requests cannot join them safely. This sample excludes those polygons instead of accepting false joins. That makes the African and South American estimates less representative.

GlobalBuildingAtlas uses the maximum predicted height pixel inside each building footprint. Its raw observed maximum is therefore unsuitable as a cap. P95 and P99 are retained in the CSV and JSON outputs.

GBA.LoD1 and the non-ODbL polygons are CC BY-NC 4.0. ODbL polygons have their own ODbL terms. Derived constants need a license review before a commercial distribution.

## Coverage

| Region | Joined buildings |
|---|---:|
| north_america | 60,853 |
| north_america_ne | 65,526 |
| north_america_west | 61,594 |
| europe | 69,842 |
| scandinavia | 73,205 |
| mediterranean | 72,690 |
| asia | 69,070 |
| se_asia | 75,921 |
| africa | 40,813 |
| australia_oceania | 60,180 |
| south_america | 31,817 |

| Class | Joined buildings |
|---|---:|
| C1 tiny residential | 312,757 |
| C2 small residential | 170,476 |
| C3 compact residential | 103,760 |
| C4 medium footprint | 40,223 |
| C5 small apartment | 31,555 |
| C6 apartment block | 9,450 |
| C7 large footprint | 10,567 |
| C8 extra-large footprint | 2,723 |

## Class thresholds

The thresholds copied from the runtime are 90, 170, 270, 450, 850, 1,650, and 7,000 square metres. Classes 4 through 7 also use maximum side limits of 28, 45, 55, and 100 metres.

## Source

Zhu, Chen, Zhang, Shi, and Wang, GlobalBuildingAtlas, Earth System Science Data 17, 6647 to 6668, 2025. DOI 10.5194/essd-17-6647-2025.

Dataset record: https://mediatum.ub.tum.de/1782307
Paper: https://essd.copernicus.org/articles/17/6647/2025/
Google Open Buildings 2.5D cross-check: https://sites.research.google/gr/open-buildings/temporal/
Microsoft global density and height cross-check: https://github.com/microsoft/buildings
