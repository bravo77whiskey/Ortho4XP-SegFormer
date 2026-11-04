# Smaller and quantised overlay models (2026-11-03)

Goal: win back overlay time lost to the quality work by swapping in smaller
models of the same family or quantised versions of the current ones. Judged by
eye in the review viewer (hotspots set, A = `mdl-baseline`); numbers below are
for time and for how far each variant's raw output drifts from production.

## Where the model time goes

Isolated, one model on the GPU at a time, average per dense ZL16 texture
(RTX 4080): SegFormer 4.2 s (0.9 s building pass + 3.3 s vegetation's 2x
pass), HeightNet ~2.4 s, building YOLO 1.1 s, stock YOLO 0.8 s. About 8.5 s of
model time per texture; the rest of an overlay is CPU work (placement,
greenhouses, suppression, vegetation polygons). In a real build the GPU is
shared by the building and vegetation overlays (and parallel tiles), so short
GPU steps also cut the time other steps spend waiting for VRAM.

## Results

Isolated timings over the 23 hotspot textures (`E:\Ortho4XP_Review\model_size\logs3`):

| Model | Production | Candidate | Time | Output vs production |
|---|---|---|---|---|
| SegFormer-B4 (native / 2x pass) | 21.2 / 36.1 s | fused attention (SDPA) | 18.6 / 31.5 s | 99.96 % of pixels identical |
| | | TensorRT fp16 | 17.5 / 28.4 s | 99.89 % |
| | | TensorRT int8 | 14.7 / 23.9 s | 98.9 % (unsure class IoU 0.87) |
| HeightNet v2s (ConvNeXt-S) | 0.236 ms/det | TensorRT fp16 | 0.169 ms/det | MAE 0.016 m, 99.5 % same floors |
| | | TensorRT int8 | 0.173 ms/det | MAE 0.54 m, 85 % same floors |
| | | 112 / 96 / 64 px crops | 0.175 / 0.125 / 0.060 | MAE 1.4 / 2.3 / 4.9 m |
| | | legacy reg40 | 0.044 ms/det | median 23.6 m vs 10.9 m |
| Stock YOLO-OBB (landmarks) | x 18.7 s | l / m / s | 11.3 / 9.2 / 5.4 s | 45 / 32 / 28 % of landmarks matched |
| Building YOLO-OBB | x, 8 classes 25.2 s | yolo26m-obb-buildings (1 class) | 12.7 s | 1.1-4x more raw boxes |
| Canopy height (ZL18/19) | ViT-H 39.6 s | Meta ViT-L (compressed_SSLlarge) | 29.2 s | tree mask IoU 0.63-0.68 |

Review runs (2 workers, full pipeline, same imagery):

| Run | Wall | Bld stages | Buildings | Tree polygons |
|---|---|---|---|---|
| `mdl-baseline` | 844 s | 1099 s | 201,054 | 107,993 |
| `mdl-sdpa` | 759 s | 1085 s | -0.0 % | +0.1 % |
| `mdl-trt16` | (split run) | 746 s | -0.0 % | +0.1 % |
| `mdl-trt8` | 656 s | 743 s | +0.0 % | +0.4 % |
| `mdl-stock-l` | 778 s | 1009 s | -0.0 %, landmarks 244 -> 182 | +0.0 % |
| `mdl-canopy-vitl` (ZL18/19) | - | - | - | cover same, 5-35 % fewer polygons |
| `mdl-bld-yolo26m` | 806 s | 1181 s | +46 % | -0.1 % |

Not available: no smaller SegFormer trained on our 9 classes exists (the only
B2 landcover model on Hugging Face has 7 LoveDA classes, no rangeland or
developed, 0.51 mIoU); no ConvNeXt-Tiny HeightNet checkpoint exists. Meta's
compressed ViT-H is int8 for CPU — same speed on the GPU, only a smaller file.

## Recommendation

1. TensorRT fp16 for SegFormer and HeightNet (chosen): building stages -32 %
   in the review runs, output practically identical. Needs the `tensorrt` package in
   the overlay venv (~1 GB) and engines built once per GPU/driver (minutes).
2. If no new dependency is wanted: SDPA attention, -10 % wall time, identical
   output, a few lines of code.
3. Not worth it: smaller stock YOLO (-0.3 s/texture, a quarter of the tanks and
   pools gone), smaller HeightNet crops or the legacy model (heights drift
   more than the model's own error), int8 HeightNet (no faster than fp16).
4. Look at, not for speed: yolo26m buildings (+46 % buildings, slower overall)
   and the ViT-L canopy (fewer single trees at ZL18/19, ~1.3 s/texture saved).

## In production (2026-11-04)

TensorRT fp16 was chosen after the visual review ("barely any difference").

- `O4_TensorRT` builds each engine on first use (ONNX export, then TensorRT:
  ~260 s for SegFormer, ~80 s for HeightNet on the RTX 4080) and keeps it in
  `Overlay_cache/_tensorrt`, keyed by weights, input shape, GPU and TensorRT
  version. One process builds at a time; the others stay on torch meanwhile.
- Setting `overlay_tensorrt` (default on; env `O4_TENSORRT`, cache folder
  `O4_TENSORRT_DIR`). `O4_BLD_HEIGHT_FP16=0` also keeps HeightNet in torch.
- `tensorrt-cu12==10.9.0.34` and `onnx==1.17.0` install into the overlay venv
  on the first overlay run when there is an NVIDIA GPU. Without them, or if a
  build fails, the models run in torch as before.
- TensorRT scratch memory comes from torch on every call, so the VRAM guard
  sees it; with TensorRT's default static allocation four overlay processes
  held ~9 GB and spilled into system memory.
- The other variants (SDPA, smaller crops, ViT-L canopy, single-class YOLO)
  were removed; the canopy and YOLO hooks are kept as a patch in
  `E:\Ortho4XP_Review\model_size\unchosen_hooks.patch`.

## Found on the way

- Vegetation could crash a whole tile with `PermissionError` reading
  `<texture>_veg.npy` while the building overlay swapped it in (twice today).
  The read now retries.
- The first baseline run left five textures without buildings, apparently
  CUDA OOM under the VRAM guard's per-process cap ("648 MiB allowed"); its
  logs were overwritten, and the repeat run was clean.
