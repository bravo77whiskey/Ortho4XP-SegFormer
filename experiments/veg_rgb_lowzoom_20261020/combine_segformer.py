"""G = F intersect (SegFormer forest union rangeland), local cached weights only."""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
import O4_Landcover_Inference as SEG

OUT = Path(r"E:\Ortho4XP_Review\rgb_lowzoom_20261020")


def combine(f_mask, class_map):
    if f_mask.shape != class_map.shape:
        raise ValueError("F and SegFormer must use the same native image grid")
    zones = (class_map == SEG.CLASS_TREE) | (class_map == SEG.CLASS_RANGELAND)
    return f_mask.astype(bool) & zones, zones


def main():
    from huggingface_hub import snapshot_download
    import torch

    # Resolve only the installed snapshot. Fail rather than fetching anything.
    model_dir = Path(snapshot_download(SEG._MODEL_VEG_REPO, local_files_only=True))
    SEG._MODEL_VEG_REPO = str(model_dir)
    model, processor, device = SEG.load_vegetation_model(torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"))
    payload = json.loads((OUT / "metrics.json").read_text())
    signature = {
        "model_snapshot": model_dir.name,
        "inference_sha256": hashlib.sha256(Path(SEG.__file__).read_bytes()).hexdigest(),
        "threshold": SEG.segformer_confidence_threshold,
        "patch": SEG.segformer_patch_size, "overlap": SEG.segformer_overlap,
        "native_resolution": True, "tree_support": False,
    }
    for i, rec in enumerate(payload["samples"], 1):
        folder = OUT / rec["id"]
        cache, meta = folder / "segformer_classes.npy", folder / "segformer_classes.json"
        key = dict(signature, image_sha256=rec["sha256"])
        actual_hash = hashlib.sha256(Path(rec["path"]).read_bytes()).hexdigest()
        if actual_hash != rec["sha256"]:
            raise ValueError(f"Source changed: {rec['path']}")
        started = time.perf_counter()
        if cache.exists() and meta.exists() and json.loads(meta.read_text()) == key:
            classes = np.load(cache, allow_pickle=False)
            origin = "reused experiment cache"
        else:
            rgb = np.array(Image.open(rec["path"]).convert("RGB"))
            classes = SEG.run_inference(model, device, rgb, processor)
            np.save(cache, classes, allow_pickle=False)
            meta.write_text(json.dumps(key, indent=2), encoding="utf-8")
            origin = "fresh inference"
        inference_seconds = time.perf_counter() - started
        f_mask = np.array(Image.open(folder / "field_suppressed_native.png"), dtype=bool)
        result, zones = combine(f_mask, classes)
        assert not (result & ~f_mask).any()
        assert not (result & ~zones).any()
        for name, mask in (("segformer_f", result), ("segformer_zones", zones)):
            Image.fromarray(mask).save(folder / (name + "_native.png"), compress_level=1)
            rec["coverage_pct"][name] = float(mask.mean() * 100)
        rec["segformer"] = dict(signature, source=origin, seconds=inference_seconds,
                               class_map_sha256=hashlib.sha256(cache.read_bytes()).hexdigest())
        print(f"[{i}/{len(payload['samples'])}] {rec['key']}: "
              f"F={rec['coverage_pct']['field_suppressed']:.2f}% "
              f"zones={rec['coverage_pct']['segformer_zones']:.2f}% "
              f"G={rec['coverage_pct']['segformer_f']:.2f}% "
              f"({inference_seconds:.1f}s, {origin})", flush=True)
        # Checkpoint metadata so an interrupted run can resume its own maps.
        temp = OUT / "metrics.next.json"
        temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp.replace(OUT / "metrics.json")
    from research import gallery
    gallery(OUT, payload["samples"])


if __name__ == "__main__":
    main()
