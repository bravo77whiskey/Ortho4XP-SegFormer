"""Compare production G against every frozen native preview mask, offline."""
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import O4_Veg_Lowzoom as LOW

OUT = Path(r"E:\Ortho4XP_Review\rgb_lowzoom_20261020")


def main():
    cv2.setNumThreads(4)
    samples = json.loads((OUT / "metrics.json").read_text())["samples"]
    results = []
    for sample in samples:
        folder = OUT / sample["id"]
        source = Path(sample["path"])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == sample["sha256"]
        rgb = np.asarray(Image.open(source).convert("RGB"))
        classes = np.load(folder / "segformer_classes.npy", allow_pickle=False)
        expected = np.asarray(Image.open(folder / "segformer_f_native.png"), dtype=bool)
        south, west, north, east = sample["bbox"]
        mpp = LOW.metres_per_pixel(north, south, west, east, *classes.shape)
        assert abs(mpp - sample["mpp"]) < 1e-12
        started = time.perf_counter()
        actual = LOW.tree_mask(rgb, classes, mpp)
        different = int(np.count_nonzero(actual != expected))
        record = dict(id=sample["id"], zl=sample["zl"], differing_pixels=different,
                      seconds=time.perf_counter() - started, pixels=int(actual.size),
                      tree_pixels=int(actual.sum()))
        results.append(record)
        print(json.dumps(record), flush=True)
        assert different == 0, sample["id"]
    report = dict(signature=LOW.SIGNATURE, samples=results, all_exact=True,
                  module_sha256=hashlib.sha256(Path(LOW.__file__).read_bytes()).hexdigest())
    (OUT / "production-validation.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
