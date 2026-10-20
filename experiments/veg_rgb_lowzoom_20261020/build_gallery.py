"""Rebuild the preview without running image processing or model inference."""
import json
from pathlib import Path

from research import gallery

if __name__ == "__main__":
    out = Path(r"E:\Ortho4XP_Review\rgb_lowzoom_20261020")
    gallery(out, json.loads((out / "metrics.json").read_text())["samples"])
    print(f"Updated {out / 'index.html'}")
