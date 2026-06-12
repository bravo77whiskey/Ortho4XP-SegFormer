"""Monte-Carlo coverage check for the procgen dimension grid.

Builds the REAL scored candidate index (_build_yolo_object_candidate_index)
from the manifest's enabled assets and verifies that synthetic YOLO
detections sampled across the grid's domain always find a candidate that
passes the runtime pre-filter:

    required_length <= det_length, required_width <= det_width,
    asset_area / det_area >= per-class min coverage

This is the empirical proof that the geometric ladders (r=1.22 / r=1.10)
satisfy the 0.5 / 0.65 / 0.8 coverage floors.
"""

import json
import math
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROCGEN = ROOT / "scripts" / "asset_pipeline" / "procgen"
SRC = ROOT / "src"
for path in (str(PROCGEN), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

import O4_SFR_Building_Overlay as overlay  # noqa: E402
from archetypes import ARCHETYPES  # noqa: E402

SAMPLES_PER_CLASS = 10_000

# Classes whose grid coverage depends on archetypes that land in M2; skip
# (not xfail) until they exist so the suite stays green per milestone.
CLASS_REQUIRES_ARCHETYPE = {7: "warehouse"}

# Sampling domain mirrors config.yaml: the grid covers detections from the
# smallest rung (3.4 m) up to the 7000 m^2 / aspect caps.  Detections outside
# this domain fall back to default/simHeaven pools or facades by design.
# The pool gate rejects assets under 12 m^2, so the smallest legal square
# asset is ~3.5 m a side; detections narrower than that have no candidate
# by design (they fall back to default pools / facades).
MIN_SIDE_M = 3.55
TIER_A_MAX_ASPECT = 3.0
TIER_B_MAX_ASPECT = 4.0
TIER_SPLIT_AREA_M2 = 270.0
MAX_AREA_M2 = 7000.0
MAX_SIDE_M = 105.0

CLASS_AREA_BANDS = {
    1: (12.0, 90.0),
    2: (90.0, 170.0),
    3: (170.0, 270.0),
    4: (270.0, 450.0),
    5: (450.0, 850.0),
    6: (850.0, 1650.0),
    7: (1650.0, 7000.0),
}


@pytest.fixture(scope="module")
def candidates():
    with open(PROCGEN / "procgen_manifest.json", "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    pools = {cls: [] for cls in overlay.BLD_PLACEMENT_CLASSES}
    added = 0
    for asset in manifest["assets"]:
        if not asset.get("enabled"):
            continue
        hx = asset["length_m"] / 2.0
        hy = asset["width_m"] / 2.0
        if overlay._append_object_asset(
            pools, asset["virtual_path"], (-hx, hx, -hy, hy),
            "O4SFR_Library", region_priority=1,
        ):
            added += 1
    assert added > 0
    index = overlay._build_yolo_object_candidate_index(pools)
    flat = [
        cand
        for class_candidates in index["assets_by_class"].values()
        for cand in class_candidates
    ]
    assert flat
    return flat


def _sample_detection(rng, zone_class):
    area_lo, area_hi = CLASS_AREA_BANDS[zone_class]
    max_aspect = (
        TIER_A_MAX_ASPECT if area_hi <= TIER_SPLIT_AREA_M2
        else TIER_B_MAX_ASPECT
    )
    for _ in range(100):
        area = rng.uniform(area_lo, min(area_hi, MAX_AREA_M2))
        aspect = rng.uniform(1.0, max_aspect)
        width = math.sqrt(area / aspect)
        length = width * aspect
        if width >= MIN_SIDE_M and length <= MAX_SIDE_M:
            return length, width
    return None


@pytest.mark.parametrize("zone_class", sorted(CLASS_AREA_BANDS))
def test_grid_covers_class(zone_class, candidates):
    needed = CLASS_REQUIRES_ARCHETYPE.get(zone_class)
    if needed and needed not in ARCHETYPES:
        pytest.skip(f"class {zone_class} grid needs archetype {needed!r} (M2)")
    rng = random.Random(1234 + zone_class)
    min_cov = overlay._yolo_object_min_coverage_for_class(zone_class)
    misses = []
    sampled = 0
    for _ in range(SAMPLES_PER_CLASS):
        dims = _sample_detection(rng, zone_class)
        if dims is None:
            continue
        sampled += 1
        det_l, det_w = dims
        det_area = det_l * det_w
        found = any(
            cand["required_length_m"] <= det_l + 1e-9
            and cand["required_width_m"] <= det_w + 1e-9
            and cand["coverage_area_m2"] / det_area >= min_cov - 1e-9
            for cand in candidates
        )
        if not found and len(misses) < 5:
            misses.append((round(det_l, 2), round(det_w, 2)))
    assert sampled > SAMPLES_PER_CLASS * 0.9
    assert not misses, (
        f"class {zone_class} (min_cov {min_cov}): no candidate for "
        f"detections like {misses}"
    )
