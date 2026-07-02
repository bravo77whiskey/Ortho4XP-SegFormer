import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROCGEN = ROOT / "scripts" / "asset_pipeline" / "procgen"
SRC = ROOT / "src"
for path in (str(PROCGEN), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

import O4_SFR_Building_Overlay as overlay  # noqa: E402
from archetypes.styles import CLASS_PROFILE_BY_CLASS, reference_styles, \
    style_for_profile  # noqa: E402


EXPECTED_REGIONS = {
    "generic",
    "europe",
    "north_america",
    "mediterranean",
    "asia",
    "africa",
    "south_america",
    "australia_oceania",
}


def test_reference_styles_cover_current_procgen_regions():
    data = reference_styles()
    assert set(data["regions"]) == EXPECTED_REGIONS
    for region, spec in data["regions"].items():
        assert spec["cues"], region
        assert "families" in spec["weights"], region
        assert "roofs" in spec["weights"], region


def test_reference_styles_cover_all_runtime_placement_classes():
    data = reference_styles()
    expected_classes = set(overlay.BLD_PLACEMENT_CLASSES)
    found_classes = {
        int(profile["class"])
        for profile in data["class_profiles"].values()
    }
    assert found_classes == expected_classes
    assert set(CLASS_PROFILE_BY_CLASS) == expected_classes


def test_style_for_profile_merges_region_and_class_weights():
    style = style_for_profile("africa", "extra_large",
                              industrial_bias=True)
    family_weights = dict(style["families"])
    roof_weights = dict(style["roofs"])
    assert family_weights["concrete"] > family_weights.get("siding", 0)
    assert roof_weights["roof_metal"] > roof_weights.get("roof_shingle", 0)


def test_reference_styles_yaml_round_trips():
    path = PROCGEN / "reference_styles.yaml"
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert loaded["version"] == 1
