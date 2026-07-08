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


def test_clean_flavors_stay_clean():
    """USER DIRECTIVE (2026-07-08): asia textures must NEVER carry grime.

    Grime resurfaced four times because it stacks from independent layers
    (FLAVOR_PATTERNS, reference_styles.yaml overrides, page tweaks, the
    compositor's page-1 grunge floor, staining baked into AI sources).
    This test pins the kill-switch across every group and page: if it
    fails, someone re-added weathering to a CLEAN flavor -- remove it.
    """
    import atlas
    import compose_atlases
    from archetypes.combo_styles import CLEAN_FLAVORS, GROUPS

    assert "asia" in CLEAN_FLAVORS
    references = atlas._load_reference_styles()
    for flavor in CLEAN_FLAVORS:
        for group in GROUPS:
            for page in range(3):
                pattern = atlas.pattern_for_combo(
                    flavor, group, references, page)
                assert pattern["grime"] == 0.0, (flavor, group, page)
                assert pattern["streaks"] == 0.0, (flavor, group, page)
                assert pattern["roof_streaks"] == 0.0, (flavor, group, page)
                assert pattern["rust"] is False, (flavor, group, page)
        # The compositor must fully disable its grunge overlay (the page-1
        # formula has a constant floor even at grime=0).
        dirt = compose_atlases.FLAVOR_DIRT.get(flavor) or {}
        assert float(dirt.get("grunge_scale", 1.0)) == 0.0, flavor
