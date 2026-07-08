"""Per-region-flavor and footprint-class material weighting.

Post-overhaul (July 2026) the base DISTRIBUTIONS of wall families and roof
types live per (flavor, class-group) in combo_styles; this module layers
the reference class-profile weights on top and exposes the massing lookup
(per flavor x class profile) the builders consume.  Weights are consumed
with the asset's seeded RNG so variants stay deterministic.
"""

from __future__ import annotations

from functools import lru_cache
import json
import os

from .combo_styles import (
    GROUP_FAMILIES, GROUP_SHADES, GROUP_STRIPS,
    combo_family_weights, combo_massing, combo_roof_weights,
    group_for_archetype,
)

# (choice, weight) pairs; families must have wall_/ground_/plain_ strips.
FLAVOR_STYLES = {
    "generic": {
        "families": (("siding", 2), ("brick", 2), ("stucco", 2)),
        "roofs": (("roof_shingle", 2), ("roof_tile", 2), ("roof_metal", 1)),
    },
    "europe": {
        "families": (("stucco", 3), ("brick", 2), ("siding", 1)),
        "roofs": (("roof_tile", 3), ("roof_shingle", 1), ("roof_metal", 1)),
    },
    "north_america": {
        "families": (("siding", 3), ("brick", 1), ("stucco", 1)),
        "roofs": (("roof_shingle", 3), ("roof_metal", 1), ("roof_tile", 1)),
    },
    "mediterranean": {
        "families": (("stucco", 4), ("brick", 1)),
        "roofs": (("roof_tile", 4), ("roof_metal", 1)),
    },
    "asia": {
        "families": (("concrete", 2), ("stucco", 2), ("brick", 1)),
        "roofs": (("roof_metal", 2), ("roof_tile", 2), ("roof_shingle", 1)),
    },
    "africa": {
        "families": (("stucco", 3), ("concrete", 1)),
        "roofs": (("roof_metal", 3), ("roof_tile", 1)),
    },
    "south_america": {
        "families": (("stucco", 3), ("brick", 2)),
        "roofs": (("roof_tile", 2), ("roof_metal", 2)),
    },
    "australia_oceania": {
        "families": (("brick", 2), ("siding", 2)),
        "roofs": (("roof_metal", 3), ("roof_tile", 1), ("roof_shingle", 1)),
    },
}


# NOTE: the old flavor-only FLAVOR_MASSING table moved to combo_styles
# (_MASSING_BASE + MASSING_PROFILES) where it is differentiated per class
# profile; flavor_massing(flavor, profile) below is the lookup.


def flavor_style(flavor: str) -> dict:
    return FLAVOR_STYLES.get(flavor, FLAVOR_STYLES["generic"])


def flavor_massing(flavor: str, profile: str | None = None) -> dict:
    """Massing parameters per (flavor, class profile) -- combo_styles is
    the source of truth; the old flavor-only table is gone."""
    return combo_massing(flavor, profile)


@lru_cache(maxsize=1)
def reference_styles() -> dict:
    """Load the reference-backed style manifest used by tests and builders."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "reference_styles.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return json.loads(text)
    return yaml.safe_load(text)


def _pairs_from_mapping(value) -> tuple:
    if not value:
        return ()
    if isinstance(value, dict):
        return tuple((str(k), int(v)) for k, v in value.items())
    return tuple((str(k), int(v)) for k, v in value)


def _merge_pairs(*groups) -> tuple:
    weights = {}
    for group in groups:
        for choice, weight in _pairs_from_mapping(group):
            weights[choice] = weights.get(choice, 0) + int(weight)
    return tuple((choice, weight) for choice, weight in weights.items()
                 if weight > 0)


def _group_roof_strips(group: str) -> set:
    return {row[0] for row in GROUP_STRIPS[group] if row[1] == "roof"}


def _filtered(pairs, allowed) -> tuple:
    return tuple((choice, weight) for choice, weight
                 in _pairs_from_mapping(pairs) if choice in allowed)


def style_weights(flavor: str, group: str,
                  profile: str | None = None) -> dict:
    """Families/roofs weights for one (flavor, group, class profile).

    Base distributions are the combo tables (per flavor x group); the
    reference region/class-profile weights merge on top, filtered to the
    families and roof strips that actually exist in the group's layout --
    this is where two classes inside one group still diverge.
    """
    families_base = combo_family_weights(flavor, group)
    roofs_base = combo_roof_weights(flavor, group)
    allowed_fams = set(GROUP_FAMILIES[group])
    allowed_roofs = _group_roof_strips(group)
    refs = reference_styles()
    region_w = ((refs.get("regions") or {}).get(flavor, {})
                .get("weights") or {})
    profile_w = ((refs.get("class_profiles") or {}).get(profile or "", {})
                 .get("weights") or {})
    families = _merge_pairs(
        families_base,
        _filtered(region_w.get("families"), allowed_fams),
        _filtered(profile_w.get("families"), allowed_fams),
    )
    roofs = _merge_pairs(
        roofs_base,
        _filtered(region_w.get("roofs"), allowed_roofs),
        _filtered(profile_w.get("roofs"), allowed_roofs),
    )
    return {
        "families": families or families_base,
        "roofs": roofs or roofs_base,
    }


def style_for_profile(flavor: str, profile: str | None = None,
                      *, apartment_bias: bool = False,
                      industrial_bias: bool = False) -> dict:
    """Back-compat wrapper: biases map onto the group axis."""
    group = "ind" if industrial_bias else ("apt" if apartment_bias else "res")
    return style_weights(flavor, group, profile)


def placement_class_for_footprint(length_m: float, width_m: float) -> int:
    """Mirror the runtime footprint-only class thresholds for procgen assets."""
    area = float(length_m) * float(width_m)
    max_side = max(float(length_m), float(width_m))
    if area <= 90.0:
        return 1
    if area <= 170.0:
        return 2
    if area <= 270.0:
        return 3
    if area <= 450.0 and max_side <= 28.0:
        return 4
    if area <= 850.0 and max_side <= 45.0:
        return 5
    if area <= 1650.0 and max_side <= 55.0:
        return 6
    if area <= 7000.0 and max_side <= 100.0:
        return 7
    return 8


CLASS_PROFILE_BY_CLASS = {
    1: "tiny_residential",
    2: "small_residential",
    3: "compact_residential",
    4: "medium",
    5: "small_apartment",
    6: "apartment_block",
    7: "large",
    8: "extra_large",
}


def profile_for_asset(length_m: float, width_m: float, *,
                      bucket: str = "residential",
                      archetype: str = "") -> str:
    """Choose the reference profile for an asset's footprint and role."""
    zone_class = placement_class_for_footprint(length_m, width_m)
    if bucket in {"industrial"} or archetype in {"warehouse", "bigbox"}:
        return "extra_large" if zone_class == 8 else "large"
    if bucket in {"commercial"} and zone_class < 5:
        return "medium"
    return CLASS_PROFILE_BY_CLASS[zone_class]


def weighted_choice(rng, pairs):
    total = sum(weight for _choice, weight in pairs)
    roll = rng.uniform(0.0, total)
    acc = 0.0
    for choice, weight in pairs:
        acc += weight
        if roll <= acc:
            return choice
    return pairs[-1][0]
