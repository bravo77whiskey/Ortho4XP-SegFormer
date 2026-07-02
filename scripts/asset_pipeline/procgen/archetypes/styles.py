"""Per-region-flavor and footprint-class material weighting.

The atlas gives every flavor the same strip LAYOUT with different palettes,
so colors regionalize for free; this table regionalizes the DISTRIBUTION of
wall families and roof types (Mediterranean = clay tile on stucco, Africa =
corrugated metal on render, Asia = concrete + metal/tile, ...).  Weights are
consumed with the asset's seeded RNG so variants stay deterministic.
"""

from __future__ import annotations

from functools import lru_cache
import json
import os

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


# Per-flavor MASSING parameters: region identity comes from silhouettes as
# much as materials. pitch/hip_pitch in degrees, overhang as a multiplier on
# the base eave overhang, chimney_prob gates the seeded rooftop chimney,
# tank_prob gates a rooftop water tank (flat-roof archetypes).
FLAVOR_MASSING = {
    "generic": {
        "pitch": (32.0, 42.0), "hip_pitch": (26.0, 34.0),
        "overhang": 1.0, "chimney_prob": 0.55, "tank_prob": 0.25,
    },
    "europe": {
        "pitch": (38.0, 48.0), "hip_pitch": (30.0, 38.0),
        "overhang": 0.9, "chimney_prob": 0.75, "tank_prob": 0.05,
    },
    "north_america": {
        "pitch": (34.0, 45.0), "hip_pitch": (28.0, 36.0),
        "overhang": 1.0, "chimney_prob": 0.60, "tank_prob": 0.05,
    },
    "mediterranean": {
        "pitch": (17.0, 25.0), "hip_pitch": (15.0, 22.0),
        "overhang": 1.1, "chimney_prob": 0.25, "tank_prob": 0.35,
    },
    "asia": {
        "pitch": (21.0, 30.0), "hip_pitch": (18.0, 26.0),
        "overhang": 1.5, "chimney_prob": 0.05, "tank_prob": 0.55,
    },
    "africa": {
        "pitch": (14.0, 24.0), "hip_pitch": (13.0, 20.0),
        "overhang": 1.2, "chimney_prob": 0.05, "tank_prob": 0.55,
    },
    "south_america": {
        "pitch": (17.0, 27.0), "hip_pitch": (15.0, 23.0),
        "overhang": 1.1, "chimney_prob": 0.15, "tank_prob": 0.45,
    },
    "australia_oceania": {
        "pitch": (19.0, 29.0), "hip_pitch": (17.0, 25.0),
        "overhang": 1.3, "chimney_prob": 0.30, "tank_prob": 0.15,
    },
}


def flavor_style(flavor: str) -> dict:
    return FLAVOR_STYLES.get(flavor, FLAVOR_STYLES["generic"])


def flavor_massing(flavor: str) -> dict:
    return FLAVOR_MASSING.get(flavor, FLAVOR_MASSING["generic"])


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


def style_for_profile(flavor: str, profile: str | None = None,
                      *, apartment_bias: bool = False,
                      industrial_bias: bool = False) -> dict:
    """Return region + class weighted families/roofs for a generated asset."""
    base = flavor_style(flavor)
    refs = reference_styles()
    region_ref = (refs.get("regions") or {}).get(flavor, {})
    profile_ref = (refs.get("class_profiles") or {}).get(profile or "", {})
    region_weights = region_ref.get("weights") or {}
    profile_weights = profile_ref.get("weights") or {}

    family_bias = ()
    roof_bias = ()
    if apartment_bias:
        family_bias += (("concrete", 3),)
        roof_bias += (("roof_flat", 2),)
    if industrial_bias:
        family_bias += (("concrete", 5),)
        roof_bias += (("roof_metal", 3), ("roof_flat", 2))

    families = _merge_pairs(
        base.get("families"),
        region_weights.get("families"),
        profile_weights.get("families"),
        family_bias,
    )
    roofs = _merge_pairs(
        base.get("roofs"),
        region_weights.get("roofs"),
        profile_weights.get("roofs"),
        roof_bias,
    )
    return {
        "families": families or base["families"],
        "roofs": roofs or base["roofs"],
    }


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
