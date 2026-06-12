"""Per-region-flavor material weighting for archetype style picks.

The atlas gives every flavor the same strip LAYOUT with different palettes,
so colors regionalize for free; this table regionalizes the DISTRIBUTION of
wall families and roof types (Mediterranean = clay tile on stucco, Africa =
corrugated metal on render, Asia = concrete + metal/tile, ...).  Weights are
consumed with the asset's seeded RNG so variants stay deterministic.
"""

from __future__ import annotations

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


def weighted_choice(rng, pairs):
    total = sum(weight for _choice, weight in pairs)
    roll = rng.uniform(0.0, total)
    acc = 0.0
    for choice, weight in pairs:
        acc += weight
        if roll <= acc:
            return choice
    return pairs[-1][0]
