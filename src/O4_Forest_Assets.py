"""Fixed forest-asset policy for future overlay generation.

This module intentionally avoids any dynamic ``library.txt`` generation or
runtime asset discovery. Future overlays emit only the explicit, fixed
allowlist below.

The current cap is 24 meters, based on measured X-Plane 12 and Global Forests
v2 ``TREE`` records from local reference installs. Older overlays for tiles
outside the current regeneration set remain untouched on disk.
"""

from __future__ import annotations

import re

import O4_SFR_Climate_Regions as CLIMATE

HEIGHT_CUTOFF_METERS = 24.0

GFV2_REGIONS = (
    "tropical",
    "subtropical",
    "northsouth",
    "northmiddle",
    "northnorth",
)

GFV2_ROLE_DLEVELS = {
    "tree": (25, 50, 75, 100),
    "woodland": (25, 50, 75, 100),
    "cropland": (25,),
}

DEFAULT_SHORT_TREE_BY_REGION = {
    "tropical": ("lib/vegetation/forests/broadleaves/warm_dry.for",),
    # Warm dry conifers proved noticeably heavier than the GFv2 woodland
    # assets they displaced, so keep warm-climate defaults to broadleaf only.
    "subtropical": ("lib/vegetation/forests/broadleaves/warm_dry.for",),
    "northsouth": ("lib/vegetation/forests/broadleaves/warm_dry.for",),
    "northmiddle": (
        "lib/vegetation/forests/broadleaves/cold_low.for",
        "lib/vegetation/forests/conifers/cold_low.for",
    ),
    "northnorth": (
        "lib/vegetation/forests/broadleaves/cold_low.for",
        "lib/vegetation/forests/conifers/cold_low.for",
    ),
}

_GFV2_ROLE_SPECS = {
    "tropical": {
        "tree": (("woodland", (1, 2)),),
        "woodland": (("woodland", (1, 2)),),
        "cropland": (("cropland", (1, 2)),),
    },
    "subtropical": {
        "tree": (("woodland", (1, 2)),),
        "woodland": (("woodland", (1, 2)),),
        "cropland": (("cropland", (1, 2)),),
    },
    "northsouth": {
        "tree": (("woodland", (1,)),),
        "woodland": (("woodland", (1,)),),
        "cropland": (("cropland", (1, 2)),),
    },
    "northmiddle": {
        "tree": (("woodland", (1,)),),
        "woodland": (("woodland", (1,)),),
        "cropland": (("cropland", (1, 2)),),
    },
    "northnorth": {
        "tree": (
            ("decidbroad", (1,)),
            ("everbroad", (1,)),
            ("mixed", (1,)),
        ),
        "woodland": (("mixed", (1,)),),
        "cropland": (("mixed", (1,)),),
    },
}

KNOWN_TALL_DEFAULT_FOREST_PATHS = (
    "lib/vegetation/forests/broadleaves/hot.for",
    "lib/vegetation/forests/broadleaves/warm.for",
    "lib/vegetation/forests/broadleaves/temperate.for",
    "lib/vegetation/forests/conifers/cold.for",
    "lib/vegetation/forests/mixed/temperate.for",
)

KNOWN_TALL_GFV2_FOREST_PATHS = (
    "forests/tropical/woodland/tropical_woodland_100_y3.for",
    "forests/subtropical/woodland/subtropical_woodland_75_y3.for",
    "forests/northmiddle/woodland/northmiddle_woodland_75_y2.for",
    "forests/northnorth/woodland/northnorth_woodland_75_y1.for",
    "forests/northnorth/cropland/northnorth_cropland_25_y1.for",
)

MESH_DEFS_BY_PATH = {
    "lib/vegetation/forests/broadleaves/warm_dry.for": 45,
    "lib/vegetation/forests/broadleaves/cold_low.for": 39,
    "lib/vegetation/forests/conifers/cold_low.for": 46,
}

GFV2_MESH_DEFS_DEFAULT = 12
TREE_LOW_IMPACT_CONTEXTS = frozenset({"managed", "treeline"})
GFV2_PATH_RE = re.compile(
    r"^forests/(?P<region>[^/]+)/(?P<family>[^/]+)/"
    r"(?P=region)_(?P=family)_(?P<dlevel>\d+)_y(?P<variant>\d+)\.for$",
    re.IGNORECASE,
)
GFV2_TYPE_EXCLUDE_TOKENS = frozenset(
    {
        "palm",
        "coconut",
        "shrub",
        "shrubs",
        "scrub",
        "brush",
        "bush",
    }
)


def climate_region(lat: float, lon: float | None = None) -> str:
    """Return the climate bucket used by the SFR vegetation pipeline.

    When longitude is provided this uses the vendored Koppen-Geiger climate
    grid; latitude-only callers keep the old latitude-band fallback.
    """
    if lon is not None:
        return CLIMATE.forest_region_for_latlon(lat, lon)
    return CLIMATE.latitude_band_region(lat + 0.5)


def default_short_tree_candidates_for_region(region: str) -> tuple[str, ...]:
    """Return the approved default tree assets for a climate region."""
    try:
        return DEFAULT_SHORT_TREE_BY_REGION[region]
    except KeyError as exc:
        raise ValueError(f"Unsupported climate region: {region!r}") from exc


def default_short_tree_candidates_for_lat(lat: float) -> tuple[str, ...]:
    """Return the approved default tree assets for a latitude."""
    return default_short_tree_candidates_for_region(climate_region(lat))


def default_short_tree_for_lat(lat: float) -> str:
    """Return the canonical default short-tree path for a latitude."""
    return default_short_tree_candidates_for_lat(lat)[0]


def _gfv2_path(region: str, family: str, dlevel: int, variant: int) -> str:
    fname = f"{region}_{family}_{dlevel}_y{variant}.for"
    return f"forests/{region}/{family}/{fname}"


def short_gfv2_candidates(region: str, role: str, dlevel: int) -> tuple[str, ...]:
    """Return the fixed measured allowlist of GFv2 assets for a role."""
    if region not in GFV2_REGIONS:
        raise ValueError(f"Unsupported climate region: {region!r}")
    try:
        allowed_dlevels = GFV2_ROLE_DLEVELS[role]
    except KeyError as exc:
        raise ValueError(f"Unsupported forest role: {role!r}") from exc
    if dlevel not in allowed_dlevels:
        raise ValueError(f"Unsupported density level {dlevel!r} for {role!r}")
    return tuple(
        _gfv2_path(region, family, dlevel, variant)
        for family, variants in _GFV2_ROLE_SPECS[region][role]
        for variant in variants
    )


def short_tree_candidates(region: str, dlevel: int) -> tuple[str, ...]:
    """Combine measured short GFv2 tree assets with approved default trees."""
    return (
        short_gfv2_candidates(region, "tree", dlevel)
        + default_short_tree_candidates_for_region(region)
    )


def gfv2_tree_candidates(region: str, dlevel: int) -> tuple[str, ...]:
    """Return the measured GFv2-only tree pool for bulk canopy placement."""
    return short_gfv2_candidates(region, "tree", dlevel)


def mesh_defs_for_path(path: str) -> int:
    """Return the measured or budgeted mesh-def count for an allowed asset."""
    if path.startswith("forests/"):
        return GFV2_MESH_DEFS_DEFAULT
    return MESH_DEFS_BY_PATH.get(path, 99)


def tree_candidate_weights(candidates) -> tuple[int, ...]:
    """Bias tree selection toward lower-mesh assets when defaults are allowed."""
    return tuple(max(1, 60 - mesh_defs_for_path(path)) for path in candidates)


def choose_tree_path(region: str, dlevel: int, rng=None, context: str = "bulk") -> str:
    """Choose a tree path with mesh-aware context-sensitive selection."""
    if context in TREE_LOW_IMPACT_CONTEXTS:
        candidates = short_tree_candidates(region, dlevel)
        weights = tree_candidate_weights(candidates)
    else:
        candidates = gfv2_tree_candidates(region, dlevel)
        weights = None
    return choose_path(candidates, rng, weights=weights)


def parse_gfv2_path(path: str) -> dict | None:
    """Return GFv2 path metadata, or None when the path is not a GFv2 asset."""
    normalized = (path or "").replace("\\", "/").lower().lstrip("/")
    match = GFV2_PATH_RE.match(normalized)
    if not match:
        return None
    try:
        dlevel = int(match.group("dlevel"))
        variant = int(match.group("variant"))
    except ValueError:
        return None
    return {
        "region": match.group("region"),
        "family": match.group("family"),
        "dlevel": dlevel,
        "variant": variant,
        "path": normalized,
    }


def is_acceptable_gfv2_type_source(path: str) -> bool:
    """Return True when a GFv2 polygon may influence generated tree type."""
    metadata = parse_gfv2_path(path)
    if metadata is None:
        return False
    searchable = f"{metadata['family']} {metadata['path']}"
    return not any(token in searchable for token in GFV2_TYPE_EXCLUDE_TOKENS)


def is_tree_gfv2_type_source(path: str) -> bool:
    """Return True when a GFv2 polygon may drive generated TREE asset type.

    Cropland windbreak rows routinely outnumber woodland polygons in
    agricultural tiles; letting them win the tile vote turned every generated
    forest into near-empty cropland assets, so tree polygons only accept
    tree-like families as type sources.
    """
    if not is_acceptable_gfv2_type_source(path):
        return False
    metadata = parse_gfv2_path(path)
    return metadata is not None and metadata["family"].lower() != "cropland"


def _role_for_gfv2_family(family: str) -> str:
    family = (family or "").lower()
    if family == "cropland":
        return "cropland"
    if family == "woodland":
        return "woodland"
    return "tree"


def gfv2_type_hint_candidates(
    source_path: str,
    fallback_region: str,
    dlevel: int,
) -> tuple[str, ...]:
    """Map a nearby acceptable GFv2 source path to approved generated assets.

    The source path fixes the region and family (species look) so generated
    forests blend with the surrounding GFv2 coverage, while the polygon's own
    measured density level is kept (snapped to the closest allowed level for
    the role) — inheriting the source polygon's density flattened every
    forest in a tile to one density regardless of actual canopy fill.
    """
    metadata = parse_gfv2_path(source_path)
    if metadata is None or not is_acceptable_gfv2_type_source(source_path):
        return gfv2_tree_candidates(fallback_region, dlevel)

    region = metadata["region"]
    role = _role_for_gfv2_family(metadata["family"])
    allowed_dlevels = GFV2_ROLE_DLEVELS.get(role, ())
    use_dlevel = (
        dlevel if dlevel in allowed_dlevels
        else min(allowed_dlevels, key=lambda d: abs(d - int(dlevel)))
        if allowed_dlevels else dlevel
    )
    try:
        return short_gfv2_candidates(region, role, use_dlevel)
    except (KeyError, ValueError):
        return gfv2_tree_candidates(fallback_region, dlevel)


def all_approved_generated_forest_paths() -> tuple[str, ...]:
    """Return the complete fixed set of future-emittable forest asset paths."""
    approved = set()
    for paths in DEFAULT_SHORT_TREE_BY_REGION.values():
        approved.update(paths)
    for region in GFV2_REGIONS:
        for role, dlevels in GFV2_ROLE_DLEVELS.items():
            for dlevel in dlevels:
                approved.update(short_gfv2_candidates(region, role, dlevel))
    return tuple(sorted(approved))


APPROVED_GENERATED_FOREST_PATHS = all_approved_generated_forest_paths()


def is_approved_generated_forest_path(path: str) -> bool:
    """Return True when a path is allowed for newly generated overlays."""
    return path in APPROVED_GENERATED_FOREST_PATHS


def choose_path(candidates, rng=None, weights=None) -> str:
    """Choose one path deterministically when given a reproducible RNG."""
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("No forest asset candidates available")
    if len(candidates) == 1 or rng is None:
        return candidates[0]
    if weights is not None:
        weights = tuple(int(w) for w in weights)
        if len(weights) != len(candidates):
            raise ValueError("Weights must match candidate count")
        total = sum(max(0, w) for w in weights)
        if total <= 0:
            raise ValueError("Weights must contain at least one positive value")
        if hasattr(rng, "integers"):
            pick = int(rng.integers(0, total))
        elif hasattr(rng, "randrange"):
            pick = rng.randrange(total)
        else:
            pick = 0
        upto = 0
        for candidate, weight in zip(candidates, weights):
            weight = max(0, weight)
            upto += weight
            if pick < upto:
                return candidate
        return candidates[-1]
    if hasattr(rng, "integers"):
        return candidates[int(rng.integers(0, len(candidates)))]
    if hasattr(rng, "randrange"):
        return candidates[rng.randrange(len(candidates))]
    if hasattr(rng, "choice"):
        return rng.choice(candidates)
    return candidates[0]
