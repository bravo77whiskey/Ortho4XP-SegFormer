"""simHeaven forest-asset policy for vegetation overlays.

New overlays use stable X-World Vegetation Library paths. The Global Forests
v2 parser remains here solely so migration tools can recognize old DSFs.
"""

from __future__ import annotations

import re

import O4_SFR_Climate_Regions as CLIMATE

HEIGHT_CUTOFF_METERS = 24.0

# X-World regionalizes these stable virtual paths through climate bitmaps and
# provides season-specific physical forests behind each alias.
DEFAULT_SHORT_TREE_BY_REGION = {
    "tropical": ("simheaven/forests/broad.for",),
    "subtropical": ("simheaven/forests/broad.for",),
    "northsouth": (
        "simheaven/forests/broad.for",
        "simheaven/forests/mixed.for",
    ),
    "northmiddle": (
        "simheaven/forests/broad.for",
        "simheaven/forests/coni.for",
        "simheaven/forests/mixed.for",
    ),
    "northnorth": (
        "simheaven/forests/broad.for",
        "simheaven/forests/coni.for",
        "simheaven/forests/mixed.for",
    ),
}

KNOWN_TALL_DEFAULT_FOREST_PATHS = (
    "lib/vegetation/forests/broadleaves/hot.for",
    "lib/vegetation/forests/broadleaves/warm.for",
    "lib/vegetation/forests/broadleaves/temperate.for",
    "lib/vegetation/forests/conifers/cold.for",
    "lib/vegetation/forests/mixed/temperate.for",
)

MESH_DEFS_BY_PATH = {
    "simheaven/forests/broad.for": 45,
    "simheaven/forests/coni.for": 46,
    "simheaven/forests/mixed.for": 49,
}

SIMHEAVEN_TREE_ASSET_PATHS = frozenset(MESH_DEFS_BY_PATH)

GFV2_PATH_RE = re.compile(
    r"^forests/(?P<region>[^/]+)/(?P<family>[^/]+)/"
    r"(?P=region)_(?P=family)_(?P<dlevel>\d+)_y(?P<variant>\d+)\.for$",
    re.IGNORECASE,
)
def climate_region(lat: float, lon: float | None = None) -> str:
    """Return the climate bucket used by the vegetation pipeline."""
    if lon is not None:
        return CLIMATE.forest_region_for_latlon(lat, lon)
    return CLIMATE.latitude_band_region(lat + 0.5)


def default_short_tree_candidates_for_region(region: str) -> tuple[str, ...]:
    """Return the approved simHeaven tree assets for a climate region."""
    try:
        return DEFAULT_SHORT_TREE_BY_REGION[region]
    except KeyError as exc:
        raise ValueError(f"Unsupported climate region: {region!r}") from exc


def default_short_tree_candidates_for_lat(lat: float) -> tuple[str, ...]:
    """Return the approved simHeaven tree assets for a latitude."""
    return default_short_tree_candidates_for_region(climate_region(lat))


def default_short_tree_for_lat(lat: float) -> str:
    """Return the canonical simHeaven tree path for a latitude."""
    return default_short_tree_candidates_for_lat(lat)[0]


def short_tree_candidates(region: str, dlevel: int) -> tuple[str, ...]:
    """Return simHeaven candidates for compatibility with older callers."""
    del dlevel
    return default_short_tree_candidates_for_region(region)


def mesh_defs_for_path(path: str) -> int:
    """Return the measured mesh-definition count for an allowed asset."""
    return MESH_DEFS_BY_PATH.get(path, 99)


def tree_candidate_weights(candidates) -> tuple[int, ...]:
    """Bias selection toward lower-mesh simHeaven assets."""
    return tuple(max(1, 60 - mesh_defs_for_path(path)) for path in candidates)


def choose_tree_path(region: str, dlevel: int, rng=None, context: str = "bulk") -> str:
    """Choose a stable simHeaven forest path."""
    del dlevel, context
    candidates = default_short_tree_candidates_for_region(region)
    return choose_path(candidates, rng, weights=tree_candidate_weights(candidates))


def _normalize_legacy_gfv2_path(path: str) -> str:
    normalized = (path or "").replace("\\", "/").strip().lower().lstrip("/")
    bundled_marker = "veg_overlay_short_forests/gfv2/"
    if normalized.startswith(bundled_marker):
        normalized = normalized[len(bundled_marker) :]
    return normalized


def parse_gfv2_path(path: str) -> dict | None:
    """Return metadata for a legacy Global Forests v2 asset path."""
    normalized = _normalize_legacy_gfv2_path(path)
    match = GFV2_PATH_RE.match(normalized)
    if not match:
        return None
    return {
        "region": match.group("region").lower(),
        "family": match.group("family").lower(),
        "dlevel": int(match.group("dlevel")),
        "variant": int(match.group("variant")),
        "path": normalized,
    }


def normalize_simheaven_tree_path(path: str | None) -> str | None:
    """Return a canonical selectable X-World forest path, when recognized."""
    normalized = (path or "").replace("\\", "/").strip().lower().lstrip("/")
    if normalized in SIMHEAVEN_TREE_ASSET_PATHS:
        return normalized
    return None


def simheaven_type_hint_candidates(
    source_path: str,
    fallback_region: str,
    dlevel: int,
) -> tuple[str, ...]:
    """Use a nearby X-World tree family or the climate-based fallback."""
    del dlevel
    normalized = normalize_simheaven_tree_path(source_path)
    if normalized is not None:
        return (normalized,)
    return default_short_tree_candidates_for_region(fallback_region)


def all_approved_generated_forest_paths() -> tuple[str, ...]:
    """Return every forest path that new overlays may emit."""
    return tuple(
        sorted(
            {
                path
                for paths in DEFAULT_SHORT_TREE_BY_REGION.values()
                for path in paths
            }
        )
    )


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
        weights = tuple(int(weight) for weight in weights)
        if len(weights) != len(candidates):
            raise ValueError("Weights must match candidate count")
        total = sum(max(0, weight) for weight in weights)
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
            upto += max(0, weight)
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
