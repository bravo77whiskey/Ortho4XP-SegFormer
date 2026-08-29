"""Regional building-height priors used to regularize HeightNet output.

The profiles are derived from a stratified sample of Microsoft Global
Building Atlas height estimates. Each class tuple contains the regional
median, P95, and a conservative hard ceiling. The ceiling is the sampled P99
plus the published GBA height RMSE, rounded up to the next five metres.

These priors are intentionally compact runtime statistics. The reproducible
research inputs and methodology live under ``experiments/height_priors``.
"""

from __future__ import annotations

from dataclasses import dataclass


POLICY_VERSION = 1
P95_EXCESS_RETAIN = 0.5


@dataclass(frozen=True)
class HeightPrior:
    """Height distribution summary for one region and footprint class."""

    region: str
    class_id: int
    median_m: float
    p95_m: float
    hard_ceiling_m: float


# Class order is C1 through C8. Values are rounded to millimetre precision,
# which is finer than the source model's useful accuracy.
_PROFILE_VALUES = {
    "north_america": ((3.506, 7.908, 20.0), (4.269, 8.422, 20.0), (5.021, 8.440, 20.0), (5.809, 8.665, 20.0), (5.649, 9.238, 20.0), (6.335, 11.847, 25.0), (7.029, 17.813, 50.0), (7.086, 17.575, 35.0)),
    "north_america_ne": ((4.404, 14.824, 30.0), (4.929, 15.142, 30.0), (5.253, 15.416, 30.0), (5.243, 13.164, 30.0), (5.365, 13.350, 30.0), (6.099, 15.489, 30.0), (6.188, 16.060, 30.0), (7.642, 23.307, 40.0)),
    "north_america_west": ((3.110, 8.667, 20.0), (4.345, 9.512, 20.0), (4.577, 9.043, 20.0), (5.021, 9.561, 20.0), (5.565, 11.070, 25.0), (6.639, 15.430, 40.0), (7.032, 18.968, 50.0), (7.957, 20.563, 50.0)),
    "europe": ((2.960, 6.093, 15.0), (3.903, 6.917, 15.0), (4.273, 7.964, 20.0), (4.611, 8.592, 20.0), (4.682, 8.079, 15.0), (5.025, 8.825, 20.0), (5.462, 10.012, 20.0), (6.730, 12.895, 20.0)),
    "scandinavia": ((3.498, 9.481, 20.0), (4.529, 8.873, 20.0), (4.762, 8.698, 20.0), (5.366, 10.608, 25.0), (5.663, 10.817, 20.0), (6.364, 11.872, 20.0), (7.242, 13.631, 25.0), (8.058, 16.761, 35.0)),
    "mediterranean": ((2.697, 5.643, 15.0), (3.333, 6.172, 15.0), (4.181, 7.362, 15.0), (4.879, 8.383, 15.0), (4.815, 8.477, 20.0), (5.266, 9.189, 20.0), (5.432, 9.936, 20.0), (5.693, 10.719, 20.0)),
    "asia": ((2.766, 4.931, 15.0), (3.510, 6.110, 15.0), (3.926, 7.855, 20.0), (4.510, 10.591, 30.0), (5.307, 27.039, 50.0), (6.261, 26.728, 45.0), (7.352, 29.655, 50.0), (8.335, 22.758, 40.0)),
    "se_asia": ((4.213, 7.364, 20.0), (5.000, 8.011, 20.0), (5.397, 8.434, 20.0), (5.724, 9.114, 20.0), (6.355, 10.918, 20.0), (7.025, 11.314, 20.0), (7.729, 14.482, 30.0), (8.841, 16.625, 35.0)),
    "africa": ((2.288, 5.176, 20.0), (2.822, 5.929, 20.0), (3.042, 6.155, 20.0), (3.243, 6.668, 20.0), (3.259, 6.975, 20.0), (3.850, 8.836, 20.0), (4.193, 8.928, 20.0), (5.190, 6.590, 20.0)),
    "australia_oceania": ((3.341, 8.996, 20.0), (4.049, 7.738, 15.0), (4.258, 6.812, 15.0), (4.422, 7.034, 15.0), (4.610, 7.481, 15.0), (5.270, 9.355, 20.0), (5.356, 10.209, 20.0), (5.465, 16.898, 30.0)),
    "south_america": ((2.933, 7.983, 25.0), (3.200, 8.206, 25.0), (3.731, 9.465, 25.0), (4.053, 8.856, 25.0), (4.314, 9.942, 25.0), (4.533, 16.229, 40.0), (6.519, 10.397, 25.0), (6.091, 12.825, 25.0)),
    "generic": ((3.141, 7.593, 25.0), (4.127, 8.836, 25.0), (4.609, 9.079, 25.0), (5.183, 9.013, 25.0), (5.231, 10.684, 30.0), (6.037, 14.428, 40.0), (6.688, 19.138, 45.0), (7.718, 18.878, 40.0)),
}


def normalize_region(region: object) -> str:
    """Return a known runtime region, falling back to the global profile."""

    normalized = str(region or "generic").strip().lower()
    return normalized if normalized in _PROFILE_VALUES else "generic"


def height_prior(region: object, class_id: object) -> HeightPrior:
    """Return the regional prior for a C1-C8 footprint class."""

    normalized_region = normalize_region(region)
    try:
        normalized_class = int(class_id)
    except (TypeError, ValueError):
        normalized_class = 4
    if normalized_class < 1 or normalized_class > 8:
        normalized_class = 4
    median_m, p95_m, hard_ceiling_m = _PROFILE_VALUES[normalized_region][
        normalized_class - 1
    ]
    return HeightPrior(
        region=normalized_region,
        class_id=normalized_class,
        median_m=median_m,
        p95_m=p95_m,
        hard_ceiling_m=hard_ceiling_m,
    )


def regularize_height_m(height_m: float, prior: HeightPrior) -> tuple[float, str]:
    """Shrink P95 excess and apply the regional hard ceiling."""

    height_m = float(height_m)
    adjustment = "none"
    if height_m > prior.p95_m:
        height_m = prior.p95_m + (
            height_m - prior.p95_m
        ) * P95_EXCESS_RETAIN
        adjustment = "soft_p95"
    if height_m > prior.hard_ceiling_m:
        height_m = prior.hard_ceiling_m
        adjustment = "hard_ceiling"
    return height_m, adjustment
