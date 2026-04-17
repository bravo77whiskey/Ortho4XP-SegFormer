"""Helpers for fast repeated bounding-box queries on prepared geometry."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def build_bounds_index(items: Sequence[object] | None):
    """Return a vectorized bbox index for items carrying ``_bounds`` metadata.

    Each item must expose ``_bounds`` as ``(south, north, west, east)``.
    Returns ``None`` for empty input so callers can keep simple fast-paths.
    """
    if not items:
        return None

    south = np.empty(len(items), dtype=np.float64)
    north = np.empty(len(items), dtype=np.float64)
    west = np.empty(len(items), dtype=np.float64)
    east = np.empty(len(items), dtype=np.float64)
    obj_items = np.empty(len(items), dtype=object)

    for idx, item in enumerate(items):
        bounds = item.get("_bounds") if isinstance(item, dict) else None
        if bounds is None:
            raise ValueError("bounds index items must provide '_bounds'")
        south[idx], north[idx], west[idx], east[idx] = bounds
        obj_items[idx] = item

    return {
        "items": obj_items,
        "south": south,
        "north": north,
        "west": west,
        "east": east,
    }


def query_bounds(index, south: float, north: float, west: float, east: float):
    """Return indexed items whose bounds intersect the query box."""
    if index is None:
        return []
    keep = (
        (index["north"] >= south) &
        (index["south"] <= north) &
        (index["east"] >= west) &
        (index["west"] <= east)
    )
    if not bool(np.any(keep)):
        return []
    return index["items"][keep].tolist()
