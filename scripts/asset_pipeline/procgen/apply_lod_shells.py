"""Patch banded LOD shells into generated OBJ8s in place.

Band 1 (0..d0): the full Blender-exported model.
Band 2 (d0..d1): a silhouette shell rebuilt here in pure Python from the
archetype's own metadata (exact ridge height + the same atlas strips) with
PER-FLOOR wall quads, so window rows keep their band-1 scale at the swap.
Band 3 (d1..d2): a flat-top box whose walls use the windowless plain_*
strip (at that range facades read as wall color, and a stretched window
strip would paint one giant window row). Band 4 (d2..d3): a roof-colored
quad; beyond d3 the object culls. No Blender run needed -- this rewrites
OBJ8 text directly, and already-banded files are re-banded in place.

Run:  python apply_lod_shells.py --output <pkg> [--bands 1200 3500 7000 14000]
      python apply_lod_shells.py --output <pkg> --strip     # remove bands
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from archetypes import build_archetype  # noqa: E402
from add_lod import strip_lod  # noqa: E402
from lod_shell import merge_bands_into_obj8, roof_quad_mesh, shell_mesh  # noqa: E402

# Band boundaries in metres: full mesh, silhouette shell, flat-top box,
# roof-colored quad, then culled. Flat-roofed archetypes skip the box stage
# (their shell already is a flat-top box) and span shell -> quad directly.
#
# Bands are BASE values for a ~30 m-diagonal building and scale with the
# 3D bounding diagonal (footprint + height, see _band_scale): the horizon
# frame cost is dominated by the tens of thousands of small residentials,
# which are subpixel long before the old fixed 25 km cull.  A 10 m house
# still culls ~6 km out, a 110 m warehouse or a 12-floor tower reaches
# further.
DEFAULT_BANDS = (1200, 3500, 7000, 14000)

_BAND_SCALE_REF_DIAG_M = 30.0
_BAND_SCALE_MIN = 0.45
_BAND_SCALE_MAX = 1.9


def _band_scale(length_m: float, width_m: float,
                height_m: float = 0.0) -> float:
    # 3D diagonal: a slim 12-floor tower is as visible as a wide warehouse,
    # so height extends the bands the same way footprint does.
    diag = math.hypot(math.hypot(float(length_m), float(width_m)),
                      float(height_m))
    return min(_BAND_SCALE_MAX,
               max(_BAND_SCALE_MIN, diag / _BAND_SCALE_REF_DIAG_M))


def _strip_band(layout: dict, name: str):
    row = layout["strips"][name]
    return (float(row["v0"]), float(row["v1"]), float(row["world_w_m"]))


def _plain_strip_band(layout: dict, wall_strip: str):
    """Windowless plain_* strip matching a wall strip's material family."""
    parts = wall_strip.split("_")
    if len(parts) >= 2:
        plain = f"plain_{parts[1]}"
        if plain in layout["strips"]:
            return _strip_band(layout, plain)
    return _strip_band(layout, wall_strip)


def patch_shell(obj_path: str, archetype: str, length_m: float,
                width_m: float, floors: int, seed: int, flavor: str,
                layout: dict, bands=DEFAULT_BANDS) -> str:
    spec = build_archetype(archetype, length_m, width_m, floors, seed,
                           layout, flavor)
    meta = spec.meta
    if not meta:
        return "no-meta"
    scale = _band_scale(length_m, width_m, floors * 3.2)
    d0, d1, d2, d3 = (int(round(b * scale)) for b in bands)
    wall_band = _strip_band(layout, meta["wall_strip"])
    plain_band = _plain_strip_band(layout, meta["wall_strip"])
    roof_band = _strip_band(layout, meta["roof_strip"])

    # Band 2: silhouette shell with per-floor windowed walls; band 3: flat
    # box with windowless plain walls; band 4: roof quad.  Pitched and flat
    # archetypes share the structure (the flat band-2 shell already is a
    # box, but its walls carry real window rows unlike band 3's).
    lod_bands = []
    if meta["kind"] == "pitched":
        lod_bands.append((*shell_mesh(
            length_m, width_m, floors, "pitched",
            meta["ridge_z"], meta["eave_z"], wall_band, roof_band,
            wall_mode="floors",
        ), d0, d1))
        box_top = meta["eave_z"]
        quad_z = meta["eave_z"]
    else:
        lod_bands.append((*shell_mesh(
            length_m, width_m, floors, "flat",
            meta["ridge_z"], meta["eave_z"], wall_band, roof_band,
            wall_mode="floors",
        ), d0, d1))
        box_top = meta["ridge_z"]
        quad_z = meta["ridge_z"]
    lod_bands.append((*shell_mesh(
        length_m, width_m, floors, "flat",
        box_top, meta["eave_z"], plain_band, roof_band,
        wall_mode="stretch",
    ), d1, d2))
    lod_bands.append((*roof_quad_mesh(length_m, width_m, quad_z, roof_band),
                      d2, d3))
    return merge_bands_into_obj8(obj_path, lod_bands)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default=os.path.join(HERE, "procgen_manifest.json")
    )
    parser.add_argument(
        "--layout", default=os.path.join(HERE, "atlas_layout.json")
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--bands", type=int, nargs=4,
                        default=list(DEFAULT_BANDS),
                        metavar=("FULL", "SHELL", "BOX", "QUAD"),
                        help="Band boundaries in metres: full mesh end, "
                             "shell end, flat-box end, roof-quad end (cull).")
    parser.add_argument("--strip", action="store_true",
                        help="Remove LOD bands instead of adding them "
                             "(NOTE: band-2 shell geometry stays in the "
                             "file but is never drawn; regenerate for a "
                             "byte-clean library).")
    args = parser.parse_args(argv)

    with open(args.manifest, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    with open(args.layout, "r", encoding="utf-8") as fh:
        layout = json.load(fh)

    stats = {}
    seen = set()
    for asset in manifest["assets"]:
        if not asset.get("enabled"):
            continue
        for variant in asset["variants"]:
            physical = variant["physical_path"]
            if physical in seen:
                continue
            seen.add(physical)
            obj_path = os.path.join(args.output, physical)
            if not os.path.isfile(obj_path):
                stats["missing"] = stats.get("missing", 0) + 1
                continue
            if args.strip:
                status = strip_lod(obj_path)
            else:
                status = patch_shell(
                    obj_path, variant["archetype"],
                    asset["length_m"], asset["width_m"], asset["floors"],
                    variant["seed"], asset["flavor"],
                    layout, tuple(args.bands),
                )
            stats[status] = stats.get(status, 0) + 1

    print("LOD shells: " + "  ".join(
        f"{key}={value}" for key, value in sorted(stats.items())
    ))
    failures = sum(
        value for key, value in stats.items()
        if key in ("no-tris", "no-meta", "unsupported")
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
