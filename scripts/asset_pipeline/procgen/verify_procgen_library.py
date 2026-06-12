"""Validate a built procgen library against its manifest and pool contracts.

Checks per OBJ:
  * measured VT footprint bounds match manifest dims within --tol (default
    0.05 m) and are centered within 1 cm (mirrors _read_obj8_bounds /
    _required_centered_dimensions_for_bounds in O4_SFR_Building_Overlay);
  * triangle count within budget;
  * exactly one TEXTURE directive pointing at an existing atlas PNG.

Checks per virtual path (importing the REAL token tuples from
O4_SFR_Building_Overlay so this can never drift):
  * passes _is_optional_library_building_candidate;
  * not rejected as special landmark / very tall;
  * height parses to floors * 3.2.

Run:  python verify_procgen_library.py --output <pkg>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

TRI_BUDGET = 1500
CENTER_TOL_M = 0.01


def _read_obj8(path):
    xs, zs = [], []
    tris = 0
    texture_lines = []
    lod_lines = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            parts = raw.split()
            if not parts:
                continue
            if parts[0] == "VT" and len(parts) >= 4:
                xs.append(float(parts[1]))
                zs.append(float(parts[3]))
            elif parts[0] == "TRIS" and len(parts) >= 3:
                tris += int(parts[2]) // 3
            elif parts[0] == "TEXTURE":
                texture_lines.append(parts[1] if len(parts) > 1 else "")
            elif parts[0] == "ATTR_LOD":
                lod_lines.append(parts[1:])
    if not xs:
        return None
    return {
        "xmin": min(xs), "xmax": max(xs),
        "zmin": min(zs), "zmax": max(zs),
        "tris": tris,
        "textures": texture_lines,
        "lods": lod_lines,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default=os.path.join(HERE, "procgen_manifest.json")
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--tol", type=float, default=0.05)
    parser.add_argument("--missing-ok", action="store_true",
                        help="Only verify OBJs that exist (partial builds).")
    args = parser.parse_args(argv)

    import O4_SFR_Building_Overlay as overlay

    with open(args.manifest, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    errors = []
    checked_objs = 0
    checked_paths = 0
    for asset in manifest["assets"]:
        if not asset.get("enabled"):
            continue
        virtual = asset["virtual_path"]

        # --- virtual-path contract audit (real overlay predicates) ---
        path_problems = []
        if not overlay._is_optional_library_building_candidate(virtual):
            path_problems.append("fails include/exclude token filter")
        if overlay._path_has_any(
            virtual, overlay.OPTIONAL_LIBRARY_SPECIAL_LANDMARK_TOKENS
        ):
            path_problems.append("hits special-landmark token")
        height = overlay._object_estimated_height_m(virtual)
        expected_height = asset["floors"] * 3.2
        if height is None or abs(height - expected_height) > 0.01:
            path_problems.append(
                f"height parse {height} != floors*3.2 ({expected_height})"
            )
        if overlay._is_very_tall_building_asset(virtual, height):
            path_problems.append("rejected as very tall")
        regions = overlay._optional_library_asset_regions(virtual, "o4sfr")
        if not regions:
            path_problems.append("unclassified region")
        elif asset["region"] == "generic" and "generic" not in regions:
            path_problems.append(f"generic asset classified as {regions}")
        elif asset["region"] != "generic" and asset["region"] not in regions:
            path_problems.append(
                f"region {asset['region']} not in classified {regions}"
            )
        for problem in path_problems:
            errors.append(f"{virtual}: {problem}")
        checked_paths += 1

        # --- physical OBJ checks ---
        hx, hy = asset["length_m"] / 2.0, asset["width_m"] / 2.0
        for variant in asset["variants"]:
            obj_path = os.path.join(args.output, variant["physical_path"])
            if not os.path.isfile(obj_path):
                if not args.missing_ok:
                    errors.append(f"{variant['physical_path']}: missing OBJ")
                continue
            data = _read_obj8(obj_path)
            if data is None:
                errors.append(f"{variant['physical_path']}: no VT vertices")
                continue
            checked_objs += 1
            for got, want, label in (
                (data["xmin"], -hx, "xmin"), (data["xmax"], hx, "xmax"),
                (data["zmin"], -hy, "zmin"), (data["zmax"], hy, "zmax"),
            ):
                if abs(got - want) > args.tol:
                    errors.append(
                        f"{variant['physical_path']}: {label}={got:.3f} "
                        f"expected {want:.3f}"
                    )
            if abs(data["xmin"] + data["xmax"]) > CENTER_TOL_M or \
                    abs(data["zmin"] + data["zmax"]) > CENTER_TOL_M:
                errors.append(f"{variant['physical_path']}: not centered")
            if data["tris"] > TRI_BUDGET:
                errors.append(
                    f"{variant['physical_path']}: {data['tris']} tris "
                    f"over budget {TRI_BUDGET}"
                )
            if len(data["textures"]) != 1:
                errors.append(
                    f"{variant['physical_path']}: {len(data['textures'])} "
                    f"TEXTURE directives (want exactly 1)"
                )
            n_lods = len(data["lods"])
            if n_lods not in (0, 2, 3, 4):
                errors.append(
                    f"{variant['physical_path']}: {n_lods} ATTR_LOD "
                    f"directives (want 0 = X-Plane auto LOD, or 2-4 "
                    f"contiguous bands; the engine limit is 4)"
                )
            elif n_lods:
                if data["lods"][0][0] != "0":
                    errors.append(
                        f"{variant['physical_path']}: first LOD band must "
                        f"start at 0"
                    )
                for prev, cur in zip(data["lods"], data["lods"][1:]):
                    if prev[1] != cur[0]:
                        errors.append(
                            f"{variant['physical_path']}: LOD bands not "
                            f"contiguous: {prev} then {cur}"
                        )
                        break
            else:
                texture_abs = os.path.normpath(os.path.join(
                    os.path.dirname(obj_path), data["textures"][0]
                ))
                if not os.path.isfile(texture_abs):
                    errors.append(
                        f"{variant['physical_path']}: texture missing "
                        f"{data['textures'][0]}"
                    )

    print(f"verified {checked_paths} virtual paths, {checked_objs} OBJ files")
    if errors:
        for line in errors[:40]:
            print(f"  ! {line}")
        if len(errors) > 40:
            print(f"  ... and {len(errors) - 40} more")
        print(f"FAILED with {len(errors)} problems")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
