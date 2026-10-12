"""Switch the decal of already-built tiles without rebuilding them.

    python set_terrain_decal.py <tile dir> [<tile dir> ...] [--decal o4xp|xplane]

Rewrites the ``DECAL_LIB`` line of every ``terrain/*.ter`` that has one
(tiles built with use_decal_on_terrain).  Terrain files without a decal are
left alone; nothing else in the tile is touched.
"""

import argparse
import os

# Same table as O4_DSF_Utils.TERRAIN_DECALS (importing that module loads
# the providers and prints errors when run from outside the repo root).
TERRAIN_DECALS = {
    "o4xp": "o4xp/decals/ortho_detail.dcl",
    "xplane": "lib/g10/decals/maquify_2_green_key.dcl",
}


def retarget(tile_dir, decal_path):
    """Return (changed, with_decal) .ter counts for one tile."""
    terrain = os.path.join(tile_dir, "terrain")
    changed = with_decal = 0
    for name in os.listdir(terrain):
        if not name.endswith(".ter"):
            continue
        path = os.path.join(terrain, name)
        try:
            with open(path, encoding="ascii") as f:
                lines = f.read().split("\n")
        except (OSError, UnicodeDecodeError) as e:
            print(f"  skipped unreadable {path}: {e}")
            continue
        hits = [i for i, ln in enumerate(lines) if ln.startswith("DECAL_LIB ")]
        if not hits:
            continue
        with_decal += 1
        new = "DECAL_LIB " + decal_path
        if all(lines[i] == new for i in hits):
            continue
        for i in hits:
            lines[i] = new
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="ascii", newline="\n") as f:
            f.write("\n".join(lines))
        os.replace(tmp, path)
        changed += 1
    return changed, with_decal


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tiles", nargs="+", help="tile directories (zOrtho4XP_+NN+EEE)")
    ap.add_argument("--decal", default="o4xp", choices=sorted(TERRAIN_DECALS))
    args = ap.parse_args()
    for tile in args.tiles:
        changed, total = retarget(tile, TERRAIN_DECALS[args.decal])
        print(f"{tile}: {changed} of {total} decal .ter files switched to {args.decal}")


if __name__ == "__main__":
    main()
