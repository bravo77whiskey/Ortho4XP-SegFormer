"""Tile-wide vegetation replacement: inventory, embedded-tree cleanup, restore.

The generated vegetation overlay carries a ``sim/exclude_for`` rectangle over
the whole degree square (see O4_Vegetation_Overlay), which suppresses every
*forest polygon* in lower-priority scenery. It does nothing about trees that a
package draws some other way: standalone tree objects, trees attached to road
networks, trees baked into autogen blocks and strings. This module removes
those, for the selected tiles only.

How the removal works
---------------------
Only one kind of edit is ever made, at two levels:

    a reference to a *proven* tree object is repointed at an empty object

  * in the tile's source DSF, by substituting one entry of the DEFN/OBJT table
  * inside a referencing text asset (.net, .ags, .agb, .agp, .str, .lin), by
    substituting the path on the line that declares it, in a private copy of
    that asset which only the processed tile's DSF points at

Nothing is ever deleted, reordered or reindexed, so buildings, facades, ground
surfaces, roads, lights and every unrelated object survive by construction, and
no coordinate is requantized. An asset whose trees cannot be isolated this way
is left exactly as it is and named in the report.

"Proven" is the whole game, and ``classify_object`` is deliberately strict: a
filename containing "tree" is never enough on its own to change a mixed model.

Autogen format: https://developer.x-plane.com/article/x-plane-10-autogen-scenery-file-format/
DSF usage:      https://developer.x-plane.com/article/dsf-usage-in-x-plane/
"""

import json
import os
import re
import shutil
import time
import uuid

import O4_DSF_Edit as DSF
from O4_Scenery_Packs import (
    active_scenery_pack_dirs,
    resolve_custom_scenery_dir,
)

#: Bumped whenever a transformation would produce different output from the
#: same source, so private assets from an older run are never reused silently.
TRANSFORM_VERSION = 1

#: Where the managed state lives, relative to the Ortho4XP data directory.
MANAGED_DIR_NAME = "yOrtho4XP_Veg_Replacement"

#: The overlay package whose DSFs carry the generated trees and the exclusion.
OVERLAY_PACK_NAME = "yOrtho4XP_Veg_Overlays"

#: Namespaced folder added inside a modified package for its private assets.
#: Nothing shipped by that package is touched; this folder only ever gains
#: files, and removing it restores the package to its original content.
PRIVATE_DIR_NAME = "o4xp_veg_replacement"

EXCLUSION_PROPERTY = "sim/exclude_for"

#: Text assets whose object references this module knows how to repoint.
REFERENCING_ASSET_SUFFIXES = (".net", ".ags", ".agb", ".agp", ".str", ".lin")

#: Path components that signal vegetation. Matched against whole, normalised
#: path components and texture filename stems — never as a bare substring.
_VEGETATION_TOKENS = frozenset(
    (
        "tree", "trees", "forest", "forests", "vegetation", "veg",
        "bush", "bushes", "shrub", "shrubs", "foliage", "hedge", "hedges",
        "canopy", "woodland", "woods",
    )
)

#: An object doing any of this is not a tree, whatever it is called. Trees do
#: not carry collision surfaces, drape onto terrain, animate, emit light or
#: expose a cockpit or manipulator.
_DISQUALIFYING_OBJ_COMMANDS = frozenset(
    (
        "ATTR_hard", "ATTR_hard_deck", "ATTR_draped", "ATTR_no_draped",
        "ANIM_begin", "ATTR_cockpit", "ATTR_cockpit_region",
        "ATTR_solid_camera", "EMITTER", "SMOKE_BLACK", "SMOKE_WHITE",
        "LIGHT_NAMED", "LIGHT_CUSTOM", "LIGHT_PARAM", "LIGHT_SPILL_CUSTOM",
        "LIGHTS", "MAGNET", "PARTICLE_SYSTEM",
    )
)

_TEXTURE_COMMANDS = ("TEXTURE", "TEXTURE_LIT", "TEXTURE_NORMAL", "TEXTURE_DRAPED")

_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def tile_name(lat, lon):
    """Return the ``+47+008`` style name of the tile containing ``lat, lon``."""
    import math

    lat_i = int(math.floor(float(lat)))
    lon_i = int(math.floor(float(lon)))
    return (
        f"{'+' if lat_i >= 0 else '-'}{abs(lat_i):02d}"
        f"{'+' if lon_i >= 0 else '-'}{abs(lon_i):03d}"
    )


def tile_dsf_relpath(lat, lon):
    """Return ``Earth nav data/+40+000/+47+008.dsf`` for a tile."""
    import math

    lat_i = int(math.floor(float(lat)))
    lon_i = int(math.floor(float(lon)))
    lat_g = int(math.floor(lat_i / 10)) * 10
    lon_g = int(math.floor(lon_i / 10)) * 10
    block = (
        f"{'+' if lat_g >= 0 else '-'}{abs(lat_g):02d}"
        f"{'+' if lon_g >= 0 else '-'}{abs(lon_g):03d}"
    )
    return os.path.join("Earth nav data", block, f"{tile_name(lat, lon)}.dsf")


def _tokens(text):
    return {token for token in _TOKEN_SPLIT_RE.split((text or "").lower()) if token}


def _path_components(path):
    return [part for part in re.split(r"[\\/]+", (path or "")) if part]


def _has_vegetation_component(path):
    """True when a *directory* component of ``path`` names vegetation."""
    components = _path_components(path)
    for component in components[:-1]:
        if _tokens(component) & _VEGETATION_TOKENS:
            return True
    return False


def _looks_like_vegetation_texture(path):
    """True when a texture path names vegetation in a folder or its stem."""
    if _has_vegetation_component(path):
        return True
    components = _path_components(path)
    if not components:
        return False
    stem = os.path.splitext(components[-1])[0]
    return bool(_tokens(stem) & _VEGETATION_TOKENS)


def _real_resolve(path):
    """Resolve junctions and symlinks so a source is identified by real file."""
    return os.path.realpath(os.path.abspath(path))


# ─────────────────────────────────────────────────────────────────────────────
# Library resolution
# ─────────────────────────────────────────────────────────────────────────────

class LibraryIndex:
    """Virtual library paths exported by the enabled scenery packages.

    Packs are consulted in ``scenery_packs.ini`` order, which is X-Plane's own
    priority order, so the first export of a virtual path wins. Regional
    exports (``EXPORT_EXCLUDE``/``EXPORT_BACKUP``/``EXPORT_RATIO``) and the
    alternative variants of a path are all recorded, because an inventory that
    saw only the first variant would under-report what a tile actually draws.
    """

    _EXPORT_RE = re.compile(
        r"^(EXPORT|EXPORT_EXCLUDE|EXPORT_BACKUP|EXPORT_RATIO|EXPORT_EXTEND)\s+(.*)$"
    )

    def __init__(self):
        self._exports = {}

    @classmethod
    def from_packs(cls, packs):
        """Build the index from ``[(folder_name, package_dir), ...]``."""
        index = cls()
        for _folder_name, package_dir in packs:
            index.add_package(package_dir)
        return index

    def add_package(self, package_dir):
        library = os.path.join(package_dir, "library.txt")
        if not os.path.isfile(library):
            return
        try:
            with open(library, "r", encoding="utf-8", errors="ignore") as handle:
                lines = handle.readlines()
        except OSError:
            return
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            match = self._EXPORT_RE.match(line)
            if not match:
                continue
            command, rest = match.group(1), match.group(2).strip()
            parts = rest.split()
            if command == "EXPORT_RATIO":
                parts = parts[1:]  # drop the leading ratio
            if len(parts) < 2:
                continue
            virtual_path = parts[0].replace("\\", "/")
            real_relative = " ".join(parts[1:]).replace("\\", "/")
            real_path = os.path.join(package_dir, *real_relative.split("/"))
            self._exports.setdefault(virtual_path.lower(), []).append(real_path)

    def variants(self, virtual_path):
        """Return every real file exported under ``virtual_path``."""
        return list(self._exports.get((virtual_path or "").replace("\\", "/").lower(), ()))

    def resolve(self, reference, package_dir=None, relative_to=None):
        """Resolve a DSF or asset reference to a real file on disk.

        X-Plane looks a reference up in the library first, then relative to the
        package (for a DSF reference) or to the referencing file. All three are
        tried, and the first that exists wins.
        """
        reference = (reference or "").replace("\\", "/").strip()
        if not reference:
            return None
        for candidate in self.variants(reference):
            if os.path.isfile(candidate):
                return candidate
        bases = []
        if relative_to:
            bases.append(os.path.dirname(relative_to))
        if package_dir:
            bases.append(package_dir)
        for base in bases:
            candidate = os.path.join(base, *reference.split("/"))
            if os.path.isfile(candidate):
                return candidate
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Tree-object classification
# ─────────────────────────────────────────────────────────────────────────────

class Verdict:
    """Why one asset reference was or was not treated as a tree."""

    __slots__ = ("is_tree", "evidence", "reason")

    def __init__(self, is_tree, evidence, reason):
        self.is_tree = is_tree
        self.evidence = list(evidence)
        self.reason = reason

    def as_dict(self):
        return {"is_tree": self.is_tree, "evidence": self.evidence, "reason": self.reason}


def _read_obj_commands(obj_path, max_bytes=8 * 1024 * 1024):
    """Return ``(commands, textures)`` from an OBJ8 file, or None if unreadable.

    Only the command name of each line is kept, plus the texture paths; the
    vertex and index tables are the bulk of the file and say nothing about
    whether this is a tree.
    """
    try:
        if os.path.getsize(obj_path) > max_bytes:
            return None
        with open(obj_path, "r", encoding="utf-8", errors="ignore") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return None
    commands = []
    textures = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        command = parts[0]
        commands.append(command)
        if command in _TEXTURE_COMMANDS and len(parts) > 1:
            textures.append(parts[1].strip())
    return commands, textures


def classify_object(reference, real_path, rules=None):
    """Decide whether one object reference may be treated as a removable tree.

    The bar is deliberately high, because the same call decides whether a
    third-party model keeps rendering. An object qualifies only when it does
    nothing a tree would not do, *and* all of its artwork is vegetation
    artwork, *and* it is either filed under a vegetation folder or built like a
    billboard (two-sided, no culling). A name is never evidence on its own.

    ``rules`` may carry user-confirmed decisions as
    ``{"tree": [...], "not_tree": [...]}``; ``not_tree`` always wins.
    """
    reference_key = (reference or "").replace("\\", "/").lower()
    rules = rules or {}
    if reference_key in {item.lower() for item in rules.get("not_tree", ())}:
        return Verdict(False, ["rule:not_tree"], "excluded by asset rules")
    if reference_key in {item.lower() for item in rules.get("tree", ())}:
        return Verdict(True, ["rule:tree"], "confirmed by asset rules")

    if not real_path or not os.path.isfile(real_path):
        return Verdict(False, [], "object file not found on disk")

    parsed = _read_obj_commands(real_path)
    if parsed is None:
        return Verdict(False, [], "object file unreadable or too large to inspect")
    commands, textures = parsed
    if not commands:
        return Verdict(False, [], "object file has no commands")

    disqualifying = sorted(set(commands) & _DISQUALIFYING_OBJ_COMMANDS)
    if disqualifying:
        return Verdict(
            False,
            [f"disqualifying:{name}" for name in disqualifying],
            "object does more than draw vegetation "
            f"({', '.join(disqualifying)})",
        )

    evidence = []
    if not textures:
        return Verdict(False, [], "object references no texture to judge")
    if all(_looks_like_vegetation_texture(texture) for texture in textures):
        evidence.append("textures:vegetation")
    else:
        return Verdict(
            False,
            [],
            "at least one texture is not vegetation artwork: "
            + ", ".join(sorted(textures)),
        )

    if _has_vegetation_component(reference) or _has_vegetation_component(real_path):
        evidence.append("path:vegetation-folder")
    if "ATTR_no_cull" in commands:
        evidence.append("geometry:two-sided")

    if len(evidence) < 2:
        return Verdict(
            False,
            evidence,
            "only one independent signal; a vegetation texture alone is not "
            "enough to alter a third-party model",
        )
    return Verdict(True, evidence, "vegetation artwork plus " + evidence[-1])


# ─────────────────────────────────────────────────────────────────────────────
# Private assets
# ─────────────────────────────────────────────────────────────────────────────

_EMPTY_OBJECT_SOURCE = """A
800
OBJ

# Written by Ortho4XP tile vegetation replacement (transform v{version}).
# An empty object: the placement record in the DSF is preserved exactly, so
# nothing is reindexed, but nothing is drawn where the tree used to be.

POINT_COUNTS 0 0 0 0
"""


def empty_object_source():
    return _EMPTY_OBJECT_SOURCE.format(version=TRANSFORM_VERSION)


def private_asset_name(source_bytes, original_name):
    """Return the content-addressed filename of a private asset copy.

    Keyed on the source content and the transformation version, so two tiles
    needing the same modified asset from the same package share one file, and a
    package update or a changed transformation produces a new one instead of
    quietly reusing the old.
    """
    digest = DSF.sha256_bytes(
        f"v{TRANSFORM_VERSION}\0".encode("utf-8") + source_bytes
    )
    stem, suffix = os.path.splitext(os.path.basename(original_name))
    return f"{stem}_vegfree_{digest[:16]}{suffix}"


def repoint_text_asset(text, mapping):
    """Repoint object references inside a text asset, preserving every line.

    ``mapping`` maps a reference (as written in the file, case-insensitively)
    to its replacement. Only the matched token is substituted; the keyword, the
    surrounding parameters, the spacing and the line order are untouched, so
    index-based references inside the format keep working.

    Returns ``(new_text, replacements)``.
    """
    lookup = {key.replace("\\", "/").lower(): value for key, value in mapping.items()}
    replacements = []
    out_lines = []
    for raw_line in text.splitlines(keepends=True):
        stripped = raw_line.rstrip("\r\n")
        newline = raw_line[len(stripped):]
        # Split keeping the separators, so a whole token is swapped and the
        # original spacing survives. A plain str.replace would also rewrite a
        # token that merely starts with the one being replaced.
        parts = re.split(r"(\s+)", stripped)
        for index, part in enumerate(parts):
            key = part.replace("\\", "/").lower()
            if key in lookup:
                replacements.append((part, lookup[key]))
                parts[index] = lookup[key]
        out_lines.append("".join(parts) + newline)
    return "".join(out_lines), replacements


# ─────────────────────────────────────────────────────────────────────────────
# Scenery order
# ─────────────────────────────────────────────────────────────────────────────

def _is_airport_pack(package_dir, max_dsfs=8):
    """True when a package is a custom airport rather than area scenery."""
    earth_nav_data = os.path.join(package_dir, "Earth nav data")
    if not os.path.isfile(os.path.join(earth_nav_data, "apt.dat")):
        return False
    seen = 0
    for _root, _dirs, files in os.walk(earth_nav_data):
        for name in files:
            if name.lower().endswith(".dsf"):
                seen += 1
                if seen > max_dsfs:
                    return False
    return True


def validate_scenery_order(custom_scenery_dir, lat, lon,
                           overlay_pack_name=OVERLAY_PACK_NAME):
    """Report scenery-order problems for one tile; never rearrange anything.

    A tile-wide exclusion only reaches packages *below* the overlay, and must
    not reach an airport the user wants to keep. This checks that the installed
    order actually gives that result and returns what is wrong, leaving
    ``scenery_packs.ini`` to the user.
    """
    conflicts = []
    packs = active_scenery_pack_dirs(custom_scenery_dir)
    if not packs:
        return [{
            "severity": "error",
            "message": "no enabled scenery packages found; is scenery_packs.ini "
                       f"present in {resolve_custom_scenery_dir(custom_scenery_dir)!r}?",
        }]

    overlay_index = None
    for index, (folder_name, _package_dir) in enumerate(packs):
        if folder_name.lower() == overlay_pack_name.lower():
            overlay_index = index
            break
    if overlay_index is None:
        conflicts.append({
            "severity": "error",
            "message": f"{overlay_pack_name} is not an enabled scenery package; "
                       "the generated overlay and its exclusion will not load",
        })

    relpath = tile_dsf_relpath(lat, lon)
    for index, (folder_name, package_dir) in enumerate(packs):
        if index == overlay_index:
            continue
        if not os.path.isfile(os.path.join(package_dir, relpath)):
            continue
        airport = _is_airport_pack(package_dir)
        if overlay_index is None:
            continue
        if airport and index > overlay_index:
            conflicts.append({
                "severity": "warning",
                "package": folder_name,
                "message": f"airport package {folder_name} sits BELOW "
                           f"{overlay_pack_name}; its authored forests will be "
                           "excluded too. Move it above the overlay to keep it.",
            })
        elif not airport and index < overlay_index:
            conflicts.append({
                "severity": "warning",
                "package": folder_name,
                "message": f"regional package {folder_name} sits ABOVE "
                           f"{overlay_pack_name}; its forests will NOT be "
                           "replaced. Move the overlay above it.",
            })
    return conflicts


# ─────────────────────────────────────────────────────────────────────────────
# Inventory
# ─────────────────────────────────────────────────────────────────────────────

#: How each finding will be treated. Mirrors the planning table in the design.
TREATMENTS = {
    "forest_polygon": "suppressed by the generated forest exclusion",
    "tree_object": "repointed to an empty object in this tile's DSF",
    "embedded_tree_reference": "repointed inside a private copy of the asset",
    "airport": "out of scope: higher-priority airport scenery",
    "unsupported": "preserved; reported",
}


def _classify_polygon_definition(path):
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".for":
        return "forest_polygon"
    if suffix in REFERENCING_ASSET_SUFFIXES:
        return "referencing_asset"
    return "other"


def inventory_tile(custom_scenery_dir, lat, lon, rules=None,
                   overlay_pack_name=OVERLAY_PACK_NAME):
    """Inspect every enabled package that draws this tile, before anything moves.

    Returns a dict with one entry per source DSF: what it defines, which of
    those definitions are removable trees and why, and what is being left alone.
    Nothing is written and no source file is modified.
    """
    custom_scenery_dir = resolve_custom_scenery_dir(custom_scenery_dir)
    packs = active_scenery_pack_dirs(custom_scenery_dir)
    library = LibraryIndex.from_packs(packs)
    relpath = tile_dsf_relpath(lat, lon)

    overlay_index = next(
        (index for index, (name, _) in enumerate(packs)
         if name.lower() == overlay_pack_name.lower()),
        None,
    )

    report = {
        "tile": tile_name(lat, lon),
        "custom_scenery_dir": custom_scenery_dir,
        "transform_version": TRANSFORM_VERSION,
        "sources": [],
        "skipped": [],
        "errors": [],
    }

    seen_real = set()
    for index, (folder_name, package_dir) in enumerate(packs):
        if index == overlay_index:
            continue
        dsf_path = os.path.join(package_dir, relpath)
        if not os.path.isfile(dsf_path):
            continue
        real_path = _real_resolve(dsf_path)
        if real_path in seen_real:
            continue
        seen_real.add(real_path)
        if overlay_index is not None and index < overlay_index:
            report["skipped"].append({
                "package": folder_name, "path": dsf_path,
                "reason": "higher priority than the overlay; the exclusion "
                          "does not reach it",
            })
            continue
        if _is_airport_pack(package_dir):
            report["skipped"].append({
                "package": folder_name, "path": dsf_path,
                "reason": TREATMENTS["airport"],
            })
            continue
        try:
            report["sources"].append(
                _inventory_source(
                    folder_name, package_dir, dsf_path, real_path, library, rules
                )
            )
        except DSF.DsfError as exc:
            report["errors"].append({
                "package": folder_name, "path": dsf_path, "error": str(exc),
            })
    return report


def _inventory_source(folder_name, package_dir, dsf_path, real_path, library, rules):
    with open(real_path, "rb") as handle:
        data = handle.read()
    dsf = DSF.DsfFile(data)
    usage = DSF.count_definition_usage(data)

    entry = {
        "package": folder_name,
        "package_dir": package_dir,
        "path": dsf_path,
        "resolved": real_path,
        "size": len(data),
        "sha256": DSF.sha256_bytes(data),
        "forests": [],
        "tree_objects": [],
        "referencing_assets": [],
        "unsupported": [],
    }

    for index, reference in enumerate(dsf.definitions("object")):
        placements = usage["object"].get(index, 0)
        resolved = library.resolve(reference, package_dir=package_dir)
        verdict = classify_object(reference, resolved, rules)
        record = {
            "index": index, "reference": reference, "resolved": resolved,
            "placements": placements, **verdict.as_dict(),
        }
        if verdict.is_tree:
            entry["tree_objects"].append(record)
        elif _tokens(os.path.basename(reference)) & _VEGETATION_TOKENS:
            # Named like vegetation but not proven: exactly the case the
            # report exists for, so the user can confirm it in asset rules.
            record["treatment"] = TREATMENTS["unsupported"]
            entry["unsupported"].append(record)

    for index, reference in enumerate(dsf.definitions("polygon")):
        placements = usage["polygon"].get(index, 0)
        kind = _classify_polygon_definition(reference)
        if kind == "forest_polygon":
            entry["forests"].append({
                "index": index, "reference": reference,
                "placements": placements,
                "treatment": TREATMENTS["forest_polygon"],
            })
        elif kind == "referencing_asset":
            entry["referencing_assets"].append(
                _inventory_referencing_asset(
                    "polygon", index, reference, placements,
                    package_dir, library, rules
                )
            )

    for index, reference in enumerate(dsf.definitions("network")):
        placements = usage["network"].get(index, 0)
        entry["referencing_assets"].append(
            _inventory_referencing_asset(
                "network", index, reference, placements,
                package_dir, library, rules
            )
        )
    return entry


def _inventory_referencing_asset(kind, index, reference, placements,
                                 package_dir, library, rules):
    """Look inside one text asset for object references that are trees."""
    record = {
        "kind": kind, "index": index, "reference": reference,
        "placements": placements, "resolved": None,
        "tree_references": [], "supported": False, "reason": "",
    }
    resolved = library.resolve(reference, package_dir=package_dir)
    record["resolved"] = resolved
    if not resolved:
        record["reason"] = "asset not found on disk"
        return record
    try:
        with open(resolved, "r", encoding="utf-8", errors="ignore") as handle:
            text = handle.read()
    except OSError as exc:
        record["reason"] = f"asset unreadable ({exc})"
        return record
    if "\0" in text[:4096]:
        record["reason"] = "asset is not a text file"
        return record

    candidates = set()
    for token in re.findall(r"[^\s\"']+\.obj\b", text, flags=re.IGNORECASE):
        candidates.add(token)
    if not candidates:
        record["reason"] = "asset places no objects"
        return record

    for candidate in sorted(candidates):
        target = library.resolve(
            candidate, package_dir=package_dir, relative_to=resolved
        )
        verdict = classify_object(candidate, target, rules)
        if verdict.is_tree:
            record["tree_references"].append({
                "reference": candidate, "resolved": target, **verdict.as_dict(),
            })
    if record["tree_references"]:
        record["supported"] = True
        record["reason"] = TREATMENTS["embedded_tree_reference"]
    else:
        record["reason"] = "no proven tree object referenced"
    return record


# ─────────────────────────────────────────────────────────────────────────────
# Managed state: verified backups, journal, per-tile manifests
# ─────────────────────────────────────────────────────────────────────────────

class ConflictError(RuntimeError):
    """An installed file is not what the recorded state says it should be."""


class ManagedStore:
    """Everything this workflow must remember, on disk and verifiable.

    Backups are content-addressed by the SHA-256 of the original bytes, so a
    repeated run over the same source version reuses one verified copy, and a
    scenery update that changes a source is simply a second version with its
    own backup rather than an overwrite of the first.
    """

    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.backups_dir = os.path.join(self.root, "backups")
        self.tiles_dir = os.path.join(self.root, "tiles")
        self.journal_path = os.path.join(self.root, "journal.jsonl")
        self.pending_path = os.path.join(self.root, "pending.json")
        self.index_path = os.path.join(self.backups_dir, "index.jsonl")
        self.rules_path = os.path.join(self.root, "asset_rules.json")

    # ── layout ───────────────────────────────────────────────────────────
    def ensure_dirs(self):
        os.makedirs(self.backups_dir, exist_ok=True)
        os.makedirs(self.tiles_dir, exist_ok=True)

    def backup_path(self, digest):
        return os.path.join(self.backups_dir, digest[:2], digest + ".bin")

    def rules(self):
        if not os.path.isfile(self.rules_path):
            return {}
        try:
            with open(self.rules_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return {}

    # ── backups ──────────────────────────────────────────────────────────
    def has_backup(self, digest):
        path = self.backup_path(digest)
        return os.path.isfile(path) and DSF.sha256_file(path) == digest

    def ensure_backup(self, source_path, digest, metadata):
        """Store and verify the original bytes of one source, then return its path.

        Called before every destructive change. An existing backup is re-hashed
        rather than trusted, and one that does not verify is a hard error: it is
        the only route back, and overwriting it with the current (possibly
        already modified) file would destroy the recovery path.
        """
        path = self.backup_path(digest)
        if os.path.isfile(path):
            existing = DSF.sha256_file(path)
            if existing != digest:
                raise ConflictError(
                    "existing backup " + path + " hashes to " + existing
                    + ", not " + digest + "; refusing to touch it"
                )
            return path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(source_path, "rb") as handle:
            data = handle.read()
        if DSF.sha256_bytes(data) != digest:
            raise ConflictError(source_path + " changed while being backed up")
        DSF.atomic_write(path, data)
        if DSF.sha256_file(path) != digest:
            raise ConflictError("backup verification failed for " + path)
        self._append(
            self.index_path,
            dict(metadata, sha256=digest, backup=path, recorded=time.time()),
        )
        return path

    def restore_from_backup(self, digest, target_path):
        path = self.backup_path(digest)
        if not os.path.isfile(path):
            raise ConflictError("no backup stored for " + digest)
        if DSF.sha256_file(path) != digest:
            raise ConflictError("backup " + path + " failed verification")
        with open(path, "rb") as handle:
            data = handle.read()
        DSF.atomic_write(target_path, data)
        if DSF.sha256_file(target_path) != digest:
            raise ConflictError("restore verification failed for " + target_path)

    # ── journal ──────────────────────────────────────────────────────────
    @staticmethod
    def _append(path, row):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def pending(self):
        if not os.path.isfile(self.pending_path):
            return None
        try:
            with open(self.pending_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError) as exc:
            raise ConflictError(
                self.pending_path + " is unreadable (" + str(exc) + "); an "
                "interrupted operation cannot be resolved automatically"
            )

    def begin(self, operation):
        """Record an operation as in flight before its first file changes.

        Atomic replacement protects one file. This is what protects the run: if
        the process dies between two DSFs, the next run finds this file and
        knows which tile was left half-done.
        """
        self.ensure_dirs()
        existing = self.pending()
        if existing and existing.get("op_id") != operation.get("op_id"):
            raise ConflictError(
                "operation " + str(existing.get("op_id")) + " on tile "
                + str(existing.get("tile")) + " was interrupted and is still "
                "open; resume or restore it first"
            )
        # Resuming the same operation keeps what the interrupted run already
        # completed; starting over from an empty list would lose the record of
        # files it had finished with.
        completed = list(operation.get("completed") or ())
        if existing:
            completed = list(existing.get("completed") or ()) + completed
        payload = dict(
            operation,
            started=(existing or {}).get("started")
            or operation.get("started", time.time()),
            completed=completed,
        )
        DSF.atomic_write(
            self.pending_path,
            json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
        )
        self._append(self.journal_path, dict(
            {key: payload[key] for key in ("op_id", "action", "tile", "started")
             if key in payload},
            event="begin",
        ))
        return payload

    def record(self, operation, entry):
        """Mark one completed change, in the journal and in the pending file."""
        operation.setdefault("completed", []).append(entry)
        DSF.atomic_write(
            self.pending_path,
            json.dumps(operation, indent=2, sort_keys=True).encode("utf-8"),
        )
        self._append(self.journal_path, dict(
            entry, event="change", op_id=operation.get("op_id")
        ))

    def fail(self, operation):
        """Journal a failed operation and leave its pending file in place."""
        self._append(self.journal_path, {
            "event": "failed",
            "op_id": operation.get("op_id"),
            "tile": operation.get("tile"),
            "action": operation.get("action"),
            "finished": time.time(),
            "changes": len(operation.get("completed", [])),
        })

    def end(self, operation, outcome="completed"):
        self._append(self.journal_path, {
            "event": "end",
            "op_id": operation.get("op_id"),
            "tile": operation.get("tile"),
            "action": operation.get("action"),
            "outcome": outcome,
            "finished": time.time(),
            "changes": len(operation.get("completed", [])),
        })
        if os.path.isfile(self.pending_path):
            os.remove(self.pending_path)

    # ── per-tile manifests ───────────────────────────────────────────────
    def tile_manifest_path(self, tile):
        return os.path.join(self.tiles_dir, tile + ".json")

    def load_tile(self, tile):
        path = self.tile_manifest_path(tile)
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def save_tile(self, tile, manifest):
        DSF.atomic_write(
            self.tile_manifest_path(tile),
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
        )

    def delete_tile(self, tile):
        path = self.tile_manifest_path(tile)
        if os.path.isfile(path):
            os.remove(path)

    def tiles(self):
        if not os.path.isdir(self.tiles_dir):
            return []
        return sorted(
            os.path.splitext(name)[0]
            for name in os.listdir(self.tiles_dir)
            if name.endswith(".json")
        )

    def private_asset_users(self, asset_path, excluding_tile=None):
        """Return the tiles whose installed replacement still needs an asset."""
        users = []
        target = _real_resolve(asset_path)
        for tile in self.tiles():
            if excluding_tile and tile == excluding_tile:
                continue
            manifest = self.load_tile(tile) or {}
            for asset in manifest.get("private_assets", ()):
                if _real_resolve(asset.get("path", "")) == target:
                    users.append(tile)
                    break
        return users


def adopt_backups_from_strip_output(store, output_dir):
    """Reuse verified originals left by the older simHeaven cleanup run.

    That tool recorded both the original and the stripped hash of every file it
    touched. Only a file whose bytes hash to a recorded *original* is adopted,
    so an already stripped DSF can never be taken for an untouched one.
    """
    manifest_path = os.path.join(output_dir, "manifest.jsonl")
    originals_dir = os.path.join(output_dir, "originals")
    adopted, skipped = [], []
    if not os.path.isfile(manifest_path):
        return {"adopted": adopted, "skipped": skipped,
                "error": "no manifest.jsonl in " + output_dir}
    with open(manifest_path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    store.ensure_dirs()
    for row in rows:
        candidate = os.path.join(originals_dir, row["path"])
        if not os.path.isfile(candidate):
            skipped.append({"path": row["path"], "reason": "no stored original"})
            continue
        digest = DSF.sha256_file(candidate)
        if digest == row.get("cleaned_sha256"):
            skipped.append({"path": row["path"],
                            "reason": "stored file is the stripped version"})
            continue
        if digest != row.get("original_sha256"):
            skipped.append({"path": row["path"],
                            "reason": "hash matches neither recorded version"})
            continue
        if store.has_backup(digest):
            adopted.append({"path": row["path"], "sha256": digest,
                            "reused": True})
            continue
        store.ensure_backup(candidate, digest, {
            "source": row["path"], "origin": "strip_simheaven_vegetation",
            "output_dir": os.path.abspath(output_dir),
        })
        adopted.append({"path": row["path"], "sha256": digest, "reused": False})
    return {"adopted": adopted, "skipped": skipped}


# ─────────────────────────────────────────────────────────────────────────────
# Planning
# ─────────────────────────────────────────────────────────────────────────────

def _package_relative(package_dir, path):
    """Return ``path`` as a package-relative reference, or None if outside it."""
    package_real = _real_resolve(package_dir)
    target_real = _real_resolve(path)
    try:
        relative = os.path.relpath(target_real, package_real)
    except ValueError:  # different drives on Windows
        return None
    if relative.startswith(".."):
        return None
    return relative.replace(os.sep, "/")


def plan_tile(inventory, store):
    """Turn an inventory into the exact set of changes, with a storage estimate.

    Nothing is written. The result lists, per source DSF, which definition
    entries get repointed and which private assets have to exist first, plus
    everything that will be left alone and why.
    """
    plan = {
        "tile": inventory["tile"],
        "transform_version": TRANSFORM_VERSION,
        "sources": [],
        "private_assets": {},
        "unsupported": [],
        "forests_excluded": 0,
        "trees_removed": 0,
    }
    empty_source = empty_object_source().encode("utf-8")

    for source in inventory.get("sources", []):
        package_dir = source["package_dir"]
        changes = {"object": {}}
        assets = []
        removed = 0

        empty_path = os.path.join(package_dir, PRIVATE_DIR_NAME,
                                  "empty_v%d.obj" % TRANSFORM_VERSION)
        empty_reference = _package_relative(package_dir, empty_path)

        for record in source["tree_objects"]:
            changes["object"][record["index"]] = empty_reference
            removed += record["placements"]
            plan["private_assets"][empty_path] = {
                "path": empty_path, "bytes": len(empty_source),
                "kind": "empty_object", "sha256": DSF.sha256_bytes(empty_source),
            }

        for record in source["referencing_assets"]:
            if not record["supported"]:
                # Silence is only right when the asset was understood and simply
                # holds no trees; anything we could not read or resolve has to
                # be named, because its trees will still be there.
                if record["reason"] not in (
                    "no proven tree object referenced", "asset places no objects"
                ):
                    plan["unsupported"].append(dict(record, package=source["package"]))
                continue
            built = _plan_private_copy(record, package_dir, empty_path, empty_source)
            if built is None:
                plan["unsupported"].append(dict(
                    record, package=source["package"],
                    reason="its private copy could not be referenced from this "
                           "tile alone without changing behaviour elsewhere",
                ))
                continue
            kind = record["kind"]
            changes.setdefault(kind, {})[record["index"]] = built["reference"]
            assets.append(built["asset"])
            plan["private_assets"][built["asset"]["path"]] = built["asset"]
            plan["private_assets"][empty_path] = {
                "path": empty_path, "bytes": len(empty_source),
                "kind": "empty_object", "sha256": DSF.sha256_bytes(empty_source),
            }
            removed += record["placements"]

        plan["forests_excluded"] += sum(
            record["placements"] for record in source["forests"]
        )
        plan["unsupported"].extend(
            dict(record, package=source["package"])
            for record in source["unsupported"]
        )
        if not any(changes.get(kind) for kind in changes):
            continue
        plan["trees_removed"] += removed
        plan["sources"].append({
            "tile": inventory["tile"],
            "package": source["package"],
            "package_dir": package_dir,
            "path": source["path"],
            "resolved": source["resolved"],
            "sha256": source["sha256"],
            "size": source["size"],
            "changes": {kind: value for kind, value in changes.items() if value},
            "private_assets": [asset["path"] for asset in assets],
            "placements_removed": removed,
        })

    plan["private_assets"] = list(plan["private_assets"].values())
    plan["neighbour_tiles"] = neighbour_tiles_sharing_assets(
        plan, *_tile_latlon(inventory["tile"])
    )
    plan["estimate"] = estimate_storage(plan, store)
    return plan


def _plan_private_copy(record, package_dir, empty_path, empty_source):
    """Build the vegetation-free private copy of one referencing text asset.

    The copy is written beside the original so every *relative* reference in it
    — textures, sibling assets — resolves exactly as it did before. That is
    only possible when the original lives inside the package whose DSF points
    at it; a library asset owned by another package cannot be redirected for
    one tile without affecting every other tile that uses it, so those are
    reported as unsupported instead.
    """
    original = record["resolved"]
    if _package_relative(package_dir, original) is None:
        return None
    try:
        with open(original, "r", encoding="utf-8", errors="ignore") as handle:
            text = handle.read()
    except OSError:
        return None

    asset_dir = os.path.dirname(original)
    empty_relative = os.path.relpath(empty_path, asset_dir).replace(os.sep, "/")
    mapping = {
        tree["reference"]: empty_relative for tree in record["tree_references"]
    }
    new_text, replacements = repoint_text_asset(text, mapping)
    if not replacements:
        return None
    data = new_text.encode("utf-8")
    name = private_asset_name(text.encode("utf-8"), original)
    private_path = os.path.join(asset_dir, name)
    reference = _package_relative(package_dir, private_path)
    if reference is None:
        return None
    return {
        "reference": reference,
        "asset": {
            "path": private_path,
            "bytes": len(data),
            "kind": "vegetation_free_copy",
            "sha256": DSF.sha256_bytes(data),
            "source": original,
            "source_sha256": DSF.sha256_bytes(text.encode("utf-8")),
            "transform_version": TRANSFORM_VERSION,
            "replacements": len(replacements),
            "content": new_text,
        },
    }


def neighbour_tiles_sharing_assets(plan, lat, lon):
    """Report neighbouring tiles that still draw an asset this tile repoints.

    A DSF reference is tile-local, but the object it places is not: a tree at
    the edge of this tile renders geometry that reaches into the next one, and
    the neighbour's own DSF keeps pointing at the original asset until that
    tile is processed too. That shows up as a seam, so it is worth naming
    rather than discovering in the air.
    """
    findings = []
    for source in plan.get("sources", []):
        package_dir = source["package_dir"]
        repointed = set()
        try:
            original = DSF.DsfFile.read(source["resolved"])
        except (DSF.DsfError, OSError):
            continue
        for kind, entries in source["changes"].items():
            paths = original.definitions(kind)
            for index in entries:
                position = int(index)
                if position < len(paths):
                    repointed.add((kind, paths[position]))
        if not repointed:
            continue
        for delta_lat in (-1, 0, 1):
            for delta_lon in (-1, 0, 1):
                if delta_lat == 0 and delta_lon == 0:
                    continue
                neighbour_lat = int(lat) + delta_lat
                neighbour_lon = int(lon) + delta_lon
                neighbour = os.path.join(
                    package_dir, tile_dsf_relpath(neighbour_lat, neighbour_lon)
                )
                if not os.path.isfile(neighbour):
                    continue
                try:
                    other = DSF.DsfFile.read(neighbour)
                except (DSF.DsfError, OSError):
                    continue
                shared = sorted(
                    reference for kind, reference in repointed
                    if reference in other.definitions(kind)
                )
                if shared:
                    findings.append({
                        "tile": tile_name(neighbour_lat, neighbour_lon),
                        "package": source["package"],
                        "path": neighbour,
                        "references": shared,
                    })
    return findings


def estimate_storage(plan, store):
    """Estimate backup, private-asset and peak temporary space before any write.

    Sizes come from the actual files: a regional DSF carries terrain and raster
    data as well as the vegetation, so a per-placement guess would be far out.
    """
    backup_bytes = sum(
        source["size"] for source in plan["sources"]
        if not store.has_backup(source["sha256"])
    )
    asset_bytes = sum(
        asset["bytes"] for asset in plan["private_assets"]
        if not os.path.isfile(asset["path"])
    )
    peak_temp = max(
        [source["size"] for source in plan["sources"]]
        + [asset["bytes"] for asset in plan["private_assets"]]
        + [0]
    )
    volumes = {}
    for path in ([store.root]
                 + [source["resolved"] for source in plan["sources"]]
                 + [asset["path"] for asset in plan["private_assets"]]):
        anchor = path
        while anchor and not os.path.isdir(anchor):
            parent = os.path.dirname(anchor)
            if parent == anchor:
                break
            anchor = parent
        if not anchor or not os.path.isdir(anchor):
            continue
        key = os.path.splitdrive(os.path.abspath(anchor))[0] or os.sep
        volumes.setdefault(key, anchor)
    free = {}
    for key, anchor in volumes.items():
        try:
            free[key] = shutil.disk_usage(anchor).free
        except OSError:
            free[key] = None
    required = backup_bytes + asset_bytes + peak_temp
    return {
        "backup_bytes": backup_bytes,
        "private_asset_bytes": asset_bytes,
        "peak_temp_bytes": peak_temp,
        "required_bytes": required,
        "free_bytes": free,
        "sufficient": all(
            value is None or value > required + (1 << 30) for value in free.values()
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_transform(original_bytes, modified_bytes, changes):
    """Prove a rewritten DSF changed only the definition entries it was meant to.

    Compared atom by atom against the original rather than re-derived from the
    same code that produced the rewrite, so a bug in the writer shows up here
    instead of being confirmed by itself. Returns a problem string, or None.
    """
    try:
        before = DSF.DsfFile(original_bytes)
        after = DSF.DsfFile(modified_bytes)
    except DSF.DsfError as exc:
        return "modified file does not parse as a DSF (%s)" % exc

    before_atoms = before.atom_map()
    after_atoms = after.atom_map()
    if set(before_atoms) != set(after_atoms):
        return "the set of top-level atoms changed"
    for atom_id, payload in before_atoms.items():
        if atom_id == DSF.ATOM_DEFN:
            continue
        if after_atoms[atom_id] != payload:
            return "atom %r was modified but should not have been" % atom_id

    if before.properties() != after.properties():
        return "header properties were modified"

    for kind in DSF.DEFINITION_ATOMS:
        old_paths = before.definitions(kind)
        new_paths = after.definitions(kind)
        if len(old_paths) != len(new_paths):
            return "%s definition count changed" % kind
        expected = dict(changes.get(kind, {}))
        for index, (old_path, new_path) in enumerate(zip(old_paths, new_paths)):
            if index in expected:
                wanted = expected.pop(index)
            elif str(index) in expected:
                wanted = expected.pop(str(index))
            else:
                wanted = None
            if wanted is None:
                if old_path != new_path:
                    return "%s definition %d changed unexpectedly" % (kind, index)
            elif new_path != wanted:
                return "%s definition %d is %r, expected %r" % (
                    kind, index, new_path, wanted
                )
        if expected:
            return "%s definitions %s were not applied" % (
                kind, sorted(expected)
            )
    return None


def scenery_busy():
    """Return a reason not to install right now, or None.

    X-Plane holding a DSF open, or a generation run about to rewrite one, both
    turn an atomic replacement into a race. This is a best-effort check on
    Windows; when it cannot tell, it says nothing rather than blocking.
    """
    import subprocess
    import sys

    if not sys.platform.startswith("win"):
        return None
    command = (
        "Get-CimInstance Win32_Process | Where-Object { "
        "$_.Name -match '^(python|pythonw|X-Plane|Ortho4XP)\\.exe$' -and "
        "($_.Name -eq 'X-Plane.exe' -or $_.CommandLine -match "
        # O4_SFR_ matches overlay runs started by builds from before the rename.
        "'O4_(Vegetation|Building)_Overlay|O4_SFR_|generate_(veg|bld)_overlay') } | "
        "Select-Object -ExpandProperty ProcessId"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True, text=True, timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    running = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    # This process is one of them whenever the app itself drives the install.
    running = [pid for pid in running if pid != str(os.getpid())]
    if running:
        return ("X-Plane or a scenery-generation process is running (pid "
                + ", ".join(running) + "); stop it before installing")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Installing
# ─────────────────────────────────────────────────────────────────────────────

def apply_plan(plan, store, force_space=False, ignore_busy=False, log=print):
    """Install one tile's embedded-tree cleanup, one source DSF at a time.

    Private assets are written first — they only ever add files — and each
    source DSF is then backed up, rewritten to a temporary file, verified and
    swapped in. The journal records every completed change as it happens, so an
    interruption leaves a resumable, restorable state rather than a mystery.
    """
    busy = None if ignore_busy else scenery_busy()
    if busy:
        raise ConflictError(busy)

    estimate = plan.get("estimate") or estimate_storage(plan, store)
    if not estimate["sufficient"] and not force_space:
        raise ConflictError(
            "not enough free space: %d bytes needed (%d backup, %d private "
            "assets, %d peak temporary), free by volume %s"
            % (estimate["required_bytes"], estimate["backup_bytes"],
               estimate["private_asset_bytes"], estimate["peak_temp_bytes"],
               estimate["free_bytes"])
        )

    operation = store.begin({
        "op_id": uuid.uuid4().hex,
        "action": "install",
        "tile": plan["tile"],
        "transform_version": TRANSFORM_VERSION,
        "sources": [source["path"] for source in plan["sources"]],
    })

    manifest = {
        "tile": plan["tile"],
        "transform_version": TRANSFORM_VERSION,
        "installed": time.time(),
        "op_id": operation["op_id"],
        "sources": [],
        "private_assets": [],
        "unsupported": plan.get("unsupported", []),
        "trees_removed": plan.get("trees_removed", 0),
        "forests_excluded": plan.get("forests_excluded", 0),
    }

    try:
        for asset in plan["private_assets"]:
            written = _install_private_asset(asset, log)
            manifest["private_assets"].append(written)
            store.record(operation, {"kind": "private_asset", **written})

        for source in plan["sources"]:
            installed = _install_source(source, store, log)
            manifest["sources"].append(installed)
            store.record(operation, {"kind": "source_dsf", **installed})
    except Exception:
        # Deliberately NOT store.end(): the pending file is what tells the next
        # run that this tile was left half-done. It is removed only once the
        # operation has either finished or been restored.
        store.fail(operation)
        raise

    store.save_tile(plan["tile"], manifest)
    store.end(operation)
    log("[veg cleanup] tile %s: %d source DSFs rewritten, %d tree placements "
        "removed, %d private assets"
        % (plan["tile"], len(manifest["sources"]), manifest["trees_removed"],
           len(manifest["private_assets"])))
    return manifest


def _install_private_asset(asset, log):
    """Write one private asset, or confirm the shared copy already matches."""
    path = asset["path"]
    if asset["kind"] == "empty_object":
        data = empty_object_source().encode("utf-8")
    else:
        data = asset["content"].encode("utf-8")
    digest = DSF.sha256_bytes(data)
    if digest != asset["sha256"]:
        raise ConflictError("private asset %s no longer matches its plan" % path)
    if os.path.isfile(path):
        existing = DSF.sha256_file(path)
        if existing != digest:
            raise ConflictError(
                "private asset %s exists with different content (%s); its name "
                "is content-addressed, so this should not happen" % (path, existing)
            )
        log("[veg cleanup] reusing private asset %s" % path)
    else:
        DSF.atomic_write(path, data)
        if DSF.sha256_file(path) != digest:
            raise ConflictError("private asset verification failed for %s" % path)
        log("[veg cleanup] wrote private asset %s" % path)
    return {
        "path": path, "sha256": digest, "kind": asset["kind"],
        "source": asset.get("source"),
        "source_sha256": asset.get("source_sha256"),
        "transform_version": TRANSFORM_VERSION,
    }


def _install_source(source, store, log):
    """Back up, rewrite and swap in one source DSF."""
    target = source["resolved"]
    with open(target, "rb") as handle:
        original = handle.read()
    digest = DSF.sha256_bytes(original)
    if digest != source["sha256"]:
        raise ConflictError(
            "%s changed since it was inventoried (now %s, planned against %s)"
            % (target, digest, source["sha256"])
        )

    store.ensure_backup(target, digest, {
        "source": source["path"], "resolved": target,
        "tile": source.get("tile"), "package": source["package"],
        "transform": "repoint_tree_definitions_v%d" % TRANSFORM_VERSION,
    })

    dsf = DSF.DsfFile(original)
    for kind, entries in source["changes"].items():
        paths = list(dsf.definitions(kind))
        for index, replacement in entries.items():
            paths[int(index)] = replacement
        dsf.set_definitions(kind, paths)
    modified = dsf.to_bytes()

    problem = verify_transform(original, modified, source["changes"])
    if problem:
        raise ConflictError("refusing to install %s: %s" % (target, problem))

    # Recheck immediately before the swap: everything above took time.
    if DSF.sha256_file(target) != digest:
        raise ConflictError("%s changed just before replacement" % target)
    DSF.atomic_write(target, modified)
    modified_digest = DSF.sha256_bytes(modified)
    if DSF.sha256_file(target) != modified_digest:
        raise ConflictError("installed verification failed for %s" % target)
    log("[veg cleanup] rewrote %s (%d tree placements)"
        % (target, source.get("placements_removed", 0)))
    return {
        "path": source["path"],
        "resolved": target,
        "package": source["package"],
        "original_sha256": digest,
        "modified_sha256": modified_digest,
        "changes": source["changes"],
        "placements_removed": source.get("placements_removed", 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# The managed overlay's exclusion property
# ─────────────────────────────────────────────────────────────────────────────

def overlay_exclusion_value(lat, lon):
    import math

    lat_i = int(math.floor(float(lat)))
    lon_i = int(math.floor(float(lon)))
    return "%d/%d/%d/%d" % (lon_i, lat_i, lon_i + 1, lat_i + 1)


def check_overlay_suitable(overlay_dsf, lat, lon):
    """Return why an existing overlay cannot carry a tile-wide exclusion, or None.

    Excluding every forest below an overlay that has no trees of its own would
    strip the tile bare, so the overlay has to be a real generated result for
    this tile before its exclusion is turned on.
    """
    if not os.path.isfile(overlay_dsf):
        return "no generated vegetation overlay at %s" % overlay_dsf
    try:
        dsf = DSF.DsfFile.read(overlay_dsf)
    except DSF.DsfError as exc:
        return "%s is not a readable DSF (%s)" % (overlay_dsf, exc)
    properties = dict(dsf.properties())
    if properties.get("sim/overlay") != "1":
        return "%s is not an overlay DSF" % overlay_dsf
    import math

    lat_i = int(math.floor(float(lat)))
    lon_i = int(math.floor(float(lon)))
    bounds = {
        "sim/west": lon_i, "sim/east": lon_i + 1,
        "sim/south": lat_i, "sim/north": lat_i + 1,
    }
    for key, expected in bounds.items():
        try:
            actual = int(float(properties.get(key, "nan")))
        except ValueError:
            return "%s has no usable %s property" % (overlay_dsf, key)
        if actual != expected:
            return "%s covers %s=%s, not tile %s" % (
                overlay_dsf, key, actual, tile_name(lat, lon)
            )
    forests = [
        path for path in dsf.definitions("polygon")
        if path.lower().endswith(".for")
    ]
    if not forests:
        return ("%s defines no forests, so replacing the layers below it would "
                "leave the tile bare" % overlay_dsf)
    return None


def set_overlay_exclusion(overlay_dsf, lat, lon, enabled, store,
                          preview=False, log=print):
    """Add or remove the tile-wide forest exclusion on an existing overlay.

    Only the header property table is rewritten; the definitions, the command
    stream and the pools are copied through untouched, so an overlay does not
    have to be regenerated to join or leave the replacement workflow.
    """
    value = overlay_exclusion_value(lat, lon)
    if enabled:
        problem = check_overlay_suitable(overlay_dsf, lat, lon)
        if problem:
            raise ConflictError(problem)
    if not os.path.isfile(overlay_dsf):
        return {"changed": False, "reason": "no overlay to change"}

    with open(overlay_dsf, "rb") as handle:
        original = handle.read()
    dsf = DSF.DsfFile(original)
    properties = dsf.properties()
    existing = [item for item in properties if item[0] == EXCLUSION_PROPERTY]
    if enabled:
        updated = DSF.upsert_property(properties, EXCLUSION_PROPERTY, value)
    else:
        updated = DSF.remove_property(properties, EXCLUSION_PROPERTY)
    if updated == properties:
        return {"changed": False, "reason": "already in the wanted state",
                "exclusion": existing}
    if preview:
        return {"changed": False, "preview": True, "from": existing,
                "to": [item for item in updated if item[0] == EXCLUSION_PROPERTY]}

    dsf.set_properties(updated)
    modified = dsf.to_bytes()
    check = DSF.DsfFile(modified)
    if check.all_definitions() != DSF.DsfFile(original).all_definitions():
        raise ConflictError("%s: definitions changed during a header edit"
                            % overlay_dsf)
    before_atoms = DSF.DsfFile(original).atom_map()
    after_atoms = check.atom_map()
    for atom_id, payload in before_atoms.items():
        if atom_id == DSF.ATOM_HEAD:
            continue
        if after_atoms.get(atom_id) != payload:
            raise ConflictError("%s: atom %r changed during a header edit"
                                % (overlay_dsf, atom_id))
    got = [item for item in check.properties() if item[0] == EXCLUSION_PROPERTY]
    if enabled and got != [(EXCLUSION_PROPERTY, value)]:
        raise ConflictError("%s: exclusion not written as expected (%s)"
                            % (overlay_dsf, got))
    if not enabled and got:
        raise ConflictError("%s: exclusion still present" % overlay_dsf)

    digest = DSF.sha256_bytes(original)
    store.ensure_backup(overlay_dsf, digest, {
        "source": overlay_dsf, "resolved": _real_resolve(overlay_dsf),
        "tile": tile_name(lat, lon), "package": OVERLAY_PACK_NAME,
        "transform": "exclusion_%s_v%d" % (
            "on" if enabled else "off", TRANSFORM_VERSION
        ),
    })
    DSF.atomic_write(overlay_dsf, modified)
    if DSF.sha256_file(overlay_dsf) != DSF.sha256_bytes(modified):
        raise ConflictError("overlay verification failed for %s" % overlay_dsf)
    log("[veg cleanup] %s exclusion %s"
        % ("added" if enabled else "removed", overlay_dsf))
    return {
        "changed": True, "enabled": bool(enabled), "value": value if enabled else None,
        "original_sha256": digest, "modified_sha256": DSF.sha256_bytes(modified),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Restoring
# ─────────────────────────────────────────────────────────────────────────────

def restore_tile(store, tile, overlay_dsf=None, ignore_busy=False, log=print):
    """Undo both halves of one tile's replacement, refusing to overwrite surprises.

    Every installed file is hashed first. A file that matches the recorded
    original is already restored; one that matches what was installed is
    restored from its verified backup; anything else means another tool or a
    scenery update got there first, and that is reported instead of clobbered.
    """
    manifest = store.load_tile(tile)
    if manifest is None:
        return {"tile": tile, "restored": [], "conflicts": [],
                "reason": "no managed replacement recorded for this tile"}
    busy = None if ignore_busy else scenery_busy()
    if busy:
        raise ConflictError(busy)

    conflicts = []
    for source in manifest.get("sources", []):
        target = source["resolved"]
        if not os.path.isfile(target):
            conflicts.append({"path": target, "reason": "file is gone"})
            continue
        current = DSF.sha256_file(target)
        if current in (source["original_sha256"], source["modified_sha256"]):
            continue
        conflicts.append({
            "path": target,
            "reason": "installed file is neither the original nor what this "
                      "workflow installed (now %s)" % current,
        })
    if conflicts:
        return {"tile": tile, "restored": [], "conflicts": conflicts}

    operation = store.begin({
        "op_id": uuid.uuid4().hex, "action": "restore", "tile": tile,
        "sources": [source["path"] for source in manifest.get("sources", [])],
    })
    restored = []
    try:
        for source in manifest.get("sources", []):
            target = source["resolved"]
            if DSF.sha256_file(target) == source["original_sha256"]:
                continue
            store.restore_from_backup(source["original_sha256"], target)
            restored.append(target)
            store.record(operation, {"kind": "restore_dsf", "path": target})
            log("[veg cleanup] restored %s" % target)

        removed_assets = []
        for asset in manifest.get("private_assets", []):
            path = asset["path"]
            users = store.private_asset_users(path, excluding_tile=tile)
            if users:
                log("[veg cleanup] keeping %s; still used by %s"
                    % (path, ", ".join(users)))
                continue
            if os.path.isfile(path):
                if DSF.sha256_file(path) != asset["sha256"]:
                    log("[veg cleanup] leaving %s in place: content changed "
                        "since it was written" % path)
                    continue
                os.remove(path)
                removed_assets.append(path)
                store.record(operation, {"kind": "remove_private_asset",
                                         "path": path})
            parent = os.path.dirname(path)
            if os.path.basename(parent) == PRIVATE_DIR_NAME and os.path.isdir(parent):
                try:
                    os.rmdir(parent)
                except OSError:
                    pass

        overlay_result = None
        if overlay_dsf:
            overlay_result = set_overlay_exclusion(
                overlay_dsf, *_tile_latlon(tile), enabled=False,
                store=store, log=log
            )
            store.record(operation, {"kind": "overlay_exclusion_off",
                                     "path": overlay_dsf})
    except Exception:
        store.fail(operation)
        raise

    store.delete_tile(tile)
    store.end(operation)
    return {
        "tile": tile, "restored": restored, "conflicts": [],
        "removed_private_assets": removed_assets,
        "overlay": overlay_result,
    }


def _tile_latlon(tile):
    """Parse ``+47+008`` back into ``(47, 8)``."""
    match = re.match(r"^([+-]\d{2})([+-]\d{3})$", tile)
    if not match:
        raise ValueError("not a tile name: %r" % tile)
    return int(match.group(1)), int(match.group(2))


def reconcile_tile(store, tile, overlay_dsf=None, ignore_busy=False, log=print):
    """Undo a managed replacement that the current settings no longer ask for.

    Called when generation produced nothing for a tile, or when replacement was
    turned off: the checkbox alone must never leave an old exclusion or an old
    embedded-tree cleanup silently installed.
    """
    manifest = store.load_tile(tile)
    if manifest is None:
        if overlay_dsf and os.path.isfile(overlay_dsf):
            return {"tile": tile, "action": "overlay_only",
                    **set_overlay_exclusion(overlay_dsf, *_tile_latlon(tile),
                                            enabled=False, store=store, log=log)}
        return {"tile": tile, "action": "nothing_to_do"}
    log("[veg cleanup] tile %s still carries a managed replacement; undoing it"
        % tile)
    return restore_tile(store, tile, overlay_dsf=overlay_dsf,
                        ignore_busy=ignore_busy, log=log)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _human(size):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return "%.1f %s" % (size, unit)
        size /= 1024.0
    return "%.1f TiB" % size


def format_report(plan, inventory=None, conflicts=(), applied=None, store=None):
    """Render the completion (or preview) report as plain text."""
    lines = []
    lines.append("Tile %s — vegetation replacement %s"
                 % (plan["tile"], "applied" if applied else "preview"))
    lines.append("")
    if conflicts:
        lines.append("Scenery order:")
        for conflict in conflicts:
            lines.append("  [%s] %s" % (conflict["severity"], conflict["message"]))
        lines.append("")
    lines.append("Forest polygons suppressed by the tile exclusion: %d"
                 % plan.get("forests_excluded", 0))
    lines.append("Embedded tree placements removed: %d"
                 % plan.get("trees_removed", 0))
    lines.append("")
    if plan["sources"]:
        lines.append("Source DSFs %s:" % ("rewritten" if applied else "to rewrite"))
        for source in plan["sources"]:
            lines.append("  %s  (%s, %d placements)"
                         % (source["path"], source["package"],
                            source["placements_removed"]))
            for kind, entries in source["changes"].items():
                for index, replacement in sorted(entries.items(),
                                                 key=lambda item: int(item[0])):
                    lines.append("      %s[%s] -> %s" % (kind, index, replacement))
    else:
        lines.append("No source DSF needs an embedded-tree edit.")
    lines.append("")
    if plan["private_assets"]:
        lines.append("Private assets:")
        for asset in plan["private_assets"]:
            lines.append("  %s  (%s, %s)"
                         % (asset["path"], asset["kind"], _human(asset["bytes"])))
        lines.append("")
    if plan.get("unsupported"):
        lines.append("Preserved and unsupported (left visible):")
        for record in plan["unsupported"]:
            lines.append("  %s — %s"
                         % (record.get("reference", "?"),
                            record.get("reason") or record.get("treatment", "")))
        lines.append("")
    estimate = plan.get("estimate") or {}
    if estimate:
        lines.append("Storage: %s backups + %s private assets, peak temporary %s"
                     % (_human(estimate.get("backup_bytes", 0)),
                        _human(estimate.get("private_asset_bytes", 0)),
                        _human(estimate.get("peak_temp_bytes", 0))))
        lines.append("  free by volume: %s"
                     % ", ".join("%s %s" % (key, _human(value or 0))
                                 for key, value in
                                 sorted((estimate.get("free_bytes") or {}).items())))
        lines.append("")
    neighbours = plan.get("neighbour_tiles") or ()
    if neighbours:
        lines.append("Assets also drawn by neighbouring tiles (process those "
                     "tiles too, or expect a seam at the border):")
        for finding in neighbours:
            lines.append("  %s — %s" % (finding["tile"],
                                        ", ".join(finding["references"])))
        lines.append("")
    if inventory and inventory.get("skipped"):
        lines.append("Out of scope:")
        for skipped in inventory["skipped"]:
            lines.append("  %s — %s" % (skipped["package"], skipped["reason"]))
        lines.append("")
    if inventory and inventory.get("errors"):
        lines.append("Unreadable sources:")
        for error in inventory["errors"]:
            lines.append("  %s — %s" % (error["path"], error["error"]))
        lines.append("")
    if store is not None:
        lines.append("Backups: %s" % store.backups_dir)
        lines.append("Journal: %s" % store.journal_path)
        lines.append("Restore this tile:")
        lines.append("  python src/scripts/veg_tile_replacement.py restore "
                     "--tile %s" % plan["tile"])
    lines.append("")
    lines.append("This report describes file-level changes that were verified by "
                 "hash and by re-parsing each DSF. It does not state what the "
                 "tile looks like in X-Plane; fly it to check that.")
    return "\n".join(lines)
