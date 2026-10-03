"""Tile-wide vegetation replacement: classification, install, verify, restore.

Everything here runs against a synthetic Custom Scenery tree built in a temp
folder — no real scenery is read or written.
"""

import hashlib
import json
import os
import re
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_DSF_Edit as DSFEDIT
import O4_Veg_Replacement as CLEANUP


# ── synthetic assets ─────────────────────────────────────────────────────────

TREE_OBJ = """A
800
OBJ

TEXTURE forests/broadleaf_summer.png
POINT_COUNTS 8 0 0 12
ATTR_no_cull
TRIS 0 12
"""

# Named like a tree, but it drapes and carries a light: a mixed model.
STREET_TREE_LAMP_OBJ = """A
800
OBJ

TEXTURE objects/street_tree_lamp.png
POINT_COUNTS 40 0 0 60
ATTR_draped
LIGHT_NAMED airport_beacon 0 10 0
TRIS 0 60
"""

# Vegetation artwork but nothing else: one signal only, so not proven.
LONE_SIGNAL_OBJ = """A
800
OBJ

TEXTURE props/tree_stand.png
POINT_COUNTS 4 0 0 6
TRIS 0 6
"""

HANGAR_OBJ = """A
800
OBJ

TEXTURE buildings/hangar.png
POINT_COUNTS 20 0 0 30
TRIS 0 30
"""

AUTOGEN_BLOCK = """A
1000
AG_BLOCK

TEXTURE textures/block.png
TILE 0 0 64 64
OBJ_GRADED 4 2 vegetation/oak.obj
OBJ_GRADED 8 6 buildings/house.obj
"""


def _atom(tag, payload):
    return struct.pack("<4sI", tag, len(payload) + 8) + payload


def _strings(values):
    return b"".join(value.encode("utf-8") + b"\0" for value in values)


def _commands(placements):
    """Build a command stream placing objects: ``[(definition, count), ...]``."""
    out = bytearray()
    for definition, count in placements:
        out += bytes((3, definition))          # SetDefinition8
        for index in range(count):
            out += bytes((7,)) + struct.pack("<H", index)   # Object
    return bytes(out)


def _dsf(properties, definitions, commands=b""):
    head = _atom(DSFEDIT.ATOM_HEAD, _atom(DSFEDIT.ATOM_PROP, _strings(
        [item for pair in properties for item in pair]
    )))
    defn_children = b""
    for kind, tag in (("terrain", DSFEDIT.ATOM_TERT),
                      ("object", DSFEDIT.ATOM_OBJT),
                      ("polygon", DSFEDIT.ATOM_POLY),
                      ("network", DSFEDIT.ATOM_NETW)):
        defn_children += _atom(tag, _strings(definitions.get(kind, [])))
    body = (b"XPLNEDSF" + struct.pack("<I", 1) + head
            + _atom(DSFEDIT.ATOM_DEFN, defn_children)
            + _atom(DSFEDIT.ATOM_CMDS, commands))
    return body + hashlib.md5(body).digest()


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class SceneryFixture:
    """A Custom Scenery folder with an overlay pack above one regional pack."""

    def __init__(self, base, lat=47, lon=8):
        self.lat, self.lon = lat, lon
        self.custom_scenery = Path(base) / "Custom Scenery"
        self.overlay_pack = self.custom_scenery / CLEANUP.OVERLAY_PACK_NAME
        self.region_pack = self.custom_scenery / "zzz_Regional_Scenery"
        self.managed_root = Path(base) / "managed"
        self.relpath = CLEANUP.tile_dsf_relpath(lat, lon)
        self._build()

    def _build(self):
        (self.custom_scenery).mkdir(parents=True, exist_ok=True)
        (self.custom_scenery / "scenery_packs.ini").write_text(
            "I\n1000 Version\nSCENERY\n\n"
            f"SCENERY_PACK Custom Scenery/{CLEANUP.OVERLAY_PACK_NAME}/\n"
            f"SCENERY_PACK Custom Scenery/{self.region_pack.name}/\n",
            encoding="utf-8",
        )

        _write(self.region_pack / "trees" / "oak.obj", TREE_OBJ)
        _write(self.region_pack / "objects" / "street_tree_lamp.obj",
               STREET_TREE_LAMP_OBJ)
        _write(self.region_pack / "props" / "tree_stand.obj", LONE_SIGNAL_OBJ)
        _write(self.region_pack / "buildings" / "hangar.obj", HANGAR_OBJ)
        _write(self.region_pack / "autogen" / "suburb.agb", AUTOGEN_BLOCK)
        # The autogen block's own reference resolves relative to itself.
        _write(self.region_pack / "autogen" / "vegetation" / "oak.obj", TREE_OBJ)
        _write(self.region_pack / "autogen" / "buildings" / "house.obj", HANGAR_OBJ)

        self.source_dsf = self.region_pack / self.relpath
        self.source_dsf.parent.mkdir(parents=True, exist_ok=True)
        self.source_dsf.write_bytes(_dsf(
            [("sim/planet", "earth"), ("sim/overlay", "1")],
            {
                "object": ["trees/oak.obj", "objects/street_tree_lamp.obj",
                           "props/tree_stand.obj", "buildings/hangar.obj"],
                "polygon": ["forests/mixed.for", "autogen/suburb.agb"],
            },
            _commands([(0, 5), (1, 2), (3, 3)]),
        ))

        self.overlay_dsf = self.overlay_pack / self.relpath
        self.overlay_dsf.parent.mkdir(parents=True, exist_ok=True)
        self.overlay_dsf.write_bytes(_dsf(
            [("sim/planet", "earth"), ("sim/overlay", "1"),
             ("sim/west", str(self.lon)), ("sim/east", str(self.lon + 1)),
             ("sim/south", str(self.lat)), ("sim/north", str(self.lat + 1))],
            {"polygon": ["simHeaven/forests/broadleaf.for"]},
        ))

    def store(self):
        return CLEANUP.ManagedStore(str(self.managed_root))


class ClassifierTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_billboard_under_a_vegetation_folder_is_a_tree(self):
        path = _write(self.base / "trees" / "oak.obj", TREE_OBJ)
        verdict = CLEANUP.classify_object("trees/oak.obj", str(path))
        self.assertTrue(verdict.is_tree, verdict.reason)
        self.assertIn("textures:vegetation", verdict.evidence)

    def test_mixed_model_named_tree_is_never_removed(self):
        path = _write(self.base / "objects" / "street_tree_lamp.obj",
                      STREET_TREE_LAMP_OBJ)
        verdict = CLEANUP.classify_object(
            "objects/street_tree_lamp.obj", str(path))
        self.assertFalse(verdict.is_tree)
        self.assertIn("draped", verdict.reason)

    def test_one_signal_is_not_enough(self):
        path = _write(self.base / "props" / "tree_stand.obj", LONE_SIGNAL_OBJ)
        verdict = CLEANUP.classify_object("props/tree_stand.obj", str(path))
        self.assertFalse(verdict.is_tree)
        self.assertIn("one independent signal", verdict.reason)

    def test_asset_rules_can_confirm_and_veto(self):
        path = _write(self.base / "props" / "tree_stand.obj", LONE_SIGNAL_OBJ)
        confirmed = CLEANUP.classify_object(
            "props/tree_stand.obj", str(path), {"tree": ["props/tree_stand.obj"]})
        self.assertTrue(confirmed.is_tree)
        vetoed = CLEANUP.classify_object(
            "trees/oak.obj", str(_write(self.base / "trees" / "oak.obj", TREE_OBJ)),
            {"not_tree": ["trees/oak.obj"]})
        self.assertFalse(vetoed.is_tree)


class ExclusionValueTests(unittest.TestCase):
    def test_positive_and_negative_tiles(self):
        self.assertEqual(CLEANUP.overlay_exclusion_value(47, 8), "8/47/9/48")
        self.assertEqual(CLEANUP.overlay_exclusion_value(-34, -71),
                         "-71/-34/-70/-33")

    def test_fractional_coordinates_floor_to_the_containing_tile(self):
        self.assertEqual(CLEANUP.overlay_exclusion_value(-33.8, -70.2),
                         "-71/-34/-70/-33")
        self.assertEqual(CLEANUP.tile_name(-33.8, -70.2), "-34-071")


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = SceneryFixture(self.tmp.name)

    def test_inventory_separates_proven_trees_from_everything_else(self):
        report = CLEANUP.inventory_tile(
            str(self.fixture.custom_scenery), 47, 8)
        self.assertEqual(len(report["sources"]), 1)
        source = report["sources"][0]
        self.assertEqual([item["reference"] for item in source["tree_objects"]],
                         ["trees/oak.obj"])
        self.assertEqual(source["tree_objects"][0]["placements"], 5)
        unsupported = {item["reference"] for item in source["unsupported"]}
        self.assertIn("objects/street_tree_lamp.obj", unsupported)
        self.assertIn("props/tree_stand.obj", unsupported)
        self.assertEqual([item["reference"] for item in source["forests"]],
                         ["forests/mixed.for"])

    def test_autogen_block_is_supported_through_its_object_reference(self):
        report = CLEANUP.inventory_tile(
            str(self.fixture.custom_scenery), 47, 8)
        assets = report["sources"][0]["referencing_assets"]
        block = next(item for item in assets
                     if item["reference"].endswith(".agb"))
        self.assertTrue(block["supported"], block["reason"])
        self.assertEqual([tree["reference"] for tree in block["tree_references"]],
                         ["vegetation/oak.obj"])

    def test_overlay_pack_is_never_inventoried_as_a_source(self):
        report = CLEANUP.inventory_tile(
            str(self.fixture.custom_scenery), 47, 8)
        packages = {source["package"] for source in report["sources"]}
        self.assertNotIn(CLEANUP.OVERLAY_PACK_NAME, packages)


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = SceneryFixture(self.tmp.name)
        self.store = self.fixture.store()
        self.original = self.fixture.source_dsf.read_bytes()
        self.inventory = CLEANUP.inventory_tile(
            str(self.fixture.custom_scenery), 47, 8)
        self.plan = CLEANUP.plan_tile(self.inventory, self.store)

    def _apply(self):
        return CLEANUP.apply_plan(self.plan, self.store, force_space=True,
                                  ignore_busy=True, log=lambda *_: None)

    def test_plan_estimates_storage_from_real_file_sizes(self):
        estimate = self.plan["estimate"]
        self.assertEqual(estimate["backup_bytes"], len(self.original))
        self.assertGreater(estimate["private_asset_bytes"], 0)
        self.assertGreaterEqual(estimate["peak_temp_bytes"], len(self.original))

    def test_apply_rewrites_only_the_proven_tree_definitions(self):
        self._apply()
        after = DSFEDIT.DsfFile.read(str(self.fixture.source_dsf))
        objects = after.definitions("object")
        self.assertTrue(objects[0].endswith("empty_v1.obj"), objects[0])
        self.assertEqual(objects[1], "objects/street_tree_lamp.obj")
        self.assertEqual(objects[2], "props/tree_stand.obj")
        self.assertEqual(objects[3], "buildings/hangar.obj")

    def test_apply_preserves_every_non_vegetation_record(self):
        self._apply()
        before = DSFEDIT.DsfFile(self.original).atom_map()
        after = DSFEDIT.DsfFile.read(str(self.fixture.source_dsf)).atom_map()
        self.assertEqual(before[DSFEDIT.ATOM_CMDS], after[DSFEDIT.ATOM_CMDS])
        self.assertEqual(before[DSFEDIT.ATOM_HEAD], after[DSFEDIT.ATOM_HEAD])
        self.assertEqual(
            DSFEDIT.DsfFile(self.original).definitions("terrain"),
            DSFEDIT.DsfFile.read(str(self.fixture.source_dsf)).definitions("terrain"),
        )

    def test_autogen_gets_a_private_copy_beside_the_original(self):
        self._apply()
        after = DSFEDIT.DsfFile.read(str(self.fixture.source_dsf))
        polygons = after.definitions("polygon")
        self.assertEqual(polygons[0], "forests/mixed.for")
        self.assertNotEqual(polygons[1], "autogen/suburb.agb")
        private = self.fixture.region_pack / polygons[1]
        self.assertTrue(private.is_file())
        text = private.read_text(encoding="utf-8")
        self.assertNotIn("vegetation/oak.obj", text)
        self.assertIn("buildings/house.obj", text)
        self.assertIn("OBJ_GRADED 4 2 ", text)
        # The original autogen file is untouched.
        self.assertIn(
            "vegetation/oak.obj",
            (self.fixture.region_pack / "autogen" / "suburb.agb")
            .read_text(encoding="utf-8"),
        )

    def test_private_assets_are_content_addressed_and_shared(self):
        self._apply()
        names = [os.path.basename(asset["path"])
                 for asset in self.plan["private_assets"]]
        self.assertTrue(any("_vegfree_" in name for name in names), names)
        # Re-planning against the installed state produces the same asset names.
        again = CLEANUP.plan_tile(
            CLEANUP.inventory_tile(str(self.fixture.custom_scenery), 47, 8),
            self.store,
        )
        self.assertEqual(again["sources"], [])

    def test_backup_is_verified_and_reused(self):
        manifest = self._apply()
        digest = manifest["sources"][0]["original_sha256"]
        backup = Path(self.store.backup_path(digest))
        self.assertTrue(backup.is_file())
        self.assertEqual(backup.read_bytes(), self.original)
        self.assertTrue(self.store.has_backup(digest))

    def test_journal_records_each_change_and_closes(self):
        self._apply()
        rows = [json.loads(line) for line in
                Path(self.store.journal_path).read_text().splitlines()]
        events = [row["event"] for row in rows]
        self.assertEqual(events[0], "begin")
        self.assertEqual(events[-1], "end")
        self.assertIn("change", events)
        self.assertIsNone(self.store.pending())

    def test_there_is_no_third_collection_of_stripped_dsfs(self):
        self._apply()
        stored = [
            name for _root, _dirs, files in os.walk(self.store.root)
            for name in files
        ]
        # One backup per source DSF, plus bookkeeping — never a second full set.
        backups = [
            name for _root, _dirs, files in os.walk(self.store.backups_dir)
            for name in files if name.endswith(".bin")
        ]
        self.assertEqual(len(backups), 1)
        self.assertNotIn("cleaned", stored)

    def test_repeated_apply_is_refused_when_the_source_moved_on(self):
        self._apply()
        with self.assertRaises(CLEANUP.ConflictError):
            CLEANUP.apply_plan(self.plan, self.store, force_space=True,
                               ignore_busy=True, log=lambda *_: None)

    def test_verify_transform_rejects_an_unplanned_edit(self):
        dsf = DSFEDIT.DsfFile(self.original)
        paths = dsf.definitions("object")
        paths[3] = "somewhere/else.obj"
        dsf.set_definitions("object", paths)
        problem = CLEANUP.verify_transform(
            self.original, dsf.to_bytes(), {"object": {0: "private/empty.obj"}})
        self.assertIsNotNone(problem)


class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = SceneryFixture(self.tmp.name)
        self.store = self.fixture.store()
        self.original = self.fixture.source_dsf.read_bytes()
        plan = CLEANUP.plan_tile(
            CLEANUP.inventory_tile(str(self.fixture.custom_scenery), 47, 8),
            self.store,
        )
        self.manifest = CLEANUP.apply_plan(
            plan, self.store, force_space=True, ignore_busy=True,
            log=lambda *_: None)
        self.private = [Path(asset["path"])
                        for asset in self.manifest["private_assets"]]

    def test_restore_puts_the_original_bytes_back(self):
        result = CLEANUP.restore_tile(self.store, "+47+008", ignore_busy=True,
                                      log=lambda *_: None)
        self.assertEqual(result["conflicts"], [])
        self.assertEqual(self.fixture.source_dsf.read_bytes(), self.original)
        self.assertIsNone(self.store.load_tile("+47+008"))

    def test_restore_removes_private_assets_nothing_else_uses(self):
        CLEANUP.restore_tile(self.store, "+47+008", ignore_busy=True,
                             log=lambda *_: None)
        for path in self.private:
            self.assertFalse(path.exists(), path)

    def test_restore_keeps_a_private_asset_another_tile_still_needs(self):
        other = dict(self.manifest, tile="+48+008")
        self.store.save_tile("+48+008", other)
        CLEANUP.restore_tile(self.store, "+47+008", ignore_busy=True,
                             log=lambda *_: None)
        for path in self.private:
            self.assertTrue(path.exists(), path)

    def test_restore_refuses_to_overwrite_a_file_something_else_changed(self):
        self.fixture.source_dsf.write_bytes(self.original[:-16] + b"\0" * 16)
        result = CLEANUP.restore_tile(self.store, "+47+008", ignore_busy=True,
                                      log=lambda *_: None)
        self.assertTrue(result["conflicts"])
        self.assertIsNotNone(self.store.load_tile("+47+008"))

    def test_restore_of_an_unmanaged_tile_says_so(self):
        result = CLEANUP.restore_tile(self.store, "+49+008", ignore_busy=True,
                                      log=lambda *_: None)
        self.assertIn("no managed replacement", result["reason"])


class OverlayExclusionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = SceneryFixture(self.tmp.name)
        self.store = self.fixture.store()
        self.store.ensure_dirs()

    def test_adding_and_removing_the_exclusion_leaves_everything_else_alone(self):
        before = self.fixture.overlay_dsf.read_bytes()
        CLEANUP.set_overlay_exclusion(str(self.fixture.overlay_dsf), 47, 8, True,
                                      self.store, log=lambda *_: None)
        after = DSFEDIT.DsfFile.read(str(self.fixture.overlay_dsf))
        self.assertIn((CLEANUP.EXCLUSION_PROPERTY, "8/47/9/48"),
                      after.properties())
        self.assertEqual(after.definitions("polygon"),
                         DSFEDIT.DsfFile(before).definitions("polygon"))
        CLEANUP.set_overlay_exclusion(str(self.fixture.overlay_dsf), 47, 8, False,
                                      self.store, log=lambda *_: None)
        self.assertEqual(self.fixture.overlay_dsf.read_bytes(), before)

    def test_an_overlay_with_no_forests_is_refused(self):
        self.fixture.overlay_dsf.write_bytes(_dsf(
            [("sim/planet", "earth"), ("sim/overlay", "1"),
             ("sim/west", "8"), ("sim/east", "9"),
             ("sim/south", "47"), ("sim/north", "48")],
            {"object": ["something.obj"]},
        ))
        problem = CLEANUP.check_overlay_suitable(
            str(self.fixture.overlay_dsf), 47, 8)
        self.assertIn("no forests", problem)
        with self.assertRaises(CLEANUP.ConflictError):
            CLEANUP.set_overlay_exclusion(
                str(self.fixture.overlay_dsf), 47, 8, True, self.store)

    def test_an_overlay_for_another_tile_is_refused(self):
        problem = CLEANUP.check_overlay_suitable(
            str(self.fixture.overlay_dsf), 48, 8)
        self.assertIn("covers", problem)

    def test_a_missing_overlay_is_refused(self):
        problem = CLEANUP.check_overlay_suitable(
            str(self.fixture.overlay_dsf.parent / "nope.dsf"), 47, 8)
        self.assertIn("no generated vegetation overlay", problem)


class InterruptionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = CLEANUP.ManagedStore(os.path.join(self.tmp.name, "managed"))

    def test_an_open_operation_blocks_a_different_one(self):
        self.store.begin({"op_id": "aaa", "action": "install", "tile": "+47+008"})
        with self.assertRaises(CLEANUP.ConflictError) as caught:
            self.store.begin({"op_id": "bbb", "action": "install",
                              "tile": "+48+008"})
        self.assertIn("interrupted", str(caught.exception))

    def test_the_same_operation_may_resume(self):
        first = self.store.begin({"op_id": "aaa", "action": "install",
                                  "tile": "+47+008"})
        self.store.record(first, {"kind": "source_dsf", "path": "a.dsf"})
        resumed = self.store.begin({"op_id": "aaa", "action": "install",
                                    "tile": "+47+008"})
        self.assertEqual(resumed["op_id"], "aaa")
        self.store.end(resumed)
        self.assertIsNone(self.store.pending())


class AdoptBackupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.store = CLEANUP.ManagedStore(str(self.base / "managed"))
        self.output = self.base / "old_cleanup"
        (self.output / "originals" / "pack").mkdir(parents=True)

    def _record(self, name, stored_bytes, original_bytes, cleaned_bytes):
        (self.output / "originals" / "pack" / name).write_bytes(stored_bytes)
        row = {
            "path": f"pack/{name}",
            "original_sha256": DSFEDIT.sha256_bytes(original_bytes),
            "cleaned_sha256": DSFEDIT.sha256_bytes(cleaned_bytes),
        }
        with (self.output / "manifest.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def test_a_verified_original_is_adopted(self):
        original, cleaned = b"original bytes", b"stripped bytes"
        self._record("a.dsf", original, original, cleaned)
        result = CLEANUP.adopt_backups_from_strip_output(self.store, str(self.output))
        self.assertEqual(len(result["adopted"]), 1)
        self.assertTrue(self.store.has_backup(DSFEDIT.sha256_bytes(original)))

    def test_an_already_stripped_file_is_never_taken_for_an_original(self):
        original, cleaned = b"original bytes", b"stripped bytes"
        self._record("b.dsf", cleaned, original, cleaned)
        result = CLEANUP.adopt_backups_from_strip_output(self.store, str(self.output))
        self.assertEqual(result["adopted"], [])
        self.assertIn("stripped version", result["skipped"][0]["reason"])
        self.assertFalse(self.store.has_backup(DSFEDIT.sha256_bytes(original)))


class SceneryOrderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = SceneryFixture(self.tmp.name)

    def test_correct_order_reports_nothing(self):
        conflicts = CLEANUP.validate_scenery_order(
            str(self.fixture.custom_scenery), 47, 8)
        self.assertEqual(conflicts, [])

    def test_a_regional_pack_above_the_overlay_is_reported(self):
        (self.fixture.custom_scenery / "scenery_packs.ini").write_text(
            "I\n1000 Version\nSCENERY\n\n"
            f"SCENERY_PACK Custom Scenery/{self.fixture.region_pack.name}/\n"
            f"SCENERY_PACK Custom Scenery/{CLEANUP.OVERLAY_PACK_NAME}/\n",
            encoding="utf-8",
        )
        conflicts = CLEANUP.validate_scenery_order(
            str(self.fixture.custom_scenery), 47, 8)
        self.assertEqual(len(conflicts), 1)
        self.assertIn("will NOT be replaced", conflicts[0]["message"])

    def test_a_missing_overlay_pack_is_an_error(self):
        (self.fixture.custom_scenery / "scenery_packs.ini").write_text(
            "I\n1000 Version\nSCENERY\n\n"
            f"SCENERY_PACK Custom Scenery/{self.fixture.region_pack.name}/\n",
            encoding="utf-8",
        )
        conflicts = CLEANUP.validate_scenery_order(
            str(self.fixture.custom_scenery), 47, 8)
        self.assertEqual(conflicts[0]["severity"], "error")


class NeighbourTileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = SceneryFixture(self.tmp.name)
        self.store = self.fixture.store()

    def _plan(self):
        return CLEANUP.plan_tile(
            CLEANUP.inventory_tile(str(self.fixture.custom_scenery), 47, 8),
            self.store,
        )

    def test_a_neighbour_drawing_the_same_tree_is_reported(self):
        neighbour = self.fixture.region_pack / CLEANUP.tile_dsf_relpath(47, 9)
        neighbour.parent.mkdir(parents=True, exist_ok=True)
        neighbour.write_bytes(_dsf(
            [("sim/planet", "earth")],
            {"object": ["trees/oak.obj"]},
            _commands([(0, 2)]),
        ))
        findings = self._plan()["neighbour_tiles"]
        self.assertEqual([item["tile"] for item in findings], ["+47+009"])
        self.assertEqual(findings[0]["references"], ["trees/oak.obj"])

    def test_no_neighbours_means_nothing_to_report(self):
        self.assertEqual(self._plan()["neighbour_tiles"], [])


class ApplyInterruptionTests(unittest.TestCase):
    """An interrupted install must stay findable, not silently vanish."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = SceneryFixture(self.tmp.name)
        self.store = self.fixture.store()
        self.plan = CLEANUP.plan_tile(
            CLEANUP.inventory_tile(str(self.fixture.custom_scenery), 47, 8),
            self.store,
        )

    def test_a_failure_leaves_the_pending_operation_behind(self):
        # Make the source vanish between planning and installing.
        self.fixture.source_dsf.unlink()
        with self.assertRaises(Exception):
            CLEANUP.apply_plan(self.plan, self.store, force_space=True,
                               ignore_busy=True, log=lambda *_: None)
        pending = self.store.pending()
        self.assertIsNotNone(pending)
        self.assertEqual(pending["tile"], "+47+008")
        self.assertEqual(pending["action"], "install")
        self.assertIsNone(self.store.load_tile("+47+008"))

    def test_the_journal_records_the_failure(self):
        self.fixture.source_dsf.unlink()
        with self.assertRaises(Exception):
            CLEANUP.apply_plan(self.plan, self.store, force_space=True,
                               ignore_busy=True, log=lambda *_: None)
        events = [json.loads(line)["event"] for line in
                  Path(self.store.journal_path).read_text().splitlines()]
        self.assertIn("failed", events)
        self.assertNotIn("end", events)

    def test_resuming_the_same_operation_keeps_recorded_progress(self):
        first = self.store.begin({"op_id": "abc", "action": "install",
                                  "tile": "+47+008"})
        self.store.record(first, {"kind": "private_asset", "path": "a.obj"})
        resumed = self.store.begin({"op_id": "abc", "action": "install",
                                    "tile": "+47+008"})
        self.assertEqual(len(resumed["completed"]), 1)

    def test_space_shortage_stops_before_any_modification(self):
        before = self.fixture.source_dsf.read_bytes()
        self.plan["estimate"]["sufficient"] = False
        with self.assertRaises(CLEANUP.ConflictError) as caught:
            CLEANUP.apply_plan(self.plan, self.store, ignore_busy=True,
                               log=lambda *_: None)
        self.assertIn("not enough free space", str(caught.exception))
        self.assertEqual(self.fixture.source_dsf.read_bytes(), before)
        self.assertIsNone(self.store.pending())


class ReconcileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = SceneryFixture(self.tmp.name)
        self.store = self.fixture.store()
        self.original = self.fixture.source_dsf.read_bytes()
        plan = CLEANUP.plan_tile(
            CLEANUP.inventory_tile(str(self.fixture.custom_scenery), 47, 8),
            self.store,
        )
        CLEANUP.apply_plan(plan, self.store, force_space=True, ignore_busy=True,
                           log=lambda *_: None)
        CLEANUP.set_overlay_exclusion(str(self.fixture.overlay_dsf), 47, 8, True,
                                      self.store, log=lambda *_: None)

    def _exclusion_keys(self):
        overlay = DSFEDIT.DsfFile.read(str(self.fixture.overlay_dsf))
        return [key for key, _ in overlay.properties()]

    def test_reconcile_undoes_a_replacement_the_settings_no_longer_want(self):
        result = CLEANUP.reconcile_tile(
            self.store, "+47+008", overlay_dsf=str(self.fixture.overlay_dsf),
            ignore_busy=True, log=lambda *_: None)
        self.assertEqual(result["conflicts"], [])
        self.assertEqual(self.fixture.source_dsf.read_bytes(), self.original)
        self.assertNotIn(CLEANUP.EXCLUSION_PROPERTY, self._exclusion_keys())

    def test_reconcile_of_an_untouched_tile_still_clears_a_stray_exclusion(self):
        CLEANUP.restore_tile(self.store, "+47+008", ignore_busy=True,
                             log=lambda *_: None)
        CLEANUP.set_overlay_exclusion(str(self.fixture.overlay_dsf), 47, 8, True,
                                      self.store, log=lambda *_: None)
        result = CLEANUP.reconcile_tile(
            self.store, "+47+008", overlay_dsf=str(self.fixture.overlay_dsf),
            ignore_busy=True, log=lambda *_: None)
        self.assertEqual(result["action"], "overlay_only")
        self.assertNotIn(CLEANUP.EXCLUSION_PROPERTY, self._exclusion_keys())


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = SceneryFixture(self.tmp.name)
        self.store = self.fixture.store()
        self.inventory = CLEANUP.inventory_tile(
            str(self.fixture.custom_scenery), 47, 8)
        self.plan = CLEANUP.plan_tile(self.inventory, self.store)

    def _report(self):
        return CLEANUP.format_report(self.plan, self.inventory, (),
                                     store=self.store)

    def test_the_report_names_unsupported_sources_and_the_restore_command(self):
        text = self._report()
        self.assertIn("street_tree_lamp.obj", text)
        self.assertIn("veg_tile_replacement.py restore --tile +47+008", text)
        self.assertIn(self.store.backups_dir, text)

    def test_the_report_separates_file_checks_from_what_x_plane_shows(self):
        self.assertIn("does not state what the tile looks like in X-Plane",
                      self._report())

    def test_the_report_counts_forests_left_to_the_exclusion(self):
        self.assertIn("Forest polygons suppressed by the tile exclusion:",
                      self._report())


class RepointTextAssetTests(unittest.TestCase):
    def test_a_longer_token_starting_with_the_match_is_left_alone(self):
        text = "OBJ_GRADED 4 2 a/oak.obj\nOBJ_GRADED 8 6 a/oak.obj.backup\n"
        out, replacements = CLEANUP.repoint_text_asset(
            text, {"a/oak.obj": "p/empty.obj"})
        self.assertIn("a/oak.obj.backup", out)
        self.assertEqual(out.count("p/empty.obj"), 1)
        self.assertEqual(replacements, [("a/oak.obj", "p/empty.obj")])

    def test_spacing_line_order_and_untouched_lines_survive(self):
        text = "OBJ  9  1   a/oak.obj  # note\r\nTILE 0 0 64 64\n"
        out, _ = CLEANUP.repoint_text_asset(text, {"a/oak.obj": "p/empty.obj"})
        self.assertEqual(
            out, "OBJ  9  1   p/empty.obj  # note\r\nTILE 0 0 64 64\n")

    def test_matching_ignores_case_and_separator_style(self):
        text = "OBJ A\\Oak.OBJ\n"
        out, replacements = CLEANUP.repoint_text_asset(
            text, {"a/oak.obj": "p/empty.obj"})
        self.assertEqual(out, "OBJ p/empty.obj\n")
        self.assertEqual(len(replacements), 1)

    def test_nothing_to_map_means_nothing_changes(self):
        text = "OBJ b/house.obj\n"
        out, replacements = CLEANUP.repoint_text_asset(
            text, {"a/oak.obj": "p/empty.obj"})
        self.assertEqual(out, text)
        self.assertEqual(replacements, [])


class BatchDeleteRestoreTests(unittest.TestCase):
    """Erasing a tile must hand its edited scenery back to its owner.

    The Tiles window is Tk, so the wiring is checked against the source rather
    than by driving a window; the behaviour it calls into is exercised below
    against a real managed store.
    """

    def _gui_source(self):
        return (SRC / "O4_GUI_Utils.py").read_text(encoding="utf-8")

    def test_the_delete_list_offers_the_restore(self):
        source = self._gui_source()
        self.assertIn('"Veg replacement",', source)

    def test_erasing_a_whole_tile_restores_without_being_asked(self):
        source = self._gui_source()
        self.assertIn(
            'restore_veg = bool(self.v_["Veg replacement"].get()) or (\n'
            '            bool(self.v_["Tile (whole)"].get()) and not self.grouped\n'
            '        )',
            source,
        )

    def test_the_restore_runs_before_the_tile_is_erased(self):
        source = self._gui_source()
        restore_at = source.index("self.restore_veg_replacement(lat, lon, summary)")
        delete_at = source.index("self.delete_tile_whole(lat, lon)")
        # The backup is keyed on the tile, not on the build folder, but running
        # first keeps a failure recoverable: the tile is still there to retry.
        self.assertLess(restore_at, delete_at)

    def test_a_tile_with_nothing_recorded_is_skipped_silently(self):
        source = self._gui_source()
        self.assertIn("if not PIPELINE.has_veg_tile_replacement(lat, lon):", source)
        self.assertIn('summary["skipped"] += 1', source)

    def test_an_unresolved_conflict_is_raised_to_a_dialog(self):
        source = self._gui_source()
        report = source[source.index("def report_veg_restore"):]
        report = report[:report.index("\n    def ")]
        self.assertIn("messagebox.showwarning", report)
        self.assertIn("UI.lvprint(0,", report)


class HasReplacementTests(unittest.TestCase):
    """The cheap check the delete path uses to stay quiet."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = CLEANUP.ManagedStore(os.path.join(self.tmp.name, "managed"))

    def test_absent_before_anything_is_installed(self):
        self.assertIsNone(self.store.load_tile("+47+008"))

    def test_present_once_a_manifest_exists_and_gone_after_restore(self):
        self.store.save_tile("+47+008", {"tile": "+47+008", "sources": [],
                                         "private_assets": []})
        self.assertIsNotNone(self.store.load_tile("+47+008"))
        CLEANUP.restore_tile(self.store, "+47+008", ignore_busy=True,
                             log=lambda *_: None)
        self.assertIsNone(self.store.load_tile("+47+008"))


class DeleteThenRestoreTests(unittest.TestCase):
    """The whole delete-time path, against installed scenery."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = SceneryFixture(self.tmp.name)
        self.store = self.fixture.store()
        self.original = self.fixture.source_dsf.read_bytes()
        plan = CLEANUP.plan_tile(
            CLEANUP.inventory_tile(str(self.fixture.custom_scenery), 47, 8),
            self.store,
        )
        CLEANUP.apply_plan(plan, self.store, force_space=True, ignore_busy=True,
                           log=lambda *_: None)
        CLEANUP.set_overlay_exclusion(str(self.fixture.overlay_dsf), 47, 8, True,
                                      self.store, log=lambda *_: None)

    def test_restoring_at_delete_time_returns_the_package_to_shipped_state(self):
        result = CLEANUP.restore_tile(
            self.store, "+47+008", overlay_dsf=str(self.fixture.overlay_dsf),
            ignore_busy=True, log=lambda *_: None)
        self.assertEqual(result["conflicts"], [])
        self.assertEqual(self.fixture.source_dsf.read_bytes(), self.original)
        self.assertFalse(
            (self.fixture.region_pack / CLEANUP.PRIVATE_DIR_NAME).exists())
        self.assertIsNone(self.store.load_tile("+47+008"))

    def test_a_changed_source_is_reported_and_left_alone(self):
        installed = self.fixture.source_dsf.read_bytes()
        self.fixture.source_dsf.write_bytes(installed[:-16] + b"\0" * 16)
        result = CLEANUP.restore_tile(
            self.store, "+47+008", overlay_dsf=str(self.fixture.overlay_dsf),
            ignore_busy=True, log=lambda *_: None)
        self.assertTrue(result["conflicts"])
        # The record survives so the tile can still be recovered by hand.
        self.assertIsNotNone(self.store.load_tile("+47+008"))


class BatchDeleteOutcomeTests(unittest.TestCase):
    """Batch Delete reports what the whole run did, not what its last step did.

    Each delete helper returns True or None, and they used to be assigned to
    the same flag in turn, so whichever ran last decided the closing message on
    its own. A run that erased OSM data and imagery still said "No cached data
    found" if the final step happened to find nothing.
    """

    def _trash_body(self):
        source = (SRC / "O4_GUI_Utils.py").read_text(encoding="utf-8")
        start = source.index("    def trash(self) -> None:")
        return source[start:source.index("\n    def ", start + 1)]

    def test_no_step_overwrites_the_outcome_of_an_earlier_one(self):
        body = self._trash_body()
        self.assertNotIn("data_deleted = self.", body)
        self.assertNotIn("data_deleted = data_deleted", body)

    def test_every_delete_step_contributes_to_the_outcome(self):
        body = self._trash_body()
        calls = set(re.findall(r"self\.(delete_\w+)\(lat, lon\)", body))
        self.assertTrue(calls)
        for call in sorted(calls):
            self.assertIn(
                "data_deleted |= bool(self.%s(lat, lon))" % call,
                body,
                "%s does not accumulate into data_deleted" % call,
            )

    def test_a_restore_on_its_own_still_counts_as_work_done(self):
        body = self._trash_body()
        self.assertIn(
            "data_deleted |= self.restore_veg_replacement(lat, lon, summary)",
            body,
        )


if __name__ == "__main__":
    unittest.main()
