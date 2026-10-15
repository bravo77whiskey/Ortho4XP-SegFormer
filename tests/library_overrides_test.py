"""Tests for scripts/library_audit/build_library_overrides.py on a fake Custom Scenery."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "library_audit"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_library_overrides as LO  # noqa: E402

CFG = {
    "pack_name": "O4XP_Library_Overrides",
    "target_pack": "SFD Global Autogen",
    "max_area_ratio": 1.5,
    "fit_slack": 1.1,
    "min_fill": 0.45,
    "same_style_within": 0.6,
    "closest_max_overshoot": 1.3,
    "simheaven_packs_by_region": {"africa": ["Africa"]},
    "manual": [],
}


def write_obj(path, w, d, texture="tex.png"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"I\n800\nOBJ\nTEXTURE {texture}\n"
        f"VT {-w / 2} 0 {-d / 2} 0 1 0 0 0\nVT {w / 2} 3 {d / 2} 0 1 0 1 1\n"
    )
    (path.parent / texture).write_bytes(b"png")


class LibraryOverridesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        sfd = root / "SFD Global Autogen"
        write_obj(sfd / "Africa" / "Big.obj", 12, 14)
        write_obj(sfd / "Africa" / "Small.obj", 8.8, 7.5)
        (sfd / "africa.png").write_bytes(b"png")
        (sfd / "library.txt").write_text(
            "A\n800\nLIBRARY\nREGION_DEFINE africa\nREGION_BITMAP africa.png\n"
            "REGION_DEFINE all\nREGION_ALL\n"
            "REGION africa\n"
            "EXPORT simheaven/houses/house_06x04x2.obj Africa/Big.obj\n"
            "EXPORT simheaven/houses/house_09x08x1.obj Africa/Big.obj\n"
            "EXPORT simheaven/houses/house_10x10x1.obj Africa/Small.obj\n"
            "EXPORT simheaven/houses/house_10x09x1.obj Africa/Small.obj\n"
            "EXPORT simheaven/houses/house_10x09x1.obj Africa/Big.obj\n"
            "EXPORT simheaven/houses/house_05x05x1.obj Africa/Big.obj\n"
            "REGION all\n"
            "EXPORT_BACKUP simheaven/houses/house_09x08x1.obj Africa/Big.obj\n"
        )
        sh = root / "simHeaven_X-World_Africa-6-scenery"
        write_obj(sh / "objects" / "houses" / "06x04.obj", 6.5, 4.5, "GR.dds")
        write_obj(sh / "objects" / "houses" / "04x06.obj", 4, 6, "GR.dds")
        (sh / "library.txt").write_text(
            "A\n800\nLIBRARY\nREGION_DEFINE africa\nREGION_BITMAP region_africa.png\n"
            "REGION africa\n"
            "EXPORT_EXCLUDE simheaven/houses/house_06x04x2.obj objects/houses/06x04.obj\n"
            "EXPORT_EXCLUDE simheaven/houses/house_04x06x1.obj objects/houses/04x06.obj\n"
        )
        (root / "scenery_packs.ini").write_text(
            "I\n1000 Version\nSCENERY\n\n"
            "SCENERY_PACK Custom Scenery/yOrtho4XP_Bld_Overlays/\n"
            "SCENERY_PACK Custom Scenery/SFD Global Autogen/\n"
            "SCENERY_PACK Custom Scenery/simHeaven_X-World_Africa-6-scenery/\n"
        )
        self.root = str(root)

    def tearDown(self):
        self.tmp.cleanup()

    def groups(self, cfg=CFG):
        return {g.virtual.split("/")[-1]: g for g in LO.plan(self.root, cfg)}

    def test_picks(self):
        g = self.groups()
        # 10x10 drawn 8.8x7.5 is undersized, not overridden
        self.assertNotIn("house_10x10x1.obj", g)
        # simHeaven's own mesh where it has one
        self.assertEqual(g["house_06x04x2.obj"].action, "simheaven")
        self.assertEqual([m.rel for m in g["house_06x04x2.obj"].picks], ["objects/houses/06x04.obj"])
        # else the largest fitting mesh: 8.8x7.5 fits inside 9x8
        self.assertEqual(g["house_09x08x1.obj"].action, "fit")
        self.assertEqual([m.rel for m in g["house_09x08x1.obj"].picks], ["Africa/Small.obj"])
        # one oversized export overrides the whole (region, path)
        self.assertIn("house_10x09x1.obj", g)
        # nothing fits inside 5x5: the 4x6 house overshoots least (1.2x)
        self.assertEqual(g["house_05x05x1.obj"].action, "closest")
        self.assertEqual([m.rel for m in g["house_05x05x1.obj"].picks], ["objects/houses/04x06.obj"])

    def test_manual_rules_win(self):
        cfg = dict(CFG, manual=[
            {"virtual": "simheaven/houses/house_09x08x1.obj", "mesh": "keep"},
            {"virtual": "simheaven/houses/house_06x04x2.obj", "region": "africa",
             "mesh": "simHeaven_X-World_Africa-6-scenery/objects/houses/04x06.obj"},
        ])
        g = self.groups(cfg)
        self.assertEqual(g["house_09x08x1.obj"].action, "keep (manual)")
        self.assertEqual(g["house_09x08x1.obj"].picks, [])
        self.assertEqual([m.rel for m in g["house_06x04x2.obj"].picks], ["objects/houses/04x06.obj"])

    def test_install_writes_pack_above_target(self):
        groups = LO.plan(self.root, CFG)
        files = LO.files_to_place(self.root, groups)
        LO.install(self.root, groups, files, CFG)
        pack = Path(self.root) / "O4XP_Library_Overrides"
        lib = (pack / "library.txt").read_text()
        self.assertIn("REGION_BITMAP africa.png", lib)
        self.assertIn(
            "EXPORT_EXCLUDE simheaven/houses/house_06x04x2.obj "
            "objects/simHeaven_X-World_Africa-6-scenery/objects/houses/06x04.obj", lib)
        self.assertNotIn("house_10x10x1", lib)
        self.assertTrue((pack / "africa.png").is_file())
        self.assertTrue((pack / "objects/SFD_Global_Autogen/Africa/Small.obj").is_file())
        self.assertTrue((pack / "objects/SFD_Global_Autogen/Africa/tex.png").is_file())
        self.assertTrue((pack / "objects/simHeaven_X-World_Africa-6-scenery/objects/houses/GR.dds").is_file())
        ini = (Path(self.root) / "scenery_packs.ini").read_text().splitlines()
        self.assertEqual(ini.index("SCENERY_PACK Custom Scenery/O4XP_Library_Overrides/") + 1,
                         ini.index("SCENERY_PACK Custom Scenery/SFD Global Autogen/"))
        self.assertTrue(Path(self.root, "scenery_packs.ini.bak_before_o4xp_library_overrides").is_file())
        marker = json.loads((pack / LO.MARKER).read_text())
        self.assertEqual(marker["target_pack"], "SFD Global Autogen")
        # a rebuild replaces the pack and keeps one ini line
        LO.install(self.root, groups, files, CFG)
        ini = (Path(self.root) / "scenery_packs.ini").read_text().splitlines()
        self.assertEqual(ini.count("SCENERY_PACK Custom Scenery/O4XP_Library_Overrides/"), 1)

    def test_refuses_foreign_pack_folder(self):
        os.makedirs(os.path.join(self.root, "O4XP_Library_Overrides"))
        groups = LO.plan(self.root, CFG)
        with self.assertRaises(SystemExit):
            LO.install(self.root, groups, LO.files_to_place(self.root, groups), CFG)


if __name__ == "__main__":
    unittest.main()
