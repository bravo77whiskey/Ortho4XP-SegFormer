"""Guard the PyInstaller spec against missing overlay module bundles.

The packaged app runs the overlays in a .venv subprocess that imports loose
.py files from _internal/overlay_scripts/src. Files land there ONLY when listed
in Ortho4XP.spec's OVERLAY_MODULES / OVERLAY_SCRIPTS. A module imported at the
top of a shipped module but absent from the spec breaks every deployed build
with ModuleNotFoundError (this happened with the height model, 2026-07-07).

This test statically closes the import graph: every top-level ``import O4_*``
/ ``from O4_* import`` in a shipped loose module must resolve to another
shipped loose module. Indented (lazy/guarded) imports are exempt — they are
allowed to resolve against the frozen exe or optional environments.
"""

import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "Ortho4XP.spec"
SRC = ROOT / "src"

_TOP_IMPORT_RE = re.compile(r"^(?:import|from)\s+(O4_\w+)", re.MULTILINE)


def _spec_list(name):
    """Return the literal list assigned to ``name`` in Ortho4XP.spec."""
    tree = ast.parse(SPEC.read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Ortho4XP.spec has no {name} list")


class SpecBundlesOverlayModulesTest(unittest.TestCase):
    def test_shipped_files_exist(self):
        for name in _spec_list("OVERLAY_MODULES"):
            self.assertTrue((SRC / f"{name}.py").is_file(), f"src/{name}.py missing")
        for name in _spec_list("OVERLAY_DATA_FILES"):
            self.assertTrue((SRC / name).is_file(), f"src/{name} missing")
        for name in _spec_list("OVERLAY_SCRIPTS"):
            self.assertTrue(
                (SRC / "scripts" / name).is_file(), f"src/scripts/{name} missing"
            )

    def test_loose_import_graph_is_closed(self):
        shipped = set(_spec_list("OVERLAY_MODULES"))
        self.assertIn("O4_Building_Overlay", shipped)
        sources = [SRC / f"{module}.py" for module in sorted(shipped)]
        sources += [SRC / "scripts" / name for name in _spec_list("OVERLAY_SCRIPTS")]
        missing = {}
        for path in sources:
            source = path.read_text(encoding="utf-8", errors="ignore")
            for imported in _TOP_IMPORT_RE.findall(source):
                if imported in shipped:
                    continue
                if not (SRC / f"{imported}.py").is_file():
                    # Not a repo module (or a package) — out of scope.
                    continue
                missing.setdefault(imported, []).append(path.name)
        self.assertFalse(
            missing,
            "Top-level imports of repo modules that are NOT bundled into "
            "overlay_scripts/src by Ortho4XP.spec (deployed builds will crash "
            f"with ModuleNotFoundError): {missing}",
        )

    def test_overlay_and_app_module_lists_do_not_overlap(self):
        overlap = set(_spec_list("OVERLAY_MODULES")) & set(_spec_list("APP_MODULES"))
        self.assertFalse(overlap, f"listed twice in Ortho4XP.spec: {sorted(overlap)}")


if __name__ == "__main__":
    unittest.main()
