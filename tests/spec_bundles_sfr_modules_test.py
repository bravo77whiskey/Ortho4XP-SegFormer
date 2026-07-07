"""Guard the PyInstaller spec against missing SFR module bundles.

The packaged app runs the SFR overlay in a venv subprocess that imports loose
.py files from _internal/sfr_scripts/src — files land there ONLY when listed
in Ortho4XP.spec's added_datas. A new module imported at the top of a shipped
module but absent from the spec breaks every deployed build with
ModuleNotFoundError (this happened with O4_SFR_Height_Model, 2026-07-07).

This test statically closes the import graph: every top-level ``import O4_*``
/ ``from O4_* import`` in a shipped loose module must resolve to another
shipped loose module. Indented (lazy/guarded) imports are exempt — they are
allowed to resolve against the frozen exe or optional environments.
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "Ortho4XP.spec"
SRC = ROOT / "src"

_DATA_RE = re.compile(
    r'os\.path\.join\(SPEC_DIR,\s*"src",\s*"([^"]+\.py)"\),\s*"sfr_scripts/src"'
)
_TOP_IMPORT_RE = re.compile(
    r"^(?:import|from)\s+(O4_\w+)", re.MULTILINE
)
_HIDDEN_RE = re.compile(r'^\s*"(O4_\w+)",', re.MULTILINE)


class SpecBundlesSfrModulesTest(unittest.TestCase):
    def _shipped_loose_modules(self, spec_text: str) -> set[str]:
        return {Path(name).stem for name in _DATA_RE.findall(spec_text)}

    def test_shipped_module_files_exist(self):
        spec_text = SPEC.read_text(encoding="utf-8")
        for name in _DATA_RE.findall(spec_text):
            self.assertTrue(
                (SRC / name).is_file(),
                f"Ortho4XP.spec bundles src/{name} which does not exist",
            )

    def test_loose_sfr_import_graph_is_closed(self):
        spec_text = SPEC.read_text(encoding="utf-8")
        shipped = self._shipped_loose_modules(spec_text)
        self.assertIn("O4_SFR_Building_Overlay", shipped)
        missing = {}
        for module in sorted(shipped):
            path = SRC / f"{module}.py"
            if not path.is_file():
                continue
            source = path.read_text(encoding="utf-8", errors="ignore")
            for imported in _TOP_IMPORT_RE.findall(source):
                if imported in shipped:
                    continue
                if not (SRC / f"{imported}.py").is_file():
                    # Not a repo module (or a package) — out of scope.
                    continue
                missing.setdefault(imported, []).append(module)
        self.assertFalse(
            missing,
            "Top-level imports of repo modules that are NOT bundled into "
            "sfr_scripts/src by Ortho4XP.spec (deployed builds will crash "
            f"with ModuleNotFoundError): {missing}",
        )

    def test_loose_sfr_modules_are_also_hidden_imports(self):
        # The frozen exe lists the same modules as hiddenimports; keep the two
        # lists in sync for the O4_SFR_* set so analysis never silently drops
        # one path.
        spec_text = SPEC.read_text(encoding="utf-8")
        shipped_sfr = {
            name for name in self._shipped_loose_modules(spec_text)
            if name.startswith("O4_SFR_")
        }
        hidden = set(_HIDDEN_RE.findall(spec_text))
        missing = sorted(shipped_sfr - hidden)
        self.assertFalse(
            missing,
            f"sfr_scripts modules absent from hiddenimports: {missing}",
        )


if __name__ == "__main__":
    unittest.main()
