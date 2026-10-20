"""Quiet overlay runs retain progress and failures across worker boundaries."""

import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import O4_Building_Overlay as BLD
import O4_Cfg_Vars as CFG
import O4_Grass_Overlay as GRASS
import O4_Landcover_Inference as SEG
import O4_Leaf_Type as LEAF
import O4_Overlay_Pipeline as PIPE
import O4_Texture_Selection as TEX
import O4_Vegetation_Overlay as VEG
from O4_Overlay_Logging import verbose_print, with_verbose_logging


class OverlayLoggingTests(unittest.TestCase):
    def test_building_setup_honours_toggle_and_keeps_texture_warning(self):
        report = {"notes": ["bad tile config"], "partial": 1}
        for verbose in (False, True):
            with self.subTest(verbose=verbose), tempfile.TemporaryDirectory() as tmp:
                out = StringIO()
                with (
                    mock.patch.object(SEG, "collect_source_texture_files",
                                      return_value=(["1_2_BI16.dds"], "dds", None)),
                    mock.patch.object(TEX, "select_textures",
                                      return_value=(["1_2_BI16.dds"], {}, report)),
                    mock.patch.object(BLD._BuildingOverlayRun, "_start_dsf_predecode",
                                      side_effect=RuntimeError("stop before inference")),
                    redirect_stdout(out),
                    self.assertRaisesRegex(RuntimeError, "stop before inference"),
                ):
                    BLD.run(tmp, 47, 8, str(Path(tmp) / "out.dsf"), 10, 1, 1,
                            cache_dir=tmp, verbose_log=verbose)
                text = out.getvalue()
                self.assertIn("Tile: lat=47", text)
                self.assertIn("WARNING bad tile config", text)
                self.assertEqual("Params:" in text, verbose)
                self.assertEqual("partially owned" in text, verbose)

    def test_vegetation_entry_point_honours_toggle_and_keeps_empty_result(self):
        for verbose in (False, True):
            with self.subTest(verbose=verbose), tempfile.TemporaryDirectory() as tmp:
                out = StringIO()
                with mock.patch.object(SEG, "collect_source_texture_files",
                                       return_value=([], "dds", None)), redirect_stdout(out):
                    result = VEG.run(tmp, 47, 8, str(Path(tmp) / "out.dsf"), tmp,
                                     10, 3, 50, 3, False, exclude_lower_forests=True,
                                     verbose_log=verbose)
                self.assertEqual(result, 0)
                self.assertIn("No DDS files found", out.getvalue())
                self.assertEqual("forced off for this run" in out.getvalue(), verbose)
                self.assertIn("[Veg] Tile +47+008: tree polygons=0; grass polygons=0;", out.getvalue())
                self.assertIn("total generation time=", out.getvalue())

    def test_building_tile_summary_is_unconditional_and_covers_the_whole_run(self):
        for verbose in (False, True):
            for count in (None, 1234):
                with self.subTest(verbose=verbose, count=count):
                    run = SimpleNamespace(verbose_log=verbose, lat=-33.8, lon=-70.2,
                                          _par=None, _execute=mock.Mock(return_value=count))
                    out = StringIO()
                    with mock.patch.object(BLD.time, "perf_counter", side_effect=(100, 280)), redirect_stdout(out):
                        self.assertEqual(BLD._BuildingOverlayRun.execute(run), count)
                    self.assertIn(f"[Bld] Tile -34-071: building placements={count or 0:,};", out.getvalue())
                    self.assertIn("total generation time=180.0s", out.getvalue())

    def test_building_texture_helper_does_not_print_a_tile_summary(self):
        run = SimpleNamespace(verbose_log=False, _par=SimpleNamespace(helper=True),
                              _execute=mock.Mock(return_value=None))
        out = StringIO()
        with redirect_stdout(out):
            BLD._BuildingOverlayRun.execute(run)
        self.assertEqual(out.getvalue(), "")

    def _cached_vegetation_run(self, verbose, trees, grass_count):
        with tempfile.TemporaryDirectory() as tmp:
            name = "34336_26384_BI16.dds"
            classes = np.zeros((32, 32), dtype=np.int8)
            if trees:
                classes[8:24, 8:24] = SEG.CLASS_TREE
            np.save(Path(tmp) / name.replace(".dds", "_veg.npy"), classes)
            text = []
            clock = [0.0]
            output = Path(tmp) / "overlay" / "Earth nav data" / "+40+000" / "+47+008.dsf"

            def collect_fixture(*_args):
                clock[0] += 60.0
                return [name], "dds", None

            def compile_fixture(source, target):
                text.append(Path(source).read_text(encoding="utf-8"))
                Path(target).write_bytes(b"compiled fixture")
                clock[0] += 120.0
                return True

            def grass_fixture(offset):
                # Seven polygons may share one forest definition.
                body = (f"BEGIN_POLYGON {offset} 230 2\nBEGIN_WINDING\n"
                        "POLYGON_POINT 8.0 47.0\nPOLYGON_POINT 8.1 47.0\n"
                        "POLYGON_POINT 8.1 47.1\nEND_WINDING\nEND_POLYGON\n")
                return ["grass/tile/c00.for"], body * grass_count, {"polygons": grass_count}

            hook = mock.Mock(sources={name: None})
            hook.wants.return_value = False
            hook.build.side_effect = grass_fixture
            out = StringIO()
            with (
                mock.patch.object(SEG, "collect_source_texture_files",
                                  side_effect=collect_fixture),
                mock.patch.object(VEG.time, "perf_counter", side_effect=lambda: clock[0]),
                mock.patch.object(TEX, "select_textures", return_value=([name], {}, {})),
                mock.patch.object(VEG, "dds_bounds", return_value=(47.1, 47.0, 8.0, 8.1)),
                mock.patch.object(SEG, "tree_support_enabled", return_value=False),
                mock.patch.object(SEG, "load_dds_or_none", return_value=np.zeros((32, 32, 3), np.uint8)),
                mock.patch.object(VEG.LOWZOOM, "tree_mask", return_value=(classes == SEG.CLASS_TREE).astype(np.uint8)),
                mock.patch.object(SEG, "load_vegetation_model", side_effect=AssertionError("uncached inference")),
                mock.patch.object(VEG, "_load_mesh_water_index", return_value=None),
                mock.patch.object(VEG, "_load_osm_roads", return_value=[]),
                mock.patch.object(SEG, "compile_dsf", side_effect=compile_fixture),
                mock.patch.object(VEG, "validate_compiled_overlay", return_value=None),
                mock.patch.object(GRASS, "GrassHook", return_value=hook),
                mock.patch.object(GRASS, "remove_tile_assets"),
                redirect_stdout(out),
            ):
                count = VEG.run(tmp, 47, 8, str(output), tmp,
                                10, 3, 50, 3, False, verbose_log=verbose,
                                use_simheaven=False, use_simheaven_asset_proximity=False,
                                avoid_simheaven_buildings=False, avoid_simheaven_forests=False,
                                use_leaf_type_maps=False, use_xp_forests=False, bld_excl_m=0,
                                grass_settings=GRASS.GrassSettings() if grass_count else None)
            # Trees and grass are separate packs side by side.
            grass_output = (Path(tmp) / VEG.GRASS_PACK_NAME / "Earth nav data"
                            / "+40+000" / "+47+008.dsf")
            for path, wanted in ((output, trees), (grass_output, grass_count > 0)):
                if wanted:
                    self.assertEqual(path.read_bytes(), b"compiled fixture")
                else:
                    self.assertFalse(path.exists())
            grass_sources = [source for source in text if "grass/" in source]
            self.assertEqual(len(grass_sources), int(grass_count > 0))
            self.assertEqual(len(text) - len(grass_sources), int(trees))
            for source in grass_sources:
                self.assertNotIn("sim/exclude", source)
                self.assertEqual(source.count("BEGIN_POLYGON 0 230 2"), grass_count)
            return count, text, out.getvalue()

    def test_complete_vegetation_tile_summary_and_output_in_both_modes(self):
        for trees, grass_count in ((True, 0), (True, 7), (False, 7), (False, 0)):
            generated = []
            for verbose in (False, True):
                with self.subTest(trees=trees, grass_count=grass_count, verbose=verbose):
                    count, text, log = self._cached_vegetation_run(verbose, trees, grass_count)
                    self.assertEqual(count > 0, trees)
                    self.assertIn(f"[Veg] Tile +47+008: tree polygons={count:,}; grass polygons={grass_count};", log)
                    self.assertEqual(log.count("[Veg] Tile +47+008:"), 1)
                    elapsed = 60.0 + 120.0 * trees + 120.0 * (grass_count > 0)
                    self.assertIn(f"total generation time={elapsed:.1f}s", log)
                    self.assertIn("(cached)", log)
                    self.assertEqual(f"DSF compiled → " in log.replace("Grass DSF compiled", ""), trees)
                    self.assertEqual("Grass DSF compiled" in log, grass_count > 0)
                    self.assertEqual("[Veg timing]" in log, verbose and trees)
                    self.assertEqual("vegetation context:" in log, verbose)
                    generated.append((count, text))
            self.assertEqual(generated[0], generated[1])

    def test_inference_thread_uses_its_runs_toggle_and_restores_context(self):
        stage = BLD._TextureInference(SimpleNamespace(verbose_log=False), 1, "fixture.dds", None)

        def infer():
            verbose_print("batch diagnostics")
            print("inference failed")
            raise RuntimeError("fixture")

        with ThreadPoolExecutor(max_workers=1) as pool, mock.patch.object(stage, "_start", side_effect=infer):
            for verbose in (False, True):
                stage.run.verbose_log = verbose
                out = StringIO()
                with redirect_stdout(out):
                    with self.assertRaisesRegex(RuntimeError, "fixture"):
                        pool.submit(stage.execute).result()
                    pool.submit(verbose_print, "outside overlay").result()
                self.assertEqual("batch diagnostics" in out.getvalue(), verbose)
                self.assertIn("inference failed", out.getvalue())
                self.assertIn("outside overlay", out.getvalue())

    def test_successful_dsftool_stdout_is_verbose_but_failures_stay_visible(self):
        @with_verbose_logging
        def compile_overlay(verbose_log=False):
            return SEG.compile_dsf("source.txt", "output/tile.dsf")

        for rc in (0, 1):
            for verbose in (False, True):
                out = StringIO()
                result = SimpleNamespace(returncode=rc, stdout="compiler details", stderr="warning")
                with (mock.patch.object(SEG.subprocess, "run", return_value=result),
                      mock.patch.object(SEG.os, "makedirs"), redirect_stdout(out)):
                    self.assertEqual(compile_overlay(verbose), rc == 0)
                self.assertEqual("compiler details" in out.getvalue(), verbose or rc != 0)
                self.assertIn("warning", out.getvalue())

    def test_quiet_batch_message_does_not_suppress_later_verbose_run(self):
        @with_verbose_logging
        def announce(verbose_log=False):
            SEG._print_batch_size_once("batch=4")

        out = StringIO()
        with mock.patch.object(SEG, "_last_batch_size_message", None), redirect_stdout(out):
            announce(False)
            self.assertEqual(out.getvalue(), "")
            announce(True)
            announce(True)
        self.assertEqual(out.getvalue(), "batch=4\n")

    def test_leaf_type_failures_survive_quiet_callback(self):
        @with_verbose_logging
        def resolve(verbose_log=False):
            source = mock.Mock(side_effect=OSError("source unreadable"))
            return LEAF.TileLeafTypes(47, 8, "unused", grid_sources=(("fixture", source),),
                                         log=verbose_print, warning_log=print)

        out = StringIO()
        with redirect_stdout(out):
            resolve()
        self.assertIn("source unreadable", out.getvalue())
        self.assertNotIn("chain:", out.getvalue())

    def test_pipeline_gates_settings_and_passes_both_toggles_to_subprocess(self):
        for kind in ("bld", "veg"):
            for verbose in (False, True):
                with self.subTest(kind=kind, verbose=verbose), tempfile.TemporaryDirectory() as tmp:
                    out = StringIO()
                    with ExitStack() as stack:
                        for name, value in {
                            "_check_tile_imagery": True, "_auto_setup": True,
                            "_overlay_cache_dir": tmp, "_dsftool_path": "dsftool",
                            "_dsf_output_path": str(Path(tmp) / "out.dsf"),
                            "_scenery_paths": tmp, "_resolve_remote_host": None,
                            "_cleanup_derived_cache": None, "_finish_veg_tile_replacement": None,
                        }.items():
                            stack.enter_context(mock.patch.object(PIPE, name, return_value=value))
                        stack.enter_context(mock.patch.object(PIPE, f"{kind}_verbose_log", verbose))
                        stack.enter_context(mock.patch.object(PIPE, f"{kind}_disable_cache", False))
                        runner = stack.enter_context(mock.patch.object(PIPE, "_run_venv", return_value=0))
                        stack.enter_context(redirect_stdout(out))
                        getattr(PIPE, f"process_{kind}_tile")(47, 8, tmp)
                    code = runner.call_args.args[0]
                    compile(code, "overlay subprocess", "exec")
                    calls = [n for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Call)
                             and isinstance(n.func, ast.Attribute) and n.func.attr == "run"]
                    arg = next(k.value for k in calls[0].keywords if k.arg == "verbose_log")
                    self.assertIs(ast.literal_eval(arg), verbose)
                    self.assertEqual("Effective settings:" in out.getvalue(), verbose)
                    self.assertIn("Starting for tile", out.getvalue())
                    self.assertIn("Done for tile", out.getvalue())

    def test_vegetation_toggle_is_configured_for_gui_and_batch(self):
        self.assertIs(CFG.cfg_vars["veg_verbose_log"]["default"], False)
        self.assertIn("veg_verbose_log", CFG.list_veg_vars)
        self.assertIn("veg_verbose_log", CFG.list_veg_pipeline_vars)

    def test_vegetation_cli_accepts_verbose_flag(self):
        with mock.patch("sys.argv", ["veg", "textures", "47", "8", "--verbose-log"]):
            self.assertTrue(VEG.parse_args().verbose_log)


if __name__ == "__main__":
    unittest.main()
