import copy
import os
import time
import shutil
import queue
import threading
from collections import defaultdict
import O4_UI_Utils as UI
import O4_Cfg_Vars as CFGVARS
import O4_File_Names as FNAMES
import O4_Imagery_Utils as IMG
import O4_Color_Build as CCB
import O4_Vector_Map as VMAP
import O4_Mesh_Utils as MESH
import O4_Mask_Utils as MASK
import O4_DSF_Utils as DSF
import O4_Overlay_Utils as OVL
import O4_Overlay_Pipeline as PIPELINE
import O4_Scenery_Links as SLINK
import O4_Build_State as BSTATE
import O4_Terrain_Materials as TMAT
from O4_Parallel_Utils import parallel_launch, parallel_join

max_download_slots = 1
max_convert_slots = 4
skip_downloads = False
skip_converts = False

################################################################################
def download_textures(
    tile,
    download_queue,
    convert_queue,
    workers=None,
    producer_done_event=None,
):
    worker_count = max(1, workers or max_download_slots)
    UI.vprint(1, f"-> Opening download queue with {worker_count} worker(s).")

    progress_lock = threading.Lock()
    progress_state = {"done": 0, "failed": 0, "pending": 0}
    attempts = defaultdict(int)
    interrupted = False
    max_attempts = 3
    stop_worker = object()

    def _update_progress_locked():
        completed = progress_state["done"] + progress_state["failed"]
        denom = (
            completed
            + progress_state["pending"]
            + download_queue.qsize()
        )
        UI.progress_bar(2, int(100 * completed / denom) if denom else 100)

    def _download_task(*attrs):
        nonlocal interrupted

        UI.check_pause()
        if UI.red_flag:
            interrupted = True
            return 0

        attrs = tuple(attrs)
        with progress_lock:
            progress_state["pending"] += 1
            _update_progress_locked()

        try:
            # A texture colour-corrected from its existing DDS needs no download.
            ok = CCB.uses_existing_dds(tile, attrs) or IMG.build_jpeg_ortho(tile, *attrs)
        except Exception as err:
            UI.vprint(2, f"Download failed: {err}")
            ok = 0

        should_retry = False
        with progress_lock:
            progress_state["pending"] -= 1
            if ok:
                progress_state["done"] += 1
                attempts.pop(attrs, None)
            else:
                attempt = attempts[attrs] + 1
                attempts[attrs] = attempt
                should_retry = attempt < max_attempts and not UI.red_flag
                if not should_retry:
                    attempts.pop(attrs, None)
                    progress_state["failed"] += 1
            _update_progress_locked()

        if ok:
            convert_queue.put((tile, *attrs))
        elif should_retry:
            download_queue.put(attrs)
            with progress_lock:
                _update_progress_locked()

        if UI.red_flag:
            interrupted = True

        return 1 if ok else 0

    def _download_worker():
        while True:
            attrs = download_queue.get()
            try:
                if attrs is stop_worker:
                    return
                _download_task(*attrs)
            finally:
                download_queue.task_done()

    if producer_done_event is None:
        producer_done_event = threading.Event()
        producer_done_event.set()

    workers_list = []
    for worker_idx in range(worker_count):
        worker = threading.Thread(
            target=_download_worker,
            name=f"texture-download-{worker_idx + 1}",
        )
        worker.start()
        workers_list.append(worker)

    # The producer can continue adding textures while workers are active. Wait
    # for it to finish before joining the queue so no late work can arrive.
    producer_done_event.wait()

    # A retry is queued before its original task calls task_done(), keeping the
    # unfinished-task count non-zero across the handoff. This avoids the prior
    # empty()/pending polling race that could strand a retry behind sentinels.
    download_queue.join()

    for _ in range(worker_count):
        download_queue.put(stop_worker)

    download_queue.join()
    parallel_join(workers_list)

    UI.progress_bar(2, 100)
    if interrupted or UI.red_flag:
        UI.vprint(1, "Download process interrupted.")
        return 0
    if progress_state["done"]:
        UI.vprint(1, " *Download of textures completed.")
    if progress_state["failed"]:
        UI.vprint(
            1,
            f"WARNING: {progress_state['failed']} texture download(s) failed "
            f"after {max_attempts} attempts.",
        )
        return 0
    return 1

################################################################################
def _finalize_terrain_materials(tile):
    """Make the land .ter files and their XP12 material maps agree.

    Converted textures got their map during the conversion; this makes the
    maps of the textures Step 3 kept (existing tiles), drops the lines of any
    .ter whose map cannot be made, and deletes maps no .ter uses any more.
    After a Stop it only drops the lines of the .ter files without a map, as
    the DSF is activated all the same.
    """
    try:
        if not TMAT.enabled(tile):
            n = 0 if tile.grouped else TMAT.prune_tile(tile.build_dir)
            if n:
                UI.vprint(1, f"   Terrain material maps: removed {n} (terrain_materials is off).")
            return
        timer = time.time()
        stats = TMAT.finalize_tile(
            tile.build_dir,
            TMAT.relief_of(tile),
            IMG.dds_convert_cmd,
            workers=max_convert_slots,
            prune=not tile.grouped and not UI.red_flag,
            make=not UI.red_flag,
            should_stop=lambda: UI.red_flag,
            log=lambda msg: UI.vprint(1, msg),
        )
        UI.vprint(
            1,
            f" *Terrain material maps: {stats['made']} made, {stats['kept']} up to date"
            + (f", {stats['removed']} .ter left without one" if stats["removed"] else "")
            + (f", {stats['failed']} could not be remade" if stats["failed"] else "")
            + (f", {stats['pruned']} unused removed" if stats["pruned"] else "")
            + f" ({time.time() - timer:.0f}s).",
        )
    except Exception as e:
        UI.lvprint(1, f"WARNING: terrain material maps failed ({e}).")


def build_tile(tile):
    if UI.is_working:
        return 0
    UI.is_working = 1
    UI.red_flag = False
    UI.logprint(
        "Step 3 for tile lat=", tile.lat, ", lon=", tile.lon, ": starting."
    )
    UI.vprint(
        0,
        "\nStep 3 : Building DSF/Imagery for tile "
        + FNAMES.short_latlon(tile.lat, tile.lon)
        + " : \n--------\n",
    )

    if not os.path.isfile(FNAMES.mesh_file(tile.build_dir, tile.lat, tile.lon)):
        UI.lvprint(
            0, "ERROR: A mesh file must first be constructed for the tile!"
        )
        UI.exit_message_and_bottom_line("")
        return 0

    timer = time.time()

    tile.write_to_config()

    if not IMG.initialize_local_combined_providers_dict(tile):
        UI.exit_message_and_bottom_line("")
        return 0

    try:
        if not os.path.exists(
            os.path.join(
                tile.build_dir,
                "Earth nav data",
                FNAMES.round_latlon(tile.lat, tile.lon),
            )
        ):
            os.makedirs(
                os.path.join(
                    tile.build_dir,
                    "Earth nav data",
                    FNAMES.round_latlon(tile.lat, tile.lon),
                )
            )
        if not os.path.isdir(os.path.join(tile.build_dir, "textures")):
            os.makedirs(os.path.join(tile.build_dir, "textures"))
        if UI.cleaning_level > 1 and not tile.grouped:
            for f in os.listdir(os.path.join(tile.build_dir, "textures")):
                if f[-4:] != ".png":
                    continue
                try:
                    os.remove(os.path.join(tile.build_dir, "textures", f))
                except:
                    pass
        if not tile.grouped:
            try:
                shutil.rmtree(os.path.join(tile.build_dir, "terrain"))
            except:
                pass
        if not os.path.isdir(os.path.join(tile.build_dir, "terrain")):
            os.makedirs(os.path.join(tile.build_dir, "terrain"))
    except Exception as e:
        UI.lvprint(0, "ERROR: Cannot create tile subdirectories.")
        UI.vprint(3, e)
        UI.exit_message_and_bottom_line("")
        return 0

    download_queue = queue.Queue()
    convert_queue = queue.Queue()

    download_launched = False
    convert_launched = False
    download_workers = max_download_slots

    build_dsf_thread = threading.Thread(
        target=DSF.build_dsf, args=[tile, download_queue]
    )
    producer_done_event = threading.Event()

    download_thread = threading.Thread(
        target=download_textures,
        args=[
            tile,
            download_queue,
            convert_queue,
            download_workers,
            producer_done_event,
        ],
    )
    # Colour correction fits the whole tile before anything is converted, so
    # the converters wait for the downloads instead of overlapping with them.
    color_correction = not skip_downloads and not skip_converts and CCB.enabled(tile)
    build_dsf_thread.start()
    if not skip_downloads:
        download_thread.start()
        download_launched = True
        if not skip_converts and not color_correction:
            UI.vprint(
                1,
                "-> Opening convert queue and",
                max_convert_slots,
                "conversion workers.",
            )
            dico_conv_progress = {"done": 0, "bar": 3}
            convert_workers = parallel_launch(
                IMG.convert_texture,
                convert_queue,
                max_convert_slots,
                progress=dico_conv_progress,
            )
            convert_launched = True
    build_dsf_thread.join()
    producer_done_event.set()
    if download_launched:
        download_thread.join()
        if color_correction and not UI.red_flag:
            pending = []
            while True:
                try:
                    pending.append(convert_queue.get_nowait())
                except queue.Empty:
                    break
            # Textures corrected from their existing DDS have no orthophoto for
            # the normal converter: if the correction failed they keep their DDS.
            keep_dds = {tuple(item[1:]) for item in pending
                        if CCB.uses_existing_dds(tile, tuple(item[1:]))}
            done = CCB.run(tile, [tuple(item[1:]) for item in pending], max_convert_slots)
            pending = [item for item in pending
                       if tuple(item[1:]) not in done and tuple(item[1:]) not in keep_dds]
            if pending and not UI.red_flag:
                UI.vprint(1, "-> Converting", len(pending), "remaining texture(s) with",
                          max_convert_slots, "conversion workers.")
                dico_conv_progress = {"done": 0, "bar": 3}
                convert_workers = parallel_launch(
                    IMG.convert_texture,
                    convert_queue,
                    max_convert_slots,
                    progress=dico_conv_progress,
                )
                convert_launched = True
                for item in pending:
                    convert_queue.put(item)
        if convert_launched:
            for _ in range(max_convert_slots):
                convert_queue.put("quit")
            parallel_join(convert_workers)
            if UI.red_flag:
                UI.vprint(1, "DDS conversion process interrupted.")
            elif dico_conv_progress["done"] >= 1:
                UI.vprint(1, " *DDS conversion of textures completed.")
    _finalize_terrain_materials(tile)
    UI.vprint(1, " *Activating DSF file.")
    dsf_file_name = os.path.join(
        tile.build_dir,
        "Earth nav data",
        FNAMES.long_latlon(tile.lat, tile.lon) + ".dsf",
    )
    try:
        os.replace(dsf_file_name + ".tmp", dsf_file_name)
    except:
        UI.vprint(0, "ERROR : could not rename DSF file, tile is not active.")
    if UI.red_flag:
        UI.exit_message_and_bottom_line()
        return 0
    if UI.cleaning_level > 1:
        try:
            os.remove(FNAMES.alt_file(tile))
        except:
            pass
        try:
            os.remove(FNAMES.input_node_file(tile))
        except:
            pass
        try:
            os.remove(FNAMES.input_poly_file(tile))
        except:
            pass
    if UI.cleaning_level > 2:
        try:
            os.remove(FNAMES.mesh_file(tile.build_dir, tile.lat, tile.lon))
        except:
            pass
        try:
            os.remove(FNAMES.apt_file(tile))
        except:
            pass
    if UI.cleaning_level > 1 and not tile.grouped:
        remove_unwanted_textures(tile)
    SLINK.auto_link_tile(tile)
    UI.timings_and_bottom_line(timer)
    UI.logprint(
        "Step 3 for tile lat=", tile.lat, ", lon=", tile.lon, ": normal exit."
    )
    return 1

################################################################################
def build_all(tile):
    VMAP.build_poly_file(tile)
    if UI.stop_requested():
        UI.exit_message_and_bottom_line("")
        return 0
    MESH.build_mesh(tile)
    if UI.stop_requested():
        UI.exit_message_and_bottom_line("")
        return 0
    MASK.build_masks(tile)
    if UI.stop_requested():
        UI.exit_message_and_bottom_line("")
        return 0
    build_tile(tile)
    tile_coords = FNAMES.short_latlon(tile.lat, tile.lon)
    if tile_coords in IMG.incomplete_imgs:
        UI.lvprint(
            1,
            f"Attempting to rebuild textures with white squares: "
            f"{IMG.incomplete_imgs[tile_coords]}",
        )
        delete_incomplete_imgs(tile)
        build_tile(tile)
    if UI.red_flag:
        UI.exit_message_and_bottom_line("")
        return 0
    UI.is_working = 0
    if IMG.incomplete_imgs:
        UI.lvprint(
            0,
            f"\nERROR: Parts of the following images could not be obtained "
            f"and have been filled with white: {IMG.incomplete_imgs}",
        )
    return 1

################################################################################
def _clear_todo_marker(lat, lon):
    """Drop the red 'to do' square once a tile needs no further work."""
    try:
        UI.gui.earth_window.canvas.delete(
            UI.gui.earth_window.dico_tiles_todo[(lat, lon)]
        )
        UI.gui.earth_window.dico_tiles_todo.pop((lat, lon), None)
    except:
        pass

################################################################################
def _batch_pipelining():
    """Run a tile's overlays while the batch builds the next tile (O4_BATCH_PIPELINE=0: don't)."""
    return os.environ.get("O4_BATCH_PIPELINE", "").strip().lower() not in ("0", "false", "no", "off")


def _build_tile_overlays(tile, do_bld, do_veg, link):
    """Building and/or vegetation overlays of one tile, then link and clear it."""
    lat, lon = tile.lat, tile.lon
    name = FNAMES.short_latlon(lat, lon)
    try:
        if do_bld:
            UI.lvprint(0, f"\nBuilding overlay for {name} :\n--------\n")
            PIPELINE.apply_tile_settings(tile, CFGVARS.list_bld_pipeline_vars)
        if do_veg:
            UI.lvprint(0, f"\nVegetation overlay for {name} :\n--------\n")
            PIPELINE.apply_tile_settings(tile, CFGVARS.list_veg_pipeline_vars)
        errors = PIPELINE.process_overlays_tile(
            lat, lon, tile.build_dir, do_bld=do_bld, do_veg=do_veg,
        )
    except Exception as exc:
        UI.lvprint(0, f"[Overlay] Overlays failed for {name}: {exc}")
        return
    if UI.red_flag:
        return
    if "bld" in errors:
        done = errors["bld"] is None
        if not done:
            UI.lvprint(0, f"[Overlay] Building overlay failed for {name}: {errors['bld']}")
        if done:
            BSTATE.mark_done(lat, lon, "bld_overlay")
    if "veg" in errors:
        done = errors["veg"] is None
        if not done:
            UI.lvprint(0, f"[Overlay] Vegetation overlay failed for {name}: {errors['veg']}")
        if done:
            BSTATE.mark_done(lat, lon, "veg_overlay")
    if link:
        # build_tile() links the tile itself; catch the runs which only
        # refreshed overlays over an already built tile.
        SLINK.auto_link_tile(tile)
    _clear_todo_marker(lat, lon)


class _TileOverlays:
    """One tile's overlays in a background thread while the batch builds the next tile.

    The overlays are GPU and CPU work in their own processes; the next
    tile's OSM, mesh, mask and DSF steps are mostly downloads and other CPU
    work, so the two overlap well. Only one tile's overlays run at a time,
    and they get their own copy of the tile settings.
    """

    def __init__(self):
        self.thread = None

    def start(self, tile, do_bld, do_veg, link):
        self.wait()
        self.thread = threading.Thread(
            target=_build_tile_overlays,
            args=(copy.copy(tile), do_bld, do_veg, link),
            name="tile-overlays",
            daemon=True,
        )
        self.thread.start()
        if not _batch_pipelining():
            self.wait()

    def wait(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None


################################################################################
def _end_journal_on_stop():
    """Settle the journal of a batch that red_flag cut short.

    Stop means the user is done with the batch, so it is not offered for
    resume. Anything else that raised red_flag - closing the window, mid-build
    or while paused - leaves it resumable, as a crash does.
    """
    if UI.user_stopped:
        BSTATE.complete()
    else:
        BSTATE.set_status("stopped")

################################################################################
def build_tile_list(
    tile, list_lat_lon, do_osm, do_mesh, do_mask, do_dsf, do_ovl,
    do_bld_overlay=False, do_veg_overlay=False, override_cfg=False, resume_path=None
):
    try:
        return _build_tile_list(
            tile, list_lat_lon, do_osm, do_mesh, do_mask, do_dsf, do_ovl,
            do_bld_overlay, do_veg_overlay, override_cfg, resume_path,
        )
    finally:
        # However this ended - finished, stopped, or an error nobody caught -
        # the journal is no longer live work, so this instance (or another)
        # can offer to resume it.
        BSTATE.release()

################################################################################
def _build_tile_list(
    tile, list_lat_lon, do_osm, do_mesh, do_mask, do_dsf, do_ovl,
    do_bld_overlay=False, do_veg_overlay=False, override_cfg=False, resume_path=None
):
    if UI.is_working:
        return 0
    UI.red_flag = 0
    UI.user_stopped = False
    timer = time.time()
    # The journal lets a batch that was stopped (or that died with the app)
    # pick up at the exact step it reached rather than rebuild from scratch.
    steps = {
        "osm": bool(do_osm),
        "mesh": bool(do_mesh),
        "mask": bool(do_mask),
        "dsf": bool(do_dsf),
        "ovl": bool(do_ovl),
        "bld_overlay": bool(do_bld_overlay),
        "veg_overlay": bool(do_veg_overlay),
    }
    BSTATE.begin(
        list_lat_lon,
        steps,
        tile.custom_build_dir,
        override_cfg,
        resume_path=resume_path,
    )
    UI.lvprint(
        0,
        "Batch build" + (" resumed" if resume_path else " launched"),
        "for a number of",
        len(list_lat_lon),
        "tiles.",
    )
    overlays = _TileOverlays()
    try:
        return _build_tiles(
            tile, list_lat_lon, do_osm, do_mesh, do_mask, do_dsf, do_ovl,
            do_bld_overlay, do_veg_overlay, override_cfg, steps, timer, overlays,
        )
    finally:
        overlays.wait()


def _build_tiles(
    tile, list_lat_lon, do_osm, do_mesh, do_mask, do_dsf, do_ovl,
    do_bld_overlay, do_veg_overlay, override_cfg, steps, timer, overlays,
):
    k = 0
    for (lat, lon) in list_lat_lon:
        k += 1
        UI.check_pause()
        if UI.red_flag:
            _end_journal_on_stop()
            UI.exit_message_and_bottom_line()
            return 0
        if any(steps.values()) and all(
            BSTATE.is_done(lat, lon, step)
            for step, wanted in steps.items()
            if wanted
        ):
            UI.vprint(
                1,
                "Skipping tile",
                FNAMES.short_latlon(lat, lon),
                "- already built by the batch being resumed.",
            )
            _clear_todo_marker(lat, lon)
            continue
        UI.vprint(
            1,
            "Dealing with tile ",
            k,
            "/",
            len(list_lat_lon),
            ":",
            FNAMES.short_latlon(lat, lon),
        )
        UI.lvprint(1, f"[Batch] Steps: osm={do_osm} mesh={do_mesh} mask={do_mask} "
                   f"dsf={do_dsf} ovl={do_ovl} "
                   f"veg_overlay={do_veg_overlay} bld_overlay={do_bld_overlay}")
        (tile.lat, tile.lon) = (lat, lon)
        tile.build_dir = FNAMES.build_dir(
            tile.lat, tile.lon, tile.custom_build_dir
        )
        tile.dem = None
        if override_cfg:
            tile.read_from_config(use_global=True)
        else:
            tile.read_from_config()
        if do_osm or do_mesh or do_dsf:
            tile.make_dirs()
        if do_osm and not BSTATE.is_done(lat, lon, "osm"):
            UI.check_pause()
            done = VMAP.build_poly_file(tile)
            if UI.red_flag:
                _end_journal_on_stop()
                UI.exit_message_and_bottom_line()
                return 0
            if done:
                BSTATE.mark_done(lat, lon, "osm")
        if do_mesh and not BSTATE.is_done(lat, lon, "mesh"):
            UI.check_pause()
            done = MESH.build_mesh(tile)
            if UI.red_flag:
                _end_journal_on_stop()
                UI.exit_message_and_bottom_line()
                return 0
            if done:
                BSTATE.mark_done(lat, lon, "mesh")
        if do_mask and not BSTATE.is_done(lat, lon, "mask"):
            UI.check_pause()
            done = MASK.build_masks(tile)
            if UI.red_flag:
                _end_journal_on_stop()
                UI.exit_message_and_bottom_line()
                return 0
            if done:
                BSTATE.mark_done(lat, lon, "mask")
        if do_dsf and not BSTATE.is_done(lat, lon, "dsf"):
            UI.check_pause()
            tile_coords = FNAMES.short_latlon(lat, lon)
            done = build_tile(tile)
            if tile_coords in IMG.incomplete_imgs:
                UI.lvprint(
                    1,
                    f"Attempting to rebuild textures with white squares: "
                    f"{IMG.incomplete_imgs[tile_coords]}",
                )
                delete_incomplete_imgs(tile)
                done = build_tile(tile)
            if UI.red_flag:
                _end_journal_on_stop()
                UI.exit_message_and_bottom_line()
                return 0
            if done:
                BSTATE.mark_done(lat, lon, "dsf")
        if do_ovl and not BSTATE.is_done(lat, lon, "ovl"):
            UI.check_pause()
            done = OVL.build_overlay(lat, lon)
            if UI.red_flag:
                _end_journal_on_stop()
                UI.exit_message_and_bottom_line()
                return 0
            if done:
                BSTATE.mark_done(lat, lon, "ovl")
        want_bld = do_bld_overlay and not BSTATE.is_done(lat, lon, "bld_overlay")
        want_veg = do_veg_overlay and not BSTATE.is_done(lat, lon, "veg_overlay")
        # One tile's overlays at a time: wait for the previous tile's, which
        # ran while this tile's steps above were built.
        overlays.wait()
        if UI.red_flag:
            _end_journal_on_stop()
            UI.exit_message_and_bottom_line()
            return 0
        if want_bld or want_veg:
            UI.check_pause()
            overlays.start(tile, want_bld, want_veg, link=not do_dsf)
        else:
            if not do_dsf:
                SLINK.auto_link_tile(tile)
            _clear_todo_marker(lat, lon)
    overlays.wait()
    if UI.red_flag:
        _end_journal_on_stop()
        UI.exit_message_and_bottom_line()
        return 0
    BSTATE.complete()
    UI.lvprint(
        0, "Batch process completed in", UI.nicer_timer(time.time() - timer)
    )
    if IMG.incomplete_imgs:
        UI.lvprint(
            0,
            f"\nERROR: Parts of the following images could not be obtained "
            f"and have been filled with white: {IMG.incomplete_imgs}",
        )
    return 1

################################################################################
def remove_unwanted_textures(tile):
    texture_list = []
    for f in os.listdir(os.path.join(tile.build_dir, "terrain")):
        if f[-4:] != ".ter":
            continue
        if f[-5] == "y":  # water overlay
            texture_list.append("_".join(f[:-4].split("_")[:-2]) + ".dds")
        if f[-5] == "a":  # sea
            texture_list.append("_".join(f[:-4].split("_")[:-1]) + ".dds")
        else:
            texture_list.append(f.replace(".ter", ".dds"))
    for f in os.listdir(os.path.join(tile.build_dir, "textures")):
        if f[-4:] != ".dds" or TMAT.is_map_name(f):  # maps: TMAT.prune_tile
            continue
        if f not in texture_list:
            print("Removing obsolete texture", f)
            try:
                os.remove(os.path.join(tile.build_dir, "textures", f))
            except:
                pass

def delete_incomplete_imgs(tile):
    """Delete orthophoto jpegs and dds that have white squares."""
    tile_coords = FNAMES.short_latlon(tile.lat, tile.lon)
    if tile_coords not in IMG.incomplete_imgs:
        return
    file_name_list = IMG.incomplete_imgs[tile_coords]
    for file_name in file_name_list:
        # Delete the orthophoto jpegs with white squares
        for root, _, files in os.walk(FNAMES.Imagery_dir):
            if file_name in files:
                file_path = os.path.join(root, file_name)
                os.remove(file_path)
                UI.lvprint(1, f"Deleted: {file_name} in {file_path}")

        # Delete the tile dds textures with white squares
        # file_name has .jpg extension, so create a variable for .dds extension as well
        base_name, _ = os.path.splitext(file_name)
        file_name_dds = f"{base_name}.dds"
        for root, _, files in os.walk(tile.build_dir):
            if file_name_dds in files:
                file_path = os.path.join(root, file_name_dds)
                os.remove(file_path)
                UI.lvprint(1, f"Deleted: {file_name_dds} in {file_path}")

    IMG.incomplete_imgs.pop(tile_coords, None)
