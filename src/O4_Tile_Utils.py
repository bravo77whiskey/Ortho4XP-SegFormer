import copy
import os
import time
import shutil
import queue
import threading
from collections import defaultdict
import O4_UI_Utils as UI
import O4_Airport_Zones as AZONE
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
import O4_Progress as PROG
from O4_Parallel_Utils import parallel_launch, parallel_join

max_download_slots = 1
max_convert_slots = 4
skip_downloads = False
skip_converts = False

################################################################################
def _texture_name(attrs):
    """The DDS name of a texture, for the progress panel; never raises."""
    try:
        return FNAMES.dds_file_name_from_attributes(*attrs)
    except Exception:
        return " ".join(str(a) for a in attrs)


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
        PROG.items(part="Download", done=completed, total=denom)

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
                    if not UI.red_flag:
                        PROG.issue(
                            _texture_name(attrs),
                            "download", f"failed after {max_attempts} attempts",
                        )
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
            target=PROG.bind(_download_worker),
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


def _xplane_dir():
    import O4_Config_Utils as CFG
    from O4_Scenery_Packs import resolve_custom_scenery_dir

    custom_scenery_dir = resolve_custom_scenery_dir(CFG.custom_scenery_dir)
    return os.path.dirname(custom_scenery_dir) if custom_scenery_dir else ""


def plan_airport_zones(tile, list_lat_lon, settings_for):
    """Airport zones of every tile in list_lat_lon, {} when none are asked for.

    settings_for(lat, lon) -> (icaos, {zl: metres}) for a tile of the list.
    """
    xplane_dir = _xplane_dir()

    def runway_lookup(icaos):
        if not xplane_dir:
            UI.lvprint(
                1,
                "WARNING: custom_scenery_dir is not set: airport zones need the "
                "runways in X-Plane's apt.dat, none drawn for",
                ", ".join(icaos) + ".",
            )
            return {}
        return AZONE.runways_by_icao(
            xplane_dir,
            icaos,
            cache_dir=os.path.join(FNAMES.Overlay_cache_dir, "_airports"),
            log=lambda msg: UI.vprint(1, "   " + msg),
        )

    try:
        return AZONE.plan_zones(
            list_lat_lon, settings_for, runway_lookup,
            log=lambda msg: UI.vprint(1, msg),
        )
    except Exception as e:
        UI.lvprint(1, f"WARNING: airport zones could not be drawn ({e}).")
        return {}


def airport_tile_years(airports, years):
    """Which colour reference year each airport tile gets.

    An airport's year goes to every tile its zones reach. Where two airports
    with different years reach one tile, the airport the tile is home to
    wins, then the one listed first. Returns ({tile: year}, conflicts) with
    conflicts as (tile, (icao, year) used, (icao, year) not used).
    """
    owner = {}
    conflicts = []
    ordered = [(a, True) for a in airports] + [(a, False) for a in airports]
    for airport, homes_only in ordered:
        year = years.get(airport["icao"])
        if year is None:
            continue
        tiles = [airport["home"]] if homes_only else airport["tiles"]
        for tile in tiles:
            held = owner.setdefault(tile, (airport["icao"], year))
            if held[0] != airport["icao"] and held[1] != year:
                conflict = (tile, held, (airport["icao"], year))
                if conflict not in conflicts:
                    conflicts.append(conflict)
    return {tile: year for tile, (_icao, year) in owner.items()}, conflicts


def prepare_airport_batch(icaos, custom_build_dir, years=None):
    """The tiles some airports' zones reach, with configs ready to build them.

    icaos: one or more ICAO codes, comma- or space-separated. Each airport's
    own tile gets its code added to airport_zone_icaos, so its zones are
    drawn; every other reached tile without a config gets one from the global
    settings. Configs other tiles already have are left as they are, except
    for the colour reference year: years {icao: year} sets it on every tile an
    airport reaches (see airport_tile_years for tiles two airports share).

    Every code is checked before anything is written: one that apt.dat does
    not know raises ValueError (with a message for the user) and no config
    changes. Returns {"airports": [{"icao", "home", "tiles"}], "tiles",
    "center", "created", "written", "listed", "year_conflicts"}: "tiles" is
    every reached tile once, in airport order; "center" the tile in the middle
    of them; "written" every tile whose config was created or changed;
    "listed" the (icao, home tile) pairs newly added to a tile's airports.
    """
    import O4_Config_Utils as CFG

    codes = AZONE.parse_icaos(icaos)
    if not codes:
        raise ValueError("Type the ICAO code of an airport first.")
    xplane_dir = _xplane_dir()
    if not xplane_dir:
        raise ValueError(
            "custom_scenery_dir is not set: the runways are read from "
            "X-Plane's apt.dat."
        )
    runways = AZONE.runways_by_icao(
        xplane_dir,
        codes,
        cache_dir=os.path.join(FNAMES.Overlay_cache_dir, "_airports"),
        log=lambda msg: UI.vprint(1, "   " + msg),
    )
    unknown = [code for code in codes if not runways.get(code)]
    if unknown:
        raise ValueError(
            ", ".join(unknown)
            + (" has" if len(unknown) == 1 else " have")
            + " no land runway in apt.dat. Nothing was changed."
        )

    loaded = {}

    def load(lat, lon):
        if (lat, lon) not in loaded:
            tile = CFG.Tile(lat, lon, custom_build_dir)
            tile.read_from_config()
            cfg = os.path.join(
                tile.build_dir,
                "Ortho4XP_" + FNAMES.short_latlon(lat, lon) + ".cfg",
            )
            loaded[(lat, lon)] = (tile, os.path.isfile(cfg))
        return loaded[(lat, lon)]

    airports = []
    tiles = []
    for code in codes:
        home = AZONE.home_tile(runways[code])
        home_tile, _exists = load(*home)
        reached = AZONE.tiles_reached(runways[code], AZONE.distances_from(home_tile))
        for lat, lon in reached:
            load(lat, lon)
        airports.append({"icao": code, "home": home, "tiles": reached})
        tiles += [tile for tile in reached if tile not in tiles]
    if any(not exists and not tile.default_website
           for tile, exists in loaded.values()):
        raise ValueError(
            "Choose an imagery provider in the main window first: the new "
            "tile configs need one."
        )

    listed = []
    for airport in airports:
        tile, _exists = loaded[airport["home"]]
        listed_here = AZONE.parse_icaos(tile.airport_zone_icaos)
        if airport["icao"] not in listed_here:
            tile.airport_zone_icaos = ",".join(listed_here + [airport["icao"]])
            listed.append((airport["icao"], airport["home"]))
    changed = {home for _icao, home in listed}
    tile_years, year_conflicts = airport_tile_years(airports, years or {})
    for key, year in tile_years.items():
        tile, _exists = loaded[key]
        if int(tile.color_reference_year) != int(year):
            tile.color_reference_year = int(year)
            changed.add(key)
    created = []
    written = []
    for (lat, lon) in tiles:
        tile, exists = loaded[(lat, lon)]
        if exists and (lat, lon) not in changed:
            continue
        tile.make_dirs()
        if not tile.write_to_config():
            raise ValueError(
                "Could not write the config of tile "
                + FNAMES.short_latlon(lat, lon) + "."
            )
        written.append((lat, lon))
        if not exists:
            created.append((lat, lon))
    lats = [lat for lat, _lon in tiles]
    lons = [lon for _lat, lon in tiles]
    return {
        "airports": airports,
        "tiles": tiles,
        "center": ((min(lats) + max(lats)) // 2, (min(lons) + max(lons)) // 2),
        "created": created,
        "written": written,
        "listed": listed,
        "year_conflicts": year_conflicts,
    }


def _batch_airport_zone_plan(tile, list_lat_lon, override_cfg):
    """Airport zones for a whole batch, each tile's settings read as it will be built."""
    def settings_for(lat, lon):
        other = copy.copy(tile)
        (other.lat, other.lon) = (lat, lon)
        other.build_dir = FNAMES.build_dir(lat, lon, tile.custom_build_dir)
        other.read_from_config(use_global=override_cfg)
        return AZONE.parse_icaos(other.airport_zone_icaos), AZONE.distances_from(other)

    return plan_airport_zones(tile, list_lat_lon, settings_for)


def _airport_zones(tile):
    """This tile's airport zones as zone_list entries.

    Step 3 saves them in the tile config as airport_zone_list, which the
    overlays replay to know which textures the DSF uses.

    A batch plans them for all its tiles up front (tile.airport_zone_plan), so
    zones reach across tile edges; a tile built on its own only gets those of
    its own airports.
    """
    plan = getattr(tile, "airport_zone_plan", None)
    if plan is None:
        plan = plan_airport_zones(
            tile,
            [(tile.lat, tile.lon)],
            lambda lat, lon: (
                AZONE.parse_icaos(tile.airport_zone_icaos),
                AZONE.distances_from(tile),
            ),
        )
    zones = []
    by_icao = defaultdict(list)
    for coords, zl, icao in plan.get((tile.lat, tile.lon), []):
        if zl > tile.mesh_zl:
            UI.vprint(
                1,
                f"   Airport zones: {icao} ZL{zl} lowered to mesh_zl={tile.mesh_zl}.",
            )
            zl = tile.mesh_zl
        zones.append([coords, zl, tile.default_website])
        by_icao[icao].append(zl)
    for icao, zls in by_icao.items():
        UI.vprint(
            1,
            f"-> Airport zones from {icao}: "
            + ", ".join(f"ZL{zl}" for zl in zls) + ".",
        )
    return zones


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

    tile.airport_zone_list = _airport_zones(tile)
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
        target=PROG.bind(DSF.build_dsf), args=[tile, download_queue]
    )
    producer_done_event = threading.Event()

    download_thread = threading.Thread(
        target=PROG.bind(download_textures),
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
            # This is the work of the batch's Colour (and Clouds) steps: their
            # rows run from here, and those steps finish whatever is left.
            with PROG.ahead(tile.lat, tile.lon,
                            ("color", "clouds") if getattr(tile, "color_fill_clouds", False)
                            else ("color",)):
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
def build_color(tile, fill):
    """Batch step Colour correction (fill=False) or Cloud removal (fill=True).

    Works on the textures of a tile Step 3 has built: corrects every one not
    yet corrected to these settings, from its orthophoto, or from its DDS when
    the orthophoto was cleared. Cloud removal includes the correction, as the
    fill is colour-matched through the same fit. Returns 1 when every texture
    is done.
    """
    name = FNAMES.short_latlon(tile.lat, tile.lon)
    UI.vprint(0, f"\n{'Cloud removal' if fill else 'Colour correction'} for tile {name} :\n--------\n")
    why_not = CCB.unavailable_reason()
    if why_not:
        UI.lvprint(0, f"ERROR: Colour correction is not available: {why_not}.")
        return 0
    if CCB.tile_textures(tile) is None:
        UI.lvprint(0, "ERROR: Build imagery/DSF must first be run for the tile!")
        return 0
    try:
        todo = CCB.plan_tile(tile, fill)
        if not todo:
            UI.vprint(1, " *Every texture is already done.")
            return 1
        reasons = defaultdict(int)
        for _, reason, _ in todo:
            reasons[reason] += 1
        UI.vprint(1, f"-> {len(todo)} texture(s) to do: "
                     + ", ".join(f"{n} {r}" for r, n in sorted(reasons.items())) + ".")
        fetch = [attrs for attrs, _, download in todo if download]
        if fetch:
            # Already-corrected textures are redone from their orthophoto.
            download_queue = queue.Queue()
            for attrs in fetch:
                download_queue.put(attrs)
            download_textures(tile, download_queue, queue.Queue(), max_download_slots)
            if UI.red_flag:
                return 0
        attrs_list = [attrs for attrs, _, _ in todo if CCB.has_source(tile, attrs)]
        done = CCB.run(tile, attrs_list, max_convert_slots, fill=fill) if attrs_list else set()
        if UI.red_flag:
            return 0
        left = [attrs for attrs, _, _ in todo if attrs not in done]
        if left:
            UI.lvprint(1, f"WARNING: {len(left)} texture(s) could not be corrected and keep "
                          "their current imagery; retry them from the progress panel.")
            for attrs in left:
                PROG.issue(_texture_name(attrs),
                           "cloud removal" if fill else "colour correction",
                           "could not be corrected; keeps its current imagery")
        # Done with issues while anything was corrected; failed when nothing was.
        return 1 if len(left) < len(todo) else 0
    finally:
        CCB.forget(tile)

################################################################################
def build_all(tile):
    lat, lon = tile.lat, tile.lon
    PROG.begin_run([(lat, lon)], ["osm", "mesh", "mask", "dsf"], title="All in one")
    try:
        _step(lat, lon, "osm", VMAP.build_poly_file, tile)
        if UI.stop_requested():
            UI.exit_message_and_bottom_line("")
            return 0
        _step(lat, lon, "mesh", MESH.build_mesh, tile)
        if UI.stop_requested():
            UI.exit_message_and_bottom_line("")
            return 0
        _step(lat, lon, "mask", MASK.build_masks, tile)
        if UI.stop_requested():
            UI.exit_message_and_bottom_line("")
            return 0
        _step(lat, lon, "dsf", _build_tile_retrying_white, tile)
        if UI.red_flag:
            UI.exit_message_and_bottom_line("")
            return 0
        UI.is_working = 0
        _print_summary()
        return 1
    finally:
        PROG.end_run()

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
    with PROG.scope(PROG.tile_name(tile.lat, tile.lon), None):
        _build_tile_overlays_tagged(tile, do_bld, do_veg, link)


def _build_tile_overlays_tagged(tile, do_bld, do_veg, link):
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
    do_bld_overlay=False, do_veg_overlay=False, override_cfg=False, resume_path=None,
    do_color=False, do_clouds=False, retry=None, tile_steps=None,
):
    """Build the steps asked for on every tile of the list.

    retry: {(lat, lon): step keys} - only these steps of these tiles run (the
    progress panel's Retry); the batch's other steps count as done already.
    tile_steps: {(lat, lon): step keys} - each tile builds only its own steps
    (an airport batch); the progress panel shows only those.
    """
    try:
        return _build_tile_list(
            tile, list_lat_lon, do_osm, do_mesh, do_mask, do_dsf, do_ovl,
            do_bld_overlay, do_veg_overlay, override_cfg, resume_path,
            do_color, do_clouds, retry, tile_steps,
        )
    finally:
        # However this ended - finished, stopped, or an error nobody caught -
        # the journal is no longer live work, so this instance (or another)
        # can offer to resume it.
        BSTATE.release()
        PROG.end_run()

################################################################################
def _build_tile_list(
    tile, list_lat_lon, do_osm, do_mesh, do_mask, do_dsf, do_ovl,
    do_bld_overlay=False, do_veg_overlay=False, override_cfg=False, resume_path=None,
    do_color=False, do_clouds=False, retry=None, tile_steps=None,
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
        "color": bool(do_color),
        "clouds": bool(do_clouds),
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
        tile_steps=tile_steps,
    )
    only = retry if retry is not None else tile_steps
    if only is not None:
        for (lat, lon) in list_lat_lon:
            redo = only.get((lat, lon), ())
            for step, wanted in steps.items():
                if wanted and step not in redo:
                    BSTATE.leave_out(lat, lon, step)
    PROG.begin_run(
        list_lat_lon,
        [step for step in BSTATE.STEPS if steps[step]],
        title="Retry" if retry is not None else "Batch",
        launch={"custom_build_dir": tile.custom_build_dir, "override_cfg": bool(override_cfg)},
        tile_steps=tile_steps,
    )
    UI.lvprint(
        0,
        "Retry" if retry is not None else
        "Batch build" + (" resumed" if resume_path else " launched"),
        "for a number of",
        len(list_lat_lon),
        "tiles.",
    )
    overlays = _TileOverlays()
    if do_dsf:
        # Zones of an airport near a tile edge reach into the neighbouring
        # tiles of the batch, so they are planned for all tiles at once.
        tile.airport_zone_plan = _batch_airport_zone_plan(
            tile, list_lat_lon, override_cfg
        )
    try:
        return _build_tiles(
            tile, list_lat_lon, do_osm, do_mesh, do_mask, do_dsf, do_ovl,
            do_bld_overlay, do_veg_overlay, override_cfg, steps, timer, overlays,
            do_color, do_clouds,
        )
    finally:
        tile.__dict__.pop("airport_zone_plan", None)
        overlays.wait()


def _build_tiles(
    tile, list_lat_lon, do_osm, do_mesh, do_mask, do_dsf, do_ovl,
    do_bld_overlay, do_veg_overlay, override_cfg, steps, timer, overlays,
    do_color=False, do_clouds=False,
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
                "- nothing left to build for it in this batch.",
            )
            PROG.tile_skipped(lat, lon)
            _clear_todo_marker(lat, lon)
            continue
        for step, wanted in steps.items():
            if wanted and BSTATE.is_done(lat, lon, step):
                PROG.step_skipped(lat, lon, step)
        UI.vprint(
            1,
            "Dealing with tile ",
            k,
            "/",
            len(list_lat_lon),
            ":",
            FNAMES.short_latlon(lat, lon),
        )
        UI.lvprint(1, "[Batch] Steps for this tile: " + ", ".join(
            PROG.STEP_NAMES[step] for step, wanted in steps.items()
            if wanted and not BSTATE.is_done(lat, lon, step)
        ) + ".")
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
            done = _step(lat, lon, "osm", VMAP.build_poly_file, tile)
            if UI.red_flag:
                _end_journal_on_stop()
                UI.exit_message_and_bottom_line()
                return 0
            if done:
                BSTATE.mark_done(lat, lon, "osm")
        if do_mesh and not BSTATE.is_done(lat, lon, "mesh"):
            UI.check_pause()
            done = _step(lat, lon, "mesh", MESH.build_mesh, tile)
            if UI.red_flag:
                _end_journal_on_stop()
                UI.exit_message_and_bottom_line()
                return 0
            if done:
                BSTATE.mark_done(lat, lon, "mesh")
        if do_mask and not BSTATE.is_done(lat, lon, "mask"):
            UI.check_pause()
            done = _step(lat, lon, "mask", MASK.build_masks, tile)
            if UI.red_flag:
                _end_journal_on_stop()
                UI.exit_message_and_bottom_line()
                return 0
            if done:
                BSTATE.mark_done(lat, lon, "mask")
        want_color = do_color and not BSTATE.is_done(lat, lon, "color")
        want_clouds = do_clouds and not BSTATE.is_done(lat, lon, "clouds")
        # Step 3 corrects inline when the colour steps follow it, so no
        # texture is converted twice; the steps below finish what it could not.
        tile.color_correction = want_color or want_clouds
        tile.color_fill_clouds = want_clouds
        # Step 3 links the tile itself; when it did not run here (not asked
        # for, or already done) the tile is linked once its other steps end.
        dsf_ran = do_dsf and not BSTATE.is_done(lat, lon, "dsf")
        if dsf_ran:
            UI.check_pause()
            done = _step(lat, lon, "dsf", _build_tile_retrying_white, tile)
            if UI.red_flag:
                _end_journal_on_stop()
                UI.exit_message_and_bottom_line()
                return 0
            if done:
                BSTATE.mark_done(lat, lon, "dsf")
        if want_color:
            UI.check_pause()
            # With Cloud removal also selected, one pass does both.
            done = _step(lat, lon, "color", build_color, tile, want_clouds)
            if UI.red_flag:
                _end_journal_on_stop()
                UI.exit_message_and_bottom_line()
                return 0
            if done:
                BSTATE.mark_done(lat, lon, "color")
        if want_clouds:
            UI.check_pause()
            done = _step(lat, lon, "clouds", build_color, tile, True)
            if UI.red_flag:
                _end_journal_on_stop()
                UI.exit_message_and_bottom_line()
                return 0
            if done:
                BSTATE.mark_done(lat, lon, "clouds")
        if do_ovl and not BSTATE.is_done(lat, lon, "ovl"):
            UI.check_pause()
            done = _step(lat, lon, "ovl", OVL.build_overlay, lat, lon)
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
            overlays.start(tile, want_bld, want_veg, link=not dsf_ran)
        else:
            if not dsf_ran:
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
    _print_summary()
    return 1


def _print_summary():
    summary = PROG.summary_lines()
    if summary:
        UI.lvprint(0, "\nIssues and failures (right-click them in the progress panel to retry):")
        for line in summary:
            UI.lvprint(0, line)


def _step(lat, lon, key, function, *args):
    """Run one step of a tile under the progress panel; returns its result."""
    with PROG.step(lat, lon, key) as scope:
        return scope.result(function(*args))


def _build_tile_retrying_white(tile):
    """Step 3, once more for textures that came back with white squares.

    Textures left white by an earlier build of this session are redone up
    front; any still white after the second build are the step's issues.
    """
    tile_coords = FNAMES.short_latlon(tile.lat, tile.lon)
    delete_incomplete_imgs(tile)
    done = build_tile(tile)
    if tile_coords in IMG.incomplete_imgs and not UI.red_flag:
        UI.lvprint(
            1,
            f"Attempting to rebuild textures with white squares: "
            f"{IMG.incomplete_imgs[tile_coords]}",
        )
        delete_incomplete_imgs(tile)
        done = build_tile(tile)
    if not UI.red_flag:
        for name in IMG.incomplete_imgs.get(tile_coords, []):
            PROG.issue(name, "download", "parts could not be obtained; filled with white")
    return done

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
