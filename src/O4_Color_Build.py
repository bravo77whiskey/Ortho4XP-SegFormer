"""Driver for the reference-anchored colour correction (main process).

Colour correction and cloud removal are batch steps, run after Step 3 and
before the overlays (O4_Tile_Utils.build_color).  When the batch builds the
tile's Step 3 too, Step 3 does the correction inline (tile.color_correction /
tile.color_fill_clouds, set by the batch) so no texture is converted twice;
the step afterwards then only finds what the inline pass could not do.

The correction itself (O4_Color_Correction.py) needs torch, which is not
bundled into the exe, so it runs in the overlay .venv python like the
overlays do.  This side does everything that needs Ortho4XP's imagery
machinery:

  1. group the textures the build is about to convert by (provider, ZL);
  2. download the Sentinel-2 cloudless reference (EOX, ZL14) around each group;
  3. worker job "fit": fit every group, report cloudy textures;
  4. download the fill provider's textures for those (Bing <-> Google);
  5. worker job "apply": correct, fill and write the DDS of every texture.

Whatever the worker did not convert (combined providers, providers with
colour filters, an error, a missing .venv) goes through the normal
conversion, so a failure here never loses a texture.
"""
import json
import os
import re
import shutil
import struct
import time

import O4_File_Names as FNAMES
import O4_Imagery_Utils as IMG
import O4_Terrain_Materials as TMAT
import O4_UI_Utils as UI

REF_PROVIDER = "EOX"      # base layer; the year comes from color_reference_year
REF_CAPABILITIES = "https://tiles.maps.eox.at/wmts/1.0.0/WMTSCapabilities.xml"
REF_DEFAULT_YEAR = 2025
REF_ZL = 14
MARGIN_TILES = 1
FILL_PROVIDERS = {"BI": "GO2", "GO2": "BI"}


# Corrected DDS carry a stamp in their header (DDS_HEADER.dwReserved1[0..2];
# nvcompress itself uses [7..10]), so Step 3 knows on a rebuild whether an
# existing texture already matches the colour settings.  Bump MARK_VERSION when
# the correction changes enough that existing textures should be redone.
MARK = b"O4CC"
MARK_VERSION = 1
_MARK_OFFSET = 32


def unavailable_reason():
    """Why colour correction cannot run on this install, or None."""
    if not os.path.basename(IMG.dds_convert_cmd).lower().startswith("nvcompress"):
        return "it needs nvcompress"          # the worker drives nvcompress directly
    if REF_PROVIDER not in IMG.providers_dict:
        return f"the {REF_PROVIDER} provider is missing"
    import O4_Overlay_Pipeline as PIPELINE
    if not PIPELINE._venv_exists():
        return "the overlay .venv is not set up"
    return None


def enabled(tile):
    """True when Step 3 should correct inline (the batch's colour steps follow it)."""
    return bool(getattr(tile, "color_correction", False)) and unavailable_reason() is None


def _fill(tile):
    return bool(getattr(tile, "color_fill_clouds", False))


def _mark_flags(tile, fill):
    # bit 0: cloud fill; bits 16+: Sentinel-2 year, so another year redoes the texture
    return (_reference_year(tile) << 16) | (1 if fill else 0)


def _reference_year(tile):
    return int(getattr(tile, "color_reference_year", REF_DEFAULT_YEAR))


_server_years = None


def server_years():
    """Sentinel-2 cloudless years the EOX server publishes, or None when it can't be reached."""
    global _server_years
    if _server_years is None:
        try:
            r = IMG.requests.get(REF_CAPABILITIES, headers=IMG.request_headers_generic, timeout=30)
            r.raise_for_status()
            _server_years = sorted({int(y) for y in re.findall(
                r"<ows:Identifier>s2cloudless-(\d{4})_3857<", r.text)})
        except Exception as e:
            UI.vprint(1, f"   Could not read the Sentinel-2 years from the EOX server ({e}).")
            return None
    return _server_years


def check_year(year):
    """(ok, message) for a Sentinel-2 cloudless year; ok is None when the server can't be reached."""
    years = server_years()
    if years is None:
        return None, f"Could not reach the EOX server to check Sentinel-2 cloudless {year}."
    if year in years:
        return True, f"Sentinel-2 cloudless {year} is available on the EOX server."
    return False, (f"Sentinel-2 cloudless {year} is not on the EOX server.\n"
                   f"Available years: {', '.join(map(str, years))}.")


def _reference_provider(tile):
    """Provider code of the configured yearly mosaic, registered on the fly from EOX.lay.

    The year is in the code so the downloaded reference JPEGs of different
    years never mix.
    """
    year = _reference_year(tile)
    code = f"{REF_PROVIDER}{year}"
    if code not in IMG.providers_dict:
        base = IMG.providers_dict[REF_PROVIDER]
        prov = dict(base)
        prov["code"] = code
        prov["url_template"] = re.sub(r"s2cloudless-\d{4}_", f"s2cloudless-{year}_", base["url_template"])
        IMG.providers_dict[code] = prov
    return code


def dds_mark(path):
    """(version, flags) of a colour-corrected DDS, or None."""
    try:
        with open(path, "rb") as f:
            head = f.read(_MARK_OFFSET + 12)
    except OSError:
        return None
    if len(head) < _MARK_OFFSET + 12 or head[_MARK_OFFSET:_MARK_OFFSET + 4] != MARK:
        return None
    return struct.unpack_from("<II", head, _MARK_OFFSET + 4)


# Textures to correct from their existing DDS because the downloaded
# orthophoto is gone (people clear Orthophotos): {(build_dir, attrs): dds path}.
# Re-encoding a BC1 texture adds ~6% to its compression error, far cheaper
# than downloading the whole tile again.
_from_dds = {}


def correction_reason(tile, attrs, dds_path, fill):
    """Why a texture still needs the colour step (with cloud removal when fill), or None.

    A texture already filled is not redone by a colour step without cloud
    removal: not running a step never undoes it.
    """
    if not _eligible(attrs):
        return None
    mark = dds_mark(dds_path)
    if mark is None:
        return "not colour-corrected yet"
    version, flags = mark
    if version != MARK_VERSION:
        return "colour correction changed"
    if flags >> 16 != _reference_year(tile):
        return "corrected against another Sentinel-2 year"
    if fill and not flags & 1:
        return "clouds not removed yet"
    return None


def _use_existing_dds_if_needed(tile, attrs, dds_path):
    """Correct from the DDS when the orthophoto is gone and the DDS is still uncorrected."""
    if dds_mark(dds_path) is None and not os.path.isfile(_jpeg_path(tile, *attrs)):
        _from_dds[(tile.build_dir, tuple(attrs))] = dds_path
        return True
    return False


def rebuild_reason(tile, attrs, dds_path):
    """Why Step 3 must redo an otherwise reusable DDS for its inline correction, or None."""
    if not enabled(tile):
        return None
    reason = correction_reason(tile, attrs, dds_path, _fill(tile))
    if reason and _use_existing_dds_if_needed(tile, attrs, dds_path):
        reason += ", correcting the existing texture"
    return reason


def uses_existing_dds(tile, attrs):
    """True when this texture is corrected from its DDS: nothing to download or convert normally."""
    return bool(_from_dds) and (tile.build_dir, tuple(attrs)) in _from_dds


def has_source(tile, attrs):
    """True when there is imagery to correct the texture from."""
    return uses_existing_dds(tile, attrs) or os.path.isfile(_jpeg_path(tile, *attrs))


def forget(tile):
    """Drop the tile's correct-from-DDS records."""
    for key in [k for k in _from_dds if k[0] == tile.build_dir]:
        del _from_dds[key]


def tile_textures(tile):
    """(attrs, dds path) of the orthophoto textures the tile's terrain files use.

    None when the tile has no terrain folder (Step 3 never ran).
    """
    ter_dir = os.path.join(tile.build_dir, "terrain")
    if not os.path.isdir(ter_dir):
        return None
    found = {}
    prefix = "BASE_TEX_NOWRAP ../textures/"
    for name in os.listdir(ter_dir):
        if not name.endswith(".ter"):
            continue
        try:
            with open(os.path.join(ter_dir, name)) as f:
                tex = next((ln[len(prefix):].strip() for ln in f if ln.startswith(prefix)), None)
        except OSError:
            continue
        m = re.fullmatch(r"(\d+)_(\d+)_(.+)(\d\d)\.dds", tex or "")
        if not m:
            continue
        ty, tx, prov, zl = int(m[1]), int(m[2]), m[3], int(m[4])
        if prov not in IMG.providers_dict or prov in IMG.combined_providers_dict:
            continue
        found[(tx, ty, zl, prov)] = os.path.join(tile.build_dir, "textures", tex)
    return sorted(found.items())


def plan_tile(tile, fill):
    """Textures of a built tile the colour step must do: [(attrs, reason, needs_download)]."""
    todo = []
    for attrs, dds in tile_textures(tile) or []:
        if not os.path.isfile(dds):
            continue
        reason = correction_reason(tile, attrs, dds, fill)
        if reason:
            from_dds = _use_existing_dds_if_needed(tile, attrs, dds)
            todo.append((attrs, reason, not from_dds and not os.path.isfile(_jpeg_path(tile, *attrs))))
    return todo


def _eligible(attrs):
    tx, ty, zl, prov = attrs
    return (zl >= REF_ZL and prov in IMG.providers_dict
            and prov not in IMG.local_combined_providers_dict
            and IMG.providers_dict[prov].get("color_filters", "none") == "none"
            and prov != REF_PROVIDER)


def _jpeg_path(tile, tx, ty, zl, prov):
    return os.path.join(
        FNAMES.jpeg_file_dir_from_attributes(tile.lat, tile.lon, zl, IMG.providers_dict[prov]),
        FNAMES.jpeg_file_name_from_attributes(tx, ty, zl, prov))


def _reference_textures(zl, textures):
    """ZL14 reference textures covering a group plus one ZL14 tile of margin.

    Must match reference_window/reference_textures in O4_Color_Correction
    (the worker imports nothing from Ortho4XP, so the arithmetic lives twice).
    Worked in ZL14 pixels: a ZL19 group can start mid-tile.
    """
    f14 = 2 ** (zl - REF_ZL)
    xs = [t[0] for t in textures]
    ys = [t[1] for t in textures]
    x0, y0 = min(xs), min(ys)
    nx, ny = (max(xs) - x0) // 16 + 1, (max(ys) - y0) // 16 + 1
    m = MARGIN_TILES * 256
    rx0, ry0 = x0 * 256 // f14 - m, y0 * 256 // f14 - m
    rw, rh = nx * 4096 // f14 + 2 * m, ny * 4096 // f14 + 2 * m
    return [(tx, ty) for ty in range((ry0 // 4096) * 16, (ry0 + rh - 1) // 4096 * 16 + 1, 16)
            for tx in range((rx0 // 4096) * 16, (rx0 + rw - 1) // 4096 * 16 + 1, 16)]


def _run_worker(job, job_path):
    import O4_Overlay_Pipeline as PIPELINE
    with open(job_path, "w") as f:
        json.dump(job, f)
    code = ("import O4_Color_Correction as CC\n"
            f"CC.main({job_path!r})\n")
    rc = PIPELINE._run_venv(code)
    if rc != 0 or not os.path.isfile(job["result"]):
        return None
    with open(job["result"]) as f:
        return json.load(f)


def run(tile, texture_attrs, workers=4, fill=None):
    """Colour-correct and convert what it can; return the set of converted texture attrs.

    fill: remove clouds too; None takes the tile's inline setting.
    """
    import O4_Overlay_Pipeline as PIPELINE
    timer = time.time()
    fill = _fill(tile) if fill is None else bool(fill)
    groups = {}
    for attrs in texture_attrs:
        if _eligible(attrs):
            groups.setdefault((attrs[3], attrs[2]), set()).add((attrs[0], attrs[1]))
    if not groups:
        return set()
    if not PIPELINE._venv_exists():
        UI.vprint(1, "   Colour correction skipped: the overlay .venv is not set up.")
        return set()
    UI.vprint(1, "-> Colour correction against the Sentinel-2 reference "
                 f"({sum(len(v) for v in groups.values())} textures).")
    work = os.path.join(FNAMES.resource_path("tmp"), "colour_" + FNAMES.short_latlon(tile.lat, tile.lon))
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    try:
        # --- reference --------------------------------------------------
        year = _reference_year(tile)
        ok, msg = check_year(year)
        if ok is False:
            UI.lvprint(1, "   Colour correction skipped: " + msg.replace("\n", " ")
                       + " Set color_reference_year to one of them.")
            return set()
        ref_prov = _reference_provider(tile)
        UI.vprint(1, f"   Reference: Sentinel-2 cloudless {year}.")
        jobs = []
        for (prov, zl), tex in sorted(groups.items()):
            refs = []
            for (rtx, rty) in _reference_textures(zl, tex):
                if not IMG.build_jpeg_ortho(tile, rtx, rty, REF_ZL, ref_prov):
                    raise RuntimeError(f"could not download reference texture {rty}_{rtx}")
                refs.append([rtx, rty, _jpeg_path(tile, rtx, rty, REF_ZL, ref_prov)])
                if UI.red_flag:
                    return set()
            alt = FILL_PROVIDERS.get(prov)
            jobs.append({
                "id": f"{prov}_{zl}", "provider": prov, "zl": zl, "lat": tile.lat + 0.5,
                "textures": [[tx, ty, _from_dds.get((tile.build_dir, (tx, ty, zl, prov)))
                              or _jpeg_path(tile, tx, ty, zl, prov)] for (tx, ty) in sorted(tex)],
                "reference": refs,
                "clouds": fill and alt in IMG.providers_dict,
            })
        # --- fit ----------------------------------------------------------
        fit = _run_worker({"phase": "fit", "state_dir": work, "groups": jobs,
                           "result": os.path.join(work, "fit.json")}, os.path.join(work, "fit_job.json"))
        if fit is None or UI.red_flag:
            if not UI.red_flag:
                UI.lvprint(1, "   Colour correction failed (fit); converting without it.")
            return set()
        failed = [gj for gj in jobs if "failed" in fit.get(gj["id"], {})]
        if failed:
            UI.lvprint(1, "   Colour correction failed for " + ", ".join(
                f"{gj['provider']} ZL{gj['zl']}" for gj in failed) + "; those textures are converted without it.")
            jobs = [gj for gj in jobs if gj not in failed]
            if not jobs:
                return set()
        # --- fill provider ------------------------------------------------
        for gj in jobs:
            need = fit.get(gj["id"], {}).get("need", [])
            alt = FILL_PROVIDERS.get(gj["provider"])
            if not (need and gj["clouds"]):
                continue
            UI.vprint(1, f"   Colour correction: {len(fit[gj['id']]['cloudy'])} {gj['provider']} "
                         f"ZL{gj['zl']} texture(s) with cloud or missing tiles, fetching {len(need)} "
                         f"{alt} texture(s) to fill them.")
            got = []
            for tx, ty in need:
                if IMG.build_jpeg_ortho(tile, tx, ty, gj["zl"], alt):
                    got.append([tx, ty, _jpeg_path(tile, tx, ty, gj["zl"], alt)])
                if UI.red_flag:
                    return set()
            gj["fill"] = {"provider": alt, "textures": got} if got else None
        # --- apply --------------------------------------------------------
        for gj in jobs:
            targets = []
            for tx, ty, src in gj["textures"]:
                dds = FNAMES.dds_file_name_from_attributes(tx, ty, gj["zl"], gj["provider"])
                mask = None
                if tile.imprint_masks_to_dds:
                    mpath = os.path.join(tile.build_dir, "textures",
                                         FNAMES.mask_file(tx, ty, gj["zl"], gj["provider"]))
                    mask = mpath if os.path.exists(mpath) else None
                targets.append([tx, ty, src, os.path.join(tile.build_dir, "textures", dds), mask])
            gj["targets"] = targets
        res = _run_worker({"phase": "apply", "state_dir": work, "groups": jobs, "workers": workers,
                           "nvcompress": IMG.dds_convert_cmd, "tmp_dir": work,
                           "mark": [MARK.decode("ascii"), MARK_VERSION, _mark_flags(tile, fill)],
                           "materials": ({"relief": TMAT.relief_of(tile)} if TMAT.enabled(tile) else None),
                           "result": os.path.join(work, "apply.json")}, os.path.join(work, "apply_job.json"))
        if res is None:
            if not UI.red_flag:
                UI.lvprint(1, "   Colour correction failed (apply); converting the rest without it.")
            return set()
        done = {tuple(d) for d in res.get("done", [])}
        UI.vprint(1, f" *Colour correction: {len(done)} textures corrected and converted in "
                     f"{time.time() - timer:.0f}s.")
        return done
    except Exception as e:
        UI.lvprint(1, f"   Colour correction failed ({e}); converting without it.")
        return set()
    finally:
        shutil.rmtree(work, ignore_errors=True)
        forget(tile)
