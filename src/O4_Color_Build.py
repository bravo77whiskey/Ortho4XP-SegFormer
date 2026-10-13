"""Step 3 driver for the reference-anchored colour correction (main process).

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
import shutil
import struct
import time

import O4_File_Names as FNAMES
import O4_Imagery_Utils as IMG
import O4_Terrain_Materials as TMAT
import O4_UI_Utils as UI

REF_PROVIDER = "EOX"
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


def enabled(tile):
    if not getattr(tile, "color_correction", True):
        return False
    if not os.path.basename(IMG.dds_convert_cmd).lower().startswith("nvcompress"):
        return False          # the worker drives nvcompress directly
    if REF_PROVIDER not in IMG.providers_dict:
        return False
    import O4_Overlay_Pipeline as PIPELINE
    return PIPELINE._venv_exists()


def _mark_flags(tile):
    return 1 if getattr(tile, "color_fill_clouds", True) else 0


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


def rebuild_reason(tile, attrs, dds_path):
    """Why an otherwise reusable DDS must be redone for the colour settings, or None."""
    want = enabled(tile) and _eligible(attrs)
    mark = dds_mark(dds_path)
    reason = None
    if want and mark is None:
        reason = "not colour-corrected yet"
    elif want and mark != (MARK_VERSION, _mark_flags(tile)):
        reason = "colour correction settings changed"
    elif not want and mark is not None and getattr(tile, "color_correction", True) is False:
        # Undoing the correction needs the original imagery: normal path.
        return "colour-corrected, but colour correction is off"
    if reason and mark is None and not os.path.isfile(_jpeg_path(tile, *attrs)):
        _from_dds[(tile.build_dir, tuple(attrs))] = dds_path
        reason += ", correcting the existing texture"
    return reason


def uses_existing_dds(tile, attrs):
    """True when this texture is corrected from its DDS: nothing to download or convert normally."""
    return (tile.build_dir, tuple(attrs)) in _from_dds


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


def run(tile, texture_attrs, workers=4):
    """Colour-correct and convert what it can; return the set of converted texture attrs."""
    import O4_Overlay_Pipeline as PIPELINE
    timer = time.time()
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
        jobs = []
        for (prov, zl), tex in sorted(groups.items()):
            refs = []
            for (rtx, rty) in _reference_textures(zl, tex):
                if not IMG.build_jpeg_ortho(tile, rtx, rty, REF_ZL, REF_PROVIDER):
                    raise RuntimeError(f"could not download reference texture {rty}_{rtx}")
                refs.append([rtx, rty, _jpeg_path(tile, rtx, rty, REF_ZL, REF_PROVIDER)])
                if UI.red_flag:
                    return set()
            alt = FILL_PROVIDERS.get(prov)
            jobs.append({
                "id": f"{prov}_{zl}", "provider": prov, "zl": zl, "lat": tile.lat + 0.5,
                "textures": [[tx, ty, _from_dds.get((tile.build_dir, (tx, ty, zl, prov)))
                              or _jpeg_path(tile, tx, ty, zl, prov)] for (tx, ty) in sorted(tex)],
                "reference": refs,
                "clouds": bool(getattr(tile, "color_fill_clouds", True)) and alt in IMG.providers_dict,
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
                           "mark": [MARK.decode("ascii"), MARK_VERSION, _mark_flags(tile)],
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
        for key in [k for k in _from_dds if k[0] == tile.build_dir]:
            del _from_dds[key]
