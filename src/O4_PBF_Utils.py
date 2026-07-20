"""Local planet.osm.pbf data source.

Maintains a user-managed planet file (or any regional .osm.pbf extract),
keeps it current through the OSM daily replication diffs, and slices the
per-tile OSM cache files out of a filtered "scenery" extract so tile builds
never need the Overpass servers.

All heavy lifting is delegated to osmium-tool (conda-forge win-64 build,
bootstrapped through micromamba on first use).  The stock osmctools Windows
binaries are not used because they cannot process files larger than 2 GB
together with the --complete-* options.
"""

import os
import sys
import json
import time
import shutil
import hashlib
import datetime
import threading
import subprocess
import requests
import O4_UI_Utils as UI
import O4_File_Names as FNAMES
import O4_Version

# Config-bound (module "PBF"); empty means the feature is disabled.
osm_pbf_dir = ""

PLANET_URL = "https://planet.openstreetmap.org/pbf/planet-latest.osm.pbf"
REPLICATION_BASE = "https://planet.openstreetmap.org/replication/day/"
MICROMAMBA_URL = (
    "https://github.com/mamba-org/micromamba-releases/releases/latest/"
    "download/micromamba-win-64"
)
DOWNLOAD_CHUNK = 1024 * 1024
AVG_DAILY_DIFF_BYTES = 95_000_000
# osmium apply-changes loads all given change files into RAM; cap how many
# daily diffs are applied per pass (each pass rewrites the planet file).
max_diffs_per_pass = int(os.environ.get("O4_PBF_DIFF_BATCH", "10"))
max_download_tentatives = 5

_user_agent = (
    "Ortho4XP/" + O4_Version.version
    + " (SegFormer fork; +https://github.com/oscarpilote/Ortho4XP)"
)

# Per-layer osmium tags-filter expressions.  Supersets of the Overpass
# queries are fine: cached files are re-filtered through input_tags on read.
LAYER_FILTERS = {
    "airports": ["nwr/aeroway"],
    "big_roads": [
        "w/highway=motorway,trunk,primary,secondary",
        "w/railway=rail,narrow_gauge",
    ],
    # Always all road levels so the cache stays valid across road_level changes.
    "small_roads": [
        "w/highway=tertiary,unclassified,residential,service,track"
    ],
    "coastline": ["w/natural=coastline"],
    "water": [
        "wr/natural=water",
        "wr/waterway=riverbank",
        "w/waterway=dock",
    ],
}
SCENERY_FILTER = [
    "nwr/aeroway",
    "w/highway=motorway,trunk,primary,secondary,tertiary,unclassified,"
    "residential,service,track",
    "w/railway=rail,narrow_gauge",
    "w/natural=coastline",
    "wr/natural=water",
    "wr/waterway=riverbank",
    "w/waterway=dock",
]

maintenance_lock = threading.Lock()
maintenance_in_progress = False
_not_ready_notified = False
_tile_extract_cache = {}

_CREATE_NO_WINDOW = (
    subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
)

################################################################################
# Paths and state
################################################################################

def planet_path():
    return os.path.join(osm_pbf_dir, "planet.osm.pbf")


def extract_path():
    return os.path.join(osm_pbf_dir, "scenery.osm.pbf")


def state_path():
    return os.path.join(osm_pbf_dir, "planet_state.json")


def diffs_dir():
    return os.path.join(osm_pbf_dir, "diffs")


def _load_state():
    try:
        with open(state_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(seq, timestamp):
    data = {
        "seq": seq,
        "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
        if isinstance(timestamp, datetime.datetime)
        else timestamp,
    }
    with open(state_path(), "w", encoding="utf-8") as f:
        json.dump(data, f)

################################################################################
# Tool provisioning (osmium-tool via micromamba)
################################################################################

def _tools_root():
    return os.path.join(FNAMES.user_path("OSM_tools"))


def _micromamba_exe():
    return os.path.join(_tools_root(), "micromamba.exe")


def _osmium_env_exe():
    if sys.platform == "win32":
        return os.path.join(
            _tools_root(), "envs", "osmium", "Library", "bin", "osmium.exe"
        )
    return os.path.join(_tools_root(), "envs", "osmium", "bin", "osmium")


def osmium_path():
    system_osmium = shutil.which("osmium")
    if system_osmium:
        return system_osmium
    env_osmium = _osmium_env_exe()
    if os.path.isfile(env_osmium):
        return env_osmium
    return None


def tools_available():
    return osmium_path() is not None


def ensure_tools(confirm=None, progress_cb=None):
    """Provision osmium-tool. Never called from the tile-build path."""
    if tools_available():
        return True
    if sys.platform != "win32":
        UI.lvprint(
            1,
            "Please install osmium-tool through your package manager",
            "(e.g. apt install osmium-tool).",
        )
        return False
    if confirm is not None and not confirm():
        return False
    with maintenance_lock:
        if tools_available():
            return True
        os.makedirs(_tools_root(), exist_ok=True)
        mm_exe = _micromamba_exe()
        if not os.path.isfile(mm_exe):
            UI.vprint(1, "    Downloading micromamba (tool bootstrapper)...")
            sha_ok = _download_file(
                MICROMAMBA_URL + ".sha256",
                mm_exe + ".sha256",
                resume=False,
            )
            expected = None
            if sha_ok:
                try:
                    with open(mm_exe + ".sha256", "r", encoding="utf-8") as f:
                        expected = f.read().split()[0].strip()
                except Exception:
                    expected = None
            if not _download_file(
                MICROMAMBA_URL,
                mm_exe,
                resume=False,
                progress_cb=progress_cb,
                expected_sha256=expected,
            ):
                UI.lvprint(1, "Could not download micromamba.")
                return False
        UI.vprint(
            1, "    Installing osmium-tool from conda-forge (~80 MB)..."
        )
        rc = _run_tool(
            [
                mm_exe,
                "create",
                "-y",
                "-r",
                _tools_root(),
                "-n",
                "osmium",
                "-c",
                "conda-forge",
                "osmium-tool",
            ]
        )
        if rc != 0 or not os.path.isfile(_osmium_env_exe()):
            UI.lvprint(1, "osmium-tool installation failed.")
            return False
        rc, out = _run_tool_capture([_osmium_env_exe(), "--version"])
        if rc != 0:
            UI.lvprint(1, "osmium-tool does not run:", out)
            return False
        UI.vprint(1, "    osmium-tool ready:", out.strip().splitlines()[0])
        return True

################################################################################
# Subprocess helpers
################################################################################

def _run_tool(cmd, log_prefix="      "):
    """Run an external tool, streaming output; killable via the Stop button."""
    try:
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=_CREATE_NO_WINDOW,
        )
    except Exception as exc:
        UI.lvprint(1, "Could not start", cmd[0], ":", exc)
        return -1
    UI.register_subprocess(p)
    try:
        for line in p.stdout:
            line = line.decode("utf-8", errors="replace").rstrip()
            if line:
                UI.vprint(2, log_prefix + line)
        p.wait()
    finally:
        UI.unregister_subprocess(p)
    return p.returncode


def _run_tool_capture(cmd):
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=_CREATE_NO_WINDOW,
            timeout=120,
        )
        return (
            result.returncode,
            result.stdout.decode("utf-8", errors="replace"),
        )
    except Exception as exc:
        return (-1, str(exc))

################################################################################
# Downloads
################################################################################

def _download_file(
    url, dest, resume=True, progress_cb=None, expected_sha256=None
):
    """Streamed download with .part file, HTTP Range resume and retries.

    Returns True on a byte-complete (and hash-verified, when requested)
    download; a red_flag abort keeps the .part file for a later resume.
    """
    part = dest + ".part"
    tentative = 0
    while True:
        tentative += 1
        total = 0
        stream_complete = False
        pos = (
            os.path.getsize(part)
            if (resume and os.path.isfile(part))
            else 0
        )
        headers = {"User-Agent": _user_agent}
        if pos:
            headers["Range"] = "bytes=%d-" % pos
        try:
            with requests.get(
                url, headers=headers, stream=True, timeout=60
            ) as r:
                if pos and r.status_code == 200:
                    # Server ignored the Range header: restart from scratch.
                    pos = 0
                if r.status_code not in (200, 206):
                    UI.vprint(
                        1,
                        "      Download of",
                        os.path.basename(dest),
                        "got HTTP status",
                        r.status_code,
                    )
                    if r.status_code == 404 or tentative >= max_download_tentatives:
                        return False
                    time.sleep(2 ** tentative)
                    continue
                if r.status_code == 206:
                    content_range = r.headers.get("Content-Range", "")
                    total = (
                        int(content_range.rsplit("/", 1)[-1])
                        if "/" in content_range
                        else 0
                    )
                else:
                    total = int(r.headers.get("Content-Length", 0) or 0)
                mode = "ab" if pos else "wb"
                with open(part, mode) as f:
                    for chunk in r.iter_content(DOWNLOAD_CHUNK):
                        if UI.red_flag:
                            return False
                        f.write(chunk)
                        pos += len(chunk)
                        if progress_cb and total:
                            progress_cb(int(100 * pos / total))
                stream_complete = True
        except Exception as exc:
            UI.vprint(
                1,
                "      Download of",
                os.path.basename(dest),
                "interrupted (",
                exc,
                ")",
            )
        if UI.red_flag:
            return False
        if stream_complete and (not total or pos >= total):
            break
        if tentative >= max_download_tentatives:
            return False
        time.sleep(2 ** tentative)
    if expected_sha256:
        digest = hashlib.sha256()
        with open(part, "rb") as f:
            for block in iter(lambda: f.read(DOWNLOAD_CHUNK), b""):
                digest.update(block)
        if digest.hexdigest().lower() != expected_sha256.lower():
            UI.lvprint(
                1,
                "SHA256 mismatch for",
                os.path.basename(dest),
                "- discarding the file.",
            )
            os.remove(part)
            return False
    os.replace(part, dest)
    return True


def download_planet(progress_cb=None):
    """Download planet-latest.osm.pbf (resumable) into osm_pbf_dir."""
    global maintenance_in_progress
    if not osm_pbf_dir:
        UI.lvprint(1, "No OSM planet directory configured.")
        return 0
    with maintenance_lock:
        maintenance_in_progress = True
        try:
            os.makedirs(osm_pbf_dir, exist_ok=True)
            if not _preflight_disk(osm_pbf_dir, 100 * 2 ** 30):
                return 0
            UI.vprint(
                1,
                "-> Downloading",
                PLANET_URL,
                "(~90 GB, resumable; this takes hours).",
            )
            if not _download_file(
                PLANET_URL, planet_path(), progress_cb=progress_cb
            ):
                UI.lvprint(
                    1,
                    "Planet download interrupted; partial data was kept",
                    "and the download will resume next time.",
                )
                return 0
            timestamp = get_pbf_timestamp(planet_path())
            _save_state(None, timestamp or "")
            UI.vprint(1, "   Planet file downloaded.")
            return 1
        finally:
            maintenance_in_progress = False

################################################################################
# Replication state
################################################################################

def _seq_url(seq):
    return REPLICATION_BASE + "%03d/%03d/%03d" % (
        seq // 1_000_000,
        (seq // 1000) % 1000,
        seq % 1000,
    )


def _parse_state_text(text):
    seq = None
    timestamp = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("sequenceNumber="):
            seq = int(line.split("=", 1)[1])
        elif line.startswith("timestamp="):
            raw = line.split("=", 1)[1].replace("\\:", ":")
            timestamp = datetime.datetime.strptime(
                raw, "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=datetime.timezone.utc)
    if seq is None or timestamp is None:
        raise ValueError("Unparsable replication state file")
    return seq, timestamp


def fetch_state(seq=None):
    """Return (sequence, timestamp UTC) of the current or given daily state."""
    url = (
        REPLICATION_BASE + "state.txt"
        if seq is None
        else _seq_url(seq) + ".state.txt"
    )
    r = requests.get(url, headers={"User-Agent": _user_agent}, timeout=30)
    r.raise_for_status()
    return _parse_state_text(r.text)


def find_start_sequence(local_ts, current_seq, margin_hours=2):
    """Largest daily sequence not newer than local_ts - margin, plus one."""
    target = local_ts - datetime.timedelta(hours=margin_hours)
    lo = max(1, current_seq - 4000)
    hi = current_seq
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        seq_ts = None
        for probe in range(mid, max(lo - 1, mid - 5), -1):
            try:
                _, seq_ts = fetch_state(probe)
                mid = probe
                break
            except Exception:
                continue
        if seq_ts is None:
            break
        if seq_ts <= target:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best + 1


def get_pbf_timestamp(path):
    """Replication timestamp from the PBF header, or None."""
    osmium = osmium_path()
    if not osmium or not os.path.isfile(path):
        return None
    rc, out = _run_tool_capture(
        [
            osmium,
            "fileinfo",
            "-g",
            "header.option.osmosis_replication_timestamp",
            path,
        ]
    )
    if rc != 0:
        return None
    raw = out.strip()
    if not raw:
        return None
    try:
        return datetime.datetime.strptime(
            raw, "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=datetime.timezone.utc)
    except ValueError:
        return None


def pending_updates():
    """Summary of not-yet-applied daily diffs, or None when up to date."""
    current_seq, current_ts = fetch_state()
    state = _load_state()
    if state.get("seq"):
        start = state["seq"] + 1
        local_ts = state.get("timestamp")
    else:
        local_dt = get_pbf_timestamp(planet_path())
        if local_dt is None:
            UI.lvprint(
                1,
                "Cannot determine the timestamp of the local planet file.",
            )
            return None
        local_ts = local_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        start = find_start_sequence(local_dt, current_seq)
    if start > current_seq:
        return None
    count = current_seq - start + 1
    return {
        "start": start,
        "end": current_seq,
        "count": count,
        "est_bytes": count * AVG_DAILY_DIFF_BYTES,
        "local_ts": local_ts,
        "remote_ts": current_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

################################################################################
# Update installation
################################################################################

def _preflight_disk(path, needed_bytes):
    try:
        free = shutil.disk_usage(path).free
    except Exception:
        return True
    if free < needed_bytes:
        UI.lvprint(
            1,
            "Not enough free disk space on",
            path,
            ":",
            UI.human_print(needed_bytes),
            "needed,",
            UI.human_print(free),
            "available.",
        )
        return False
    return True


def install_updates(progress_cb=None):
    """Download and apply all pending daily diffs, then refresh the extract."""
    global maintenance_in_progress
    if UI.is_working:
        UI.lvprint(
            1,
            "A tile build is in progress; finish or stop it before",
            "updating the planet file.",
        )
        return 0
    osmium = osmium_path()
    if not osmium or not os.path.isfile(planet_path()):
        UI.lvprint(1, "Planet file or osmium-tool missing.")
        return 0
    with maintenance_lock:
        maintenance_in_progress = True
        try:
            pending = pending_updates()
            if not pending:
                UI.vprint(1, "   Planet file is up to date.")
                return 1
            planet_size = os.path.getsize(planet_path())
            if not _preflight_disk(
                osm_pbf_dir,
                int(1.05 * planet_size) + pending["est_bytes"] + 2 ** 30,
            ):
                return 0
            UI.vprint(
                1,
                "-> Applying",
                pending["count"],
                "daily diff(s)",
                "(" + UI.human_print(pending["est_bytes"]) + " to download).",
            )
            os.makedirs(diffs_dir(), exist_ok=True)
            sequences = list(range(pending["start"], pending["end"] + 1))
            diff_files = []
            for i, seq in enumerate(sequences):
                if UI.red_flag:
                    return 0
                diff_file = os.path.join(diffs_dir(), "%09d.osc.gz" % seq)
                if not os.path.isfile(diff_file):
                    UI.vprint(
                        1,
                        "   Downloading diff",
                        str(i + 1) + "/" + str(len(sequences)),
                    )
                    if not _download_file(
                        _seq_url(seq) + ".osc.gz",
                        diff_file,
                        progress_cb=progress_cb,
                    ):
                        return 0
                diff_files.append((seq, diff_file))
            # Apply in batches: osmium loads all given change files in RAM.
            for batch_start in range(
                0, len(diff_files), max_diffs_per_pass
            ):
                if UI.red_flag:
                    return 0
                batch = diff_files[
                    batch_start : batch_start + max_diffs_per_pass
                ]
                last_seq = batch[-1][0]
                UI.vprint(
                    1,
                    "   Applying diffs",
                    str(batch[0][0]),
                    "to",
                    str(last_seq),
                    "(planet rewrite, this takes a while)...",
                )
                new_planet = planet_path() + ".new"
                rc = _run_tool(
                    [osmium, "apply-changes", planet_path()]
                    + [f for (_, f) in batch]
                    + ["-O", "-f", "pbf", "-o", new_planet]
                )
                if rc != 0 or not os.path.isfile(new_planet):
                    _remove_quietly(new_planet)
                    UI.lvprint(1, "Applying diffs failed; planet unchanged.")
                    return 0
                if not _replace_with_retries(new_planet, planet_path()):
                    _remove_quietly(new_planet)
                    return 0
                try:
                    _, batch_ts = fetch_state(last_seq)
                except Exception:
                    batch_ts = ""
                _save_state(last_seq, batch_ts)
            if not regenerate_extract(
                progress_cb=progress_cb, _lock_held=True
            ):
                return 0
            shutil.rmtree(diffs_dir(), ignore_errors=True)
            UI.vprint(1, "   Planet file is now up to date.")
            return 1
        finally:
            maintenance_in_progress = False


def _remove_quietly(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _replace_with_retries(src, dst, retries=3, delay=5):
    for tentative in range(retries):
        try:
            os.replace(src, dst)
            return True
        except PermissionError:
            if tentative == retries - 1:
                UI.lvprint(
                    1,
                    "Could not replace",
                    dst,
                    "- another program is holding it open.",
                )
                return False
            time.sleep(delay)
    return False

################################################################################
# Scenery extract and per-tile slicing
################################################################################

def regenerate_extract(source=None, progress_cb=None, _lock_held=False):
    """One tags-filter pass producing the small scenery extract."""
    global maintenance_in_progress
    if not _lock_held and UI.is_working:
        UI.lvprint(
            1,
            "A tile build is in progress; finish or stop it before",
            "rebuilding the scenery extract.",
        )
        return 0
    osmium = osmium_path()
    source = source or planet_path()
    if not osmium or not os.path.isfile(source):
        UI.lvprint(1, "Planet file or osmium-tool missing.")
        return 0

    def do_regen():
        if not _preflight_disk(osm_pbf_dir, 20 * 2 ** 30):
            return 0
        UI.vprint(
            1,
            "-> Building the filtered scenery extract from",
            source,
            "(one full read of the file; hours for a planet on HDD).",
        )
        new_extract = extract_path() + ".new"
        # Referenced objects (way nodes, relation members) are included by
        # default in osmium tags-filter.  The explicit -f is required because
        # the format is not derivable from the temporary .new suffix.
        rc = _run_tool(
            [osmium, "tags-filter", source]
            + SCENERY_FILTER
            + ["-O", "-f", "pbf", "-o", new_extract]
        )
        if rc != 0 or not os.path.isfile(new_extract):
            _remove_quietly(new_extract)
            UI.lvprint(1, "Scenery extract build failed.")
            return 0
        if not _replace_with_retries(new_extract, extract_path()):
            _remove_quietly(new_extract)
            return 0
        _tile_extract_cache.clear()
        UI.vprint(
            1,
            "   Scenery extract ready:",
            UI.human_print(os.path.getsize(extract_path())),
        )
        return 1

    if _lock_held:
        return do_regen()
    with maintenance_lock:
        maintenance_in_progress = True
        try:
            return do_regen()
        finally:
            maintenance_in_progress = False


def pbf_ready():
    """True when tile builds can be served from the local data."""
    global _not_ready_notified
    if not osm_pbf_dir:
        return False
    if maintenance_in_progress:
        return False
    if not os.path.isfile(extract_path()) or not tools_available():
        if not _not_ready_notified:
            _not_ready_notified = True
            UI.vprint(
                1,
                "    An OSM planet directory is configured but the scenery",
                "extract or osmium-tool is missing - using Overpass servers.",
            )
        return False
    return True


def _tile_extract(lat, lon):
    """Clip the tile bbox out of the scenery extract, once per session."""
    key = (lat, lon)
    cached = _tile_extract_cache.get(key)
    if cached and os.path.isfile(cached):
        return cached
    osmium = osmium_path()
    if not osmium:
        return None
    os.makedirs(FNAMES.Tmp_dir, exist_ok=True)
    tile_pbf = os.path.join(
        FNAMES.Tmp_dir,
        FNAMES.short_latlon(lat, lon) + "_pbf_tile.osm.pbf",
    )
    rc = _run_tool(
        [
            osmium,
            "extract",
            "-b",
            "%d,%d,%d,%d" % (lon, lat, lon + 1, lat + 1),
            "-s",
            "smart",
            extract_path(),
            "-O",
            "-o",
            tile_pbf,
        ]
    )
    if rc != 0 or not os.path.isfile(tile_pbf):
        _remove_quietly(tile_pbf)
        return None
    _tile_extract_cache[key] = tile_pbf
    return tile_pbf


def slice_tile_layer(lat, lon, cached_suffix):
    """Produce the standard per-tile OSM cache file from the local extract.

    Returns the cache file path, or None on any failure (in which case no
    cache file is written and the caller falls back to Overpass).
    """
    if cached_suffix not in LAYER_FILTERS:
        return None
    if not pbf_ready():
        return None
    tile_pbf = _tile_extract(lat, lon)
    if not tile_pbf:
        return None
    osmium = osmium_path()
    cache_file = FNAMES.osm_cached(lat, lon, cached_suffix)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    tmp_out = cache_file + ".tmp.osm.bz2"
    rc = _run_tool(
        [osmium, "tags-filter", tile_pbf]
        + LAYER_FILTERS[cached_suffix]
        + ["-O", "-f", "osm.bz2", "-o", tmp_out]
    )
    if rc != 0 or not os.path.isfile(tmp_out):
        _remove_quietly(tmp_out)
        return None
    os.replace(tmp_out, cache_file)
    return cache_file

################################################################################
# Status for the GUI
################################################################################

def planet_status():
    status = {
        "configured": bool(osm_pbf_dir),
        "tools_present": tools_available(),
        "planet_present": False,
        "planet_size": 0,
        "planet_timestamp": "",
        "planet_partial": False,
        "extract_present": False,
        "extract_size": 0,
    }
    if not osm_pbf_dir:
        return status
    if os.path.isfile(planet_path()):
        status["planet_present"] = True
        status["planet_size"] = os.path.getsize(planet_path())
        state = _load_state()
        if state.get("timestamp"):
            status["planet_timestamp"] = state["timestamp"]
        else:
            stamp = get_pbf_timestamp(planet_path())
            if stamp:
                status["planet_timestamp"] = stamp.strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
    elif os.path.isfile(planet_path() + ".part"):
        status["planet_partial"] = True
        status["planet_size"] = os.path.getsize(planet_path() + ".part")
    if os.path.isfile(extract_path()):
        status["extract_present"] = True
        status["extract_size"] = os.path.getsize(extract_path())
    return status
