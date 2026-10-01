"""Overpass downloads for the SFR overlays (stdlib only, runs in sfr_venv).

The overlays cannot import O4_OSM_Utils (GUI + requests + shapely), so this
mirrors its hardened client: identifying User-Agent, POST body, rotation over
the server pool, Retry-After on 429/504, and a check that the answer is
complete before it is cached.
"""

import bz2
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

# overpass-api.de answers 406 to the default "Python-urllib" agent.
try:
    import O4_Version
    _VERSION = O4_Version.version
except ImportError:
    _VERSION = "1.40"
USER_AGENT = (
    "Ortho4XP/" + _VERSION
    + " (SegFormer fork; +https://github.com/oscarpilote/Ortho4XP)"
)

# overpass.osm.jp was dropped: its certificate does not match the hostname.
SERVERS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)
# A dense 1-degree tile of buildings can take minutes server-side.
QUERY_TIMEOUT = 300
MAX_ROUNDS = 2
MAX_RETRY_AFTER_WAIT = 60

_RUNTIME_ERROR = re.compile(rb"<remark>\s*runtime error", re.IGNORECASE)


def tile_query(lat, lon, statements, timeout=QUERY_TIMEOUT):
    """Return an XML Overpass query for the 1-degree tile at ``lat, lon``.

    ``statements`` are filters such as ``'way["highway"]'``; the tile bbox is
    appended to each.
    """
    bbox = f"({int(lat)},{int(lon)},{int(lat) + 1},{int(lon) + 1})"
    body = "".join(f"{statement}{bbox};" for statement in statements)
    return f"[out:xml][timeout:{timeout}];({body});(._;>;);out body;"


def _check_answer(data):
    """Return None when ``data`` is a complete OSM answer, else the reason."""
    tail = data[-64:].rstrip()
    if not tail.endswith(b"</osm>"):
        return "truncated answer (no closing </osm>)"
    # Overpass returns 200 with partial data when the query runs out of time
    # or memory; caching that would hide the gap forever.
    if _RUNTIME_ERROR.search(data[-4096:]) or _RUNTIME_ERROR.search(data[:4096]):
        return "server runtime error (query too big or timed out), partial data"
    return None


def _post(server, query, timeout):
    request = urllib.request.Request(
        server,
        data=urllib.parse.urlencode({"data": query}).encode("utf-8"),
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def download_to_cache(query, cache_path, label, timeout=QUERY_TIMEOUT):
    """Fetch ``query`` and store it bz2-compressed at ``cache_path``.

    Returns True when the cache exists afterwards. Each failure names the
    server and the reason; the next server is tried straight away and a full
    round of failures backs off before the next round.
    """
    if os.path.exists(cache_path):
        return True
    read_timeout = timeout + 60
    for round_index in range(MAX_ROUNDS):
        wait = 2 ** (round_index + 2)
        for server in SERVERS:
            host = urllib.parse.urlsplit(server).netloc
            try:
                data = _post(server, query, read_timeout)
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 504):
                    try:
                        wait = max(wait, min(int(exc.headers.get("Retry-After") or 0),
                                             MAX_RETRY_AFTER_WAIT))
                    except ValueError:
                        pass
                print(f"  [{label}] {host} failed: HTTP {exc.code} {exc.reason}", flush=True)
                continue
            except Exception as exc:
                print(f"  [{label}] {host} failed: {exc}", flush=True)
                continue
            problem = _check_answer(data)
            if problem:
                print(f"  [{label}] {host} failed: {problem}", flush=True)
                continue
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            tmp_path = cache_path + ".part"
            with bz2.open(tmp_path, "wb") as handle:
                handle.write(data)
            os.replace(tmp_path, cache_path)
            print(f"  [{label}] Downloaded {len(data) // 1024} KB from {host} "
                  f"-> {os.path.basename(cache_path)}", flush=True)
            return True
        if round_index + 1 < MAX_ROUNDS:
            print(f"  [{label}] all servers failed, retrying in {wait} s", flush=True)
            time.sleep(wait)
    return False
