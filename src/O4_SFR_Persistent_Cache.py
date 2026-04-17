"""Small helpers for persistent sidecar caches used by SFR overlay parsing."""

from __future__ import annotations

import hashlib
import os
import pickle


def cache_file_for_source(source_path, cache_dir, namespace, *, version):
    """Return a stable cache file path for one source file revision."""
    if not cache_dir or not source_path or not os.path.exists(source_path):
        return None

    stat = os.stat(source_path)
    cache_key = hashlib.sha1(
        (
            f"{os.path.realpath(source_path)}|{stat.st_size}|"
            f"{stat.st_mtime_ns}|{version}"
        ).encode("utf-8")
    ).hexdigest()
    return os.path.join(cache_dir, namespace, f"{cache_key}.pkl")


def load(cache_path):
    """Load a cache payload, returning ``None`` if missing or unreadable."""
    if not cache_path or not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def save(cache_path, payload):
    """Persist a cache payload, ignoring write failures."""
    if not cache_path:
        return
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass


def load_or_build(source_path, cache_dir, namespace, builder, *, version):
    """Load a cached payload for ``source_path`` or build and persist it."""
    cache_path = cache_file_for_source(
        source_path, cache_dir, namespace, version=version
    )
    cached = load(cache_path)
    if cached is not None:
        return cached

    payload = builder()
    save(cache_path, payload)
    return payload
