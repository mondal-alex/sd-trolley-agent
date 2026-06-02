"""San Diego MTS GTFS feed loader.

MTS publishes its schedules as a **static GTFS** feed (a zip of CSVs) at a fixed
URL. This module downloads that zip, caches it on disk, and parses it once per
process into a ``gtfs_kit.Feed`` for the trolley schedule tool to query.

MTS does not offer a public real-time (GTFS-Realtime) feed, so everything here
is the *scheduled* timetable. See:
https://www.sdmts.com/business-center/app-developers
"""

import functools
import time
from pathlib import Path

import gtfs_kit as gk

# Always points at the newest MTS feed.
GTFS_URL = "http://www.sdmts.com/google_transit_files/google_transit.zip"

# On-disk cache: the zip is large and updates infrequently, so we avoid
# re-downloading it on every call (or every process start).
_CACHE_DIR = Path(__file__).resolve().parents[2] / ".gtfs_cache"
_CACHE_FILE = _CACHE_DIR / "google_transit.zip"
_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60  # refresh weekly


def _cache_is_fresh() -> bool:
    """True if the cached zip exists and is younger than the TTL."""
    if not _CACHE_FILE.exists():
        return False
    age = time.time() - _CACHE_FILE.stat().st_mtime
    return age < _CACHE_TTL_SECONDS


def _download_feed() -> None:
    """Download the GTFS zip to the cache (uses requests, already a dependency)."""
    import requests

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    response = requests.get(GTFS_URL, timeout=60)
    response.raise_for_status()
    _CACHE_FILE.write_bytes(response.content)


def gtfs_path(force_refresh: bool = False) -> Path:
    """Return the path to the cached GTFS zip, downloading it if needed."""
    if force_refresh or not _cache_is_fresh():
        _download_feed()
    return _CACHE_FILE


@functools.lru_cache(maxsize=1)
def get_feed() -> gk.Feed:
    """Load and cache the parsed MTS GTFS feed for this process.

    Parsing the feed is relatively expensive, so the result is memoized. Call
    ``get_feed.cache_clear()`` if you need to force a fresh parse (e.g. after a
    refresh, or in tests).
    """
    return gk.read_feed(gtfs_path(), dist_units="mi")
