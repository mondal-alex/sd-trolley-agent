"""Shared, cached Google API clients.

All Google-backed tools build their clients from the same
``GOOGLE_MAPS_API_KEY``. Centralizing construction here avoids duplication and,
via ``functools.lru_cache``, builds each client only once and reuses it across
tool calls (important for the gRPC-backed Routes client, whose channel is
expensive to recreate).

Clients are created lazily on first use, so importing a tool module never fails
just because the key isn't set yet.

Testing tip: prefer patching the tool's reference (e.g.
``patch("src.tools.location.maps_client")``). If you ever patch a client here
directly, call ``<factory>.cache_clear()`` in a fixture so a cached instance
doesn't leak between tests.
"""

import functools
import os

import googlemaps
from google.maps import routing_v2


def _api_key() -> str:
    """Return the Google Maps API key or raise if it's missing."""
    key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY environment variable is not set")
    return key


@functools.lru_cache(maxsize=1)
def maps_client() -> googlemaps.Client:
    """Legacy Google Maps client, used for Geocoding + Geolocation."""
    return googlemaps.Client(key=_api_key())


@functools.lru_cache(maxsize=1)
def routes_client() -> routing_v2.RoutesClient:
    """Routes API client (drive/walk times)."""
    return routing_v2.RoutesClient(client_options={"api_key": _api_key()})
