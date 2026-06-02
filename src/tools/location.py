"""User location tool.

The agent often needs a starting point ("near me"). This tool resolves the
user's approximate current location. Because location is sensitive, treat it as
requiring explicit user approval before use.

Implementation notes:
- The Google **Geolocation API** estimates a location from network signals and
  returns lat/lng (it does not require GPS). Geolocation has no "new" successor
  API; this uses the ``googlemaps`` client (``gmaps.geolocate(...)``).
- The coordinates are reverse-geocoded to a human-readable address via the
  **Geocoding API** (``gmaps.reverse_geocode(...)``), also still current.
- Requires the ``GOOGLE_MAPS_API_KEY`` environment variable.
"""

from langchain_core.tools import tool

from ._clients import maps_client


@tool
def get_current_location() -> str:
    """Get the user's approximate current location (requires user approval).

    Returns:
        A human-readable location (address or lat/lng) to use as a trip origin.

    Note:
        Location is sensitive: only call this after the user has agreed to share
        their location. If approval is unclear, ask the user for a starting
        address instead.
    """
    try:
        gmaps = maps_client()

        # Geolocation API: estimate position from network signals (no GPS).
        result = gmaps.geolocate()
        location = result.get("location")
        if not location:
            return "Could not determine the current location."

        lat, lng = location["lat"], location["lng"]
        accuracy = result.get("accuracy")

        # Reverse-geocode to a human-readable address when possible.
        address = f"{lat:.5f}, {lng:.5f}"
        reverse = gmaps.reverse_geocode((lat, lng))
        if reverse:
            address = reverse[0].get("formatted_address", address)

        summary = f"Approximate current location: {address} ({lat:.5f}, {lng:.5f})"
        if accuracy:
            summary += f", accuracy ~{round(accuracy)} m"
        return summary

    except Exception as e:
        return f"Error determining current location: {e}"
