"""Routing tools backed by the Google Maps Routes API.

These tools answer "how long does it take to get from A to B" for the two
travel modes the agent cares about: driving (to a trolley station / parking)
and walking (from a station to the final destination).

Implementation notes:
- Uses the **Routes API** via the ``google-maps-routing`` client
  (``from google.maps import routing_v2``), which replaces the legacy
  Directions / Distance Matrix APIs.
- The client is built once and cached in ``_clients.routes_client``.
- Routes API requires a **field mask** sent as request metadata
  (``x-goog-fieldmask``) or the response comes back empty.
- ``routing_preference`` (traffic-aware) is only valid for DRIVE, so the walking
  tool must not set it.
- Tools return a short, model-friendly string rather than raw protobuf.
"""

from datetime import datetime

from google.maps import routing_v2
from langchain_core.tools import tool

from ._clients import routes_client

# Only the fields we actually read; keeps responses small and avoids the
# "empty response" gotcha that happens when no field mask is supplied.
_FIELD_MASK = "routes.duration,routes.distanceMeters"
_METERS_PER_MILE = 1609.344


def _duration_seconds(duration) -> float:
    """Get seconds from a Routes API duration.

    proto-plus maps protobuf ``Duration`` to ``datetime.timedelta``; fall back to
    a raw ``.seconds`` field just in case.
    """
    if hasattr(duration, "total_seconds"):
        return duration.total_seconds()
    return float(getattr(duration, "seconds", 0))


def _format_duration(seconds: float) -> str:
    """Format seconds as e.g. "9 min" or "1 hr 23 min"."""
    total_minutes = round(seconds / 60)
    hours, minutes = divmod(total_minutes, 60)
    if hours and minutes:
        return f"{hours} hr {minutes} min"
    if hours:
        return f"{hours} hr"
    return f"{minutes} min"


def _format_distance(meters: float) -> str:
    """Format meters as miles, e.g. "7.4 mi"."""
    return f"{meters / _METERS_PER_MILE:.1f} mi"

@tool
def get_driving_time(origin: str, destination: str, depart_at: str | None = None) -> str:
    """Get the driving time and distance between two locations.

    Args:
        origin: Starting address or place name.
        destination: Destination address or place name.
        depart_at: Optional ISO-8601 departure time used for traffic-aware
            estimates. Defaults to now when omitted.

    Returns:
        A short human-readable summary, e.g. "Driving: 18 min (7.4 mi)".
    """
    return _compute_route(
        origin,
        destination,
        mode=routing_v2.RouteTravelMode.DRIVE,
        label="Driving",
        traffic_aware=True,
        depart_at=depart_at,
    )


@tool
def get_walking_time(origin: str, destination: str) -> str:
    """Get the walking time and distance between two locations.

    Args:
        origin: Starting address or place name (often a trolley station).
        destination: Destination address or place name.

    Returns:
        A short human-readable summary, e.g. "Walking: 9 min (0.4 mi)".
    """
    return _compute_route(
        origin,
        destination,
        mode=routing_v2.RouteTravelMode.WALK,
        label="Walking",
    )


def _compute_route(
    origin: str,
    destination: str,
    *,
    mode: routing_v2.RouteTravelMode,
    label: str,
    traffic_aware: bool = False,
    depart_at: str | None = None,
) -> str:
    """Shared Routes API call + formatting for the driving/walking tools."""
    try:
        request = routing_v2.ComputeRoutesRequest(
            origin=routing_v2.Waypoint(address=origin),
            destination=routing_v2.Waypoint(address=destination),
            travel_mode=mode,
        )

        # Traffic-aware routing (and a departure time) only apply to driving.
        if traffic_aware:
            request.routing_preference = routing_v2.RoutingPreference.TRAFFIC_AWARE
            if depart_at:
                request.departure_time = datetime.fromisoformat(depart_at)

        response = routes_client().compute_routes(
            request=request,
            metadata=[("x-goog-fieldmask", _FIELD_MASK)],
        )

        if not response.routes:
            return f"No {label.lower()} route found from {origin} to {destination}."

        route = response.routes[0]
        duration = _format_duration(_duration_seconds(route.duration))
        distance = _format_distance(route.distance_meters)
        return f"{label}: {duration} ({distance})"

    except Exception as e:
        return f"Error getting {label.lower()} time: {e}"
