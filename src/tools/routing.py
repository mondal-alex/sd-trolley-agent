"""Routing tools backed by the Google Maps APIs.

These tools answer "how long does it take to get from A to B" for the two
travel modes the agent cares about: driving (to a trolley station / parking)
and walking (from a station to the final destination).

Implementation notes (for when you fill these in):
- Use the Google Maps Directions or Distance Matrix API.
- Read the API key from the ``GOOGLE_MAPS_API_KEY`` environment variable.
- Prefer returning a short, model-friendly string (duration + distance) rather
  than a giant raw JSON blob. The LLM has to read whatever you return.
- Support a ``departure_time`` / ``arrival_time`` so the model can reason about
  traffic ("leave by 5pm" type questions).
"""

from langchain_core.tools import tool


@tool
def get_driving_time(origin: str, destination: str, depart_at: str | None = None) -> str:
    """Get the driving time and distance between two locations.

    Args:
        origin: Starting address or place name.
        destination: Destination address or place name.
        depart_at: Optional ISO-8601 departure time used for traffic-aware
            estimates. Defaults to now when omitted.

    Returns:
        A short human-readable summary, e.g. "Driving: 18 mins (7.4 mi)".
    """
    # TODO: Call the Google Maps Directions/Distance Matrix API and format the
    # duration + distance into a short string.
    raise NotImplementedError


@tool
def get_walking_time(origin: str, destination: str) -> str:
    """Get the walking time and distance between two locations.

    Args:
        origin: Starting address or place name (often a trolley station).
        destination: Destination address or place name.

    Returns:
        A short human-readable summary, e.g. "Walking: 9 mins (0.4 mi)".
    """
    # TODO: Call the Google Maps API in walking mode and format the result.
    raise NotImplementedError
