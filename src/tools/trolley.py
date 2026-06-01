"""San Diego Trolley (MTS) information tools.

These tools give the agent knowledge of the trolley system itself: which
stations exist near a location, the published timetable, and (if available)
live/next-arrival information.

Data source strategy (decide during implementation):
- Preferred: an official feed. San Diego MTS publishes GTFS (static timetables)
  and GTFS-Realtime (live vehicle positions / trip updates). GTFS is a zip of
  CSVs; GTFS-Realtime is a protobuf feed.
- Fallback: scrape the public MTS schedule pages if a feed is unavailable.

Keep the network/parsing logic in private helpers and let each ``@tool`` return
a compact, model-friendly summary.
"""

from langchain_core.tools import tool


@tool
def find_nearby_trolley_stations(location: str, radius_meters: int = 1500) -> str:
    """Find San Diego Trolley stations near a location.

    Args:
        location: Address or place name to search around.
        radius_meters: Search radius in meters.

    Returns:
        A short list of nearby stations with their lines and walking distance.
    """
    # TODO: Resolve `location` to coordinates, then find nearby trolley stops
    # (via GTFS stops.txt, a places search, or scraping). Return a short list.
    raise NotImplementedError


@tool
def get_trolley_schedule(station: str, line: str | None = None) -> str:
    """Get the published trolley timetable for a station.

    Args:
        station: Trolley station name (e.g. "Old Town").
        line: Optional line filter (e.g. "Blue", "Green", "Orange").

    Returns:
        A summary of scheduled departures for the station/line.
    """
    # TODO: Look up scheduled departures from GTFS static data (stop_times.txt
    # joined with trips.txt / routes.txt) or the schedule webpage.
    raise NotImplementedError


@tool
def get_next_trolley_arrivals(station: str, line: str | None = None) -> str:
    """Get the next (ideally live) trolley arrivals at a station.

    Args:
        station: Trolley station name.
        line: Optional line filter.

    Returns:
        The next few arrival times, marked live vs. scheduled.

    Note:
        If no realtime feed is available, fall back to the static schedule and
        clearly label the result as scheduled rather than live.
    """
    # TODO: Read GTFS-Realtime trip updates if available; otherwise fall back to
    # the static schedule and label the times as scheduled.
    raise NotImplementedError
