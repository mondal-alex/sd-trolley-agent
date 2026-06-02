"""San Diego Trolley (MTS) information tools.

These tools give the agent knowledge of the trolley system itself: which
stations exist near a location, the parking situation, and the published
timetable.

Data sources:
- Stations near a point and the timetable both come from the MTS **static GTFS**
  feed, loaded via ``gtfs.get_feed`` (see ``gtfs.py``). Using GTFS for both means
  station names are identical across tools. Geocoding an address to coordinates
  still uses the Google ``Geocoding API`` (``maps_client``).
- MTS does not publish a public real-time feed, so schedules are the *scheduled*
  times only: https://www.sdmts.com/business-center/app-developers

Network/parsing logic lives in private helpers; each ``@tool`` returns a
compact, model-friendly summary.
"""

from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from gtfs_kit.helpers import timestr_to_seconds
from langchain_core.tools import tool

from ._clients import maps_client
from .gtfs import get_feed

# GTFS route_type 0 == tram / streetcar / light rail (the trolley).
_LIGHT_RAIL_ROUTE_TYPE = 0
_MAX_STATIONS = 5
_MAX_DEPARTURES = 8
_METERS_PER_MILE = 1609.344


def _haversine_meters(lat0: float, lon0: float, lats, lons):
    """Great-circle distance (meters) from one point to arrays of points."""
    r = 6_371_000.0
    phi0 = np.radians(lat0)
    phi = np.radians(lats)
    dphi = np.radians(lats - lat0)
    dlmb = np.radians(lons - lon0)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi0) * np.cos(phi) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _trolley_stations(feed) -> pd.DataFrame:
    """Return light-rail (trolley) *stations* with name + coordinates.

    ``stop_times`` reference platform-level stops (location_type 0); we resolve
    each to its ``parent_station`` (location_type 1) when present so we report
    one clean station rather than each platform.
    """
    routes = feed.routes
    lr_routes = routes.loc[
        pd.to_numeric(routes["route_type"], errors="coerce") == _LIGHT_RAIL_ROUTE_TYPE,
        "route_id",
    ]
    lr_trips = feed.trips.loc[feed.trips["route_id"].isin(lr_routes), "trip_id"]
    served = feed.stop_times.loc[
        feed.stop_times["trip_id"].isin(lr_trips), "stop_id"
    ].unique()

    stops = feed.stops
    served_stops = stops[stops["stop_id"].isin(served)]

    # Resolve to parent station id where available, else the stop itself.
    if "parent_station" in served_stops.columns:
        parent = served_stops["parent_station"].fillna("").replace("", np.nan)
        station_ids = parent.fillna(served_stops["stop_id"]).unique()
    else:
        station_ids = served_stops["stop_id"].unique()

    stations = stops[stops["stop_id"].isin(station_ids)].copy()
    stations["stop_lat"] = pd.to_numeric(stations["stop_lat"], errors="coerce")
    stations["stop_lon"] = pd.to_numeric(stations["stop_lon"], errors="coerce")
    return stations.dropna(subset=["stop_lat", "stop_lon"])


def _now_pacific() -> datetime:
    """Current time in San Diego. Factored out so tests can patch it."""
    return datetime.now(ZoneInfo("America/Los_Angeles"))


def _format_clock(dep_seconds: float) -> str:
    """Format GTFS departure seconds as a 12-hour clock, e.g. "5:12 PM".

    GTFS times can exceed 24:00:00 (a post-midnight trip on the same service
    day), so wrap into a normal day with modulo.
    """
    seconds = int(dep_seconds) % 86_400
    return (
        dtime(hour=seconds // 3600, minute=(seconds % 3600) // 60)
        .strftime("%I:%M %p")
        .lstrip("0")
    )


@tool
def find_nearby_trolley_stations(location: str, radius_meters: int = 1500) -> str:
    """Find San Diego Trolley stations near a location.

    Args:
        location: Address or place name to search around.
        radius_meters: Search radius in meters.

    Returns:
        The closest trolley stations within the radius, with straight-line
        distance. Station names come from MTS GTFS, so they match the names used
        by get_trolley_schedule.
    """
    try:
        # 1. Resolve the location string to coordinates (Geocoding API).
        geocode = maps_client().geocode(location)
        if not geocode:
            return f"Could not find a location matching '{location}'."
        coords = geocode[0]["geometry"]["location"]

        # 2. Rank trolley stations (from GTFS) by distance from that point.
        stations = _trolley_stations(get_feed())
        if stations.empty:
            return "No trolley stations are available in the GTFS feed."

        stations = stations.assign(
            _dist_m=_haversine_meters(
                coords["lat"], coords["lng"], stations["stop_lat"], stations["stop_lon"]
            )
        )
        within = stations[stations["_dist_m"] <= radius_meters].sort_values("_dist_m")
        if within.empty:
            return f"No trolley stations found within {radius_meters} m of {location}."

        lines = [
            f"- {row['stop_name']} ({row['_dist_m'] / _METERS_PER_MILE:.1f} mi)"
            for _, row in within.head(_MAX_STATIONS).iterrows()
        ]
        return f"Trolley stations near {location}:\n" + "\n".join(lines)

    except Exception as e:
        return f"Error finding trolley stations: {e}"


@tool
def get_trolley_schedule(station: str, line: str | None = None) -> str:
    """Get the published (scheduled) trolley timetable for a station.

    Args:
        station: Trolley station name (e.g. "Old Town").
        line: Optional line filter (e.g. "Blue", "Green", "Orange").

    Returns:
        The next scheduled departures from the station today, from MTS static
        GTFS. These are scheduled times, not real-time arrivals.
    """
    try:
        feed = get_feed()

        # 1. Match the station name to GTFS stop_id(s) (case-insensitive).
        stops = feed.stops
        matches = stops[
            stops["stop_name"].str.contains(station, case=False, na=False)
        ]
        if matches.empty:
            return f"No trolley station matching '{station}' was found."
        stop_ids = matches["stop_id"].tolist()

        # 2. Build today's timetable for each matched stop.
        now = _now_pacific()
        today = now.strftime("%Y%m%d")
        frames = [
            tt
            for sid in stop_ids
            if not (tt := feed.build_stop_timetable(sid, [today])).empty
        ]
        if not frames:
            return f"No scheduled trolley service at '{station}' today."
        timetable = pd.concat(frames, ignore_index=True)

        # 3. Restrict to light-rail (trolley) routes so buses sharing a stop
        #    name are excluded, and attach the line names.
        routes = feed.routes[
            ["route_id", "route_short_name", "route_long_name", "route_type"]
        ]
        timetable = timetable.merge(routes, on="route_id", how="left")
        timetable = timetable[
            pd.to_numeric(timetable["route_type"], errors="coerce")
            == _LIGHT_RAIL_ROUTE_TYPE
        ]

        # 4. Optional line filter (matches short or long route name).
        if line:
            needle = line.lower()
            timetable = timetable[
                timetable["route_short_name"].fillna("").str.lower().str.contains(needle)
                | timetable["route_long_name"].fillna("").str.lower().str.contains(needle)
            ]

        if timetable.empty:
            target = f"the {line} line at '{station}'" if line else f"'{station}'"
            return f"No scheduled trolley service for {target} today."

        # 5. Keep upcoming departures (handles GTFS times >= 24:00:00).
        now_seconds = now.hour * 3600 + now.minute * 60 + now.second
        timetable = timetable.assign(
            _dep=timetable["departure_time"].map(timestr_to_seconds)
        )
        upcoming = timetable[timetable["_dep"] >= now_seconds].sort_values("_dep")
        if upcoming.empty:
            return f"No more scheduled trolley departures from '{station}' today."

        # 6. Format the next few departures.
        lines = []
        for _, row in upcoming.head(_MAX_DEPARTURES).iterrows():
            label = row.get("route_long_name") or row.get("route_short_name") or "Trolley"
            headsign = row.get("trip_headsign")
            when = _format_clock(row["_dep"])
            lines.append(f"- {when}  {label}" + (f"  (to {headsign})" if headsign else ""))

        return (
            f"Scheduled trolley departures from {station} today:\n"
            + "\n".join(lines)
            + "\n(Scheduled times from MTS static GTFS; real-time arrivals are not available.)"
        )

    except Exception as e:
        return f"Error getting trolley schedule: {e}"