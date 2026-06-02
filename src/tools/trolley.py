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
_MAX_TRIP_OPTIONS = 8
_METERS_PER_MILE = 1609.344
_STATION_BUFFER_MINUTES = 8


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


def _parse_service_date(value: str) -> str:
    """Parse a service date into GTFS ``YYYYMMDD`` form.

    Accepts ISO ``YYYY-MM-DD``, compact ``YYYYMMDD``, or ``MM/DD/YYYY``.
    """
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(value.strip(), fmt).strftime("%Y%m%d")
        except ValueError:
            continue
    raise ValueError(f"Could not parse date '{value}'; use YYYY-MM-DD (e.g. 2026-06-05).")


def _match_stop_ids(station: str, stops: pd.DataFrame) -> list[str]:
    """Return GTFS stop_ids whose name contains ``station`` (case-insensitive)."""
    matches = stops[stops["stop_name"].str.contains(station, case=False, na=False)]
    return matches["stop_id"].tolist()


def _light_rail_trip_ids(feed) -> set[str]:
    routes = feed.routes
    lr_routes = routes.loc[
        pd.to_numeric(routes["route_type"], errors="coerce") == _LIGHT_RAIL_ROUTE_TYPE,
        "route_id",
    ]
    return set(feed.trips.loc[feed.trips["route_id"].isin(lr_routes), "trip_id"])


def _trips_on_date(feed, date: str) -> set[str]:
    """Trip IDs in service on ``date`` (YYYYMMDD)."""
    day_trips = feed.get_trips(date)
    if day_trips is None or day_trips.empty:
        return set()
    return set(day_trips["trip_id"])


def _parse_clock_to_seconds(value: str) -> int:
    """Parse a clock string into seconds since midnight.

    Accepts 24-hour ``HH:MM`` or 12-hour ``H:MM AM/PM`` forms.
    """
    v = value.strip().upper()
    for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p", "%I %p"):
        try:
            t = datetime.strptime(v, fmt)
            return t.hour * 3600 + t.minute * 60
        except ValueError:
            continue
    raise ValueError(f"Could not parse time '{value}'; use HH:MM (e.g. 17:00).")


@tool
def find_nearby_trolley_stations(location: str, radius_meters: int = 15000) -> str:
    """Find San Diego Trolley stations near a location.

    Args:
        location: Address, zip code, or place name to search around.
        radius_meters: Search radius in meters. Default 15000 (about 9 mi) suits
            users who will *drive* to a park-and-ride station. Use a smaller value
            only for walking-distance checks.

    Returns:
        The closest trolley stations within the radius, with straight-line
        distance. If none fall inside the radius, the nearest stations are still
        listed. Station names come from MTS GTFS and match get_trolley_schedule.
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
            # User may drive to a station — still return the nearest options instead
            # of implying the trolley is unavailable.
            nearest = stations.sort_values("_dist_m").head(_MAX_STATIONS)
            lines = [
                f"- {row['stop_name']} ({row['_dist_m'] / _METERS_PER_MILE:.1f} mi)"
                for _, row in nearest.iterrows()
            ]
            return (
                f"No trolley stations within {radius_meters} m of {location}. "
                f"Closest stations (straight-line; use get_driving_time to reach them):\n"
                + "\n".join(lines)
            )

        lines = [
            f"- {row['stop_name']} ({row['_dist_m'] / _METERS_PER_MILE:.1f} mi)"
            for _, row in within.head(_MAX_STATIONS).iterrows()
        ]
        return f"Trolley stations near {location}:\n" + "\n".join(lines)

    except Exception as e:
        return f"Error finding trolley stations: {e}"


@tool
def get_trolley_schedule(
    station: str,
    line: str | None = None,
    service_date: str | None = None,
    after: str | None = None,
    before: str | None = None,
) -> str:
    """Get the published (scheduled) trolley timetable for a station.

    Args:
        station: Trolley station name (e.g. "Old Town").
        line: Optional line filter (e.g. "Blue", "Green", "Orange").
        service_date: Optional service date as "YYYY-MM-DD" (defaults to today
            in San Diego). Use this to plan for a future day, e.g. "this Friday".
        after: Optional earliest departure time as "HH:MM" (24-hour) or
            "H:MM AM/PM". Departures before this are excluded. When omitted and
            the date is today, the current time is used so only upcoming
            departures are shown.
        before: Optional latest departure time as "HH:MM" / "H:MM AM/PM".
            Departures after this are excluded. Useful to find the last trolley
            that still arrives by a target time.

    Returns:
        The scheduled departures from the station for the requested date and
        time window, from MTS static GTFS. These are scheduled times, not
        real-time arrivals.
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

        # 2. Resolve the service date (defaults to today) and parse the optional
        #    time-window bounds up front so bad input fails with a clear message.
        now = _now_pacific()
        today = now.strftime("%Y%m%d")
        date = _parse_service_date(service_date) if service_date else today
        is_today = date == today
        # "today" reads naturally on its own; a specific date wants an "on" prefix.
        if is_today:
            date_label = "today"
            when_label = "today"
        else:
            date_label = datetime.strptime(date, "%Y%m%d").strftime("%A %Y-%m-%d")
            when_label = f"on {date_label}"
        after_seconds = _parse_clock_to_seconds(after) if after else None
        before_seconds = _parse_clock_to_seconds(before) if before else None

        # 3. Build the requested day's timetable for each matched stop.
        frames = [
            tt
            for sid in stop_ids
            if not (tt := feed.build_stop_timetable(sid, [date])).empty
        ]
        if not frames:
            return f"No scheduled trolley service at '{station}' {when_label}."
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

        # 5. Apply the time window (handles GTFS times >= 24:00:00).
        #    Lower bound: explicit `after`, else "now" only when planning today.
        #    Upper bound: explicit `before`. No bound otherwise.
        timetable = timetable.assign(
            _dep=timetable["departure_time"].map(timestr_to_seconds)
        )
        if after_seconds is not None:
            lower = after_seconds
        elif is_today:
            lower = now.hour * 3600 + now.minute * 60 + now.second
        else:
            lower = None

        window = timetable
        if lower is not None:
            window = window[window["_dep"] >= lower]
        if before_seconds is not None:
            window = window[window["_dep"] <= before_seconds]
        window = window.sort_values("_dep")

        if window.empty:
            if lower is not None and before_seconds is None and is_today:
                return f"No more scheduled trolley departures from '{station}' today."
            return (
                f"No scheduled trolley departures from '{station}' "
                f"{when_label} in the requested time window."
            )

        # 6. Format the matching departures.
        lines = []
        for _, row in window.head(_MAX_DEPARTURES).iterrows():
            label = row.get("route_long_name") or row.get("route_short_name") or "Trolley"
            headsign = row.get("trip_headsign")
            when = _format_clock(row["_dep"])
            lines.append(f"- {when}  {label}" + (f"  (to {headsign})" if headsign else ""))

        return (
            f"Scheduled trolley departures from {station} {when_label}:\n"
            + "\n".join(lines)
            + "\n(Scheduled times from MTS static GTFS; real-time arrivals are not available.)"
        )

    except Exception as e:
        return f"Error getting trolley schedule: {e}"


@tool
def get_trolley_trips_between_stations(
    from_station: str,
    to_station: str,
    service_date: str | None = None,
    arrive_by: str | None = None,
    after_departure: str | None = None,
) -> str:
    """List scheduled trolley trips from one station to another (same vehicle).

    Use this when the user has an **arrive-by** deadline. Each line shows
    departure from ``from_station`` and scheduled **arrival at ``to_station``**
    on the same trip. Only trips that reach ``to_station`` by ``arrive_by`` are
    included (when ``arrive_by`` is set).

    Args:
        from_station: Boarding station name (e.g. "Mission San Diego").
        to_station: Alighting station near the destination (e.g. "12th & Imperial"
            for Petco Park). This is where the rider exits the trolley, not the
            final venue — add walking time separately with ``get_walking_time``.
        service_date: Service date "YYYY-MM-DD" (defaults to today).
        arrive_by: Latest acceptable **scheduled arrival at to_station** as
            "HH:MM" or "H:MM AM/PM". For "arrive at Petco by 6:00 PM", first
            subtract walking time from the venue to ``to_station``, then pass that
            earlier time here.
        after_departure: Optional earliest departure from ``from_station``.

    Returns:
        Matching trips sorted by departure (latest departures first when
        ``arrive_by`` is set — pick the top line for the tightest feasible plan).
    """
    try:
        feed = get_feed()
        from_ids = _match_stop_ids(from_station, feed.stops)
        to_ids = _match_stop_ids(to_station, feed.stops)
        if not from_ids:
            return f"No trolley station matching '{from_station}' was found."
        if not to_ids:
            return f"No trolley station matching '{to_station}' was found."

        now = _now_pacific()
        today = now.strftime("%Y%m%d")
        date = _parse_service_date(service_date) if service_date else today
        if date == today:
            when_label = "today"
        else:
            when_label = datetime.strptime(date, "%Y%m%d").strftime("%A %Y-%m-%d")

        arrive_by_seconds = (
            _parse_clock_to_seconds(arrive_by) if arrive_by else None
        )
        after_dep_seconds = (
            _parse_clock_to_seconds(after_departure) if after_departure else None
        )

        lr_trips = _light_rail_trip_ids(feed)
        active = _trips_on_date(feed, date)
        if not active:
            return f"No scheduled trolley service {when_label}."
        trip_ids = lr_trips & active

        st = feed.stop_times.loc[feed.stop_times["trip_id"].isin(trip_ids)].copy()
        st["_dep"] = st["departure_time"].map(timestr_to_seconds)
        st["_arr"] = st["arrival_time"].map(timestr_to_seconds)

        rows = []
        for trip_id, group in st.groupby("trip_id"):
            group = group.sort_values("stop_sequence")
            from_rows = group[group["stop_id"].isin(from_ids)]
            to_rows = group[group["stop_id"].isin(to_ids)]
            if from_rows.empty or to_rows.empty:
                continue
            o_seq = from_rows["stop_sequence"].min()
            d_seq = to_rows["stop_sequence"].min()
            if d_seq <= o_seq:
                continue
            dep_row = from_rows.sort_values("stop_sequence").iloc[0]
            arr_row = to_rows.sort_values("stop_sequence").iloc[0]
            dep_s = int(dep_row["_dep"])
            arr_s = int(arr_row["_arr"])
            if after_dep_seconds is not None and dep_s < after_dep_seconds:
                continue
            if arrive_by_seconds is not None and arr_s > arrive_by_seconds:
                continue
            trip = feed.trips.loc[feed.trips["trip_id"] == trip_id].iloc[0]
            route = feed.routes.loc[feed.routes["route_id"] == trip["route_id"]].iloc[0]
            label = route.get("route_long_name") or route.get("route_short_name") or "Trolley"
            rows.append(
                {
                    "_dep": dep_s,
                    "_arr": arr_s,
                    "line": label,
                    "headsign": trip.get("trip_headsign"),
                    "dep": dep_s,
                    "arr": arr_s,
                }
            )

        if not rows:
            msg = (
                f"No scheduled trolley trips from '{from_station}' to '{to_station}' "
                f"{when_label}"
            )
            if arrive_by:
                msg += f" with arrival at '{to_station}' by {arrive_by}"
            msg += "."
            return msg

        # Latest departure first when filtering by arrive_by (tightest feasible).
        rows.sort(key=lambda r: r["_dep"], reverse=bool(arrive_by_seconds))
        lines = []
        for r in rows[:_MAX_TRIP_OPTIONS]:
            dep = _format_clock(r["dep"])
            arr = _format_clock(r["arr"])
            ride_min = max(1, round((r["arr"] - r["dep"]) / 60))
            head = f"  (to {r['headsign']})" if r.get("headsign") else ""
            lines.append(
                f"- Depart {dep} → Arrive {arr} ({ride_min} min ride)  {r['line']}{head}"
            )

        header = (
            f"Scheduled trolley trips from {from_station} to {to_station} {when_label}"
        )
        if arrive_by:
            header += f" (arrival at {to_station} by {arrive_by})"
        return (
            header
            + ":\n"
            + "\n".join(lines)
            + "\n(Scheduled GTFS times; ride duration is on the trolley only — use "
            f"`get_walking_time` for the walk after alighting. "
            f"{_STATION_BUFFER_MINUTES} min buffer at origin station before boarding.)"
        )

    except Exception as e:
        return f"Error getting trolley trips between stations: {e}"