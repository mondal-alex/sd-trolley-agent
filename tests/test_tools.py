"""Unit tests for the agent's tools.

Tools call external services (Google Maps, MTS feeds), so in tests you'll want
to mock those network calls and assert on the formatted string each tool
returns. Tools are decorated with ``@tool``; invoke them with ``.invoke({...})``
or call the underlying function via ``.func(...)``.
"""

import json
import zipfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import gtfs_kit as gk
import pytest
from google.maps import routing_v2

from src.tools import ALL_TOOLS
from src.tools.clock import get_current_time
from src.tools.parking import get_station_parking_info
from src.tools.routing import get_driving_time, get_walking_time
from src.tools.timeline import build_trip_timeline
from src.tools.trolley import (
    find_nearby_trolley_stations,
    get_trolley_schedule,
    get_trolley_trips_between_stations,
)


def test_all_tools_exposed():
    """Every tool should be registered for the agent to bind."""
    assert len(ALL_TOOLS) == 8
    names = {t.name for t in ALL_TOOLS}
    assert "get_driving_time" in names
    assert "get_current_time" in names
    assert "get_station_parking_info" in names
    assert "get_trolley_schedule" in names
    assert "get_trolley_trips_between_stations" in names
    assert "build_trip_timeline" in names
    # The live-arrivals tool was removed (MTS has no public real-time feed).
    assert "get_next_trolley_arrivals" not in names


class TestGetCurrentTime:
    """Tests for the get_current_time tool."""

    def test_returns_nonempty_string_with_year(self):
        """A live call should return a formatted string containing the year."""
        result = get_current_time.invoke({})
        assert isinstance(result, str)
        assert result
        assert str(datetime.now(ZoneInfo("America/Los_Angeles")).year) in result

    def test_formats_in_pacific_timezone(self):
        """The time should be formatted from a fixed Pacific datetime."""
        fixed = datetime(2026, 6, 1, 14, 20, tzinfo=ZoneInfo("America/Los_Angeles"))

        with patch("src.tools.clock.datetime") as mock_datetime:
            mock_datetime.now.return_value = fixed
            result = get_current_time.invoke({})

        # Matches the documented "Monday 2026-06-01 14:20 PDT" style output.
        assert result == "Monday 2026-06-01 14:20 PDT"
        mock_datetime.now.assert_called_once_with(ZoneInfo("America/Los_Angeles"))


class TestRoutingTools:
    """Tests for the Routes API driving/walking tools (client mocked)."""

    @staticmethod
    def _mock_client_returning(duration, distance_meters):
        """A mock RoutesClient whose compute_routes returns one route."""
        route = Mock()
        route.duration = duration
        route.distance_meters = distance_meters
        client = Mock()
        client.compute_routes.return_value = Mock(routes=[route])
        return client

    def test_driving_time_formats_summary_and_is_traffic_aware(self):
        client = self._mock_client_returning(timedelta(minutes=18), 11909)

        with patch("src.tools.routing.routes_client", return_value=client):
            result = get_driving_time.invoke({"origin": "A", "destination": "B"})

        assert result == "Driving: 18 min (7.4 mi)"

        kwargs = client.compute_routes.call_args.kwargs
        request = kwargs["request"]
        assert request.travel_mode == routing_v2.RouteTravelMode.DRIVE
        assert request.routing_preference == routing_v2.RoutingPreference.TRAFFIC_AWARE
        # Field mask must be supplied or the Routes API returns nothing.
        assert ("x-goog-fieldmask", "routes.duration,routes.distanceMeters") in kwargs[
            "metadata"
        ]

    def test_walking_time_formats_summary_and_no_traffic_preference(self):
        client = self._mock_client_returning(timedelta(minutes=9), 644)

        with patch("src.tools.routing.routes_client", return_value=client):
            result = get_walking_time.invoke({"origin": "A", "destination": "B"})

        assert result == "Walking: 9 min (0.4 mi)"

        request = client.compute_routes.call_args.kwargs["request"]
        assert request.travel_mode == routing_v2.RouteTravelMode.WALK
        # routing_preference is DRIVE-only; walking must leave it unset.
        assert request.routing_preference != routing_v2.RoutingPreference.TRAFFIC_AWARE

    def test_no_route_found(self):
        client = Mock()
        client.compute_routes.return_value = Mock(routes=[])

        with patch("src.tools.routing.routes_client", return_value=client):
            result = get_driving_time.invoke({"origin": "A", "destination": "B"})

        assert "No driving route found" in result

    def test_errors_are_returned_as_strings(self):
        with patch(
            "src.tools.routing.routes_client",
            side_effect=RuntimeError("boom"),
        ):
            result = get_walking_time.invoke({"origin": "A", "destination": "B"})

        assert "Error getting walking time" in result


# A tiny but valid GTFS feed:
# - one trolley line (light rail, route_type 0) serving two stations,
# - one bus line (route_type 3) serving a bus-only stop plus the Old Town stop,
# - departures spanning the day plus one post-midnight (>= 24:00:00) trip.
_GTFS_FILES = {
    "agency.txt": (
        "agency_id,agency_name,agency_url,agency_timezone\n"
        "MTS,MTS,https://sdmts.com,America/Los_Angeles\n"
    ),
    "routes.txt": (
        "route_id,agency_id,route_short_name,route_long_name,route_type\n"
        "blue,MTS,Blue,UC San Diego Blue Line,0\n"
        "bus1,MTS,900,Some Bus,3\n"
    ),
    "trips.txt": (
        "route_id,service_id,trip_id,trip_headsign,direction_id\n"
        "blue,WEEK,t0,UTC,0\n"
        "blue,WEEK,t1,UTC,0\n"
        "blue,WEEK,t2,UTC,0\n"
        "blue,WEEK,t3,UTC,0\n"
        "bus1,WEEK,b1,Downtown,0\n"
    ),
    "calendar.txt": (
        "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,"
        "start_date,end_date\n"
        "WEEK,1,1,1,1,1,1,1,20260101,20271231\n"
    ),
    # s1/s2 are trolley stations; sbus is a bus-only stop near s1.
    "stops.txt": (
        "stop_id,stop_name,stop_lat,stop_lon\n"
        "s1,Old Town,32.7551,-117.1995\n"
        "s2,Santa Fe Depot,32.7187,-117.1699\n"
        "sbus,Bus Only Stop,32.7550,-117.1990\n"
    ),
    "stop_times.txt": (
        "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
        "t0,06:00:00,06:00:00,s1,1\n"
        "t1,17:10:00,17:10:00,s1,1\n"
        "t1,17:20:00,17:20:00,s2,2\n"
        "t2,17:25:00,17:25:00,s1,1\n"
        "t3,24:30:00,24:30:00,s1,1\n"
        "b1,17:15:00,17:15:00,sbus,1\n"
        "b1,17:18:00,17:18:00,s1,2\n"
    ),
}


@pytest.fixture
def gtfs_feed(tmp_path):
    """Build the tiny GTFS fixture and read it with gtfs-kit."""
    zip_path = tmp_path / "gtfs.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name, content in _GTFS_FILES.items():
            zf.writestr(name, content)
    return gk.read_feed(zip_path, dist_units="mi")


class TestFindNearbyTrolleyStations:
    """Tests for find_nearby_trolley_stations (geocode mocked + GTFS feed)."""

    @staticmethod
    def _geocoder(lat=32.7551, lng=-117.1995):
        gmaps = Mock()
        gmaps.geocode.return_value = [{"geometry": {"location": {"lat": lat, "lng": lng}}}]
        return gmaps

    def test_lists_nearest_trolley_stations(self, gtfs_feed):
        gmaps = self._geocoder()  # right at the Old Town stop

        with patch("src.tools.trolley.maps_client", return_value=gmaps), patch(
            "src.tools.trolley.get_feed", return_value=gtfs_feed
        ):
            result = find_nearby_trolley_stations.invoke(
                {"location": "Old Town", "radius_meters": 10000}
            )

        assert "Trolley stations near Old Town" in result
        assert "Old Town" in result
        assert "Santa Fe Depot" in result
        assert "mi)" in result  # distance shown
        # Bus-only stops are excluded (light-rail only).
        assert "Bus Only Stop" not in result
        # Closest station (Old Town) is listed before the farther one.
        assert result.index("Old Town") < result.index("Santa Fe Depot")

    def test_lists_closest_when_none_within_radius(self, gtfs_feed):
        # Point far from fixture stations so a small radius finds nothing nearby.
        gmaps = self._geocoder(lat=32.60, lng=-117.10)

        with patch("src.tools.trolley.maps_client", return_value=gmaps), patch(
            "src.tools.trolley.get_feed", return_value=gtfs_feed
        ):
            result = find_nearby_trolley_stations.invoke(
                {"location": "remote point", "radius_meters": 500}
            )

        assert "No trolley stations within 500 m" in result
        assert "Closest stations" in result
        assert "Old Town" in result

    def test_excludes_stations_outside_radius(self, gtfs_feed):
        gmaps = self._geocoder()

        with patch("src.tools.trolley.maps_client", return_value=gmaps), patch(
            "src.tools.trolley.get_feed", return_value=gtfs_feed
        ):
            result = find_nearby_trolley_stations.invoke(
                {"location": "Old Town", "radius_meters": 1500}
            )

        # 1500 m radius includes Old Town but not the ~5 km Santa Fe Depot.
        assert "Old Town" in result
        assert "Santa Fe Depot" not in result

    def test_location_not_found(self, gtfs_feed):
        gmaps = Mock()
        gmaps.geocode.return_value = []

        with patch("src.tools.trolley.maps_client", return_value=gmaps), patch(
            "src.tools.trolley.get_feed", return_value=gtfs_feed
        ):
            result = find_nearby_trolley_stations.invoke({"location": "nowhere"})

        assert "Could not find a location" in result

    def test_errors_are_returned_as_strings(self):
        with patch("src.tools.trolley.maps_client", side_effect=RuntimeError("boom")):
            result = find_nearby_trolley_stations.invoke({"location": "Old Town"})

        assert "Error finding trolley stations" in result


class TestGetTrolleySchedule:
    """Tests for get_trolley_schedule against a tiny in-memory GTFS feed."""

    # Monday 5:00 PM Pacific -> "today" has service and 6 AM is already past.
    _NOW = datetime(2026, 6, 1, 17, 0, 0, tzinfo=ZoneInfo("America/Los_Angeles"))

    def _run(self, feed, **kwargs):
        with patch("src.tools.trolley.get_feed", return_value=feed), patch(
            "src.tools.trolley._now_pacific", return_value=self._NOW
        ):
            return get_trolley_schedule.invoke({"station": "Old Town", **kwargs})

    def test_lists_upcoming_light_rail_departures(self, gtfs_feed):
        result = self._run(gtfs_feed)

        assert "Scheduled trolley departures from Old Town today:" in result
        assert "5:10 PM" in result
        assert "5:25 PM" in result
        assert "UC San Diego Blue Line" in result
        assert "(to UTC)" in result
        # Post-midnight trip (24:30:00) wraps to a 12-hour clock.
        assert "12:30 AM" in result
        # Past departure is excluded.
        assert "6:00 AM" not in result
        # Buses sharing the stop are filtered out (light-rail only).
        assert "Some Bus" not in result
        assert "Downtown" not in result
        # Clear that these are scheduled, not live.
        assert "real-time arrivals are not available" in result

    def test_line_filter_matches(self, gtfs_feed):
        result = self._run(gtfs_feed, line="Blue")
        assert "UC San Diego Blue Line" in result

    def test_line_filter_no_match(self, gtfs_feed):
        result = self._run(gtfs_feed, line="Green")
        assert "No scheduled trolley service for the Green line" in result

    def test_future_service_date_ignores_now(self, gtfs_feed):
        # A future Friday: the current time must NOT filter out earlier
        # departures (the morning 6:00 AM trip should still be listed).
        result = self._run(gtfs_feed, service_date="2026-06-05")
        assert (
            "Scheduled trolley departures from Old Town on Friday 2026-06-05:"
            in result
        )
        assert "6:00 AM" in result
        assert "5:10 PM" in result
        assert "5:25 PM" in result

    def test_before_window_limits_departures(self, gtfs_feed):
        # Today, only departures up to 5:20 PM -> just the 5:10 PM trolley
        # (5:25 PM is too late, 6:00 AM already past "now" of 5:00 PM).
        result = self._run(gtfs_feed, before="17:20")
        assert "5:10 PM" in result
        assert "5:25 PM" not in result

    def test_after_and_before_window_on_future_date(self, gtfs_feed):
        # 12-hour clock input is accepted; window keeps only the 5:10 PM trip.
        result = self._run(
            gtfs_feed, service_date="2026-06-05", after="5:00 PM", before="5:20 PM"
        )
        assert "5:10 PM" in result
        assert "6:00 AM" not in result
        assert "5:25 PM" not in result

    def test_empty_window_message(self, gtfs_feed):
        result = self._run(
            gtfs_feed, service_date="2026-06-05", after="19:00", before="20:00"
        )
        assert "No scheduled trolley departures" in result
        assert "in the requested time window" in result

    def test_invalid_date_returns_error_string(self, gtfs_feed):
        result = self._run(gtfs_feed, service_date="not-a-date")
        assert "Error getting trolley schedule" in result

    def test_unknown_station(self, gtfs_feed):
        with patch("src.tools.trolley.get_feed", return_value=gtfs_feed), patch(
            "src.tools.trolley._now_pacific", return_value=self._NOW
        ):
            result = get_trolley_schedule.invoke({"station": "Nowhere"})
        assert "No trolley station matching 'Nowhere' was found." in result

    def test_errors_are_returned_as_strings(self):
        with patch("src.tools.trolley.get_feed", side_effect=RuntimeError("boom")):
            result = get_trolley_schedule.invoke({"station": "Old Town"})
        assert "Error getting trolley schedule" in result


class TestGetTrolleyTripsBetweenStations:
    """Tests for arrive-by-aware trip listing between two stations."""

    def _run(self, feed, **kwargs):
        with patch("src.tools.trolley.get_feed", return_value=feed):
            return get_trolley_trips_between_stations.invoke(
                {"from_station": "Old Town", "to_station": "Santa Fe Depot", **kwargs}
            )

    def test_lists_trip_with_arrival_by(self, gtfs_feed):
        result = self._run(gtfs_feed, service_date="2026-06-05", arrive_by="17:30")
        assert "Depart 5:10 PM" in result
        assert "Arrive 5:20 PM" in result
        assert "10 min ride" in result
        assert "UC San Diego Blue Line" in result

    def test_arrive_by_excludes_late_trips(self, gtfs_feed):
        result = self._run(gtfs_feed, service_date="2026-06-05", arrive_by="17:15")
        assert "No scheduled trolley trips" in result

    def test_unknown_station(self, gtfs_feed):
        with patch("src.tools.trolley.get_feed", return_value=gtfs_feed):
            result = get_trolley_trips_between_stations.invoke(
                {
                    "from_station": "Nowhere",
                    "to_station": "Santa Fe Depot",
                }
            )
        assert "No trolley station matching 'Nowhere'" in result


class TestBuildTripTimeline:
    """Tests for deterministic arrive-by timeline assembly."""

    _LEGS = json.dumps(
        [
            {
                "depart": "5:05 PM",
                "arrive": "5:35 PM",
                "description": "Blue Line Old Town → America Plaza",
            }
        ]
    )

    def test_valid_single_leg_plan(self):
        result = build_trip_timeline.invoke(
            {
                "target_arrival": "6:00 PM",
                "destination": "Petco Park",
                "origin_label": "home",
                "boarding_station": "Old Town",
                "exit_station": "America Plaza",
                "drive_minutes": 7,
                "walk_minutes": 10,
                "trolley_legs_json": self._LEGS,
            }
        )
        assert "Timeline validation failed" not in result
        assert "Leave home: 4:50 PM" in result
        assert "Final arrival at Petco Park: 5:45 PM" in result
        assert "15 min before your 6:00 PM deadline" in result

    def test_rejects_late_plan_like_screenshot(self):
        legs = json.dumps(
            [
                {
                    "depart": "5:54 PM",
                    "arrive": "6:06 PM",
                    "description": "Blue Line Mission San Diego → America Plaza",
                },
                {
                    "depart": "6:10 PM",
                    "arrive": "6:20 PM",
                    "description": "Green Line America Plaza → 12th & Imperial",
                },
            ]
        )
        result = build_trip_timeline.invoke(
            {
                "target_arrival": "6:00 PM",
                "destination": "Petco Park",
                "origin_label": "home",
                "boarding_station": "Mission San Diego",
                "exit_station": "12th & Imperial Transit Center",
                "drive_minutes": 4,
                "walk_minutes": 4,
                "trolley_legs_json": legs,
            }
        )
        assert "Timeline validation failed" in result
        assert "6:24 PM" in result

    def test_rejects_overlapping_transfer_legs(self):
        legs = json.dumps(
            [
                {"depart": "5:00 PM", "arrive": "5:20 PM", "description": "Leg 1"},
                {"depart": "5:15 PM", "arrive": "5:35 PM", "description": "Leg 2"},
            ]
        )
        result = build_trip_timeline.invoke(
            {
                "target_arrival": "6:00 PM",
                "destination": "Petco Park",
                "origin_label": "home",
                "boarding_station": "Old Town",
                "exit_station": "12th & Imperial",
                "drive_minutes": 7,
                "walk_minutes": 4,
                "trolley_legs_json": legs,
            }
        )
        assert "Timeline validation failed" in result
        assert "before leg 1 arrives" in result


class TestGetStationParkingInfo:
    """Tests for get_station_parking_info (curated MTS park-and-ride data)."""

    def test_free_lot_with_capacity(self):
        result = get_station_parking_info.invoke({"station": "Old Town"})
        assert "free park-and-ride" in result
        assert "412 spaces" in result
        assert "paid" not in result.replace("park-and-ride", "")

    def test_paid_lot_flagged(self):
        result = get_station_parking_info.invoke({"station": "UTC Transit Center"})
        assert "paid park-and-ride" in result
        assert "333 spaces" in result

    def test_matches_gtfs_style_name_with_suffix(self):
        # find_nearby/schedule may pass names like "Nobel Drive Station".
        result = get_station_parking_info.invoke({"station": "Nobel Drive Station"})
        assert "Nobel Drive" in result
        assert "free park-and-ride" in result

    def test_varies_capacity(self):
        # GTFS-style name should match the curated "Santee Town Center" lot.
        result = get_station_parking_info.invoke({"station": "Santee Town Center Station"})
        assert "capacity varies" in result

    def test_unknown_station_is_explicit(self):
        result = get_station_parking_info.invoke({"station": "America Plaza"})
        assert "No MTS park-and-ride lot is listed" in result
        assert "only street or paid parking" in result
