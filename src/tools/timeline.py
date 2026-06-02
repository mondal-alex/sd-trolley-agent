"""Deterministic trip timeline math for arrive-by planning.

The LLM gathers drive times, trolley schedules, and walk times from other tools,
then passes the chosen legs here. This module computes clock times forward and
backward and rejects plans whose final arrival exceeds the user's deadline.
"""

import json

from langchain_core.tools import tool

from .trolley import _STATION_BUFFER_MINUTES, _format_clock, _parse_clock_to_seconds


def _add_minutes(clock: str, minutes: int) -> str:
    seconds = _parse_clock_to_seconds(clock) + minutes * 60
    return _format_clock(seconds)


def _subtract_minutes(clock: str, minutes: int) -> str:
    seconds = _parse_clock_to_seconds(clock) - minutes * 60
    return _format_clock(seconds)


def _minutes_between(start: str, end: str) -> int:
    return round((_parse_clock_to_seconds(end) - _parse_clock_to_seconds(start)) / 60)


def _build_timeline(
    *,
    target_arrival: str,
    destination: str,
    origin_label: str,
    boarding_station: str,
    exit_station: str,
    drive_minutes: int,
    walk_minutes: int,
    trolley_legs: list[dict],
    station_buffer_minutes: int = _STATION_BUFFER_MINUTES,
) -> str:
    if drive_minutes < 0 or walk_minutes < 0 or station_buffer_minutes < 0:
        raise ValueError("drive_minutes, walk_minutes, and station_buffer_minutes must be >= 0.")
    if not trolley_legs:
        raise ValueError("trolley_legs must contain at least one leg.")

    parsed_legs = []
    for i, leg in enumerate(trolley_legs):
        depart = leg.get("depart")
        arrive = leg.get("arrive")
        if not depart or not arrive:
            raise ValueError(f"Leg {i + 1} must include 'depart' and 'arrive' times.")
        desc = leg.get("description") or f"Trolley leg {i + 1}"
        parsed_legs.append(
            {
                "depart": depart,
                "arrive": arrive,
                "description": desc,
                "ride_minutes": _minutes_between(depart, arrive),
            }
        )

    first_depart = parsed_legs[0]["depart"]
    last_arrive = parsed_legs[-1]["arrive"]
    leave_home = _subtract_minutes(first_depart, station_buffer_minutes + drive_minutes)
    arrive_at_station = _add_minutes(leave_home, drive_minutes)
    buffer_end = _add_minutes(arrive_at_station, station_buffer_minutes)
    final_arrival = _add_minutes(last_arrive, walk_minutes)

    target_seconds = _parse_clock_to_seconds(target_arrival)
    final_seconds = _parse_clock_to_seconds(final_arrival)
    first_dep_seconds = _parse_clock_to_seconds(first_depart)
    leave_seconds = _parse_clock_to_seconds(leave_home)
    needed_before_board = drive_minutes + station_buffer_minutes

    errors = []
    if final_seconds > target_seconds:
        errors.append(
            f"Final arrival at {destination} would be {final_arrival}, which is after "
            f"the target of {target_arrival}. Pick an earlier trolley trip or another route."
        )
    if leave_seconds + needed_before_board * 60 > first_dep_seconds:
        errors.append(
            f"Cannot reach {boarding_station} in time to board at {first_depart}: "
            f"leave {leave_home} + {drive_minutes} min drive + {station_buffer_minutes} min "
            f"buffer ends at {buffer_end}, after departure."
        )
    for i in range(1, len(parsed_legs)):
        prev_arrive = parsed_legs[i - 1]["arrive"]
        next_depart = parsed_legs[i]["depart"]
        if _parse_clock_to_seconds(next_depart) < _parse_clock_to_seconds(prev_arrive):
            errors.append(
                f"Leg {i + 1} departs at {next_depart} before leg {i} arrives at "
                f"{prev_arrive}."
            )

    if errors:
        return "Timeline validation failed:\n- " + "\n- ".join(errors)

    lines = [
        f"Trip plan to {destination} (arrive by {target_arrival}):",
        f"- Leave {origin_label}: {leave_home}",
        f"- Drive to {boarding_station}: {drive_minutes} min → arrive {arrive_at_station}",
        (
            f"- Station buffer at {boarding_station}: {station_buffer_minutes} min "
            f"({arrive_at_station}–{buffer_end})"
        ),
    ]
    for leg in parsed_legs:
        lines.append(
            f"- {leg['description']}: depart {leg['depart']} → arrive {leg['arrive']} "
            f"({leg['ride_minutes']} min)"
        )
    lines.append(
        f"- Walk from {exit_station} to {destination}: {walk_minutes} min "
        f"({last_arrive}–{final_arrival})"
    )
    lines.append(f"- Final arrival at {destination}: {final_arrival}")
    if final_seconds < target_seconds:
        slack = round((target_seconds - final_seconds) / 60)
        lines.append(f"  ({slack} min before your {target_arrival} deadline)")
    lines.append("")
    lines.append(
        "Present this timeline to the user. Do not change any clock times — they are "
        "computed from the tool inputs."
    )
    return "\n".join(lines)


@tool
def build_trip_timeline(
    target_arrival: str,
    destination: str,
    origin_label: str,
    boarding_station: str,
    exit_station: str,
    drive_minutes: int,
    walk_minutes: int,
    trolley_legs_json: str,
    station_buffer_minutes: int = _STATION_BUFFER_MINUTES,
) -> str:
    """Build and validate a full trip timeline from tool-gathered data.

    Call this **after** you have drive time, walk time, and chosen trolley trip(s)
    from the other tools. Pass the exact departure/arrival times from
    ``get_trolley_trips_between_stations`` (one object per trolley leg, including
    transfers). This tool computes leave-home time, station buffer, and final
    arrival — and rejects plans that miss the deadline.

    Args:
        target_arrival: User's latest acceptable arrival at ``destination``, e.g.
            "6:00 PM".
        destination: Final venue name, e.g. "Petco Park".
        origin_label: Where the user starts, e.g. "home (4109 Park Pl)".
        boarding_station: First trolley station where they board.
        exit_station: Last trolley station before the final walk.
        drive_minutes: Minutes from ``get_driving_time`` (origin → boarding_station).
        walk_minutes: Minutes from ``get_walking_time`` (exit_station → destination).
        trolley_legs_json: JSON array of legs, each with ``depart``, ``arrive``, and
            optional ``description``. Example:
            ``[{"depart": "5:05 PM", "arrive": "5:35 PM", "description": "Blue Line Old Town → America Plaza"}]``
        station_buffer_minutes: Minutes at the boarding station before first departure
            (default 8: park, walk to platform, ticket ready).

    Returns:
        A validated chronological timeline with explicit clock times, or an error
        explaining why the chosen trips do not meet the deadline.
    """
    try:
        legs = json.loads(trolley_legs_json)
        if not isinstance(legs, list):
            return "trolley_legs_json must be a JSON array of leg objects."
        return _build_timeline(
            target_arrival=target_arrival,
            destination=destination,
            origin_label=origin_label,
            boarding_station=boarding_station,
            exit_station=exit_station,
            drive_minutes=drive_minutes,
            walk_minutes=walk_minutes,
            trolley_legs=legs,
            station_buffer_minutes=station_buffer_minutes,
        )
    except json.JSONDecodeError as e:
        return f"Invalid trolley_legs_json: {e}"
    except ValueError as e:
        return f"Invalid timeline input: {e}"
    except Exception as e:
        return f"Error building trip timeline: {e}"
