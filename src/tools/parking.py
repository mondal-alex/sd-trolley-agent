"""Trolley station park-and-ride information.

Unlike schedules/stations (GTFS) or routing (Routes API), MTS does not publish a
structured feed of park-and-ride details, so this data is curated by hand from
the official MTS page:

    https://www.sdmts.com/transit-services/transit-station-parking

Key facts that drive "avoid paid parking" style questions:
- Every listed park-and-ride lot is **free** except **UTC Transit Center**,
  which is pay parking.
- Stations *not* in this table have no free MTS lot; only street or paid parking
  may be nearby. We say so explicitly rather than guessing.

Station names here are kept close to GTFS ``stop_name`` so the agent can pass a
name straight from ``find_nearby_trolley_stations`` / ``get_trolley_schedule``.
Matching is by exact name after normalization (case-insensitive, ignoring
"Station"/"Transit Center"); unrecognized names return "no listed lot" rather
than guessing.
"""

import re

from langchain_core.tools import tool


class _Lot:
    """One station's park-and-ride entry."""

    def __init__(self, name, lines, address, spaces, paid=False, note=""):
        self.name = name
        self.lines = lines
        self.address = address
        self.spaces = spaces  # int, or None when capacity varies/unknown
        self.paid = paid
        self.note = note


# Trolley (light-rail) park-and-ride lots. Rapid bus lots on the MTS page are
# omitted since this agent is trolley-focused.
_LOTS = [
    # --- UC San Diego Blue Line ---
    _Lot("UTC Transit Center", ["UC San Diego Blue Line"], "4545 La Jolla Village Drive", 333, paid=True, note="Pay parking (the only MTS lot that is not free)."),
    _Lot("Nobel Drive", ["UC San Diego Blue Line"], "3449 Nobel Drive", 289, note="Levels 3A-5. Often full by 7:30am on weekdays. Transit riders only."),
    _Lot("Balboa Avenue", ["UC San Diego Blue Line"], "3690 Morena Blvd", 227, note="Transit riders only."),
    _Lot("Tecolote Road", ["UC San Diego Blue Line"], "1364 W Morena Blvd", 279),
    _Lot("Old Town Transit Center", ["UC San Diego Blue Line", "Green Line"], "4009 Taylor St.", 412, note="Transit riders only."),
    _Lot("8th Street", ["UC San Diego Blue Line"], "555 W. 8th Street", 123, note="Transit riders only."),
    _Lot("24th Street", ["UC San Diego Blue Line"], "506 W. 22nd St.", 156, note="Transit riders only."),
    _Lot("E Street", ["UC San Diego Blue Line"], "750 E St.", 267),
    _Lot("H Street", ["UC San Diego Blue Line"], "745 H St.", 295),
    _Lot("Palomar Street", ["UC San Diego Blue Line"], "1265 Industrial Blvd.", 305),
    _Lot("Palm Avenue", ["UC San Diego Blue Line"], "2340 Palm Ave.", 499),
    _Lot("Iris Avenue", ["UC San Diego Blue Line"], "3120 Iris Ave.", 192, note="Transit riders only."),
    _Lot("Beyer Blvd", ["UC San Diego Blue Line"], "4035 Beyer Blvd.", 131, note="Transit riders only."),
    # --- Orange Line ---
    _Lot("47th Street", ["Orange Line"], "350 47th St.", 129, note="Transit riders only."),
    _Lot("Euclid Ave", ["Orange Line"], "450 Euclid Ave.", 115, note="Transit riders only."),
    _Lot("Encanto/62nd St", ["Orange Line"], "6249 Akins Dr.", 158, note="In parking garage."),
    _Lot("Massachusetts Ave", ["Orange Line"], "1787 San Altos Pl.", 241),
    _Lot("Spring Street", ["Orange Line"], "4250 Spring St.", 324, note="Starting mid-June 2026, no transit center parking due to construction."),
    _Lot("Grossmont Transit Center", ["Orange Line", "Green Line"], "8601 Fletcher Pkwy.", 220, note="1st floor garage."),
    _Lot("Amaya Drive", ["Orange Line", "Green Line"], "9100 Amaya Dr.", 236),
    _Lot("El Cajon Transit Center", ["Orange Line", "Green Line", "Copper Line"], "352 S. Marshall Ave.", 469),
    # --- Green Line ---
    _Lot("Morena/Linda Vista", ["Green Line"], "5198 Friars Rd.", 199, note="Transit riders only."),
    _Lot("Fashion Valley Transit Center", ["Green Line"], "1205 Fashion Valley Rd.", 63, note="Transit riders only."),
    _Lot("Union Grantville", ["Green Line"], "4510 Alvarado Canyon Rd.", 100, note="Transit riders only."),
    _Lot("70th Street", ["Green Line"], "7255 Alvarado Rd.", 125, note="Transit riders only."),
    # --- Copper Line ---
    _Lot("Arnele Avenue", ["Copper Line"], "762 1/2 N. Marshall Ave.", 65),
    _Lot("Gillespie Field", ["Copper Line"], "1990 1/2 N. Cuyamaca St.", 175),
    _Lot("Santee Town Center", ["Copper Line"], "Santee Town Center", None, note="Capacity varies."),
]

# Words that add no signal when matching a station name.
_STOPWORDS = {"station", "transit", "center", "trolley", "the", "line"}


def _normalize(name: str) -> str:
    """Lowercase, strip punctuation, and drop boilerplate station words.

    This is what lets a GTFS-style name ("Nobel Drive Station") match a curated
    entry ("Nobel Drive"): both reduce to "nobel drive".
    """
    tokens = re.sub(r"[^a-z0-9]+", " ", name.lower()).split()
    return " ".join(t for t in tokens if t not in _STOPWORDS)


# Normalized-name -> lot, for exact lookup.
_LOTS_BY_NAME = {_normalize(lot.name): lot for lot in _LOTS}


def _find_lot(station: str) -> _Lot | None:
    """Match a station name to a curated lot by exact (normalized) name.

    Matching is intentionally strict: an unrecognized name returns ``None`` (the
    tool then reports "no listed lot") rather than risking a wrong, looser match.
    """
    return _LOTS_BY_NAME.get(_normalize(station))


@tool
def get_station_parking_info(station: str) -> str:
    """Get parking availability and cost for a trolley station.

    Args:
        station: Trolley station name (e.g. "Old Town").

    Returns:
        A summary of whether the station has a park-and-ride lot, its
        capacity if known, and whether parking is free or paid.

    Note:
        This matters for "avoid paid parking" style constraints, so be explicit
        about free vs. paid and say so when the information is unknown.
    """
    lot = _find_lot(station)
    if lot is None:
        return (
            f"No MTS park-and-ride lot is listed for '{station}'. Free station "
            "parking is not available there; only street or paid parking may be "
            "nearby. (Source: MTS Transit Station Parking.)"
        )

    cost = "paid" if lot.paid else "free"
    if lot.spaces is not None:
        capacity = f"{lot.spaces} spaces"
    else:
        capacity = "capacity varies"

    lines = ", ".join(lot.lines)
    summary = (
        f"{lot.name} has a {cost} park-and-ride lot ({capacity}). "
        f"Lines: {lines}. Address: {lot.address}."
    )
    if lot.note:
        summary += f" {lot.note}"
    summary += " Parking is limited to 24 hours; no overnight RV/camper parking."
    return summary
