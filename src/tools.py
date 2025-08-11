"""Tools for the SD Trolley Agent."""

from langchain_core.tools import tool

@tool
def get_trolley_schedule(date: str) -> str:
    """Get the trolley schedule for a given date."""
    return ...

@tool
def get_current_location() -> str:
    """Get the user's current location."""
    return ...

@tool
def get_current_time() -> str:
    """Get the current time."""
    return ...

@tool
def get_walking_time(destination: str) -> str:
    """Get the walking time to a given destination."""
    return ...

@tool
def get_driving_time(destination: str) -> str:
    """Get the driving time to a given destination."""
    return ...