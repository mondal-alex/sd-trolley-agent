"""Time awareness tool.

The agent reasons about "leave by" / "arrive by" times, so it needs to know
the current date and time. The LLM does *not* reliably know "now", so expose it
as a tool the model can call.
"""

from zoneinfo import ZoneInfo
from langchain_core.tools import tool
from datetime import datetime


@tool
def get_current_time() -> str:
    """Get the current date and time in San Diego (America/Los_Angeles).

    Returns:
        The current local date, time, and day of week, e.g.
        "Monday 2026-06-01 14:20 PDT".
    """
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%A %Y-%m-%d %H:%M %Z")
