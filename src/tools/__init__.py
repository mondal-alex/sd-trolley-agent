"""Tools available to the SD Trolley Agent.

The agent's LLM decides *when* to call these tools; LangGraph's tool node is
what actually *executes* them. Each tool is a plain function decorated with
``@tool`` so LangChain can expose its name, args, and docstring to the model.

Keep the docstrings descriptive: the model reads them to decide which tool to
use and how to fill in the arguments.
"""

from .routing import get_driving_time, get_walking_time
from .trolley import (
    find_nearby_trolley_stations,
    get_trolley_schedule,
    get_next_trolley_arrivals,
)

# The agent binds this list to the LLM and hands it to the tool node.
ALL_TOOLS = [
    get_driving_time,
    get_walking_time,
    find_nearby_trolley_stations,
    get_trolley_schedule,
    get_next_trolley_arrivals,
]

__all__ = [
    "ALL_TOOLS",
    "get_driving_time",
    "get_walking_time",
    "find_nearby_trolley_stations",
    "get_trolley_schedule",
    "get_next_trolley_arrivals",
]
