"""Unit tests for the agent's tools.

Tools call external services (Google Maps, MTS feeds), so in tests you'll want
to mock those network calls and assert on the formatted string each tool
returns. Tools are decorated with ``@tool``; invoke them with ``.invoke({...})``
or call the underlying function via ``.func(...)``.
"""

import pytest

from src.tools import ALL_TOOLS


def test_all_tools_exposed():
    """Every tool should be registered for the agent to bind."""
    assert len(ALL_TOOLS) == 5
    names = {t.name for t in ALL_TOOLS}
    assert "get_driving_time" in names


# TODO: add tests for each tool once implemented, e.g.:
#
# from unittest.mock import patch
#
# @patch("src.tools.routing.<google_maps_client>")
# def test_get_driving_time(mock_client):
#     mock_client.directions.return_value = [...]
#     result = get_driving_time.invoke({"origin": "A", "destination": "B"})
#     assert "Driving" in result
@pytest.mark.skip(reason="implement once routing tools are written")
def test_get_driving_time():
    raise NotImplementedError
