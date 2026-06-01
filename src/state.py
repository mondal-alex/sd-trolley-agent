"""Graph state for the SD Trolley Agent.

In LangGraph, every node receives the current state and returns a partial
update that gets merged in. The state schema below is the single source of
truth for what flows through the graph.

The key idea is the ``add_messages`` reducer: instead of *replacing* the
``messages`` list on every update, LangGraph *appends* to it. That's what lets
the conversation (human message -> AI tool call -> tool result -> AI answer)
accumulate over multiple steps of the ReAct loop.
"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State passed between nodes in the agent graph.

    Attributes:
        messages: The running conversation. The ``add_messages`` reducer means
            node updates are appended rather than overwriting the list.
    """

    messages: Annotated[list, add_messages]

    # TODO (optional): add more fields as your agent grows, e.g.
    #   user_location: str
    #   target_arrival_time: str
    # Any plain field replaces on update; wrap with Annotated[..., reducer] to
    # customize merge behavior like `messages` does.
