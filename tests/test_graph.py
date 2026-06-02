"""Tests for the LangGraph agent graph.

The graph calls an LLM, so we patch ``src.graph.get_llm`` with a fake chat model
that returns a scripted sequence of messages. This keeps tests fast/offline and
lets us assert on how the graph *routes*:
- a model response with no tool calls should end the run,
- a model response with tool calls should route through the tool node and loop
  back to the llm node.
"""

from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src import graph as graph_module
from src.graph import build_agent, llm_node
from src.tools import ALL_TOOLS


class FakeChatModel:
    """A stand-in chat model that returns pre-scripted responses.

    ``bind_tools`` returns self (so the node can chain it), and each ``invoke``
    pops the next scripted message. ``bound_tools`` / ``seen_messages`` are
    recorded for assertions.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.bound_tools = None
        self.seen_messages = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.seen_messages.append(messages)
        return self._responses.pop(0)


def test_build_agent_compiles_with_expected_nodes():
    """build_agent() returns a compiled graph wired with both nodes."""
    compiled = build_agent()
    assert compiled is not None
    nodes = set(compiled.get_graph().nodes)
    assert {"llm_node", "tool_node"} <= nodes


def test_llm_node_prepends_system_prompt_and_binds_tools():
    """llm_node should bind tools and prepend the system prompt before invoking."""
    fake = FakeChatModel([AIMessage(content="hello")])

    with patch("src.graph.get_llm", return_value=fake):
        state = {"messages": [HumanMessage(content="hi there")]}
        result = llm_node(state)

    # Tools were bound for tool-calling.
    assert fake.bound_tools == ALL_TOOLS
    # The system prompt is prepended ahead of the conversation.
    sent = fake.seen_messages[0]
    assert isinstance(sent[0], SystemMessage)
    assert isinstance(sent[1], HumanMessage)
    assert sent[1].content == "hi there"
    # The response is returned under "messages" for the reducer to append.
    assert result["messages"][0].content == "hello"


def test_graph_ends_when_no_tool_calls():
    """A plain answer (no tool calls) should route straight to END."""
    fake = FakeChatModel([AIMessage(content="final answer")])

    with patch("src.graph.get_llm", return_value=fake):
        # The graph is compiled with a checkpointer, so invoking it requires a
        # thread_id to scope the persisted conversation state.
        result = build_agent().invoke(
            {"messages": [HumanMessage(content="hi")]},
            {"configurable": {"thread_id": "test"}},
        )

    assert result["messages"][-1].content == "final answer"


def test_graph_routes_through_tool_node_then_back():
    """A response with tool calls should run the tool, then loop back to the llm."""
    tool_name = ALL_TOOLS[0].name  # get_driving_time
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": tool_name,
                    "args": {"origin": "A", "destination": "B"},
                    "id": "call-1",
                }
            ],
        ),
        AIMessage(content="here is your plan"),
    ]
    fake = FakeChatModel(responses)

    with patch("src.graph.get_llm", return_value=fake):
        result = build_agent().invoke(
            {"messages": [HumanMessage(content="how long to drive A to B?")]},
            {"configurable": {"thread_id": "test"}},
        )

    # The tool node executed (ToolNode captures the NotImplementedError into a
    # ToolMessage by default), proving we routed through "tool_node".
    assert any(isinstance(m, ToolMessage) for m in result["messages"])
    # After looping back, the second llm response ends the run.
    assert result["messages"][-1].content == "here is your plan"


def test_memory_persists_across_turns_on_same_thread():
    """Reusing a thread_id should let the second turn see the first exchange."""
    fake = FakeChatModel([AIMessage(content="first reply"), AIMessage(content="second reply")])
    config = {"configurable": {"thread_id": "mem-same-thread"}}

    with patch("src.graph.get_llm", return_value=fake):
        agent = build_agent()
        agent.invoke({"messages": [HumanMessage(content="my name is Alex")]}, config)
        agent.invoke({"messages": [HumanMessage(content="what is my name?")]}, config)

    # On the second turn the llm node should have been handed the full history:
    # the first human + AI messages, plus the new human message.
    second_turn_input = [m.content for m in fake.seen_messages[-1]]
    assert "my name is Alex" in second_turn_input
    assert "first reply" in second_turn_input
    assert "what is my name?" in second_turn_input


def test_memory_is_isolated_across_threads():
    """A different thread_id should not see another thread's history."""
    fake = FakeChatModel([AIMessage(content="reply A"), AIMessage(content="reply B")])

    with patch("src.graph.get_llm", return_value=fake):
        agent = build_agent()
        agent.invoke(
            {"messages": [HumanMessage(content="thread one message")]},
            {"configurable": {"thread_id": "mem-thread-A"}},
        )
        agent.invoke(
            {"messages": [HumanMessage(content="thread two message")]},
            {"configurable": {"thread_id": "mem-thread-B"}},
        )

    second_turn_input = [m.content for m in fake.seen_messages[-1]]
    assert "thread two message" in second_turn_input
    # The other thread's conversation must not leak in.
    assert "thread one message" not in second_turn_input
    assert "reply A" not in second_turn_input
