"""The LangGraph agent graph (agent orchestration).

It wires up a classic ReAct (reason -> act -> observe) loop:

    START -> agent -> (tools? ) -> tools -> agent -> ... -> END

How the loop works:
1. The ``agent`` node calls the LLM. The LLM either answers directly or emits
   one or more tool calls.
2. A conditional edge inspects the last message:
     - if it contains tool calls  -> go to the ``tools`` node
     - otherwise (a final answer)  -> go to END
3. The ``tools`` node executes the requested tool(s) and appends the results
   to ``messages``.
4. An edge routes from ``tools`` back to ``agent`` so the model can read the
   tool output and continue. This back-edge is what makes it a *loop*.

Useful imports:
    from langgraph.graph import StateGraph, START, END
    from langgraph.prebuilt import ToolNode, tools_condition
    from langchain_core.messages import SystemMessage

Tip: ``langgraph.prebuilt`` gives you ``ToolNode`` (runs the tools) and
``tools_condition`` (the "are there tool calls?" router) so you don't have to
write them by hand. There's also ``create_react_agent`` for a one-liner version
once you understand the manual wiring below.

Docs / references:
- LangGraph quickstart: https://langchain-ai.github.io/langgraph/tutorials/introduction/
- StateGraph API: https://langchain-ai.github.io/langgraph/reference/graphs/
- Prebuilt ToolNode / tools_condition: https://langchain-ai.github.io/langgraph/reference/prebuilt/
- Prebuilt create_react_agent (one-liner): https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent
- State & reducers (add_messages): https://langchain-ai.github.io/langgraph/concepts/low_level/#state
- Persistence / checkpointers (memory): https://langchain-ai.github.io/langgraph/concepts/persistence/
- Tool calling in LangChain: https://python.langchain.com/docs/concepts/tool_calling/
- ChatOllama: https://python.langchain.com/docs/integrations/chat/ollama/
"""

from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition

from .llm import get_llm
from .prompts import SYSTEM_PROMPT
from .state import AgentState
from .tools import ALL_TOOLS


def llm_node(state: AgentState) -> dict:
    """The reasoning node: call the LLM and return its message.

    Steps to implement:
        1. Build the model with tools bound:  ``get_llm().bind_tools(ALL_TOOLS)``.
        2. Prepend the system prompt to ``state["messages"]`` (a SystemMessage).
        3. Invoke the model on those messages.
        4. Return ``{"messages": [response]}`` so the reducer appends it.
    """

    # Bind the tools to the LLM model.
    llm_model_with_tools = get_llm().bind_tools(ALL_TOOLS)

    # The system mesage.
    system_message = SystemMessage(
        content=SYSTEM_PROMPT
    )

    # Get the input messages from the AgentState.
    input_messages = state['messages']

    # Invoke the LLM model on the input messages.
    response = llm_model_with_tools.invoke([system_message] + input_messages) 

    return {"messages": [response]}


def build_agent() -> CompiledStateGraph:
    """Construct and compile the agent graph.

    Suggested steps:
        1. ``graph = StateGraph(AgentState)``
        2. Add the reasoning node:   ``graph.add_node("agent", agent_node)``
        3. Add the tool node:        ``graph.add_node("tools", ToolNode(ALL_TOOLS))``
        4. Set the entry point:      ``graph.add_edge(START, "agent")``
        5. Conditional routing from "agent" using ``tools_condition`` so tool
           calls go to "tools" and a final answer goes to END.
        6. Loop back:                ``graph.add_edge("tools", "agent")``
        7. ``return graph.compile()``  (optionally pass a checkpointer for memory)
    """

    # Add nodes
    agent_builder = StateGraph(AgentState)
    agent_builder.add_node("llm_node", llm_node)
    agent_builder.add_node("tool_node", ToolNode(ALL_TOOLS))

    # Add edges
    agent_builder.add_edge(START, "llm_node")
    # tools_condition returns "tools" (tool calls present) or "__end__".
    # Map those literals to our actual node names.
    agent_builder.add_conditional_edges(
        "llm_node",
        tools_condition,
        {"tools": "tool_node", "__end__": END},
    )
    agent_builder.add_edge("tool_node", "llm_node")

    agent = agent_builder.compile()
    return agent



# A module-level compiled graph is handy for the CLI and `langgraph dev`.
graph = build_agent()
