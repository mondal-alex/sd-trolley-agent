"""The SD Trolley Agent."""

import os
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model

os.environ["OPENAI_API_KEY"] = "sk-..."

llm = init_chat_model("openai:gpt-4.1")

class SDTrolleyState(TypedDict):
    """State for the SD Trolley Agent."""
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(SDTrolleyState)

# TODO: use locally-running LLM agent instead.
# TODO: How to add system and user prompts?
llm = init_chat_model("openai:gpt-4.1")

def chat_node(state: SDTrolleyState) -> SDTrolleyState:
    return {"messages": llm.invoke(state["messages"])}

