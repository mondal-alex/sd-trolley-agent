"""LLM setup for the agent.

The graph needs a chat model that supports *tool calling* so the model can emit
structured requests to run your tools. With a local Ollama model use a
tool-calling-capable model (e.g. llama3.1) via ``ChatOllama``; you can swap in a
hosted model later without changing the graph.
"""

import os

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama


def get_llm() -> BaseChatModel:
    """Build and return the chat model the agent should use.

    Returns:
        A chat model instance. Remember to call ``.bind_tools(ALL_TOOLS)`` on it
        (the graph does this) so the model can request tool calls.

    Implementation hints:
        - Local:  ``from langchain_ollama import ChatOllama`` then
          ``ChatOllama(model="llama3.1", base_url=...)``.
        - Hosted: ``from langchain.chat_models import init_chat_model`` then
          ``init_chat_model("openai:gpt-4o")``.
        - Pull config (model name, base url) from env vars for flexibility.
    """
    _model = os.getenv("OLLAMA_MODEL", "llama3.1")
    _base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    _temperature = 0.2 # Keep low since this agent calls tools and must be reliable.
    
    llm = ChatOllama(
                model=_model,
                base_url=_base_url,
                validate_model_on_init=True,
                temperature=_temperature)

    return llm