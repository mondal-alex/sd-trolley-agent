#!/usr/bin/env python3
"""Simple CLI to chat with the SD Trolley Agent.

A thin wrapper around the compiled LangGraph agent: each turn feeds the user's
input into the graph (scoped to one conversation via a ``thread_id``) and prints
the reply. Conversation memory comes from the graph's checkpointer, so follow-up
questions in the same session retain context.
"""

import sys
import uuid
from pathlib import Path

# Allow running this file directly (so `import src...` resolves).
sys.path.insert(0, str(Path(__file__).parent))

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

# Cap how many reason/act steps a single turn may take before we bail out, so a
# model that loops on tool calls fails fast with a friendly message.
RECURSION_LIMIT = 25


def main() -> None:
    """Run an interactive chat loop against the agent graph."""
    from src.graph import build_agent

    graph = build_agent()
    # One thread per CLI run -> one continuous conversation for this session.
    config = {
        "recursion_limit": RECURSION_LIMIT,
        "configurable": {"thread_id": uuid.uuid4().hex},
    }

    print("SD Trolley Agent — type 'quit' to exit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        # The graph appends to `messages`; the last message is the agent reply.
        try:
            result = graph.invoke(
                {"messages": [HumanMessage(content=user_input)]}, config
            )
        except GraphRecursionError:
            print(
                "\nAgent: I got stuck taking too many steps on that one. Try "
                "rephrasing it or breaking it into smaller questions.\n"
            )
            continue
        except Exception as e:
            print(f"\nAgent: Sorry, something went wrong: {e}\n")
            continue

        print(f"\nAgent: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
