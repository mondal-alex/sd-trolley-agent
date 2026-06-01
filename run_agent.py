#!/usr/bin/env python3
"""Simple CLI to chat with the SD Trolley Agent.

This is a thin wrapper around the compiled LangGraph graph. Once you've
implemented ``build_graph()`` in ``src/graph.py`` (and uncommented the
module-level ``graph``), this loop will feed user input into the graph and
print the agent's reply.
"""

import sys
from pathlib import Path

# Allow running this file directly (so `import src...` resolves).
sys.path.insert(0, str(Path(__file__).parent))

from langchain_core.messages import HumanMessage


def main() -> None:
    """Run an interactive chat loop against the agent graph."""
    # TODO: import the compiled graph once build_graph() is implemented:
    #   from src.graph import graph
    # (kept local so the CLI imports cleanly before the graph is built).
    from src.graph import build_agent

    graph = build_agent()

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
        result = graph.invoke({"messages": [HumanMessage(content=user_input)]})
        print(f"\nAgent: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
