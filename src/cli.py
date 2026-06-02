"""Command-line interface for the SD Trolley Agent.

Two ways to use it:

- Interactive REPL (no arguments)::

      sd-trolley
      uv run python run_agent.py

- One-shot question (arguments are joined into a single question, answered
  once, then the process exits) -- handy for scripting and quick lookups::

      sd-trolley "leave time to reach Petco Park by 6pm from La Jolla"
      uv run python run_agent.py "what trolley stations are near UTC?"

Conversation memory comes from the graph's checkpointer, scoped to one
``thread_id`` per process, so follow-up questions in a REPL session retain
context.
"""

import itertools
import sys
import threading
import time
import uuid

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

# Cap how many reason/act steps a single turn may take before we bail out, so a
# model that loops on tool calls fails fast with a friendly message.
RECURSION_LIMIT = 25

_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class _Spinner:
    """A tiny threaded terminal spinner shown while the agent is working.

    Writes to a stream (stderr by default, so it never mixes into the piped
    stdout answer) and only animates when that stream is an interactive TTY,
    keeping one-shot/redirected output clean.
    """

    def __init__(self, message: str = "Thinking", stream=sys.stderr) -> None:
        self._message = message
        self._stream = stream
        self._enabled = hasattr(stream, "isatty") and stream.isatty()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_Spinner":
        if self._enabled:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def _spin(self) -> None:
        for frame in itertools.cycle(_SPINNER_FRAMES):
            if self._stop.is_set():
                break
            self._stream.write(f"\r{frame} {self._message}… ")
            self._stream.flush()
            time.sleep(0.08)

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self._enabled:
            # Wipe the spinner line so the answer prints on a clean row.
            self._stream.write("\r" + " " * (len(self._message) + 6) + "\r")
            self._stream.flush()


def _ask(graph, config, question: str) -> None:
    """Run one question through the graph and print the agent's reply."""
    try:
        with _Spinner():
            result = graph.invoke(
                {"messages": [HumanMessage(content=question)]}, config
            )
    except GraphRecursionError:
        print(
            "\nAgent: I got stuck taking too many steps on that one. Try "
            "rephrasing it or breaking it into smaller questions.\n"
        )
        return
    except Exception as e:
        print(f"\nAgent: Sorry, something went wrong: {e}\n")
        return

    # The graph appends to `messages`; the last message is the agent reply.
    print(f"\nAgent: {result['messages'][-1].content}\n")


def main() -> None:
    """Entry point: one-shot if a question is passed on argv, else a REPL."""
    from src.graph import build_agent

    graph = build_agent()
    # One thread per process -> one continuous conversation for this session.
    config = {
        "recursion_limit": RECURSION_LIMIT,
        "configurable": {"thread_id": uuid.uuid4().hex},
    }

    # One-shot mode: everything after the program name is the question.
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip()
        if question:
            _ask(graph, config, question)
        return

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

        _ask(graph, config, user_input)


if __name__ == "__main__":
    main()
