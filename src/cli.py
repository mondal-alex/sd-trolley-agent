"""Command-line interface for the SD Trolley Agent.

Run once and chat in the same session (memory lasts until you quit):

    sd-trolley
    sd-trolley "When should I leave to reach Petco Park by 6pm this Friday?"

The optional first argument is your opening question; you can answer follow-ups
at the ``User:`` prompt without re-running the command.

For scripting (non-interactive terminal), pass a question and stdout is not a TTY
— the process answers once and exits.
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

    def set_message(self, message: str) -> None:
        """Update the label while the spinner is running."""
        self._message = message

    def __enter__(self) -> "_Spinner":
        if self._enabled:
            # Draw immediately so the user sees feedback before slow imports/LLM.
            self._stream.write(f"\r{_SPINNER_FRAMES[0]} {self._message}… ")
            self._stream.flush()
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


def _invoke(graph, config, question: str):
    """Run the graph for one user turn (no UI)."""
    return graph.invoke(
        {"messages": [HumanMessage(content=question)]}, config
    )


def _ask(
    graph,
    config,
    question: str,
    *,
    spinner: _Spinner | None = None,
) -> None:
    """Run one question through the graph and print the agent's reply."""
    try:
        if spinner is not None:
            spinner.set_message("Thinking")
            result = _invoke(graph, config, question)
        else:
            with _Spinner():
                result = _invoke(graph, config, question)
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


def _repl(graph, config) -> None:
    """Interactive loop: one process, one conversation."""
    print("SD Trolley Agent — type 'quit' to exit.\n")
    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        _ask(graph, config, user_input)


def main() -> None:
    """One session: optional opening question, then REPL when stdin is a TTY."""
    opening = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else ""
    interactive = sys.stdin.isatty()

    # One thread per process — all turns in this run share memory.
    config = {
        "recursion_limit": RECURSION_LIMIT,
        "configurable": {"thread_id": uuid.uuid4().hex},
    }

    # Imports and graph compile can take tens of seconds on a cold start; show the
    # spinner immediately and keep it through the first LLM turn.
    with _Spinner("Starting") as busy:
        from src.graph import build_agent

        graph = build_agent()
        if opening:
            busy.set_message("Thinking")
            _ask(graph, config, opening, spinner=busy)
    if opening and not interactive:
        return

    if interactive:
        _repl(graph, config)
    elif not opening:
        print(
            "Usage: sd-trolley [question]\n"
            "  sd-trolley              — interactive trip planner\n"
            "  sd-trolley \"...\"        — ask a question, then keep chatting\n",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
