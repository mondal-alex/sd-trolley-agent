#!/usr/bin/env python3
"""Convenience launcher for the SD Trolley Agent CLI.

Lets you run the agent without installing it as a command:

    uv run python run_agent.py                 # interactive REPL
    uv run python run_agent.py "your question" # one-shot answer

The real logic lives in ``src/cli.py`` (also exposed as the ``sd-trolley``
console script once installed via ``uv tool install .``).
"""

import sys
from pathlib import Path

# Allow running this file directly (so `import src...` resolves).
sys.path.insert(0, str(Path(__file__).parent))

from src.cli import main

if __name__ == "__main__":
    main()
