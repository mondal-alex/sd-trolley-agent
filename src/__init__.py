"""
SD Trolley Agent package.

Loads environment variables on import so `os.getenv(...)` calls in `llm.py` and
the tools pick up your Ollama and Google Maps configuration. See `.env.example`
for the expected variables.

Config is resolved from several places, in priority order (earlier wins;
nothing already set in the real environment is ever overridden):

1. Variables already exported in your shell environment.
2. A `.env` in the current working directory (handy when running from source).
3. A user-level `~/.config/sd-trolley/.env` (so the installed `sd-trolley`
   command works from any directory).

This makes the tool usable both from the project checkout and as a globally
installed command.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# 2. A `.env` in the current working directory. `find_dotenv` defaults to
#    searching from the *caller's* file location, which breaks for an installed
#    package, so search explicitly from the CWD instead.
_cwd_env = Path.cwd() / ".env"
if _cwd_env.is_file():
    load_dotenv(_cwd_env)

# 3. User-level config fallback (does not override anything set above).
_user_env = Path(
    os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")
) / "sd-trolley" / ".env"
if _user_env.is_file():
    load_dotenv(_user_env)

__version__ = "0.3.8"
