"""
SD Trolley Agent package.

Loads environment variables from a local `.env` file on import so that
`os.getenv(...)` calls in `llm.py` and the tools pick up your Ollama and Google
Maps configuration. See `.env.example` for the expected variables.
"""

from dotenv import load_dotenv

# `load_dotenv` is a no-op if `.env` is absent and never overrides variables
# already set in the real environment.
load_dotenv()

__version__ = "0.1.0"
