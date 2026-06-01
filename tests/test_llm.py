"""Unit tests for src.llm.get_llm.

`get_llm` constructs a ChatOllama, which would normally try to reach a running
Ollama server (because of validate_model_on_init=True). We patch ChatOllama so
these tests stay fast and offline, and assert that config is wired from env
vars correctly.
"""

from unittest.mock import patch

from src.llm import get_llm


def test_get_llm_reads_env_and_builds_model(monkeypatch):
    """get_llm should pass env-derived config to ChatOllama and return it."""
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example:1234")

    with patch("src.llm.ChatOllama") as mock_chat_ollama:
        sentinel = object()
        mock_chat_ollama.return_value = sentinel

        result = get_llm()

        assert result is sentinel
        mock_chat_ollama.assert_called_once()
        kwargs = mock_chat_ollama.call_args.kwargs
        assert kwargs["model"] == "test-model"
        assert kwargs["base_url"] == "http://example:1234"


def test_get_llm_defaults_when_env_missing(monkeypatch):
    """Falls back to sensible defaults when env vars are unset."""
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    with patch("src.llm.ChatOllama") as mock_chat_ollama:
        get_llm()

        kwargs = mock_chat_ollama.call_args.kwargs
        assert kwargs["model"] == "llama3.1"
        assert kwargs["base_url"] == "http://localhost:11434"
