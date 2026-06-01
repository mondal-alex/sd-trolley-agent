# San Diego Trolley Agent

A Python project for a SD Trolley agent using UV for fast dependency management.

This is intended to be a **ReAct agent** (Reasoning + Acting), which interleaves chain-of-thought reasoning with tool/action calls to iteratively work toward a goal. The agent loop will be implemented using either **LangChain** or **LangGraph**.

## Setup

This project uses [UV](https://github.com/astral-sh/uv) for dependency management.

### Prerequisites

- Python 3.11+
- UV (already installed)

### Installation

1. Install dependencies:
   ```bash
   uv sync
   ```

2. For development dependencies:
   ```bash
   uv sync --extra dev
   ```

### Development

- **Run the project:**
  ```bash
  uv run src/main.py
  ```

- **Format code:**
  ```bash
  uv run black .
  uv run isort .
  ```

- **Lint code:**
  ```bash
  uv run flake8 .
  ```

- **Type checking:**
  ```bash
  uv run mypy .
  ```

- **Run tests:**
  ```bash
  uv run pytest
  ```

- **Start Ollama**
```bash
  brew services start ollama
```

- **List Available Ollama Models**
```bash
ollama list
```

- **Check if Ollama Service is Running**
```bash
ollama ps
```

- **Update a model**
```bash
ollama pull llama3
```

- **Remove a model to save space**
```bash
ollama rm mistral
```

- **Show model information**
```bash
ollama show llama3
```

- **Stop the Ollama service**
```bash
brew services stop ollama
```

- **Check Ollama version**
```bash
ollama --version
```
### Project Structure

```
sd-trolley-agent/
├── src/                     # Source code package
│   ├── __init__.py
│   ├── state.py             # LangGraph state schema (messages + reducer)
│   ├── llm.py               # Chat model setup (tool-calling capable)
│   ├── prompts.py           # System prompt
│   ├── graph.py             # The ReAct agent graph (build_graph)
│   └── tools/               # Tools the agent can call
│       ├── __init__.py      # ALL_TOOLS registry
│       ├── routing.py       # Google Maps drive/walk times
│       └── trolley.py       # MTS trolley stations/schedule/arrivals
├── tests/                   # Test scaffolding
├── run_agent.py             # Interactive CLI entry point
├── pyproject.toml           # Project configuration and dependencies
├── README.md               # This file
└── .gitignore              # Git ignore rules
```

The agent loop lives in `src/graph.py`. The intended flow is a ReAct loop:

```
START -> agent -> (tool calls?) -> tools -> agent -> ... -> END
```

Implement the `NotImplementedError` stubs (tools, `get_llm`, `agent_node`,
`build_graph`) to bring the agent to life.

## Features

- Fast dependency resolution with UV
- Code formatting with Black
- Import sorting with isort
- Linting with flake8
- Type checking with mypy
- Testing with pytest
