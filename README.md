# SD Trolley Agent

A Python project for a SD Trolley agent using UV for fast dependency management.

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

### Project Structure

```
sd-trolley-agent/
├── src/                     # Source code package
│   ├── __init__.py
│   └── main.py
├── pyproject.toml           # Project configuration and dependencies
├── README.md               # This file
└── .gitignore              # Git ignore rules
```

## Features

- Fast dependency resolution with UV
- Code formatting with Black
- Import sorting with isort
- Linting with flake8
- Type checking with mypy
- Testing with pytest
