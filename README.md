# San Diego Trolley Agent

A Python project for a SD Trolley agent using UV for fast dependency management.

This is intended to be a **ReAct agent** (Reasoning + Acting), which interleaves chain-of-thought reasoning with tool/action calls to iteratively work toward a goal. The agent loop will be implemented using either **LangChain** or **LangGraph**.

## Setup

This project uses [UV](https://github.com/astral-sh/uv) for dependency management.

### Prerequisites

- Python 3.11+
- UV (already installed)
- [Ollama](https://ollama.com/) running locally with a tool-calling model pulled
- A Google Maps Platform API key (see below)

### Google Maps APIs

The agent's tools call several Google Maps Platform APIs. Enable these in your
[Google Cloud project](https://console.cloud.google.com/) and put the key in
`.env` as `GOOGLE_MAPS_API_KEY` (see `.env.example`). Billing must be enabled.

| API | Powers | Client library | Used for |
|-----|--------|----------------|----------|
| Routes API | `get_driving_time`, `get_walking_time` | `google-maps-routing` | Route duration/distance (traffic-aware driving) |
| Geocoding API | `find_nearby_trolley_stations` | `googlemaps` | Address ↔ lat/lng for station search |

> This project uses Google's **current** APIs. Routes API replaces the legacy
> Directions + Distance Matrix APIs and is recommended for new projects (it may
> be the only version enableable on a freshly created Cloud project).

> Geocoding has no "new" successor; it is accessed with the `googlemaps` client.

> `find_nearby_trolley_stations` does **not** use Google Places — it geocodes
> the input location, then finds the nearest trolley stations directly from the
> GTFS feed (see below). This keeps station names consistent with
> `get_trolley_schedule`.

> Gotcha: the Routes API requires a **field mask** on each request (listing the
> response fields you want) or the response comes back empty.

> Minimal start: only **Routes API** is needed to get the routing tools
> working. Add **Geocoding** for `find_nearby_trolley_stations`.

### Trolley data (GTFS)

The trolley timetable (`get_trolley_schedule`) and nearby-station search
(`find_nearby_trolley_stations`) both come from the San Diego MTS **static
GTFS** feed, downloaded automatically and cached locally under `.gtfs_cache/`
(refreshed weekly). No API key is needed for the feed itself. Parsing is done
with [`gtfs-kit`](https://pypi.org/project/gtfs-kit/). Sharing one data source
keeps station names consistent across both tools.

### Station parking

MTS does not publish structured park-and-ride data, so `get_station_parking_info`
uses a small curated table in `src/tools/parking.py`, transcribed from the
official [MTS Transit Station Parking](https://www.sdmts.com/transit-services/transit-station-parking)
page. Every listed lot is free except **UTC Transit Center** (pay parking);
stations not on the list have no free MTS lot, which the tool states explicitly
so the agent can honor "avoid paid parking" constraints.

> No real-time / live arrivals: MTS does not publish a public GTFS-Realtime
> feed — their developer page states "We hope to share our real time
> information in the future"
> ([sdmts.com/business-center/app-developers](https://www.sdmts.com/business-center/app-developers)).
> Because of this, the live-arrivals tool was intentionally removed and trolley
> times are reported as **scheduled**, not live.

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

- **Run the agent (one session, smooth follow-ups):**
  ```bash
  uv run python run_agent.py
  ```
  Or open with your first question — you stay in the same chat until you type
  `quit` (no need to re-run the command for follow-ups):
  ```bash
  uv run python run_agent.py "When should I leave for Petco Park by 6pm? I'm at 4109 Park Pl, San Diego 92116."
  # Agent replies, then:
  User: quit
  ```

- **Install as a global command** (then run `sd-trolley` from anywhere):
  ```bash
  uv tool install .
  sd-trolley
  sd-trolley "When should I leave for Petco Park by 6pm? I can drive to a station."
  ```
  > To use it from any directory, put your config where the installed command
  > can always find it (it needs Ollama running locally):
  > ```bash
  > mkdir -p ~/.config/sd-trolley
  > cp .env ~/.config/sd-trolley/.env
  > ```
  > Config is resolved in this order (first wins, real env vars are never
  > overridden): exported environment variables → `.env` in the current
  > directory → `~/.config/sd-trolley/.env`.

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
│   ├── graph.py             # The ReAct agent graph (build_agent)
│   ├── cli.py               # CLI logic (REPL + one-shot); sd-trolley entry
│   └── tools/               # Tools the agent can call
│       ├── __init__.py      # ALL_TOOLS registry
│       ├── _clients.py      # Shared, cached Google API clients
│       ├── routing.py       # Routes API drive/walk times
│       ├── trolley.py       # Nearby stations + schedule (GTFS)
│       ├── parking.py       # Curated MTS park-and-ride data (free/paid)
│       ├── gtfs.py          # MTS static GTFS feed download/cache
│       └── clock.py         # Current time in San Diego
├── tests/                   # Test scaffolding
├── run_agent.py             # Convenience launcher (REPL + one-shot)
├── pyproject.toml           # Project configuration and dependencies
├── README.md               # This file
└── .gitignore              # Git ignore rules
```

The agent loop lives in `src/graph.py`. The flow is a ReAct loop:

```
START -> agent -> (tool calls?) -> tools -> agent -> ... -> END
```

The LLM node (`llm_node`), tool registry (`ALL_TOOLS`), and `build_agent()`
are all implemented; `get_llm()` returns a tool-calling `ChatOllama` configured
from the `OLLAMA_*` env vars.

## Features

- Fast dependency resolution with UV
- Code formatting with Black
- Import sorting with isort
- Linting with flake8
- Type checking with mypy
- Testing with pytest
