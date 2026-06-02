# SD Trolley Agent Setup Guide

## Prerequisites

1. **Ollama (local LLM)**
   - Install [Ollama](https://ollama.com/) and start it (`brew services start ollama`).
   - Pull a **tool-calling-capable** model and set it as `OLLAMA_MODEL`:
     ```bash
     ollama pull qwen2.5:7b      # current default in .env
     # or a lighter/faster option on CPU-only machines:
     ollama pull llama3.2:3b
     ```
   - The agent validates the model on startup, so it must be pulled first.

2. **Google Maps API Key**
   - Go to [Google Cloud Console](https://console.cloud.google.com/).
   - Create a project (or select an existing one) and enable billing.
   - Enable these APIs:
     - **Routes API** (driving/walking times)
     - **Geocoding API** (address ↔ lat/lng)
     - **Geolocation API** (current location)
   - Create an API Key.

## Installation

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Configure environment:** copy the template and fill in your values
   (`.env` is gitignored):
   ```bash
   cp .env.example .env
   # then edit .env:
   #   OLLAMA_MODEL=qwen2.5:7b
   #   OLLAMA_BASE_URL=http://localhost:11434
   #   GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
   ```

## Running the agent

```bash
# Interactive REPL
uv run python run_agent.py

# One-shot question (good for scripting / quick lookups)
uv run python run_agent.py "what trolley stations are near UTC?"

# Install once, then run from anywhere as `sd-trolley`
uv tool install .
sd-trolley "I need to get from La Jolla to Downtown by 6 PM. When should I leave?"
```

> The first schedule/station question downloads the MTS static GTFS feed into
> `.gtfs_cache/` (no API key needed; requires network). Subsequent runs reuse
> the cache.

## Testing

1. **Install test dependencies:**
   ```bash
   uv sync --extra dev
   ```

2. **Run the unit tests** (fully mocked, no network/LLM required):
   ```bash
   uv run pytest
   ```

3. **Run with coverage:**
   ```bash
   uv run pytest --cov=src
   ```

4. **End-to-end smoke test** (needs Ollama + Google key + network):
   ```bash
   uv run python run_agent.py "what time is it in San Diego?"
   ```
   > Note: on CPU-only machines a 7B model can take minutes per answer. Use a
   > smaller model (e.g. `llama3.2:3b`) for faster iteration.

## Example Queries

```
"I live in La Jolla and want to get to Gaslamp Quarter using the trolley. What trolley stations are near me?"

"I want to take the trolley from UTC to Old Town. How long would it take to walk from UTC Transit Center to the trolley platform?"

"I need to get from La Jolla to Downtown San Diego by 6 PM using the trolley. When should I leave my house?"

"Find trolley stations near Gaslamp Quarter and tell me how long it takes to walk from there to the Convention Center."
```

## Capabilities

- ✅ Driving / walking times between locations (Routes API)
- ✅ Find nearby trolley stations from any location (Geocoding + GTFS)
- ✅ Trolley schedules (static GTFS)
- ✅ Station park-and-ride info (free vs paid)
- ✅ Current location and current San Diego time
- ❌ Real-time arrivals — MTS publishes no public GTFS-Realtime feed, so times
  are **scheduled**, not live.
