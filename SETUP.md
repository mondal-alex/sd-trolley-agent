# SD Trolley Agent Setup Guide

## Prerequisites

1. **Google Maps API Key**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing one
   - Enable the following APIs:
     - Maps JavaScript API
     - Directions API
     - Distance Matrix API
     - Places API
   - Create credentials (API Key)
   - Set up billing (Google Maps API requires billing to be enabled)

2. **OpenAI API Key**
   - Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)

## Installation

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Set environment variables:**
   ```bash
   export GOOGLE_MAPS_API_KEY="your_google_maps_api_key_here"
   export OPENAI_API_KEY="your_openai_api_key_here"
   ```

   Or create a `.env` file:
   ```
   GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Testing

1. **Install test dependencies:**
   ```bash
   uv sync --extra dev
   ```

2. **Run unit and integration tests:**
   ```bash
   pytest
   ```

3. **Run tests with coverage:**
   ```bash
   pytest --cov=src
   ```

4. **Test Google Maps integration directly:**
   ```bash
   python test_google_maps.py
   ```

5. **Run example agent:**
   ```bash
   python example_usage.py
   ```

## Usage

The agent can now:

- ✅ **Get driving times** between any two locations
- ✅ **Get walking times** between any two locations  
- ✅ **Find nearby trolley stations** from any location
- ✅ **Calculate distances and durations** for travel planning

### Example Queries

```
"I live in La Jolla and want to get to Gaslamp Quarter using the trolley. What trolley stations are near me?"

"I want to take the trolley from UTC to Old Town. How long would it take to walk from UTC Transit Center to the trolley platform?"

"I need to get from La Jolla to Downtown San Diego by 6 PM using the trolley. When should I leave my house?"

"Find trolley stations near Gaslamp Quarter and tell me how long it takes to walk from there to the Convention Center"
```

## Next Steps

- [ ] Implement trolley schedule integration
- [ ] Add real-time transit data
- [ ] Create complete trip planning workflow
- [ ] Add departure time recommendations
