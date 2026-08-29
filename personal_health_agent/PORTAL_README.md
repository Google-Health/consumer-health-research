# PHA Portal - Developer Guide

Detailed documentation for the PHA web portal. For quick start instructions, see the main [README.md](README.md).

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│   API Server    │
│   (React)       │◀────│   (FastAPI)     │
│   Port 3000     │     │   Port 8000     │
└─────────────────┘     └─────────────────┘
                               │
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
              ┌─────────┐┌─────────┐┌─────────┐
              │  PHA  ││ Parallel││  PHIA   │
              │Orchestr.││ Baseline││ ReAct   │
              └─────────┘└─────────┘└─────────┘
```

## Project Structure

```
api/
  main.py              # FastAPI application, session management
frontend/
  src/
    App.jsx            # Main React component
    components/
      SplashScreen.jsx # Configuration screen
      ChatInterface.jsx# Chat UI
    components/*.css   # Component styles
pha/
  agents/              # Agent implementations
  llm/                 # LLM backend abstraction
  utils/               # Model discovery, utilities
data/
  sample/              # Default synthetic persona
  */                   # Additional persona directories
```

## API Endpoints

### Configuration
```
GET /config
```
Returns available providers, models, baselines, and personas.

### Sessions
```
POST /sessions
Body: {
  "provider": "gemini",
  "model_id": "models/gemini-2.0-flash",
  "baseline": "pha",
  "persona_id": "sample",
  "gemini_api_key": "...",     // or openai_api_key
  "tavily_api_key": "..."      // optional
}
Response: { "session_id": "uuid", ... }

GET /sessions/{id}
DELETE /sessions/{id}

POST /sessions/{id}/chat
Body: { "message": "How is my sleep?" }
Response: { "response": "...", "processing_time": 1.23 }

GET /sessions/{id}/messages
Response: { "messages": [...] }
```

## Adding New Personas

1. Create directory: `data/my_persona/`
2. Add required CSV files:
   - `summary.csv` - Daily health metrics
   - `activities.csv` - Exercise records  
   - `profile.csv` - Demographics
   - `population_percentiles.csv` - Reference data (can copy from sample)
3. Optionally add `description.txt` with persona description
4. Restart the API server

The portal will automatically discover the new persona.

## Provider Support

| Provider | Status | Notes |
|----------|--------|-------|
| Gemini | ✅ Full | All baselines supported |
| OpenAI | ✅ Full | All baselines supported via OneTwo |
| Anthropic | 🚧 Coming Soon | Backend ready, UI disabled |

## Environment Variables

All API keys can be provided via the portal UI, but can also be set server-side:

```bash
# LLM Providers
GEMINI_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...

# Optional Services  
TAVILY_API_KEY=...     # Web search in DomainExpertAgent
```

## Development

### Frontend Development

```bash
cd frontend
npm install
npm run dev          # Development server with hot reload
npm run build        # Production build
npm run preview      # Preview production build
```

### Backend Development

```bash
# Run with auto-reload
uvicorn api.main:app --reload --port 8000

# Run tests
pytest tests/

# Type checking
mypy pha/
```

### Adding New Models

Models are automatically discovered from provider APIs when keys are available.
Fallback model lists are defined in `pha/utils/model_discovery.py`.

## User Study Mode

When `GLOBAL_USER_STUDY_FLAG = True` in `api/main.py`, the splash screen hides provider, model, and API key selection. Defaults are loaded from `config/user_study_defaults.json` (gitignored).

**Setup:**
1. Copy `config/user_study_defaults.example.json` to `config/user_study_defaults.json`
2. Fill in `provider`, `model`, and `api_keys` for your chosen provider
3. The API key stays on the server and is never sent to the frontend

**Endpoint:** `GET /user-study-defaults` returns `{ provider, model }` when user study mode is enabled (API keys are never exposed).

## For User Studies

The portal is designed for A/B testing different agent architectures:

1. **Randomize Assignment**: Create sessions with different `baseline` values
2. **Collect Data**: All messages stored in session objects via `/sessions/{id}/messages`
3. **Measure Performance**: `processing_time` returned with each response
4. **Clean Separation**: Each session is independent, can be deleted after study

## Troubleshooting

### "API key not valid" error
- Ensure correct key format (Gemini: `AIza...`, OpenAI: `sk-...`)
- Check provider selection matches the key entered
- Try refreshing and re-entering the key

### PHIA baseline fails
- PHIA supports Gemini and OpenAI (uses OneTwo ReAct, which has no Anthropic backend)
- Switch to PHA or Parallel baseline for Anthropic models

### Frontend can't connect to API
- Ensure API server is running on port 8000
- Check CORS settings if running from different domain
- Verify no firewall blocking localhost connections
