# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Heathcliff** is a voice-enabled personal AI assistant that integrates with daily services (Gmail, Google Calendar, Spotify, and more). It uses LLM-powered decision making to orchestrate tasks across multiple external APIs. Users activate it with a wake word, issue voice commands, and the assistant handles the execution through integrated tools.

## Architecture

The system has three major layers:

**1. Voice Layer** (`voice/main.py`)
- Wake word detection using pvporcupine
- Speech-to-text via Google Speech Recognition
- Runs in a separate thread listening for activation

**2. Agent Layer** (`agent.py`, `settings.py`)
- LangChain-based orchestration with Gemini Flash 2.5
- React agent pattern for tool selection and execution
- Maintains conversation history and context

**3. Tools Layer** (`tools/`)
- **gmail_tools.py**: Search, read, send, draft emails
- **calendar_tools.py**: Create, read, update, delete calendar events
- **spotify_tools.py**: Control playback, search tracks
- **alexa_tools.py**: Placeholder for voice assistant integration

The application entry point is `app.py` which initializes all components.

## Development Commands

### Environment Setup
```bash
# Install system dependency (Linux)
sudo apt install python3-pyaudio

# Install Python dependencies
pip install -r requirements.txt

# Environment variables
# Create .env file with API keys and set configuration
```

### Running the Application
```bash
python app.py
```

The application starts the voice listener in a background thread and waits for the wake word.

### Debugging
- Check `logger.py` for logging setup
- Enable debug mode by adjusting log levels in logger configuration
- Monitor voice thread output for wake word detection issues

## Key Configuration

### API Credentials
Store credentials in `.env` (not in version control). Required variables:
- `GEMINI_API_KEY`: For LLM inference (Gemini Flash 2.5)
- `GOOGLE_APPLICATION_CREDENTIALS`: JSON path for Gmail/Calendar APIs
- `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`: For Spotify control
- `TELEGRAM_BOT_TOKEN`: For Telegram integration
- `OPENWEATHERMAP_API_KEY`: For weather data
- `NEWSAPI_KEY`: For news aggregation

### Runtime Settings
Edit `settings.py` to customize:
- `APPLICATION_NAME`: Display name (default: "Heathcliff")
- `MASTER`: Your name and credentials
- `APPLICATION_VERSION`: Version tracking

## Code Patterns & Conventions

### Shared Agent Memory (`plan/MEMORY.md`)

**CRITICAL for Multi-Agent Coordination**: When multiple agents work on this project concurrently, they use `plan/MEMORY.md` as a central knowledge base.

- Eliminates duplicate analysis and investigation work
- Reduces API costs by preventing redundant discovery
- Maintains consistent understanding across concurrent agents
- Serves as handoff point for agent-to-agent context
- Before starting work: Always check `plan/MEMORY.md` first for discoveries and decisions
- After completing work: Update `plan/MEMORY.md` with new findings immediately
- Reuse previous agent analyses instead of re-analyzing code
- Share discovered API patterns and implementation strategies
- Reference prior issue investigations to avoid duplicate debugging
- Leverage previous architectural reviews instead of redesigning

**Memory sections** (`plan/MEMORY.md`):
- **Codebase Discoveries**: Project structure, architecture patterns, code organization
- **API Credentials & Configuration**: Required variables and configuration files
- **Development Decisions & Standards**: Code standards, naming conventions, testing requirements
- **Dependencies & Versions**: Package versions and compatibility notes
- **Known Issues & Considerations**: Thread safety, rate limits, accuracy issues, error handling
- **Planned Features & Roadmap**: Prioritized development tasks
- **Agent Coordination Notes**: When to check/update memory, guidelines
- **Recent Agent Activity**: Timestamped log of completed work by agents

### Task Tracking

**CRITICAL**: Always update `plan/TODO.md` as work progresses:

- When starting a task, check the box: `- [x] Task description`
- Update immediately after completing each item
- Keep the TODO list synchronized with actual progress
- This is mandatory for all development work

### Code Organization

**Maintain centralized utilities**:

- All shared/reusable code MUST go in `utils/` or `helpers/` directory
- NEVER duplicate utility functions across modules
- Before writing a helper function, check if it already exists in `utils/`
- Common utilities include: API clients, data transformers, validators, formatters

### Tool Implementation
Tools follow LangChain's Tool pattern:
```python
from langchain.agents import Tool

tool = Tool(
    name="tool_name",
    func=function_reference,
    description="What the tool does"
)
```

### Agent Chain Setup
The base agent uses:
- `LLMChain` for basic request-response
- `create_react_agent` for multi-step reasoning with tools
- Prompt template with input variables: `input`, `context`, `history`

### Voice Processing Flow
1. `VoiceListener.start()` initializes Porcupine and PyAudio
2. Listens for wake word in blocking loop
3. On detection, calls registered callback with recognized text
4. Agent processes and returns response
5. Response is spoken back to user

## Testing Considerations

Since this is a voice-enabled assistant:
- Test tool integrations independently before connecting to voice pipeline
- Verify API credentials are correct (errors from external APIs won't be obvious during voice interaction)
- Test LLM prompt effectiveness with various input patterns
- Monitor for audio device issues (PyAudio setup is platform-dependent)

## Important Notes

- **Early Stage**: Project is in v0.0.1 with core infrastructure. Active feature development ongoing.
- **Thread Safety**: Voice listener runs in separate thread; ensure agent handles concurrent requests if needed.
- **API Rate Limits**: Gmail, Calendar, and Spotify have rate limits—consider adding backoff logic.
- **Wake Word Accuracy**: Porcupine's free tier has limited accuracy; consider paid tier for production use.
- **Context Management**: Conversation history is maintained in-memory; add persistence layer for long-term memory.

## Dependencies

Core packages:
- **langchain** (0.3.19): Agent orchestration
- **langchain-google-genai**: Gemini Flash 2.5 integration
- **langgraph**: StateGraph for advanced agent workflows
- **langchain-community** (0.3.18): Pre-built tool integrations
- **chromadb**: Vector database for memory/conversation storage
- **pvporcupine** (3.0.5): Wake word detection
- **SpeechRecognition** (3.14.1): STT
- **pyttsx3**: Text-to-speech
- **PyYAML**, **python-dotenv**: Configuration management
- **spotipy**: Spotify API client
- **python-telegram-bot**: Telegram Bot API
- **streamlit**: Dashboard UI

## Next Steps for Development

1. Complete `main.py` orchestrator
2. Implement persistent conversation memory with ChromaDB (3 collections: memories, chats, my_data)
3. Add error handling and graceful degradation for API failures
4. Build Streamlit UI for dashboard
5. Add more tool integrations (Twitter/X, Google Drive, News, Weather)
6. Implement LangGraph StateGraph for advanced agent workflows

See `plan/TODO.md` for detailed task breakdown and progress tracking.
