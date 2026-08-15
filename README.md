# Heathcliff 🎤

**Voice-Activated AI Assistant that just does stuff -- no BS.**

Heathcliff is a voice-enabled personal AI assistant that integrates with your daily services. Wake it up with "Heathcliff", give it commands, and watch it orchestrate tasks across Gmail, Calendar, Spotify, Weather, News, and more using Gemini-powered decision making.

## Quick Start

**Get up and running in 5 minutes:**

```bash
# 1. Clone and navigate
git clone <your-repo-url>
cd heathcliff

# 2. Install system dependencies (Linux/WSL)
sudo apt install python3-pyaudio portaudio19-dev espeak

# 3. Set up Python environment with uv
curl -LsSf https://astral.sh/uv/install.sh | sh  # skip if you already have uv
uv sync  # creates .venv from pyproject.toml / uv.lock

# 4. Configure API keys
cp .env.example .env
# Edit .env and add your AI_KEY + service keys
# (Optional) Add LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY
#          + LANGFUSE_BASE_URL (https://cloud.langfuse.com or us.cloud...) for observability

# 4.5 Personalize your profile
# Edit master_info.toml (IMPORTANT/OPTIONAL sections are commented inline)
# Config reads this file path via MASTER_INFO_LOC in config/config.py

# 5. Run in text mode (the default; no voice hardware needed)
uv run python main.py

# Enable voice mode when audio hardware is configured
uv run python main.py --voice

# OR launch the Streamlit dashboard
uv run streamlit run ui/Home.py
```

**That's it!** For detailed setup including Google OAuth, Spotify, and other integrations, see [SETUP.md](SETUP.md).

## Key Features

### 🎤 Voice Interface

- Wake word detection ("Heathcliff")
- Speech-to-text and text-to-speech
- Conversational memory and context

### 🔧 Integrations

- **Gmail**: Read, search, send emails
- **Google Calendar**: View schedule, create events
- **Spotify**: Play music, control playback
- **Weather**: Real-time weather updates (OpenWeatherMap via LangChain wrapper)
- **News**: Latest headlines by topic (NewsAPI)
- **Web Search**: DuckDuckGo (primary) + Google Search (fallback) + Wikipedia
- **Web Reader**: Extract and read content from any webpage URL
- **Telegram**: Send notifications
- **Google Drive**: Read files

### 🧠 Intelligence

- Gemini 3 Flash Preview (supervisor) + Gemini 2.5 Pro (tool calls)
- LangGraph agent orchestration with supervisor + subagents
- Mem0 long-term memory (Gemini LLM + Gemini embeddings + ChromaDB vector store)
- Multi-turn conversation context with pair-based history
- Semantic + chronological context retrieval
- Dynamic skills framework (persona, safety rules, profile info)

### 📈 Observability

- Built-in Langfuse tracing for every conversation
- LangChain callback handler automatically captures Gemini prompts/completions
- Tool usage + errors are streamed to Langfuse events for debugging

### 💻 Interfaces

- Voice mode (`main.py`)
- Text mode by default (`main.py`; `--text` remains supported)
- Streamlit web dashboard (`ui/Home.py`)
- 3D Blob web UI (`ui/server.py` → `assets/`)

## Tech Stack

- **LLM Framework**: LangChain + LangGraph with Gemini 3 Flash Preview / Gemini 2.5 Pro
- **Memory**: Mem0 (long-term) + ChromaDB (chat history)
- **Voice**: pvporcupine (wake word), Google STT, espeak TTS
- **Integrations**: Gmail, Google Calendar, Spotify, Telegram, Google Drive, OpenWeatherMap, NewsAPI
- **Search**: DuckDuckGo (primary), Google Custom Search (fallback), Wikipedia
- **Audio**: PyAudio
- **Observability**: Langfuse

## Architecture

Heathcliff uses a **coordinator + subagents** architecture:

1. **Coordinator** (`HeathcliffAgent`): Singleton orchestrator that receives user input, retrieves context (Mem0 memories + ChromaDB chat history), then plans, executes, and aggregates domain subagent work.
2. **Subagents** (`core/subagents/`): Six domain agents — Info, Music, Email, Calendar, Contacts, Comms — each with its own `tools.py` and `agent.py`.
3. **Skills** (`skills/`): Three dynamic capabilities loaded at runtime — `master_info` (user profile), `british_persona` (voice/tone guide), and `email_safety` (email composition rules).
4. **Memory Layer**: Pair-based semantic + chronological chat context injected as `HumanMessage`/`AIMessage` pairs; Mem0 recall injected into `USER_PROMPT_TEMPLATE` with XML delimiters.

Coordinator execution now enforces task/runtime budgets and exposes stream completion metadata via `agents_used` and `agent_count` (replacing `tools_used`).

## Usage Modes

### 1. Voice Mode (Default)

```bash
uv run python main.py
```

- Say **"Heathcliff"** to activate
- Speak your command
- Heathcliff responds via audio

**Example:**

```text
[You say]: "Heathcliff"
[Heathcliff]: *listening beep*
[You say]: "What's the weather in London?"
[Heathcliff]: "The current weather in London is 72°F and partly cloudy..."
```

### 2. Text Mode (Testing/No Audio)

```bash
uv run python main.py --text
```

- Type commands in terminal
- Great for debugging and testing
- No microphone/speakers required

**Example:**

```text
You: What's the weather?
Heathcliff: The current weather in New York is 68°F and sunny...

You: Add an event to my calendar for tomorrow at 2pm
Heathcliff: I've added an event to your calendar for tomorrow at 2:00 PM...
```

### 3. Streamlit Dashboard

```bash
uv run streamlit run ui/Home.py
```

Access at `http://localhost:8501`

**Dashboard Pages:**

- **Home**: Chat interface with Heathcliff
- **Memories**: View, search, and add long-term memories
- **Analytics**: Usage statistics and conversation insights
- **Settings**: View API configuration and system status
- **Chat History**: Browse and search past conversation sessions

### 4. 3D Blob UI (FastAPI)

```bash
uv run python ui/server.py
```

Access at `http://localhost:8600`

- GPU simplex noise animated blob with 4 states
- Chat overlay bridging to `HeathcliffAgent`

## Programmatic Usage

### Using the Agent Core (Programmatic)

```python
from core import MemoryManager, HeathcliffAgent

# Initialize components (Config is read internally)
memory = MemoryManager()
agent = HeathcliffAgent(memory_manager=memory)

# Single turn conversation
response = agent.invoke("Hello! What can you do?")
print(response)

# Multi-turn conversation (same session maintains context)
session_id = "my-session-123"
response1 = agent.invoke("My name is Alex", conversation_id=session_id)
response2 = agent.invoke("What's my name?", conversation_id=session_id)
# response2 will know your name is Alex
```

### Using the Memory Manager

```python
from core import MemoryManager

memory = MemoryManager()  # Uses ChromaDB config from Config singleton

# Store a long-term memory
memory_id = memory.add_memory("User prefers dark mode", category="preferences")

# Recall relevant memories
results = memory.recall("what are user preferences?", n=3)
print(results["documents"])

# Save chat conversation
memory.save_turn(
    user_msg="What's the weather?",
    assistant_msg="It's sunny and 72F",
    conversation_id="session-123"
)
```

## Langfuse Observability

Heathcliff now ships with first-class Langfuse instrumentation:

1. Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and (optionally) `LANGFUSE_HOST` / `LANGFUSE_RELEASE` in `.env`.
   - If you're on Langfuse Cloud US/EU, also set `LANGFUSE_BASE_URL` to `https://us.cloud.langfuse.com` or `https://cloud.langfuse.com`.
2. Start the assistant like normal; every agent run creates a Langfuse trace named `heathcliff.agent`, tagged with a configurable `LANGFUSE_USER_ID` (see `LangFuseConf` in `config/config.py`).
3. Gemini prompt/response pairs automatically stream through the Langfuse LangChain callback handler.
4. Each external tool invocation is logged as a Langfuse event, so you can inspect failures and latency directly in the Langfuse UI.

### Troubleshooting Tips

- If no traces appear, run `python -m utils.langfuse_client` or start Heathcliff with `LOG_LEVEL=DEBUG` to confirm the Langfuse callback is registering.
- Double-check the Langfuse dashboard filters (environment/project) match the `ENVIRONMENT` value in `LangFuseConf` in `config/config.py`.
- Serverless/text-only sessions may exit before the SDK flushes; add `LANGFUSE_DISABLE_BACKGROUND_FLUSH=false` or keep the process alive for a few seconds.
- The Langfuse callback handler automatically reads keys from environment variables. Passing `public_key`/`secret_key` directly will fail on newer Langfuse releases, so be sure the env vars are loaded before the process starts.

Disable observability anytime by removing or unsetting the `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` environment variables.

## Example Conversations

```text
User: Hello!
Heathcliff: Hello! I'm Heathcliff, your personal AI assistant. How can I help you today?

User: What's the weather in London?
Heathcliff: The weather in London is 72F and sunny.

User: My name is Alex and I work as a software engineer.
Heathcliff: Nice to meet you, Alex! I'll remember that you're a software engineer.

User: What do you know about me?
Heathcliff: Based on what I know, your name is Alex and you work as a software engineer.
```

## Contributing

See [plan/EXECUTION.md](plan/EXECUTION.md) for architecture details, [plan/TODO.md](plan/TODO.md) for remaining tasks, and [AGENTS.md](AGENTS.md) for coding guidelines.

## Documentation

- **[SETUP.md](SETUP.md)**: Complete setup guide with API credentials and troubleshooting
- **[AGENTS.md](AGENTS.md)**: Repository guidelines and coding conventions
- **[plan/INIT.md](plan/INIT.md)**: Initial architecture and design decisions
- **[plan/EXECUTION.md](plan/EXECUTION.md)**: Detailed implementation plan
- **[plan/TODO.md](plan/TODO.md)**: Task tracking and project status
- **[plan/MEMORY.md](plan/MEMORY.md)**: Shared agent working memory and discoveries
- **[plan/DB-SETUP.md](plan/DB-SETUP.md)**: ChromaDB multi-collection setup
- **[docs/ARCHITECTURE.mmd](docs/ARCHITECTURE.mmd)**: Mermaid architecture diagram

## License

MIT
