# Heathcliff 🎤

**Voice-Activated AI Assistant that just does stuff -- no BS.**

Heathcliff is a voice-enabled personal AI assistant that integrates with your daily services. Wake it up with "Heathcliff", give it commands, and watch it orchestrate tasks across Gmail, Calendar, Spotify, Weather, News, and more using Gemini Flash 2.5-powered decision making.

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
# Edit .env and add your GEMINI_API_KEY + service keys
# (Optional) Add LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY
#          + LANGFUSE_BASE_URL (https://cloud.langfuse.com or us.cloud...) for observability

# 5. Run in text mode (no voice hardware needed)
uv run python main.py --text

# OR run in voice mode
uv run python main.py

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
- **Weather**: Real-time weather updates
- **News**: Latest headlines by topic
- **Web Search**: DuckDuckGo + Wikipedia
- **Telegram**: Send notifications
- **Google Drive**: Read files

### 🧠 Intelligence

- Gemini Flash 2.5 LLM
- LangGraph agent orchestration
- ChromaDB vector memory
- Multi-turn conversation context
- Long-term memory storage

### 📈 Observability

- Built-in Langfuse tracing for every conversation
- LangChain callback handler automatically captures Gemini prompts/completions
- Tool usage + errors are streamed to Langfuse events for debugging

### 💻 Interfaces

- Voice mode (main.py)
- Text mode for testing
- Streamlit web dashboard

## Tech Stack

- **LLM Framework**: LangChain + LangGraph with Gemini 2.0 Flash
- **Memory**: ChromaDB for persistent vector storage
- **Voice**: pvporcupine (wake word), Google STT
- **Integrations**: Gmail, Google Calendar, Spotify APIs
- **Audio**: PyAudio

## Architecture

Heathcliff uses a LangGraph-based agent architecture with 4 nodes:

1. **Retrieval Node**: Fetches relevant context and memories from ChromaDB
2. **Reasoning Node**: Processes input with Gemini LLM, determines actions
3. **Tool Calling Node**: Executes requested tools (weather, time, etc.)
4. **Output Node**: Saves conversation to memory, returns response

## Usage Modes

### 1. Voice Mode (Default)

```bash
python main.py
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
python main.py --text
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
streamlit run ui/Home.py
```

Access at `http://localhost:8501`

**Dashboard Features:**

- **Home**: Chat interface with Heathcliff
- **Memories**: View, search, and add long-term memories
- **Analytics**: Usage statistics and conversation insights
- **Settings**: View API configuration and system status

## Programmatic Usage

### Using the Agent Core (Programmatic)

```python
from core import MemoryManager, HeathcliffAgent
from config import Config

# Initialize components
config = Config
memory = MemoryManager(config=config)
agent = HeathcliffAgent(config=config, memory_manager=memory)

# Single turn conversation
response = agent.invoke("Hello! What can you do?")
print(response)

# Multi-turn conversation (same session maintains context)
session_id = "my-session-123"
response1 = agent.invoke("My name is Adi", session_id=session_id)
response2 = agent.invoke("What's my name?", session_id=session_id)
# response2 will know your name is Adi
```

### Using the Memory Manager

```python
from core import MemoryManager

memory = MemoryManager(persist_dir="./chroma_db")

# Store a long-term memory
memory_id = memory.add_memory("User prefers dark mode", category="preferences")

# Recall relevant memories
results = memory.recall("what are user preferences?", n=3)
print(results["documents"])

# Save chat conversation
memory.save_chat(
    user_msg="What's the weather?",
    assistant_msg="It's sunny and 72F",
    session_id="session-123"
)

# Retrieve chat context
context = memory.get_chat_context("weather", session_id="session-123")
```

### Voice Mode

```bash
python app.py
```

## Langfuse Observability

Heathcliff now ships with first-class Langfuse instrumentation:

1. Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and (optionally) `LANGFUSE_HOST` / `LANGFUSE_RELEASE` in `.env`.
   - If you're on Langfuse Cloud US/EU, also set `LANGFUSE_BASE_URL` to `https://us.cloud.langfuse.com` or `https://cloud.langfuse.com`.
2. Start the assistant like normal; every agent run creates a Langfuse trace named `heathcliff.agent`, tagged with `user_id=adiagarwal` (configurable via `observability.langfuse.user_id`).
3. Gemini prompt/response pairs automatically stream through the Langfuse LangChain callback handler.
4. Each external tool invocation is logged as a Langfuse event, so you can inspect failures and latency directly in the Langfuse UI.

**Troubleshooting tips**

- If no traces appear, run `python -m utils.langfuse_client` or start Heathcliff with `LOG_LEVEL=DEBUG` to confirm the Langfuse callback is registering.
- Double-check the Langfuse dashboard filters (environment/project) match the `observability.langfuse.environment` value in `config/config.py`.
- Serverless/text-only sessions may exit before the SDK flushes; add `LANGFUSE_DISABLE_BACKGROUND_FLUSH=false` or keep the process alive for a few seconds.
- The Langfuse callback handler automatically reads keys from environment variables. Passing `public_key`/`secret_key` directly will fail on newer Langfuse releases, so be sure the env vars are loaded before the process starts.

Disable observability anytime by setting `observability.langfuse.enabled` to `false` in `config/config.py`.

Say "Heathcliff" to activate, then give your command.

## Example Conversations

```text
User: Hello!
Heathcliff: Hello! I'm Heathcliff, your personal AI assistant. How can I help you today?

User: What's the weather in London?
Heathcliff: The weather in London is 72F and sunny.

User: My name is Adi and I work as a software engineer.
Heathcliff: Nice to meet you, Adi! I'll remember that you're a software engineer.

User: What do you know about me?
Heathcliff: Based on what I know, your name is Adi and you work as a software engineer.
```

## Project Structure

```text
heathcliff/
├── main.py                  # Main entry point (voice/text modes)
├── core/
│   ├── memory_manager.py    # ChromaDB-backed memory storage
│   ├── agent_core.py        # LangGraph agent orchestrator
│   └── audio_handler.py     # Voice I/O (wake word, STT, TTS)
├── config/
│   ├── config.py            # Configuration classes
│   └── __init__.py          # Config singleton
├── tools/                   # Tool integrations
│   ├── email_tool.py        # Gmail integration
│   ├── calendar_tool.py     # Google Calendar
│   ├── spotify_tool.py      # Spotify playback
│   ├── info_tools.py        # Weather, news, web search
│   └── comm_tools.py        # Telegram, Google Drive
├── utils/
│   └── google_auth.py       # OAuth2 credential manager
├── ui/                      # Streamlit dashboard
│   ├── Home.py              # Main chat interface
│   └── pages/
│       ├── 1_🧠_Memories.py
│       ├── 2_📊_Analytics.py
│       └── 3_⚙️_Settings.py
├── plan/                    # Planning docs
│   ├── INIT.md
│   ├── TODO.md
│   └── EXECUTION.md
├── .env.example             # API key template
├── config/config.py         # Runtime configuration
├── requirements.txt         # Python dependencies
├── SETUP.md                 # Detailed setup guide
└── README.md
```

## Development Status

**Phase 4 Complete (v1.0.0)** ✅

- ✅ Foundation Setup (Config, Memory, Audio)
- ✅ Core Agent (LangGraph with Gemini Flash 2.5)
- ✅ Tools Integration (Gmail, Calendar, Spotify, Weather, News, etc.)
- ✅ UI & Integration (Voice mode, Text mode, Streamlit dashboard)
- ⏳ Testing & Polish (Pending)

**Ready for production use with basic testing!**

## Contributing

See [plan/EXECUTION.md](plan/EXECUTION.md) for architecture details and [plan/TODO.md](plan/TODO.md) for remaining tasks.

## Documentation

- **[SETUP.md](SETUP.md)**: Complete setup guide with API credentials and troubleshooting
- **[plan/INIT.md](plan/INIT.md)**: Initial architecture and design decisions
- **[plan/EXECUTION.md](plan/EXECUTION.md)**: Detailed implementation plan

## License

MIT
