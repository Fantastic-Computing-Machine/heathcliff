# Heathcliff Project Plan

> **Historical Note**: This was the initial project plan from project inception. The codebase has since evolved significantly. See `plan/MEMORY.md` for the current architecture and `plan/TODO.md` for current task tracking.

## Architecture Overview

```txt
┌─────────────────┐
│  Audio Input    │ → Wake Word → Voice-to-Text
└─────────────────┘              ↓
                          ┌──────────────┐
                          │ Agent Core   │ ← Gemini Flash 2.5
                          │ (LangGraph)  │
                          └──────────────┘
                                 ↓
                    ┌────────────┼────────────┐
                    ↓            ↓            ↓
              [Tools Layer]  [Memory]   [Streamlit UI]
                    ↓            ↓
              Vector Store  Conversation
              (ChromaDB)    History
```

## Tech Stack Decisions

### Core

- **STT**: Whisper (local) or Google Speech-to-Text
- **Wake word**: Porcupine/Picovoice (free tier) or pvporcupine
- **LLM**: Gemini 3 Flash Preview (supervisor) + Gemini 2.5 Pro (tools) via LangChain
- **Agent**: LangGraph supervisor + subagents orchestration
- **Vector DB**: ChromaDB (simple, local)
- **TTS**: Google TTS or ElevenLabs

### Tools Integration

- Gmail API
- Spotify Web API
- OpenWeatherMap
- Google Calendar API
- Telegram Bot API
- Google Drive API
- NewsAPI
- DuckDuckGo Search (primary)
- Google Custom Search (fallback)
- Wikipedia API
- Web content reader (BeautifulSoup)

## Module Structure

```txt
heathcliff/
├── core/
│   ├── agent_core.py           # Singleton supervisor agent
│   ├── audio_handler.py        # STT, TTS, wake word
│   ├── memory_manager.py       # ChromaDB + Mem0 memory
│   ├── approval_handler.py     # Human-in-the-loop approval
│   └── subagents/              # Domain-specific subagents
│       ├── calendar/           # Google Calendar tools
│       ├── comms/              # Telegram, Google Drive
│       ├── contacts/           # Contact management
│       ├── email/              # Gmail tools
│       ├── info/               # Weather, News, Web search
│       └── music/              # Spotify playback
├── skills/                     # Dynamic skills loaded at runtime
│   ├── skill_tools.py
│   ├── skills.py
│   └── master_info.py
├── ui/
│   ├── Home.py                 # Streamlit dashboard
│   └── server.py               # FastAPI blob server
├── assets/                     # 3D Blob web frontend
├── config/
│   └── config.py               # Class-based config (reads .env + master_info.toml)
├── instructions/
│   └── prompts.py              # System/user prompt templates
├── voice/
│   └── main.py                 # Wake-word voice entry point
├── utils/                      # Auth, observability, helpers
├── master_info.toml            # User profile & preferences
└── main.py
```

## Implementation Phases

### Phase 1: Foundation (Week 1)

- Wake word detection loop
- Basic STT/TTS pipeline
- Gemini integration with LangChain
- Simple text-based interaction (no tools yet)
- ChromaDB setup for conversation storage

### Phase 2: Core Tools (Week 2)

- Email read/send
- Calendar operations (read yours, CRUD its own)
- Telegram sender
- Weather + jokes + time

### Phase 3: External APIs (Week 3)

- Spotify control
- Web search + Wikipedia
- News aggregator (configurable sources)
- X posting
- GDrive file reader

### Phase 4: Intelligence (Week 4)

- LangGraph tool orchestration
- Context-aware responses
- Memory retrieval from vector store
- Conversation continuity

### Phase 5: UI (Week 5)

- Streamlit dashboard
- Conversation history viewer
- Audio playback
- Settings configurator

## Key Implementation Notes

**Wake Word Logic**

```python
while True:
    audio = listen()
    if detect_wake_word(audio):
        conversation = []
        while in_session:
            user_input = stt(listen())
            response = agent.invoke(user_input, conversation)
            tts(response)
            save_to_vector_db(user_input, response)
```

### Agent Pattern

- Use LangGraph StateGraph
- Each tool = node
- Routing based on intent classification
- Maintain conversation state across turns

### Memory Strategy

- Embed conversations with sentence-transformers
- Query vector DB for relevant context before LLM call
- Inject top-k similar conversations into prompt

### Config Management

```yaml
apis:
  gemini_key: xxx
  spotify: {client_id: xxx, secret: xxx}
news:
  sources: [bbc, techcrunch]
  topics: [ai, tech]
```

## MVP Scope (Start Here)

1. Wake word + STT/TTS loop
2. Gemini integration
3. 3 tools: weather, email read, Telegram send
4. Basic Streamlit UI
5. Conversation logging (no vector search yet)

**Estimated**: 2-3 days for working MVP

Want to dive into any specific component first?
