# Heathcliff Project Plan

## Architecture Overview

```
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

**Core**

- **STT**: Whisper (local) or Google Speech-to-Text
- **Wake word**: OpenWakeWord (local, free) - pre-trained models for "hey_jarvis", "alexa", etc.
- **LLM**: Gemini Flash 2.5 via LangChain
- **Agent**: LangGraph for tool orchestration
- **Vector DB**: ChromaDB (simple, local)
- **TTS**: Google TTS or ElevenLabs

**Tools Integration**

- Gmail API
- Spotify Web API
- OpenWeatherMap
- Twitter/X API
- Google Calendar API
- Telegram Bot API
- Google Drive API
- NewsAPI

## Module Structure

```
heathcliff/
├── core/
│   ├── audio_handler.py      # STT, TTS, wake word
│   ├── agent_core.py          # LangGraph agent
│   └── memory_manager.py      # Vector store ops
├── tools/
│   ├── email_tool.py
│   ├── spotify_tool.py
│   ├── calendar_tool.py
│   ├── news_tool.py
│   ├── web_search_tool.py
│   ├── telegram_tool.py
│   └── gdrive_tool.py
├── ui/
│   └── streamlit_app.py
├── config/
│   └── settings.yaml          # API keys, configs
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

**Agent Pattern**

- Use LangGraph StateGraph
- Each tool = node
- Routing based on intent classification
- Maintain conversation state across turns

**Memory Strategy**

- Embed conversations with sentence-transformers
- Query vector DB for relevant context before LLM call
- Inject top-k similar conversations into prompt

**Config Management**

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
