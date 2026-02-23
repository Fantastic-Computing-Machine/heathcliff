# Heathcliff Project TODO

## Project Status: 🟡 In Progress

---

## Phase 1: Foundation Setup

### Task 1: Project Setup & Config Manager

- [x] Create project structure (`heathcliff/` directories)
- [x] Create `.env.example` with all API key placeholders
- [x] Create `config/config.py` with default settings
- [x] Implement config singleton in `config/__init__.py`
- [x] Manage dependencies via `pyproject.toml` / `uv.lock`
- [x] Test config loading from both env and TOML

### Task 2: Memory Manager (ChromaDB)

- [x] Implement `core/memory_manager.py`
- [x] Create 3 collections: memories, chat_messages, my_data
- [x] Implement `add_memory()` method
- [x] Implement `recall()` method
- [x] Implement `save_chat()` method
- [x] Implement `get_chat_context()` method
- [x] Implement `index_document()` method
- [x] Implement `get_all_sessions()` method
- [x] Implement `get_session_history()` method
- [x] Implement `delete_all_chats()` method
- [x] Test all CRUD operations
- [x] Verify persistence across restarts

### Task 3: Audio Handler (STT/TTS/Wake Word)

- [x] Implement `core/audio_handler.py`
- [x] Set up Porcupine wake word detection
- [x] Implement `listen_for_wake_word()` method
- [x] Implement `speech_to_text()` using Google Speech Recognition
- [x] Implement `text_to_speech()` using espeak
- [x] Implement `listen_loop()` orchestration
- [x] Test wake word detection accuracy
- [x] Test STT/TTS pipeline end-to-end

---

## Phase 2: Core Agent

### Task 4: Gemini Agent Core

- [x] Implement `core/agent_core.py`
- [x] Set up LangChain Google GenAI integration
- [x] Build supervisor agent with `create_agent()`
- [x] Implement singleton `HeathcliffAgent` with self-wiring
- [x] Implement `invoke()` method
- [x] Implement `stream_invoke()` method
- [x] Integrate MemoryManager for context (pair-based retrieval)
- [x] Integrate Mem0 for long-term memory
- [x] Inject temporal context (date/time) into user prompt
- [x] Test basic conversation flow

---

## Phase 3: Tools Integration (Subagents)

### Task 5: Email Subagent

- [x] Implement `core/subagents/email/tools.py`
- [x] Implement `core/subagents/email/agent.py`
- [x] Set up Gmail API OAuth2
- [x] Implement `read_emails()` tool
- [x] Implement `send_email()` tool
- [x] Register tools with LangChain
- [x] Test email reading
- [x] Test email sending

### Task 6: Calendar Subagent

- [x] Implement `core/subagents/calendar/tools.py`
- [x] Implement `core/subagents/calendar/agent.py`
- [x] Set up Calendar API OAuth2
- [x] Implement `read_my_calendar()` tool (user calendar, read-only)
- [x] Implement `add_heathcliff_event()` tool
- [x] Implement `list_heathcliff_events()` tool
- [x] Register tools with LangChain
- [x] Test calendar operations

### Task 7: Music Subagent (Spotify)

- [x] Implement `core/subagents/music/tools.py`
- [x] Implement `core/subagents/music/agent.py`
- [x] Set up Spotify OAuth
- [x] Implement `play_track()` tool
- [x] Implement `pause_playback()` tool
- [x] Implement `current_track()` tool
- [x] Register tools with LangChain
- [x] Test playback control

### Task 8: Info Subagent (News/Weather/Web)

- [x] Implement `core/subagents/info/tools.py`
- [x] Implement `core/subagents/info/agent.py`
- [x] Implement `get_weather()` using OpenWeatherMap (via LangChain wrapper)
- [x] Implement `get_news()` using NewsAPI
- [x] Implement `search_web()` tool (DuckDuckGo primary + Google fallback)
- [x] Implement `wikipedia_search()` tool
- [x] Implement `read_website()` tool (webpage content extraction)
- [x] Register tools with LangChain
- [x] Test all info tools

### Task 9: Communication Subagent (Telegram/GDrive)

- [x] Implement `core/subagents/comms/tools.py`
- [x] Implement `core/subagents/comms/agent.py`
- [x] Set up Telegram Bot API
- [x] Implement `send_to_telegram()` tool
- [x] Set up Google Drive API OAuth2
- [x] Implement `read_gdrive_file()` tool
- [x] Register tools with LangChain
- [x] Test Telegram messaging
- [x] Test GDrive file reading

### Task 9b: Contacts Subagent

- [x] Implement `core/subagents/contacts/tools.py`
- [x] Implement `core/subagents/contacts/agent.py`

---

## Phase 4: UI & Integration

### Task 10: Streamlit Dashboard

- [x] Implement `ui/Home.py`
- [x] Create Home chat page with background initialization
- [x] Create Memories page (view/add/delete)
- [x] Create Analytics page
- [x] Create Settings page (config viewer)
- [x] Create Chat History page (session browser)
- [x] Add human-in-the-loop approval UI for sensitive tools
- [x] Test UI with real data

### Task 10b: 3D Blob UI

- [x] Implement `assets/` (Three.js standalone frontend)
- [x] Implement `ui/server.py` (FastAPI backend)
- [x] GPU simplex noise blob with 4 animation states
- [x] Chat overlay bridging to HeathcliffAgent

### Task 11: Main Orchestrator

- [x] Implement `main.py`
- [x] Initialize all components
- [x] Connect audio → agent → memory pipeline
- [x] Implement session management (UUID per wake)
- [x] Add `--text` mode for testing
- [x] Add error handling and logging
- [x] Test full end-to-end flow

### Task 12: Skills Framework

- [x] Implement `skills/skills.py` (3 skills: master_info, british_persona, email_safety)
- [x] Implement `skills/skill_tools.py` (tool registration)
- [x] Implement `skills/master_info.py` (master profile skill from TOML)
- [x] Wire skills into `HeathcliffAgent._assemble_default_tools()`

### Task 13: Observability

- [x] Implement `utils/langfuse_client.py`
- [x] Register Langfuse callback handler in HeathcliffAgent
- [x] Log tool invocations as Langfuse events

### Task 14: User Reference Generalization

- [x] Replace all hardcoded user references with generic placeholders
- [x] Update config files, skills, agent logic, UI, README, and tests

---

## Phase 5: Testing & Polish

### Integration Testing

- [ ] Test wake word → response flow
- [ ] Test tool calling accuracy
- [ ] Test memory recall accuracy
- [x] Test multi-turn conversations (pair-based context)
- [ ] Test error recovery

### Documentation

- [x] Write README.md with setup instructions
- [x] Document all API key requirements
- [x] Document config/config.py options
- [x] Add usage examples
- [ ] Create troubleshooting guide

### Deployment

- [ ] Create Docker container (optional)
- [ ] Set up systemd service (optional)
- [ ] Configure auto-start on boot
- [ ] Set up logging rotation

---

## Notes & Blockers

### Completed

- ✅ Architecture design
- ✅ Tech stack selection
- ✅ Task breakdown and dependency mapping
- ✅ Subagents architecture refactor (6 domains under `core/subagents/`)
- ✅ Skills framework (dynamic runtime loading)
- ✅ Mem0 long-term memory integration
- ✅ Pair-based semantic/chronological context retrieval
- ✅ USER_PROMPT_TEMPLATE with XML delimiters and structured sections
- ✅ Streamlit background initialization (non-blocking)
- ✅ Chat History page with session management
- ✅ Master profile migration to TOML (`master_info.toml`)
- ✅ Langfuse observability (traces, events, tool logging)
- ✅ 3D Blob UI (Three.js + FastAPI)
- ✅ Human-in-the-loop approval for sensitive tools
- ✅ Weather API refactored to LangChain OpenWeatherMapAPIWrapper
- ✅ User reference generalization (hardcoded names → generic placeholders)

### In Progress

- 🔄 Integration testing & error recovery

### Blocked

- ⛔

### Questions/Decisions Needed

-

---

**Last Updated**: 2026-02-23
