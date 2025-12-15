# JARVIS Project TODO

## Project Status: 🟡 In Progress

---

## Phase 1: Foundation Setup

### Task 1: Project Setup & Config Manager
- [x] Create project structure (`jarvis/` directories)
- [x] Create `.env.example` with all API key placeholders
- [x] Create `config.yaml` with default settings
- [x] Implement `config/config_loader.py`
- [x] Generate `requirements.txt`
- [x] Test config loading from both env and yaml

### Task 2: Memory Manager (ChromaDB)
- [x] Implement `core/memory_manager.py`
- [x] Create 3 collections: memories, chat_messages, my_data
- [x] Implement `add_memory()` method
- [x] Implement `recall()` method
- [x] Implement `save_chat()` method
- [x] Implement `get_chat_context()` method
- [x] Implement `index_document()` method
- [x] Test all CRUD operations
- [x] Verify persistence across restarts

### Task 3: Audio Handler (STT/TTS/Wake Word)
- [x] Implement `core/audio_handler.py`
- [x] Set up Porcupine wake word detection
- [x] Implement `listen_for_wake_word()` method
- [x] Implement `speech_to_text()` using Google Speech Recognition
- [x] Implement `text_to_speech()` using pyttsx3
- [x] Implement `listen_loop()` orchestration
- [x] Test wake word detection accuracy
- [x] Test STT/TTS pipeline end-to-end

---

## Phase 2: Core Agent

### Task 4: Gemini Agent Core
- [ ] Implement `core/agent_core.py`
- [ ] Set up LangChain Google GenAI integration
- [ ] Define `AgentState` TypedDict
- [ ] Build LangGraph StateGraph
- [ ] Add retrieval node
- [ ] Add reasoning node
- [ ] Add tool calling node
- [ ] Implement `invoke()` method
- [ ] Integrate MemoryManager for context
- [ ] Test basic conversation flow

---

## Phase 3: Tools Integration

### Task 5: Email Tool
- [x] Implement `tools/email_tool.py`
- [x] Set up Gmail API OAuth2
- [x] Implement `read_emails()` tool
- [x] Implement `send_email()` tool
- [x] Register tools with LangChain
- [x] Test email reading
- [x] Test email sending

### Task 6: Calendar Tool
- [x] Implement `tools/calendar_tool.py`
- [x] Set up Calendar API OAuth2
- [x] Implement `read_my_calendar()` tool (user calendar, read-only)
- [x] Implement `add_jarvis_event()` tool
- [x] Implement `list_jarvis_events()` tool
- [x] Register tools with LangChain
- [x] Test calendar operations

### Task 7: Spotify Tool
- [x] Implement `tools/spotify_tool.py`
- [x] Set up Spotify OAuth
- [x] Implement `play_track()` tool
- [x] Implement `pause_playback()` tool
- [x] Implement `current_track()` tool
- [x] Register tools with LangChain
- [x] Test playback control

### Task 8: Info Tools (News/Weather/Web)
- [x] Implement `tools/info_tools.py`
- [x] Implement `get_weather()` using OpenWeatherMap
- [x] Implement `get_news()` using NewsAPI
- [x] Implement `search_web()` tool
- [x] Implement `wikipedia_search()` tool
- [x] Register tools with LangChain
- [x] Test all info tools

### Task 9: Communication Tools (Telegram/GDrive)
- [x] Implement `tools/comm_tools.py`
- [x] Set up Telegram Bot API
- [x] Implement `send_to_telegram()` tool
- [x] Set up Google Drive API OAuth2
- [x] Implement `read_gdrive_file()` tool
- [x] Register tools with LangChain
- [x] Test Telegram messaging
- [x] Test GDrive file reading

---

## Phase 4: UI & Integration

### Task 10: Streamlit Dashboard
- [ ] Implement `ui/streamlit_app.py`
- [ ] Create Chat History tab
- [ ] Create Memories tab (view/add/delete)
- [ ] Create Settings tab (config editor)
- [ ] Add timestamp display
- [ ] Add search/filter functionality
- [ ] Test UI with real data

### Task 11: Main Orchestrator
- [ ] Implement `main.py`
- [ ] Initialize all components
- [ ] Connect audio → agent → memory pipeline
- [ ] Implement session management (UUID per wake)
- [ ] Add graceful shutdown handling
- [ ] Add error handling and logging
- [ ] Test full end-to-end flow

---

## Phase 5: Testing & Polish

### Integration Testing
- [ ] Test wake word → response flow
- [ ] Test tool calling accuracy
- [ ] Test memory recall accuracy
- [ ] Test multi-turn conversations
- [ ] Test error recovery

### Documentation
- [ ] Write README.md with setup instructions
- [ ] Document all API key requirements
- [ ] Document config.yaml options
- [ ] Add usage examples
- [ ] Create troubleshooting guide

### Deployment
- [ ] Create Docker container (optional)
- [ ] Set up systemd service (optional)
- [ ] Configure auto-start on boot
- [ ] Set up logging rotation

---

## MVP Milestone (First 2-3 Days)

**Goal**: Basic working voice assistant

- [ ] Wake word detection working
- [ ] STT/TTS pipeline functional
- [ ] Gemini integration responding
- [ ] 3 basic tools: weather, email read, Telegram send
- [ ] Basic Streamlit UI showing conversations
- [ ] Conversation logging (no vector search yet)

---

## Notes & Blockers

### Completed
- ✅ Architecture design
- ✅ Tech stack selection
- ✅ Task breakdown and dependency mapping

### In Progress
- 🔄

### Blocked
- ⛔

### Questions/Decisions Needed
-

---

**Last Updated**: 2025-12-13
