# JARVIS Project TODO

## Project Status: 🟡 In Progress

---

## Phase 1: Foundation Setup

### Task 1: Project Setup & Config Manager
- [ ] Create project structure (`jarvis/` directories)
- [ ] Create `.env.example` with all API key placeholders
- [ ] Create `config.yaml` with default settings
- [ ] Implement `config/config_loader.py`
- [ ] Generate `requirements.txt`
- [ ] Test config loading from both env and yaml

### Task 2: Memory Manager (ChromaDB)
- [ ] Implement `core/memory_manager.py`
- [ ] Create 3 collections: memories, chat_messages, my_data
- [ ] Implement `add_memory()` method
- [ ] Implement `recall()` method
- [ ] Implement `save_chat()` method
- [ ] Implement `get_chat_context()` method
- [ ] Implement `index_document()` method
- [ ] Test all CRUD operations
- [ ] Verify persistence across restarts

### Task 3: Audio Handler (STT/TTS/Wake Word)
- [ ] Implement `core/audio_handler.py`
- [ ] Set up Porcupine wake word detection
- [ ] Implement `listen_for_wake_word()` method
- [ ] Implement `speech_to_text()` using Google Speech Recognition
- [ ] Implement `text_to_speech()` using pyttsx3
- [ ] Implement `listen_loop()` orchestration
- [ ] Test wake word detection accuracy
- [ ] Test STT/TTS pipeline end-to-end

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
- [ ] Implement `tools/email_tool.py`
- [ ] Set up Gmail API OAuth2
- [ ] Implement `read_emails()` tool
- [ ] Implement `send_email()` tool
- [ ] Register tools with LangChain
- [ ] Test email reading
- [ ] Test email sending

### Task 6: Calendar Tool
- [ ] Implement `tools/calendar_tool.py`
- [ ] Set up Calendar API OAuth2
- [ ] Implement `read_my_calendar()` tool (user calendar, read-only)
- [ ] Implement `add_jarvis_event()` tool
- [ ] Implement `list_jarvis_events()` tool
- [ ] Register tools with LangChain
- [ ] Test calendar operations

### Task 7: Spotify Tool
- [ ] Implement `tools/spotify_tool.py`
- [ ] Set up Spotify OAuth
- [ ] Implement `play_track()` tool
- [ ] Implement `pause_playback()` tool
- [ ] Implement `current_track()` tool
- [ ] Register tools with LangChain
- [ ] Test playback control

### Task 8: Info Tools (News/Weather/Web)
- [ ] Implement `tools/info_tools.py`
- [ ] Implement `get_weather()` using OpenWeatherMap
- [ ] Implement `get_news()` using NewsAPI
- [ ] Implement `search_web()` tool
- [ ] Implement `wikipedia_search()` tool
- [ ] Register tools with LangChain
- [ ] Test all info tools

### Task 9: Communication Tools (Telegram/GDrive)
- [ ] Implement `tools/comm_tools.py`
- [ ] Set up Telegram Bot API
- [ ] Implement `send_to_telegram()` tool
- [ ] Set up Google Drive API OAuth2
- [ ] Implement `read_gdrive_file()` tool
- [ ] Register tools with LangChain
- [ ] Test Telegram messaging
- [ ] Test GDrive file reading

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
