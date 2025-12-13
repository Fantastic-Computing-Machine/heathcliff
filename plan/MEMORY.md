# Shared Agent Memory & Discovery

This file serves as the **working memory** for all coding agents on the Heathcliff project. It tracks discoveries, issues, and recent activity. For complete project documentation, see `.claude/CLAUDE.md`.

## How Agents Use This File

- Before starting work: Check this file for ongoing issues, recent discoveries, and previous agent findings
- After completing work: Update this file with new issues, workarounds, code patterns discovered, and activity log
- Reference `.claude/CLAUDE.md` for project overview, architecture, configuration, and development standards
- Cost optimization: Reuse previous agent discoveries instead of re-investigating
- Share API integration workarounds and debugging strategies discovered during work

---

## Code Patterns & Implementation Notes

### Phase 1 Foundation - COMPLETED
- **Config Management**: Singleton pattern in `config/config_loader.py` with `get_config()` function
- **Memory Manager**: ChromaDB with 3 collections (memories, chats, my_data) in `core/memory_manager.py`
  - `add_memory()` for long-term facts with categories
  - `recall()` for semantic search of memories
  - `save_chat()` stores user/assistant message pairs
  - `get_chat_context()` retrieves relevant conversation history
  - `index_document()` for emails/files in my_data collection
- **Audio Handler**: Complete STT/TTS/wake word pipeline in `core/audio_handler.py`
  - Porcupine for wake word detection with PyAudio stream
  - Google Speech Recognition for STT
  - pyttsx3 for configurable TTS
  - `listen_loop()` orchestrates wake → listen → process → speak cycle
  - Background threading support with `start_background_listener()`

### Configuration Files
- `.env.example`: Template for all required API keys
- `config.yaml`: Runtime settings (wake word, TTS config, news sources, LLM params)
- `requirements.txt`: All dependencies including langchain-google-genai, chromadb, pvporcupine, etc.

### Project Structure
```
heathcliff/
├── core/          # Foundation components (MemoryManager, AudioHandler)
├── tools/         # External API integrations (to be implemented)
├── ui/            # Streamlit dashboard (to be implemented)
├── config/        # Configuration loader
├── utils/         # Shared utilities (empty, populate as needed)
└── plan/          # Project planning docs
```

### LangChain Integration Notes
- Using Gemini Flash 2.5 via `langchain-google-genai`
- LangGraph StateGraph for agent orchestration (to be implemented in Phase 2)
- React agent pattern requires tool descriptions to be clear and specific
- Prompt template needs input variables: `input`, `context`, `history`
- Voice thread callback should handle requests concurrently (may need queuing)
- Tool implementations should validate inputs at system boundaries only

---

## Known Issues & Workarounds

### Phase 1 Implementation
- **RESOLVED**: Conversation history persistence - now using ChromaDB with `save_chat()` method
- Voice listener in separate thread - concurrent request handling not fully tested
- Gmail, Calendar, Spotify rate limits - need backoff/retry logic implementation
- Porcupine free tier false positive rate affects UX - consider paid tier for production
- API errors silent during voice interaction - difficult to debug without comprehensive logging
- PyAudio setup is platform-dependent - test on target systems early (requires `sudo apt install python3-pyaudio` on Linux)

### Dependencies Installation Notes
- Install system dependencies first: `sudo apt install python3-pyaudio` (Linux)
- ChromaDB persistence directory defaults to `./chroma_db`
- Porcupine access key optional for free tier (pass to AudioHandler constructor for paid tier)

---

## Discovered API Patterns

### ChromaDB
- Collections auto-create on first access with `get_or_create_collection()`
- Metadata filtering with `where` parameter in queries
- IDs must be unique strings (using UUID + prefixes: `mem_`, `doc_`, etc.)
- Embeddings generated automatically from document text
- Query returns dict with `documents`, `metadatas`, `distances`, `ids` keys

### Configuration Loading
- `get_config()` returns singleton instance to avoid re-reading files
- YAML config accessible via dot notation: `config.get('llm.temperature')`
- Environment variables override YAML when both present
- Validation with `config.validate()` checks required API keys

### Audio Processing
- PyAudio stream must match Porcupine sample rate (16000 Hz)
- Wake word detection processes frame chunks (512 samples)
- `adjust_for_ambient_noise()` essential before STT to reduce errors
- TTS engine properties set once during initialization for performance

---

## Recent Agent Activity

- **2025-12-13**: **Phase 1 Foundation Complete** (3/3 tasks)
  - Implemented project structure and config management system
  - Created ChromaDB memory manager with 3 collections
  - Built complete audio handler with wake word, STT, TTS
  - Updated all requirements and configuration files
  - Updated CLAUDE.md with Gemini integration, task tracking, code org standards
  - Created comprehensive TODO.md with all project tasks
  - Files created:
    - `config/config_loader.py` - centralized configuration
    - `core/memory_manager.py` - ChromaDB integration
    - `core/audio_handler.py` - voice I/O pipeline
    - `.env.example`, `config.yaml` - configuration templates
    - Updated `requirements.txt` with all dependencies
  - Ready for Phase 2: Gemini Agent Core with LangGraph
