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
- **Config Management**: Singleton `Config` instance exported from `config/__init__.py`
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
- `config/config.py`: Runtime settings (wake word, TTS config, news sources, LLM params)
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
- **CRITICAL API UPDATE (Dec 2025)**: Modern LangChain uses `langchain.agents.create_agent` NOT deprecated `AgentExecutor`
  - Import: `from langchain.agents import create_agent` (NOT `from langgraph.prebuilt import create_react_agent`)
  - Parameters: `create_agent(model, tools, system_prompt=...)` where `system_prompt` can be SystemMessage or str
  - Returns: `CompiledStateGraph` (LangGraph graph)
  - Invocation: `graph.invoke({"messages": [HumanMessage(content="...")]})` NOT `executor.invoke({"input": "..."})`
  - Response extraction: `result["messages"][-1].content` (may be structured format from Gemini)
  - **Gemini Response Format**: Content may be `[{'type': 'text', 'text': '...'}]` - extract text parts before saving
- React agent pattern requires tool descriptions to be clear and specific
- Voice thread callback should handle requests concurrently (may need queuing)
- Tool implementations should validate inputs at system boundaries only
- Tool registry prefers LangChain community toolkits (Gmail/Calendar/Search) when available, with config toggle `tools.prefer_langchain_toolkits` and Google Custom Search helpers.
- Credentials fetched via `utils/google_auth.get_google_credentials()` are cached per scope/token tuple to avoid repeated disk reads; they still refresh automatically when expired.
- `tools/__init__.py` lazy-loads tool modules via `__getattr__`, keeping startup lighter while still favoring LangChain toolkits, with fallbacks to custom implementations.
- Gmail and Calendar integrations now rely exclusively on LangChain's community toolkits; the older bespoke `email_tool.py` and `calendar_tool.py` modules were removed to prevent duplicate behavior.

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
- `Config` is a singleton instance to avoid re-reading files
- Class-based config values are accessed via attributes (example: `config.TEMPERATURE`)
- Environment variables override config defaults at load time
- Validation with `config.validate()` checks required API keys

### Audio Processing
- PyAudio stream must match Porcupine sample rate (16000 Hz)
- Wake word detection processes frame chunks (512 samples)
- `adjust_for_ambient_noise()` essential before STT to reduce errors
- TTS engine properties set once during initialization for performance

---

## Recent Agent Activity

- **2025-12-28**: **Mem0 Chroma Config Fix** ✅
  - Mem0 Chroma config expects `path` (or host/port) for local usage; `persist_directory` fails validation.
  - Avoid passing a `chromadb.CloudClient` object in Mem0 config; provide `api_key` + `tenant` (+ host/port) instead.

- **2025-12-28**: **Config Attribute Migration Cleanup** ✅
  - Updated runtime usages to rely on class-based config attributes instead of `config.get`.
  - Simplified middleware to a no-op stack for the new config layout.
  - Aligned audio, greetings, Langfuse, settings UI, and test mocks with attribute access.

- **2025-12-15**: **LangChain Agent Refactoring - Phase 1 Complete** ✅
  - **Goal**: Migrate from custom LangGraph StateGraph to modern LangChain native agent framework
  - **Key Discovery**: LangChain 1.1.3 uses `langchain.agents.create_agent` (NOT deprecated `AgentExecutor` or `create_react_agent` from langgraph.prebuilt)
  - **Changes Made**:
    - Removed ~468 lines of custom orchestration code from `core/agent_core.py`
    - Replaced custom nodes (\_retrieval_node, \_reasoning_node, \_tool_calling_node, \_output_node) with `create_agent` call
    - Updated `_build_prompt_template()` to return SystemMessage (not ChatPromptTemplate)
    - Updated `_build_agent()` to use `create_agent(model, tools, system_prompt=...)`
    - Rewrote `invoke()` to use `graph.invoke({"messages": [...]})` format
    - Rewrote `stream_invoke()` to parse LangGraph's message-based streaming
    - Added Gemini structured response parsing: `[{'type': 'text', 'text': '...'}]` → extract text
    - Fixed `core/__init__.py` to remove deleted `AgentState` export
  - **Testing**: Both tool-based (weather query) and non-tool queries working correctly
  - **Benefits**:
    - Code reduction: 973 lines → 492 lines (~49% reduction)
    - Simpler, more maintainable code using official LangChain patterns
    - Built-in iteration control and error handling from LangGraph
    - Ready for Phase 2: Human-in-the-loop callback integration
  - **Next**: Phase 2 - Implement StreamlitApprovalHandler for human approval on sensitive operations

- **2025-12-15**: **Human-in-the-Loop Approval System - Phase 2 Complete** ✅
  - **Goal**: Implement generalized approval system for sensitive operations using LangChain callbacks
  - **Key Discovery**: `langchain_community.callbacks.human.HumanApprovalCallbackHandler` provides `on_tool_start` hook for intercepting tool execution
  - **Changes Made**:
    - Created `core/approval_handler.py` (~200 lines):
      - `StreamlitApprovalHandler` extends `HumanApprovalCallbackHandler`
      - Defines `SENSITIVE_TOOLS` set (send_email, create_event, cancel_event, send_to_telegram, etc.)
      - Integrates with Streamlit session state for approval workflow
      - Provides helper functions: `is_approval_pending()`, `approve_request()`, `reject_request()`, `clear_approval()`
    - Updated `core/agent_core.py`:
      - Added `additional_callbacks` parameter to both `invoke()` and `stream_invoke()`
      - Merged callbacks (Langfuse + approval handler) before graph execution
      - Pass callbacks via `config={"callbacks": all_callbacks}` to graph
    - Updated `ui/Home.py`:
      - Added approval UI with Approve/Modify/Reject buttons
      - Instantiates `StreamlitApprovalHandler(st.session_state)` and passes to agent
      - Shows tool details (name, args) before execution
      - Supports inline modification of tool arguments
    - Updated `tools/gmail_tools.py`:
      - Changed from `GmailCreateDraft` to `GmailSendMessage` (actually sends email)
      - Simplified docstring - callback handles approval, not drafts
      - Reduced safety instructions from ~24 lines to ~5 lines
    - Updated `instructions/prompts.py`:
      - Removed "🚨 CRITICAL EMAIL RULES" section (~24 lines)
      - Simplified to "⚠️ EMAIL SAFETY" (~5 lines)
      - Updated send_email description: "Creates draft email (user sends manually)" → "Sends email (requires user approval)"
  - **Testing**: Test script passed - agent invoke/stream working correctly with callback architecture
  - **Benefits**:
    - Generalized approval works for ANY tool, not just Gmail
    - User can approve, modify args, or reject in-app
    - Centralized approval logic - no per-tool workarounds
    - Cleaner prompts - callback handles approval workflow
    - Significantly reduced custom safety code
  - **Next**: Phase 2.5 - Test email approval workflow end-to-end in Streamlit UI with real Gmail sending

- **2025-12-15**: **Middleware Framework & Critical Bug Fix** ✅
  - **Goal**: Add execution control middleware (tool selector, rate limiting, summarization) and fix response extraction bug
  - **Key Discovery**: LangChain middleware (designed for deprecated AgentExecutor) are NOT compatible with create_agent's LangGraph architecture
  - **Changes Made**:
    - Created `core/middleware.py` (~230 lines):
      - `create_middleware_stack()` function with 6 middleware types
      - LLMToolSelectorMiddleware - intelligently filter tools before model call
      - ModelCallLimitMiddleware - prevent infinite loops (run_limit, thread_limit)
      - ToolCallLimitMiddleware - control tool execution with per-tool limits
      - ToolRetryMiddleware - exponential backoff for transient failures
      - SummarizationMiddleware - manage conversation history when approaching token limits
      - ModerationMiddleware - placeholder for custom content filtering
    - Updated `config/config.py`:
      - Added comprehensive middleware configuration section (lines 130-182)
      - Documented all settings with inline comments
      - Set all middleware `enabled: false` with NOTE explaining LangGraph incompatibility
    - Fixed type error in `core/agent_core.py`:
      - Changed `_build_agent()` to pass `model=self.llm` instead of `self.llm_with_tools`
      - create_agent expects BaseChatModel, not Runnable with bound tools
    - **CRITICAL BUG FIX**: Fixed response extraction in `stream_invoke()` (line 497-499):
      - **Bug**: Condition `if last_msg.type == "ai" and not hasattr(last_msg, "tool_calls")` failed when AI messages had empty tool_calls list
      - **Symptom**: LangGraph execution successful but UI showed "I encountered an error processing your request"
      - **Fix**: Changed condition to `if last_msg.type == "ai" and (not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls)`
      - Now correctly extracts AI responses with empty or missing tool_calls attribute
      - Added debug logging: `logger.debug(f"Stream chunk - msg type: {last_msg.type}, has tool_calls: {hasattr(last_msg, 'tool_calls')}, tool_calls value: {getattr(last_msg, 'tool_calls', None)}")`
  - **Middleware Incompatibility Details**:
    - LangChain middleware objects lack required callback interface attributes (raise_error, ignore_chain)
    - LangGraph callback manager expects different interface than AgentExecutor
    - Error: `AttributeError: 'LLMToolSelectorMiddleware' object has no attribute 'raise_error'`
    - Solution: Disabled all middleware in config/config.py until LangGraph-compatible integration method found
  - **Testing**: Agent working correctly after bug fix - responses now properly extracted and displayed
  - **Benefits**:
    - Middleware framework ready for future integration (when LangGraph compatibility available)
    - Response extraction bug fixed - UI now displays agent responses correctly
    - Comprehensive middleware configuration documented in config/config.py
  - **Next**: Test response extraction fix in Streamlit UI, then proceed with Phase 2.5 email approval testing

- **2025-12-15**: **Critical Tool Selection Fix** ✅
  - **Issue**: Agent was picking tools for previous queries instead of current query
  - **Root Cause Analysis**:
    - Agent was receiving 5 recent chat messages (`n=5` in `get_recent_chats()`)
    - This created cumulative context where old queries influenced tool selection
    - Example: User asks "list company names" but agent still saw "play song" context
    - Debug logs showed agent confused by multiple queries in history
  - **Changes Made** in [core/agent_core.py](core/agent_core.py):
    1. **Reduced Context Window** (lines 282-286, 422-426):
       - Changed from `n=5` to `n=2` in both `invoke()` and `stream_invoke()` methods
       - Only retrieves last 1 turn (user query + assistant response)
       - Prevents context bleed between unrelated queries
    2. **Added Focus Instruction** (lines 244-253):
       - Added system message in `_format_chat_history()` after context messages
       - Explicitly tells agent: "Focus ONLY on the user's CURRENT query"
       - Instructs agent to NOT use tools based on previous queries
  - **Testing**: Restarted Streamlit at 16:47 to apply changes
  - **Expected Result**: Agent should now correctly select tools for current query only
  - **Trade-offs**:
    - Reduced context window (n=2) means less conversation history available
    - May lose some long-term context benefits
    - Could implement conversation summarization later if needed
  - **Next**: User should test with multi-turn conversations to verify fix works

- **2025-12-13**: **Phase 1 Foundation Complete** (3/3 tasks)
  - Implemented project structure and config management system
  - Created ChromaDB memory manager with 3 collections
  - Built complete audio handler with wake word, STT, TTS
  - Updated all requirements and configuration files
  - Updated CLAUDE.md with Gemini integration, task tracking, code org standards
  - Created comprehensive TODO.md with all project tasks
  - Files created:
    - `config/config.py` / `config/__init__.py` - centralized configuration
    - `core/memory_manager.py` - ChromaDB integration
    - `core/audio_handler.py` - voice I/O pipeline
    - `.env.example`, `config/config.py` - configuration templates
    - Updated `requirements.txt` with all dependencies
  - Ready for Phase 2: Gemini Agent Core with LangGraph

- **2025-12-13**: **Phase 2 Core Agent** (Pre-existing)
  - `core/agent_core.py` already implemented with LangGraph StateGraph
  - HeathcliffAgent class with 4-node graph: retrieval → reasoning → tool_node/output_node
  - Gemini Flash 2.5 integration via langchain-google-genai
  - Multi-turn conversation support with session IDs

- **2025-12-13**: **Phase 3 Tools Integration Complete** (8/8 tasks)
  - Created centralized OAuth manager (`utils/google_auth.py`) with token caching
  - Implemented all tool modules:
    - `tools/email_tool.py` - Gmail read/send/search
    - `tools/calendar_tool.py` - Google Calendar read/write (Heathcliff's calendar)
    - `tools/spotify_tool.py` - Spotify playback control
    - `tools/info_tools.py` - Weather (OpenWeatherMap), News (NewsAPI), Web search (DuckDuckGo + Wikipedia)
    - `tools/comm_tools.py` - Telegram notifications, Google Drive file reading
  - Created `tools/__init__.py` with lazy loading and deduplication
  - Renamed all "JARVIS" references to "Heathcliff" throughout codebase

- **2025-12-13**: **Tooling Enhancements**
  - Added config-driven LangChain toolkit preferences plus Google Custom Search API support
  - Introduced deduplicating tool registry (`tools/__init__.py`) that merges LangChain Gmail/Calendar toolkits with custom fallbacks
  - Upgraded `tools/info_tools.search_web` to leverage Google/DuckDuckGo LangChain integrations with automatic fallback to Wikipedia summaries
  - Cached Google OAuth credentials in-memory to avoid repeated disk reads per tool call
  - Made `tools/__init__.py` lazy-load tool modules via `__getattr__`, trimming startup costs while retaining toolkit prioritization
  - Deferred heavy LangChain search imports until runtime
  - Removed redundant `tools/email_tool.py` / `tools/calendar_tool.py` in favor of LangChain Gmail/Calendar toolkits

- **2025-12-14**: **Langfuse Observability Integration**
  - Introduced `utils/langfuse_client.py` to centralize Langfuse client + callback creation and safe logging helpers.
  - HeathcliffAgent now registers the Langfuse LangChain callback handler, streams LangGraph executions with `graph.invoke(..., callbacks=...)`, and reports tool events + final responses through Langfuse traces/events.
  - Added configuration knobs (`observability.langfuse.*`) plus `.env` entries (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, etc.) and documented setup in README/SETUP.
  - Requirements now include `langfuse==2.28.1`; remember to add keys or disable observability via `config/config.py` if running without Langfuse.

- **2025-12-13**: **Phase 4 UI & Integration Complete** (3/3 tasks)
  - Created `main.py` - Main orchestrator with voice mode and text mode
    - Voice mode: wake word → listen → agent → speak response
    - Text mode: terminal-based chat for testing without audio hardware
    - Handles graceful shutdown and error recovery
  - Built Streamlit multipage dashboard:
    - `ui/Home.py` - Main chat interface with agent integration
    - `ui/pages/1_🧠_Memories.py` - Memory management (search, view, add)
    - `ui/pages/2_📊_Analytics.py` - Usage statistics and insights
    - `ui/pages/3_⚙️_Settings.py` - Configuration and API status viewer
  - Created `SETUP.md` - Comprehensive setup guide with:
    - System dependencies for Linux/macOS/Windows
    - API credential setup for all services (Google Cloud, Gemini, Spotify, etc.)
    - Configuration instructions
    - Troubleshooting section
  - Updated `README.md` with quick start guide and updated status

## Project Completion Status

**Phase 1-4 Complete** ✅ **Ready for Production Testing**

- ✅ **Phase 1**: Foundation Setup (Config, Memory, Audio)
- ✅ **Phase 2**: Core Agent (LangGraph + Gemini Flash 2.5)
- ✅ **Phase 3**: Tools Integration (All 8 tools implemented)
- ✅ **Phase 4**: UI & Integration (Voice mode, Text mode, Streamlit dashboard)
- ⏳ **Phase 5**: Testing & Polish (Pending)

**Next Steps:**

- Integration testing (wake word → response flow, tool calling, memory recall)
- Multi-turn conversation testing
- Error recovery testing
- Documentation finalization
- Optional: Docker containerization, systemd service setup

---

## Recent Agent Activity (Cont.)

- **2025-12-28**: **Mem0 Migration Planning** 📝
  - Goal: Replace custom memory extraction with Mem0 OSS using Gemini + Chroma Cloud.
  - Decision: Single-user deployment; no per-user isolation needed.
  - Decision: OK to use Mem0 SQLite history as ephemeral (reset on redeploy).
  - Preference: Use REST server if simplest; disable `/docs` if possible.

- **2025-12-28**: **Mem0 REST Implementation (In-Repo)** ✅
  - Replaced REST server with Mem0 SDK in-process.
  - Configured Gemini LLM + Gemini embeddings + Chroma Cloud via `config/config.py`.
  - Heathcliff now uses Mem0 SDK for memory add/search while chat/docs stay in Chroma Cloud.
  - Dockerfile + docker-compose now run `heathcliff` only.
