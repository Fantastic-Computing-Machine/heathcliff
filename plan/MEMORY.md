# Shared Agent Memory & Discovery

This file serves as the **working memory** for all coding agents on the Heathcliff project. It tracks discoveries, issues, and recent activity. For complete project documentation, see `AGENTS.md`.

## How Agents Use This File

- Before starting work: Check this file for ongoing issues, recent discoveries, and previous agent findings
- After completing work: Update this file with new issues, workarounds, code patterns discovered, and activity log
- Reference `AGENTS.md` for project overview, architecture, configuration, and development standards
- Cost optimization: Reuse previous agent discoveries instead of re-investigating
- Share API integration workarounds and debugging strategies discovered during work

---

## Current Architecture (as of 2026-02-23)

### Project Structure

```txt
heathcliff/
├── core/
│   ├── __init__.py          # Exports MemoryManager, HeathcliffAgent, AudioHandler
│   ├── agent_core.py        # Singleton HeathcliffAgent supervisor
│   ├── approval_handler.py  # Human-in-the-loop approval (StreamlitApprovalHandler)
│   ├── audio_handler.py     # STT/TTS/wake word pipeline
│   ├── memory_manager.py    # ChromaDB (chats, my_data) + Mem0 (long-term memories)
│   ├── middleware.py         # Middleware framework (disabled — LangGraph incompatible)
│   └── subagents/            # Domain-specific subagents
│       ├── __init__.py       # Registry: get_all_subagent_tools()
│       ├── calendar/         # Google Calendar tools
│       ├── comms/            # Telegram, Google Drive
│       ├── contacts/         # Contact management
│       ├── email/            # Gmail tools
│       ├── info/             # Weather, News, Web search, Wikipedia, Website reader
│       └── music/            # Spotify playback
├── skills/                   # Dynamic skills loaded at runtime
│   ├── __init__.py           # Skills package init
│   ├── skill_tools.py        # get_skill_tools() — LangChain tool wrappers
│   ├── skills.py             # Skill definitions (master_info, british_persona, email_safety)
│   └── master_info.py        # Master profile data loading from TOML
├── assets/                   # 3D Blob UI (standalone web frontend)
│   ├── index.html/style.css  # Warm lavender palette, glassmorphism
│   ├── blob.js               # GPU simplex noise vertex shader, 4 states
│   └── chat.js               # Chat overlay, /api/chat POST
├── config/                   # Singleton Config (class-based attribute access)
│   ├── __init__.py
│   └── config.py             # Inherits: RuntimeConf, MasterConf, ChromaConf, Mem0Conf, etc.
├── ui/                       # Streamlit dashboard + blob FastAPI server
│   ├── Home.py               # Streamlit chat dashboard (background init)
│   ├── server.py             # FastAPI blob server (static files + /api/chat → HeathcliffAgent)
│   └── pages/                # 4 Streamlit pages (Memories, Analytics, Settings, Chat History)
├── utils/
│   ├── __init__.py
│   ├── google_auth.py        # OAuth manager with token caching
│   ├── langfuse_client.py    # Langfuse observability client
│   ├── heathcliff_greetings.py # Greeting utilities
│   ├── errors.py             # Custom error types (AgentMemoryError)
│   └── retry.py              # Retry utilities
├── voice/                    # Wake-word voice entry point
│   └── main.py               # VoiceListener class (Porcupine + PyAudio)
├── instructions/             # System prompt templates (prompts.py, XML delimiters)
│   ├── __init__.py
│   └── prompts.py
├── tests/                    # Pytest test suite (6 test files)
├── logger.py                 # Logging setup and formats
├── main.py                   # Entry point (voice/text mode)
├── master_info.toml          # User profile & preferences (TOML)
└── pyproject.toml            # Python >=3.11, uv-managed deps
```

### Key Patterns

- **Agent**: `HeathcliffAgent` is a singleton (`.instance()` or constructor). Self-wires tools via `_assemble_default_tools()` (subagents + skills). Extensible with `extra_tools`.
- **LLM**: `google_genai:gemini-3-flash-preview` (supervisor) + `google_genai:gemini-2.5-pro` (tool calls) via `langchain-google-genai`. Agent built with `langchain.agents.create_agent(model, tools, system_prompt=...)` returning a `CompiledStateGraph`. Mem0 uses `gemini-2.5-flash-lite` for its LLM and `gemini-embedding-001` for embeddings.
- **Invocation**: `graph.invoke({"messages": [HumanMessage(...)]})` → response at `result["messages"][-1].content`.
- **Gemini Response Format**: Content may be `[{'type': 'text', 'text': '...'}]` — extract text parts before saving.
- **Memory**: Mem0 SDK in-process for memory add/search; ChromaDB for chat/docs. Chroma Cloud backend.
- **Prompt Context Injection (2026-02-22)**: Pair-based semantic/recent chat context is injected as preceding `HumanMessage`/`AIMessage` objects, while Mem0 recall is injected into `USER_PROMPT_TEMPLATE` under `Long-term Memory Context` (not as a `SystemMessage`).
- **Temporal Context Injection (2026-02-22)**: Current date/time metadata (including month/year) is now injected into `USER_PROMPT_TEMPLATE` at invoke-time via `get_current_temporal_context()`.
- **User Prompt XML Delimiters (2026-02-22)**: `USER_PROMPT_TEMPLATE` now wraps long-term memory and current user query in XML tags (`<USER_MEMORY_CONTEXT>`, `<USER_QUERY>`) to improve boundary adherence.
- **Credentials**: `utils/google_auth.get_google_credentials()` — cached per scope/token tuple.
- **Approval**: `StreamlitApprovalHandler` intercepts `SENSITIVE_TOOLS` (send_email, create_event, etc.) via `on_tool_start` hook. Approve/Modify/Reject in Streamlit UI.
- **Middleware**: Framework exists in `core/middleware.py` (6 types) but **all disabled** — LangGraph callback interface incompatible with LangChain middleware.
- **Middleware Tool Selection (2026-02-23)**: `LLMToolSelectorMiddleware` now sets `always_include=["recent_context"]` so recency snippets remain selectable even when other tools are filtered.
- **Context Window**: Retrieval now uses pair-aware history (`build_message_history`) with semantic pairs first and recent chronological pairs next.
- **Info Tooling (2026-02-23)**: `core/subagents/info/tools.py` now includes optional LangChain community wrappers for Yahoo Finance news and YouTube search, plus a `recent_context` tool backed by a small in-memory recency buffer populated from successful info-tool outputs.

### Operational Notes

- Mem0 + Chroma: Use `path` (or host/port) in Chroma config; `persist_directory` fails validation. Do not pass `chromadb.CloudClient`; provide `api_key` + `tenant` instead.
- ChromaDB usage: Collections auto-create via `get_or_create_collection()`. IDs must be unique strings (use `mem_*`/`doc_*`). Query result keys: `documents`, `metadatas`, `distances`, `ids`; filter with `where`.
- Audio: PyAudio stream must be 16000 Hz; wake word frames are 512 samples. Call `adjust_for_ambient_noise()` before STT; set TTS engine properties once at init.
- Dependencies: On Linux install `python3-pyaudio` first (`sudo apt install python3-pyaudio`).

### Known Issues

- Middleware disabled due to LangGraph incompatibility (missing `raise_error`/`ignore_chain` attributes).
- Voice listener concurrency not fully tested.
- Gmail/Calendar/Spotify rate limits need backoff/retry logic.
- PyAudio is platform-dependent — test on target systems early.
- `pyproject.toml` description is still a placeholder.

---

## Timeline (Latest Activity)

- **2026-02-23**: **Tool selector always includes recent context** ✅
  - Updated `core/middleware.py` to define `ALWAYS_INCLUDE_TOOL_NAMES = ["recent_context"]`.
  - Wired `LLMToolSelectorMiddleware(..., always_include=ALWAYS_INCLUDE_TOOL_NAMES)` so `recent_context` is consistently available to the model.

- **2026-02-23**: **Recent context extracted to dedicated module** ✅
  - Moved `_RECENT_SNIPPETS`, `_add_recent_snippet`, `_capture_recent_result`, `RecentContextArgs`, and `recent_context()` into `core/subagents/info/recent_context.py`.
  - Updated `core/subagents/info/tools.py` to import `recent_context` and `_capture_recent_result` from the new module.
  - Kept tool registration behavior unchanged via `get_info_tools()`.

- **2026-02-23**: **Info tool import style adjustment** ✅
  - Moved Yahoo Finance and YouTube tool imports to module scope in `core/subagents/info/tools.py`.
  - Removed local imports inside `finance_news_tool()` and `yt_search_tool()` per code-style request.

- **2026-02-23**: **Info subagent tool expansion (finance + YouTube + recent context)** ✅
  - Added `finance_news_tool()` returning `YahooFinanceNewsTool()` when available.
  - Added `yt_search_tool()` returning `YouTubeSearchTool()` when available.
  - Added recency buffer (`_RECENT_SNIPPETS`) and `recent_context(n)` LangChain tool for short-term grounding.
  - Wired snippet capture into `get_weather`, `get_news`, `search_web`, `wikipedia_search`, and `read_website`.
  - Updated `get_info_tools()` to register `recent_context` plus optional finance/YouTube tools.

- **2026-02-23**: **Generalized user references** ✅
  - Replaced all hardcoded references to specific user names with generic placeholders ("User", "Alex") across the codebase.
  - Updated config files, skill implementations, agent logic, UI elements, README examples, and test cases.
  - Ensures the application is replicable and personalized for any user.

- **2026-02-23**: **Weather API refactored to LangChain wrapper** ✅
  - Replaced direct PyOWM/OpenWeatherMap API calls with `OpenWeatherMapAPIWrapper` from `langchain-community`.
  - Location format requires "City,CountryCode" (e.g., "Paris,FR").

- **2026-02-23**: **Master info constant naming + docs update** ✅
  - Standardized config constant name to `MASTER_INFO_LOC` (replacing `MASTER_INFO_TOML_LOC`) in `config/config.py`.
  - Documented master-profile flow in `README.md` Quick Start and `SETUP.md` Configuration sections.
  - Docs now explicitly note `master_info.toml` as the profile source and point to `MASTER_INFO_LOC` for path overrides.

- **2026-02-23**: **Removed deprecated master-info files** ✅
  - Deleted legacy `config/master_info.py` and deprecated `master_info.json` after TOML migration.
  - Kept `master_info.toml` as the single source of truth for `Config.MASTER_INFO`.
  - Updated `skills/master_info.py` seed-data comment to reference TOML-based config loading.

- **2026-02-23**: **Switched master profile format to TOML** ✅
  - Updated `config/config.py` to load `master_info.toml` with `tomllib` (Python 3.11+), replacing JSON parsing.
  - Preserved graceful startup exit behavior for missing/invalid/empty master profile files.
  - Added a commented `master_info.toml` template with IMPORTANT and OPTIONAL sections for easier user editing.
  - Updated `settings.py` comments to reflect TOML-based source of truth.

- **2026-02-23**: **Annotated master_info template sections** ✅
  - Updated `master_info.json` to label template blocks as IMPORTANT vs OPTIONAL using JSON-safe `_comment_*` keys.
  - Kept file valid JSON (no native comments) and preserved existing schema fields.

- **2026-02-23**: **Master info source migrated to JSON-only** ✅
  - Updated config loading so `Config.MASTER_INFO` initializes from root `master_info.json` at startup.
  - Added graceful startup failure when `master_info.json` is missing, invalid JSON, empty, null, or not a JSON object.
  - Removed seed import dependency on `config/master_info.py` in `config/config.py` (file kept for deprecation compatibility).
  - Created a full `master_info.json` template using the prior `config/master_info.py` schema and added extra profile credential fields from `settings.py` under `master_credentials`.

- **2026-02-22**: **Fixed Chat History page memory API mismatch** ✅
  - Added `MemoryManager.get_all_sessions()` to aggregate session summaries (`session_id`, `start_time`, `msg_count`) from `chat_messages` metadata for `ui/pages/4_💬_Chat_History.py`.
  - Added `MemoryManager.get_session_history(session_id)` to return full session messages sorted chronologically via existing `_sort_key`.
  - Added `MemoryManager.delete_all_chats()` to support global history deletion used by Chat History danger-zone UI.
  - Added unit tests in `tests/test_memory_manager.py` covering normal + error paths for all three methods.
  - Verified with `uv run pytest tests/test_memory_manager.py -v` (39 passed).

- **2026-02-22**: **Streamlit Home non-blocking initialization** ✅
  - Refactored `ui/Home.py` to remove blocking top-level `init_components()` execution that showed Streamlit's `init_components` spinner before UI render.
  - Added background initialization thread (`_initialize_components_background`) with guarded shared state so the chat input renders immediately on page load.
  - Sidebar now shows `Initializing core components in background...` while booting, then switches to Ready metrics when initialization completes.
  - Added `st.fragment(run_every=1)` watcher to auto-rerun once background init finishes/fails so UI status flips to Ready/Error without manual interaction.
  - Chat submission now handles init state gracefully: shows a wait warning during warmup and an explicit failure message if initialization errors.
  - Updated `Reload Agent` behavior to reset background init state and restart initialization asynchronously.

- **2026-02-22**: **User prompt redesign + datetime move** ✅
  - Reformatted `USER_PROMPT_TEMPLATE` in `instructions/prompts.py` using structured sections (`Task`, `Current Date and Time`, `Long-term Memory Context`, `Response Requirements`, `Current User Query`) for stronger prompt adherence.
  - Moved runtime date/time details out of `build_system_prompt()` into the user prompt path.
  - Added `get_current_temporal_context()` and wired it in `core/agent_core.py` for both `invoke()` and `stream_invoke()`.
  - Updated tests in `tests/test_agent_core.py` to assert date/month/year metadata appears in the final user prompt payload.
  - Added XML wrappers around long-term memory and current query blocks in `USER_PROMPT_TEMPLATE`; updated tests to assert the new tags.
  - Updated `instructions/README.md` to match the current context injection design.

- **2026-02-22**: **Prompt injection update for Mem0 recall** ✅
  - Updated `core/agent_core.py` so Mem0 recall is formatted into a dynamic `memories_block` and injected through `USER_PROMPT_TEMPLATE`.
  - Removed long-term memory injection as `SystemMessage`; chat history now contains pair-based messages only before the final user message.
  - Updated `instructions/prompts.py` with explicit sections (`Long-term Memory Context`, `Current User Query`) and aligned context guidance.
  - Added/updated tests in `tests/test_agent_core.py` to assert memory prompt injection and absence of memory `SystemMessage` context.

- **2026-02-21**: **Blob UI cleanup** — Removed dead Streamlit component protocol from `blob.js` (standalone blob has no Streamlit dependency). Updated `MEMORY.md` paths.

- **2026-02-20**: **Architecture Refactor — Subagents & Singleton Supervisor** ✅
  - Removed old `tools/` and `core/sub_agents/`. New `core/subagents/` with 6 domains (calendar, comms, contacts, email, info, music), each with `tools.py` + `agent.py`.
  - HeathcliffAgent singleton with self-wiring. Skills framework in `skills/`.
  - 101 tests passing.

- **2026-02-20**: **3D Blob UI — End-to-End Verified** ✅
  - Standalone web frontend in `assets/` (Three.js). FastAPI server at `ui/server.py`. GPU simplex noise blob, 4 animation states, chat API bridging to HeathcliffAgent.
  - Run: `uv run python ui/server.py` → <http://localhost:8600>

- **2025-12-28**: **Config & Mem0 Cleanup** ✅
  - Mem0 SDK in-process (replaced REST server). Gemini LLM + Gemini embeddings + Chroma Cloud.
  - Config migrated to class-based attribute access; middleware simplified to no-op stack.

- **2025-12-15**: **Agent Modernization & Bug Fixes** ✅
  - Migrated from custom LangGraph StateGraph to `langchain.agents.create_agent`. ~49% code reduction.
  - Human-in-the-loop approval system for sensitive tools.
  - Fixed stream_invoke response extraction (empty tool_calls list bug).
  - Fixed tool selection bleed (reduced context window to n=2).

- **2025-12-14**: **Langfuse Observability** ✅
  - `utils/langfuse_client.py` for traces/events. Callback handler registered in HeathcliffAgent.

- **2025-12-13**: **Phases 1–4 Complete** ✅
  - Foundation (Config, Memory, Audio) → Core Agent (LangGraph + Gemini) → Tools (8 integrations) → UI & Integration (Voice, Text, Streamlit).

---

## Project Status

**Phases 1–4 Complete** ✅ — Subagents architecture refactor, skills framework (3 skills), Mem0 memory, 3D Blob UI, Langfuse observability, master profile TOML migration, user reference generalization, weather API LangChain wrapper all completed. **Phase 5 (Testing & Polish) In Progress** ⏳

Next steps: Integration testing, error recovery, Docker containerization, troubleshooting guide.
