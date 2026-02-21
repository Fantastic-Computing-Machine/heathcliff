# Shared Agent Memory & Discovery

This file serves as the **working memory** for all coding agents on the Heathcliff project. It tracks discoveries, issues, and recent activity. For complete project documentation, see `AGENTS.md`.

## How Agents Use This File

- Before starting work: Check this file for ongoing issues, recent discoveries, and previous agent findings
- After completing work: Update this file with new issues, workarounds, code patterns discovered, and activity log
- Reference `AGENTS.md` for project overview, architecture, configuration, and development standards
- Cost optimization: Reuse previous agent discoveries instead of re-investigating
- Share API integration workarounds and debugging strategies discovered during work

---

## Current Architecture (as of 2026-02-20)

### Project Structure

```txt
heathcliff/
├── core/
│   ├── agent_core.py       # Singleton HeathcliffAgent supervisor
│   ├── approval_handler.py # Human-in-the-loop approval (StreamlitApprovalHandler)
│   ├── audio_handler.py    # STT/TTS/wake word pipeline
│   ├── memory_manager.py   # ChromaDB (memories, chats, my_data)
│   ├── middleware.py        # Middleware framework (disabled - LangGraph incompatible)
│   └── subagents/           # Domain-specific subagents
│       ├── __init__.py      # Registry: get_all_subagent_tools()
│       ├── calendar/        # Google Calendar tools
│       ├── comms/           # Telegram, Google Drive
│       ├── contacts/        # Contact management
│       ├── email/           # Gmail tools
│       ├── info/            # Weather, News, Web search
│       └── music/           # Spotify playback
├── skills/                  # Dynamic skills loaded at runtime
│   ├── skill_tools.py       # get_skill_tools()
│   ├── skills.py            # Skill loading/management
│   └── master_info.py       # Master information
├── assets/                  # 3D Blob UI (standalone web frontend)
│   ├── index.html/style.css # Warm lavender palette, glass-morphism
│   ├── blob.js              # GPU simplex noise vertex shader, 4 states
│   ├── chat.js              # Chat overlay, /api/chat POST
│   └── server.py            # FastAPI server (static + chat API → HeathcliffAgent)
├── config/                  # Singleton Config (class-based attribute access)
├── ui/                      # Streamlit dashboard (Home, Memories, Analytics, Settings)
├── utils/
│   ├── google_auth.py       # OAuth manager with token caching
│   └── langfuse_client.py   # Langfuse observability client
├── voice/                   # Wake-word voice entry point
├── instructions/            # System prompt templates (prompts.py)
├── main.py                  # Entry point (voice/text mode)
└── pyproject.toml           # Python >=3.11, uv-managed deps
```

### Key Patterns

- **Agent**: `HeathcliffAgent` is a singleton (`.instance()` or constructor). Self-wires tools via `_assemble_default_tools()` (subagents + skills). Extensible with `extra_tools`.
- **LLM**: Gemini Flash 2.5 via `langchain-google-genai`. Agent built with `langchain.agents.create_agent(model, tools, system_prompt=...)` returning a `CompiledStateGraph`.
- **Invocation**: `graph.invoke({"messages": [HumanMessage(...)]})` → response at `result["messages"][-1].content`.
- **Gemini Response Format**: Content may be `[{'type': 'text', 'text': '...'}]` — extract text parts before saving.
- **Memory**: Mem0 SDK in-process for memory add/search; ChromaDB for chat/docs. Chroma Cloud backend.
- **Credentials**: `utils/google_auth.get_google_credentials()` — cached per scope/token tuple.
- **Approval**: `StreamlitApprovalHandler` intercepts `SENSITIVE_TOOLS` (send_email, create_event, etc.) via `on_tool_start` hook. Approve/Modify/Reject in Streamlit UI.
- **Middleware**: Framework exists in `core/middleware.py` (6 types) but **all disabled** — LangGraph callback interface incompatible with LangChain middleware.
- **Context Window**: Chat history limited to `n=2` (1 turn) in invoke/stream_invoke to prevent tool selection bleed between queries.

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

- **2026-02-20**: **Architecture Refactor — Subagents & Singleton Supervisor** ✅
  - Removed old `tools/` and `core/sub_agents/`. New `core/subagents/` with 6 domains (calendar, comms, contacts, email, info, music), each with `tools.py` + `agent.py`.
  - HeathcliffAgent singleton with self-wiring. Skills framework in `skills/`.
  - 101 tests passing.

- **2026-02-20**: **3D Blob UI — End-to-End Verified** ✅
  - Standalone web frontend in `assets/` (FastAPI + Three.js). GPU simplex noise blob, 4 animation states, chat API bridging to HeathcliffAgent.
  - Run: `uv run python assets/server.py` → <http://localhost:8600>

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

**Phases 1–4 Complete** ✅ — **Phase 5 (Testing & Polish) Pending** ⏳

Next steps: Integration testing, multi-turn conversation testing, error recovery, Docker containerization.
