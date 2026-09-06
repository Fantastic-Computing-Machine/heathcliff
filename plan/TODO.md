# Heathcliff Project TODO

## Project Status: 🟡 In Progress

---

## Spotify Playback Safety and Inline State (2026-08-23)

- [x] Prevent explicit device requests from falling back to another Spotify device
- [x] Replace natural-language field parsing with typed title, artist, and device tool arguments
- [x] Add true queue resume without searching for a new track
- [x] Reject unrelated zero-confidence Spotify search results before playback
- [x] Show verified cover, track, artist, album, status, and device inline after music turns
- [x] Add regressions and verify all 349 tests

## Langfuse Trace Integrity (2026-08-16)

- [x] Replace the shared Langfuse callback handler with a request-scoped handler
- [x] Keep one Heathcliff root trace and add explicit planner, repair, aggregate, and specialist-agent observations
- [x] Propagate tracing callbacks into every nested specialist LangGraph agent so model calls and actual tool request/results are captured
- [x] Verify the live trace hierarchy with a non-mutating Langfuse smoke run
- [x] Non-destructively label the 48 pre-repair orphaned specialist roots as legacy telemetry in Langfuse

## Type Checking Polish (2026-08-15)

- [x] Replace deprecated `@contextmanager` `Iterator` return annotation with `Generator`
- [x] Verify Ruff and all tests; remaining `ty` diagnostics are unrelated existing issues

## Streamlit Control Panel (2026-08-15)

- [x] Replace disconnected page scripts with one shared `st.navigation` shell and focused views
- [x] Add process-local runtime profiles with revisioned agent replacement and enabled-capability filtering
- [x] Add Command Center streaming timeline, approval controls, and Langfuse trace links
- [x] Restore the weather-aware Heathcliff greeting in new Command Center conversations
- [x] Show active memories by default with 50-item pagination and full-width inspection controls
- [x] Add detailed, full-width analytics sourced from persistent execution events
- [x] Store intermediary coordinator events with completed conversation turns and show them in transcript accordions
- [x] Keep the Streamlit UI deliberately default-styled for a future Next.js migration
- [x] Cover runtime profiles, event persistence, analytics, and focused Streamlit rendering with tests
- [x] Register callable view renderers with unique Streamlit routes so the control panel does not render blank pages

## Next.js UI replacement (2026-09-05)

- [x] Create the initial responsive chat shell with shadcn Sidebar 01 and AI Elements PromptInput
- [x] Add the Runtime V2 same-origin chat gateway and streamed text rendering
- [ ] Add server-side authentication
- [x] Add Memory and Settings routes to the shared application shell
- [ ] Connect live conversation history, Memory, and Settings
- [ ] Retire Streamlit only after replacement acceptance checks pass

## Spotify Reliability (2026-08-23)

- [x] Move Spotify authorization out of task-worker terminal input and into Agent Controls
- [x] Keep Spotify playback in-process so timed-out tasks cannot act later
- [x] Prevent the planner from selecting Spotify without an explicit playback request
- [x] Search Spotify's public catalogue before selecting open-ended music
- [x] Play public playlists and set an explicitly requested Spotify volume
- [x] Ignore unavailable Spotify API entries and keep public music requests in one specialist loop
- [x] Gracefully report Spotify devices that disallow remote volume control

## Failed Action-Chain Hardening (2026-08-15)

- [x] Treat explicit specialist failures as failed task results so dependent actions do not run
- [x] Keep approved Gmail, Calendar, and communications actions in-process rather than allowing an unkillable timeout worker to continue
- [x] Replace Gmail search's brittle `raw` message parser with the API's `full` payload
- [x] Default specialist agents to the lower-quota model, with an explicit `TOOL_MODEL` override

---

## Repository Audit and Action Chains (2026-08-02)

- [x] Feed retrieved conversation history into coordinator planning for follow-up clarification turns
- [x] Add coordinator regression for research → contacts → email dependency output flow
- [x] Await Telegram API sends before reporting success
- [x] Keep text-only startup independent of optional voice imports
- [x] Fix stale Memories page import after the `db/` persistence migration
- [x] Align repository validation instructions with the installed Ruff toolchain
- [x] Enforce approval policy for delegated email, calendar, and messaging mutations
- [x] Replace callback-only approvals with LangGraph interrupt/checkpointer resume
- [ ] Add parallel execution when a measured workload requires it (the coordinator is intentionally sequential today)
- [x] Restore and document optional voice dependencies (`pvporcupine`, `pyaudio`, `pyttsx3`)
- [ ] Clear remaining `uvx ty check` diagnostics, prioritizing `db/conversation_manager.py`

## Ponytail Simplification (2026-08-14)

- [x] Remove unused document index, nonlocal adapters, legacy callback approvals, and duplicate voice entry point
- [x] Collapse coordinator dispatch/quality/retry nodes into plan → execute → aggregate
- [x] Remove unused supervisor tool/middleware wiring and centralise simple subagent response handling
- [x] Replace manual CLI parsing; preserve the full weather-aware butler greeting system
- [x] Default the CLI to text mode while keeping `--voice` opt-in
- [x] Resume pending approval actions from the CLI (`approve` / `reject`)
- [x] Play named Spotify playlists without falling back to a track search
- [x] Replace flaky Wikipedia package queries with Wikimedia REST search
- [x] Consolidate model configuration on provider-neutral `AI_KEY`
- [x] Run full formatter, type check, and test suite (`312 passed`; `ty` has 34 pre-existing diagnostics)

## Live Integration Testing (2026-08-15)

- [x] Add an opt-in text-mode runner that writes per-query coordinator events, responses, approvals, and specialist tool traces to JSONL
- [x] Run a bounded real-service pass (weather, news, research, Gmail, Calendar, Contacts, and Spotify); preserve its local JSONL artifact
- [ ] Fix Google Calendar read-event parsing (`Expecting property name enclosed in double quotes` from `langchain_google_community`)
- [ ] Make timed-out specialist calls stop cooperatively instead of continuing in the background
- [ ] Complete Spotify's interactive OAuth refresh before headless/live playback checks
- [ ] Rerun the full JSONL suite after Gemini free-tier quota resets; include rejection-only draft and calendar approval cases

## Research Quality (2026-08-15)

- [x] Remove all keyword/phrase-based research routing; one semantic info agent decides depth from the request
- [x] Keep source-driven research standards without hiding tools based on hard-coded triggers
- [x] Add optional Tavily search and extraction through the official LangChain integration

## Semantic Decisions (2026-08-15)

- [x] Remove keyword capability matching from planner fallback routing
- [x] Replace delegated-action approval regexes with exact tool/agent identity policy

## Outbound Identity (2026-08-15)

- [x] Add the master-name Heathcliff signature and autonomous-system disclaimer to Gmail drafts, Gmail sends, and Telegram messages
- [x] Render Gmail drafts and sends as clean HTML (headings, bullets, paragraphs, emphasis, links, and signature)
- [x] Apply Heathcliff's navy-and-gold branded card layout to all outgoing Gmail drafts and sends

## Email Resend Reliability (2026-08-16)

- [x] Read full Gmail HTML bodies when the email agent retrieves a previously sent message
- [x] Prevent resends from using Gmail's truncated search snippets
- [x] Require a fresh recipient-specific greeting when sending a prior message to somebody else

---

## Coordinator Stability Remediation (2026-05-12)

- [x] Fix dependency-chain execution crash path (invalid `TaskSpec.parallelizable` usage removed)
- [x] Enforce dependency validity + cycle failure semantics (`DEPENDENCY_FAILED`)
- [x] Restore callback bridge parity for coordinator subtask execution
- [x] Map approval denial to `APPROVAL_REJECTED`
- [x] Enforce task-count, parallel-cap, per-task timeout, and max-runtime coordinator budgets
- [x] Add strict planner schema validation with one repair pass and fallback
- [x] Finalize stream completion payload to `agents_used` / `agent_count`
- [x] Update `ui/Home.py` to consume `agents_used`
- [x] Add structured per-task telemetry logging (`status`, `error_type`, `latency_ms`)
- [x] Add coordinator stability regression tests and streaming contract tests

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
- [x] Replace custom Wikipedia parsing with LangChain `WikipediaQueryRun`
- [x] Add `wikidata_search()` tool (LangChain Wikidata integration)
- [x] Add `stackexchange_search()` tool (LangChain StackExchange integration)
- [x] Add NASA toolkit-backed tools (`nasa_media_search`, `nasa_media_manifest`, `nasa_media_metadata`, `nasa_video_captions`)
- [x] Implement `read_website()` tool (webpage content extraction)
- [x] Add Yahoo Finance news tool integration
- [x] Add YouTube search tool integration
- [x] Add `recent_context()` tool for recency-grounded answers
- [x] Move `recent_context()` into dedicated module (`core/subagents/info/recent_context.py`)
- [x] Upgrade `recent_context` to JSON-backed persistent store (TTL, max items, atomic writes, thread lock, auto-path)
- [x] Add `RecentContextConfig` to config with 5 env-var-backed params
- [x] Add adaptive fast/deep info routing with per-mode recursion limits and graceful recursion-loop fallback
- [x] Write 22-test suite for `recent_context` (`tests/test_recent_context.py`)
- [x] Always include `recent_context` in LLM tool-selector middleware
- [x] Fix all test suites for middleware mock + updated tool counts (184 passed)
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

### Prompt Optimization (Latency Reduction)

- [x] Phase 0: Emergency hardening — tool alias middleware + info param compat + prompt regression tests
- [x] Phase 1: System prompt consolidation — XML-delimited sections, few-shot routing examples, positive-only enforcement
- [x] Phase 2: Tool description normalization — standardized all 9 tool docstrings to `Use for:` / `Provide:` / `Returns:` / `Example:` template
- [x] Phase 3: Subagent prompt slimming — reduced 6 subagent prompts from ~30–90 to ~8–15 lines, removed verbose Reasoning blocks
- [x] Phase 4: Test updates — added `TestToolDescriptionConsistency` (11 tests), 7 XML tag tests, updated existing tests (237 passed)

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
- ✅ Prompt optimization (XML system prompt, tool description normalization, subagent prompt slimming, middleware alias rewriting — 237 tests passing)

### In Progress

- 🔄 Integration testing & error recovery
- 🔄 Runtime V2 cutover: validate provider/OAuth integrations and compare the
  new native Langfuse traces with legacy callbacks before enabling broad client
  traffic. PostgreSQL/S3 remains the portable multi-host alternative.

### Blocked

- ⛔

### Questions/Decisions Needed

-

---

**Last Updated**: 2026-08-29
