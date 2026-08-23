# Shared Agent Memory & Discovery

This file serves as the **working memory** for all coding agents on the Heathcliff project. It tracks discoveries, issues, and recent activity. For complete project documentation, see `AGENTS.md`.

## 2026-08-23 Streamlit Blank-Page Repair

- The control-panel router had registered each view's source file with `st.Page`, but the files intentionally only define `render()` and therefore displayed no elements when Streamlit executed them. The page could load with an empty body.
- `ui/Home.py` now registers the `render` callables directly and assigns unique URL paths to every non-default page (callables otherwise all infer the conflicting `render` path). The default route renders Command Center. Verified with Streamlit `AppTest` against the real entry point and the full suite (`332 passed`).

## 2026-08-16 Langfuse Trace Integrity

- Root cause of duplicate/disconnected rows: `HeathcliffAgent` opened a manual root observation while a process-global LangChain `CallbackHandler` emitted each outer specialist dispatch as another root. The real nested specialist agents never received that handler, so their model/tool loops were absent.
- The callback handler is now request-scoped and explicitly bound to the configured Langfuse project. Heathcliff records one root agent observation with `coordinator.plan`, `coordinator.plan_repair`, `coordinator.aggregate`, and one direct agent observation per specialist task. The handler is forwarded into every nested LangGraph specialist, which records its model generations and native tool calls, inputs, and outputs below that specialist.
- The outer manual callback bridge still serves non-Langfuse callbacks (including legacy approval tests), but excludes the marked Langfuse handler so it cannot invert or duplicate the trace tree.
- Coordinator observations refuse to start without an active Heathcliff root, preventing direct graph calls and test runs from producing unrelated Langfuse root traces.
- Live non-mutating Langfuse verification produced one trace with `heathcliff.trace-smoke-v2 -> info_agent_tool -> nested_agent_loop`; full test suite passed with 331 tests. No AI request or external action was made for the smoke check.

## 2026-08-23 Legacy Langfuse Trace Annotation

- Langfuse cannot reparent or repair already-ingested observations. The 48 objectively affected pre-repair standalone specialist roots (Info, Music, Email, Calendar, and Contacts, dated 2026-08-14 through 2026-08-15 UTC) were retained and non-destructively labelled through the Langfuse Scores API.
- Each has the categorical `trace_integrity=legacy_orphaned_specialist_root` score, an explanation that it is an incomplete child of a Heathcliff request, and metadata identifying the repair. The labels were verified in the configured Langfuse project; no trace was deleted, rewritten, or re-ingested.

## 2026-08-15 Context Manager Annotation

- Updated `core/subagents/_runner.py:capture_agent_invocations()` from `Iterator[...]` to `Generator[..., None, None]` for `@contextmanager` compatibility. Ruff passed and all 314 tests passed; `ty` still reports the same 34 unrelated diagnostics.

## 2026-08-15 Failed Action-Chain Hardening

- A live Korea-trip research-and-email attempt correctly paused for approval, but Gemini's daily quota made the research and contacts specialists return their explicit `... failed:` strings. The old coordinator treated those strings as successes, proceeded to Gmail, timed out, and left an unkillable worker alive; no email was sent before that worker then crashed in the third-party Gmail `raw` parser.
- The coordinator now converts the shared specialist failure/unavailable contract into `EXECUTION_ERROR`, so every dependent task becomes `DEPENDENCY_FAILED` and never invokes the next external action.
- Approval-gated delegated actions now run in-process after approval. Python cannot cancel an active `ThreadPoolExecutor` thread, so this prevents a send/draft/calendar/message action from continuing after Heathcliff reports a timeout. Non-mutating work retains bounded timeout behavior.
- Replaced toolkit Gmail search with `SafeGmailSearch`, which requests the supported `full` response and extracts `text/plain` MIME parts without requiring a `raw` field. Added regression tests for missing-raw payloads, failed dependency chains, and sensitive timeout handling.
- `TOOL_MODEL` defaults to `google_genai:gemini-3.1-flash-lite-preview`, matching the working supervisor default and remaining overrideable through `.env`; this avoids selecting the exhausted `gemini-3-flash-preview` by default.
- Verification: Ruff and `git diff --check` passed; the full suite passed (317 tests). `uvx ty check` reports the same 34 unrelated existing diagnostics and none from this repair.

## 2026-08-15 Outbound Email Formatting

- Gmail sends had always used an HTML MIME part but inserted the agent's Markdown/plain text directly, so lists and headings rendered as unstyled text; drafts were plain text only.
- Added the stdlib-only `format_outbound_email()` renderer: headings, ordered/unordered lists, paragraphs, bold/italic Markdown, safe links, and the mandatory Heathcliff disclosure all render as email HTML. Gmail drafts now include both plain-text and HTML alternatives; sends use the same HTML.
- Verification: Ruff and `git diff --check` passed; all 318 tests passed. `uvx ty check` remains at the same 34 unrelated diagnostics.
- Upgraded the email HTML into a branded, responsive Heathcliff card: navy header, gold monogram/accent, white content panel, restrained disclosure footer, and email-client-safe table/inline styling. No images or external assets are needed, and plain-text MIME alternatives remain available.

## 2026-08-15 Research Quality

- The adaptive fast/deep keyword classifier was removed at the user's request. The info specialist is now a single agent that chooses answer depth semantically from the whole request, while the prompt defines the source-driven standard for substantial answers. All information tools remain available; no words or phrases change the tool set.
- Removed `AgentDescriptor.matches_goal()` and capability-keyword fallback routing. If LLM planning fails, Heathcliff uses the general information agent rather than guessing from text overlap.
- Removed all delegated email/calendar/comms approval regexes. The approval policy now uses exact known tool identities and conservatively pauses every mutation-capable delegated agent regardless of the request text; this preserves safety without phrase matching.
- Verification: Ruff, formatting, and `git diff --check` passed; all 312 tests passed. `uvx ty check` remains at the same 34 unrelated diagnostics.
- Added the official `langchain-tavily` integration. When `TAVILY_API_KEY` is configured, the info agent receives Tavily's `tavily_search` and `tavily_extract` tools, alongside its existing free public sources. Search is deliberately limited to Tavily: the other LangChain-listed free-tier web-search vendors need separate accounts/keys and duplicate Tavily's purpose. The key is not configured in this checkout yet.
- Verification: `uv lock`/`uv sync` installed `langchain-tavily==0.2.17`; Ruff, formatting, and `git diff --check` passed; all 314 tests passed. `uvx ty check` remains at the same 34 unrelated diagnostics.

## 2026-08-14 Ponytail Simplification

- Restored `utils/heathcliff_greetings.py` exactly from the supplied pre-deletion source (apart from its CRLF line endings): all time/weather variations, weather commentary, and return-greeting behaviour are preserved.
- Added `tests/test_greetings.py`; the full suite now passes 302 tests. `uvx ty check` has remaining unrelated project-wide diagnostics.
- Restored missing runtime dependency declarations for the voice stack (`pvporcupine`, `pyaudio`, `pyttsx3`) and the directly used Google OAuth clients. `uv sync` now installs them; importing the voice stack succeeds.
- Made text mode the default CLI path: `uv run main.py` no longer initializes or imports audio; `--voice` explicitly enables it. Added CLI/lazy-import regressions and verified a real text-mode startup/quit flow.
- Added CLI approval resume: after a sensitive action pauses, `approve`/`sure`/`yes` resumes the original checkpointed action, while `reject`/`no`/`cancel` rejects it. The original request is retained for the resume and no second planner invocation occurs.
- Added Spotify playlist playback: named playlists now resolve against the user's playlists and use Spotify playlist-context playback, never a track search. Spotify requests playlist-read scopes and will prompt once to refresh the cached authorization if needed.
- Replaced the brittle third-party Wikipedia client with Wikimedia's REST search API. Mount Fuji research now returns live source snippets without the JSON decode traceback; request failures are reported as a concise temporary-unavailable message.
- Consolidated every model and Mem0 configuration on provider-neutral `AI_KEY` via `Config.get_ai_api_key()`. `GOOGLE_API_KEY` and `GEMINI_API_KEY` remain migration fallbacks, while Custom Search receives only `GOOGLE_CSE_API_KEY` directly and cannot overwrite the AI key.
- Removed unused document indexing (`DocumentManager`/`my_data`), context store, nonlocal/deep delegation adapters, the duplicate `voice/main.py` entry point, and unused direct Playwright/LangSmith dependencies.
- Replaced the unused supervisor tool assembly and LangChain middleware stack with a small `DelegationBudget` used directly by the coordinator.
- Reduced the coordinator to `plan → execute_subtasks → aggregate`; streaming keeps a derived sequential dispatch event. Dependency validation, timeouts, callbacks, and the process-local resumable approval interrupt remain intact.
- Kept approval as the pure `requires_approval()` policy plus the LangGraph resume UI; removed the obsolete Streamlit callback handler and its session mutation helpers.
- Collapsed the five simple domain-agent response wrappers into `core/subagents/_runner.py`, simplified CLI parsing, and updated persistence/UI/docs to remove document-index claims.
- Verification: `uv run ruff check --fix . && uv run ruff format .`, `uv lock`, `uv run pytest tests -v -s` (310 passed), `uv run python main.py --help`, and `git diff --check` all passed. `uvx ty check` reports 37 existing diagnostics concentrated in optional audio dependencies and third-party/LangGraph type narrowing.

## 2026-08-15 Real-Service Integration Pass

- Added `scripts/run_live_integration.py`, a text-mode, bounded real-service runner. It saves one JSONL record per query with timestamps, session ID, coordinator stream events, response, approval payload, and specialist tool traces. It deliberately rejects approval-gated draft/calendar actions before any mutation.
- Added opt-in diagnostic tracing in `core/subagents/_runner.py`, and passed the trace context across the coordinator's worker thread. The custom info agent now records its tool messages too. This is inactive during normal operation.
- The first real run is local-only at `artifacts/live-integration-20260815.jsonl` (ignored because it includes private service output). Seven calls completed before the free-tier quota was exhausted; its events and responses are preserved, but its per-tool trace arrays are empty because it was generated immediately before the worker-context trace fix.
- Confirmed successes: live Jersey City weather, three current technology headlines, and Gmail unread-message retrieval. The latter necessarily wrote sender/subject data only to the ignored local audit artifact.
- Confirmed failures: deep Mount Fuji research exceeded the coordinator 60-second deadline but continued tool/model work after the coordinator returned; Calendar read failed inside `langchain_google_community.calendar.search_events` with malformed JSON parsing; Spotify attempted interactive OAuth and then returned an EOF-derived misleading “nothing playing” response in a noninteractive runner; later Contacts/complex queries hit Gemini free-tier `RESOURCE_EXHAUSTED` (model `gemini-3-flash`, daily quota value 20).
- Do not rerun the full live suite until the Gemini quota resets. Then run `uv run python scripts/run_live_integration.py --include-approvals --output artifacts/live-integration-<timestamp>.jsonl`; the two mutation-intent cases are rejected by the harness.
- Final validation: Ruff clean and `uv run pytest tests -v -s` passed (312 tests). `uvx ty check` has 34 pre-existing diagnostics; the trace additions introduced none.

## 2026-08-15 Outbound Identity

- Added `utils/outbound_signature.py`: every Gmail draft, Gmail send, and Telegram message now receives exactly one footer using `Config.MASTER_INFO["full_name"]` (falling back to `name`): `Heathcliff o.b.o {master_name}` followed by `This is sent by Heathcliff an Autonomous Intelligence system. It may make mistakes.`
- Gmail enforcement happens at the toolkit send/draft message builders, and Telegram enforcement at `send_to_telegram`; it cannot be omitted by the model prompt. Calendar descriptions and regular assistant chat responses are intentionally unchanged.
- Added MIME-level draft/send regression coverage plus Telegram coverage in `tests/test_outbound_signature.py` and `tests/test_action_execution.py`. `tests/test_greetings.py` now validates required greeting content rather than a random exact phrase. Full validation passed (314 tests); `ty` remains at 34 existing diagnostics with none from this feature.

## 2026-08-13 Approval Policy and Resumable Actions

- Added a shared `requires_approval(tool_name, tool_input)` policy covering exact sensitive Gmail/Calendar/Telegram tools and mutation intent routed through the outer email, calendar, and communications agents. Read/search/list requests remain unblocked.
- Replaced the Streamlit callback dead-end with a LangGraph human-in-the-loop flow: the coordinator compiles with `InMemorySaver`, pauses with `interrupt()`, and resumes the same run with `Command(resume=...)` under the same conversation/thread ID.
- Sensitive tasks are preflighted before the execute node performs any subagent call. This avoids LangGraph node-replay duplicating earlier side effects when the run resumes.
- Runtime callback objects now travel through LangGraph runtime context rather than checkpointed state, avoiding msgpack serialization failures while preserving callback telemetry and legacy rejection behavior.
- Streamlit Approve, Modify, and Reject controls call `HeathcliffAgent.resume_approval()`; interrupted turns are saved to conversation memory only after terminal completion.
- Added `tests/test_approval_handler.py` and `tests/test_approval_resume.py` for mutation/read classification, callback parity, exact-once execution, rejection mapping, no-save-on-pause, and UI resume wiring.
- Current persistence ceiling: `InMemorySaver` survives Streamlit reruns within the running process but not a process restart. Add a configured SQLite/Postgres checkpointer before claiming restart-durable approvals.
- Verification: Ruff check/format passed; `uv run pytest tests -v -s` passed all 376 tests. The captured pytest command still fails before collection because its temporary capture file disappears. `uvx ty check` reports 48 existing project-wide diagnostics, including optional voice imports, LangGraph typing, registry tool types, and DB result narrowing.

## 2026-08-02 Repository Audit and Action-Chain Hardening

- Audited the staged coordinator/delegation migration together with the unstaged `db/` persistence migration.
- Fixed coordinator planning to consume retrieved LangChain conversation messages. Short replies such as an email address can now be interpreted in the context of the prior requested action instead of being routed as isolated new requests.
- Extended the planner instruction to require explicit dependencies for data-producing chains such as research + contact lookup → email.
- Added `tests/test_action_execution.py` covering clarification history, research/contact outputs reaching the email task, and awaited Telegram sends.
- Fixed `send_to_telegram()` to await `python-telegram-bot`'s async `send_message()` before reporting success.
- Decoupled text-only startup from optional audio imports; fixed `AudioHandler` to use the class-based `Config` API; fixed the Memories page import after `core/memory_manager.py` moved to `db/memory_manager.py`.
- Updated `AGENTS.md` from removed Black/isort commands to the configured Ruff toolchain.
- Verification: `uv run pytest tests -v -s` passed 331 tests; Ruff check/format passed; text-mode import and touched-file compilation passed.
- Resolved 2026-08-13: delegated mutation policy and LangGraph interrupt/resume now pause and resume the original action. Checkpoints remain process-local until a persistent saver is configured.
- Remaining coordinator issue: tasks marked `parallelizable` are still executed by a sequential loop; `effective_parallel` is calculated but not used for scheduling.
- Remaining validation issue: `uvx ty check` reports 52 diagnostics, concentrated in optional voice dependencies, LangGraph/registry typing, and new DB result typing.
- Environment note: pytest's default capture crashed on a missing capture tempfile in this workspace; running with `-s` completed normally.

## 2026-05-13 New Architecture Redesign Draft Started

- Created `docs/new_architecture.md` as the shared draft for redesigning Heathcliff from the ground up.
- Initial direction: Heathcliff is a personal butler / Jarvis-like top-level orchestrator, not one giant agent with every tool.
- Drafted hierarchical multi-agent concept: `Heathcliff Agent -> task-specific agents with focused tools`.
- Added planner-led fan-out with shared state, inspired by ADK multi-agent and parallel-agent workflow patterns.
- Added controlled handoff concept: specialist agents may emit structured handoff requests, but Heathcliff validates and routes them to avoid unbounded agent-to-agent loops.
- Current agreement: use LangChain + LangGraph to implement ADK-inspired concepts rather than adopting ADK itself.
- Added context-management decision: Heathcliff model input should use real LangChain `SystemMessage`, `HumanMessage`, and `AIMessage` objects; never serialize prior chat history into the current `HumanMessage`.
- Current `HumanMessage` may include current user query, memories, runtime context, active task state, and selected subagent context/results. Full subagent outputs live in shared state and enter prompts through a context builder.
- Specialist agents should receive focused task packets with strict-but-not-restrictive instructions, relevant context only, allowed tools, and an expected structured output shape.
- Added DB/module design decision: conversation history should live in a Chroma collection, extracted memories should live in a separate memories collection, and memory extraction should run asynchronously after conversation writes.
- Proposed DB module split: `base` for connection and generic CRUD/collection lifecycle, `memory_manager` using Mem0 and the memories collection, and `conversation_manager` for storing/retrieving multimodal conversation messages as LangChain `HumanMessage` / `AIMessage` blocks.
- Added concrete redesign principle: the architecture should be explicitly object-oriented and service-based, using concrete classes, typed data structures, explicit service boundaries, controlled singletons, and caching where it improves runtime behavior.
- Expanded OOP/service design in `docs/new_architecture.md`: proposed services include `HeathcliffRuntime`, `HeathcliffAgent`, `AgentRegistry`, `ContextBuilder`, `HandoffResolver`, `ApprovalService`, `ConversationManager`, and `MemoryManager`; singleton use is limited to stable resources, while per-turn state remains request-scoped.
- Reorganized `docs/new_architecture.md` so OOP/service structure is baked into the architecture flow: core direction, runtime shape, multi-agent topology, fan-out/shared state, controlled handoffs, context management, lifecycle/caching, and architecture rules.
- Cleaned up `docs/new_architecture.md` to remove duplication between context management and DB design: persistence/storage details now live in `Persistence And Memory Services`, while `Context Management` only describes prompt assembly and service boundaries.
- Added prompting and grounding architecture to `docs/new_architecture.md`: prompts are versioned architecture artifacts under a `prompts/` module with `PromptRegistry`; stable instructions precede dynamic context; Heathcliff prompt focuses on orchestration; specialist prompts focus on narrow execution; factual/actionable responses must be grounded in memories, history, tool outputs, or cited research.

## How Agents Use This File

- Before starting work: Check this file for ongoing issues, recent discoveries, and previous agent findings
- After completing work: Update this file with new issues, workarounds, code patterns discovered, and activity log
- Reference `AGENTS.md` for project overview, architecture, configuration, and development standards
- Cost optimization: Reuse previous agent discoveries instead of re-investigating
- Share API integration workarounds and debugging strategies discovered during work

---

## 2026-05-12 Coordinator Stability Remediation

- Hardened `core/coordinator_graph.py` with strict planner schema validation (`extra="forbid"`) and single repair pass before fallback.
- Enforced dependency semantics:
  - out-of-range, forward-ref, self-ref -> task marked `DEPENDENCY_FAILED`
  - cycle-involved tasks -> `DEPENDENCY_FAILED`
  - acyclic tasks continue execution.
- Removed invalid dependent-task `TaskSpec.parallelizable` construction that previously caused dependency-chain crashes.
- Added coordinator callback bridge around local adapter execution:
  - `on_tool_start` / `on_tool_end` / `on_tool_error` hooks are called
  - approval rejection maps to `TaskStatus.APPROVAL_REJECTED` + `ErrorType.APPROVAL_REJECTED`.
- Enforced coordinator budgets/timeouts from `Config`:
  - post-plan task-count budget check with truncation
  - pre-execution parallel cap clamping via `effective_parallel`
  - per-task timeout -> `TIMEOUT` and continue
  - total-runtime cutoff -> stop scheduling new work.
- Kept text-based dependency-context injection but hardened with normalization, control-char stripping, literal code-block wrapping, and 1600-char total cap (evenly split).
- Updated stream contract and UI:
  - coordinator stream emits `plan`, `dispatch`, `subtask_complete`, `quality_retry`, `response`, `complete`
  - completion payload uses `agents_used` + `agent_count` (deduped first-seen order)
  - `ui/Home.py` now consumes `agents_used` instead of `tools_used`.
- Added structured per-task telemetry logging fields: `status`, `error_type`, `latency_ms`.
- Added/rewrote `tests/test_coordinator.py` coverage for dependency chains, invalid deps/cycles, unknown target-agent semantics, timeout/runtime behavior, approval-rejection mapping, planner repair path, and `agents_used` stream semantics.

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
│       ├── comms/            # Telegram messaging
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
├── tests/                    # Pytest test suite (7 test files)
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
- **Middleware (2026-02-23)**: Framework exists in `core/middleware.py` — `LLMToolSelectorMiddleware`, `ToolCallLimitMiddleware`, `TodoListMiddleware`, and `RobustLLMToolSelectorMiddleware` (alias rewriting for 12+ hallucinated tool names) are active. `always_include=["recent_context"]` ensures recency tool stays available. Tests mock `create_middleware_stack` to avoid `langchain_openai` import dependency.
- **Middleware Tool Selection (2026-02-23)**: `LLMToolSelectorMiddleware` now sets `always_include=["recent_context"]` so recency snippets remain selectable even when other tools are filtered.
- **Context Window**: Retrieval now uses pair-aware history (`build_message_history`) with semantic pairs first and recent chronological pairs next.
- **Info Tooling (2026-04-05)**: `core/subagents/info/tools.py` now uses LangChain wrappers/toolkits for Wikipedia (`WikipediaQueryRun`), Wikidata (`WikidataQueryRun`), StackExchange (`StackExchangeTool`), and NASA (`NasaToolkit` via typed facade tools), while preserving `recent_context` capture across new tools. `search_web()` fallback now routes through the LangChain Wikipedia wrapper instead of manual `wikipedia` package calls.

### Operational Notes

- Mem0 + Chroma: Use `path` (or host/port) in Chroma config; `persist_directory` fails validation. Do not pass `chromadb.CloudClient`; provide `api_key` + `tenant` instead.
- ChromaDB usage: Collections auto-create via `get_or_create_collection()`. IDs must be unique strings (use `mem_*`/`doc_*`). Query result keys: `documents`, `metadatas`, `distances`, `ids`; filter with `where`.
- Audio: PyAudio stream must be 16000 Hz; wake word frames are 512 samples. Call `adjust_for_ambient_noise()` before STT; set TTS engine properties once at init.
- Dependencies: On Linux install `python3-pyaudio` first (`sudo apt install python3-pyaudio`).

### Known Issues

- Middleware disabled due to LangGraph incompatibility (missing `raise_error`/`ignore_chain` attributes). — **Resolved 2026-02-23**: middleware stack is now active (`ToolCallLimitMiddleware`, `TodoListMiddleware`, `LLMToolSelectorMiddleware`).
- Tests require `create_middleware_stack` to be mocked (the `LLMToolSelectorMiddleware` internally imports `langchain_openai` which isn't installed in the test env).
- Voice listener concurrency not fully tested.
- Gmail/Calendar/Spotify rate limits need backoff/retry logic.
- PyAudio is platform-dependent — test on target systems early.

---

## Timeline (Latest Activity)

- **2026-04-06**: **Info agent adaptive routing + recursion hardening** ✅
  - Implemented dual-mode routing in `core/subagents/info/agent.py`:
    - `fast` mode for short factual requests.
    - `deep` mode for analysis/source-heavy requests.
  - Added request classifier (`_choose_research_mode`) using explicit deep-intent keywords and structural complexity signals.
  - Added per-mode recursion budgets via invoke config (`recursion_limit`) to prevent routine queries from hitting deep research limits.
  - Added optional fast→deep one-time escalation when fast mode hits `GraphRecursionError`.
  - Added graceful fallback on recursion loops using `recent_context` snippets; raw recursion exception text is no longer returned to users.
  - Added new config knobs in `config/config.py`:
    - `INFO_ADAPTIVE_ROUTING_ENABLED`
    - `INFO_FAST_TO_DEEP_ESCALATION_ENABLED`
    - `INFO_FAST_RECURSION_LIMIT`
    - `INFO_DEEP_RECURSION_LIMIT`
  - Added test coverage in `tests/test_subagents.py` (`TestInfoAgentAdaptiveRouting`) for mode selection, recursion-limit config passing, escalation behavior, and graceful fallback.
  - Verification run:
    - `uv run pytest tests/test_subagents.py -v` (52 passed)
    - `uv run pytest tests/test_agent_integration.py -k "MiddlewareAliasNormalization or PromptRulesNoInnerToolNames" -v` (16 passed)

- **2026-04-05**: **Info agent knowledge-tool refresh (Wikipedia + Wikidata + StackExchange + NASA)** ✅
  - Replaced custom `wikipedia_search` parsing/disambiguation with LangChain `WikipediaQueryRun` + `WikipediaAPIWrapper` in `core/subagents/info/tools.py`.
  - Added new info tools: `wikidata_search`, `stackexchange_search`, `nasa_media_search`, `nasa_media_manifest`, `nasa_media_metadata`, `nasa_video_captions`.
  - Wired NASA through `NasaToolkit.from_nasa_api_wrapper(...)` and mapped toolkit modes to stable snake_case tool functions for better prompting/reliability.
  - Updated `search_web` final fallback to use the same LangChain Wikipedia wrapper path for consistent behavior.
  - Expanded middleware alias rewrites in `core/middleware.py` for new info tool names (`wikidata_search`, `stackexchange_search`, `nasa_*`) to route safely to `info_agent_tool` at supervisor level.
  - Updated info subagent prompt and description in `core/subagents/info/agent.py` to advertise new tools/capabilities.
  - Added alias regression coverage in `tests/test_agent_integration.py`.
  - Verification run:
    - `uv run pytest tests/test_agent_integration.py -v` (56 passed, 2 failed) — failures are pre-existing coordinator-graph mocking behavior in `TestAgentWithMockedMemory::test_full_flow_retrieval_to_output` and `TestToolCallingIntegration::test_tool_request_and_response` (response is a mocked coordinator object, not a string).
    - `uv run pytest tests/test_agent_integration.py -k "MiddlewareAliasNormalization" -v` (16 passed)
    - `uv run pytest tests/test_subagents.py -v` (47 passed)
    - `uv run pytest tests/test_recent_context.py -v` (22 passed)
    - Manual smoke test script invoked all new tools successfully (Wikipedia, Wikidata, StackExchange, NASA).

- **2026-04-05**: **Spotify device preference + LLM selector** ✅
  - `play_track` now calls a structured-output LLM (`DeviceSelection`) to pick the target device from available Spotify Connect devices; falls back to default device when ambiguous.
  - If the requested device name isn't found, responds: "I can play '<track>' by <artist>, but I can't control the specific device it plays on. Shall I play it on the default device?" and does not start playback.
  - Music agent prompt updated to preserve device preferences and ask before switching to default.
  - Added tests `TestMusicDeviceSelection` covering missing-device fallback and requested-device match.

- **2026-04-05**: **Spotify playback hardening (deterministic parsing + confidence gate)** ✅
  - Device parsing is deterministic; LLM parsing is removed from the control path. Device phrases are stripped before Spotify search.
  - Track search uses cleaned music query and picks the best-scored candidate; if the user specified an artist and confidence is low, it asks before playing.
  - User-facing response now prefers tool output to avoid hallucinated confirmations.
  - Tests updated for device fallback, device match, and ensuring device phrases are excluded from search.

- **2026-02-24**: **Prompt optimization Phases 0–4 complete — latency reduction & test suite at 237** ✅
  - **Root cause**: System prompt referenced raw inner tools (`get_weather`, `search_web`) instead of supervisor-level tools (`info_agent_tool`, etc.), causing hallucinated tool calls and ~48s weather queries.
  - **Phase 0 (Emergency hardening)**: Added `TOOL_NAME_ALIASES` dict in `core/middleware.py` mapping 12+ hallucinated tool names to canonical supervisor tools via `RobustLLMToolSelectorMiddleware`. Fixed `info_agent_tool` to accept both `request` and `query` params. Added 30 tests (info param compat + middleware alias + prompt regression).
  - **Phase 1 (System prompt consolidation)**: Complete rewrite of `build_system_prompt()` in `instructions/prompts.py` with XML-delimited sections (`<role>`, `<tools>`, `<routing_examples>`, `<execution_rules>`, `<response_style>`, `<user_profile>`), 6 few-shot routing examples, positive-only enforcement.
  - **Phase 2 (Tool description normalization)**: Standardized all 9 supervisor-visible tool `@tool` docstrings to `Use for:` / `Provide:` / `Returns:` / `Example:` template. Fixed email tool (recipient conditional), comms tool (removed Google Drive references).
  - **Phase 3 (Subagent prompt slimming)**: Reduced all 6 subagent `_SYSTEM_PROMPT` constants from ~30–90 lines to ~8–15 lines using XML tags (`<task>`, `<rules>`, `<workflow>`). Removed verbose `**Reasoning**` output blocks.
  - **Phase 4 (Test updates)**: Added `TestToolDescriptionConsistency` class (11 tests), 7 new XML tag validation tests, updated existing tests for new format.
  - **Files edited**: `instructions/prompts.py`, `core/middleware.py`, all 6 subagent `agent.py` files, `skills/skill_tools.py`, `skills/master_info.py`, `core/subagents/info/recent_context.py`, `tests/test_agent_integration.py`, `tests/test_subagents.py`.
  - **Full suite: 237 passed, 0 failed.**

- **2026-02-23**: **JSON-backed persistent recent context store + test suite green** ✅
  - Rewrote `core/subagents/info/recent_context.py` from in-memory list to JSON-backed persistent store: `temp/recent_memory.json` with configurable TTL (2h), max items (100), atomic writes (`.tmp` + `os.replace()`), `threading.Lock`, stale cleanup on every read/write, corrupt-file recovery, and auto-path setup on module load.
  - Added `RecentContextConfig` to `config/config.py` with 5 env-var-backed params (`RECENT_CONTEXT_TTL_SECONDS`, `RECENT_CONTEXT_MAX_ITEMS`, `RECENT_CONTEXT_MAX_SNIPPET_CHARS`, `RECENT_CONTEXT_MAX_RETURN`, `RECENT_CONTEXT_STORE_PATH`).
  - Created `tests/test_recent_context.py` with 22 tests (persistence roundtrip, TTL expiry, max-items pruning, corrupt JSON fallback, return ordering/clamping, content filtering, auto-path setup).
  - Fixed test helper `_read_store()` to handle file-not-created case (empty/error content filtered before first write).
  - Updated `tests/test_subagents.py` to expect 7 tools (6 agents + `recent_context`).
  - Updated `tests/test_agent_core.py`, `tests/test_agent_e2e.py`, `tests/test_agent_integration.py` to mock `create_middleware_stack` (avoids `langchain_openai` import in test env).
  - Updated `tests/test_agent_core.py::TestToolRegistration` to expect 9 supervisor tools (added `recent_context`).
  - **Full suite: 184 passed, 0 failed, 0 errors.**

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

- **2026-08-16: Email resend reliability** ✅
  - Diagnosed a real resend to Ram: Gmail search ignored `text/html` bodies, so the email agent copied Gmail's truncated preview (`...`) and retained the original Aditya/Nilesh greeting.
  - `SafeGmailSearch` now extracts readable text from full HTML mail payloads, and the email-agent prompt directs semantic resend requests to use the complete body with a recipient-specific greeting.
  - Added regression coverage for HTML body extraction; no outbound message was sent during diagnosis or verification.

- **2026-08-15: Streamlit control panel rebuild** ✅
  - Replaced legacy page scripts with a shared `st.navigation` shell and focused Command Center, Agent Controls, Runs & Traces, Analytics, Memories, and Conversations views.
  - Runtime settings now use a process-local, revisioned `RuntimeProfile`; future requests get a fresh coordinator while a pending approval keeps its original agent snapshot.
  - Specialist agents bind their model to the run profile, rather than mutating global `Config` values.
  - The Command Center restores Heathcliff's weather-aware British-butler greeting for new conversations and keeps approval actions resumable.
  - Completed streamed turns persist their coordinator events in conversation metadata. Conversations show those events in an accordion after the relevant message; Analytics presents them in full-width tables.
  - The Memories view always shows active data, 50 per page, without custom CSS. The UI intentionally stays close to default Streamlit pending a future Next.js migration.
  - Verified `uv run ruff check --fix . && uv run ruff format .` and `uv run pytest tests -q -s` (`325 passed`). `uvx ty check` has 25 pre-existing diagnostics outside this change set.

**Phases 1–4 Complete** ✅ — Subagents architecture refactor, skills framework (3 skills), Mem0 memory, 3D Blob UI, Langfuse observability, master profile TOML migration, user reference generalization, weather API LangChain wrapper, JSON-backed recent context store, middleware stack (tool selector + call limits + todo list + alias rewriting), prompt optimization (XML-delimited system prompt, normalized tool descriptions, slimmed subagent prompts) all completed. Full test suite: 237 passed. **Phase 5 (Testing & Polish) In Progress** ⏳

Next steps: Integration testing, error recovery, Docker containerization, troubleshooting guide.
