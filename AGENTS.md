# Repository Guidelines

## Purpose

- Guidance for agentic coding tools working in this repository.
- Keep this file aligned with repo conventions and updated when workflows change.

## Project Overview

- Heathcliff is a voice-enabled personal AI assistant integrating Gmail, Google Calendar, Spotify, Weather, News, and more.
- Wake-word voice loop runs in its own thread; text-only mode is available for dev and tests.
- Agent orchestration uses LangChain + LangGraph with Gemini models; memory persists via ChromaDB.

## Architecture Snapshot

- Voice layer: `voice/main.py` + `core/audio_handler.py` handle wake word, STT, and TTS.
- Agent layer: `core/agent_core.py` (singleton supervisor) orchestrates subagents.
- Subagents layer: `core/subagents/` contains domain-specific agents (`info`, `music`, etc.) each with its own `tools.py` and `agent.py`.
- Skills layer: `skills/` contains dynamic capabilities loaded at runtime.

## Project Structure & Module Organization

- `core/` contains subagents, memory, audio handlers, and the singleton supervisor agent.
- `skills/` contains dynamic skills that extend the agent's capabilities.
- `config/` holds the singleton `Config` and middleware settings.
- `ui/` and `voice/` encapsulate Streamlit dashboard and wake-word entry points.
- Mirrors of these modules sit under `tests/` for parity and test organization.
- `instructions/` owns system prompt templates and anti-redundancy rules for tool calls.

## Environment Setup

- System deps (Linux): `sudo apt install python3-pyaudio portaudio19-dev espeak`
- Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Sync deps: `uv sync` (creates `.venv` from `pyproject.toml` / `uv.lock`)
- Configure env: `cp .env.example .env` then set API keys and OAuth files

## Run Commands

- Voice mode (primary): `uv run python main.py`
- Text-only mode: `uv run python main.py --text`
- Legacy entry (if used): `uv run python app.py`
- Streamlit UI: `uv run streamlit run ui/Home.py`

## Debugging

- Check `logger.py` for logging setup and formats.
- Enable verbose logs with `LOG_LEVEL=DEBUG`.
- Monitor voice thread output for wake word detection issues.

## Lint/Format

- Format: `uv run black .`
- Sort imports: `uv run isort .`

## Tests

- All tests: `uv run pytest tests -v`
- Single file: `uv run pytest tests/test_memory_manager.py -v`
- Single test: `uv run pytest tests/test_memory_manager.py::TestMemoryManager::test_add_memory -v`
- Pattern filter: `uv run pytest -k "memory" -v`

## Coding Style & Conventions

- Python 3.11+; 4-space indentation; keep functions small and focused.
- Imports: stdlib first, third-party next, local last; use isort; avoid unused imports.
- Formatting: black defaults (88-char line length).
- Types: add type hints for public functions and class APIs; use `Optional[T]`, `list[T]`, `dict[str, T]`.
- Naming: modules/files snake_case, classes PascalCase, functions verb phrases, constants UPPER_SNAKE.
- Docstrings: concise Google-style on public entry points; keep internal helpers minimal.
- Config access: import `Config` from the `config` package; avoid re-reading `.env` elsewhere.
- Side effects: keep LangGraph nodes pure; do I/O inside tools or dedicated helpers.
- Utilities: centralize shared helpers in `utils/` and reuse instead of duplicating.
- Prompt changes: edit `instructions/prompts.py` and restart the agent to apply.

## Error Handling & Logging

- Use `logger` from `logger.py` for consistent formatting and LOG_LEVEL control.
- Raise `ValueError` for invalid configuration (see `Config.validate()` pattern).
- Wrap external API calls in tools with try/except; return actionable errors.
- Avoid swallowing exceptions silently; log context without leaking secrets.
- Prefer explicit fallbacks over broad `except Exception` unless guarding API boundaries.

## Agent/Prompt Behavior

- Prompts require single-pass tool execution; avoid redundant tool calls.
- Always check tool feedback before re-calling a tool; retry only with improved arguments.
- Keep voice responses concise (1-3 sentences) and context-aware.

## Voice Processing Flow

- `VoiceListener.start()` initializes Porcupine and PyAudio.
- Listener blocks on wake word in a loop.
- On detection, callback receives recognized text.
- Agent processes and returns a response.
- Response is spoken back to the user.

## Tool Implementation Pattern

- Tools follow LangChain's Tool pattern; keep descriptions clear and precise.

```python
from langchain.agents import Tool

tool = Tool(
    name="tool_name",
    func=function_reference,
    description="What the tool does",
)
```

## Testing Guidelines

- Pytest discovery: `tests/test_*.py`, classes `Test*`, functions `test_*` (see `pytest.ini`).
- Place fast unit tests near their subject; add integration/E2E suites for cross-service behavior.
- Mock Gmail/Calendar/Spotify clients and ChromaDB for deterministic tests.
- Record expected prompts/responses as fixtures when testing agent behavior.

## Testing Considerations

- Test tool integrations independently before connecting to the voice pipeline.
- Verify API credentials early; failures can be non-obvious in voice mode.
- Exercise prompts with varied inputs to validate tool selection.
- Watch for audio device issues (PyAudio is platform-dependent).

## Configuration & Secrets

- Secrets live in `.env` and OAuth files (e.g., `credentials.json`); never commit them.
- Validate config via `Config.validate()` before running tool-heavy flows.
- Langfuse observability uses env vars (`LANGFUSE_*`); disable via config when needed.

## Observability Notes

- Langfuse trace name defaults to `heathcliff.agent`; user id set in config.
- If traces are missing, verify `LANGFUSE_*` env vars and enable DEBUG logging.

## Commit & PR Guidelines

- Use imperative, descriptive commit subjects (e.g., "Add Gmail sync tool scaffolding").
- Mention config/dependency changes explicitly; include UI/audio artifacts when relevant.
- Before review, run formatters and `uv run pytest tests -v`.

## Agent Coordination & Knowledge Base

- **CRITICAL:** Always read `plan/MEMORY.md` before starting work to understand current context.
- **MANDATORY:** You MUST update `plan/MEMORY.md` with new discoveries, decisions, and progress after every session.
- Keep `plan/TODO.md` updated as tasks progress; treat it as source of truth.
- Refer to `plan/EXECUTION.md` alongside `plan/TODO.md` for implementation guidance.

## Shared Agent Memory (`plan/MEMORY.md`)

- Codebase discoveries: structure, architecture, code organization.
- API credentials and configuration details.
- Development decisions and standards.
- Dependencies and versions.
- Known issues and considerations.
- Planned features and roadmap items.
- Agent coordination notes and handoffs.
- Recent agent activity log.

## Task Tracking

- Update `plan/TODO.md` immediately after completing each item.
- Check off tasks as you start and finish work.
- Treat the TODO list as the live source of ownership.

## Documentation Map

- `README.md` for usage modes and quick start.
- `SETUP.md` for detailed environment and credential setup.
- `plan/EXECUTION.md` for implementation guidance.

## Cursor/Copilot Rules

- No `.cursor/rules/`, `.cursorrules`, or `.github/copilot-instructions.md` found in this repo.

## Additional Notes

- Use text mode for debugging when audio hardware is unavailable.
- Be mindful of API rate limits for Gmail, Calendar, Spotify, Weather, and News.
- Always use web.fetch and web.search when discussing and planning.
