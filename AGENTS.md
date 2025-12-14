# Repository Guidelines

## Project Structure & Module Organization
- `core/` contains LangGraph orchestration (`agent_core.py`), persistent memory (`memory_manager.py`), and audio helpers; keep agent nodes isolated and side-effect free.
- Runtime knobs live in `config/` and `config.yaml`; always load through `config/config_loader.py` so overrides flow consistently from `.env`.
- `tools/`, `utils/`, `ui/`, and `voice/` encapsulate integrations, helpers, Streamlit UI, and wake-word assets, while mirrors of those modules sit under `tests/` for parity.

## Architecture Snapshot
- **Voice layer** (`voice/main.py`) owns wake-word detection (Porcupine) and Google STT; it runs in its own thread and invokes the agent callback on activation.
- **Agent layer** (`agent.py`, `settings.py`) wires LangChain/Gemini, follows the ReAct pattern, and persists conversation context via LangGraph state.
- **Tools layer** (`tools/gmail_tools.py`, `tools/calendar_tools.py`, `tools/spotify_tools.py`, `tools/alexa_tools.py`) encapsulates API-specific logic; expand these modules rather than inlining API calls elsewhere.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` keeps the heavy audio/LLM stack isolated; install deps via `pip install -r requirements.txt` afterward.
- `cp .env.example .env` then set `GEMINI_API_KEY` and OAuth secrets before running anything that touches Google or Spotify APIs.
- `python app.py` starts the wake-word driven voice loop using the active config; `Ctrl+C` shuts down gracefully and flushes logs.
- `python -m pytest tests -v` runs the entire suite; filter with `-k <pattern>` for focused modules during iteration.

## Coding Style & Naming Conventions
- Python 3 code uses 4-space indentation, type hints, and concise Google-style docstrings for public entry points.
- Run `black .` (88-char default) and `isort .` before committing to keep diffs small; avoid manual formatting tweaks afterward.
- Modules, files, and test names are snake_case, classes are PascalCase, and functions should be verb phrases that describe the side effect or return value.

## Testing Guidelines
- Pytest discovers files as `tests/test_*.py`, classes `Test*`, and functions `test_*` per `pytest.ini`; stick with that template when adding suites.
- Place fast unit tests next to their subject (`test_memory_manager.py`), and extend integration/E2E flows (`test_agent_integration.py`, `test_agent_e2e.py`) when adding cross-service behavior.
- Mock Gmail/Calendar/Spotify clients plus ChromaDB in tests to keep runs deterministic and CI-friendly; record expected prompts/responses as fixtures.

## Commit & Pull Request Guidelines
- Follow the existing history by writing imperative, descriptive subjects (`Add Gmail sync tool scaffolding`) and optional body bullets for context.
- Mention config or dependency changes explicitly, link tracking issues, and include screenshots or transcripts when UI or audio behavior changes.
- Before requesting review, ensure formatters and `python -m pytest tests -v` pass, and summarize that verification plus manual steps in the PR description.

## Agent Coordination & Knowledge Base
- Read `plan/MEMORY.md` before starting work and append new discoveries (APIs, bugs, design choices) after finishing so concurrent agents stay synchronized.
- Keep `plan/TODO.md` in sync by checking off items as you progress; treat it as the live source of truth for task ownership.
- Centralize repeatable helpers inside `utils/` to avoid divergence between agents; reference prior approaches recorded in memory before re-implementing tooling.
- refer plan/TODO.md and plan/EXECUTION.md simultaneously
