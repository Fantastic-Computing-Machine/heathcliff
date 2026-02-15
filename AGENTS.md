# Repository Guidelines for AI Agents

## Project Overview

Heathcliff is a voice-activated AI assistant built with Python 3.11+, using LangChain/LangGraph for orchestration and Gemini as the LLM backbone. Integrates with Gmail, Google Calendar, Spotify, Weather APIs, and more.

## Project Structure

```txt
heathcliff/
├── main.py              # Entry point (--text for text mode, default is voice)
├── core/                # Agent orchestration and memory
│   ├── agent_core.py    # HeathcliffAgent - Unified LangChain agent
│   └── memory_manager.py # ChromaDB + Mem0 persistent memory
├── config/              # Configuration (import Config from config/__init__.py)
├── tools/               # API integrations (gmail, calendar, spotify, etc.)
├── utils/               # Shared helpers (google_auth, retry, errors)
├── ui/                  # Streamlit web dashboard
├── voice/               # Wake-word detection (OpenWakeWord) and Google STT
└── tests/               # pytest test suite
```

## Build & Development Commands

```bash
# Setup: Install uv, sync dependencies, configure environment
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
cp .env.example .env  # Set GEMINI_API_KEY and OAuth secrets

# Running the application
uv run python main.py           # Voice mode (wake-word activated)
uv run python main.py --text    # Text-only mode
uv run streamlit run ui/Home.py # Streamlit dashboard

# Formatting (run before committing)
black .                         # Code formatter (88-char line length)
isort .                         # Import sorter
```

## Testing Commands

```bash
# Run entire test suite
uv run pytest tests -v

# Run single test file
uv run pytest tests/test_memory_manager.py -v

# Run single test class
uv run pytest tests/test_memory_manager.py::TestAddMemory -v

# Run single test function
uv run pytest tests/test_memory_manager.py::TestAddMemory::test_add_memory_returns_id -v

# Run tests matching a pattern
uv run pytest tests -v -k "recall"

# Run with stdout output
uv run pytest tests -v -s
```

## Code Style Guidelines

### Formatting

- 4-space indentation
- 88-character line length (black default)
- Run `black .` and `isort .` before committing

### Imports (3-section organization)

```python
# 1. Standard library
import os
from typing import Any, Dict, List, Optional

# 2. Third-party
from langchain.tools import tool
import requests

# 3. Local/project
from config import Config
from logger import logger
```

### Type Hints & Docstrings

Use comprehensive type hints on all public APIs with Google-style docstrings:

```python
def add_memory(
    self, text: str, category: str = "general", metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Store a long-term fact or preference.
    Args:
        text: The memory content
        category: Category of memory (e.g., 'preference', 'fact')
    Returns:
        ID of the stored memory
    """
```

### Naming Conventions

| Element      | Convention          | Example                          |
|--------------|---------------------|----------------------------------|
| Modules      | snake_case          | `memory_manager.py`              |
| Classes      | PascalCase          | `HeathcliffAgent`                |
| Functions    | snake_case verbs    | `get_calendar_events()`          |
| Constants    | UPPER_SNAKE_CASE    | `GMAIL_SCOPES`                   |
| Private      | `_` prefix          | `_global_chroma_client`          |

### Module Headers

Every file should start with ABOUTME comments:

```python
# ABOUTME: Gmail integration using LangChain's GmailToolkit
# ABOUTME: Uses draft creation for safety instead of direct sending
```

### Error Handling

Use custom exceptions from `utils/errors.py`. Catch specific exceptions, log with context:

```python
from utils.errors import AgentMemoryError, ToolExecutionError

try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}", exc_info=True)
    raise AgentMemoryError(f"Memory operation failed: {e}")
```

### Logging

Import the centralized logger (`from logger import logger`) and use appropriate levels:

- `logger.debug()` for verbose details, `logger.info()` for normal operations
- `logger.warning()` for recoverable issues, `logger.error(..., exc_info=True)` for errors

## Testing Guidelines

- Mock external clients (Gmail, Calendar, Spotify, ChromaDB) for deterministic tests
- Use fixtures from `tests/conftest.py` (DummyMem0, DummyConfig, config_factory)
- Place unit tests alongside their subject (`test_memory_manager.py` for `memory_manager.py`)
- Extend `test_agent_integration.py` or `test_agent_e2e.py` for cross-service behavior

## Architecture Notes

- **Voice layer** (`voice/main.py`): Wake-word detection (OpenWakeWord) and Google STT, runs in its own thread
- **Agent layer** (`core/agent_core.py`): LangChain/Gemini with `create_agent` framework, persists context via LangGraph
- **Tools layer** (`tools/`): API-specific logic; expand these modules rather than inlining API calls

## Commit Guidelines

- Write imperative, descriptive subjects: `Add Gmail sync tool scaffolding`
- Mention config or dependency changes explicitly
- Ensure `black .`, `isort .`, and `uv run pytest tests -v` pass before committing

## Agent Coordination

- Read `plan/MEMORY.md` before starting work; append discoveries after finishing
- Keep `plan/TODO.md` in sync; treat it as the source of truth for task ownership
- Refer to `plan/TODO.md` and `plan/EXECUTION.md` simultaneously
- Centralize reusable helpers in `utils/` to avoid duplication
