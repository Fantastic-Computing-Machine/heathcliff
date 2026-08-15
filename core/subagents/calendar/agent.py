# ABOUTME: Calendar / Google Calendar sub-agent — create, search, update, delete events
# ABOUTME: Wraps tools/calendar_tools.py; exposed to supervisor as a single @tool

from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from config import Config
from core.subagents._runner import run_agent
from core.subagents.calendar.tools import get_calendar_toolkit_tools
from logger import logger

_SYSTEM_PROMPT = """\
You are a Google Calendar management specialist.

<task>
Create, search, update, and delete calendar events using the available tools.
</task>

<rules>
1. Format all dates and times in ISO 8601 (e.g. 2024-01-15T14:00:00).
2. Return a concise confirmation listing the event title, start/end times, and attendees.
</rules>
"""

_agent = None


def _build() -> Any:
    try:
        return create_agent(
            model=init_chat_model(
                api_key=Config.get_ai_api_key(),
                model=Config.TOOL_MODEL,
                temperature=0.2,
                max_tokens=Config.MAX_TOKENS,
                timeout=Config.TIMEOUT_SECONDS,
                max_retries=Config.MAX_RETRIES,
            ),
            tools=get_calendar_toolkit_tools(),
            system_prompt=_SYSTEM_PROMPT,
            name="Expert Calendar Agent",
        )
    except Exception as exc:
        logger.warning(f"[calendar_agent] build failed: {exc}")
        return None


@tool(
    description=(
        "Use for: creating, searching, updating, and deleting Google Calendar events.\n"
        "Provide: A natural-language request with complete time/date context.\n"
        "Returns: A confirmation or list of matching events.\n"
        "Example: calendar_agent_tool(request=\"Create 'Design Review' tomorrow at 2pm for 1 hour\")\n"
        'Example: calendar_agent_tool(request="What events do I have this Friday?")'
    ),
)
def calendar_agent_tool(request: str) -> str:
    """Manage Google Calendar: create, search, update, and delete events."""
    global _agent
    if _agent is None:
        _agent = _build()
    if _agent is None:
        return "Calendar agent is currently unavailable."
    return run_agent(_agent, request, "calendar_agent", "Calendar operation failed")
