# ABOUTME: Calendar / Google Calendar sub-agent — create, search, update, delete events
# ABOUTME: Wraps tools/calendar_tools.py; exposed to supervisor as a single @tool

from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from config import Config
from logger import logger

_SYSTEM_PROMPT = """\
You are a specialist Google Calendar management agent.
Your job: create, search, update, and delete calendar events.
Always use ISO 8601 datetime format for times (e.g. 2024-01-15T14:00:00).
When creating events confirm: title, start, end, and attendees (if any).
Return a clear confirmation with event details.
"""

_agent = None


def _build() -> Any:
    try:
        from core.subagents.calendar.tools import get_calendar_toolkit_tools

        return create_agent(
            model=ChatGoogleGenerativeAI(
                model=Config.MODEL,
                google_api_key=Config.GEMINI_API_KEY,
                temperature=0.2,
                max_output_tokens=Config.MAX_TOKENS,
            ),
            tools=get_calendar_toolkit_tools(),
            system_prompt=_SYSTEM_PROMPT,
        )
    except Exception as exc:
        logger.warning(f"[calendar_agent] build failed: {exc}")
        return None


@tool
def calendar_agent_tool(request: str) -> str:
    """Manage Google Calendar: create, search, update, and delete events.

    Use for all calendar tasks:
    - Check what's on the calendar today or this week
    - Create a new event or meeting
    - Update or reschedule an existing event
    - Delete or cancel an event

    Input: Full natural-language request with complete time/date context.
    Example: "Create 'Design Review' tomorrow at 2pm for 1 hour"
    Example: "What events do I have this Friday?"
    Example: "Move my 3pm meeting to 4pm"
    """
    global _agent
    if _agent is None:
        _agent = _build()
    if _agent is None:
        return "Calendar agent is currently unavailable."
    try:
        logger.info(f"[calendar_agent] {request[:80]}")
        result = _agent.invoke({"messages": [{"role": "user", "content": request}]})
        return result["messages"][-1].content
    except Exception as exc:
        logger.error(f"[calendar_agent] error: {exc}", exc_info=True)
        return f"Calendar operation failed: {exc}"
