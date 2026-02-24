# ABOUTME: Calendar / Google Calendar sub-agent — create, search, update, delete events
# ABOUTME: Wraps tools/calendar_tools.py; exposed to supervisor as a single @tool

from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from config import Config
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
                api_key=Config.AI_KEY,
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
    try:
        logger.info(f"[calendar_agent] {request[:80]}")
        result = _agent.invoke({"messages": [{"role": "user", "content": request}]})

        messages = result.get("messages", [])
        if not messages:
            return "No response generated."

        last_msg = messages[-1]
        content = last_msg.content
        if isinstance(content, list):
            resp = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        else:
            resp = str(content) if content else ""

        resp = resp.strip()

        # Fallback: if AI yielded empty string, use the last tool's output
        if not resp:
            for msg in reversed(messages):
                if getattr(msg, "type", "") == "tool":
                    resp = str(msg.content)
                    break
            if not resp:
                resp = "Action completed, but no text response was generated."

        return resp
    except Exception as exc:
        logger.error(f"[calendar_agent] error: {exc}", exc_info=True)
        return f"Calendar operation failed: {exc}"
