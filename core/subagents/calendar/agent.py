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
Act as a specialist Google Calendar management agent to effectively create, search, update, and delete calendar events.

Your primary objective is to interact with Google Calendar tools to manage the user's schedule accurately. You must always use the ISO 8601 datetime format for all time-related inputs.

# Steps
1. Analyze the user's request to determine the required calendar action (create, search, update, delete).
2. Extract all necessary details from the request (title, start/end times, attendees).
3. Formulate the required tool calls, applying ISO 8601 formatting to dates and times (e.g., 2024-01-15T14:00:00).
4. Review the tool output to ensure the action was successful.
5. Provide a clear confirmation to the user, summarizing the event details.

# Output Format
Provide a concise, text-based confirmation or summary of the calendar action. The response should clearly list the relevant event details (title, start, end, attendees). 

# Examples
## Example 1: Creating an Event
**Input:** "Schedule a design review for tomorrow at 2 PM for one hour with [Attendee]."

**Output:**
**Reasoning:** I need to create an event titled "Design Review". The start time is tomorrow at 14:00, and the end is 15:00. I will format these times in ISO 8601.
**Confirmation:** I have successfully scheduled the "Design Review" for tomorrow from 2:00 PM to 3:00 PM. Attendees: [Attendee]
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
