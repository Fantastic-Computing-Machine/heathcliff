# ABOUTME: Communications sub-agent — Telegram messaging
# ABOUTME: Wraps tools/comm_tools.py; exposed as a single @tool

from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from config import Config
from core.subagents.comms.tools import get_comm_tools
from logger import logger

# try:
#     from core.subagents.comms.tools import get_drive_tools
# except ImportError as drive_exc:
#     get_drive_tools = None
#     logger.warning(f"[comms_agent] drive tools unavailable: {drive_exc}")

_SYSTEM_PROMPT = """\
You are a Telegram messaging specialist.

<task>
Send messages and notifications via Telegram using the available tools.
</task>

<rules>
1. Extract the intended message content from the request.
2. Execute the Telegram tool and return a brief confirmation of what was sent.
</rules>
"""

_agent = None


def _build() -> Any:
    try:
        tools = list(get_comm_tools())
        # if get_drive_tools:
        #     try:
        #         tools.extend(get_drive_tools())
        #     except Exception as drive_exc:
        #         logger.warning(f"[comms_agent] drive tools error: {drive_exc}")

        return create_agent(
            model=init_chat_model(
                api_key=Config.AI_KEY,
                model=Config.TOOL_MODEL,
                temperature=0.6,
                max_tokens=Config.MAX_TOKENS,
                timeout=Config.TIMEOUT_SECONDS,
                max_retries=Config.MAX_RETRIES,
            ),
            tools=tools,
            system_prompt=_SYSTEM_PROMPT,
            name="Expert Communications Agent",
        )
    except Exception as exc:
        logger.warning(f"[comms_agent] build failed: {exc}")
        return None


@tool(
    description=(
        "Use for: sending messages and notifications via Telegram.\n"
        "Provide: A natural-language request with the message content.\n"
        "Returns: Confirmation of the message sent.\n"
        "Example: comms_agent_tool(request=\"Send a Telegram message: 'Build finished!'\")"
    ),
)
def comms_agent_tool(request: str) -> str:
    """Send Telegram messages and notifications."""
    global _agent
    if _agent is None:
        _agent = _build()
    if _agent is None:
        return "Communications agent is currently unavailable."
    try:
        logger.info(f"[comms_agent] {request[:80]}")
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
        logger.error(f"[comms_agent] error: {exc}", exc_info=True)
        return f"Communications failed: {exc}"
