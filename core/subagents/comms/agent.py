# ABOUTME: Communications sub-agent — Telegram messaging
# ABOUTME: Wraps tools/comm_tools.py; exposed as a single @tool

from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from config import Config
from core.runtime_profile import current_tool_model
from core.subagents._runner import run_agent
from core.subagents.comms.tools import get_comm_tools
from logger import logger

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

_agents: dict[str, Any] = {}
# Compatibility seam for existing integrations and unit tests.
_agent = None


def _build(model_name: str) -> Any:
    try:
        return create_agent(
            model=init_chat_model(
                api_key=Config.get_ai_api_key(),
                model=model_name,
                temperature=0.6,
                max_tokens=Config.MAX_TOKENS,
                timeout=Config.TIMEOUT_SECONDS,
                max_retries=Config.MAX_RETRIES,
            ),
            tools=get_comm_tools(),
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
    model_name = current_tool_model(Config.TOOL_MODEL)
    if _agent is not None:
        agent = _agent
    elif model_name not in _agents:
        _agents[model_name] = _build(model_name)
        agent = _agents[model_name]
    else:
        agent = _agents[model_name]
    if agent is None:
        return "Communications agent is currently unavailable."
    return run_agent(agent, request, "comms_agent", "Communications failed")
