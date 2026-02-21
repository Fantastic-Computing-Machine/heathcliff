# ABOUTME: Communications sub-agent — Telegram messaging and Google Drive files
# ABOUTME: Wraps tools/comm_tools.py + tools/drive_tools.py; exposed as a single @tool

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
Act as a specialist communications agent to send Telegram messages and read/access Google Drive files.

Your role is to accurately dispatch messages to users via Telegram and retrieve contents from Google Drive as requested.

# Steps
1. Analyze the user's request to determine if it is a messaging task (Telegram) or a file access task (Drive).
2. For Telegram: Extract the intended message content and recipient (if applicable).
3. For Drive: Extract the target filename or query parameters.
4. Execute the appropriate tool call.
5. Verify the tool execution was successful and confirm the message sent or the file data retrieved.

# Output Format
Provide a concise text response. If a message was sent, confirm what was sent. If a file was retrieved, present the file's contents clearly to the user.

# Examples
## Example 1: Sending a Telegram Message
**Input:** "Send a Telegram message saying 'Build finished!'"

**Output:**
**Reasoning:** The user wants to send a Telegram notification. The message body is "Build finished!". I will invoke the Telegram tool with this exact text.
**Confirmation:** The message "Build finished!" has been sent successfully via Telegram.
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


@tool
def comms_agent_tool(request: str) -> str:
    """Send Telegram messages or access Google Drive files.

    Use for:
    - Sending a notification or message via Telegram
    - Reading a file or document from Google Drive

    Input: Full natural-language communications request.
    Example: "Send a Telegram message: 'Build finished successfully'"
    Example: "Read the file 'project_notes.txt' from Google Drive"
    """
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
