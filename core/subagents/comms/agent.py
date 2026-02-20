# ABOUTME: Communications sub-agent — Telegram messaging and Google Drive files
# ABOUTME: Wraps tools/comm_tools.py + tools/drive_tools.py; exposed as a single @tool

from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from config import Config
from logger import logger

_SYSTEM_PROMPT = """\
You are a specialist communications agent.
Your job: send Telegram messages and read/access Google Drive files.
Confirm message content before sending.
Return a clear confirmation or the file contents retrieved.
"""

_agent = None


def _build() -> Any:
    try:
        from core.subagents.comms.tools import get_comm_tools

        tools = list(get_comm_tools())
        try:
            from core.subagents.comms.tools import get_drive_tools

            tools.extend(get_drive_tools())
        except Exception as drive_exc:
            logger.warning(f"[comms_agent] drive tools unavailable: {drive_exc}")
        return create_agent(
            model=ChatGoogleGenerativeAI(
                model=Config.MODEL,
                google_api_key=Config.GEMINI_API_KEY,
                temperature=0.2,
                max_output_tokens=Config.MAX_TOKENS,
            ),
            tools=tools,
            system_prompt=_SYSTEM_PROMPT,
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
