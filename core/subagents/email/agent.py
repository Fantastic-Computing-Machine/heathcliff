# ABOUTME: Email / Gmail sub-agent — search, read, draft, send
# ABOUTME: Wraps tools/gmail_tools.py; exposed to supervisor as a single @tool

from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from config import Config
from logger import logger

_SYSTEM_PROMPT = """\
You are a specialist Gmail email management agent.
Your job: search email, read messages and threads, create drafts, and send emails.

CRITICAL SAFETY RULES:
- NEVER invent, guess, or assume an email address
- Only use addresses explicitly given in the request
- If no address is provided, return: "I need a valid email address to proceed."

Return a clear summary of what was done or found.
"""

_agent = None


def _build() -> Any:
    try:
        from core.subagents.email.tools import get_gmail_toolkit_tools

        return create_agent(
            model=ChatGoogleGenerativeAI(
                model=Config.MODEL,
                google_api_key=Config.GEMINI_API_KEY,
                temperature=0.2,
                max_output_tokens=Config.MAX_TOKENS,
            ),
            tools=get_gmail_toolkit_tools(),
            system_prompt=_SYSTEM_PROMPT,
        )
    except Exception as exc:
        logger.warning(f"[email_agent] build failed: {exc}")
        return None


@tool
def email_agent_tool(request: str) -> str:
    """Manage Gmail: search, read, draft, and send emails.

    Use for all email tasks:
    - Search emails from a person or about a topic
    - Read a specific email or thread
    - Draft or send an email

    IMPORTANT: The request MUST include the recipient's email address.
    If you don't have it, call contacts_agent_tool first.

    Input: Full natural-language email request including recipient address.
    Example: "Send an email to philip@example.com: subject 'Meeting tomorrow', body '...'"
    Example: "Search for emails from my manager received this week"
    """
    global _agent
    if _agent is None:
        _agent = _build()
    if _agent is None:
        return "Email agent is currently unavailable."
    try:
        logger.info(f"[email_agent] {request[:80]}")
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
        logger.error(f"[email_agent] error: {exc}", exc_info=True)
        return f"Email operation failed: {exc}"
