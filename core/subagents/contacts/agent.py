# ABOUTME: Contacts / Google People sub-agent — look up email and phone numbers
# ABOUTME: Wraps tools/people_tools.py; exposed to supervisor as a single @tool
# ABOUTME: Called BEFORE email_agent_tool whenever an email address is unknown

from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from config import Config
from core.subagents.contacts.tools import get_people_tools
from logger import logger

_SYSTEM_PROMPT = """\
You are a Google Contacts lookup specialist.

<task>
Search for contact details (email addresses, phone numbers) using the available tools.
</task>

<rules>
1. Only return contact information found in the search results. If no match is found, return: "No contact found for '[query]'. Please provide the email address directly."
2. When a contact is found, return their name, email(s), and phone(s) clearly.
</rules>
"""

_agent = None


def _build() -> Any:
    try:
        return create_agent(
            model=init_chat_model(
                api_key=Config.AI_KEY,
                model=Config.TOOL_MODEL,
                temperature=0.4,
                max_tokens=Config.MAX_TOKENS,
                timeout=Config.TIMEOUT_SECONDS,
                max_retries=Config.MAX_RETRIES,
            ),
            tools=get_people_tools(),
            system_prompt=_SYSTEM_PROMPT,
            name="Expert Contacts Agent",
        )
    except Exception as exc:
        logger.warning(f"[contacts_agent] build failed: {exc}")
        return None


@tool(
    description=(
        "Use for: looking up email addresses, phone numbers, and contact details "
        "from Google Contacts.\n"
        "Provide: A name or query to search for.\n"
        "Returns: Contact name, email(s), and phone(s), or a 'No contact found' message.\n"
        'Example: contacts_agent_tool(request="Find Philip\'s email address")\n'
        "Tip: Call this before email_agent_tool when you need a recipient address. "
        "If the result says 'No contact found', ask the user for the email."
    ),
)
def contacts_agent_tool(request: str) -> str:
    """Look up contacts from Google Contacts by name, email, or phone."""
    global _agent
    if _agent is None:
        _agent = _build()
    if _agent is None:
        return "Contacts agent is currently unavailable."
    try:
        logger.info(f"[contacts_agent] {request[:80]}")
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
        logger.error(f"[contacts_agent] error: {exc}", exc_info=True)
        return f"Contacts lookup failed: {exc}"
