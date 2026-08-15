# ABOUTME: Contacts / Google People sub-agent — look up email and phone numbers
# ABOUTME: Wraps tools/people_tools.py; exposed to supervisor as a single @tool
# ABOUTME: Called BEFORE email_agent_tool whenever an email address is unknown

from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from config import Config
from core.subagents._runner import run_agent
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
                api_key=Config.get_ai_api_key(),
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
    return run_agent(_agent, request, "contacts_agent", "Contacts lookup failed")
