# ABOUTME: Contacts / Google People sub-agent — look up email and phone numbers
# ABOUTME: Wraps tools/people_tools.py; exposed to supervisor as a single @tool
# ABOUTME: Called BEFORE email_agent_tool whenever an email address is unknown

from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from config import Config
from logger import logger

_SYSTEM_PROMPT = """\
You are a specialist Google Contacts lookup agent.
Your job: find email addresses, phone numbers, and contact details.

If a matching contact IS found: Return their name, email(s), and phone(s) clearly.
If NO contact is found: Return exactly this message:
  "No contact found for '[query]'. Please provide the email address directly."

Do NOT guess or invent contact information under any circumstances.
"""

_agent = None


def _build() -> Any:
    try:
        from core.subagents.contacts.tools import get_people_tools

        return create_agent(
            model=ChatGoogleGenerativeAI(
                model=Config.MODEL,
                google_api_key=Config.GEMINI_API_KEY,
                temperature=0.1,
                max_output_tokens=Config.MAX_TOKENS,
            ),
            tools=get_people_tools(),
            system_prompt=_SYSTEM_PROMPT,
        )
    except Exception as exc:
        logger.warning(f"[contacts_agent] build failed: {exc}")
        return None


@tool
def contacts_agent_tool(request: str) -> str:
    """Look up contacts from Google Contacts by name, email, or phone.

    Call this BEFORE email_agent_tool when you need someone's email address.
    If the result says "No contact found" — ask Adi for the email before proceeding.

    Input: Natural language contact lookup request.
    Example: "Find Philip's email address"
    Example: "What is the phone number for Sarah Johnson?"
    Example: "Look up contact details for John Smith"
    """
    global _agent
    if _agent is None:
        _agent = _build()
    if _agent is None:
        return "Contacts agent is currently unavailable."
    try:
        logger.info(f"[contacts_agent] {request[:80]}")
        result = _agent.invoke({"messages": [{"role": "user", "content": request}]})
        return result["messages"][-1].content
    except Exception as exc:
        logger.error(f"[contacts_agent] error: {exc}", exc_info=True)
        return f"Contacts lookup failed: {exc}"
