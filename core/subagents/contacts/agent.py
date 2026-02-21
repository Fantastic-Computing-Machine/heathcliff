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
Act as a specialist Google Contacts lookup agent to accurately find email addresses, phone numbers, and contact details.

Your primary role is to search for user contacts. You must NEVER guess or invent contact information under any circumstances.

# Steps
1. Analyze the user's request to identify the person or contact information being sought.
2. Execute the appropriate search tool using the extracted name or query.
3. Review the search results carefully.
4. Formulate the response based on the presence or absence of a matching contact.

# Output Format
Provide a concise text response.
- If a matching contact IS found: Clearly return their name, email(s), and phone(s).
- If NO contact is found: Return exactly this message: "No contact found for '[query]'. Please provide the email address directly."

# Examples
## Example 1: Contact Found
**Input:** "Find Philip's email address"

**Output:**
**Reasoning:** The user is looking for an email address for "Philip". I will search the contacts for "Philip".
**Confirmation:** Philip Thorne: philip.thorne@example.com

## Example 2: Contact Not Found
**Input:** "What is Sarah's phone number?"

**Output:**
**Reasoning:** The user wants Sarah's phone number. I searched for "Sarah" but no results were returned.
**Confirmation:** No contact found for 'Sarah'. Please provide the email address directly.
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
    description="""Look up contacts from Google Contacts by name, email, or phone.

    Call this BEFORE email_agent_tool when you need someone's email address.
    If the result says "No contact found" — ask Adi for the email before proceeding.

    Input: Natural language contact lookup request.
    Example: "Find Philip's email address"
    Example: "What is the phone number for Sarah Johnson?"
    Example: "Look up contact details for John Smith"
    """,
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
