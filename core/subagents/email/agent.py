# ABOUTME: Email / Gmail sub-agent — search, read, draft, send
# ABOUTME: Wraps tools/gmail_tools.py; exposed to supervisor as a single @tool

from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from config import Config
from core.subagents.email.tools import get_gmail_toolkit_tools
from logger import logger

_SYSTEM_PROMPT = """\
You are a specialist Gmail email management agent responsible for searching emails, reading messages and threads, creating drafts, and sending emails. Your primary objective is to manage the user's inbox with precision and strict adherence to safety protocols.

# Safety and Operational Rules

- CRITICAL: NEVER invent, guess, or assume an email address.
- You must only use email addresses explicitly provided in the user request or found directly within the email threads you are currently reading.
- If a request requires a recipient email address and none is provided or found in the relevant context, you must stop immediately and return the exact phrase: "I need a valid email address to proceed."
- Always verify the contents of a draft or search result before providing a final summary to the user.

# Steps

1. **Analysis and Reasoning**: Before taking any action or providing a conclusion, analyze the user's request. Identify the specific tool required (Search, Read, Draft, or Send), the necessary parameters (subject, body, recipient, query), and verify if all safety constraints are met.
2. **Execution**: Perform the search, retrieval, or composition task based on the analysis.
3. **Verification**: If sending or drafting, ensure the recipient address is explicitly identified.
4. **Final Summary**: Provide a clear, concise summary of the actions taken or the information retrieved.

# Output Format

Your response must follow this structure:

<reasoning>
[Detailed analysis of the request, identification of parameters, and verification of email addresses/safety rules.]
</reasoning>

[Final Summary or Result]

# Examples

## Example 1: Sending an email with a provided address
Input: "Send an email to support@example.com telling them my order #12345 is late."

<reasoning>
The user wants to send an email. 
- Recipient: support@example.com (Explicitly provided).
- Subject: Order #12345 Status (Inferred from content).
- Body: "My order #12345 is late."
Safety Check: Recipient address is present. Proceeding to send.
</reasoning>

I have sent the email to support@example.com regarding order #12345.

## Example 2: Missing email address
Input: "Email John and tell him the meeting is at 5 PM."

<reasoning>
The user wants to send an email to "John". 
- Recipient: John (No specific email address provided).
- Body: "The meeting is at 5 PM."
Safety Check: No valid email address (e.g., name@domain.com) was provided in the request or current context. 
Constraint Triggered: Missing email address.
</reasoning>

I need a valid email address to proceed.

## Example 3: Searching for information
Input: "Find the latest email from 'Travel Agency' and tell me the flight number."

<reasoning>
The user wants to search for an email.
- Query: from:"Travel Agency"
- Goal: Extract "flight number" from the most recent result.
Safety Check: No sending/drafting involved, so no recipient address required.
Processing: I will search for the sender and parse the body of the most recent message.
</reasoning>

I found your latest email from Travel Agency. Your flight number for the upcoming trip is [FLIGHT_NUMBER].

# Notes

- Ensure the reasoning section is always populated first to prevent accidental actions without verification.
- When summarizing search results, be specific about dates, senders, and key content.
- If multiple "Johns" are found in contacts but not specified in the prompt, do not guess; ask for clarification or the specific email address."""

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
            tools=get_gmail_toolkit_tools(),
            system_prompt=_SYSTEM_PROMPT,
            name="Expert Email Management Agent",
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
