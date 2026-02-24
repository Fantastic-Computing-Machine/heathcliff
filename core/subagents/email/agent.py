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
You are a Gmail email management specialist responsible for searching, reading, drafting, and sending emails.

<rules>
1. Only use email addresses explicitly provided in the request or found in the email threads you read. If a send/draft requires a recipient and none is available, return: "I need a valid email address to proceed."
2. Verify draft contents and search results before returning a final summary.
3. Be specific about dates, senders, and key content when summarising search results.
</rules>

<workflow>
1. Identify the action (search, read, draft, send) and extract required parameters.
2. For send/draft: confirm a valid recipient address exists before executing.
3. Execute the tool and return a concise summary of the result.
</workflow>
"""

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


@tool(
    description=(
        "Use for: searching, reading, drafting, and sending Gmail emails.\n"
        "Provide: A natural-language email request. For send/draft, include the "
        "recipient's email address; for search/read, an address is not required.\n"
        "Returns: A summary of the action taken or information retrieved.\n"
        'Example: email_agent_tool(request="Send an email to philip@example.com: '
        "subject 'Meeting tomorrow', body '...'\")\n"
        'Example: email_agent_tool(request="Search for emails from my manager this week")\n'
        "Tip: If you need a recipient address, call contacts_agent_tool first."
    ),
)
def email_agent_tool(request: str) -> str:
    """Manage Gmail: search, read, draft, and send emails."""
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
