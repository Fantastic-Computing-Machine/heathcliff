# ABOUTME: Email / Gmail sub-agent — search, read, draft, send
# ABOUTME: Wraps tools/gmail_tools.py; exposed to supervisor as a single @tool

from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from config import Config
from core.subagents._runner import run_agent
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
                api_key=Config.get_ai_api_key(),
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
    return run_agent(_agent, request, "email_agent", "Email operation failed")
