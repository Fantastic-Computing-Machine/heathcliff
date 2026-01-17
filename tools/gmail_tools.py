# ABOUTME: Gmail integration using LangChain's GmailToolkit with modifications
# ABOUTME: Uses GmailToolkit but replaces direct sending with draft creation for safety.

import re
from typing import Any, List

from googleapiclient.discovery import build
from langchain_core.tools import Tool
from langchain_google_community import GmailToolkit

from logger import logger
from utils.google_auth import GMAIL_SCOPES, get_google_credentials

# Email validation regex
EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


def validate_email(email: str) -> bool:
    """Validate email address format."""
    if not email or not isinstance(email, str):
        return False
    return EMAIL_REGEX.match(email.strip()) is not None


def _get_gmail_service():
    """Get authenticated Gmail API service."""
    creds = get_google_credentials(GMAIL_SCOPES)
    return build("gmail", "v1", credentials=creds)


_gmail_api_resource = None


def _get_api_resource():
    global _gmail_api_resource
    if not _gmail_api_resource:
        _gmail_api_resource = _get_gmail_service()
    return _gmail_api_resource


def create_email_draft_function(to: str, subject: str, message: str) -> str:
    """
    Creates a draft email for the user to review and send manually.

    ⚠️ CRITICAL SAFETY RULES - READ CAREFULLY:
    1. This tool creates DRAFTS, not sent emails - user must send manually
    2. NEVER hallucinate or invent email addresses
    3. ONLY use email addresses explicitly provided by the user
    4. If email address is unclear or not provided, STOP and ask: "What email address should this be for?"
    5. Validate email format before creating draft

    When user says "send an email", interpret this as "create a draft email".
    User will review and send the draft themselves from Gmail.

    Args:
        to: Recipient email address (MUST be explicitly provided by user)
        subject: Email subject line
        message: Email body content

    Returns:
        Success message confirming draft created, or error if email invalid
    """
    # Validate email format
    if not validate_email(to):
        logger.warning(f"Invalid email address provided: {to}")
        return (
            f"⚠️ Email address '{to}' appears invalid.\n\n"
            f"Please provide a valid email address. I cannot create a draft with an invalid address."
        )

    try:
        # We manually use the create_draft tool logic here or reuse the one from toolkit if available.
        # But for simplicity and control, we can reuse the toolkit's create draft tool if we can find it,
        # or just instantiate it manually.
        # Since we are inside the function, let's just use the helper we built.

        # Instantiate just the draft tool for this execution
        from langchain_community.tools.gmail import GmailCreateDraft

        draft_tool = GmailCreateDraft(api_resource=_get_api_resource())

        result = draft_tool.run(
            {
                "to": [to],
                "subject": subject,
                "message": message,
            }
        )

        logger.info(f"✅ Draft created successfully for {to}")
        return (
            f"✅ I've created a draft email to {to} with subject '{subject}'.\n"
            f"You can review and send it from your Gmail drafts."
        )

    except Exception as e:
        logger.error(f"Failed to create draft for {to}: {e}", exc_info=True)
        return f"❌ Failed to create draft: {str(e)}"


# Explicit Tool for creating drafts
create_email_draft_tool = Tool(
    name="create_email_draft",
    description=create_email_draft_function.__doc__,
    func=create_email_draft_function,
)


def get_gmail_toolkit_tools() -> List[Any]:
    """
    Get all Gmail tools from the toolkit, replacing send_message with create_email_draft.

    Returns:
        List of LangChain tools
    """
    # Initialize toolkit with our authenticated resource
    toolkit = GmailToolkit(api_resource=_get_api_resource())

    # Get all tools
    all_tools = toolkit.get_tools()

    # Filter out 'send_message' or similar dangerous tools
    # And 'create_draft' because we provide our own wrapper with validation/safety prompts
    safe_tools = [
        t
        for t in all_tools
        if "send_message" not in t.name and "create_draft" not in t.name
    ]

    # Add our custom draft tool
    safe_tools.append(create_email_draft_tool)

    return safe_tools
