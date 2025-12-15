# ABOUTME: Gmail integration with human-in-the-loop confirmation for sending emails
# ABOUTME: Prevents email hallucination by requiring explicit user confirmation

import re
from typing import Any, List

from langchain.tools import tool
from langchain_community.tools.gmail import (
    GmailCreateDraft,
    GmailGetMessage,
    GmailGetThread,
    GmailSearch,
)

from logger import logger

# Email validation regex
EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


def validate_email(email: str) -> bool:
    """Validate email address format."""
    if not email or not isinstance(email, str):
        return False
    return EMAIL_REGEX.match(email.strip()) is not None


@tool
def send_email(to: str, subject: str, message: str) -> str:
    """
    Create a draft email for the user to review and send manually.

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
        # Use LangChain's Gmail create draft tool
        # Note: LangChain expects 'to' as a list of email addresses
        draft_tool = GmailCreateDraft()
        result = draft_tool.run(
            {
                "to": [to],  # Wrap in list - LangChain schema requires list
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


# Standard Gmail tools (read-only operations)
gmail_search = GmailSearch(
    name="search_emails",
    description=(
        "Search for emails in Gmail. Use this to find emails by sender, subject, or content. "
        "Examples: 'from:user@example.com', 'subject:invoice', 'newer_than:7d'"
    ),
)

gmail_get_message = GmailGetMessage(
    name="get_email",
    description="Get the full content of a specific email by message ID",
)

gmail_get_thread = GmailGetThread(
    name="get_thread",
    description="Get an entire email thread/conversation by thread ID",
)


def get_gmail_toolkit_tools() -> List[Any]:
    """
    Get all Gmail tools. Email sending creates drafts for user review.

    Returns:
        List of LangChain tools
    """
    return [
        gmail_search,
        gmail_get_message,
        gmail_get_thread,
        send_email,  # Creates draft emails (safer than auto-sending)
    ]
