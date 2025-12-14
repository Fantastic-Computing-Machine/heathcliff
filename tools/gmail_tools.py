from typing import Any, List

from langchain_community.tools.gmail import (
    GmailCreateDraft,
    GmailGetMessage,
    GmailGetThread,
    GmailSearch,
    GmailSendMessage,
)

gmail_tools = [
    GmailSearch(name="search_emails", description="Search for emails in Gmail"),
    GmailGetMessage(
        name="get_email", description="Get the content of a specific email"
    ),
    GmailCreateDraft(name="create_draft", description="Create a draft email"),
    GmailSendMessage(name="send_email", description="Send an email"),
    GmailGetThread(name="get_thread", description="Get an email thread"),
]

def get_gmail_toolkit_tools() -> List[Any]:
    """Expose LangChain Gmail tools for registry consumption."""

    return list(gmail_tools)
