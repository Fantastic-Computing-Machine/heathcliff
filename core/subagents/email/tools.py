# ABOUTME: Gmail integration via LangChain Gmail toolkit
# ABOUTME: Provides search, read, draft tools using Google OAuth credentials

from typing import Any, List

from langchain_community.agent_toolkits import GmailToolkit
from utils.google_auth import get_google_credentials, GMAIL_SCOPES
from googleapiclient.discovery import build


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


def get_gmail_toolkit_tools() -> List[Any]:
    """Get all Gmail tools from the LangChain toolkit.

    Returns:
        List of LangChain tools (search, read, get thread, create draft)
    """
    toolkit = GmailToolkit(api_resource=_get_api_resource())
    return toolkit.get_tools()
