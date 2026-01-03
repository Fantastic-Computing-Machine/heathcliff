# ABOUTME: Calendar integration using LangChain's CalendarToolkit
# ABOUTME: Provides full suite of Google Calendar tools (Create, Search, Update, Delete)

from typing import Any, List

from langchain_google_community import CalendarToolkit

from googleapiclient.discovery import build
from utils.google_auth import get_google_credentials, CALENDAR_SCOPES


def _get_calendar_service():
    """Get authenticated Calendar API service."""
    creds = get_google_credentials(CALENDAR_SCOPES)
    return build("calendar", "v3", credentials=creds)


_calendar_api_resource = None


def _get_api_resource():
    global _calendar_api_resource
    if not _calendar_api_resource:
        _calendar_api_resource = _get_calendar_service()
    return _calendar_api_resource


def get_calendar_toolkit_tools() -> List[Any]:
    """
    Get all Google Calendar tools from the toolkit.

    Returns:
        List of LangChain tools
    """
    # Initialize toolkit with our authenticated resource
    toolkit = CalendarToolkit(api_resource=_get_api_resource())

    return toolkit.get_tools()
