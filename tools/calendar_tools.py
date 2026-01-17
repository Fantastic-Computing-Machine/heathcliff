# ABOUTME: Calendar integration using LangChain's CalendarToolkit
# ABOUTME: Provides full suite of Google Calendar tools (Create, Search, Update, Delete)

from typing import Any, List

from googleapiclient.discovery import build
from langchain_google_community.calendar.create_event import CalendarCreateEvent
from langchain_google_community.calendar.current_datetime import GetCurrentDatetime
from langchain_google_community.calendar.delete_event import CalendarDeleteEvent
from langchain_google_community.calendar.get_calendars_info import GetCalendarsInfo
from langchain_google_community.calendar.move_event import CalendarMoveEvent
from langchain_google_community.calendar.search_events import CalendarSearchEvents
from langchain_google_community.calendar.update_event import CalendarUpdateEvent

from utils.google_auth import CALENDAR_SCOPES, get_google_credentials

_calendar_api_resource = None


def _get_calendar_service():
    """Get authenticated Calendar API service."""
    creds = get_google_credentials(CALENDAR_SCOPES)
    return build("calendar", "v3", credentials=creds)


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

    tools = []
    api_resource = _get_api_resource()

    create_calendar_event = CalendarCreateEvent(api_resource=api_resource)
    create_calendar_event.description += (
        " Note: Default event duration is 1 hour if end time is not specified."
    )

    tools.append(create_calendar_event)
    tools.append(CalendarSearchEvents(api_resource=api_resource))
    tools.append(CalendarUpdateEvent(api_resource=api_resource))
    tools.append(GetCalendarsInfo(api_resource=api_resource))
    tools.append(CalendarMoveEvent(api_resource=api_resource))
    tools.append(CalendarDeleteEvent(api_resource=api_resource))

    return tools
