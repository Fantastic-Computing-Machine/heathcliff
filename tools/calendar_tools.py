# ABOUTME: Calendar integration using LangChain's CalendarToolkit
# ABOUTME: Provides full suite of Google Calendar tools (Create, Search, Update, Delete)

from typing import Any, List

from googleapiclient.discovery import build
from langchain_google_community.calendar.create_event import CalendarCreateEvent
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
    create_calendar_event.description = (
        "Create a new Google Calendar event. "
        "Requires: summary (title), start_datetime, end_datetime in 'YYYY-MM-DD HH:MM:SS' format. "
        "Default duration is 1 hour if end time not specified. "
        "Returns the created event details including event_id."
    )

    search_events = CalendarSearchEvents(api_resource=api_resource)
    search_events.description = (
        "Search for calendar events within a time range. "
        "WORKFLOW: First call get_calendars_info, then use its output as calendars_info parameter. "
        "Parameters: calendars_info (JSON from get_calendars_info), "
        "min_datetime and max_datetime in 'YYYY-MM-DD HH:MM:SS' format, "
        "optional query for title search. "
        "Returns list of events with 'id', 'summary', 'start', 'end'. "
        "Use the 'id' field for delete or update operations."
    )

    delete_event = CalendarDeleteEvent(api_resource=api_resource)
    delete_event.description = (
        "Delete a calendar event by its event_id. "
        "WORKFLOW: First search for the event using search_events to get the event_id, "
        "then call this with that event_id. "
        "Parameters: event_id (required), calendar_id (default 'primary')."
    )

    update_event = CalendarUpdateEvent(api_resource=api_resource)
    update_event.description = (
        "Update an existing calendar event. "
        "WORKFLOW: First search for the event to get event_id, then update. "
        "Parameters: event_id (required), plus any fields to update (summary, start, end, etc)."
    )

    get_calendars_info = GetCalendarsInfo(api_resource=api_resource)
    get_calendars_info.description = (
        "Get list of user's calendars with their IDs and timezones. "
        "MUST be called BEFORE search_events. The output is used as calendars_info parameter. "
        "Returns JSON with calendar id, summary, and timeZone."
    )

    tools.append(create_calendar_event)
    tools.append(search_events)
    tools.append(update_event)
    tools.append(get_calendars_info)
    tools.append(CalendarMoveEvent(api_resource=api_resource))
    tools.append(delete_event)

    return tools
