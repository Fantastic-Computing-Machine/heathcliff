from typing import Any, List

from googleapiclient.discovery import build
from langchain_google_community import (
    CalendarCreateEvent,
    CalendarDeleteEvent,
    CalendarSearchEvents,
    CalendarUpdateEvent,
)
from utils.google_auth import get_google_credentials, CALENDAR_SCOPES

def _get_calendar_service():
    """Get authenticated Calendar API service."""
    creds = get_google_credentials(CALENDAR_SCOPES)
    return build("calendar", "v3", credentials=creds)

class CalendarTools:
    def __init__(self):
        # Initialize tools with account configurations (if required)
        self.service = _get_calendar_service()
        self.create_event_tool = CalendarCreateEvent(api_resource=self.service)
        self.get_date_events_tool = CalendarSearchEvents(api_resource=self.service)
        self.update_event_tool = CalendarUpdateEvent(api_resource=self.service)
        self.cancel_event_tool = CalendarDeleteEvent(api_resource=self.service)

    def create_event(
        self,
        summary: str,
        start_time: str,
        end_time: str,
        description: str = None,
        location: str = None,
        attendees: list = [],
    ):
        event_data = {
            "summary": summary,
            "start": start_time,
            "end": end_time,
            "description": description,
            "location": location,
            "attendees": attendees,
        }
        return self.create_event_tool.run(event_data)

    def get_date_events(self, date: str):  # Format: 'YYYY-MM-DD'
        return self.get_date_events_tool.run({"date": date})

    def get_upcoming_events(self):
        return self.get_date_events_tool.run({})  # Default fetches upcoming events

    def update_event(
        self,
        event_id: str,
        summary: str = None,
        start_time: str = None,
        end_time: str = None,
        description: str = None,
        location: str = None,
        attendees: list = [],
    ):
        update_data = {
            "event_id": event_id,
            "summary": summary,
            "start": start_time,
            "end": end_time,
            "description": description,
            "location": location,
            "attendees": attendees,
        }
        return self.update_event_tool.run(update_data)

    def cancel_event(self, event_id: str):
        return self.cancel_event_tool.run({"event_id": event_id})


# Initialize the tool instance for usage
calendar_tools = CalendarTools()


def get_calendar_toolkit_tools() -> List[Any]:
    """Expose Google Calendar LangChain tools as a list."""

    return [
        calendar_tools.create_event_tool,
        calendar_tools.get_date_events_tool,
        calendar_tools.update_event_tool,
        calendar_tools.cancel_event_tool,
    ]
