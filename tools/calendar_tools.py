from langchain_community.tools.calendar import (
    GoogleCalendarCreateTool,
    GoogleCalendarGetTool,
    GoogleCalendarUpdateTool,
    GoogleCalendarDeleteTool,
)


class CalendarTools:
    def __init__(self):
        # Initialize tools with account configurations (if required)
        self.create_event_tool = GoogleCalendarCreateTool()
        self.get_date_events_tool = GoogleCalendarGetTool()
        self.update_event_tool = GoogleCalendarUpdateTool()
        self.cancel_event_tool = GoogleCalendarDeleteTool()

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
