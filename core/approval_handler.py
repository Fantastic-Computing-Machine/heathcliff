# ABOUTME: Pure policy for deciding whether an action needs user approval.

SENSITIVE_TOOLS = {
    "send_email",
    "GmailSendMessage",
    "GmailCreateDraft",
    "send_gmail_message",
    "create_gmail_draft",
    "create_event",
    "GoogleCalendarCreateTool",
    "create_calendar_event",
    "update_event",
    "GoogleCalendarUpdateTool",
    "update_calendar_event",
    "move_calendar_event",
    "cancel_event",
    "GoogleCalendarDeleteTool",
    "delete_calendar_event",
    "send_to_telegram",
}

SENSITIVE_AGENTS = {"email_agent_tool", "calendar_agent_tool", "comms_agent_tool"}


def requires_approval(tool_name: str, tool_input: str = "") -> bool:
    """Approve exact mutation tools and all mutation-capable delegated agents."""
    del tool_input
    return tool_name in SENSITIVE_TOOLS or tool_name in SENSITIVE_AGENTS
