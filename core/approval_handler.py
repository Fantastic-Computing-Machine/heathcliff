# ABOUTME: Pure policy for deciding whether an action needs user approval.

SENSITIVE_TOOLS = {
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

SENSITIVE_AGENTS = {"calendar_agent_tool", "comms_agent_tool"}
SERIAL_EXTERNAL_AGENTS = {
    "email_agent_tool",
    "calendar_agent_tool",
    "comms_agent_tool",
}


def requires_approval(tool_name: str, tool_input: str = "") -> bool:
    """Approve only interactions that retain a deliberate confirmation policy."""
    del tool_input
    return tool_name in SENSITIVE_TOOLS or tool_name in SENSITIVE_AGENTS


def requires_serial_execution(tool_name: str) -> bool:
    """Keep external operations in-process so a timed-out worker cannot outlive us."""
    return tool_name in SERIAL_EXTERNAL_AGENTS
