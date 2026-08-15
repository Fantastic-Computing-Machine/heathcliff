# ABOUTME: Pure policy for deciding whether an action needs user approval.

import re

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

_ACTION_PREFIX = r"(?:^|[\"':]\s*|\b(?:please|kindly|can you|could you|would you)\s+)"

_DELEGATED_MUTATION_PATTERNS = {
    "email_agent_tool": re.compile(
        r"\b(send|compose|write|reply|respond|forward)\b"
        rf"|{_ACTION_PREFIX}(?:draft|email)\b"
        r"|\bcreate\s+(?:an?\s+)?draft\b",
        re.IGNORECASE,
    ),
    "calendar_agent_tool": re.compile(
        r"\b(create|add|book|update|edit|change|reschedule|move|delete|remove|cancel)\b"
        r"|\bset\s+up\b"
        rf"|{_ACTION_PREFIX}(?:schedule|invite)\b",
        re.IGNORECASE,
    ),
    "comms_agent_tool": re.compile(
        r"\b(send|notify|post|publish|reply|forward|share)\b"
        rf"|{_ACTION_PREFIX}(?:tell|message|text)\b",
        re.IGNORECASE,
    ),
}


def requires_approval(tool_name: str, tool_input: str = "") -> bool:
    """Return whether a direct or delegated tool call may mutate external state."""
    if tool_name in SENSITIVE_TOOLS:
        return True
    pattern = _DELEGATED_MUTATION_PATTERNS.get(tool_name)
    return bool(pattern and pattern.search(tool_input or ""))
