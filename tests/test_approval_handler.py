# ABOUTME: Approval-policy regressions for direct tools and delegated subagents
# ABOUTME: Ensures mutations pause while read-only requests continue without approval

import pytest

from core.approval_handler import requires_approval


@pytest.mark.parametrize(
    "tool_name",
    [
        "send_email",
        "send_gmail_message",
        "create_gmail_draft",
        "create_calendar_event",
        "update_calendar_event",
        "move_calendar_event",
        "delete_calendar_event",
        "send_to_telegram",
    ],
)
def test_exact_sensitive_inner_tools_require_approval(tool_name):
    assert requires_approval(tool_name)


@pytest.mark.parametrize(
    ("tool_name", "tool_input"),
    [
        ("email_agent_tool", "Send an email to alex@example.com"),
        ("email_agent_tool", "Draft a reply to the latest message"),
        ("email_agent_tool", "Email Alice the report"),
        ("calendar_agent_tool", "Create a calendar event tomorrow at 2pm"),
        ("calendar_agent_tool", "Schedule lunch tomorrow at noon"),
        ("calendar_agent_tool", "Set up a meeting with Alice tomorrow"),
        ("calendar_agent_tool", "Invite Alice to the design review"),
        ("calendar_agent_tool", "Reschedule the design review to Friday"),
        ("calendar_agent_tool", "Cancel my 3pm appointment"),
        ("comms_agent_tool", "Send a Telegram message that the build finished"),
        ("comms_agent_tool", "Tell Alice the build finished"),
        ("comms_agent_tool", "Message Alex that the build finished"),
        ("comms_agent_tool", "Notify me on Telegram when the task completes"),
        ("comms_agent_tool", "Share the report on Telegram"),
        ("comms_agent_tool", "Forward this on Telegram"),
    ],
)
def test_delegated_mutations_require_approval(tool_name, tool_input):
    assert requires_approval(tool_name, tool_input)


@pytest.mark.parametrize(
    ("tool_name", "tool_input"),
    [
        ("email_agent_tool", "Search for emails from my manager this week"),
        ("email_agent_tool", "Read the latest email from alex@example.com"),
        ("email_agent_tool", "Search for draft emails from this week"),
        ("email_agent_tool", "Get the draft email I saved yesterday"),
        ("calendar_agent_tool", "What events are on my calendar this Friday?"),
        ("calendar_agent_tool", "Show my schedule for tomorrow"),
        ("calendar_agent_tool", "Search for the design review event"),
        ("calendar_agent_tool", "Check the invitation from Alice"),
        ("calendar_agent_tool", "List canceled events from last month"),
        ("comms_agent_tool", "Read a Telegram message from Alex"),
        ("comms_agent_tool", "Read my recent Telegram messages"),
        ("comms_agent_tool", "Search Telegram messages for the build result"),
        ("comms_agent_tool", "Find the message Alice forwarded yesterday"),
        ("info_agent_tool", "Search the web for calendar apps"),
    ],
)
def test_read_only_delegated_requests_do_not_require_approval(tool_name, tool_input):
    assert not requires_approval(tool_name, tool_input)


def test_sensitive_tool_names_are_exact():
    assert not requires_approval("send_email_preview", "Send nothing")
    assert not requires_approval("SEND_GMAIL_MESSAGE")
