# ABOUTME: Approval-policy regressions for direct tools and delegated subagents
# ABOUTME: Uses exact tool and agent identities, never natural-language keyword matching

import pytest

from core.approval_handler import requires_approval, requires_serial_execution


@pytest.mark.parametrize(
    "tool_name",
    [
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
def test_delegated_agents_require_approval_independent_of_request_text(
    tool_name, tool_input
):
    assert requires_approval(tool_name, tool_input)


@pytest.mark.parametrize(
    ("tool_name", "tool_input"),
    [
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
def test_mutation_capable_delegated_agents_require_approval_for_reads_too(
    tool_name, tool_input
):
    assert requires_approval(tool_name, tool_input) == (
        tool_name in {"calendar_agent_tool", "comms_agent_tool"}
    )


def test_sensitive_tool_names_are_exact():
    assert not requires_approval("send_email")
    assert not requires_approval("email_agent_tool")
    assert not requires_approval("send_email_preview", "Send nothing")
    assert not requires_approval("SEND_GMAIL_MESSAGE")


def test_email_is_noninteractive_but_still_serialized():
    assert not requires_approval("email_agent_tool")
    assert requires_serial_execution("email_agent_tool")
