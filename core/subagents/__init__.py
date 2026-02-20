# ABOUTME: Domain-grouped subagents package for Heathcliff supervisor
# ABOUTME: Each subpackage (info, music, email, calendar, contacts, comms)
# ABOUTME: contains tools.py (raw LC tools) + agent.py (@tool wrapper for supervisor)

from core.subagents.info.agent import info_agent_tool
from core.subagents.music.agent import music_agent_tool
from core.subagents.email.agent import email_agent_tool
from core.subagents.calendar.agent import calendar_agent_tool
from core.subagents.contacts.agent import contacts_agent_tool
from core.subagents.comms.agent import comms_agent_tool
from typing import Any, List

__all__ = [
    "info_agent_tool",
    "music_agent_tool",
    "email_agent_tool",
    "calendar_agent_tool",
    "contacts_agent_tool",
    "comms_agent_tool",
    "get_all_subagent_tools",
]


def get_all_subagent_tools() -> List[Any]:
    """Return all domain sub-agent @tool wrappers for supervisor registration."""
    return [
        info_agent_tool,
        music_agent_tool,
        email_agent_tool,
        calendar_agent_tool,
        contacts_agent_tool,
        comms_agent_tool,
    ]
