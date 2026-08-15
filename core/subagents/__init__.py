# ABOUTME: Domain subagent tool wrappers.

from core.subagents.calendar.agent import calendar_agent_tool
from core.subagents.comms.agent import comms_agent_tool
from core.subagents.contacts.agent import contacts_agent_tool
from core.subagents.email.agent import email_agent_tool
from core.subagents.info.agent import info_agent_tool
from core.subagents.info.recent_context import recent_context
from core.subagents.music.agent import music_agent_tool

__all__ = [
    "info_agent_tool",
    "music_agent_tool",
    "email_agent_tool",
    "calendar_agent_tool",
    "contacts_agent_tool",
    "comms_agent_tool",
    "recent_context",
]
