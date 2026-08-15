# ABOUTME: Capability registry mapping agent names to descriptors with invoke functions
# ABOUTME: Supplies registered agent metadata to the LLM planner

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from logger import logger


@dataclass
class AgentDescriptor:
    """Describes an agent's capabilities and how to invoke it."""

    name: str
    capabilities: List[str]
    invoke_fn: Callable[..., Any]
    locality: str = "local"  # "local" | "remote" | "deep"
    timeout_ms: int = 60000
    sensitive_actions: List[str] = field(default_factory=list)


class CapabilityRegistry:
    """Registry of agent descriptors supplied to the LLM planner."""

    def __init__(self) -> None:
        self._agents: Dict[str, AgentDescriptor] = {}

    def register(self, descriptor: AgentDescriptor) -> None:
        """Register an agent descriptor."""
        self._agents[descriptor.name] = descriptor
        logger.debug(
            "Registered agent %r with capabilities: %s",
            descriptor.name,
            descriptor.capabilities,
        )

    def get(self, name: str) -> Optional[AgentDescriptor]:
        """Get a descriptor by agent name."""
        return self._agents.get(name)

    def all_agents(self) -> List[AgentDescriptor]:
        """Return all registered agent descriptors."""
        return list(self._agents.values())

    def agent_names(self) -> List[str]:
        """Return all registered agent names."""
        return list(self._agents.keys())


def build_default_registry() -> CapabilityRegistry:
    """Build registry populated with all local subagent descriptors.

    Lazily imports subagents and skills to avoid circular imports.
    """
    from core.subagents.calendar.agent import calendar_agent_tool
    from core.subagents.comms.agent import comms_agent_tool
    from core.subagents.contacts.agent import contacts_agent_tool
    from core.subagents.email.agent import email_agent_tool
    from core.subagents.info.agent import info_agent_tool
    from core.subagents.info.recent_context import recent_context
    from core.subagents.music.agent import music_agent_tool
    from skills.skill_tools import get_skill_tools

    registry = CapabilityRegistry()

    descriptors = [
        AgentDescriptor(
            name="info_agent_tool",
            capabilities=[
                "search",
                "weather",
                "news",
                "wikipedia",
                "research",
                "web",
                "yahoo",
                "youtube",
                "url",
                "read",
                "information",
            ],
            invoke_fn=info_agent_tool,
            sensitive_actions=[],
        ),
        AgentDescriptor(
            name="music_agent_tool",
            capabilities=["music", "spotify", "play", "song", "track", "playlist"],
            invoke_fn=music_agent_tool,
            sensitive_actions=[],
        ),
        AgentDescriptor(
            name="email_agent_tool",
            capabilities=["email", "gmail", "mail", "inbox", "send email", "draft"],
            invoke_fn=email_agent_tool,
            sensitive_actions=["send_email", "GmailSendMessage", "GmailCreateDraft"],
        ),
        AgentDescriptor(
            name="calendar_agent_tool",
            capabilities=[
                "calendar",
                "event",
                "schedule",
                "meeting",
                "appointment",
            ],
            invoke_fn=calendar_agent_tool,
            sensitive_actions=[
                "create_event",
                "update_event",
                "cancel_event",
            ],
        ),
        AgentDescriptor(
            name="contacts_agent_tool",
            capabilities=["contact", "phone", "address", "person", "people"],
            invoke_fn=contacts_agent_tool,
            sensitive_actions=[],
        ),
        AgentDescriptor(
            name="comms_agent_tool",
            capabilities=[
                "telegram",
                "message",
                "send message",
                "communicate",
                "text",
            ],
            invoke_fn=comms_agent_tool,
            sensitive_actions=["send_to_telegram"],
        ),
        AgentDescriptor(
            name="recent_context",
            capabilities=["recent", "context", "previous", "earlier", "prior"],
            invoke_fn=recent_context,
            sensitive_actions=[],
        ),
    ]

    for desc in descriptors:
        registry.register(desc)

    # Register skill tools
    try:
        for skill_tool in get_skill_tools():
            name = getattr(skill_tool, "name", str(skill_tool))
            registry.register(
                AgentDescriptor(
                    name=name,
                    capabilities=[name.replace("_", " ")],
                    invoke_fn=skill_tool,
                    sensitive_actions=[],
                )
            )
    except Exception as exc:
        logger.warning("Failed to register skill tools: %s", exc)

    logger.info(
        "Capability registry built with %d agents: %s",
        len(registry.agent_names()),
        registry.agent_names(),
    )
    return registry
