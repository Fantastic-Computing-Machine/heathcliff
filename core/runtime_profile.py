# ABOUTME: Process-local, non-secret runtime tuning for Heathcliff operators.

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Generator, Iterable

from config import Config

DEFAULT_AGENT_NAMES = (
    "info_agent_tool",
    "music_agent_tool",
    "email_agent_tool",
    "calendar_agent_tool",
    "contacts_agent_tool",
    "comms_agent_tool",
    "recent_context",
    "load_skill",
    "update_master_info",
)

_tool_model_override: ContextVar[str | None] = ContextVar(
    "heathcliff_tool_model_override", default=None
)


@contextmanager
def use_tool_model(model: str) -> Generator[None, None, None]:
    """Make a profile's tool model available while one specialist executes."""
    token = _tool_model_override.set(model)
    try:
        yield
    finally:
        _tool_model_override.reset(token)


def current_tool_model(default: str) -> str:
    """Return the profile-bound model, or the deployment default outside a run."""
    return _tool_model_override.get() or default


@dataclass(frozen=True)
class RuntimeProfile:
    """Validated, non-secret values applied to newly created agent instances."""

    supervisor_model: str
    tool_model: str
    temperature: float
    max_tokens: int
    max_tasks: int
    per_task_timeout_ms: int
    max_total_runtime_ms: int
    enabled_agents: tuple[str, ...]

    @classmethod
    def defaults(cls) -> "RuntimeProfile":
        return cls(
            supervisor_model=Config.SUPERVISOR_MODEL,
            tool_model=Config.TOOL_MODEL,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
            max_tasks=Config.MAX_TASKS_PER_REQUEST,
            per_task_timeout_ms=Config.PER_TASK_TIMEOUT_MS,
            max_total_runtime_ms=Config.MAX_TOTAL_RUNTIME_MS,
            enabled_agents=DEFAULT_AGENT_NAMES,
        )

    @classmethod
    def from_values(cls, values: dict[str, Any]) -> "RuntimeProfile":
        enabled = tuple(
            name for name in DEFAULT_AGENT_NAMES if name in values["enabled_agents"]
        )
        profile = cls(
            supervisor_model=str(values["supervisor_model"]).strip(),
            tool_model=str(values["tool_model"]).strip(),
            temperature=float(values["temperature"]),
            max_tokens=int(values["max_tokens"]),
            max_tasks=int(values["max_tasks"]),
            per_task_timeout_ms=int(values["per_task_timeout_ms"]),
            max_total_runtime_ms=int(values["max_total_runtime_ms"]),
            enabled_agents=enabled,
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        if not self.supervisor_model or not self.tool_model:
            raise ValueError("Supervisor and tool model IDs are required.")
        if not 0 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2.")
        if not 256 <= self.max_tokens <= 32768:
            raise ValueError("Max tokens must be between 256 and 32,768.")
        if not 1 <= self.max_tasks <= 20:
            raise ValueError("Maximum tasks must be between 1 and 20.")
        if not 5_000 <= self.per_task_timeout_ms <= 600_000:
            raise ValueError("Per-task timeout must be between 5 and 600 seconds.")
        if not self.per_task_timeout_ms <= self.max_total_runtime_ms <= 900_000:
            raise ValueError(
                "Total runtime must be at least the task timeout and at most 900 seconds."
            )
        if not self.enabled_agents:
            raise ValueError("Enable at least one agent capability.")

    def metadata(self, revision: int) -> dict[str, str]:
        return {
            "runtime_profile_revision": str(revision),
            "supervisor_model": self.supervisor_model,
            "tool_model": self.tool_model,
            "enabled_agent_count": str(len(self.enabled_agents)),
        }


def enabled_agent_names(names: Iterable[str] | None) -> set[str] | None:
    """Normalize an optional agent allow-list without accepting unknown names."""
    if names is None:
        return None
    allowed = set(names)
    unknown = allowed.difference(DEFAULT_AGENT_NAMES)
    if unknown:
        raise ValueError(f"Unknown agent capabilities: {', '.join(sorted(unknown))}")
    return allowed
