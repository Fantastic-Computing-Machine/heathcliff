# ABOUTME: Shared Streamlit runtime for process-local agent controls and components.

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from config import Config
from core.agent_core import HeathcliffAgent
from core.runtime.http_client import RuntimeV2HttpClient
from core.runtime_profile import RuntimeProfile
from db.memory_manager import MemoryManager


@dataclass(frozen=True)
class AgentHandle:
    """An agent snapshot that remains valid after later profile changes."""

    agent: Any
    revision: int
    profile: RuntimeProfile


class AppRuntime:
    """Own the shared profile and profile-specific agent snapshots."""

    def __init__(self) -> None:
        self.memory = MemoryManager()
        self._lock = threading.RLock()
        self._profile = RuntimeProfile.defaults()
        self._revision = 0
        self._agents: dict[int, Any] = {}

    def snapshot(self) -> tuple[RuntimeProfile, int]:
        with self._lock:
            return self._profile, self._revision

    def apply(self, values: dict[str, Any]) -> tuple[RuntimeProfile, int]:
        profile = RuntimeProfile.from_values(values)
        with self._lock:
            if profile != self._profile:
                self._profile = profile
                self._revision += 1
            return self._profile, self._revision

    def reset_profile(self) -> tuple[RuntimeProfile, int]:
        with self._lock:
            self._profile = RuntimeProfile.defaults()
            self._revision += 1
            return self._profile, self._revision

    def current_agent(self) -> AgentHandle:
        with self._lock:
            agent = self._agents.get(self._revision)
            if agent is None:
                if Config.RUNTIME_V2_ENABLED:
                    agent = RuntimeV2HttpClient(Config.RUNTIME_V2_URL)
                else:
                    # ponytail: retain old snapshots only for in-process approval resumes.
                    HeathcliffAgent.reset()
                    agent = HeathcliffAgent(
                        memory_manager=self.memory,
                        runtime_profile=self._profile,
                        runtime_profile_revision=self._revision,
                    )
                self._agents[self._revision] = agent
            return AgentHandle(agent, self._revision, self._profile)

    def agent_for_revision(self, revision: int) -> AgentHandle:
        with self._lock:
            agent = self._agents.get(revision)
            if agent is None:
                raise RuntimeError(
                    "The approval's original agent is no longer available."
                )
            profile = (
                self._profile
                if revision == self._revision or Config.RUNTIME_V2_ENABLED
                else agent.runtime_profile
            )
            return AgentHandle(agent, revision, profile)
