# ABOUTME: Runtime-profile validation and registry filtering regressions.

import pytest

from core.runtime_profile import (
    DEFAULT_AGENT_NAMES,
    RuntimeProfile,
    enabled_agent_names,
)


def test_runtime_profile_defaults_enable_every_registered_capability():
    profile = RuntimeProfile.defaults()

    assert profile.enabled_agents == DEFAULT_AGENT_NAMES


def test_runtime_profile_rejects_empty_capability_set():
    values = RuntimeProfile.defaults().__dict__.copy()
    values["enabled_agents"] = []

    with pytest.raises(ValueError, match="at least one"):
        RuntimeProfile.from_values(values)


def test_runtime_profile_rejects_invalid_total_runtime():
    values = RuntimeProfile.defaults().__dict__.copy()
    values["max_total_runtime_ms"] = 1

    with pytest.raises(ValueError, match="Total runtime"):
        RuntimeProfile.from_values(values)


def test_enabled_agent_names_reject_unknown_capabilities():
    with pytest.raises(ValueError, match="Unknown"):
        enabled_agent_names(["not-a-real-agent"])
