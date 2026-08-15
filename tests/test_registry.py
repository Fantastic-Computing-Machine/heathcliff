# ABOUTME: Unit tests for CapabilityRegistry — registration, resolution, agent lookup
# ABOUTME: Tests capability matching and default registry population

import os
import sys
from unittest.mock import Mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.delegation.registry import AgentDescriptor, CapabilityRegistry


class TestCapabilityRegistry:
    def test_register_and_get(self):
        registry = CapabilityRegistry()
        desc = AgentDescriptor(
            name="test_agent", capabilities=["test"], invoke_fn=Mock()
        )
        registry.register(desc)
        assert registry.get("test_agent") is desc

    def test_get_unknown_returns_none(self):
        registry = CapabilityRegistry()
        assert registry.get("nonexistent") is None

    def test_registry_does_not_route_from_goal_keywords(self):
        registry = CapabilityRegistry()
        registry.register(
            AgentDescriptor(name="info", capabilities=["weather"], invoke_fn=Mock())
        )
        assert not hasattr(registry, "resolve")

    def test_all_agents(self):
        registry = CapabilityRegistry()
        for name in ["a", "b", "c"]:
            registry.register(
                AgentDescriptor(name=name, capabilities=[name], invoke_fn=Mock())
            )
        assert len(registry.all_agents()) == 3

    def test_agent_names(self):
        registry = CapabilityRegistry()
        registry.register(
            AgentDescriptor(name="alpha", capabilities=["a"], invoke_fn=Mock())
        )
        registry.register(
            AgentDescriptor(name="beta", capabilities=["b"], invoke_fn=Mock())
        )
        assert set(registry.agent_names()) == {"alpha", "beta"}

    def test_duplicate_registration_overwrites(self):
        registry = CapabilityRegistry()
        fn1 = Mock()
        fn2 = Mock()
        registry.register(
            AgentDescriptor(name="dup", capabilities=["x"], invoke_fn=fn1)
        )
        registry.register(
            AgentDescriptor(name="dup", capabilities=["y"], invoke_fn=fn2)
        )
        assert registry.get("dup").invoke_fn is fn2
        assert len(registry.all_agents()) == 1
