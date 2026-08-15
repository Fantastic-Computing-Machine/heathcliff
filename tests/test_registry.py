# ABOUTME: Unit tests for CapabilityRegistry — registration, resolution, agent lookup
# ABOUTME: Tests capability matching and default registry population

import os
import sys
from unittest.mock import Mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.delegation.registry import AgentDescriptor, CapabilityRegistry


class TestAgentDescriptor:
    def test_matches_goal_positive(self):
        desc = AgentDescriptor(
            name="info_agent_tool",
            capabilities=["weather", "news", "search"],
            invoke_fn=Mock(),
        )
        assert desc.matches_goal("What's the weather like?")
        assert desc.matches_goal("Search for AI news")

    def test_matches_goal_negative(self):
        desc = AgentDescriptor(
            name="music_agent_tool",
            capabilities=["music", "spotify", "play"],
            invoke_fn=Mock(),
        )
        assert not desc.matches_goal("Send an email to John")

    def test_matches_goal_case_insensitive(self):
        desc = AgentDescriptor(
            name="email_agent_tool",
            capabilities=["email", "gmail"],
            invoke_fn=Mock(),
        )
        assert desc.matches_goal("Send EMAIL to boss")


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

    def test_resolve_finds_matching(self):
        registry = CapabilityRegistry()
        info = AgentDescriptor(
            name="info", capabilities=["weather", "search"], invoke_fn=Mock()
        )
        music = AgentDescriptor(
            name="music", capabilities=["play", "spotify"], invoke_fn=Mock()
        )
        registry.register(info)
        registry.register(music)

        matches = registry.resolve("What's the weather?")
        assert len(matches) == 1
        assert matches[0].name == "info"

    def test_resolve_multiple_matches(self):
        registry = CapabilityRegistry()
        a = AgentDescriptor(name="a", capabilities=["search"], invoke_fn=Mock())
        b = AgentDescriptor(name="b", capabilities=["search", "web"], invoke_fn=Mock())
        registry.register(a)
        registry.register(b)

        matches = registry.resolve("search the web")
        assert len(matches) == 2

    def test_resolve_no_match(self):
        registry = CapabilityRegistry()
        registry.register(
            AgentDescriptor(name="a", capabilities=["xyz"], invoke_fn=Mock())
        )
        assert registry.resolve("completely unrelated") == []

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
