# ABOUTME: Comprehensive tests for core/subagents/ — each domain agent wrapper
# ABOUTME: Tests tool registration, lazy init, graceful degradation, tool metadata

import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Registry: get_all_subagent_tools()
# ---------------------------------------------------------------------------


class TestSubagentRegistry:
    """Tests for the core/subagents/__init__.py registry."""

    def test_get_all_subagent_tools_returns_list(self):
        from core.subagents import get_all_subagent_tools

        tools = get_all_subagent_tools()
        assert isinstance(tools, list)

    def test_get_all_subagent_tools_returns_6_tools(self):
        from core.subagents import get_all_subagent_tools

        tools = get_all_subagent_tools()
        assert len(tools) == 6

    def test_all_expected_tools_present(self):
        from core.subagents import get_all_subagent_tools

        names = {t.name for t in get_all_subagent_tools()}
        expected = {
            "info_agent_tool",
            "music_agent_tool",
            "email_agent_tool",
            "calendar_agent_tool",
            "contacts_agent_tool",
            "comms_agent_tool",
        }
        assert names == expected

    def test_all_tools_have_name(self):
        from core.subagents import get_all_subagent_tools

        for tool in get_all_subagent_tools():
            assert hasattr(tool, "name"), f"Tool missing .name: {tool}"
            assert len(tool.name) > 0

    def test_all_tools_have_description(self):
        from core.subagents import get_all_subagent_tools

        for tool in get_all_subagent_tools():
            assert hasattr(
                tool, "description"
            ), f"Tool missing .description: {tool.name}"
            assert len(tool.description) > 0

    def test_no_duplicate_tool_names(self):
        from core.subagents import get_all_subagent_tools

        names = [t.name for t in get_all_subagent_tools()]
        assert len(names) == len(set(names))

    def test_tools_are_langchain_tools(self):
        """Each tool should be invokable — has .invoke method."""
        from core.subagents import get_all_subagent_tools

        for tool in get_all_subagent_tools():
            assert hasattr(tool, "invoke"), f"{tool.name} lacks .invoke()"


# ---------------------------------------------------------------------------
# Individual tool metadata
# ---------------------------------------------------------------------------


class TestToolMetadata:

    @pytest.mark.parametrize(
        "tool_name,module_path",
        [
            ("info_agent_tool", "core.subagents.info.agent"),
            ("music_agent_tool", "core.subagents.music.agent"),
            ("email_agent_tool", "core.subagents.email.agent"),
            ("calendar_agent_tool", "core.subagents.calendar.agent"),
            ("contacts_agent_tool", "core.subagents.contacts.agent"),
            ("comms_agent_tool", "core.subagents.comms.agent"),
        ],
    )
    def test_tool_exported_from_module(self, tool_name, module_path):
        """Each agent tool is importable from its own module."""
        import importlib

        mod = importlib.import_module(module_path)
        tool = getattr(mod, tool_name)
        assert tool.name == tool_name

    def test_info_agent_description_mentions_search(self):
        from core.subagents.info.agent import info_agent_tool

        desc = info_agent_tool.description.lower()
        assert any(
            kw in desc for kw in ["search", "research", "weather", "news", "wikipedia"]
        )

    def test_music_agent_description_mentions_spotify(self):
        from core.subagents.music.agent import music_agent_tool

        desc = music_agent_tool.description.lower()
        assert any(kw in desc for kw in ["spotify", "music", "play", "pause"])

    def test_email_agent_description_mentions_gmail(self):
        from core.subagents.email.agent import email_agent_tool

        desc = email_agent_tool.description.lower()
        assert any(kw in desc for kw in ["email", "gmail", "send", "draft"])

    def test_email_agent_description_requires_address(self):
        """Email agent description must warn about needing email address."""
        from core.subagents.email.agent import email_agent_tool

        desc = email_agent_tool.description.lower()
        assert "address" in desc or "email" in desc

    def test_calendar_agent_description_mentions_events(self):
        from core.subagents.calendar.agent import calendar_agent_tool

        desc = calendar_agent_tool.description.lower()
        assert any(kw in desc for kw in ["calendar", "event", "schedule", "meeting"])

    def test_contacts_agent_description_mentions_contacts(self):
        from core.subagents.contacts.agent import contacts_agent_tool

        desc = contacts_agent_tool.description.lower()
        assert any(kw in desc for kw in ["contact", "email", "phone", "lookup"])

    def test_contacts_description_mentions_fallback_behavior(self):
        """Contacts tool must describe what to do if contact not found."""
        from core.subagents.contacts.agent import contacts_agent_tool

        desc = contacts_agent_tool.description.lower()
        assert "not found" in desc or "ask" in desc or "provide" in desc

    def test_comms_agent_description_mentions_telegram(self):
        from core.subagents.comms.agent import comms_agent_tool

        desc = comms_agent_tool.description.lower()
        assert any(
            kw in desc for kw in ["telegram", "message", "drive", "notification"]
        )


# ---------------------------------------------------------------------------
# Graceful degradation (agent builds fail at import time)
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Tests that sub-agents return meaningful errors when their tools are unavailable."""

    def test_info_agent_degrades_when_tools_unavailable(self):
        import core.subagents.info.agent as info_mod

        original = info_mod._agent
        info_mod._agent = None

        with patch("core.subagents.info.agent._build", return_value=None):
            result = info_mod.info_agent_tool.invoke({"request": "weather"})
        assert "unavailable" in result.lower()

        info_mod._agent = original

    def test_music_agent_degrades_when_tools_unavailable(self):
        import core.subagents.music.agent as music_mod

        original = music_mod._agent
        music_mod._agent = None

        with patch("core.subagents.music.agent._build", return_value=None):
            result = music_mod.music_agent_tool.invoke({"request": "play something"})
        assert "unavailable" in result.lower()

        music_mod._agent = original

    def test_email_agent_degrades_when_tools_unavailable(self):
        import core.subagents.email.agent as email_mod

        original = email_mod._agent
        email_mod._agent = None

        with patch("core.subagents.email.agent._build", return_value=None):
            result = email_mod.email_agent_tool.invoke({"request": "send email"})
        assert "unavailable" in result.lower()

        email_mod._agent = original

    def test_calendar_agent_degrades_when_tools_unavailable(self):
        import core.subagents.calendar.agent as cal_mod

        original = cal_mod._agent
        cal_mod._agent = None

        with patch("core.subagents.calendar.agent._build", return_value=None):
            result = cal_mod.calendar_agent_tool.invoke({"request": "check calendar"})
        assert "unavailable" in result.lower()

        cal_mod._agent = original

    def test_contacts_agent_degrades_when_tools_unavailable(self):
        import core.subagents.contacts.agent as contacts_mod

        original = contacts_mod._agent
        contacts_mod._agent = None

        with patch("core.subagents.contacts.agent._build", return_value=None):
            result = contacts_mod.contacts_agent_tool.invoke({"request": "find Philip"})
        assert "unavailable" in result.lower()

        contacts_mod._agent = original

    def test_comms_agent_degrades_when_tools_unavailable(self):
        import core.subagents.comms.agent as comms_mod

        original = comms_mod._agent
        comms_mod._agent = None

        with patch("core.subagents.comms.agent._build", return_value=None):
            result = comms_mod.comms_agent_tool.invoke({"request": "send telegram"})
        assert "unavailable" in result.lower()

        comms_mod._agent = original


# ---------------------------------------------------------------------------
# Tool invocation with mocked underlying agent
# ---------------------------------------------------------------------------


class TestToolInvocationWithMockedAgent:
    """Tests that each tool correctly proxies requests to its underlying agent."""

    def _make_mock_agent(self, response_text: str):
        mock_agent = Mock()
        mock_msg = Mock()
        mock_msg.content = response_text
        mock_agent.invoke = Mock(return_value={"messages": [mock_msg]})
        return mock_agent

    def test_info_tool_returns_agent_response(self):
        import core.subagents.info.agent as info_mod

        mock_agent = self._make_mock_agent("Jersey City temp is 12°C, overcast.")
        info_mod._agent = mock_agent
        result = info_mod.info_agent_tool.invoke({"request": "weather in Jersey City"})
        assert "Jersey City" in result or "12" in result
        info_mod._agent = None

    def test_music_tool_returns_agent_response(self):
        import core.subagents.music.agent as music_mod

        mock_agent = self._make_mock_agent("Now playing: Taylor Swift - Love Story")
        music_mod._agent = mock_agent
        result = music_mod.music_agent_tool.invoke(
            {"request": "play Taylor Swift - Love Story"}
        )
        assert "Taylor Swift" in result
        music_mod._agent = None

    def test_email_tool_returns_agent_response(self):
        import core.subagents.email.agent as email_mod

        mock_agent = self._make_mock_agent("Email sent to philip@example.com.")
        email_mod._agent = mock_agent
        result = email_mod.email_agent_tool.invoke(
            {"request": "send email to philip@example.com about the sea level research"}
        )
        assert "philip" in result.lower() or "sent" in result.lower()
        email_mod._agent = None

    def test_calendar_tool_returns_agent_response(self):
        import core.subagents.calendar.agent as cal_mod

        mock_agent = self._make_mock_agent(
            "Event 'Design Review' created for tomorrow at 2pm."
        )
        cal_mod._agent = mock_agent
        result = cal_mod.calendar_agent_tool.invoke(
            {"request": "create Design Review tomorrow 2pm"}
        )
        assert "Design Review" in result or "created" in result.lower()
        cal_mod._agent = None

    def test_contacts_tool_returns_agent_response_when_found(self):
        import core.subagents.contacts.agent as contacts_mod

        mock_agent = self._make_mock_agent("Philip Thorne — philip.thorne@example.com")
        contacts_mod._agent = mock_agent
        result = contacts_mod.contacts_agent_tool.invoke(
            {"request": "Find Philip's email"}
        )
        assert "philip" in result.lower()
        contacts_mod._agent = None

    def test_contacts_tool_returns_not_found_message(self):
        import core.subagents.contacts.agent as contacts_mod

        mock_agent = self._make_mock_agent(
            "No contact found for 'Philip'. Please provide the email address directly."
        )
        contacts_mod._agent = mock_agent
        result = contacts_mod.contacts_agent_tool.invoke(
            {"request": "Find Philip's email"}
        )
        assert "no contact found" in result.lower() or "not found" in result.lower()
        contacts_mod._agent = None

    def test_agent_invocation_passes_request_in_messages(self):
        """Verify the request string is forwarded as a user message."""
        import core.subagents.info.agent as info_mod

        mock_agent = self._make_mock_agent("Sunny.")
        info_mod._agent = mock_agent
        info_mod.info_agent_tool.invoke({"request": "What is the weather in Denver?"})
        call_args = mock_agent.invoke.call_args[0][0]
        messages = call_args.get("messages", [])
        assert any("Denver" in str(m) for m in messages)
        info_mod._agent = None

    def test_comms_tool_returns_agent_response(self):
        import core.subagents.comms.agent as comms_mod

        mock_agent = self._make_mock_agent("Telegram message sent: Build finished.")
        comms_mod._agent = mock_agent
        result = comms_mod.comms_agent_tool.invoke(
            {"request": "Send telegram: Build finished"}
        )
        assert (
            "telegram" in result.lower()
            or "sent" in result.lower()
            or "build" in result.lower()
        )
        comms_mod._agent = None


# ---------------------------------------------------------------------------
# Multi-step chaining scenario (integration-style, all mocked)
# ---------------------------------------------------------------------------


class TestMultiStepChaining:
    """
    Tests the multi-step pattern: info → contacts → email.
    All agents mocked — no LLM calls.
    """

    def test_research_then_email_pattern(self):
        """
        Simulates: 'research rising sea levels then email Philip a summary'
        Step 1: info_agent_tool → returns research
        Step 2: contacts_agent_tool → returns Philip's email
        Step 3: email_agent_tool → sends email
        """
        import core.subagents.contacts.agent as contacts_mod
        import core.subagents.email.agent as email_mod
        import core.subagents.info.agent as info_mod

        def _mock_agent(text):
            m = Mock()
            m.content = text
            a = Mock()
            a.invoke = Mock(return_value={"messages": [m]})
            return a

        # Wire mocks
        info_mod._agent = _mock_agent(
            "Rising sea levels: oceans rising ~3.7mm/year, projections show +1m by 2100."
        )
        contacts_mod._agent = _mock_agent("Philip Thorne — philip.thorne@example.com")
        email_mod._agent = _mock_agent("Email sent to philip.thorne@example.com.")

        # Execute the chain
        research = info_mod.info_agent_tool.invoke(
            {"request": "research rising sea levels 2025"}
        )
        contact = contacts_mod.contacts_agent_tool.invoke(
            {"request": "find Philip's email"}
        )
        email_result = email_mod.email_agent_tool.invoke(
            {
                "request": f"Send email to philip.thorne@example.com: subject 'Rising Sea Levels Summary', body '{research}'"
            }
        )

        # Assertions across the chain
        assert "sea level" in research.lower() or "ocean" in research.lower()
        assert "philip" in contact.lower()
        assert "sent" in email_result.lower() or "philip" in email_result.lower()

        # Cleanup
        info_mod._agent = None
        contacts_mod._agent = None
        email_mod._agent = None

    def test_contacts_fallback_when_email_unknown(self):
        """
        Simulates: email requested, contacts not found, supervisor MUST ask user.
        Verifies contacts returns the canonical not-found message.
        """
        import core.subagents.contacts.agent as contacts_mod

        m = Mock()
        m.content = (
            "No contact found for 'Philip'. Please provide the email address directly."
        )
        mock_agent = Mock()
        mock_agent.invoke = Mock(return_value={"messages": [m]})
        contacts_mod._agent = mock_agent

        result = contacts_mod.contacts_agent_tool.invoke(
            {"request": "find Philip's email"}
        )

        # Supervisor should detect this and ask user
        assert (
            "no contact found" in result.lower() or "please provide" in result.lower()
        )

        contacts_mod._agent = None

    def test_music_then_info_independent_agents(self):
        """
        Simulates: 'play Taylor Swift and tell me the weather'
        Two independent agent calls — both should succeed.
        """
        import core.subagents.info.agent as info_mod
        import core.subagents.music.agent as music_mod

        def _mk(text):
            m = Mock()
            m.content = text
            a = Mock()
            a.invoke = Mock(return_value={"messages": [m]})
            return a

        music_mod._agent = _mk("Now playing: Taylor Swift - Shake It Off")
        info_mod._agent = _mk("Jersey City: 14°C, partly cloudy")

        music_result = music_mod.music_agent_tool.invoke(
            {"request": "play Taylor Swift - Shake It Off"}
        )
        info_result = info_mod.info_agent_tool.invoke(
            {"request": "weather in Jersey City"}
        )

        assert "taylor swift" in music_result.lower()
        assert "jersey city" in info_result.lower() or "14" in info_result

        music_mod._agent = None
        info_mod._agent = None
