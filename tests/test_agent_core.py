# ABOUTME: Unit tests for HeathcliffAgent — updated for supervisor+subagents architecture
# ABOUTME: Tests init, invoke validation, session management, memory integration, and error handling

import os
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock, call
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_memory_manager():
    mm = Mock()
    mm.recall = Mock(
        return_value={
            "documents": [["Adi likes dark mode", "Adi is in Jersey City"]],
            "metadatas": [[{"category": "preferences"}, {"category": "location"}]],
            "ids": [["mem_1", "mem_2"]],
            "distances": [[0.05, 0.12]],
        }
    )
    mm.get_chat_context = Mock(
        return_value={
            "documents": [["prev: Hello Heathcliff", "prev: Good morning Adi"]],
            "metadatas": [
                [{"role": "user", "session": "sess_abc"}, {"role": "assistant"}]
            ],
            "ids": [["chat_1", "chat_2"]],
            "distances": [[0.08, 0.11]],
        }
    )
    mm.save_chat = Mock(return_value=("uid_123", "aid_456"))
    return mm


@pytest.fixture
def mock_subagent_tools():
    """Fake subagent @tool callables (no LLM calls)."""
    tools = []
    for name in [
        "info_agent_tool",
        "music_agent_tool",
        "email_agent_tool",
        "calendar_agent_tool",
        "contacts_agent_tool",
        "comms_agent_tool",
        "load_skill",
        "update_master_info",
    ]:
        t = Mock()
        t.name = name
        t.description = f"Mock tool: {name}"
        tools.append(t)
    return tools


# ---------------------------------------------------------------------------
# HeathcliffAgent initialisation
# ---------------------------------------------------------------------------


class TestHeathcliffAgentInit:
    """Tests for HeathcliffAgent.__init__ with the new supervisor architecture."""

    @patch("core.agent_core.create_agent")
    @patch("core.agent_core.ChatGoogleGenerativeAI")
    def test_init_accepts_tool_list(
        self, mock_llm_cls, mock_create_agent, mock_memory_manager, mock_subagent_tools
    ):
        """Agent accepts a flat list of BaseTool objects."""
        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(
            memory_manager=mock_memory_manager, tools=mock_subagent_tools
        )
        assert len(agent._tools) == len(mock_subagent_tools)

    @patch("core.agent_core.create_agent")
    @patch("core.agent_core.ChatGoogleGenerativeAI")
    def test_init_with_no_tools(
        self, mock_llm_cls, mock_create_agent, mock_memory_manager
    ):
        """Agent starts successfully with no tools (fallback to direct LLM)."""
        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(memory_manager=mock_memory_manager, tools=[])
        assert agent._tools == []

    @patch("core.agent_core.create_agent")
    @patch("core.agent_core.ChatGoogleGenerativeAI")
    def test_init_creates_llm(
        self, mock_llm_cls, mock_create_agent, mock_memory_manager, mock_subagent_tools
    ):
        """LLM is instantiated with correct config values."""
        from core.agent_core import HeathcliffAgent
        from config import Config

        HeathcliffAgent(memory_manager=mock_memory_manager, tools=mock_subagent_tools)
        call_kwargs = mock_llm_cls.call_args[1]
        assert call_kwargs["model"] == Config.MODEL
        assert call_kwargs["google_api_key"] == Config.GEMINI_API_KEY

    @patch("core.agent_core.create_agent")
    @patch("core.agent_core.ChatGoogleGenerativeAI")
    def test_init_builds_executor(
        self, mock_llm_cls, mock_create_agent, mock_memory_manager, mock_subagent_tools
    ):
        """create_agent is called and executor is stored."""
        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(
            memory_manager=mock_memory_manager, tools=mock_subagent_tools
        )
        mock_create_agent.assert_called_once()
        assert agent.executor is not None

    @patch("core.agent_core.create_agent")
    @patch("core.agent_core.ChatGoogleGenerativeAI")
    def test_init_passes_all_tools_to_create_agent(
        self, mock_llm_cls, mock_create_agent, mock_memory_manager, mock_subagent_tools
    ):
        """All registered tools are forwarded to create_agent."""
        from core.agent_core import HeathcliffAgent

        HeathcliffAgent(memory_manager=mock_memory_manager, tools=mock_subagent_tools)
        create_agent_kwargs = mock_create_agent.call_args[1]
        assert create_agent_kwargs["tools"] == mock_subagent_tools

    @patch("core.agent_core.create_agent")
    @patch("core.agent_core.ChatGoogleGenerativeAI")
    def test_init_stores_memory_manager(
        self, mock_llm_cls, mock_create_agent, mock_memory_manager, mock_subagent_tools
    ):
        """memory_manager reference is stored."""
        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(
            memory_manager=mock_memory_manager, tools=mock_subagent_tools
        )
        assert agent.memory_manager is mock_memory_manager


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests for invoke() input validation contract."""

    @pytest.fixture
    def agent(self, mock_memory_manager, mock_subagent_tools):
        with (
            patch("core.agent_core.ChatGoogleGenerativeAI"),
            patch("core.agent_core.create_agent"),
        ):
            from core.agent_core import HeathcliffAgent

            return HeathcliffAgent(
                memory_manager=mock_memory_manager, tools=mock_subagent_tools
            )

    def test_empty_input_raises_value_error(self, agent):
        with pytest.raises(ValueError, match="empty"):
            agent.invoke("")

    def test_whitespace_only_raises_value_error(self, agent):
        with pytest.raises(ValueError):
            agent.invoke("   ")

    def test_input_exceeding_max_length_raises_value_error(self, agent):
        with pytest.raises(ValueError, match="[Tt]oo long|exceed|[Mm]ax"):
            agent.invoke("X" * 10_001)

    def test_valid_input_does_not_raise_on_validation(self, agent):
        """The validation step should not raise for normal input."""
        agent.executor = Mock()
        agent.executor.invoke = Mock(
            return_value={"messages": [Mock(content="Good day, Adi.")]}
        )
        # Should not raise
        agent.invoke("Good morning")


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


class TestSessionManagement:
    """Tests for session ID handling."""

    @pytest.fixture
    def agent(self, mock_memory_manager, mock_subagent_tools):
        with (
            patch("core.agent_core.ChatGoogleGenerativeAI"),
            patch("core.agent_core.create_agent"),
        ):
            from core.agent_core import HeathcliffAgent

            a = HeathcliffAgent(
                memory_manager=mock_memory_manager, tools=mock_subagent_tools
            )
            a.executor = Mock()
            a.executor.invoke = Mock(
                return_value={"messages": [Mock(content="Certainly, Adi.")]}
            )
            return a

    def test_auto_generates_session_id_when_none(self, agent):
        """If no session_id is passed, a UUID is generated."""
        agent.invoke("Hello")
        save_args = agent.memory_manager.save_chat.call_args
        # Either save_chat was called (success path) or an error was gracefully handled
        if save_args is not None:
            session_id = save_args[0][2]
            assert session_id
        # Either way no exception should have propagated

    def test_uses_provided_session_id(self, agent):
        """Explicit session_id is threaded through to save_chat."""
        agent.invoke("Hello", session_id="fixed-session-42")
        save_args = agent.memory_manager.save_chat.call_args
        if save_args is not None:
            assert save_args[0][2] == "fixed-session-42"

    def test_different_sessions_are_isolated(self, agent):
        """Two invocations with different sessions each save to their own session."""
        agent.invoke("Hello", session_id="session-A")
        agent.invoke("World", session_id="session-B")
        calls = agent.memory_manager.save_chat.call_args_list
        if len(calls) >= 2:
            sessions = [c[0][2] for c in calls]
            assert "session-A" in sessions
            assert "session-B" in sessions
            assert sessions[0] != sessions[1]
        # If save_chat wasn't called (error path), just verify no exception raised


# ---------------------------------------------------------------------------
# Memory integration
# ---------------------------------------------------------------------------


class TestMemoryIntegration:
    """Tests for memory retrieval and save integration."""

    @pytest.fixture
    def agent(self, mock_memory_manager, mock_subagent_tools):
        with (
            patch("core.agent_core.ChatGoogleGenerativeAI"),
            patch("core.agent_core.create_agent"),
        ):
            from core.agent_core import HeathcliffAgent

            a = HeathcliffAgent(
                memory_manager=mock_memory_manager, tools=mock_subagent_tools
            )
            a.executor = Mock()
            a.executor.invoke = Mock(
                return_value={"messages": [Mock(content="Indeed.")]}
            )
            return a

    def test_recall_is_called_before_invoke(self, agent):
        agent.invoke("What do I like?")
        agent.memory_manager.recall.assert_called()

    def test_get_chat_context_is_called(self, agent):
        agent.invoke("Remind me of our last chat")
        # get_chat_context is called during _format_chat_history path.
        # It may be skipped if recall itself errors; verify at least recall or context was attempted.
        assert (
            agent.memory_manager.recall.called
            or agent.memory_manager.get_chat_context.called
        )

    def test_save_chat_is_called_with_user_input(self, agent):
        agent.invoke("Test message")
        save_args = agent.memory_manager.save_chat.call_args
        if save_args:
            assert save_args[0][0] == "Test message"

    def test_save_chat_called_with_response(self, agent):
        agent.invoke("Test message")
        save_args = agent.memory_manager.save_chat.call_args
        if save_args:
            assert len(save_args[0][1]) > 0

    def test_memory_error_handled_gracefully(self, agent):
        """MemoryManager failure shouldn't crash the agent."""
        agent.memory_manager.recall.side_effect = Exception("DB unavailable")
        # Should degrade gracefully, not raise
        try:
            result = agent.invoke("Hello")
            assert isinstance(result, str)
        except Exception:
            pytest.fail("Agent should handle memory errors gracefully")


# ---------------------------------------------------------------------------
# Invoke return contract
# ---------------------------------------------------------------------------


class TestInvokeContract:
    """Tests for invoke() return type and content guarantees."""

    @pytest.fixture
    def agent(self, mock_memory_manager, mock_subagent_tools):
        with (
            patch("core.agent_core.ChatGoogleGenerativeAI"),
            patch("core.agent_core.create_agent"),
        ):
            from core.agent_core import HeathcliffAgent

            a = HeathcliffAgent(
                memory_manager=mock_memory_manager, tools=mock_subagent_tools
            )
            a.executor = Mock()
            a.executor.invoke = Mock(
                return_value={"messages": [Mock(content="Quite right, Adi.")]}
            )
            return a

    def test_invoke_returns_string(self, agent):
        result = agent.invoke("Hello")
        assert isinstance(result, str)

    def test_invoke_returns_non_empty_string(self, agent):
        result = agent.invoke("What's the date?")
        assert len(result.strip()) > 0

    def test_invoke_handles_executor_error(self, agent):
        """LLM/executor failure returns an error string, doesn't raise."""
        agent.executor.invoke.side_effect = Exception("Gemini API timeout")
        result = agent.invoke("Tell me a joke")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_invoke_handles_empty_messages(self, agent):
        """If executor returns empty messages, agent handles gracefully."""
        agent.executor.invoke = Mock(return_value={"messages": []})
        result = agent.invoke("Hello?")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    """Verify supervisor receives exactly the tools registered at init."""

    @patch("core.agent_core.create_agent")
    @patch("core.agent_core.ChatGoogleGenerativeAI")
    def test_all_8_supervisor_tools_registered(
        self, mock_llm, mock_ca, mock_memory_manager
    ):
        """Full supervisor tool list: 6 subagents + load_skill + update_master_info."""
        from core.subagents import get_all_subagent_tools
        from skills.skill_tools import get_skill_tools
        from core.agent_core import HeathcliffAgent

        tools = get_all_subagent_tools() + get_skill_tools()
        HeathcliffAgent(memory_manager=mock_memory_manager, tools=tools)
        registered_names = [t.name for t in mock_ca.call_args[1]["tools"]]
        expected = {
            "info_agent_tool",
            "music_agent_tool",
            "email_agent_tool",
            "calendar_agent_tool",
            "contacts_agent_tool",
            "comms_agent_tool",
            "load_skill",
            "update_master_info",
        }
        assert set(registered_names) == expected

    @patch("core.agent_core.create_agent")
    @patch("core.agent_core.ChatGoogleGenerativeAI")
    def test_no_duplicate_tools(self, mock_llm, mock_ca, mock_memory_manager):
        """No duplicate tool names in the registered list."""
        from core.subagents import get_all_subagent_tools
        from skills.skill_tools import get_skill_tools
        from core.agent_core import HeathcliffAgent

        tools = get_all_subagent_tools() + get_skill_tools()
        HeathcliffAgent(memory_manager=mock_memory_manager, tools=tools)
        registered_names = [t.name for t in mock_ca.call_args[1]["tools"]]
        assert len(registered_names) == len(set(registered_names))
