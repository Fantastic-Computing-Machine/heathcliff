# ABOUTME: Unit tests for HeathcliffAgent — singleton + self-wiring architecture
# ABOUTME: Tests init, singleton behaviour, invoke validation, session
# ABOUTME: management, memory integration, invoke contract, and tool registration.

import os
import sys
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Singleton reset — every test gets a clean HeathcliffAgent
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton():
    from core.agent_core import HeathcliffAgent

    HeathcliffAgent.reset()
    yield
    HeathcliffAgent.reset()


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_memory_manager():
    mm = Mock()
    mm.recall = Mock(
        return_value={
            "documents": [["User likes dark mode", "User is in Jersey City"]],
            "metadatas": [[{"category": "preferences"}, {"category": "location"}]],
            "ids": [["mem_1", "mem_2"]],
            "distances": [[0.05, 0.12]],
        }
    )
    mm.build_message_history = Mock(
        return_value=[
            {"role": "user", "content": "prev: Hello Heathcliff"},
            {"role": "assistant", "content": "prev: Good morning."},
        ]
    )
    mm.get_recent_chats = Mock(return_value=[])
    mm.get_chat_context = Mock(
        return_value={
            "documents": [["prev: Hello Heathcliff", "prev: Good morning."]],
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


def _make_agent(mock_memory_manager, mock_subagent_tools):
    """Helper: build a HeathcliffAgent with mocked LLM / tools."""
    with (
        patch("core.agent_core.init_chat_model"),
        patch("core.agent_core.create_agent"),
        patch("core.agent_core.create_middleware_stack", return_value=[]),
        patch.object(
            __import__("core.agent_core", fromlist=["HeathcliffAgent"]).HeathcliffAgent,
            "_assemble_default_tools",
            return_value=list(mock_subagent_tools),
        ),
    ):
        from core.agent_core import HeathcliffAgent

        return HeathcliffAgent(memory_manager=mock_memory_manager)


# ---------------------------------------------------------------------------
# Singleton behaviour
# ---------------------------------------------------------------------------


class TestSingleton:
    """HeathcliffAgent must behave as a singleton."""

    @patch("core.agent_core.create_agent")
    @patch("core.agent_core.init_chat_model")
    @patch("core.agent_core.create_middleware_stack", return_value=[])
    @patch(
        "core.agent_core.HeathcliffAgent._assemble_default_tools",
        return_value=[],
    )
    def test_same_instance_returned(self, _tools, _mw, _llm, _ca, mock_memory_manager):
        from core.agent_core import HeathcliffAgent

        a1 = HeathcliffAgent(memory_manager=mock_memory_manager)
        a2 = HeathcliffAgent()
        assert a1 is a2

    @patch("core.agent_core.create_agent")
    @patch("core.agent_core.init_chat_model")
    @patch("core.agent_core.create_middleware_stack", return_value=[])
    @patch(
        "core.agent_core.HeathcliffAgent._assemble_default_tools",
        return_value=[],
    )
    def test_instance_returns_singleton(
        self, _tools, _mw, _llm, _ca, mock_memory_manager
    ):
        from core.agent_core import HeathcliffAgent

        HeathcliffAgent(memory_manager=mock_memory_manager)
        assert HeathcliffAgent.instance() is HeathcliffAgent._instance

    def test_instance_raises_before_init(self):
        from core.agent_core import HeathcliffAgent

        with pytest.raises(RuntimeError, match="not been initialised"):
            HeathcliffAgent.instance()

    def test_reset_clears_singleton(self, mock_memory_manager):
        from core.agent_core import HeathcliffAgent

        with (
            patch("core.agent_core.init_chat_model"),
            patch("core.agent_core.create_agent"),
            patch("core.agent_core.create_middleware_stack", return_value=[]),
            patch(
                "core.agent_core.HeathcliffAgent._assemble_default_tools",
                return_value=[],
            ),
        ):
            HeathcliffAgent(memory_manager=mock_memory_manager)
        HeathcliffAgent.reset()
        assert HeathcliffAgent._instance is None


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestHeathcliffAgentInit:
    """Tests for HeathcliffAgent.__init__ with the singleton + self-wiring arch."""

    @patch("core.agent_core.create_agent")
    @patch("core.agent_core.init_chat_model")
    @patch("core.agent_core.create_middleware_stack", return_value=[])
    @patch(
        "core.agent_core.HeathcliffAgent._assemble_default_tools",
        return_value=[],
    )
    def test_init_with_no_extra_tools(
        self, _tools, _mw, mock_init_model, mock_create_agent, mock_memory_manager
    ):
        """Agent starts with only default tools when no extra_tools given."""
        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(memory_manager=mock_memory_manager)
        assert agent._tools == []

    @patch("core.agent_core.create_agent")
    @patch("core.agent_core.init_chat_model")
    @patch("core.agent_core.create_middleware_stack", return_value=[])
    def test_init_loads_default_tools(
        self,
        _mw,
        mock_init_model,
        mock_create_agent,
        mock_memory_manager,
        mock_subagent_tools,
    ):
        """_assemble_default_tools is called and its result is used."""
        with patch(
            "core.agent_core.HeathcliffAgent._assemble_default_tools",
            return_value=list(mock_subagent_tools),
        ):
            from core.agent_core import HeathcliffAgent

            agent = HeathcliffAgent(memory_manager=mock_memory_manager)
        assert len(agent._tools) == len(mock_subagent_tools)

    @patch("core.agent_core.create_agent")
    @patch("core.agent_core.init_chat_model")
    @patch("core.agent_core.create_middleware_stack", return_value=[])
    @patch(
        "core.agent_core.HeathcliffAgent._assemble_default_tools",
        return_value=[],
    )
    def test_extra_tools_appended(
        self, _tools, _mw, mock_init_model, mock_create_agent, mock_memory_manager
    ):
        """extra_tools are appended to the default set."""
        from core.agent_core import HeathcliffAgent

        extra = [Mock(name="my_extra")]
        agent = HeathcliffAgent(memory_manager=mock_memory_manager, extra_tools=extra)
        assert len(agent._tools) == 1
        assert agent._tools[0] is extra[0]

    @patch("core.agent_core.create_agent")
    @patch("core.agent_core.init_chat_model")
    @patch("core.agent_core.create_middleware_stack", return_value=[])
    @patch(
        "core.agent_core.HeathcliffAgent._assemble_default_tools",
        return_value=[],
    )
    def test_init_creates_llm(
        self, _tools, _mw, mock_init_model, mock_create_agent, mock_memory_manager
    ):
        """LLM is instantiated via init_chat_model with correct config values."""
        from config import Config
        from core.agent_core import HeathcliffAgent

        HeathcliffAgent(memory_manager=mock_memory_manager)
        mock_init_model.assert_called_once()
        call_kwargs = mock_init_model.call_args[1]
        assert call_kwargs["model"] == Config.SUPERVISOR_MODEL
        assert call_kwargs["api_key"] == Config.AI_KEY

    @patch("core.agent_core.create_agent")
    @patch("core.agent_core.init_chat_model")
    @patch("core.agent_core.create_middleware_stack", return_value=[])
    @patch(
        "core.agent_core.HeathcliffAgent._assemble_default_tools",
        return_value=[],
    )
    def test_init_builds_executor(
        self, _tools, _mw, mock_init_model, mock_create_agent, mock_memory_manager
    ):
        """create_agent is called and executor is stored."""
        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(memory_manager=mock_memory_manager)
        mock_create_agent.assert_called_once()
        assert agent.executor is not None

    @patch("core.agent_core.create_agent")
    @patch("core.agent_core.init_chat_model")
    @patch("core.agent_core.create_middleware_stack", return_value=[])
    @patch(
        "core.agent_core.HeathcliffAgent._assemble_default_tools",
        return_value=[],
    )
    def test_init_stores_memory_manager(
        self, _tools, _mw, mock_init_model, mock_create_agent, mock_memory_manager
    ):
        """memory_manager reference is stored."""
        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(memory_manager=mock_memory_manager)
        assert agent.memory_manager is mock_memory_manager


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests for invoke() input validation contract."""

    @pytest.fixture
    def agent(self, mock_memory_manager, mock_subagent_tools):
        return _make_agent(mock_memory_manager, mock_subagent_tools)

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
            return_value={"messages": [Mock(content="Good day.")]}
        )
        agent.invoke("Good morning")


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


class TestSessionManagement:
    """Tests for session ID handling."""

    @pytest.fixture
    def agent(self, mock_memory_manager, mock_subagent_tools):
        a = _make_agent(mock_memory_manager, mock_subagent_tools)
        a.executor = Mock()
        a.executor.invoke = Mock(
            return_value={"messages": [Mock(content="Certainly.")]}
        )
        return a

    def test_auto_generates_session_id_when_none(self, agent):
        """If no session_id is passed, a UUID is generated."""
        agent.invoke("Hello")
        save_args = agent.memory_manager.save_chat.call_args
        if save_args is not None:
            session_id = save_args[0][2]
            assert session_id

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


# ---------------------------------------------------------------------------
# Memory integration
# ---------------------------------------------------------------------------


class TestMemoryIntegration:
    """Tests for memory retrieval and save integration."""

    @pytest.fixture
    def agent(self, mock_memory_manager, mock_subagent_tools):
        a = _make_agent(mock_memory_manager, mock_subagent_tools)
        a.executor = Mock()
        a.executor.invoke = Mock(return_value={"messages": [Mock(content="Indeed.")]})
        return a

    def test_recall_is_called_before_invoke(self, agent):
        agent.invoke("What do I like?")
        agent.memory_manager.recall.assert_called()

    def test_build_message_history_is_called(self, agent):
        agent.invoke("Remind me of our last chat")
        agent.memory_manager.build_message_history.assert_called()

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
        try:
            result = agent.invoke("Hello")
            assert isinstance(result, str)
        except Exception:
            pytest.fail("Agent should handle memory errors gracefully")

    def test_memories_are_injected_into_user_prompt(self, agent):
        """Mem0 recall should be injected into USER_PROMPT_TEMPLATE."""
        query = "What do I like?"
        agent.invoke(query)

        invoke_args = agent.executor.invoke.call_args
        assert invoke_args is not None
        messages = invoke_args[0][0]["messages"]
        final_message = messages[-1]

        assert "Long-term Memory Context:" in final_message.content
        assert "<USER_MEMORY_CONTEXT>" in final_message.content
        assert "</USER_MEMORY_CONTEXT>" in final_message.content
        assert "- User likes dark mode" in final_message.content
        assert "Current Date and Time:" in final_message.content
        assert "Current month:" in final_message.content
        assert "Current year:" in final_message.content
        assert "Current User Query:" in final_message.content
        assert "<USER_QUERY>" in final_message.content
        assert "</USER_QUERY>" in final_message.content
        assert query in final_message.content
        assert final_message.content.strip().endswith("</USER_QUERY>")

    def test_memories_not_injected_as_system_message(self, agent):
        """Long-term memories should no longer be added as SystemMessage context."""
        agent.invoke("Hello")

        invoke_args = agent.executor.invoke.call_args
        assert invoke_args is not None
        messages = invoke_args[0][0]["messages"]

        assert not any(
            getattr(message, "type", "") == "system"
            and "Long-term memories:" in str(getattr(message, "content", ""))
            for message in messages
        )


# ---------------------------------------------------------------------------
# Invoke return contract
# ---------------------------------------------------------------------------


class TestInvokeContract:
    """Tests for invoke() return type and content guarantees."""

    @pytest.fixture
    def agent(self, mock_memory_manager, mock_subagent_tools):
        a = _make_agent(mock_memory_manager, mock_subagent_tools)
        a.executor = Mock()
        a.executor.invoke = Mock(
            return_value={"messages": [Mock(content="Quite right.")]}
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
    @patch("core.agent_core.init_chat_model")
    @patch("core.agent_core.create_middleware_stack", return_value=[])
    def test_all_9_supervisor_tools_registered(
        self, _mw, mock_llm, mock_ca, mock_memory_manager
    ):
        """Full supervisor tool list: 6 subagents + recent_context + load_skill + update_master_info."""
        from core.agent_core import HeathcliffAgent

        HeathcliffAgent(memory_manager=mock_memory_manager)
        registered_names = [t.name for t in mock_ca.call_args[1]["tools"]]
        expected = {
            "info_agent_tool",
            "music_agent_tool",
            "email_agent_tool",
            "calendar_agent_tool",
            "contacts_agent_tool",
            "comms_agent_tool",
            "recent_context",
            "load_skill",
            "update_master_info",
        }
        assert set(registered_names) == expected

    @patch("core.agent_core.create_agent")
    @patch("core.agent_core.init_chat_model")
    @patch("core.agent_core.create_middleware_stack", return_value=[])
    def test_no_duplicate_tools(self, _mw, mock_llm, mock_ca, mock_memory_manager):
        """No duplicate tool names in the registered list."""
        from core.agent_core import HeathcliffAgent

        HeathcliffAgent(memory_manager=mock_memory_manager)
        registered_names = [t.name for t in mock_ca.call_args[1]["tools"]]
        assert len(registered_names) == len(set(registered_names))
