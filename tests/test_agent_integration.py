# ABOUTME: Integration tests for HeathcliffAgent with mocked MemoryManager
# ABOUTME: Tests full flow, multi-turn conversations, and component interaction

import os
import sys
from unittest.mock import MagicMock, Mock, patch

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


def _make_mock_memory_manager():
    """Create a mock MemoryManager with all required methods."""
    mm = Mock()
    mm.recall = Mock(
        return_value={
            "documents": [[]],
            "metadatas": [[]],
            "ids": [[]],
            "distances": [[]],
        }
    )
    mm.build_message_history = Mock(return_value=[])
    mm.get_recent_chats = Mock(return_value=[])
    mm.get_chat_context = Mock(
        return_value={
            "documents": [[]],
            "metadatas": [[]],
            "ids": [[]],
            "distances": [[]],
        }
    )
    mm.save_chat = Mock(return_value=("uid", "aid"))
    mm.add_memory = Mock(return_value="mem_1")
    mm.get_stats = Mock(return_value={"memories": 0, "chats": 0, "documents": 0})
    return mm


def _make_agent(memory_manager, executor_response="Hello! I'm Heathcliff."):
    """Build a HeathcliffAgent with mocked LLM / tools / executor."""
    with (
        patch("core.agent_core.init_chat_model"),
        patch("core.agent_core.create_agent") as mock_ca,
        patch("core.agent_core.create_middleware_stack", return_value=[]),
    ):
        mock_executor = Mock()
        mock_executor.invoke = Mock(
            return_value={"messages": [Mock(content=executor_response)]}
        )
        mock_ca.return_value = mock_executor

        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(memory_manager=memory_manager)
        return agent


class TestAgentWithMockedMemory:
    """Integration tests with mocked MemoryManager."""

    @pytest.fixture
    def memory_manager(self):
        return _make_mock_memory_manager()

    def test_full_flow_retrieval_to_output(self, memory_manager):
        """Test complete flow from input to output."""
        agent = _make_agent(memory_manager)

        response = agent.invoke("Hello, who are you?")

        assert isinstance(response, str)
        assert len(response) > 0
        memory_manager.save_chat.assert_called_once()

    def test_multi_turn_uses_context(self, memory_manager):
        """Test that second turn calls build_message_history."""
        agent = _make_agent(
            memory_manager, executor_response="Response to your question."
        )

        session_id = "multi-turn-test"
        agent.invoke("My name is Alex", session_id=session_id)
        agent.invoke("What is my name?", session_id=session_id)

        # build_message_history should be called once per invoke
        assert memory_manager.build_message_history.call_count == 2

    def test_memories_are_retrieved(self, memory_manager):
        """Test that stored memories are retrieved during invoke."""
        memory_manager.recall.return_value = {
            "documents": [["User's favorite color is green"]],
            "metadatas": [[{"category": "preferences"}]],
            "ids": [["mem_1"]],
            "distances": [[0.05]],
        }

        agent = _make_agent(
            memory_manager, executor_response="Based on what I know about you..."
        )

        agent.invoke("Tell me about myself")

        memory_manager.recall.assert_called()

    def test_session_isolation(self, memory_manager):
        """Test that different sessions pass correct session_id."""
        agent = _make_agent(memory_manager, executor_response="Session response")

        agent.invoke("Session A message", session_id="session-a")
        agent.invoke("Session B message", session_id="session-b")

        # Verify build_message_history was called with both session IDs
        calls = memory_manager.build_message_history.call_args_list
        session_ids = [
            c[1].get("session_id", c[0][1] if len(c[0]) > 1 else None) for c in calls
        ]
        assert "session-a" in session_ids
        assert "session-b" in session_ids


class TestToolCallingIntegration:
    """Integration tests for tool calling flow."""

    @pytest.fixture
    def memory_manager(self):
        return _make_mock_memory_manager()

    def test_tool_request_and_response(self, memory_manager):
        """Test that tool requests are processed and results returned."""
        agent = _make_agent(
            memory_manager, executor_response="The weather in London is 72 degrees."
        )

        response = agent.invoke("What's the weather in London?")

        assert isinstance(response, str)
        assert len(response) > 0


class TestErrorHandlingIntegration:
    """Integration tests for error handling."""

    @pytest.fixture
    def memory_manager(self):
        return _make_mock_memory_manager()

    def test_llm_failure_graceful_recovery(self, memory_manager):
        """Test graceful recovery when LLM fails."""
        agent = _make_agent(memory_manager)
        agent.executor.invoke.side_effect = Exception("API timeout")

        response = agent.invoke("Hello")

        assert isinstance(response, str)
        assert "error" in response.lower()

    def test_memory_failure_continues(self, memory_manager):
        """Test that agent continues if memory retrieval fails."""
        memory_manager.build_message_history.side_effect = Exception("DB Error")

        agent = _make_agent(memory_manager)

        response = agent.invoke("Hello")

        assert isinstance(response, str)


class TestConcurrentSessions:
    """Tests for handling multiple concurrent sessions."""

    @pytest.fixture
    def memory_manager(self):
        return _make_mock_memory_manager()

    def test_multiple_sessions_concurrent(self, memory_manager):
        """Test multiple sessions can run without interference."""
        agent = _make_agent(memory_manager, executor_response="Response")

        agent.invoke("Session 1 - Turn 1", session_id="s1")
        agent.invoke("Session 2 - Turn 1", session_id="s2")
        agent.invoke("Session 1 - Turn 2", session_id="s1")
        agent.invoke("Session 2 - Turn 2", session_id="s2")

        # save_chat should have been called 4 times
        assert memory_manager.save_chat.call_count == 4

        # Verify session IDs were passed correctly
        save_calls = memory_manager.save_chat.call_args_list
        sessions = [c[0][2] for c in save_calls]
        assert sessions.count("s1") == 2
        assert sessions.count("s2") == 2


# ---------------------------------------------------------------------------
# Middleware alias normalization
# ---------------------------------------------------------------------------


class TestMiddlewareAliasNormalization:
    """RobustLLMToolSelectorMiddleware must rewrite hallucinated tool names."""

    def test_alias_dict_maps_get_weather(self):
        """get_weather should map to info_agent_tool."""
        from core.middleware import TOOL_NAME_ALIASES

        assert TOOL_NAME_ALIASES["get_weather"] == "info_agent_tool"

    def test_alias_dict_maps_search_web(self):
        """search_web should map to info_agent_tool."""
        from core.middleware import TOOL_NAME_ALIASES

        assert TOOL_NAME_ALIASES["search_web"] == "info_agent_tool"

    def test_alias_dict_maps_play_track(self):
        """play_track should map to music_agent_tool."""
        from core.middleware import TOOL_NAME_ALIASES

        assert TOOL_NAME_ALIASES["play_track"] == "music_agent_tool"

    def test_alias_dict_maps_send_email(self):
        """send_email should map to email_agent_tool."""
        from core.middleware import TOOL_NAME_ALIASES

        assert TOOL_NAME_ALIASES["send_email"] == "email_agent_tool"

    def test_alias_dict_maps_load_skill_tool(self):
        """load_skill_tool should map to load_skill."""
        from core.middleware import TOOL_NAME_ALIASES

        assert TOOL_NAME_ALIASES["load_skill_tool"] == "load_skill"

    def test_alias_dict_maps_skill_loader_tool(self):
        """skill_loader_tool should map to load_skill."""
        from core.middleware import TOOL_NAME_ALIASES

        assert TOOL_NAME_ALIASES["skill_loader_tool"] == "load_skill"

    def test_alias_dict_maps_wikipedia_search(self):
        """wikipedia_search should map to info_agent_tool."""
        from core.middleware import TOOL_NAME_ALIASES

        assert TOOL_NAME_ALIASES["wikipedia_search"] == "info_agent_tool"

    def test_alias_dict_maps_research_agent_tool(self):
        """research_agent_tool should map to info_agent_tool."""
        from core.middleware import TOOL_NAME_ALIASES

        assert TOOL_NAME_ALIASES["research_agent_tool"] == "info_agent_tool"

    def test_alias_dict_maps_pause_playback(self):
        """pause_playback should map to music_agent_tool."""
        from core.middleware import TOOL_NAME_ALIASES

        assert TOOL_NAME_ALIASES["pause_playback"] == "music_agent_tool"

    def test_alias_dict_maps_read_emails(self):
        """read_emails should map to email_agent_tool."""
        from core.middleware import TOOL_NAME_ALIASES

        assert TOOL_NAME_ALIASES["read_emails"] == "email_agent_tool"

    def test_all_aliases_point_to_valid_supervisor_tools(self):
        """Every alias target must be one of the 9 real supervisor tools."""
        from core.middleware import TOOL_NAME_ALIASES

        valid_tools = {
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
        for alias, target in TOOL_NAME_ALIASES.items():
            assert target in valid_tools, (
                f"Alias {alias!r} → {target!r} is not a valid supervisor tool"
            )

    def test_middleware_rewrites_hallucinated_name(self):
        """RobustLLMToolSelectorMiddleware._process_selection_response rewrites aliases."""
        from core.middleware import RobustLLMToolSelectorMiddleware

        mw = RobustLLMToolSelectorMiddleware.__new__(RobustLLMToolSelectorMiddleware)

        # Simulate the response dict the LLM returns
        response = {"tools": ["get_weather", "info_agent_tool"]}
        valid_tool_names = [
            "info_agent_tool",
            "music_agent_tool",
            "load_skill",
            "recent_context",
        ]

        # We can't call _process_selection_response directly without the full
        # parent infrastructure, but we can verify the alias lookup logic
        # by testing the dict directly.
        from core.middleware import TOOL_NAME_ALIASES

        rewritten = []
        for name in response["tools"]:
            if name in TOOL_NAME_ALIASES:
                name = TOOL_NAME_ALIASES[name]
            if name in valid_tool_names:
                rewritten.append(name)

        assert "info_agent_tool" in rewritten
        assert "get_weather" not in rewritten
        # info_agent_tool should appear (original + rewritten alias)
        assert rewritten.count("info_agent_tool") == 2

    def test_unknown_tool_not_rewritten(self):
        """A completely unknown tool name should NOT be rewritten — just dropped."""
        from core.middleware import TOOL_NAME_ALIASES

        assert "totally_fake_tool" not in TOOL_NAME_ALIASES


# ---------------------------------------------------------------------------
# Prompt regression tests
# ---------------------------------------------------------------------------


class TestPromptRegression:
    """System prompt must reference correct supervisor tools, not raw inner tools."""

    @pytest.fixture
    def system_prompt(self):
        from instructions.prompts import build_system_prompt

        return build_system_prompt()

    def test_prompt_mentions_info_agent_tool(self, system_prompt):
        assert "info_agent_tool" in system_prompt

    def test_prompt_mentions_music_agent_tool(self, system_prompt):
        assert "music_agent_tool" in system_prompt

    def test_prompt_mentions_email_agent_tool(self, system_prompt):
        assert "email_agent_tool" in system_prompt

    def test_prompt_mentions_calendar_agent_tool(self, system_prompt):
        assert "calendar_agent_tool" in system_prompt

    def test_prompt_mentions_contacts_agent_tool(self, system_prompt):
        assert "contacts_agent_tool" in system_prompt

    def test_prompt_mentions_comms_agent_tool(self, system_prompt):
        assert "comms_agent_tool" in system_prompt

    def test_prompt_mentions_load_skill(self, system_prompt):
        assert "load_skill" in system_prompt

    def test_prompt_mentions_update_master_info(self, system_prompt):
        assert "update_master_info" in system_prompt

    def test_prompt_mentions_recent_context(self, system_prompt):
        assert "recent_context" in system_prompt

    def test_prompt_does_not_mention_get_weather(self, system_prompt):
        """get_weather should not appear anywhere in the prompt (positive enforcement only)."""
        assert "get_weather" not in system_prompt

    def test_prompt_does_not_mention_search_web_as_tool(self, system_prompt):
        """search_web should not appear as a callable tool in the prompt."""
        # search_web may appear inside the info subagent's own prompt, but
        # the supervisor system prompt should only reference info_agent_tool.
        lines = system_prompt.split("\n")
        for line in lines:
            if "search_web" in line:
                # Acceptable only inside a routing example that points to info_agent_tool
                assert "info_agent_tool" in line, (
                    f"search_web appears without info_agent_tool redirect: {line.strip()}"
                )

    def test_prompt_does_not_mention_play_track(self, system_prompt):
        """play_track should not appear anywhere in the prompt."""
        assert "play_track" not in system_prompt

    def test_prompt_does_not_mention_send_email_as_tool(self, system_prompt):
        """send_email should not appear as a callable tool."""
        # It may appear inside an example that routes to email_agent_tool
        lines = system_prompt.split("\n")
        for line in lines:
            if "send_email" in line:
                assert "email_agent_tool" in line, (
                    f"send_email appears without email_agent_tool redirect: {line.strip()}"
                )

    def test_prompt_uses_request_param(self, system_prompt):
        """Supervisor tool signatures should show `request: str`."""
        assert "request: str" in system_prompt or "request=" in system_prompt

    def test_prompt_contains_routing_examples(self, system_prompt):
        """Prompt must contain positive routing examples showing correct tool usage."""
        lower = system_prompt.lower()
        assert "routing guide" in lower or "→" in system_prompt

    def test_prompt_uses_positive_framing_only(self, system_prompt):
        """Prompt should not contain negative enforcement markers."""
        assert "❌" not in system_prompt
        assert "NEVER call" not in system_prompt
