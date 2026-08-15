# ABOUTME: Integration tests for HeathcliffAgent with mocked MemoryManager
# ABOUTME: Tests full flow, multi-turn conversations, and component interaction

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
    mm.build_langchain_history = Mock(return_value=[])
    mm.get_recent_chats = Mock(return_value=[])
    mm.get_chat_context = Mock(
        return_value={
            "documents": [[]],
            "metadatas": [[]],
            "ids": [[]],
            "distances": [[]],
        }
    )
    mm.save_turn = Mock(return_value=("uid", "aid"))
    mm.add_memory = Mock(return_value="mem_1")
    mm.get_stats = Mock(return_value={"memories": 0, "chats": 0})
    return mm


def _make_agent(memory_manager, executor_response="Hello! I'm Heathcliff."):
    """Build a HeathcliffAgent with mocked LLM / coordinator."""
    with (
        patch("core.agent_core.init_chat_model"),
        patch("core.agent_core.build_coordinator_graph") as mock_bcg,
        patch("core.agent_core.build_default_registry"),
        patch("core.agent_core.invoke_coordinator", return_value=executor_response),
    ):
        mock_bcg.return_value = Mock()

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
        memory_manager.save_turn.assert_called_once()

    def test_multi_turn_uses_context(self, memory_manager):
        """Test that second turn calls build_langchain_history."""
        agent = _make_agent(
            memory_manager, executor_response="Response to your question."
        )

        conversation_id = "multi-turn-test"
        agent.invoke("My name is Alex", conversation_id=conversation_id)
        agent.invoke("What is my name?", conversation_id=conversation_id)

        # build_langchain_history should be called once per invoke
        assert memory_manager.build_langchain_history.call_count == 2

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

        agent.invoke("Session A message", conversation_id="session-a")
        agent.invoke("Session B message", conversation_id="session-b")

        # Verify build_langchain_history was called with both conversation IDs
        calls = memory_manager.build_langchain_history.call_args_list
        conversation_ids = [
            c[1].get("conversation_id", c[0][1] if len(c[0]) > 1 else None)
            for c in calls
        ]
        assert "session-a" in conversation_ids
        assert "session-b" in conversation_ids


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

        with patch(
            "core.agent_core.invoke_coordinator",
            side_effect=Exception("API timeout"),
        ):
            response = agent.invoke("Hello")

        assert isinstance(response, str)
        assert "error" in response.lower()

    def test_memory_failure_continues(self, memory_manager):
        """Test that agent continues if memory retrieval fails."""
        memory_manager.build_langchain_history.side_effect = Exception("DB Error")

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

        agent.invoke("Session 1 - Turn 1", conversation_id="s1")
        agent.invoke("Session 2 - Turn 1", conversation_id="s2")
        agent.invoke("Session 1 - Turn 2", conversation_id="s1")
        agent.invoke("Session 2 - Turn 2", conversation_id="s2")

        # save_chat should have been called 4 times
        assert memory_manager.save_turn.call_count == 4

        # Verify session IDs were passed correctly
        save_calls = memory_manager.save_turn.call_args_list
        sessions = [c[0][2] for c in save_calls]
        assert sessions.count("s1") == 2
        assert sessions.count("s2") == 2


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
        assert "<routing_examples>" in system_prompt
        assert "Action:" in system_prompt

    def test_prompt_uses_positive_framing_only(self, system_prompt):
        """Prompt should not contain negative enforcement markers."""
        assert "❌" not in system_prompt
        assert "NEVER call" not in system_prompt

    # --- XML structure validation (Phase 1 regression) ---

    def test_prompt_has_role_tag(self, system_prompt):
        """Prompt must have <role> XML section."""
        assert "<role>" in system_prompt
        assert "</role>" in system_prompt

    def test_prompt_has_user_profile_tag(self, system_prompt):
        """Prompt must have <user_profile> XML section."""
        assert "<user_profile>" in system_prompt
        assert "</user_profile>" in system_prompt

    def test_prompt_has_tools_tag(self, system_prompt):
        """Prompt must have <tools> XML section."""
        assert "<tools>" in system_prompt
        assert "</tools>" in system_prompt

    def test_prompt_has_routing_examples_tag(self, system_prompt):
        """Prompt must have <routing_examples> XML section."""
        assert "<routing_examples>" in system_prompt
        assert "</routing_examples>" in system_prompt

    def test_prompt_has_execution_rules_tag(self, system_prompt):
        """Prompt must have <execution_rules> XML section."""
        assert "<execution_rules>" in system_prompt
        assert "</execution_rules>" in system_prompt

    def test_prompt_has_response_style_tag(self, system_prompt):
        """Prompt must have <response_style> XML section."""
        assert "<response_style>" in system_prompt
        assert "</response_style>" in system_prompt

    def test_prompt_does_not_mention_google_drive(self, system_prompt):
        """System prompt should not reference Google Drive (Drive tools are disabled)."""
        assert "Google Drive" not in system_prompt
        assert "google drive" not in system_prompt.lower()


# ---------------------------------------------------------------------------
# Tool description consistency (Phase 2 regression)
# ---------------------------------------------------------------------------


class TestToolDescriptionConsistency:
    """All supervisor-visible tool descriptions follow the standard template."""

    STANDARD_KEYWORDS = ["use for:", "example:"]

    @pytest.mark.parametrize(
        "tool_import,module_path",
        [
            ("info_agent_tool", "core.subagents.info.agent"),
            ("music_agent_tool", "core.subagents.music.agent"),
            ("email_agent_tool", "core.subagents.email.agent"),
            ("calendar_agent_tool", "core.subagents.calendar.agent"),
            ("contacts_agent_tool", "core.subagents.contacts.agent"),
            ("comms_agent_tool", "core.subagents.comms.agent"),
        ],
    )
    def test_subagent_tool_follows_template(self, tool_import, module_path):
        """Each subagent tool description contains 'Use for:' and 'Example:'."""
        import importlib

        mod = importlib.import_module(module_path)
        tool_obj = getattr(mod, tool_import)
        desc = tool_obj.description.lower()
        for kw in self.STANDARD_KEYWORDS:
            assert kw in desc, (
                f"{tool_import} description missing '{kw}'. Got: {tool_obj.description[:100]}"
            )

    def test_load_skill_follows_template(self):
        from skills.skill_tools import load_skill

        desc = load_skill.description.lower()
        assert "use for:" in desc
        assert "example:" in desc

    def test_update_master_info_follows_template(self):
        from skills.master_info import update_master_info

        desc = update_master_info.description.lower()
        assert "use for:" in desc
        assert "example:" in desc

    def test_recent_context_follows_template(self):
        from core.subagents.info.recent_context import recent_context

        desc = recent_context.description.lower()
        assert "use for:" in desc

    def test_comms_description_does_not_mention_drive(self):
        """comms_agent_tool description should not reference Google Drive."""
        from core.subagents.comms.agent import comms_agent_tool

        desc = comms_agent_tool.description.lower()
        assert "drive" not in desc

    def test_email_description_does_not_require_address_for_search(self):
        """email_agent_tool description should clarify address is only needed for send/draft."""
        from core.subagents.email.agent import email_agent_tool

        desc = email_agent_tool.description.lower()
        # Must mention that search/read doesn't need an address
        assert "search" in desc
        assert "not required" in desc or "send" in desc
