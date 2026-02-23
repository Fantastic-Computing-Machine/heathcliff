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
