# ABOUTME: End-to-end tests for HeathcliffAgent with mocked components
# ABOUTME: Tests complete conversation flows with mocked Gemini API

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
# Shared helpers
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
    """Build a HeathcliffAgent with mocked LLM / coordinator.

    Patches persist until the test's reset_singleton fixture runs.
    """
    patchers = [
        patch("core.agent_core.init_chat_model"),
        patch("core.agent_core.build_coordinator_graph", return_value=Mock()),
        patch("core.agent_core.build_default_registry"),
        patch("core.agent_core.invoke_coordinator", return_value=executor_response),
    ]
    for p in patchers:
        p.start()

    from core.agent_core import HeathcliffAgent

    agent = HeathcliffAgent(memory_manager=memory_manager)

    # Stop construction-only patches; keep invoke_coordinator active
    for p in patchers[:-1]:
        p.stop()

    # Store the invoke mock patcher so tests can stop/replace it
    agent._invoke_patcher = patchers[-1]  # type: ignore[attr-defined]
    return agent


class TestBasicConversation:
    """E2E tests for basic conversation scenarios."""

    @pytest.fixture
    def memory_manager(self):
        return _make_mock_memory_manager()

    def test_greeting_response(self, memory_manager):
        """Test that agent responds to greetings appropriately."""
        agent = _make_agent(
            memory_manager,
            executor_response="Hello! I'm Heathcliff, your personal AI assistant. How can I help you today?",
        )

        response = agent.invoke("Hello!")

        assert isinstance(response, str)
        assert len(response) > 0
        assert any(
            word in response.lower() for word in ["hello", "hi", "help", "heathcliff"]
        )

    def test_question_response(self, memory_manager):
        """Test that agent responds to questions."""
        agent = _make_agent(
            memory_manager,
            executor_response="Based on what I know, I can help you with various tasks like managing your calendar, checking the weather, and more.",
        )

        response = agent.invoke("What can you do?")

        assert isinstance(response, str)
        assert len(response) > 10

    def test_context_aware_response(self, memory_manager):
        """Test that agent uses context in responses."""
        memory_manager.recall.return_value = {
            "documents": [["User's name is Alex"]],
            "metadatas": [[{"category": "facts"}]],
            "ids": [["mem_1"]],
            "distances": [[0.05]],
        }

        agent = _make_agent(
            memory_manager, executor_response="Hello Alex! Great to see you again."
        )

        response = agent.invoke("Hi there!")

        assert isinstance(response, str)
        memory_manager.recall.assert_called()


class TestConversationContinuity:
    """E2E tests for multi-turn conversation continuity."""

    @pytest.fixture
    def memory_manager(self):
        return _make_mock_memory_manager()

    def test_remembers_previous_turn(self, memory_manager):
        """Test that agent calls build_langchain_history for context on each turn."""
        turn_count = [0]

        def dynamic_response(*args, **kwargs):
            turn_count[0] += 1
            if turn_count[0] == 1:
                return "Nice to meet you, Alex!"
            return "Your name is Alex, as you mentioned earlier."

        agent = _make_agent(memory_manager)

        session = "continuity-test"

        with patch("core.agent_core.invoke_coordinator", side_effect=dynamic_response):
            agent.invoke("My name is Alex", conversation_id=session)
            agent.invoke("What's my name?", conversation_id=session)

        assert memory_manager.build_langchain_history.call_count == 2

    def test_three_turn_conversation(self, memory_manager):
        """Test a three-turn conversation maintains context."""
        turn = [0]

        def multi_turn_response(*args, **kwargs):
            turn[0] += 1
            return f"Response to turn {turn[0]}"

        agent = _make_agent(memory_manager)

        session = "three-turn"

        with patch(
            "core.agent_core.invoke_coordinator", side_effect=multi_turn_response
        ):
            r1 = agent.invoke("First message", conversation_id=session)
            r2 = agent.invoke("Second message", conversation_id=session)
            r3 = agent.invoke("Third message", conversation_id=session)

        assert r1 is not None
        assert r2 is not None
        assert r3 is not None

        # save_turn should be called 3 times
        assert memory_manager.save_turn.call_count == 3


class TestErrorRecovery:
    """E2E tests for error recovery scenarios."""

    @pytest.fixture
    def memory_manager(self):
        return _make_mock_memory_manager()

    def test_recovers_from_transient_error(self, memory_manager):
        """Test recovery from a transient API error."""
        call_count = [0]

        def flaky_coordinator(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Temporary network error")
            return "Recovered successfully!"

        agent = _make_agent(memory_manager)

        with patch("core.agent_core.invoke_coordinator", side_effect=flaky_coordinator):
            r1 = agent.invoke("Hello")
            assert "error" in r1.lower()

            r2 = agent.invoke("Try again")
            assert isinstance(r2, str)

    def test_handles_invalid_input_gracefully(self, memory_manager):
        """Test handling of edge case inputs."""
        agent = _make_agent(memory_manager, executor_response="I understand")

        r1 = agent.invoke("?")
        assert isinstance(r1, str)

        r2 = agent.invoke("Hello! @#$%^&*()")
        assert isinstance(r2, str)

        r3 = agent.invoke("Hello ")
        assert isinstance(r3, str)


class TestConversationQuality:
    """E2E tests for conversation quality metrics."""

    @pytest.fixture
    def memory_manager(self):
        return _make_mock_memory_manager()

    def test_response_not_empty(self, memory_manager):
        """Test that responses are never empty."""
        agent = _make_agent(
            memory_manager, executor_response="Here's a helpful response."
        )

        for query in ["Hello", "What time is it?", "Tell me a joke"]:
            response = agent.invoke(query)
            assert response is not None
            assert len(response.strip()) > 0

    def test_response_reasonable_length(self, memory_manager):
        """Test that responses are reasonably sized."""
        agent = _make_agent(
            memory_manager,
            executor_response="This is a reasonable length response that provides helpful information.",
        )

        response = agent.invoke("Tell me about yourself")

        assert len(response) > 10
        assert len(response) < 10000
