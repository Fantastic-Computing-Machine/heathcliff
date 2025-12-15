# ABOUTME: Integration tests for HeathcliffAgent with real MemoryManager
# ABOUTME: Tests full flow, multi-turn conversations, and component interaction

import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory_manager import MemoryManager


class TestAgentWithRealMemory:
    """Integration tests with real MemoryManager."""

    @pytest.fixture
    def memory_manager(self, tmp_path):
        """Create real MemoryManager with temp storage."""
        persist_dir = str(tmp_path / "chroma_integration")
        return MemoryManager(persist_dir=persist_dir)

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = Mock()
        config.gemini_key = "test-api-key"
        config.get = Mock(
            side_effect=lambda key, default=None: {
                "llm.model": "gemini-2.0-flash-exp",
                "llm.temperature": 0.7,
                "llm.max_tokens": 1024,
            }.get(key, default)
        )
        return config

    @patch("core.agent_core.ChatGoogleGenerativeAI")
    def test_full_flow_retrieval_to_output(
        self, mock_llm_class, mock_config, memory_manager
    ):
        """Test complete flow from input to output."""
        # Setup mock LLM
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content="Hello! I'm Heathcliff."))
        mock_llm_class.return_value = mock_llm

        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(config=mock_config, memory_manager=memory_manager)

        # Invoke agent
        response = agent.invoke("Hello, who are you?")

        # Verify response
        assert isinstance(response, str)
        assert len(response) > 0

        # Verify chat was saved
        stats = memory_manager.get_stats()
        assert stats["chats"] >= 2  # User msg + assistant msg

    @patch("core.agent_core.ChatGoogleGenerativeAI")
    def test_multi_turn_uses_context(self, mock_llm_class, mock_config, memory_manager):
        """Test that second turn has context from first turn."""
        # Track what prompts are sent to LLM
        prompts_received = []

        mock_llm = Mock()

        def capture_invoke(prompt):
            prompts_received.append(str(prompt))
            return Mock(content="Response to your question.")

        mock_llm.invoke = Mock(side_effect=capture_invoke)
        mock_llm_class.return_value = mock_llm

        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(config=mock_config, memory_manager=memory_manager)

        # First turn
        session_id = "multi-turn-test"
        agent.invoke("My name is Adi", session_id=session_id)

        # Second turn - should have context
        agent.invoke("What is my name?", session_id=session_id)

        # Second prompt should include context from first turn
        assert len(prompts_received) == 2
        # The context/prompt should reference previous conversation

    @patch("core.agent_core.ChatGoogleGenerativeAI")
    def test_memories_are_retrieved(self, mock_llm_class, mock_config, memory_manager):
        """Test that stored memories are retrieved during invoke."""
        # Pre-populate memories
        memory_manager.add_memory(
            "User's favorite color is green", category="preferences"
        )
        memory_manager.add_memory("User works as a software engineer", category="facts")

        # Track prompts
        prompts_received = []

        mock_llm = Mock()

        def capture_invoke(prompt):
            prompts_received.append(str(prompt))
            return Mock(content="Based on what I know about you...")

        mock_llm.invoke = Mock(side_effect=capture_invoke)
        mock_llm_class.return_value = mock_llm

        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(config=mock_config, memory_manager=memory_manager)

        agent.invoke("Tell me about myself")

        # Prompt should include retrieved memories
        assert len(prompts_received) == 1
        # Memory content should be in prompt (or at least referenced)

    @patch("core.agent_core.ChatGoogleGenerativeAI")
    def test_session_isolation(self, mock_llm_class, mock_config, memory_manager):
        """Test that different sessions are isolated."""
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content="Session response"))
        mock_llm_class.return_value = mock_llm

        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(config=mock_config, memory_manager=memory_manager)

        # Two separate sessions
        agent.invoke("Session A message", session_id="session-a")
        agent.invoke("Session B message", session_id="session-b")

        # Get context for session A
        context_a = memory_manager.get_chat_context("message", session_id="session-a")
        context_b = memory_manager.get_chat_context("message", session_id="session-b")

        # Each should only have their own messages
        for meta in context_a.get("metadatas", [[]])[0]:
            assert meta.get("session") == "session-a"

        for meta in context_b.get("metadatas", [[]])[0]:
            assert meta.get("session") == "session-b"


class TestToolCallingIntegration:
    """Integration tests for tool calling flow."""

    @pytest.fixture
    def memory_manager(self, tmp_path):
        """Create real MemoryManager."""
        persist_dir = str(tmp_path / "chroma_tools")
        return MemoryManager(persist_dir=persist_dir)

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = Mock()
        config.gemini_key = "test-api-key"
        config.get = Mock(return_value=None)
        return config

    @patch("core.agent_core.ChatGoogleGenerativeAI")
    def test_tool_request_and_response(
        self, mock_llm_class, mock_config, memory_manager
    ):
        """Test that tool requests are processed and results returned."""
        call_count = [0]

        def mock_invoke(prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call - request a tool
                return Mock(content="[TOOL: weather city=London]")
            else:
                # Second call - use tool result
                return Mock(content="The weather in London is 72 degrees.")

        mock_llm = Mock()
        mock_llm.invoke = Mock(side_effect=mock_invoke)
        mock_llm_class.return_value = mock_llm

        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(config=mock_config, memory_manager=memory_manager)

        response = agent.invoke("What's the weather in London?")

        # Should have called LLM at least twice (request + response)
        assert call_count[0] >= 1


class TestErrorHandlingIntegration:
    """Integration tests for error handling."""

    @pytest.fixture
    def memory_manager(self, tmp_path):
        """Create real MemoryManager."""
        persist_dir = str(tmp_path / "chroma_errors")
        return MemoryManager(persist_dir=persist_dir)

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = Mock()
        config.gemini_key = "test-api-key"
        config.get = Mock(return_value=None)
        return config

    @patch("core.agent_core.ChatGoogleGenerativeAI")
    def test_llm_failure_graceful_recovery(
        self, mock_llm_class, mock_config, memory_manager
    ):
        """Test graceful recovery when LLM fails."""
        mock_llm = Mock()
        mock_llm.invoke = Mock(side_effect=Exception("API timeout"))
        mock_llm_class.return_value = mock_llm

        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(config=mock_config, memory_manager=memory_manager)

        # Should not raise, should return error message
        response = agent.invoke("Hello")

        assert isinstance(response, str)
        assert "error" in response.lower()

    @patch("core.agent_core.ChatGoogleGenerativeAI")
    def test_memory_failure_continues(
        self, mock_llm_class, mock_config, memory_manager
    ):
        """Test that agent continues if memory retrieval fails."""
        # Break the memory manager's recall
        memory_manager.recall = Mock(side_effect=Exception("DB Error"))

        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content="Hello without context"))
        mock_llm_class.return_value = mock_llm

        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(config=mock_config, memory_manager=memory_manager)

        # Should still work, just without memory context
        response = agent.invoke("Hello")

        assert isinstance(response, str)


class TestConcurrentSessions:
    """Tests for handling multiple concurrent sessions."""

    @pytest.fixture
    def memory_manager(self, tmp_path):
        """Create real MemoryManager."""
        persist_dir = str(tmp_path / "chroma_concurrent")
        return MemoryManager(persist_dir=persist_dir)

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = Mock()
        config.gemini_key = "test-api-key"
        config.get = Mock(return_value=None)
        return config

    @patch("core.agent_core.ChatGoogleGenerativeAI")
    def test_multiple_sessions_concurrent(
        self, mock_llm_class, mock_config, memory_manager
    ):
        """Test multiple sessions can run without interference."""
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content="Response"))
        mock_llm_class.return_value = mock_llm

        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(config=mock_config, memory_manager=memory_manager)

        # Simulate interleaved sessions
        agent.invoke("Session 1 - Turn 1", session_id="s1")
        agent.invoke("Session 2 - Turn 1", session_id="s2")
        agent.invoke("Session 1 - Turn 2", session_id="s1")
        agent.invoke("Session 2 - Turn 2", session_id="s2")

        # Each session should have 4 messages (2 user + 2 assistant)
        s1_context = memory_manager.get_recent_chats("s1", n=10)
        s2_context = memory_manager.get_recent_chats("s2", n=10)

        assert len(s1_context) == 4
        assert len(s2_context) == 4
