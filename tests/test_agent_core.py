# ABOUTME: Unit tests for HeathcliffAgent - the master-class orchestrator
# ABOUTME: Tests initialization, invoke/ask methods, and factory functions

import os
import sys
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestHeathcliffAgentInit:
    """Tests for HeathcliffAgent initialization."""

    @pytest.fixture
    def mock_memory_manager(self):
        """Create a mock memory manager."""
        mm = Mock()
        mm.recall = Mock(
            return_value={
                "documents": [["User likes coffee"]],
                "metadatas": [[{"category": "preferences"}]],
                "ids": [["mem_123"]],
                "distances": [[0.1]],
            }
        )
        mm.get_recent_chats = Mock(return_value=[])
        mm.save_chat = Mock(return_value=("user_id", "asst_id"))
        return mm

    @patch("core.agent_core.ChatGoogleGenerativeAI")
    @patch("core.agent_core.get_all_tools")
    @patch("core.agent_core.create_agent")
    def test_agent_init_with_memory_manager(
        self, mock_create_agent, mock_get_tools, mock_llm_class, mock_memory_manager
    ):
        """Test agent initialization with provided memory manager."""
        mock_get_tools.return_value = []
        mock_create_agent.return_value = Mock()

        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(memory_manager=mock_memory_manager)

        assert agent.memory_manager == mock_memory_manager

    @patch("core.agent_core.ChatGoogleGenerativeAI")
    @patch("core.agent_core.get_all_tools")
    @patch("core.agent_core.create_agent")
    @patch("core.memory_manager.MemoryManager")
    def test_agent_init_auto_creates_memory_manager(
        self, mock_mm_class, mock_create_agent, mock_get_tools, mock_llm_class
    ):
        """Test that agent auto-creates MemoryManager if not provided."""
        mock_get_tools.return_value = []
        mock_create_agent.return_value = Mock()
        mock_mm_instance = Mock()
        mock_mm_instance.recall = Mock(
            return_value={"documents": [[]], "metadatas": [[]], "ids": [[]]}
        )
        mock_mm_instance.get_recent_chats = Mock(return_value=[])
        mock_mm_instance.save_chat = Mock(return_value=("uid", "aid"))
        mock_mm_class.return_value = mock_mm_instance

        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent()

        mock_mm_class.assert_called_once()
        assert agent.memory_manager == mock_mm_instance

    @patch("core.agent_core.ChatGoogleGenerativeAI")
    @patch("core.agent_core.get_all_tools")
    @patch("core.agent_core.create_agent")
    def test_agent_init_creates_llm(
        self, mock_create_agent, mock_get_tools, mock_llm_class, mock_memory_manager
    ):
        """Test that agent creates LLM instance."""
        mock_get_tools.return_value = []
        mock_create_agent.return_value = Mock()

        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(memory_manager=mock_memory_manager)

        mock_llm_class.assert_called_once()
        call_kwargs = mock_llm_class.call_args[1]
        assert "model" in call_kwargs
        assert "google_api_key" in call_kwargs

    @patch("core.agent_core.ChatGoogleGenerativeAI")
    @patch("core.agent_core.get_all_tools")
    @patch("core.agent_core.create_agent")
    def test_agent_has_executor(
        self, mock_create_agent, mock_get_tools, mock_llm_class, mock_memory_manager
    ):
        """Test that agent builds an executor."""
        mock_get_tools.return_value = []
        mock_executor = Mock()
        mock_create_agent.return_value = mock_executor

        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(memory_manager=mock_memory_manager)

        assert hasattr(agent, "executor")
        assert agent.executor == mock_executor


class TestAgentFactoryMethod:
    """Tests for HeathcliffAgent.create() factory method."""

    @patch("core.agent_core.ChatGoogleGenerativeAI")
    @patch("core.agent_core.get_all_tools")
    @patch("core.agent_core.create_agent")
    @patch("core.memory_manager.MemoryManager")
    def test_create_returns_agent(
        self, mock_mm_class, mock_create_agent, mock_get_tools, mock_llm_class
    ):
        """Test that create() returns a HeathcliffAgent instance."""
        mock_get_tools.return_value = []
        mock_create_agent.return_value = Mock()
        mock_mm_instance = Mock()
        mock_mm_instance.recall = Mock(
            return_value={"documents": [[]], "metadatas": [[]], "ids": [[]]}
        )
        mock_mm_instance.get_recent_chats = Mock(return_value=[])
        mock_mm_instance.save_chat = Mock(return_value=("uid", "aid"))
        mock_mm_class.return_value = mock_mm_instance

        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent.create()

        assert isinstance(agent, HeathcliffAgent)

    @patch("core.agent_core.ChatGoogleGenerativeAI")
    @patch("core.agent_core.get_all_tools")
    @patch("core.agent_core.create_agent")
    @patch("core.memory_manager.MemoryManager")
    def test_create_auto_initializes_memory(
        self, mock_mm_class, mock_create_agent, mock_get_tools, mock_llm_class
    ):
        """Test that create() auto-initializes memory manager."""
        mock_get_tools.return_value = []
        mock_create_agent.return_value = Mock()
        mock_mm_instance = Mock()
        mock_mm_instance.recall = Mock(
            return_value={"documents": [[]], "metadatas": [[]], "ids": [[]]}
        )
        mock_mm_instance.get_recent_chats = Mock(return_value=[])
        mock_mm_instance.save_chat = Mock(return_value=("uid", "aid"))
        mock_mm_class.return_value = mock_mm_instance

        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent.create()

        mock_mm_class.assert_called_once()
        assert agent.memory_manager == mock_mm_instance


class TestAskMethod:
    """Tests for ask() alias method."""

    @pytest.fixture
    def agent_with_mocks(self):
        """Create fully mocked agent."""
        with (
            patch("core.agent_core.ChatGoogleGenerativeAI") as mock_llm_class,
            patch("core.agent_core.get_all_tools") as mock_get_tools,
            patch("core.agent_core.create_agent") as mock_create_agent,
        ):
            mock_get_tools.return_value = []
            mock_executor = Mock()
            mock_executor.invoke = Mock(
                return_value={"messages": [Mock(content="I'm Heathcliff!")]}
            )
            mock_create_agent.return_value = mock_executor

            mm = Mock()
            mm.recall = Mock(
                return_value={"documents": [[]], "metadatas": [[]], "ids": [[]]}
            )
            mm.get_recent_chats = Mock(return_value=[])
            mm.save_chat = Mock(return_value=("uid", "aid"))

            from core.agent_core import HeathcliffAgent

            agent = HeathcliffAgent(memory_manager=mm)

        return agent

    def test_ask_returns_string(self, agent_with_mocks):
        """Test that ask() returns a string response."""
        response = agent_with_mocks.ask("Hello")

        assert isinstance(response, str)
        assert len(response) > 0

    def test_ask_calls_invoke(self, agent_with_mocks):
        """Test that ask() delegates to invoke()."""
        with patch.object(
            agent_with_mocks, "invoke", return_value="Test response"
        ) as mock_invoke:
            response = agent_with_mocks.ask("Hello", session_id="test-session")

            mock_invoke.assert_called_once_with(
                "Hello", session_id="test-session", additional_callbacks=None
            )


class TestInvoke:
    """Tests for invoke method."""

    @pytest.fixture
    def agent_with_mocks(self):
        """Create fully mocked agent."""
        with (
            patch("core.agent_core.ChatGoogleGenerativeAI") as mock_llm_class,
            patch("core.agent_core.get_all_tools") as mock_get_tools,
            patch("core.agent_core.create_agent") as mock_create_agent,
        ):
            mock_get_tools.return_value = []
            mock_executor = Mock()
            mock_executor.invoke = Mock(
                return_value={"messages": [Mock(content="I'm Heathcliff!")]}
            )
            mock_create_agent.return_value = mock_executor

            mm = Mock()
            mm.recall = Mock(
                return_value={"documents": [[]], "metadatas": [[]], "ids": [[]]}
            )
            mm.get_recent_chats = Mock(return_value=[])
            mm.save_chat = Mock(return_value=("uid", "aid"))

            from core.agent_core import HeathcliffAgent

            agent = HeathcliffAgent(memory_manager=mm)

        return agent

    def test_invoke_returns_string(self, agent_with_mocks):
        """Test that invoke returns a string response."""
        response = agent_with_mocks.invoke("Hello")

        assert isinstance(response, str)
        assert len(response) > 0

    def test_invoke_validates_empty_input(self, agent_with_mocks):
        """Test that invoke rejects empty input."""
        with pytest.raises(ValueError):
            agent_with_mocks.invoke("")

    def test_invoke_validates_whitespace_input(self, agent_with_mocks):
        """Test that invoke rejects whitespace-only input."""
        with pytest.raises(ValueError):
            agent_with_mocks.invoke("   ")

    def test_invoke_validates_long_input(self, agent_with_mocks):
        """Test that invoke rejects too-long input."""
        long_input = "A" * 10001  # Over 10k chars

        with pytest.raises(ValueError):
            agent_with_mocks.invoke(long_input)

    def test_invoke_generates_session_id(self, agent_with_mocks):
        """Test that invoke generates session_id if not provided."""
        response = agent_with_mocks.invoke("Hello")

        # Should not raise and should return response
        assert response is not None

    def test_invoke_uses_provided_session_id(self, agent_with_mocks):
        """Test that invoke uses provided session_id."""
        response = agent_with_mocks.invoke("Hello", session_id="my-session")

        # Verify session_id was used in save_chat
        save_call = agent_with_mocks.memory_manager.save_chat.call_args
        assert save_call[0][2] == "my-session"

    def test_invoke_queries_memory(self, agent_with_mocks):
        """Test that invoke queries the memory manager."""
        response = agent_with_mocks.invoke("Hello")

        agent_with_mocks.memory_manager.recall.assert_called()
        agent_with_mocks.memory_manager.get_recent_chats.assert_called()

    def test_invoke_saves_chat(self, agent_with_mocks):
        """Test that invoke saves the conversation."""
        response = agent_with_mocks.invoke("Hello", session_id="test-session")

        agent_with_mocks.memory_manager.save_chat.assert_called_once()
        save_args = agent_with_mocks.memory_manager.save_chat.call_args[0]
        assert save_args[0] == "Hello"  # user input
        assert save_args[2] == "test-session"  # session_id
