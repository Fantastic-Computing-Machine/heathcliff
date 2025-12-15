# ABOUTME: Unit tests for HeathcliffAgent LangGraph-based orchestrator
# ABOUTME: Tests state management, graph nodes, and invoke functionality

import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAgentState:
    """Tests for AgentState TypedDict structure."""

    def test_agent_state_structure(self):
        """Test that AgentState has all required fields."""
        from core.agent_core import AgentState

        # Create a valid state
        state: AgentState = {
            "messages": [],
            "user_input": "Hello",
            "session_id": "test-session",
            "context": "",
            "memories": [],
            "tool_calls": [],
            "final_response": "",
        }

        assert "messages" in state
        assert "user_input" in state
        assert "session_id" in state
        assert "context" in state
        assert "memories" in state
        assert "tool_calls" in state
        assert "final_response" in state


class TestHeathcliffAgentInit:
    """Tests for HeathcliffAgent initialization."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config object."""
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

    @pytest.fixture
    def mock_memory_manager(self, tmp_path):
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
        mm.get_chat_context = Mock(
            return_value={
                "documents": [["Previous conversation"]],
                "metadatas": [[{"role": "user", "session": "test"}]],
                "ids": [["chat_123"]],
                "distances": [[0.2]],
            }
        )
        mm.save_chat = Mock(return_value=("user_id", "asst_id"))
        return mm

    @patch("core.agent_core.ChatGoogleGenerativeAI")
    def test_agent_init_with_config(
        self, mock_llm_class, mock_config, mock_memory_manager
    ):
        """Test agent initialization with config."""
        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(config=mock_config, memory_manager=mock_memory_manager)

        assert agent.memory_manager == mock_memory_manager
        assert agent.graph is not None

    @patch("core.agent_core.ChatGoogleGenerativeAI")
    def test_agent_init_creates_llm(
        self, mock_llm_class, mock_config, mock_memory_manager
    ):
        """Test that agent creates LLM instance."""
        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(config=mock_config, memory_manager=mock_memory_manager)

        mock_llm_class.assert_called_once()
        call_kwargs = mock_llm_class.call_args[1]
        assert call_kwargs["model"] == "gemini-2.0-flash-exp"
        assert call_kwargs["google_api_key"] == "test-api-key"

    @patch("core.agent_core.ChatGoogleGenerativeAI")
    def test_agent_has_graph(self, mock_llm_class, mock_config, mock_memory_manager):
        """Test that agent builds a graph."""
        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(config=mock_config, memory_manager=mock_memory_manager)

        assert hasattr(agent, "graph")
        assert agent.graph is not None


class TestRetrievalNode:
    """Tests for _retrieval_node method."""

    @pytest.fixture
    def agent_with_mocks(self, tmp_path):
        """Create agent with mocked dependencies."""
        config = Mock()
        config.gemini_key = "test-key"
        config.get = Mock(return_value=None)

        mm = Mock()
        mm.recall = Mock(
            return_value={
                "documents": [["Memory 1", "Memory 2"]],
                "metadatas": [[{}, {}]],
                "ids": [["id1", "id2"]],
                "distances": [[0.1, 0.2]],
            }
        )
        mm.get_chat_context = Mock(
            return_value={
                "documents": [["Chat 1"]],
                "metadatas": [[{"role": "user"}]],
                "ids": [["chat1"]],
                "distances": [[0.15]],
            }
        )

        with patch("core.agent_core.ChatGoogleGenerativeAI"):
            from core.agent_core import HeathcliffAgent

            agent = HeathcliffAgent(config=config, memory_manager=mm)

        return agent

    def test_retrieval_queries_memory(self, agent_with_mocks):
        """Test that retrieval node queries memory manager."""
        state = {
            "messages": [],
            "user_input": "What's my name?",
            "session_id": "test-session",
            "context": "",
            "memories": [],
            "tool_calls": [],
            "final_response": "",
        }

        result = agent_with_mocks._retrieval_node(state)

        agent_with_mocks.memory_manager.recall.assert_called()
        agent_with_mocks.memory_manager.get_chat_context.assert_called()

    def test_retrieval_updates_context(self, agent_with_mocks):
        """Test that retrieval node updates context field."""
        state = {
            "messages": [],
            "user_input": "Test query",
            "session_id": "test-session",
            "context": "",
            "memories": [],
            "tool_calls": [],
            "final_response": "",
        }

        result = agent_with_mocks._retrieval_node(state)

        assert "context" in result
        assert result["context"] != ""

    def test_retrieval_updates_memories(self, agent_with_mocks):
        """Test that retrieval node updates memories field."""
        state = {
            "messages": [],
            "user_input": "Test query",
            "session_id": "test-session",
            "context": "",
            "memories": [],
            "tool_calls": [],
            "final_response": "",
        }

        result = agent_with_mocks._retrieval_node(state)

        assert "memories" in result
        assert len(result["memories"]) > 0


class TestReasoningNode:
    """Tests for _reasoning_node method."""

    @pytest.fixture
    def agent_with_mocks(self):
        """Create agent with mocked LLM."""
        config = Mock()
        config.gemini_key = "test-key"
        config.get = Mock(return_value=None)

        mm = Mock()
        mm.recall = Mock(
            return_value={"documents": [[]], "metadatas": [[]], "ids": [[]]}
        )
        mm.get_chat_context = Mock(
            return_value={"documents": [[]], "metadatas": [[]], "ids": [[]]}
        )

        with patch("core.agent_core.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = Mock()
            mock_llm.invoke = Mock(return_value=Mock(content="Test response"))
            mock_llm_class.return_value = mock_llm

            from core.agent_core import HeathcliffAgent

            agent = HeathcliffAgent(config=config, memory_manager=mm)

        return agent

    def test_reasoning_calls_llm(self, agent_with_mocks):
        """Test that reasoning node calls the LLM."""
        state = {
            "messages": [],
            "user_input": "Hello",
            "session_id": "test",
            "context": "Some context",
            "memories": ["Memory 1"],
            "tool_calls": [],
            "final_response": "",
        }

        result = agent_with_mocks._reasoning_node(state)

        agent_with_mocks.llm.invoke.assert_called()

    def test_reasoning_sets_final_response(self, agent_with_mocks):
        """Test that reasoning sets final_response when no tools needed."""
        state = {
            "messages": [],
            "user_input": "Hello",
            "session_id": "test",
            "context": "Context",
            "memories": [],
            "tool_calls": [],
            "final_response": "",
        }

        result = agent_with_mocks._reasoning_node(state)

        assert "final_response" in result


class TestToolCallingNode:
    """Tests for _tool_calling_node method."""

    @pytest.fixture
    def agent_with_mocks(self):
        """Create agent with mocked dependencies."""
        config = Mock()
        config.gemini_key = "test-key"
        config.get = Mock(return_value=None)

        mm = Mock()

        with patch("core.agent_core.ChatGoogleGenerativeAI"):
            from core.agent_core import HeathcliffAgent

            agent = HeathcliffAgent(config=config, memory_manager=mm)

        return agent

    def test_tool_calling_clears_tool_calls(self, agent_with_mocks):
        """Test that tool node clears tool_calls after execution."""
        state = {
            "messages": [],
            "user_input": "What's the weather?",
            "session_id": "test",
            "context": "",
            "memories": [],
            "tool_calls": [{"name": "weather", "args": {"city": "SF"}}],
            "final_response": "",
        }

        result = agent_with_mocks._tool_calling_node(state)

        assert result["tool_calls"] == []

    def test_tool_calling_adds_to_messages(self, agent_with_mocks):
        """Test that tool results are added to messages."""
        state = {
            "messages": [],
            "user_input": "What's the time?",
            "session_id": "test",
            "context": "",
            "memories": [],
            "tool_calls": [{"name": "time", "args": {}}],
            "final_response": "",
        }

        result = agent_with_mocks._tool_calling_node(state)

        assert len(result["messages"]) > 0


class TestOutputNode:
    """Tests for _output_node method."""

    @pytest.fixture
    def agent_with_mocks(self):
        """Create agent with mocked memory manager."""
        config = Mock()
        config.gemini_key = "test-key"
        config.get = Mock(return_value=None)

        mm = Mock()
        mm.save_chat = Mock(return_value=("uid", "aid"))

        with patch("core.agent_core.ChatGoogleGenerativeAI"):
            from core.agent_core import HeathcliffAgent

            agent = HeathcliffAgent(config=config, memory_manager=mm)

        return agent

    def test_output_saves_chat(self, agent_with_mocks):
        """Test that output node saves conversation to memory."""
        state = {
            "messages": [],
            "user_input": "Hello there",
            "session_id": "test-session",
            "context": "",
            "memories": [],
            "tool_calls": [],
            "final_response": "Hi! How can I help?",
        }

        result = agent_with_mocks._output_node(state)

        agent_with_mocks.memory_manager.save_chat.assert_called_once_with(
            "Hello there", "Hi! How can I help?", "test-session"
        )


class TestInvoke:
    """Tests for invoke method."""

    @pytest.fixture
    def agent_with_mocks(self):
        """Create fully mocked agent."""
        config = Mock()
        config.gemini_key = "test-key"
        config.get = Mock(return_value=None)

        mm = Mock()
        mm.recall = Mock(
            return_value={"documents": [[]], "metadatas": [[]], "ids": [[]]}
        )
        mm.get_chat_context = Mock(
            return_value={"documents": [[]], "metadatas": [[]], "ids": [[]]}
        )
        mm.save_chat = Mock(return_value=("uid", "aid"))

        with patch("core.agent_core.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = Mock()
            mock_llm.invoke = Mock(return_value=Mock(content="I'm Heathcliff!"))
            mock_llm_class.return_value = mock_llm

            from core.agent_core import HeathcliffAgent

            agent = HeathcliffAgent(config=config, memory_manager=mm)

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

    def test_invoke_handles_llm_error(self, agent_with_mocks):
        """Test that invoke handles LLM errors gracefully."""
        agent_with_mocks.llm.invoke.side_effect = Exception("API Error")

        response = agent_with_mocks.invoke("Hello")

        # Should return error message, not raise
        assert "error" in response.lower()


class TestGraphRouting:
    """Tests for graph routing logic."""

    @pytest.fixture
    def agent_with_mocks(self):
        """Create agent for routing tests."""
        config = Mock()
        config.gemini_key = "test-key"
        config.get = Mock(return_value=None)

        mm = Mock()
        mm.recall = Mock(
            return_value={"documents": [[]], "metadatas": [[]], "ids": [[]]}
        )
        mm.get_chat_context = Mock(
            return_value={"documents": [[]], "metadatas": [[]], "ids": [[]]}
        )
        mm.save_chat = Mock(return_value=("uid", "aid"))

        with patch("core.agent_core.ChatGoogleGenerativeAI") as mock_llm_class:
            mock_llm = Mock()
            mock_llm.invoke = Mock(return_value=Mock(content="Test response"))
            mock_llm_class.return_value = mock_llm

            from core.agent_core import HeathcliffAgent

            agent = HeathcliffAgent(config=config, memory_manager=mm)

        return agent

    def test_routing_to_output_when_no_tools(self, agent_with_mocks):
        """Test that state with no tool_calls routes to output."""
        state = {
            "messages": [],
            "user_input": "Hello",
            "session_id": "test",
            "context": "",
            "memories": [],
            "tool_calls": [],
            "final_response": "Hi there!",
        }

        route = agent_with_mocks._route_after_reasoning(state)
        assert route == "output_node"

    def test_routing_to_tools_when_tools_requested(self, agent_with_mocks):
        """Test that state with tool_calls routes to tool node."""
        state = {
            "messages": [],
            "user_input": "What's the weather?",
            "session_id": "test",
            "context": "",
            "memories": [],
            "tool_calls": [{"name": "weather"}],
            "final_response": "",
        }

        route = agent_with_mocks._route_after_reasoning(state)
        assert route == "tool_node"
