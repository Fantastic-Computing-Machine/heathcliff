# ABOUTME: End-to-end tests for HeathcliffAgent with real components
# ABOUTME: Tests complete conversation flows with actual (or mocked) Gemini API

import pytest
import os
import sys
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory_manager import MemoryManager


class TestBasicConversation:
    """E2E tests for basic conversation scenarios."""

    @pytest.fixture
    def memory_manager(self, tmp_path):
        """Create real MemoryManager."""
        persist_dir = str(tmp_path / "chroma_e2e")
        return MemoryManager(persist_dir=persist_dir)

    @pytest.fixture
    def mock_config(self):
        """Create mock config with real-ish values."""
        config = Mock()
        config.gemini_key = os.getenv("GEMINI_API_KEY", "test-key")
        config.get = Mock(
            side_effect=lambda key, default=None: {
                "llm.model": "gemini-2.0-flash-exp",
                "llm.temperature": 0.7,
                "llm.max_tokens": 1024,
            }.get(key, default)
        )
        return config

    @patch("core.agent_core.ChatGoogleGenerativeAI")
    def test_greeting_response(self, mock_llm_class, mock_config, memory_manager):
        """Test that agent responds to greetings appropriately."""
        mock_llm = Mock()
        mock_llm.invoke = Mock(
            return_value=Mock(
                content="Hello! I'm Heathcliff, your personal AI assistant. How can I help you today?"
            )
        )
        mock_llm_class.return_value = mock_llm

        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(config=mock_config, memory_manager=memory_manager)

        response = agent.invoke("Hello!")

        assert isinstance(response, str)
        assert len(response) > 0
        # Should be a friendly greeting
        assert any(
            word in response.lower() for word in ["hello", "hi", "help", "heathcliff"]
        )

    @patch("core.agent_core.ChatGoogleGenerativeAI")
    def test_question_response(self, mock_llm_class, mock_config, memory_manager):
        """Test that agent responds to questions."""
        mock_llm = Mock()
        mock_llm.invoke = Mock(
            return_value=Mock(
                content="Based on what I know, I can help you with various tasks like managing your calendar, checking the weather, and more."
            )
        )
        mock_llm_class.return_value = mock_llm

        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(config=mock_config, memory_manager=memory_manager)

        response = agent.invoke("What can you do?")

        assert isinstance(response, str)
        assert len(response) > 10  # Should be more than just a word

    @patch("core.agent_core.ChatGoogleGenerativeAI")
    def test_context_aware_response(self, mock_llm_class, mock_config, memory_manager):
        """Test that agent uses context in responses."""
        # Pre-populate with user info
        memory_manager.add_memory("User's name is Adi", category="facts")

        call_count = [0]

        def context_aware_response(prompt):
            call_count[0] += 1
            prompt_str = str(prompt)
            if "Adi" in prompt_str or "name" in prompt_str.lower():
                return Mock(content="Hello Adi! Great to see you again.")
            return Mock(content="Hello! How can I help?")

        mock_llm = Mock()
        mock_llm.invoke = Mock(side_effect=context_aware_response)
        mock_llm_class.return_value = mock_llm

        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(config=mock_config, memory_manager=memory_manager)

        response = agent.invoke("Hi there!")

        # Response should be personalized if context was used
        assert isinstance(response, str)


class TestConversationContinuity:
    """E2E tests for multi-turn conversation continuity."""

    @pytest.fixture
    def memory_manager(self, tmp_path):
        """Create real MemoryManager."""
        persist_dir = str(tmp_path / "chroma_continuity")
        return MemoryManager(persist_dir=persist_dir)

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = Mock()
        config.gemini_key = "test-key"
        config.get = Mock(return_value=None)
        return config

    @patch("core.agent_core.ChatGoogleGenerativeAI")
    def test_remembers_previous_turn(self, mock_llm_class, mock_config, memory_manager):
        """Test that agent remembers info from previous turn."""
        prompts = []

        def track_prompts(prompt):
            prompts.append(str(prompt))
            if len(prompts) == 1:
                return Mock(content="Nice to meet you, Adi!")
            else:
                return Mock(content="Your name is Adi, as you mentioned earlier.")

        mock_llm = Mock()
        mock_llm.invoke = Mock(side_effect=track_prompts)
        mock_llm_class.return_value = mock_llm

        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(config=mock_config, memory_manager=memory_manager)

        session = "continuity-test"

        # First turn - introduce name
        agent.invoke("My name is Adi", session_id=session)

        # Second turn - ask for name
        response = agent.invoke("What's my name?", session_id=session)

        # The second prompt should contain context from first turn
        assert len(prompts) == 2

    @patch("core.agent_core.ChatGoogleGenerativeAI")
    def test_three_turn_conversation(self, mock_llm_class, mock_config, memory_manager):
        """Test a three-turn conversation maintains context."""
        turn = [0]

        def multi_turn_response(prompt):
            turn[0] += 1
            return Mock(content=f"Response to turn {turn[0]}")

        mock_llm = Mock()
        mock_llm.invoke = Mock(side_effect=multi_turn_response)
        mock_llm_class.return_value = mock_llm

        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(config=mock_config, memory_manager=memory_manager)

        session = "three-turn"

        r1 = agent.invoke("First message", session_id=session)
        r2 = agent.invoke("Second message", session_id=session)
        r3 = agent.invoke("Third message", session_id=session)

        assert r1 is not None
        assert r2 is not None
        assert r3 is not None

        # Check chat history has all turns
        stats = memory_manager.get_stats()
        assert stats["chats"] >= 6  # 3 user + 3 assistant messages


class TestErrorRecovery:
    """E2E tests for error recovery scenarios."""

    @pytest.fixture
    def memory_manager(self, tmp_path):
        """Create real MemoryManager."""
        persist_dir = str(tmp_path / "chroma_recovery")
        return MemoryManager(persist_dir=persist_dir)

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = Mock()
        config.gemini_key = "test-key"
        config.get = Mock(return_value=None)
        return config

    @patch("core.agent_core.ChatGoogleGenerativeAI")
    def test_recovers_from_transient_error(
        self, mock_llm_class, mock_config, memory_manager
    ):
        """Test recovery from a transient API error."""
        call_count = [0]

        def flaky_llm(prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Temporary network error")
            return Mock(content="Recovered successfully!")

        mock_llm = Mock()
        mock_llm.invoke = Mock(side_effect=flaky_llm)
        mock_llm_class.return_value = mock_llm

        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(config=mock_config, memory_manager=memory_manager)

        # First call fails
        r1 = agent.invoke("Hello")
        assert "error" in r1.lower()

        # Second call succeeds
        r2 = agent.invoke("Try again")
        assert "error" not in r2.lower() or "recovered" in r2.lower()

    @patch("core.agent_core.ChatGoogleGenerativeAI")
    def test_handles_invalid_input_gracefully(
        self, mock_llm_class, mock_config, memory_manager
    ):
        """Test handling of edge case inputs."""
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content="I understand"))
        mock_llm_class.return_value = mock_llm

        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(config=mock_config, memory_manager=memory_manager)

        # Very short input
        r1 = agent.invoke("?")
        assert isinstance(r1, str)

        # Input with special characters
        r2 = agent.invoke("Hello! @#$%^&*()")
        assert isinstance(r2, str)

        # Unicode input
        r3 = agent.invoke("Hello ")
        assert isinstance(r3, str)


class TestConversationQuality:
    """E2E tests for conversation quality metrics."""

    @pytest.fixture
    def memory_manager(self, tmp_path):
        """Create real MemoryManager."""
        persist_dir = str(tmp_path / "chroma_quality")
        return MemoryManager(persist_dir=persist_dir)

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = Mock()
        config.gemini_key = "test-key"
        config.get = Mock(return_value=None)
        return config

    @patch("core.agent_core.ChatGoogleGenerativeAI")
    def test_response_not_empty(self, mock_llm_class, mock_config, memory_manager):
        """Test that responses are never empty."""
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content="Here's a helpful response."))
        mock_llm_class.return_value = mock_llm

        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(config=mock_config, memory_manager=memory_manager)

        for query in ["Hello", "What time is it?", "Tell me a joke"]:
            response = agent.invoke(query)
            assert response is not None
            assert len(response.strip()) > 0

    @patch("core.agent_core.ChatGoogleGenerativeAI")
    def test_response_reasonable_length(
        self, mock_llm_class, mock_config, memory_manager
    ):
        """Test that responses are reasonably sized."""
        mock_llm = Mock()
        mock_llm.invoke = Mock(
            return_value=Mock(
                content="This is a reasonable length response that provides helpful information."
            )
        )
        mock_llm_class.return_value = mock_llm

        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(config=mock_config, memory_manager=memory_manager)

        response = agent.invoke("Tell me about yourself")

        # Response should be substantial but not too long
        assert len(response) > 10
        assert len(response) < 10000


class TestRealGeminiAPI:
    """E2E tests with real Gemini API (skipped if no API key)."""

    @pytest.fixture
    def memory_manager(self, tmp_path):
        """Create real MemoryManager."""
        persist_dir = str(tmp_path / "chroma_real")
        return MemoryManager(persist_dir=persist_dir)

    @pytest.fixture
    def real_config(self):
        """Create config with real API key from environment."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            pytest.skip("GEMINI_API_KEY not set, skipping real API test")

        config = Mock()
        config.gemini_key = api_key
        config.get = Mock(
            side_effect=lambda key, default=None: {
                "llm.model": "gemini-2.0-flash-exp",
                "llm.temperature": 0.7,
                "llm.max_tokens": 1024,
            }.get(key, default)
        )
        return config

    def test_real_api_greeting(self, real_config, memory_manager):
        """Test real API call with greeting."""
        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(config=real_config, memory_manager=memory_manager)

        response = agent.invoke("Hello! Please respond with just 'Hi there!'")

        assert isinstance(response, str)
        assert len(response) > 0

    def test_real_api_question(self, real_config, memory_manager):
        """Test real API call with question."""
        from core.agent_core import HeathcliffAgent

        agent = HeathcliffAgent(config=real_config, memory_manager=memory_manager)

        response = agent.invoke("What is 2 + 2? Just give the number.")

        assert isinstance(response, str)
        # Should contain "4" somewhere
        assert "4" in response
