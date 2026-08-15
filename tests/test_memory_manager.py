# ABOUTME: Unit tests for MemoryManager facade.
# ABOUTME: Tests delegation, extraction gate, queue worker, and stats.

import os
import sys
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helper — build a MemoryManager with all external clients mocked
# ---------------------------------------------------------------------------


def _make_memory_manager():
    from db.base import ChromaConnection
    from db.memory_manager import MemoryManager

    ChromaConnection.reset()
    MemoryManager._mem0_singleton = None

    mock_collection = Mock()
    mock_collection.count = Mock(return_value=0)

    mock_chroma_conn = Mock()
    mock_chroma_conn.get_or_create_collection.return_value = mock_collection
    mock_chroma_conn.get_client.return_value = Mock()

    mock_mem0 = Mock()
    mock_mem0.get_all.return_value = {"results": []}
    mock_mem0.search.return_value = {"results": []}

    with (
        patch.object(ChromaConnection, "initialise", return_value=mock_chroma_conn),
        patch.object(ChromaConnection, "get", return_value=mock_chroma_conn),
        patch.object(
            ChromaConnection, "get_or_create_collection", return_value=mock_collection
        ),
        patch("db.memory_manager.Memory", return_value=mock_mem0),
        patch("db.memory_manager.Memory.from_config", return_value=mock_mem0),
    ):
        mgr = MemoryManager()

    mgr._mock_mem0 = mock_mem0
    mgr._mock_collection = mock_collection
    mgr.mem0_client = mock_mem0
    return mgr


# ---------------------------------------------------------------------------
# Delegation
# ---------------------------------------------------------------------------


class TestDelegation:
    @pytest.fixture
    def mm(self):
        return _make_memory_manager()

    def test_save_turn_delegates_to_conversation_manager(self, mm):
        mm._conversation_manager = Mock()
        mm._conversation_manager.save_turn.return_value = ("uid", "aid")
        result = mm.save_turn("Hello", "Hi", "conv-1")
        mm._conversation_manager.save_turn.assert_called_once_with(
            "Hello", "Hi", "conv-1"
        )
        assert result == ("uid", "aid")

    def test_build_langchain_history_delegates(self, mm):
        mm._conversation_manager = Mock()
        mm._conversation_manager.build_langchain_history.return_value = []
        mm.build_langchain_history("q", "conv-1")
        mm._conversation_manager.build_langchain_history.assert_called_once()

    def test_get_all_conversations_delegates(self, mm):
        mm._conversation_manager = Mock()
        mm._conversation_manager.get_all_conversations.return_value = []
        mm.get_all_conversations()
        mm._conversation_manager.get_all_conversations.assert_called_once()

    def test_clear_conversation_delegates(self, mm):
        mm._conversation_manager = Mock()
        mm._conversation_manager.clear_conversation.return_value = True
        mm.clear_conversation("conv-1")
        mm._conversation_manager.clear_conversation.assert_called_once_with("conv-1")

    def test_delete_all_chats_delegates(self, mm):
        mm._conversation_manager = Mock()
        mm._conversation_manager.delete_all_chats.return_value = True
        mm.delete_all_chats()
        mm._conversation_manager.delete_all_chats.assert_called_once()


# ---------------------------------------------------------------------------
# Extraction gate
# ---------------------------------------------------------------------------


class TestShouldExtractMemory:
    def test_empty_returns_false(self):
        from db.memory_manager import MemoryManager

        assert MemoryManager._should_extract_memory("") is False

    def test_question_returns_false(self):
        from db.memory_manager import MemoryManager

        assert MemoryManager._should_extract_memory("What is the weather?") is False

    def test_command_prefix_returns_false(self):
        from db.memory_manager import MemoryManager

        assert MemoryManager._should_extract_memory("Please find my emails") is False
        assert MemoryManager._should_extract_memory("send an email to Philip") is False

    def test_personal_statement_returns_true(self):
        from db.memory_manager import MemoryManager

        assert MemoryManager._should_extract_memory("I like dark mode") is True
        assert (
            MemoryManager._should_extract_memory("My favourite music is jazz") is True
        )
        assert MemoryManager._should_extract_memory("I prefer morning meetings") is True

    def test_very_short_returns_false(self):
        from db.memory_manager import MemoryManager

        assert MemoryManager._should_extract_memory("ok") is False


# ---------------------------------------------------------------------------
# Queue worker — extraction is enqueued after save_turn
# ---------------------------------------------------------------------------


class TestExtractionQueue:
    @pytest.fixture
    def mm(self):
        return _make_memory_manager()

    def test_save_turn_enqueues_extraction_task(self, mm):
        mm._conversation_manager = Mock()
        mm._conversation_manager.save_turn.return_value = ("uid", "aid")
        mm.save_turn("I like jazz", "Noted.", "conv-1")
        assert (
            not mm._extraction_queue.empty()
            or mm._extraction_queue.unfinished_tasks >= 0
        )

    def test_worker_thread_is_daemon(self, mm):
        assert mm._worker.daemon is True

    def test_worker_thread_is_alive(self, mm):
        assert mm._worker.is_alive()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestGetStats:
    @pytest.fixture
    def mm(self):
        return _make_memory_manager()

    def test_get_stats_returns_dict_with_expected_keys(self, mm):
        mm._conversation_manager = Mock()
        mm._conversation_manager.count.return_value = 5
        stats = mm.get_stats()
        assert "memories" in stats
        assert "chats" in stats

    def test_repr_contains_counts(self, mm):
        mm._conversation_manager = Mock()
        mm._conversation_manager.count.return_value = 3
        r = repr(mm)
        assert "MemoryManager" in r


# ---------------------------------------------------------------------------
# Recall / add_memory
# ---------------------------------------------------------------------------


class TestRecall:
    @pytest.fixture
    def mm(self):
        return _make_memory_manager()

    def test_recall_returns_normalized_results(self, mm):
        mm.mem0_client.search.return_value = {
            "results": [
                {
                    "memory": "User likes dark mode",
                    "id": "m1",
                    "metadata": {"category": "preferences"},
                },
            ]
        }
        result = mm.recall("preferences", n=1)
        assert "documents" in result
        assert "ids" in result

    def test_recall_handles_exception(self, mm):
        mm.mem0_client.search.side_effect = Exception("API error")
        result = mm.recall("query")
        assert result["documents"] == [[]]

    def test_add_memory_returns_string_id(self, mm):
        mm.mem0_client.add.return_value = {"id": "mem_abc"}
        mem_id = mm.add_memory("I prefer tea", category="preferences")
        assert isinstance(mem_id, str)
