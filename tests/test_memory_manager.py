# ABOUTME: Unit tests for MemoryManager ChromaDB-backed memory storage
# ABOUTME: Tests all CRUD operations, persistence, and edge cases

import pytest
import tempfile
import shutil
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory_manager import MemoryManager


class TestMemoryManagerInit:
    """Tests for MemoryManager initialization."""

    def test_init_creates_collections(self, tmp_path):
        """Test that initialization creates all three collections."""
        persist_dir = str(tmp_path / "chroma_test")
        mm = MemoryManager(persist_dir=persist_dir)

        # Verify collections exist
        assert mm.memories is not None
        assert mm.chats is not None
        assert mm.my_data is not None

    def test_init_with_default_path(self):
        """Test initialization with default persistence directory."""
        mm = MemoryManager()
        assert mm.client is not None

    def test_collections_initially_empty(self, tmp_path):
        """Test that new collections start empty."""
        persist_dir = str(tmp_path / "chroma_test")
        mm = MemoryManager(persist_dir=persist_dir)

        stats = mm.get_stats()
        assert stats["memories"] == 0
        assert stats["chats"] == 0
        assert stats["documents"] == 0


class TestAddMemory:
    """Tests for add_memory method."""

    @pytest.fixture
    def memory_manager(self, tmp_path):
        """Create a fresh MemoryManager for each test."""
        persist_dir = str(tmp_path / "chroma_test")
        return MemoryManager(persist_dir=persist_dir)

    def test_add_memory_returns_id(self, memory_manager):
        """Test that add_memory returns a valid ID."""
        memory_id = memory_manager.add_memory("User's name is Adi")
        assert memory_id is not None
        assert memory_id.startswith("mem_")

    def test_add_memory_with_category(self, memory_manager):
        """Test adding memory with category."""
        memory_id = memory_manager.add_memory(
            "Prefers dark mode", category="preferences"
        )
        assert memory_id is not None

        # Verify memory was stored
        results = memory_manager.recall("dark mode", n=1)
        assert len(results["documents"][0]) > 0

    def test_add_memory_increments_count(self, memory_manager):
        """Test that adding memories increases collection count."""
        initial_count = memory_manager.get_stats()["memories"]

        memory_manager.add_memory("Fact 1")
        memory_manager.add_memory("Fact 2")

        final_count = memory_manager.get_stats()["memories"]
        assert final_count == initial_count + 2

    def test_add_memory_with_metadata(self, memory_manager):
        """Test adding memory with custom metadata."""
        memory_id = memory_manager.add_memory(
            "Works at Google", category="facts", metadata={"importance": "high"}
        )
        assert memory_id is not None


class TestRecall:
    """Tests for recall method."""

    @pytest.fixture
    def memory_manager_with_data(self, tmp_path):
        """Create MemoryManager with pre-populated data."""
        persist_dir = str(tmp_path / "chroma_test")
        mm = MemoryManager(persist_dir=persist_dir)

        # Add some memories
        mm.add_memory("User's favorite color is blue", category="preferences")
        mm.add_memory("User lives in San Francisco", category="facts")
        mm.add_memory("User prefers Python over JavaScript", category="preferences")

        return mm

    def test_recall_returns_relevant_results(self, memory_manager_with_data):
        """Test that recall returns semantically relevant results."""
        results = memory_manager_with_data.recall(
            "What is the user's favorite color?", n=1
        )

        assert "documents" in results
        assert len(results["documents"][0]) > 0
        # The most relevant result should mention color
        assert (
            "color" in results["documents"][0][0].lower()
            or "blue" in results["documents"][0][0].lower()
        )

    def test_recall_with_category_filter(self, memory_manager_with_data):
        """Test recall with category filter."""
        results = memory_manager_with_data.recall(
            "preferences", n=5, category="preferences"
        )

        # All results should be preferences
        for metadata in results.get("metadatas", [[]])[0]:
            assert metadata.get("category") == "preferences"

    def test_recall_returns_correct_structure(self, memory_manager_with_data):
        """Test that recall returns expected dictionary structure."""
        results = memory_manager_with_data.recall("color", n=2)

        assert "documents" in results
        assert "metadatas" in results
        assert "ids" in results

    def test_recall_with_empty_query(self, memory_manager_with_data):
        """Test recall with empty query still returns results."""
        results = memory_manager_with_data.recall("", n=1)
        # ChromaDB should handle empty queries gracefully
        assert "documents" in results

    def test_recall_n_parameter(self, memory_manager_with_data):
        """Test that n parameter limits results."""
        results = memory_manager_with_data.recall("user", n=2)
        assert len(results["documents"][0]) <= 2


class TestSaveChat:
    """Tests for save_chat method."""

    @pytest.fixture
    def memory_manager(self, tmp_path):
        """Create a fresh MemoryManager for each test."""
        persist_dir = str(tmp_path / "chroma_test")
        return MemoryManager(persist_dir=persist_dir)

    def test_save_chat_returns_ids(self, memory_manager):
        """Test that save_chat returns tuple of IDs."""
        result = memory_manager.save_chat(
            user_msg="Hello", assistant_msg="Hi there!", session_id="test-session-1"
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] is not None  # user_id
        assert result[1] is not None  # asst_id

    def test_save_chat_stores_both_messages(self, memory_manager):
        """Test that both user and assistant messages are stored."""
        memory_manager.save_chat(
            user_msg="What's the weather?",
            assistant_msg="It's sunny today.",
            session_id="test-session",
        )

        stats = memory_manager.get_stats()
        assert stats["chats"] == 2  # Two messages stored

    def test_save_chat_with_session_isolation(self, memory_manager):
        """Test that different sessions don't mix."""
        memory_manager.save_chat("Hello session 1", "Hi session 1", "session-1")
        memory_manager.save_chat("Hello session 2", "Hi session 2", "session-2")

        # Get context for session 1
        results = memory_manager.get_chat_context("Hello", n=5, session_id="session-1")

        # Should only return session-1 messages
        for metadata in results.get("metadatas", [[]])[0]:
            assert metadata.get("session") == "session-1"


class TestGetChatContext:
    """Tests for get_chat_context method."""

    @pytest.fixture
    def memory_manager_with_chats(self, tmp_path):
        """Create MemoryManager with chat history."""
        persist_dir = str(tmp_path / "chroma_test")
        mm = MemoryManager(persist_dir=persist_dir)

        # Add chat history
        mm.save_chat("What's the weather?", "It's sunny today.", "session-1")
        mm.save_chat("Play some music", "Playing your playlist.", "session-1")
        mm.save_chat("Different session", "Different response", "session-2")

        return mm

    def test_get_chat_context_returns_relevant(self, memory_manager_with_chats):
        """Test that context retrieval finds relevant messages."""
        results = memory_manager_with_chats.get_chat_context("weather", n=2)

        assert "documents" in results
        assert len(results["documents"][0]) > 0

    def test_get_chat_context_with_session_filter(self, memory_manager_with_chats):
        """Test context retrieval with session filter."""
        results = memory_manager_with_chats.get_chat_context(
            "hello", n=5, session_id="session-1"
        )

        # All results should be from session-1
        for metadata in results.get("metadatas", [[]])[0]:
            assert metadata.get("session") == "session-1"

    def test_get_chat_context_structure(self, memory_manager_with_chats):
        """Test that returned structure matches expected format."""
        results = memory_manager_with_chats.get_chat_context("music", n=2)

        assert "documents" in results
        assert "metadatas" in results
        assert "ids" in results


class TestIndexDocument:
    """Tests for index_document method."""

    @pytest.fixture
    def memory_manager(self, tmp_path):
        """Create a fresh MemoryManager for each test."""
        persist_dir = str(tmp_path / "chroma_test")
        return MemoryManager(persist_dir=persist_dir)

    def test_index_document_returns_id(self, memory_manager):
        """Test that index_document returns a document ID."""
        doc_id = memory_manager.index_document(
            content="Meeting notes from standup",
            source="meeting_notes.txt",
            doc_type="file",
        )

        assert doc_id is not None
        assert "file" in doc_id

    def test_index_document_searchable(self, memory_manager):
        """Test that indexed documents are searchable."""
        memory_manager.index_document(
            content="Email from John about project deadline",
            source="john@example.com",
            doc_type="email",
        )

        results = memory_manager.search_my_data("project deadline", n=1)
        assert len(results["documents"][0]) > 0
        assert "deadline" in results["documents"][0][0].lower()

    def test_index_document_with_type_filter(self, memory_manager):
        """Test searching documents with type filter."""
        memory_manager.index_document("Email content", "a@b.com", "email")
        memory_manager.index_document("File content", "doc.txt", "file")

        results = memory_manager.search_my_data("content", doc_type="email", n=5)

        # All results should be emails
        for metadata in results.get("metadatas", [[]])[0]:
            assert metadata.get("type") == "email"


class TestPersistence:
    """Tests for data persistence across sessions."""

    def test_data_persists_after_close(self, tmp_path):
        """Test that data persists after closing and reopening."""
        persist_dir = str(tmp_path / "chroma_persist")

        # Create manager and add data
        mm1 = MemoryManager(persist_dir=persist_dir)
        mm1.add_memory("Persistent memory test")
        mm1.save_chat("Persistent chat", "Persistent response", "persist-session")

        initial_stats = mm1.get_stats()

        # Delete first instance (simulates closing)
        del mm1

        # Reopen with new instance
        mm2 = MemoryManager(persist_dir=persist_dir)

        # Verify data still exists
        final_stats = mm2.get_stats()
        assert final_stats["memories"] >= 1
        assert final_stats["chats"] >= 2

    def test_search_works_after_reopen(self, tmp_path):
        """Test that search works correctly after reopening."""
        persist_dir = str(tmp_path / "chroma_persist")

        # Create and populate
        mm1 = MemoryManager(persist_dir=persist_dir)
        mm1.add_memory("User's birthday is March 15th", category="facts")
        del mm1

        # Reopen and search
        mm2 = MemoryManager(persist_dir=persist_dir)
        results = mm2.recall("birthday", n=1)

        assert len(results["documents"][0]) > 0
        assert "birthday" in results["documents"][0][0].lower()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def memory_manager(self, tmp_path):
        """Create a fresh MemoryManager for each test."""
        persist_dir = str(tmp_path / "chroma_test")
        return MemoryManager(persist_dir=persist_dir)

    def test_empty_collection_search(self, memory_manager):
        """Test searching an empty collection returns empty results."""
        results = memory_manager.recall("anything", n=5)
        # Should return empty but valid structure
        assert "documents" in results
        assert len(results["documents"][0]) == 0

    def test_large_text_storage(self, memory_manager):
        """Test storing large text content."""
        large_text = "A" * 10000  # 10K characters
        memory_id = memory_manager.add_memory(large_text)
        assert memory_id is not None

    def test_special_characters_in_memory(self, memory_manager):
        """Test storing text with special characters."""
        special_text = "User said: \"Hello! @#$%^&*() \n\t 'quotes' \""
        memory_id = memory_manager.add_memory(special_text)
        assert memory_id is not None

        results = memory_manager.recall("Hello", n=1)
        assert len(results["documents"][0]) > 0

    def test_unicode_support(self, memory_manager):
        """Test storing unicode text."""
        unicode_text = "User speaks Japanese: "
        memory_id = memory_manager.add_memory(unicode_text)
        assert memory_id is not None

    def test_delete_memory(self, memory_manager):
        """Test deleting a memory."""
        memory_id = memory_manager.add_memory("To be deleted")
        initial_count = memory_manager.get_stats()["memories"]

        result = memory_manager.delete_memory(memory_id)

        assert result is True
        final_count = memory_manager.get_stats()["memories"]
        assert final_count == initial_count - 1

    def test_clear_session(self, memory_manager):
        """Test clearing all messages from a session."""
        memory_manager.save_chat("Msg 1", "Response 1", "clear-me")
        memory_manager.save_chat("Msg 2", "Response 2", "clear-me")
        memory_manager.save_chat("Msg 3", "Response 3", "keep-me")

        result = memory_manager.clear_session("clear-me")
        assert result is True

        # Verify clear-me session is gone
        results = memory_manager.get_chat_context("Msg", n=10, session_id="clear-me")
        assert len(results["documents"][0]) == 0

        # Verify keep-me session still exists
        results = memory_manager.get_chat_context("Msg", n=10, session_id="keep-me")
        assert len(results["documents"][0]) > 0

    def test_repr(self, memory_manager):
        """Test string representation."""
        repr_str = repr(memory_manager)
        assert "MemoryManager" in repr_str
        assert "memories=" in repr_str
