# ABOUTME: Unit tests for ChromaConnection singleton lifecycle and collection helpers.

import os
import sys
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def reset_singleton():
    from db.base import ChromaConnection

    ChromaConnection.reset()
    yield
    ChromaConnection.reset()


class TestChromaConnectionSingleton:
    def test_initialise_returns_instance(self):
        from db.base import ChromaConnection

        mock_config = Mock()
        mock_config.USE_REMOTE_CHROMA = False
        mock_config.CHROMA_PERSIST_DIRECTORY = "/tmp/test_chroma"
        with patch("db.base.chromadb.PersistentClient", return_value=Mock()):
            conn = ChromaConnection.initialise(mock_config)
        assert conn is not None

    def test_initialise_is_idempotent(self):
        from db.base import ChromaConnection

        mock_config = Mock()
        mock_config.USE_REMOTE_CHROMA = False
        mock_config.CHROMA_PERSIST_DIRECTORY = "/tmp/test_chroma"
        with patch("db.base.chromadb.PersistentClient", return_value=Mock()):
            conn1 = ChromaConnection.initialise(mock_config)
            conn2 = ChromaConnection.initialise(mock_config)
        assert conn1 is conn2

    def test_get_raises_if_not_initialised(self):
        from db.base import ChromaConnection
        from utils.errors import AgentMemoryError

        with pytest.raises(AgentMemoryError):
            ChromaConnection.get()

    def test_get_returns_instance_after_initialise(self):
        from db.base import ChromaConnection

        mock_config = Mock()
        mock_config.USE_REMOTE_CHROMA = False
        mock_config.CHROMA_PERSIST_DIRECTORY = "/tmp/test_chroma"
        with patch("db.base.chromadb.PersistentClient", return_value=Mock()):
            ChromaConnection.initialise(mock_config)
        conn = ChromaConnection.get()
        assert conn is not None

    def test_reset_clears_singleton(self):
        from db.base import ChromaConnection
        from utils.errors import AgentMemoryError

        mock_config = Mock()
        mock_config.USE_REMOTE_CHROMA = False
        mock_config.CHROMA_PERSIST_DIRECTORY = "/tmp/test_chroma"
        with patch("db.base.chromadb.PersistentClient", return_value=Mock()):
            ChromaConnection.initialise(mock_config)
        ChromaConnection.reset()
        with pytest.raises(AgentMemoryError):
            ChromaConnection.get()


class TestChromaConnectionCollections:
    @pytest.fixture
    def conn(self):
        from db.base import ChromaConnection

        mock_client = Mock()
        mock_client.get_or_create_collection = Mock(return_value=Mock())
        conn = ChromaConnection(mock_client)
        ChromaConnection._instance = conn
        return conn

    def test_get_or_create_collection_delegates(self, conn):
        conn.get_or_create_collection("test_col", {"desc": "test"})
        conn._client.get_or_create_collection.assert_called_once_with(
            name="test_col", metadata={"desc": "test"}
        )

    def test_get_or_create_collection_defaults_empty_metadata(self, conn):
        conn.get_or_create_collection("test_col")
        conn._client.get_or_create_collection.assert_called_once_with(
            name="test_col", metadata={}
        )

    def test_delete_collection_delegates(self, conn):
        conn.delete_collection("test_col")
        conn._client.delete_collection.assert_called_once_with(name="test_col")

    def test_delete_collection_swallows_errors(self, conn):
        conn._client.delete_collection.side_effect = Exception("not found")
        conn.delete_collection("missing_col")  # should not raise

    def test_get_client_returns_client(self, conn):
        assert conn.get_client() is conn._client


class TestCollectionConstants:
    def test_constants_have_expected_values(self):
        from db.base import (
            CONVERSATIONS_COLLECTION,
            MEMORIES_COLLECTION,
        )

        assert CONVERSATIONS_COLLECTION == "heathcliff_conversations"
        assert MEMORIES_COLLECTION == "heathcliff_memories"
