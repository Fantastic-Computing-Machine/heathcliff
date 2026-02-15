# ABOUTME: Unit tests for new chat history/session APIs in MemoryManager

import os
import sys
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory_manager import MemoryManager


class TestMemoryManagerSessions:
    """Tests for get_all_sessions and related session APIs."""

    @pytest.fixture
    def memory_manager(self):
        """Create a MemoryManager with mocked dependencies."""
        mock_client = Mock()

        with (
            patch("core.memory_manager.chroma_client"),
            patch("core.memory_manager._global_chroma_client", mock_client),
            patch("core.memory_manager.MemoryManager._get_mem0_client"),
            patch("core.memory_manager.Config") as MockConfig,
        ):
            # Mock Config to avoid real connection attempts
            MockConfig.USE_REMOTE_CHROMA = False

            mm = MemoryManager()

            # Mock the chats collection
            mm.chats = Mock()
            # Default get return value (empty)
            mm.chats.get.return_value = {"metadatas": []}

            return mm

    def test_get_all_sessions_empty(self, memory_manager):
        """Test getting sessions when no data exists."""
        sessions = memory_manager.get_all_sessions()
        assert sessions == []

    def test_get_all_sessions_aggregation(self, memory_manager):
        """Test aggregation of messages into sessions."""
        # Mock the get response with flattened metadata from multiple sessions
        # simulating what Chroma would return
        memory_manager.chats.get.return_value = {
            "metadatas": [
                # Session 1 messages
                {
                    "session": "session-1",
                    "timestamp": "2023-01-01T10:00:00",
                    "role": "user",
                },
                {
                    "session": "session-1",
                    "timestamp": "2023-01-01T10:00:01",
                    "role": "assistant",
                },
                {
                    "session": "session-1",
                    "timestamp": "2023-01-01T10:00:02",
                    "role": "user",
                },
                {
                    "session": "session-1",
                    "timestamp": "2023-01-01T10:00:03",
                    "role": "assistant",
                },
                # Session 2 messages
                {
                    "session": "session-2",
                    "timestamp": "2023-01-02T10:00:00",
                    "role": "user",
                },
                {
                    "session": "session-2",
                    "timestamp": "2023-01-02T10:00:01",
                    "role": "assistant",
                },
            ]
        }

        sessions = memory_manager.get_all_sessions()

        assert len(sessions) == 2

        # Check session 1 stats
        s1 = next(s for s in sessions if s["session_id"] == "session-1")
        assert s1["msg_count"] == 4
        assert s1["start_time"] == "2023-01-01T10:00:00"
        assert s1["end_time"] == "2023-01-01T10:00:03"

        # Check session 2 stats
        s2 = next(s for s in sessions if s["session_id"] == "session-2")
        assert s2["msg_count"] == 2
        assert s2["start_time"] == "2023-01-02T10:00:00"
        assert s2["end_time"] == "2023-01-02T10:00:01"

    def test_get_all_sessions_sorting(self, memory_manager):
        """Test that sessions are sorted by end_time (recent first)."""
        memory_manager.chats.get.return_value = {
            "metadatas": [
                {"session": "old-session", "timestamp": "2023-01-01T10:00:00"},
                {"session": "new-session", "timestamp": "2023-02-01T10:00:00"},
            ]
        }

        sessions = memory_manager.get_all_sessions()

        assert len(sessions) == 2
        assert sessions[0]["session_id"] == "new-session"
        assert sessions[1]["session_id"] == "old-session"

    def test_get_all_sessions_pagination(self, memory_manager):
        """Test that get_all_sessions handles pagination loops."""

        # Mock responses for 2 batches then empty
        # Note: We need to mock the side_effect on the instance's chats.get
        memory_manager.chats.get.side_effect = [
            # Batch 1
            {
                "metadatas": [
                    {"session": "sess-1", "timestamp": "t1"},
                    {"session": "sess-1", "timestamp": "t2"},
                ]
            },
            # Batch 2
            {"metadatas": [{"session": "sess-2", "timestamp": "t3"}]},
            # Batch 3 (empty/end)
            {"metadatas": []},
        ]

        sessions = memory_manager.get_all_sessions(limit=2)

        # Should have called get 2 times (stopped after partial batch)
        assert memory_manager.chats.get.call_count == 2
        assert len(sessions) == 2
