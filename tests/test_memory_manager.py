# ABOUTME: Unit tests for MemoryManager ChromaDB-backed memory storage
# ABOUTME: Tests pair-based history methods, save_chat turn_id, and edge cases

import os
import sys
import uuid
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers — build a MemoryManager with mocked ChromaDB / Mem0
# ---------------------------------------------------------------------------


def _make_memory_manager():
    """Build a MemoryManager with mocked external clients.

    Patches the global ChromaDB client init, injects in-memory
    mock collections, and stubs out Mem0.
    """
    import core.memory_manager as mm_mod

    # Pre-set the global chroma client so __init__ skips network calls
    mock_chroma_client = Mock()
    mock_chats_collection = Mock()
    mock_my_data_collection = Mock()

    mock_chroma_client.get_or_create_collection = Mock(
        side_effect=lambda name, **kw: {
            "chat_messages": mock_chats_collection,
            "my_data": mock_my_data_collection,
        }.get(name, Mock())
    )

    original_global = mm_mod._global_chroma_client
    mm_mod._global_chroma_client = mock_chroma_client

    with patch.object(mm_mod, "chroma_client"):
        with patch.object(
            mm_mod.MemoryManager, "_get_mem0_client", return_value=Mock()
        ):
            from core.memory_manager import MemoryManager

            mgr = MemoryManager()

    # Restore (each test will get its own manager)
    mm_mod._global_chroma_client = original_global

    # Expose internals for test assertions
    mgr._mock_chats = mock_chats_collection
    return mgr


# ---------------------------------------------------------------------------
# save_chat — turn_id pairing
# ---------------------------------------------------------------------------


class TestSaveChat:
    """Tests for save_chat including turn_id pairing."""

    @pytest.fixture
    def mm(self):
        return _make_memory_manager()

    def test_save_chat_returns_tuple_of_ids(self, mm):
        result = mm.save_chat("Hello", "Hi there!", "session-1")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_save_chat_stores_both_messages(self, mm):
        mm.save_chat("Hello", "Hi there!", "session-1")
        call_args = mm.chats.add.call_args
        docs = call_args[1].get("documents", call_args[0][0] if call_args[0] else None)
        assert len(docs) == 2

    def test_save_chat_uses_shared_turn_id(self, mm):
        mm.save_chat("What's the weather?", "It's sunny.", "session-1")
        call_args = mm.chats.add.call_args
        metadatas = call_args[1].get("metadatas")
        assert metadatas is not None
        assert len(metadatas) == 2
        # Both messages should share the same turn_id
        assert metadatas[0]["turn_id"] == metadatas[1]["turn_id"]
        assert len(metadatas[0]["turn_id"]) > 0

    def test_save_chat_user_role_first(self, mm):
        mm.save_chat("Question", "Answer", "s1")
        metadatas = mm.chats.add.call_args[1]["metadatas"]
        assert metadatas[0]["role"] == "user"
        assert metadatas[1]["role"] == "assistant"

    def test_save_chat_order_is_monotonic(self, mm):
        mm.save_chat("Q", "A", "s1")
        metadatas = mm.chats.add.call_args[1]["metadatas"]
        assert metadatas[0]["order"] < metadatas[1]["order"]


# ---------------------------------------------------------------------------
# _sort_key
# ---------------------------------------------------------------------------


class TestSortKey:
    """Tests for the _sort_key static method."""

    def test_order_takes_precedence(self):
        from core.memory_manager import MemoryManager

        msg = {"order": 100.0, "timestamp": "2025-01-01T00:00:00", "role": "user"}
        key = MemoryManager._sort_key(msg)
        assert key[0] == 0  # order-based path

    def test_fallback_to_timestamp(self):
        from core.memory_manager import MemoryManager

        msg = {"timestamp": "2025-01-01T00:00:00", "role": "user"}
        key = MemoryManager._sort_key(msg)
        assert key[0] == 1  # fallback path

    def test_user_before_assistant_in_fallback(self):
        from core.memory_manager import MemoryManager

        user = {"timestamp": "2025-01-01T00:00:00", "role": "user"}
        asst = {"timestamp": "2025-01-01T00:00:00", "role": "assistant"}
        assert MemoryManager._sort_key(user) < MemoryManager._sort_key(asst)


# ---------------------------------------------------------------------------
# _fetch_partner_by_turn_id
# ---------------------------------------------------------------------------


class TestFetchPartnerByTurnId:
    """Tests for _fetch_partner_by_turn_id."""

    @pytest.fixture
    def mm(self):
        return _make_memory_manager()

    def test_finds_partner(self, mm):
        turn_id = "tid-123"
        mm.chats.get = Mock(
            return_value={
                "documents": ["Q", "A"],
                "metadatas": [
                    {
                        "role": "user",
                        "turn_id": turn_id,
                        "timestamp": "t1",
                        "order": 1.0,
                    },
                    {
                        "role": "assistant",
                        "turn_id": turn_id,
                        "timestamp": "t2",
                        "order": 2.0,
                    },
                ],
                "ids": ["id_user", "id_asst"],
            }
        )
        partner = mm._fetch_partner_by_turn_id(turn_id, "id_user")
        assert partner is not None
        assert partner["role"] == "assistant"
        assert partner["id"] == "id_asst"

    def test_returns_none_when_no_partner(self, mm):
        mm.chats.get = Mock(
            return_value={
                "documents": ["Q"],
                "metadatas": [{"role": "user", "turn_id": "tid", "timestamp": "t1"}],
                "ids": ["id_user"],
            }
        )
        partner = mm._fetch_partner_by_turn_id("tid", "id_user")
        assert partner is None

    def test_returns_none_on_exception(self, mm):
        mm.chats.get = Mock(side_effect=Exception("DB error"))
        partner = mm._fetch_partner_by_turn_id("tid", "id_user")
        assert partner is None


# ---------------------------------------------------------------------------
# _resolve_pair
# ---------------------------------------------------------------------------


class TestResolvePair:
    """Tests for _resolve_pair."""

    @pytest.fixture
    def mm(self):
        return _make_memory_manager()

    def test_resolve_pair_from_user_msg(self, mm):
        turn_id = "tid-1"
        mm.chats.get = Mock(
            return_value={
                "documents": ["Q", "A"],
                "metadatas": [
                    {
                        "role": "user",
                        "turn_id": turn_id,
                        "timestamp": "t1",
                        "order": 1.0,
                    },
                    {
                        "role": "assistant",
                        "turn_id": turn_id,
                        "timestamp": "t2",
                        "order": 2.0,
                    },
                ],
                "ids": ["id_u", "id_a"],
            }
        )

        user_msg = {
            "role": "user",
            "content": "Q",
            "turn_id": turn_id,
            "id": "id_u",
            "timestamp": "t1",
            "order": 1.0,
        }
        pair = mm._resolve_pair(user_msg)
        assert pair is not None
        assert pair[0]["role"] == "user"
        assert pair[1]["role"] == "assistant"

    def test_resolve_pair_from_assistant_msg(self, mm):
        turn_id = "tid-2"
        mm.chats.get = Mock(
            return_value={
                "documents": ["Q", "A"],
                "metadatas": [
                    {
                        "role": "user",
                        "turn_id": turn_id,
                        "timestamp": "t1",
                        "order": 1.0,
                    },
                    {
                        "role": "assistant",
                        "turn_id": turn_id,
                        "timestamp": "t2",
                        "order": 2.0,
                    },
                ],
                "ids": ["id_u", "id_a"],
            }
        )

        asst_msg = {
            "role": "assistant",
            "content": "A",
            "turn_id": turn_id,
            "id": "id_a",
            "timestamp": "t2",
            "order": 2.0,
        }
        pair = mm._resolve_pair(asst_msg)
        assert pair is not None
        assert pair[0]["role"] == "user"
        assert pair[1]["role"] == "assistant"

    def test_resolve_pair_returns_none_when_no_partner(self, mm):
        mm.chats.get = Mock(return_value={"documents": [], "metadatas": [], "ids": []})
        msg = {"role": "user", "content": "Q", "turn_id": "", "id": "x"}
        pair = mm._resolve_pair(msg)
        assert pair is None


# ---------------------------------------------------------------------------
# get_recent_pairs
# ---------------------------------------------------------------------------


class TestGetRecentPairs:
    """Tests for get_recent_pairs."""

    @pytest.fixture
    def mm(self):
        return _make_memory_manager()

    def _setup_session_messages(self, mm, n_turns):
        """Set up mock chats.get to return n_turns worth of messages.

        The mock dispatches on the ``where`` kwarg so that
        ``_fetch_partner_by_turn_id`` receives only the two messages that
        share a ``turn_id``, while the initial session-level fetch returns
        everything.
        """
        all_docs = []
        all_metas = []
        all_ids = []
        # Index messages by turn_id for fast lookup
        by_turn: dict[str, dict] = {}
        now = datetime.now()
        for i in range(n_turns):
            turn_id = f"tid-{i}"
            t_user = now + timedelta(seconds=i * 2)
            t_asst = t_user + timedelta(microseconds=1)
            u_meta = {
                "role": "user",
                "session": "s1",
                "timestamp": t_user.isoformat(),
                "order": t_user.timestamp(),
                "turn_id": turn_id,
            }
            a_meta = {
                "role": "assistant",
                "session": "s1",
                "timestamp": t_asst.isoformat(),
                "order": t_asst.timestamp(),
                "turn_id": turn_id,
            }
            u_id = f"s1_{i}_user"
            a_id = f"s1_{i}_asst"

            all_docs.extend([f"Q{i}", f"A{i}"])
            all_metas.extend([u_meta, a_meta])
            all_ids.extend([u_id, a_id])

            by_turn[turn_id] = {
                "documents": [f"Q{i}", f"A{i}"],
                "metadatas": [u_meta, a_meta],
                "ids": [u_id, a_id],
            }

        full_result = {
            "documents": all_docs,
            "metadatas": all_metas,
            "ids": all_ids,
        }

        def _get_side_effect(**kwargs):
            where = kwargs.get("where", {})
            tid = where.get("turn_id")
            if tid and tid in by_turn:
                return by_turn[tid]
            return full_result

        mm.chats.get = Mock(side_effect=_get_side_effect)

    def test_returns_correct_number_of_pairs(self, mm):
        self._setup_session_messages(mm, 8)
        pairs = mm.get_recent_pairs("s1", n_pairs=3)
        assert len(pairs) == 3

    def test_pairs_are_chronological(self, mm):
        self._setup_session_messages(mm, 4)
        pairs = mm.get_recent_pairs("s1", n_pairs=4)
        for i in range(len(pairs) - 1):
            assert mm._sort_key(pairs[i][0]) <= mm._sort_key(pairs[i + 1][0])

    def test_each_pair_has_user_and_assistant(self, mm):
        self._setup_session_messages(mm, 3)
        pairs = mm.get_recent_pairs("s1", n_pairs=3)
        for user_msg, asst_msg in pairs:
            assert user_msg["role"] == "user"
            assert asst_msg["role"] == "assistant"

    def test_empty_session_returns_empty(self, mm):
        mm.chats.get = Mock(return_value={"documents": [], "metadatas": [], "ids": []})
        pairs = mm.get_recent_pairs("empty-session", n_pairs=3)
        assert pairs == []


# ---------------------------------------------------------------------------
# get_semantic_pairs
# ---------------------------------------------------------------------------


class TestGetSemanticPairs:
    """Tests for get_semantic_pairs."""

    @pytest.fixture
    def mm(self):
        return _make_memory_manager()

    def test_returns_pairs(self, mm):
        turn_id = "tid-sem-1"
        # query returns a candidate hit
        mm.chats.query = Mock(
            return_value={
                "documents": [["What's the weather?"]],
                "metadatas": [
                    [
                        {
                            "role": "user",
                            "session": "old-s",
                            "timestamp": "t1",
                            "order": 1.0,
                            "turn_id": turn_id,
                        }
                    ]
                ],
                "ids": [["id_u"]],
                "distances": [[0.05]],
            }
        )
        # partner lookup
        mm.chats.get = Mock(
            return_value={
                "documents": ["What's the weather?", "It's sunny."],
                "metadatas": [
                    {
                        "role": "user",
                        "turn_id": turn_id,
                        "timestamp": "t1",
                        "order": 1.0,
                    },
                    {
                        "role": "assistant",
                        "turn_id": turn_id,
                        "timestamp": "t2",
                        "order": 2.0,
                    },
                ],
                "ids": ["id_u", "id_a"],
            }
        )

        pairs = mm.get_semantic_pairs("weather", n_pairs=3)
        assert len(pairs) >= 1
        assert pairs[0][0]["role"] == "user"
        assert pairs[0][1]["role"] == "assistant"

    def test_empty_query_returns_empty(self, mm):
        pairs = mm.get_semantic_pairs("", n_pairs=3)
        assert pairs == []

    def test_whitespace_query_returns_empty(self, mm):
        pairs = mm.get_semantic_pairs("   ", n_pairs=3)
        assert pairs == []

    def test_handles_query_exception(self, mm):
        mm.chats.query = Mock(side_effect=Exception("search failed"))
        pairs = mm.get_semantic_pairs("test", n_pairs=3)
        assert pairs == []


# ---------------------------------------------------------------------------
# build_message_history
# ---------------------------------------------------------------------------


class TestBuildMessageHistory:
    """Tests for build_message_history."""

    @pytest.fixture
    def mm(self):
        return _make_memory_manager()

    def test_returns_flat_list_of_dicts(self, mm):
        # No recent or semantic results
        mm.chats.get = Mock(return_value={"documents": [], "metadatas": [], "ids": []})
        mm.chats.query = Mock(
            return_value={
                "documents": [[]],
                "metadatas": [[]],
                "ids": [[]],
                "distances": [[]],
            }
        )

        history = mm.build_message_history("test", "session-1")
        assert isinstance(history, list)
        for msg in history:
            assert "role" in msg
            assert "content" in msg

    def test_semantic_pairs_come_before_recent(self, mm):
        """Semantic pairs should appear before recent chronological pairs."""
        now = datetime.now()

        # Set up get_recent_pairs to return one recent pair
        mm.chats.get = Mock(
            return_value={
                "documents": ["Recent Q", "Recent A"],
                "metadatas": [
                    {
                        "role": "user",
                        "session": "s1",
                        "timestamp": now.isoformat(),
                        "order": now.timestamp(),
                        "turn_id": "recent-tid",
                    },
                    {
                        "role": "assistant",
                        "session": "s1",
                        "timestamp": (now + timedelta(microseconds=1)).isoformat(),
                        "order": (now + timedelta(microseconds=1)).timestamp(),
                        "turn_id": "recent-tid",
                    },
                ],
                "ids": ["r_u", "r_a"],
            }
        )

        # Set up get_semantic_pairs to return one semantic pair
        old_time = now - timedelta(days=1)
        sem_tid = "semantic-tid"
        mm.chats.query = Mock(
            return_value={
                "documents": [["Old Q"]],
                "metadatas": [
                    [
                        {
                            "role": "user",
                            "session": "old-s",
                            "timestamp": old_time.isoformat(),
                            "order": old_time.timestamp(),
                            "turn_id": sem_tid,
                        }
                    ]
                ],
                "ids": [["sem_u"]],
                "distances": [[0.05]],
            }
        )

        # Partner lookup for semantic pair
        original_get = mm.chats.get

        def smart_get(**kwargs):
            where = kwargs.get("where", {})
            if where.get("turn_id") == sem_tid:
                return {
                    "documents": ["Old Q", "Old A"],
                    "metadatas": [
                        {
                            "role": "user",
                            "turn_id": sem_tid,
                            "timestamp": old_time.isoformat(),
                            "order": old_time.timestamp(),
                        },
                        {
                            "role": "assistant",
                            "turn_id": sem_tid,
                            "timestamp": (
                                old_time + timedelta(microseconds=1)
                            ).isoformat(),
                            "order": (old_time + timedelta(microseconds=1)).timestamp(),
                        },
                    ],
                    "ids": ["sem_u", "sem_a"],
                }
            return original_get(**kwargs)

        mm.chats.get = Mock(side_effect=smart_get)

        history = mm.build_message_history(
            "test query", "s1", n_recent_pairs=1, n_semantic_pairs=1
        )

        # Should have 4 messages: 2 semantic + 2 recent
        assert len(history) == 4
        # First pair (semantic) should be "Old Q" / "Old A"
        assert history[0]["content"] == "Old Q"
        assert history[1]["content"] == "Old A"
        # Second pair (recent) should be "Recent Q" / "Recent A"
        assert history[2]["content"] == "Recent Q"
        assert history[3]["content"] == "Recent A"

    def test_deduplicates_overlapping_pairs(self, mm):
        """Pairs present in both semantic and recent should only appear once (in recent)."""
        now = datetime.now()
        shared_tid = "shared-tid"

        # Both recent and semantic return the same pair
        mm.chats.get = Mock(
            return_value={
                "documents": ["Q", "A"],
                "metadatas": [
                    {
                        "role": "user",
                        "session": "s1",
                        "timestamp": now.isoformat(),
                        "order": now.timestamp(),
                        "turn_id": shared_tid,
                    },
                    {
                        "role": "assistant",
                        "session": "s1",
                        "timestamp": (now + timedelta(microseconds=1)).isoformat(),
                        "order": (now + timedelta(microseconds=1)).timestamp(),
                        "turn_id": shared_tid,
                    },
                ],
                "ids": ["u1", "a1"],
            }
        )

        mm.chats.query = Mock(
            return_value={
                "documents": [["Q"]],
                "metadatas": [
                    [
                        {
                            "role": "user",
                            "session": "s1",
                            "timestamp": now.isoformat(),
                            "order": now.timestamp(),
                            "turn_id": shared_tid,
                        }
                    ]
                ],
                "ids": [["u1"]],
                "distances": [[0.01]],
            }
        )

        history = mm.build_message_history(
            "Q", "s1", n_recent_pairs=1, n_semantic_pairs=1
        )

        # Should only have 2 messages (the deduped pair from recent)
        assert len(history) == 2
        assert history[0]["content"] == "Q"
        assert history[1]["content"] == "A"

    def test_empty_history(self, mm):
        mm.chats.get = Mock(return_value={"documents": [], "metadatas": [], "ids": []})
        mm.chats.query = Mock(
            return_value={
                "documents": [[]],
                "metadatas": [[]],
                "ids": [[]],
                "distances": [[]],
            }
        )

        history = mm.build_message_history("hello", "empty-session")
        assert history == []


# ---------------------------------------------------------------------------
# get_recent_chats (existing method, verify turn_id inclusion)
# ---------------------------------------------------------------------------


class TestGetRecentChats:
    """Tests for get_recent_chats including turn_id in output."""

    @pytest.fixture
    def mm(self):
        return _make_memory_manager()

    def test_includes_turn_id(self, mm):
        mm.chats.get = Mock(
            return_value={
                "documents": ["Hello"],
                "metadatas": [
                    {
                        "role": "user",
                        "session": "s1",
                        "timestamp": "t1",
                        "order": 1.0,
                        "turn_id": "tid-abc",
                    }
                ],
                "ids": ["id_1"],
            }
        )
        messages = mm.get_recent_chats("s1", n=5)
        assert len(messages) == 1
        assert messages[0]["turn_id"] == "tid-abc"

    def test_empty_session(self, mm):
        mm.chats.get = Mock(return_value={"documents": [], "metadatas": [], "ids": []})
        messages = mm.get_recent_chats("empty", n=5)
        assert messages == []

    def test_returns_no_results(self, mm):
        mm.chats.get = Mock(return_value=None)
        messages = mm.get_recent_chats("none", n=5)
        assert messages == []


# ---------------------------------------------------------------------------
# Chat history page helpers
# ---------------------------------------------------------------------------


class TestChatHistoryHelpers:
    """Tests for get_all_sessions/get_session_history/delete_all_chats."""

    @pytest.fixture
    def mm(self):
        return _make_memory_manager()

    def test_get_all_sessions_groups_and_counts_messages(self, mm):
        mm.chats.get = Mock(
            return_value={
                "documents": ["Q1", "A1", "Q2"],
                "metadatas": [
                    {
                        "session": "s1",
                        "role": "user",
                        "timestamp": "2026-02-22T10:00:00",
                    },
                    {
                        "session": "s1",
                        "role": "assistant",
                        "timestamp": "2026-02-22T10:00:01",
                    },
                    {
                        "session": "s2",
                        "role": "user",
                        "timestamp": "2026-02-22T12:00:00",
                    },
                ],
                "ids": ["1", "2", "3"],
            }
        )

        sessions = mm.get_all_sessions()
        assert len(sessions) == 2

        by_id = {s["session_id"]: s for s in sessions}
        assert by_id["s1"]["msg_count"] == 2
        assert by_id["s1"]["start_time"] == "2026-02-22T10:00:00"
        assert by_id["s2"]["msg_count"] == 1

    def test_get_all_sessions_returns_empty_on_error(self, mm):
        mm.chats.get = Mock(side_effect=Exception("DB error"))
        assert mm.get_all_sessions() == []

    def test_get_session_history_returns_sorted_messages(self, mm):
        mm.chats.get = Mock(
            return_value={
                "documents": ["A1", "Q1"],
                "metadatas": [
                    {
                        "role": "assistant",
                        "session": "s1",
                        "timestamp": "2026-02-22T10:00:01",
                        "order": 2.0,
                        "turn_id": "tid-1",
                    },
                    {
                        "role": "user",
                        "session": "s1",
                        "timestamp": "2026-02-22T10:00:00",
                        "order": 1.0,
                        "turn_id": "tid-1",
                    },
                ],
                "ids": ["a1", "u1"],
            }
        )

        messages = mm.get_session_history("s1")

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Q1"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "A1"

    def test_get_session_history_returns_empty_for_missing_session(self, mm):
        mm.chats.get = Mock(return_value={"documents": [], "metadatas": [], "ids": []})
        assert mm.get_session_history("missing") == []

    def test_delete_all_chats_deletes_ids(self, mm):
        mm.chats.get = Mock(
            side_effect=[
                {"ids": ["id_1", "id_2"]},
                {"ids": []},
            ]
        )

        assert mm.delete_all_chats() is True
        mm.chats.delete.assert_called_once_with(ids=["id_1", "id_2"])

    def test_delete_all_chats_noop_when_empty(self, mm):
        mm.chats.get = Mock(return_value={"ids": []})

        assert mm.delete_all_chats() is True
        mm.chats.delete.assert_not_called()

    def test_delete_all_chats_returns_false_on_error(self, mm):
        mm.chats.get = Mock(side_effect=Exception("DB error"))
        assert mm.delete_all_chats() is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def mm(self):
        return _make_memory_manager()

    def test_should_extract_memory_empty(self, mm):
        assert mm._should_extract_memory("") is False
        assert mm._should_extract_memory("   ") is False
        assert mm._should_extract_memory("hi") is False  # too short

    def test_should_extract_memory_questions(self, mm):
        assert mm._should_extract_memory("What is the weather?") is False
        assert mm._should_extract_memory("Can you help me?") is False

    def test_should_extract_memory_triggers(self, mm):
        assert mm._should_extract_memory("Remember that I like pizza") is True
        assert mm._should_extract_memory("My name is Adi") is True
        assert mm._should_extract_memory("I prefer dark mode") is True
