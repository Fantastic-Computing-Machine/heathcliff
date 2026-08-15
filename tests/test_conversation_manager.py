# ABOUTME: Unit tests for ConversationManager and ConversationMessageRecord.
# ABOUTME: Tests save_turn, pair retrieval, build_langchain_history, and conversation management.

import os
import sys
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helper — build a ConversationManager with mocked Chroma
# ---------------------------------------------------------------------------


def _make_manager():
    from db.base import ChromaConnection
    from db.conversation_manager import ConversationManager

    ChromaConnection.reset()
    mock_collection = Mock()
    mock_conn = Mock()
    mock_conn.get_or_create_collection.return_value = mock_collection
    ChromaConnection._instance = ChromaConnection.__new__(ChromaConnection)
    ChromaConnection._instance._client = mock_conn.get_or_create_collection.return_value

    # Patch get_or_create_collection on the instance
    with patch.object(
        ChromaConnection, "get_or_create_collection", return_value=mock_collection
    ):
        mgr = ConversationManager()

    mgr._mock_collection = mock_collection
    return mgr


# ---------------------------------------------------------------------------
# ConversationMessageRecord
# ---------------------------------------------------------------------------


class TestConversationMessageRecord:
    def test_default_artifact_uris_is_empty(self):
        from db.conversation_manager import ConversationMessageRecord

        r = ConversationMessageRecord(
            id="r1",
            conversation_id="c1",
            turn_id="t1",
            message_index=0,
            role="user",
            searchable_text="hello",
            message_payload={"type": "text", "text": "hello"},
            created_at="2025-01-01T00:00:00",
        )
        assert r.artifact_uris == []

    def test_message_payload_is_dict(self):
        from db.conversation_manager import ConversationMessageRecord

        payload = {"type": "text", "text": "hello"}
        r = ConversationMessageRecord(
            id="r1",
            conversation_id="c1",
            turn_id="t1",
            message_index=0,
            role="user",
            searchable_text="hello",
            message_payload=payload,
            created_at="2025-01-01T00:00:00",
        )
        assert r.message_payload == payload

    def test_message_index_zero_for_user(self):
        from db.conversation_manager import ConversationMessageRecord

        r = ConversationMessageRecord(
            id="r1",
            conversation_id="c1",
            turn_id="t1",
            message_index=0,
            role="user",
            searchable_text="q",
            message_payload={"type": "text", "text": "q"},
            created_at="2025-01-01T00:00:00",
        )
        assert r.message_index == 0

    def test_message_index_one_for_assistant(self):
        from db.conversation_manager import ConversationMessageRecord

        r = ConversationMessageRecord(
            id="r1",
            conversation_id="c1",
            turn_id="t1",
            message_index=1,
            role="assistant",
            searchable_text="a",
            message_payload={"type": "text", "text": "a"},
            created_at="2025-01-01T00:00:00",
        )
        assert r.message_index == 1


# ---------------------------------------------------------------------------
# save_turn
# ---------------------------------------------------------------------------


class TestSaveTurn:
    @pytest.fixture
    def mgr(self):
        return _make_manager()

    def test_returns_tuple_of_ids(self, mgr):
        result = mgr.save_turn("Hello", "Hi there!", "conv-1")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_stores_two_documents(self, mgr):
        mgr.save_turn("Hello", "Hi there!", "conv-1")
        call_args = mgr._mock_collection.add.call_args
        docs = call_args[1].get("documents") or call_args[0][0]
        assert len(docs) == 2

    def test_shared_turn_id(self, mgr):
        mgr.save_turn("Question?", "Answer.", "conv-1")
        metadatas = mgr._mock_collection.add.call_args[1]["metadatas"]
        assert metadatas[0]["turn_id"] == metadatas[1]["turn_id"]
        assert len(metadatas[0]["turn_id"]) > 0

    def test_user_role_first(self, mgr):
        mgr.save_turn("Q", "A", "conv-1")
        metadatas = mgr._mock_collection.add.call_args[1]["metadatas"]
        assert metadatas[0]["role"] == "user"
        assert metadatas[1]["role"] == "assistant"

    def test_order_is_monotonic(self, mgr):
        mgr.save_turn("Q", "A", "conv-1")
        metadatas = mgr._mock_collection.add.call_args[1]["metadatas"]
        assert metadatas[0]["order"] < metadatas[1]["order"]

    def test_conversation_id_stored(self, mgr):
        mgr.save_turn("Q", "A", "conv-42")
        metadatas = mgr._mock_collection.add.call_args[1]["metadatas"]
        assert metadatas[0]["conversation_id"] == "conv-42"
        assert metadatas[1]["conversation_id"] == "conv-42"

    def test_message_payload_stored(self, mgr):
        mgr.save_turn("hello", "world", "conv-1")
        metadatas = mgr._mock_collection.add.call_args[1]["metadatas"]
        assert "message_payload" in metadatas[0]
        assert "message_payload" in metadatas[1]


# ---------------------------------------------------------------------------
# _sort_key
# ---------------------------------------------------------------------------


class TestSortKey:
    def test_order_takes_precedence(self):
        from db.conversation_manager import ConversationManager

        msg = {"order": 100.0, "created_at": "2025-01-01T00:00:00", "role": "user"}
        key = ConversationManager._sort_key(msg)
        assert key[0] == 0

    def test_fallback_to_created_at(self):
        from db.conversation_manager import ConversationManager

        msg = {"created_at": "2025-01-01T00:00:00", "role": "user"}
        key = ConversationManager._sort_key(msg)
        assert key[0] == 1

    def test_user_before_assistant_in_fallback(self):
        from db.conversation_manager import ConversationManager

        user = {"created_at": "2025-01-01T00:00:00", "role": "user"}
        asst = {"created_at": "2025-01-01T00:00:00", "role": "assistant"}
        assert ConversationManager._sort_key(user) < ConversationManager._sort_key(asst)


# ---------------------------------------------------------------------------
# _fetch_partner_by_turn_id
# ---------------------------------------------------------------------------


class TestFetchPartnerByTurnId:
    @pytest.fixture
    def mgr(self):
        return _make_manager()

    def test_finds_partner(self, mgr):
        turn_id = "tid-123"
        mgr._mock_collection.get = Mock(
            return_value={
                "documents": ["Q", "A"],
                "metadatas": [
                    {
                        "role": "user",
                        "turn_id": turn_id,
                        "created_at": "t1",
                        "order": 1.0,
                        "conversation_id": "c1",
                        "message_index": 0,
                    },
                    {
                        "role": "assistant",
                        "turn_id": turn_id,
                        "created_at": "t2",
                        "order": 2.0,
                        "conversation_id": "c1",
                        "message_index": 1,
                    },
                ],
                "ids": ["id_user", "id_asst"],
            }
        )
        partner = mgr._fetch_partner_by_turn_id(turn_id, "id_user")
        assert partner is not None
        assert partner["role"] == "assistant"
        assert partner["id"] == "id_asst"

    def test_returns_none_when_only_one_message(self, mgr):
        mgr._mock_collection.get = Mock(
            return_value={
                "documents": ["Q"],
                "metadatas": [
                    {
                        "role": "user",
                        "turn_id": "tid",
                        "created_at": "t1",
                        "order": 1.0,
                        "conversation_id": "c1",
                        "message_index": 0,
                    }
                ],
                "ids": ["id_user"],
            }
        )
        partner = mgr._fetch_partner_by_turn_id("tid", "id_user")
        assert partner is None

    def test_returns_none_on_exception(self, mgr):
        mgr._mock_collection.get = Mock(side_effect=Exception("DB error"))
        partner = mgr._fetch_partner_by_turn_id("tid", "id_user")
        assert partner is None


# ---------------------------------------------------------------------------
# _resolve_pair
# ---------------------------------------------------------------------------


class TestResolvePair:
    @pytest.fixture
    def mgr(self):
        return _make_manager()

    def _make_chroma_result(self, turn_id):
        return {
            "documents": ["Q", "A"],
            "metadatas": [
                {
                    "role": "user",
                    "turn_id": turn_id,
                    "created_at": "t1",
                    "order": 1.0,
                    "conversation_id": "c1",
                    "message_index": 0,
                },
                {
                    "role": "assistant",
                    "turn_id": turn_id,
                    "created_at": "t2",
                    "order": 2.0,
                    "conversation_id": "c1",
                    "message_index": 1,
                },
            ],
            "ids": ["id_u", "id_a"],
        }

    def test_resolve_pair_from_user_msg(self, mgr):
        turn_id = "tid-1"
        mgr._mock_collection.get = Mock(return_value=self._make_chroma_result(turn_id))
        user_msg = {
            "role": "user",
            "content": "Q",
            "turn_id": turn_id,
            "id": "id_u",
            "created_at": "t1",
            "order": 1.0,
        }
        pair = mgr._resolve_pair(user_msg)
        assert pair is not None
        assert pair[0]["role"] == "user"
        assert pair[1]["role"] == "assistant"

    def test_resolve_pair_from_assistant_msg(self, mgr):
        turn_id = "tid-2"
        mgr._mock_collection.get = Mock(return_value=self._make_chroma_result(turn_id))
        asst_msg = {
            "role": "assistant",
            "content": "A",
            "turn_id": turn_id,
            "id": "id_a",
            "created_at": "t2",
            "order": 2.0,
        }
        pair = mgr._resolve_pair(asst_msg)
        assert pair is not None
        assert pair[0]["role"] == "user"
        assert pair[1]["role"] == "assistant"

    def test_resolve_pair_returns_none_when_no_partner(self, mgr):
        mgr._mock_collection.get = Mock(
            return_value={"documents": [], "metadatas": [], "ids": []}
        )
        msg = {
            "role": "user",
            "content": "Q",
            "turn_id": "no-match",
            "id": "id_u",
            "created_at": "t1",
            "order": 1.0,
            "conversation_id": "c1",
        }
        pair = mgr._resolve_pair(msg)
        assert pair is None


# ---------------------------------------------------------------------------
# build_langchain_history
# ---------------------------------------------------------------------------


class TestBuildLangchainHistory:
    @pytest.fixture
    def mgr(self):
        return _make_manager()

    def test_returns_list_of_langchain_messages(self, mgr):
        turn_id = "t1"
        chroma_result = {
            "documents": ["Hello", "Hi there"],
            "metadatas": [
                {
                    "role": "user",
                    "turn_id": turn_id,
                    "created_at": "t1",
                    "order": 1.0,
                    "conversation_id": "conv-1",
                    "message_index": 0,
                },
                {
                    "role": "assistant",
                    "turn_id": turn_id,
                    "created_at": "t2",
                    "order": 2.0,
                    "conversation_id": "conv-1",
                    "message_index": 1,
                },
            ],
            "ids": ["id_u", "id_a"],
        }
        mgr._mock_collection.get = Mock(return_value=chroma_result)
        mgr._mock_collection.query = Mock(
            return_value={"documents": [[]], "metadatas": [[]], "ids": [[]]}
        )

        history = mgr.build_langchain_history(
            "Hello", "conv-1", n_recent_pairs=1, n_semantic_pairs=0
        )

        assert isinstance(history, list)
        for msg in history:
            assert isinstance(msg, (HumanMessage, AIMessage))

    def test_returns_empty_for_no_history(self, mgr):
        mgr._mock_collection.get = Mock(
            return_value={"documents": [], "metadatas": [], "ids": []}
        )
        mgr._mock_collection.query = Mock(
            return_value={"documents": [[]], "metadatas": [[]], "ids": [[]]}
        )

        history = mgr.build_langchain_history(
            "hi", "empty-conv", n_recent_pairs=3, n_semantic_pairs=0
        )
        assert history == []

    def test_user_message_is_human_message(self, mgr):
        turn_id = "t1"
        chroma_result = {
            "documents": ["User question", "Assistant answer"],
            "metadatas": [
                {
                    "role": "user",
                    "turn_id": turn_id,
                    "created_at": "t1",
                    "order": 1.0,
                    "conversation_id": "conv-1",
                    "message_index": 0,
                },
                {
                    "role": "assistant",
                    "turn_id": turn_id,
                    "created_at": "t2",
                    "order": 2.0,
                    "conversation_id": "conv-1",
                    "message_index": 1,
                },
            ],
            "ids": ["id_u", "id_a"],
        }
        mgr._mock_collection.get = Mock(return_value=chroma_result)
        mgr._mock_collection.query = Mock(
            return_value={"documents": [[]], "metadatas": [[]], "ids": [[]]}
        )

        history = mgr.build_langchain_history(
            "q", "conv-1", n_recent_pairs=1, n_semantic_pairs=0
        )

        human_msgs = [m for m in history if isinstance(m, HumanMessage)]
        ai_msgs = [m for m in history if isinstance(m, AIMessage)]
        assert len(human_msgs) >= 1
        assert len(ai_msgs) >= 1


# ---------------------------------------------------------------------------
# clear_conversation / delete_all_chats
# ---------------------------------------------------------------------------


class TestConversationManagement:
    @pytest.fixture
    def mgr(self):
        return _make_manager()

    def test_clear_conversation_calls_delete(self, mgr):
        mgr.clear_conversation("conv-1")
        mgr._mock_collection.delete.assert_called_once_with(
            where={"conversation_id": "conv-1"}
        )

    def test_clear_conversation_returns_true_on_success(self, mgr):
        assert mgr.clear_conversation("conv-1") is True

    def test_clear_conversation_returns_false_on_error(self, mgr):
        mgr._mock_collection.delete.side_effect = Exception("error")
        assert mgr.clear_conversation("conv-1") is False

    def test_delete_all_chats_returns_true_when_empty(self, mgr):
        mgr._mock_collection.get = Mock(return_value={"ids": []})
        assert mgr.delete_all_chats() is True

    def test_get_all_conversations_returns_list(self, mgr):
        mgr._mock_collection.get = Mock(
            return_value={"documents": [], "metadatas": [], "ids": []}
        )
        result = mgr.get_all_conversations()
        assert isinstance(result, list)
