# ABOUTME: ConversationManager — saves, retrieves, and reconstructs conversation turns.
# ABOUTME: Uses heathcliff_conversations Chroma collection; returns real LangChain messages.

import json
import uuid
import zlib
from base64 import b64decode, b64encode
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from db.base import CONVERSATIONS_COLLECTION, ChromaConnection
from logger import logger

CHROMA_QUERY_BATCH_SIZE = 200
CHROMA_METADATA_CHUNK_BYTES = 7000


def _execution_events_metadata(events_json: str) -> Dict[str, Any]:
    """Keep complete execution history below Chroma's per-value byte quota."""
    if len(events_json.encode("utf-8")) <= CHROMA_METADATA_CHUNK_BYTES:
        return {"execution_events_json": events_json}

    encoded = b64encode(zlib.compress(events_json.encode("utf-8"))).decode("ascii")
    chunks = [
        encoded[index : index + CHROMA_METADATA_CHUNK_BYTES]
        for index in range(0, len(encoded), CHROMA_METADATA_CHUNK_BYTES)
    ]
    return {
        "execution_events_encoding": "zlib-base64",
        "execution_events_chunks": len(chunks),
        **{f"execution_events_{index}": chunk for index, chunk in enumerate(chunks)},
    }


class ConversationMessageRecord(BaseModel):
    id: str
    conversation_id: str
    turn_id: str
    message_index: int  # 0 = user, 1 = assistant
    role: str  # "user" | "assistant"
    searchable_text: str
    message_payload: dict  # {"type": "text", "text": ...} or multimodal blocks
    artifact_uris: list[str] = []
    created_at: str  # ISO timestamp
    metadata: dict = {}


class ConversationManager:
    """Persists and retrieves conversation turns for a single Chroma collection."""

    def __init__(self) -> None:
        self._collection = ChromaConnection.get().get_or_create_collection(
            name=CONVERSATIONS_COLLECTION,
            metadata={"description": "Heathcliff conversation history"},
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_turn(
        self,
        user_msg: str,
        assistant_msg: str,
        conversation_id: str,
        execution_events: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[str, str]:
        """Persist a user/assistant turn as two ConversationMessageRecords.

        Returns:
            Tuple of (user_record_id, assistant_record_id).
        """
        now = datetime.now()
        turn_id = str(uuid.uuid4())
        user_id = f"{conversation_id}_{uuid.uuid4()}_user"
        asst_id = f"{conversation_id}_{uuid.uuid4()}_assistant"
        asst_time = now + timedelta(microseconds=1)

        user_record = ConversationMessageRecord(
            id=user_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            message_index=0,
            role="user",
            searchable_text=user_msg,
            message_payload={"type": "text", "text": user_msg},
            created_at=now.isoformat(),
            metadata={"order": now.timestamp()},
        )
        events_json = json.dumps(execution_events or [], default=str)
        events_metadata = _execution_events_metadata(events_json)
        asst_record = ConversationMessageRecord(
            id=asst_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            message_index=1,
            role="assistant",
            searchable_text=assistant_msg,
            message_payload={"type": "text", "text": assistant_msg},
            created_at=asst_time.isoformat(),
            metadata={
                "order": asst_time.timestamp(),
                "execution_events_json": events_json,
            },
        )

        self._collection.add(
            documents=[user_record.searchable_text, asst_record.searchable_text],
            metadatas=[
                {
                    "role": user_record.role,
                    "conversation_id": user_record.conversation_id,
                    "turn_id": user_record.turn_id,
                    "message_index": user_record.message_index,
                    "created_at": user_record.created_at,
                    "order": user_record.metadata["order"],
                    "artifact_uris": str(user_record.artifact_uris),
                },
                {
                    "role": asst_record.role,
                    "conversation_id": asst_record.conversation_id,
                    "turn_id": asst_record.turn_id,
                    "message_index": asst_record.message_index,
                    "created_at": asst_record.created_at,
                    "order": asst_record.metadata["order"],
                    "artifact_uris": str(asst_record.artifact_uris),
                    **events_metadata,
                },
            ],
            ids=[user_id, asst_id],
        )

        return user_id, asst_id

    # ------------------------------------------------------------------
    # LangChain history reconstruction
    # ------------------------------------------------------------------

    def build_langchain_history(
        self,
        query: str,
        conversation_id: str,
        n_recent_pairs: Optional[int] = None,
        n_semantic_pairs: Optional[int] = None,
    ) -> List[HumanMessage | AIMessage]:
        """Return deduplicated conversation history as real LangChain message objects.

        Order: semantic pairs first (cross-session context), then recent chronological
        pairs (current-session continuity). Current HumanMessage is NOT included —
        the caller appends it.
        """
        from config import Config as _Cfg

        n_recent = (
            n_recent_pairs if n_recent_pairs is not None else _Cfg.RECENT_PAIRS_COUNT
        )
        n_semantic = (
            n_semantic_pairs
            if n_semantic_pairs is not None
            else _Cfg.SEMANTIC_PAIRS_COUNT
        )

        recent_pairs = self.get_recent_pairs(conversation_id, n_pairs=n_recent)
        semantic_pairs = self.get_semantic_pairs(query, n_pairs=n_semantic)

        # Collect dedup keys from recent pairs
        recent_keys: set = set()
        for u, a in recent_pairs:
            tid = u.get("turn_id") or a.get("turn_id") or ""
            if tid:
                recent_keys.add(tid)
            recent_keys.add((u.get("content", ""), a.get("content", "")))

        # Filter semantic pairs: drop any already in recent
        filtered_semantic: List[tuple] = []
        for u, a in semantic_pairs:
            tid = u.get("turn_id") or a.get("turn_id") or ""
            content_key = (u.get("content", ""), a.get("content", ""))
            if tid and tid in recent_keys:
                continue
            if content_key in recent_keys:
                continue
            filtered_semantic.append((u, a))

        messages: List[HumanMessage | AIMessage] = []
        for u, a in filtered_semantic:
            messages.append(HumanMessage(content=u.get("content", "")))
            messages.append(AIMessage(content=a.get("content", "")))
        for u, a in recent_pairs:
            messages.append(HumanMessage(content=u.get("content", "")))
            messages.append(AIMessage(content=a.get("content", "")))

        return messages

    # ------------------------------------------------------------------
    # Pair retrieval
    # ------------------------------------------------------------------

    def get_recent_pairs(self, conversation_id: str, n_pairs: int = 6) -> List[tuple]:
        """Return the last n_pairs turns from conversation_id, chronologically."""
        messages = self._get_recent_messages(conversation_id, n=n_pairs * 2)
        return self._build_pairs(messages, n_pairs)

    def get_semantic_pairs(self, query: str, n_pairs: int = 3) -> List[tuple]:
        """Search all history semantically and return matching pairs."""
        if not query or not query.strip():
            return []

        n_candidates = n_pairs * 4
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=n_candidates,
            )
        except Exception as exc:
            logger.warning("Semantic conversation search failed: %s", exc)
            return []

        if not results or not results.get("documents") or not results["documents"][0]:
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        ids = results["ids"][0]

        seen_turn_ids: set = set()
        pairs: List[tuple] = []

        for i, doc in enumerate(docs):
            meta = metas[i]
            msg = self._meta_to_dict(doc, meta, ids[i])
            tid = msg.get("turn_id") or ""
            if tid and tid in seen_turn_ids:
                continue
            pair = self._resolve_pair(msg)
            if pair:
                pair_tid = pair[0].get("turn_id") or pair[1].get("turn_id") or ""
                if pair_tid and pair_tid in seen_turn_ids:
                    continue
                pairs.append(pair)
                if pair_tid:
                    seen_turn_ids.add(pair_tid)
            if len(pairs) >= n_pairs:
                break

        pairs.sort(key=lambda p: self._sort_key(p[0]))
        return pairs

    # ------------------------------------------------------------------
    # Conversation management
    # ------------------------------------------------------------------

    def get_all_conversations(self) -> List[Dict[str, Any]]:
        """Return conversation summaries for the chat history UI."""
        try:
            pages = self._iter_pages()
        except Exception as exc:
            logger.warning("Error retrieving conversations: %s", exc)
            return []

        conversations: Dict[str, Dict[str, Any]] = {}
        for page in pages:
            for metadata in (page or {}).get("metadatas", []) or []:
                if not isinstance(metadata, dict):
                    continue
                cid = str(metadata.get("conversation_id") or "").strip()
                if not cid:
                    continue
                timestamp = str(metadata.get("created_at") or "")
                if cid not in conversations:
                    conversations[cid] = {
                        "conversation_id": cid,
                        "start_time": timestamp,
                        "msg_count": 1,
                    }
                else:
                    conversations[cid]["msg_count"] += 1
                    existing = conversations[cid].get("start_time", "")
                    if timestamp and (not existing or timestamp < existing):
                        conversations[cid]["start_time"] = timestamp

        return list(conversations.values())

    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Return all messages from a conversation in chronological order."""
        try:
            pages = self._iter_pages(where={"conversation_id": conversation_id})
        except Exception as exc:
            logger.warning("Error retrieving conversation history: %s", exc)
            return []

        messages: List[Dict[str, Any]] = []
        for page in pages:
            docs = page.get("documents", []) or []
            metas = page.get("metadatas", []) or []
            ids = page.get("ids", []) or []
            for i, doc in enumerate(docs):
                meta = metas[i] if i < len(metas) and metas[i] else {}
                mid = ids[i] if i < len(ids) else ""
                messages.append(self._meta_to_dict(doc, meta, mid))

        messages.sort(key=self._sort_key)
        return messages

    def clear_conversation(self, conversation_id: str) -> bool:
        """Delete all messages from a conversation."""
        try:
            self._collection.delete(where={"conversation_id": conversation_id})
            return True
        except Exception as exc:
            logger.warning("Error clearing conversation %s: %s", conversation_id, exc)
            return False

    def delete_all_chats(self) -> bool:
        """Delete all conversation history."""
        try:
            previous_batch_ids: List[str] = []
            while True:
                results = self._collection.get(limit=CHROMA_QUERY_BATCH_SIZE, offset=0)
                ids = (results or {}).get("ids", []) or []
                if not ids:
                    return True
                if ids == previous_batch_ids:
                    logger.warning("delete_all_chats made no progress; stopping")
                    return False
                self._collection.delete(ids=ids)
                previous_batch_ids = ids
        except Exception as exc:
            logger.warning("Error deleting all chats: %s", exc)
            return False

    def count(self) -> int:
        return self._collection.count()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_recent_messages(
        self, conversation_id: str, n: int
    ) -> List[Dict[str, Any]]:
        results = self._collection.get(
            where={"conversation_id": conversation_id},
            limit=n * 2,
        )
        if not results or not results.get("documents"):
            return []
        messages = [
            self._meta_to_dict(doc, results["metadatas"][i], results["ids"][i])
            for i, doc in enumerate(results["documents"])
        ]
        messages.sort(key=self._sort_key)
        return messages[-n:]

    def _build_pairs(self, messages: List[Dict[str, Any]], n_pairs: int) -> List[tuple]:
        seen_turn_ids: set = set()
        pairs: List[tuple] = []
        for msg in messages:
            tid = msg.get("turn_id") or ""
            if tid and tid in seen_turn_ids:
                continue
            pair = self._resolve_pair(msg)
            if pair:
                pair_tid = pair[0].get("turn_id") or pair[1].get("turn_id") or ""
                if pair_tid and pair_tid in seen_turn_ids:
                    continue
                pairs.append(pair)
                if pair_tid:
                    seen_turn_ids.add(pair_tid)
        pairs.sort(key=lambda p: self._sort_key(p[0]))
        return pairs[-n_pairs:]

    def _resolve_pair(self, msg: Dict[str, Any]) -> Optional[tuple]:
        """Return (user_msg, assistant_msg) for the given message, or None."""
        turn_id = msg.get("turn_id") or ""
        partner: Optional[Dict[str, Any]] = None

        if turn_id:
            partner = self._fetch_partner_by_turn_id(turn_id, msg.get("id", ""))
        if not partner:
            partner = self._fetch_partner_by_adjacency(msg)
        if not partner:
            return None

        if msg.get("role") == "user":
            return (msg, partner)
        return (partner, msg)

    def _fetch_partner_by_turn_id(
        self, turn_id: str, exclude_id: str
    ) -> Optional[Dict[str, Any]]:
        try:
            results = self._collection.get(where={"turn_id": turn_id}, limit=2)
        except Exception:
            return None
        if not results or not results.get("documents"):
            return None
        for i, doc in enumerate(results["documents"]):
            pid = results["ids"][i]
            if pid != exclude_id:
                return self._meta_to_dict(doc, results["metadatas"][i], pid)
        return None

    def _fetch_partner_by_adjacency(
        self, msg: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        cid = msg.get("conversation_id") or ""
        if not cid:
            return None
        try:
            results = self._collection.get(where={"conversation_id": cid}, limit=200)
        except Exception:
            return None
        if not results or not results.get("documents"):
            return None

        all_msgs = [
            self._meta_to_dict(doc, results["metadatas"][i], results["ids"][i])
            for i, doc in enumerate(results["documents"])
        ]
        all_msgs.sort(key=self._sort_key)

        target_role = "assistant" if msg.get("role") == "user" else "user"
        msg_id = msg.get("id")
        idx = next((i for i, m in enumerate(all_msgs) if m["id"] == msg_id), None)
        if idx is None:
            return None

        if msg.get("role") == "user":
            if idx + 1 < len(all_msgs) and all_msgs[idx + 1]["role"] == target_role:
                return all_msgs[idx + 1]
        else:
            if idx - 1 >= 0 and all_msgs[idx - 1]["role"] == target_role:
                return all_msgs[idx - 1]
        return None

    def _iter_pages(self, where: Optional[Dict[str, Any]] = None) -> List[Any]:
        pages: List[Any] = []
        offset = 0
        while True:
            kwargs: Dict[str, Any] = {
                "limit": CHROMA_QUERY_BATCH_SIZE,
                "offset": offset,
            }
            if where:
                kwargs["where"] = where
            results = self._collection.get(**kwargs)
            if not results:
                break
            ids = (results or {}).get("ids", []) or []
            if not ids:
                break
            pages.append(results)
            if len(ids) < CHROMA_QUERY_BATCH_SIZE:
                break
            offset += CHROMA_QUERY_BATCH_SIZE
        return pages

    @staticmethod
    def _meta_to_dict(doc: str, meta: Dict[str, Any], msg_id: str) -> Dict[str, Any]:
        raw_events = meta.get("execution_events_json", "[]")
        if meta.get("execution_events_encoding") == "zlib-base64":
            try:
                chunk_count = int(meta.get("execution_events_chunks", 0))
                encoded = "".join(
                    str(meta.get(f"execution_events_{index}", ""))
                    for index in range(chunk_count)
                )
                raw_events = zlib.decompress(b64decode(encoded)).decode("utf-8")
            except Exception:
                raw_events = "[]"
        try:
            execution_events = json.loads(raw_events)
        except (TypeError, json.JSONDecodeError):
            execution_events = []
        if not isinstance(execution_events, list):
            execution_events = []
        return {
            "id": msg_id,
            "role": meta.get("role", "unknown"),
            "content": doc,
            "turn_id": meta.get("turn_id", ""),
            "conversation_id": meta.get("conversation_id", ""),
            "created_at": meta.get("created_at", ""),
            "order": meta.get("order"),
            "message_index": meta.get("message_index", 0),
            "execution_events": execution_events,
        }

    @staticmethod
    def _sort_key(msg: Dict[str, Any]) -> tuple:
        order_value = msg.get("order")
        if order_value is not None:
            return (0, order_value)
        created_at = msg.get("created_at", "")
        role_priority = 0 if msg.get("role") == "user" else 1
        return (1, created_at, role_priority)
