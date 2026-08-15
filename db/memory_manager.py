# ABOUTME: MemoryManager — public facade for all persistence concerns.
# ABOUTME: Owns ConversationManager, Mem0 client, and the async
# ABOUTME: extraction queue. Call sites interact only with this class.

import queue
import threading
import uuid
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional

from mem0 import Memory

from config import Config
from db.base import MEMORIES_COLLECTION, ChromaConnection
from db.conversation_manager import ConversationManager
from logger import logger
from utils.errors import AgentMemoryError

CHROMA_QUERY_BATCH_SIZE = 200


class _Mem0CollectionAdapter:
    """Chroma-like get() interface over Mem0 memories (for UI compatibility)."""

    def __init__(self, client: Memory, user_id: str, agent_id: str) -> None:
        self._client = client
        self._user_id = user_id
        self._agent_id = agent_id

    def get(self, limit: int = 100) -> Dict[str, List[Any]]:
        if limit > CHROMA_QUERY_BATCH_SIZE:
            limit = CHROMA_QUERY_BATCH_SIZE
        try:
            result = self._client.get_all(
                user_id=self._user_id, agent_id=self._agent_id, limit=limit
            )
        except Exception as exc:
            logger.warning("Mem0 get_all failed: %s", exc)
            return {"documents": [], "metadatas": [], "ids": []}

        items = result.get("results", result) if isinstance(result, dict) else result
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                text = (
                    item.get("memory") or item.get("text") or item.get("content") or ""
                )
                documents.append(str(text))
                meta = dict(item.get("metadata", {}) or {})
                for key in ("category", "type", "timestamp", "user_id", "agent_id"):
                    if key in item and key not in meta:
                        meta[key] = item.get(key)
                metadatas.append(meta)
                ids.append(str(item.get("id") or item.get("memory_id") or ""))

        return {"documents": documents, "metadatas": metadatas, "ids": ids}


class MemoryManager:
    """
    Facade for all Heathcliff persistence concerns.

    Initialises ChromaConnection, ConversationManager, and Mem0.
    Starts a background daemon thread that processes memory extraction tasks
    from a queue without blocking user-facing turns.

    Usage:
        mm = MemoryManager()
        mm.save_turn(user_msg, response, conversation_id)
        history = mm.build_langchain_history(query, conversation_id)
    """

    _mem0_singleton: Optional[Memory] = None

    def __init__(self) -> None:
        ChromaConnection.initialise(Config)

        self._conversation_manager = ConversationManager()

        self.mem0_user_id = Config.USER_ID
        self.mem0_agent_id = Config.MEM0_AGENT_ID
        self.mem0_client = self._get_mem0_client()

        if not self.mem0_client:
            raise AgentMemoryError("Mem0 client failed to initialise.")

        self.memories = _Mem0CollectionAdapter(
            self.mem0_client, self.mem0_user_id, self.mem0_agent_id
        )

        self._extraction_queue: queue.Queue = queue.Queue()
        self._worker = threading.Thread(
            target=self._extraction_worker, daemon=True, name="mem0-extractor"
        )
        self._worker.start()
        logger.info("MemoryManager initialised; extraction worker started.")

    # ------------------------------------------------------------------
    # Mem0 client
    # ------------------------------------------------------------------

    def _get_mem0_client(self) -> Memory:
        if MemoryManager._mem0_singleton is not None:
            return MemoryManager._mem0_singleton

        chroma_client = ChromaConnection.get().get_client()
        config = deepcopy(Config.MEM0_CONFIG)
        try:
            config["vector_store"]["config"] = {
                "collection_name": MEMORIES_COLLECTION,
                "client": chroma_client,
                "host": Config.CHROMA_HOST,
                "port": Config.CHROMA_PORT,
                "path": Config.CHROMA_PERSIST_DIRECTORY,
            }
            MemoryManager._mem0_singleton = Memory.from_config(config)
            logger.info("Mem0 client initialised.")
            return MemoryManager._mem0_singleton
        except Exception as exc:
            logger.warning("Mem0 SDK initialisation failed: %s", exc)
            raise AgentMemoryError("Mem0 client failed to initialise.") from exc

    # ------------------------------------------------------------------
    # Conversation — delegates to ConversationManager
    # ------------------------------------------------------------------

    def save_turn(
        self, user_msg: str, assistant_msg: str, conversation_id: str
    ) -> tuple[str, str]:
        """Persist turn and enqueue background memory extraction."""
        user_id, asst_id = self._conversation_manager.save_turn(
            user_msg, assistant_msg, conversation_id
        )
        # Fire-and-forget extraction via the queue worker
        self._extraction_queue.put(
            (user_msg, assistant_msg, conversation_id, self._last_turn_id(user_id))
        )
        return user_id, asst_id

    @staticmethod
    def _last_turn_id(record_id: str) -> str:
        """Extract turn_id embedded in record id (best-effort; fallback to new uuid)."""
        # record_id format: {conversation_id}_{uuid}_user — turn_id is stored in Chroma
        # We pass a new uuid here; the worker only uses it for Mem0 metadata.
        return str(uuid.uuid4())

    def build_langchain_history(
        self,
        query: str,
        conversation_id: str,
        n_recent_pairs: Optional[int] = None,
        n_semantic_pairs: Optional[int] = None,
    ):
        return self._conversation_manager.build_langchain_history(
            query=query,
            conversation_id=conversation_id,
            n_recent_pairs=n_recent_pairs,
            n_semantic_pairs=n_semantic_pairs,
        )

    def get_all_conversations(self) -> List[Dict[str, Any]]:
        return self._conversation_manager.get_all_conversations()

    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        return self._conversation_manager.get_conversation_history(conversation_id)

    def clear_conversation(self, conversation_id: str) -> bool:
        return self._conversation_manager.clear_conversation(conversation_id)

    def delete_all_chats(self) -> bool:
        return self._conversation_manager.delete_all_chats()

    # ------------------------------------------------------------------
    # Memories — Mem0
    # ------------------------------------------------------------------

    def add_memory(
        self,
        text: str,
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        meta = {
            "type": "fact",
            "category": category,
            "timestamp": datetime.now().isoformat(),
            **(metadata or {}),
        }
        memory_id = None
        try:
            result = self.mem0_client.add(
                [{"role": "user", "content": text}],
                user_id=self.mem0_user_id,
                metadata=meta,
                agent_id=self.mem0_agent_id,
            )
            if isinstance(result, dict):
                memory_id = (
                    result.get("id")
                    or result.get("memory_id")
                    or (result.get("ids") or [None])[0]
                )
        except Exception as exc:
            logger.warning("Mem0 add_memory failed: %s", exc)
        return memory_id or f"mem0_{uuid.uuid4()}"

    @staticmethod
    def _empty_query_results() -> Dict[str, List[List[Any]]]:
        return {"documents": [[]], "metadatas": [[]], "ids": [[]], "distances": [[]]}

    @staticmethod
    def _normalize_mem0_results(raw: Any) -> Dict[str, List[List[Any]]]:
        items = raw.get("results", raw) if isinstance(raw, dict) else raw
        documents, metadatas, ids, distances = [], [], [], []
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                text = (
                    item.get("memory") or item.get("text") or item.get("content") or ""
                )
                documents.append(str(text))
                metadatas.append(item.get("metadata", {}) or {})
                ids.append(str(item.get("id") or item.get("memory_id") or ""))
                distances.append(
                    item.get("distance") or item.get("score") or item.get("similarity")
                )
        return {
            "documents": [documents],
            "metadatas": [metadatas],
            "ids": [ids],
            "distances": [distances],
        }

    def recall(
        self, query: str, n: int = 3, category: Optional[str] = None
    ) -> Dict[str, Any]:
        filters = {"category": category} if category else None
        try:
            result = self.mem0_client.search(
                query, user_id=self.mem0_user_id, limit=n, filters=filters
            )
        except Exception as exc:
            logger.warning("Mem0 recall failed: %s", exc)
            return self._empty_query_results()
        return self._normalize_mem0_results(result)

    def delete_memory(self, memory_id: str) -> bool:
        try:
            if hasattr(self.mem0_client, "delete"):
                try:
                    self.mem0_client.delete(memory_id=memory_id)
                except TypeError:
                    self.mem0_client.delete(memory_id)
            return True
        except Exception as exc:
            logger.warning("Error deleting memory %s: %s", memory_id, exc)
            return False

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, int]:
        return {
            "memories": len(
                self.memories.get(limit=CHROMA_QUERY_BATCH_SIZE).get("documents", [])
            ),
            "chats": self._conversation_manager.count(),
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"<MemoryManager memories={stats['memories']} chats={stats['chats']}>"

    # ------------------------------------------------------------------
    # Background extraction worker
    # ------------------------------------------------------------------

    def _extraction_worker(self) -> None:
        """Daemon worker that processes memory extraction tasks from the queue."""
        while True:
            task = self._extraction_queue.get()
            try:
                user_msg, assistant_msg, conversation_id, turn_id = task
                if self._should_extract_memory(user_msg):
                    self.mem0_client.add(
                        [
                            {"role": "user", "content": user_msg},
                            {"role": "assistant", "content": assistant_msg},
                        ],
                        user_id=self.mem0_user_id,
                        metadata={
                            "conversation_id": conversation_id,
                            "turn_id": turn_id,
                        },
                        agent_id=self.mem0_agent_id,
                    )
            except Exception as exc:
                logger.warning("Memory extraction failed: %s", exc)
            finally:
                self._extraction_queue.task_done()

    @staticmethod
    def _should_extract_memory(user_msg: str) -> bool:
        """Heuristic gate — only enqueue turns likely to contain personal facts."""
        if not user_msg or not user_msg.strip():
            return False
        text = user_msg.strip().lower()
        if len(text) < 5:
            return False
        if text.endswith("?"):
            return False
        if text.startswith(
            (
                "can you",
                "could you",
                "please ",
                "tell me",
                "find ",
                "search ",
                "send ",
                "email ",
            )
        ):
            return False
        triggers = (
            "remember",
            "i like",
            "i love",
            "i prefer",
            "my favorite",
            "my favourite",
            "i am ",
            "i'm ",
            "my name is",
            "call me ",
            "i live",
            "i work",
            "my email",
            "my phone",
            "my address",
            "my birthday",
            "my birthdate",
            "i have",
            "i don't",
            "i do not",
            "i hate",
            "my diet",
            "my allergies",
            "my timezone",
        )
        return any(t in text for t in triggers)
