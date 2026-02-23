# ABOUTME: Memory manager using ChromaDB for persistent storage of conversations and facts
# ABOUTME: Manages three collections: memories (long-term facts), chats (conversation history), and my_data (documents)

import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import chromadb
from chromadb.api import ClientAPI
from mem0 import Memory

from config import Config
from logger import logger
from utils.errors import AgentMemoryError
from utils.retry import retry

CHROMA_CLIENT = Union[ClientAPI, None]
CHROMA_QUERY_BATCH_SIZE = 200

_global_chroma_client: CHROMA_CLIENT = None


@retry(
    max_retries=3,
    error_message="ChromaDB client initialization failed",
    exponential_backoff=True,
)
def chroma_client(
    host: str, port: int, api_key: str, tenant: str, db_name: str
) -> None:
    """Initialize global ChromaDB client."""

    global _global_chroma_client

    if _global_chroma_client is not None:
        return

    if not (host and port and api_key and tenant and db_name):
        raise AgentMemoryError(
            "Memory not found: remote Chroma requires CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE.",
            exec,
        )

    try:
        if Config.USE_REMOTE_CHROMA:
            _global_chroma_client = chromadb.CloudClient(
                api_key=api_key, tenant=tenant, database=db_name, cloud_host=host
            )
            logger.info("Using ChromaDB Cloud client for memory storage")
        else:
            _global_chroma_client = chromadb.PersistentClient(
                path=Config.CHROMA_PERSIST_DIRECTORY
            )
    except Exception as exc:
        logger.warning(f"ChromaDB Cloud client initialization failed: {exc}")
        raise AgentMemoryError(
            "Memory not found: ChromaDB Cloud client failed to initialize.",
            exec,
        ) from exc


class _Mem0CollectionAdapter:
    """Adapter to provide a Chroma-like get() interface over Mem0 memories."""

    def __init__(self, client: Memory, user_id: str, agent_id: str):
        self._client = client
        self._user_id = user_id
        self._agent_id = agent_id

    def get(self, limit: int = 100) -> Dict[str, List[Any]]:
        if limit > CHROMA_QUERY_BATCH_SIZE:
            logger.warning("Mem0 get() limit capped at %s", CHROMA_QUERY_BATCH_SIZE)
            limit = CHROMA_QUERY_BATCH_SIZE
        try:
            result = self._client.get_all(
                user_id=self._user_id, agent_id=self._agent_id, limit=limit
            )
        except Exception as exc:
            logger.warning(f"Mem0 get_all failed: {exc}")
            return {"documents": [], "metadatas": [], "ids": []}

        items = result.get("results", result) if isinstance(result, dict) else result
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                memory_text = (
                    item.get("memory") or item.get("text") or item.get("content") or ""
                )
                documents.append(str(memory_text))

                metadata = dict(item.get("metadata", {}) or {})
                for key in ("category", "type", "timestamp", "user_id", "agent_id"):
                    if key in item and key not in metadata:
                        metadata[key] = item.get(key)
                metadatas.append(metadata)

                ids.append(str(item.get("id") or item.get("memory_id") or ""))

        return {"documents": documents, "metadatas": metadatas, "ids": ids}


class MemoryManager:
    """
    Manages persistent memory storage.
    - Mem0 handles long-term memories (Gemini + Chroma Cloud).
    - Chroma stores chat_messages and my_data collections.
    """

    _mem0_singleton: Optional[Memory] = None

    def __init__(self) -> None:
        """Initialize ChromaDB client and create/load collections."""

        chroma_client(
            host=Config.CHROMA_HOST,
            port=Config.CHROMA_PORT,
            api_key=Config.CHROMA_API_KEY,
            tenant=Config.CHROMA_TENANT,
            db_name=Config.CHROMA_DATABASE,
        )

        self.client: CHROMA_CLIENT = _global_chroma_client

        self.mem0_user_id = Config.USER_ID
        self.mem0_agent_id = Config.MEM0_AGENT_ID
        self.mem0_config = Config.MEM0_CONFIG
        self.mem0_client = self._get_mem0_client(self.client)

        if not self.mem0_client:
            raise AgentMemoryError(
                "Memory not found: Mem0 client failed to initialize."
            )

        # Create or load collections (non-memory data only)
        self.memories = _Mem0CollectionAdapter(
            self.mem0_client, self.mem0_user_id, self.mem0_agent_id
        )
        self.chats = self.client.get_or_create_collection(
            name="chat_messages", metadata={"description": "Conversation history"}
        )
        self.my_data = self.client.get_or_create_collection(
            name="my_data",
            metadata={"description": "Indexed user documents and emails"},
        )

    def _get_mem0_client(self, client: CHROMA_CLIENT) -> Memory:
        """Initialize or return singleton Mem0 client."""

        # TODO: Improve client initialization (consolidate from `chroma_client`), one func that returns chroma client and mem0 client.
        # This function should also handle both remote and local chroma clients.
        # This should also handle singleton pattern for both clients.

        if not _global_chroma_client:
            raise AgentMemoryError(
                "Memory not found: ChromaDB client is not initialized."
            )

        from copy import deepcopy

        config = deepcopy(Config.MEM0_CONFIG)

        try:
            config["vector_store"]["config"] = {
                "collection_name": Config.MEMORY_COLLECTION,
                "client": client or _global_chroma_client,
                "host": Config.CHROMA_HOST,
                "port": Config.CHROMA_PORT,
                "path": Config.CHROMA_PERSIST_DIRECTORY,
            }

            MemoryManager._mem0_singleton = Memory.from_config(config)
            logger.info("Mem0 client initialized for memory management.")

            return MemoryManager._mem0_singleton
        except Exception as exc:
            logger.warning(f"Mem0 SDK initialization failed: {exc}")
            raise AgentMemoryError(
                "Memory not found: Mem0 client failed to initialize."
            ) from exc

    def add_memory(
        self,
        text: str,
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a long-term fact or preference.

        Args:
            text: The memory content
            category: Category of memory (e.g., 'preference', 'fact', 'reminder')
            metadata: Additional metadata to store

        Returns:
            ID of the stored memory
        """
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
            logger.warning(f"Mem0 add_memory failed: {exc}")

        return memory_id or f"mem0_{uuid.uuid4()}"

    @staticmethod
    def _empty_query_results() -> Dict[str, List[List[Any]]]:
        return {"documents": [[]], "metadatas": [[]], "ids": [[]], "distances": [[]]}

    @staticmethod
    def _normalize_mem0_results(raw_results: Any) -> Dict[str, List[List[Any]]]:
        items = (
            raw_results.get("results", raw_results)
            if isinstance(raw_results, dict)
            else raw_results
        )
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []
        distances: List[Optional[float]] = []

        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                memory_text = (
                    item.get("memory") or item.get("text") or item.get("content") or ""
                )
                documents.append(str(memory_text))
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
        """
        Search for relevant memories.

        Args:
            query: Search query
            n: Number of results to return
            category: Optional category filter

        Returns:
            Dictionary containing documents, metadatas, ids, and distances
        """
        filters = {"category": category} if category else None
        try:
            result = self.mem0_client.search(
                query, user_id=self.mem0_user_id, limit=n, filters=filters
            )
        except Exception as exc:
            logger.warning(f"Mem0 recall failed: {exc}")
            return self._empty_query_results()

        return self._normalize_mem0_results(result)

    def save_chat(
        self, user_msg: str, assistant_msg: str, session_id: str
    ) -> tuple[str, str]:
        """
        Store a conversation turn (user message + assistant response).

        Both messages share a ``turn_id`` so they can be reliably paired
        when reconstructing history (semantic or chronological).

        Args:
            user_msg: User's message
            assistant_msg: Assistant's response
            session_id: Session identifier

        Returns:
            Tuple of (user_msg_id, assistant_msg_id)
        """
        now = datetime.now()
        user_timestamp = now.isoformat()
        assistant_time = now + timedelta(microseconds=1)
        assistant_timestamp = assistant_time.isoformat()

        # Monotonic order index keeps chat history deterministic even if storage order varies
        user_order = now.timestamp()
        assistant_order = assistant_time.timestamp()

        turn_id = str(uuid.uuid4())
        user_id = f"{session_id}_{uuid.uuid4()}_user"
        asst_id = f"{session_id}_{uuid.uuid4()}_assistant"

        self.chats.add(
            documents=[user_msg, assistant_msg],
            metadatas=[
                {
                    "role": "user",
                    "session": session_id,
                    "timestamp": user_timestamp,
                    "order": user_order,
                    "turn_id": turn_id,
                },
                {
                    "role": "assistant",
                    "session": session_id,
                    "timestamp": assistant_timestamp,
                    "order": assistant_order,
                    "turn_id": turn_id,
                },
            ],
            ids=[user_id, asst_id],
        )

        try:
            if self._should_extract_memory(user_msg):
                # Only send user content for extraction to avoid assistant acknowledgements
                self.mem0_client.add(
                    [{"role": "user", "content": user_msg}],
                    user_id=self.mem0_user_id,
                    metadata={"session_id": session_id},
                    agent_id=self.mem0_agent_id,
                )
        except Exception as exc:
            logger.warning(f"Mem0 memory extraction failed: {exc}")

        return user_id, asst_id

    def get_chat_context(
        self, query: str, n: int = 5, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieve relevant chat history for context.

        Args:
            query: Search query to find relevant context
            n: Number of messages to retrieve
            session_id: Optional session filter

        Returns:
            Dictionary containing documents, metadatas, and distances
        """
        where_filter = {"session": session_id} if session_id else None

        results = self.chats.query(query_texts=[query], n_results=n, where=where_filter)

        return results

    def get_recent_chats(self, session_id: str, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent chat messages from a session chronologically.

        Args:
            session_id: Session identifier
            n: Maximum number of messages to retrieve

        Returns:
            List of chat messages with role, content, and timestamp
        """
        # Note: ChromaDB doesn't support ordering by metadata, so we retrieve more
        # than needed and sort manually
        results = self.chats.get(
            where={"session": session_id},
            limit=n * 2,  # Get extra to ensure we have enough after filtering
        )

        if not results or not results.get("documents"):
            return []

        # Combine results into structured format
        messages = []
        for i, doc in enumerate(results["documents"]):
            metadata = results["metadatas"][i]
            messages.append(
                {
                    "role": metadata.get("role", "unknown"),
                    "content": doc,
                    "timestamp": metadata.get("timestamp", ""),
                    "order": metadata.get("order"),
                    "turn_id": metadata.get("turn_id", ""),
                    "id": results["ids"][i],
                }
            )

        # Sort chronologically; fallback keeps legacy entries stable
        messages.sort(key=self._sort_key)
        return messages[-n:]

    # ------------------------------------------------------------------
    # Pair-aware history retrieval
    # ------------------------------------------------------------------

    @staticmethod
    def _sort_key(msg: Dict[str, Any]):
        """Sort key for chronological ordering of chat messages."""
        order_value = msg.get("order")
        if order_value is not None:
            return (0, order_value)
        timestamp = msg.get("timestamp", "")
        role_priority = 0 if msg.get("role") == "user" else 1
        return (1, timestamp, role_priority)

    def _fetch_partner_by_turn_id(
        self, turn_id: str, exclude_id: str
    ) -> Optional[Dict[str, Any]]:
        """Look up the other message in a pair using the shared turn_id."""
        try:
            partner_results = self.chats.get(
                where={"turn_id": turn_id},
                limit=2,
            )
        except Exception:
            return None
        if not partner_results or not partner_results.get("documents"):
            return None
        for i, doc in enumerate(partner_results["documents"]):
            pid = partner_results["ids"][i]
            if pid != exclude_id:
                meta = partner_results["metadatas"][i]
                return {
                    "role": meta.get("role", "unknown"),
                    "content": doc,
                    "timestamp": meta.get("timestamp", ""),
                    "order": meta.get("order"),
                    "turn_id": meta.get("turn_id", ""),
                    "id": pid,
                }
        return None

    def _fetch_partner_by_adjacency(
        self, msg: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Fallback: find the partner of a message using order-based adjacency.

        For legacy rows that don't carry a ``turn_id``, we look for the
        nearest message in the same session with the opposite role.
        """
        session = msg.get("session") or msg.get("metadata", {}).get("session")
        if not session:
            return None
        try:
            results = self.chats.get(
                where={"session": session},
                limit=200,
            )
        except Exception:
            return None
        if not results or not results.get("documents"):
            return None

        all_msgs: List[Dict[str, Any]] = []
        for i, doc in enumerate(results["documents"]):
            meta = results["metadatas"][i]
            all_msgs.append(
                {
                    "role": meta.get("role", "unknown"),
                    "content": doc,
                    "timestamp": meta.get("timestamp", ""),
                    "order": meta.get("order"),
                    "turn_id": meta.get("turn_id", ""),
                    "id": results["ids"][i],
                    "session": meta.get("session", ""),
                }
            )
        all_msgs.sort(key=self._sort_key)

        target_role = "assistant" if msg.get("role") == "user" else "user"
        msg_id = msg.get("id")

        # Find position of our message
        idx = None
        for i, m in enumerate(all_msgs):
            if m["id"] == msg_id:
                idx = i
                break
        if idx is None:
            return None

        # Look for nearest neighbor with opposite role (prefer after, then before)
        if msg.get("role") == "user":
            # Partner is the next assistant message
            if idx + 1 < len(all_msgs) and all_msgs[idx + 1]["role"] == target_role:
                return all_msgs[idx + 1]
        else:
            # Partner is the previous user message
            if idx - 1 >= 0 and all_msgs[idx - 1]["role"] == target_role:
                return all_msgs[idx - 1]
        return None

    def _resolve_pair(self, msg: Dict[str, Any]) -> Optional[tuple]:
        """Return a (user_msg, assistant_msg) pair for the given message.

        Tries ``turn_id`` first, falls back to adjacency for legacy data.
        Returns None if the partner cannot be found.
        """
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
        else:
            return (partner, msg)

    def get_recent_pairs(self, session_id: str, n_pairs: int = 6) -> List[tuple]:
        """
        Get the last *n_pairs* conversation pairs from the current session,
        in chronological order.  Each pair is ``(user_msg, assistant_msg)``.

        Args:
            session_id: Active session identifier.
            n_pairs: Number of pairs to return.

        Returns:
            List of (user_dict, assistant_dict) tuples ordered chronologically.
        """
        messages = self.get_recent_chats(session_id, n=n_pairs * 2)

        # Group into pairs via turn_id when available, else adjacency
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

        # Sort pairs chronologically by user message order
        pairs.sort(key=lambda p: self._sort_key(p[0]))
        return pairs[-n_pairs:]

    def get_semantic_pairs(self, query: str, n_pairs: int = 3) -> List[tuple]:
        """
        Search *all* chat history semantically and return matching pairs.

        Queries both user and assistant messages against the user's query,
        then resolves each hit to its full (user, assistant) pair.

        Args:
            query: The current user query for semantic similarity search.
            n_pairs: Maximum number of pairs to return.

        Returns:
            List of (user_dict, assistant_dict) tuples, ordered by the
            original chronological order of each pair.
        """
        if not query or not query.strip():
            return []

        # Fetch more candidates than needed so we can de-duplicate after pairing
        n_candidates = n_pairs * 4
        try:
            results = self.chats.query(
                query_texts=[query],
                n_results=n_candidates,
            )
        except Exception as exc:
            logger.warning("Semantic chat search failed: %s", exc)
            return []

        if not results or not results.get("documents") or not results["documents"][0]:
            return []

        seen_turn_ids: set = set()
        pairs: List[tuple] = []

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        ids = results["ids"][0]

        for i, doc in enumerate(docs):
            meta = metas[i]
            msg = {
                "role": meta.get("role", "unknown"),
                "content": doc,
                "timestamp": meta.get("timestamp", ""),
                "order": meta.get("order"),
                "turn_id": meta.get("turn_id", ""),
                "id": ids[i],
                "session": meta.get("session", ""),
            }

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

        # Sort pairs by chronological order of the user message
        pairs.sort(key=lambda p: self._sort_key(p[0]))
        return pairs

    def build_message_history(
        self,
        query: str,
        session_id: str,
        n_recent_pairs: Optional[int] = None,
        n_semantic_pairs: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build the full message history list for the agent.

        Structure (in order):
          1. Semantic history pairs (historical context from all sessions)
          2. Recent chronological pairs (conversation continuity from current session)

        The caller should append the current ``HumanMessage`` after this list
        so the user's latest query is always the final message.

        Duplicate pairs that appear in both semantic and recent sets are
        kept only in the recent (chronological) section.

        Args:
            query: Current user query (used for semantic search).
            session_id: Active session identifier.
            n_recent_pairs: Override for Config.RECENT_PAIRS_COUNT.
            n_semantic_pairs: Override for Config.SEMANTIC_PAIRS_COUNT.

        Returns:
            Flat list of message dicts ``{"role": ..., "content": ...}``
            ready to be converted to LangChain message objects.
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

        recent_pairs = self.get_recent_pairs(session_id, n_pairs=n_recent)
        semantic_pairs = self.get_semantic_pairs(query, n_pairs=n_semantic)

        # Collect turn_ids of recent pairs for dedup
        recent_turn_ids: set = set()
        for user_msg, asst_msg in recent_pairs:
            tid = user_msg.get("turn_id") or asst_msg.get("turn_id") or ""
            if tid:
                recent_turn_ids.add(tid)
            # Also deduplicate by content+role if no turn_id
            recent_turn_ids.add(
                (user_msg.get("content", ""), asst_msg.get("content", ""))
            )

        # Filter semantic pairs: remove any that are already in recent
        filtered_semantic: List[tuple] = []
        for user_msg, asst_msg in semantic_pairs:
            tid = user_msg.get("turn_id") or asst_msg.get("turn_id") or ""
            content_key = (user_msg.get("content", ""), asst_msg.get("content", ""))
            if tid and tid in recent_turn_ids:
                continue
            if content_key in recent_turn_ids:
                continue
            filtered_semantic.append((user_msg, asst_msg))

        # Build flat message list: semantic first, then chronological
        history: List[Dict[str, Any]] = []

        for user_msg, asst_msg in filtered_semantic:
            history.append({"role": "user", "content": user_msg.get("content", "")})
            history.append(
                {"role": "assistant", "content": asst_msg.get("content", "")}
            )

        for user_msg, asst_msg in recent_pairs:
            history.append({"role": "user", "content": user_msg.get("content", "")})
            history.append(
                {"role": "assistant", "content": asst_msg.get("content", "")}
            )

        return history

    def index_document(
        self,
        content: str,
        source: str,
        doc_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Index a document (email, file, etc.) for later retrieval.

        Args:
            content: Document content
            source: Source identifier (e.g., email ID, file path)
            doc_type: Type of document (e.g., 'email', 'gdrive', 'note')
            metadata: Additional metadata

        Returns:
            ID of the indexed document
        """
        doc_id = f"{doc_type}_{source}_{uuid.uuid4()}"
        meta = {
            "source": source,
            "type": doc_type,
            "timestamp": datetime.now().isoformat(),
            **(metadata or {}),
        }

        self.my_data.add(documents=[content], metadatas=[meta], ids=[doc_id])

        return doc_id

    def search_my_data(
        self, query: str, doc_type: Optional[str] = None, n: int = 3
    ) -> Dict[str, Any]:
        """
        Search indexed documents.

        Args:
            query: Search query
            doc_type: Optional document type filter
            n: Number of results to return

        Returns:
            Dictionary containing documents, metadatas, and distances
        """
        where_filter = {"type": doc_type} if doc_type else None

        results = self.my_data.query(
            query_texts=[query], n_results=n, where=where_filter
        )

        return results

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a specific memory.

        Args:
            memory_id: ID of the memory to delete

        Returns:
            True if successful
        """
        try:
            if hasattr(self.mem0_client, "delete"):
                try:
                    self.mem0_client.delete(memory_id=memory_id)
                except TypeError:
                    self.mem0_client.delete(memory_id)
            return True
        except Exception as e:
            logger.warning(f"Error deleting memory {memory_id}: {e}")
            return False

    def clear_session(self, session_id: str) -> bool:
        """
        Clear all chat messages from a session.

        Args:
            session_id: Session identifier

        Returns:
            True if successful
        """
        try:
            self.chats.delete(where={"session": session_id})
            return True
        except Exception as e:
            logger.warning(f"Error clearing session {session_id}: {e}")
            return False

    def _iter_chat_pages(self, where: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Fetch chat collection rows in paginated batches."""
        pages: List[Any] = []
        offset = 0

        while True:
            if where is None:
                results = self.chats.get(limit=CHROMA_QUERY_BATCH_SIZE, offset=offset)
            else:
                results = self.chats.get(
                    where=where,
                    limit=CHROMA_QUERY_BATCH_SIZE,
                    offset=offset,
                )

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

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Return session summaries for the chat history UI."""
        try:
            pages = self._iter_chat_pages()
        except Exception as e:
            logger.warning(f"Error retrieving sessions: {e}")
            return []

        sessions: Dict[str, Dict[str, Any]] = {}

        for page in pages:
            metadatas = (page or {}).get("metadatas", []) or []
            for metadata in metadatas:
                if not isinstance(metadata, dict):
                    continue

                session_id = str(metadata.get("session") or "").strip()
                if not session_id:
                    continue

                timestamp = str(metadata.get("timestamp") or "")

                if session_id not in sessions:
                    sessions[session_id] = {
                        "session_id": session_id,
                        "start_time": timestamp,
                        "msg_count": 1,
                    }
                    continue

                sessions[session_id]["msg_count"] += 1
                existing_start = sessions[session_id].get("start_time", "")
                if timestamp and (not existing_start or timestamp < existing_start):
                    sessions[session_id]["start_time"] = timestamp

        return list(sessions.values())

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Return all messages from a session in chronological order."""
        try:
            pages = self._iter_chat_pages(where={"session": session_id})
        except Exception as e:
            logger.warning(f"Error retrieving session history for {session_id}: {e}")
            return []

        messages: List[Dict[str, Any]] = []

        for page in pages:
            documents = page.get("documents", []) or []
            metadatas = page.get("metadatas", []) or []
            ids = page.get("ids", []) or []

            for i, doc in enumerate(documents):
                metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
                message_id = ids[i] if i < len(ids) else ""

                messages.append(
                    {
                        "role": metadata.get("role", "unknown"),
                        "content": doc,
                        "timestamp": metadata.get("timestamp", ""),
                        "order": metadata.get("order"),
                        "turn_id": metadata.get("turn_id", ""),
                        "id": message_id,
                    }
                )

        if not messages:
            return []

        messages.sort(key=self._sort_key)
        return messages

    def delete_all_chats(self) -> bool:
        """Delete all chat history entries."""
        try:
            previous_batch_ids: List[str] = []
            while True:
                results = self.chats.get(limit=CHROMA_QUERY_BATCH_SIZE, offset=0)
                chat_ids = (results or {}).get("ids", []) or []

                if not chat_ids:
                    return True

                if chat_ids == previous_batch_ids:
                    logger.warning(
                        "Delete all chats made no progress; stopping to avoid infinite loop"
                    )
                    return False

                self.chats.delete(ids=chat_ids)
                previous_batch_ids = chat_ids

            return True
        except Exception as e:
            logger.warning(f"Error deleting all chats: {e}")
            return False

    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about stored data.

        Returns:
            Dictionary with counts for each collection
        """
        return {
            "memories": len(
                self.memories.get(limit=CHROMA_QUERY_BATCH_SIZE).get("documents", [])
            ),
            "chats": self.chats.count(),
            "documents": self.my_data.count(),
        }

    def __repr__(self) -> str:
        """String representation with stats."""
        stats = self.get_stats()
        return f"<MemoryManager memories={stats['memories']} chats={stats['chats']} docs={stats['documents']}>"

    @staticmethod
    def _should_extract_memory(user_msg: str) -> bool:
        """Heuristic gate to reduce noisy memories from questions or commands."""
        if not user_msg or not user_msg.strip():
            return False

        text = user_msg.strip().lower()
        if len(text) < 5:
            return False

        # Skip obvious questions/commands to avoid tool-related noise.
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
        return any(trigger in text for trigger in triggers)
