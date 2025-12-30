# ABOUTME: Memory manager using ChromaDB for persistent storage of conversations and facts
# ABOUTME: Manages three collections: memories (long-term facts), chats (conversation history), and my_data (documents)

import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import chromadb
from mem0 import Memory
from logger import logger
from utils.errors import AgentMemoryError
from config import Config


class _Mem0CollectionAdapter:
    """Adapter to provide a Chroma-like get() interface over Mem0 memories."""

    def __init__(self, client: Memory, user_id: str, agent_id: str):
        self._client = client
        self._user_id = user_id
        self._agent_id = agent_id

    def get(self, limit: int = 100) -> Dict[str, List[Any]]:
        if limit > 300:
            limit = 300
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
    _chroma_client: Optional[Any] = None

    def __init__(
        self,
    ):
        """
        Initialize ChromaDB client and create/load collections.

        Args:
            client: Optional pre-configured ChromaDB client
            config: Config instance for runtime settings
        """

        self.client = self._build_client()

        self.mem0_user_id = Config.USER_ID
        self.mem0_agent_id = Config.MEM0_AGENT_ID
        self.mem0_config = Config.CONFIG
        self.mem0_client = self._get_mem0_client()

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

    def _get_mem0_client(self) -> Memory:
        if MemoryManager._mem0_singleton is not None:
            return MemoryManager._mem0_singleton

        try:
            MemoryManager._mem0_singleton = Memory.from_config(self.mem0_config)
            return MemoryManager._mem0_singleton
        except Exception as exc:
            logger.warning(f"Mem0 SDK initialization failed: {exc}")
            raise AgentMemoryError(
                "Memory not found: Mem0 client failed to initialize."
            ) from exc

    def _build_client(self) -> Any:
        """Create a Chroma client (remote if USE_REMOTE else local)."""

        if MemoryManager._chroma_client:
            return MemoryManager._chroma_client

        if not Config.USE_REMOTE_CHROMA:

            persist_dir = Config.CHROMA_PERSIST_DIRECTORY
            if not persist_dir:
                raise AgentMemoryError(
                    "Memory not found: CHROMA_PERSIST_DIRECTORY must be set for local Chroma."
                )

            logger.info("Using ChromaDB persistent client for memory storage")
            return chromadb.PersistentClient(path=persist_dir)

        api_key = Config.CHROMA_API_KEY
        tenant = Config.CHROMA_TENANT
        database = Config.CHROMA_DATABASE
        host = Config.CHROMA_HOST

        if not api_key or not tenant or not database:
            raise AgentMemoryError(
                "Memory not found: remote Chroma requires CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE."
            )

        logger.info("Using ChromaDB Cloud client for memory storage")
        MemoryManager._chroma_client = chromadb.CloudClient(
            api_key=api_key, tenant=tenant, database=database, cloud_host=host
        )
        return MemoryManager._chroma_client

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
            Dictionary containing documents, metadatas, and distances
        """
        filters = {"category": category} if category else None
        try:
            result = self.mem0_client.search(
                query, user_id=self.mem0_user_id, limit=n, filters=filters
            )
        except Exception as exc:
            logger.warning(f"Mem0 recall failed: {exc}")
            return {"documents": [[]], "metadatas": [[]], "ids": [[]]}

        results = result.get("results", result)
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        if isinstance(results, list):
            for item in results:
                if not isinstance(item, dict):
                    continue
                memory_text = (
                    item.get("memory") or item.get("text") or item.get("content")
                )
                if memory_text:
                    documents.append(str(memory_text))
                metadatas.append(item.get("metadata", {}) or {})
                ids.append(str(item.get("id") or item.get("memory_id") or ""))

        return {"documents": [documents], "metadatas": [metadatas], "ids": [ids]}

        where_filter = {"category": category} if category else None
        results = self.memories.query(
            query_texts=[query], n_results=n, where=where_filter
        )
        return results

    def save_chat(
        self, user_msg: str, assistant_msg: str, session_id: str
    ) -> tuple[str, str]:
        """
        Store a conversation turn (user message + assistant response).

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
                },
                {
                    "role": "assistant",
                    "session": session_id,
                    "timestamp": assistant_timestamp,
                    "order": assistant_order,
                },
            ],
            ids=[user_id, asst_id],
        )

        try:
            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
            self.mem0_client.add(
                messages,
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
                    "id": results["ids"][i],
                }
            )

        def _sort_key(msg: Dict[str, Any]):
            order_value = msg.get("order")
            if order_value is not None:
                return (0, order_value)

            timestamp = msg.get("timestamp", "")
            role_priority = 0 if msg.get("role") == "user" else 1
            return (1, timestamp, role_priority)

        # Sort chronologically; fallback keeps legacy entries stable
        messages.sort(key=_sort_key)
        return messages[-n:]

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
                self.mem0_client.delete(memory_id)
            return True
        except Exception as e:
            print(f"Error deleting memory {memory_id}: {e}")
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
            print(f"Error clearing session {session_id}: {e}")
            return False

    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about stored data.

        Returns:
            Dictionary with counts for each collection
        """
        return {
            "memories": len(self.memories.get(limit=300).get("documents", [])),
            "chats": self.chats.count(),
            "documents": self.my_data.count(),
        }

    def __repr__(self) -> str:
        """String representation with stats."""
        stats = self.get_stats()
        return f"<MemoryManager memories={stats['memories']} chats={stats['chats']} docs={stats['documents']}>"
