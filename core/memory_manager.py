# ABOUTME: Memory manager using ChromaDB for persistent storage of conversations and facts
# ABOUTME: Manages three collections: memories (long-term facts), chats (conversation history), and my_data (documents)

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import chromadb


class MemoryManager:
    """
    Manages persistent memory storage using ChromaDB with three collections:
    - memories: Long-term facts and preferences
    - chat_messages: Conversation history
    - my_data: Indexed documents (emails, files, etc.)
    """

    def __init__(self, persist_dir: str = "./chroma_db"):
        """
        Initialize ChromaDB client and create/load collections.

        Args:
            persist_dir: Directory path for ChromaDB persistence
        """
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Create or load collections
        self.memories = self.client.get_or_create_collection(
            name="memories",
            metadata={"description": "Long-term facts and user preferences"},
        )
        self.chats = self.client.get_or_create_collection(
            name="chat_messages", metadata={"description": "Conversation history"}
        )
        self.my_data = self.client.get_or_create_collection(
            name="my_data",
            metadata={"description": "Indexed user documents and emails"},
        )

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
        memory_id = f"mem_{uuid.uuid4()}"
        meta = {
            "type": "fact",
            "category": category,
            "timestamp": datetime.now().isoformat(),
            **(metadata or {}),
        }

        self.memories.add(documents=[text], metadatas=[meta], ids=[memory_id])

        return memory_id

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
        timestamp = datetime.now().isoformat()
        id_base = f"{session_id}_{int(datetime.now().timestamp())}"

        user_id = f"{id_base}_user"
        asst_id = f"{id_base}_asst"

        self.chats.add(
            documents=[user_msg, assistant_msg],
            metadatas=[
                {"role": "user", "session": session_id, "timestamp": timestamp},
                {"role": "assistant", "session": session_id, "timestamp": timestamp},
            ],
            ids=[user_id, asst_id],
        )

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
            messages.append(
                {
                    "role": results["metadatas"][i]["role"],
                    "content": doc,
                    "timestamp": results["metadatas"][i].get("timestamp", ""),
                    "id": results["ids"][i],
                }
            )

        # Sort by timestamp and limit
        messages.sort(key=lambda x: x["timestamp"])
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
            self.memories.delete(ids=[memory_id])
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
            "memories": self.memories.count(),
            "chats": self.chats.count(),
            "documents": self.my_data.count(),
        }

    def __repr__(self) -> str:
        """String representation with stats."""
        stats = self.get_stats()
        return f"<MemoryManager memories={stats['memories']} chats={stats['chats']} docs={stats['documents']}>"
