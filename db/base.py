# ABOUTME: ChromaConnection singleton and collection name constants for the db layer.
# ABOUTME: All Chroma client lifecycle lives here; consumers call ChromaConnection.get().

from typing import Any, Dict, Optional

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from logger import logger
from utils.errors import AgentMemoryError
from utils.retry import retry

CONVERSATIONS_COLLECTION = "heathcliff_conversations"
MEMORIES_COLLECTION = "heathcliff_memories"


class ChromaConnection:
    """Singleton Chroma client. Call ChromaConnection.initialise() once at startup."""

    _instance: Optional["ChromaConnection"] = None

    def __init__(self, client: ClientAPI) -> None:
        self._client = client

    # ------------------------------------------------------------------
    # Singleton lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def initialise(cls, config: Any) -> "ChromaConnection":
        """Initialise the singleton from Config. Idempotent — returns existing instance."""
        if cls._instance is not None:
            return cls._instance
        client = cls._connect(config)
        cls._instance = cls(client)
        return cls._instance

    @classmethod
    def get(cls) -> "ChromaConnection":
        """Return the singleton. Raises if not yet initialised."""
        if cls._instance is None:
            raise AgentMemoryError(
                "ChromaConnection has not been initialised. "
                "Call ChromaConnection.initialise(config) first."
            )
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Destroy the singleton. Used in tests only."""
        cls._instance = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @staticmethod
    @retry(
        max_retries=3,
        error_message="ChromaDB client initialisation failed",
        exponential_backoff=True,
    )
    def _connect(config: Any) -> ClientAPI:
        use_remote = getattr(config, "USE_REMOTE_CHROMA", False)
        if use_remote:
            host = getattr(config, "CHROMA_HOST", None)
            port = getattr(config, "CHROMA_PORT", None)
            api_key = getattr(config, "CHROMA_API_KEY", None)
            tenant = getattr(config, "CHROMA_TENANT", None)
            db_name = getattr(config, "CHROMA_DATABASE", None)
            if not all([host, port, api_key, tenant, db_name]):
                raise AgentMemoryError(
                    "Remote Chroma requires CHROMA_HOST, CHROMA_PORT, CHROMA_API_KEY, "
                    "CHROMA_TENANT, and CHROMA_DATABASE."
                )
            try:
                client = chromadb.CloudClient(
                    api_key=api_key,
                    tenant=tenant,
                    database=db_name,
                    cloud_host=host,
                )
                logger.info("ChromaConnection: using Cloud client")
                return client
            except Exception as exc:
                raise AgentMemoryError(
                    "ChromaDB Cloud client failed to initialise."
                ) from exc
        else:
            path = getattr(config, "CHROMA_PERSIST_DIRECTORY", "./chroma_db")
            client = chromadb.PersistentClient(path=path)
            logger.info("ChromaConnection: using PersistentClient at %s", path)
            return client

    # ------------------------------------------------------------------
    # Collection helpers
    # ------------------------------------------------------------------

    def get_client(self) -> ClientAPI:
        return self._client

    def get_or_create_collection(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Collection:
        return self._client.get_or_create_collection(name=name, metadata=metadata or {})

    def delete_collection(self, name: str) -> None:
        try:
            self._client.delete_collection(name=name)
        except Exception as exc:
            logger.warning(
                "ChromaConnection: could not delete collection %s: %s", name, exc
            )
