"""Production construction for the Runtime V2 daemon."""

from __future__ import annotations

from config import Config
from core.providers.gemini import GeminiProvider
from core.runtime.engine import HeathcliffRuntime
from core.runtime.gemini_memory import GeminiEmbedder, GeminiMemoryClassifier
from core.runtime.legacy_tools import build_legacy_tool_bridge
from core.runtime.memory import MemoryJobWorker, SemanticMemoryPipeline
from db.artifact_store import LocalArtifactStore, S3ArtifactStore
from db.runtime_store import PostgresRuntimeStore, SqliteRuntimeStore


async def create_runtime() -> HeathcliffRuntime:
    """Connect and migrate the canonical store before accepting a request."""
    if Config.RUNTIME_STORAGE_BACKEND == "sqlite":
        store = SqliteRuntimeStore(Config.RUNTIME_SQLITE_PATH)
        artifact_store = LocalArtifactStore(Config.RUNTIME_ARTIFACT_DIRECTORY)
    elif Config.RUNTIME_STORAGE_BACKEND == "postgres":
        if not Config.DATABASE_URL:
            raise ValueError("DATABASE_URL is required for the PostgreSQL runtime")
        if not Config.RUNTIME_S3_BUCKET:
            raise ValueError("RUNTIME_S3_BUCKET is required for the PostgreSQL runtime")
        store = PostgresRuntimeStore(Config.DATABASE_URL)
        artifact_store = S3ArtifactStore(
            Config.RUNTIME_S3_BUCKET, Config.RUNTIME_S3_ENDPOINT
        )
    else:
        raise ValueError("RUNTIME_STORAGE_BACKEND must be 'sqlite' or 'postgres'")
    await store.connect()
    await store.migrate()
    api_key = Config.get_ai_api_key()
    model = Config.SUPERVISOR_MODEL.removeprefix("google_genai:")
    provider = GeminiProvider(
        api_key=api_key,
        model=model,
        context_window=Config.RUNTIME_CONTEXT_WINDOW,
    )
    memory_worker = MemoryJobWorker(
        store,
        SemanticMemoryPipeline(
            store,
            GeminiMemoryClassifier(api_key, model),
            GeminiEmbedder(api_key),
        ),
    )
    return HeathcliffRuntime(
        store=store,
        provider=provider,
        tools=build_legacy_tool_bridge(Config.TOOL_MODEL),
        system_instruction=(
            "You are Heathcliff, a careful personal assistant. Use only typed tools "
            "and never claim an external action succeeded unless its result verifies it."
        ),
        instance_id=Config.RUNTIME_INSTANCE_ID or None,
        max_model_steps=Config.RUNTIME_MAX_MODEL_STEPS,
        lease_seconds=Config.RUNTIME_LEASE_SECONDS,
        memory_worker=memory_worker,
        artifact_store=artifact_store,
    )
