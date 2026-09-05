"""Durable semantic-memory classification without phrase-based gates."""

from __future__ import annotations

from typing import Protocol

from core.runtime.contracts import MemoryCandidate, PersonalMemory, RuntimeEvent
from db.runtime_store import RuntimeStore


class MemoryClassifier(Protocol):
    async def classify(self, source: RuntimeEvent) -> list[MemoryCandidate]: ...


class MemoryEmbedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...


class SemanticMemoryPipeline:
    """Accept only provenance-bearing candidates from a structured classifier."""

    def __init__(
        self,
        store: RuntimeStore,
        classifier: MemoryClassifier,
        embedder: MemoryEmbedder,
    ) -> None:
        self.store = store
        self.classifier = classifier
        self.embedder = embedder

    async def process(self, source: RuntimeEvent) -> list[PersonalMemory]:
        accepted: list[PersonalMemory] = []
        for candidate in await self.classifier.classify(source):
            if (
                candidate.kind == "non_memory"
                or candidate.source_kind == "assistant_inference"
                or candidate.contains_secret
                or candidate.source_event_id != source.id
            ):
                continue
            memory = PersonalMemory(
                **candidate.model_dump(),
                embedding=await self.embedder.embed(candidate.content),
            )
            await self.store.save_memory(memory)
            accepted.append(memory)
        return accepted


class MemoryJobWorker:
    """Claims committed jobs; a crash leaves unclaimed work durable in PostgreSQL."""

    def __init__(self, store: RuntimeStore, pipeline: SemanticMemoryPipeline) -> None:
        self.store = store
        self.pipeline = pipeline

    async def run_once(self) -> bool:
        job = await self.store.claim_memory_job()
        if job is None:
            return False
        source = await self.store.event(job.source_event_id)
        if source is None:
            await self.store.complete_memory_job(job.id, succeeded=False)
            return True
        try:
            await self.pipeline.process(source)
        except Exception:
            await self.store.complete_memory_job(job.id, succeeded=False)
            raise
        await self.store.complete_memory_job(job.id, succeeded=True)
        return True
