"""Durable semantic-memory classification without phrase-based gates."""

from __future__ import annotations

import asyncio
from typing import Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field

from core.runtime.contracts import EventKind, MemoryCandidate, PersonalMemory, RuntimeEvent, utcnow
from core.runtime.recall import RecallService, history_source
from db.recall_store import RecallSource
from db.runtime_store import RuntimeStore


class MemoryClassifier(Protocol):
    async def classify(self, source: RuntimeEvent) -> list[MemoryCandidate]: ...


class MemoryEmbedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...


class EvidenceMemoryCandidate(MemoryCandidate):
    """Extraction evidence lives here to keep shared runtime contracts stable."""

    evidence_quote: str = Field(default="", max_length=2000)
    supersedes_id: UUID | None = None


class SemanticMemoryPipeline:
    """Accept only provenance-bearing candidates from a structured classifier."""

    def __init__(
        self,
        store: RuntimeStore,
        classifier: MemoryClassifier,
        embedder: MemoryEmbedder,
        *,
        recall: RecallService | None = None,
        scope: str = "default",
    ) -> None:
        self.store = store
        self.classifier = classifier
        self.embedder = embedder
        self.recall = recall
        self.scope = scope

    async def process(self, source: RuntimeEvent) -> list[PersonalMemory]:
        if self.recall is not None:
            return await self._process_durable(source)
        # Legacy constructor remains compatible; daemon wiring selects durable recall.
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

    async def _process_durable(self, source: RuntimeEvent) -> list[PersonalMemory]:
        assert self.recall is not None
        repository = self.recall.repository
        committed = await self.store.event(source.id)
        if committed is None or await repository.extracted(source.id):
            return []
        source = committed  # Never trust a caller's replacement payload/provenance.
        if source.kind != EventKind.INPUT_ADMITTED:
            return []
        text = source.payload.get("content")
        if (not isinstance(text, str) or not text.strip()
                or source.payload.get("contains_secret") or source.payload.get("privacy") == "secret"):
            return []
        bounded = source.model_copy(update={"payload": {"content": text[:8000]}})
        async with asyncio.timeout(30):
            candidates = (await self.classifier.classify(bounded))[:8]
        records = []
        history = history_source(bounded, self.scope)
        # A semantic secret classification also prevents indexing the raw source.
        if history is not None and not any(c.contains_secret for c in candidates):
            records.append(history)
        now = utcnow()
        for candidate in candidates:
            quote = getattr(candidate, "evidence_quote", "")
            target = getattr(candidate, "supersedes_id", None)
            if (candidate.kind == "non_memory" or candidate.contains_secret
                    or candidate.source_kind != "user" or candidate.source_event_id != source.id
                    or candidate.confidence < 0.8 or not quote.strip() or quote not in text[:8000]
                    or not candidate.content.strip() or len(candidate.content) > 2000
                    or not candidate.subject.strip() or len(candidate.subject) > 200
                    or (candidate.valid_until is not None and candidate.valid_until <= now)
                    or (candidate.valid_from is not None and candidate.valid_until is not None
                        and candidate.valid_from >= candidate.valid_until)):
                continue
            if candidate.kind == "correction":
                old = await repository.get(target) if target else None
                if (old is None or old.kind != "fact" or old.scope != self.scope
                        or not old.valid(now) or old.subject != candidate.subject):
                    continue
            elif target is not None:
                continue
            record = RecallSource(
                id=uuid5(NAMESPACE_URL, f"heathcliff:fact:{self.scope}:{source.id}:{candidate.kind}:{candidate.subject.casefold().strip()}"),
                scope=self.scope, kind="fact", source_event_id=source.id,
                thread_id=source.thread_id, turn_id=source.turn_id,
                content=candidate.content, subject=candidate.subject,
                memory_kind=candidate.kind, confidence=candidate.confidence,
                evidence_quote=quote, supersedes_id=target,
                valid_from=candidate.valid_from, valid_until=candidate.valid_until,
                created_at=source.created_at,
            )
            records.append(record)
        saved = await repository.accept(records, source.id)
        return [PersonalMemory(
            id=s.id, kind=s.memory_kind, subject=s.subject, content=s.content,
            confidence=s.confidence, source_event_id=s.source_event_id,
            source_kind=s.source_kind, valid_from=s.valid_from, valid_until=s.valid_until,
            supersedes_id=s.supersedes_id, created_at=s.created_at,
        ) for s in saved if s.kind == "fact"]

    async def process_turn(self, sources: list[RuntimeEvent]) -> list[PersonalMemory]:
        """Bounded committed turn evidence; no assistant inference becomes a fact."""
        accepted = []
        for source in sources[:8]:
            accepted.extend(await self.process(source))
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
