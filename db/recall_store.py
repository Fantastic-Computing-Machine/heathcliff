"""Authoritative recall records and a transactional, leased index outbox."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from core.runtime.contracts import utcnow


class RecallSource(BaseModel):
    id: UUID
    scope: str = Field(default="default", min_length=1, max_length=200)
    kind: Literal["history", "fact"]
    source_event_id: UUID
    thread_id: UUID
    turn_id: UUID | None = None
    content: str = Field(min_length=1, max_length=8000)
    revision: int = Field(default=1, ge=1)
    subject: str = Field(default="", max_length=200)
    memory_kind: str = ""
    source_kind: str = "user"
    evidence_quote: str = Field(default="", max_length=2000)
    confidence: float = Field(default=1, ge=0, le=1)
    supersedes_id: UUID | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    deleted_at: datetime | None = None
    created_at: datetime = Field(default_factory=utcnow)

    def valid(self, now: datetime) -> bool:
        return (
            self.deleted_at is None
            and (self.valid_from is None or self.valid_from <= now)
            and (self.valid_until is None or self.valid_until > now)
        )


class IndexJob(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    source_id: UUID
    revision: int
    action: Literal["upsert", "delete"]
    attempts: int = 0
    token: UUID | None = None


class PostgresRecallStore:
    """Borrow an asyncpg pool. All source mutations include their outbox writes."""

    def __init__(self, pool):
        self.pool = pool

    @staticmethod
    def _source(row) -> RecallSource | None:
        if row is None:
            return None
        payload = row["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return RecallSource.model_validate({**payload, "revision": row["revision"],
                                            "deleted_at": row["deleted_at"]})

    async def _enqueue(self, conn, source: RecallSource, action: str) -> None:
        await conn.execute("""
            INSERT INTO runtime_recall_jobs (id, source_id, revision, action)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (source_id, revision, action) DO UPDATE
            SET status = 'pending', available_at = now(), claim_token = NULL
            WHERE runtime_recall_jobs.status = 'completed'
        """, uuid4(), source.id, source.revision, action)

    async def _version(self, conn, source: RecallSource) -> None:
        await conn.execute("""
            UPDATE runtime_recall_sources SET revision=$2, payload=$3::jsonb,
                valid_until=$4, deleted_at=$5 WHERE id=$1
        """, source.id, source.revision, source.model_dump_json(), source.valid_until, source.deleted_at)
        await conn.execute("""
            INSERT INTO runtime_recall_revisions (source_id, revision, payload)
            VALUES ($1, $2, $3::jsonb)
        """, source.id, source.revision, source.model_dump_json())

    async def _verify_provenance(self, conn, source: RecallSource) -> None:
        row = await conn.fetchrow("SELECT thread_id, turn_id, kind, payload FROM runtime_events WHERE id=$1", source.source_event_id)
        if row is None or row["thread_id"] != source.thread_id or row["turn_id"] != source.turn_id:
            raise ValueError("Recall source must reference its committed event")
        payload = json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"]
        if payload.get("contains_secret") or payload.get("privacy") == "secret":
            raise ValueError("Secret events cannot be indexed")
        text = payload.get("content") or payload.get("text") or ""
        if source.kind == "history":
            if row["kind"] not in {"input.admitted", "turn.completed"} or source.content != text[:8000]:
                raise ValueError("History must project committed text")
        elif (row["kind"] != "input.admitted" or source.source_kind != "user"
              or not source.evidence_quote.strip() or source.evidence_quote not in text[:8000]):
            raise ValueError("Facts require explicit committed user evidence")

    async def _save(self, conn, source: RecallSource, expected_revision: int | None = None):
        source = RecallSource.model_validate(source.model_dump())
        old = self._source(await conn.fetchrow("SELECT * FROM runtime_recall_sources WHERE id=$1 FOR UPDATE", source.id))
        if old is None:
            if expected_revision is not None:
                raise ValueError("Recall source revision conflict")
            await self._verify_provenance(conn, source)
            source = source.model_copy(update={"revision": 1})
            inserted = await conn.fetchval("""
                INSERT INTO runtime_recall_sources
                    (id, scope, kind, source_event_id, revision, payload, valid_until, deleted_at)
                VALUES ($1,$2,$3,$4,1,$5::jsonb,$6,$7) ON CONFLICT DO NOTHING RETURNING id
            """, source.id, source.scope, source.kind, source.source_event_id,
                source.model_dump_json(), source.valid_until, source.deleted_at)
            if inserted is not None:
                await self._version(conn, source)
                await self._enqueue(conn, source, "upsert")
                return source
            old = self._source(await conn.fetchrow("SELECT * FROM runtime_recall_sources WHERE id=$1 FOR UPDATE", source.id))
        if old.deleted_at is not None:
            return None
        if expected_revision is None:
            return old
        if old.revision != expected_revision:
            raise ValueError("Recall source revision conflict")
        if (old.scope, old.kind, old.source_event_id) != (source.scope, source.kind, source.source_event_id):
            raise ValueError("Recall provenance is immutable")
        await self._verify_provenance(conn, source)
        await self._enqueue(conn, old, "delete")
        source = source.model_copy(update={"revision": old.revision + 1})
        await self._version(conn, source)
        await self._enqueue(conn, source, "upsert")
        return source

    async def save(self, source: RecallSource, *, expected_revision: int | None = None):
        async with self.pool.acquire() as conn, conn.transaction():
            return await self._save(conn, source, expected_revision)

    async def extracted(self, event_id: UUID) -> bool:
        async with self.pool.acquire() as conn:
            return bool(await conn.fetchval("SELECT 1 FROM runtime_recall_extractions WHERE source_event_id=$1", event_id))

    async def accept(self, sources: list[RecallSource], event_id: UUID) -> list[RecallSource]:
        async with self.pool.acquire() as conn, conn.transaction():
            claimed = await conn.fetchval("""
                INSERT INTO runtime_recall_extractions (source_event_id) VALUES ($1)
                ON CONFLICT DO NOTHING RETURNING source_event_id
            """, event_id)
            if claimed is None:
                return []
            targets = sorted({s.supersedes_id for s in sources if s.supersedes_id})
            rows = await conn.fetch("SELECT * FROM runtime_recall_sources WHERE id=ANY($1::uuid[]) ORDER BY id FOR UPDATE", targets)
            prior = {row["id"]: self._source(row) for row in rows}
            for source in sources:
                if source.source_event_id != event_id:
                    raise ValueError("Extraction provenance mismatch")
                if source.supersedes_id:
                    old = prior.get(source.supersedes_id)
                    if old is None or old.kind != "fact" or old.scope != source.scope or not old.valid(utcnow()):
                        raise ValueError("Correction target is not an active fact")
            accepted = []
            for source in sources:
                saved = await self._save(conn, source)
                if saved is not None:
                    accepted.append(saved)
                    if source.supersedes_id:
                        await self._delete(conn, source.supersedes_id)
            return accepted

    async def get(self, source_id: UUID) -> RecallSource | None:
        return (await self.get_many([source_id])).get(source_id)

    async def get_many(self, ids: list[UUID]) -> dict[UUID, RecallSource]:
        if not ids:
            return {}
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT s.* FROM runtime_recall_sources s JOIN runtime_events e ON e.id=s.source_event_id
                WHERE s.id=ANY($1::uuid[]) AND s.payload->>'thread_id'=e.thread_id::text
                    AND (s.payload->>'turn_id') IS NOT DISTINCT FROM e.turn_id::text
            """, ids)
        return {row["id"]: self._source(row) for row in rows}

    async def _delete(self, conn, source_id: UUID) -> bool:
        source = self._source(await conn.fetchrow("SELECT * FROM runtime_recall_sources WHERE id=$1 FOR UPDATE", source_id))
        if source is None or source.deleted_at is not None:
            return False
        await self._enqueue(conn, source, "delete")
        await self._version(conn, source.model_copy(update={"revision": source.revision + 1, "deleted_at": utcnow()}))
        await conn.execute("INSERT INTO runtime_recall_extractions (source_event_id) VALUES ($1) ON CONFLICT DO NOTHING", source.source_event_id)
        return True

    async def delete(self, source_id: UUID, *, scope: str = "default") -> bool:
        async with self.pool.acquire() as conn, conn.transaction():
            found = await conn.fetchval("SELECT id FROM runtime_recall_sources WHERE id=$1 AND scope=$2 FOR UPDATE", source_id, scope)
            return await self._delete(conn, source_id) if found else False

    async def expire(self, *, limit: int = 100) -> int:
        async with self.pool.acquire() as conn, conn.transaction():
            ids = await conn.fetch("""SELECT id FROM runtime_recall_sources
                WHERE deleted_at IS NULL AND valid_until<=now()
                ORDER BY valid_until LIMIT $1 FOR UPDATE SKIP LOCKED
            """, limit)
            count = 0
            for row in ids:
                count += await self._delete(conn, row["id"])
            return count

    async def claim(self, *, lease_seconds: int = 60) -> IndexJob | None:
        async with self.pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow("""
                WITH due AS (
                    SELECT id FROM runtime_recall_jobs
                    WHERE status <> 'completed' AND available_at<=now()
                    ORDER BY available_at, created_at LIMIT 1 FOR UPDATE SKIP LOCKED
                )
                UPDATE runtime_recall_jobs j SET status='running', attempts=attempts+1,
                    claim_token=$1, available_at=now()+($2 * interval '1 second')
                FROM due WHERE j.id=due.id
                RETURNING j.id, source_id, revision, action, attempts, claim_token AS token
            """, uuid4(), lease_seconds)
        return IndexJob(**dict(row)) if row else None

    async def finish(self, job: IndexJob, *, succeeded: bool) -> bool:
        async with self.pool.acquire() as conn:
            return bool(await conn.fetchval("""
                UPDATE runtime_recall_jobs SET status=$3, claim_token=NULL,
                    available_at=now()+($4 * interval '1 second')
                WHERE id=$1 AND claim_token=$2 AND status='running' RETURNING id
            """, job.id, job.token, "completed" if succeeded else "pending",
                min(300, 2 ** min(job.attempts, 8))))


class InMemoryRecallStore:
    """Deterministic offline repository. Production uses PostgresRecallStore."""

    def __init__(self, *, clock=utcnow):
        self.clock = clock
        self.sources: dict[UUID, RecallSource] = {}
        self.jobs: dict[UUID, dict[str, Any]] = {}
        self.receipts: set[UUID] = set()
        self._lock = asyncio.Lock()

    def _enqueue(self, source: RecallSource, action: str) -> None:
        for row in self.jobs.values():
            job = row["job"]
            if (job.source_id, job.revision, job.action) == (source.id, source.revision, action):
                if row["status"] == "completed":
                    row.update(status="pending", ready_at=self.clock())
                return
        job = IndexJob(source_id=source.id, revision=source.revision, action=action)
        self.jobs[job.id] = {"job": job, "status": "pending", "ready_at": self.clock()}

    def _save(self, source: RecallSource, expected_revision: int | None = None):
        source = RecallSource.model_validate(source.model_dump())
        old = self.sources.get(source.id)
        if old:
            if old.deleted_at:
                return None
            if expected_revision is None:
                return old.model_copy(deep=True)
            if old.revision != expected_revision:
                raise ValueError("Recall source revision conflict")
            if (old.scope, old.source_event_id, old.kind) != (source.scope, source.source_event_id, source.kind):
                raise ValueError("Recall provenance is immutable")
            self._enqueue(old, "delete")
            source = source.model_copy(update={"revision": old.revision + 1})
        elif expected_revision is not None:
            raise ValueError("Recall source revision conflict")
        else:
            source = source.model_copy(update={"revision": 1})
        self.sources[source.id] = source
        self._enqueue(source, "upsert")
        return source.model_copy(deep=True)

    async def save(self, source: RecallSource, *, expected_revision: int | None = None):
        async with self._lock:
            return self._save(source, expected_revision)

    async def extracted(self, event_id: UUID) -> bool:
        return event_id in self.receipts

    async def accept(self, sources: list[RecallSource], event_id: UUID) -> list[RecallSource]:
        async with self._lock:
            if event_id in self.receipts:
                return []
            # Validate the complete batch before changing any state.
            for source in sources:
                if source.source_event_id != event_id:
                    raise ValueError("Extraction provenance mismatch")
                if source.supersedes_id:
                    old = self.sources.get(source.supersedes_id)
                    if not old or old.kind != "fact" or old.scope != source.scope or not old.valid(self.clock()):
                        raise ValueError("Correction target is not an active fact")
            accepted = []
            for source in sources:
                saved = self._save(source)
                if saved:
                    accepted.append(saved)
                    if source.supersedes_id:
                        self._delete(source.supersedes_id)
            self.receipts.add(event_id)
            return accepted

    async def get(self, source_id: UUID) -> RecallSource | None:
        source = self.sources.get(source_id)
        return source.model_copy(deep=True) if source else None

    async def get_many(self, ids: list[UUID]) -> dict[UUID, RecallSource]:
        return {key: self.sources[key].model_copy(deep=True) for key in ids if key in self.sources}

    def _delete(self, source_id: UUID) -> bool:
        source = self.sources.get(source_id)
        if not source or source.deleted_at:
            return False
        self._enqueue(source, "delete")
        self.sources[source_id] = source.model_copy(update={
            "revision": source.revision + 1, "deleted_at": self.clock(),
        })
        # Keep the receipt even after deleting the content from the index.
        self.receipts.add(source.source_event_id)
        return True

    async def delete(self, source_id: UUID, *, scope: str = "default") -> bool:
        async with self._lock:
            source = self.sources.get(source_id)
            if source is None or source.scope != scope:
                return False
            return self._delete(source_id)

    async def expire(self, *, limit: int = 100) -> int:
        async with self._lock:
            ids = [s.id for s in self.sources.values() if s.deleted_at is None
                   and s.valid_until is not None and s.valid_until <= self.clock()][:limit]
            return sum(self._delete(key) for key in ids)

    async def claim(self, *, lease_seconds: int = 60) -> IndexJob | None:
        async with self._lock:
            now = self.clock()
            for row in self.jobs.values():
                if row["status"] != "completed" and row["ready_at"] <= now:
                    job = row["job"].model_copy(update={
                        "attempts": row["job"].attempts + 1, "token": uuid4(),
                    })
                    row.update(job=job, status="running", ready_at=now + timedelta(seconds=lease_seconds))
                    return job.model_copy(deep=True)
        return None

    async def finish(self, job: IndexJob, *, succeeded: bool) -> bool:
        async with self._lock:
            row = self.jobs[job.id]
            if row["status"] != "running" or row["job"].token != job.token:
                return False
            row.update(status="completed" if succeeded else "pending",
                       ready_at=self.clock() + timedelta(seconds=min(300, 2 ** min(job.attempts, 8))))
            return True
