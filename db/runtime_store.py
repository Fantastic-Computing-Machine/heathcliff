"""PostgreSQL-backed canonical Runtime V2 event store."""

from __future__ import annotations

import asyncio
import base64
import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import UUID, uuid4

from core.runtime.contracts import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    ContextCheckpoint,
    EventKind,
    MemoryJob,
    PendingInput,
    PersonalMemory,
    RuntimeEvent,
    RuntimeLease,
    Thread,
    Turn,
    TurnStatus,
    utcnow,
)


class RuntimeStore(Protocol):
    async def create_thread(self, thread: Thread | None = None) -> Thread: ...
    async def ensure_thread(self, thread_id: UUID) -> Thread: ...
    async def admit_input(self, pending: PendingInput) -> PendingInput: ...
    async def create_turn(self, turn: Turn) -> Turn: ...
    async def set_turn_status(self, turn_id: UUID, status: TurnStatus) -> None: ...
    async def append_event(self, event: RuntimeEvent) -> RuntimeEvent: ...
    async def events(self, thread_id: UUID, after: int = 0) -> list[RuntimeEvent]: ...
    async def event(self, event_id: UUID) -> RuntimeEvent | None: ...
    async def latest_checkpoint(self, thread_id: UUID) -> ContextCheckpoint | None: ...
    async def save_checkpoint(self, checkpoint: ContextCheckpoint) -> None: ...
    async def save_approval(self, approval: ApprovalRequest) -> None: ...
    async def decide_approval(self, decision: ApprovalDecision) -> ApprovalRequest: ...
    async def get_approval(self, approval_id: UUID) -> ApprovalRequest | None: ...
    async def pending_approval(self, thread_id: UUID) -> ApprovalRequest | None: ...
    async def acquire_lease(self, lease: RuntimeLease) -> bool: ...
    async def release_lease(self, name: str, holder: str) -> None: ...
    async def enqueue_memory_job(self, source_event_id: UUID) -> None: ...
    async def claim_memory_job(self) -> MemoryJob | None: ...
    async def complete_memory_job(self, job_id: UUID, succeeded: bool) -> None: ...
    async def save_memory(self, memory: PersonalMemory) -> None: ...
    async def ready(self) -> bool: ...


def _json(value: Any) -> str:
    return json.dumps(_json_value(value))


def _json_value(value: Any) -> Any:
    """Convert opaque provider/tool values into a durable JSON representation."""
    if isinstance(value, bytes):
        return {"encoding": "base64", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, (UUID, datetime)):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class PostgresRuntimeStore:
    """Small transactional store using plain SQL and asyncpg."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._pool: Any = None

    async def connect(self) -> None:
        try:
            import asyncpg
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("asyncpg is required for Runtime V2") from exc
        self._pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=5)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def migrate(self) -> None:
        sql = (Path(__file__).parent / "migrations" / "001_runtime_v2.sql").read_text()
        async with self._pool.acquire() as conn:
            await conn.execute(sql)

    async def ready(self) -> bool:
        if self._pool is None:
            return False
        try:
            async with self._pool.acquire() as conn:
                return (await conn.fetchval("SELECT 1")) == 1
        except Exception:
            return False

    async def create_thread(self, thread: Thread | None = None) -> Thread:
        thread = thread or Thread()
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO runtime_threads (id, created_at, next_event_seq) VALUES ($1, $2, $3)",
                thread.id,
                thread.created_at,
                thread.next_event_seq,
            )
        return thread

    async def ensure_thread(self, thread_id: UUID) -> Thread:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO runtime_threads (id) VALUES ($1) ON CONFLICT DO NOTHING",
                thread_id,
            )
            row = await conn.fetchrow(
                "SELECT id, created_at, next_event_seq FROM runtime_threads WHERE id = $1",
                thread_id,
            )
        return Thread(**dict(row))

    async def admit_input(self, pending: PendingInput) -> PendingInput:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                WITH inserted AS (
                    INSERT INTO runtime_inputs
                    (id, thread_id, content, idempotency_key, admitted_at)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT DO NOTHING
                    RETURNING id, thread_id, content, idempotency_key, admitted_at, true AS was_admitted
                )
                SELECT * FROM inserted
                UNION ALL
                SELECT id, thread_id, content, idempotency_key, admitted_at, false AS was_admitted
                FROM runtime_inputs
                WHERE thread_id = $2 AND idempotency_key = $4
                  AND NOT EXISTS (SELECT 1 FROM inserted)
                LIMIT 1
                """,
                pending.id,
                pending.thread_id,
                pending.content,
                pending.idempotency_key,
                pending.admitted_at,
            )
        return PendingInput(**dict(row))

    async def create_turn(self, turn: Turn) -> Turn:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                WITH inserted AS (
                    INSERT INTO runtime_turns
                    (id, thread_id, input_id, status, created_at, completed_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (input_id) DO NOTHING
                    RETURNING id, thread_id, input_id, status, created_at, completed_at, true AS was_created
                )
                SELECT * FROM inserted
                UNION ALL
                SELECT id, thread_id, input_id, status, created_at, completed_at, false AS was_created
                FROM runtime_turns
                WHERE input_id = $3 AND NOT EXISTS (SELECT 1 FROM inserted)
                LIMIT 1
                """,
                turn.id,
                turn.thread_id,
                turn.input_id,
                turn.status.value,
                turn.created_at,
                turn.completed_at,
            )
        return Turn(**{**dict(row), "status": TurnStatus(row["status"])})

    async def set_turn_status(self, turn_id: UUID, status: TurnStatus) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE runtime_turns
                SET status = $2,
                    completed_at = CASE
                        WHEN $2 IN ('completed', 'cancelled', 'failed') THEN now()
                        ELSE completed_at
                    END
                WHERE id = $1
                """,
                turn_id,
                status.value,
            )

    async def append_event(self, event: RuntimeEvent) -> RuntimeEvent:
        async with self._pool.acquire() as conn, conn.transaction():
            sequence = await conn.fetchval(
                """
                UPDATE runtime_threads SET next_event_seq = next_event_seq + 1
                WHERE id = $1 RETURNING next_event_seq - 1
                """,
                event.thread_id,
            )
            if sequence is None:
                raise ValueError(f"Unknown thread: {event.thread_id}")
            await conn.execute(
                """
                INSERT INTO runtime_events (id, thread_id, turn_id, sequence, kind, payload, created_at)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
                """,
                event.id,
                event.thread_id,
                event.turn_id,
                sequence,
                event.kind.value,
                _json(event.payload),
                event.created_at,
            )
        return event.model_copy(update={"sequence": sequence})

    async def events(self, thread_id: UUID, after: int = 0) -> list[RuntimeEvent]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, thread_id, turn_id, sequence, kind, payload, created_at
                FROM runtime_events WHERE thread_id = $1 AND sequence > $2
                ORDER BY sequence
                """,
                thread_id,
                after,
            )
        return [self._event_from_row(row) for row in rows]

    @staticmethod
    def _event_from_row(row: Any) -> RuntimeEvent:
        data = cast(dict[str, Any], dict(row))
        data["kind"] = EventKind(data["kind"])
        data["payload"] = dict(data["payload"])
        return RuntimeEvent(**data)

    async def event(self, event_id: UUID) -> RuntimeEvent | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, thread_id, turn_id, sequence, kind, payload, created_at
                FROM runtime_events WHERE id = $1
                """,
                event_id,
            )
        if not row:
            return None
        return self._event_from_row(row)

    async def latest_checkpoint(self, thread_id: UUID) -> ContextCheckpoint | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT payload FROM runtime_checkpoints
                WHERE thread_id = $1 ORDER BY through_sequence DESC LIMIT 1
                """,
                thread_id,
            )
        return ContextCheckpoint(**dict(row["payload"])) if row else None

    async def save_checkpoint(self, checkpoint: ContextCheckpoint) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO runtime_checkpoints (id, thread_id, through_sequence, payload)
                VALUES ($1, $2, $3, $4::jsonb)
                """,
                checkpoint.id,
                checkpoint.thread_id,
                checkpoint.through_sequence,
                _json(checkpoint.model_dump(mode="json")),
            )

    async def save_approval(self, approval: ApprovalRequest) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO runtime_approvals
                (id, thread_id, turn_id, tool_call, resource_scope, expires_at, status)
                VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6, $7)
                """,
                approval.id,
                approval.thread_id,
                approval.turn_id,
                _json(approval.tool_call.model_dump(mode="json")),
                _json(approval.resource_scope.model_dump(mode="json")),
                approval.expires_at,
                approval.status.value,
            )

    async def get_approval(self, approval_id: UUID) -> ApprovalRequest | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM runtime_approvals WHERE id = $1", approval_id
            )
        if not row:
            return None
        data = dict(row)
        return ApprovalRequest(
            id=data["id"],
            thread_id=data["thread_id"],
            turn_id=data["turn_id"],
            tool_call=data["tool_call"],
            resource_scope=data["resource_scope"],
            expires_at=data["expires_at"],
            status=data["status"],
        )

    async def pending_approval(self, thread_id: UUID) -> ApprovalRequest | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM runtime_approvals
                WHERE thread_id = $1 AND status = 'pending' AND expires_at > now()
                ORDER BY expires_at LIMIT 1
                """,
                thread_id,
            )
        if not row:
            return None
        data = dict(row)
        return ApprovalRequest(
            id=data["id"],
            thread_id=data["thread_id"],
            turn_id=data["turn_id"],
            tool_call=data["tool_call"],
            resource_scope=data["resource_scope"],
            expires_at=data["expires_at"],
            status=data["status"],
        )

    async def decide_approval(self, decision: ApprovalDecision) -> ApprovalRequest:
        status = (
            ApprovalStatus.APPROVED if decision.approved else ApprovalStatus.REJECTED
        )
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE runtime_approvals SET status = $2, decided_at = $3
                WHERE id = $1 AND status = 'pending' AND expires_at > now()
                RETURNING *
                """,
                decision.approval_id,
                status.value,
                decision.decided_at,
            )
        if not row:
            raise ValueError("Approval is no longer pending or does not exist")
        data = dict(row)
        return ApprovalRequest(
            id=data["id"],
            thread_id=data["thread_id"],
            turn_id=data["turn_id"],
            tool_call=data["tool_call"],
            resource_scope=data["resource_scope"],
            expires_at=data["expires_at"],
            status=data["status"],
        )

    async def acquire_lease(self, lease: RuntimeLease) -> bool:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO runtime_leases (name, holder, expires_at) VALUES ($1, $2, $3)
                ON CONFLICT (name) DO UPDATE SET holder = EXCLUDED.holder, expires_at = EXCLUDED.expires_at
                WHERE runtime_leases.expires_at < now() OR runtime_leases.holder = EXCLUDED.holder
                RETURNING holder
                """,
                lease.name,
                lease.holder,
                lease.expires_at,
            )
        return row is not None and row["holder"] == lease.holder

    async def release_lease(self, name: str, holder: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM runtime_leases WHERE name = $1 AND holder = $2",
                name,
                holder,
            )

    async def enqueue_memory_job(self, source_event_id: UUID) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO runtime_memory_jobs (id, source_event_id) VALUES ($1, $2)
                ON CONFLICT (source_event_id) DO NOTHING
                """,
                uuid4(),
                source_event_id,
            )

    async def claim_memory_job(self) -> MemoryJob | None:
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                """
                SELECT id, source_event_id, attempts FROM runtime_memory_jobs
                WHERE status = 'pending' ORDER BY created_at
                FOR UPDATE SKIP LOCKED LIMIT 1
                """
            )
            if not row:
                return None
            claimed = await conn.fetchrow(
                """
                UPDATE runtime_memory_jobs
                SET status = 'running', attempts = attempts + 1, claimed_at = now()
                WHERE id = $1
                RETURNING id, source_event_id, attempts
                """,
                row["id"],
            )
        return MemoryJob(**dict(claimed))

    async def complete_memory_job(self, job_id: UUID, succeeded: bool) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE runtime_memory_jobs
                SET status = $2, completed_at = now() WHERE id = $1
                """,
                job_id,
                "completed" if succeeded else "failed",
            )

    async def save_memory(self, memory: PersonalMemory) -> None:
        embedding = (
            "[" + ",".join(str(value) for value in memory.embedding) + "]"
            if memory.embedding is not None
            else None
        )
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO personal_memories
                (id, kind, subject, content, confidence, source_event_id, source_kind,
                 valid_from, valid_until, supersedes_id, embedding, created_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11::vector,$12)
                """,
                memory.id,
                memory.kind,
                memory.subject,
                memory.content,
                memory.confidence,
                memory.source_event_id,
                memory.source_kind,
                memory.valid_from,
                memory.valid_until,
                memory.supersedes_id,
                embedding,
                memory.created_at,
            )


class InMemoryRuntimeStore:
    """Deterministic test double; never selected by production configuration."""

    def __init__(self) -> None:
        self.threads: dict[UUID, Thread] = {}
        self.inputs: dict[tuple[UUID, str], PendingInput] = {}
        self.turns: dict[UUID, Turn] = {}
        self.turns_by_input: dict[UUID, UUID] = {}
        self.event_log: dict[UUID, list[RuntimeEvent]] = defaultdict(list)
        self.approvals: dict[UUID, ApprovalRequest] = {}
        self.checkpoints: dict[UUID, ContextCheckpoint] = {}
        self.leases: dict[str, RuntimeLease] = {}
        self.memory_jobs: dict[UUID, MemoryJob] = {}
        self.memory_job_status: dict[UUID, str] = {}
        self.memories: dict[UUID, PersonalMemory] = {}

    async def create_thread(self, thread: Thread | None = None) -> Thread:
        thread = thread or Thread()
        self.threads[thread.id] = thread
        return thread

    async def ensure_thread(self, thread_id: UUID) -> Thread:
        return await self.create_thread(
            self.threads.get(thread_id) or Thread(id=thread_id)
        )

    async def admit_input(self, pending: PendingInput) -> PendingInput:
        key = (pending.thread_id, pending.idempotency_key)
        existing = self.inputs.get(key)
        if existing:
            return existing.model_copy(update={"was_admitted": False})
        admitted = pending.model_copy(update={"was_admitted": True})
        self.inputs[key] = admitted
        return admitted

    async def create_turn(self, turn: Turn) -> Turn:
        existing_id = self.turns_by_input.get(turn.input_id)
        if existing_id:
            return self.turns[existing_id].model_copy(update={"was_created": False})
        stored = turn.model_copy(update={"was_created": True})
        self.turns[stored.id] = stored
        self.turns_by_input[stored.input_id] = stored.id
        return stored

    async def set_turn_status(self, turn_id: UUID, status: TurnStatus) -> None:
        turn = self.turns[turn_id]
        self.turns[turn_id] = turn.model_copy(update={"status": status})

    async def append_event(self, event: RuntimeEvent) -> RuntimeEvent:
        thread = self.threads[event.thread_id]
        stored = event.model_copy(update={"sequence": thread.next_event_seq})
        self.threads[event.thread_id] = thread.model_copy(
            update={"next_event_seq": thread.next_event_seq + 1}
        )
        self.event_log[event.thread_id].append(stored)
        return stored

    async def events(self, thread_id: UUID, after: int = 0) -> list[RuntimeEvent]:
        return [event for event in self.event_log[thread_id] if event.sequence > after]

    async def event(self, event_id: UUID) -> RuntimeEvent | None:
        return next(
            (
                event
                for events in self.event_log.values()
                for event in events
                if event.id == event_id
            ),
            None,
        )

    async def latest_checkpoint(self, thread_id: UUID) -> ContextCheckpoint | None:
        return self.checkpoints.get(thread_id)

    async def save_checkpoint(self, checkpoint: ContextCheckpoint) -> None:
        self.checkpoints[checkpoint.thread_id] = checkpoint

    async def save_approval(self, approval: ApprovalRequest) -> None:
        self.approvals[approval.id] = approval

    async def get_approval(self, approval_id: UUID) -> ApprovalRequest | None:
        return self.approvals.get(approval_id)

    async def pending_approval(self, thread_id: UUID) -> ApprovalRequest | None:
        return next(
            (
                approval
                for approval in self.approvals.values()
                if approval.thread_id == thread_id
                and approval.status == ApprovalStatus.PENDING
                and approval.expires_at > utcnow()
            ),
            None,
        )

    async def decide_approval(self, decision: ApprovalDecision) -> ApprovalRequest:
        approval = self.approvals[decision.approval_id]
        if approval.status != ApprovalStatus.PENDING or approval.expires_at <= utcnow():
            raise ValueError("Approval is no longer pending")
        updated = approval.model_copy(
            update={
                "status": ApprovalStatus.APPROVED
                if decision.approved
                else ApprovalStatus.REJECTED
            }
        )
        self.approvals[updated.id] = updated
        return updated

    async def acquire_lease(self, lease: RuntimeLease) -> bool:
        existing = self.leases.get(lease.name)
        if (
            existing
            and existing.expires_at > utcnow()
            and existing.holder != lease.holder
        ):
            return False
        self.leases[lease.name] = lease
        return True

    async def release_lease(self, name: str, holder: str) -> None:
        if self.leases.get(name) and self.leases[name].holder == holder:
            del self.leases[name]

    async def enqueue_memory_job(self, source_event_id: UUID) -> None:
        if any(
            job.source_event_id == source_event_id for job in self.memory_jobs.values()
        ):
            return
        job = MemoryJob(source_event_id=source_event_id)
        self.memory_jobs[job.id] = job
        self.memory_job_status[job.id] = "pending"

    async def claim_memory_job(self) -> MemoryJob | None:
        for job_id, job in self.memory_jobs.items():
            if self.memory_job_status[job_id] == "pending":
                claimed = job.model_copy(update={"attempts": job.attempts + 1})
                self.memory_jobs[job_id] = claimed
                self.memory_job_status[job_id] = "running"
                return claimed
        return None

    async def complete_memory_job(self, job_id: UUID, succeeded: bool) -> None:
        self.memory_job_status[job_id] = "completed" if succeeded else "failed"

    async def save_memory(self, memory: PersonalMemory) -> None:
        self.memories[memory.id] = memory

    async def ready(self) -> bool:
        return True


class SqliteRuntimeStore(InMemoryRuntimeStore):
    """Durable single-host Runtime V2 store using the standard library."""

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path).expanduser()
        self._connected = False
        self._write_lock = asyncio.Lock()

    async def connect(self) -> None:
        def open_database() -> str | None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.path) as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS runtime_state "
                    "(id INTEGER PRIMARY KEY CHECK (id = 1), snapshot TEXT NOT NULL)"
                )
                row = connection.execute(
                    "SELECT snapshot FROM runtime_state WHERE id = 1"
                ).fetchone()
                return row[0] if row else None

        snapshot = await asyncio.to_thread(open_database)
        if snapshot:
            self._restore(json.loads(snapshot))
        self._connected = True

    async def close(self) -> None:
        self._connected = False

    async def migrate(self) -> None:
        if not self._connected:
            await self.connect()

    async def _persist(self) -> None:
        snapshot = json.dumps(self._snapshot(), separators=(",", ":"))

        def save() -> None:
            with sqlite3.connect(self.path) as connection:
                connection.execute(
                    "INSERT INTO runtime_state (id, snapshot) VALUES (1, ?) "
                    "ON CONFLICT(id) DO UPDATE SET snapshot = excluded.snapshot",
                    (snapshot,),
                )

        await asyncio.to_thread(save)

    def _snapshot(self) -> dict[str, Any]:
        return {
            "threads": {
                str(key): _json_value(value.model_dump(mode="python"))
                for key, value in self.threads.items()
            },
            "inputs": [
                _json_value(value.model_dump(mode="python"))
                for value in self.inputs.values()
            ],
            "turns": {
                str(key): _json_value(value.model_dump(mode="python"))
                for key, value in self.turns.items()
            },
            "events": [
                _json_value(event.model_dump(mode="python"))
                for events in self.event_log.values()
                for event in events
            ],
            "approvals": {
                str(key): _json_value(value.model_dump(mode="python"))
                for key, value in self.approvals.items()
            },
            "checkpoints": {
                str(key): _json_value(value.model_dump(mode="python"))
                for key, value in self.checkpoints.items()
            },
            "leases": {
                key: _json_value(value.model_dump(mode="python"))
                for key, value in self.leases.items()
            },
            "memory_jobs": {
                str(key): _json_value(value.model_dump(mode="python"))
                for key, value in self.memory_jobs.items()
            },
            "memory_job_status": {
                str(key): value for key, value in self.memory_job_status.items()
            },
            "memories": {
                str(key): _json_value(value.model_dump(mode="python"))
                for key, value in self.memories.items()
            },
        }

    def _restore(self, snapshot: dict[str, Any]) -> None:
        self.threads = {
            UUID(key): Thread.model_validate(value)
            for key, value in snapshot.get("threads", {}).items()
        }
        self.inputs = {}
        for value in snapshot.get("inputs", []):
            pending = PendingInput.model_validate(value)
            self.inputs[(pending.thread_id, pending.idempotency_key)] = pending
        self.turns = {
            UUID(key): Turn.model_validate(value)
            for key, value in snapshot.get("turns", {}).items()
        }
        self.turns_by_input = {turn.input_id: turn.id for turn in self.turns.values()}
        self.event_log = defaultdict(list)
        for value in snapshot.get("events", []):
            event = RuntimeEvent.model_validate(value)
            self.event_log[event.thread_id].append(event)
        for events in self.event_log.values():
            events.sort(key=lambda event: event.sequence)
        self.approvals = {
            UUID(key): ApprovalRequest.model_validate(value)
            for key, value in snapshot.get("approvals", {}).items()
        }
        self.checkpoints = {
            UUID(key): ContextCheckpoint.model_validate(value)
            for key, value in snapshot.get("checkpoints", {}).items()
        }
        self.leases = {
            key: RuntimeLease.model_validate(value)
            for key, value in snapshot.get("leases", {}).items()
        }
        self.memory_jobs = {
            UUID(key): MemoryJob.model_validate(value)
            for key, value in snapshot.get("memory_jobs", {}).items()
        }
        self.memory_job_status = {
            UUID(key): value
            for key, value in snapshot.get("memory_job_status", {}).items()
        }
        self.memories = {
            UUID(key): PersonalMemory.model_validate(value)
            for key, value in snapshot.get("memories", {}).items()
        }

    async def create_thread(self, thread: Thread | None = None) -> Thread:
        async with self._write_lock:
            result = await super().create_thread(thread)
            await self._persist()
            return result

    async def admit_input(self, pending: PendingInput) -> PendingInput:
        async with self._write_lock:
            result = await super().admit_input(pending)
            if result.was_admitted:
                await self._persist()
            return result

    async def create_turn(self, turn: Turn) -> Turn:
        async with self._write_lock:
            result = await super().create_turn(turn)
            if result.was_created:
                await self._persist()
            return result

    async def set_turn_status(self, turn_id: UUID, status: TurnStatus) -> None:
        async with self._write_lock:
            await super().set_turn_status(turn_id, status)
            await self._persist()

    async def append_event(self, event: RuntimeEvent) -> RuntimeEvent:
        async with self._write_lock:
            result = await super().append_event(event)
            await self._persist()
            return result

    async def save_checkpoint(self, checkpoint: ContextCheckpoint) -> None:
        async with self._write_lock:
            await super().save_checkpoint(checkpoint)
            await self._persist()

    async def save_approval(self, approval: ApprovalRequest) -> None:
        async with self._write_lock:
            await super().save_approval(approval)
            await self._persist()

    async def decide_approval(self, decision: ApprovalDecision) -> ApprovalRequest:
        async with self._write_lock:
            result = await super().decide_approval(decision)
            await self._persist()
            return result

    async def acquire_lease(self, lease: RuntimeLease) -> bool:
        async with self._write_lock:
            result = await super().acquire_lease(lease)
            if result:
                await self._persist()
            return result

    async def release_lease(self, name: str, holder: str) -> None:
        async with self._write_lock:
            await super().release_lease(name, holder)
            await self._persist()

    async def enqueue_memory_job(self, source_event_id: UUID) -> None:
        async with self._write_lock:
            await super().enqueue_memory_job(source_event_id)
            await self._persist()

    async def claim_memory_job(self) -> MemoryJob | None:
        async with self._write_lock:
            result = await super().claim_memory_job()
            if result:
                await self._persist()
            return result

    async def complete_memory_job(self, job_id: UUID, succeeded: bool) -> None:
        async with self._write_lock:
            await super().complete_memory_job(job_id, succeeded)
            await self._persist()

    async def save_memory(self, memory: PersonalMemory) -> None:
        async with self._write_lock:
            await super().save_memory(memory)
            await self._persist()

    async def ready(self) -> bool:
        if not self._connected:
            return False

        def check() -> bool:
            with sqlite3.connect(self.path) as connection:
                return connection.execute("SELECT 1").fetchone() == (1,)

        try:
            return await asyncio.to_thread(check)
        except sqlite3.Error:
            return False
