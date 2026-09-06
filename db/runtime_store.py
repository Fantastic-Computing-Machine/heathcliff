"""PostgreSQL-backed canonical Runtime V2 event store."""

from __future__ import annotations

import asyncio
import base64
import json
import sqlite3
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
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
    async def start_turn(self, turn_id: UUID) -> Turn: ...
    async def finish_turn(self, turn_id: UUID, status: TurnStatus, event_kind: EventKind, payload: dict[str, Any]) -> Turn: ...
    async def get_turn(self, turn_id: UUID) -> Turn | None: ...
    async def get_input(self, input_id: UUID) -> PendingInput | None: ...
    async def pending_inputs(self, thread_id: UUID | None = None) -> list[PendingInput]: ...
    async def request_cancel(self, turn_id: UUID) -> bool: ...
    async def cancel_requested(self, turn_id: UUID) -> bool: ...
    async def submit_job(self, kind: str, payload: dict[str, Any], idempotency_key: str, *, max_attempts: int = 3) -> dict[str, Any]: ...
    async def claim_job(self, holder: str, kinds: list[str] | None = None, lease_seconds: int = 60) -> dict[str, Any] | None: ...
    async def finish_job(self, job_id: UUID, holder: str, *, generation: int, succeeded: bool = True, result: dict[str, Any] | None = None, error: str | None = None, retry: bool = False) -> bool: ...
    async def renew_job(self, job_id: UUID, holder: str, generation: int, lease_seconds: int = 60) -> bool: ...
    async def get_job(self, job_id: UUID) -> dict[str, Any] | None: ...
    async def cancel_job(self, job_id: UUID) -> bool: ...
    async def get_lease(self, name: str) -> dict[str, Any] | None: ...
    async def renew_lease(self, name: str, holder: str, expires_at: datetime, *, generation: int | None = None) -> bool: ...
    async def owns_lease(self, name: str, holder: str, *, generation: int | None = None) -> bool: ...
    async def quarantine_resource(self, name: str, holder: str, *, generation: int, reason: str) -> bool: ...
    async def resource_quarantined(self, name: str) -> bool: ...
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
    async def release_lease(self, name: str, holder: str, *, generation: int | None = None) -> None: ...
    async def enqueue_memory_job(self, source_event_id: UUID) -> None: ...
    async def claim_memory_job(self) -> MemoryJob | None: ...
    async def complete_memory_job(self, job_id: UUID, succeeded: bool) -> None: ...
    async def save_memory(self, memory: PersonalMemory) -> None: ...
    async def ready(self) -> bool: ...


def _json(value: Any) -> str:
    return json.dumps(_json_value(value))


def _json_object(value: Any) -> dict[str, Any]:
    return json.loads(value) if isinstance(value, str) else dict(value)


TERMINAL = {TurnStatus.COMPLETED, TurnStatus.CANCELLED, TurnStatus.FAILED}


def _terminal_status(status: TurnStatus, kind: EventKind) -> None:
    if status not in TERMINAL or kind.value != f"turn.{status.value}":
        raise ValueError("Terminal status and event kind must match")


def _positive(value: int) -> None:
    if value <= 0:
        raise ValueError("Attempts and lease duration must be positive")


def _job_row(row: Any) -> dict[str, Any] | None:
    if row is None:
        return None
    result = dict(row)
    for field in ("payload", "result"):
        if result.get(field) is not None:
            result[field] = _json_object(result[field])
    return result


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
        async with self._pool.acquire() as conn, conn.transaction():
            # Serialize migration runners within this database/schema.
            await conn.execute("SELECT pg_advisory_xact_lock(hashtext(current_schema()), 7412)")
            for path in sorted((Path(__file__).parent / "migrations").glob("*.sql")):
                await conn.execute(path.read_text())

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
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                """
                INSERT INTO runtime_inputs
                    (id, thread_id, content, idempotency_key, admitted_at)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (thread_id, idempotency_key) DO NOTHING
                    RETURNING id, thread_id, content, idempotency_key, admitted_at, true AS was_admitted
                """,
                pending.id,
                pending.thread_id,
                pending.content,
                pending.idempotency_key,
                pending.admitted_at,
            )
            if row:
                await self._append_event(conn, RuntimeEvent(
                    thread_id=pending.thread_id, kind=EventKind.INPUT_ADMITTED,
                    payload={"input_id": str(pending.id), "content": pending.content},
                ))
            else:
                # A new statement sees the winner of a concurrent unique-key insert.
                row = await conn.fetchrow(
                    "SELECT *, false AS was_admitted FROM runtime_inputs WHERE thread_id=$1 AND idempotency_key=$2",
                    pending.thread_id, pending.idempotency_key,
                )
        return PendingInput(**dict(row))

    async def create_turn(self, turn: Turn) -> Turn:
        async with self._pool.acquire() as conn, conn.transaction():
            if not await conn.fetchval("SELECT 1 FROM runtime_inputs WHERE id=$1 AND thread_id=$2", turn.input_id, turn.thread_id):
                raise ValueError("Input does not belong to thread")
            row = await conn.fetchrow(
                """
                    INSERT INTO runtime_turns
                    (id, thread_id, input_id, status, created_at, completed_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (input_id) DO NOTHING
                    RETURNING id, thread_id, input_id, status, created_at, completed_at, true AS was_created
                """,
                turn.id,
                turn.thread_id,
                turn.input_id,
                turn.status.value,
                turn.created_at,
                turn.completed_at,
            )
            if row is None:
                row = await conn.fetchrow("SELECT *, false AS was_created FROM runtime_turns WHERE input_id=$1", turn.input_id)
        return Turn(**{**dict(row), "status": TurnStatus(row["status"])})

    async def get_turn(self, turn_id: UUID) -> Turn | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM runtime_turns WHERE id=$1", turn_id)
        return Turn(**dict(row)) if row else None

    async def get_input(self, input_id: UUID) -> PendingInput | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM runtime_inputs WHERE id=$1", input_id)
        return PendingInput(**dict(row)) if row else None

    async def pending_inputs(self, thread_id: UUID | None = None) -> list[PendingInput]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT i.* FROM runtime_inputs i LEFT JOIN runtime_turns t ON t.input_id=i.id
                WHERE ($1::uuid IS NULL OR i.thread_id=$1)
                  AND (t.id IS NULL OR t.status IN ('admitted','running'))
                ORDER BY i.admitted_at, i.id
            """, thread_id)
        return [PendingInput(**dict(row)) for row in rows]

    async def _locked_turn(self, conn: Any, turn_id: UUID) -> Turn:
        await conn.fetchval("""SELECT t.id FROM runtime_threads t
            JOIN runtime_turns r ON r.thread_id=t.id WHERE r.id=$1 FOR UPDATE OF t""", turn_id)
        row = await conn.fetchrow("SELECT * FROM runtime_turns WHERE id=$1 FOR UPDATE", turn_id)
        if row is None:
            raise ValueError("Unknown turn")
        return Turn(**dict(row))

    async def start_turn(self, turn_id: UUID) -> Turn:
        async with self._pool.acquire() as conn, conn.transaction():
            turn = await self._locked_turn(conn, turn_id)
            if turn.status in TERMINAL or turn.status == TurnStatus.RUNNING:
                return turn
            if await conn.fetchval("SELECT 1 FROM runtime_turns WHERE thread_id=$1 AND id<>$2 AND status IN ('running','waiting_for_approval') LIMIT 1", turn.thread_id, turn.id):
                raise ValueError("Another turn is active in this thread")
            row = await conn.fetchrow("UPDATE runtime_turns SET status='running' WHERE id=$1 RETURNING *", turn.id)
            if not await conn.fetchval("SELECT 1 FROM runtime_events WHERE turn_id=$1 AND kind='turn.started'", turn.id):
                content = await conn.fetchval("SELECT content FROM runtime_inputs WHERE id=$1", turn.input_id)
                await self._append_event(conn, RuntimeEvent(thread_id=turn.thread_id, turn_id=turn.id, kind=EventKind.TURN_STARTED, payload={"input_id": str(turn.input_id), "content": content}))
            return Turn(**dict(row))

    async def finish_turn(self, turn_id: UUID, status: TurnStatus, event_kind: EventKind, payload: dict[str, Any]) -> Turn:
        _terminal_status(status, event_kind)
        async with self._pool.acquire() as conn, conn.transaction():
            turn = await self._locked_turn(conn, turn_id)
            if turn.status in TERMINAL:
                return turn
            row = await conn.fetchrow("UPDATE runtime_turns SET status=$2, completed_at=now() WHERE id=$1 RETURNING *", turn_id, status.value)
            await self._append_event(conn, RuntimeEvent(thread_id=turn.thread_id, turn_id=turn.id, kind=event_kind, payload=payload))
            source = await conn.fetchval("SELECT id FROM runtime_events WHERE thread_id=$1 AND kind='input.admitted' AND payload->>'input_id'=$2 ORDER BY sequence LIMIT 1", turn.thread_id, str(turn.input_id))
            if source:
                await self._enqueue_memory_job(conn, source)
            return Turn(**dict(row))

    async def request_cancel(self, turn_id: UUID) -> bool:
        async with self._pool.acquire() as conn:
            return bool(await conn.fetchval("UPDATE runtime_turns SET cancel_requested=true WHERE id=$1 AND status NOT IN ('completed','cancelled','failed') RETURNING true", turn_id))

    async def cancel_requested(self, turn_id: UUID) -> bool:
        async with self._pool.acquire() as conn:
            return bool(await conn.fetchval("SELECT cancel_requested FROM runtime_turns WHERE id=$1", turn_id))

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
            return await self._append_event(conn, event)

    async def _append_event(self, conn: Any, event: RuntimeEvent) -> RuntimeEvent:
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
        data["payload"] = _json_object(data["payload"])
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
        return ContextCheckpoint(**_json_object(row["payload"])) if row else None

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
            tool_call=_json_object(data["tool_call"]),
            resource_scope=_json_object(data["resource_scope"]),
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
            tool_call=_json_object(data["tool_call"]),
            resource_scope=_json_object(data["resource_scope"]),
            expires_at=data["expires_at"],
            status=data["status"],
        )

    async def decide_approval(self, decision: ApprovalDecision) -> ApprovalRequest:
        status = (
            ApprovalStatus.APPROVED if decision.approved else ApprovalStatus.REJECTED
        )
        async with self._pool.acquire() as conn, conn.transaction():
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
            await self._submit_job(conn, "approval.resume", {"approval_id": str(row["id"]), "turn_id": str(row["turn_id"]), "approved": decision.approved}, str(row["id"]), 3)
        data = dict(row)
        return ApprovalRequest(
            id=data["id"],
            thread_id=data["thread_id"],
            turn_id=data["turn_id"],
            tool_call=_json_object(data["tool_call"]),
            resource_scope=_json_object(data["resource_scope"]),
            expires_at=data["expires_at"],
            status=data["status"],
        )

    async def acquire_lease(self, lease: RuntimeLease) -> bool:
        if not lease.holder or lease.expires_at <= utcnow():
            raise ValueError("Lease needs a holder and future expiry")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO runtime_leases (name, holder, expires_at)
                SELECT $1, $2, $3 WHERE NOT EXISTS (SELECT 1 FROM runtime_resource_quarantine WHERE name=$1)
                ON CONFLICT (name) DO UPDATE SET holder = EXCLUDED.holder, expires_at = EXCLUDED.expires_at,
                    generation = runtime_leases.generation + 1
                WHERE runtime_leases.expires_at <= now()
                  AND NOT EXISTS (SELECT 1 FROM runtime_resource_quarantine WHERE name=$1)
                RETURNING holder
                """,
                lease.name,
                lease.holder,
                lease.expires_at,
            )
        return row is not None and row["holder"] == lease.holder

    async def get_lease(self, name: str) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM runtime_leases WHERE name=$1", name)
        return dict(row) if row else None

    async def owns_lease(self, name: str, holder: str, *, generation: int | None = None) -> bool:
        async with self._pool.acquire() as conn:
            return bool(await conn.fetchval("SELECT true FROM runtime_leases WHERE name=$1 AND holder=$2 AND expires_at>now() AND ($3::bigint IS NULL OR generation=$3)", name, holder, generation))

    async def renew_lease(self, name: str, holder: str, expires_at: datetime, *, generation: int | None = None) -> bool:
        if expires_at <= utcnow():
            raise ValueError("Lease expiry must be in the future")
        async with self._pool.acquire() as conn:
            return bool(await conn.fetchval("UPDATE runtime_leases SET expires_at=$3 WHERE name=$1 AND holder=$2 AND expires_at>now() AND ($4::bigint IS NULL OR generation=$4) RETURNING true", name, holder, expires_at, generation))

    async def release_lease(self, name: str, holder: str, *, generation: int | None = None) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE runtime_leases SET expires_at=now() WHERE name=$1 AND holder=$2 AND ($3::bigint IS NULL OR generation=$3)",
                name, holder, generation,
            )

    async def quarantine_resource(self, name: str, holder: str, *, generation: int, reason: str) -> bool:
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow("SELECT * FROM runtime_leases WHERE name=$1 AND holder=$2 AND generation=$3 FOR UPDATE", name, holder, generation)
            if not row:
                return False
            await conn.execute("INSERT INTO runtime_resource_quarantine(name,holder,generation,reason) VALUES($1,$2,$3,$4) ON CONFLICT(name) DO NOTHING", name, holder, generation, reason)
            return True

    async def resource_quarantined(self, name: str) -> bool:
        async with self._pool.acquire() as conn:
            return bool(await conn.fetchval("SELECT true FROM runtime_resource_quarantine WHERE name=$1", name))

    async def submit_job(self, kind: str, payload: dict[str, Any], idempotency_key: str, *, max_attempts: int = 3) -> dict[str, Any]:
        async with self._pool.acquire() as conn, conn.transaction():
            return await self._submit_job(conn, kind, payload, idempotency_key, max_attempts)

    async def _submit_job(self, conn: Any, kind: str, payload: dict[str, Any], idempotency_key: str, max_attempts: int) -> dict[str, Any]:
        _positive(max_attempts)
        if not kind or not idempotency_key or not isinstance(payload, dict):
            raise ValueError("Job requires kind, idempotency key and object payload")
        row = await conn.fetchrow("INSERT INTO runtime_jobs(id,kind,payload,idempotency_key,max_attempts) VALUES($1,$2,$3::jsonb,$4,$5) ON CONFLICT(kind,idempotency_key) DO NOTHING RETURNING *", uuid4(), kind, _json(payload), idempotency_key, max_attempts)
        if row is None:
            row = await conn.fetchrow("SELECT * FROM runtime_jobs WHERE kind=$1 AND idempotency_key=$2", kind, idempotency_key)
        return cast(dict[str, Any], _job_row(row))

    async def get_job(self, job_id: UUID) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            return _job_row(await conn.fetchrow("SELECT * FROM runtime_jobs WHERE id=$1", job_id))

    async def claim_job(self, holder: str, kinds: list[str] | None = None, lease_seconds: int = 60) -> dict[str, Any] | None:
        _positive(lease_seconds)
        if not holder:
            raise ValueError("Claim holder is required")
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute("""UPDATE runtime_jobs SET status=CASE WHEN cancel_requested THEN 'cancelled' ELSE 'failed' END, completed_at=now()
                WHERE (status='running' AND expires_at<=now() OR status='pending')
                AND (cancel_requested OR attempts>=max_attempts)""")
            return _job_row(await conn.fetchrow("""
                WITH candidate AS (
                    SELECT id FROM runtime_jobs
                    WHERE (status='pending' OR status='running' AND expires_at<=now())
                      AND NOT cancel_requested AND attempts<max_attempts
                      AND ($2::text[] IS NULL OR kind=ANY($2))
                    ORDER BY created_at,id FOR UPDATE SKIP LOCKED LIMIT 1
                )
                UPDATE runtime_jobs j SET status='running', holder=$1,
                    generation=generation+1, attempts=attempts+1,
                    expires_at=now()+$3*interval '1 second'
                FROM candidate c WHERE j.id=c.id RETURNING j.*
            """, holder, kinds, lease_seconds))

    async def renew_job(self, job_id: UUID, holder: str, generation: int, lease_seconds: int = 60) -> bool:
        _positive(lease_seconds)
        async with self._pool.acquire() as conn:
            return bool(await conn.fetchval("UPDATE runtime_jobs SET expires_at=now()+$4*interval '1 second' WHERE id=$1 AND holder=$2 AND generation=$3 AND status='running' AND expires_at>now() AND NOT cancel_requested RETURNING true", job_id, holder, generation, lease_seconds))

    async def finish_job(self, job_id: UUID, holder: str, *, generation: int, succeeded: bool = True, result: dict[str, Any] | None = None, error: str | None = None, retry: bool = False) -> bool:
        async with self._pool.acquire() as conn:
            return bool(await conn.fetchval("""UPDATE runtime_jobs
                SET status=CASE WHEN cancel_requested THEN 'cancelled' WHEN $4 THEN 'completed'
                    WHEN $7 AND attempts<max_attempts THEN 'pending' ELSE 'failed' END,
                    result=$5::jsonb, error=$6, expires_at=NULL, holder=NULL,
                    completed_at=CASE WHEN NOT cancel_requested AND NOT $4 AND $7 AND attempts<max_attempts THEN NULL ELSE now() END
                WHERE id=$1 AND holder=$2 AND generation=$3 AND status='running' AND expires_at>now()
                RETURNING true""", job_id, holder, generation, succeeded, _json(result) if result is not None else None, error, retry))

    async def cancel_job(self, job_id: UUID) -> bool:
        async with self._pool.acquire() as conn:
            return bool(await conn.fetchval("UPDATE runtime_jobs SET cancel_requested=true, status=CASE WHEN status='pending' THEN 'cancelled' ELSE status END, completed_at=CASE WHEN status='pending' THEN now() ELSE completed_at END WHERE id=$1 AND status IN ('pending','running') RETURNING true", job_id))

    async def enqueue_memory_job(self, source_event_id: UUID) -> None:
        async with self._pool.acquire() as conn:
            await self._enqueue_memory_job(conn, source_event_id)

    async def _enqueue_memory_job(self, conn: Any, source_event_id: UUID) -> None:
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
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO personal_memories
                (id, kind, subject, content, confidence, source_event_id, source_kind,
                 valid_from, valid_until, supersedes_id, created_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
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
