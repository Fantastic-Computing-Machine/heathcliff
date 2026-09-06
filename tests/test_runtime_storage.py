"""Storage durability; PostgreSQL runs only against an explicitly supplied test DSN."""

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import timedelta
from uuid import uuid4

import pytest

os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
os.environ["LANGFUSE_TRACING_ENABLED"] = "false"

from core.runtime.contracts import (
    ApprovalDecision, ApprovalRequest, ContextCheckpoint, EventKind, PendingInput,
    PersonalMemory, ResourceScope, RuntimeLease, ToolCall, Turn, TurnStatus, utcnow,
)
from db.runtime_store import InMemoryRuntimeStore, PostgresRuntimeStore, SqliteRuntimeStore


@asynccontextmanager
async def storage(backend, tmp_path):
    if backend == "postgres":
        # Never fall back to application configuration or load .env.
        dsn = os.environ.get("TEST_POSTGRES_DSN")
        if not dsn:
            pytest.skip("TEST_POSTGRES_DSN is not explicitly set")
        import asyncpg
        schema = "test_runtime_" + uuid4().hex
        admin = await asyncpg.connect(dsn)
        store = PostgresRuntimeStore(dsn)
        try:
            await admin.execute(f'CREATE SCHEMA "{schema}"')
            store._pool = await asyncpg.create_pool(
                dsn, min_size=1, max_size=5, server_settings={"search_path": schema}
            )
            await store.migrate()
            yield store
        finally:
            await store.close()
            # Only this test's generated namespace is ever removed.
            await admin.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            await admin.close()
    else:
        store = SqliteRuntimeStore(tmp_path / "runtime.db") if backend == "sqlite" else InMemoryRuntimeStore()
        if backend == "sqlite":
            await store.connect()
        try:
            yield store
        finally:
            if backend == "sqlite":
                await store.close()


BACKENDS = pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])


async def admitted_turn(store, thread=None, key="one"):
    thread = thread or await store.create_thread()
    pending = await store.admit_input(PendingInput(thread_id=thread.id, content="hello", idempotency_key=key))
    turn = await store.create_turn(Turn(thread_id=thread.id, input_id=pending.id))
    return thread, pending, turn


@BACKENDS
def test_transactional_turn_lifecycle(backend, tmp_path):
    async def run():
        async with storage(backend, tmp_path) as store:
            thread, pending, turn = await admitted_turn(store)
            duplicate = await store.admit_input(pending.model_copy(update={"content": "ignored"}))
            assert not duplicate.was_admitted and duplicate.content == "hello"
            assert (await store.get_input(pending.id)).content == "hello"
            assert [p.id for p in await store.pending_inputs()] == [pending.id]
            await store.start_turn(turn.id)
            await store.start_turn(turn.id)
            events = await store.events(thread.id)
            assert [e.kind for e in events] == [EventKind.INPUT_ADMITTED, EventKind.TURN_STARTED]
            assert events[-1].payload == {"input_id": str(pending.id), "content": "hello"}
            await store.finish_turn(turn.id, TurnStatus.COMPLETED, EventKind.TURN_COMPLETED, {"response": "hi"})
            await store.finish_turn(turn.id, TurnStatus.FAILED, EventKind.TURN_FAILED, {})
            assert (await store.get_turn(turn.id)).status == TurnStatus.COMPLETED
            assert (await store.get_turn(turn.id)).completed_at is not None
            assert await store.pending_inputs() == []
            assert len(await store.events(thread.id)) == 3
            job = await store.claim_memory_job()
            assert job.source_event_id == events[0].id
            assert await store.claim_memory_job() is None
    asyncio.run(run())


@BACKENDS
def test_jobs_fence_expired_claims_and_bound_retries(backend, tmp_path):
    async def run():
        async with storage(backend, tmp_path) as store:
            job = await store.submit_job("work", {"nested": [1, {"yes": True}]}, "one", max_attempts=2)
            assert (await store.submit_job("work", {}, "one"))["id"] == job["id"]
            claims = await asyncio.gather(*(store.claim_job("owner", ["work"], 1) for _ in range(4)))
            claimed = next(c for c in claims if c)
            assert sum(c is not None for c in claims) == 1
            assert claimed["payload"] == {"nested": [1, {"yes": True}]}
            assert not await store.finish_job(job["id"], "intruder", generation=claimed["generation"])
            await asyncio.sleep(1.05)
            assert not await store.renew_job(job["id"], "owner", claimed["generation"])
            replacement = await store.claim_job("owner", ["work"], 60)
            assert replacement["generation"] > claimed["generation"]
            assert not await store.finish_job(job["id"], "owner", generation=claimed["generation"])
            assert await store.finish_job(job["id"], "owner", generation=replacement["generation"], succeeded=False, retry=True, error="failed")
            assert (await store.get_job(job["id"]))["status"] == "failed"
            assert await store.claim_job("owner") is None
    asyncio.run(run())


@BACKENDS
def test_leases_cancellation_and_resource_quarantine(backend, tmp_path):
    async def run():
        async with storage(backend, tmp_path) as store:
            thread, pending, turn = await admitted_turn(store)
            await store.start_turn(turn.id)
            _, _, second = await admitted_turn(store, thread, "two")
            with pytest.raises(ValueError):
                await store.start_turn(second.id)
            assert await store.request_cancel(turn.id)
            assert await store.cancel_requested(turn.id)
            lease = RuntimeLease(name="resource:calendar", holder="a", expires_at=utcnow() + timedelta(seconds=60))
            assert await store.acquire_lease(lease)
            generation = (await store.get_lease(lease.name))["generation"]
            assert not await store.acquire_lease(lease.model_copy(update={"holder": "b"}))
            assert not await store.renew_lease(lease.name, "a", lease.expires_at, generation=generation + 1)
            assert await store.renew_lease(lease.name, "a", lease.expires_at, generation=generation)
            await store.quarantine_resource(lease.name, "a", generation=generation, reason="unknown outcome")
            await store.release_lease(lease.name, "a", generation=generation)
            assert not await store.acquire_lease(lease.model_copy(update={"holder": "b"}))
            assert await store.resource_quarantined(lease.name)
            job = await store.submit_job("work", {}, "cancel")
            assert await store.cancel_job(job["id"])
            assert await store.claim_job("a") is None
    asyncio.run(run())


@BACKENDS
def test_approval_decision_durably_queues_resume(backend, tmp_path):
    async def run():
        async with storage(backend, tmp_path) as store:
            thread, _, turn = await admitted_turn(store)
            approval = ApprovalRequest(thread_id=thread.id, turn_id=turn.id, tool_call=ToolCall(name="test", arguments={"x": [1]}), resource_scope=ResourceScope(resource="test"))
            await store.save_approval(approval)
            assert (await store.get_approval(approval.id)).tool_call.arguments == {"x": [1]}
            assert (await store.pending_approval(thread.id)).id == approval.id
            await store.decide_approval(ApprovalDecision(approval_id=approval.id, approved=True))
            job = await store.claim_job("a", ["approval.resume"])
            assert job["payload"] == {"approval_id": str(approval.id), "turn_id": str(turn.id), "approved": True}
            with pytest.raises(ValueError):
                await store.decide_approval(ApprovalDecision(approval_id=approval.id, approved=False))
            checkpoint = ContextCheckpoint(thread_id=thread.id, through_sequence=1, entities={"nested": [1]})
            await store.save_checkpoint(checkpoint)
            assert (await store.latest_checkpoint(thread.id)).entities == {"nested": [1]}
    asyncio.run(run())


def test_sqlite_reopens_pending_work_and_cancellation(tmp_path):
    async def run():
        path = tmp_path / "runtime.db"
        store = SqliteRuntimeStore(path)
        await store.connect()
        _, pending, turn = await admitted_turn(store)
        await store.request_cancel(turn.id)
        job = await store.submit_job("work", {"input_id": str(pending.id)}, "one")
        await store.close()
        restored = SqliteRuntimeStore(path)
        await restored.connect()
        assert await restored.cancel_requested(turn.id)
        assert (await restored.get_job(job["id"]))["payload"]["input_id"] == str(pending.id)
        assert [p.id for p in await restored.pending_inputs()] == [pending.id]
        await restored.close()
    asyncio.run(run())


def test_postgres_rolls_back_terminal_event_failure(tmp_path):
    async def run():
        async with storage("postgres", tmp_path) as store:
            thread, _, turn = await admitted_turn(store)
            await store.start_turn(turn.id)
            async with store._pool.acquire() as conn:
                await conn.execute("ALTER TABLE runtime_events ADD CONSTRAINT reject_terminal CHECK (kind <> 'turn.completed')")
            with pytest.raises(Exception):
                await store.finish_turn(turn.id, TurnStatus.COMPLETED, EventKind.TURN_COMPLETED, {})
            assert (await store.get_turn(turn.id)).status == TurnStatus.RUNNING
            assert len(await store.events(thread.id)) == 2
            assert await store.claim_memory_job() is None
    asyncio.run(run())
