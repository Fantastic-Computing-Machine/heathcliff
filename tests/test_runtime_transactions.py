"""Store contracts shared by the daemon and recovery worker."""

import asyncio

from core.runtime.contracts import EventKind, PendingInput, Turn, TurnStatus
from db.runtime_store import InMemoryRuntimeStore


def test_admission_owns_the_input_event_and_duplicate_does_not_add_one():
    async def run():
        store = InMemoryRuntimeStore()
        thread = await store.create_thread()
        pending = PendingInput(thread_id=thread.id, content="hello", idempotency_key="one")
        await store.admit_input(pending)
        await store.admit_input(pending)
        events = await store.events(thread.id)
        assert len(events) == 1
        assert events[0].kind == EventKind.INPUT_ADMITTED
    asyncio.run(run())


def test_terminal_commit_is_single_and_enqueues_memory():
    async def run():
        store = InMemoryRuntimeStore()
        thread = await store.create_thread()
        pending = await store.admit_input(PendingInput(thread_id=thread.id, content="hello", idempotency_key="one"))
        turn = await store.create_turn(Turn(thread_id=thread.id, input_id=pending.id))
        assert hasattr(store, "finish_turn"), "terminal commits need a transactional store operation"
        await store.start_turn(turn.id)
        await store.finish_turn(turn.id, TurnStatus.COMPLETED, EventKind.TURN_COMPLETED, {"response": "hi"})
        await store.finish_turn(turn.id, TurnStatus.FAILED, EventKind.TURN_FAILED, {})
        assert (await store.get_turn(turn.id)).status == TurnStatus.COMPLETED
        assert len(store.memory_jobs) == 1
        assert [e.kind for e in await store.events(thread.id)] == [
            EventKind.INPUT_ADMITTED, EventKind.TURN_STARTED, EventKind.TURN_COMPLETED,
        ]
    asyncio.run(run())
