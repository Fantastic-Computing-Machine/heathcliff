"""HTTP transport checks for the Runtime V2 UI gateway."""

import asyncio

from core.runtime.contracts import EventKind, ModelEvent, PreparedModelCall, ProviderCapabilities
from core.runtime.engine import HeathcliffRuntime
from core.runtime.tools import ToolRegistry
from db.runtime_store import InMemoryRuntimeStore
from ui.runtime_server import browser_event, create_app


class _Provider:
    capabilities = ProviderCapabilities(provider="test", model="test")

    async def prepare(self, request):
        return PreparedModelCall(request=request)

    async def stream(self, call):
        del call
        yield ModelEvent(kind="text_delta", data={"text": "Hello"})


def test_event_stream_exposes_incremental_text_before_turn_completion():
    """The browser receives text while a model stream is still running."""

    async def run():
        runtime = HeathcliffRuntime(
            InMemoryRuntimeStore(), _Provider(), ToolRegistry(), "test"
        )
        thread = await runtime.create_thread()
        pending = await runtime.admit_input(thread.id, "hello", "streaming")
        task = asyncio.create_task(runtime.run_input(pending))
        await task
        events = await runtime.store.events(thread.id)
        kinds = [event.kind for event in events]
        assert EventKind.MODEL_TEXT_DELTA in kinds
        assert kinds.index(EventKind.MODEL_TEXT_DELTA) < kinds.index(EventKind.TURN_COMPLETED)

    asyncio.run(run())


def test_browser_event_redacts_provider_continuation_state():
    event = browser_event(
        EventKind.MODEL_COMPLETED,
        {"text": "Hello", "provider_state": {"thought_signature": "secret", "usage": {"total_tokens": 4}}},
    )

    assert event == {
        "kind": "model.completed",
        "payload": {"text": "Hello", "usage": {"total_tokens": 4}},
    }
