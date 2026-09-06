"""Regression tests for the actual context delivered to the native provider."""

import asyncio
from uuid import uuid4

from core.providers.gemini import GeminiProvider
from core.runtime.context import build_context
from core.runtime.contracts import (
    ApprovalDecision, ApprovalPolicy, EventKind, ModelEvent, ModelRequest, PreparedModelCall, ProviderCapabilities,
    ResourceScope, RuntimeEvent, RuntimeItem, ToolContract,
)
from core.runtime.engine import HeathcliffRuntime
from core.runtime.tools import ToolRegistry
from db.runtime_store import InMemoryRuntimeStore


class RecordingProvider:
    capabilities = ProviderCapabilities(provider="test", model="test")

    def __init__(self, fail=False):
        self.requests = []
        self.fail = fail

    async def prepare(self, request):
        self.requests.append(request)
        return PreparedModelCall(request=request)

    async def stream(self, call):
        if self.fail:
            raise RuntimeError("provider unavailable")
        yield ModelEvent(kind="text_delta", data={"text": "Hello"})


def test_current_user_input_is_delivered_once_and_operational_events_are_excluded():
    async def run():
        provider = RecordingProvider()
        runtime = HeathcliffRuntime(InMemoryRuntimeStore(), provider, ToolRegistry(), "test")
        thread = await runtime.create_thread()
        pending = await runtime.admit_input(thread.id, "Remember the blue teapot", "first")
        await runtime.run_input(pending)
        items = provider.requests[0].context
        assert [item.content for item in items if item.kind == "user_message"] == [
            {"text": "Remember the blue teapot"}
        ]
        assert all(item.kind not in {"turn.started", "model.started"} for item in items)
        if hasattr(runtime, "close"):
            await runtime.close()
    asyncio.run(run())


def test_projection_uses_source_identity_not_random_ids():
    thread, turn, input_id = uuid4(), uuid4(), uuid4()
    events = [
        RuntimeEvent(thread_id=thread, sequence=1, kind=EventKind.INPUT_ADMITTED,
                     payload={"input_id": str(input_id), "content": "hello"}),
        RuntimeEvent(thread_id=thread, turn_id=turn, sequence=2, kind=EventKind.TURN_STARTED,
                     payload={"input_id": str(input_id)}),
    ]
    first = build_context(thread, events)
    assert first == build_context(thread, events)
    assert len(first) == 1
    assert first[0].id == events[0].id


def test_native_contents_preserve_model_parts_call_id_and_function_response():
    provider = GeminiProvider(api_key="test", model="configured-model")
    thread, turn = uuid4(), uuid4()
    model_content = {"role": "model", "parts": [{
        "function_call": {"id": "provider-call", "name": "read_state", "args": {}},
        "thought_signature": "c2lnbmF0dXJl",
    }]}
    request = ModelRequest(thread_id=thread, turn_id=turn, system_instruction="test",
        provider=provider.capabilities, context=[
            RuntimeItem(thread_id=thread, turn_id=turn, kind="user_message", content={"text": "Check state"}),
            RuntimeItem(thread_id=thread, turn_id=turn, kind="model_message", content=model_content),
            RuntimeItem(thread_id=thread, turn_id=turn, kind="tool_result", content={
                "name": "read_state", "provider_call_id": "provider-call", "response": {"status": "paused"}
            }),
        ])
    contents = provider._contents(request)
    assert contents[0] == {"role": "user", "parts": [{"text": "Check state"}]}
    assert contents[1] == model_content
    assert contents[2] == {"role": "user", "parts": [{"function_response": {
        "id": "provider-call", "name": "read_state", "response": {"status": "paused"}
    }}]}


def test_provider_failure_commits_one_failed_terminal_instead_of_leaving_running():
    async def run():
        runtime = HeathcliffRuntime(InMemoryRuntimeStore(), RecordingProvider(fail=True), ToolRegistry(), "test")
        thread = await runtime.create_thread()
        pending = await runtime.admit_input(thread.id, "hello", "failure")
        turn = await runtime.run_input(pending)
        assert turn.status.value == "failed"
        events = await runtime.store.events(thread.id)
        assert len([event for event in events if event.kind == EventKind.TURN_FAILED]) == 1
        if hasattr(runtime, "close"):
            await runtime.close()
    asyncio.run(run())


def test_approval_pause_retains_all_calls_and_executes_each_once_after_resume():
    async def run():
        class Provider(RecordingProvider):
            async def stream(self, call):
                if len(self.requests) == 1:
                    for name in ["needs_approval", "read_other"]:
                        yield ModelEvent(kind="tool_call", data={"name": name, "arguments": {}})
                else:
                    yield ModelEvent(kind="text_delta", data={"text": "done"})
        executed = []
        tools = ToolRegistry()
        for name, policy in [("needs_approval", ApprovalPolicy.ALWAYS), ("read_other", ApprovalPolicy.NEVER)]:
            async def handler(args, name=name):
                executed.append(name)
                return {"ok": True}
            tools.register(ToolContract(name=name, description=name, input_schema={"type": "object"},
                resource_scope=ResourceScope(resource=name), approval_policy=policy), handler)
        runtime = HeathcliffRuntime(InMemoryRuntimeStore(), Provider(), tools, "test")
        thread = await runtime.create_thread()
        pending = await runtime.admit_input(thread.id, "do both", "approval")
        paused = await runtime.run_input(pending)
        assert paused.status.value == "waiting_for_approval"
        events = await runtime.store.events(thread.id)
        assert len([event for event in events if event.kind == EventKind.TOOL_PROPOSED]) == 2
        approval = await runtime.store.pending_approval(thread.id)
        completed = await runtime.decide_approval(ApprovalDecision(approval_id=approval.id, approved=True))
        assert completed.status.value == "completed"
        assert sorted(executed) == ["needs_approval", "read_other"]
        if hasattr(runtime, "close"):
            await runtime.close()
    asyncio.run(run())


def test_cancel_interrupts_an_active_provider_stream():
    async def run():
        started = asyncio.Event()
        stopped = asyncio.Event()
        class Provider(RecordingProvider):
            async def stream(self, call):
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped.set()
                yield ModelEvent(kind="completed")
        store = InMemoryRuntimeStore()
        runtime = HeathcliffRuntime(store, Provider(), ToolRegistry(), "test")
        thread = await runtime.create_thread()
        pending = await runtime.admit_input(thread.id, "work", "cancel")
        running = asyncio.create_task(runtime.run_input(pending))
        await started.wait()
        turn_id = store.turns_by_input[pending.id]
        await runtime.cancel(turn_id)
        try:
            await asyncio.wait_for(stopped.wait(), 0.2)
        except asyncio.TimeoutError:
            running.cancel()
            await running
            raise AssertionError("Cancellation did not interrupt the provider")
        assert (await running).status.value == "cancelled"
        if hasattr(runtime, "close"):
            await runtime.close()
    asyncio.run(run())
