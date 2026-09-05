"""Focused Runtime V2 invariants independent of external services."""

import asyncio
import base64
import os
from collections.abc import AsyncIterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.parse import urlparse

from core.providers.gemini import GeminiProvider
from core.runtime.contracts import (
    ApprovalDecision,
    ApprovalPolicy,
    EventKind,
    MemoryCandidate,
    ModelEvent,
    ModelRequest,
    PendingInput,
    PreparedModelCall,
    ProviderCapabilities,
    ResourceScope,
    RuntimeEvent,
    ToolCall,
    ToolContract,
    ToolEffect,
    TurnStatus,
)
from core.runtime.crypto import CredentialCipher
from core.runtime.engine import HeathcliffRuntime
from core.runtime.http_client import RuntimeV2HttpClient
from core.runtime.memory import MemoryJobWorker, SemanticMemoryPipeline
from core.runtime.tools import ToolRegistry
from db.artifact_store import LocalArtifactStore
from db.runtime_store import InMemoryRuntimeStore, SqliteRuntimeStore, _json_value


class _Provider:
    def __init__(self, rounds: list[list[ModelEvent]]) -> None:
        self.rounds = rounds
        self.calls = 0
        self.capabilities = ProviderCapabilities(provider="test", model="test-model")

    async def prepare(self, request: ModelRequest) -> PreparedModelCall:
        return PreparedModelCall(
            request=request, effective_config={"model": "test-model"}
        )

    async def stream(self, call: PreparedModelCall) -> AsyncIterator[ModelEvent]:
        del call
        events = self.rounds[self.calls]
        self.calls += 1
        for event in events:
            yield event


def _runtime(provider: _Provider, tools: ToolRegistry | None = None):
    return HeathcliffRuntime(
        store=InMemoryRuntimeStore(),
        provider=provider,
        tools=tools or ToolRegistry(),
        system_instruction="test",
    )


def test_runtime_commits_ordered_events_and_final_response():
    async def run():
        runtime = _runtime(
            _Provider([[ModelEvent(kind="text_delta", data={"text": "done"})]])
        )
        thread = await runtime.create_thread()
        pending = await runtime.admit_input(thread.id, "hello", "first")
        turn = await runtime.run_input(pending)
        events = await runtime.store.events(thread.id)
        assert turn.status == TurnStatus.COMPLETED
        assert [event.sequence for event in events] == list(range(1, len(events) + 1))
        assert events[-1].kind == EventKind.TURN_COMPLETED
        assert events[-1].payload["response"] == "done"
        assert len(runtime.store.memory_jobs) == 1

    asyncio.run(run())


def test_input_admission_and_turn_creation_are_idempotent():
    async def run():
        runtime = _runtime(
            _Provider([[ModelEvent(kind="text_delta", data={"text": "done"})]])
        )
        thread = await runtime.create_thread()
        first = await runtime.admit_input(thread.id, "hello", "request-1")
        duplicate = await runtime.admit_input(thread.id, "ignored", "request-1")
        assert first.id == duplicate.id
        assert first.was_admitted and not duplicate.was_admitted
        completed = await runtime.run_input(first)
        repeated = await runtime.run_input(duplicate)
        assert completed.id == repeated.id
        assert not repeated.was_created
        events = await runtime.store.events(thread.id)
        assert [event.kind for event in events].count(EventKind.INPUT_ADMITTED) == 1

    asyncio.run(run())


def test_runtime_waits_for_durable_approval_then_resumes():
    async def send(arguments):
        return {"sent": arguments["message"]}

    async def run():
        tools = ToolRegistry()
        tools.register(
            ToolContract(
                name="send_message",
                description="Send a message",
                input_schema={"type": "object"},
                effect=ToolEffect.EXTERNAL_SIDE_EFFECT,
                approval_policy=ApprovalPolicy.EXTERNAL_SIDE_EFFECTS,
                resource_scope=ResourceScope(resource="message"),
            ),
            send,
        )
        runtime = _runtime(
            _Provider(
                [
                    [
                        ModelEvent(
                            kind="tool_call",
                            data={
                                "name": "send_message",
                                "arguments": {"message": "hi"},
                            },
                        )
                    ],
                    [ModelEvent(kind="text_delta", data={"text": "sent"})],
                ]
            ),
            tools,
        )
        thread = await runtime.create_thread()
        pending = await runtime.admit_input(thread.id, "send it", "one")
        paused = await runtime.run_input(pending)
        approval = await runtime.store.pending_approval(thread.id)
        assert paused.status == TurnStatus.WAITING_FOR_APPROVAL
        assert approval is not None
        completed = await runtime.decide_approval(
            ApprovalDecision(approval_id=approval.id, approved=True)
        )
        assert completed.status == TurnStatus.COMPLETED
        assert any(
            event.kind == EventKind.APPROVAL_DECIDED
            for event in await runtime.store.events(thread.id)
        )

    asyncio.run(run())


def test_runtime_tool_results_keep_model_call_order():
    async def read(arguments):
        return arguments["value"]

    async def run():
        tools = ToolRegistry()
        for name in ("read_one", "read_two"):
            tools.register(
                ToolContract(
                    name=name,
                    description=name,
                    input_schema={"type": "object"},
                    resource_scope=ResourceScope(resource=name),
                ),
                read,
            )
        results = await tools.execute_ready(
            [
                ToolCall(name="read_one", arguments={"value": 1}),
                ToolCall(name="read_two", arguments={"value": 2}),
            ]
        )
        assert [result.output for result in results] == [1, 2]

    asyncio.run(run())


def test_side_effects_are_serial_even_if_misdeclared_parallel_safe():
    active = 0
    maximum = 0

    async def mutate(arguments):
        nonlocal active, maximum
        del arguments
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0)
        active -= 1

    async def run():
        tools = ToolRegistry()
        tools.register(
            ToolContract(
                name="mutate",
                description="mutation",
                input_schema={"type": "object"},
                effect=ToolEffect.EXTERNAL_SIDE_EFFECT,
                resource_scope=ResourceScope(resource="account"),
            ),
            mutate,
        )
        await tools.execute_ready([ToolCall(name="mutate"), ToolCall(name="mutate")])
        assert maximum == 1

    asyncio.run(run())


def test_runtime_v2_traces_native_model_and_tool_steps():
    root = Mock()
    observations = []

    @contextmanager
    def runtime_trace(**kwargs):
        assert kwargs["user_input"] == "check"
        yield root

    @contextmanager
    def step_trace(name, **kwargs):
        observation = Mock()
        observations.append((name, kwargs, observation))
        yield observation

    async def read(arguments):
        return {"value": arguments["value"]}

    async def run():
        tools = ToolRegistry()
        tools.register(
            ToolContract(
                name="read",
                description="Read a value",
                input_schema={"type": "object"},
                resource_scope=ResourceScope(resource="test"),
            ),
            read,
        )
        runtime = _runtime(
            _Provider(
                [
                    [
                        ModelEvent(
                            kind="tool_call",
                            data={"name": "read", "arguments": {"value": 1}},
                        )
                    ],
                    [ModelEvent(kind="text_delta", data={"text": "done"})],
                ]
            ),
            tools,
        )
        thread = await runtime.create_thread()
        pending = await runtime.admit_input(thread.id, "check", "trace")
        await runtime.run_input(pending)

    with (
        patch("core.runtime.engine.trace_runtime_turn", runtime_trace),
        patch("core.runtime.engine.trace_observation", step_trace),
    ):
        asyncio.run(run())

    assert root.update.called
    assert [name for name, _, _ in observations] == [
        "runtime.model",
        "runtime.tool.read",
        "runtime.model",
    ]
    assert observations[1][1]["input"]["arguments"] == {"value": 1}


def test_gemini_function_schema_removes_unsupported_additional_properties():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "nested": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"value": {"type": "string"}},
            }
        },
    }
    assert GeminiProvider._function_schema(schema) == {
        "type": "object",
        "properties": {
            "nested": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
            }
        },
    }


def test_runtime_serializes_binary_provider_state():
    assert GeminiProvider._thought_signature(b"opaque") == "b3BhcXVl"
    assert _json_value({"state": b"opaque"}) == {
        "state": {"encoding": "base64", "data": "b3BhcXVl"}
    }


def test_sqlite_store_survives_restart(tmp_path):
    async def run():
        path = tmp_path / "runtime.sqlite3"
        first = SqliteRuntimeStore(path)
        await first.connect()
        thread = await first.create_thread()
        admitted = await first.admit_input(
            PendingInput(
                thread_id=thread.id,
                content="remember this",
                idempotency_key="local-input",
            )
        )
        event = await first.append_event(
            RuntimeEvent(
                thread_id=thread.id,
                kind=EventKind.INPUT_ADMITTED,
                payload={"input_id": str(admitted.id)},
            )
        )
        await first.close()

        restored = SqliteRuntimeStore(path)
        await restored.connect()
        assert (await restored.events(thread.id))[0].id == event.id
        assert not (await restored.admit_input(admitted)).was_admitted

    asyncio.run(run())


def test_local_artifact_store_is_content_addressed(tmp_path):
    async def run():
        store = LocalArtifactStore(tmp_path / "artifacts")
        artifact = await store.put(b"local artifact", "text/plain")
        assert artifact.content_hash
        assert await store.ready()
        assert Path(urlparse(artifact.uri).path).read_bytes() == b"local artifact"

    asyncio.run(run())


def test_semantic_memory_rejects_inference_and_secrets():
    class Classifier:
        async def classify(self, source):
            return [
                MemoryCandidate(
                    kind="preference",
                    subject="user",
                    content="likes tea",
                    confidence=1,
                    source_event_id=source.id,
                    source_kind="user",
                ),
                MemoryCandidate(
                    kind="stable_fact",
                    subject="user",
                    content="guess",
                    confidence=1,
                    source_event_id=source.id,
                    source_kind="assistant_inference",
                ),
            ]

    class Embedder:
        async def embed(self, text):
            return [float(len(text))]

    async def run():
        store = InMemoryRuntimeStore()
        thread = await store.create_thread()
        event = await store.append_event(
            RuntimeEvent(thread_id=thread.id, kind=EventKind.MODEL_COMPLETED)
        )
        memories = await SemanticMemoryPipeline(
            store, Classifier(), Embedder()
        ).process(event)
        assert [memory.content for memory in memories] == ["likes tea"]

    asyncio.run(run())


def test_memory_worker_only_processes_a_committed_input_event():
    class Classifier:
        async def classify(self, source):
            return [
                MemoryCandidate(
                    kind="preference",
                    subject="user",
                    content="prefers tea",
                    confidence=1,
                    source_event_id=source.id,
                    source_kind="user",
                )
            ]

    class Embedder:
        async def embed(self, text):
            return [float(len(text))]

    async def run():
        store = InMemoryRuntimeStore()
        thread = await store.create_thread()
        source = await store.append_event(
            RuntimeEvent(
                thread_id=thread.id,
                kind=EventKind.INPUT_ADMITTED,
                payload={"content": "I prefer tea"},
            )
        )
        await store.enqueue_memory_job(source.id)
        worker = MemoryJobWorker(
            store, SemanticMemoryPipeline(store, Classifier(), Embedder())
        )
        assert await worker.run_once()
        assert [memory.content for memory in store.memories.values()] == ["prefers tea"]
        assert not await worker.run_once()

    asyncio.run(run())


def test_credential_cipher_uses_authenticated_encryption():
    key = base64.urlsafe_b64encode(os.urandom(32)).decode()
    cipher = CredentialCipher(key)
    encrypted = cipher.encrypt(b"refresh-token")
    assert encrypted != b"refresh-token"
    assert cipher.decrypt(encrypted) == b"refresh-token"


def test_http_client_maps_sse_events_and_durable_approvals():
    class Client(RuntimeV2HttpClient):
        def __init__(self):
            super().__init__("http://runtime.test")
            self.calls = []

        def _request(self, method, path, payload=None):
            self.calls.append((method, path, payload))
            if path.endswith("/turns"):
                return {"thread_id": path.split("/")[3], "event_cursor": "2"}
            return {"response": "sent"}

        def _events(self, thread_id, after):
            del thread_id, after
            yield {
                "event": "approval.required",
                "data": {
                    "payload": {
                        "id": "approval-1",
                        "tool_call": {"name": "send", "arguments": {"to": "a"}},
                    }
                },
            }

    client = Client()
    events = list(client.stream_invoke("send", "session-a"))
    approval = next(event for event in events if event["type"] == "approval_required")
    assert approval["data"]["tool_name"] == "send"
    assert client.resume_approval("session-a", approved=True) == "sent"
    assert client.calls[-1][2] == {"approved": True}
